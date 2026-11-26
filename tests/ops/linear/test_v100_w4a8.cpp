#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#include "ops/linear/sm70/v100_w4a8.h"
#include "ops/op_tester.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
namespace sm = ops::detail::sm70;
constexpr auto policy = ops::LinearPolicy::AllowNvfp4Int8;

int quantization() {
    // Exact halfway values with an exactly representable scale, independent tokens,
    // all-zero groups, a subnormal scale and an infinity in a single group.
    constexpr int k = 128, t = 3;
    std::array<std::uint16_t, k*t> input{};
    for (int column = 0; column < t; ++column) {
        const float scale = std::ldexp(1.0F, column);
        for (int i = 0; i < 32; ++i) {
            input[column*k+i] = test::f32_to_bf16((i == 31 ? 127.0F : i-14.5F)*scale);
        }
        input[column*k+64] = 1;
        input[column*k+96] = 0x7f80;
    }
    DeviceBuffer x(sizeof(input));
    x.copy_from_host(input.data(), x.bytes);
    Tensor operand(x.p, DType::BF16, {k, t});
    const auto bytes = sm::w4a8_workspace_bytes(k, t);
    test::GuardedDeviceBuffer scratch(bytes);
    WorkspaceArena workspace(DeviceSpan{scratch.data(), bytes});
    const auto a = sm::w4a8_prepare(operand, workspace, nullptr);
    std::array<signed char, k*t> codes;
    std::array<float, k*t/32> scales;
    CUDA_CHECK(cudaMemcpy(codes.data(), a.w4a8_codes, sizeof(codes), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales.data(), a.w4a8_scales, sizeof(scales), cudaMemcpyDeviceToHost));
    int errors = scratch.verify_guards("W4A8 quantizer");
    {
        DeviceBuffer wide_x(k * (sm::kW4A8MaxTokens + 1) * 2);
        Tensor wide_operand(wide_x.p, DType::BF16, {k, sm::kW4A8MaxTokens + 1});
        WorkspaceArena wide_ws(sm::w4a8_workspace_bytes(k, sm::kW4A8MaxTokens + 1));
        bool rejected = false;
        try {
            (void)sm::w4a8_prepare(wide_operand, wide_ws, nullptr);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) { ++errors; }
    }
    for (int column = 0; column < t; ++column) {
        const float factor = std::ldexp(1.0F, column);
        if (scales[column*4] != factor || scales[column*4+1] != 0 ||
            scales[column*4+2] >= 0 || scales[column*4+3] >= 0) ++errors;
        for (int i = 0; i < 64; ++i) {
            const auto expected = std::nearbyint(test::bf16_to_f32(input[column*k+i])/factor);
            if (codes[column*k+i] != expected) ++errors;
        }
    }
    return errors;
}

int extreme_zero_product() {
    constexpr int n = 128, k = 64;
    auto host = test::linear::make_nvfp4_weight(n,k,443U);
    std::fill_n(host.payload.begin(), host.code_plane_bytes, 0);
    std::fill_n(host.payload.begin()+host.scale_plane_offset, host.scale_plane_bytes, 0x7e);
    DeviceBuffer bytes(host.payload.size()), x(k*2), y(n*4);
    bytes.copy_from_host(host.payload.data(),bytes.bytes);
    const auto weight = host.device_weight(bytes.p);
    std::array<std::uint16_t,k> bits{};
    bits[0] = 0x7f7f;
    x.copy_from_host(bits.data(),x.bytes);
    WorkspaceArena ws(ops::linear_workspace_capacity_bytes(QType::NVFP4,n,k,policy,1,1));
    Tensor input(x.p,DType::BF16,{k,1}), output(y.p,DType::FP32,{n,1});
    ops::linear(input,weight,output,policy,ws,nullptr);
    std::array<float,n> result;
    y.copy_to_host(result.data(),y.bytes);
    return std::any_of(result.begin(),result.end(),[](float v) { return v != 0; });
}

enum class Operation { Linear, Residual, SwiGLU, Attention, Gdn };

int projection(Operation op, int k, const std::vector<int>& parts, bool packed,
               std::span<const int> widths, bool integer_fixture = false) {
    int n = 0;
    std::vector<std::int32_t> rows;
    for (int part : parts) {
        rows.insert(rows.end(), {n, n+1, n+part-1});
        n += part;
    }
    auto host = test::linear::make_nvfp4_weight(n, k, 439U);
    if (integer_fixture) {
        // Scale the E4M3 blocks by 2. Then every E2M1 nibble represents an
        // integer and a single active block exposes its exact DP4A dot.
        std::fill_n(host.payload.begin()+host.scale_plane_offset, host.scale_plane_bytes, 0x40);
        const float divisor = 1;
        std::memcpy(host.payload.data()+host.weight_divisor_offset, &divisor, sizeof(divisor));
    } else if (op != Operation::Gdn) {
        // Two serialized source factors in one parent, sampled across the row boundary.
        const std::array<float,2> divisors{0.375F,1.25F};
        host.payload.resize(host.weight_divisor_offset+sizeof(divisors));
        std::memcpy(host.payload.data()+host.weight_divisor_offset,divisors.data(),sizeof(divisors));
        host.weight.weight_divisor_rows = n/2;
        host.weight.payload_bytes = host.payload.size();
    }
    const auto represented = test::quantized_weight::materialize_rows_fp32(host, rows);
    DeviceBuffer bytes(host.payload.size());
    bytes.copy_from_host(host.payload.data(), bytes.bytes);
    auto weight = host.device_weight(bytes.p);
    if (packed) ops::detail::nvfp4_prepack_qpn_sm70(weight);
    int failures = 0;
    DeviceContext device;
    for (int t : widths) {
        const bool narrow = t <= sm::kW4A8MaxTokens;
        std::vector<std::uint16_t> xbits(std::size_t(k)*t);
        std::vector<float> xref(xbits.size());
        for (std::size_t g = 0; g < xbits.size(); g += 32) {
            const float scale = integer_fixture ? 1.0F : std::ldexp(1.0F, -6-int(g/32%3));
            for (int i = 0; i < 32; ++i) {
                float v = (i == 0 ? 127.0F : float(int((g+13*i)%63)-31)+0.5F)*scale;
                if (integer_fixture && ((g % k) != 0 || i >= 16)) v = 0;
                xbits[g+i] = test::f32_to_bf16(v);
                v = test::bf16_to_f32(xbits[g+i]);
                xref[g+i] = narrow ? std::nearbyint(v/scale)*scale : v;
            }
        }
        std::vector<double> dots(rows.size()*t);
        test::linear::cpu_linear_gemm_fp64(represented.data(), xref.data(), dots.data(),
                                          int(rows.size()), k, t);
        constexpr auto fmt = QType::NVFP4;
        std::size_t capacity = 0;
        switch (op) {
        case Operation::Linear: capacity = ops::linear_workspace_capacity_bytes(fmt,n,k,policy,t,t); break;
        case Operation::Residual: capacity = ops::linear_add_workspace_capacity_bytes(fmt,n,k,policy,t,t); break;
        case Operation::SwiGLU: capacity = ops::linear_swiglu_workspace_capacity_bytes(fmt,n,k,policy,t,t); break;
        case Operation::Attention: capacity = ops::attn_input_proj_workspace_capacity_bytes(fmt,n,k,policy,t,t); break;
        case Operation::Gdn: capacity = ops::gdn_input_proj_workspace_capacity_bytes(fmt,n,k,policy,t,t); break;
        }
        test::GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity,256));
        WorkspaceArena ws(DeviceSpan{scratch.data(),std::max<std::size_t>(capacity,256)});
        DeviceBuffer x(xbits.size()*2);
        x.copy_from_host(xbits.data(),x.bytes);
        Tensor input(x.p,DType::BF16,{k,t});
        const auto output_count = op == Operation::SwiGLU ? 1U : parts.size();
        std::vector<std::unique_ptr<test::GuardedDeviceBuffer>> buffers;
        std::vector<Tensor> outputs;
        for (std::size_t j=0; j<output_count; ++j) {
            const std::size_t elements = std::size_t(parts[j])*t;
            const bool fp32 = integer_fixture && op == Operation::Linear;
            auto b = std::make_unique<test::GuardedDeviceBuffer>(elements*(fp32 ? 4 : 2));
            const std::vector<std::uint16_t> init(elements, test::f32_to_bf16(0.5F));
            CUDA_CHECK(cudaMemcpy(b->data(),init.data(),elements*2,cudaMemcpyHostToDevice));
            outputs.emplace_back(b->data(),fp32 ? DType::FP32 : DType::BF16,std::initializer_list<std::int32_t>{parts[j],t});
            buffers.push_back(std::move(b));
        }
        auto launch = [&] {
            switch (op) {
            case Operation::Linear: ops::linear(input,weight,outputs[0],policy,ws,device.stream); break;
            case Operation::Residual: ops::linear_add(input,weight,outputs[0],policy,ws,device.stream); break;
            case Operation::SwiGLU: ops::linear_swiglu(input,weight,outputs[0],policy,ws,device.stream); break;
            case Operation::Attention: ops::attn_input_proj(input,weight,outputs[0],outputs[2],outputs[1],
                                                           outputs[3],policy,ws,device.stream); break;
            case Operation::Gdn: ops::gdn_input_proj(input,weight,outputs[0],outputs[1],policy,ws,device.stream); break;
            }
        };
        const auto before = ops::v100_kernel_route_counts();
        if (t == 4) {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable graph;
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
            graph.launch(device.stream);
            test::cuda_synchronize(device.stream);
        } else { launch(); test::cuda_synchronize(device.stream); }
        const auto after = ops::v100_kernel_route_counts();
        if ((narrow ? after.w4a8_narrow == before.w4a8_narrow : after.w4a8_wide == before.w4a8_wide)) ++failures;
        for (std::size_t j=0; j<output_count; ++j) {
            std::vector<std::uint16_t> result(std::size_t(parts[j])*t);
            std::vector<float> exact(integer_fixture ? result.size() : 0);
            if (integer_fixture) buffers[j]->copy_to_host(exact.data(),exact.size()*4);
            else buffers[j]->copy_to_host(result.data(),result.size()*2);
            std::vector<double> expected, got;
            for (int token=0; token<t; ++token) {
                for (int i=0; i<3; ++i) {
                    const auto pos = token*rows.size()+j*3+i;
                    double value = dots[pos];
                    if (op == Operation::Residual) value += 0.5;
                    if (op == Operation::SwiGLU) value = (value/(1+std::exp(-value)))*dots[pos+3];
                    const int index = i == 2 ? parts[j]-1 : i;
                    const auto location = token*parts[j]+index;
                    got.push_back(integer_fixture ? exact[location] : test::bf16_to_f32(result[location]));
                    expected.push_back(value);
                    if (integer_fixture && got.back() != value) ++failures;
                }
            }
            failures += test::verify_reduction("W4A8 scaled epilogue",got,expected,
                                               {1.0/256.0,1.0/256.0,2.0/256.0});
            failures += buffers[j]->verify_guards("W4A8 output");
        }
        failures += scratch.verify_guards("W4A8 workspace");
    }
    return failures;
}

int mixed_operand_fallbacks() {
    constexpr int rows = 128, k = 512, t = 3;
    auto gate_host = test::linear::make_nvfp4_weight(rows, k, 501U);
    auto up_host = test::linear::make_nvfp4_weight(rows, k, 503U);
    auto t2_host = test::linear::make_t2_g128_fp16_weight(rows, k, 509U);
    DeviceBuffer gate_bytes(gate_host.payload.size()), up_bytes(up_host.payload.size()),
        t2_bytes(t2_host.payload.size());
    gate_bytes.copy_from_host(gate_host.payload.data(), gate_bytes.bytes);
    up_bytes.copy_from_host(up_host.payload.data(), up_bytes.bytes);
    t2_bytes.copy_from_host(t2_host.payload.data(), t2_bytes.bytes);
    auto gate = gate_host.device_weight(gate_bytes.p);
    const auto up = up_host.device_weight(up_bytes.p);
    const auto t2 = t2_host.device_weight(t2_bytes.p);
    ops::detail::nvfp4_prepack_qpn_sm70(gate); // gate is VoltaQpnPrepacked, up stays BlockScale

    std::vector<std::uint16_t> xbits(std::size_t(k) * t);
    std::vector<float> xref(xbits.size());
    for (std::size_t i = 0; i < xbits.size(); ++i) {
        xbits[i] = test::f32_to_bf16(float(int((i * 11 + 5) % 41) - 20) / 64.0F);
        xref[i] = test::bf16_to_f32(xbits[i]);
    }
    DeviceBuffer x(xbits.size() * 2), out_swiglu(std::size_t(rows) * t * 2),
        out_t2(std::size_t(rows) * t * 2);
    x.copy_from_host(xbits.data(), x.bytes);
    Tensor input(x.p, DType::BF16, {k, t});
    Tensor swiglu_tensor(out_swiglu.p, DType::BF16, {rows, t});
    WorkspaceArena ws(std::max<std::size_t>(
        256, sm::covered_bytes(QType::NVFP4, rows, k, t, t, policy)));
    const auto activation = sm::prepare_for(
        input, rows, ws, nullptr, QType::NVFP4, false, sm::ProjectionPurpose::Generic, gate.layout,
        policy);
    int failures = 0;
    const auto before = ops::v100_kernel_route_counts();
    // Mismatched split SwiGLU layouts decline W4A8 cleanly and run the BF16 direct small-T route.
    sm::swiglu_for(activation, gate, &up, swiglu_tensor, nullptr);
    // A T2 sibling must not mistake the W4A8 activation plane for its own integer format.
    const sm::Destination dest{static_cast<__nv_bfloat16*>(out_t2.p), rows, 0, 0, rows, false,
                               nullptr};
    sm::project_for(activation, t2, {&dest, 1}, nullptr);
    test::cuda_synchronize();
    const auto after = ops::v100_kernel_route_counts();
    if (after.w4a8_narrow != before.w4a8_narrow || after.w4a8_wide != before.w4a8_wide) {
        ++failures;
    }

    std::vector<std::int32_t> all_rows(rows);
    for (int r = 0; r < rows; ++r) { all_rows[r] = r; }
    const auto gate_ref = test::quantized_weight::materialize_rows_fp32(gate_host, all_rows);
    const auto up_ref = test::quantized_weight::materialize_rows_fp32(up_host, all_rows);
    const auto t2_ref = test::quantized_weight::materialize_rows_fp32(t2_host, all_rows);
    const auto elements = std::size_t(rows) * t;
    std::vector<double> expected_gate(elements), expected_up(elements), got_swiglu(elements);
    std::vector<double> expected_t2(elements), got_t2(elements);
    test::linear::cpu_linear_gemm_fp64(gate_ref.data(), xref.data(), expected_gate.data(), rows, k, t);
    test::linear::cpu_linear_gemm_fp64(up_ref.data(), xref.data(), expected_up.data(), rows, k, t);
    test::linear::cpu_linear_gemm_fp64(t2_ref.data(), xref.data(), expected_t2.data(), rows, k, t);
    const auto actual_swiglu = test::from_device<std::uint16_t>(out_swiglu, elements);
    const auto actual_t2 = test::from_device<std::uint16_t>(out_t2, elements);
    for (std::size_t i = 0; i < elements; ++i) {
        const double gate_value = expected_gate[i];
        expected_gate[i] = gate_value / (1.0 + std::exp(-gate_value)) * expected_up[i];
        got_swiglu[i] = test::bf16_to_f32(actual_swiglu[i]);
        got_t2[i] = test::bf16_to_f32(actual_t2[i]);
    }
    failures += test::verify_reduction("W4A8 mixed-layout fallback", got_swiglu, expected_gate,
                                       {1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0});
    failures += test::verify_reduction("W4A8 mixed-format fallback", got_t2, expected_t2,
                                       {1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0});
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (!test::linear::cuda_available() || !sm::enabled()) return 77;
    try {
        ops::set_v100_kernel_policy(true,true);
        if (argc == 2 && std::string_view(argv[1]) == "--mixed-fallback-only") {
            return mixed_operand_fallbacks() == 0 ? 0 : 1;
        }
        const int crossover = sm::cublas_min_tokens();
        const std::array widths{1,2,3,4,5,32,33,crossover-1,crossover,crossover+1};
        const std::array fused{1,4,5,32,crossover-1,crossover};
        int failures = quantization() + extreme_zero_product() + mixed_operand_fallbacks();
        for (bool packed : {false,true}) {
            failures += projection(Operation::Linear,512,{256},packed,widths,true);
            failures += projection(Operation::Residual,17408,{5120},packed,fused);
            failures += projection(Operation::SwiGLU,5120,{17408,17408},packed,fused);
        }
        failures += projection(Operation::Attention,5120,{6144,1024,6144,1024},false,fused);
        failures += projection(Operation::Gdn,5120,{10240,6144},false,fused);
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
