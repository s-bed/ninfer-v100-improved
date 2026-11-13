// Native RowScale FP8 qualification against exact stored-weight FP64 oracles.
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/sm70/v100_fp8.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace ninfer;
namespace lt = test::linear;
namespace qw = test::quantized_weight;

qw::PackedWeight broad_fp8_weight(int n,int k,std::uint32_t seed) {
    // The existing fixture includes both signed zeros, subnormal E4M3 codes,
    // all exponent boundaries and +/-448. No NaN weight code is valid here.
    auto result=lt::make_fp8_weight(n,k,seed);
    constexpr std::array<std::uint16_t,12> scales{
        0x0000,0x3580,0x3781,0x3983,0x3a01,0x3b43,
        0x3d99,0x3e81,0x3f81,0x4003,0x4105,0x4189};
    for (int row=0;row<n;++row) {
        const auto index=(std::uint32_t(row)*7U+(std::uint32_t(row)>>5)+seed)%scales.size();
        qw::detail::store_u16_le(result.payload,result.scale_plane_offset+std::size_t(row)*2,
                                scales[index]);
    }
    return result;
}

std::vector<lt::Invocation> calls(bool head=false) {
    std::vector<lt::Invocation> result;
    for (int t : {1,2,3,4,5,6,8,16,32,33,64,65,128,512,513})
        if (!head || t<=32) result.push_back({t});
    for (int t : {3,6,32,33,65,513})
        if (!head || t<=32)
            result.push_back({t,lt::CallForm::Policy,ops::LinearPolicy::A16Only,true});
    return result;
}

int route_scratch_contracts() {
    namespace sm=ops::detail::sm70;
    using Purpose=sm::ProjectionPurpose;
    struct Case { int n,k,t; Purpose purpose; };
    const Case cases[]{
        {16384,5120,1,Purpose::Generic},
        {16384,5120,32,Purpose::Generic},
        {16384,5120,33,Purpose::Generic},
        {16384,5120,4096,Purpose::Generic},
        {34816,5120,1,Purpose::Generic},
        {34816,5120,8,Purpose::Generic},
        {14336,5120,33,Purpose::Generic},
        {248320,5120,1,Purpose::Generic},
        {16384,5120,1,Purpose::GdnInput},
        {16384,5120,8,Purpose::GdnInput},
        {16384,5120,32,Purpose::GdnInput},
        {16384,5120,33,Purpose::GdnInput},
        {16384,5120,64,Purpose::GdnInput},
        {16384,5120,65,Purpose::GdnInput},
        {16384,5120,4096,Purpose::GdnInput},
        {16384,5120,4097,Purpose::GdnInput},
        {16384,5120,8192,Purpose::GdnInput},
        {14336,5120,33,Purpose::GdnInput},
        {34816,5120,1,Purpose::SwiGLU},
        {34816,5120,8,Purpose::SwiGLU},
        {34816,5120,9,Purpose::SwiGLU},
        {34816,5120,32,Purpose::SwiGLU},
        {34816,5120,33,Purpose::SwiGLU},
        {16384,5120,1,Purpose::SwiGLU},
    };
    int failures=0;
    for (const auto& c:cases) {
        const auto bytes=sm::v100_fp8_scratch_bytes(
            QType::FP8_E4M3FN_ROW_BF16,c.n,c.k,c.t,c.purpose);
        const auto interval=sm::v100_fp8_scratch_capacity(
            QType::FP8_E4M3FN_ROW_BF16,c.n,c.k,c.t,c.t,c.purpose);
        const bool shape=(c.k==5120 && (c.n==16384 || c.n==14336 || c.n==34816)) ||
                         (c.n==5120 && (c.k==6144 || c.k==17408));
        const bool reachable=(shape && c.t<=32) ||
            (c.purpose==Purpose::GdnInput && c.n==16384 && c.k==5120 && c.t>=33 && c.t<=4096);
        const bool expected=reachable && ops::v100_ported_kernels_enabled();
        if ((bytes!=0)!=expected || interval!=bytes) {
            std::cerr<<"FP8 purpose/scratch contract ["<<c.n<<','<<c.k<<"] T="<<c.t<<'\n';
            ++failures;
        }
    }
    // A declined right endpoint must retain scratch needed by an admitted
    // earlier width. These are policy bands, independent of allocation formulas.
    for (const auto purpose : {Purpose::GdnInput,Purpose::SwiGLU}) {
        const int n=purpose==Purpose::GdnInput?16384:34816;
        for (const auto [first,last] : {std::pair{8,9},std::pair{32,33},
                                       std::pair{4096,4097}}) {
            const auto left=sm::v100_fp8_scratch_bytes(
                QType::FP8_E4M3FN_ROW_BF16,n,5120,first,purpose);
            const auto right=sm::v100_fp8_scratch_bytes(
                QType::FP8_E4M3FN_ROW_BF16,n,5120,last,purpose);
            const auto capacity=sm::v100_fp8_scratch_capacity(
                QType::FP8_E4M3FN_ROW_BF16,n,5120,first,last,purpose);
            if (capacity!=std::max(left,right)) {
                std::cerr<<"FP8 route boundary scratch omitted a reachable width\n";
                ++failures;
            }
        }
    }
    return failures;
}

int plain_projections(bool packed=false) {
    const auto before=ops::v100_kernel_route_counts();
    int failures=0;
    for (auto [n,k] : {std::pair{16384,5120},std::pair{14336,5120},
                       std::pair{5120,6144},std::pair{34816,5120},
                       std::pair{5120,17408},std::pair{248320,5120}}) {
        if (packed && n==248320) continue; // Production eligibility excludes the head.
        const auto invocations=calls(n==248320);
        lt::ShapeCase shape{n,k,3601U,lt::Comparison::SampledRows,true,invocations};
        shape.v100_prepack=packed;
        failures+=lt::run_shape("V100 FP8 BF16",lt::ActivationCompute::A16,broad_fp8_weight,shape);
        shape.out=DType::FP32;
        failures+=lt::run_shape("V100 FP8 FP32",lt::ActivationCompute::A16,
                                broad_fp8_weight,shape);
        if (n==248320) {
            // Repair is a full N*K reduction: one outlier token covers the head
            // contract without turning every narrow head width into scalar stress.
            const std::array<lt::Invocation,1> outlier_calls{{{1}}};
            shape.invocations=outlier_calls;
            shape.activation_signs=lt::ActivationSigns::CenteredWideRange;
            failures+=lt::run_shape("V100 FP8 head FP32 outlier",lt::ActivationCompute::A16,
                                    broad_fp8_weight,shape);
        }
        if (n!=248320) failures+=lt::verify_workspace_envelopes(
            QType::FP8_E4M3FN_ROW_BF16,n,k);
    }
    const std::array<lt::Invocation,5> outlier_calls{{{1},{3},{5},{33},{65}}};
    lt::ShapeCase outliers{16384,5120,3605U,lt::Comparison::SampledRows,true,outlier_calls};
    outliers.v100_prepack=packed;
    outliers.activation_signs=lt::ActivationSigns::CenteredWideRange;
    failures+=lt::run_shape("V100 FP8 BF16 outliers",lt::ActivationCompute::A16,
                            broad_fp8_weight,outliers);
    outliers.out=DType::FP32;
    failures+=lt::run_shape("V100 FP8 FP32 outliers",lt::ActivationCompute::A16,
                            broad_fp8_weight,outliers);
    // This valid public shape has K=64, outside the donor QPN/CUTLASS domains.
    const auto fallback_calls=calls();
    if (!packed) failures+=lt::run_shape("V100 FP8 unsupported shape fallback",lt::ActivationCompute::A16,
        broad_fp8_weight,{256,64,3607U,lt::Comparison::Full,true,fallback_calls});
    if ((ops::v100_kernel_route_counts().ported!=before.ported)!=(packed && ops::v100_ported_kernels_enabled())) {
        std::cerr<<"generic FP8 projection selected an unqualified donor route\n";
        ++failures;
    }
    return failures;
}

int nan_propagation(bool packed=false) {
    constexpr int n=16384,k=5120;
    auto host=broad_fp8_weight(n,k,3613U);
    DeviceBuffer weight_bytes(host.payload.size());
    weight_bytes.copy_from_host(host.payload.data(),weight_bytes.bytes);
    Weight weight=host.device_weight(weight_bytes.p);
    if (packed) ops::detail::fp8_prepack_qpn_sm70(weight, nullptr);
    int failures=0;
    for (int width : {1,3,8,33,65}) {
        std::vector<std::uint16_t> values(std::size_t(k)*width,0x3b80);
        for (int t=0;t<width;++t) values[std::size_t(t)*k+(t*127)%k]=0x7fc0;
        DeviceBuffer x(values.size()*2);
        x.copy_from_host(values.data(),x.bytes);
        for (DType dtype : {DType::BF16,DType::FP32}) {
            const std::size_t capacity=ops::linear_workspace_capacity_bytes(
                weight.qtype,n,k,ops::LinearPolicy::A16Only,width,width);
            WorkspaceArena workspace(std::max<std::size_t>(256,capacity));
            const std::size_t element_bytes=dtype==DType::BF16?2:4;
            test::GuardedDeviceBuffer y(std::size_t(n)*width*element_bytes);
            Tensor input(x.p,DType::BF16,{k,width}),output(y.data(),dtype,{n,width});
            y.fill(0x00); // Finite poison makes unwritten NaN outputs fail.
            test::cuda_synchronize();
            ops::linear(input,weight,output,ops::LinearPolicy::A16Only,workspace,nullptr);
            test::cuda_synchronize();
            bool correct=true;
            if (dtype==DType::BF16) {
                std::vector<std::uint16_t> bits(std::size_t(n)*width);
                y.copy_to_host(bits.data(),bits.size()*2);
                for (auto value:bits)
                    correct&=(value&0x7f80)==0x7f80 && (value&0x007f)!=0;
            } else {
                std::vector<float> values_out(std::size_t(n)*width);
                y.copy_to_host(values_out.data(),values_out.size()*4);
                for (float value:values_out) correct&=std::isnan(value);
            }
            if (!correct) {
                std::cerr<<"FP8 activation NaN propagation failed T="<<width<<'\n';
                ++failures;
            }
            failures+=y.verify_guards("FP8 NaN output");
        }
    }
    return failures;
}

// Complete epilogues use the same stored-weight FP64 oracle as plain Linear. Only
// output rows are sampled; every input coordinate and every live token contributes.
int epilogue(int n,int k,bool fused,bool packed=false) {
    constexpr int max_t=513;
    const int out_rows=fused?n/2:n;
    const std::array<std::int32_t,9> rows{0,1,31,32,127,128,out_rows/2,out_rows-2,out_rows-1};
    std::vector<std::int32_t> projection_rows(rows.begin(),rows.end());
    if (fused) for (int row:rows) projection_rows.push_back(n/2+row);
    auto host=broad_fp8_weight(n,k,3623U);
    const auto reference_weight=qw::materialize_rows_fp32(host,projection_rows);
    std::vector<std::uint16_t> input(std::size_t(k)*max_t),negative(input.size());
    std::vector<float> reference_input(input.size());
    for (std::size_t i=0;i<input.size();++i) {
        input[i]=test::f32_to_bf16(float(int((i*17+13)%127)-63)/8192.0F);
        negative[i]=input[i]^0x8000U;
        reference_input[i]=test::bf16_to_f32(input[i]);
    }
    std::vector<double> projection(projection_rows.size()*max_t);
    lt::cpu_linear_gemm_fp64(reference_weight.data(),reference_input.data(),projection.data(),
                             int(projection_rows.size()),k,max_t);
    DeviceBuffer weights(host.payload.size()),x(input.size()*2);
    weights.copy_from_host(host.payload.data(),weights.bytes);
    x.copy_from_host(input.data(),x.bytes);
    Weight weight=host.device_weight(weights.p);
    if (packed) ops::detail::fp8_prepack_qpn_sm70(weight, nullptr);
    DeviceContext context;
    int failures=0;
    for (int width : {1,2,3,4,5,6,8,9,16,32,33,64,65,128,512,513}) {
        const auto before=ops::v100_kernel_route_counts();
        const std::size_t capacity=fused?
            ops::linear_swiglu_workspace_capacity_bytes(weight.qtype,n,k,
                ops::LinearPolicy::A16Only,width,width):
            ops::linear_add_workspace_capacity_bytes(weight.qtype,n,k,
                ops::LinearPolicy::A16Only,width,width);
        test::GuardedDeviceBuffer scratch(std::max<std::size_t>(256,capacity));
        WorkspaceArena workspace(DeviceSpan{scratch.data(),std::max<std::size_t>(256,capacity)});
        test::GuardedDeviceBuffer y(std::size_t(out_rows)*width*2);
        std::vector<std::uint16_t> residual(std::size_t(out_rows)*width);
        for (std::size_t i=0;i<residual.size();++i)
            residual[i]=test::f32_to_bf16(float(int((i*23+7)%127)-63)/32.0F);
        Tensor input_tensor(x.p,DType::BF16,{k,width});
        Tensor output(y.data(),DType::BF16,{out_rows,width});
        const auto launch=[&] {
            if (fused) ops::linear_swiglu(input_tensor,weight,output,ops::LinearPolicy::A16Only,
                                        workspace,context.stream);
            else ops::linear_add(input_tensor,weight,output,ops::LinearPolicy::A16Only,
                                  workspace,context.stream);
        };
        const bool graph_case=width==3 || width==6 || width==33 || width==65 || width==513;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        // Warm the public Op before capture, including any native handle initialization.
        if (!fused) y.copy_from_host(residual.data(),residual.size()*2);
        launch();
        test::cuda_synchronize(context.stream);
        if (graph_case) {
            definition.capture(context.stream,launch);
            graph.instantiate(definition);
        }
        for (int phase=0;phase<(graph_case?2:1);++phase) {
            const auto& active_input=phase?negative:input;
            x.copy_from_host(active_input.data(),x.bytes);
            if (!fused) y.copy_from_host(residual.data(),residual.size()*2);
            else y.fill(0xff);
            test::cuda_synchronize();
            if (graph_case) graph.launch(context.stream);
            else launch();
            test::cuda_synchronize(context.stream);
            std::vector<std::uint16_t> bits(residual.size());
            y.copy_to_host(bits.data(),bits.size()*2);
            std::vector<double> got(rows.size()*width),expected(got.size());
            const double sign=phase?-1.0:1.0;
            for (int t=0;t<width;++t) for (std::size_t r=0;r<rows.size();++r) {
                const auto logical=std::size_t(t)*rows.size()+r;
                const auto physical=std::size_t(t)*out_rows+rows[r];
                got[logical]=test::bf16_to_f32(bits[physical]);
                const auto source=std::size_t(t)*projection_rows.size()+r;
                const double gate=sign*projection[source];
                if (fused) {
                    const double up=sign*projection[source+rows.size()];
                    const double silu=gate>=0 ? gate/(1.0+std::exp(-gate))
                        : gate*std::exp(gate)/(1.0+std::exp(gate));
                    expected[logical]=silu*up;
                } else expected[logical]=gate+test::bf16_to_f32(residual[physical]);
            }
            const std::string label=std::string("FP8 ")+(fused?"SwiGLU":"residual")+
                " T="+std::to_string(width)+(graph_case?" graph":"");
            const test::ReductionCriterion tolerance=fused?
                test::ReductionCriterion{3.3e-3,5.0e-3,test::kBf16GrossRelativeFloor}:
                test::ReductionCriterion{1.0/256.0,1.0/256.0,2.0/256.0};
            failures+=test::verify_reduction(label,got,expected,tolerance);
            failures+=y.verify_guards(label);
            failures+=scratch.verify_guards(label);
            if (workspace.peak_used()>capacity || workspace.used()!=0) {
                std::cerr<<label<<": workspace query/scope mismatch\n";
                ++failures;
            }
        }
        const auto after=ops::v100_kernel_route_counts();
        const bool expected=((packed && width<=32) || (!packed && fused && width<=8)) && ops::v100_ported_kernels_enabled();
        if ((after.ported!=before.ported)!=expected) {
            std::cerr<<"FP8 epilogue selected an unqualified route T="<<width<<'\n';
            ++failures;
        }
        x.copy_from_host(input.data(),x.bytes);
    }
    return failures;
}
}

int main(int argc,char** argv) {
    const bool native=argc==2 && std::string_view(argv[1])=="--native";
    if (argc!=1 && !native) {
        std::cerr<<"usage: ninfer_v100_fp8_test [--native]\n";
        return 2;
    }
    if (!lt::cuda_available() || !ops::detail::sm70::enabled()) return 77;
    try {
        ops::set_v100_kernel_policy(!native);
        int failures=route_scratch_contracts()+plain_projections()+nan_propagation();
        failures+=epilogue(5120,6144,false);
        failures+=epilogue(5120,17408,false);
        failures+=epilogue(34816,5120,true);
        failures+=plain_projections(true);
        failures+=nan_propagation(true);
        failures+=epilogue(5120,6144,false,true);
        failures+=epilogue(5120,17408,false,true);
        failures+=epilogue(34816,5120,true,true);
        const auto counts=ops::v100_kernel_route_counts();
        std::cout<<"FP8 routes ported="<<counts.ported<<" fallback="<<counts.fallback<<'\n';
        if (native ? counts.ported!=0 : counts.ported==0 || counts.fallback==0) ++failures;
        return failures?1:0;
    } catch (const std::exception& e) {
        std::cerr<<"FP8 qualification: "<<e.what()<<'\n';
        return 1;
    }
}
