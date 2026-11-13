#include "ops/common/v100_kernel_policy.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_add.h"
#include "ops/op_tester.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/linear/sm70/v100_qpn.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <string>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
ninfer::test::quantized_weight::PackedWeight stacked(int n,int k,std::uint32_t seed) {
    auto p=ninfer::test::linear::make_nvfp4_weight(n,k,seed);
    p.payload.resize(p.weight_divisor_offset+8);
    const std::array<float,2> divisors{0.3F,1.7F};
    std::memcpy(p.payload.data()+p.weight_divisor_offset,divisors.data(),8);
    p.weight.weight_divisor_rows=n/2;
    p.weight.payload_bytes=p.payload.size();
    return p;
}

int nan_propagation() {
    using namespace ninfer;
    int failures=0;
    for (auto [n,k] : {std::pair{256,512},std::pair{5120,17408}})
    for (bool packed : {false,true}) {
        auto host=test::linear::make_nvfp4_weight(n,k,23U);
        DeviceBuffer weights(host.payload.size());
        weights.copy_from_host(host.payload.data(),weights.bytes);
        auto weight=host.device_weight(weights.p);
        if (packed) ops::detail::nvfp4_prepack_qpn_sm70(weight);
        for (int width : {1,9,65}) {
            // One quiet NaN in every column propagates through the represented dot product.
            std::vector<std::uint16_t> activation(std::size_t(k)*width,0x3f80);
            for (int c=0;c<width;++c) activation[std::size_t(c)*k]=0x7fc0;
            DeviceBuffer x(activation.size()*2),y(std::size_t(n)*width*2);
            x.copy_from_host(activation.data(),x.bytes);
            WorkspaceArena workspace(std::max<std::size_t>(256,ops::linear_workspace_capacity_bytes(
                QType::NVFP4,n,k,ops::LinearPolicy::A16Only,width,width)));
            Tensor input(x.p,DType::BF16,{k,width}), output(y.p,DType::BF16,{n,width});
            ops::linear(input,weight,output,ops::LinearPolicy::A16Only,workspace,nullptr);
            std::vector<std::uint16_t> values(std::size_t(n)*width);
            y.copy_to_host(values.data(),y.bytes);
            for (auto bits : values) {
                if ((bits&0x7f80)!=0x7f80 || (bits&0x007f)==0) {
                    std::cerr<<"ported NVFP4 failed NaN propagation\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    return failures;
}

int residual_output() {
    using namespace ninfer;
    constexpr int n=5120,k=17408,max_t=513;
    constexpr std::array<int,4> rows{0,1,n/2,n-1};
    constexpr std::array<int,13> widths{2,3,4,5,6,7,15,31,33,65,511,512,513};
    int failures=0;
    for (bool packed : {false,true}) {
        auto host=stacked(n,k,29U);
        const auto reference_weight=test::quantized_weight::materialize_rows_fp32(host,rows);
        std::vector<std::uint16_t> input_bits(std::size_t(k)*max_t);
        std::vector<float> reference_input(input_bits.size());
        for (std::size_t i=0;i<input_bits.size();++i) {
            input_bits[i]=test::f32_to_bf16(float(int((i*17+13)%127)-63)/256.0F);
            reference_input[i]=test::bf16_to_f32(input_bits[i]);
        }
        std::vector<double> projection(rows.size()*max_t);
        test::linear::cpu_linear_gemm_fp64(reference_weight.data(),reference_input.data(),
                                          projection.data(),int(rows.size()),k,max_t);
        std::vector<std::uint16_t> residual(std::size_t(n)*max_t);
        for (std::size_t i=0;i<residual.size();++i)
            residual[i]=test::f32_to_bf16(float(int((i*23+7)%127)-63)/32.0F);
        DeviceBuffer weight_bytes(host.payload.size()),x(input_bits.size()*2),y(residual.size()*2);
        weight_bytes.copy_from_host(host.payload.data(),weight_bytes.bytes);
        x.copy_from_host(input_bits.data(),x.bytes);
        auto weight=host.device_weight(weight_bytes.p);
        if (packed) ops::detail::nvfp4_prepack_qpn_sm70(weight);
        for (int width:widths) {
            y.copy_from_host(residual.data(),std::size_t(n)*width*2);
            WorkspaceArena ws(std::max<std::size_t>(256,ops::linear_add_workspace_capacity_bytes(
                QType::NVFP4,n,k,ops::LinearPolicy::A16Only,width,width)));
            Tensor input(x.p,DType::BF16,{k,width}),output(y.p,DType::BF16,{n,width});
            ops::linear_add(input,weight,output,ops::LinearPolicy::A16Only,ws,nullptr);
            std::vector<std::uint16_t> bits(std::size_t(n)*width);
            y.copy_to_host(bits.data(),bits.size()*2);
            std::vector<double> got(rows.size()*width),expected(got.size());
            for (int token=0;token<width;++token)
                for (std::size_t row=0;row<rows.size();++row) {
                    const auto logical=std::size_t(token)*rows.size()+row;
                    const auto physical=std::size_t(token)*n+rows[row];
                    got[logical]=test::bf16_to_f32(bits[physical]);
                    expected[logical]=projection[logical]+test::bf16_to_f32(residual[physical]);
                }
            failures+=test::verify_reduction("ported NVFP4 residual",got,expected,
                                               {1.0/256.0,1.0/256.0,2.0/256.0});
        }
    }
    return failures;
}

int split_swiglu_outliers() {
    using namespace ninfer;
    constexpr int n=34816,k=5120,half=n/2;
    constexpr std::array<std::int32_t,9> rows{0,1,31,32,127,128,half/2,half-2,half-1};
    std::vector<std::int32_t> sampled(rows.begin(),rows.end());
    for (int row:rows) sampled.push_back(half+row);
    int failures=0;
    for (bool packed : {false,true}) {
        auto host=stacked(n,k,3671U);
        const auto represented=test::quantized_weight::materialize_rows_fp32(host,sampled);
        DeviceBuffer weight_bytes(host.payload.size());
        weight_bytes.copy_from_host(host.payload.data(),weight_bytes.bytes);
        auto weight=host.device_weight(weight_bytes.p);
        if (packed) ops::detail::nvfp4_prepack_qpn_sm70(weight);
        for (int width : {1,3}) {
            std::vector<std::uint16_t> bits(std::size_t(k)*width,0);
            std::vector<float> reference_input(bits.size());
            for (int t=0;t<width;++t) {
                bits[std::size_t(t)*k]=test::f32_to_bf16(1048576.0F);
                bits[std::size_t(t)*k+k/2]=test::f32_to_bf16(0x1p-30F);
                bits[std::size_t(t)*k+k-1]=test::f32_to_bf16(float(t+1)/64.0F);
            }
            for (std::size_t i=0;i<bits.size();++i)
                reference_input[i]=test::bf16_to_f32(bits[i]);
            std::vector<double> sums(sampled.size()*width);
            test::linear::cpu_linear_gemm_fp64(represented.data(),reference_input.data(),sums.data(),
                                               int(sampled.size()),k,width);
            DeviceBuffer x(bits.size()*2);
            x.copy_from_host(bits.data(),x.bytes);
            test::GuardedDeviceBuffer y(std::size_t(half)*width*2);
            const auto capacity=ops::linear_swiglu_workspace_capacity_bytes(
                QType::NVFP4,n,k,ops::LinearPolicy::A16Only,width,width);
            test::GuardedDeviceBuffer scratch(std::max<std::size_t>(256,capacity));
            WorkspaceArena workspace(DeviceSpan{scratch.data(),std::max<std::size_t>(256,capacity)});
            DeviceContext context;
            Tensor input(x.p,DType::BF16,{k,width}),output(y.data(),DType::BF16,{half,width});
            const auto launch=[&] {
                ops::linear_swiglu(input,weight,output,ops::LinearPolicy::A16Only,workspace,
                                   context.stream);
            };
            DecodeGraphDefinition definition;
            DecodeGraphExecutable graph;
            test::cuda_synchronize();
            definition.capture(context.stream,launch);
            graph.instantiate(definition);
            for (int phase=0;phase<3;++phase) {
                auto active=bits;
                if (phase==1) for (auto& value:active) value^=0x8000U;
                if (phase==2) for (int t=0;t<width;++t) active[std::size_t(t)*k]=0x7fc0;
                x.copy_from_host(active.data(),x.bytes);
                // Finite poison makes an unwritten result fail the NaN check.
                y.fill(phase==2 ? 0x00 : 0xff);
                test::cuda_synchronize();
                graph.launch(context.stream);
                test::cuda_synchronize(context.stream);
                std::vector<std::uint16_t> actual(std::size_t(half)*width);
                y.copy_to_host(actual.data(),actual.size()*2);
                const std::string label=std::string("NVFP4 split SwiGLU graph ")+
                    (packed?"packed":"native")+" T="+std::to_string(width);
                if (phase==2) {
                    for (auto value:actual) {
                        if ((value&0x7f80)!=0x7f80 || (value&0x007f)==0) {
                            std::cerr<<label<<": NaN did not reach both half projections\n";
                            ++failures;
                            break;
                        }
                    }
                } else {
                    const double sign=phase?-1.0:1.0;
                    std::vector<double> got(rows.size()*width),expected(got.size());
                    for (int t=0;t<width;++t) for (std::size_t r=0;r<rows.size();++r) {
                        const auto at=std::size_t(t)*sampled.size()+r;
                        const double gate=sign*sums[at],up=sign*sums[at+rows.size()];
                        const double silu=gate>=0 ? gate/(1.0+std::exp(-gate))
                            : gate*std::exp(gate)/(1.0+std::exp(gate));
                        const auto selected=std::size_t(t)*rows.size()+r;
                        got[selected]=test::bf16_to_f32(actual[std::size_t(t)*half+rows[r]]);
                        // This focused check compares the final public BF16 value.
                        // Keep contraction and SiLU in FP64, then apply its output cast;
                        // a correct nearest BF16 cast can itself exceed 0.33% error.
                        expected[selected]=test::bf16_to_f32(
                            test::f32_to_bf16(static_cast<float>(silu*up)));
                    }
                    failures+=test::verify_reduction(label,got,expected,
                        {3.3e-3,5.0e-3,test::kBf16GrossRelativeFloor});
                }
                failures+=y.verify_guards(label);
                failures+=scratch.verify_guards(label);
                if (workspace.peak_used()>capacity || workspace.used()!=0) {
                    std::cerr<<label<<": workspace query/scope mismatch\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

// Representation-specific preparation and layout-blind query union. This checks
// route boundaries without retaining a second weight plane or launching a GEMM;
// the existing independent FP64 down suites qualify the selected arithmetic.
int down_staging_contract() {
    using namespace ninfer;
    namespace sm=ops::detail::sm70;
    if(!ops::v100_ported_kernels_enabled()) {
        if(sm::v100_qpn_scratch_bytes(QType::NVFP4,5120,17408,32)!=0) {
            std::cerr<<"native policy reserved ported down staging\n";return 1;
        }
        return 0;
    }
    const auto align=[](std::size_t bytes){return(bytes+255)/256*256;};
    int failures=0;
    for(auto [rows,k]:{std::pair{5120,17408},std::pair{34816,5120},std::pair{256,512}})
    for(int tokens:{1,32,33})
    for(auto layout:{QuantLayout::BlockScaleK16M128x4,QuantLayout::VoltaQpnPrepacked}) {
        const bool mlp=rows==34816 && k==5120;
        const bool down=rows==5120 && k==17408;
        const bool union_natural=(mlp||down) && tokens<=32;
        const bool selected_natural=(mlp || (down && layout==QuantLayout::VoltaQpnPrepacked)) && tokens<=32;
        const auto expected=(union_natural?sm::activation_bytes(k,tokens):align(std::size_t(tokens)*4))+
            (mlp?align(std::size_t(rows)*std::min(tokens,32)*sizeof(float)):0);
        const auto capacity=sm::v100_qpn_scratch_bytes(QType::NVFP4,rows,k,tokens);
        if(capacity!=expected) {
            std::cerr<<"NVFP4 down staging query union mismatch T="<<tokens<<'\n';++failures;
        }
        std::vector<std::uint16_t> bits(std::size_t(k)*tokens,test::f32_to_bf16(0.25F));
        bits[0]=test::f32_to_bf16(1048576.0F);
        bits[std::size_t(tokens-1)*k+1]=0x7fc0;
        DeviceBuffer input(bits.size()*2);input.copy_from_host(bits.data(),input.bytes);
        test::GuardedDeviceBuffer scratch(std::max<std::size_t>(256,capacity));
        WorkspaceArena arena(DeviceSpan{scratch.data(),std::max<std::size_t>(256,capacity)});
        {
            auto scope=arena.scope();
            sm::Activation activation;
            activation.bf16=static_cast<const __nv_bfloat16*>(input.p);
            activation.k=k;activation.tokens=tokens;activation.max_rows=rows;
            activation.source_layout=layout;
            sm::v100_qpn_prepare(activation,arena,nullptr,QType::NVFP4);
            test::cuda_synchronize();
            if(bool(activation.qpn_f16)!=selected_natural) {
                std::cerr<<"NVFP4 down staging actual-layout guard failed T="<<tokens<<'\n';++failures;
            }
            std::vector<std::uint32_t> flags(tokens);
            CUDA_CHECK(cudaMemcpy(flags.data(),activation.qpn_beyond,flags.size()*4,cudaMemcpyDeviceToHost));
            for(int column=0;column<tokens;++column)
                if(flags[column]!=(column==0 || column==tokens-1?1U:0U)) {
                    std::cerr<<"NVFP4 down staging range/NaN flag mismatch\n";++failures;break;
                }
            if(selected_natural) {
                std::array<std::uint16_t,3> half_bits{};
                CUDA_CHECK(cudaMemcpy(half_bits.data(),activation.qpn_f16,half_bits.size()*2,cudaMemcpyDeviceToHost));
                if(half_bits[0]!=0x7bff || half_bits[2]!=0x3400) {
                    std::cerr<<"NVFP4 down staging natural FP16 values mismatch\n";++failures;
                }
            }
            if(arena.peak_used()>capacity)++failures;
        }
        if(arena.used()!=0)++failures;
        failures+=scratch.verify_guards("NVFP4 down staging scratch");
    }
    const auto interval=sm::v100_qpn_scratch_capacity(QType::NVFP4,5120,17408,31,33);
    std::size_t point_max=0;
    for(int tokens:{31,32,33})point_max=std::max(point_max,
        sm::v100_qpn_scratch_bytes(QType::NVFP4,5120,17408,tokens));
    if(interval!=point_max) {
        std::cerr<<"NVFP4 down staging interval missed T32 boundary\n";++failures;
    }
    return failures;
}

}
int main(int argc,char** argv) {
    using namespace ninfer;
    namespace lt=test::linear;
    const bool native_packed=argc==2 && std::string_view(argv[1])=="--native-packed";
    if (argc!=1 && !native_packed) {
        std::cerr<<"usage: ninfer_v100_port_test [--native-packed]\n";
        return 2;
    }
    if (!lt::cuda_available() || !ops::detail::sm70::enabled()) return 77;
    try {
        ops::set_v100_kernel_policy(!native_packed);
        constexpr std::array<lt::Invocation,26> cases{{{1},{2},{3},{4},{5},{6},{7},{8},{9},{15},{16},{17},{31},{32},{33},{48},{63},{64},{65},{128},{256},{511},{512},{513},{4,lt::CallForm::Policy,ops::LinearPolicy::A16Only,true},{1,lt::CallForm::A16Convenience}}};
        int failures=down_staging_contract()+nan_propagation()+residual_output()+split_swiglu_outliers();
        failures+=lt::verify_workspace_envelopes(QType::NVFP4,34816,5120);
        failures+=lt::verify_workspace_envelopes(QType::NVFP4,5120,17408);
        for (bool packed : {false,true}) {
            for (auto [n,k] : {std::pair{34816,5120},std::pair{5120,17408},std::pair{256,512}}) {
                lt::ShapeCase c{n,k,1701U,lt::Comparison::SampledRows,true,cases};
                c.v100_prepack=packed;
                failures+=lt::run_shape("v100 ported NVFP4",lt::ActivationCompute::A16,stacked,c);
                c.out=DType::FP32;
                c.activation_signs=lt::ActivationSigns::CenteredWideRange;
                failures+=lt::run_shape("v100 ported FP32 outliers",lt::ActivationCompute::A16,stacked,c);
            }
        }
        // Unsupported QPN K=64 remains on the native implementation.
        lt::ShapeCase fallback{256,64,19U,lt::Comparison::Full,true,cases};
        failures+=lt::run_shape("v100 native shape fallback",lt::ActivationCompute::A16,lt::make_nvfp4_weight,fallback);
        namespace st=test::linear_swiglu;
        constexpr std::array<int,22> widths{1,2,3,4,5,6,7,8,9,15,16,31,32,33,48,63,64,65,256,511,512,513};
        constexpr std::array<int,11> graphs{2,3,4,5,6,7,15,31,33,65,513};
        for (bool packed : {false,true})
            failures+=st::run_profile("v100 ported SwiGLU",{QType::NVFP4,34816,5120,17408,1751U,st::ActivationCompute::A16,true},widths,graphs,packed);
        const auto routes=ops::v100_kernel_route_counts();
        std::cout<<"routes ported="<<routes.ported<<" fallback="<<routes.fallback<<'\n';
        if (native_packed) {
            if (routes.ported) {
                std::cerr<<"native-packed oracle selected a ported route\n";
                ++failures;
            }
        } else if (!routes.ported || !routes.fallback) {
            ++failures;
        }
        return failures?1:0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
