// CUDA-event operator comparison only. Model throughput and numerical qualification use the
// separate inference harness and ninfer_v100_fp8_test, respectively.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;

class Events {
public:
    Events() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~Events() {
        if (stop_) cudaEventDestroy(stop_);
        if (start_) cudaEventDestroy(start_);
    }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;

    template <class Call> float sample(Call&& call) {
        CUDA_CHECK(cudaEventRecord(start_));
        call();
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }
private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

struct Options {
    std::vector<int> widths{1,4,8,32,33,128,2048,4096};
    bool all_shapes=false;
    bool packed=false;
};

Options selected_options(int argc,char** argv) {
    Options options;
    bool saw_widths=false;
    for (int argument=2;argument<argc;++argument) {
        const std::string_view flag(argv[argument]);
        if (flag=="--all-shapes" && !options.all_shapes) options.all_shapes=true;
        else if (flag=="--packed" && !options.packed) options.packed=true;
        else if (flag=="--widths" && !saw_widths && argument+1<argc) {
            saw_widths=true;
            options.widths.clear();
            std::istringstream input(argv[++argument]);
            for (std::string token;std::getline(input,token,',');) {
                std::size_t consumed=0;
                const int value=std::stoi(token,&consumed);
                if (consumed!=token.size() || value<1 || value>4096)
                    throw std::invalid_argument("FP8 benchmark widths must be integers in [1,4096]");
                options.widths.push_back(value);
            }
            if (options.widths.empty()) throw std::invalid_argument("FP8 benchmark widths are empty");
        } else throw std::invalid_argument(
            "usage: ninfer_v100_fp8_bench native|ported [--widths 1,4,8,...] [--all-shapes] [--packed]");
    }
    return options;
}

template <class Call>
void measure(std::string_view mode, std::string_view op, int n, int k, int tokens, Call&& call) {
    // Allocation, fixture uploads, prepack and one-time cuBLAS handle creation are
    // outside the event samples. Each sample measures one complete public operator call.
    for (int i=0; i<2; ++i) call();
    CUDA_CHECK(cudaDeviceSynchronize());
    Events events;
    const float pilot = events.sample(call);
    const int minimum = tokens >= 2048 ? 5 : 10;
    const int maximum = tokens >= 2048 ? 5 : (tokens >= 128 ? 64 : 500);
    const int repetitions = std::clamp(int(std::ceil(50.0 / std::max(0.001F,pilot))),
                                       minimum,maximum);
    const auto before = ops::v100_kernel_route_counts();
    std::vector<float> samples;
    samples.reserve(repetitions);
    for (int i=0; i<repetitions; ++i) samples.push_back(events.sample(call));
    const auto after = ops::v100_kernel_route_counts();
    const double total = std::accumulate(samples.begin(),samples.end(),0.0);
    std::sort(samples.begin(),samples.end());
    const double median = samples.size()%2 ? samples[samples.size()/2]
        : (double(samples[samples.size()/2-1])+samples[samples.size()/2])/2;
    std::cout << std::setprecision(9)
              << "{\"mode\":\"" << mode << "\",\"op\":\"" << op << "\",\"shape\":["
              << n << ',' << k << "],\"T\":" << tokens << ",\"ms\":" << median
              << ",\"mean_ms\":" << total/repetitions << ",\"min_ms\":" << samples.front()
              << ",\"max_ms\":" << samples.back() << ",\"samples\":" << repetitions
              << ",\"total_ms\":" << total << ",\"ported\":" << after.ported-before.ported
              << ",\"fallback\":" << after.fallback-before.fallback << "}\n" << std::flush;
}

void benchmark_shape(std::string_view mode, int n, int k, const std::vector<int>& widths,bool packed) {
    auto host = test::quantized_weight::make_patterned_weight(
        QType::FP8_E4M3FN_ROW_BF16,n,k,31U);
    DeviceBuffer weights(host.payload.size());
    weights.copy_from_host(host.payload.data(),weights.bytes);
    auto weight = host.device_weight(weights.p);
    if(packed)ops::detail::fp8_prepack_qpn_sm70(weight);
    // Holding only the encoded parent avoids retaining a dequantized host matrix.
    host.payload.clear();
    host.payload.shrink_to_fit();
    for (const int tokens : widths) {
        // The full head is a narrow-only qualification; never query/reserve its
        // multi-GiB full FP16 wide-weight scratch just to benchmark decode.
        if (n==248320 && tokens>32) continue;
        std::vector<std::uint16_t> values(std::size_t(k)*tokens);
        for (std::size_t i=0; i<values.size(); ++i) {
            const float value = float(int((i*17+13)%127)-63) / 1024.0F;
            const std::uint32_t bits = test::quantized_weight::detail::float_bits(value);
            values[i] = static_cast<std::uint16_t>((bits+0x7fff+((bits>>16)&1))>>16);
        }
        DeviceBuffer x(values.size()*2), y(std::size_t(n)*tokens*2);
        x.copy_from_host(values.data(),x.bytes);
        Tensor input(x.p,DType::BF16,{k,tokens}),output(y.p,DType::BF16,{n,tokens});
        const auto linear_bytes = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
        WorkspaceArena workspace(std::max<std::size_t>(256,linear_bytes));
        measure(mode,"linear",n,k,tokens,[&] {
            auto scope=workspace.scope();
            ops::linear(input,weight,output,ops::LinearPolicy::A16Only,workspace,nullptr);
        });
        if (n == 16384 && k == 5120) {
            DeviceBuffer qkv(std::size_t(10240)*tokens*2),z(std::size_t(6144)*tokens*2);
            Tensor qkv_output(qkv.p,DType::BF16,{10240,tokens});
            Tensor z_output(z.p,DType::BF16,{6144,tokens});
            const auto gdn_bytes=ops::gdn_input_proj_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
            WorkspaceArena gdn_workspace(std::max<std::size_t>(256,gdn_bytes));
            measure(mode,"gdn_input_proj",n,k,tokens,[&] {
                auto scope=gdn_workspace.scope();
                ops::gdn_input_proj(input,weight,qkv_output,z_output,
                                    ops::LinearPolicy::A16Only,gdn_workspace,nullptr);
            });
        }
        if (n == 14336 && k == 5120) {
            DeviceBuffer q(std::size_t(6144)*tokens*2),g(std::size_t(6144)*tokens*2);
            DeviceBuffer key(std::size_t(1024)*tokens*2),value(std::size_t(1024)*tokens*2);
            Tensor query(q.p,DType::BF16,{6144,tokens}),gate(g.p,DType::BF16,{6144,tokens});
            Tensor key_output(key.p,DType::BF16,{1024,tokens});
            Tensor value_output(value.p,DType::BF16,{1024,tokens});
            const auto attn_bytes=ops::attn_input_proj_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
            WorkspaceArena attn_workspace(std::max<std::size_t>(256,attn_bytes));
            measure(mode,"attn_input_proj",n,k,tokens,[&] {
                auto scope=attn_workspace.scope();
                ops::attn_input_proj(input,weight,query,gate,key_output,value_output,
                                     ops::LinearPolicy::A16Only,attn_workspace,nullptr);
            });
        }
        if (n == 5120 && (k == 6144 || k == 17408)) {
            y.fill();
            const auto residual_bytes=ops::linear_add_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
            WorkspaceArena residual_workspace(std::max<std::size_t>(256,residual_bytes));
            measure(mode,"linear_add",n,k,tokens,[&] {
                auto scope=residual_workspace.scope();
                ops::linear_add(input,weight,output,ops::LinearPolicy::A16Only,residual_workspace,nullptr);
            });
        }
        if (n == 34816 && k == 5120) {
            const auto fused_bytes = ops::linear_swiglu_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
            WorkspaceArena fused_workspace(std::max<std::size_t>(256,fused_bytes));
            Tensor fused_output(y.p,DType::BF16,{n/2,tokens});
            measure(mode,"linear_swiglu",n,k,tokens,[&] {
                auto scope=fused_workspace.scope();
                ops::linear_swiglu(input,weight,fused_output,ops::LinearPolicy::A16Only,
                                   fused_workspace,nullptr);
            });
        }
    }
}
}

int main(int argc,char** argv) {
    if (argc == 1) {
        std::cerr<<"Pass native|ported to run this opt-in GPU benchmark.\n";
        return 77;
    }
    try {
        const std::string_view mode(argv[1]);
        if (mode!="native" && mode!="ported")
            throw std::invalid_argument("mode must be native or ported");
        const auto options=selected_options(argc,argv);
        if(options.packed && mode!="ported")throw std::invalid_argument("packed public GDN benchmark requires ported mode");
        const std::string record_mode=std::string(mode)+(options.packed?"-packed":"");
        int count=0;
        if (cudaGetDeviceCount(&count)!=cudaSuccess || count<1) return 77;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties,0));
        if (properties.major!=7 || properties.minor!=0) {
            std::cerr<<"This comparison targets an sm_70 device.\n";
            return 77;
        }
        ops::set_v100_kernel_policy(mode=="ported");
        benchmark_shape(record_mode,16384,5120,options.widths,options.packed);
        if (options.all_shapes) {
            benchmark_shape(record_mode,14336,5120,options.widths,options.packed);
            benchmark_shape(record_mode,5120,6144,options.widths,options.packed);
            benchmark_shape(record_mode,34816,5120,options.widths,options.packed);
            benchmark_shape(record_mode,5120,17408,options.widths,options.packed);
            if(!options.packed)benchmark_shape(record_mode,248320,5120,options.widths,false);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
