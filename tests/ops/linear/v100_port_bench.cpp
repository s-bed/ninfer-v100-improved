// CUDA-event operator comparison only. Model throughput and numerical qualification use the
// separate inference harness and ninfer_v100_port_test, respectively.
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ops/linear/sm70/v100_cutlass.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
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

std::vector<int> selected_widths(int argc, char** argv) {
    if (argc == 2) return {1,4,8,32,33,128,2048,4096};
    if (argc != 4 || std::string_view(argv[2]) != "--widths")
        throw std::invalid_argument("usage: ninfer_v100_port_bench native|native-packed|ported-mma|ported [--widths 1,4,8,...]");
    std::vector<int> result;
    std::istringstream input(argv[3]);
    for (std::string token; std::getline(input, token, ',');) {
        std::size_t consumed = 0;
        const int value = std::stoi(token,&consumed);
        if (consumed != token.size() || value < 1 || value > 4096)
            throw std::invalid_argument("benchmark widths must be integers in [1,4096]");
        result.push_back(value);
    }
    if (result.empty()) throw std::invalid_argument("benchmark width list is empty");
    return result;
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

void benchmark_shape(std::string_view mode, int n, int k, const std::vector<int>& widths) {
    auto host = test::quantized_weight::make_patterned_weight(
        QType::NVFP4,n,k,31U,{.weight_scale_divisor=1.0F,.input_scale_divisor=1.0F});
    DeviceBuffer weights(host.payload.size());
    weights.copy_from_host(host.payload.data(),weights.bytes);
    auto weight = host.device_weight(weights.p);
    // native-packed isolates the existing native route on the same packed representation.
    if (mode != "native") ops::detail::nvfp4_prepack_qpn_sm70(weight);
    // Holding only the encoded parent avoids retaining a dequantized host matrix.
    host.payload.clear();
    host.payload.shrink_to_fit();
    for (const int tokens : widths) {
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
            QType::NVFP4,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
        WorkspaceArena workspace(std::max<std::size_t>(256,linear_bytes));
        measure(mode,"linear",n,k,tokens,[&] {
            auto scope=workspace.scope();
            ops::linear(input,weight,output,ops::LinearPolicy::A16Only,workspace,nullptr);
        });
        if (n == 5120 && k == 17408) {
            y.fill();
            const auto residual_bytes=ops::linear_add_workspace_capacity_bytes(
                QType::NVFP4,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
            WorkspaceArena residual_workspace(std::max<std::size_t>(256,residual_bytes));
            measure(mode,"linear_add",n,k,tokens,[&] {
                auto scope=residual_workspace.scope();
                ops::linear_add(input,weight,output,ops::LinearPolicy::A16Only,residual_workspace,nullptr);
            });
        }
        if (n == 34816 && k == 5120) {
            const auto fused_bytes = ops::linear_swiglu_workspace_capacity_bytes(
                QType::NVFP4,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
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
        std::cerr<<"Pass native|native-packed|ported-mma|ported to run this opt-in GPU benchmark.\n";
        return 77;
    }
    try {
        const std::string_view mode(argv[1]);
        if (mode != "native" && mode != "native-packed" && mode != "ported" && mode != "ported-mma")
            throw std::invalid_argument("mode must be native, native-packed, ported-mma or ported");
        const auto widths=selected_widths(argc,argv);
        int count=0;
        if (cudaGetDeviceCount(&count)!=cudaSuccess || count<1) return 77;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties,0));
        if (properties.major!=7 || properties.minor!=0) {
            std::cerr<<"This comparison targets an sm_70 device.\n";
            return 77;
        }
        ops::detail::sm70::set_v100_wide_backend_mma_for_benchmark(mode=="ported-mma");
        ops::set_v100_kernel_policy(mode=="ported" || mode=="ported-mma");
        benchmark_shape(mode,34816,5120,widths);
        benchmark_shape(mode,5120,17408,widths);
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
