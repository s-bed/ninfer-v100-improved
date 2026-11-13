// QPN schedule and exact load-time permutation imported from geoffwatts/ninfer-v100 b37d0dd3.
// NInfer-all keeps its own parent geometry, per-source FP32 divisors and output contracts.
#include "ops/linear/sm70/v100_qpn.h"
#include "ops/linear/sm70/v100_cutlass.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_mma_gemm.cuh"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/sm70/sm70_small.h"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <algorithm>

// Host launch stubs and the sm_70 image own these kernels. Other device passes in a
// 70;75/... build must not parse calls to donor leaves that only exist for sm_70.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
namespace ninfer::ops::detail::sm70 {
namespace {
constexpr int kMaxQpnTokens = 32;
std::size_t aligned(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// The donor's actual MLP decode/verification route stages natural FP16 once,
// then uses two separately scheduled 17408-row products. The generic small-T
// plane is P8-permuted and cannot be consumed by the donor's natural loads.
bool natural_mlp_shape(int rows,int k,int tokens) {
    return rows==34816 && k==5120 && tokens>=1 && tokens<=kMaxQpnTokens;
}

// Contingency candidate: measured separately before selecting it in a release.
// The query reserves either representation; only packed down parents prepare it.
bool natural_down_shape(int rows,int k,int tokens) {
    return rows==5120 && k==17408 && tokens>=1 && tokens<=kMaxQpnTokens;
}

__global__ void column_flags(const __nv_bfloat16* x, int k, int tokens, std::uint32_t* flags) {
    const int t = int(blockIdx.x);
    bool any = false;
    for (int e = int(threadIdx.x); e < k; e += int(blockDim.x)) {
        const float f = __bfloat162float(x[std::int64_t(t) * k + e]);
        any |= !isfinite(f) || fabsf(f) > 65504.0F;
    }
    const int warp_any = __any_sync(0xffffffffu, any);
    if ((threadIdx.x & 31) == 0 && warp_any) atomicOr(flags + t, 1u);
}

// One represented weight, without its per-source divisor. This path runs only for BF16
// activation outliers that the FP16 tensor-core product saturated; ordinary calls skip it.
__device__ float represented_weight(const Weight& w, int row, int e) {
    const auto* codes = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    std::uint8_t nibble, scale;
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        const int lane = ((row & 31) / 8) * 4 + (row & 3) + ((row & 4) << 2);
        const std::int64_t tuple = (std::int64_t(row / 32) * (w.k / 16) + e / 16) * 32 + lane;
        const int q = e & 15;
        const int at = (q & 8) + ((q & 1) ? 4 : 0) + ((q & 7) >> 1);
        nibble = (codes[tuple * 8 + at / 2] >> ((at & 1) * 4)) & 15;
        scale = scales[tuple];
    } else {
        nibble = (codes[std::int64_t(row) * (w.k / 2) + e / 2] >> ((e & 1) * 4)) & 15;
        const int g = e / 16, inner = row & 127;
        scale = scales[(std::int64_t(row / 128) * (w.k / 64) + g / 4) * 512 +
                       (inner & 31) * 16 + (inner >> 5) * 4 + (g & 3)];
    }
    __nv_fp4x2_e2m1 f; f.__x = nibble;
    __nv_fp8_e4m3 s; s.__x = scale;
    return static_cast<float2>(f).x * float(s);
}

struct QpnOutput {
    DestinationEpilogue destination;
    Weight weight;
    const __nv_bfloat16* x;
    const std::uint32_t* beyond;
    int first;
    int row_base = 0;
    __device__ void store(int row, int token, float value) const {
        // Keep the full parent's coordinates for both per-source divisors and
        // saturation repair when the donor launch reads one physical half.
        const int parent_row = row_base + row;
        const int column = first + token;
        if (beyond[column]) {
            float excess = 0.0F;
            for (int e = 0; e < weight.k; ++e) {
                const float a = __bfloat162float(x[std::int64_t(column) * weight.k + e]);
                const float delta = saturated_excess(a);
                if (delta != 0.0F) excess = fmaf(represented_weight(weight, parent_row, e), delta, excess);
            }
            value += excess;
        }
        destination(parent_row, column, value);
    }
};

bool eligible(const Activation& x, const Weight& w) {
    return v100_ported_kernels_enabled() && enabled() && w.qtype == QType::NVFP4 &&
           (w.layout == QuantLayout::BlockScaleK16M128x4 || w.layout == QuantLayout::VoltaQpnPrepacked) &&
           w.n > 0 && w.n % 128 == 0 && w.k >= 512 && w.k % 64 == 0 &&
           x.tokens >= 1 && x.qpn_beyond != nullptr;
}

void project_qpn(const Activation& x, const Weight& w, std::span<const Destination> parts,
                 cudaStream_t stream, bool force_wide = false) {
    DestinationEpilogue epilogue;
    epilogue.factor = row_factor(w);
    for (auto d : parts) epilogue.set[epilogue.count++] = d;
    if (force_wide || !small_takes(x.tokens)) {
        using S = Nvfp4VoltaMmaSchedule;
        const dim3 grid(unsigned((w.n + S::kRowsPerCta-1)/S::kRowsPerCta),1,unsigned((x.tokens+S::kTTile-1)/S::kTTile));
        nvfp4_volta_mma_gemm_kernel<<<grid,S::kThreads,0,stream>>>(static_cast<const std::uint8_t*>(w.qdata),static_cast<const std::uint8_t*>(w.scales),x.bf16,nullptr,QpnOutput{epilogue,w,x.bf16,x.qpn_beyond,0},w.n,w.k,x.tokens,1.0F,1,w.layout == QuantLayout::VoltaQpnPrepacked);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (int first = 0; first < x.tokens; first += kMaxQpnTokens) {
        const int count = std::min(kMaxQpnTokens, x.tokens - first);
        Tensor input(const_cast<__nv_bfloat16*>(x.bf16 + std::int64_t(first) * x.k), DType::BF16, {x.k, count});
        const QpnOutput output{epilogue,w,x.bf16,x.qpn_beyond,first};
        if (x.qpn_f16) {
            launch_nvfp4_volta_qpn_with_fp16_activation(input,w,
                x.qpn_f16+std::int64_t(first)*x.k,output,w.n,1.0F,stream);
        } else {
            launch_nvfp4_volta_qpn_with_output(input,w,output,w.n,1.0F,stream);
        }
    }
}

__global__ void combine(const float* sums, int rows, int tokens, __nv_bfloat16* out) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::int64_t(rows) * tokens) return;
    const int t = int(i / rows), r = int(i % rows);
    const float g = sums[std::int64_t(t) * 2 * rows + r];
    const float u = sums[std::int64_t(t) * 2 * rows + rows + r];
    out[i] = __float2bfloat16_rn(silu(g) * u);
}
}

std::size_t v100_qpn_scratch_bytes(QType format,int max_rows,int k,int tokens) {
    if (format != QType::NVFP4 || !v100_ported_kernels_enabled() || !enabled() || tokens <= 0) return 0;
    return ((natural_mlp_shape(max_rows,k,tokens) || natural_down_shape(max_rows,k,tokens))
                ? activation_bytes(k,tokens)
                                                : aligned(std::size_t(tokens) * sizeof(std::uint32_t))) +
           ((max_rows == 34816 && k == 5120) ? aligned(std::size_t(max_rows) * std::min(tokens,kMaxQpnTokens) * sizeof(float)) : 0) +
           (v100_cutlass_wide_enabled() ? v100_cutlass_workspace_bytes(max_rows,k,tokens) : 0);
}

std::size_t v100_qpn_scratch_capacity(QType format,int rows,int k,int first,int last) {
    std::size_t result=0;
    const auto consider=[&](int t) {
        if (t>=first && t<=last) result=std::max(result,v100_qpn_scratch_bytes(format,rows,k,t));
    };
    consider(first); consider(last);
    // Extrema at each closed route band; CUTLASS has bounded 512-column products and
    // natural staging is reused above the native small/wide boundary.
    for (int t : {32,33,63,64,cublas_min_tokens()-1,cublas_min_tokens(),512,
                  2047,2048,kWideColumns}) consider(t);
    return result;
}

void v100_qpn_prepare(Activation& x,WorkspaceArena& ws,cudaStream_t stream,QType format) {
    if (format != QType::NVFP4 || !v100_ported_kernels_enabled() || !enabled() || x.tokens <= 0) return;
    if (natural_mlp_shape(x.max_rows,x.k,x.tokens) ||
        (x.source_layout==QuantLayout::VoltaQpnPrepacked &&
         natural_down_shape(x.max_rows,x.k,x.tokens))) {
        auto* natural=static_cast<__half*>(ws.alloc_bytes(activation_bytes(x.k,x.tokens)).data);
        convert_activation(x.bf16,x.k,x.tokens,natural,stream);
        x.qpn_f16=natural;
        // convert_activation already initializes these flags and preserves
        // original BF16 values for the existing excess/NaN correction.
        x.qpn_beyond=beyond_columns(natural,x.k,x.tokens);
    } else {
        auto* flags=static_cast<std::uint32_t*>(ws.alloc_bytes(
            aligned(std::size_t(x.tokens)*sizeof(std::uint32_t))).data);
        CUDA_CHECK(cudaMemsetAsync(flags,0,std::size_t(x.tokens)*sizeof(std::uint32_t),stream));
        column_flags<<<x.tokens,128,0,stream>>>(x.bf16,x.k,x.tokens,flags);
        CUDA_CHECK(cudaGetLastError());
        x.qpn_beyond=flags;
    }
    if (x.max_rows == 34816 && x.k == 5120)
        x.qpn_sums = static_cast<float*>(ws.alloc_bytes(aligned(std::size_t(x.max_rows) * std::min(x.tokens,kMaxQpnTokens) * sizeof(float))).data);
    if (v100_cutlass_wide_enabled()) {
        x.cutlass_bytes=v100_cutlass_workspace_bytes(x.max_rows,x.k,x.tokens);
        if (x.cutlass_bytes) x.cutlass_storage=ws.alloc_bytes(x.cutlass_bytes).data;
    }
}

bool v100_qpn_project(const Activation& x, const Weight& w, std::span<const Destination> destinations, cudaStream_t stream) {
    if (v100_cutlass_project(x,w,destinations,stream)) {
        record_v100_kernel_route(true);
        return true;
    }
    if (!eligible(x,w) || destinations.size() > kMaxDestinations) return false;
    // The standalone MMA comparison is opt-in. A production call outside the
    // qualified CUTLASS bands keeps the native packed-aware implementation.
    if (!small_takes(x.tokens) && v100_cutlass_wide_enabled()) return false;
    project_qpn(x,w,destinations,stream);
    record_v100_kernel_route(true);
    return true;
}

bool v100_qpn_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out, cudaStream_t stream) {
    if (v100_cutlass_swiglu(x,gate,up,out,stream)) {
        record_v100_kernel_route(true);
        return true;
    }
    if (up != nullptr || !eligible(x,gate) || x.qpn_sums == nullptr || gate.n != 34816 ||
        out.ne[0] != gate.n / 2 || out.ne[1] != x.tokens || out.dtype != DType::BF16) return false;
    if (!small_takes(x.tokens) && v100_cutlass_wide_enabled()) return false;
    Destination d{nullptr,gate.n,0,0,gate.n,false,x.qpn_sums};
    if (x.qpn_f16 && natural_mlp_shape(gate.n,gate.k,x.tokens)) {
        DestinationEpilogue epilogue;
        epilogue.factor=row_factor(gate);
        epilogue.set[epilogue.count++]=d;
        const int half_rows=gate.n/2;
        const Tensor input(const_cast<__nv_bfloat16*>(x.bf16),DType::BF16,{x.k,x.tokens});
        for (int row_base : {0,half_rows}) {
            // Both native M128 scales and QPN N32 panels have complete tile
            // boundaries at row17408. Their physical half-plane sizes agree.
            Weight half=gate;
            half.n=half.shape[0]=half.padded_shape[0]=half_rows;
            half.qdata=static_cast<const std::uint8_t*>(gate.qdata)+
                       std::int64_t(row_base)*(gate.k/2);
            half.scales=static_cast<const std::uint8_t*>(gate.scales)+
                        std::int64_t(row_base)*(gate.k/16);
            launch_nvfp4_volta_qpn_with_fp16_activation(input,half,x.qpn_f16,
                QpnOutput{epilogue,gate,x.bf16,x.qpn_beyond,0,row_base},
                half_rows,1.0F,stream);
        }
        const std::int64_t elements=std::int64_t(half_rows)*x.tokens;
        combine<<<unsigned((elements+255)/256),256,0,stream>>>(x.qpn_sums,half_rows,x.tokens,
            static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        record_v100_kernel_route(true);
        return true;
    }
    // Bounded FP32 projection tile: SiLU sees unrounded gate/up, while prompt width does not
    // reserve a full [34816,T] FP32 plane. Wide calls keep the donor's wide MMA schedule.
    for (int first=0; first<x.tokens; first+=kMaxQpnTokens) {
        Activation part=x;
        part.tokens=std::min(kMaxQpnTokens,x.tokens-first);
        part.bf16=x.bf16+std::int64_t(first)*x.k;
        part.qpn_beyond=x.qpn_beyond+first;
        project_qpn(part,gate,{&d,1},stream,!small_takes(x.tokens));
        const std::int64_t elements=std::int64_t(out.ne[0])*part.tokens;
        combine<<<unsigned((elements+255)/256),256,0,stream>>>(x.qpn_sums,out.ne[0],part.tokens,static_cast<__nv_bfloat16*>(out.data)+std::int64_t(first)*out.ne[0]);
        CUDA_CHECK(cudaGetLastError());
    }
    record_v100_kernel_route(true);
    return true;
}
}

#endif // host pass or sm_70 device image
