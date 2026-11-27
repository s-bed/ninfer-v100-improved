// Volta Q4/Q8 panel GEMV adapted from RodrigoR78/ninfer-all-v100 (f0c49290, 55a696f4).
#include "ops/linear/sm70/v100_drafter_gemv.h"

#include "core/device.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/common/math.cuh"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/sm70/sm70_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
namespace ninfer::ops::detail::sm70 {
namespace {

constexpr int kWarps       = 8;
constexpr int kPanelRows   = kRowSplitPanelRows;
constexpr int kRecordBytes = 32; // one G32 Q8 group, one G64 Q4 group

// Both formats keep a 32-byte record per (row, group); a lane reads one 16-byte half of one.
struct Q8Format {
    static constexpr int kGroup = 32;
    // 16 signed codes against 16 activations from k0.
    __device__ __forceinline__ static float dot(const uint4 codes, const __nv_bfloat16* x) {
        const uint4 xa = __ldg(reinterpret_cast<const uint4*>(x));
        const uint4 xb = __ldg(reinterpret_cast<const uint4*>(x) + 1);
        const unsigned c[4]  = {codes.x, codes.y, codes.z, codes.w};
        const unsigned xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float q = static_cast<float>(static_cast<signed char>(c[i] >> (8 * b)));
                const unsigned pair = xv[(4 * i + b) / 2];
                const float a = __uint_as_float((b & 1) ? (pair & 0xffff0000u) : (pair << 16));
                sum = fmaf(q, a, sum);
            }
        }
        return sum;
    }
};

struct Q4Format {
    static constexpr int kGroup = 64;
    // 32 signed nibbles (code 2b in byte b's low nibble, 2b + 1 in its high one) from k0.
    __device__ __forceinline__ static float dot(const uint4 codes, const __nv_bfloat16* x) {
        const unsigned c[4] = {codes.x, codes.y, codes.z, codes.w};
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 xv4 = __ldg(reinterpret_cast<const uint4*>(x) + i); // values 8i..8i+7
            const unsigned xv[4] = {xv4.x, xv4.y, xv4.z, xv4.w};
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int nib = static_cast<int>((c[i] >> (4 * n)) & 0xfu);
                const float q = static_cast<float>((nib ^ 8) - 8);
                const unsigned pair = xv[n / 2];
                const float a = __uint_as_float((n & 1) ? (pair & 0xffff0000u) : (pair << 16));
                sum = fmaf(q, a, sum);
            }
        }
        return sum;
    }
};

struct View {
    const std::uint8_t* base = nullptr;
    const __half* scales     = nullptr;
    std::int32_t groups      = 0; // per row
    std::int32_t shift       = 0; // panel shift (0: row-major)

    __device__ __forceinline__ const uint4* record(std::int32_t row, std::int32_t group) const {
        const std::int64_t mask = (std::int64_t{1} << shift) - 1;
        return reinterpret_cast<const uint4*>(
            base + ((row & ~mask) * groups + (row & mask)) * kRecordBytes +
            static_cast<std::int64_t>(group) * (kRecordBytes << shift));
    }
    __device__ __forceinline__ float scale(std::int32_t row, std::int32_t group) const {
        return __half2float(scales[static_cast<std::int64_t>(row) * groups + group]);
    }
};

// The partial sums of the lane's row (panel row (lane % 8) / 2) over groups [g0, g1) for the T
// columns of the [T][k] plane x, the lane taking group g0 + 4 s + lane / 8 and record half lane % 2
// at step s: each record is read once and reused across the activation columns.
template <class Format, int T>
__device__ __forceinline__ void panel_partial(const View& v, std::int32_t row, std::int32_t g0,
                                              std::int32_t g1, const __nv_bfloat16* __restrict__ x,
                                              std::int32_t k, int lane, float (&acc)[T]) {
    const int half = lane & 1;
#pragma unroll
    for (int t = 0; t < T; ++t) { acc[t] = 0.0f; }
    std::int32_t g = g0 + (lane >> 3);
#pragma unroll 4
    for (; g < g1; g += 4) {
        const uint4 codes = __ldg(v.record(row, g) + half);
        const float scale = v.scale(row, g);
        const std::int64_t k0 =
            static_cast<std::int64_t>(g) * Format::kGroup + half * (Format::kGroup / 2);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            acc[t] = fmaf(Format::dot(codes, x + static_cast<std::int64_t>(t) * k + k0), scale,
                          acc[t]);
        }
    }
    // Lanes of one row: lane % 8 in {2r, 2r + 1}, any lane / 8.
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 1);
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 8);
        acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 16);
    }
}

// Warp w of the CTA: panel blockIdx.x * (kWarps / KSplit) + w / KSplit, K share w % KSplit. With
// Swiglu the panel's gate rows [4p, 4p + 4) and up rows rows + [4p, 4p + 4) both, out = silu(g) * u
// into the BF16 [T][rows] plane.
template <class Format, int T, int KSplit, bool Swiglu>
__global__ __launch_bounds__(kWarps * 32) void gemv_kernel(View v, std::int32_t panels,
                                                           std::int32_t rows, std::int32_t k,
                                                           const __nv_bfloat16* __restrict__ x,
                                                           DestinationEpilogue epilogue,
                                                           __nv_bfloat16* __restrict__ out) {
    constexpr int kPanelsPerCta = kWarps / KSplit;
    __shared__ float partial[kWarps][Swiglu ? 2 : 1][T][kPanelRows];
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int share        = warp % KSplit;
    const std::int32_t pan = static_cast<std::int32_t>(blockIdx.x) * kPanelsPerCta + warp / KSplit;
    const bool live        = pan < panels;
    const std::int32_t row = pan * kPanelRows + ((lane & 7) >> 1);
    const std::int32_t per = (v.groups + KSplit - 1) / KSplit;
    const std::int32_t g0  = share * per;
    const std::int32_t g1  = min(v.groups, g0 + per);
    float a[T];
    float b[T];
    if (live) {
        panel_partial<Format, T>(v, row, g0, g1, x, k, lane, a);
        if constexpr (Swiglu) {
            panel_partial<Format, T>(v, row + rows, g0, g1, x, k, lane, b);
        }
    }
    const bool writer = live && lane < 8 && (lane & 1) == 0;
    if constexpr (KSplit > 1) {
        if (writer) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                partial[warp][0][t][lane >> 1] = a[t];
                if constexpr (Swiglu) partial[warp][1][t][lane >> 1] = b[t];
            }
        }
        __syncthreads();
        if (share != 0 || !writer) { return; }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            a[t] = 0.0f;
            if constexpr (Swiglu) b[t] = 0.0f;
#pragma unroll
            for (int s = 0; s < KSplit; ++s) {
                a[t] += partial[warp + s][0][t][lane >> 1];
                if constexpr (Swiglu) b[t] += partial[warp + s][1][t][lane >> 1];
            }
        }
    }
    if (!writer) { return; }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        if constexpr (Swiglu) {
            out[static_cast<std::int64_t>(t) * rows + row] = __float2bfloat16_rn(
                silu(a[t] * epilogue.factor(row)) * (b[t] * epilogue.factor(row + rows)));
        } else {
            epilogue(row, t, a[t]);
        }
    }
}

bool supports(const Activation& x, const Weight& w) {
    if (!v100_candidate_route_enabled(V100CandidateRoute::Drafter) || !enabled() ||
        x.tokens < 1 || x.tokens > 4 || w.k != x.k || x.bf16 == nullptr ||
        reinterpret_cast<std::uintptr_t>(x.bf16) % 16 != 0) {
        return false;
    }
    if (w.qtype != QType::Q8_G32_FP16 && w.qtype != QType::Q4_G64_FP16) { return false; }
    if (w.layout != QuantLayout::RowSplit && w.layout != QuantLayout::RowSplitPanel) {
        return false;
    }
    const int group = w.qtype == QType::Q8_G32_FP16 ? 32 : 64;
    return w.n > 0 && w.k > 0 && w.group_size == group && w.scale_dtype == DType::FP16 &&
           w.qhigh == nullptr && w.padded_shape[1] == w.k && w.k % group == 0 &&
           w.n % kPanelRows == 0 && w.qdata != nullptr && w.scales != nullptr &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0;
}

View view_of(const Weight& w) {
    View v;
    v.base   = static_cast<const std::uint8_t*>(w.qdata);
    v.scales = static_cast<const __half*>(w.scales);
    v.groups = w.k / (w.qtype == QType::Q8_G32_FP16 ? 32 : 64);
    v.shift  = row_split_panel_shift(w.layout);
    return v;
}

template <class Format, bool Swiglu>
void launch(const View& v, std::int32_t panels, std::int32_t rows, std::int32_t k,
            std::int32_t tokens, const __nv_bfloat16* x, const DestinationEpilogue& epilogue,
            __nv_bfloat16* out, cudaStream_t stream) {
    // Enough warps to fill 80 SMs: split K when the panels alone fall short.
    const int split = panels >= 4096 ? 1 : panels >= 2048 ? 2 : 4;
    const auto go = [&]<int T, int KSplit>() {
        constexpr int per = kWarps / KSplit;
        const unsigned grid = static_cast<unsigned>((panels + per - 1) / per);
        gemv_kernel<Format, T, KSplit, Swiglu>
            <<<grid, kWarps * 32, 0, stream>>>(v, panels, rows, k, x, epilogue, out);
    };
    const auto with_split = [&]<int T>() {
        if (split == 1) {
            go.template operator()<T, 1>();
        } else if (split == 2) {
            go.template operator()<T, 2>();
        } else {
            go.template operator()<T, 4>();
        }
    };
    switch (tokens) {
    case 1: with_split.template operator()<1>(); break;
    case 2: with_split.template operator()<2>(); break;
    case 3: with_split.template operator()<3>(); break;
    default: with_split.template operator()<4>(); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Swiglu>
void dispatch(const Weight& w, std::int32_t panels, std::int32_t rows, std::int32_t tokens,
              const __nv_bfloat16* x, const DestinationEpilogue& epilogue, __nv_bfloat16* out,
              cudaStream_t stream) {
    const View v = view_of(w);
    if (w.qtype == QType::Q8_G32_FP16) {
        launch<Q8Format, Swiglu>(v, panels, rows, w.k, tokens, x, epilogue, out, stream);
    } else {
        launch<Q4Format, Swiglu>(v, panels, rows, w.k, tokens, x, epilogue, out, stream);
    }
}

} // namespace

bool v100_drafter_gemv_project(const Activation& x, const Weight& w,
                               std::span<const Destination> destinations, cudaStream_t stream) {
    if (!supports(x, w) || destinations.empty() || destinations.size() > kMaxDestinations) {
        return false;
    }
    DestinationEpilogue epilogue;
    epilogue.factor = row_factor(w);
    for (const Destination& d : destinations) { epilogue.set[epilogue.count++] = d; }
    dispatch<false>(w, w.n / kPanelRows, w.n, x.tokens, x.bf16, epilogue, nullptr, stream);
    record_v100_selection(V100ObservedRoute::Drafter);
    record_v100_kernel_route(true);
    return true;
}

bool v100_drafter_gemv_swiglu(const Activation& x, const Weight& gate, const Weight* up,
                              Tensor& out, cudaStream_t stream) {
    const int rows = out.ne[0];
    if (up != nullptr || !supports(x, gate) || out.dtype != DType::BF16 ||
        rows <= 0 || rows % kPanelRows != 0 || rows != gate.n / 2 || out.ne[1] != x.tokens) {
        return false;
    }
    DestinationEpilogue epilogue;
    epilogue.factor = row_factor(gate);
    dispatch<true>(gate, rows / kPanelRows, rows, x.tokens, x.bf16, epilogue,
                   static_cast<__nv_bfloat16*>(out.data), stream);
    record_v100_selection(V100ObservedRoute::Drafter);
    record_v100_kernel_route(true);
    return true;
}

} // namespace ninfer::ops::detail::sm70
#endif
