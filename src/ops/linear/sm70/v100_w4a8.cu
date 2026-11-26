// Group-of-32 activation quantization and exact 2*E2M1 integer weights, as in
// iamwavecut/ninfer-all 58e3df9f. Volta's DP4A operates on signed bytes.
#include "ops/linear/sm70/v100_w4a8.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/sm70/sm70_epilogue.cuh"

#include <cuda_fp8.h>
#include <stdexcept>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
namespace ninfer::ops::detail::sm70 {
namespace {

constexpr int kWarps = 4;
constexpr unsigned kMask = 0xffffffffU;

// Keep the FP32 scale with its codes. A negative scale marks a group containing
// a non-finite value or a scale below the normal FP32 range. Its dot uses BF16.
__global__ void quantize(const __nv_bfloat16* input, int groups, signed char* codes, float* scales) {
    const int group = blockIdx.x * kWarps + threadIdx.y;
    if (group >= groups) { return; }
    const int lane = threadIdx.x;
    const auto index = (static_cast<std::size_t>(blockIdx.y) * groups + group) * 32 + lane;
    const float value = __bfloat162float(input[index]);
    unsigned maximum_bits = __float_as_uint(value) & 0x7fffffffU;
    const bool exceptional = __any_sync(kMask, !isfinite(value));
    for (int offset = 16; offset; offset >>= 1) {
        maximum_bits = max(maximum_bits, __shfl_xor_sync(kMask, maximum_bits, offset));
    }
    const float maximum = __uint_as_float(maximum_bits);
    const float scale = maximum / 127.0F;
    const bool repair = exceptional || (maximum_bits != 0 && scale < 0x1p-126F);
    if (lane == 0) { scales[index / 32] = repair ? -1.0F : scale; }
    const float scaled = !repair && maximum > 0.0F ? value * (127.0F / maximum) : 0.0F;
    codes[index] = static_cast<signed char>(max(-127, min(127, __float2int_rn(scaled))));
}

__device__ __forceinline__ int twice_e2m1(unsigned nibble) {
    // Select one byte from 0,1,2,3,4,6,8,12. E2M1 has both signs of zero.
    const int magnitude = __byte_perm(0x03020100, 0x0c080604, nibble & 7U) & 255;
    return (nibble & 8U) ? -magnitude : magnitude;
}

// Widen eight FP4 nibbles to two signed-byte DP4A operands in parallel. Adapted
// from e2m1_word in iamwavecut/ninfer-all 58e3df9f. PRMT's sign-replicate
// selector is available in PTX even when __byte_perm masks that selector bit.
__device__ __forceinline__ unsigned sign_bytes(unsigned value) {
    unsigned result;
    asm("prmt.b32 %0, %1, 0, 0xba98;" : "=r"(result) : "r"(value));
    return result;
}

template <bool Packed>
__device__ __forceinline__ void widen_word(unsigned word, int& first, int& second) {
    const unsigned even_signs = sign_bytes(word << 4);
    const unsigned odd_signs = sign_bytes(word);
    const auto four = [&](unsigned selector, unsigned signs) {
        const unsigned positive = __byte_perm(0x03020100U, 0x0c080604U, selector);
        const unsigned negative = __byte_perm(0xfdfeff00U, 0xf4f8fafcU, selector);
        return (positive & ~signs) | (negative & signs);
    };
    const unsigned lo = four(word & 0x7777U, __byte_perm(even_signs, odd_signs, 0x5140));
    const unsigned hi = four((word >> 16) & 0x7777U, __byte_perm(even_signs, odd_signs, 0x7362));
    if constexpr (Packed) {
        // QPN stores four even positions followed by four odd positions.
        first = static_cast<int>(__byte_perm(lo, hi, 0x5140));
        second = static_cast<int>(__byte_perm(lo, hi, 0x7362));
    } else {
        first = static_cast<int>(lo);
        second = static_cast<int>(hi);
    }
}

struct Group {
    uint2 words;
    float scale;
};

template <bool Packed>
__device__ __forceinline__ Group load_group(const Weight& w, int row, int group) {
    const auto* codes = static_cast<const uint2*>(w.qdata);
    const auto* scales = static_cast<const unsigned char*>(w.scales);
    std::size_t data_index, scale_index;
    if constexpr (Packed) {
        const int lane = ((row & 31) / 8) * 4 + (row & 3) + ((row & 4) << 2);
        data_index = (static_cast<std::size_t>(row / 32) * (w.k / 16) + group) * 32 + lane;
        scale_index = data_index;
    } else {
        data_index = static_cast<std::size_t>(row) * (w.k / 16) + group;
        scale_index = (static_cast<std::size_t>(row / 128) * (w.k / 64) + group / 4) * 512 +
                      (row & 31) * 16 + ((row & 127) >> 5) * 4 + (group & 3);
    }
    __nv_fp8_e4m3 scale;
    scale.__x = scales[scale_index];
    return {codes[data_index], float(scale) * 0.5F};
}

template <bool Packed>
__device__ __forceinline__ int integer_weight(uint2 words, int e) {
    const int i = Packed ? ((e & 8) + ((e & 1) << 2) + ((e & 7) >> 1)) : e;
    return twice_e2m1(((i < 8 ? words.x : words.y) >> (4 * (i & 7))) & 15U);
}

template <int Columns, bool Packed>
__device__ __forceinline__ void accumulate_row(const Activation& x, const Weight& w, int row,
                                                int token, float (&sum)[Columns]) {
    const int groups = w.k / 16;
    for (int g = threadIdx.x; g < groups; g += 32) {
        const Group weights = load_group<Packed>(w, row, g);
        int packed[4];
        widen_word<Packed>(weights.words.x, packed[0], packed[1]);
        widen_word<Packed>(weights.words.y, packed[2], packed[3]);
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (token + c >= x.tokens) { break; }
            const auto offset = static_cast<std::size_t>(token + c) * w.k + g * 16;
            const float scale = x.w4a8_scales[offset / 32];
            if (scale < 0.0F) {
                float partial = 0.0F;
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    partial = fmaf(float(integer_weight<Packed>(weights.words, j)) * weights.scale,
                                   __bfloat162float(x.bf16[offset + j]), partial);
                }
                sum[c] += partial;
            } else {
                const int4 bytes = *reinterpret_cast<const int4*>(x.w4a8_codes + offset);
                int dot = __dp4a(bytes.x, packed[0], 0);
                dot = __dp4a(bytes.y, packed[1], dot);
                dot = __dp4a(bytes.z, packed[2], dot);
                dot = __dp4a(bytes.w, packed[3], dot);
                // For the largest finite BF16 inputs the product of the two scales can
                // overflow even when an exact integer cancellation makes the dot zero.
                if (dot != 0 || !isfinite(weights.scale)) {
                    sum[c] = fmaf(float(dot), scale * weights.scale, sum[c]);
                }
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < Columns; ++c) {
        for (int offset = 16; offset; offset >>= 1) {
            sum[c] += __shfl_down_sync(kMask, sum[c], offset);
        }
    }
}

// Weight groups are read once for up to four activation columns. For SwiGLU
// each warp owns both rows and emits only the final BF16 activation.
template <int Columns, bool Packed, bool SwiGLU>
__global__ void multiply(Activation x, Weight gate, Weight up, RowFactor up_factor, int up_first,
                         int rows, DestinationEpilogue epilogue, __nv_bfloat16* output) {
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= rows) { return; }
    const int token = blockIdx.y * Columns;
    float sums[Columns]{}, ups[Columns]{};
    accumulate_row<Columns, Packed>(x, gate, row, token, sums);
    if constexpr (SwiGLU) {
        accumulate_row<Columns, Packed>(x, up, up_first + row, token, ups);
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int c = 0; c < Columns && token + c < x.tokens; ++c) {
            if constexpr (SwiGLU) {
                const float g = sums[c] * epilogue.factor(row);
                const float u = ups[c] * up_factor(up_first + row);
                output[static_cast<std::size_t>(token + c) * rows + row] = __float2bfloat16_rn(silu(g)*u);
            } else {
                epilogue(row, token + c, sums[c]);
            }
        }
    }
}

bool w4a8_eligible_weight(const Activation& x, const Weight& w) {
    return x.w4a8_codes != nullptr && x.w4a8_scales != nullptr && x.bf16 != nullptr &&
           x.tokens >= 1 && x.tokens <= kW4A8MaxTokens && w.qtype == QType::NVFP4 &&
           w.k == x.k && w.n > 0 && w.qdata != nullptr && w.scales != nullptr &&
           (w.layout == QuantLayout::VoltaQpnPrepacked ||
            w.layout == QuantLayout::BlockScaleK16M128x4);
}

template <bool Fused>
void launch(const Activation& x, const Weight& w, const Weight& up, int up_first, int rows,
            DestinationEpilogue epilogue, __nv_bfloat16* output, cudaStream_t stream) {
    const RowFactor factor = row_factor(up);
    const auto issue = [&]<int Columns, bool Packed>() {
        multiply<Columns, Packed, Fused><<<dim3((rows + kWarps-1)/kWarps, (x.tokens + Columns-1)/Columns),
                                          dim3(32,kWarps), 0, stream>>>(
            x, w, up, factor, up_first, rows, epilogue, output);
    };
    const auto schedule = [&]<bool Packed>() {
        if (x.tokens == 1) { issue.template operator()<1, Packed>(); }
        else if (x.tokens == 2) { issue.template operator()<2, Packed>(); }
        else { issue.template operator()<4, Packed>(); }
    };
    if (w.layout == QuantLayout::VoltaQpnPrepacked) { schedule.template operator()<true>(); }
    else { schedule.template operator()<false>(); }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t w4a8_workspace_bytes(int k, int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(static_cast<std::size_t>(tokens) * k);
    (void)layout.alloc_bytes(static_cast<std::size_t>(tokens) * (k / 32) * sizeof(float));
    return layout.peak_bytes(1);
}

Activation w4a8_prepare(const Tensor& x, WorkspaceArena& workspace, cudaStream_t stream) {
    Activation a;
    if (x.dtype != DType::BF16 || x.data == nullptr || !x.is_contiguous() || x.ne[0] <= 0 ||
        x.ne[0] % 64 || x.ne[1] < 1 || x.ne[1] > kW4A8MaxTokens || x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument(
            "Volta W4A8 needs a contiguous BF16 [K, 1..4] matrix with K divisible by 64");
    }
    a.bf16 = static_cast<const __nv_bfloat16*>(x.data);
    a.tokens = x.ne[1];
    a.k = x.ne[0];
    const auto elements = static_cast<std::size_t>(a.tokens) * a.k;
    auto* codes = static_cast<signed char*>(workspace.alloc_bytes(elements).data);
    auto* scales = static_cast<float*>(workspace.alloc_bytes(elements / 32 * sizeof(float)).data);
    quantize<<<dim3((a.k / 32 + kWarps - 1) / kWarps, a.tokens), dim3(32, kWarps), 0, stream>>>(
        a.bf16, a.k / 32, codes, scales);
    CUDA_CHECK(cudaGetLastError());
    a.w4a8_codes = codes;
    a.w4a8_scales = scales;
    return a;
}

bool w4a8_project(const Activation& x, const Weight& w, std::span<const Destination> destinations,
                  cudaStream_t stream) {
    if (!w4a8_eligible_weight(x, w)) { return false; }
    if (destinations.empty() || destinations.size() > kMaxDestinations) {
        throw std::invalid_argument("invalid W4A8 projection destinations");
    }
    DestinationEpilogue epilogue;
    epilogue.factor = row_factor(w);
    for (const auto& d : destinations) { epilogue.set[epilogue.count++] = d; }
    launch<false>(x, w, w, 0, w.n, epilogue, nullptr, stream);
    return true;
}

bool w4a8_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                 cudaStream_t stream) {
    const int rows = out.ne[0];
    const bool joined = up == nullptr;
    if (!w4a8_eligible_weight(x, gate) || out.dtype != DType::BF16 || out.data == nullptr ||
        rows <= 0 || out.ne[1] != x.tokens ||
        (joined ? gate.n != 2 * rows
                : (!w4a8_eligible_weight(x, *up) || up->layout != gate.layout || gate.n != rows ||
                   up->n != rows))) {
        return false;
    }
    DestinationEpilogue epilogue;
    epilogue.factor = row_factor(gate);
    launch<true>(x, gate, joined ? gate : *up, joined ? rows : 0, rows, epilogue,
                 static_cast<__nv_bfloat16*>(out.data), stream);
    return true;
}

} // namespace ninfer::ops::detail::sm70
#endif
