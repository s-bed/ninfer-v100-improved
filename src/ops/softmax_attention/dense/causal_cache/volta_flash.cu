// Adapted from JimmyMax/ninfer-v100 ec1d53a9 (Apache-2.0). See third_party/ggml-quants.
#include "ops/softmax_attention/dense/causal_cache/volta_flash.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/device_tier.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/softmax_attention/dense/causal_cache/volta_flash_bridge.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim = 256;
constexpr int kKeyTile = 256;

constexpr int padded_keys(int keys) {
    return ((keys + kKeyTile - 1) / kKeyTile) * kKeyTile;
}

struct Scratch {
    Tensor keys;
    Tensor values;
    Tensor mask;
    Tensor query;
    Tensor output;
    float2* stream_partials;
    std::uint32_t* native_required;
};

template <class Allocator>
Scratch allocate(Allocator& arena, int q_heads, int kv_heads, int visible_keys) {
    const int keys = padded_keys(visible_keys);
    const std::size_t partials = volta_flash_meta_elements(q_heads);
    return {
        arena.alloc(DType::FP16, {kHeadDim, kv_heads, keys}),
        arena.alloc(DType::FP16, {kHeadDim, kv_heads, keys}),
        arena.alloc(DType::FP16, {keys, kVoltaFlashQueryBlock}),
        arena.alloc(DType::FP32, {kHeadDim, q_heads, kVoltaFlashQueryBlock}),
        arena.alloc(DType::FP32, {kHeadDim, q_heads, kVoltaFlashQueryBlock}),
        static_cast<float2*>(arena.alloc_bytes(partials * sizeof(float2)).data),
        static_cast<std::uint32_t*>(arena.alloc_bytes(sizeof(std::uint32_t)).data),
    };
}

__device__ bool outside_fp16(float value) {
    return !(fabsf(value) <= 65504.0f);
}

__device__ void require_native(bool out_of_range, std::uint32_t* native_required) {
    if (__any_sync(0xffffffffu, out_of_range) && (threadIdx.x % 32) == 0) {
        atomicExch(native_required, 1u);
    }
}

// The bridge takes contiguous key-major rows: [key][head][dimension]. The paged K and V planes
// both hold BF16 bits. Padding is zero even where the causal mask later drops it.
template <int KVHeads>
__global__ void gather_kv(const __nv_bfloat16* keys, const __nv_bfloat16* values,
                          const std::int32_t* tables, const std::int32_t* table_rows,
                          int table_stride, int visible_keys, half* gathered_keys,
                          half* gathered_values, std::uint32_t* native_required) {
    const int key = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int dimension = static_cast<int>(threadIdx.x);
    const std::int64_t dst =
        (static_cast<std::int64_t>(key) * KVHeads + head) * kHeadDim + dimension;
    if (key >= visible_keys) {
        gathered_keys[dst] = __float2half_rn(0.0f);
        gathered_values[dst] = __float2half_rn(0.0f);
        return;
    }

    const int row = table_rows == nullptr ? 0 : table_rows[0];
    const auto* table = tables + static_cast<std::int64_t>(row) * table_stride;
    const std::int64_t src =
        paged_kv_element_offset<kHeadDim, KVHeads>(table, head, key, dimension);
    const float k = __bfloat162float(keys[src]);
    const float v = __bfloat162float(values[src]);
    const bool k_outside = outside_fp16(k);
    const bool v_outside = outside_fp16(v);
    require_native(k_outside || v_outside, native_required);
    gathered_keys[dst] = __float2half_rn(k_outside ? 0.0f : k);
    gathered_values[dst] = __float2half_rn(v_outside ? 0.0f : v);
}

__global__ void convert_query(const __nv_bfloat16* query, float* staged_query,
                              std::uint32_t* native_required) {
    // Query extents are multiples of the fixed 256-element head dimension.
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float value = __bfloat162float(query[index]);
    const bool outside = outside_fp16(value);
    require_native(outside, native_required);
    staged_query[index] = outside ? 0.0f : value;
}

__global__ void causal_mask(const std::int32_t* positions, int tokens, half* mask, int keys) {
    const int row = static_cast<int>(blockIdx.x);
    const int last = row < tokens ? positions[row] : -1;
    for (int key = threadIdx.x; key < keys; key += blockDim.x) {
        mask[static_cast<std::int64_t>(row) * keys + key] =
            __float2half_rn(key <= last ? 0.0f : -CUDART_INF_F);
    }
}

__global__ void convert_output(const float* staged, __nv_bfloat16* out) {
    const std::int64_t shard_row =
        (static_cast<std::int64_t>(threadIdx.x / kVoltaFlashValueShard) * gridDim.x +
         blockIdx.x) * kVoltaFlashValueShard;
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    out[index] = __float2bfloat16_rn(staged[shard_row + threadIdx.x % kVoltaFlashValueShard]);
}

template <int KVHeads>
void gather(const PagedKVBatchLayerView& cache, const std::int32_t* table_rows, int visible_keys,
            const Scratch& scratch, cudaStream_t stream) {
    gather_kv<KVHeads><<<dim3(padded_keys(visible_keys), KVHeads), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(cache.k_pages.data),
        static_cast<const __nv_bfloat16*>(cache.v_pages.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), table_rows,
        cache.block_tables.ne[0], visible_keys, static_cast<half*>(scratch.keys.data),
        static_cast<half*>(scratch.values.data), scratch.native_required);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool volta_flash_prompt_enabled(std::int32_t q_heads, std::int32_t kv_heads,
                               KvCacheStorage storage, std::int32_t batch) {
    return batch == 1 && storage == KvCacheStorage::BFloat16 &&
           ((q_heads == CausalD256H24Kv4::ModelQHeads && kv_heads == CausalD256H24Kv4::ModelKVHeads) ||
            (q_heads == CausalD256H16Kv2::ModelQHeads && kv_heads == CausalD256H16Kv2::ModelKVHeads)) &&
           v100_candidate_route_enabled(V100CandidateRoute::Flash) && current_device_volta();
}

std::size_t volta_flash_prompt_workspace_bytes(std::int32_t q_heads, std::int32_t kv_heads,
                                                std::int32_t visible_keys) {
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, q_heads, kv_heads, visible_keys);
    return layout.peak_bytes(1);
}

const std::uint32_t* volta_flash_prompt_launch(
    const Tensor& q, const Tensor& positions, const Tensor* table_rows, float scale,
    std::int32_t visible_keys, const PagedKVBatchLayerView& cache, WorkspaceArena& workspace,
    Tensor& out, cudaStream_t stream) {
    const int q_heads = q.ne[1];
    const int width = q.ne[2];
    const int kv_heads = cache.num_kv_heads;
    record_v100_selection(V100ObservedRoute::Flash);
    record_v100_kernel_route(true);
    const Scratch scratch = allocate(workspace, q_heads, kv_heads, visible_keys);
    const auto* rows = table_rows == nullptr ? nullptr
                                            : static_cast<const std::int32_t*>(table_rows->data);
    CUDA_CHECK(cudaMemsetAsync(scratch.native_required, 0, sizeof(std::uint32_t), stream));
    with_causal_geometry(q_heads, kv_heads, [&]<typename Geometry>() {
        gather<Geometry::ModelKVHeads>(cache, rows, visible_keys, scratch, stream);
    });

    for (int begin = 0; begin < width; begin += kVoltaFlashQueryBlock) {
        const int tokens = std::min(width - begin, kVoltaFlashQueryBlock);
        // Live positions are sequential with max(p + 1) <= visible_keys, so queries in
        // [begin, begin + tokens) never attend past visible_keys - width + begin + tokens.
        const int block_keys = padded_keys(visible_keys - width + begin + tokens);
        const int rows = ((tokens + 7) / 8) * 8;
        const auto offset = static_cast<std::int64_t>(begin) * q_heads * kHeadDim;
        const int blocks = tokens * q_heads;
        auto* query = static_cast<float*>(scratch.query.data);
        auto* output = static_cast<float*>(scratch.output.data);
        auto* mask = static_cast<half*>(scratch.mask.data);
        convert_query<<<blocks, kHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data) + offset, query, scratch.native_required);
        causal_mask<<<rows, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data) + begin, tokens, mask, block_keys);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(output, 0, std::size_t(blocks) * kHeadDim * sizeof(float),
                                   stream));
        volta_flash_mma_launch(query, static_cast<const half*>(scratch.keys.data),
                               static_cast<const half*>(scratch.values.data), mask, output,
                               scratch.stream_partials, q_heads, tokens, block_keys, scale, stream);
        convert_output<<<blocks, kHeadDim, 0, stream>>>(
            output, static_cast<__nv_bfloat16*>(out.data) + offset);
    }
    CUDA_CHECK(cudaGetLastError());
    return scratch.native_required;
}

} // namespace ninfer::ops::detail
#endif
