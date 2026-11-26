#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kVoltaFlashQueryBlock = 1024;
inline constexpr std::int32_t kVoltaFlashMaximumKeys = 200000;

// Runtime policy and geometry shared by startup workspace planning and normal dispatch.
[[nodiscard]] bool volta_flash_prompt_enabled(std::int32_t q_heads, std::int32_t kv_heads,
                                              KvCacheStorage storage, std::int32_t batch);

[[nodiscard]] std::size_t volta_flash_prompt_workspace_bytes(std::int32_t q_heads,
                                                            std::int32_t kv_heads,
                                                            std::int32_t visible_keys);

// The visible extent is exact, every query is live, and the view covers the whole cache row.
// Returns a stream-ordered device flag. On out-of-range BF16 operands the caller runs the native
// prompt kernel with this flag so its output replaces the internal FP16 result.
[[nodiscard]] const std::uint32_t* volta_flash_prompt_launch(
    const Tensor& q, const Tensor& positions, const Tensor* table_rows, float scale,
    std::int32_t visible_keys, const PagedKVBatchLayerView& cache, WorkspaceArena& workspace,
    Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
