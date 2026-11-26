#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t volta_flash_meta_elements(std::int32_t q_heads, std::int32_t tokens);
void volta_flash_mma_launch(const float* q_f32, const half* keys, const half* values,
                           const half* mask, float* output, float2* meta, std::int32_t q_heads,
                           std::int32_t tokens, std::int32_t keys_count, float scale,
                           cudaStream_t stream);

} // namespace ninfer::ops::detail
