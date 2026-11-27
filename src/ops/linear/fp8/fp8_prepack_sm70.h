#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include <cuda_runtime.h>
#include <cstdint>
namespace ninfer::ops::detail {
constexpr bool fp8_qpn_head_shape(std::uint64_t rows, std::uint64_t columns) noexcept {
    return rows == 248320 && columns == 5120;
}
bool fp8_qpn_prepack_eligible(const WeightGeometry&) noexcept;
void fp8_prepack_qpn_sm70(Weight&,cudaStream_t stream=nullptr);
void fp8_prepack_parent_sm70(const WeightGeometry&,std::byte*,cudaStream_t stream);
}
