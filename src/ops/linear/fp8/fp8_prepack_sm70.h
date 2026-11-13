#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
bool fp8_qpn_prepack_eligible(const WeightGeometry&) noexcept;
void fp8_prepack_qpn_sm70(Weight&,cudaStream_t stream=nullptr);
void fp8_prepack_parent_sm70(const WeightGeometry&,std::byte*,cudaStream_t stream);
}
