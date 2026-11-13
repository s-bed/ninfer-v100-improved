#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place, load-time permutation from the portable NVFP4 artifact layout to the fragment order
// consumed by Volta QPN2. The payload size is unchanged and no transformed copy is retained.
void nvfp4_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);
// Only the dense MLP parents qualified for the port; never embedding gathers.
bool nvfp4_qpn_prepack_eligible(const WeightGeometry& geometry) noexcept;
void nvfp4_prepack_parent_sm70(const WeightGeometry& geometry, std::byte* data, cudaStream_t stream);

} // namespace ninfer::ops::detail
