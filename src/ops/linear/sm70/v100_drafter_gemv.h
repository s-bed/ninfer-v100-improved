#pragma once

#include "ops/linear/sm70/sm70_project.h"

namespace ninfer::ops::detail::sm70 {

bool v100_drafter_gemv_project(const Activation& x, const Weight& w,
                               std::span<const Destination> destinations, cudaStream_t stream);
bool v100_drafter_gemv_swiglu(const Activation& x, const Weight& gate, const Weight* up,
                              Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail::sm70
