#pragma once

#include "ops/linear/sm70/sm70_project.h"

namespace ninfer::ops::detail::sm70 {

std::size_t w4a8_workspace_bytes(int k, int tokens);
Activation w4a8_prepare(const Tensor& x, WorkspaceArena& workspace, cudaStream_t stream);
void w4a8_project(const Activation& x, const Weight& w, std::span<const Destination> destinations,
                  cudaStream_t stream);
void w4a8_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail::sm70
