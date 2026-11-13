#pragma once
#include "ops/linear/sm70/sm70_project.h"
namespace ninfer::ops::detail::sm70 {
// The server always uses CUTLASS in ported mode. The standalone operator benchmark may
// freeze the previous MMA implementation before any workspace query or execution.
void set_v100_wide_backend_mma_for_benchmark(bool mma);
bool v100_cutlass_wide_enabled();
bool v100_cutlass_shape(std::int32_t n,std::int32_t k,std::int32_t tokens);
std::size_t v100_cutlass_workspace_bytes(std::int32_t n,std::int32_t k,std::int32_t tokens);
bool v100_cutlass_project(const Activation&,const Weight&,std::span<const Destination>,cudaStream_t);
bool v100_cutlass_swiglu(const Activation&,const Weight&,const Weight*,Tensor&,cudaStream_t);
}
