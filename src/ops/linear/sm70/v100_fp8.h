#pragma once

#include "ops/linear/sm70/sm70_project.h"

namespace ninfer::ops::detail::sm70 {

// Native RowScale operands only. Embedding bytes and all parent geometry stay intact.
bool v100_fp8_enabled();
std::size_t v100_fp8_scratch_bytes(QType,int,int,int,ProjectionPurpose);
std::size_t v100_fp8_scratch_capacity(QType,int,int,int,int,ProjectionPurpose);
void v100_fp8_prepare(Activation&,WorkspaceArena&,cudaStream_t,QType);
bool v100_fp8_project(const Activation&,const Weight&,std::span<const Destination>,cudaStream_t);
bool v100_fp8_swiglu(const Activation&,const Weight&,const Weight*,Tensor&,cudaStream_t);

bool v100_fp8_cutlass_shape(int,int,int);
std::size_t v100_fp8_cutlass_workspace_bytes(int,int,int);
bool v100_fp8_cutlass_project(const Activation&,const Weight&,std::span<const Destination>,cudaStream_t);

} // namespace ninfer::ops::detail::sm70
