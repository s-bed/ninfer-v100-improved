#pragma once
#include "ops/linear/sm70/sm70_project.h"
namespace ninfer::ops::detail::sm70 {
// Donor QPN projection leaves. False means the native route owns this contract.
bool v100_qpn_project(const Activation&, const Weight&, std::span<const Destination>, cudaStream_t);
bool v100_qpn_swiglu(const Activation&, const Weight&, const Weight*, Tensor&, cudaStream_t);
std::size_t v100_qpn_scratch_bytes(QType format,std::int32_t max_rows,std::int32_t k,std::int32_t tokens);
std::size_t v100_qpn_scratch_capacity(QType format,std::int32_t max_rows,std::int32_t k,
                                      std::int32_t first,std::int32_t last);
void v100_qpn_prepare(Activation&,WorkspaceArena&,cudaStream_t,QType format);
}
