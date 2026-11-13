// The V100 opt-in implementation is absent from builds that carry no sm_70 image.
// These keep shared native callers/link interfaces usable without CUTLASS headers.
#include "ops/linear/sm70/v100_qpn.h"
#include "ops/linear/sm70/v100_cutlass.h"
#include "ops/linear/sm70/v100_fp8.h"
#include <stdexcept>

namespace ninfer::ops::detail::sm70 {
void set_v100_wide_backend_mma_for_benchmark(bool mma) {
    if (mma) throw std::invalid_argument("V100 MMA benchmark needs an sm_70 build");
}
bool v100_cutlass_wide_enabled() { return false; }
bool v100_cutlass_shape(std::int32_t,std::int32_t,std::int32_t) { return false; }
std::size_t v100_cutlass_workspace_bytes(std::int32_t,std::int32_t,std::int32_t) { return 0; }
bool v100_cutlass_project(const Activation&,const Weight&,std::span<const Destination>,cudaStream_t) {
    return false;
}
bool v100_cutlass_swiglu(const Activation&,const Weight&,const Weight*,Tensor&,cudaStream_t) {
    return false;
}
std::size_t v100_qpn_scratch_bytes(QType,std::int32_t,std::int32_t,std::int32_t) { return 0; }
std::size_t v100_qpn_scratch_capacity(QType,std::int32_t,std::int32_t,std::int32_t,std::int32_t) {
    return 0;
}
void v100_qpn_prepare(Activation&,WorkspaceArena&,cudaStream_t,QType) {}
bool v100_qpn_project(const Activation&,const Weight&,std::span<const Destination>,cudaStream_t) {
    return false;
}
bool v100_qpn_swiglu(const Activation&,const Weight&,const Weight*,Tensor&,cudaStream_t) {
    return false;
}
bool v100_fp8_enabled() { return false; }
std::size_t v100_fp8_scratch_bytes(QType,int,int,int,ProjectionPurpose) { return 0; }
std::size_t v100_fp8_scratch_capacity(QType,int,int,int,int,ProjectionPurpose) { return 0; }
void v100_fp8_prepare(Activation&,WorkspaceArena&,cudaStream_t,QType) {}
bool v100_fp8_project(const Activation&,const Weight&,std::span<const Destination>,cudaStream_t) { return false; }
bool v100_fp8_swiglu(const Activation&,const Weight&,const Weight*,Tensor&,cudaStream_t) { return false; }
bool v100_fp8_cutlass_shape(int,int,int) { return false; }
std::size_t v100_fp8_cutlass_workspace_bytes(int,int,int) { return 0; }
bool v100_fp8_cutlass_project(const Activation&,const Weight&,std::span<const Destination>,cudaStream_t) { return false; }
} // namespace ninfer::ops::detail::sm70
