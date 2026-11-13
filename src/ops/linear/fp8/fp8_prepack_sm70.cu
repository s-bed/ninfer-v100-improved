#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/fp8/fp8_format.h"
#include "core/arena.h"
#include "core/device.h"
#include <stdexcept>
namespace ninfer::ops::detail {
namespace {
__global__ void prepack(const std::uint8_t* input,std::uint8_t* output,int n,int k) {
    const std::int64_t i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int blocks=k/128;
    if(i>=std::int64_t(n/32)*blocks*8*32)return;
    const int lane=int(i%32),slice=int((i/32)%8),block=int((i/256)%blocks);
    const int tile=int(i/(std::int64_t(blocks)*256));
    const int row=tile*32+((lane>>2)&3)*8+(lane&3)+(lane&16?4:0);
    reinterpret_cast<uint4*>(output)[i]=*reinterpret_cast<const uint4*>(input+std::int64_t(row)*k+block*128+slice*16);
}
}
bool fp8_qpn_prepack_eligible(const WeightGeometry& g) noexcept {
    return g.format==QType::FP8_E4M3FN_ROW_BF16 && g.layout==QuantLayout::RowScale &&
        g.shape.size()==2 && ((g.shape[0]==16384 && g.shape[1]==5120) ||
        (g.shape[0]==14336 && g.shape[1]==5120) || (g.shape[0]==5120 && g.shape[1]==6144) ||
        (g.shape[0]==34816 && g.shape[1]==5120) || (g.shape[0]==5120 && g.shape[1]==17408));
}
void fp8_prepack_qpn_sm70(Weight& w,cudaStream_t stream) {
    if(w.layout!=QuantLayout::RowScale || w.n%32 || w.k%128)
        throw std::invalid_argument("FP8 QPN prepack requires native N32 K128 parent");
    const auto g=validate_fp8_weight(w,"FP8 QPN prepack");
    DeviceBuffer scratch(std::size_t(g.code_plane_bytes));
    const std::int64_t tuples=std::int64_t(w.n)*w.k/16;
    prepack<<<unsigned((tuples+255)/256),256,0,stream>>>(static_cast<const std::uint8_t*>(w.qdata),static_cast<std::uint8_t*>(scratch.p),w.n,w.k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(w.qdata),scratch.p,std::size_t(g.code_plane_bytes),cudaMemcpyDeviceToDevice,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    w.layout=QuantLayout::VoltaQpnPrepacked;
}
void fp8_prepack_parent_sm70(const WeightGeometry& g,std::byte* data,cudaStream_t stream) {
    if(!fp8_qpn_prepack_eligible(g))throw std::invalid_argument("unsupported FP8 QPN parent");
    WeightParent parent{g,data,1.0F};
    WeightView view{g.shape,{{&parent,0,g.elements}}};
    auto w=native_weight(view,1.0F);
    fp8_prepack_qpn_sm70(w,stream);
}
}
