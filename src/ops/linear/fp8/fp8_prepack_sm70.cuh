#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace ninfer::ops::detail {
// QType owns the code width and scale geometry of the device-only QPN family.
__host__ __device__ inline std::int64_t fp8_qpn_prepacked_offset(int row,int column,int k) {
    const int lane=((row&31)/8)*4+(row&3)+((row&4)<<2);
    return (((std::int64_t(row/32)*(k/128)+column/128)*8+(column%128)/16)*32+lane)*16+column%16;
}
}
