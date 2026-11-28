#pragma once

#include "ops/common/memory.cuh"

namespace ninfer::ops {
// The same copies with a "memory" clobber, for kernels that stage with raw asm and rely on the
// compiler not moving shared-memory accesses across the commit/wait. These keep the sm_80+ code
// those kernels already had; on Volta they are the synchronous emulation (see cp_async).
template <int Bytes, Cache Policy = Cache::cg>
__device__ __forceinline__ void cp_async_mem(void* smem_dst, const void* gmem_src) {
#if defined(NINFER_PRE_SM80_DEVICE)
    cp_async<Bytes, Policy>(smem_dst, gmem_src);
#else
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src)
                     : "memory");
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes)
                     : "memory");
    }
#endif
}

// Copies src_bytes (0..Bytes) and zero-fills the rest; the source is not read when src_bytes is 0.
template <int Bytes, Cache Policy = Cache::cg>
__device__ __forceinline__ void cp_async_zfill_mem(void* smem_dst, const void* gmem_src,
                                                   int src_bytes) {
#if defined(NINFER_PRE_SM80_DEVICE)
    cp_async_zfill<Bytes, Policy>(smem_dst, gmem_src, src_bytes);
#else
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes)
                     : "memory");
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes)
                     : "memory");
    }
#endif
}

__device__ __forceinline__ void cp_commit_mem() {
#if !defined(NINFER_PRE_SM80_DEVICE)
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait_mem() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if !defined(NINFER_PRE_SM80_DEVICE)
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups) : "memory");
#endif
}

} // namespace ninfer::ops
