// Donor's production wide NVFP4 route, adapted from geoffwatts/ninfer-v100 b37d0dd3.
// CUTLASS v4.4.2 Sm70 FP16 operands / FP32 accumulators. Output stays FP32 until
// NInfer-all applies per-source row factors, saturation repair and the fused output.
#include "ops/linear/sm70/v100_cutlass.h"
#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.cuh"
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/sm70/sm70_gemm.cuh"
#include "ops/linear/sm70/sm70_small.h"

// The host pass emits launch stubs; a mixed build compiles these leaves only into sm_70.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace ninfer::ops::detail::sm70 {
namespace {
constexpr int kProductColumns=512;
// -1: no route queried yet; 0: CUTLASS; 1: prior MMA candidate, benchmark only.
std::atomic<int> wide_backend{-1};
using OutputOp=cutlass::epilogue::thread::LinearCombination<float,4,float,float>;
using Gemm=cutlass::gemm::device::Gemm<
    cutlass::half_t,cutlass::layout::RowMajor,
    cutlass::half_t,cutlass::layout::ColumnMajor,
    float,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<128,128,32>,
    cutlass::gemm::GemmShape<64,64,32>,
    cutlass::gemm::GemmShape<8,8,4>,OutputOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2>;

Gemm::Arguments sizing_arguments(int n,int k,int tokens) {
    return {{std::min(tokens,kProductColumns),n,k},{nullptr,k},{nullptr,k},
            {nullptr,n},{nullptr,n},{1.0F,0.0F},1};
}

struct Scratch {
    __half* weight=nullptr;
    __half* activation=nullptr;
    float* product=nullptr;
    void* gemm=nullptr;
};

template <class Allocator>
Scratch allocate(Allocator& ws,int n,int k,int tokens) {
    Scratch result;
    result.weight=static_cast<__half*>(ws.alloc_bytes(std::size_t(n)*k*sizeof(__half)).data);
    // The generic small-T staging is P8-permuted, so T=33..64 needs a natural plane.
    // At wider T the generic route has already prepared a reusable natural FP16 plane.
    if (small_takes(tokens))
        result.activation=static_cast<__half*>(ws.alloc_bytes(activation_bytes(k,tokens)).data);
    result.product=static_cast<float*>(ws.alloc_bytes(
        std::size_t(n)*std::min(tokens,kProductColumns)*sizeof(float)).data);
    const auto bytes=Gemm::get_workspace_size(sizing_arguments(n,k,tokens));
    if (bytes) result.gemm=ws.alloc_bytes(bytes).data;
    return result;
}

bool accepts(const Activation& x,const Weight& w) {
    return v100_ported_kernels_enabled() && enabled() && v100_cutlass_wide_enabled() &&
           w.qtype==QType::NVFP4 && v100_cutlass_shape(w.n,w.k,x.tokens) &&
           dequantizable(w) && x.qpn_beyond && x.cutlass_storage &&
           x.cutlass_bytes>=v100_cutlass_workspace_bytes(w.n,w.k,x.tokens);
}

template <class Consume>
void run(const Activation& x,const Weight& w,cudaStream_t stream,Consume&& consume) {
    // Borrowed arena: no cudaMalloc, persistent storage or host/device allocation on execution.
    WorkspaceArena arena(DeviceSpan{x.cutlass_storage,x.cutlass_bytes});
    const Scratch scratch=allocate(arena,w.n,w.k,x.tokens);
    dequantize_rows(w,0,w.n,scratch.weight,stream); // once across every T tile
    const __half* activation=x.data;
    if (scratch.activation) {
        convert_activation(x.bf16,x.k,x.tokens,scratch.activation,stream);
        activation=scratch.activation;
    }
    if (!activation) throw std::logic_error("V100 CUTLASS input has no natural FP16 plane");
    for (int first=0;first<x.tokens;first+=kProductColumns) {
        const int columns=std::min(kProductColumns,x.tokens-first);
        Gemm operation;
        Gemm::Arguments arguments{
            {columns,w.n,w.k},
            {reinterpret_cast<const cutlass::half_t*>(activation+std::size_t(first)*w.k),w.k},
            {reinterpret_cast<const cutlass::half_t*>(scratch.weight),w.k},
            {scratch.product,w.n},{scratch.product,w.n},{1.0F,0.0F},1};
        if (operation.can_implement(arguments)!=cutlass::Status::kSuccess)
            throw std::runtime_error("V100 CUTLASS cannot implement qualified NVFP4 shape");
        if (operation.initialize(arguments,scratch.gemm,stream)!=cutlass::Status::kSuccess)
            throw std::runtime_error("V100 CUTLASS initialize failed");
        if (operation(stream)!=cutlass::Status::kSuccess)
            throw std::runtime_error("V100 CUTLASS GEMM failed");
        CUDA_CHECK(cudaGetLastError());
        consume(scratch.product,scratch.weight,first,columns);
    }
}

__global__ void fused_output(const float* plane,const __half* weights,
                             Saturation saturation,int n,int columns,RowFactor factor,
                             __nv_bfloat16* out) {
    const std::int64_t i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int rows=n/2;
    if (i>=std::int64_t(rows)*columns) return;
    const int t=int(i/rows),r=int(i%rows);
    const float gate=saturated_product(plane,weights,saturation,n,r,t)*factor(r);
    const float up=saturated_product(plane,weights,saturation,n,r+rows,t)*factor(r+rows);
    out[i]=__float2bfloat16_rn(silu(gate)*up);
}
}

void set_v100_wide_backend_mma_for_benchmark(bool mma) {
    int expected=-1;
    const int requested=mma?1:0;
    if (!wide_backend.compare_exchange_strong(expected,requested) && expected!=requested)
        throw std::logic_error("V100 wide backend is fixed before its first query");
}

bool v100_cutlass_wide_enabled() {
    int expected=-1;
    wide_backend.compare_exchange_strong(expected,0);
    return wide_backend.load()==0;
}

bool v100_cutlass_shape(int n,int k,int tokens) {
    // Event measurements on the actual model shapes favor segmented QPN at
    // T33/48, CUTLASS at T64/65/96/128, and native packed down products at
    // T2048/4096. The up product has no established packed-relative wide gain
    // either. Keep the unqualified wide MMA candidate out of production.
    return tokens>=64 && tokens<2048 &&
           ((n==34816 && k==5120) || (n==5120 && k==17408));
}

std::size_t v100_cutlass_workspace_bytes(int n,int k,int tokens) {
    if (!v100_cutlass_shape(n,k,tokens)) return 0;
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout,n,k,tokens);
    return layout.peak_bytes(1);
}

bool v100_cutlass_project(const Activation& x,const Weight& w,
                          std::span<const Destination> destinations,cudaStream_t stream) {
    if (!accepts(x,w) || destinations.empty() || destinations.size()>kMaxDestinations) return false;
    run(x,w,stream,[&](const float* product,const __half* weights,int first,int columns) {
        DestinationEpilogue output;
        output.factor=row_factor(w);
        for (auto d:destinations) {
            const auto offset=std::size_t(first)*d.leading;
            if (d.fp32) d.fp32+=offset; else d.data+=offset;
            output.set[output.count++]=d;
        }
        const Saturation saturation{x.bf16+std::size_t(first)*w.k,x.qpn_beyond+first,w.k};
        const dim3 grid(unsigned((w.n+255)/256),unsigned(columns));
        epilogue_kernel<<<grid,256,0,stream>>>(product,weights,saturation,w.n,columns,0,output);
        CUDA_CHECK(cudaGetLastError());
    });
    return true;
}

bool v100_cutlass_swiglu(const Activation& x,const Weight& gate,const Weight* up,
                         Tensor& out,cudaStream_t stream) {
    if (up || !accepts(x,gate) || gate.n!=34816 || out.dtype!=DType::BF16 ||
        out.ne[0]!=gate.n/2 || out.ne[1]!=x.tokens) return false;
    run(x,gate,stream,[&](const float* product,const __half* weights,int first,int columns) {
        const Saturation saturation{x.bf16+std::size_t(first)*gate.k,x.qpn_beyond+first,gate.k};
        const std::int64_t elements=std::int64_t(gate.n/2)*columns;
        fused_output<<<unsigned((elements+255)/256),256,0,stream>>>(
            product,weights,saturation,gate.n,columns,row_factor(gate),
            static_cast<__nv_bfloat16*>(out.data)+std::size_t(first)*(gate.n/2));
        CUDA_CHECK(cudaGetLastError());
    });
    return true;
}
}

#endif // host pass or sm_70 device image
