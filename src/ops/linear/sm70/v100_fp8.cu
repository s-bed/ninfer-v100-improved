// FP8 QPN from donor b37d0dd3 with current FP32 RowFactor/output contracts.
#include "ops/linear/sm70/v100_fp8.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/common/math.cuh"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/fp8/fp8_prepack_sm70.cuh"
#include "ops/linear/sm70/v100_fp8_qpn.cuh"
#include <cuda_fp8.h>
#include <algorithm>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
namespace ninfer::ops::detail::sm70 {
namespace {
constexpr int kQpnColumns=32;
bool packed_shape(int rows,int k,int tokens) {
    return tokens>=1 && tokens<=kQpnColumns &&
        ((rows==16384 && k==5120) || (rows==14336 && k==5120) ||
         (rows==5120 && k==6144) || (rows==34816 && k==5120) ||
         (rows==5120 && k==17408));
}
bool raw_swiglu(int rows,int k,int tokens,ProjectionPurpose purpose) {
    return purpose==ProjectionPurpose::SwiGLU && rows==34816 && k==5120 && tokens>=1 && tokens<=8;
}
bool wide_gdn(int rows,int k,int tokens,ProjectionPurpose purpose) {
    return purpose==ProjectionPurpose::GdnInput && v100_fp8_cutlass_shape(rows,k,tokens);
}
bool sums_needed(int rows,int k,ProjectionPurpose purpose) {
    return purpose==ProjectionPurpose::SwiGLU && rows==34816 && k==5120;
}
std::size_t aligned(std::size_t n){return(n+255)/256*256;}
__global__ void column_flags(const __nv_bfloat16* x,int k,int tokens,std::uint32_t* flags) {
    const int t=int(blockIdx.x);bool beyond=false;
    for(int e=int(threadIdx.x);e<k;e+=int(blockDim.x)) {
        const float value=__bfloat162float(x[std::int64_t(t)*k+e]);
        beyond|=!isfinite(value)||fabsf(value)>65504.0F;
    }
    if(__any_sync(0xffffffffU,beyond) && (threadIdx.x&31)==0)atomicOr(flags+t,1U);
}
struct Fp8Output {
    DestinationEpilogue destination;
    Weight weight;
    const __nv_bfloat16* activation;
    const std::uint32_t* beyond;
    int first;
    __device__ void store(int row,int token,float value)const {
        const int column=first+token;
        if(beyond[column]) {
            const auto* codes=static_cast<const std::uint8_t*>(weight.qdata);
            float correction=0.0F;
            for(int e=0;e<weight.k;++e) {
                const float delta=saturated_excess(__bfloat162float(activation[std::int64_t(column)*weight.k+e]));
                if(delta!=0.0F) {
                    const auto at=weight.layout==QuantLayout::VoltaQpnPrepacked
                        ? fp8_qpn_prepacked_offset(row,e,weight.k):std::int64_t(row)*weight.k+e;
                    __nv_fp8_e4m3 code;code.__x=codes[at];
                    correction=fmaf(float(code),delta,correction);
                }
            }
            value+=correction;
        }
        destination(row,column,value);
    }
};
bool eligible(const Activation& x,const Weight& w) {
    return v100_fp8_enabled() && w.qtype==QType::FP8_E4M3FN_ROW_BF16 &&
        (w.layout==QuantLayout::RowScale || w.layout==QuantLayout::VoltaQpnPrepacked) &&
        w.n>0 && w.n%32==0 && w.k==x.k && w.k>=1024 && w.k%128==0 &&
        x.tokens>=1 && x.qpn_beyond && w.qdata && w.scales;
}
void project_qpn(const Activation& x,const Weight& w,std::span<const Destination> destinations,cudaStream_t stream) {
    DestinationEpilogue output;output.factor=row_factor(w);
    for(auto destination:destinations)output.set[output.count++]=destination;
    for(int first=0;first<x.tokens;first+=kQpnColumns) {
        const int count=std::min(kQpnColumns,x.tokens-first);
        Tensor input(const_cast<__nv_bfloat16*>(x.bf16+std::int64_t(first)*x.k),DType::BF16,{x.k,count});
        const Fp8Output store{output,w,x.bf16,x.qpn_beyond,first};
        if(w.layout==QuantLayout::VoltaQpnPrepacked && x.qpn_f16)
            launch_fp8_volta_qpn_with_fp16_activation(input,w,x.qpn_f16+std::int64_t(first)*x.k,store,w.n,stream);
        else launch_fp8_volta_qpn_with_output(input,w,store,w.n,stream);
    }
}
__global__ void combine(const float* sums,int rows,int tokens,__nv_bfloat16* out) {
    const std::int64_t i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=std::int64_t(rows)*tokens)return;
    const int t=int(i/rows),row=int(i%rows);
    const float gate=sums[std::int64_t(t)*2*rows+row],up=sums[std::int64_t(t)*2*rows+rows+row];
    out[i]=__float2bfloat16_rn(silu(gate)*up);
}
}
bool v100_fp8_enabled(){return v100_ported_kernels_enabled()&&enabled();}
std::size_t v100_fp8_scratch_bytes(QType format,int rows,int k,int tokens,ProjectionPurpose purpose) {
    if(format!=QType::FP8_E4M3FN_ROW_BF16 || !v100_fp8_enabled())return 0;
    // Public capacity queries lack the actual layout: reserve the union, while
    // preparation below stages natural FP16 only for an actual packed parent.
    if(packed_shape(rows,k,tokens))return activation_bytes(k,tokens)+
        (sums_needed(rows,k,purpose)?aligned(std::size_t(rows)*tokens*sizeof(float)):0);
    if(wide_gdn(rows,k,tokens,purpose))return aligned(std::size_t(tokens)*4)+
        v100_fp8_cutlass_workspace_bytes(rows,k,tokens);
    return 0;
}
std::size_t v100_fp8_scratch_capacity(QType format,int rows,int k,int first,int last,ProjectionPurpose purpose) {
    std::size_t bytes=0;
    const auto consider=[&](int t){if(t>=first&&t<=last)bytes=std::max(bytes,v100_fp8_scratch_bytes(format,rows,k,t,purpose));};
    consider(first);consider(last);
    for(int t:{8,9,32,33,cublas_min_tokens()-1,cublas_min_tokens(),512,4096,4097,kWideColumns})consider(t);
    return bytes;
}
void v100_fp8_prepare(Activation& x,WorkspaceArena& ws,cudaStream_t stream,QType format) {
    if(format!=QType::FP8_E4M3FN_ROW_BF16 || !v100_fp8_enabled())return;
    const bool packed=x.source_layout==QuantLayout::VoltaQpnPrepacked && packed_shape(x.max_rows,x.k,x.tokens);
    const bool raw=raw_swiglu(x.max_rows,x.k,x.tokens,x.purpose);
    const bool wide=wide_gdn(x.max_rows,x.k,x.tokens,x.purpose);
    if(!packed && !raw && !wide)return;
    if(packed) {
        auto* natural=static_cast<__half*>(ws.alloc_bytes(activation_bytes(x.k,x.tokens)).data);
        convert_activation(x.bf16,x.k,x.tokens,natural,stream);
        x.qpn_f16=natural;x.qpn_beyond=beyond_columns(natural,x.k,x.tokens);
    }else {
        auto* flags=static_cast<std::uint32_t*>(ws.alloc_bytes(aligned(std::size_t(x.tokens)*4)).data);
        CUDA_CHECK(cudaMemsetAsync(flags,0,std::size_t(x.tokens)*4,stream));
        column_flags<<<x.tokens,128,0,stream>>>(x.bf16,x.k,x.tokens,flags);CUDA_CHECK(cudaGetLastError());
        x.qpn_beyond=flags;
    }
    if((packed||raw)&&sums_needed(x.max_rows,x.k,x.purpose))
        x.qpn_sums=static_cast<float*>(ws.alloc_bytes(aligned(std::size_t(x.max_rows)*x.tokens*4)).data);
    x.cutlass_bytes=wide?v100_fp8_cutlass_workspace_bytes(x.max_rows,x.k,x.tokens):0;
    if(x.cutlass_bytes)x.cutlass_storage=ws.alloc_bytes(x.cutlass_bytes).data;
}
bool v100_fp8_project(const Activation& x,const Weight& w,std::span<const Destination> destinations,cudaStream_t stream) {
    if(destinations.empty()||destinations.size()>kMaxDestinations)return false;
    if(v100_fp8_cutlass_project(x,w,destinations,stream)){record_v100_kernel_route(true);return true;}
    if(!eligible(x,w) || w.layout!=QuantLayout::VoltaQpnPrepacked || !x.qpn_f16 || !packed_shape(w.n,w.k,x.tokens))return false;
    project_qpn(x,w,destinations,stream);record_v100_kernel_route(true);return true;
}
bool v100_fp8_swiglu(const Activation& x,const Weight& gate,const Weight* up,Tensor& out,cudaStream_t stream) {
    const bool packed=gate.layout==QuantLayout::VoltaQpnPrepacked && x.qpn_f16 && packed_shape(gate.n,gate.k,x.tokens);
    const bool raw=gate.layout==QuantLayout::RowScale && raw_swiglu(gate.n,gate.k,x.tokens,x.purpose);
    if(up||(!packed&&!raw)||!eligible(x,gate)||!x.qpn_sums||gate.n!=34816||
       out.dtype!=DType::BF16||out.ne[0]!=gate.n/2||out.ne[1]!=x.tokens)return false;
    const Destination destination{nullptr,gate.n,0,0,gate.n,false,x.qpn_sums};
    project_qpn(x,gate,{&destination,1},stream);
    const std::int64_t elements=std::int64_t(out.ne[0])*x.tokens;
    combine<<<unsigned((elements+255)/256),256,0,stream>>>(x.qpn_sums,out.ne[0],x.tokens,static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());record_v100_kernel_route(true);return true;
}
}
#endif
