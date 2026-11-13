// TEST/BENCH ONLY: directly instantiate donor FP8 QPN with a private byte permutation.
// Packed bytes never enter a public Weight, loader, artifact reader or layout validator.
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/sm70/v100_fp8_qpn.cuh"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/op_check.h"
#include <cuda_fp8.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;
namespace sm = ops::detail::sm70;

void ck(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class Buffer {
    static constexpr std::size_t guard=256;
    std::size_t bytes_;
    std::uint8_t* base_=nullptr;
public:
    explicit Buffer(std::size_t bytes):bytes_(bytes) {
        ck(cudaMalloc(&base_,bytes+2*guard));
        ck(cudaMemset(base_,0xa5,bytes+2*guard));
    }
    ~Buffer() { if(base_) cudaFree(base_); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    template<class T=std::uint8_t> T* data()const {return reinterpret_cast<T*>(base_+guard);}
    std::size_t size()const {return bytes_;}
    void put(const void* host,std::size_t bytes) {if(bytes>bytes_)throw std::runtime_error("upload overflow");ck(cudaMemcpy(data(),host,bytes,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());}
    void get(void* host,std::size_t bytes,std::size_t offset=0)const {if(offset+bytes>bytes_)throw std::runtime_error("download overflow");ck(cudaMemcpy(host,data()+offset,bytes,cudaMemcpyDeviceToHost));}
    void poison(int byte=0xff) {ck(cudaMemset(data(),byte,bytes_));ck(cudaDeviceSynchronize());}
    void check(std::string_view label)const {
        std::array<std::uint8_t,2*guard> bytes{};
        ck(cudaMemcpy(bytes.data(),base_,guard,cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(bytes.data()+guard,base_+guard+bytes_,guard,cudaMemcpyDeviceToHost));
        if(std::any_of(bytes.begin(),bytes.end(),[](auto v){return v!=0xa5;}))throw std::runtime_error(std::string(label)+" guard changed");
    }
};

float from_bf16(std::uint16_t bits) {return std::bit_cast<float>(std::uint32_t(bits)<<16);}
std::uint16_t to_bf16(float value) {
    auto bits=std::bit_cast<std::uint32_t>(value);
    if((bits&0x7fffffffU)>0x7f800000U)return std::uint16_t((bits>>16)|0x0040U);
    bits+=0x7fffU+((bits>>16)&1U);
    return std::uint16_t(bits>>16);
}
// Independent E4M3FN arithmetic; no CUDA decoder, production materializer or FP16 shift.
double code_value(std::uint8_t code) {
    const int exponent=(code>>3)&15,mantissa=code&7;
    const double magnitude=exponent==0?std::ldexp(double(mantissa),-9):
        (exponent==15 && mantissa==7?std::numeric_limits<double>::quiet_NaN():
         std::ldexp(1.0+double(mantissa)/8.0,exponent-7));
    return code&128?-magnitude:magnitude;
}

// CPU forward and inverse address formulas are independently expressed.
std::size_t packed_address(int row,int column,int k) {
    const int local=row%32;
    const int lane=(local/8)*4+local%4+((local%8)/4)*16;
    return (((std::size_t(row/32)*(k/128)+column/128)*8+(column%128)/16)*32+lane)*16+column%16;
}
std::size_t native_tuple(std::size_t tuple,int k) {
    const int lane=int(tuple%32),slice=int((tuple/32)%8);
    const int block=int((tuple/256)%(k/128));
    const int tile=int(tuple/(std::size_t(k/128)*256));
    const int row=tile*32+((lane>>2)&3)*8+(lane&3)+(lane&16?4:0);
    return std::size_t(row)*k+block*128+slice*16;
}
void cpu_round_trip() {
    constexpr int n=64,k=384;
    std::vector<std::uint8_t> raw(n*k),packed(n*k),restored(n*k),written(n*k);
    for(std::size_t i=0;i<raw.size();++i)raw[i]=std::uint8_t(i*73+(i/127)*19);
    for(int row=0;row<n;++row)for(int col=0;col<k;++col) {
        const auto at=packed_address(row,col,k);
        if(at>=packed.size() || written[at]++)throw std::runtime_error("CPU permutation is not bijective");
        packed[at]=raw[std::size_t(row)*k+col];
    }
    for(std::size_t tuple=0;tuple<packed.size()/16;++tuple)
        std::memcpy(restored.data()+native_tuple(tuple,k),packed.data()+tuple*16,16);
    if(raw!=restored)throw std::runtime_error("exact CPU FP8 permutation round trip failed");
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
// Donor b37d0dd3 fp8_prepack_sm70.cu, test-local, out of place; scales remain logical.
__global__ void prepack(const std::uint8_t* input,std::uint8_t* output,int n,int k) {
    const std::int64_t index=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int blocks=k/128;
    if(index>=std::int64_t(n/32)*blocks*8*32)return;
    const int lane=int(index%32),slice=int((index/32)%8);
    const int block=int((index/256)%blocks),tile=int(index/(std::int64_t(blocks)*256));
    const int row=tile*32+((lane>>2)&3)*8+(lane&3)+(lane&16?4:0);
    reinterpret_cast<uint4*>(output)[index]=*reinterpret_cast<const uint4*>(input+std::int64_t(row)*k+block*128+slice*16);
}

__global__ void column_flags(const __nv_bfloat16* x,int k,int tokens,std::uint32_t* flags) {
    const int token=int(blockIdx.x);
    bool beyond=false;
    for(int e=int(threadIdx.x);e<k;e+=int(blockDim.x)) {
        const float value=__bfloat162float(x[std::int64_t(token)*k+e]);
        beyond|=!isfinite(value)||fabsf(value)>65504.0F;
    }
    if(__any_sync(0xffffffffU,beyond) && (threadIdx.x&31)==0)atomicOr(flags+token,1U);
}
// Same vectorized saturation/flag contract as sm70_gemm.cu, without library dependencies.
__global__ void natural_activation(const uint4* x,uint4* out,std::int64_t vectors,int k,std::uint32_t* flags) {
    const std::int64_t i=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=vectors)return;
    const uint4 in=x[i];
    out[i]=make_uint4(sm::bf16x2_to_f16x2(in.x),sm::bf16x2_to_f16x2(in.y),sm::bf16x2_to_f16x2(in.z),sm::bf16x2_to_f16x2(in.w));
    if(sm::bf16x2_beyond_f16(in.x)||sm::bf16x2_beyond_f16(in.y)||sm::bf16x2_beyond_f16(in.z)||sm::bf16x2_beyond_f16(in.w)) {
        const unsigned words[4]={in.x,in.y,in.z,in.w};
        for(int e=0;e<8;++e) {
            const unsigned bits=e&1?words[e/2]&0xffff0000U:words[e/2]<<16;
            if(sm::bf16x2_beyond_f16(bits))atomicOr(flags+(i*8+e)/k,1U);
        }
    }
}

struct Output {
    sm::DestinationEpilogue destination;
    const std::uint8_t* native_codes;
    const __nv_bfloat16* original;
    const std::uint32_t* beyond;
    int k;
    __device__ void store(int row,int token,float value)const {
        if(beyond[token]) {
            float correction=0.0F;
            for(int e=0;e<k;++e) {
                const float delta=sm::saturated_excess(__bfloat162float(original[std::int64_t(token)*k+e]));
                if(delta!=0.0F) {
                    __nv_fp8_e4m3 code;code.__x=native_codes[std::int64_t(row)*k+e];
                    correction=fmaf(float(code),delta,correction);
                }
            }
            value+=correction;
        }
        // Leaf has undone 2^-8 before this FP32 correction. Apply RowFactor once,
        // outside the contraction and before the existing BF16/FP32 destination.
        destination(row,token,value);
    }
};

enum class Variant {RawBf16,PackedBf16,PackedF16,RawF16};
constexpr std::array<Variant,4> variants{Variant::RawBf16,Variant::PackedBf16,Variant::PackedF16,Variant::RawF16};
std::string_view variant_name(Variant v) {
    switch(v){case Variant::RawBf16:return "raw-bf16";case Variant::PackedBf16:return "packed-bf16";
    case Variant::PackedF16:return "packed-f16";case Variant::RawF16:return "raw-f16";}
    return "invalid";
}
bool half_activation(Variant v){return v==Variant::PackedF16||v==Variant::RawF16;}

template<bool Packed,class Activation>
void qpn(const std::uint8_t* codes,const __nv_bfloat16* scales,const Activation* activation,int n,int k,int t,Output output,cudaStream_t stream) {
    const dim3 grid(unsigned(n/32));
#define LAUNCH(TILES,SPLIT) ops::detail::fp8_volta_qpn_gemm_kernel<TILES,SPLIT,1,Packed><<<grid,SPLIT*32,0,stream>>>(codes,scales,activation,n,k,t,output)
    if(t<=8){if(k==6144){LAUNCH(1,16);}else{LAUNCH(1,8);}}
    else if(t<=16){LAUNCH(2,8);}else{LAUNCH(4,8);}
#undef LAUNCH
    ck(cudaGetLastError());
}

struct Fixture {
    int n,k,max_t;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
    std::vector<int> rows;
    Buffer raw,packed,device_scales,x,natural,flags,out;
    cudaStream_t stream=nullptr;
    Fixture(int n_,int k_,int max_t_):n(n_),k(k_),max_t(max_t_),codes(std::size_t(n)*k),scales(n),
        raw(codes.size()),packed(codes.size()),device_scales(std::size_t(n)*2),x(std::size_t(k)*max_t*2),
        natural(std::size_t(k)*max_t*2),flags(std::size_t(max_t)*4),out(std::size_t(n)*max_t*4) {
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        constexpr std::array<std::uint16_t,12> broad{0x0000,0x3580,0x3781,0x3983,0x3a01,0x3b43,0x3d99,0x3e81,0x3f81,0x4003,0x4105,0x4189};
        for(int row=0;row<n;++row) {
            scales[row]=broad[(row*7+(row>>5)+3)%broad.size()];
            for(int e=0;e<k;++e) {
                const unsigned index=(unsigned(row)*13U+unsigned(e)*17U+(unsigned(e)>>6)*29U+(unsigned(row)>>5)*7U)%254U;
                codes[std::size_t(row)*k+e]=std::uint8_t(index+(index>=127));
            }
        }
        for(int row:{0,1,7,8,15,16,31,32,33,63,64,127,128,n/2,n-33,n-32,n-31,n-2,n-1})
            if(row>=0&&row<n&&std::find(rows.begin(),rows.end(),row)==rows.end())rows.push_back(row);
        raw.put(codes.data(),codes.size());device_scales.put(scales.data(),scales.size()*2);
        const std::int64_t tuples=std::int64_t(n)*k/16;
        prepack<<<unsigned((tuples+255)/256),256,0,stream>>>(raw.data(),packed.data(),n,k);
        ck(cudaGetLastError());ck(cudaStreamSynchronize(stream));
        // Compare every GPU-packed byte to the independent CPU inverse mapping,
        // using bounded host memory even for the 1.27 GB vocabulary head.
        constexpr std::size_t chunk_bytes=1U<<22;
        std::vector<std::uint8_t> chunk(std::min(chunk_bytes,codes.size()));
        for(std::size_t offset=0;offset<codes.size();offset+=chunk.size()) {
            const std::size_t bytes=std::min(chunk.size(),codes.size()-offset);
            packed.get(chunk.data(),bytes,offset);
            for(std::size_t at=0;at<bytes;at+=16)
                if(std::memcmp(chunk.data()+at,codes.data()+native_tuple((offset+at)/16,k),16)!=0)
                    throw std::runtime_error("GPU prepack differs from exact CPU permutation");
        }
        std::vector<std::uint16_t> unchanged(n);device_scales.get(unchanged.data(),unchanged.size()*2);
        if(unchanged!=scales)throw std::runtime_error("prepack changed logical row scales");
    }
    ~Fixture(){if(stream)cudaStreamDestroy(stream);}
    void prepare(Variant variant,int t) {
        ck(cudaMemsetAsync(flags.data(),0,std::size_t(t)*4,stream));
        if(half_activation(variant)) {
            const std::int64_t vectors=std::int64_t(k)*t/8;
            natural_activation<<<unsigned((vectors+255)/256),256,0,stream>>>(x.data<uint4>(),natural.data<uint4>(),vectors,k,flags.data<std::uint32_t>());
        }else column_flags<<<t,128,0,stream>>>(x.data<__nv_bfloat16>(),k,t,flags.data<std::uint32_t>());
        ck(cudaGetLastError());
    }
    void launch(Variant variant,int t,bool bf16,bool with_prepare=true) {
        if(with_prepare)prepare(variant,t);
        sm::DestinationEpilogue destination;
        destination.count=1;
        destination.factor.kind=sm::RowFactor::Kind::Bf16Row;
        destination.factor.data=device_scales.data();
        destination.set[0]=sm::Destination{bf16?out.data<__nv_bfloat16>():nullptr,n,0,0,n,false,bf16?nullptr:out.data<float>()};
        const Output output{destination,raw.data(),x.data<__nv_bfloat16>(),flags.data<std::uint32_t>(),k};
        switch(variant) {
        case Variant::RawBf16:qpn<false>(raw.data(),device_scales.data<__nv_bfloat16>(),x.data<__nv_bfloat16>(),n,k,t,output,stream);break;
        case Variant::PackedBf16:qpn<true>(packed.data(),device_scales.data<__nv_bfloat16>(),x.data<__nv_bfloat16>(),n,k,t,output,stream);break;
        case Variant::PackedF16:qpn<true>(packed.data(),device_scales.data<__nv_bfloat16>(),natural.data<half>(),n,k,t,output,stream);break;
        case Variant::RawF16:qpn<false>(raw.data(),device_scales.data<__nv_bfloat16>(),natural.data<half>(),n,k,t,output,stream);break;
        }
    }
    std::vector<double> reference(std::span<const std::uint16_t> input,int t)const {
        std::vector<double> values(std::size_t(t)*rows.size());
        for(int token=0;token<t;++token)for(std::size_t r=0;r<rows.size();++r) {
            const int row=rows[r];double sum=0;
            for(int e=0;e<k;++e)sum+=code_value(codes[std::size_t(row)*k+e])*double(from_bf16(input[std::size_t(token)*k+e]));
            values[std::size_t(token)*rows.size()+r]=sum*double(from_bf16(scales[row]));
        }
        return values;
    }
    void verify(Variant variant,int t,bool bf16,std::span<const double> expected,bool nan,std::string_view phase) {
        std::vector<double> got(expected.size());
        if(bf16) {
            std::vector<std::uint16_t> host(std::size_t(n)*t);out.get(host.data(),host.size()*2);
            if(nan) {for(auto bits:host)if((bits&0x7f80)!=0x7f80 || (bits&0x007f)==0)throw std::runtime_error("NaN propagation failed");}
            else for(int token=0;token<t;++token)for(std::size_t r=0;r<rows.size();++r)got[std::size_t(token)*rows.size()+r]=from_bf16(host[std::size_t(token)*n+rows[r]]);
        }else {
            std::vector<float> host(std::size_t(n)*t);out.get(host.data(),host.size()*4);
            if(nan) {for(float value:host)if(!std::isnan(value))throw std::runtime_error("NaN propagation failed");}
            else for(int token=0;token<t;++token)for(std::size_t r=0;r<rows.size();++r)got[std::size_t(token)*rows.size()+r]=host[std::size_t(token)*n+rows[r]];
        }
        if(!nan) {
            constexpr test::ReductionCriterion tolerance{1.0/256.0,1.0/256.0,2.0/256.0};
            const auto stats=test::compute_reduction_stats(got.data(),expected.data(),std::int64_t(got.size()));
            if(!test::reduction_passes(stats,std::int64_t(got.size()),tolerance)) {
                std::cerr<<"FAIL "<<variant_name(variant)<<" N="<<n<<" K="<<k<<" T="<<t<<" bf16="<<bf16<<" "<<phase<<" relative_l2="<<stats.relative_l2<<" max_error="<<stats.maximum_absolute_error<<" gross_limit="<<test::gross_error_limit(stats,tolerance)<<'\n';
                throw std::runtime_error("independent FP64 A16 criterion failed");
            }
        }
        out.check("output");
    }
    void qualify(std::vector<std::uint16_t> input,int t,bool nan,std::string_view phase) {
        auto expected=reference(input,t);
        std::vector<std::uint16_t> negative=input;
        for(auto& value:negative)value^=0x8000U;
        std::vector<double> negative_expected=expected;for(auto& value:negative_expected)value=-value;
        for(Variant variant:variants)for(bool bf16:{false,true}) {
            x.put(input.data(),input.size()*2);out.poison(nan?0x00:0xff);
            launch(variant,t,bf16);ck(cudaStreamSynchronize(stream));
            verify(variant,t,bf16,expected,nan,phase);
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            launch(variant,t,bf16);
            ck(cudaStreamEndCapture(stream,&graph));
            ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            // Replay twice with different bytes at the same captured pointers.
            for(int replay=0;replay<2;++replay) {
                const auto& active=replay?input:negative;
                x.put(active.data(),active.size()*2);out.poison(nan?0x00:0xff);
                ck(cudaGraphLaunch(executable,stream));ck(cudaStreamSynchronize(stream));
                verify(variant,t,bf16,replay?std::span<const double>(expected):std::span<const double>(negative_expected),nan,"graph");
            }
            ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));
        }
        raw.check("native codes");packed.check("packed codes");device_scales.check("row scales");x.check("input");natural.check("natural FP16");flags.check("flags");
    }
};

std::vector<std::uint16_t> activation(int k,int t,int stress=0) {
    std::vector<std::uint16_t> values(std::size_t(k)*t);
    for(int token=0;token<t;++token)for(int e=0;e<k;++e) {
        const int centered=int((std::uint64_t(e)*17+std::uint64_t(token)*23+13)%127)-63;
        const float value=token&1?(float(centered+80)/256.0F):float(centered)/128.0F;
        values[std::size_t(token)*k+e]=to_bf16(value);
    }
    if(stress==1)for(int token=0;token<t;++token) {
        values[std::size_t(token)*k+3]=to_bf16(std::ldexp(1.0F,20));
        values[std::size_t(token)*k+19]=to_bf16(-std::ldexp(1.0F,20));
        values[std::size_t(token)*k+97]=to_bf16(std::ldexp(1.0F,-30));
    }
    if(stress==2)for(int token=0;token<t;++token)values[std::size_t(token)*k+11]=0x7fc0;
    return values;
}

struct Timing {std::string phase,variant;int n,k,t,repeats;float minimum,median,maximum;};
Timing measure(Fixture& fixture,Variant variant,int t,bool complete) {
    auto input=activation(fixture.k,t);fixture.x.put(input.data(),input.size()*2);
    fixture.prepare(variant,t);fixture.launch(variant,t,false,complete);ck(cudaStreamSynchronize(fixture.stream));
    cudaEvent_t begin=nullptr,end=nullptr;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    const auto sample=[&] {
        ck(cudaEventRecord(begin,fixture.stream));fixture.launch(variant,t,false,complete);
        ck(cudaEventRecord(end,fixture.stream));ck(cudaEventSynchronize(end));
        float elapsed=0;ck(cudaEventElapsedTime(&elapsed,begin,end));return elapsed;
    };
    const float pilot=sample();
    const int count=std::clamp(int(std::ceil(60.0/std::max(0.001F,pilot))),10,200);
    std::vector<float> samples;for(int i=0;i<count;++i)samples.push_back(sample());
    ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));
    std::sort(samples.begin(),samples.end());
    const float median=count%2?samples[count/2]:(samples[count/2-1]+samples[count/2])/2;
    return Timing{complete?"complete":"kernel",std::string(variant_name(variant)),fixture.n,fixture.k,t,count,samples.front(),median,samples.back()};
}
#endif
}

int main(int argc,char** argv) {
    bool bench=false;std::string selected;
    for(int argument=1;argument<argc;++argument) {
        const std::string_view flag(argv[argument]);
        if(flag=="--bench"&&!bench)bench=true;
        else if(flag=="--shape"&&selected.empty()&&argument+1<argc)selected=argv[++argument];
        else {std::cerr<<"usage: ninfer_v100_fp8_prepack_experiment [--bench] [--shape gdn|attn|out|mlp|down|head]\n";return 2;}
    }
    try {
        cpu_round_trip();
        int count=0;const auto available=cudaGetDeviceCount(&count);
        if(available!=cudaSuccess || count==0)return 77;
        cudaDeviceProp properties{};ck(cudaGetDeviceProperties(&properties,0));
        if(properties.major!=7||properties.minor!=0)return 77;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
        struct Shape{std::string_view name;int n,k;};
        constexpr std::array<Shape,6> shapes{{{"gdn",16384,5120},{"attn",14336,5120},{"out",5120,6144},{"mlp",34816,5120},{"down",5120,17408},{"head",248320,5120}}};
        if(!selected.empty()&&std::none_of(shapes.begin(),shapes.end(),[&](auto shape){return shape.name==selected;}))throw std::invalid_argument("unknown shape");
        std::vector<Timing> timings;
        for(auto shape:shapes) {
            if(!selected.empty()&&shape.name!=selected)continue;
            const int max_t=shape.name=="head"?8:32;
            Fixture fixture(shape.n,shape.k,max_t);
            for(int t:{1,4,8,16,32})if(t<=max_t)fixture.qualify(activation(shape.k,t),t,false,"finite");
            fixture.qualify(activation(shape.k,1,1),1,false,"BF16 range");
            fixture.qualify(activation(shape.k,1,2),1,true,"NaN");
            std::cout<<"QUALIFIED test-local FP8 prepack "<<shape.name<<" N="<<shape.n<<" K="<<shape.k<<" four variants, BF16/FP32, independent FP64, outliers/NaN, changed-input graph replay\n";
            if(bench)for(int t:{1,4,8,16,32})if(t<=max_t)for(Variant variant:variants)for(bool complete:{false,true})timings.push_back(measure(fixture,variant,t,complete));
        }
        // No event result is published unless every selected shape passed first.
        if(bench) {
            std::cout<<"phase,variant,N,K,T,repetitions,min_ms,median_ms,max_ms\n"<<std::fixed<<std::setprecision(6);
            for(const auto& row:timings)std::cout<<row.phase<<','<<row.variant<<','<<row.n<<','<<row.k<<','<<row.t<<','<<row.repeats<<','<<row.minimum<<','<<row.median<<','<<row.maximum<<'\n';
        }
        std::cout<<"PASS private FP8 prepack experiment; no production weight-layout compatibility claimed\n";
        return 0;
#else
        return 77;
#endif
    }catch(const std::exception& error){std::cerr<<"FP8 prepack experiment: "<<error.what()<<'\n';return 1;}
}
