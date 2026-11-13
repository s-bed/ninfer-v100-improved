// Replays frozen represented weights and activations through the same public Ops.
#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/attn_input_proj.h"
#ifdef PRECISION_R5
#include "ops/common/v100_kernel_policy.h"
#endif
#include "ninfer_precision_ops.h"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
using namespace ninfer;
using json=nlohmann::json;
namespace fs=std::filesystem;
static std::vector<char> read(const fs::path& p) {
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("missing "+p.string());
    return {std::istreambuf_iterator<char>(f),{}};
}
int main(int argc,char**argv) {
  try {
    if(argc==2 && std::string(argv[1])=="--help") {std::cout<<"precision-wide MANIFEST OUT\n";return 0;}
    if(argc!=3)throw std::runtime_error("missing manifest/output");
    DeviceContext device;
#ifdef PRECISION_R5
    ops::set_v100_kernel_policy(true);
#endif
    auto jobs=json::parse(std::ifstream(argv[1]));std::ofstream records(argv[2],std::ios::app);
    for(auto job:jobs) {
        fs::path dir=job.at("operator_dir").get<std::string>();
        auto m=json::parse(std::ifstream(dir/"weight.json"));
        fs::path fixture=m.at("fixture").get<std::string>();
        int captured_columns=json::parse(std::ifstream(fixture/"activation-columns.json")).at("columns");
        int columns=2048;
        auto codes=read(dir/"codes.bin"),scales=read(dir/"scales.bin"),activation=read(fixture/"activation.bf16");
        const auto original_activation=activation;
        while(activation.size()<std::size_t(m.at("k").get<int>())*columns*2)
            activation.insert(activation.end(),original_activation.begin(),original_activation.end());
        Weight w;w.n=m.at("n");w.k=m.at("k");int qtype=m.at("qtype");
        w.qtype=QType(qtype);w.layout=QuantLayout(m.at("layout").get<int>());
        std::vector<float> divisors=m.at("weight_divisors").get<std::vector<float>>();
        if(w.n<=0 || w.k<=0 || (qtype!=7 && qtype!=8))throw std::runtime_error("unsupported frozen weight geometry");
        const std::size_t elements=std::size_t(w.n)*w.k;
        if(codes.size()!=elements/(qtype==7?2:1) || scales.size()!=(qtype==7?elements/16:std::size_t(w.n)*2))
            throw std::runtime_error("frozen weight plane size differs from its representation");
        if(divisors.empty())throw std::runtime_error("missing captured divisor metadata");
        int divisor_rows=m.at("weight_divisor_rows");
        if(qtype==7) {
            if(w.n%128 || w.k%64 || divisor_rows<=0 || divisor_rows%128 || w.n%divisor_rows ||
               divisors.size()!=std::size_t(w.n/divisor_rows))throw std::runtime_error("invalid frozen NVFP4 divisor geometry");
            for(float divisor:divisors)if(!std::isfinite(divisor) || divisor<=0)throw std::runtime_error("invalid frozen NVFP4 divisor");
        }
        // Native and QPN layouts keep the same enclosing payload geometry. The
        // public Ops validate both the descriptor and the relative plane pointers;
        // separate code/scale allocations are not a valid Weight contract.
        const std::size_t scale_offset=(codes.size()+255)&~std::size_t(255);
        const std::size_t divisor_offset=scale_offset+scales.size();
        std::vector<char> payload(divisor_offset+(qtype==7?divisors.size()*sizeof(float):0),0);
        std::copy(codes.begin(),codes.end(),payload.begin());
        std::copy(scales.begin(),scales.end(),payload.begin()+scale_offset);
        if(qtype==7)std::memcpy(payload.data()+divisor_offset,divisors.data(),divisors.size()*sizeof(float));
        DeviceBuffer resident(payload.size()),dx(activation.size());
        resident.copy_from_host(payload.data(),payload.size());dx.copy_from_host(activation.data(),activation.size());
        const auto* base=static_cast<const std::byte*>(resident.p);
        w.payload=base;w.payload_bytes=payload.size();w.high_plane_bytes=0;
        w.qdata=base;w.scales=base+scale_offset;w.qhigh=nullptr;
        w.ndim=2;w.shape[0]=w.padded_shape[0]=w.n;w.shape[1]=w.padded_shape[1]=w.k;
        w.shape[2]=w.shape[3]=w.padded_shape[2]=w.padded_shape[3]=1;
        w.group_size=qtype==7?16:w.k;w.group=int(w.group_size);
        w.scale_dtype=qtype==7?DType::FP8_E4M3FN:DType::BF16;
        if(qtype==8) {
            w.scale_ne[0]=w.n;w.scale_ne[1]=w.scale_ne[2]=w.scale_ne[3]=1;
            w.scale_nb[0]=2;w.scale_nb[1]=w.scale_nb[2]=w.scale_nb[3]=std::int64_t(w.n)*2;
        }
        w.input_scale_divisor=m.at("input_scale_divisor");
        w.weight_scale_divisor=divisors.at(0);
#ifndef PRECISION_V100
        if(qtype==7){w.weight_divisors=base+divisor_offset;w.weight_divisor_rows=divisor_rows;}
#endif
        auto policy=ops::LinearPolicy(m.at("policy").get<int>());
        std::string family=m.at("family");int rows=m.at("output_rows"),element=m.at("output_element_bytes");
        DeviceBuffer output(std::size_t(rows)*columns*element);
        std::vector<char> residual=family=="linear_add"?read(fixture/"residual.bin"):std::vector<char>{};
        if(!residual.empty()) {
            const auto original=residual;
            while(residual.size()<std::size_t(rows)*columns*element)residual.insert(residual.end(),original.begin(),original.end());
        }
        for(int width:{1,128,captured_columns,columns}) {
            Tensor x(dx.p,DType::BF16,{w.k,width});Tensor y(output.p,element==2?DType::BF16:DType::FP32,{rows,width});
            std::size_t capacity=0;
            if(family=="linear_add")capacity=ops::linear_add_workspace_capacity_bytes(w.qtype,w.n,w.k,policy,width,width);
            else if(family=="linear_swiglu")capacity=ops::linear_swiglu_workspace_capacity_bytes(w.qtype,w.n,w.k,policy,width,width);
            else if(family=="gdn_input_proj")capacity=ops::gdn_input_proj_workspace_capacity_bytes(w.qtype,w.n,w.k,policy,width,width);
            else if(family=="attn_input_proj")capacity=ops::attn_input_proj_workspace_capacity_bytes(w.qtype,w.n,w.k,policy,width,width);
            else capacity=ops::linear_workspace_capacity_bytes(w.qtype,w.n,w.k,policy,width,width);
            WorkspaceArena scratch(std::max<std::size_t>(256,capacity));
            if(family=="linear_add") {
                output.copy_from_host(residual.data(),std::size_t(rows)*width*element);
                ops::linear_add(x,w,y,policy,scratch,nullptr);
            } else if(family=="linear_swiglu")ops::linear_swiglu(x,w,y,policy,scratch,nullptr);
            else if(family=="gdn_input_proj") {
                DeviceBuffer a(std::size_t(10240)*width*2),b(std::size_t(6144)*width*2);
                Tensor qkv(a.p,DType::BF16,{10240,width}),z(b.p,DType::BF16,{6144,width});
                ops::gdn_input_proj(x,w,qkv,z,policy,scratch,nullptr);
                ninfer_precision::pack_part(qkv,y,0,nullptr);ninfer_precision::pack_part(z,y,10240,nullptr);
            } else if(family=="attn_input_proj") {
                DeviceBuffer a(std::size_t(6144)*width*2),b(std::size_t(1024)*width*2),c(std::size_t(6144)*width*2),d(std::size_t(1024)*width*2);
                Tensor q(a.p,DType::BF16,{6144,width}),key(b.p,DType::BF16,{1024,width}),gate(c.p,DType::BF16,{6144,width}),value(d.p,DType::BF16,{1024,width});
                ops::attn_input_proj(x,w,q,gate,key,value,policy,scratch,nullptr);
                ninfer_precision::pack_part(q,y,0,nullptr);ninfer_precision::pack_part(key,y,6144,nullptr);
                ninfer_precision::pack_part(gate,y,7168,nullptr);ninfer_precision::pack_part(value,y,13312,nullptr);
            } else ops::linear(x,w,y,policy,scratch,nullptr);
            if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("wide replay synchronization failed");
            std::vector<char> actual(y.bytes());
            if(cudaMemcpy(actual.data(),y.data,actual.size(),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("wide replay download failed");
            fs::path target=dir/("width-"+std::to_string(width)+".bin");
            if(width<=128) {
                if(actual!=read(target))throw std::runtime_error("reconstructed Op differs from frozen narrow replay: "+target.string());
            } else {
                if(fs::exists(target))throw std::runtime_error("wide output already exists");
                std::ofstream f(target,std::ios::binary);f.write(actual.data(),actual.size());
            }
        }
        records<<json{{"operator",dir.filename().string()},{"wide_columns",{captured_columns,columns}},
            {"input_mapping","Tile frozen 1024-column real activations and residuals twice for width 2048; no new values"},
            {"narrow_1_and_128_exact",true}}.dump()<<'\n'<<std::flush;
        std::cerr<<"WIDE "<<dir.filename()<<" OK\n"<<std::flush;
    }
  }catch(std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
