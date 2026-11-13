// Test-only real-activation capture and identical-input operator replay.
#pragma once
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#ifdef NINFER_PRECISION_JSON_HEADER
#include NINFER_PRECISION_JSON_HEADER
#else
#include <nlohmann/json.hpp>
#endif
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace ninfer_precision {
inline thread_local bool operator_replaying=false;
inline void pack_part(const ninfer::Tensor& part,ninfer::Tensor& output,unsigned row,cudaStream_t stream) {
    if(cudaMemcpy2DAsync(static_cast<char*>(output.data)+row*2,output.ne[0]*2,
                         part.data,part.ne[0]*2,part.ne[0]*2,part.ne[1],cudaMemcpyDeviceToDevice,stream)!=cudaSuccess)
        throw std::runtime_error("operator output packing failed");
}
inline void transfer_file(const std::filesystem::path& file,const void* device,std::size_t bytes) {
    std::ofstream output(file,std::ios::binary);
    std::vector<char> buffer(std::min<std::size_t>(bytes,4<<20));
    for(std::size_t offset=0;offset<bytes;offset+=buffer.size()) {
        auto count=std::min(buffer.size(),bytes-offset);
        if(cudaMemcpy(buffer.data(),static_cast<const char*>(device)+offset,count,cudaMemcpyDeviceToHost)!=cudaSuccess)
            throw std::runtime_error("operator capture transfer failed");
        output.write(buffer.data(),count);
    }
}
inline std::vector<char> read_prefix(const std::filesystem::path& file,std::size_t bytes) {
    std::ifstream input(file,std::ios::binary);
    std::vector<char> data(bytes);
    input.read(data.data(),bytes);
    if(std::size_t(input.gcount())!=bytes) throw std::runtime_error("truncated frozen operator activation");
    return data;
}
class OperatorCapture {
    ninfer::Tensor x_,out_;
    ninfer::Weight weight_;
    cudaStream_t stream_;
    std::string family_,key_;
    std::filesystem::path directory_,fixture_;
    bool active_=false;
    std::function<std::size_t(int)> capacity_;
    std::function<void(const ninfer::Tensor&,ninfer::Tensor&,ninfer::WorkspaceArena&)> replay_;
    std::vector<char> residual_;
public:
    OperatorCapture(const ninfer::Tensor& x,const ninfer::Weight& w,ninfer::Tensor& out,
                    cudaStream_t stream,std::string family,int policy,
                    std::function<std::size_t(int)> capacity,
                    std::function<void(const ninfer::Tensor&,ninfer::Tensor&,ninfer::WorkspaceArena&)> replay)
        :x_(x),out_(out),weight_(w),stream_(stream),family_(family),capacity_(capacity),replay_(replay) {
        const char* directory=std::getenv("NINFER_PRECISION_OPERATOR_OUT");
        const char* fixtures=std::getenv("NINFER_PRECISION_OPERATOR_FIXTURES");
        if(!directory || !fixtures || operator_replaying || (int(w.qtype)!=7 && int(w.qtype)!=8) || x.ne[1]<128) return;
        if(!((w.n==14336 && w.k==5120)||(w.n==16384 && w.k==5120)||
             (w.n==34816 && w.k==5120)||(w.n==5120 && w.k==6144)||(w.n==5120 && w.k==17408))) return;
        cudaStreamCaptureStatus capture;
        if(cudaStreamIsCapturing(stream,&capture)!=cudaSuccess || capture!=cudaStreamCaptureStatusNone) return;
        static thread_local std::map<std::string,std::set<const void*>> seen;
        std::string shape=family+"-q"+std::to_string(int(w.qtype))+"-n"+std::to_string(w.n)+"-k"+std::to_string(w.k);
        auto& pointers=seen[shape];
        if(pointers.contains(w.qdata)) return;
        unsigned index=pointers.size();pointers.insert(w.qdata);
        // 64 MLP layers; 48 GDN layers; 16 full-attention layers. Keep their
        // first, middle and last occurrences without consulting measured errors.
        unsigned occurrences=0;
        const char* counts_file=std::getenv("NINFER_PRECISION_OPERATOR_COUNTS");
        if(!counts_file) throw std::runtime_error("operator occurrence counts missing");
        static thread_local nlohmann::json counts=nlohmann::json::parse(std::ifstream(counts_file));
        occurrences=counts.at("q"+std::to_string(int(w.qtype))+"-n"+std::to_string(w.n)+"-k"+std::to_string(w.k));
        if(index!=0 && index!=occurrences/2 && index!=occurrences-1) return;
        key_=shape+"-occ"+std::to_string(index);
        directory_=std::filesystem::path(directory)/key_;
        fixture_=std::filesystem::path(fixtures)/key_;
        std::filesystem::create_directories(directory_);
        std::filesystem::create_directories(fixture_);
        if(cudaStreamSynchronize(stream)!=cudaSuccess) throw std::runtime_error("operator pre-sync failed");
        const int capture_columns=std::min<int>(2048,x.ne[1]);
        std::size_t activation_bytes=std::size_t(w.k)*capture_columns*2;
        const char* capture_mode=std::getenv("NINFER_PRECISION_OPERATOR_CAPTURE");
        if(capture_mode && std::string(capture_mode)=="1") {
            if(std::filesystem::exists(fixture_/"activation.bf16")) throw std::runtime_error("operator fixture exists");
            transfer_file(fixture_/"activation.bf16",x.data,activation_bytes);
            std::ofstream capture_meta(fixture_/"activation-columns.json");
            capture_meta<<nlohmann::json{{"columns",capture_columns}}.dump();
            if(family=="linear_add") transfer_file(fixture_/"residual.bin",out.data,std::size_t(out.ne[0])*capture_columns*(out.bytes()/out.numel()));
        }
        if(!std::filesystem::exists(fixture_/"activation.bf16")) {
            std::ofstream missing(directory_/"missing-fixture.json");
            missing<<nlohmann::json{{"key",key_},{"reason","baseline did not capture this family/occurrence"}}.dump();
            return;
        }
        const int frozen_columns=nlohmann::json::parse(std::ifstream(fixture_/"activation-columns.json")).at("columns");
        if(family=="linear_add") residual_=read_prefix(fixture_/"residual.bin",std::size_t(out.ne[0])*frozen_columns*(out.bytes()/out.numel()));
        auto code_bytes=std::size_t(w.n)*w.k/(int(w.qtype)==7?2:1);
        auto scale_bytes=int(w.qtype)==7?std::size_t(w.n)*w.k/16:std::size_t(w.n)*2;
        transfer_file(directory_/"codes.bin",w.qdata,code_bytes);
        transfer_file(directory_/"scales.bin",w.scales,scale_bytes);
        unsigned divisor_rows=w.n;
        std::vector<float> divisors{w.weight_scale_divisor};
#ifndef PRECISION_V100
        if(w.weight_divisors && w.weight_divisor_rows>0 && int(w.qtype)==7) {
            divisor_rows=w.weight_divisor_rows;
            divisors.resize((w.n+divisor_rows-1)/divisor_rows);
            if(cudaMemcpy(divisors.data(),w.weight_divisors,divisors.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess)
                throw std::runtime_error("operator divisor capture failed");
        }
#endif
        std::ofstream meta(directory_/"weight.json");
        meta<<nlohmann::json{{"family",family},{"qtype",int(w.qtype)},{"layout",int(w.layout)},
             {"n",w.n},{"k",w.k},{"policy",policy},{"output_rows",out.ne[0]},{"output_element_bytes",out.bytes()/out.numel()},
             {"input_scale_divisor",w.input_scale_divisor},{"weight_divisors",divisors},
             {"weight_divisor_rows",divisor_rows},{"occurrence",index},{"expected_occurrences",occurrences},
             {"fixture",fixture_.string()},{"actual_capture_columns",x.ne[1]}}.dump(2);
        active_=true;
    }
    ~OperatorCapture() noexcept(false) {
        if(!active_ || std::uncaught_exceptions()) return;
        operator_replaying=true;
        try {
            int columns=nlohmann::json::parse(std::ifstream(fixture_/"activation-columns.json")).at("columns");
            auto input=read_prefix(fixture_/"activation.bf16",std::size_t(weight_.k)*columns*2);
            ninfer::DeviceBuffer x(input.size()),y(std::size_t(out_.ne[0])*columns*(out_.bytes()/out_.numel()));
            x.copy_from_host(input.data(),input.size());
            std::vector<int> widths{1,2,4,6,8,16,32,33,63,64,65,128};
            if(columns>=2048)widths.push_back(2048);
            for(int width:widths) {
                ninfer::WorkspaceArena workspace(std::max<std::size_t>(256,capacity_(width)));
                ninfer::Tensor in(x.p,ninfer::DType::BF16,{weight_.k,width});
                ninfer::Tensor output(y.p,out_.dtype,{int(out_.ne[0]),width});
                if(family_=="linear_add") y.copy_from_host(residual_.data(),std::size_t(out_.ne[0])*width*(out_.bytes()/out_.numel()));
                replay_(in,output,workspace);
                if(cudaStreamSynchronize(stream_)!=cudaSuccess) throw std::runtime_error("operator replay failed");
                transfer_file(directory_/("width-"+std::to_string(width)+".bin"),y.p,output.bytes());
            }
        } catch(...) {operator_replaying=false;throw;}
        operator_replaying=false;
    }
};
}
