// Test-only observer: invoked after completion, never inside graph capture.
#pragma once
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <type_traits>
namespace ninfer_precision {
struct Script {
    std::string identity;
    std::vector<int> tokens;
    std::size_t cursor=0;
    unsigned round=0;
};
inline Script& script() {
    static thread_local Script s;
    const char* value=std::getenv("NINFER_PRECISION_FORCED");
    const char* directory=std::getenv("NINFER_PRECISION_OBSERVER");
    std::string identity=directory?directory:"";
    if(s.identity!=identity) {
        s=Script{};s.identity=identity;
        if(value && *value) s.tokens=nlohmann::json::parse(value).get<std::vector<int>>();
    }
    return s;
}
inline bool forcing() {return !script().tokens.empty();}
inline bool observing_state() {
    const char* directory=std::getenv("NINFER_PRECISION_OBSERVER");
    return directory && *directory;
}
inline void force_prefill(int* tokens) {
    auto& s=script();
    if(!s.tokens.empty()) {
        if(s.cursor!=0) throw std::logic_error("script prefill cursor is not zero");
        tokens[0]=s.tokens.at(s.cursor++);
    }
}
template<class Context> void force_ordinary(Context& state) {
    auto& s=script();
    if(s.tokens.empty()) return;
    std::filesystem::create_directories(s.identity);
    if(cudaStreamSynchronize(state.execution.device.stream)!=cudaSuccess) throw std::runtime_error("forced ordinary sync failed");
    state.host_egress.sampled_tokens[0]=s.tokens.at(s.cursor++);
}
template<class Context> void proposals(Context& state,unsigned width) {
    auto& s=script();
    if(s.tokens.empty()) return;
    std::filesystem::create_directories(s.identity);
    // This host ingress is uploaded by the actual eager/captured MTP body.
    // Alternate an unmodified fixed proposal with a deliberately wrong suffix.
    unsigned current=state.host_ingress.current_extents[0];
    for(unsigned i=0;i<current;i++) {
        int token=s.tokens.at(std::min(s.cursor+i,s.tokens.size()-1));
        if(s.round%3==1 && i>=current/2) token=42;
        if(s.round%3==2) token=43;
        auto& ingress=const_cast<std::remove_cvref_t<decltype(state.host_ingress)>&>(state.host_ingress);
        ingress.current_drafts[i]=token;
    }
    std::ofstream meta(std::filesystem::path(s.identity)/"proposal-inputs.jsonl",std::ios::app);
    meta<<nlohmann::json{{"cursor",s.cursor},{"round",s.round},{"width",width},
                         {"extent",current},{"frontier",state.host_ingress.base_frontiers[0]},
                         {"drafts",std::vector<int>(state.host_ingress.current_drafts.begin(),state.host_ingress.current_drafts.begin()+current)}}.dump()<<'\n';
}
template<class Context> void force_bonus(Context& state,unsigned width) {
    auto& s=script();
    if(s.tokens.empty()) return;
    if(cudaStreamSynchronize(state.execution.device.stream)!=cudaSuccess) throw std::runtime_error("forced verify sync failed");
    int count=state.host_egress.licensed_counts[0], accepted=state.host_egress.accepted_drafts[0];
    if(count<1 || count>int(width+1) || count!=accepted+1) throw std::logic_error("invalid actual acceptance");
    // Accepted proposals have already been executed; retain their exact bytes.
    // The final correction/bonus has not entered target state yet and is scripted
    // before the normal caller commits it. Acceptance and fold logic stay intact.
    for(int i=0;i<accepted;i++) {
        if(state.host_egress.licensed_tokens[i]!=s.tokens.at(s.cursor+i))
            throw std::logic_error("accepted injected wrong proposal; cannot force a different committed prefix");
    }
    const int unforced_bonus=state.host_egress.licensed_tokens[accepted];
    state.host_egress.licensed_tokens[accepted]=s.tokens.at(s.cursor+accepted);
    std::ofstream meta(std::filesystem::path(s.identity)/"acceptance.jsonl",std::ios::app);
    meta<<nlohmann::json{{"cursor",s.cursor},{"round",s.round},{"accepted",accepted},{"count",count},
                         {"frontier",state.host_ingress.base_frontiers[0]},
                         {"unforced_bonus",unforced_bonus},
                         {"extent",state.host_ingress.current_extents[0]}}.dump()<<'\n';
    s.cursor+=count;s.round++;
}
template<class Pool> void state(Pool& pool,int slot,unsigned execution,unsigned ledger) {
    const char* directory=std::getenv("NINFER_PRECISION_OBSERVER");
    if(!directory || !*directory) return;
    std::filesystem::path dir(directory);
    std::filesystem::create_directories(dir);
    auto path=dir/"states.bin";
    auto offset=std::filesystem::exists(path)?std::filesystem::file_size(path):0;
    std::ofstream out(path,std::ios::binary|std::ios::app);
    nlohmann::json parts=nlohmann::json::array();
    for(unsigned layer:{0U,pool.layer_count()/2,pool.layer_count()-1}) {
        auto recurrent=pool.recurrent_slot(layer,slot);
        auto conv=pool.conv_slot(layer,slot);
        for(auto pair:{std::pair{recurrent,"recurrent"},std::pair{conv,"conv"}}) {
            auto count=std::min<std::size_t>(pair.first.bytes(),4096*4);
            std::vector<char> buffer(count);
            if(cudaMemcpy(buffer.data(),pair.first.data,count,cudaMemcpyDeviceToHost)!=cudaSuccess)
                throw std::runtime_error("state observer download failed");
            out.write(buffer.data(),count);
            parts.push_back({{"layer",layer},{"kind",pair.second},{"bytes",count},{"dtype",int(pair.first.dtype)}});
        }
    }
    std::ofstream meta(dir/"states.jsonl",std::ios::app);
    meta<<nlohmann::json{{"execution_frontier",execution},{"ledger_frontier",ledger},
                         {"slot",slot},{"offset",offset},{"parts",parts},{"script_cursor",script().cursor}}.dump()<<'\n';
}
template<class T> void logits(const T& tensor, cudaStream_t stream, const char* phase,
                              unsigned public_vocab, bool graph=false, int input_frontier=-1) {
    const char* directory=std::getenv("NINFER_PRECISION_OBSERVER");
    if(!directory || !*directory) return;
    cudaStreamCaptureStatus status;
    if(cudaStreamIsCapturing(stream,&status)!=cudaSuccess || status!=cudaStreamCaptureStatusNone)
        throw std::logic_error("precision observer invoked during capture");
    if(cudaStreamSynchronize(stream)!=cudaSuccess) throw std::runtime_error("observer sync failed");
    std::filesystem::path dir(directory);
    std::filesystem::create_directories(dir);
    auto path=dir/"logits.bf16";
    std::uint64_t offset=std::filesystem::exists(path)?std::filesystem::file_size(path):0;
    // One tail column per scoring tile; all columns for actual Decode/Verify.
    unsigned columns=tensor.ne[1];
    unsigned start=std::string(phase)=="score"?columns-1:0;
    std::vector<std::uint16_t> host(public_vocab);
    std::ofstream out(path,std::ios::binary|std::ios::app);
    for(unsigned col=start;col<columns;col++) {
        const auto* src=static_cast<const char*>(tensor.data)+std::size_t(col)*tensor.ne[0]*2;
        if(cudaMemcpy(host.data(),src,host.size()*2,cudaMemcpyDeviceToHost)!=cudaSuccess)
            throw std::runtime_error("observer logits download failed");
        out.write(reinterpret_cast<const char*>(host.data()),host.size()*2);
    }
    std::ofstream meta(dir/"observations.jsonl",std::ios::app);
    meta << nlohmann::json{{"phase",phase},{"offset",offset},{"vocab",public_vocab},
                           {"tensor_rows",tensor.ne[0]},{"columns",columns-start},
                           {"first_column",start},{"actual_graph_launch",graph},
                           {"input_frontier",input_frontier},
                           {"script_cursor",script().cursor}}.dump()<<'\n';
}
}
