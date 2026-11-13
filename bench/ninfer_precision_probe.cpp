// Isolated benchmark executable. Linked to each engine independently; never deployed.
#include "ninfer/engine.h"
#ifdef PRECISION_V100
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
namespace frontend = ninfer::targets::qwen3_6::frontend_internal;
#else
#include "models/qwen3_5/frontend/tokenizer.h"
namespace frontend = ninfer::models::qwen3_5::frontend;
#endif
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
using json = nlohmann::json;
namespace fs = std::filesystem;
static std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>(in), {}};
}
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "precision-probe tokenize RESOURCES JOBS OUT | score|generate|native ARTIFACT JOBS OUT [K] [CONTEXT]\n";
        return 0;
    }
    if (argc < 5) throw std::invalid_argument("missing arguments");
    std::string mode = argv[1];
    std::ifstream jobs(argv[3]);
    if (!jobs) throw std::runtime_error("jobs missing");
    std::ofstream out(argv[4], std::ios::app);
    std::string line;
    if (mode == "tokenize") {
        std::string a=read(fs::path(argv[2])/"tokenizer.json"), b=read(fs::path(argv[2])/"tokenizer_config.json"), c=read(fs::path(argv[2])/"generation_config.json");
        frontend::Tokenizer tokenizer({a,b,c});
#ifdef PRECISION_V100
        constexpr std::size_t public_vocab=248077; // pinned target's kTokenDomain
        if(!tokenizer.has_exact_token_domain(public_vocab)) throw std::runtime_error("original tokenizer domain differs");
#else
        const std::size_t public_vocab=tokenizer.vocab_size();
#endif
        while (std::getline(jobs,line)) {
            auto job=json::parse(line);
            std::string text = job.contains("text") ? job["text"].get<std::string>() : read(job["path"].get<std::string>());
            auto ids=tokenizer.encode(text);
            std::size_t maximum=job.value("max_tokens", ids.size());
            if (ids.size() > maximum) ids.resize(maximum);
            out << json{{"id",job.at("id")},{"tokens",ids},{"public_vocab",public_vocab}}.dump() << '\n' << std::flush;
        }
        return 0;
    }
    ninfer::EngineOptions opts;
    opts.artifact_path=argv[2];
    opts.purpose=mode=="score" ? ninfer::EnginePurpose::CausalScoring : ninfer::EnginePurpose::Generation;
    opts.max_context=argc>6 ? std::stoul(argv[6]) : 8192;
    opts.kv_capacity=ninfer::KvCapacityPolicy::explicit_capacity(opts.max_context);
    opts.kv_cache=ninfer::KvCacheStorage::Int8Group64;
    opts.prefill_chunk=2048;
    opts.context_cache.host_state_slots=0;
    opts.context_cache.host_kv_capacity_bytes=0;
    if (mode=="native") opts.context_cache.enabled=false;
    opts.speculative.proposal_head=ninfer::ProposalHead::Full;
    int k=argc>5 ? std::stoi(argv[5]) : 0;
    if (k) {
        opts.speculative.backend=ninfer::SpeculativeBackend::Mtp;
        opts.speculative.draft_tokens=k;
        opts.speculative.proposal_head=ninfer::ProposalHead::Optimized;
    }
#ifdef PRECISION_R5
    opts.v100_kernels=ninfer::V100KernelPolicy::Ported;
#endif
#ifndef PRECISION_V100
    opts.device_profile="off";
#endif
    ninfer::Engine engine(std::move(opts));
    while(std::getline(jobs,line)) {
      auto job=json::parse(line);
      auto started=std::chrono::steady_clock::now();
      json result={{"id",job.at("id")},{"mode",mode},{"k",k}};
      try {
        std::vector<ninfer::TokenId> ids=job.at("tokens").get<std::vector<ninfer::TokenId>>();
        if (job.contains("observer_dir")) setenv("NINFER_PRECISION_OBSERVER",job["observer_dir"].get<std::string>().c_str(),1);
        else unsetenv("NINFER_PRECISION_OBSERVER");
        if (job.contains("forced_tokens")) setenv("NINFER_PRECISION_FORCED",job["forced_tokens"].dump().c_str(),1);
        else unsetenv("NINFER_PRECISION_FORCED");
        if (mode=="score") {
          auto probs=engine.score_tokens(ids,job.at("first_target"));
          for(float probability:probs) if(!std::isfinite(probability)) throw std::runtime_error("non-finite target-token log probability");
          result["logprobs"]=probs;
          result["first_target"]=job.at("first_target");
          result["input_tokens"]=ids.size();
        } else {
          ninfer::RequestOptions request;
          request.execution.requested_output_tokens=job.value("max_tokens",128);
          request.execution.sampling.temperature=0;
          request.execution.sampling.top_k=0;
          request.execution.sampling.top_p=1;
          request.execution.sampling.min_p=0;
          request.execution.sampling.presence_penalty=0;
          request.execution.sampling.frequency_penalty=0;
          request.execution.sampling.seed=1729;
          request.execution.allow_prefix_reuse=job.value("reuse",false);
          request.output.raw=!job.contains("messages");
          auto prepared=engine.prepare_tokens(ids);
          if(job.contains("messages")) {
            ninfer::PromptInput input;
            input.options.enable_thinking=false;
            input.options.preserve_thinking=true;
            for(const auto& m:job["messages"]) {
              ninfer::ChatMessage message;
              auto role=m.at("role").get<std::string>();
              message.role=role=="system"?ninfer::ChatRole::System:role=="assistant"?ninfer::ChatRole::Assistant:ninfer::ChatRole::User;
              ninfer::MessagePart part;
              part.kind=ninfer::MessagePartKind::Text;
              part.text=m.at("content").get<std::string>();
              message.parts.push_back(std::move(part));
              input.messages.push_back(std::move(message));
            }
            prepared=engine.prepare(std::move(input));
          }
          auto res=engine.generate(std::move(prepared), request);
          result["tokens"]=res.generated_token_ids;
          result["content"]=res.content;
          result["reasoning"]=res.reasoning;
          result["reused_tokens"]=res.reused_prompt_tokens;
          result["drafted_tokens"]=res.speculative.drafted_tokens;
          result["accepted_tokens"]=res.speculative.accepted_tokens;
          result["prefill_seconds"]=res.timings.prefill_seconds;
        }
      } catch(const std::exception& e) {result["error"]=e.what();}
      result["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
      out << result.dump() << '\n' << std::flush;
      std::cerr << "PRECISION " << job.at("id") << " " << result["seconds"] << (result.contains("error")?" ERROR":" OK") << '\n' << std::flush;
      if(result.contains("error"))return 2;
    }
  } catch (const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
