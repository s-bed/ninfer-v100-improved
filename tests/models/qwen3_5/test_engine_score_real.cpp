#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
#if defined(NINFER_SM8X_COMPAT)
    options.kv_cache      = ninfer::KvCacheStorage::BFloat16;
#else
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
#endif
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != options.kv_cache) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const float value : all) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(suffix[i]) || suffix[i] > 0.0F) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }
    // Stop the warmup at predictor 128, then score two groups of four and one tail.
    // Truncating immediately after the first group must give exactly its four probabilities.
    std::vector<ninfer::TokenId> short_window(tokens.begin(), tokens.begin() + 138);
    const auto narrow = engine.score_tokens(short_window, 129, 4);
    short_window.resize(133);
    const auto first_group = engine.score_tokens(short_window, 129, 4);
    if (narrow.size() != 9 || first_group.size() != 4 ||
        !std::all_of(narrow.begin(), narrow.end(),
                     [](float x) { return std::isfinite(x) && x <= 0; }) ||
        !std::equal(first_group.begin(), first_group.end(), narrow.begin())) {
        std::cerr << "narrow causal scoring changed the target range or its first group\n";
        return 1;
    }
    std::cout << "OK causal_score_real\n";
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
