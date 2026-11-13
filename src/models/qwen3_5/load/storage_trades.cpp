#include "models/qwen3_5/load/bindings.h"

#include "artifact/permute_row_split.h"
#include "ops/common/v100_kernel_policy.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"

#include <initializer_list>

#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace ninfer::models::qwen3_5::loading {

// Device-memory trades selected at startup. Each stores a Q8 matrix at a narrower group-64 width
// while the weights load; the artifact is unchanged and the measured quality cost of each is in
// docs/maintainer/quality-trade-experiments.md.
void apply_storage_trades(Bindings& bindings, const Config& config, const ModelWeights& weights,
                          const LoadOptions& options) {
    if (options.lm_head_q4 && options.lm_head_q6) {
        throw std::invalid_argument("--lm-head-q4 and --lm-head-q6 are mutually exclusive");
    }
    if (options.embedding_q4 && options.embedding_q6) {
        throw std::invalid_argument("--embedding-q4 and --embedding-q6 are mutually exclusive");
    }
    if (options.embedding_residency == EmbeddingResidency::Host) {
        // The transcodes only shrink a device copy, and a host table holds no device memory.
        if (options.embedding_q4 || options.embedding_q6) {
            throw std::invalid_argument(
                "--embedding-residency host and --embedding-q4/--embedding-q6 cannot be combined");
        }
        // A tied table is also the output head, which a projection reads whole every step.
        const auto& embedding = bindings.at(weights.text.token_embedding).reference.binding;
        const auto& head      = bindings.at(weights.text.output_head).reference.binding;
        for (const auto& a : embedding.parts) {
            for (const auto& b : head.parts) {
                if (a.object.index == b.object.index) {
                    throw std::invalid_argument(
                        "--embedding-residency host needs an untied token embedding; this "
                        "artifact stores it in the same object as the output head");
                }
            }
        }
    }
    if (options.lm_head_q4 || options.lm_head_q6) {
        if (options.masked_draft()) {
            // DFlash's candidate top-k reads the Q8 output head through its own kernel.
            throw std::invalid_argument(
                "--lm-head-q4/--lm-head-q6 are not supported with DFlash or DFlash2");
        }
        // A head that already stores Q6 needs no work for --lm-head-q6; anything else must be
        // row-split Q8 (checked by Bindings::transcode).
        (void)bindings.transcode(weights.text.output_head,
                                 options.lm_head_q4 ? QType::Q4_G64_FP16 : QType::Q6_G64_FP16,
                                 options.lm_head_q4 ? "--lm-head-q4" : "--lm-head-q6");
    }
    if (options.embedding_q4 || options.embedding_q6) {
        (void)bindings.transcode(weights.text.token_embedding,
                                 options.embedding_q4 ? QType::Q4_G64_FP16 : QType::Q6_G64_FP16,
                                 options.embedding_q4 ? "--embedding-q4" : "--embedding-q6");
    }
    if (options.mtp_experts_q4) {
        if (!std::holds_alternative<MoeConfig>(config.text.ffn)) {
            throw std::invalid_argument(
                "--mtp-experts-q4 applies to a mixture-of-experts MTP layer (Qwen3_5MoeForCausalLM)");
        }
        // The MTP layer is bound only when MTP is selected; without it there is nothing to store.
        if (weights.mtp) {
            // Stored like the Text layers of the official 35B-A3B recipe: Q4 gate/up, Q6 down.
            // Drafts are verified exactly, so this can change acceptance but never the output
            // distribution.
            const auto& moe = std::get<MoeWeights>(weights.mtp->layer.ffn);
            for (const auto& expert : moe.experts) {
                (void)bindings.transcode(expert.gate, QType::Q4_G64_FP16, "--mtp-experts-q4");
                (void)bindings.transcode(expert.up, QType::Q4_G64_FP16, "--mtp-experts-q4");
                (void)bindings.transcode(expert.down, QType::Q6_G64_FP16, "--mtp-experts-q4");
            }
        }
    }
}

void apply_device_layouts(Bindings& bindings, const ModelWeights& weights,
                          std::span<const ops::DeviceTier> rank_tiers) {
    if constexpr (!ops::kBuildVolta) {
        (void)bindings;
        (void)weights;
        (void)rank_tiers;
        return;
    }
    // Row-split panel selection retains its existing projection ownership. FP8
    // QPN additionally proves the exact group that native preparation joins.
    const auto volta_rank = [&](std::size_t rank) {
        if (rank >= rank_tiers.size()) {
            throw std::logic_error("device layouts: no device tier for pipeline rank " +
                                   std::to_string(rank));
        }
        return rank_tiers[rank] == ops::DeviceTier::Volta;
    };
    const bool fp8_qpn=ops::v100_ported_kernels_enabled();
    std::vector<bool> selected(bindings.weights.size(), false);
    std::vector<WeightId> projections;
    std::vector<std::vector<WeightId>> fp8_groups;
    std::vector<bool> fp8_forbidden(bindings.weights.size(),false);
    const auto exclude=[&](WeightId id) {
        if (id.index<fp8_forbidden.size()) fp8_forbidden[id.index]=true;
    };
    // Role aliases may reuse a WeightId as well as a physical object. Preserve
    // the embedding/head veto even when no second PendingWeight records it.
    exclude(weights.text.token_embedding);
    exclude(weights.text.output_head);
    if (weights.mtp) {
        exclude(weights.mtp->token_embedding);
        exclude(weights.mtp->output_head);
    }
    if (weights.draft) {
        exclude(weights.draft->token_embedding);
        exclude(weights.draft->output_head);
    }
    if (weights.proposal) exclude(weights.proposal->head);
    const auto take = [&](WeightId id) {
        if (id.index < selected.size() && !selected[id.index]) {
            selected[id.index] = true;
            projections.push_back(id);
        }
    };
    const auto group=[&](std::initializer_list<WeightId> ids) {
        if (fp8_qpn) fp8_groups.emplace_back(ids);
    };
    const auto dense = [&](const DenseWeights& d) {
        take(d.gate);
        take(d.up);
        take(d.down);
        group({d.gate,d.up});
        group({d.down});
    };
    const auto ffn = [&](const BlockWeights& block) {
        if (const auto* d = std::get_if<DenseWeights>(&block.ffn)) {
            dense(*d);
        } else {
            const auto& moe=std::get<MoeWeights>(block.ffn);
            dense(moe.shared);
            exclude(moe.router);
            exclude(moe.shared_score);
            for (const auto& expert:moe.experts)
                for (const WeightId id:{expert.gate,expert.up,expert.down}) exclude(id);
        }
    };
    for (const auto& block : weights.text.layers) {
        if (const auto* a = std::get_if<AttentionWeights>(&block.mixer)) {
            for (const WeightId id : {a->query, a->key, a->gate, a->value, a->output}) { take(id); }
            group({a->query,a->key,a->gate,a->value});
            group({a->output});
        } else {
            const auto& g = std::get<GdnWeights>(block.mixer);
            for (const WeightId id : {g.query, g.key, g.value, g.z, g.output}) { take(id); }
            group({g.query,g.key,g.value,g.z});
            group({g.output});
        }
        ffn(block);
    }
    if (weights.mtp) {
        take(weights.mtp->input_projection);
        group({weights.mtp->input_projection});
        ffn(weights.mtp->layer);
    }
    std::unordered_map<std::size_t, bool> only_projections;
    for (std::size_t i = 0; i < bindings.weights.size(); ++i) {
        for (const auto& part : bindings.weights[i].reference.binding.parts) {
            const auto [entry, inserted] = only_projections.emplace(part.object.index, selected[i]);
            if (!inserted) { entry->second = entry->second && selected[i]; }
        }
    }
    // Every actual FP8 consumer group must concatenate a complete physical parent
    // in the same order as prepare_attn/gdn/linear_swiglu_weight. A union across
    // independent partial projections does not establish that contract.
    const auto complete_group=[&](std::span<const WeightId> ids) {
        std::optional<artifact::ObjectHandle> parent;
        std::uint64_t covered=0,elements=0,columns=0;
        for (const WeightId id:ids) {
            if (id.index>=bindings.weights.size()) return false;
            const auto& parameter=bindings.at(id);
            if (parameter.reference.residency!=artifact::Residency::Device ||
                !volta_rank(parameter.rank) || parameter.reference.shape.size()!=2) return false;
            for (const auto& part:parameter.reference.binding.parts) {
                if (bindings.binder.selected(part.object)) return false;
                const auto geometry=bindings.binder.geometry(part.object);
                if (!ops::detail::fp8_qpn_prepack_eligible(geometry) ||
                    parameter.reference.shape[1]!=geometry.shape[1]) return false;
                if (!parent) {
                    parent=part.object;
                    elements=geometry.elements;
                    columns=geometry.shape[1];
                }
                if (part.object!=*parent || geometry.shape[1]!=columns ||
                    part.begin!=covered || part.end<=part.begin || part.end>elements) return false;
                covered=part.end;
            }
        }
        return parent && covered==elements;
    };
    std::vector<bool> group_seen(bindings.weights.size(),false);
    std::vector<bool> group_safe(bindings.weights.size(),true);
    for (const auto& ids:fp8_groups) {
        const bool complete=complete_group(ids);
        for (const WeightId id:ids) {
            if (id.index>=bindings.weights.size()) continue;
            group_seen[id.index]=true;
            group_safe[id.index]=group_safe[id.index] && complete;
        }
    }
    std::unordered_map<std::size_t,bool> fp8_only_complete_groups;
    if (fp8_qpn) {
        for (std::size_t i=0;i<bindings.weights.size();++i) {
            for (const auto& part:bindings.weights[i].reference.binding.parts) {
                // Unselected embeddings, heads, experts and any other consumer
                // veto the whole object, including tied/shared physical parents.
                const bool safe=selected[i] && !fp8_forbidden[i] && group_seen[i] && group_safe[i];
                const auto [entry,inserted]=fp8_only_complete_groups.emplace(part.object.index,safe);
                if (!inserted) entry->second=entry->second && safe;
            }
        }
    }
    for (const WeightId id : projections) {
        const auto& parameter = bindings.at(id);
        if (parameter.reference.residency != artifact::Residency::Device ||
            !volta_rank(parameter.rank)) {
            continue;
        }
        for (const auto& part : parameter.reference.binding.parts) {
            const auto geometry = bindings.binder.geometry(part.object);
            if (fp8_qpn && ops::detail::fp8_qpn_prepack_eligible(geometry) &&
                fp8_only_complete_groups.at(part.object.index)) {
                bindings.binder.fp8_qpn_device(part.object);
            }
            const bool format =
                geometry.format == QType::T2_G128_FP16 || geometry.format == QType::Q2_G64_FP16 ||
                geometry.format == QType::Q4_G64_FP16 || geometry.format == QType::Q5_G64_FP16 ||
                geometry.format == QType::Q6_G64_FP16;
            if (format && only_projections.at(part.object.index) &&
                geometry.layout == QuantLayout::RowSplit &&
                artifact::row_split_panel_supported(geometry)) {
                bindings.binder.panel_device(part.object);
            }
        }
    }
}

} // namespace ninfer::models::qwen3_5::loading
