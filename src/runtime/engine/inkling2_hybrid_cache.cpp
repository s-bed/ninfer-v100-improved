#include "runtime/engine/inkling2_hybrid_cache.h"
#include "core/copy_batch.h"
#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {
namespace pc = prefix_cache;

Inkling2HybridCache::Inkling2HybridCache(Text& text, Draft* draft, std::size_t budget,
                                         std::uint32_t capacity, cudaStream_t stream)
    : text_(text), draft_(draft), stream_(stream), global_(text.global_cache_rings()) {
    if (draft) {
        auto rings = draft->global_cache_rings();
        global_.insert(global_.end(), rings.begin(), rings.end());
    }
    for (const auto& t : global_) { slab_bytes_ += pc::kBlockTokens * t.nb[1]; }
    if (!slab_bytes_) { throw std::logic_error("Inkling-2 hybrid cache has no global KV planes"); }
    const auto image_bytes = text.snapshot_bytes() + (draft ? draft->snapshot_bytes() : 0);
    const auto image_slabs = (image_bytes + slab_bytes_ - 1) / slab_bytes_;
    const auto count = budget / slab_bytes_;
    const auto device_blocks = capacity / pc::kBlockTokens;
    if (count + device_blocks + 1 >= pc::kNoId / 2 || image_slabs >= pc::kNoId) {
        throw std::invalid_argument("Inkling-2 host cache budget exceeds the index limits");
    }
    pc::PrefixIndexConfig config;
    config.host_slabs = static_cast<std::uint32_t>(count);
    config.max_nodes = config.host_slabs + device_blocks + 1;
    config.image_slabs = static_cast<std::uint32_t>(image_slabs);
    config.max_snapshots = config.host_slabs + 1;
    config.block_bytes = slab_bytes_;
    config.image_bytes = image_bytes;
    host_.allocate(config.host_slabs, slab_bytes_);
    index_.emplace(config, static_cast<pc::PrefixIndexBackend&>(*this));
}

std::vector<DeviceSpan> Inkling2HybridCache::block(std::uint32_t ordinal) const {
    std::vector<DeviceSpan> regions;
    regions.reserve(global_.size());
    for (const auto& ring : global_) {
        const auto bytes = static_cast<std::size_t>(ring.nb[1]) * pc::kBlockTokens;
        regions.push_back({static_cast<std::byte*>(ring.data) + ordinal * bytes, bytes});
    }
    return regions;
}

std::vector<DeviceSpan> Inkling2HybridCache::image(std::uint32_t frontier) const {
    auto spans = text_.cache_image(frontier);
    if (draft_) {
        auto draft = draft_->cache_image(frontier);
        spans.insert(spans.end(), draft.begin(), draft.end());
    }
    return spans;
}

std::size_t Inkling2HybridCache::transfer(std::span<const DeviceSpan> regions,
                                          std::span<const std::uint32_t> slabs, bool restore) {
    std::size_t offset = 0;
    CopyBatch copies(stream_);
    for (auto region : regions) {
        auto* device = static_cast<std::byte*>(region.data);
        while (region.bytes) {
            if (offset / slab_bytes_ >= slabs.size()) {
                throw std::logic_error("Inkling cache image exceeds its slabs");
            }
            const auto in_slab = offset % slab_bytes_;
            auto* host = host_.data(slabs[offset / slab_bytes_]) + in_slab;
            const auto n = std::min(region.bytes, slab_bytes_ - in_slab);
            copies.add(restore ? device : host, restore ? host : device, n);
            offset += n;
            device += n;
            region.bytes -= n;
        }
    }
    try {
        copies.flush();
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    } catch (...) {
        (void)cudaStreamSynchronize(stream_);
        throw;
    }
    return offset;
}

std::uint32_t Inkling2HybridCache::resume(std::span<const TokenId> tokens,
                                          std::span<const std::uint64_t> keys,
                                          bool reuse) {
    finish_request();
    extras_.assign(tokens.size() / pc::kBlockTokens, 0);
    for (std::size_t i = 0; i < extras_.size() * pc::kBlockTokens; ++i) {
        if (keys[i] != static_cast<std::uint64_t>(tokens[i])) {
            auto& extra = extras_[i / pc::kBlockTokens];
            extra = (extra ^ keys[i]) * 1099511628211ULL;
        }
    }
    hashes_ = pc::block_lookup_hashes(tokens, extras_);
    const auto match = index_->match(tokens, hashes_, extras_, static_cast<std::uint32_t>(tokens.size()));
    const auto choice = index_->choose(match, static_cast<std::uint32_t>(tokens.size()));
    if (!reuse || !choice.candidate) {
        index_->evict_device_blocks(pc::kNoId);
        return 0;
    }
    const auto& source = match.candidates[*choice.candidate];
    path_.assign(match.path.begin(), match.path.begin() + source.path_blocks);
    index_->acquire_path(path_);
    index_->evict_device_blocks(pc::kNoId);
    for (std::uint32_t i = 0; i < path_.size(); ++i) {
        const auto view = index_->node(path_[i]);
        if (view.device == pc::CopyState::Resident) { continue; }
        counters_.hybrid_host_restore_bytes += transfer(block(i), {&view.host_slab, 1}, true);
        index_->begin_device_fill(path_[i], i);
        index_->complete_device_fill(path_[i]);
        ++counters_.hybrid_host_block_restores;
    }
    const auto snapshot = index_->snapshot(source.snapshot);
    counters_.hybrid_host_restore_bytes += transfer(image(source.frontier), snapshot.host_slabs, true);
    text_.finish_cache_restore(source.frontier);
    if (draft_) { draft_->finish_cache_restore(source.frontier); }
    previous_ = source.snapshot;
    index_->note_hit(previous_);
    ++counters_.hybrid_host_image_restores;
    counters_.hybrid_reused_tokens += source.frontier;
    return source.frontier;
}

void Inkling2HybridCache::capture(std::span<const TokenId> tokens, std::uint32_t frontier) {
    if (frontier != text_.position() || (draft_ && frontier != draft_->position()) ||
        frontier % pc::kBlockTokens || frontier >= tokens.size()) {
        throw std::logic_error("Inkling hybrid cache capture outside a committed prompt boundary");
    }
    const auto base = index_->valid(previous_) ? index_->snapshot(previous_).frontier : 0;
    const auto state = image(frontier);
    std::uint64_t image_bytes = 0;
    for (const auto& region : state) { image_bytes += region.bytes; }
    const auto claim = index_->estimate_priority(base, frontier, false, 0, image_bytes);
    path_.reserve(frontier / pc::kBlockTokens);
    for (std::uint32_t i = path_.size(); i < frontier / pc::kBlockTokens; ++i) {
        const auto parent = path_.empty() ? pc::NodeRef{} : path_.back();
        const auto ids = tokens.subspan(i * pc::kBlockTokens, pc::kBlockTokens);
        auto inserted = index_->insert_block(parent, hashes_[i], ids, extras_[i], i);
        path_.push_back(inserted.node);
        if (inserted.inserted) { ++counters_.hybrid_blocks_inserted; }
        else if (inserted.device_attached) { ++counters_.hybrid_blocks_reattached; }
        else { ++counters_.hybrid_blocks_duplicate; }
    }
    // Retry unbacked blocks from an earlier tap whose host reservation was declined.
    for (std::uint32_t i = 0; i < path_.size(); ++i) {
        const auto node = path_[i];
        if (index_->node(node).host == pc::CopyState::Resident) { continue; }
        const auto slab = index_->begin_host_fill(node, previous_, claim);
        if (!slab) { ++counters_.hybrid_taps_skipped; return; }
        try {
            counters_.hybrid_host_write_bytes += transfer(block(i), {&*slab, 1}, false);
            index_->complete_host_fill(node);
            ++counters_.hybrid_host_block_writes;
        } catch (...) {
            index_->abort_host_fill(node);
            throw;
        }
    }
    const auto result = index_->capture_host_image(path_.back(), frontier, pc::SnapshotKind::Tap, image_bytes,
        [&](auto slabs) { counters_.hybrid_host_write_bytes += transfer(state, slabs, false); });
    if (result) {
        if (index_->valid(previous_)) { (void)index_->supersede(previous_, pc::Supersession::NextTap); }
        previous_ = *result;
        ++counters_.hybrid_host_image_writes;
        ++counters_.hybrid_taps_created;
    }
}

void Inkling2HybridCache::finish_request() {
    index_->release_path(path_);
    path_.clear();
    previous_ = {};
}

void Inkling2HybridCache::report(RuntimeStats& s) const {
    const auto v = index_->stats();
    s.hybrid_cached_blocks = v.device_resident_blocks;
    s.hybrid_evictable_blocks = v.device_evictable_blocks;
    s.hybrid_tree_blocks = v.nodes;
    s.hybrid_snapshots = v.snapshots;
    s.hybrid_host_capacity_bytes = index_->config().host_slabs * slab_bytes_;
    s.hybrid_host_used_bytes = (index_->config().host_slabs - v.host_free_slabs) * slab_bytes_;
    s.hybrid_snapshot_hits = v.snapshot_hits;
    s.hybrid_evicted_blocks = v.device_block_evictions;
    s.hybrid_host_snapshot_evictions = v.host_snapshot_evictions;
    s.hybrid_host_dead_reclaims = v.host_dead_reclaims;
    s.hybrid_unbacked_node_losses = v.unbacked_node_losses;
    s.hybrid_reused_tokens = counters_.hybrid_reused_tokens;
    s.hybrid_blocks_inserted = counters_.hybrid_blocks_inserted;
    s.hybrid_blocks_reattached = counters_.hybrid_blocks_reattached;
    s.hybrid_blocks_duplicate = counters_.hybrid_blocks_duplicate;
    s.hybrid_taps_created = counters_.hybrid_taps_created;
    s.hybrid_taps_skipped = counters_.hybrid_taps_skipped;
    s.hybrid_host_image_writes = counters_.hybrid_host_image_writes;
    s.hybrid_host_block_writes = counters_.hybrid_host_block_writes;
    s.hybrid_host_image_restores = counters_.hybrid_host_image_restores;
    s.hybrid_host_block_restores = counters_.hybrid_host_block_restores;
    s.hybrid_host_write_bytes = counters_.hybrid_host_write_bytes;
    s.hybrid_host_restore_bytes = counters_.hybrid_host_restore_bytes;
}
} // namespace ninfer::runtime
