#pragma once

#include "models/inkling2/program/dspark_program.h"
#include "models/inkling2/program/text_program.h"
#include "ninfer/observability/snapshot.h"
#include "runtime/prefix_cache/host_slabs.h"
#include "runtime/prefix_cache/prefix_index.h"

namespace ninfer::runtime {

// One worker owns this cache and its model's contiguous device lane. Block ids are the
// absolute block positions in that lane. A new request relinquishes every row it may overwrite.
class Inkling2HybridCache final : private prefix_cache::PrefixIndexBackend {
public:
    using Text = models::inkling2::TextProgram;
    using Draft = models::inkling2::DSparkProgram;
    Inkling2HybridCache(Text& text, Draft* draft, std::size_t budget, std::uint32_t capacity,
                        cudaStream_t stream);
    std::uint32_t resume(std::span<const TokenId> tokens, std::span<const std::uint64_t> keys,
                         bool reuse);
    void capture(std::span<const TokenId> tokens, std::uint32_t frontier);
    void finish_request();
    void report(RuntimeStats& stats) const;
    [[nodiscard]] std::size_t block_bytes() const { return slab_bytes_; }
private:
    void release_device_block(std::uint32_t) noexcept override {}
    [[nodiscard]] std::vector<DeviceSpan> image(std::uint32_t frontier) const;
    [[nodiscard]] std::vector<DeviceSpan> block(std::uint32_t ordinal) const;
    std::size_t transfer(std::span<const DeviceSpan> regions,
                         std::span<const std::uint32_t> slabs, bool host_to_device);

    Text& text_;
    Draft* draft_;
    cudaStream_t stream_;
    std::vector<Tensor> global_;
    std::size_t slab_bytes_ = 0;
    prefix_cache::HostSlabs host_;
    std::optional<prefix_cache::PrefixCacheIndex> index_;
    std::vector<prefix_cache::NodeRef> path_;
    std::vector<std::uint64_t> extras_, hashes_;
    prefix_cache::SnapshotRef previous_;
    RuntimeStats counters_;
};

} // namespace ninfer::runtime
