#pragma once

#include "core/host_pages.h"
#include "core/startup.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::runtime::prefix_cache {

// Whole slabs never cross a pinned allocation. Chunk ids also bound coalesced DMA runs.
class HostSlabs {
public:
    void allocate(std::uint32_t count, std::size_t bytes_per_slab, StartupPhaseScope* pin_progress = nullptr) {
        constexpr std::size_t kHostChunkBytes = std::size_t{4} << 30U;
        if (count == 0) { return; }
        if (bytes_per_slab == 0) { throw std::invalid_argument("a host cache slab cannot be empty"); }
        slab_bytes_ = bytes_per_slab;
        slabs_per_chunk_ = static_cast<std::uint32_t>(
            std::clamp<std::size_t>(kHostChunkBytes / slab_bytes_, 1U, count));
        const std::uint64_t total = static_cast<std::uint64_t>(count) * slab_bytes_;
        std::uint64_t pinned      = 0;
        host_chunks_.reserve(1U + (count - 1U) / slabs_per_chunk_);
        for (std::uint32_t first = 0; first < count;) {
            const std::uint32_t slabs = std::min(slabs_per_chunk_, count - first);
            const std::size_t bytes   = static_cast<std::size_t>(slabs) * slab_bytes_;
            try {
                host_chunks_.push_back(std::make_unique<HostPages>(bytes));
                host_chunks_.back()->pin();
            } catch (const std::exception& error) {
                throw std::runtime_error("pinning the hybrid prefix cache Host tier failed after " +
                                         std::to_string(pinned >> 20U) + " of " +
                                         std::to_string(total >> 20U) + " MiB (" + error.what() +
                                         "); lower --host-cache-mib");
            }
            first += slabs;
            pinned += bytes;
            if (pin_progress != nullptr) { pin_progress->progress(pinned, total); }
        }
    }

    [[nodiscard]] bool empty() const { return host_chunks_.empty(); }
    [[nodiscard]] std::uint32_t chunk(std::uint32_t id) const { return id / slabs_per_chunk_; }
    [[nodiscard]] std::byte* data(std::uint32_t id) const {
        return host_chunks_[chunk(id)]->data() + (id % slabs_per_chunk_) * slab_bytes_;
    }

private:
    std::vector<std::unique_ptr<HostPages>> host_chunks_;
    std::uint32_t slabs_per_chunk_ = 0;
    std::size_t slab_bytes_ = 0;
};

} // namespace ninfer::runtime::prefix_cache
