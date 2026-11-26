#include "ops/common/v100_kernel_policy.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops {
namespace {
// Commit all process-lifetime options together before preparing any weights or graphs.
constexpr unsigned kNative = 1;
constexpr unsigned kPorted = 2;
constexpr unsigned kW4A8 = 4;
constexpr unsigned kRouteShift = 3;
constexpr unsigned kAllRoutes = 15;
unsigned candidate_mask() {
    static const unsigned mask = [] {
        const char* raw = std::getenv("NINFER_V100_CANDIDATES");
        if (raw == nullptr) { return kAllRoutes; }
        std::string_view names(raw);
        if (names == "none") { return 0U; }
        constexpr std::array<std::string_view, 4> routes{"head", "drafter", "gdn", "flash"};
        unsigned bits = 0;
        for (;;) {
            const auto comma = names.find(',');
            const auto name = names.substr(0, comma);
            unsigned bit = 1;
            for (auto route : routes) {
                if (name == route) { break; }
                bit <<= 1;
            }
            if (bit > kAllRoutes || (bits & bit) != 0) {
                throw std::invalid_argument("invalid or repeated NINFER_V100_CANDIDATES route");
            }
            bits |= bit;
            if (comma == std::string_view::npos) { return bits; }
            names.remove_prefix(comma + 1);
        }
    }();
    return mask;
}
std::atomic<unsigned> policy{0};
std::array<std::atomic<std::uint64_t>, 6> selections{};
std::atomic<std::uint64_t> ported_routes{0};
std::atomic<std::uint64_t> fallback_routes{0};
} // namespace

bool v100_ported_kernels_enabled() noexcept {
    return (policy.load(std::memory_order_relaxed) & kPorted) != 0;
}

void validate_v100_kernel_architecture(bool ported, int compute_capability) {
    if (ported && compute_capability != 70) {
        throw std::invalid_argument("--v100-kernels ported requires sm_70 on every device");
    }
}

bool v100_candidate_route_enabled(V100CandidateRoute route) noexcept {
    return (policy.load(std::memory_order_relaxed) &
            (static_cast<unsigned>(route) << kRouteShift)) != 0;
}

bool v100_w4a8_enabled() noexcept {
    return (policy.load(std::memory_order_relaxed) & kW4A8) != 0;
}

std::string v100_kernel_cache_identity(bool ported, bool w4a8) {
    if (!ported) { return {}; }
    return std::string(";v100-kernels=decode-prefill-r2-") + (w4a8 ? "w4a8-" : "a16-") +
           std::to_string(candidate_mask());
}

void set_v100_kernel_policy(bool ported, bool w4a8) {
    if (w4a8 && !ported) {
        throw std::invalid_argument("--nvfp4-w4a8-int requires --v100-kernels ported");
    }
    const unsigned requested = ported ? (kPorted | (w4a8 ? kW4A8 : 0U) |
                                         (candidate_mask() << kRouteShift)) : kNative;
    unsigned expected = 0;
    if (!policy.compare_exchange_strong(expected, requested, std::memory_order_relaxed) &&
        expected != requested) {
        throw std::invalid_argument("v100 kernel policy is fixed for this process; restart to change it");
    }
}

void record_v100_kernel_route(bool ported) noexcept {
    if (!v100_ported_kernels_enabled()) { return; }
    (ported ? ported_routes : fallback_routes).fetch_add(1, std::memory_order_relaxed);
}

void record_v100_selection(V100ObservedRoute route) noexcept {
    selections[static_cast<unsigned>(route)].fetch_add(1, std::memory_order_relaxed);
}

V100KernelRouteCounts v100_kernel_route_counts() noexcept {
    return {selections[0].load(std::memory_order_relaxed), selections[1].load(std::memory_order_relaxed),
            selections[2].load(std::memory_order_relaxed), selections[3].load(std::memory_order_relaxed),
            selections[4].load(std::memory_order_relaxed), selections[5].load(std::memory_order_relaxed),
            ported_routes.load(std::memory_order_relaxed),
            fallback_routes.load(std::memory_order_relaxed)};
}

} // namespace ninfer::ops
