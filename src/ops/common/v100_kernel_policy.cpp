#include "ops/common/v100_kernel_policy.h"

#include <atomic>
#include <stdexcept>

namespace ninfer::ops {
namespace {
// 0: unselected (the native default), 1: explicitly native, 2: explicitly ported.
std::atomic<unsigned> policy{0};
std::atomic<std::uint64_t> ported_routes{0};
std::atomic<std::uint64_t> fallback_routes{0};
} // namespace

bool v100_ported_kernels_enabled() noexcept {
    return policy.load(std::memory_order_relaxed) == 2;
}

void validate_v100_kernel_architecture(bool ported, int compute_capability) {
    if (ported && compute_capability != 70) {
        throw std::invalid_argument("--v100-kernels ported requires sm_70 on every device");
    }
}

void set_v100_kernel_policy(bool ported) {
    const unsigned requested = ported ? 2 : 1;
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

V100KernelRouteCounts v100_kernel_route_counts() noexcept {
    return {ported_routes.load(std::memory_order_relaxed),
            fallback_routes.load(std::memory_order_relaxed)};
}

} // namespace ninfer::ops
