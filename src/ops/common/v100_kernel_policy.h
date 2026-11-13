#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::ops {

// Selected before weight materialization and fixed for the lifetime of the process: changing
// it would invalidate prepared weights, plans and captured CUDA Graphs.
[[nodiscard]] bool v100_ported_kernels_enabled() noexcept;
void set_v100_kernel_policy(bool ported);
void validate_v100_kernel_architecture(bool ported, int compute_capability);

// Host dispatch decisions, including decisions made while capturing a graph. These are route
// selections rather than GPU launch counts: graph replays do not visit host dispatch again.
struct V100KernelRouteCounts {
    std::uint64_t ported = 0;
    std::uint64_t fallback = 0;
};
void record_v100_kernel_route(bool ported) noexcept;
[[nodiscard]] V100KernelRouteCounts v100_kernel_route_counts() noexcept;

[[nodiscard]] constexpr std::string_view v100_kernel_policy_name(bool ported) noexcept {
    return ported ? "ported" : "native";
}

// Native retains the original fingerprint. The opt-in arithmetic needs a distinct identity
// even when it shares a product binary and artifact with native execution.
[[nodiscard]] constexpr std::string_view v100_kernel_cache_identity(bool ported) noexcept {
    return ported ? ";v100-kernels=ported-b37d0dd3-v1" : "";
}

} // namespace ninfer::ops
