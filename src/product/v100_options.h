#pragma once

#include "ninfer/engine_options.h"

#include <stdexcept>
#include <string_view>

namespace ninfer::product {

inline V100KernelPolicy parse_v100_kernel_policy(std::string_view value) {
    if (value == "native") { return V100KernelPolicy::Native; }
    if (value == "ported") { return V100KernelPolicy::Ported; }
    throw std::invalid_argument("--v100-kernels must be native or ported");
}

inline void validate_v100_options(V100KernelPolicy policy, bool w4a8) {
    if (w4a8 && policy != V100KernelPolicy::Ported) {
        throw std::invalid_argument("--nvfp4-w4a8-int requires --v100-kernels ported");
    }
}

} // namespace ninfer::product
