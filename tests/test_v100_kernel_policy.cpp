#include "ops/common/v100_kernel_policy.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    using namespace ninfer::ops;
    const bool ported = argc < 2 || std::string(argv[1]) != "native";
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            ++failures;
            std::cerr << message << '\n';
        }
    };
    check(!v100_ported_kernels_enabled(), "default policy is not native");
    check(v100_kernel_cache_identity(false).empty(), "native changed the vendor cache identity");
    check(!v100_kernel_cache_identity(true).empty(), "ported policy shares native cache identity");
    const std::string native_fingerprint = "artifact=A;build=B";
    const std::string ported_fingerprint = native_fingerprint +
                                           std::string(v100_kernel_cache_identity(true));
    check(native_fingerprint != ported_fingerprint, "same-binary native/ported cache collision");
    for (const int capability : {70, 75, 80, 86, 89, 120}) {
        validate_v100_kernel_architecture(false, capability);
        bool rejected = false;
        try { validate_v100_kernel_architecture(true, capability); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected == (capability != 70), "ported architecture admission is incorrect");
    }
    set_v100_kernel_policy(ported);
    set_v100_kernel_policy(ported);
    check(v100_ported_kernels_enabled() == ported, "selected policy was not installed");
    record_v100_kernel_route(true);
    record_v100_kernel_route(true);
    record_v100_kernel_route(false);
    const auto counts = v100_kernel_route_counts();
    check(counts.ported == (ported ? 2U : 0U) && counts.fallback == (ported ? 1U : 0U),
          "route selections were counted incorrectly");
    bool rejected = false;
    try { set_v100_kernel_policy(!ported); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected && v100_ported_kernels_enabled() == ported,
          "policy changed after preparing weights or graphs");
    return failures == 0 ? 0 : 1;
}
