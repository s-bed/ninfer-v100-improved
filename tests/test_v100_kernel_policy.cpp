#include "ops/common/v100_kernel_policy.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    using namespace ninfer::ops;
    const bool ported = argc < 2 || std::string(argv[1]) != "native";
    const bool w4a8 = argc > 1 && std::string(argv[1]) == "w4a8";
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            ++failures;
            std::cerr << message << '\n';
        }
    };
    if (argc > 1 && std::string(argv[1]) == "bad-mask") {
        try { set_v100_kernel_policy(true); }
        catch (const std::invalid_argument&) { return 0; }
        std::cerr << "malformed candidate mask was accepted\n";
        return 1;
    }
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
    check(v100_kernel_cache_identity(true, true) != v100_kernel_cache_identity(true, false),
          "W4A8 and A16 share persisted state");
    bool invalid = false;
    try { set_v100_kernel_policy(false, true); }
    catch (const std::invalid_argument&) { invalid = true; }
    check(invalid, "W4A8 accepted native execution");
    set_v100_kernel_policy(ported, w4a8);
    set_v100_kernel_policy(ported, w4a8);
    check(v100_w4a8_enabled() == w4a8, "W4A8 option lost");
    if (argc > 1 && std::string(argv[1]) == "flash-head") {
        check(v100_candidate_route_enabled(V100CandidateRoute::Head) &&
                  v100_candidate_route_enabled(V100CandidateRoute::Flash) &&
                  !v100_candidate_route_enabled(V100CandidateRoute::Drafter) &&
                  !v100_candidate_route_enabled(V100CandidateRoute::Gdn) &&
                  v100_kernel_cache_identity(true).ends_with("-9"),
              "partial route mask and cache identity disagree");
    }
    if (ported) {
        bool immutable = false;
        try { set_v100_kernel_policy(true, !w4a8); }
        catch (const std::invalid_argument&) { immutable = true; }
        check(immutable, "arithmetic changed after cache identity was fixed");
    }
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
