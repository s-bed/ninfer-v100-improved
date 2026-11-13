#include "serve/engine_watch.h"

#include <atomic>
#include <iostream>
#include <latch>
#include <thread>
#include <vector>

namespace {

using ninfer::serve::EngineWatchState;

int check(bool value, const char* message) {
    if (value) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The watchdog's production decision: a claimed serving failure writes FATAL,
// while main() chooses exit 2 only from the latched failure state.
struct Watch {
    EngineWatchState state;
    std::vector<const char*> records;
    void observe(bool engine_failed) {
        if (engine_failed && state.record_failure()) { records.push_back("FATAL"); }
    }
    int exit_status() const { return state.has_failed() ? 2 : 0; }
};

} // namespace

int main() {
    int failures = 0;
    {
        // Force the observed vendor interleaving: stop has been requested, the
        // worker latches failed while saving its cache, and the watch polls before
        // that save completes. No sleep or probabilistic scheduling is involved.
        Watch watch;
        std::atomic<bool> engine_failed{false};
        std::latch saving{1};
        std::latch checked{1};
        std::thread stop([&] {
            watch.state.begin_stop();
            engine_failed.store(true, std::memory_order_release);
            saving.count_down();
            checked.wait(); // The orderly cache save remains in progress.
        });
        saving.wait();
        watch.observe(engine_failed.load(std::memory_order_acquire));
        failures += check(watch.records.empty(), "intentional cache-save shutdown logged FATAL");
        failures += check(watch.exit_status() == 0, "intentional shutdown selected exit 2");
        checked.count_down();
        stop.join();
    }
    {
        Watch watch;
        watch.observe(false);
        failures += check(watch.records.empty() && watch.exit_status() == 0,
                          "a healthy engine was reported failed");
        watch.observe(true);
        watch.state.begin_stop(); // A supervisor stop cannot erase a real failure.
        watch.observe(true);
        failures += check(watch.records.size() == 1 && watch.exit_status() == 2,
                          "a real engine failure lost its FATAL record or exit 2");
    }
    if (failures == 0) { std::cout << "engine watch shutdown/failure tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
