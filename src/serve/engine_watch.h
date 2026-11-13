#pragma once

#include <atomic>
#include <cstdint>

namespace ninfer::serve {

// Serialize intentional HTTP shutdown against an Engine-failure observation.
// EngineCore currently latches has_failed during orderly request cancellation too;
// that must not turn a requested stop into the supervisor's failure/exit-2 path.
class EngineWatchState {
public:
    void begin_stop() noexcept {
        State expected = State::Serving;
        state_.compare_exchange_strong(expected, State::Stopping, std::memory_order_acq_rel);
    }

    // The first genuine serving failure owns the FATAL record and exit status.
    // Once failure won, a later intentional stop must not clear it.
    [[nodiscard]] bool record_failure() noexcept {
        State expected = State::Serving;
        return state_.compare_exchange_strong(expected, State::Failed, std::memory_order_acq_rel);
    }

    [[nodiscard]] bool has_failed() const noexcept {
        return state_.load(std::memory_order_acquire) == State::Failed;
    }

private:
    enum class State : std::uint8_t { Serving, Stopping, Failed };
    std::atomic<State> state_{State::Serving};
};

} // namespace ninfer::serve
