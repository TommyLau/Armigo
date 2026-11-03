// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Monotonic host sampling and cancellable deadline waits share one 19.2 MHz
// timeline: deadlines round up, samples round down, and a clock failure never
// becomes a fabricated advancing sample.

#include <armigo/runtime/guest_clock.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stop_token>
#include <type_traits>
#include <utility>
#include <variant>

namespace armigo::runtime {
namespace {

// The firmware 20.5.0 guest counter frequency in ticks per second.
// Frequency changes and clock regression are
// unsupported by this contract.
constexpr std::uint64_t counter_frequency = 19'200'000;

// Exact tick scale: 1e9 ns / 19.2e6 ticks per second equals 625 ns per 12
// ticks. Conversions stay in exact integer arithmetic on this pair.
constexpr std::uint64_t nanoseconds_per = 625;
constexpr std::uint64_t ticks_per = 12;

static_assert(std::is_same_v<std::chrono::steady_clock::duration,
                             std::chrono::nanoseconds>,
              "ClockInstant conversions assume a nanoseconds steady clock");

LaunchFailure make_failure(LaunchStage stage, LaunchError code,
                           const char* operation, std::string detail) {
    LaunchFailure result;
    result.stage = stage;
    result.code = code;
    result.operation = operation;
    result.detail = std::move(detail);
    return result;
}

LaunchFailure overflow_failure(const char* operation) {
    return make_failure(LaunchStage::services, LaunchError::resource_limit,
                        operation,
                        "the value does not convert without wraparound");
}

// floor(ticks * 625 / 12), refused when the result leaves the signed
// nanoseconds domain. Sampled ticks always convert downward.
std::optional<std::chrono::nanoseconds> ticks_to_nanoseconds_floor(
    std::uint64_t ticks) {
    const std::uint64_t whole = ticks / ticks_per;
    const std::uint64_t part =
        (ticks % ticks_per) * nanoseconds_per / ticks_per;
    const std::uint64_t limit =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()) - part;
    if (whole > limit / nanoseconds_per)
        return std::nullopt;
    return std::chrono::nanoseconds{
        static_cast<std::int64_t>(whole * nanoseconds_per + part)};
}

// floor(ns * 12 / 625) for a nonnegative nanosecond count. Sampled elapsed
// time always converts downward.
std::uint64_t nanoseconds_to_ticks_floor(std::chrono::nanoseconds duration) {
    const auto count = static_cast<std::uint64_t>(duration.count());
    const std::uint64_t whole = count / nanoseconds_per;
    return whole * ticks_per + (count % nanoseconds_per) * ticks_per /
                                   nanoseconds_per;
}

// ceil(ns * 12 / 625). Deadline conversions always round up so a deadline
// never lands before the requested time.
std::optional<std::uint64_t> nanoseconds_to_ticks_ceil(
    std::chrono::nanoseconds duration) {
    const auto count = static_cast<std::uint64_t>(duration.count());
    const std::uint64_t whole = count / nanoseconds_per;
    const std::uint64_t part =
        ((count % nanoseconds_per) * ticks_per + nanoseconds_per - 1) /
        nanoseconds_per;
    if (whole > (std::numeric_limits<std::uint64_t>::max() - part) / ticks_per)
        return std::nullopt;
    return whole * ticks_per + part;
}

class GuestClockImpl final : public GuestClock {
public:
    GuestClockImpl(GuestClockSources sources, ClockInstant origin)
        : sources_(std::move(sources)), origin_(origin) {}

    LaunchResult<ClockSample> sample() const override {
        if (auto held = latched())
            return std::move(*held);
        return sample_and_latch();
    }

    execution::GuestCounter counter() noexcept override {
        return {&GuestClockImpl::read_ticks, this, counter_frequency};
    }

    LaunchResult<std::chrono::nanoseconds> ticks_to_nanoseconds(
        std::uint64_t ticks) const override {
        if (auto held = latched())
            return std::move(*held);
        if (auto converted = ticks_to_nanoseconds_floor(ticks))
            return *converted;
        return overflow_failure("guest_clock.ticks_to_nanoseconds");
    }

    LaunchResult<std::uint64_t> nanoseconds_to_ticks(
        std::chrono::nanoseconds duration) const override {
        if (auto held = latched())
            return std::move(*held);
        if (duration.count() < 0)
            return make_failure(LaunchStage::services, LaunchError::invalid_input,
                                "guest_clock.nanoseconds_to_ticks",
                                "deadline durations are nonnegative");
        if (auto ticks = nanoseconds_to_ticks_ceil(duration))
            return *ticks;
        return overflow_failure("guest_clock.nanoseconds_to_ticks");
    }

    LaunchResult<ClockWait> wait_until(ClockInstant deadline,
                                       std::stop_token stop) override {
        if (auto held = latched())
            return std::move(*held);
        try {
            if (sources_.wait) {
                auto result = sources_.wait(deadline, std::move(stop));
                if (auto* failure = std::get_if<LaunchFailure>(&result)) {
                    latch(*failure);
                    return std::move(*failure);
                }
                return std::get<ClockWait>(std::move(result));
            }
            // Cancellable wait on the production monotonic timeline. This
            // libc++ carries no stop-token condition-variable overloads, so
            // the cancellation is wired by hand: the stop callback takes the
            // wait mutex before flagging and notifying, and the waiter
            // re-checks the flag under the same mutex after registering, so
            // no wake-up can be lost in the registration window. The callback
            // may run inline during registration; the mutex is not held yet
            // at that point.
            if (stop.stop_requested())
                return ClockWait::cancelled;
            bool stop_requested = false;
            std::stop_callback callback(stop, [&] {
                std::scoped_lock guard(wait_mutex_);
                stop_requested = true;
                wait_cv_.notify_all();
            });
            std::unique_lock lock(wait_mutex_);
            while (!stop_requested) {
                if (wait_cv_.wait_until(lock, deadline) ==
                    std::cv_status::timeout) {
                    return stop.stop_requested() ? ClockWait::cancelled
                                                 : ClockWait::reached;
                }
            }
            return ClockWait::cancelled;
        } catch (...) {
            auto failure = make_failure(LaunchStage::services,
                                        LaunchError::host_failure,
                                        "guest_clock.wait_until",
                                        "the clock wait raised an exception");
            latch(failure);
            return failure;
        }
    }

private:
    // The no-throw counter adapter. While the clock is healthy it delivers the
    // current monotonic sample. Once a source error latched, it freezes at the
    // last valid value (zero before the first valid sample): the value is a
    // valid counter reading, but it never advances again, so no guest observer
    // can mistake a fabricated value for fresh progress.
    static std::uint64_t read_ticks(void* context) noexcept {
        auto& self = *static_cast<GuestClockImpl*>(context);
        if (self.failed_.load(std::memory_order_acquire))
            return self.last_ticks_.load(std::memory_order_relaxed);
        try {
            auto sampled = self.sample();
            if (std::holds_alternative<LaunchFailure>(sampled))
                return self.last_ticks_.load(std::memory_order_relaxed);
            return std::get<ClockSample>(sampled).ticks;
        } catch (...) {
            // Failure reporting could not allocate; keep the frozen value.
            return self.last_ticks_.load(std::memory_order_relaxed);
        }
    }

    std::optional<LaunchFailure> latched() const {
        std::scoped_lock lock(latch_mutex_);
        return latched_failure_;
    }

    void latch(LaunchFailure failure) const {
        std::scoped_lock lock(latch_mutex_);
        if (!latched_failure_)
            latched_failure_ = std::move(failure);
        failed_.store(true, std::memory_order_release);
    }

    LaunchResult<ClockSample> sample_and_latch() const {
        ClockInstant instant{};
        try {
            instant = sources_.now ? sources_.now()
                                   : std::chrono::steady_clock::now();
        } catch (...) {
            auto failure = make_failure(LaunchStage::services,
                                        LaunchError::host_failure,
                                        "guest_clock.sample",
                                        "the clock source raised an exception");
            latch(failure);
            return failure;
        }
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(instant -
                                                                 origin_);
        if (elapsed.count() < 0) {
            auto failure = make_failure(
                LaunchStage::services, LaunchError::host_failure,
                "guest_clock.sample",
                "the clock source regressed before the creation origin");
            latch(failure);
            return failure;
        }
        const std::uint64_t ticks = nanoseconds_to_ticks_floor(elapsed);
        const auto counter_time = ticks_to_nanoseconds_floor(ticks);
        if (!counter_time) {
            // Unreachable for a count that fit the nanoseconds domain; kept
            // structured so no path can wrap silently.
            auto failure = overflow_failure("guest_clock.sample");
            latch(failure);
            return failure;
        }
        // Publish monotonically so a later failure freezes the no-throw
        // counter adapter at the newest committed value. A concurrent thread
        // may already have published a later sample; that is scheduling, not
        // regression, and this call still reports its own instant.
        std::uint64_t previous =
            last_ticks_.load(std::memory_order_relaxed);
        while (ticks > previous &&
               !last_ticks_.compare_exchange_weak(previous, ticks,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed)) {
        }
        ClockSample sample;
        sample.instant = instant;
        sample.ticks = ticks;
        sample.counter_time = *counter_time;
        return sample;
    }

    GuestClockSources sources_;
    ClockInstant origin_;
    mutable std::mutex latch_mutex_;
    mutable std::optional<LaunchFailure> latched_failure_;
    mutable std::atomic<bool> failed_{false};
    mutable std::atomic<std::uint64_t> last_ticks_{0};
    // Serves production waits only; injected waits replace it entirely.
    std::condition_variable_any wait_cv_;
    std::mutex wait_mutex_;
};

} // namespace

LaunchResult<std::shared_ptr<GuestClock>> create_guest_clock(
    GuestClockSources sources) {
    if (static_cast<bool>(sources.now) != static_cast<bool>(sources.wait))
        return make_failure(LaunchStage::inputs, LaunchError::invalid_input,
                            "create_guest_clock",
                            "clock sources must be both empty or both provided");
    try {
        const auto origin =
            sources.now ? sources.now() : std::chrono::steady_clock::now();
        return std::make_shared<GuestClockImpl>(std::move(sources), origin);
    } catch (const std::bad_alloc&) {
        return make_failure(LaunchStage::inputs, LaunchError::host_failure,
                            "create_guest_clock",
                            "the clock origin could not be allocated");
    } catch (...) {
        return make_failure(LaunchStage::inputs, LaunchError::host_failure,
                            "create_guest_clock",
                            "the clock source raised an exception");
    }
}

} // namespace armigo::runtime
