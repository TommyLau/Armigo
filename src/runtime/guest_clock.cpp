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
//
// Production waits end at their deadlines, as Horizon's timers do. The host
// coalesces an ordinary thread's timed wait by about half its length (a
// 16.7 ms wait ended 5 to 8 ms late on Apple Silicon at every QoS class), so
// one alarm thread, declared time-constrained so its own waits end on time,
// wakes each waiter whose registered deadline passed, and only that waiter.
// Waits of at most 500 us are not registered: the host's own wake-up ends
// them at most about 0.25 ms late. The waiting threads keep their own
// scheduling policy and QoS class: a Mach policy change would end pthread QoS
// for good.

#include <armigo/runtime/guest_clock.h>

#include <armigo/common/diagnostics.h>
#include <armigo/platform/presentation_thread.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

namespace armigo::runtime {
namespace {

// The Switch 1 guest counter frequency in ticks per second: the 23.0.1 kernel
// converts timeouts to 19.2 MHz ticks (SleepThread 0x62080, WaitSynchronization
// 0x54640). Conversion helpers stay on this ratio. counter(rate) supplies the
// platform guest counter. Clock regression is unsupported by this contract.
constexpr std::uint64_t counter_frequency = 19'200'000;

// Exact tick scale: 1e9 ns / 19.2e6 ticks per second equals 625 ns per 12
// ticks. Conversions stay in exact integer arithmetic on this pair.
constexpr std::uint64_t nanoseconds_per = 625;
constexpr std::uint64_t ticks_per = 12;

// Production waits at most this long use the host's own timed wake-up.
constexpr std::chrono::microseconds alarm_threshold{500};

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

struct ElapsedReading {
    ClockInstant instant{};
    std::chrono::nanoseconds elapsed{};
};

class GuestClockImpl;

// One stable tick source for the Switch 2 counter. Its freeze value is not the
// 19.2 MHz sample's last_ticks: the two scales must not share a counter word.
struct RatedCounter {
    GuestClockImpl* clock = nullptr;
    horizon::CounterRate rate{};
    std::atomic<std::uint64_t> last_ticks{0};
    RatedCounter(GuestClockImpl* clock_in, horizon::CounterRate rate_in)
        : clock(clock_in), rate(rate_in) {}
};

// floor(elapsed_ns * ticks / nanoseconds). The two platform rates keep every
// intermediate inside uint64 for a count that already fit the nanoseconds domain.
std::uint64_t rated_ticks_floor(std::chrono::nanoseconds elapsed,
                                horizon::CounterRate rate) noexcept {
    const auto count = static_cast<std::uint64_t>(elapsed.count());
    const auto whole = count / rate.nanoseconds;
    return whole * rate.ticks +
           (count % rate.nanoseconds) * rate.ticks / rate.nanoseconds;
}

class GuestClockImpl final : public GuestClock {
public:
    GuestClockImpl(GuestClockSources sources, ClockInstant origin)
        : sources_(std::move(sources)), origin_(origin),
          switch2_(this, horizon::switch2_counter) {}

    ~GuestClockImpl() override {
        {
            std::scoped_lock lock(alarm_mutex_);
            alarm_stopping_ = true;
        }
        alarm_cv_.notify_all();
        if (alarm_.joinable()) alarm_.join();
    }

    LaunchResult<ClockSample> sample() const override {
        if (auto held = latched())
            return std::move(*held);
        return sample_and_latch();
    }

    execution::GuestCounter counter() noexcept override {
        return {&GuestClockImpl::read_ticks, this, counter_frequency};
    }

    execution::GuestCounter counter(horizon::CounterRate rate) noexcept override {
        if (rate.frequency == horizon::switch1_counter.frequency &&
            rate.nanoseconds == horizon::switch1_counter.nanoseconds &&
            rate.ticks == horizon::switch1_counter.ticks)
            return counter();
        if (rate.frequency == horizon::switch2_counter.frequency &&
            rate.nanoseconds == horizon::switch2_counter.nanoseconds &&
            rate.ticks == horizon::switch2_counter.ticks)
            return {&GuestClockImpl::read_rated_ticks, &switch2_, rate.frequency};
        return {};
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
            Waiter waiter;
            // Declared after the waiter and before its lock, so the deadline
            // is withdrawn after the lock is released and before the waiter
            // ends.
            const Armed armed(*this, deadline, waiter);
            std::stop_callback callback(stop, [&] {
                std::scoped_lock guard(waiter.mutex);
                waiter.stopped = true;
                waiter.wake.notify_one();
            });
            std::unique_lock lock(waiter.mutex);
            while (!waiter.stopped) {
                // The alarm's wake-up arrives as a notification at the
                // deadline, before the host's own timed wake-up.
                if (waiter.wake.wait_until(lock, deadline) ==
                        std::cv_status::timeout ||
                    std::chrono::steady_clock::now() >= deadline) {
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
    // One production wait: the alarm and a stop request wake it here.
    struct Waiter {
        std::mutex mutex;
        std::condition_variable wake;
        bool stopped = false;
    };

    // A production deadline registered with the alarm for one wait.
    struct Armed {
        Armed(GuestClockImpl& clock, ClockInstant deadline, Waiter& waiter)
            : clock(clock), entry(deadline, &waiter), armed(clock.arm(entry)) {}
        ~Armed() {
            if (armed) clock.disarm(entry);
        }
        Armed(const Armed&) = delete;
        Armed& operator=(const Armed&) = delete;
        GuestClockImpl& clock;
        std::pair<ClockInstant, Waiter*> entry;
        bool armed;
    };

    // Registers a wait's deadline and starts the alarm on first use. False
    // means no alarm (a wait of at most the threshold, an unbounded wait, or
    // the alarm thread could not start): the host's own wake-up ends it.
    bool arm(const std::pair<ClockInstant, Waiter*>& entry) noexcept {
        const auto deadline = entry.first;
        if (deadline == ClockInstant::max() ||
            deadline - std::chrono::steady_clock::now() <= alarm_threshold)
            return false;
        std::scoped_lock lock(alarm_mutex_);
        if (alarm_failed_) return false;
        try {
            if (!alarm_.joinable()) alarm_ = std::thread([this] { run_alarm(); });
            const bool earliest = deadlines_.empty() || deadline < deadlines_.begin()->first;
            deadlines_.insert(entry);
            if (earliest) alarm_cv_.notify_one();
            return true;
        } catch (...) {
            alarm_failed_ = true;
            armigo::diag::write(armigo::diag::Level::error,
                "guest_clock: the deadline alarm could not start; timed waits end at the host's wake-up\n");
            return false;
        }
    }

    // After this returns the alarm no longer refers to the waiter.
    void disarm(const std::pair<ClockInstant, Waiter*>& entry) noexcept {
        std::scoped_lock lock(alarm_mutex_);
        deadlines_.erase(entry);
    }

    // Wakes each waiter whose deadline passed. The alarm mutex is held while
    // a waiter is notified, so the waiter cannot end before that.
    void run_alarm() noexcept {
#if defined(__APPLE__)
        pthread_setname_np("armigo-clock-alarm");
#endif
        if (!platform::declare_periodic_thread(std::chrono::nanoseconds(16'666'667),
                                               std::chrono::nanoseconds(1'000'000)))
            armigo::diag::write(armigo::diag::Level::error,
                "guest_clock: the deadline alarm was not declared time-constrained; timed waits may end late\n");
        std::unique_lock lock(alarm_mutex_);
        while (!alarm_stopping_) {
            if (deadlines_.empty()) {
                alarm_cv_.wait(lock, [&] { return alarm_stopping_ || !deadlines_.empty(); });
                continue;
            }
            const auto first = deadlines_.begin()->first;
            if (std::chrono::steady_clock::now() < first) {
                alarm_cv_.wait_until(lock, first, [&] {
                    return alarm_stopping_ || deadlines_.empty() || deadlines_.begin()->first < first;
                });
                continue;
            }
            const auto now = std::chrono::steady_clock::now();
            while (!deadlines_.empty() && deadlines_.begin()->first <= now) {
                auto* waiter = deadlines_.begin()->second;
                deadlines_.erase(deadlines_.begin());
                // A waiter that has not entered its wait yet holds its mutex
                // and sees the passed deadline itself.
                { std::scoped_lock wake(waiter->mutex); }
                waiter->wake.notify_one();
            }
        }
    }

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

    // Same failure latch as sample(), on this rate's own freeze word. Returns
    // this call's ticks; a concurrent later sample does not replace them.
    static std::uint64_t read_rated_ticks(void* context) noexcept {
        auto& slot = *static_cast<RatedCounter*>(context);
        auto& self = *slot.clock;
        if (self.failed_.load(std::memory_order_acquire))
            return slot.last_ticks.load(std::memory_order_relaxed);
        try {
            const auto reading = self.try_elapsed();
            if (!reading)
                return slot.last_ticks.load(std::memory_order_relaxed);
            const auto ticks = rated_ticks_floor(reading->elapsed, slot.rate);
            auto previous = slot.last_ticks.load(std::memory_order_relaxed);
            while (ticks > previous &&
                   !slot.last_ticks.compare_exchange_weak(
                       previous, ticks, std::memory_order_release,
                       std::memory_order_relaxed)) {
            }
            return ticks;
        } catch (...) {
            return slot.last_ticks.load(std::memory_order_relaxed);
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

    std::optional<ElapsedReading> try_elapsed() const {
        ClockInstant instant{};
        try {
            instant = sources_.now ? sources_.now()
                                   : std::chrono::steady_clock::now();
        } catch (...) {
            latch(make_failure(LaunchStage::services, LaunchError::host_failure,
                               "guest_clock.sample",
                               "the clock source raised an exception"));
            return std::nullopt;
        }
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::nanoseconds>(instant -
                                                                 origin_);
        if (elapsed.count() < 0) {
            latch(make_failure(LaunchStage::services, LaunchError::host_failure,
                               "guest_clock.sample",
                               "the clock source regressed before the creation origin"));
            return std::nullopt;
        }
        return ElapsedReading{instant, elapsed};
    }

    LaunchResult<ClockSample> sample_and_latch() const {
        const auto reading = try_elapsed();
        if (!reading) {
            if (auto held = latched())
                return *held;
            return make_failure(LaunchStage::services, LaunchError::host_failure,
                                "guest_clock.sample",
                                "the clock source raised an exception");
        }
        const auto instant = reading->instant;
        const auto elapsed = reading->elapsed;
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
    RatedCounter switch2_;
    // Registered production waits, earliest deadline first.
    std::mutex alarm_mutex_;
    std::condition_variable alarm_cv_;
    std::set<std::pair<ClockInstant, Waiter*>> deadlines_;
    bool alarm_stopping_ = false;
    bool alarm_failed_ = false;
    std::thread alarm_;
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
