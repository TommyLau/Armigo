// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Host display pacing uses the same injected monotonic timeline as the guest
// clock. Late frames skip missed grid points. This policy does not model
// firmware display timing.

#include <armigo/runtime/presentation_clock.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace armigo::runtime {
namespace {

static_assert(std::is_same_v<std::chrono::steady_clock::duration,
                             std::chrono::nanoseconds>,
              "Grid arithmetic assumes a nanoseconds steady clock");

LaunchFailure make_failure(LaunchStage stage, LaunchError code,
                           const char* operation, std::string detail) {
    LaunchFailure result;
    result.stage = stage;
    result.code = code;
    result.operation = operation;
    result.detail = std::move(detail);
    return result;
}

// The grid is the ray {epoch + n * refresh_period : n >= 0} on the guest
// clock's monotonic timeline. Index zero is the epoch itself, so every
// schedule reports an index whose grid point matches its deadline exactly.

class PresentationClockImpl final : public PresentationClock {
public:
    PresentationClockImpl(std::shared_ptr<GuestClock> clock,
                          PresentationClockConfiguration configuration)
        : clock_(std::move(clock)),
          configuration_(std::move(configuration)) {}

    std::shared_ptr<GuestClock> clock() const noexcept override {
        return clock_;
    }

    PresentationClockConfiguration configuration() const noexcept override {
        const std::lock_guard lock(state_mutex_);
        return configuration_;
    }

    bool set_refresh_period(std::chrono::nanoseconds period) noexcept override {
        if (period <= std::chrono::nanoseconds::zero()) return false;
        auto sampled = clock_->sample();
        if (!std::holds_alternative<ClockSample>(sampled)) return false;
        const auto instant = std::get<ClockSample>(sampled).instant;
        const std::lock_guard lock(state_mutex_);
        auto completed = completed_index(instant);
        if (!std::holds_alternative<std::uint64_t>(completed)) return false;
        std::int64_t offset = 0;
        std::int64_t epoch = 0;
        if (__builtin_mul_overflow(std::get<std::uint64_t>(completed), period.count(), &offset) ||
            __builtin_sub_overflow(instant.time_since_epoch().count(), offset, &epoch)) return false;
        configuration_.epoch = ClockInstant{std::chrono::nanoseconds{epoch}};
        configuration_.refresh_period = period;
        phase_.store(0, std::memory_order_relaxed);
        locked_.store(false, std::memory_order_relaxed);
        return true;
    }

    LaunchResult<PresentationSchedule> schedule(
        std::optional<std::uint64_t> last_refresh,
        std::int32_t swap_interval) const override {
        const std::lock_guard lock(state_mutex_);
        if (swap_interval < 0)
            return make_failure(LaunchStage::presentation,
                                LaunchError::invalid_input,
                                "presentation_clock.schedule",
                                "swap intervals are nonnegative");
        auto sampled = clock_->sample();
        if (auto* failure = std::get_if<LaunchFailure>(&sampled))
            return std::move(*failure);
        const auto& sample = std::get<ClockSample>(sampled);
        if (swap_interval == 0) {
            PresentationSchedule immediate;
            immediate.deadline = sample.instant;
            immediate.immediate = true;
            auto current = completed_index(sample.instant);
            if (auto* failure = std::get_if<LaunchFailure>(&current))
                return std::move(*failure);
            // An immediate completion records the index of the grid point it
            // delivers: the last point at or before now, or the epoch point
            // when now precedes it.
            immediate.refresh_index = std::get<std::uint64_t>(current);
            return immediate;
        }
        std::uint64_t earliest = 0;
        bool overflowed = false;
        if (last_refresh) {
            overflowed = __builtin_add_overflow(
                *last_refresh, static_cast<std::uint64_t>(swap_interval),
                &earliest);
        } else {
            auto current = completed_index(sample.instant);
            if (auto* failure = std::get_if<LaunchFailure>(&current))
                return std::move(*failure);
            overflowed = __builtin_add_overflow(
                std::get<std::uint64_t>(current),
                static_cast<std::uint64_t>(swap_interval), &earliest);
        }
        if (overflowed)
            return make_failure(LaunchStage::presentation,
                                LaunchError::resource_limit,
                                "presentation_clock.schedule",
                                "the interval overflows the refresh index");
        auto at_or_after = index_at_or_after(sample.instant);
        if (auto* failure = std::get_if<LaunchFailure>(&at_or_after))
            return std::move(*failure);
        // The later of "the first grid point at or after now" and "at least
        // the requested interval of refreshes after the last completed
        // target". A late frame jumps straight to the surviving point; missed
        // points are skipped, never accumulated into sleep debt.
        const std::uint64_t target =
            std::max(std::get<std::uint64_t>(at_or_after), earliest);
        auto deadline = grid_point(target);
        if (auto* failure = std::get_if<LaunchFailure>(&deadline))
            return std::move(*failure);
        PresentationSchedule planned;
        planned.deadline = std::get<ClockInstant>(deadline);
        planned.refresh_index = target;
        planned.immediate = false;
        return planned;
    }

    void observe_display_refresh(ClockInstant vblank, std::chrono::nanoseconds display_period) noexcept override {
        const std::lock_guard lock(state_mutex_);
        const std::int64_t period = configuration_.refresh_period.count();
        const std::int64_t display = display_period.count();
        if (display <= 0) return;
        // Only a display whose refresh divides the grid's evenly (60 Hz on a
        // 60, 120 or 240 Hz display, within half a percent) carries the grid;
        // any other rate (144 Hz, 90 Hz) leaves it on the host clock alone.
        const auto multiple = std::llround(double(period) / double(display));
        if (multiple < 1 || std::llabs(multiple * display - period) * 200 > period) return;
        // A grid point just after a vblank: a release is never early, only
        // late (waits, a busy GPU queue, the compositor's lead), so the
        // present released there has nearly a whole refresh to be committed
        // before the next vblank, which it lands on every time. The grid keeps
        // its period; only its phase follows the display, in steps small
        // enough that two refreshes never move closer by more than a fraction
        // of a millisecond.
        const std::int64_t target = vblank.time_since_epoch().count() + display_lead_ns;
        const std::int64_t phase = phase_.load(std::memory_order_relaxed);
        // The offset of the grid point after the target within one grid
        // period, then within one display refresh: measured at that point, not
        // at the epoch, so a display period that is not an exact divisor
        // (59.94 Hz) or that varies by a tick does not accumulate with time.
        std::int64_t offset = (configuration_.epoch.time_since_epoch().count() + phase - target) % period;
        if (offset < 0) offset += period;
        std::int64_t error = offset % display;
        if (error >= display / 2) error -= display;
        locked_.store(true, std::memory_order_relaxed);
        if (std::llabs(error) <= phase_deadband_ns) return;
        const std::int64_t step = phase_step_ns / multiple;
        phase_.store(phase - std::clamp(error, -step, step), std::memory_order_relaxed);
    }

    std::optional<ClockInstant> display_refresh_near(ClockInstant instant) const noexcept override {
        const std::lock_guard lock(state_mutex_);
        if (!locked_.load(std::memory_order_relaxed)) return std::nullopt;
        const std::int64_t period = configuration_.refresh_period.count();
        const std::int64_t epoch = epoch_ns();
        const std::int64_t delta = instant.time_since_epoch().count() - epoch;
        std::int64_t index = delta / period;
        if (delta % period < 0) --index;
        if (delta - index * period > period / 2) ++index;
        return ClockInstant{std::chrono::nanoseconds{epoch + index * period}};
    }

    LaunchResult<PresentationSchedule> next_refresh() const override {
        const std::lock_guard lock(state_mutex_);
        auto next = next_strictly_future_index();
        if (auto* failure = std::get_if<LaunchFailure>(&next))
            return std::move(*failure);
        const std::uint64_t index = std::get<std::uint64_t>(next);
        auto deadline = grid_point(index);
        if (auto* failure = std::get_if<LaunchFailure>(&deadline))
            return std::move(*failure);
        PresentationSchedule planned;
        planned.deadline = std::get<ClockInstant>(deadline);
        planned.refresh_index = index;
        planned.immediate = false;
        return planned;
    }

private:
    // The next strictly future grid point, for display events independent of
    // buffers. Before the epoch that is the epoch point itself; on a grid
    // point it is the following one.
    LaunchResult<std::uint64_t> next_strictly_future_index() const {
        auto sampled = clock_->sample();
        if (auto* failure = std::get_if<LaunchFailure>(&sampled))
            return std::move(*failure);
        const auto& sample = std::get<ClockSample>(sampled);
        std::int64_t delta = 0;
        if (__builtin_sub_overflow(sample.instant.time_since_epoch().count(),
                                   epoch_ns(),
                                   &delta))
            return make_failure(LaunchStage::presentation,
                                LaunchError::resource_limit,
                                "presentation_clock.grid",
                                "the grid span is not representable");
        if (delta < 0)
            return std::uint64_t{0};
        std::uint64_t whole = 0;
        if (delta > 0) {
            whole = static_cast<std::uint64_t>(delta) /
                    static_cast<std::uint64_t>(
                        configuration_.refresh_period.count());
        }
        std::uint64_t next = 0;
        if (__builtin_add_overflow(whole, std::uint64_t{1}, &next))
            return make_failure(LaunchStage::presentation,
                                LaunchError::resource_limit,
                                "presentation_clock.next_refresh",
                                "the grid has no further representable point");
        return next;
    }
    LaunchResult<ClockInstant> grid_point(std::uint64_t index) const {
        std::int64_t sum = 0;
        std::int64_t offset = 0;
        if (__builtin_mul_overflow(index, configuration_.refresh_period.count(),
                                   &offset) ||
            __builtin_add_overflow(
                epoch_ns(), offset,
                &sum))
            return make_failure(LaunchStage::presentation,
                                LaunchError::resource_limit,
                                "presentation_clock.grid",
                                "the grid point is not representable");
        return ClockInstant{std::chrono::nanoseconds{sum}};
    }

    // The last grid point at or before the instant: the largest completed
    // refresh. Zero covers every instant at or before the epoch, whose point
    // has not completed yet.
    LaunchResult<std::uint64_t> completed_index(ClockInstant instant) const {
        std::int64_t delta = 0;
        if (__builtin_sub_overflow(instant.time_since_epoch().count(),
                                   epoch_ns(),
                                   &delta))
            return make_failure(LaunchStage::presentation,
                                LaunchError::resource_limit,
                                "presentation_clock.grid",
                                "the grid span is not representable");
        if (delta <= 0)
            return std::uint64_t{0};
        const std::uint64_t whole =
            static_cast<std::uint64_t>(delta) /
            static_cast<std::uint64_t>(
                configuration_.refresh_period.count());
        return whole;
    }

    // The first grid point at or after the instant. Zero covers every instant
    // at or before the epoch, whose point is the epoch itself.
    LaunchResult<std::uint64_t> index_at_or_after(ClockInstant instant) const {
        std::int64_t delta = 0;
        if (__builtin_sub_overflow(instant.time_since_epoch().count(),
                                   epoch_ns(),
                                   &delta))
            return make_failure(LaunchStage::presentation,
                                LaunchError::resource_limit,
                                "presentation_clock.grid",
                                "the grid span is not representable");
        if (delta <= 0)
            return std::uint64_t{0};
        const std::uint64_t period =
            static_cast<std::uint64_t>(
                configuration_.refresh_period.count());
        const std::uint64_t whole =
            static_cast<std::uint64_t>(delta) / period;
        return whole + (static_cast<std::uint64_t>(delta) % period ? 1 : 0);
    }

    // The configured epoch moved by the display phase. Read once per helper:
    // a phase step between two reads moves a deadline by at most one step.
    std::int64_t epoch_ns() const noexcept {
        return configuration_.epoch.time_since_epoch().count() + phase_.load(std::memory_order_relaxed);
    }

    static constexpr std::int64_t phase_step_ns = 200'000;
    static constexpr std::int64_t phase_deadband_ns = 50'000;
    static constexpr std::int64_t display_lead_ns = 500'000;

    std::shared_ptr<GuestClock> clock_;
    mutable std::mutex state_mutex_;
    PresentationClockConfiguration configuration_;
    // Offset of the grid from the configured epoch, written only by the
    // display observer.
    std::atomic<std::int64_t> phase_{0};
    // Set once a display that divides the grid has been observed.
    std::atomic<bool> locked_{false};
};

} // namespace

LaunchResult<std::shared_ptr<PresentationClock>> create_presentation_clock(
    std::shared_ptr<GuestClock> clock,
    PresentationClockConfiguration configuration) {
    if (!clock)
        return make_failure(LaunchStage::presentation,
                            LaunchError::invalid_input,
                            "create_presentation_clock",
                            "a guest clock is required");
    if (configuration.refresh_period <= std::chrono::nanoseconds::zero())
        return make_failure(LaunchStage::presentation,
                            LaunchError::invalid_input,
                            "create_presentation_clock",
                            "the refresh period must be positive");
    return std::make_shared<PresentationClockImpl>(std::move(clock),
                                                   configuration);
}

} // namespace armigo::runtime
