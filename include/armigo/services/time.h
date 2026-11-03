// ============================================================================
// Armigo Services - Host Time Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>

namespace armigo::services::time {

// Host views of the public time values. Each has the exact size and field
// offsets of its little-endian guest representation, so an ABI binding can
// copy it to or from validated guest memory without conversion.
struct SteadyClockTimePoint {
    std::int64_t seconds = 0;
    std::array<std::uint8_t, 16> source_id{};
};

struct CalendarTime {
    std::int16_t year = 0;
    std::int8_t month = 0;
    std::int8_t day = 0;
    std::int8_t hour = 0;
    std::int8_t minute = 0;
    std::int8_t second = 0;
    std::uint8_t reserved = 0;
};

struct CalendarAdditionalInfo {
    std::uint32_t day_of_week = 0;
    std::uint32_t day_of_year = 0;
    // Zero padded; a full eight-byte name has no terminator.
    std::array<char, 8> time_zone_name{};
    std::uint8_t is_daylight_saving_time = 0;
    std::array<std::uint8_t, 3> reserved{};
    std::int32_t utc_offset_seconds = 0;
};

static_assert(std::is_trivially_copyable_v<SteadyClockTimePoint> &&
              std::is_standard_layout_v<SteadyClockTimePoint> &&
              sizeof(SteadyClockTimePoint) == 24 &&
              offsetof(SteadyClockTimePoint, source_id) == 8);
static_assert(std::is_trivially_copyable_v<CalendarTime> &&
              std::is_standard_layout_v<CalendarTime> && sizeof(CalendarTime) == 8 &&
              offsetof(CalendarTime, month) == 2 && offsetof(CalendarTime, second) == 6 &&
              offsetof(CalendarTime, reserved) == 7);
static_assert(std::is_trivially_copyable_v<CalendarAdditionalInfo> &&
              std::is_standard_layout_v<CalendarAdditionalInfo> &&
              sizeof(CalendarAdditionalInfo) == 24 &&
              offsetof(CalendarAdditionalInfo, day_of_year) == 4 &&
              offsetof(CalendarAdditionalInfo, time_zone_name) == 8 &&
              offsetof(CalendarAdditionalInfo, is_daylight_saving_time) == 16 &&
              offsetof(CalendarAdditionalInfo, utc_offset_seconds) == 20);

// Guest-visible Result values reported by the supported public clock APIs.
inline constexpr std::uint32_t result_success = 0;
inline constexpr std::uint32_t result_not_initialized = 0x74;
inline constexpr std::uint32_t result_network_clock_unavailable = 0xcc74;
inline constexpr std::uint32_t result_span_null_output = 0x70a74;
inline constexpr std::uint32_t result_span_source_mismatch = 0x19074;
inline constexpr std::uint32_t result_span_overflow = 0x19274;

enum class StatusKind : std::uint8_t {
    // `result` is the value the guest API returns.
    guest_result,
    // The request is outside the supported behavior. No guest Result is
    // known; the binding must stop with `detail` rather than invent one.
    unsupported,
    // A host facility failed. The binding must stop with `detail`.
    host_failure,
    // The host caller supplied an invalid configuration or argument.
    invalid_argument,
};

struct Status {
    StatusKind kind = StatusKind::guest_result;
    std::uint32_t result = result_success;
    // Static diagnostic text for non-guest outcomes; null for guest results.
    const char* detail = nullptr;

    static constexpr Status guest(std::uint32_t value) noexcept {
        return {StatusKind::guest_result, value, nullptr};
    }
    static constexpr Status failure(StatusKind kind, const char* detail) noexcept {
        return {kind, result_success, detail};
    }
    constexpr bool ok() const noexcept {
        return kind == StatusKind::guest_result && result == result_success;
    }
    constexpr bool guest_visible() const noexcept { return kind == StatusKind::guest_result; }
};

// Replaceable clock and entropy sources. Production code supplies the guest
// steady counter and leaves the others empty to use host realtime and host
// entropy. Tests replace them with deterministic sources. Clock callbacks may
// run concurrently; their captured state needs its own synchronization. The
// entropy callback runs under the lifecycle lock and must not reenter it.
struct TimeSources {
    // Required: the guest steady counter in nanoseconds.
    std::function<std::uint64_t()> guest_steady_nanoseconds;
    // Optional: whole POSIX seconds of the user clock. Returns false on a
    // host failure. Defaults to the host realtime clock.
    std::function<bool(std::int64_t& seconds)> realtime_seconds;
    // Optional: fills the span with random bytes. Returns false on a host
    // failure. Defaults to host cryptographic entropy.
    std::function<bool(std::span<std::uint8_t> bytes)> entropy;
    // Optional: a fixed user clock value that replaces realtime_seconds.
    std::optional<std::int64_t> fixed_wall_seconds;
};

class TimeService;
struct TimeServiceCreation {
    Status status;
    std::unique_ptr<TimeService> service;
    bool ok() const noexcept { return status.ok() && service != nullptr; }
};

// Per-process host adapter for the public time APIs. Initialization is
// reference counted. The steady clock source ID is drawn from entropy at the
// first successful initialization and remains fixed for the object lifetime,
// including across finalization. All members are thread-safe. Every output
// is written only on success and is otherwise left unchanged.
class TimeService final {
public:
    static TimeServiceCreation create(TimeSources sources);

    TimeService(const TimeService&) = delete;
    TimeService& operator=(const TimeService&) = delete;

    Status initialize();
    // Releases one reference. Succeeds even when not initialized.
    Status finalize() noexcept;
    bool is_initialized() const noexcept;

    // Whole seconds of the guest steady counter with this process's source ID.
    Status steady_clock_now(SteadyClockTimePoint& out) const;
    // The user clock: host realtime seconds or the configured fixed value.
    Status user_clock_now(std::int64_t& out) const;
    // No network-synchronized clock exists on the host; this reports it as
    // unavailable once initialized. There is no output.
    Status network_clock_now() const noexcept;
    // Seconds from `first` to `second`. A pure operation that does not
    // require initialization. A null `out` reports result_span_null_output;
    // a source mismatch is checked before subtraction overflow.
    static Status span_between(const SteadyClockTimePoint& first,
                               const SteadyClockTimePoint& second,
                               std::int64_t* out) noexcept;
    // Converts POSIX seconds in the host time zone. `info` is optional.
    Status to_calendar_time(std::int64_t posix_seconds, CalendarTime& out,
                            CalendarAdditionalInfo* info) const;

private:
    explicit TimeService(TimeSources sources);

    TimeSources sources_;
    mutable std::mutex lifecycle_;
    std::atomic<std::uint32_t> references_{0};
    bool source_id_ready_ = false;
    std::array<std::uint8_t, 16> source_id_{};
};

}  // namespace armigo::services::time
