// ============================================================================
// Armigo Services - Host Time Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/time.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <utility>

namespace armigo::services::time {
namespace {

constexpr std::uint64_t nanoseconds_per_second = 1'000'000'000;
constexpr int source_id_attempts = 4;

static_assert(std::numeric_limits<std::uint64_t>::max() / nanoseconds_per_second <=
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
static_assert(sizeof(std::time_t) == sizeof(std::int64_t) && std::time_t(-1) < 0,
              "the host time_t must be a signed 64-bit value");

bool host_realtime_seconds(std::int64_t& seconds) {
    timespec now{};
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return false;
    seconds = static_cast<std::int64_t>(now.tv_sec);
    return true;
}

bool host_entropy(std::span<std::uint8_t> bytes) {
    arc4random_buf(bytes.data(), bytes.size());
    return true;
}

bool in_range(int value, int low, int high) { return value >= low && value <= high; }

}  // namespace

TimeServiceCreation TimeService::create(TimeSources sources) {
    if (!sources.guest_steady_nanoseconds)
        return {Status::failure(StatusKind::invalid_argument,
                                "time service requires a guest steady counter source"),
                nullptr};
    if (!sources.realtime_seconds) sources.realtime_seconds = host_realtime_seconds;
    if (!sources.entropy) sources.entropy = host_entropy;
    return {Status::guest(result_success),
            std::unique_ptr<TimeService>(new TimeService(std::move(sources)))};
}

TimeService::TimeService(TimeSources sources) : sources_(std::move(sources)) {}

Status TimeService::initialize() {
    std::lock_guard lock(lifecycle_);
    const auto references = references_.load(std::memory_order_relaxed);
    if (references == std::numeric_limits<std::uint32_t>::max())
        return Status::failure(StatusKind::host_failure,
                               "time service initialization reference count overflow");
    if (!source_id_ready_) {
        constexpr std::array<std::uint8_t, 16> zero{};
        std::array<std::uint8_t, 16> candidate{};
        bool drawn = false;
        for (int attempt = 0; attempt < source_id_attempts && !drawn; ++attempt) {
            if (!sources_.entropy(candidate))
                return Status::failure(StatusKind::host_failure,
                                       "host entropy failed while creating the steady clock source ID");
            drawn = candidate != zero;
        }
        if (!drawn)
            return Status::failure(StatusKind::host_failure,
                                   "host entropy produced only all-zero steady clock source IDs");
        source_id_ = candidate;
        source_id_ready_ = true;
        tzset();
    }
    references_.store(references + 1, std::memory_order_release);
    return Status::guest(result_success);
}

Status TimeService::finalize() noexcept {
    std::lock_guard lock(lifecycle_);
    const auto references = references_.load(std::memory_order_relaxed);
    if (references != 0) references_.store(references - 1, std::memory_order_release);
    return Status::guest(result_success);
}

bool TimeService::is_initialized() const noexcept {
    return references_.load(std::memory_order_acquire) != 0;
}

Status TimeService::steady_clock_now(SteadyClockTimePoint& out) const {
    // The acquire load publishes source_id_, which never changes once set.
    if (!is_initialized()) return Status::guest(result_not_initialized);
    SteadyClockTimePoint point;
    point.seconds = static_cast<std::int64_t>(sources_.guest_steady_nanoseconds() /
                                              nanoseconds_per_second);
    point.source_id = source_id_;
    out = point;
    return Status::guest(result_success);
}

Status TimeService::user_clock_now(std::int64_t& out) const {
    if (!is_initialized()) return Status::guest(result_not_initialized);
    if (sources_.fixed_wall_seconds) {
        out = *sources_.fixed_wall_seconds;
        return Status::guest(result_success);
    }
    std::int64_t seconds = 0;
    if (!sources_.realtime_seconds(seconds))
        return Status::failure(StatusKind::host_failure, "host realtime clock failed");
    out = seconds;
    return Status::guest(result_success);
}

Status TimeService::network_clock_now() const noexcept {
    if (!is_initialized()) return Status::guest(result_not_initialized);
    return Status::guest(result_network_clock_unavailable);
}

Status TimeService::span_between(const SteadyClockTimePoint& first,
                                 const SteadyClockTimePoint& second,
                                 std::int64_t* out) noexcept {
    if (out == nullptr) return Status::guest(result_span_null_output);
    if (first.source_id != second.source_id) return Status::guest(result_span_source_mismatch);
    std::int64_t span = 0;
    if (__builtin_sub_overflow(second.seconds, first.seconds, &span))
        return Status::guest(result_span_overflow);
    *out = span;
    return Status::guest(result_success);
}

Status TimeService::to_calendar_time(std::int64_t posix_seconds, CalendarTime& out,
                                     CalendarAdditionalInfo* info) const {
    if (!is_initialized()) return Status::guest(result_not_initialized);
    const std::time_t host_time = static_cast<std::time_t>(posix_seconds);
    std::tm local{};
    if (localtime_r(&host_time, &local) == nullptr)
        return Status::failure(StatusKind::host_failure,
                               "host cannot convert the time to the local time zone");
    if (!in_range(local.tm_mon, 0, 11) || !in_range(local.tm_mday, 1, 31) ||
        !in_range(local.tm_hour, 0, 23) || !in_range(local.tm_min, 0, 59) ||
        !in_range(local.tm_sec, 0, 60) || !in_range(local.tm_wday, 0, 6) ||
        !in_range(local.tm_yday, 0, 365) || local.tm_isdst < 0)
        return Status::failure(StatusKind::host_failure,
                               "host returned an out-of-range local time");
    int year = 0;
    if (__builtin_add_overflow(local.tm_year, 1900, &year) ||
        !in_range(year, std::numeric_limits<std::int16_t>::min(),
                  std::numeric_limits<std::int16_t>::max()))
        return Status::failure(StatusKind::unsupported,
                               "calendar year is outside the signed 16-bit range");
    if (info != nullptr && (local.tm_gmtoff < std::numeric_limits<std::int32_t>::min() ||
                           local.tm_gmtoff > std::numeric_limits<std::int32_t>::max()))
        return Status::failure(StatusKind::host_failure,
                               "host UTC offset does not fit 32 bits");

    CalendarTime calendar;
    calendar.year = static_cast<std::int16_t>(year);
    calendar.month = static_cast<std::int8_t>(local.tm_mon + 1);
    calendar.day = static_cast<std::int8_t>(local.tm_mday);
    calendar.hour = static_cast<std::int8_t>(local.tm_hour);
    calendar.minute = static_cast<std::int8_t>(local.tm_min);
    calendar.second = static_cast<std::int8_t>(local.tm_sec);

    CalendarAdditionalInfo extra;
    extra.day_of_week = static_cast<std::uint32_t>(local.tm_wday);
    extra.day_of_year = static_cast<std::uint32_t>(local.tm_yday);
    // Short abbreviations are zero padded; long ones are truncated to the field.
    if (local.tm_zone != nullptr)
        std::memcpy(extra.time_zone_name.data(), local.tm_zone,
                    strnlen(local.tm_zone, extra.time_zone_name.size()));
    extra.is_daylight_saving_time = local.tm_isdst > 0 ? 1 : 0;
    extra.utc_offset_seconds = static_cast<std::int32_t>(local.tm_gmtoff);

    out = calendar;
    if (info != nullptr) *info = extra;
    return Status::guest(result_success);
}

}  // namespace armigo::services::time
