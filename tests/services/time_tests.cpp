// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/time.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace armigo::services::time;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; } } while (false)

constexpr std::uint64_t ns = 1'000'000'000;

struct FakeClocks {
    std::atomic<std::uint64_t> guest{0};
    std::atomic<std::int64_t> wall{1'790'000'123};
    std::atomic<bool> wall_fails{false};
    std::atomic<unsigned> entropy_calls{0};
    std::atomic<unsigned> zero_draws{0};  // Leading all-zero entropy draws.
    std::atomic<bool> entropy_fails{false};
    std::uint8_t next_byte = 0x10;

    TimeSources sources() {
        TimeSources result;
        result.guest_steady_nanoseconds = [this] { return guest.load(); };
        result.realtime_seconds = [this](std::int64_t& seconds) {
            if (wall_fails) return false;
            seconds = wall.load();
            return true;
        };
        result.entropy = [this](std::span<std::uint8_t> bytes) {
            ++entropy_calls;
            if (entropy_fails) return false;
            if (zero_draws > 0) {
                --zero_draws;
                std::memset(bytes.data(), 0, bytes.size());
                return true;
            }
            for (auto& byte : bytes) byte = next_byte++;
            return true;
        };
        return result;
    }
};

std::unique_ptr<TimeService> make(TimeSources sources) {
    auto created = TimeService::create(std::move(sources));
    CHECK(created.ok());
    return std::move(created.service);
}

bool is_failure(const Status& status, StatusKind kind) {
    return status.kind == kind && status.detail != nullptr && !status.ok();
}

// Proleptic Gregorian civil date to POSIX seconds.
std::int64_t utc_seconds(std::int64_t year, int month, int day, int hour, int minute, int second) {
    year -= month <= 2;
    const std::int64_t era = (year >= 0 ? year : year - 399) / 400;
    const std::int64_t year_of_era = year - era * 400;
    const std::int64_t day_of_year = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const std::int64_t days = era * 146097 + year_of_era * 365 + year_of_era / 4 -
                              year_of_era / 100 + day_of_year - 719468;
    return days * 86400 + hour * 3600 + minute * 60 + second;
}

void set_zone(const char* zone) {
    setenv("TZ", zone, 1);
    tzset();
}

void check_calendar(const TimeService& service, std::int64_t posix, int year, int month, int day,
                    int hour, int minute, int second, unsigned week_day, unsigned year_day,
                    std::string_view zone, bool dst, std::int32_t offset) {
    CalendarTime calendar;
    std::memset(&calendar, 0xa5, sizeof calendar);
    CalendarAdditionalInfo info;
    std::memset(&info, 0x5a, sizeof info);
    CHECK(service.to_calendar_time(posix, calendar, &info).ok());
    CHECK(calendar.year == year && calendar.month == month && calendar.day == day &&
          calendar.hour == hour && calendar.minute == minute && calendar.second == second &&
          calendar.reserved == 0);
    CHECK(info.day_of_week == week_day && info.day_of_year == year_day);
    CHECK(info.is_daylight_saving_time == (dst ? 1 : 0) && info.utc_offset_seconds == offset);
    std::array<char, 8> name{};
    std::memcpy(name.data(), zone.data(), std::min(zone.size(), name.size()));
    CHECK(info.time_zone_name == name);
    CHECK(info.reserved == (std::array<std::uint8_t, 3>{}));

    CalendarTime without_info;
    CHECK(service.to_calendar_time(posix, without_info, nullptr).ok());
    CHECK(std::memcmp(&without_info, &calendar, sizeof calendar) == 0);
}

void test_creation() {
    TimeSources missing;
    const auto created = TimeService::create(std::move(missing));
    CHECK(!created.ok() && created.service == nullptr);
    CHECK(is_failure(created.status, StatusKind::invalid_argument));

    // Production defaults: host realtime and entropy with only a guest counter.
    TimeSources defaults;
    defaults.guest_steady_nanoseconds = [] { return 7 * ns; };
    auto service = make(std::move(defaults));
    CHECK(service->initialize().ok());
    SteadyClockTimePoint point;
    CHECK(service->steady_clock_now(point).ok() && point.seconds == 7);
    CHECK(point.source_id != (std::array<std::uint8_t, 16>{}));
    std::int64_t now = 0;
    CHECK(service->user_clock_now(now).ok());
    const auto host_now = static_cast<std::int64_t>(std::time(nullptr));
    CHECK(now >= host_now - 5 && now <= host_now + 5);
}

void test_uninitialized() {
    FakeClocks clocks;
    auto service = make(clocks.sources());
    CHECK(!service->is_initialized());

    SteadyClockTimePoint point{123, {}};
    point.source_id[0] = 9;
    const auto original_point = point;
    std::int64_t value = 0x5555;
    CalendarTime calendar{1, 2, 3, 4, 5, 6, 7};
    CalendarAdditionalInfo info;
    info.day_of_week = 77;
    CHECK(service->steady_clock_now(point).result == result_not_initialized);
    CHECK(service->user_clock_now(value).result == result_not_initialized);
    CHECK(service->network_clock_now().result == result_not_initialized);
    CHECK(service->to_calendar_time(0, calendar, &info).result == result_not_initialized);
    CHECK(std::memcmp(&point, &original_point, sizeof point) == 0 && value == 0x5555);
    CHECK(calendar.year == 1 && calendar.reserved == 7 && info.day_of_week == 77);
    CHECK(clocks.entropy_calls == 0);

    // Finalization without initialization succeeds; the span is pure.
    CHECK(service->finalize().ok() && !service->is_initialized());
    const SteadyClockTimePoint first{5, {}}, second{12, {}};
    CHECK(TimeService::span_between(first, second, &value).ok() && value == 7);
}

void test_lifecycle_and_steady_clock() {
    FakeClocks clocks;
    auto service = make(clocks.sources());
    for (int i = 0; i < 3; ++i) CHECK(service->initialize().ok() && service->is_initialized());
    CHECK(clocks.entropy_calls == 1);

    clocks.guest = 5 * ns + (ns - 1);
    SteadyClockTimePoint first;
    CHECK(service->steady_clock_now(first).ok() && first.seconds == 5);
    CHECK(first.source_id != (std::array<std::uint8_t, 16>{}));

    for (int i = 0; i < 3; ++i) {
        CHECK(service->is_initialized());
        CHECK(service->finalize().ok());
    }
    CHECK(!service->is_initialized() && service->finalize().ok() && !service->is_initialized());
    SteadyClockTimePoint untouched = first;
    CHECK(service->steady_clock_now(untouched).result == result_not_initialized);

    // Reinitialization keeps the source ID and draws no new entropy.
    CHECK(service->initialize().ok());
    clocks.guest = std::numeric_limits<std::uint64_t>::max();
    SteadyClockTimePoint again;
    CHECK(service->steady_clock_now(again).ok());
    CHECK(again.seconds == static_cast<std::int64_t>(std::numeric_limits<std::uint64_t>::max() / ns));
    CHECK(again.source_id == first.source_id && clocks.entropy_calls == 1);

    // Independent processes draw independent sources.
    FakeClocks other_clocks;
    other_clocks.next_byte = 0x80;
    auto other = make(other_clocks.sources());
    CHECK(other->initialize().ok());
    SteadyClockTimePoint other_point;
    CHECK(other->steady_clock_now(other_point).ok() && other_point.source_id != first.source_id);
    std::int64_t span = 0x1234;
    CHECK(TimeService::span_between(first, other_point, &span).result ==
          result_span_source_mismatch && span == 0x1234);
}

void test_concurrent_lifecycle() {
    FakeClocks clocks;
    clocks.guest = 42 * ns;
    auto service = make(clocks.sources());
    CHECK(service->initialize().ok());
    SteadyClockTimePoint reference;
    CHECK(service->steady_clock_now(reference).ok());

    std::vector<std::thread> threads;
    std::atomic<unsigned> worker_failures{0};
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&] {
            for (int i = 0; i < 2000; ++i) {
                SteadyClockTimePoint point;
                std::int64_t wall = 0;
                if (!service->initialize().ok() || !service->steady_clock_now(point).ok() ||
                    point.source_id != reference.source_id || point.seconds != 42 ||
                    !service->user_clock_now(wall).ok() || !service->finalize().ok())
                    ++worker_failures;
            }
            if (!service->initialize().ok()) ++worker_failures;
        });
    for (auto& thread : threads) thread.join();
    CHECK(worker_failures == 0 && clocks.entropy_calls == 1);
    for (int i = 0; i < 8; ++i) CHECK(service->finalize().ok());
    CHECK(service->is_initialized());
    CHECK(service->finalize().ok() && !service->is_initialized());
}

void test_entropy_failures() {
    {
        // Leading zero draws are retried; the first non-zero ID is kept.
        FakeClocks clocks;
        clocks.zero_draws = 3;
        auto service = make(clocks.sources());
        CHECK(service->initialize().ok() && clocks.entropy_calls == 4);
        SteadyClockTimePoint point;
        CHECK(service->steady_clock_now(point).ok());
        CHECK(point.source_id != (std::array<std::uint8_t, 16>{}));
    }
    {
        FakeClocks clocks;
        clocks.zero_draws = 100;
        auto service = make(clocks.sources());
        CHECK(is_failure(service->initialize(), StatusKind::host_failure));
        CHECK(!service->is_initialized() && clocks.entropy_calls == 4);
        // A later successful draw initializes normally.
        clocks.zero_draws = 0;
        CHECK(service->initialize().ok() && service->is_initialized());
    }
    {
        FakeClocks clocks;
        clocks.entropy_fails = true;
        auto service = make(clocks.sources());
        CHECK(is_failure(service->initialize(), StatusKind::host_failure));
        CHECK(!service->is_initialized());
    }
}

void test_user_and_network_clocks() {
    FakeClocks clocks;
    auto service = make(clocks.sources());
    CHECK(service->initialize().ok());
    std::int64_t now = -5;
    CHECK(service->user_clock_now(now).ok() && now == 1'790'000'123);
    clocks.wall = -86400;
    CHECK(service->user_clock_now(now).ok() && now == -86400);
    clocks.wall_fails = true;
    now = 99;
    CHECK(is_failure(service->user_clock_now(now), StatusKind::host_failure) && now == 99);
    CHECK(service->network_clock_now().result == result_network_clock_unavailable);
    CHECK(service->network_clock_now().guest_visible());

    // A fixed wall clock replaces realtime; the steady clock keeps running.
    FakeClocks fixed_clocks;
    auto sources = fixed_clocks.sources();
    sources.fixed_wall_seconds = std::numeric_limits<std::int64_t>::max();
    auto fixed = make(std::move(sources));
    CHECK(fixed->initialize().ok());
    fixed_clocks.wall_fails = true;
    CHECK(fixed->user_clock_now(now).ok() && now == std::numeric_limits<std::int64_t>::max());
    fixed_clocks.guest = 3 * ns;
    SteadyClockTimePoint a, b;
    CHECK(fixed->steady_clock_now(a).ok() && a.seconds == 3);
    fixed_clocks.guest = 10 * ns;
    CHECK(fixed->steady_clock_now(b).ok() && b.seconds == 10);
}

void test_span() {
    SteadyClockTimePoint a{100, {}}, b{124, {}};
    a.source_id[3] = b.source_id[3] = 0x77;
    std::int64_t span = 0;
    CHECK(TimeService::span_between(a, b, &span).ok() && span == 24);
    CHECK(TimeService::span_between(b, a, &span).ok() && span == -24);
    CHECK(TimeService::span_between(a, a, &span).ok() && span == 0);

    auto other = b;
    other.source_id[15] ^= 1;
    span = 0x1234;
    CHECK(TimeService::span_between(a, other, &span).result == result_span_source_mismatch);
    CHECK(span == 0x1234);

    auto low = a, high = a;
    low.seconds = -1;
    high.seconds = std::numeric_limits<std::int64_t>::max();
    CHECK(TimeService::span_between(low, high, &span).result == result_span_overflow);
    low.seconds = 1;
    high.seconds = std::numeric_limits<std::int64_t>::min();
    CHECK(TimeService::span_between(low, high, &span).result == result_span_overflow);
    CHECK(span == 0x1234);
    low.seconds = 0;
    CHECK(TimeService::span_between(low, high, &span).ok() &&
          span == std::numeric_limits<std::int64_t>::min());

    // The source check precedes overflow; a null output is checked first.
    span = 0x1234;
    other.seconds = std::numeric_limits<std::int64_t>::max();
    low.seconds = -1;
    CHECK(TimeService::span_between(low, other, &span).result == result_span_source_mismatch);
    CHECK(span == 0x1234);
    CHECK(TimeService::span_between(a, other, nullptr).result == result_span_null_output);
}

void test_calendar() {
    FakeClocks clocks;
    auto service = make(clocks.sources());
    CHECK(service->initialize().ok());

    set_zone("UTC0");
    check_calendar(*service, 0, 1970, 1, 1, 0, 0, 0, 4, 0, "UTC", false, 0);
    check_calendar(*service, -1, 1969, 12, 31, 23, 59, 59, 3, 364, "UTC", false, 0);
    check_calendar(*service, 951'782'400, 2000, 2, 29, 0, 0, 0, 2, 59, "UTC", false, 0);
    check_calendar(*service, utc_seconds(32767, 12, 31, 23, 59, 59), 32767, 12, 31, 23, 59, 59,
                   0, 364, "UTC", false, 0);
    check_calendar(*service, utc_seconds(-32768, 1, 1, 0, 0, 0), -32768, 1, 1, 0, 0, 0, 4, 0,
                   "UTC", false, 0);

    set_zone("EST5EDT,M3.2.0,M11.1.0");
    check_calendar(*service, 1'625'140'800, 2021, 7, 1, 8, 0, 0, 4, 181, "EDT", true, -14400);
    check_calendar(*service, 1'609'459'200, 2020, 12, 31, 19, 0, 0, 4, 365, "EST", false, -18000);
    check_calendar(*service, 1'615'705'199, 2021, 3, 14, 1, 59, 59, 0, 72, "EST", false, -18000);
    check_calendar(*service, 1'615'705'200, 2021, 3, 14, 3, 0, 0, 0, 72, "EDT", true, -14400);
    set_zone("IST-5:30");
    check_calendar(*service, 0, 1970, 1, 1, 5, 30, 0, 4, 0, "IST", false, 19800);
    set_zone("LONGZONE-1");
    check_calendar(*service, 0, 1970, 1, 1, 1, 0, 0, 4, 0, "LONGZONE", false, 3600);
    set_zone("LONGZONENAME-1");
    check_calendar(*service, 0, 1970, 1, 1, 1, 0, 0, 4, 0, "LONGZONE", false, 3600);

    // Unrepresentable years and host conversion failures leave outputs unchanged.
    set_zone("UTC0");
    const std::int64_t unsupported[] = {
        utc_seconds(32768, 1, 1, 0, 0, 0),
        utc_seconds(-32768, 1, 1, 0, 0, 0) - 1,
        std::numeric_limits<std::int64_t>::max(),
        std::numeric_limits<std::int64_t>::min(),
    };
    for (const auto posix : unsupported) {
        CalendarTime calendar{1, 2, 3, 4, 5, 6, 7};
        CalendarAdditionalInfo info;
        info.utc_offset_seconds = 1234;
        const auto status = service->to_calendar_time(posix, calendar, &info);
        CHECK(!status.guest_visible() && status.detail != nullptr);
        CHECK(status.kind == StatusKind::unsupported || status.kind == StatusKind::host_failure);
        CHECK(calendar.year == 1 && calendar.reserved == 7 && info.utc_offset_seconds == 1234);
    }
    CalendarTime calendar;
    CHECK(service->to_calendar_time(utc_seconds(32768, 1, 1, 0, 0, 0), calendar, nullptr).kind ==
          StatusKind::unsupported);
}

}  // namespace

int main() {
    test_creation();
    test_uninitialized();
    test_lifecycle_and_steady_clock();
    test_concurrent_lifecycle();
    test_entropy_failures();
    test_user_and_network_clocks();
    test_span();
    test_calendar();
    if (failures != 0) {
        std::fprintf(stderr, "%u time service checks failed\n", failures);
        return 1;
    }
    std::puts("time service checks passed");
    return 0;
}
