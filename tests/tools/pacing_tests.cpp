// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pacing_clock.h"
#include "runner_shell_internal.h"
#include <armigo/services/vsync_source.h>
#include <atomic>
#include <filesystem>
#include <condition_variable>
#include <future>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace armigo;
using namespace armigo::runtime;
using namespace armigo::tools;

namespace {
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " << #x << '\n'; ++failures; } } while (false)
PacingSchedule parse(const std::string& source, double ceiling = 1000) {
    std::istringstream in(source);
    return std::get<PacingSchedule>(parse_pacing_schedule(in, ceiling));
}
struct Timeline {
    std::atomic<std::int64_t> now{0};
    std::atomic_uint samples{0};
    std::atomic<std::int64_t> waiting_for{-1};
    std::shared_ptr<GuestClock> guest;
    std::shared_ptr<PresentationClock> base;
    Timeline() {
        GuestClockSources sources;
        sources.now = [this] { ++samples; return ClockInstant{std::chrono::nanoseconds{now.load()}}; };
        sources.wait = [this](ClockInstant target, std::stop_token stop) -> LaunchResult<ClockWait> {
            waiting_for = target.time_since_epoch().count();
            while (!stop.stop_requested() && now < target.time_since_epoch().count()) std::this_thread::yield();
            return stop.stop_requested() ? ClockWait::cancelled : ClockWait::reached;
        };
        guest = std::get<std::shared_ptr<GuestClock>>(create_guest_clock(sources));
        base = std::get<std::shared_ptr<PresentationClock>>(create_presentation_clock(
            guest, {ClockInstant{}, std::chrono::nanoseconds{16'666'667}}));
    }
};
void parsers() {
    for (const auto* source : {"", "fixed 0 30", "fixed 100 0", "fixed 100 nan", "fixed 100 1001",
            "fixed 100 30 extra", "alternate 100 20 120 0", "ramp 100 30", "other 100 30",
            "fixed 1e99 30", "fixed 100 30\nhitch 0 0.000000001", "fixed 100 30\nhitch -1 50", "fixed 100 30\nhitch 0 100\nhitch 50 50"}) {
        std::istringstream in(source);
        CHECK(std::holds_alternative<std::string>(parse_pacing_schedule(in, 120)));
    }
    auto fixed = parse("# comment\nfixed 1000 120\n", 60);
    CHECK(fixed.period_at(0) == 16'666'667);
    CHECK(fixed.period_at(2'000'000'000) == 16'666'667);
    for (const auto rates : {std::pair{70., 140.}, {30., 120.}, {20., 120.}}) {
        auto alt = parse("alternate 10000 " + std::to_string(rates.first) + " " +
            std::to_string(rates.second) + " 500");
        CHECK(alt.period_at(499'999'999) == PacingSchedule::period(rates.first));
        CHECK(alt.period_at(500'000'000) == PacingSchedule::period(rates.second));
        CHECK(alt.period_at(1'000'000'000) == PacingSchedule::period(rates.first));
    }
    auto ramp = parse("ramp 1000 20 120");
    CHECK(ramp.period_at(0) == 50'000'000);
    CHECK(ramp.period_at(500'000'000) == PacingSchedule::period(70));
    CHECK(ramp.period_at(1'000'000'000) == PacingSchedule::period(120));
    auto sweep = parse("fixed 1000 29\nfixed 1000 30\nfixed 1000 31\nfixed 1000 119\nfixed 1000 120\nfixed 1000 121");
    int i = 0;
    for (double rate : {29, 30, 31, 119, 120, 121}) CHECK(sweep.period_at(std::int64_t(i++) * 1'000'000'000) == PacingSchedule::period(rate));
    for (const auto* text : {"1:0", "1:3", "-1:4", "0xffffffffffffffff:8", "0x:4", "1:4,", "1:4:8"})
        CHECK(std::holds_alternative<std::string>(parse_pacing_watches(text)));
    auto watches = std::get<std::vector<PacingWatch>>(parse_pacing_watches("0x1000:1,4098:2,4100:4,4104:8"));
    CHECK(watches.size() == 4 && watches[0].address == 4096 && watches[3].width == 8);
}
void relocatable_watches() {
    auto watches = std::get<std::vector<PacingWatch>>(parse_pacing_watches("main+0x100:8>+0x50:4,4096:4"));
    auto& watch = watches[0];
    CHECK(watch.column() == "watch_main_256_8_p80_4");
    CHECK(watches[1].column() == "watch_4096_4");
    unsigned reads = 0;
    std::uint64_t root_address = 0x1100, pointer = 0x8000, answer = 42;
    bool readable = true;
    PresentationMemoryReader reader = [&](std::uint64_t at, std::span<std::byte> out) {
        ++reads;
        std::uint64_t value;
        if (at == root_address && out.size() == 8) value = pointer;
        else if (readable && at == pointer + 0x50 && out.size() == 4) value = answer;
        else return false;
        for (std::size_t i = 0; i < out.size(); ++i) out[i] = std::byte((value >> (8 * i)) & 255);
        return true;
    };
    CHECK(!watch.read(reader) && reads == 0);
    std::array<PresentationModule, 2> modules{{{"sdk", 0x7000}, {"main", 0x1000}}};
    watch.bind(modules);
    CHECK(watch.read(reader) == 42 && reads == 2);
    modules[1].base_address = 0x4000;
    root_address = 0x4100; pointer = 0xa000; answer = 43;
    watch.bind(modules);
    CHECK(watch.read(reader) == 43 && reads == 4);
    CHECK(watch.column() == "watch_main_256_8_p80_4");
    readable = false;
    CHECK(!watch.read(reader) && reads == 6);
    pointer = 0;
    CHECK(!watch.read(reader) && reads == 7);
    pointer = UINT64_MAX - 3;
    CHECK(!watch.read(reader) && reads == 8);
    modules[1].base_address = UINT64_MAX - 1;
    watch.bind(modules);
    CHECK(!watch.read(reader) && reads == 8);
    modules[1] = {"other", 0x1000};
    watch.bind(modules);
    CHECK(!watch.read(reader));

    std::string entry = "main+0x100:8";
    for (int i = 0; i < 16; ++i) entry += ">+0x10:8";
    std::string list;
    for (int i = 0; i < 64; ++i) { if (i) list += ','; list += entry; }
    auto maximum = std::get<std::vector<PacingWatch>>(parse_pacing_watches(list));
    CHECK(maximum.size() == 64 && maximum[0].chain.size() == 16);
    reads = 0;
    modules[1] = {"main", 0x1000};
    for (auto& item : maximum) {
        item.bind(modules);
        CHECK(item.read([&](std::uint64_t, std::span<std::byte> out) {
            ++reads;
            std::fill(out.begin(), out.end(), std::byte{1});
            return true;
        }).has_value());
    }
    CHECK(reads == 64 * 17);
    CHECK(std::holds_alternative<std::string>(parse_pacing_watches(list + ",4096:4")));
    CHECK(std::holds_alternative<std::string>(parse_pacing_watches(entry + ">+0x10:8")));
    for (const auto* invalid : {"main+:4", "+0:4", "main+0:8>", "main+0:8>0:4", "main+0:8>+-1:4", "main+0:8>+1:3"})
        CHECK(std::holds_alternative<std::string>(parse_pacing_watches(invalid)));

    Timeline time;
    auto stream = std::make_unique<std::ostringstream>();
    auto* output = stream.get();
    PacingClock logged(time.base, std::nullopt, std::move(stream),
        std::get<std::vector<PacingWatch>>(parse_pacing_watches("main+0x100:8>+0x50:4")));
    CHECK(std::holds_alternative<std::monostate>(logged.observe_present(1, 1, reader)));
    CHECK(output->str().ends_with("\tNA\n"));
    root_address = 0x1100; pointer = 0x8000; answer = 42; readable = true;
    logged.bind_observation_modules(modules);
    CHECK(std::holds_alternative<std::monostate>(logged.observe_present(2, 1, reader)));
    CHECK(output->str().ends_with("\t42\n"));
}

void grid() {
    Timeline time;
    auto clock = std::make_shared<PacingClock>(time.base, parse("alternate-frame 10000 20 100"));
    auto first = std::get<PresentationSchedule>(clock->next_refresh());
    CHECK(first.refresh_index == 1 && first.deadline.time_since_epoch().count() == 50'000'000);
    auto second = std::get<PresentationSchedule>(clock->schedule(first.refresh_index, 1));
    CHECK(second.refresh_index == 2 && second.deadline.time_since_epoch().count() == 60'000'000);
    time.now = 125'000'000;
    auto late = std::get<PresentationSchedule>(clock->schedule(1, 1));
    CHECK(late.refresh_index == 5 && late.deadline.time_since_epoch().count() == 170'000'000);
    auto immediate = std::get<PresentationSchedule>(clock->schedule({}, 0));
    CHECK(immediate.immediate && immediate.refresh_index == 4 && immediate.deadline.time_since_epoch().count() == time.now);
    CHECK(std::holds_alternative<LaunchFailure>(clock->schedule({}, -1)));
    CHECK(std::holds_alternative<LaunchFailure>(clock->schedule(UINT64_MAX, 1)));
    CHECK(!clock->set_refresh_period(std::chrono::nanoseconds{1}));
    // A ceiling below both alternation rates must produce a fixed clock.
    Timeline capped_time;
    PacingClock capped(capped_time.base, parse("alternate 10000 70 140 500", 60));
    std::optional<std::uint64_t> last;
    for (int i = 1; i <= 120; ++i) {
        const auto point = std::get<PresentationSchedule>(capped.schedule(last, 1));
        CHECK(point.deadline.time_since_epoch().count() == std::int64_t(i) * 16'666'667);
        capped_time.now = point.deadline.time_since_epoch().count();
        last = point.refresh_index;
    }
    for (int hitch_ms : {50, 100, 250}) {
        Timeline t;
        PacingClock hit(t.base, parse("fixed 10000 100\nhitch 20 " + std::to_string(hitch_ms)));
        auto a = std::get<PresentationSchedule>(hit.schedule({}, 1));
        auto b = std::get<PresentationSchedule>(hit.schedule(a.refresh_index, 1));
        CHECK(a.deadline.time_since_epoch().count() == 10'000'000);
        CHECK(b.deadline.time_since_epoch().count() == (20 + hitch_ms) * 1'000'000);
        auto c = std::get<PresentationSchedule>(hit.schedule(b.refresh_index, 1));
        CHECK(c.deadline - b.deadline == std::chrono::milliseconds(10));
    }
    // VI and presentation agree on each opportunity, including a hitch.
    Timeline v;
    auto shared = std::make_shared<PacingClock>(v.base, parse("fixed 10000 100\nhitch 20 50"));
    services::vi::VsyncSource source(shared);
    bool created = false;
    auto event = source.acquire(0, created);
    CHECK(created && source.imposed_clock() == shared);
    auto next = std::get<PresentationSchedule>(shared->next_refresh());
    const auto wait_started = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (v.waiting_for < 0 && std::chrono::steady_clock::now() < wait_started) std::this_thread::yield();
    CHECK(v.waiting_for == next.deadline.time_since_epoch().count());
    const auto wait_sequence = [&](std::uint64_t sequence) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (source.sequence() < sequence && std::chrono::steady_clock::now() < until) std::this_thread::yield();
        return source.sequence() >= sequence;
    };
    v.now = 10'000'000;
    CHECK(wait_sequence(1));
    const auto second_wait = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (v.waiting_for == 10'000'000 && std::chrono::steady_clock::now() < second_wait) std::this_thread::yield();
    CHECK(v.waiting_for == 70'000'000);
    v.now = 69'000'000;
    CHECK(source.sequence() == 1);
    v.now = 70'000'000;
    CHECK(wait_sequence(2));
    CHECK(!source.failure());
}
void logging_and_off() {
    Timeline time;
    RunnerArguments arguments;
    ShellEnvironment env;
    env.create_guest_clock = [&]() -> LaunchResult<std::shared_ptr<GuestClock>> { return time.guest; };
    env.create_presentation_clock = [&](auto, auto) -> LaunchResult<std::shared_ptr<PresentationClock>> { return time.base; };
    auto off = shell_detail::create_shell_clocks(arguments, env);
    CHECK(std::get<std::shared_ptr<PresentationClock>>(off) == time.base);
    CHECK(!time.base->imposed_pacing() && !time.base->observes_presents());
    const auto path = std::filesystem::temp_directory_path() / ("armigo-pacing-" + std::to_string(getpid()) + ".tsv");
    {
        auto watches = std::get<std::vector<PacingWatch>>(parse_pacing_watches("4096:4,8192:8"));
        PacingClock log(time.base, std::nullopt, std::make_unique<std::ofstream>(path), watches);
        CHECK(!log.imposed_pacing());
        int reads = 0;
        PresentationMemoryReader buffers([&](std::uint64_t address, std::span<std::byte> bytes) {
            ++reads;
            if (address != 4096) return false;
            std::fill(bytes.begin(), bytes.end(), std::byte{1});
            return true;
        });
        time.now = 10'000'000;
        log.observe_vsync({ClockInstant{std::chrono::nanoseconds{9'000'000}}, 1, false});
        CHECK(std::holds_alternative<std::monostate>(log.observe_present(1, 1, buffers)));
        time.now = 110'000'000;
        CHECK(std::holds_alternative<std::monostate>(log.observe_present(2, 1, buffers)));
        CHECK(reads == 4);
    }
    std::ifstream input(path);
    std::string header, first, second;
    std::getline(input, header); std::getline(input, first); std::getline(input, second);
    CHECK(header.find("watch_4096_4") != std::string::npos);
    CHECK(first.find("\t16843009\tNA") != std::string::npos);
    CHECK(second.find("\t100000000\t50000000\t") != std::string::npos);
    CHECK(second.find("\t10000000\t9000000\t1") != std::string::npos);
    std::filesystem::remove(path);
}
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false;
    void block() {
        std::unique_lock lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [&] { return released; });
    }
    bool await_entry() {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(2), [&] { return entered; });
    }
    void release() {
        const std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};
struct BlockingBuffer : std::stringbuf {
    Gate gate;
    bool block = false;
    int sync() override {
        if (block) gate.block();
        return std::stringbuf::sync();
    }
};

void stalled_observers_leave_vi_and_planning_running(bool stall_copy) {
    Timeline time;
    BlockingBuffer output;
    Gate copy_gate;
    auto clock = std::make_shared<PacingClock>(time.base, parse("fixed 10000 100"),
        std::make_unique<std::ostream>(&output), std::vector<PacingWatch>{{4096, 8}});
    services::vi::VsyncSource source(clock);
    bool created = false;
    auto event = source.acquire(0, created);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (time.waiting_for < 0 && std::chrono::steady_clock::now() < until) std::this_thread::yield();
    CHECK(time.waiting_for == 10'000'000);
    output.block = !stall_copy;
    auto observer = std::async(std::launch::async, [&] {
        return clock->observe_present(1, 1, [&](std::uint64_t, std::span<std::byte>) {
            if (stall_copy) copy_gate.block();
            return false;
        });
    });
    CHECK(stall_copy ? copy_gate.await_entry() : output.gate.await_entry());
    time.now = 15'000'000;
    auto refresh = std::async(std::launch::async, [&] { return source.wait_sequence(1); });
    auto plan = std::async(std::launch::async, [&] { return clock->next_refresh(); });
    auto poll = std::async(std::launch::async, [&] { return clock->observation_failure(); });
    CHECK(refresh.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready);
    CHECK(plan.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready);
    CHECK(poll.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready);
    // Always unblock before future destructors, including on regression failure.
    copy_gate.release();
    output.gate.release();
    CHECK(refresh.get());
    CHECK(std::holds_alternative<PresentationSchedule>(plan.get()));
    CHECK(!poll.get());
    CHECK(std::holds_alternative<std::monostate>(observer.get()));
    CHECK(std::holds_alternative<std::monostate>(clock->observe_present(2, 1, {})));
    // The signal-completion timestamp is the actual manual time after the
    // delayed wake, not the scheduled 10 ms deadline or the blocked log time.
    CHECK(output.str().find("\t15000000\t10000000\t1") != std::string::npos);
}

void concurrent_observers_sample_in_append_order() {
    Timeline time;
    BlockingBuffer output;
    PacingClock clock(time.base, std::nullopt, std::make_unique<std::ostream>(&output));
    output.block = true;
    time.now = 10'000'000;
    auto first = std::async(std::launch::async, [&] { return clock.observe_present(1, 1, {}); });
    CHECK(output.gate.await_entry());
    time.now = 20'000'000;
    const auto sampled = time.samples.load();
    std::promise<void> entered;
    auto second = std::async(std::launch::async, [&] {
        entered.set_value();
        return clock.observe_present(2, 1, {});
    });
    entered.get_future().wait();
    CHECK(second.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
    CHECK(time.samples == sampled);
    time.now = 30'000'000;
    output.gate.release();
    CHECK(std::holds_alternative<std::monostate>(first.get()));
    CHECK(std::holds_alternative<std::monostate>(second.get()));
    std::istringstream lines(output.str());
    std::string header, one, two;
    std::getline(lines, header); std::getline(lines, one); std::getline(lines, two);
    CHECK(one.starts_with("1\t10000000\t10000000\t0\t0\t"));
    CHECK(two.starts_with("2\t30000000\t30000000\t20000000\t20000000\t"));
}

}
int main() {
    parsers(); relocatable_watches(); grid(); logging_and_off();
    stalled_observers_leave_vi_and_planning_running(false);
    stalled_observers_leave_vi_and_planning_running(true);
    concurrent_observers_sample_in_append_order();
    return failures ? 1 : 0;
}
