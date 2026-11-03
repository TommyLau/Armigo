// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/horizon/startup_queries.h>

#include <array>
#include <cstdio>
#include <chrono>
#include <future>
#include <limits>
#include <thread>

namespace {
using namespace armigo::horizon;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)

void result_is(const EventResult& result, std::uint32_t expected) {
    CHECK(result.disposition == EventDisposition::handled);
    CHECK(result.result == expected && result.detail == nullptr);
}

void unsupported(const EventResult& result) {
    CHECK(result.disposition == EventDisposition::unsupported);
    CHECK(!result.index && result.detail != nullptr);
}

void test_ready_reset_and_poll() {
    HandleTable table(4);
    auto event = std::make_shared<Event>();
    const std::array handles{table.insert(event).value()};
    auto result = wait_event(table, handles, 0);
    result_is(result, 0xea01);
    CHECK(!result.index && !event->is_signaled());
    result_is(reset_event(table, handles[0]), 0xfa01);
    result_is(wait_event(table, handles, 1), 0xea01);
    CHECK(!event->is_signaled());

    event->signal();
    event->signal();
    for (const auto timeout : {std::int64_t{0}, std::int64_t{-1}, std::int64_t{1},
                               std::numeric_limits<std::int64_t>::max()}) {
        result = wait_event(table, handles, timeout);
        result_is(result, 0);
        CHECK(result.index == 0 && event->is_signaled());
    }
    result = reset_event(table, handles[0]);
    result_is(result, 0);
    CHECK(!result.index && !event->is_signaled());
    CHECK(!event->reset());
    result = wait_event(table, handles, 0);
    result_is(result, 0xea01);
    CHECK(!result.index);
    result_is(reset_event(table, handles[0]), 0xfa01);
    event->signal();
    result_is(wait_event(table, handles, -1), 0);
}

void test_blocking_signal_and_timeout() {
    using namespace std::chrono_literals;
    HandleTable table(1);
    const auto event = std::make_shared<Event>();
    const std::array handles{table.insert(event).value()};
    for (const auto timeout : {std::int64_t{-1}, std::int64_t{1'000'000'000},
                               std::numeric_limits<std::int64_t>::max()}) {
        event->reset();
        std::promise<void> started;
        std::promise<EventResult> finished;
        auto ready = started.get_future();
        auto result = finished.get_future();
        std::thread waiter([&] {
            started.set_value();
            finished.set_value(wait_event(table, handles, timeout));
        });
        ready.wait();
        CHECK(result.wait_for(10ms) == std::future_status::timeout);
        event->signal();
        const auto completed = result.get();
        waiter.join();
        result_is(completed, 0);
        CHECK(completed.index == 0 && event->is_signaled());
        result_is(wait_event(table, handles, 0), 0);
    }
    event->reset();
    const auto start = std::chrono::steady_clock::now();
    const auto timeout = wait_event(table, handles, 20'000'000);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    result_is(timeout, 0xea01);
    CHECK(!timeout.index && !event->is_signaled());
    CHECK(elapsed >= 20ms);
}

class OtherObject final : public KernelObject {};

void test_unsupported_and_invalid_handles() {
    HandleTable table(4);
    const std::array ready{table.insert(std::make_shared<Event>(true)).value()};
    const std::array two{ready[0], ready[0]};
    unsupported(wait_event(table, {}, 0));
    unsupported(wait_event(table, two, -1));
    CHECK(table.lookup_as<Event>(ready[0])->is_signaled());
    const auto other = table.insert(std::make_shared<OtherObject>()).value();
    const std::array other_handles{other};
    unsupported(wait_event(table, other_handles, 0));
    unsupported(reset_event(table, other));
    const std::array<Handle, 1> invalid{0};
    const auto bad_wait = wait_event(table, invalid, 0);
    result_is(bad_wait, 0xe401);
    CHECK(!bad_wait.index);
    result_is(reset_event(table, 0), 0xe401);
    CHECK(table.close(ready[0]));
    result_is(wait_event(table, ready, -1), 0xe401);
    result_is(reset_event(table, ready[0]), 0xe401);
}

void test_owned_signal_and_handle_pinning() {
    HandleTable table(2);
    auto parent_event = std::make_shared<Event>(true);
    std::weak_ptr<Event> weak = parent_event;
    const auto first = table.insert(parent_event).value();
    const auto second = table.insert(parent_event).value();
    auto pin = table.lookup_as<Event>(first);
    CHECK(table.close(first));
    parent_event.reset();
    const std::array handles{second};
    result_is(wait_event(table, handles, -1), 0);
    CHECK(pin->reset());
    CHECK(!pin->is_signaled());
    pin->signal();
    CHECK(table.close(second) && !weak.expired());
    CHECK(pin->is_signaled());
    result_is(reset_event(table, second), 0xe401);
    CHECK(pin->is_signaled());
    pin.reset();
    CHECK(weak.expired());
}

void test_atomic_manual_reset() {
    HandleTable table(1);
    const auto event = std::make_shared<Event>(true);
    const auto handle = table.insert(event).value();
    CHECK(table.lookup_as<Event>(handle) == event);
    std::array<bool, 2> results;
    std::thread first([&] { results[0] = event->reset(); });
    std::thread second([&] { results[1] = event->reset(); });
    first.join();
    second.join();
    CHECK(results[0] != results[1]);
    CHECK(!event->is_signaled());
}
}  // namespace

void test_thread_exit_wait() {
    using namespace std::chrono_literals;
    HandleTable table(4);
    ProcessConfiguration configuration;
    configuration.heap_address = 0x1000000000;
    configuration.alias_address = 0x2000000000;
    configuration.stack_address = 0x6000000000;
    configuration.allowed_core_mask = 1;
    const auto built = ProcessRecord::create(configuration, {});
    CHECK(built.ok());
    auto thread = std::make_shared<ThreadRecord>(built.process, 7, 32);
    const std::array handles{table.insert(thread).value()};
    // A live thread times out on a zero or short wait and writes no index.
    auto result = wait_event(table, handles, 0);
    result_is(result, 0xea01);
    CHECK(!result.index);
    result = wait_event(table, handles, 1'000'000);
    result_is(result, 0xea01);
    CHECK(!result.index);
    // The thread object wakes its waiters when it finishes.
    std::thread exiting([&] {
        std::this_thread::sleep_for(5ms);
        thread->mark_exited();
    });
    result = wait_event(table, handles, -1);
    result_is(result, 0);
    CHECK(result.index == 0 && thread->exited());
    exiting.join();
    // An exited thread reports completion immediately, including a zero-timeout poll.
    result = wait_event(table, handles, 0);
    result_is(result, 0);
    CHECK(result.index == 0);
    // ResetSignal remains an event operation.
    CHECK(reset_event(table, handles[0]).disposition == EventDisposition::unsupported);
}

int main() {
    test_ready_reset_and_poll();
    test_blocking_signal_and_timeout();
    test_unsupported_and_invalid_handles();
    test_owned_signal_and_handle_pinning();
    test_atomic_manual_reset();
    test_thread_exit_wait();
    if (failures) return 1;
    std::puts("startup-event checks passed");
    return 0;
}
