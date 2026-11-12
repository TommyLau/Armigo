// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/startup_queries.h>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace armigo::horizon;
using namespace std::chrono_literals;

constexpr std::uint64_t word_address = 0x12345000;
constexpr std::uint64_t wide_word_address = 0x12346000;
constexpr std::uint32_t invalid_memory = 0xd401;
constexpr std::uint32_t invalid_address = 0xcc01;
constexpr std::uint32_t invalid_combination = 0xf001;
constexpr std::uint32_t invalid_state = 0xfa01;
constexpr std::uint32_t timed_out = 0xea01;

static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

std::atomic<unsigned> failures{0};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; } } while (false)

[[noreturn]] void bail(unsigned line, const char* reason) {
    std::fprintf(stderr, "Check failed at line %u: %s; aborting to avoid a stuck worker\n",
                 line, reason);
    std::fflush(stderr);
    std::abort();
}

template <typename Predicate>
bool settles_within(std::chrono::steady_clock::duration timeout, Predicate&& predicate) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

#define REQUIRE_SETTLES(condition)                                                \
    do {                                                                          \
        if (!settles_within(5s, [&] { return (condition); }))                     \
            bail(__LINE__, #condition);                                           \
    } while (false)

template <typename Future>
void require_ready(Future& future, unsigned line) {
    if (future.wait_for(5s) != std::future_status::ready)
        bail(line, "worker did not finish within the test deadline");
}

#define REQUIRE_READY(future) require_ready((future), __LINE__)

ProcessConfiguration configuration() {
    ProcessConfiguration result;
    result.heap_address = 0x1000000000;
    result.alias_address = 0x2000000000;
    result.stack_address = 0x6000000000;
    result.allowed_core_mask = 1;
    result.entropy = {1, 2, 3, 4};
    return result;
}

MemoryAccounting accounting() {
    return {0, 0, 0, 0, 0x100000, 0};
}

struct AtomicBacking {
    std::atomic<std::uint32_t> word{0};
    std::atomic<std::uint64_t> wide_word{0};
    std::atomic<unsigned> reads{0};
    std::atomic<unsigned> writes{0};
    std::atomic<unsigned> exchanges{0};
    std::atomic<unsigned> exchange_mutations{0};
    std::atomic<unsigned> wide_reads{0};

    std::mutex gate_mutex;
    std::condition_variable gate_changed;
    bool pause_next_read = false;
    bool read_paused = false;
    bool resume_read = false;
    bool pause_next_exchange = false;
    bool exchange_paused = false;
    bool resume_exchange = false;
    bool inject_next_exchange_mismatch = false;
    std::uint32_t injected_exchange_value = 0;

    void reset_counts() noexcept {
        reads.store(0, std::memory_order_relaxed);
        writes.store(0, std::memory_order_relaxed);
        exchanges.store(0, std::memory_order_relaxed);
        exchange_mutations.store(0, std::memory_order_relaxed);
        wide_reads.store(0, std::memory_order_relaxed);
    }

    void arm_read_pause() {
        const std::lock_guard lock(gate_mutex);
        pause_next_read = true;
        read_paused = false;
        resume_read = false;
    }

    void arm_exchange_pause() {
        const std::lock_guard lock(gate_mutex);
        pause_next_exchange = true;
        exchange_paused = false;
        resume_exchange = false;
    }

    void inject_exchange_mismatch(std::uint32_t value) {
        const std::lock_guard lock(gate_mutex);
        inject_next_exchange_mismatch = true;
        injected_exchange_value = value;
    }

    bool wait_for_read_pause() {
        std::unique_lock lock(gate_mutex);
        return gate_changed.wait_for(lock, 5s, [&] { return read_paused; });
    }

    bool wait_for_exchange_pause() {
        std::unique_lock lock(gate_mutex);
        return gate_changed.wait_for(lock, 5s, [&] { return exchange_paused; });
    }

    void release_read() {
        {
            const std::lock_guard lock(gate_mutex);
            resume_read = true;
        }
        gate_changed.notify_all();
    }

    void release_exchange() {
        {
            const std::lock_guard lock(gate_mutex);
            resume_exchange = true;
        }
        gate_changed.notify_all();
    }

    static bool read(void* raw, std::uint64_t address,
                     std::span<std::byte, 4> output) noexcept {
        auto& backing = *static_cast<AtomicBacking*>(raw);
        if (address != word_address) return false;
        backing.reads.fetch_add(1, std::memory_order_relaxed);
        const auto observed = backing.word.load(std::memory_order_acquire);
        {
            std::unique_lock lock(backing.gate_mutex);
            if (backing.pause_next_read) {
                backing.pause_next_read = false;
                backing.read_paused = true;
                backing.gate_changed.notify_all();
                backing.gate_changed.wait(lock, [&] { return backing.resume_read; });
                backing.resume_read = false;
                backing.read_paused = false;
            }
        }
        std::memcpy(output.data(), &observed, output.size());
        return true;
    }

    static bool write(void* raw, std::uint64_t address,
                      std::span<const std::byte, 4> input) noexcept {
        auto& backing = *static_cast<AtomicBacking*>(raw);
        if (address != word_address) return false;
        std::uint32_t desired = 0;
        std::memcpy(&desired, input.data(), input.size());
        backing.writes.fetch_add(1, std::memory_order_relaxed);
        backing.word.store(desired, std::memory_order_release);
        return true;
    }

    static ProcessWideKey::ExchangeResult compare_exchange(
        void* raw, std::uint64_t address, std::uint32_t& expected,
        std::uint32_t desired) noexcept {
        auto& backing = *static_cast<AtomicBacking*>(raw);
        if (address != word_address)
            return ProcessWideKey::ExchangeResult::invalid_address;
        backing.exchanges.fetch_add(1, std::memory_order_relaxed);
        {
            std::unique_lock lock(backing.gate_mutex);
            if (backing.pause_next_exchange) {
                backing.pause_next_exchange = false;
                backing.exchange_paused = true;
                backing.gate_changed.notify_all();
                backing.gate_changed.wait(lock, [&] { return backing.resume_exchange; });
                backing.resume_exchange = false;
                backing.exchange_paused = false;
            }
            if (backing.inject_next_exchange_mismatch) {
                backing.inject_next_exchange_mismatch = false;
                backing.word.store(backing.injected_exchange_value, std::memory_order_release);
                expected = backing.injected_exchange_value;
                return ProcessWideKey::ExchangeResult::mismatch;
            }
        }
        const auto requested_expected = expected;
        if (backing.word.compare_exchange_strong(expected, desired,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
            if (desired != requested_expected)
                backing.exchange_mutations.fetch_add(1, std::memory_order_relaxed);
            return ProcessWideKey::ExchangeResult::exchanged;
        }
        return ProcessWideKey::ExchangeResult::mismatch;
    }

    static bool read_double(void* raw, std::uint64_t address,
                            std::uint64_t& output) noexcept {
        auto& backing = *static_cast<AtomicBacking*>(raw);
        if (address != wide_word_address) return false;
        backing.wide_reads.fetch_add(1, std::memory_order_relaxed);
        output = backing.wide_word.load(std::memory_order_acquire);
        return true;
    }
};

GuestCopyOut callbacks(AtomicBacking& backing, bool with_exchange = true) {
    GuestCopyOut result{
        .write_word = AtomicBacking::write,
        .read_word = AtomicBacking::read,
        .context = &backing,
        .compare_exchange_word = with_exchange ? AtomicBacking::compare_exchange : nullptr,
        .read_double_word = AtomicBacking::read_double,
    };
    return result;
}

struct Fixture {
    std::shared_ptr<ProcessRecord> process;
    HandleTable handles{8};
    std::shared_ptr<ThreadRecord> thread;
    AtomicBacking backing;
    StartupQueries queries;

    explicit Fixture(bool with_exchange = true)
        : process(ProcessRecord::create(configuration(), accounting()).process),
          thread(std::make_shared<ThreadRecord>(process, 1, 17)),
          queries(process, handles, thread, callbacks(backing, with_exchange)) {}
};

QueryRegisters wait_arguments(std::uint32_t type, std::uint64_t value,
                              std::int64_t timeout) {
    QueryRegisters registers{};
    registers[0] = type == 3 ? wide_word_address : word_address;
    registers[1] = type;
    registers[2] = value;
    registers[3] = static_cast<std::uint64_t>(timeout);
    return registers;
}

QueryRegisters signal_arguments(std::uint32_t type, std::uint32_t value,
                                std::int32_t count) {
    QueryRegisters registers{};
    registers[0] = word_address;
    registers[1] = type;
    registers[2] = value;
    registers[3] = static_cast<std::uint32_t>(count);
    return registers;
}

QueryRegisters dispatch_wait(Fixture& fixture, std::uint32_t type, std::uint64_t value,
                             std::int64_t timeout) {
    auto registers = wait_arguments(type, value, timeout);
    CHECK(fixture.queries.dispatch(0x34, registers) == QueryDisposition::handled);
    return registers;
}

QueryRegisters dispatch_signal(Fixture& fixture, std::uint32_t type, std::uint32_t value,
                               std::int32_t count) {
    auto registers = signal_arguments(type, value, count);
    CHECK(fixture.queries.dispatch(0x35, registers) == QueryDisposition::handled);
    return registers;
}

using WaitFuture = std::future<QueryRegisters>;

std::vector<WaitFuture> enroll_waiters(Fixture& fixture, unsigned count,
                                       std::uint32_t expected) {
    std::vector<WaitFuture> waiters;
    waiters.reserve(count);
    for (unsigned index = 0; index != count; ++index) {
        waiters.emplace_back(std::async(std::launch::async, [&fixture, expected] {
            return dispatch_wait(fixture, 2, expected, -1);
        }));
    }
    REQUIRE_SETTLES(fixture.queries.arbitration()->waiters(word_address) == count);
    return waiters;
}

void release_all(Fixture& fixture) {
    const auto result = dispatch_signal(fixture, 0, 0, -1);
    CHECK(result[0] == 0);
}

void require_successful_waits(std::vector<WaitFuture>& waiters) {
    for (auto& waiter : waiters) {
        REQUIRE_READY(waiter);
        CHECK(waiter.get()[0] == 0);
    }
}

void test_count_limited_signal() {
    Fixture fixture;
    fixture.backing.word.store(7, std::memory_order_release);
    auto waiters = enroll_waiters(fixture, 2, 7);

    const auto signal = dispatch_signal(fixture, 0, 0, 1);
    CHECK(signal[0] == 0);
    REQUIRE_SETTLES(waiters[0].wait_for(0s) == std::future_status::ready ||
                    waiters[1].wait_for(0s) == std::future_status::ready);
    const auto first = waiters[0].wait_for(0s) == std::future_status::ready ? 0u : 1u;
    const auto second = first ^ 1u;
    CHECK(waiters[second].wait_for(100ms) == std::future_status::timeout);
    CHECK(fixture.queries.arbitration()->waiters(word_address) == 1);

    release_all(fixture);
    require_successful_waits(waiters);
    CHECK(fixture.queries.arbitration()->waiters(word_address) == 0);
}

void test_compare_enroll_signal_serialization() {
    Fixture fixture;
    fixture.backing.word.store(1, std::memory_order_release);
    fixture.backing.arm_read_pause();
    auto waiter = std::async(std::launch::async, [&] {
        return dispatch_wait(fixture, 2, 1, -1);
    });
    if (!fixture.backing.wait_for_read_pause()) bail(__LINE__, "wait comparison did not pause");

    std::atomic<bool> signal_started{false};
    auto signaler = std::async(std::launch::async, [&] {
        signal_started.store(true, std::memory_order_release);
        return dispatch_signal(fixture, 1, 1, -1);
    });
    REQUIRE_SETTLES(signal_started.load(std::memory_order_acquire));
    CHECK(signaler.wait_for(100ms) == std::future_status::timeout);
    fixture.backing.release_read();

    REQUIRE_READY(waiter);
    REQUIRE_READY(signaler);
    CHECK(waiter.get()[0] == 0);
    CHECK(signaler.get()[0] == 0);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 2);
    CHECK(fixture.queries.arbitration()->waiters(word_address) == 0);
}

void test_native_exchange_wins_before_signal_exchange() {
    Fixture fixture;
    fixture.backing.word.store(1, std::memory_order_release);
    auto waiter = enroll_waiters(fixture, 1, 1);
    fixture.backing.arm_exchange_pause();
    auto signaler = std::async(std::launch::async, [&] {
        return dispatch_signal(fixture, 1, 1, -1);
    });
    if (!fixture.backing.wait_for_exchange_pause())
        bail(__LINE__, "signal compare-exchange did not pause");

    std::uint32_t expected = 1;
    CHECK(fixture.backing.word.compare_exchange_strong(expected, 9,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_acquire));
    fixture.backing.release_exchange();
    REQUIRE_READY(signaler);
    CHECK(signaler.get()[0] == invalid_state);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 9);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 0);
    CHECK(waiter[0].wait_for(100ms) == std::future_status::timeout);
    CHECK(fixture.queries.arbitration()->waiters(word_address) == 1);
    release_all(fixture);
    require_successful_waits(waiter);
}

void test_decrement_retry_rechecks_signed_predicate() {
    Fixture fixture;
    fixture.backing.word.store(0, std::memory_order_release);
    fixture.backing.inject_exchange_mismatch(2);
    const auto wait = dispatch_wait(fixture, 1, 2, 0);
    CHECK(wait[0] == invalid_state);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 2);
    CHECK(fixture.backing.exchanges.load(std::memory_order_relaxed) == 1);
    CHECK(fixture.queries.arbitration()->waiters(word_address) == 0);

    // A raced value that is still below the threshold is decremented on retry.
    fixture.backing.word.store(0, std::memory_order_release);
    fixture.backing.reset_counts();
    fixture.backing.inject_exchange_mismatch(1);
    const auto retried = dispatch_wait(fixture, 1, 3, 0);
    CHECK(retried[0] == timed_out);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 0);
    CHECK(fixture.backing.exchanges.load(std::memory_order_relaxed) == 2);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 1);
}

void test_type2_signal_formula() {
    Fixture fixture;

    // N=0: V+1, including modulo wrap.
    fixture.backing.word.store(std::numeric_limits<std::uint32_t>::max(),
                               std::memory_order_release);
    fixture.backing.reset_counts();
    CHECK(dispatch_signal(fixture, 2, std::numeric_limits<std::uint32_t>::max(), 1)[0] == 0);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 0);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 1);

    // N>0 and C<=0: V-1 and all waiters complete.
    fixture.backing.word.store(5, std::memory_order_release);
    auto all = enroll_waiters(fixture, 2, 5);
    fixture.backing.reset_counts();
    CHECK(dispatch_signal(fixture, 2, 5, 0)[0] == 0);
    require_successful_waits(all);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 4);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 1);

    // A negative count follows the same all-waiters rule.
    fixture.backing.word.store(5, std::memory_order_release);
    auto negative_all = enroll_waiters(fixture, 2, 5);
    fixture.backing.reset_counts();
    CHECK(dispatch_signal(fixture, 2, 5, -1)[0] == 0);
    require_successful_waits(negative_all);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 4);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 1);

    // 0<N<=C and C>0: V-1 and every present waiter completes.
    fixture.backing.word.store(0, std::memory_order_release);
    auto enough = enroll_waiters(fixture, 2, 0);
    fixture.backing.reset_counts();
    CHECK(dispatch_signal(fixture, 2, 0, 2)[0] == 0);
    require_successful_waits(enough);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) ==
          std::numeric_limits<std::uint32_t>::max());
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 1);

    // The same decrement applies when the positive count strictly exceeds N.
    fixture.backing.word.store(6, std::memory_order_release);
    auto surplus = enroll_waiters(fixture, 2, 6);
    fixture.backing.reset_counts();
    CHECK(dispatch_signal(fixture, 2, 6, 3)[0] == 0);
    require_successful_waits(surplus);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 5);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 1);

    // N>C and C>0: compare V without writing it, then release only C waiters.
    fixture.backing.word.store(9, std::memory_order_release);
    auto partial = enroll_waiters(fixture, 2, 9);
    fixture.backing.reset_counts();
    CHECK(dispatch_signal(fixture, 2, 9, 1)[0] == 0);
    REQUIRE_SETTLES(partial[0].wait_for(0s) == std::future_status::ready ||
                    partial[1].wait_for(0s) == std::future_status::ready);
    const auto first = partial[0].wait_for(0s) == std::future_status::ready ? 0u : 1u;
    CHECK(partial[first ^ 1u].wait_for(100ms) == std::future_status::timeout);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 9);
    CHECK(fixture.backing.reads.load(std::memory_order_relaxed) == 1);
    CHECK(fixture.backing.exchanges.load(std::memory_order_relaxed) == 0);
    CHECK(fixture.backing.writes.load(std::memory_order_relaxed) == 0);
    release_all(fixture);
    require_successful_waits(partial);
}

void test_modulo_decrement() {
    Fixture fixture;
    fixture.backing.word.store(std::numeric_limits<std::uint32_t>::max(),
                               std::memory_order_release);
    CHECK(dispatch_signal(fixture, 1, std::numeric_limits<std::uint32_t>::max(), -1)[0] == 0);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 0);

    fixture.backing.word.store(0x80000000u, std::memory_order_release);
    fixture.backing.reset_counts();
    const auto wait = dispatch_wait(fixture, 1, 0, 0);
    CHECK(wait[0] == timed_out);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 0x7fffffffu);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 1);
}

void test_atomic_wide_read() {
    Fixture fixture;
    fixture.backing.wide_word.store(0x1122334455667788ull, std::memory_order_release);
    fixture.backing.reset_counts();
    const auto wait = dispatch_wait(fixture, 3, 0x1122334455667788ull, 0);
    CHECK(wait[0] == timed_out);
    CHECK(fixture.backing.wide_reads.load(std::memory_order_relaxed) == 1);
    CHECK(fixture.backing.reads.load(std::memory_order_relaxed) == 0);
}

void test_missing_exchange_is_side_effect_free() {
    Fixture fixture(false);
    fixture.backing.word.store(3, std::memory_order_release);
    fixture.backing.reset_counts();

    auto wait = wait_arguments(1, 5, 0);
    for (std::size_t index = 4; index != wait.size(); ++index)
        wait[index] = 0xabc0000000000000ull + index;
    const auto wait_before = wait;
    CHECK(fixture.queries.dispatch(0x34, wait) == QueryDisposition::unsupported_operation);
    CHECK(wait == wait_before);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 3);
    CHECK(fixture.queries.arbitration()->waiters(word_address) == 0);

    auto signal = signal_arguments(2, 3, -1);
    for (std::size_t index = 4; index != signal.size(); ++index)
        signal[index] = 0xdef0000000000000ull + index;
    const auto signal_before = signal;
    CHECK(fixture.queries.dispatch(0x35, signal) == QueryDisposition::unsupported_operation);
    CHECK(signal == signal_before);
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == 3);
    CHECK(fixture.backing.reads.load(std::memory_order_relaxed) == 0);
    CHECK(fixture.backing.writes.load(std::memory_order_relaxed) == 0);
}

void test_address_validation_and_missing_callbacks() {
    auto process = ProcessRecord::create(configuration(), accounting()).process;
    HandleTable handles(8);
    auto thread = std::make_shared<ThreadRecord>(process, 2, 17);
    StartupQueries no_callbacks(process, handles, thread);

    const auto plain_signal = [&](std::uint64_t address, std::uint32_t type = 0) {
        auto registers = signal_arguments(type, 0, 1);
        registers[0] = address;
        return std::pair{no_callbacks.dispatch(0x35, registers), registers};
    };
    // The kernel's word envelope is [0x200000, 0x40000000000), broader than the
    // mapped 39-bit space; a type-0 signal never reads its word.
    for (const auto address : {std::uint64_t{0x200000}, std::uint64_t{0x7ffffffffc},
                               std::uint64_t{0x8000000000}, std::uint64_t{0x3fffffffffc}}) {
        const auto [disposition, registers] = plain_signal(address);
        CHECK(disposition == QueryDisposition::handled);
        CHECK(registers[0] == 0);
    }

    // Numeric range validation precedes alignment and type validation.
    for (const auto address : {std::uint64_t{0}, std::uint64_t{0x1fffff},
                               std::uint64_t{0x40000000000}}) {
        auto wait = wait_arguments(99, 0, 0);
        wait[0] = address;
        CHECK(no_callbacks.dispatch(0x34, wait) == QueryDisposition::handled);
        CHECK(wait[0] == invalid_memory);
        const auto [disposition, signal] = plain_signal(address, 99);
        CHECK(disposition == QueryDisposition::handled);
        CHECK(signal[0] == invalid_memory);
    }

    auto wait = wait_arguments(99, 0, 0);
    wait[0] = word_address + 1;
    CHECK(no_callbacks.dispatch(0x34, wait) == QueryDisposition::handled);
    CHECK(wait[0] == invalid_address);
    auto [signal_disposition, signal] = plain_signal(word_address + 1, 99);
    CHECK(signal_disposition == QueryDisposition::handled);
    CHECK(signal[0] == invalid_address);

    wait = wait_arguments(99, 0, 0);
    CHECK(no_callbacks.dispatch(0x34, wait) == QueryDisposition::handled);
    CHECK(wait[0] == invalid_combination);
    auto invalid_type = plain_signal(word_address, 99);
    CHECK(invalid_type.first == QueryDisposition::handled);
    CHECK(invalid_type.second[0] == invalid_combination);

    // A valid operation that needs guest memory remains unsupported when its
    // callback capability is absent and leaves all registers unchanged.
    for (const auto type : {std::uint32_t{0}, std::uint32_t{1}, std::uint32_t{2},
                            std::uint32_t{3}}) {
        auto missing = wait_arguments(type, 1, 0);
        const auto before = missing;
        CHECK(no_callbacks.dispatch(0x34, missing) == QueryDisposition::unsupported_operation);
        CHECK(missing == before);
    }
    for (const auto type : {std::uint32_t{1}, std::uint32_t{2}}) {
        auto missing = signal_arguments(type, 1, 1);
        const auto before = missing;
        CHECK(no_callbacks.dispatch(0x35, missing) == QueryDisposition::unsupported_operation);
        CHECK(missing == before);
    }

    // A callback-reported mapping fault has the memory result, not the
    // alignment result used for malformed numeric addresses.
    Fixture fixture;
    wait = wait_arguments(2, 1, 0);
    wait[0] = word_address + 0x100;
    CHECK(fixture.queries.dispatch(0x34, wait) == QueryDisposition::handled);
    CHECK(wait[0] == invalid_memory);
    signal = signal_arguments(1, 1, 1);
    signal[0] = word_address + 0x100;
    CHECK(fixture.queries.dispatch(0x35, signal) == QueryDisposition::handled);
    CHECK(signal[0] == invalid_memory);

    // Words inside the kernel envelope but above the mapped 39-bit space pass the
    // range and alignment checks; the failed access then gives the memory result
    // for every type that touches the word, queues no waiter and changes nothing.
    const auto word_before = fixture.backing.word.load(std::memory_order_acquire);
    for (const auto address : {std::uint64_t{0x8000000000}, std::uint64_t{0x3fffffffff8}}) {
        for (const auto type : {std::uint32_t{0}, std::uint32_t{1}, std::uint32_t{2},
                                std::uint32_t{3}}) {
            wait = wait_arguments(type, 0, 0);
            wait[0] = address;
            CHECK(fixture.queries.dispatch(0x34, wait) == QueryDisposition::handled);
            CHECK(wait[0] == invalid_memory);
            CHECK(fixture.queries.arbitration()->waiters(address) == 0);
        }
        for (const auto type : {std::uint32_t{1}, std::uint32_t{2}}) {
            signal = signal_arguments(type, 0, 1);
            signal[0] = address;
            CHECK(fixture.queries.dispatch(0x35, signal) == QueryDisposition::handled);
            CHECK(signal[0] == invalid_memory);
        }
        signal = signal_arguments(0, 0, 1);
        signal[0] = address;
        CHECK(fixture.queries.dispatch(0x35, signal) == QueryDisposition::handled);
        CHECK(signal[0] == 0);
    }
    CHECK(fixture.backing.word.load(std::memory_order_acquire) == word_before);
    CHECK(fixture.backing.writes.load(std::memory_order_relaxed) == 0);
    CHECK(fixture.backing.exchange_mutations.load(std::memory_order_relaxed) == 0);
}

void test_priority_and_fifo_selection() {
    Arbitration arbitration;
    std::mutex completed_mutex;
    std::condition_variable completed_changed;
    std::vector<unsigned> completed;
    std::vector<std::future<void>> waiters;
    constexpr std::array<std::uint32_t, 3> priorities{30, 10, 10};

    for (std::size_t index = 0; index != priorities.size(); ++index) {
        waiters.emplace_back(std::async(std::launch::async, [&, index] {
            const auto outcome = arbitration.wait_transaction(
                word_address, -1, [](void*) noexcept { return ArbitrateOutcome::succeeded; },
                nullptr, priorities[index]);
            CHECK(outcome == ArbitrateOutcome::succeeded);
            {
                const std::lock_guard lock(completed_mutex);
                completed.push_back(static_cast<unsigned>(index));
            }
            completed_changed.notify_all();
        }));
        REQUIRE_SETTLES(arbitration.waiters(word_address) == index + 1);
    }

    constexpr std::array<unsigned, 3> expected_order{1, 2, 0};
    for (std::size_t selection = 0; selection != expected_order.size(); ++selection) {
        CHECK(arbitration.signal(word_address, 1) == 1);
        std::unique_lock lock(completed_mutex);
        if (!completed_changed.wait_for(lock, 5s, [&] {
                return completed.size() == selection + 1;
            })) {
            bail(__LINE__, "selected arbitration waiter did not complete");
        }
        CHECK(completed.back() == expected_order[selection]);
    }
    for (auto& waiter : waiters) {
        REQUIRE_READY(waiter);
        waiter.get();
    }
    CHECK(arbitration.waiters(word_address) == 0);
}

void test_host_thread_lost_wake_stress() {
    constexpr std::size_t iterations = 128;
    Fixture fixture;
    std::barrier phase(3);
    std::array<std::uint64_t, iterations> wait_results{};
    std::array<std::uint64_t, iterations> signal_results{};
    std::array<QueryDisposition, iterations> wait_dispositions{};
    std::array<QueryDisposition, iterations> signal_dispositions{};

    std::thread waiter([&] {
        for (std::size_t index = 0; index != iterations; ++index) {
            phase.arrive_and_wait();
            auto registers = wait_arguments(2, 1, 100'000'000);
            wait_dispositions[index] = fixture.queries.dispatch(0x34, registers);
            wait_results[index] = registers[0];
            phase.arrive_and_wait();
        }
    });
    std::thread signaler([&] {
        for (std::size_t index = 0; index != iterations; ++index) {
            phase.arrive_and_wait();
            auto registers = signal_arguments(1, 1, -1);
            signal_dispositions[index] = fixture.queries.dispatch(0x35, registers);
            signal_results[index] = registers[0];
            phase.arrive_and_wait();
        }
    });
    for (std::size_t index = 0; index != iterations; ++index) {
        fixture.backing.word.store(1, std::memory_order_release);
        phase.arrive_and_wait();
        phase.arrive_and_wait();
        CHECK(fixture.backing.word.load(std::memory_order_acquire) == 2);
        CHECK(fixture.queries.arbitration()->waiters(word_address) == 0);
    }
    waiter.join();
    signaler.join();
    // Publish the ordinary result arrays through thread joins. This also makes their
    // synchronization visible to host race detectors that do not model std::barrier.
    for (std::size_t index = 0; index != iterations; ++index) {
        CHECK(wait_dispositions[index] == QueryDisposition::handled);
        CHECK(signal_dispositions[index] == QueryDisposition::handled);
        CHECK(signal_results[index] == 0);
        CHECK(wait_results[index] == 0 || wait_results[index] == invalid_state);
        CHECK(wait_results[index] != timed_out);
    }
}

} // namespace

int main() {
    test_count_limited_signal();
    test_compare_enroll_signal_serialization();
    test_native_exchange_wins_before_signal_exchange();
    test_decrement_retry_rechecks_signed_predicate();
    test_type2_signal_formula();
    test_modulo_decrement();
    test_atomic_wide_read();
    test_missing_exchange_is_side_effect_free();
    test_address_validation_and_missing_callbacks();
    test_priority_and_fifo_selection();
    test_host_thread_lost_wake_stress();
    return failures.load(std::memory_order_relaxed) == 0 ? 0 : 1;
}
