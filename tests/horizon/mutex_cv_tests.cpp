// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/process_wide_key.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

namespace {
using namespace std::chrono_literals;
using armigo::horizon::ProcessWideKey;

constexpr std::uint32_t waiter_bit = 0x40000000u;
std::atomic<unsigned> failures{0};

#define CHECK(condition)                                                               \
    do {                                                                               \
        if (!(condition)) {                                                            \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);              \
            ++failures;                                                                \
        }                                                                              \
    } while (false)

[[noreturn]] void bail(unsigned line, const char* what) {
    std::fprintf(stderr, "FAIL %u: %s; aborting so no worker can hang the test\n", line,
                 what);
    std::fflush(stderr);
    std::abort();
}

template <typename Ready>
void require_settles(unsigned line, Ready&& ready) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) bail(line, "condition timed out");
        std::this_thread::yield();
    }
}

#define REQUIRE_SETTLES(condition) require_settles(__LINE__, [&] { return (condition); })

template <typename Result>
void require_ready(unsigned line, std::future<Result>& result) {
    if (result.wait_for(5s) != std::future_status::ready) bail(line, "worker timed out");
}

#define REQUIRE_READY(result) require_ready(__LINE__, result)

void arm_watchdog() {
    std::thread([] {
        std::this_thread::sleep_for(30s);
        std::fprintf(stderr, "FAIL: mutex/CV watchdog expired\n");
        std::fflush(stderr);
        std::abort();
    }).detach();
}

struct GuestMemory {
    static constexpr std::uint64_t base = 0x7040000000;
    static constexpr std::uint64_t stride = sizeof(std::uint32_t);
    std::array<std::atomic<std::uint32_t>, 8> words{};
    std::atomic<std::uint64_t> fail_next_write{0};

    static bool index(std::uint64_t address, std::size_t& result) noexcept {
        if (address < base || (address - base) % stride != 0) return false;
        result = static_cast<std::size_t>((address - base) / stride);
        return result < 8;
    }

    static bool read(void* context, std::uint64_t address, std::uint32_t& value) noexcept {
        auto& self = *static_cast<GuestMemory*>(context);
        std::size_t at = 0;
        if (!index(address, at)) return false;
        value = self.words[at].load(std::memory_order_acquire);
        return true;
    }

    static bool write(void* context, std::uint64_t address, std::uint32_t value) noexcept {
        auto& self = *static_cast<GuestMemory*>(context);
        std::size_t at = 0;
        if (!index(address, at)) return false;
        auto failing_address = address;
        if (self.fail_next_write.compare_exchange_strong(
                failing_address, 0, std::memory_order_acq_rel, std::memory_order_acquire))
            return false;
        self.words[at].store(value, std::memory_order_release);
        return true;
    }

    static ProcessWideKey::ExchangeResult compare_exchange(
        void* context, std::uint64_t address, std::uint32_t& expected,
        std::uint32_t desired) noexcept {
        auto& self = *static_cast<GuestMemory*>(context);
        std::size_t at = 0;
        if (!index(address, at)) return ProcessWideKey::ExchangeResult::invalid_address;
        if (self.words[at].compare_exchange_strong(expected, desired, std::memory_order_acq_rel,
                                                   std::memory_order_acquire))
            return ProcessWideKey::ExchangeResult::exchanged;
        return ProcessWideKey::ExchangeResult::mismatch;
    }

    static constexpr std::uint64_t address(std::size_t at) { return base + at * stride; }

    std::uint32_t load(std::size_t at) const {
        return words[at].load(std::memory_order_acquire);
    }

    void store(std::size_t at, std::uint32_t value) {
        words[at].store(value, std::memory_order_release);
    }
};

struct State {
    GuestMemory memory;
    ProcessWideKey keys;

    ProcessWideKey::GuestWords callbacks() {
        return {GuestMemory::read, GuestMemory::write, &memory,
                GuestMemory::compare_exchange};
    }
};

bool acquire(State& state, std::uint64_t mutex, std::uint32_t tag,
             std::int32_t priority = 32) {
    auto& word = state.memory.words[(mutex - GuestMemory::base) / GuestMemory::stride];
    for (;;) {
        auto observed = word.load(std::memory_order_acquire);
        if ((observed & ~waiter_bit) == tag) return true;
        if (observed == 0) {
            if (word.compare_exchange_weak(observed, tag, std::memory_order_acq_rel,
                                           std::memory_order_acquire))
                return true;
            continue;
        }
        const auto owner = observed & ~waiter_bit;
        if (owner == 0) continue;
        const auto marked = observed | waiter_bit;
        if (!word.compare_exchange_weak(observed, marked, std::memory_order_acq_rel,
                                        std::memory_order_acquire))
            continue;
        if (state.keys.lock(state.callbacks(), mutex, owner, tag, priority) ==
            ProcessWideKey::LockOutcome::invalid_address)
            return false;
    }
}

bool release(State& state, std::uint64_t mutex, std::uint32_t tag) {
    auto& word = state.memory.words[(mutex - GuestMemory::base) / GuestMemory::stride];
    auto expected = tag;
    if (word.compare_exchange_strong(expected, 0, std::memory_order_acq_rel,
                                     std::memory_order_acquire))
        return true;
    return state.keys.unlock(state.callbacks(), mutex);
}

void mixed_waiters_transfer_exact_tags() {
    State state;
    const auto mutex = GuestMemory::address(0);
    const auto key = GuestMemory::address(1);
    constexpr std::uint32_t owner = 0x1111;
    constexpr std::uint32_t ordinary_tag = 0x2222;
    constexpr std::uint32_t condition_tag = 0x3333;

    state.memory.store(0, condition_tag);
    auto condition = std::async(std::launch::async, [&] {
        return state.keys.wait(state.callbacks(), mutex, key, condition_tag, -1, 30);
    });
    REQUIRE_SETTLES(state.keys.waiters(key) == 1);

    state.memory.store(0, owner | waiter_bit);
    auto ordinary = std::async(std::launch::async, [&] {
        return state.keys.lock(state.callbacks(), mutex, owner, ordinary_tag, 30);
    });
    REQUIRE_SETTLES(state.keys.mutex_waiters(mutex) == 1);
    CHECK(state.keys.signal(state.callbacks(), key, 1) ==
          ProcessWideKey::SignalOutcome::signaled);
    REQUIRE_SETTLES(state.keys.mutex_waiters(mutex) == 2);

    CHECK(state.keys.unlock(state.callbacks(), mutex));
    REQUIRE_READY(ordinary);
    CHECK(ordinary.get() == ProcessWideKey::LockOutcome::succeeded);
    CHECK(state.memory.load(0) == (ordinary_tag | waiter_bit));
    CHECK(condition.wait_for(20ms) != std::future_status::ready);

    CHECK(state.keys.unlock(state.callbacks(), mutex));
    REQUIRE_READY(condition);
    CHECK(condition.get() == ProcessWideKey::WaitOutcome::signaled);
    CHECK(state.memory.load(0) == condition_tag);

    // A later condition waiter with higher priority is selected first, but its tagged
    // ownership retains the waiter marker so the ordinary waiter remains reachable.
    State priority_state;
    priority_state.memory.store(0, condition_tag);
    auto priority_condition = std::async(std::launch::async, [&] {
        return priority_state.keys.wait(priority_state.callbacks(), mutex, key, condition_tag,
                                        -1, 10);
    });
    REQUIRE_SETTLES(priority_state.keys.waiters(key) == 1);
    priority_state.memory.store(0, owner | waiter_bit);
    auto priority_ordinary = std::async(std::launch::async, [&] {
        return priority_state.keys.lock(priority_state.callbacks(), mutex, owner, ordinary_tag,
                                        30);
    });
    REQUIRE_SETTLES(priority_state.keys.mutex_waiters(mutex) == 1);
    CHECK(priority_state.keys.signal(priority_state.callbacks(), key, 1) ==
          ProcessWideKey::SignalOutcome::signaled);
    REQUIRE_SETTLES(priority_state.keys.mutex_waiters(mutex) == 2);
    CHECK(priority_state.keys.unlock(priority_state.callbacks(), mutex));
    REQUIRE_READY(priority_condition);
    CHECK(priority_condition.get() == ProcessWideKey::WaitOutcome::signaled);
    CHECK(priority_state.memory.load(0) == (condition_tag | waiter_bit));
    CHECK(priority_state.keys.unlock(priority_state.callbacks(), mutex));
    REQUIRE_READY(priority_ordinary);
    CHECK(priority_ordinary.get() == ProcessWideKey::LockOutcome::succeeded);
    CHECK(priority_state.memory.load(0) == ordinary_tag);
}

void signalled_wait_keeps_original_deadline() {
    State state;
    const auto mutex = GuestMemory::address(0);
    const auto key = GuestMemory::address(1);
    constexpr std::uint32_t owner = 0x4444;
    constexpr std::uint32_t tag = 0x5555;

    state.memory.store(0, tag);
    auto waiter = std::async(std::launch::async, [&] {
        return state.keys.wait(state.callbacks(), mutex, key, tag, 250'000'000, 10);
    });
    REQUIRE_SETTLES(state.keys.waiters(key) == 1);
    state.memory.store(0, owner);
    auto ordinary = std::async(std::launch::async, [&] {
        state.memory.words[0].fetch_or(waiter_bit, std::memory_order_acq_rel);
        return state.keys.lock(state.callbacks(), mutex, owner, 0x6666, 30);
    });
    REQUIRE_SETTLES(state.keys.mutex_waiters(mutex) == 1);
    CHECK(state.keys.signal(state.callbacks(), key, 1) ==
          ProcessWideKey::SignalOutcome::signaled);
    REQUIRE_SETTLES(state.keys.mutex_waiters(mutex) == 2);
    REQUIRE_READY(waiter);
    CHECK(waiter.get() == ProcessWideKey::WaitOutcome::timed_out);
    CHECK(state.keys.mutex_waiters(mutex) == 1);
    CHECK(state.memory.load(0) == (owner | waiter_bit));
    CHECK(state.keys.unlock(state.callbacks(), mutex));
    REQUIRE_READY(ordinary);
    CHECK(ordinary.get() == ProcessWideKey::LockOutcome::succeeded);
    CHECK(state.memory.load(0) == 0x6666);
}

void failed_handoff_completes_selected_waiter() {
    State state;
    const auto mutex = GuestMemory::address(0);
    const auto key = GuestMemory::address(1);
    state.memory.store(0, 0x2222);
    auto waiter = std::async(std::launch::async, [&] {
        return state.keys.wait(state.callbacks(), mutex, key, 0x3333, -1);
    });
    REQUIRE_SETTLES(state.keys.waiters(key) == 1);
    state.memory.store(0, 0x4444);
    CHECK(state.keys.signal(state.callbacks(), key, 1) ==
          ProcessWideKey::SignalOutcome::signaled);
    REQUIRE_SETTLES(state.keys.mutex_waiters(mutex) == 1);
    state.memory.fail_next_write.store(mutex, std::memory_order_release);
    CHECK(!state.keys.unlock(state.callbacks(), mutex));
    REQUIRE_READY(waiter);
    CHECK(waiter.get() == ProcessWideKey::WaitOutcome::invalid_address);
    CHECK(state.keys.mutex_waiters(mutex) == 0);
}

void nonpositive_signal_counts_broadcast() {
    for (const std::int64_t count : {std::int64_t{0}, std::int64_t{-1}}) {
        State state;
        const auto first_mutex = GuestMemory::address(0);
        const auto second_mutex = GuestMemory::address(1);
        const auto key = GuestMemory::address(2);
        state.memory.store(0, 0x1111);
        state.memory.store(1, 0x1111);
        auto first = std::async(std::launch::async, [&] {
            return state.keys.wait(state.callbacks(), first_mutex, key, 0x2222, -1);
        });
        auto second = std::async(std::launch::async, [&] {
            return state.keys.wait(state.callbacks(), second_mutex, key, 0x3333, -1);
        });
        REQUIRE_SETTLES(state.keys.waiters(key) == 2);
        CHECK(state.keys.signal(state.callbacks(), key, count) ==
              ProcessWideKey::SignalOutcome::signaled);
        REQUIRE_READY(first);
        REQUIRE_READY(second);
        CHECK(first.get() == ProcessWideKey::WaitOutcome::signaled);
        CHECK(second.get() == ProcessWideKey::WaitOutcome::signaled);
        CHECK(state.keys.waiters(key) == 0);
        CHECK(state.memory.load(2) == 0);
    }
}

void mutex_and_condition_stress() {
    State state;
    const auto mutex = GuestMemory::address(0);
    const auto key = GuestMemory::address(1);
    constexpr std::size_t mutex_iterations = 1000;
    constexpr std::size_t condition_iterations = 200;
    std::atomic<unsigned> inside{0};
    std::atomic<unsigned> critical_count{0};
    std::atomic<bool> stress_failed{false};

    std::atomic<std::size_t> payload{0};
    std::thread waiter([&] {
        constexpr std::uint32_t tag = 0x5001;
        for (std::size_t iteration = 1; iteration <= condition_iterations; ++iteration) {
            if (!acquire(state, mutex, tag) ||
                state.keys.wait(state.callbacks(), mutex, key, tag, 2'000'000'000) !=
                    ProcessWideKey::WaitOutcome::signaled) {
                stress_failed = true;
                return;
            }
            if ((state.memory.load(0) & ~waiter_bit) != tag) stress_failed = true;
            if (inside.fetch_add(1, std::memory_order_acq_rel) != 0) stress_failed = true;
            if (payload.load(std::memory_order_acquire) != iteration) stress_failed = true;
            if (inside.fetch_sub(1, std::memory_order_acq_rel) != 1) stress_failed = true;
            if (!release(state, mutex, tag)) {
                stress_failed = true;
                return;
            }
        }
    });
    REQUIRE_SETTLES(state.keys.waiters(key) == 1);
    std::vector<std::thread> mutex_workers;
    for (std::uint32_t index = 0; index < 4; ++index) {
        mutex_workers.emplace_back([&, tag = 0x1000u + index] {
            for (std::size_t iteration = 0; iteration < mutex_iterations; ++iteration) {
                if (!acquire(state, mutex, tag)) {
                    stress_failed = true;
                    return;
                }
                if (inside.fetch_add(1, std::memory_order_acq_rel) != 0) stress_failed = true;
                critical_count.fetch_add(1, std::memory_order_relaxed);
                if (inside.fetch_sub(1, std::memory_order_acq_rel) != 1) stress_failed = true;
                if (!release(state, mutex, tag)) {
                    stress_failed = true;
                    return;
                }
            }
        });
    }
    std::thread signaler([&] {
        constexpr std::uint32_t tag = 0x5002;
        for (std::size_t iteration = 1; iteration <= condition_iterations; ++iteration) {
            REQUIRE_SETTLES(state.keys.waiters(key) == 1);
            if (!acquire(state, mutex, tag)) {
                stress_failed = true;
                return;
            }
            if ((state.memory.load(0) & ~waiter_bit) != tag) stress_failed = true;
            if (inside.fetch_add(1, std::memory_order_acq_rel) != 0) stress_failed = true;
            payload.store(iteration, std::memory_order_release);
            const auto signal_result = state.keys.signal(state.callbacks(), key, 1);
            if (inside.fetch_sub(1, std::memory_order_acq_rel) != 1) stress_failed = true;
            if (signal_result != ProcessWideKey::SignalOutcome::signaled ||
                !release(state, mutex, tag)) {
                stress_failed = true;
                return;
            }
        }
    });
    waiter.join();
    signaler.join();
    for (auto& worker : mutex_workers) worker.join();
    CHECK(!stress_failed.load());
    CHECK(critical_count.load() == mutex_iterations * mutex_workers.size());
    CHECK(payload.load() == condition_iterations);
}
}  // namespace

int main() {
    arm_watchdog();
    mixed_waiters_transfer_exact_tags();
    signalled_wait_keeps_original_deadline();
    failed_handoff_completes_selected_waiter();
    nonpositive_signal_counts_broadcast();
    mutex_and_condition_stress();
    std::fprintf(stderr, "Mutex/CV failures: %u\n", failures.load());
    return failures.load() == 0 ? 0 : 1;
}
