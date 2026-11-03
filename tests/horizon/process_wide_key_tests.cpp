// ============================================================================
// Armigo Tests - Process-Wide Condition Variable
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/arbitration.h>
#include <armigo/horizon/process_wide_key.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

namespace {
std::atomic<unsigned> failures{0};
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

// A missed handshake or grant would leave workers blocked in unbounded guest
// waits: a joined std::thread would never exit and an std::async future would
// block in its destructor. The test records the failure and aborts instead.
[[noreturn]] void bail(unsigned line, const char* what) {
    std::fprintf(stderr, "FAIL %u: %s; aborting so no worker can hang the test\n", line, what);
    std::fflush(stderr);
    std::abort();
}

// Bounded steady-clock handshake. The live waiters count is the enqueue proof;
// a worker-set flag can be observed before the actual enqueue.
template <typename Ready>
bool settles_within(std::chrono::steady_clock::duration budget, Ready&& ready) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

#define REQUIRE_SETTLES(condition)                                              \
    do {                                                                        \
        if (!settles_within(std::chrono::seconds(5), [&] { return (condition); })) \
            bail(__LINE__, #condition);                                         \
    } while (false)

#define REQUIRE_READY(future)                                                     \
    do {                                                                          \
        if ((future).wait_for(std::chrono::seconds(5)) != std::future_status::ready) \
            bail(__LINE__, "worker future did not become ready");                 \
    } while (false)

// Last-resort watchdog for waits the gates above cannot see, such as a join
// whose grant was lost: abort with a diagnostic instead of hanging the run.
void arm_watchdog() {
    const auto deadline = std::make_shared<std::atomic<std::chrono::steady_clock::time_point>>(
        std::chrono::steady_clock::now() + std::chrono::seconds(45));
    std::thread([deadline] {
        while (true) {
            if (std::chrono::steady_clock::now() > deadline->load()) {
                std::fprintf(stderr, "FAIL: watchdog budget expired; aborting instead of hanging\n");
                std::fflush(stderr);
                std::abort();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }).detach();
}

// Guest words addressed by index. An address outside the mapped range is unmapped.
struct GuestMemory {
    static constexpr std::uint64_t base = 0x7030000000;
    static constexpr std::uint64_t slot = 4;
    mutable std::mutex mutex;
    std::array<std::uint32_t, 8> words{};
    // Indices below this are mapped; lowering it makes words inaccessible to reads,
    // writes and compare-exchanges alike.
    std::size_t mapped = words.size();
    // Armed before a signal to make the next compare-exchange observe a synthetic owner
    // claiming the word first, exactly as a concurrent guest compare-and-swap would.
    bool inject_owner_claim = false;
    std::uint32_t owner_claim_tag = 0;
    std::uint32_t valid_owner = 0;
    std::uint32_t checked_owner = 0;
    unsigned owner_checks = 0;

    static bool validate_owner(void* context, std::uint32_t owner) noexcept {
        auto& self = *static_cast<GuestMemory*>(context);
        ++self.owner_checks;
        self.checked_owner = owner;
        return owner != 0 && owner == self.valid_owner;
    }

    static bool in_range(const GuestMemory& self, std::uint64_t address, std::size_t& index) {
        if (address < base || (address - base) % slot != 0) return false;
        index = (address - base) / slot;
        return index < self.mapped;
    }
    static bool read(void* context, std::uint64_t address, std::uint32_t& value) noexcept {
        auto& self = *static_cast<GuestMemory*>(context);
        const std::lock_guard lock(self.mutex);
        std::size_t index = 0;
        if (!in_range(self, address, index)) return false;
        value = self.words[index];
        return true;
    }
    static bool write(void* context, std::uint64_t address, std::uint32_t value) noexcept {
        auto& self = *static_cast<GuestMemory*>(context);
        const std::lock_guard lock(self.mutex);
        std::size_t index = 0;
        if (!in_range(self, address, index)) return false;
        self.words[index] = value;
        return true;
    }
    static armigo::horizon::ProcessWideKey::ExchangeResult compare_exchange(
        void* context, std::uint64_t address, std::uint32_t& expected,
        std::uint32_t desired) noexcept {
        using ExchangeResult = armigo::horizon::ProcessWideKey::ExchangeResult;
        auto& self = *static_cast<GuestMemory*>(context);
        const std::lock_guard lock(self.mutex);
        std::size_t index = 0;
        if (!in_range(self, address, index)) return ExchangeResult::invalid_address;
        if (self.inject_owner_claim) {
            // The synthetic owner claims the free word before this exchange observes it.
            self.inject_owner_claim = false;
            self.words[index] = self.owner_claim_tag;
        }
        if (self.words[index] != expected) {
            expected = self.words[index];
            return ExchangeResult::mismatch;
        }
        self.words[index] = desired;
        // expected keeps the prior matching value, as the callback contract requires.
        return ExchangeResult::exchanged;
    }
    static std::uint64_t address(std::size_t index) { return base + index * slot; }
    std::uint32_t& operator[](std::size_t index) { return words[index]; }
};

struct Keys {
    GuestMemory memory;
    std::shared_ptr<armigo::horizon::Arbitration> arbitration =
        std::make_shared<armigo::horizon::Arbitration>();
    armigo::horizon::ProcessWideKey keys{arbitration};
    armigo::horizon::ProcessWideKey::GuestWords words() {
        return {GuestMemory::read, GuestMemory::write, &memory, GuestMemory::compare_exchange};
    }
};
}  // namespace

int main() {
    using armigo::horizon::ProcessWideKey;
    arm_watchdog();

    {
        Keys state;
        const auto mutex = GuestMemory::address(0), key = GuestMemory::address(1);
        state.memory[0] = 0x1111;
        // A zero timeout expires before blocking, releases the mutex and leaves the condition
        // word armed. The mutex is not taken back.
        CHECK(state.keys.wait(state.words(), mutex, key, 0x2222, 0) == ProcessWideKey::WaitOutcome::timed_out);
        CHECK(state.memory[0] == 0);
        CHECK(state.memory[1] == 1);
        CHECK(state.keys.waiters(key) == 0);
        // A positive timeout expires on its own.
        state.memory[0] = 0x1111;
        CHECK(state.keys.wait(state.words(), mutex, key, 0x2222, 1'000'000) ==
              ProcessWideKey::WaitOutcome::timed_out);
        CHECK(state.memory[0] == 0);
        // Unmapped words are refused instead of reported as a wait.
        CHECK(state.keys.wait(state.words(), GuestMemory::address(100), key, 0x2222, -1) ==
              ProcessWideKey::WaitOutcome::invalid_address);
        CHECK(!state.keys.unlock(state.words(), GuestMemory::address(100)));
    }

    {
        Keys state;
        const auto mutex = GuestMemory::address(0), key = GuestMemory::address(1);
        state.memory[0] = 0x1111;
        std::thread waiter([&] {
            CHECK(state.keys.wait(state.words(), mutex, key, 0x2222, -1) ==
                  ProcessWideKey::WaitOutcome::signaled);
        });
        REQUIRE_SETTLES(state.keys.waiters(key) == 1);
        // The condition word is armed while the thread waits, and the mutex is free.
        CHECK(state.memory[1] == 1);
        CHECK(state.memory[0] == 0);
        state.keys.signal(state.words(), key, 1);
        waiter.join();
        // The signalled thread owns the mutex through its own tag, and the condition word is
        // cleared once no waiter is left.
        CHECK(state.memory[0] == 0x2222);
        CHECK(state.memory[1] == 0);
        CHECK(state.keys.waiters(key) == 0);
        CHECK(state.keys.unlock(state.words(), mutex));
        CHECK(state.memory[0] == 0);
    }

    {
        Keys state;
        const auto mutex = GuestMemory::address(0), key = GuestMemory::address(1);
        state.memory[0] = 0x1111;
        auto waiter = std::async(std::launch::async, [&] {
            return state.keys.wait(state.words(), mutex, key, 0x3333, -1);
        });
        REQUIRE_SETTLES(state.keys.waiters(key) == 1);
        // Another thread took the mutex before the signal: the handoff is deferred, so the
        // woken guest cannot observe a mutex it does not own. The waiters-present bit is set
        // so the owner's user-mode unlock fast path fails and reaches the kernel handoff.
        state.memory[0] = 0x4444;
        state.keys.signal(state.words(), key, 1);
        // No early grant while the owner holds the mutex: the waiter stays blocked.
        CHECK(waiter.wait_for(std::chrono::milliseconds(50)) != std::future_status::ready);
        CHECK(state.keys.waiters(key) == 0);
        CHECK(state.memory[0] == 0x40004444u);
        CHECK(state.keys.unlock(state.words(), mutex));
        REQUIRE_READY(waiter);
        CHECK(waiter.get() == ProcessWideKey::WaitOutcome::signaled);
        CHECK(state.memory[0] == 0x3333);
    }

    {
        Keys state;
        const auto first_mutex = GuestMemory::address(0), second_mutex = GuestMemory::address(1);
        const auto key = GuestMemory::address(2);
        state.memory[0] = 0x1111;
        state.memory[1] = 0x1111;
        const auto body = [&](std::uint64_t mutex, std::uint32_t tag) {
            CHECK(state.keys.wait(state.words(), mutex, key, tag, -1) == ProcessWideKey::WaitOutcome::signaled);
        };
        std::thread first(body, first_mutex, 0x2222), second(body, second_mutex, 0x3333);
        REQUIRE_SETTLES(state.keys.waiters(key) == 2);
        // A count of one releases exactly one waiter.
        state.keys.signal(state.words(), key, 1);
        REQUIRE_SETTLES(state.keys.waiters(key) == 1);
        CHECK(state.memory[2] == 1);
        state.keys.signal(state.words(), key, -1);
        first.join();
        second.join();
        CHECK(state.memory[0] == 0x2222);
        CHECK(state.memory[1] == 0x3333);
        CHECK(state.memory[2] == 0);
    }

    {
        // Mutex handoffs and address waits are separate kernel queues even at the same word.
        Keys state;
        const auto mutex = GuestMemory::address(0);
        std::thread waiter([&] {
            CHECK(state.arbitration->wait(mutex, 1, 1, -1) ==
                  armigo::horizon::ArbitrateOutcome::succeeded);
        });
        REQUIRE_SETTLES(state.arbitration->waiters(mutex) == 1);
        CHECK(state.keys.unlock(state.words(), mutex));
        CHECK(state.arbitration->waiters(mutex) == 1);
        CHECK(state.arbitration->signal(mutex, 1) == 1);
        waiter.join();
        CHECK(state.memory[0] == 0);
    }

    {
        // A guest owner claims the mutex by its own compare-and-swap between the wait and
        // the signal's first handoff exchange. The handoff must observe that owner, never
        // overwrite it, and still reach the waiter once the new owner releases.
        Keys state;
        const auto mutex = GuestMemory::address(0), key = GuestMemory::address(1);
        state.memory[0] = 0x1111;
        auto waiter = std::async(std::launch::async, [&] {
            return state.keys.wait(state.words(), mutex, key, 0x2222, -1);
        });
        REQUIRE_SETTLES(state.keys.waiters(key) == 1);
        state.memory.inject_owner_claim = true;
        state.memory.owner_claim_tag = 0x4444;
        CHECK(state.keys.signal(state.words(), key, 1) ==
              ProcessWideKey::SignalOutcome::signaled);
        // The intervening owner still owns the word; the retry only added the
        // waiters-present marker instead of writing the waiter's tag over it.
        CHECK(state.memory[0] == (0x4444 | 0x40000000u));
        CHECK(state.keys.waiters(key) == 0);
        // No early grant: the waiter stays blocked while the new owner holds the mutex.
        CHECK(waiter.wait_for(std::chrono::seconds(2)) != std::future_status::ready);
        CHECK(state.keys.unlock(state.words(), mutex));
        REQUIRE_READY(waiter);
        CHECK(waiter.get() == ProcessWideKey::WaitOutcome::signaled);
        CHECK(state.memory[0] == 0x2222);
    }

    {
        // A signal without the atomic handoff reports the missing capability and changes
        // no queue, register or guest word; the same call works once the callback is there.
        Keys state;
        const auto mutex = GuestMemory::address(0), key = GuestMemory::address(1);
        state.memory[0] = 0x1111;
        auto waiter = std::async(std::launch::async, [&] {
            return state.keys.wait(state.words(), mutex, key, 0x2222, -1);
        });
        REQUIRE_SETTLES(state.keys.waiters(key) == 1);
        auto plain = state.words();
        plain.compare_exchange = nullptr;
        CHECK(state.keys.signal(plain, key, 1) ==
              ProcessWideKey::SignalOutcome::unsupported_atomic_access);
        CHECK(state.keys.waiters(key) == 1);
        CHECK(state.memory[1] == 1);
        CHECK(state.memory[0] == 0);
        CHECK(state.keys.signal(state.words(), key, 1) ==
              ProcessWideKey::SignalOutcome::signaled);
        REQUIRE_READY(waiter);
        CHECK(waiter.get() == ProcessWideKey::WaitOutcome::signaled);
    }

    for (const auto observed : {0u, 0x4444u, 0x40004444u, 0x7777u, 0x40000000u}) {
        // Validation follows the signal's atomic marker update and skips an immediately
        // acquired mutex. Invalid owners finish only the waiter, leaving no handoff queue.
        Keys state;
        const auto mutex = GuestMemory::address(0), key = GuestMemory::address(1);
        state.memory.valid_owner = 0x4444;
        state.memory[0] = 0x2222;
        std::stop_source cancellation;
        auto waiter = std::async(std::launch::async, [&] {
            return state.keys.wait(state.words(), mutex, key, 0x2222, -1,
                                   cancellation.get_token());
        });
        REQUIRE_SETTLES(state.keys.waiters(key) == 1);
        state.memory[0] = observed;
        auto checked = state.words();
        checked.validate_owner = GuestMemory::validate_owner;
        CHECK(state.keys.signal(checked, key, 1) == ProcessWideKey::SignalOutcome::signaled);
        CHECK(state.keys.waiters(key) == 0);
        CHECK(state.memory[1] == 0);
        const auto valid = (observed & ~0x40000000u) == state.memory.valid_owner;
        if (observed == 0) {
            CHECK(state.memory.owner_checks == 0);
            CHECK(state.memory[0] == 0x2222);
            CHECK(state.keys.mutex_waiters(mutex) == 0);
        } else {
            CHECK(state.memory.owner_checks == 1);
            CHECK(state.memory.checked_owner == (observed & ~0x40000000u));
            CHECK(state.memory[0] == (observed | 0x40000000u));
            CHECK(state.keys.mutex_waiters(mutex) == (valid ? 1u : 0u));
            if (valid) {
                CHECK(waiter.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
                CHECK(state.keys.unlock(state.words(), mutex));
                CHECK(state.memory[0] == 0x2222);
            }
        }
        // Once the signal or unlock completes the waiter, a later host stop cannot
        // replace that committed success or invalid-state result with cancellation.
        CHECK(cancellation.request_stop());
        REQUIRE_READY(waiter);
        CHECK(waiter.get() == (observed == 0 || valid ? ProcessWideKey::WaitOutcome::signaled
                                                     : ProcessWideKey::WaitOutcome::invalid_state));
        CHECK(state.keys.mutex_waiters(mutex) == 0);
    }

    {
        // A mutex word that becomes inaccessible between the wait and the signal fails
        // only that waiter's wait; the signal itself stays completed and no handoff is
        // parked on an address that can never be written.
        Keys state;
        const auto mutex = GuestMemory::address(0), key = GuestMemory::address(1);
        state.memory[0] = 0x1111;
        auto waiter = std::async(std::launch::async, [&] {
            return state.keys.wait(state.words(), mutex, key, 0x2222, -1);
        });
        REQUIRE_SETTLES(state.keys.waiters(key) == 1);
        state.memory.mapped = 0;
        CHECK(state.keys.signal(state.words(), key, 1) ==
              ProcessWideKey::SignalOutcome::signaled);
        REQUIRE_READY(waiter);
        CHECK(waiter.get() == ProcessWideKey::WaitOutcome::invalid_address);
        CHECK(state.keys.waiters(key) == 0);
    }

    if (failures == 0) std::fprintf(stderr, "Process-wide key failures: 0\n");
    return failures == 0 ? 0 : 1;
}
