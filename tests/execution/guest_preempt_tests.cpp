// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================


#include <armigo/execution/guest_preempt.h>

#include "../../src/execution/gate_exit.h"

#include <libkern/OSCacheControl.h>
#include <pthread.h>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

using namespace armigo::execution;
using namespace std::chrono_literals;

namespace {

int failures = 0;
#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::fprintf(stderr, "Check failed at line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                               \
        }                                                                             \
    } while (false)

// void spin(uint32_t* flag): waits for *flag != 0, no call, then returns. It
// lives in an executable page that no module and no dyld image owns.
struct SpinCode {
    void* page = nullptr;
    std::size_t size = 0;
    void (*spin)(std::atomic<std::uint32_t>*) = nullptr;
    SpinCode() {
        size = 16384;
        page = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (page == MAP_FAILED) { page = nullptr; return; }
        auto* words = static_cast<std::uint32_t*>(page);
        words[0] = 0x88dffc09;  // ldar w9, [x0]
        words[1] = 0x34ffffe9;  // cbz w9, #-4
        words[2] = 0xd65f03c0;  // ret
        if (mprotect(page, size, PROT_READ | PROT_EXEC) != 0) { page = nullptr; return; }
        sys_icache_invalidate(page, 12);
        spin = reinterpret_cast<void (*)(std::atomic<std::uint32_t>*)>(page);
    }
    ~SpinCode() { if (page) munmap(page, size); }
    GuestCodeRange range() const {
        return {reinterpret_cast<std::uint64_t>(page), reinterpret_cast<std::uint64_t>(page) + size};
    }
};

enum class Mode { spin_plain, spin_held, host_loop };

struct Bound {
    GuestPreemptSlot slot;
    std::atomic<int> ready{0};
    std::atomic<int> finished{0};
    std::atomic<std::uint32_t> flag{0};
    std::atomic<bool> restored{false};
    std::atomic<bool> held_null{false};
    int record = 0;
    SpinCode* code = nullptr;
    Mode mode = Mode::spin_plain;
    std::mutex host_mutex;
    std::atomic<std::uint32_t> stop_loop{0};
};

void body(Bound& bound) {
    const GuestPreemptBinding binding(bound.slot);
    // The stack range of an invocation: wide enough to hold this frame.
    const auto sp = reinterpret_cast<std::uint64_t>(__builtin_frame_address(0));
    const detail::GuestExitScope scope(0, &bound.record, nullptr, sp - (1u << 20), sp + (1u << 20));
    if (bound.mode == Mode::host_loop) {
        bound.ready.store(1);
        // Host code with host locks and allocation, running on the published
        // stack: what NVN entries and signal handlers do.
        while (bound.stop_loop.load() == 0) {
            const std::lock_guard lock(bound.host_mutex);
            const auto block = std::make_shared<int>(1);
            if (*block != 1) break;
        }
    } else {
        std::optional<GuestPreemptHold> hold;
        if (bound.mode == Mode::spin_held) {
            hold.emplace();
            bound.held_null.store(bound.slot.record.load() == nullptr);
        }
        bound.ready.store(1);
        bound.code->spin(&bound.flag);
        hold.reset();
        bound.restored.store(bound.slot.record.load() == &bound.record);
    }
    bound.finished.store(1);
}

void run_case(const char* name, Mode mode, bool list_page_as_guest) {
    SpinCode code;
    CHECK(code.page != nullptr);
    if (!code.page) return;
    Bound bound;
    bound.code = &code;
    bound.mode = mode;
    std::thread worker([&] { body(bound); });
    while (bound.ready.load() == 0) std::this_thread::sleep_for(1ms);
    std::this_thread::sleep_for(20ms);  // let it settle into the loop
    const GuestCodeRange guest[] = {code.range()};
    const std::span<const GuestCodeRange> listed = list_page_as_guest ? std::span<const GuestCodeRange>(guest)
                                                                       : std::span<const GuestCodeRange>();
    unsigned rejected = 0, other = 0;
    for (unsigned attempt = 0; attempt != 400; ++attempt) {
        const auto result = preempt_guest_thread(bound.slot, listed);
        if (result == GuestPreemptResult::not_in_guest) ++rejected;
        else ++other;
        if (mode == Mode::host_loop) std::this_thread::sleep_for(100us);
    }
    // A redirect would already have killed the process (the record is not a
    // real invocation); a thread left alone keeps running to its own end.
    CHECK(other == 0);
    CHECK(rejected == 400);
    CHECK(bound.finished.load() == 0);
    bound.flag.store(1);
    bound.stop_loop.store(1);
    worker.join();
    CHECK(bound.finished.load() == 1);
    if (mode == Mode::spin_held) {
        CHECK(bound.held_null.load());
        CHECK(bound.restored.load());
    }
    if (mode == Mode::host_loop) CHECK(bound.host_mutex.try_lock());
    std::printf("%s: %u attempts left the thread alone.\n", name, rejected);
}

void test_self_is_refused() {
    GuestPreemptSlot slot;
    const GuestPreemptBinding binding(slot);
    CHECK(preempt_guest_thread(slot, {}) == GuestPreemptResult::unavailable);
}

} // namespace

int main() {
    // The thread runs in an executable page that is not in the guest ranges.
    run_case("Non-guest executable page", Mode::spin_plain, false);
    // Host code with locks and allocation on the published stack.
    run_case("Host code on the guest stack", Mode::host_loop, false);
    // Guest code, but under a host callback frame: nothing is published.
    run_case("Guest code under a host callback", Mode::spin_held, true);
    test_self_is_refused();
    std::printf("Guest preemption failures: %d\n", failures);
    return failures ? 1 : 0;
}
