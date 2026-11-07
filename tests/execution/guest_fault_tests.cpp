// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================


#include <armigo/execution/guest_entry.h>
#include <armigo/execution/guest_fault.h>
#include <armigo/memory/write_tracker.h>
#include <armigo/platform/guest_stack.h>

#include "../../src/execution/gate_exit.h"

#include <libkern/OSCacheControl.h>
#include <mach-o/dyld.h>
#include <setjmp.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern char** environ;

using namespace armigo::execution;
using armigo::platform::GuardedStack;

namespace {

int failures = 0;
#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::fprintf(stderr, "Check failed at line %d: %s\n", __LINE__, #condition); \
            ++failures;                                                               \
        }                                                                             \
    } while (false)

// Synthetic guest functions in an executable page that no module owns.
constexpr std::size_t kLoad = 0;       // x5 = 0x1234; x0 = *(u64*)x0; ret
constexpr std::size_t kBreak = 16;     // brk #7; ret
constexpr std::size_t kUndefined = 32; // udf #0; ret
constexpr std::size_t kIncrement = 48; // x0 += 1; ret

struct Code {
    void* page = nullptr;
    std::size_t size = 16384;
    Code() {
        page = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (page == MAP_FAILED) { page = nullptr; return; }
        auto* words = static_cast<std::uint32_t*>(page);
        words[kLoad / 4 + 0] = 0xd2824685;      // movz x5, #0x1234
        words[kLoad / 4 + 1] = 0xf9400000;      // ldr x0, [x0]
        words[kLoad / 4 + 2] = 0xd65f03c0;      // ret
        words[kBreak / 4 + 0] = 0xd42000e0;     // brk #7
        words[kBreak / 4 + 1] = 0xd65f03c0;     // ret
        words[kUndefined / 4 + 0] = 0x00000000; // udf #0
        words[kUndefined / 4 + 1] = 0xd65f03c0; // ret
        words[kIncrement / 4 + 0] = 0x91000400; // add x0, x0, #1
        words[kIncrement / 4 + 1] = 0xd65f03c0; // ret
        if (mprotect(page, size, PROT_READ | PROT_EXEC) != 0) { page = nullptr; return; }
        sys_icache_invalidate(page, 64);
    }
    ~Code() { if (page) munmap(page, size); }
    std::uint64_t at(std::size_t offset) const { return reinterpret_cast<std::uint64_t>(page) + offset; }
    GuestCodeRange range() const { return {at(0), at(size)}; }
    std::uint64_t call(std::size_t offset, std::uint64_t x0) const {
        return reinterpret_cast<std::uint64_t (*)(std::uint64_t)>(at(offset))(x0);
    }
};

// Stands in for the handler a process had before; a chained fault lands here.
sigjmp_buf chained_jump;
volatile sig_atomic_t chained_signal = 0;
void previous_handler(int signal, siginfo_t*, void*) {
    chained_signal = signal;
    siglongjmp(chained_jump, 1);
}

std::unique_ptr<GuardedStack> make_stack() {
    auto result = GuardedStack::create({.usable_size = std::size_t{256} << 10});
    if (!result.ok()) {
        std::printf("FAIL stack creation\n");
        std::exit(1);
    }
    return std::move(result.stack);
}

GuestEntryResult enter(GuardedStack& stack, std::uint64_t function, std::uint64_t x0) {
    GuestEntryRequest request;
    request.function = function;
    request.x0 = x0;
    request.require_gate_scope = false;
    return invoke_guest_entry(stack, request);
}

void test_guest_faults_end_the_invocation(const Code& code) {
    auto stack = make_stack();
    GuestPreemptSlot slot;
    const GuestPreemptBinding binding(slot);

    const auto loaded = enter(*stack, code.at(kLoad), 0x20);
    CHECK(loaded.exited());
    const auto data = take_guest_fault(slot);
    CHECK(data.has_value());
    if (data) {
        CHECK(data->signal == SIGSEGV);
        CHECK(data->address == 0x20);
        CHECK(data->pc == code.at(kLoad + 4));
        CHECK(data->x[0] == 0x20);
        CHECK(data->x[5] == 0x1234);
        CHECK(data->sp > stack->usable_base() && data->sp <= stack->aligned_top());
        // ESR: data abort from a lower exception level, a read.
        CHECK((data->syndrome >> 26) == 0x24);
        CHECK((data->syndrome & (1u << 6)) == 0);
        std::printf("Data abort: pc offset 0x%llx, address 0x%llx, syndrome 0x%x.\n",
                    static_cast<unsigned long long>(data->pc - code.at(0)),
                    static_cast<unsigned long long>(data->address), data->syndrome);
    }
    CHECK(!take_guest_fault(slot).has_value());

    // The thread, its stack and the handler keep working after a fault.
    const auto incremented = enter(*stack, code.at(kIncrement), 41);
    CHECK(incremented.returned());
    CHECK(incremented.x0 == 42);
    CHECK(!take_guest_fault(slot).has_value());

    const auto broke = enter(*stack, code.at(kBreak), 0);
    CHECK(broke.exited());
    const auto trap = take_guest_fault(slot);
    CHECK(trap && trap->signal == SIGTRAP && trap->pc == code.at(kBreak));
    if (trap) std::printf("Breakpoint: %s, syndrome 0x%x.\n", describe_guest_fault_signal(trap->signal), trap->syndrome);

    const auto undefined = enter(*stack, code.at(kUndefined), 0);
    CHECK(undefined.exited());
    const auto illegal = take_guest_fault(slot);
    CHECK(illegal && illegal->signal == SIGILL && illegal->pc == code.at(kUndefined));
    if (illegal) std::printf("Undefined instruction: %s.\n", describe_guest_fault_signal(illegal->signal));
}

// Each case must reach the previous handler without touching the slot.
template <class Body>
void expect_chained(const char* name, GuestPreemptSlot* slot, Body body) {
    chained_signal = 0;
    if (sigsetjmp(chained_jump, 1) == 0) {
        body();
        std::fprintf(stderr, "%s: the fault did not happen\n", name);
        ++failures;
        return;
    }
    CHECK(chained_signal == SIGSEGV);
    if (slot) CHECK(!take_guest_fault(*slot).has_value());
    std::printf("%s: chained to the previous handler.\n", name);
}

void test_other_faults_are_chained(const Code& guest, const Code& host) {
    // Guest code on a thread with no slot.
    expect_chained("Unbound thread", nullptr, [&] { guest.call(kLoad, 0x20); });

    GuestPreemptSlot slot;
    const GuestPreemptBinding binding(slot);
    // Bound, but no invocation is published.
    expect_chained("No invocation", &slot, [&] { guest.call(kLoad, 0x20); });

    int record = 0;
    const auto sp = reinterpret_cast<std::uint64_t>(__builtin_frame_address(0));
    {
        // The published stack does not hold the faulting stack pointer.
        const detail::GuestExitScope scope(0, &record, nullptr, sp + (1u << 20), sp + (2u << 20));
        expect_chained("Other stack", &slot, [&] { guest.call(kLoad, 0x20); });
    }
    {
        // The published stack holds it, but the code is not guest code.
        const detail::GuestExitScope scope(0, &record, nullptr, sp - (1u << 20), sp + (1u << 20));
        expect_chained("Host code", &slot, [&] { host.call(kLoad, 0x20); });
        // Guest code under a host callback publishes no invocation.
        const GuestPreemptHold hold;
        expect_chained("Held invocation", &slot, [&] { guest.call(kLoad, 0x20); });
    }
}

// A child with the default dispositions faults once and must die of the
// signal after one line on stderr.
struct ChildOutcome {
    int signal = 0;
    std::string error;
};

ChildOutcome spawn_child(const char* mode) {
    ChildOutcome outcome;
    char path[4096];
    std::uint32_t size = sizeof path;
    int pipe_ends[2];
    if (_NSGetExecutablePath(path, &size) != 0 || pipe(pipe_ends) != 0) return outcome;
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipe_ends[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_ends[0]);
    char flag[] = "--child";
    std::string mode_text = mode;
    char* argv[] = {path, flag, mode_text.data(), nullptr};
    pid_t pid = 0;
    const int error = posix_spawn(&pid, path, &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipe_ends[1]);
    if (error == 0) {
        char buffer[512];
        for (ssize_t count; (count = read(pipe_ends[0], buffer, sizeof buffer)) > 0;)
            outcome.error.append(buffer, static_cast<std::size_t>(count));
        int status = 0;
        if (waitpid(pid, &status, 0) == pid && WIFSIGNALED(status)) outcome.signal = WTERMSIG(status);
    }
    close(pipe_ends[0]);
    return outcome;
}

[[noreturn]] void run_child(const char* mode) {
    // A fault that repeats instead of ending the child fails the parent's check.
    alarm(20);
    // The child inherits the default actions from exec, which still report
    // SA_SIGINFO. "tracker" installs the write tracker above this handler, as
    // the runtime does; "tracker-only" leaves the tracker alone in front of
    // the default action.
    if (std::strcmp(mode, "tracker-only") != 0) install_guest_fault_handler();
    if (std::strncmp(mode, "tracker", 7) == 0) armigo::memory::WriteTracker::install();
    Code code;
    const GuestCodeRange ranges[] = {code.range()};
    std::optional<GuestCodeRegistration> registration;
    if (std::strcmp(mode, "guest") == 0) registration.emplace(ranges);
    code.call(kLoad, 0x20);
    std::_Exit(0);
}

void test_default_action() {
    const auto host = spawn_child("host");
    CHECK(host.signal == SIGSEGV);
    CHECK(host.error.rfind("armigo: fatal SIGSEGV at pc 0x", 0) == 0);
    CHECK(host.error.find(" (host code), syndrome 0x") != std::string::npos);
    CHECK(host.error.find(", address 0x20\n") != std::string::npos);
    const auto guest = spawn_child("guest");
    CHECK(guest.signal == SIGSEGV);
    CHECK(guest.error.find("(guest code, not in a stoppable guest invocation)") != std::string::npos);
    const auto tracked = spawn_child("tracker");
    CHECK(tracked.signal == SIGSEGV);
    CHECK(tracked.error.find(" (host code), syndrome 0x") != std::string::npos);
    const auto tracker_only = spawn_child("tracker-only");
    CHECK(tracker_only.signal == SIGSEGV);
    CHECK(tracker_only.error.find("armigo: fatal") == std::string::npos);
    std::printf("Default action: %s", host.error.c_str());
}

void test_registration_capacity() {
    std::vector<GuestCodeRange> many(4096, GuestCodeRange{0x1000, 0x2000});
    const GuestCodeRegistration too_many(many);
    CHECK(!too_many.registered());
    const GuestCodeRange one[] = {{0x1000, 0x2000}};
    const GuestCodeRegistration fits(one);
    CHECK(fits.registered());
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) run_child(argv[2]);
    struct sigaction previous{};
    previous.sa_sigaction = previous_handler;
    previous.sa_flags = SA_SIGINFO;
    sigemptyset(&previous.sa_mask);
    for (const int signal : {SIGSEGV, SIGBUS, SIGILL, SIGTRAP}) sigaction(signal, &previous, nullptr);
    install_guest_fault_handler();
    install_guest_fault_handler();
    CHECK(guest_fault_handler_installed());

    Code guest, host;
    CHECK(guest.page != nullptr && host.page != nullptr);
    if (!guest.page || !host.page) return 1;
    const GuestCodeRange ranges[] = {guest.range()};
    const GuestCodeRegistration registration(ranges);
    CHECK(registration.registered());

    test_guest_faults_end_the_invocation(guest);
    test_other_faults_are_chained(guest, host);
    test_registration_capacity();
    test_default_action();
    std::printf("Guest fault failures: %d\n", failures);
    return failures ? 1 : 0;
}
