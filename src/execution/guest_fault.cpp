// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/execution/guest_fault.h>

#include <signal.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include <mach/arm/thread_status.h>

#include <atomic>
#include <cstring>
#include <mutex>

extern "C" void armigo_execution_guest_entry_exit(
    armigo::execution::GuestEntryRecord*, armigo::execution::GateFrame*) noexcept;

namespace armigo::execution {
namespace {

constexpr int kSignals[] = {SIGSEGV, SIGBUS, SIGILL, SIGTRAP};
constexpr std::size_t kSignalCount = sizeof kSignals / sizeof kSignals[0];
struct sigaction g_previous[kSignalCount];
std::atomic<bool> g_installed{false};
std::mutex g_install_mutex;

// Guest code ranges of every live registration. An entry is free while its
// end is zero. Each change runs under a sequence count that is odd while the
// range changes; the handler skips an entry it sees changing, so it never
// reads a range torn by another runtime's registration.
constexpr std::size_t kRangeCapacity = 512;
struct RangeEntry {
    std::atomic<std::uint32_t> sequence{0};
    std::atomic<std::uint64_t> begin{0};
    std::atomic<std::uint64_t> end{0};
};
RangeEntry g_ranges[kRangeCapacity];
std::mutex g_ranges_mutex;

std::size_t signal_index(int signal) noexcept {
    for (std::size_t index = 0; index != kSignalCount; ++index)
        if (kSignals[index] == signal) return index;
    return 0;
}

bool in_guest_code(std::uint64_t pc) noexcept {
    for (const auto& entry : g_ranges) {
        const auto sequence = entry.sequence.load(std::memory_order_acquire);
        if (sequence & 1) continue;
        const auto begin = entry.begin.load(std::memory_order_relaxed);
        const auto end = entry.end.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (entry.sequence.load(std::memory_order_relaxed) != sequence) continue;
        if (pc >= begin && pc < end) return true;
    }
    return false;
}

// Called with g_ranges_mutex held.
void store_range(RangeEntry& entry, std::uint64_t begin, std::uint64_t end) noexcept {
    const auto sequence = entry.sequence.load(std::memory_order_relaxed);
    entry.sequence.store(sequence + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    entry.begin.store(begin, std::memory_order_relaxed);
    entry.end.store(end, std::memory_order_relaxed);
    entry.sequence.store(sequence + 2, std::memory_order_release);
}

// Async-signal-safe line writer: text and hexadecimal only.
struct Line {
    char text[192];
    std::size_t length = 0;
    void add(const char* value) noexcept {
        while (*value && length + 1 < sizeof text) text[length++] = *value++;
    }
    void hex(std::uint64_t value) noexcept {
        add("0x");
        char digits[16];
        std::size_t count = 0;
        do {
            digits[count++] = "0123456789abcdef"[value & 15];
            value >>= 4;
        } while (value != 0);
        while (count != 0 && length + 1 < sizeof text) text[length++] = digits[--count];
    }
};

void describe_fatal(int signal, const siginfo_t* info, const ucontext_t* context) noexcept {
    Line line;
    line.add("armigo: fatal ");
    line.add(describe_guest_fault_signal(signal));
    if (context && context->uc_mcontext) {
        const std::uint64_t pc = arm_thread_state64_get_pc(context->uc_mcontext->__ss);
        line.add(" at pc ");
        line.hex(pc);
        line.add(in_guest_code(pc) ? " (guest code, not in a stoppable guest invocation)" : " (host code)");
        line.add(", syndrome ");
        line.hex(context->uc_mcontext->__es.__esr);
    }
    if (info) {
        line.add(", address ");
        line.hex(reinterpret_cast<std::uint64_t>(info->si_addr));
    }
    line.add("\n");
    (void)::write(STDERR_FILENO, line.text, line.length);
}

void chain_previous(int signal, siginfo_t* info, void* context) noexcept {
    const struct sigaction& previous = g_previous[signal_index(signal)];
    // The disposition comes first: a default action inherited across exec
    // still reports SA_SIGINFO.
    if (previous.sa_handler == SIG_IGN) return;
    if (previous.sa_handler != SIG_DFL) {
        if (previous.sa_flags & SA_SIGINFO) previous.sa_sigaction(signal, info, context);
        else previous.sa_handler(signal);
        return;
    }
    describe_fatal(signal, info, static_cast<const ucontext_t*>(context));
    // The default disposition takes over: the raised signal stays blocked
    // until this handler returns and then ends the process.
    struct sigaction reset{};
    reset.sa_handler = SIG_DFL;
    sigemptyset(&reset.sa_mask);
    sigaction(signal, &reset, nullptr);
    raise(signal);
}

// Ends the innermost guest invocation of the faulting thread through the
// entry bridge's exit path, as preempt_guest_thread does from outside.
bool end_guest_invocation(int signal, const siginfo_t* info, ucontext_t* context) noexcept {
    if (!context || !context->uc_mcontext) return false;
    // A signal sent by a process (kill, raise) is no fault of the guest.
    if (info && info->si_code >= SI_USER) return false;
    auto& state = context->uc_mcontext->__ss;
    const std::uint64_t pc = arm_thread_state64_get_pc(state);
    // The ranges come first: a thread outside guest code may never have
    // touched its thread-local storage, whose first use allocates.
    if ((pc & 3) != 0 || !in_guest_code(pc)) return false;
    GuestPreemptSlot* const slot = detail::bound_slot();
    if (slot == nullptr) return false;
    void* const record = slot->record.load(std::memory_order_acquire);
    const std::uint64_t low = slot->stack_low.load(std::memory_order_acquire);
    const std::uint64_t high = slot->stack_high.load(std::memory_order_acquire);
    const std::uint64_t sp = arm_thread_state64_get_sp(state);
    if (record == nullptr || sp < low || sp > high) return false;

    auto& fault = slot->fault;
    fault.signal = signal;
    fault.code = info ? info->si_code : 0;
    fault.address = info ? reinterpret_cast<std::uint64_t>(info->si_addr) : 0;
    fault.syndrome = context->uc_mcontext->__es.__esr;
    fault.pc = pc;
    fault.sp = sp;
    fault.pstate = state.__cpsr;
    for (std::size_t index = 0; index != 29; ++index) fault.x[index] = state.__x[index];
    fault.x[29] = arm_thread_state64_get_fp(state);
    fault.x[30] = arm_thread_state64_get_lr(state);
    const auto& neon = context->uc_mcontext->__ns;
    fault.fpcr = neon.__fpcr;
    fault.fpsr = neon.__fpsr;

    // The exit routine reads only FPCR and FPSR from its gate frame.
    const std::uint64_t fpsr = neon.__fpsr, fpcr = neon.__fpcr;
    std::memcpy(slot->frame + ARMIGO_GATE_FRAME_FPSR_OFFSET, &fpsr, sizeof fpsr);
    std::memcpy(slot->frame + ARMIGO_GATE_FRAME_FPCR_OFFSET, &fpcr, sizeof fpcr);
    slot->fault_signal.store(signal, std::memory_order_release);
    state.__x[0] = reinterpret_cast<std::uint64_t>(record);
    state.__x[1] = reinterpret_cast<std::uint64_t>(slot->frame);
    arm_thread_state64_set_pc_fptr(state, reinterpret_cast<void*>(&armigo_execution_guest_entry_exit));
    return true;
}

void fault_handler(int signal, siginfo_t* info, void* context) noexcept {
    if (end_guest_invocation(signal, info, static_cast<ucontext_t*>(context))) return;
    chain_previous(signal, info, context);
}

}  // namespace

void install_guest_fault_handler() noexcept {
    const std::lock_guard lock(g_install_mutex);
    if (g_installed.load(std::memory_order_acquire)) return;
    struct sigaction action{};
    action.sa_sigaction = fault_handler;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    for (std::size_t index = 0; index != kSignalCount; ++index)
        sigaction(kSignals[index], &action, &g_previous[index]);
    g_installed.store(true, std::memory_order_release);
}

bool guest_fault_handler_installed() noexcept { return g_installed.load(std::memory_order_acquire); }

GuestCodeRegistration::GuestCodeRegistration(std::span<const GuestCodeRange> ranges) {
    const std::lock_guard lock(g_ranges_mutex);
    std::vector<std::uint16_t> free;
    for (std::size_t index = 0; index != kRangeCapacity && free.size() != ranges.size(); ++index)
        if (g_ranges[index].end.load(std::memory_order_relaxed) == 0)
            free.push_back(static_cast<std::uint16_t>(index));
    if (free.size() != ranges.size()) return;
    for (std::size_t index = 0; index != ranges.size(); ++index) {
        if (ranges[index].begin >= ranges[index].end) continue;
        store_range(g_ranges[free[index]], ranges[index].begin, ranges[index].end);
        entries_.push_back(free[index]);
    }
    registered_ = true;
}

GuestCodeRegistration::~GuestCodeRegistration() {
    const std::lock_guard lock(g_ranges_mutex);
    for (const auto index : entries_) store_range(g_ranges[index], 0, 0);
}

std::optional<GuestFault> take_guest_fault(GuestPreemptSlot& slot) noexcept {
    if (slot.fault_signal.load(std::memory_order_acquire) == 0) return std::nullopt;
    const GuestFault fault = slot.fault;
    slot.fault_signal.store(0, std::memory_order_relaxed);
    return fault;
}

const char* describe_guest_fault_signal(int signal) noexcept {
    switch (signal) {
    case SIGSEGV: return "SIGSEGV";
    case SIGBUS: return "SIGBUS";
    case SIGILL: return "SIGILL";
    case SIGTRAP: return "SIGTRAP";
    }
    return "signal";
}

}  // namespace armigo::execution
