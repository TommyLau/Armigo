// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/gate.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/platform/guest_stack.h>

#include <mach-o/dyld.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C" {
void armigo_test_guest_report();
void armigo_test_guest_clobber();
void armigo_test_guest_gate();
void armigo_test_guest_call_host();
void armigo_test_guest_bad_sp();
void armigo_test_guest_corrupt_record();
void armigo_test_guest_descend();
void armigo_test_guest_process_landing();
void armigo_test_guest_process_entry();
void armigo_test_bridge_harness(armigo::execution::GuestEntryRecord* record, void* report);
}

extern char** environ;

namespace {
using namespace armigo::execution;
using namespace armigo::platform;

int g_failed = 0;
int g_total = 0;

#define CHECK(condition)                                                                \
    do {                                                                                \
        ++g_total;                                                                      \
        if (!(condition)) {                                                             \
            ++g_failed;                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);            \
        }                                                                               \
    } while (false)

template <typename Function>
std::uint64_t code(Function* function) {
    return reinterpret_cast<std::uint64_t>(reinterpret_cast<void*>(function));
}

std::uint64_t address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

std::uint64_t host_fpcr() {
    std::uint64_t value;
    asm volatile("mrs %0, fpcr" : "=r"(value));
    return value;
}

std::uint64_t host_fpsr() {
    std::uint64_t value;
    asm volatile("mrs %0, fpsr" : "=r"(value));
    return value;
}

void set_host_fpsr(std::uint64_t value) {
    asm volatile("msr fpsr, %0" : : "r"(value));
}

std::unique_ptr<GuardedStack> make_stack(std::size_t usable = std::size_t{256} << 10) {
    auto result = GuardedStack::create({.usable_size = usable});
    if (!result.ok()) {
        std::printf("FAIL stack creation: %s\n", describe(result.status.error));
        std::exit(1);
    }
    return std::move(result.stack);
}

GateStatus passive_handler(GateCall&, void*) { return GateStatus::failed; }

GateStatus exit_handler(GateCall& call, void*) {
    return call.can_exit_thread() ? GateStatus::exit_thread : GateStatus::failed;
}

// Report block written by armigo_test_guest_report.
struct EntryReport {
    std::uint64_t sp, x29, x30, fpcr, fpsr, x1;
    std::uint64_t x2_to_x17[16];
    std::uint64_t x19_to_x28[10];
};
static_assert(sizeof(EntryReport) == 256);

void test_entry_state() {
    auto stack = make_stack();
    GateScope scope(&passive_handler);
    EntryReport report{};
    std::memset(&report, 0xcc, sizeof report);
    const std::uint64_t fpcr_before = host_fpcr();
    // Sticky host flags (IXC | IOC) must neither leak into nor be lost to the guest.
    set_host_fpsr(0x11);
    const GuestEntryResult result = invoke_guest_entry(
        *stack, {.function = code(&armigo_test_guest_report), .x0 = address(&report),
                 .x1 = 0x0123456789abcdef});
    const std::uint64_t fpsr_after = host_fpsr();
    set_host_fpsr(0);
    CHECK(fpsr_after == 0x11);
    CHECK(result.ok());
    const std::uint64_t expected_sp = stack->aligned_top() - ARMIGO_GUEST_ENTRY_STACK_RESERVE;
    CHECK(report.sp == expected_sp);
    CHECK(report.sp % 16 == 0);
    CHECK(stack->contains(report.sp, ARMIGO_GUEST_ENTRY_STACK_RESERVE));
    CHECK(report.x1 == 0x0123456789abcdef);
    CHECK(result.x0 == expected_sp);
    CHECK(result.x1 == ~std::uint64_t{0x0123456789abcdef});
    CHECK(report.x29 == stack->aligned_top() - 16);
    CHECK(report.x30 != 0);
    CHECK(report.fpcr == fpcr_before);
    CHECK(report.fpsr == 0);
    bool zero = true;
    for (unsigned index = 0; index < 16; ++index) {
        const unsigned reg = index + 2;
        const std::uint64_t value = report.x2_to_x17[index];
        if (reg == 16) {
            CHECK(value == code(&armigo_test_guest_report));
        } else {
            zero &= value == 0;
        }
    }
    for (const std::uint64_t value : report.x19_to_x28) zero &= value == 0;
    CHECK(zero);
    // The bridge cleared both records after the verified return.
    const auto* reserve = reinterpret_cast<const std::uint64_t*>(expected_sp);
    CHECK(reserve[0] == 0 && reserve[1] == 0 && reserve[2] == 0 && reserve[3] == 0);
    CHECK(!stack->in_use());
    CHECK(host_fpcr() == fpcr_before);
}

struct FrameObservation {
    std::uint64_t bridge_record[2];
    std::uint64_t guest_return;
};
FrameObservation* g_frames = nullptr;

std::uint64_t observe_frames(std::uint64_t) {
    // Frame chain: this function -> armigo_test_guest_call_host -> the
    // bridge's guest-side record {host x29, bridge return}.
    const auto* own = static_cast<const std::uint64_t*>(__builtin_frame_address(0));
    const auto* call_host = reinterpret_cast<const std::uint64_t*>(own[0]);
    const auto* bridge = reinterpret_cast<const std::uint64_t*>(call_host[0]);
    g_frames->bridge_record[0] = bridge[0];
    g_frames->bridge_record[1] = bridge[1];
    g_frames->guest_return = call_host[1];
    return 0;
}

void test_frame_record_during_call() {
    // While the guest runs, x29 at entry names a frame record whose saved lr
    // is the bridge return point (the x30 the guest received) and whose saved
    // x29 is the bridge's host frame.
    auto stack = make_stack();
    FrameObservation observed{};
    g_frames = &observed;
    const GuestEntryResult result = invoke_guest_entry(
        *stack, {.function = code(&armigo_test_guest_call_host),
                 .x0 = reinterpret_cast<std::uint64_t>(&observe_frames),
                 .require_gate_scope = false});
    g_frames = nullptr;
    CHECK(result.ok());
    CHECK(observed.bridge_record[1] != 0 && observed.bridge_record[1] == observed.guest_return);
    CHECK(observed.bridge_record[0] != 0 && !stack->contains(observed.bridge_record[0]));
    CHECK(observed.bridge_record[0] % 16 == 0);
}

void test_custom_top_and_fp_state() {
    auto stack = make_stack();
    GateScope scope(&passive_handler);
    EntryReport report{};
    const std::uint64_t fpcr_before = host_fpcr();
    const std::uint64_t fpsr_before = host_fpsr();
    const std::uint64_t guest_fpcr = (std::uint64_t{1} << 25) | (std::uint64_t{2} << 22);
    const std::uint64_t guest_fpsr = 0x08000011;
    const std::uint64_t top = stack->aligned_top() - 4096;
    const GuestEntryResult result = invoke_guest_entry(
        *stack, {.function = code(&armigo_test_guest_report), .x0 = address(&report),
                 .stack_top = top, .fpcr = guest_fpcr, .fpsr = guest_fpsr});
    CHECK(result.ok());
    CHECK(report.sp == top - ARMIGO_GUEST_ENTRY_STACK_RESERVE);
    CHECK(report.fpcr == guest_fpcr);
    CHECK(report.fpsr == guest_fpsr);
    CHECK(result.fpcr == guest_fpcr);
    CHECK(result.fpsr == guest_fpsr);
    CHECK(host_fpcr() == fpcr_before);
    CHECK(host_fpsr() == fpsr_before);
}

struct HarnessReport {
    std::uint64_t x19_to_x28[10];
    std::uint64_t d8_to_d15[8];
    std::uint64_t fpcr, fpsr;
    std::uint64_t sp_before, sp_after;
    std::uint64_t x18_before, x18_after;
    std::uint64_t x29_before, x29_after;
};
static_assert(sizeof(HarnessReport) == 208);

void test_bridge_preserves_host_state() {
    auto stack = make_stack();
    const GuardedStackUse use = stack->try_acquire();
    CHECK(use.acquired());
    GuestEntryRecord record;
    record.function = code(&armigo_test_guest_clobber);
    record.stack_top = stack->aligned_top();
    record.x0 = 40;
    record.x1 = 2;
    record.fpcr = 0;
    HarnessReport report{};
    armigo_test_bridge_harness(&record, &report);

    const std::uint64_t gprs[10] = {0x1919, 0x2020, 0x2121, 0x2222, 0x2323,
                                    0x2424, 0x2525, 0x2626, 0x2727, 0x2828};
    const std::uint64_t simds[8] = {0x0808, 0x0909, 0x1010, 0x1111, 0x1212, 0x1313, 0x1414, 0x1515};
    for (unsigned index = 0; index < 10; ++index) CHECK(report.x19_to_x28[index] == gprs[index]);
    for (unsigned index = 0; index < 8; ++index) CHECK(report.d8_to_d15[index] == simds[index]);
    CHECK(report.fpcr == 0x01400000);
    CHECK(report.fpsr == 0x14);
    CHECK(report.sp_after == report.sp_before);
    CHECK(report.x18_after == report.x18_before);
    CHECK(report.x29_after == report.x29_before);
    CHECK(record.result_x0 == 42);
    CHECK(record.result_x1 == (40u ^ 2u));
    CHECK(record.result_fpcr == 0x02c00000);
    CHECK((record.result_fpsr & 0x08000003) == 0x08000003);
}

void test_validation() {
    auto stack = make_stack();
    const std::uint64_t report_function = code(&armigo_test_guest_report);
    EntryReport report{};
    const auto run = [&](GuestEntryRequest request) {
        if (request.x0 == 0) request.x0 = address(&report);
        return invoke_guest_entry(*stack, request).error;
    };
    {
        GateScope scope(&passive_handler);
        CHECK(run({.function = 0}) == GuestEntryError::null_function);
        CHECK(run({.function = report_function + 2}) == GuestEntryError::misaligned_function);
        static std::uint64_t data_word = 0;
        CHECK(run({.function = address(&data_word)}) == GuestEntryError::function_not_executable);
        CHECK(run({.function = stack->usable_base()}) == GuestEntryError::function_not_executable);
        CHECK(run({.function = stack->layout().lower_guard_base}) ==
              GuestEntryError::function_not_executable);
        CHECK(run({.function = 0x10}) == GuestEntryError::function_not_executable);
        CHECK(run({.function = report_function, .stack_top = stack->aligned_top() - 8}) ==
              GuestEntryError::misaligned_stack_top);
        CHECK(run({.function = report_function, .stack_top = stack->aligned_top() + 16}) ==
              GuestEntryError::stack_top_out_of_range);
        CHECK(run({.function = report_function, .stack_top = stack->usable_base() + 16}) ==
              GuestEntryError::stack_top_out_of_range);
        CHECK(run({.function = report_function, .stack_top = stack->usable_base() + 1024}) ==
              GuestEntryError::insufficient_stack);
        CHECK(run({.function = report_function, .minimum_stack_bytes = stack->usable_size()}) ==
              GuestEntryError::insufficient_stack);
        CHECK(run({.function = report_function, .fpcr = std::uint64_t{1} << 8}) ==
              GuestEntryError::invalid_fpcr);
        CHECK(run({.function = report_function, .fpcr = std::uint64_t{1} << 27}) ==
              GuestEntryError::invalid_fpcr);
        {
            const GuardedStackUse held = stack->try_acquire();
            CHECK(run({.function = report_function}) == GuestEntryError::stack_busy);
        }
        CHECK(run({.function = report_function}) == GuestEntryError::none);
        // A stack whose top is exactly the reserve above the base is usable
        // when the caller explicitly waives the minimum.
        CHECK(run({.function = report_function,
                   .stack_top = stack->usable_base() + ARMIGO_GUEST_ENTRY_STACK_RESERVE + 16,
                   .minimum_stack_bytes = 16}) == GuestEntryError::none);
    }
    CHECK(!GateScope::thread_has_scope());
    CHECK(run({.function = report_function}) == GuestEntryError::no_gate_scope);
    CHECK(run({.function = report_function, .require_gate_scope = false}) == GuestEntryError::none);
    CHECK(!stack->in_use());
    for (int value = 0; value <= static_cast<int>(GuestEntryError::invalid_process_landing);
         ++value) {
        const char* text = describe(static_cast<GuestEntryError>(value));
        CHECK(text != nullptr && std::strcmp(text, "unknown guest entry error") != 0);
    }
}

struct GateObservation {
    GuardedStack* stack = nullptr;
    std::uint64_t call_sp = 0;
    std::uint64_t site_sp = 0;
    std::uint64_t local = 0;
    std::uint32_t depth = 0;
    std::uint64_t return_address = 0;
};

GateStatus observing_handler(GateCall& call, void* context) {
    auto& observation = *static_cast<GateObservation*>(context);
    volatile std::uint64_t marker = 0;
    observation.call_sp = call.sp();
    observation.site_sp = call.x(1);
    observation.local = address(const_cast<std::uint64_t*>(&marker));
    observation.depth = call.depth();
    observation.return_address = call.return_address();
    call.set_x(0, call.x(0) * 3);
    return GateStatus::resume;
}

void test_gate_callback_on_host_stack() {
    auto stack = make_stack();
    GateObservation observation{stack.get()};
    GateScope scope(&observing_handler, &observation);
    const GuestEntryResult result =
        invoke_guest_entry(*stack, {.function = code(&armigo_test_guest_gate), .x0 = 14});
    CHECK(result.ok());
    CHECK(result.x0 == 42);
    CHECK(observation.depth == 1);
    CHECK(stack->contains(observation.call_sp));
    CHECK(!stack->contains(observation.local));  // the C++ handler used the saved host stack
    CHECK(observation.site_sp == stack->aligned_top() - ARMIGO_GUEST_ENTRY_STACK_RESERVE - 16);
    CHECK(observation.call_sp == observation.site_sp);
    CHECK(result.x1 == observation.site_sp);
    CHECK(observation.return_address > code(&armigo_test_guest_gate));
    CHECK(GateScope::thread_gate_depth() == 0);
}

struct NestedContext {
    GuardedStack* outer = nullptr;
    GuardedStack* inner = nullptr;
    EntryReport inner_report{};
    GuestEntryError inner_report_error = GuestEntryError::stack_busy;
    GuestEntryError reuse_outer_error = GuestEntryError::none;
    GuestEntryResult inner_gate{};
    std::uint64_t outer_local = 0;
    std::uint64_t inner_local = 0;
    std::uint64_t inner_call_sp = 0;
    std::uint32_t max_depth = 0;
};

GateStatus nested_handler(GateCall& call, void* raw) {
    auto& context = *static_cast<NestedContext*>(raw);
    volatile int marker = 0;
    const std::uint64_t local = address(const_cast<int*>(&marker));
    context.max_depth = std::max(context.max_depth, call.depth());
    if (call.depth() == 1) {
        context.outer_local = local;
        context.inner_report_error =
            invoke_guest_entry(*context.inner, {.function = code(&armigo_test_guest_report),
                                                .x0 = address(&context.inner_report)})
                .error;
        context.reuse_outer_error =
            invoke_guest_entry(*context.outer, {.function = code(&armigo_test_guest_report),
                                                .x0 = address(&context.inner_report)})
                .error;
        context.inner_gate = invoke_guest_entry(
            *context.inner, {.function = code(&armigo_test_guest_gate), .x0 = 5});
        call.set_x(0, 100 + context.inner_gate.x0);
    } else {
        context.inner_local = local;
        context.inner_call_sp = call.sp();
        call.set_x(0, call.x(0) + 1);
    }
    return GateStatus::resume;
}

void test_nested_stacks() {
    auto outer = make_stack();
    auto inner = make_stack();
    NestedContext context{outer.get(), inner.get()};
    GateScope scope(&nested_handler, &context);
    const GuestEntryResult result =
        invoke_guest_entry(*outer, {.function = code(&armigo_test_guest_gate), .x0 = 1});
    CHECK(result.ok());
    CHECK(context.inner_report_error == GuestEntryError::none);
    CHECK(context.inner_report.sp == inner->aligned_top() - ARMIGO_GUEST_ENTRY_STACK_RESERVE);
    CHECK(context.reuse_outer_error == GuestEntryError::stack_busy);
    CHECK(context.inner_gate.ok() && context.inner_gate.x0 == 6);
    CHECK(result.x0 == 106);
    CHECK(context.max_depth == 2);
    CHECK(!outer->contains(context.outer_local) && !inner->contains(context.outer_local));
    CHECK(!outer->contains(context.inner_local) && !inner->contains(context.inner_local));
    CHECK(inner->contains(context.inner_call_sp));
    CHECK(!outer->in_use() && !inner->in_use());
}

struct ConcurrentContext {
    GuardedStack* stack = nullptr;
    std::atomic<int> error{-1};
    std::uint64_t host_local = 0;
};
ConcurrentContext* g_concurrent = nullptr;

std::uint64_t try_from_other_thread(std::uint64_t) {
    volatile int marker = 0;
    g_concurrent->host_local = address(const_cast<int*>(&marker));
    std::thread([] {
        EntryReport report{};
        g_concurrent->error = static_cast<int>(
            invoke_guest_entry(*g_concurrent->stack,
                               {.function = code(&armigo_test_guest_report),
                                .x0 = address(&report), .require_gate_scope = false})
                .error);
    }).join();
    return 7;
}

void test_concurrent_refusal() {
    auto stack = make_stack();
    ConcurrentContext context{stack.get()};
    g_concurrent = &context;
    const GuestEntryResult result = invoke_guest_entry(
        *stack, {.function = code(&armigo_test_guest_call_host),
                 .x0 = reinterpret_cast<std::uint64_t>(&try_from_other_thread),
                 .require_gate_scope = false});
    CHECK(result.ok() && result.x0 == 7);
    CHECK(context.error == static_cast<int>(GuestEntryError::stack_busy));
    CHECK(stack->contains(context.host_local));
    g_concurrent = nullptr;
}

GateStatus throwing_handler(GateCall& call, void*) {
    call.set_x(0, 999);
    throw std::runtime_error("handler failure on the guest stack");
}

void test_handler_exception_contained() {
    auto stack = make_stack();
    GateScope scope(&throwing_handler, nullptr, GateFailurePolicy::resume_unchanged);
    const GuestEntryResult result =
        invoke_guest_entry(*stack, {.function = code(&armigo_test_guest_gate), .x0 = 11});
    CHECK(result.ok());
    CHECK(result.x0 == 11);  // pre-gate state restored
    CHECK(scope.failure_count() == 1);
    CHECK(scope.first_failure() != nullptr &&
          scope.first_failure()->kind == GateFailureKind::handler_threw);
}

void test_repeated_and_deep_use() {
    auto stack = make_stack(std::size_t{64} << 10);
    GateScope scope(&passive_handler);
    EntryReport report{};
    bool all = true;
    for (int iteration = 0; iteration < 2000; ++iteration) {
        const GuestEntryResult result = invoke_guest_entry(
            *stack, {.function = code(&armigo_test_guest_report), .x0 = address(&report),
                     .x1 = static_cast<std::uint64_t>(iteration)});
        all &= result.ok() && result.x1 == ~static_cast<std::uint64_t>(iteration);
    }
    CHECK(all);
    // Touch nearly the whole interior from the guest without leaving it.
    const GuestEntryResult deep = invoke_guest_entry(
        *stack, {.function = code(&armigo_test_guest_descend),
                 .x0 = stack->usable_size() - ARMIGO_GUEST_ENTRY_STACK_RESERVE - 16});
    CHECK(deep.ok() && deep.x0 == 0);
    CHECK(!stack->in_use());
}

void test_thread_entry_validation() {
    auto stack = make_stack(4096);
    std::span<std::byte> span{reinterpret_cast<std::byte*>(stack->aligned_top() - 4096), 4096};
    const auto function = code(&armigo_test_guest_clobber);
    const auto sp = address(span.data()) + span.size() - 32;
    const auto run = [&](GuestThreadEntryRequest request) {
        return invoke_guest_thread(span, request).error;
    };
    CHECK(run({.stack_pointer = sp}) == GuestEntryError::null_function);
    CHECK(run({.function = function + 2, .stack_pointer = sp}) ==
          GuestEntryError::misaligned_function);
    CHECK(run({.function = function, .stack_pointer = sp + 1}) ==
          GuestEntryError::misaligned_stack_top);
    CHECK(run({.function = function, .stack_pointer = stack->aligned_top() + 16}) ==
          GuestEntryError::stack_top_out_of_range);
    CHECK(run({.function = function, .stack_pointer = address(span.data())}) ==
          GuestEntryError::insufficient_stack);
    CHECK(run({.function = function, .stack_pointer = sp, .fpcr = 1u << 8}) ==
          GuestEntryError::invalid_fpcr);
    CHECK(run({.function = address(span.data()), .stack_pointer = sp}) ==
          GuestEntryError::function_not_executable);
    CHECK(run({.function = function, .stack_pointer = sp}) == GuestEntryError::no_gate_scope);
    CHECK(run({.function = function, .stack_pointer = sp,
               .transfer_function = code(&armigo_test_guest_process_landing)}) ==
          GuestEntryError::invalid_process_landing);
    CHECK(run({.function = function, .stack_pointer = sp,
               .original_process = true}) == GuestEntryError::invalid_process_landing);
    CHECK(invoke_guest_thread({}, {.function = function, .stack_pointer = sp}).error ==
          GuestEntryError::stack_top_out_of_range);
    std::span<std::byte> guard{
        reinterpret_cast<std::byte*>(stack->layout().lower_guard_base),
        stack->layout().guard_size};
    CHECK(invoke_guest_thread(guard, {.function = function,
                                     .stack_pointer = address(guard.data()) + 16}).error ==
          GuestEntryError::stack_not_writable);
    CHECK(std::strcmp(describe(GuestEntryError::stack_not_writable),
                      "unknown guest entry error") != 0);
}

struct ProcessEntryReport {
    std::uint64_t x[31];
    std::uint64_t sp;
    std::uint64_t fpcr;
    std::uint64_t fpsr;
    std::uint64_t nzcv;
    std::uint64_t padding;
    std::uint64_t vectors[32][2];
};
static_assert(offsetof(ProcessEntryReport, vectors) == 288);
static_assert(sizeof(ProcessEntryReport) == 800);

void test_original_process_entry_state() {
    auto stack = make_stack(4096);
    std::span<std::byte> span{
        reinterpret_cast<std::byte*>(stack->aligned_top() - 4096), 4096};
    ProcessEntryReport report;
    std::memset(&report, 0xcc, sizeof(report));
    GateScope scope(&exit_handler);
    const auto result = invoke_guest_thread(span, {
        .function = code(&armigo_test_guest_process_entry),
        .x0 = 0,
        .x1 = address(&report),
        .stack_pointer = stack->aligned_top(),
        .fpcr = 0,
        .fpsr = 0,
        .transfer_function = code(&armigo_test_guest_process_landing),
        .original_process = true,
    });
    CHECK(result.exited());
    CHECK(report.x[0] == 0 && report.x[1] == address(&report));
    bool zero_gprs = true;
    for (unsigned reg = 2; reg <= 30; ++reg) {
        if (reg != 18) zero_gprs &= report.x[reg] == 0;
    }
    CHECK(zero_gprs);
    CHECK(report.sp == stack->aligned_top());
    CHECK(report.fpcr == 0 && report.fpsr == 0 && report.nzcv == 0);
    bool zero_vectors = true;
    for (const auto& value : report.vectors) {
        zero_vectors &= value[0] == 0 && value[1] == 0;
    }
    CHECK(zero_vectors);
}

[[noreturn]] void run_child(const std::string& mode) {
    auto stack = make_stack(std::size_t{64} << 10);
    GuestEntryRequest request{.require_gate_scope = false};
    if (mode == "bad-sp") {
        request.function = code(&armigo_test_guest_bad_sp);
    } else if (mode == "corrupt-record") {
        request.function = code(&armigo_test_guest_corrupt_record);
    } else if (mode == "overflow") {
        request.function = code(&armigo_test_guest_descend);
        request.x0 = stack->usable_size() + stack->layout().guard_size;
    } else if (mode == "normal") {
        request.function = code(&armigo_test_guest_clobber);
    } else if (mode == "thread-return") {
        std::span<std::byte> span{
            reinterpret_cast<std::byte*>(stack->aligned_top() - 4096), 4096};
        invoke_guest_thread(span, {.function = code(&armigo_test_guest_clobber),
                                   .stack_pointer = stack->aligned_top() - 32,
                                   .require_gate_scope = false});
        _exit(4);
    } else {
        _exit(2);
    }
    const GuestEntryResult result = invoke_guest_entry(*stack, request);
    _exit(result.ok() ? 0 : 3);
}

struct ChildOutcome {
    bool spawned = false;
    bool exited = false;
    int exit_code = -1;
    int signal = 0;
};

ChildOutcome spawn_child(const char* mode) {
    ChildOutcome outcome;
    char path[4096];
    std::uint32_t size = sizeof path;
    if (_NSGetExecutablePath(path, &size) != 0) return outcome;
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    char arg_child[] = "--child";
    std::string mode_text = mode;
    char* argv[] = {path, arg_child, mode_text.data(), nullptr};
    // Let faults terminate the child with their signal even under a sanitizer.
    std::vector<char*> environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
        if (std::strncmp(*entry, "ASAN_OPTIONS=", 13) != 0 &&
            std::strncmp(*entry, "TSAN_OPTIONS=", 13) != 0) {
            environment.push_back(*entry);
        }
    }
    char asan_options[] = "ASAN_OPTIONS=handle_segv=0:handle_sigbus=0:handle_sigtrap=0";
    char tsan_options[] = "TSAN_OPTIONS=handle_segv=0:handle_sigbus=0:handle_sigtrap=0";
    environment.push_back(asan_options);
    environment.push_back(tsan_options);
    environment.push_back(nullptr);
    pid_t pid = 0;
    const int error = posix_spawn(&pid, path, &actions, nullptr, argv, environment.data());
    posix_spawn_file_actions_destroy(&actions);
    if (error != 0) return outcome;
    outcome.spawned = true;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return outcome;
    if (WIFEXITED(status)) {
        outcome.exited = true;
        outcome.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        outcome.signal = WTERMSIG(status);
    }
    return outcome;
}

void test_children() {
    const ChildOutcome normal = spawn_child("normal");
    CHECK(normal.spawned && normal.exited && normal.exit_code == 0);
    const ChildOutcome bad_sp = spawn_child("bad-sp");
    CHECK(bad_sp.spawned && !bad_sp.exited && bad_sp.signal == SIGTRAP);
    const ChildOutcome corrupt = spawn_child("corrupt-record");
    CHECK(corrupt.spawned && !corrupt.exited && corrupt.signal == SIGTRAP);
    const ChildOutcome thread_return = spawn_child("thread-return");
    CHECK(thread_return.spawned && !thread_return.exited && thread_return.signal == SIGTRAP);
    const ChildOutcome overflow = spawn_child("overflow");
    CHECK(overflow.spawned && !overflow.exited &&
          (overflow.signal == SIGSEGV || overflow.signal == SIGBUS));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--child") == 0) run_child(argv[2]);
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep reports if a defect crashes the process
    test_entry_state();
    test_frame_record_during_call();
    test_custom_top_and_fp_state();
    test_bridge_preserves_host_state();
    test_validation();
    test_gate_callback_on_host_stack();
    test_nested_stacks();
    test_concurrent_refusal();
    test_handler_exception_contained();
    test_repeated_and_deep_use();
    test_thread_entry_validation();
    test_original_process_entry_state();
    test_children();
    std::printf("%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
