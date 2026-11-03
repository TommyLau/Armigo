// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/kernel_queries.h>
#include <armigo/runtime/guest_clock.h>

#include <cstdio>
#include <cstring>
#include <limits>

namespace {

// Matches the existing authored guest_dispatch_fixtures.S capture layout.
struct alignas(16) RegisterImage {
    std::uint64_t x[31]{};
    std::uint64_t nzcv = 0;
    std::uint64_t x18_before = 0;
    std::uint64_t x18_after = 0;
    std::uint64_t sp_before = 0;
    std::uint64_t sp_after = 0;
};
static_assert(sizeof(RegisterImage) == 288 && offsetof(RegisterImage, nzcv) == 248);
static_assert(offsetof(RegisterImage, sp_after) == 280);

using namespace armigo;
using namespace armigo::horizon;
using namespace armigo::execution;
constexpr std::uint64_t heap_address = 0x1000000000;
constexpr std::uint64_t output_address = heap_address + 0x1000;
constexpr std::uint64_t stack_address = 0x6000400000;
constexpr std::uint64_t upper = 0xface123400000000ull;

struct Output {
    std::array<std::byte, 40> bytes{};
    unsigned words = 0;
    static bool write(void* raw, std::uint64_t address,
                      std::span<const std::byte, 4> word) noexcept {
        auto& output = *static_cast<Output*>(raw);
        if (address < output_address || address - output_address > output.bytes.size() - 4)
            return false;
        std::memcpy(output.bytes.data() + (address - output_address), word.data(), word.size());
        ++output.words;
        return true;
    }
};

bool expect(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}
} // namespace

extern "C" void armigo_test_guest_run_site(const RegisterImage*, RegisterImage*, std::uint64_t);
extern "C" const std::byte armigo_test_guest_sites[];

int main() {
    ProcessConfiguration configuration;
    configuration.heap_address = heap_address;
    configuration.alias_address = 0x2000000000;
    configuration.stack_address = 0x6000000000;
    configuration.allowed_core_mask = 5;
    configuration.entropy = {100, 200, 300, 400};
    const std::array blocks = {
        MemoryBlock{{heap_address, 0x2000}, MemoryType::normal, 0, 3},
        MemoryBlock{{stack_address, 0x4000}, MemoryType::stack, 0, 3},
    };
    const auto process = ProcessRecord::create(configuration,
        {0x2000, 0, 0, 0x4000, 0x100000, 0x6000}, blocks);
    if (!expect(process.ok(), "Synthetic process state was rejected.")) return 1;
    HandleTable handles(8);
    auto thread = std::make_shared<ThreadRecord>(process.process, 77, 17);
    const auto main_handle = handles.insert(thread);
    if (!main_handle) return 1;
    Output output;
    StartupQueries queries(process.process, handles, thread, {Output::write, nullptr, &output});
    const auto base = reinterpret_cast<std::uintptr_t>(armigo_test_guest_sites);
    const std::array calls = {0x29, 0x06, 0x0c, 0x25};
    std::vector<GuestGateSite> sites;
    for (std::size_t index = 0; index < calls.size(); ++index) {
        sites.push_back({base + index * 8 + 4,
                         aarch64::encode_svc(static_cast<std::uint16_t>(calls[index])).word});
    }
    const auto table = GuestSiteTable::build(sites);
    if (!expect(table.ok(), "Synthetic SVC sites were rejected.")) return 1;
    GuestThreadContext registers;
    registers.set_thread_pointer(0x1111);
    registers.set_read_only_thread_pointer(0x2222);
    GuestDispatchScope scope(table.table, registers, runtime::query_supervisor(queries),
                             GateFailurePolicy::resume_unchanged);
    set_thread_guest_x18(0x99887766);

    for (std::size_t index = 0; index < calls.size(); ++index) {
        RegisterImage input;
        for (unsigned reg = 0; reg < 31; ++reg) input.x[reg] = upper | (reg + 100);
        input.nzcv = 0xa0000000;
        std::uint64_t expected = 0;
        if (index == 0) {
            input.x[1] = upper | 11;
            input.x[2] = 0;
            input.x[3] = 2;
            expected = 300;
        } else if (index == 1) {
            input.x[0] = output_address;
            input.x[1] = upper | 0x12345678;
            input.x[2] = stack_address + 13;
            expected = upper;
        } else {
            input.x[0] = UINT64_MAX; // Kernel returns in registers; this is not dereferenced.
            input.x[1] = upper | *main_handle;
            expected = index == 2 ? upper | 17 : 77;
        }
        RegisterImage result;
        armigo_test_guest_run_site(&input, &result, base + index * 8);
        if (!expect(result.x[0] == 0 && result.x[1] == expected,
                    "Native SVC query returned the wrong result.")) return 1;
        for (unsigned reg = 2; reg < 18; ++reg) {
            if (!expect(result.x[reg] == 0, "A syscall scratch register was not cleared.")) return 1;
        }
        for (unsigned reg = 19; reg < 30; ++reg) {
            if (!expect(result.x[reg] == input.x[reg], "A preserved guest register changed.")) return 1;
        }
        if (!expect(result.sp_before == result.sp_after && result.nzcv == input.nzcv &&
                    thread_guest_x18() == 0x99887766,
                    "The native query changed guest stack, flags, or virtual x18.")) return 1;
    }
    if (!expect(output.words == 10 && output.bytes == encode_memory_info(
                    {{stack_address, 0x4000}, MemoryType::stack, 0, 3}),
                "The native query did not copy the complete MemoryInfo.")) return 1;
    RegisterImage unsupported;
    for (unsigned reg = 0; reg < 31; ++reg) unsupported.x[reg] = reg + 1;
    unsupported.x[1] = 9; // Known kernel type, deliberately not implemented yet.
    RegisterImage unchanged;
    armigo_test_guest_run_site(&unsupported, &unchanged, base);
    for (unsigned reg = 0; reg < 30; ++reg) {
        if (reg != 18 && !expect(unchanged.x[reg] == unsupported.x[reg],
                                 "An unsupported query partially changed registers.")) return 1;
    }
    if (!expect(scope.dispatched() == 4 && scope.failure_count() == 1 &&
                scope.gate_scope().failure_count() == 1 &&
                registers.thread_pointer() == 0x1111 &&
                registers.read_only_thread_pointer() == 0x2222,
                "Native query capability accounting or thread state is wrong.")) return 1;
    set_thread_guest_x18(0);
    std::puts("Four Horizon startup queries passed through the native guest gate.");
    if (!expect(armigo::runtime::classify_guest_sleep(1) == armigo::runtime::GuestSleep::duration &&
                    armigo::runtime::classify_guest_sleep(0) == armigo::runtime::GuestSleep::yield &&
                    armigo::runtime::classify_guest_sleep(-1) == armigo::runtime::GuestSleep::yield &&
                    armigo::runtime::classify_guest_sleep(-2) == armigo::runtime::GuestSleep::yield &&
                    armigo::runtime::classify_guest_sleep(-3) == armigo::runtime::GuestSleep::ignore &&
                    armigo::runtime::classify_guest_sleep(-4) == armigo::runtime::GuestSleep::ignore,
                "Guest sleep classification is wrong.")) return 1;

    // Reuse an authored gate site with a sleep opcode in the dispatch metadata.
    const std::array sleep_sites = {GuestGateSite{base + 4, aarch64::encode_svc(0x0b).word}};
    const auto sleep_table = GuestSiteTable::build(sleep_sites);
    if (!expect(sleep_table.ok(), "Synthetic sleep site was rejected.")) return 1;
    using namespace armigo::runtime;
    const auto now = ClockInstant(std::chrono::nanoseconds(10));
    ClockInstant observed_deadline{};
    unsigned waits = 0;
    bool cancel_wait = false;
    auto clock_result = create_guest_clock({[&] { return now; },
        [&](ClockInstant deadline, std::stop_token) -> LaunchResult<ClockWait> {
            observed_deadline = deadline;
            ++waits;
            return cancel_wait ? ClockWait::cancelled : ClockWait::reached;
        }});
    if (!expect(std::holds_alternative<std::shared_ptr<GuestClock>>(clock_result),
                "Synthetic sleep clock was rejected.")) return 1;
    auto clock = std::get<std::shared_ptr<GuestClock>>(std::move(clock_result));
    IpcDispatcher ipc(handles, nullptr);
    StartupSupervisor supervisor(queries, ipc, {}, clock.get());
    GuestDispatchScope sleep_scope(sleep_table.table, registers, supervisor.binding(),
                                   GateFailurePolicy::resume_unchanged);
    struct SleepCase { std::int64_t input; std::int64_t duration; };
    const std::array sleep_cases = {
        SleepCase{0, 0}, SleepCase{-1, 0}, SleepCase{-2, 0}, SleepCase{-3, 0},
        SleepCase{1, 156}, SleepCase{625, 729}, SleepCase{1'000'000, 1'000'104},
        SleepCase{std::numeric_limits<std::int64_t>::max(), -1},
    };
    for (const auto test : sleep_cases) {
        thread->cancel_synchronization();
        RegisterImage input;
        for (unsigned reg = 0; reg < 31; ++reg) input.x[reg] = upper | reg;
        input.x[0] = static_cast<std::uint64_t>(test.input);
        input.nzcv = 0xa0000000;
        const auto previous_waits = waits;
        RegisterImage result;
        armigo_test_guest_run_site(&input, &result, base);
        for (unsigned reg = 0; reg < 18; ++reg)
            if (!expect(result.x[reg] == 0, "Sleep did not clear its return registers.")) return 1;
        for (unsigned reg = 19; reg < 30; ++reg)
            if (!expect(result.x[reg] == input.x[reg], "Sleep changed a preserved register.")) return 1;
        if (!expect(result.sp_before == result.sp_after && result.nzcv == input.nzcv,
                    "Sleep changed stack or flags.")) return 1;
        if (test.input > 0) {
            const auto expected_deadline = test.duration < 0 ? ClockInstant::max()
                : now + std::chrono::nanoseconds(test.duration);
            if (!expect(waits == previous_waits + 1 && observed_deadline == expected_deadline,
                        "Sleep tick rounding or saturated deadline is wrong.")) return 1;
        } else if (!expect(waits == previous_waits, "A yield or ignored sleep blocked.")) return 1;
        QueryRegisters pending{0, UINT64_MAX, 0, UINT64_MAX};
        if (!expect(queries.dispatch(0x18, pending) == QueryDisposition::handled &&
                    pending[0] == 0xec01 && pending[1] == UINT64_MAX,
                    "Sleep consumed guest synchronization cancellation.")) return 1;
    }
    // Yields follow the kernel's fast path: while the thread's lone-yield count equals the
    // process's scheduled count, a yield returns at once and counts nothing. A scheduling
    // event in between makes the next yield a counted host yield, and a timed sleep is one.
    {
        auto& process = *thread->process();
        const auto run_sleep = [&](std::int64_t nanoseconds) {
            RegisterImage input;
            for (unsigned reg = 0; reg < 31; ++reg) input.x[reg] = upper | reg;
            input.x[0] = static_cast<std::uint64_t>(nanoseconds);
            RegisterImage result;
            armigo_test_guest_run_site(&input, &result, base);
        };
        thread->set_lone_yield_count(process.scheduled_count());
        const auto quiet = process.scheduled_count();
        run_sleep(0);
        run_sleep(-1);
        if (!expect(process.scheduled_count() == quiet,
                    "A yield with nothing scheduled since a lone yield was counted.")) return 1;
        process.note_scheduling();
        const auto noted = process.scheduled_count();
        run_sleep(-2);
        if (!expect(process.scheduled_count() == noted + 1,
                    "A yield after a scheduling event was skipped.")) return 1;
        const auto before_sleep = process.scheduled_count();
        run_sleep(1);
        if (!expect(process.scheduled_count() == before_sleep + 1,
                    "A timed sleep was not a scheduling event.")) return 1;
    }
    cancel_wait = true;
    RegisterImage cancelled_input;
    for (unsigned reg = 0; reg < 31; ++reg) cancelled_input.x[reg] = upper | reg;
    cancelled_input.x[0] = 1;
    RegisterImage cancelled_result;
    armigo_test_guest_run_site(&cancelled_input, &cancelled_result, base);
    for (unsigned reg = 0; reg < 18; ++reg)
        if (!expect(cancelled_result.x[reg] == cancelled_input.x[reg],
                    "Host-cancelled sleep fabricated a guest return.")) return 1;
    if (!expect(sleep_scope.failure_count() == 1,
                "Host-cancelled sleep without an exit target did not fail.")) return 1;

    return 0;
}
