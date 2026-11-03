// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/kernel_queries.h>

#include <armigo/aarch64/instruction.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

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

extern "C" void armigo_test_guest_run_site(const RegisterImage*, RegisterImage*,
                                             std::uint64_t);
extern "C" const std::byte armigo_test_guest_sites[];

namespace {

using namespace armigo;
using namespace armigo::execution;
using namespace armigo::horizon;
using namespace armigo::runtime;

constexpr std::uint64_t mib = 1024 * 1024;
constexpr std::uint64_t seed = 0x9876000000000000ULL;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

[[maybe_unused]] LaunchProfile profile() {
    LaunchProfile result;
    result.origin = ProfileOrigin::synthetic_test;
    result.process.allowed_core_mask = 1;
    result.process.heap_address = 0;
    result.process.alias_address = 0;
    result.process.stack_address = 0;
    result.process.heap_size = 8 * mib;
    result.process.base_alias_size = 2 * mib;
    result.process.stack_size = 2 * mib;
    result.permissions.handle_table_capacity = 64;
    return result;
}

[[maybe_unused]] RegisterImage input_registers() {
    RegisterImage result;
    for (unsigned index = 0; index < 31; ++index) result.x[index] = seed + index;
    result.nzcv = 0xa0000000;
    return result;
}

[[maybe_unused]] bool preserved_native_state(const RegisterImage& input,
                                             const RegisterImage& output) {
    if (output.nzcv != input.nzcv || output.sp_before != output.sp_after) return false;
    for (unsigned index = 19; index < 30; ++index) {
        if (output.x[index] != input.x[index]) return false;
    }
    return true;
}

#if defined(__APPLE__)
[[maybe_unused]] vm_prot_t native_protection(std::uint64_t address) {
    mach_vm_address_t region_address = address;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(
        mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) ::mach_port_deallocate(mach_task_self(), object);
    CHECK(result == KERN_SUCCESS && region_address <= address &&
          address - region_address < region_size);
    return result == KERN_SUCCESS ? info.protection : VM_PROT_NONE;
}
#endif

template <class Supervisor>
constexpr bool has_memory_route =
    std::is_constructible_v<Supervisor, StartupQueries&, IpcDispatcher&,
                            IpcCallContext, std::shared_ptr<GuestMemory>> &&
    requires(const Supervisor& supervisor) { supervisor.memory_failure(); };

template <class Supervisor>
int run_dispatch_integration() {
    if constexpr (!has_memory_route<Supervisor>) {
        std::puts("StartupSupervisor memory-first dispatch integration skipped: "
                  "the memory-aware constructor is unavailable.");
        return 0;
    } else {
        auto handles = std::make_shared<HandleTable>(64);
        auto made = create_guest_memory(profile(), {16 * mib}, nullptr, handles);
        CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
        if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(made)) return 1;
        auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(made));
        const auto process = memory->process();
        auto thread = std::make_shared<ThreadRecord>(process, 77, 17);
        CHECK(handles->insert(thread).has_value());
        StartupQueries queries(process, *handles, thread, memory->copy_out());
        IpcDispatcher ipc(*handles, nullptr);
        IpcCallContext caller{};
        Supervisor supervisor(queries, ipc, caller, memory);

        const auto base = reinterpret_cast<std::uintptr_t>(armigo_test_guest_sites);
        const std::array<std::uint16_t, 4> calls{1, 2, 6, 0x15};
        std::vector<GuestGateSite> sites;
        for (std::size_t index = 0; index < calls.size(); ++index) {
            sites.push_back({base + index * 8 + 4,
                             aarch64::encode_svc(calls[index]).word});
        }
        auto table = GuestSiteTable::build(sites);
        CHECK(table.ok());
        if (!table.ok()) return 1;
        GuestThreadContext thread_context;
        GuestDispatchScope scope(table.table, thread_context, supervisor.binding(),
                                 GateFailurePolicy::resume_unchanged);

        RegisterImage heap_input = input_registers();
        heap_input.x[1] = 2 * mib;
        RegisterImage heap_output;
        armigo_test_guest_run_site(&heap_input, &heap_output, base);
        const std::uint64_t heap = process->heap_region().address;
        CHECK(heap_output.x[0] == 0 && heap_output.x[1] == heap);
        CHECK(std::all_of(heap_output.x + 2, heap_output.x + 18,
                          [](std::uint64_t value) { return value == 0; }));
        CHECK(process->used_memory() == 2 * mib &&
              process->query_memory(heap).permissions == 3);
        CHECK(preserved_native_state(heap_input, heap_output));

        const auto native = platform::HostModule::native_page_size();
        CHECK(native.has_value());
        if (!native) return 1;
        RegisterImage protect_input = input_registers();
        protect_input.x[0] = heap;
        protect_input.x[1] = *native;
        protect_input.x[2] = 1;
        RegisterImage protect_output;
        armigo_test_guest_run_site(&protect_input, &protect_output, base + 8);
        CHECK(protect_output.x[0] == 0 && protect_output.x[1] == 0);
        CHECK(process->query_memory(heap).permissions == 1);
#if defined(__APPLE__)
        const auto protection = native_protection(heap);
        CHECK((protection & VM_PROT_READ) != 0 &&
              (protection & (VM_PROT_WRITE | VM_PROT_EXECUTE)) == 0);
#endif
        CHECK(preserved_native_state(protect_input, protect_output));

        const std::uint64_t output_address = heap + *native;
        RegisterImage query_input = input_registers();
        query_input.x[0] = output_address;
        query_input.x[1] = seed | 0x1234;
        query_input.x[2] = heap;
        RegisterImage query_output;
        armigo_test_guest_run_site(&query_input, &query_output, base + 16);
        CHECK(query_output.x[0] == 0 && query_output.x[1] == seed);
        auto output = memory->pin({output_address, sizeof(MemoryInfo)},
                                  memory::Permission::read);
        CHECK(std::holds_alternative<GuestMemoryPin>(output));
        if (auto* pin = std::get_if<GuestMemoryPin>(&output)) {
            const auto expected = encode_memory_info(process->query_memory(heap));
            CHECK(std::memcmp((*pin)->bytes().data(), expected.data(), expected.size()) == 0);
        }
        CHECK(preserved_native_state(query_input, query_output));

        // CreateTransferMemory reaches the startup queries, which record the
        // range and return a new handle.
        const auto handles_before = handles->size();
        RegisterImage transfer_input = input_registers();
        transfer_input.x[1] = output_address;
        transfer_input.x[2] = *native;
        transfer_input.x[3] = 3;
        RegisterImage transfer_output;
        armigo_test_guest_run_site(&transfer_input, &transfer_output, base + 24);
        CHECK(transfer_output.x[0] == 0 && transfer_output.x[1] != 0);
        CHECK(handles->size() == handles_before + 1);
        CHECK(!supervisor.memory_failure().has_value());
        CHECK(scope.dispatched() == 4 && scope.failure_count() == 0 &&
              scope.gate_scope().failure_count() == 0);
        return failures == 0 ? 0 : 1;
    }
}

} // namespace

int main() {
    return run_dispatch_integration<armigo::runtime::StartupSupervisor>();
}
