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
#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/kernel_queries.h>
#include <armigo/horizon/read_only_shared_memory.h>

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

// Fail one metadata reservation before forwarding any native operation. This
// exercises the real guest gate and its retry contract, not a direct C++ return.
class CapacityFaultMemory final : public GuestMemory {
public:
    explicit CapacityFaultMemory(std::shared_ptr<GuestMemory> backing) : backing_(std::move(backing)) {}
    bool refuse = false;
    std::shared_ptr<ProcessRecord> process() const noexcept override { return backing_->process(); }
    LaunchResult<GuestMemoryPin> allocate(std::uint64_t size, MemoryType type,
        memory::Permission access) override { return backing_->allocate(size, type, access); }
    LaunchStatus release(AddressRange range) override { return backing_->release(range); }
    LaunchResult<QueryDisposition> dispatch(std::uint16_t svc, QueryRegisters& registers) override {
        if (svc == 2 && std::exchange(refuse, false))
            return LaunchFailure{.stage = LaunchStage::memory, .code = LaunchError::resource_limit,
                .operation = "SetMemoryPermission", .detail = "Synthetic interval capacity exhaustion.",
                .guest_result = 0xce01};
        return backing_->dispatch(svc, registers);
    }
    LaunchResult<GuestMemoryPin> pin(AddressRange range, memory::Permission access) override {
        return backing_->pin(range, access);
    }
    GuestCopyOut copy_out() noexcept override { return backing_->copy_out(); }
    GuestBuffers ipc_buffers() noexcept override { return backing_->ipc_buffers(); }
    GuestMemoryMapper service_mapper() noexcept override { return backing_->service_mapper(); }
private:
    std::shared_ptr<GuestMemory> backing_;
};

template <class Supervisor>
constexpr bool has_memory_route =
    std::is_constructible_v<Supervisor, StartupQueries&, IpcDispatcher&,
                            IpcCallContext, std::shared_ptr<GuestMemory>> &&
    requires(const Supervisor& supervisor) { supervisor.memory_failure(); };

void run_application_budget_contract() {
    auto configuration = profile();
    configuration.process.platform = GuestPlatform::switch2;
    configuration.process.baseline = Baseline::horizon_23_0_1;
    auto handles = std::make_shared<HandleTable>(64);
    auto made = create_guest_memory(configuration, {4 * mib}, nullptr, handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
    if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(made)) return;
    auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(made));
    auto process = memory->process();
    const auto native = platform::HostModule::native_page_size();
    CHECK(native.has_value());
    if (!native) return;
    QueryRegisters heap{};
    heap[1] = 4 * mib;
    auto result = memory->dispatch(1, heap);
    CHECK(std::holds_alternative<QueryDisposition>(result) && heap[0] == 0);
    CHECK(process->used_memory() == 4 * mib && process->total_memory() == 4 * mib);
    CHECK(std::holds_alternative<LaunchFailure>(memory->allocate(*native,
        MemoryType::normal, memory::Permission::read | memory::Permission::write)));
    heap = {}; heap[1] = 6 * mib;
    result = memory->dispatch(1, heap);
    CHECK(std::holds_alternative<QueryDisposition>(result) && heap[0] == 0x10801);
    CHECK(process->used_memory() == 4 * mib && process->total_memory() == 4 * mib);

    // An alias of already charged pages works even with no free application
    // budget, retains coherent bytes, and never charges the source a second time.
    const auto source = process->heap_region().address;
    const auto destination = process->stack_region().address;
    {
        auto pinned = memory->pin({source, *native}, memory::Permission::read | memory::Permission::write);
        CHECK(std::holds_alternative<GuestMemoryPin>(pinned));
        if (auto* pin = std::get_if<GuestMemoryPin>(&pinned)) (*pin)->writable_bytes()[0] = std::byte{0x5a};
    }
    QueryRegisters alias{};
    alias[0] = destination; alias[1] = source; alias[2] = *native;
    result = memory->dispatch(4, alias);
    CHECK(std::holds_alternative<QueryDisposition>(result) && alias[0] == 0);
    CHECK(process->used_memory() == 4 * mib);
    {
        auto pinned = memory->pin({destination, *native}, memory::Permission::read);
        CHECK(std::holds_alternative<GuestMemoryPin>(pinned));
        if (auto* pin = std::get_if<GuestMemoryPin>(&pinned)) CHECK((*pin)->bytes()[0] == std::byte{0x5a});
    }
    alias = {}; alias[0] = destination; alias[1] = source; alias[2] = *native;
    result = memory->dispatch(5, alias);
    CHECK(std::holds_alternative<QueryDisposition>(result) && alias[0] == 0);
    CHECK(process->used_memory() == 4 * mib);
    heap = {};
    result = memory->dispatch(1, heap);
    CHECK(std::holds_alternative<QueryDisposition>(result) && heap[0] == 0);
    CHECK(process->used_memory() == 0);

    // A different application allocation consumes the very same released quota.
    auto normal = memory->allocate(4 * mib, MemoryType::normal,
        memory::Permission::read | memory::Permission::write);
    CHECK(std::holds_alternative<GuestMemoryPin>(normal));
    if (auto* pin = std::get_if<GuestMemoryPin>(&normal)) {
        CHECK(process->used_memory() == 4 * mib);
        CHECK(std::holds_alternative<LaunchFailure>(memory->allocate(*native,
            MemoryType::normal, memory::Permission::read)));
        const auto range = (*pin)->range();
        pin->reset();
        CHECK(std::holds_alternative<std::monostate>(memory->release(range)));
        CHECK(process->used_memory() == 0 && process->total_memory() == 4 * mib);
    }
}

void run_physical_mapping_contract() {
    auto config = profile();
    config.process.platform = GuestPlatform::switch2;
    config.process.baseline = Baseline::horizon_23_0_1;
    config.process.private_system_resource_bytes = 65536;
    auto handles = std::make_shared<HandleTable>(64);
    auto made = create_guest_memory(config, {65536 + 8 * 4096}, nullptr, handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
    if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(made)) return;
    auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(made));
    const auto process = memory->process();
    const auto base = process->alias_region().address;
    const auto initial = process->used_memory();
    const auto call = [&](unsigned svc, std::uint64_t address, std::uint64_t size) {
        QueryRegisters r{}; r[0] = address; r[1] = size;
        const auto result = memory->dispatch(svc, r);
        CHECK(std::holds_alternative<QueryDisposition>(result) &&
              std::get<QueryDisposition>(result) == QueryDisposition::handled);
        return r[0];
    };
    for (auto svc : {0x2cu, 0x2du}) {
        CHECK(call(svc, base + 1, 0) == 0xcc01);
        CHECK(call(svc, base, 0) == 0xca01);
        CHECK(call(svc, base, 1) == 0xca01);
        CHECK(call(svc, ~std::uint64_t{4095}, 4096) == 0xdc01);
        CHECK(call(svc, process->heap_region().address, 4096) == 0xdc01);
    }
    CHECK(call(0x2c, base + 4096, 4096) == 0);
    CHECK(process->used_memory() == initial + 4096);
    {
        auto pinned = memory->pin({base + 4096, 4096}, memory::Permission::read | memory::Permission::write);
        CHECK(std::holds_alternative<GuestMemoryPin>(pinned));
        if (auto* pin = std::get_if<GuestMemoryPin>(&pinned)) {
            CHECK(std::all_of((*pin)->bytes().begin(), (*pin)->bytes().end(),
                [](auto byte) { return byte == std::byte{}; }));
            (*pin)->writable_bytes()[0] = std::byte{0x5a};
            QueryRegisters r{}; r[0] = base + 4096; r[1] = 4096;
            CHECK(std::holds_alternative<LaunchFailure>(memory->dispatch(0x2d, r)));
        }
    }
    // Fill holes around a live subpage without clearing it or charging it twice.
    CHECK(call(0x2c, base, 4 * 4096) == 0);
    CHECK(call(0x2c, base, 4 * 4096) == 0);
    CHECK(process->used_memory() == initial + 4 * 4096);
    {
        auto pinned = memory->pin({base, 4 * 4096}, memory::Permission::read);
        CHECK(std::holds_alternative<GuestMemoryPin>(pinned));
        if (auto* pin = std::get_if<GuestMemoryPin>(&pinned)) {
            CHECK((*pin)->bytes()[4096] == std::byte{0x5a});
            CHECK((*pin)->bytes()[0] == std::byte{} && (*pin)->bytes()[8192] == std::byte{});
        }
    }
    QueryRegisters attribute{}; attribute[0] = base; attribute[1] = 4096;
    attribute[2] = 8; attribute[3] = 8;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(3, attribute)) && attribute[0] == 0);
    CHECK(call(0x2d, base, 4 * 4096) == 0xd401);
    CHECK(call(0x2c, base, 4 * 4096) == 0); // Already occupied, including attributes.
    attribute = {}; attribute[0] = base; attribute[1] = 4096; attribute[2] = 8;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(3, attribute)) && attribute[0] == 0);
    // Invalid masks, values and lock clears are invalid combinations (0xe801),
    // checked before the memory state: an unsupported bit on mapped memory and a
    // lock clear on unmapped memory both fail with it.
    attribute = {}; attribute[0] = base; attribute[1] = 4096; attribute[2] = 2;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(3, attribute)) && attribute[0] == 0xe801);
    attribute = {}; attribute[0] = base; attribute[1] = 4096; attribute[2] = 0; attribute[3] = 8;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(3, attribute)) && attribute[0] == 0xe801);
    attribute = {}; attribute[0] = base + 64 * 4096; attribute[1] = 4096; attribute[2] = 0x10;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(3, attribute)) && attribute[0] == 0xe801);
    attribute = {}; attribute[0] = base + 64 * 4096; attribute[1] = 4096; attribute[2] = 8;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(3, attribute)) && attribute[0] == 0xd401);
    // SetGpuMemoryAttribute (Switch 2 svc 0x80) changes only the GPU-accessible
    // attribute, which QueryMemory reports.
    QueryRegisters gpu{}; gpu[0] = base; gpu[1] = 4096; gpu[2] = 0x20; gpu[3] = 0x20;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(0x80, gpu)) && gpu[0] == 0);
    CHECK(process->query_memory(base).attributes == 0x20);
    gpu = {}; gpu[0] = base; gpu[1] = 4096; gpu[2] = 8; gpu[3] = 8;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(0x80, gpu)) && gpu[0] == 0xe801);
    gpu = {}; gpu[0] = base; gpu[1] = 4096; gpu[2] = 0x20;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(0x80, gpu)) && gpu[0] == 0);
    CHECK(process->query_memory(base).attributes == 0);
    CHECK(call(0x2c, base, 9 * 4096) == 0x10801);
    CHECK(process->used_memory() == initial + 4 * 4096);
    CHECK(call(0x2d, base, 4096) == 0);
    CHECK(process->query_memory(base).type == MemoryType::free);
    CHECK(process->query_memory(base + 4096).type == MemoryType::normal);
    CHECK(call(0x2c, base, 4096) == 0);
    QueryRegisters protect{}; protect[0] = base + 4096; protect[1] = 4096; protect[2] = 1;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(2, protect)) && protect[0] == 0);
    CHECK(call(0x2d, base + 4096, 4096) == 0);
    CHECK(call(0x2c, base + 4096, 4096) == 0);
    {
        auto pinned = memory->pin({base + 4096, 1}, memory::Permission::read);
        CHECK(std::holds_alternative<GuestMemoryPin>(pinned));
        if (auto* pin = std::get_if<GuestMemoryPin>(&pinned)) CHECK((*pin)->bytes()[0] == std::byte{});
    }
    CHECK(call(0x2d, base, 8 * 4096) == 0);
    CHECK(call(0x2d, base, 8 * 4096) == 0);
    CHECK(process->used_memory() == initial);
    config.process.private_system_resource_bytes = 0;
    auto no_resources = create_guest_memory(config, {mib}, nullptr, handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(no_resources));
    if (auto* other = std::get_if<std::shared_ptr<GuestMemory>>(&no_resources)) {
        QueryRegisters r{}; r[0] = (*other)->process()->alias_region().address; r[1] = 4096;
        CHECK(std::holds_alternative<QueryDisposition>((*other)->dispatch(0x2c, r)) && r[0] == 0xfa01);
    }
}

void run_remote_window_contract() {
    auto handles = std::make_shared<HandleTable>(64);
    auto configuration = profile();
    configuration.process.platform = GuestPlatform::switch2;
    configuration.process.baseline = Baseline::horizon_23_0_1;
    constexpr std::uint64_t application_budget = 9809 * mib;
    auto too_large = create_guest_memory(configuration, {application_budget + 4096}, nullptr, handles);
    CHECK(std::holds_alternative<LaunchFailure>(too_large));
    auto made = create_guest_memory(configuration, {application_budget}, nullptr, handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
    if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(made)) return;
    auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(made));
    auto process = memory->process();
    auto thread = std::make_shared<ThreadRecord>(process, 77, 17);
    StartupQueries queries(process, *handles, thread, memory->copy_out());
    const auto query = [&](unsigned type, std::uint64_t handle = current_process_handle,
                           std::uint64_t sub_id = 0) {
        QueryRegisters registers{};
        registers[1] = type; registers[2] = handle; registers[3] = sub_id;
        const auto disposition = queries.dispatch(0x29, registers);
        CHECK(disposition == QueryDisposition::handled);
        return registers;
    };
    const auto address = query(29), size = query(30);
    CHECK(query(33)[1] == 0x1000);
    CHECK(address[0] == 0 && size[0] == 0);
    CHECK(address[1] != 0 && (address[1] & 0xffff) == 0);
    CHECK(size[1] == (8ull << 30));
    if (size[1] != (8ull << 30)) return;
    const AddressRange remote{address[1], size[1]};
    for (const auto range : {process->heap_region(), process->alias_region(), process->stack_region()})
        CHECK(remote.address >= range.address + range.size ||
              range.address >= remote.address + remote.size);
    CHECK(remote.address >= ProcessRecord::aslr_start &&
          remote.address + remote.size <= ProcessRecord::query_end);
    CHECK(process->used_memory() == 0);
    CHECK(process->total_memory() == application_budget);
    CHECK(process->query_memory(remote.address).type == MemoryType::free);
    CHECK(!process->writable(remote.address, 1));
    const auto end = remote.address + remote.size;
    const auto buffers = memory->ipc_buffers();
    for (const auto address : {remote.address - 1, remote.address, end - 1, end}) {
        std::array<std::byte, 2> bytes{std::byte{0x5a}, std::byte{0xa5}};
        const auto saved = bytes;
        CHECK(std::holds_alternative<LaunchFailure>(memory->pin({address, bytes.size()},
            memory::Permission::read)));
        CHECK(!memory->copy_observation(address, bytes) && bytes == saved);
        CHECK(!buffers.read_memory(address, bytes) && bytes == saved);
        CHECK(!buffers.write_memory(address, bytes));
    }
    // Valid shared objects cannot turn either remote boundary into ordinary
    // backing. Failure must preserve accounting, query metadata and native VA.
    constexpr std::uint64_t page = 0x4000;
    auto object = std::make_shared<ReadOnlySharedMemory>(std::vector<std::byte>(page, std::byte{0x73}));
    const auto handle = handles->insert(object);
    CHECK(handle.has_value());
    if (handle) for (const auto address : {remote.address, end - page}) {
        const auto before = encode_memory_info(process->query_memory(address));
        const auto used = process->used_memory();
        QueryRegisters registers{};
        registers[0] = *handle; registers[1] = address; registers[2] = page; registers[3] = 1;
        const auto result = memory->dispatch(0x13, registers);
        CHECK(std::holds_alternative<LaunchFailure>(result) ||
              (std::get<QueryDisposition>(result) == QueryDisposition::handled && registers[0] != 0));
        CHECK(process->used_memory() == used &&
              encode_memory_info(process->query_memory(address)) == before);
        CHECK(std::holds_alternative<LaunchFailure>(memory->pin({address, page}, memory::Permission::read)));
    }
#if defined(__APPLE__)
    CHECK(native_protection(remote.address) == VM_PROT_NONE);
    CHECK(native_protection(remote.address + remote.size - 1) == VM_PROT_NONE);
#endif
    CHECK(query(29, 0)[0] == 0xe401);
    CHECK(query(30, current_process_handle, 1)[0] == 0xe801);
    const auto valid = process->configuration();
    const MemoryAccounting accounting{.resource_limit_bytes = 16 * mib};
    // The remote allocator is independent of the general allocation interval.
    auto separate = valid;
    separate.heap_address = 0x100000000;
    separate.alias_address = 0x200000000;
    separate.stack_address = 0x300000000;
    separate.heap_size = separate.base_alias_size = separate.stack_size = 0x10000;
    separate.allocation_region = {0x100000000, 0x200010000};
    separate.remote_region = {0x400000000, 8ull << 30};
    CHECK(ProcessRecord::create(separate, accounting).ok());
    for (const auto range : {process->heap_region(), process->alias_region(), process->stack_region()}) {
        auto invalid = valid;
        invalid.remote_region = {range.address, 0x10000};
        CHECK(ProcessRecord::create(invalid, accounting).error == ProcessStateError::invalid_layout);
    }
    for (const auto range : {AddressRange{remote.address + 1, remote.size},
                             AddressRange{remote.address, remote.size - 1},
                             AddressRange{UINT64_MAX - 0xffff, 0x20000},
                             AddressRange{ProcessRecord::query_end, 0x10000}}) {
        auto invalid = valid;
        invalid.remote_region = range;
        CHECK(ProcessRecord::create(invalid, accounting).error == ProcessStateError::invalid_layout);
    }
    auto switch1 = valid;
    switch1.platform = GuestPlatform::switch1;
    switch1.baseline = Baseline::horizon_23_0_1;
    CHECK(ProcessRecord::create(switch1, accounting).error == ProcessStateError::invalid_layout);
    switch1.remote_region = {};
    auto switch1_record = ProcessRecord::create(switch1, accounting);
    CHECK(switch1_record.ok());
    if (switch1_record.ok()) {
        StartupQueries switch1_queries(switch1_record.process, *handles,
            std::make_shared<ThreadRecord>(switch1_record.process, 78, 17));
        QueryRegisters registers{};
        registers[1] = 29; registers[2] = current_process_handle;
        CHECK(switch1_queries.dispatch(0x29, registers) == QueryDisposition::handled);
        CHECK(registers[0] == 0xf001);
    }
    auto missing = valid;
    missing.remote_region = {};
    auto missing_record = ProcessRecord::create(missing, accounting);
    CHECK(missing_record.ok());
    if (missing_record.ok()) {
        StartupQueries missing_queries(missing_record.process, *handles,
            std::make_shared<ThreadRecord>(missing_record.process, 79, 17));
        QueryRegisters registers{};
        registers[1] = 29; registers[2] = current_process_handle;
        CHECK(missing_queries.dispatch(0x29, registers) == QueryDisposition::unsupported_operation);
    }
    for (const auto type : {37u, 38u}) {
        QueryRegisters registers{};
        registers[1] = type; registers[2] = current_process_handle;
        CHECK(queries.dispatch(0x29, registers) == QueryDisposition::unsupported_operation);
    }
    // Remote memory size and use report zero as provisional values.
    for (const auto type : {31u, 32u}) {
        QueryRegisters registers{};
        registers[1] = type; registers[2] = current_process_handle;
        CHECK(queries.dispatch(0x29, registers) == QueryDisposition::handled &&
              registers[0] == 0 && registers[1] == 0);
    }
#if defined(__APPLE__)
    // A live reservation excludes other native allocations, then releases its VA
    // with its owner even while query metadata remains retained for inspection.
    mach_vm_address_t probe = remote.address;
    const auto collision = ::mach_vm_allocate(mach_task_self(), &probe, remote.size, VM_FLAGS_FIXED);
    CHECK(collision != KERN_SUCCESS);
    if (collision == KERN_SUCCESS)
        CHECK(::mach_vm_deallocate(mach_task_self(), probe, remote.size) == KERN_SUCCESS);
    memory.reset();
    probe = remote.address;
    const auto allocated = ::mach_vm_allocate(mach_task_self(), &probe, remote.size, VM_FLAGS_FIXED);
    CHECK(allocated == KERN_SUCCESS);
    if (allocated == KERN_SUCCESS) CHECK(::mach_vm_deallocate(mach_task_self(), probe, remote.size) == KERN_SUCCESS);
#endif
}

template <class Supervisor>
int run_dispatch_integration(bool private_resources) {
    if constexpr (!has_memory_route<Supervisor>) {
        std::puts("StartupSupervisor memory-first dispatch integration skipped: "
                  "the memory-aware constructor is unavailable.");
        return 0;
    } else {
        auto handles = std::make_shared<HandleTable>(64);
        auto configuration = profile();
        if (private_resources) {
            configuration.process.platform = GuestPlatform::switch2;
            configuration.process.baseline = Baseline::horizon_23_0_1;
            configuration.process.private_system_resource_bytes = 3 * 4096;
        }
        auto made = create_guest_memory(configuration, {16 * mib}, nullptr, handles);
        CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
        if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(made)) return 1;
        auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(made));
        const auto process = memory->process();
        auto thread = std::make_shared<ThreadRecord>(process, 77, 17);
        CHECK(handles->insert(thread).has_value());
        StartupQueries queries(process, *handles, thread, memory->copy_out());
        IpcDispatcher ipc(*handles, nullptr);
        IpcCallContext caller{};
        auto capacity = std::make_shared<CapacityFaultMemory>(memory);
        Supervisor supervisor(queries, ipc, caller, capacity);

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
        const auto used_before = process->used_memory();
        const auto total_before = process->total_memory();
        const auto map_before = process->query_memory(heap);
        capacity->refuse = true;
        armigo_test_guest_run_site(&protect_input, &protect_output, base + 8);
        CHECK(protect_output.x[0] == 0xce01);
        CHECK(std::all_of(protect_output.x + 1, protect_output.x + 18,
                          [](std::uint64_t value) { return value == 0; }));
        CHECK(preserved_native_state(protect_input, protect_output));
        CHECK(process->used_memory() == used_before && process->total_memory() == total_before);
        CHECK(encode_memory_info(process->query_memory(heap)) == encode_memory_info(map_before));
        CHECK(!supervisor.memory_failure().has_value() && scope.failure_count() == 0);
#if defined(__APPLE__)
        CHECK((native_protection(heap) & VM_PROT_WRITE) != 0);
#endif
        // Retry executes normally through the same guest gate and memory owner.
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
        CHECK(scope.dispatched() == 5 && scope.failure_count() == 0 &&
              scope.gate_scope().failure_count() == 0);
        if (private_resources) {
            std::uint64_t exhausted = 0;
            for (unsigned index = 3; index < 2 * mib / *native; index += 2) {
                auto input = input_registers();
                input.x[0] = heap + index * *native; input.x[1] = *native; input.x[2] = 1;
                RegisterImage output;
                const auto before = process->query_memory(input.x[0]);
                const auto used = process->used_memory(), total = process->total_memory();
                armigo_test_guest_run_site(&input, &output, base + 8);
                CHECK(preserved_native_state(input, output));
                if (output.x[0] == 0xce01) {
                    exhausted = input.x[0];
                    CHECK(encode_memory_info(process->query_memory(exhausted)) == encode_memory_info(before));
                    CHECK(process->used_memory() == used && process->total_memory() == total);
                    CHECK(!supervisor.memory_failure() && scope.failure_count() == 0);
                    break;
                }
                CHECK(output.x[0] == 0);
            }
            CHECK(exhausted != 0);
            if (exhausted) {
                auto recover = input_registers();
                recover.x[0] = heap + 3 * *native; recover.x[1] = *native; recover.x[2] = 3;
                RegisterImage output;
                armigo_test_guest_run_site(&recover, &output, base + 8);
                CHECK(output.x[0] == 0);
                recover.x[0] = exhausted; recover.x[2] = 1;
                armigo_test_guest_run_site(&recover, &output, base + 8);
                CHECK(output.x[0] == 0 && process->query_memory(exhausted).permissions == 1);
                CHECK(!supervisor.memory_failure() && scope.failure_count() == 0);
            }
        }
        return failures == 0 ? 0 : 1;
    }
}

} // namespace

// CreateTransferMemory locks the owner's read-write range until the last
// reference goes; a full handle table returns 0xd201 and unlocks it again.
void run_transfer_contract() {
    auto handles = std::make_shared<HandleTable>(2);
    auto made = create_guest_memory(profile(), {4 * mib}, nullptr, handles);
    CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(made));
    if (!std::holds_alternative<std::shared_ptr<GuestMemory>>(made)) return;
    auto memory = std::get<std::shared_ptr<GuestMemory>>(std::move(made));
    const auto process = memory->process();
    QueryRegisters heap{}; heap[1] = 2 * mib;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(1, heap)) && heap[0] == 0);
    const auto base = process->heap_region().address;
    const auto transfer = [&](std::uint64_t address, std::uint64_t size, std::uint64_t permission) {
        QueryRegisters r{}; r[1] = address; r[2] = size; r[3] = permission;
        const auto result = memory->dispatch(0x15, r);
        CHECK(std::holds_alternative<QueryDisposition>(result) &&
              std::get<QueryDisposition>(result) == QueryDisposition::handled);
        return r;
    };
    CHECK(transfer(base + 8, 0x1000, 0)[0] == 0xcc01);
    CHECK(transfer(base, 0, 0)[0] == 0xca01);
    CHECK(transfer(base, 0x1000, 2)[0] == 0xd801);
    const auto created = transfer(base, 0x4000, 0);
    CHECK(created[0] == 0 && created[1] != 0);
    CHECK(process->query_memory(base).attributes == 1 && process->query_memory(base).permissions == 3);
    CHECK(transfer(base + 0x1000, 0x1000, 3)[0] == 0xd401);
    // A locked range cannot be reprotected meanwhile.
    QueryRegisters protect{}; protect[0] = base; protect[1] = 0x1000; protect[2] = 1;
    CHECK(std::holds_alternative<QueryDisposition>(memory->dispatch(2, protect)) && protect[0] != 0);
    CHECK(handles->close(static_cast<Handle>(created[1])));
    CHECK(process->query_memory(base).attributes == 0);
    const auto first = handles->insert(std::make_shared<Event>());
    const auto second = handles->insert(std::make_shared<Event>());
    CHECK(first && second);
    CHECK(transfer(base, 0x4000, 1)[0] == 0xd201);
    CHECK(process->query_memory(base).attributes == 0);
}

int main() {
    run_physical_mapping_contract();
    run_application_budget_contract();
    run_remote_window_contract();
    run_transfer_contract();
    const auto legacy = run_dispatch_integration<armigo::runtime::StartupSupervisor>(false);
    const auto private_resources = run_dispatch_integration<armigo::runtime::StartupSupervisor>(true);
    return legacy || private_resources || failures;
}
