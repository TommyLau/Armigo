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

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <latch>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace {
using namespace armigo;
using namespace armigo::runtime;

constexpr std::uint64_t mib = 1024 * 1024;
bool private_mode = false;
constexpr auto rw = memory::Permission::read | memory::Permission::write;

std::atomic_uint failures{0};
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); \
    failures.fetch_add(1, std::memory_order_relaxed); \
} } while (false)

std::uint64_t native_page_size() {
    const long value = ::sysconf(_SC_PAGESIZE);
    CHECK(value > 0);
    return value > 0 ? static_cast<std::uint64_t>(value)
                     : horizon::guest_page_size;
}

#if defined(__APPLE__)
vm_prot_t native_protection(std::uint64_t address) {
    mach_vm_address_t region_address = address;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = ::mach_vm_region(
        mach_task_self(), &region_address, &region_size, VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) {
        ::mach_port_deallocate(mach_task_self(), object);
    }
    CHECK(result == KERN_SUCCESS && region_address <= address &&
          address - region_address < region_size);
    return result == KERN_SUCCESS ? info.protection : VM_PROT_NONE;
}
#endif

LaunchProfile synthetic_profile() {
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.process.allowed_core_mask = 1;
    profile.process.heap_address = 0;
    profile.process.alias_address = 0;
    profile.process.stack_address = 0;
    profile.process.heap_size = 8 * mib;
    profile.process.base_alias_size = 2 * mib;
    profile.process.stack_size = 2 * mib;
    profile.permissions.handle_table_capacity = 64;
    if (private_mode) {
        profile.process.platform = GuestPlatform::switch2;
        profile.process.baseline = horizon::Baseline::horizon_23_0_1;
        profile.process.private_system_resource_bytes = 65536;
    }
    return profile;
}

struct Fixture {
    std::shared_ptr<horizon::HandleTable> handles =
        std::make_shared<horizon::HandleTable>(64);
    std::shared_ptr<GuestMemory> memory;

    Fixture() {
        auto created = create_guest_memory(synthetic_profile(), {16 * mib}, nullptr, handles);
        CHECK(std::holds_alternative<std::shared_ptr<GuestMemory>>(created));
        if (auto* value = std::get_if<std::shared_ptr<GuestMemory>>(&created)) {
            memory = std::move(*value);
        }
    }
};

bool handled(const LaunchResult<horizon::QueryDisposition>& result) {
    const auto* disposition = std::get_if<horizon::QueryDisposition>(&result);
    return disposition != nullptr &&
           *disposition == horizon::QueryDisposition::handled;
}

bool unsupported_failure(const LaunchResult<horizon::QueryDisposition>& result) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure != nullptr && failure->stage == LaunchStage::memory &&
           failure->code == LaunchError::unsupported;
}

GuestMemoryPin take_pin(LaunchResult<GuestMemoryPin> result) {
    CHECK(std::holds_alternative<GuestMemoryPin>(result));
    return std::holds_alternative<GuestMemoryPin>(result)
               ? std::get<GuestMemoryPin>(std::move(result))
               : GuestMemoryPin{};
}

void test_pin_and_protect_race() {
    constexpr std::size_t iterations = 64;
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    auto initial = take_pin(fixture.memory->allocate(
        page, horizon::MemoryType::normal, rw));
    if (!initial) return;
    const auto range = initial->range();
    initial.reset();

    std::barrier phase(2);
    std::thread pinner([&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            auto pin = take_pin(fixture.memory->pin(range, rw));
            phase.arrive_and_wait();
            phase.arrive_and_wait();
            if (pin) {
                CHECK(pin->permissions() == rw && pin->bytes().size() == page &&
                      pin->writable_bytes().size() == page);
#if defined(__APPLE__)
                CHECK(native_protection(range.address) ==
                      (VM_PROT_READ | VM_PROT_WRITE));
#endif
            }
            pin.reset();
            phase.arrive_and_wait();
            phase.arrive_and_wait();
        }
    });
    std::thread mutator([&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            phase.arrive_and_wait();
            horizon::QueryRegisters protect{};
            protect[0] = range.address;
            protect[1] = range.size;
            protect[2] = 1;
            const auto unchanged = protect;
            CHECK(unsupported_failure(fixture.memory->dispatch(2, protect)));
            CHECK(protect == unchanged);
#if defined(__APPLE__)
            CHECK(native_protection(range.address) ==
                  (VM_PROT_READ | VM_PROT_WRITE));
#endif
            phase.arrive_and_wait();
            phase.arrive_and_wait();

            protect = {};
            protect[0] = range.address;
            protect[1] = range.size;
            protect[2] = 1;
            CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
#if defined(__APPLE__)
            CHECK(native_protection(range.address) == VM_PROT_READ);
#endif
            protect = {};
            protect[0] = range.address;
            protect[1] = range.size;
            protect[2] = 3;
            CHECK(handled(fixture.memory->dispatch(2, protect)) && protect[0] == 0);
#if defined(__APPLE__)
            CHECK(native_protection(range.address) ==
                  (VM_PROT_READ | VM_PROT_WRITE));
#endif
            phase.arrive_and_wait();
        }
    });
    pinner.join();
    mutator.join();
}

void test_pin_and_heap_resize_race() {
    constexpr std::size_t iterations = 12;
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    horizon::QueryRegisters grow{};
    grow[1] = 2 * mib;
    CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0);
    const horizon::AddressRange pinned_range{grow[1], page};

    std::barrier phase(2);
    std::thread pinner([&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            auto pin = take_pin(fixture.memory->pin(pinned_range, rw));
            phase.arrive_and_wait();
            phase.arrive_and_wait();
            if (pin) {
                CHECK(pin->bytes().size() == page &&
                      pin->bytes().front() == std::byte{0});
#if defined(__APPLE__)
                CHECK(native_protection(pinned_range.address) ==
                      (VM_PROT_READ | VM_PROT_WRITE));
#endif
            }
            pin.reset();
            phase.arrive_and_wait();
            phase.arrive_and_wait();
        }
    });
    std::thread resizer([&] {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            phase.arrive_and_wait();
            horizon::QueryRegisters shrink{};
            shrink[1] = 0;
            const auto unchanged = shrink;
            CHECK(unsupported_failure(fixture.memory->dispatch(1, shrink)));
            CHECK(shrink == unchanged);
            CHECK(fixture.memory->process()->used_memory() == 2 * mib);
            phase.arrive_and_wait();
            phase.arrive_and_wait();

            shrink = {};
            shrink[1] = 0;
            CHECK(handled(fixture.memory->dispatch(1, shrink)) && shrink[0] == 0);
            CHECK(fixture.memory->process()->used_memory() == 0 &&
                  fixture.memory->process()->query_memory(pinned_range.address).type ==
                      horizon::MemoryType::free);
#if defined(__APPLE__)
            CHECK(native_protection(pinned_range.address) == VM_PROT_NONE);
#endif
            grow = {};
            grow[1] = 2 * mib;
            CHECK(handled(fixture.memory->dispatch(1, grow)) && grow[0] == 0 &&
                  grow[1] == pinned_range.address);
            auto zeroed = take_pin(fixture.memory->pin(pinned_range, rw));
            if (zeroed) CHECK(zeroed->bytes().front() == std::byte{0});
            zeroed.reset();
#if defined(__APPLE__)
            CHECK(native_protection(pinned_range.address) ==
                  (VM_PROT_READ | VM_PROT_WRITE));
#endif
            phase.arrive_and_wait();
        }
    });
    pinner.join();
    resizer.join();
}

void test_service_mapper_lifetime() {
    const auto page = native_page_size();
    Fixture fixture;
    if (!fixture.memory) return;
    auto allocation = take_pin(fixture.memory->allocate(
        page, horizon::MemoryType::normal, rw));
    if (!allocation) return;
    const auto range = allocation->range();
    allocation.reset();

    auto mapper = fixture.memory->service_mapper();
    CHECK(mapper.map != nullptr && mapper.unmap != nullptr &&
          mapper.context != nullptr && mapper.lifetime != nullptr);
    if (!mapper.map || !mapper.unmap || !mapper.context || !mapper.lifetime) return;
    CHECK(mapper.map(mapper.context, range.address, range.size));

    std::weak_ptr<GuestMemory> memory_owner = fixture.memory;
    std::weak_ptr<horizon::HandleTable> handle_owner = fixture.handles;
    fixture.memory.reset();
    fixture.handles.reset();
    CHECK(memory_owner.expired());
    CHECK(handle_owner.expired());

    mapper.unmap(mapper.context, range.address, range.size);
    mapper.lifetime.reset();
    mapper.context = nullptr;
}

template <class Process>
constexpr bool supports_atomic_publication = requires(
    Process& process, std::span<const horizon::MemoryBlock> blocks,
    const horizon::MemoryAccounting& accounting) {
    process.commit_memory_state(
        blocks, accounting, static_cast<bool (*)(void*) noexcept>(nullptr), nullptr);
};

[[maybe_unused]] horizon::ProcessConfiguration record_configuration() {
    horizon::ProcessConfiguration configuration;
    configuration.heap_address = 0x1000000000;
    configuration.alias_address = 0x2000000000;
    configuration.stack_address = 0x6000000000;
    configuration.allowed_core_mask = 1;
    return configuration;
}

struct BlockingPublication {
    std::barrier<> gate{2};
    std::promise<void> entered;
    std::atomic_bool native_applied{false};
    bool accept = true;

    static bool apply(void* context) noexcept {
        auto& self = *static_cast<BlockingPublication*>(context);
        self.entered.set_value();
        self.gate.arrive_and_wait();
        self.gate.arrive_and_wait();
        self.native_applied.store(self.accept, std::memory_order_release);
        return self.accept;
    }
};

[[maybe_unused]] bool reject_publication(void*) noexcept { return false; }

template <class Process>
void test_process_record_atomic_publication() {
    if constexpr (!supports_atomic_publication<Process>) {
        std::puts("SKIP: ProcessRecord::commit_memory_state is not present in frozen headers.");
    } else {
        using namespace std::chrono_literals;
        const auto configuration = record_configuration();
        horizon::MemoryAccounting baseline{};
        baseline.resource_limit_bytes = mib;
        auto built = Process::create(configuration, baseline, {});
        CHECK(built.ok());
        if (!built.ok()) return;
        auto record = std::move(built.process);

        const std::vector<horizon::MemoryBlock> blocks = {{
            {configuration.heap_address, native_page_size()},
            horizon::MemoryType::normal, 0, 3}};
        auto updated = baseline;
        updated.committed_heap = native_page_size();
        updated.resource_limit_used = native_page_size();

        BlockingPublication callback;
        auto callback_entered = callback.entered.get_future();
        std::atomic_int commit_result{
            static_cast<int>(horizon::ProcessStateError::resource_limit)};
        std::thread committer([&] {
            const auto result = record->commit_memory_state(
                blocks, updated, BlockingPublication::apply, &callback);
            commit_result.store(static_cast<int>(result), std::memory_order_release);
        });
        if (callback_entered.wait_for(2s) != std::future_status::ready) {
            std::fputs("Timed out waiting for ProcessRecord publication callback.\n", stderr);
            std::_Exit(1);
        }
        callback.gate.arrive_and_wait();

        struct Observation {
            horizon::MemoryInfo info;
            std::uint64_t used = 0;
            bool native_applied = false;
        };
        std::promise<Observation> observed;
        auto observation = observed.get_future();
        std::latch observer_started{1};
        std::thread observer([&] {
            observer_started.count_down();
            const auto info = record->query_memory(configuration.heap_address);
            const auto used = record->used_memory();
            observed.set_value({info, used,
                callback.native_applied.load(std::memory_order_acquire)});
        });
        observer_started.wait();
        CHECK(observation.wait_for(25ms) == std::future_status::timeout);
        callback.gate.arrive_and_wait();
        if (observation.wait_for(2s) != std::future_status::ready) {
            std::fputs("Timed out waiting for ProcessRecord query observer.\n", stderr);
            std::_Exit(1);
        }
        committer.join();
        observer.join();
        CHECK(commit_result.load(std::memory_order_acquire) ==
              static_cast<int>(horizon::ProcessStateError::none));
        const auto result = observation.get();
        CHECK(result.native_applied && result.used == native_page_size() &&
              result.info.type == horizon::MemoryType::normal &&
              result.info.permissions == 3);

        auto failed = Process::create(configuration, baseline, {});
        CHECK(failed.ok());
        if (!failed.ok()) return;
        const auto failure = failed.process->commit_memory_state(
            blocks, updated, reject_publication, nullptr);
        CHECK(failure != horizon::ProcessStateError::none);
        CHECK(failed.process->used_memory() == 0 &&
              failed.process->query_memory(configuration.heap_address).type ==
                  horizon::MemoryType::free &&
              failed.process->memory_map().empty());
    }
}

void test_private_publication_observer() {
    using namespace std::chrono_literals;
    for (bool accept : {false, true}) {
        auto configuration = record_configuration();
        configuration.platform = GuestPlatform::switch2;
        configuration.baseline = horizon::Baseline::horizon_23_0_1;
        configuration.private_system_resource_bytes = 65536;
        horizon::MemoryAccounting baseline{};
        baseline.resource_limit_bytes = mib; baseline.resource_limit_used = 65536;
        auto built = horizon::ProcessRecord::create(configuration, baseline);
        CHECK(built.ok()); if (!built.ok()) return;
        auto record = std::move(built.process);
        const horizon::MemoryMapUpdate update{{configuration.heap_address, native_page_size()},
            horizon::MemoryBlock{{}, horizon::MemoryType::normal, 0, 3}};
        auto changed = baseline;
        changed.committed_heap = native_page_size(); changed.resource_limit_used += native_page_size();
        BlockingPublication callback; callback.accept = accept;
        auto entered = callback.entered.get_future();
        horizon::ProcessStateError committed{};
        std::thread committer([&] {
            committed = record->commit_memory_updates(std::span(&update, 1), changed,
                BlockingPublication::apply, &callback);
        });
        if (entered.wait_for(2s) != std::future_status::ready) std::_Exit(1);
        callback.gate.arrive_and_wait();
        std::promise<std::pair<horizon::MemoryInfo, std::uint64_t>> observed;
        auto future = observed.get_future();
        std::latch started{1};
        std::thread observer([&] {
            started.count_down();
            observed.set_value({record->query_memory(configuration.heap_address), record->used_memory()});
        });
        started.wait();
        CHECK(future.wait_for(25ms) == std::future_status::timeout);
        CHECK(record->used_system_resource() == 4096);
        callback.gate.arrive_and_wait();
        if (future.wait_for(2s) != std::future_status::ready) std::_Exit(1);
        committer.join(); observer.join();
        const auto result = future.get();
        CHECK(committed == (accept ? horizon::ProcessStateError::none : horizon::ProcessStateError::backend_failure));
        CHECK(result.first.type == (accept ? horizon::MemoryType::normal : horizon::MemoryType::free));
        CHECK(result.second == (accept ? native_page_size() : 0));
        CHECK(record->total_memory() == mib - 65536);
    }
}

} // namespace

int main() {
    for (bool mode : {false, true}) {
        private_mode = mode;
        test_pin_and_protect_race();
        test_pin_and_heap_resize_race();
        test_service_mapper_lifetime();
    }
    test_process_record_atomic_publication<horizon::ProcessRecord>();
    test_private_publication_observer();
    if (failures.load(std::memory_order_relaxed) != 0) return 1;
    std::puts("Guest memory pin, mutation, and mapper concurrency checks passed.");
    return 0;
}
