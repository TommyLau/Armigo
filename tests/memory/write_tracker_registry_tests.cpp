// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Registry contract test for the optional CPU write stamp surface: a registry
// created with tracking answers cpu_write_stamp()/cpu_written_since() from
// page write generations while its mappings are registered. The backing's
// host pages are a synthetic mmap owned here; native stores in the test fault
// through the production handler, and the copy backend's memcpy exercises the
// registry's own prepared write path.

#include <armigo/graphics/resource_memory_registry.h>
#include <armigo/memory/write_tracker.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <variant>

namespace {

using namespace armigo::graphics;

int failures = 0;
int checks = 0;

#define CHECK(condition)                                                 \
    do {                                                                 \
        ++checks;                                                        \
        if (!(condition)) {                                              \
            ++failures;                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #condition);                                    \
        }                                                                 \
    } while (false)

bool succeeded(const RendererStatus& result) {
    return std::holds_alternative<std::monostate>(result);
}

template <class T>
T require_value(RendererResult<T> result) {
    CHECK(std::holds_alternative<T>(result));
    if (!std::holds_alternative<T>(result)) {
        const auto command = std::get<RendererError>(result).command();
        std::fprintf(stderr, "Cannot continue: %.*s\n",
                     static_cast<int>(command.detail.size()), command.detail.data());
        std::abort();
    }
    return std::get<T>(std::move(result));
}

// One synthetic backing allocation living on dedicated mapped pages so write
// protection cannot disturb unrelated heap memory.
struct HostAllocation {
    void* pages = nullptr;
    std::size_t size = 0;
    explicit HostAllocation(std::size_t bytes)
        : size((bytes + static_cast<std::size_t>(::getpagesize()) - 1) /
               static_cast<std::size_t>(::getpagesize()) *
               static_cast<std::size_t>(::getpagesize())) {
        pages = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON,
                       -1, 0);
        if (pages == MAP_FAILED) {
            std::perror("mmap");
            std::abort();
        }
    }
    ~HostAllocation() { ::munmap(pages, size); }
    std::byte* host() noexcept { return static_cast<std::byte*>(pages); }
};

// Copies bytes against the mapped backing allocation directly. A registry
// write that reaches this memcpy unprepared would fault on protected pages.
struct CopyBackend final : ResourceMemoryBackend {
    std::byte* host = nullptr;
    RendererStatus read_cpu(BackingRange range,
                            std::span<std::byte> destination) override {
        std::memcpy(destination.data(), host + range.offset, destination.size());
        return renderer_success();
    }
    RendererStatus write_cpu(BackingRange range,
                             std::span<const std::byte> source) override {
        std::memcpy(host + range.offset, source.data(), source.size());
        return renderer_success();
    }
    RendererStatus wait_gpu(CompletionToken) override { return renderer_success(); }
};

bool read_only(const void* host) {
    mach_vm_address_t address = reinterpret_cast<mach_vm_address_t>(host);
    const auto requested = address;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info{};
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    const auto result = mach_vm_region(mach_task_self(), &address, &size,
        VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object);
    if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
    return result == KERN_SUCCESS && address <= requested && requested - address < size &&
           info.protection == VM_PROT_READ;
}

// Retains the registration owner identity for the registry's leases.
struct OwnedLease final : ResourceLease {};

void alias_stores_and_retirement() {
    using armigo::memory::WriteTracker;
    const auto page = static_cast<std::uint64_t>(getpagesize());
    HostAllocation allocation(3 * page);
    auto* host = allocation.host();
    const BackingId backing{8, 1, 1};
    const auto owner = std::make_shared<OwnedLease>();
    auto backend = std::make_shared<CopyBackend>();
    backend->host = host;
    ResourceMemoryOptions options;
    options.address_space = 8;
    options.backend = backend;
    options.cpu_write_tracking = true;
    options.host_addressable_backings = true;
    auto registry = require_value(make_resource_memory(options));
    CHECK(succeeded(registry->register_backing({backing, 3 * page, owner})));
    Mapping mapping{page, 3 * page, 0, reinterpret_cast<std::uint64_t>(host),
                    3 * page, page, owner};
    CHECK(succeeded(registry->register_mapping({{8, 1, 1}, backing, mapping})));
    CHECK(WriteTracker::has_writable_alias(host, 3 * page));
    const std::byte payload[4]{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
    // A non-host backend can track this same VA without acquiring permission
    // to bypass its own write callback through another registry's alias.
    {
        HostAllocation shadow(3 * page);
        auto shadow_backend = std::make_shared<CopyBackend>();
        shadow_backend->host = shadow.host();
        auto shadow_options = options;
        shadow_options.address_space = 9;
        shadow_options.host_addressable_backings = false;
        shadow_options.backend = shadow_backend;
        auto other = require_value(make_resource_memory(shadow_options));
        CHECK(succeeded(other->register_backing({backing, 3 * page, owner})));
        CHECK(succeeded(other->register_mapping({{9, 1, 1}, backing, mapping})));
        CHECK(other->address_space().write(page + 7, payload).ok());
        CHECK(std::memcmp(shadow.host() + 7, payload, sizeof(payload)) == 0);
        CHECK(host[7] == std::byte{});
    }
    CHECK(WriteTracker::has_writable_alias(host, 3 * page));
    CHECK(WriteTracker::protect_checked(host, 3 * page));
    const auto stamp = registry->cpu_write_stamp();
    const auto before = WriteTracker::stats();
    CHECK(registry->address_space().write(page + 7, payload).ok());
    CHECK(std::memcmp(host + 7, payload, sizeof(payload)) == 0);
    CHECK(registry->cpu_written_since({backing, 7, 4}, stamp));
    CHECK(!registry->cpu_written_since({backing, page, page}, stamp));
    CHECK(read_only(host));
    {
        auto read = require_value(registry->prepare_cpu_read({backing, 0, page}, ResolveReason::CpuRead));
    }
    const auto after = WriteTracker::stats();
    CHECK(after.protect_calls == before.protect_calls);
    CHECK(after.prepared == before.prepared && after.faults == before.faults);
    const auto guest_stamp = registry->cpu_write_stamp();
    static_cast<volatile std::byte*>(allocation.pages)[12] = std::byte{0x75};
    CHECK(WriteTracker::stats().faults == after.faults + 1);
    CHECK(registry->cpu_written_since({backing, 0, page}, guest_stamp));
    CHECK(!registry->cpu_written_since({backing, page, page}, guest_stamp));
    CHECK(WriteTracker::protect_checked(host, 3 * page));
    const auto crossing_stamp = registry->cpu_write_stamp();
    CHECK(registry->address_space().write(2 * page - 2, payload).ok());
    CHECK(registry->cpu_written_since({backing, 0, page}, crossing_stamp));
    CHECK(registry->cpu_written_since({backing, page, page}, crossing_stamp));
    CHECK(!registry->cpu_written_since({backing, 2 * page, page}, crossing_stamp));
    CHECK(read_only(host) && read_only(host + page));
    CHECK(succeeded(registry->unmap({8, 1, 1})));
    CHECK(!WriteTracker::has_writable_alias(host, 1));
    CHECK(!WriteTracker::try_alias_store(host, payload));
    // Replace the original physical pages only after the registry drops its
    // alias, then reuse the guest VA under a fresh mapping generation.
    CHECK(mmap(host, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_FIXED,
               -1, 0) == host);
    CHECK(host[7] == std::byte{});
    CHECK(succeeded(registry->register_mapping({{8, 1, 2}, backing, mapping})));
    CHECK(registry->address_space().write(page + 7, payload).ok());
    CHECK(std::memcmp(host + 7, payload, sizeof(payload)) == 0);
    CHECK(read_only(host));
    registry.reset();
    CHECK(!WriteTracker::has_writable_alias(host, 1));
    CHECK(!WriteTracker::tracked(host, 1));
    CHECK(!WriteTracker::try_alias_store(host, payload));
}

} // namespace

int main() {
    using armigo::memory::WriteTracker;
    alias_stores_and_retirement();
    const auto page = static_cast<std::uint64_t>(::getpagesize());
    HostAllocation allocation{static_cast<std::size_t>(2 * page)};

    // Without tracking the registry keeps the conservative contract defaults.
    {
        ResourceMemoryOptions plain;
        plain.address_space = 1;
        auto backend = std::make_shared<CopyBackend>();
        backend->host = allocation.host();
        plain.backend = backend;
        auto registry = require_value(make_resource_memory(std::move(plain)));
        CHECK(registry->cpu_write_stamp() == 0);
        CHECK(registry->cpu_written_since({{2, 2, 2}, 0, page}, registry->cpu_write_stamp()));
    }

    ResourceMemoryOptions options;
    options.address_space = 1;
    auto backend = std::make_shared<CopyBackend>();
    backend->host = allocation.host();
    options.backend = backend;
    options.cpu_write_tracking = true;
    auto registry = require_value(make_resource_memory(std::move(options)));

    const BackingId backing{1, 1, 1};
    const auto owner = std::make_shared<OwnedLease>();
    CHECK(succeeded(registry->register_backing({backing, 2 * page, owner})));
    const Mapping mapping{page, 2 * page, 0,
                          reinterpret_cast<std::uint64_t>(allocation.pages),
                          2 * page, static_cast<std::uint64_t>(::getpagesize()), owner};
    const MappingId mapping_id{1, 1, 1};
    CHECK(succeeded(registry->register_mapping({mapping_id, backing, mapping})));

    // Stamp, cache-style read, CPU store through the tracked range.
    const auto stamp = registry->cpu_write_stamp();
    CHECK(stamp != 0);
    {
        auto read = require_value(
            registry->prepare_cpu_read({backing, 0, page}, ResolveReason::CpuRead));
        static_cast<void>(read);
    }
    // No store since the stamp: the covered page stays clean.
    CHECK(!registry->cpu_written_since({backing, 0, page}, stamp));
    // A native store into the tracked page faults once and is observed.
    static_cast<volatile std::byte*>(allocation.pages)[0] = std::byte{0x5a};
    CHECK(registry->cpu_written_since({backing, 0, page}, stamp));
    // The second page was never prepared for a cache read and holds no store.
    CHECK(!registry->cpu_written_since({backing, page, page}, stamp));

    // The registry's own CPU write path must not fault on protected pages and
    // publishes the write to the stamp contract.
    const auto write_stamp = registry->cpu_write_stamp();
    const std::byte pattern[4] = {std::byte{1}, std::byte{2}, std::byte{3},
                                  std::byte{4}};
    {
        auto write =
            require_value(registry->prepare_cpu_write({backing, page, 16}));
        std::memcpy(allocation.host() + page + 4, pattern, sizeof(pattern));
        write->commit();
    }
    CHECK(registry->cpu_written_since({backing, page, 16}, write_stamp));

    // A fresh cache read re-arms protection; without further stores the range
    // is clean again, and an untracked or zero-stamp query stays true.
    const auto clean_stamp = registry->cpu_write_stamp();
    {
        auto reconfirm = require_value(
            registry->prepare_cpu_read({backing, 0, 2 * page}, ResolveReason::CpuRead));
        static_cast<void>(reconfirm);
    }
    CHECK(!registry->cpu_written_since({backing, 0, 2 * page}, clean_stamp));
    CHECK(registry->cpu_written_since({backing, 0, 2 * page}, 0));
    CHECK(registry->cpu_written_since({{9, 9, 9}, 0, page}, clean_stamp));

    // Unmapping drops the tracking: the stale stamp can no longer be vouched.
    CHECK(succeeded(registry->unmap(mapping_id)));
    CHECK(registry->cpu_written_since({backing, 0, 2 * page}, clean_stamp));
    CHECK(succeeded(registry->retire_backing(backing)));

    // The registry released every tracked range with its registrations.
    const auto stats = WriteTracker::stats();
    CHECK(stats.live_ranges == 0);
    CHECK(stats.canonical_pages == 0);
    CHECK(stats.shared_references == 0);

    std::printf("write_tracker_registry_tests: %d checks, %d failed\n", checks,
                failures);
    return failures != 0;
}
