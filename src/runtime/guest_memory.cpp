// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Writable pins prepare their host pages through the page write tracker.
// A syscall writing into a protected tracked page fails with EFAULT instead
// of faulting, so every host write
// into guest memory prepares tracked pages first. Bounded byte copies use a
// registered writable alias when available and publish after the copy.

#include <armigo/runtime/guest_memory.h>
#include <armigo/horizon/read_only_shared_memory.h>
#include <armigo/horizon/shared_memory.h>
#include <armigo/horizon/stack_alias.h>
#include <armigo/memory/write_tracker.h>
#include "../horizon/process_memory.h"
#include "../platform/process_memory.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace armigo::runtime {
namespace {
using namespace horizon;
namespace pm = horizon::process_memory;
using memory::Permission;
using Region = platform::ProcessMemoryRegion;
using NativeStatus = platform::ProcessMemoryStatus;
constexpr auto rw = Permission::read | Permission::write;

LaunchFailure failure(LaunchError code, const char* operation, const char* detail) {
    return {.stage = LaunchStage::memory, .code = code, .operation = operation, .detail = detail};
}
LaunchFailure unsupported(const char* operation, const char* detail) {
    return failure(LaunchError::unsupported, operation, detail);
}
LaunchFailure host_failure(const char* operation) {
    return failure(LaunchError::host_failure, operation, "Native memory transaction failed.");
}
bool native_range(AddressRange r) noexcept {
    const auto page = platform::native_page_size();
    return page && pm::valid_range(r) && !(r.address % page) && !(r.size % page);
}
bool native_source(std::uint64_t address, std::uint64_t size) noexcept {
    const auto page = platform::native_page_size();
    return page && size && !(address % page) && !(size % page) &&
        size <= std::numeric_limits<std::uint64_t>::max() - address;
}
AddressRange range_of(const Region& r) noexcept { return {r.address(), r.size()}; }
// The native pages holding a guest range. A host page larger than the guest
// page can hold several guest mappings; it is committed while any of them
// uses it and keeps the union of their permissions.
AddressRange native_pages(AddressRange r) noexcept {
    const auto page = platform::native_page_size();
    const auto first = r.address & ~(page - 1);
    return {first, ((r.address + r.size + page - 1) & ~(page - 1)) - first};
}
// Whether a mapped guest block other than `except` lies in the native page at `at`.
bool native_page_used(std::span<const MemoryBlock> blocks, std::uint64_t at, AddressRange except,
                      bool* writable = nullptr) noexcept {
    const AddressRange page{at, platform::native_page_size()};
    bool used = false;
    for (const auto& block : blocks) {
        if (block.type == MemoryType::free || !pm::overlaps(block.range, page)) continue;
        if (pm::contains(except, block.range)) continue;
        used = true;
        if (writable && (block.permissions & 2)) *writable = true;
    }
    return used;
}
bool same_range(AddressRange a, AddressRange b) noexcept {
    return a.address == b.address && a.size == b.size;
}
QueryDisposition finish(QueryRegisters& r, std::uint32_t result,
                         std::uint64_t value = 0) noexcept {
    r[0] = result;
    r[1] = value;
    std::fill(r.begin() + 2, r.end(), 0);
    return QueryDisposition::handled;
}

struct State;
class Pin final : public PinnedGuestMemory {
public:
    Pin(std::shared_ptr<State> owner, AddressRange range, Permission access)
        : owner_(std::move(owner)), range_(range), access_(access) {}
    AddressRange range() const noexcept override { return range_; }
    Permission permissions() const noexcept override { return access_; }
    std::span<const std::byte> bytes() const noexcept override {
        return {reinterpret_cast<const std::byte*>(range_.address),
                static_cast<std::size_t>(range_.size)};
    }
    std::span<std::byte> writable_bytes() noexcept override {
        if (!memory::has_permission(access_, Permission::write)) return {};
        return {reinterpret_cast<std::byte*>(range_.address),
                static_cast<std::size_t>(range_.size)};
    }
private:
    // State does not retain pins or the handle table. Destruction cannot call a
    // service, and retaining backing cannot form a process/service cycle.
    std::shared_ptr<State> owner_;
    AddressRange range_;
    Permission access_;
};
struct Allocation {
    AddressRange range;
    MemoryType type;
    std::shared_ptr<Region> region;
};
struct SharedMapping {
    AddressRange range;
    std::shared_ptr<KernelObject> object;
    std::shared_ptr<Region> region;
    Permission access;
    // The native pages a writable object's sub-page mapping aliases whole;
    // empty for other mappings.
    AddressRange envelope{};
};
struct State : std::enable_shared_from_this<State> {
    std::mutex mutex;
    std::shared_ptr<ProcessRecord> process;
    std::shared_ptr<ModuleSet> modules;
    std::shared_ptr<Region> heap, alias, stack;
    MemoryAccounting accounting;
    std::vector<MemoryBlock> blocks;
    std::vector<Allocation> allocations;
    std::vector<SharedMapping> shared;
    std::vector<StackAliasRecord> aliases;
    std::vector<std::weak_ptr<Pin>> pins;
    bool poisoned = false;

    // Whether native pages in `range` alias a writable object's sub-page
    // mapping. Those pages hold the object's neighbouring bytes, so no other
    // guest mapping may commit, copy into or protect them.
    bool envelope_used(AddressRange range) const noexcept {
        return std::any_of(shared.begin(), shared.end(), [&](const auto& m) {
            return m.envelope.size && pm::overlaps(m.envelope, range);
        });
    }
    bool pinned(AddressRange range) {
        std::erase_if(pins, [](const auto& p) { return p.expired(); });
        for (const auto& weak : pins)
            if (const auto p = weak.lock(); p && pm::overlaps(range, p->range())) return true;
        return false;
    }
    GuestMemoryPin make_pin(AddressRange range, Permission access, bool prepare_write = true) {
        std::erase_if(pins, [](const auto& p) { return p.expired(); });
        auto result = std::make_shared<Pin>(shared_from_this(), range, access);
        pins.emplace_back(result);
        // Every host write into guest memory flows through a writable pin:
        // IPC buffer writes, file-system reads into guest buffers, service
        // mapper pins, and ordered word copies. A syscall writing into a
        // protected tracked page fails with EFAULT instead of faulting, so
        // writable pins normally unprotect pages before any byte moves. A
        // bounded internal copy can defer preparation until alias lookup.
        if (prepare_write && memory::has_permission(access, Permission::write))
            memory::WriteTracker::prepare_host_write(
                reinterpret_cast<const void*>(range.address),
                static_cast<std::size_t>(range.size));
        return result;
    }
    // One short native access. It validates exactly as pin() does and then performs the
    // access under the same lock instead of creating, registering and later expiring a pin: a
    // concurrent unmap or permission change waits for the access and so cannot interleave with it,
    // which is the protection a pin gives for the duration of this call.
    template<class Access>
    bool with_word(std::uint64_t address, std::size_t size, Permission access, bool prepare_write,
                   Access&& body) noexcept {
        try {
            std::lock_guard lock(mutex);
            if (poisoned) return false;
            if (!memory::has_known_permission_bits(access) ||
                !memory::has_permission(access, Permission::read) ||
                (memory::has_permission(access, Permission::write) &&
                 memory::has_permission(access, Permission::execute)))
                return false;
            const AddressRange range{address, size};
            if (!pm::covered(blocks, range, static_cast<std::uint8_t>(access))) return false;
            if (prepare_write && memory::has_permission(access, Permission::write))
                memory::WriteTracker::prepare_host_write(reinterpret_cast<const void*>(address), size);
            body();
            return true;
        } catch (...) { return false; }
    }
    LaunchResult<GuestMemoryPin> pin(AddressRange range, Permission access, bool prepare_write = true) {
        std::lock_guard lock(mutex);
        if (poisoned) return host_failure("pin");
        if (!memory::has_known_permission_bits(access) ||
            !memory::has_permission(access, Permission::read) ||
            (memory::has_permission(access, Permission::write) &&
             memory::has_permission(access, Permission::execute)))
            return failure(LaunchError::invalid_input, "pin", "Pin access requires readable non-WX memory.");
        if (!pm::covered(blocks, range, static_cast<std::uint8_t>(access)))
            return failure(LaunchError::guest_fault, "pin", "The whole range is not mapped with the required permissions.");
        return make_pin(range, access, prepare_write);
    }
    // The free part of the process address space: guest-chosen mappings land
    // here, inside host reservations no other host allocation can take.
    std::shared_ptr<Region> mappable;
    std::shared_ptr<Region> region(AddressRange range) const {
        for (const auto& r : {heap, alias, stack, mappable})
            if (r && pm::contains(range_of(*r), range)) return r;
        for (const auto& a : allocations)
            if (pm::contains(range_of(*a.region), range)) return a.region;
        for (const auto& m : shared)
            if (pm::contains(range_of(*m.region), range)) return m.region;
        return {};
    }
    void restored(NativeStatus status) noexcept { if (!status.ok()) poisoned = true; }

    template<class Process, class Apply, class Undo>
    LaunchStatus commit(Process& record, std::vector<MemoryBlock>& next,
        MemoryAccounting updated, const char* operation, Apply&& apply, Undo&& undo) {
        // The integrator-owned ProcessRecord patch makes native publication and
        // both query views one critical section. Keep this capability check so
        // lane C can still be built against the frozen pre-integration headers.
        if constexpr (requires {
            record.commit_memory_state(next, updated,
                static_cast<bool (*)(void*) noexcept>(nullptr), nullptr);
        }) {
            struct Callback {
                std::remove_reference_t<Apply>* apply;
                bool failed = false;
                static bool invoke(void* context) noexcept {
                    auto& self = *static_cast<Callback*>(context);
                    self.failed = !(*self.apply)().ok();
                    return !self.failed;
                }
            } callback{&apply};
            const auto result = record.commit_memory_state(next, updated, Callback::invoke, &callback);
            if (callback.failed) return host_failure(operation);
            if (result != ProcessStateError::none)
                return failure(LaunchError::resource_limit, operation, "Memory state admission failed before native publication.");
            return std::monostate{};
        }
        // Validate every fallible metadata constraint before touching native pages.
        // The existing ProcessRecord publication API allocates again; retain an
        // undo mapping until that publication has succeeded. This synthetic-only
        // fallback requires serialized direct ProcessRecord observers.
        const auto checked = ProcessRecord::create(record.configuration(), updated, next);
        if (!checked.ok())
            return failure(LaunchError::resource_limit, operation, "Memory metadata or accounting admission failed.");
        if (!apply().ok()) return host_failure(operation);
        if (record.replace_memory_map(next) != ProcessStateError::none) {
            restored(undo());
            if (poisoned) return failure(LaunchError::host_failure, operation,
                "Native rollback failed; the backing authority is no longer usable.");
            return failure(LaunchError::resource_limit, operation, "Memory metadata publication failed; native rollback attempted.");
        }
        // Already validated above, and this operation does not allocate.
        if (record.set_accounting(updated) != ProcessStateError::none) {
            poisoned = true;
            return host_failure(operation);
        }
        return std::monostate{};
    }
    template<class Apply, class Undo>
    LaunchStatus transaction(std::vector<MemoryBlock> next, MemoryAccounting updated,
        const char* operation, Apply&& apply, Undo&& undo) {
        auto result = commit(*process, next, updated, operation,
                             std::forward<Apply>(apply), std::forward<Undo>(undo));
        if (std::holds_alternative<std::monostate>(result)) {
            blocks.swap(next);
            accounting = updated;
        }
        return result;
    }
};

template<class Process>
constexpr bool atomic_record_publication() {
    return requires(Process& record, std::span<const MemoryBlock> blocks, MemoryAccounting accounting) {
        record.commit_memory_state(blocks, accounting,
            static_cast<bool (*)(void*) noexcept>(nullptr), nullptr);
    };
}

template<class Mapping>
LaunchStatus register_module(State& state, const Mapping& mapping,
    std::optional<linker::ModuleRange> appended, std::uint64_t budget,
    bool original_process) {
    if constexpr (!requires { mapping.guest_regions(); }) {
        return unsupported("create_guest_memory modules",
            "Module admission requires the integrator's declared-region accessor.");
    } else {
        const auto declared = mapping.guest_regions();
        const auto page = mapping.host_page_size();
        // A native page larger than the guest page may join segments with
        // different permissions; the host then maps the union and the guest
        // memory map keeps each segment's own permissions. A native page must
        // never lack a permission one of its guest pages declares, except
        // execute in a module whose code the host never runs.
        // ModuleSet remains the sole mapping owner.
        bool host_executes = true;
        if constexpr (requires { mapping.host_executes(); }) host_executes = mapping.host_executes();
        for (const auto& native : mapping.host_regions()) {
            for (auto offset = native.offset; offset < native.offset + native.size; offset += page) {
                for (const auto& guest : declared) {
                    const auto begin = std::max(offset, guest.offset);
                    const auto end = std::min(offset + page, guest.offset + guest.size);
                    if (begin >= end) continue;
                    auto wanted = static_cast<std::uint8_t>(guest.permissions);
                    if (!host_executes) wanted &= static_cast<std::uint8_t>(~static_cast<std::uint8_t>(memory::Permission::execute));
                    if ((static_cast<std::uint8_t>(native.permissions) & wanted) != wanted)
                        return unsupported("create_guest_memory modules", "A native module page lacks a declared guest permission.");
                }
            }
        }
        const auto guest_bytes = appended ? appended->offset : mapping.mapping_size();
        if (guest_bytes > budget - state.accounting.code_bytes)
            return failure(LaunchError::resource_limit, "create_guest_memory modules", "Module pages exceed the process budget.");
        state.accounting.code_bytes += guest_bytes;
        for (const auto& region : declared) {
            if (!region.size) continue;
            if ((region.offset | region.size) % guest_page_size)
                return unsupported("create_guest_memory modules", "Module regions must contain complete guest pages.");
            const auto permissions = static_cast<std::uint8_t>(region.permissions);
            // Preparation storage belongs to this module but is not another
            // original executable image for the guest dynamic loader to scan.
            const bool preparation = original_process && appended &&
                                     region.offset >= appended->offset;
            state.blocks.push_back({{mapping.base_address() + region.offset, region.size},
                (preparation || (permissions & 2)) ? MemoryType::code_data : MemoryType::code, 0, permissions,
                0, 0, static_cast<std::uint8_t>(region.offset == 0 ? pm::merge_disable_mapping : 0)});
        }
        return std::monostate{};
    }
}

// All callbacks validate/pin before access. Pins protect mappings, not bytes;
// synchronization words use native atomics on the exact guest backing.
GuestMemoryPin checked_pin(State& state, std::uint64_t address, std::size_t size,
                           Permission access, bool prepare_write = true) noexcept {
    try {
        auto result = state.pin({address, size}, access, prepare_write);
        if (auto* p = std::get_if<GuestMemoryPin>(&result)) return std::move(*p);
    } catch (...) {}
    return {};
}
bool read_bytes(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
    if (out.empty()) return true;
    auto pin = checked_pin(*static_cast<State*>(context), address, out.size(), Permission::read);
    if (!pin) return false;
    std::memcpy(out.data(), pin->bytes().data(), out.size());
    return true;
}
bool write_bytes(void* context, std::uint64_t address, std::span<const std::byte> in) noexcept {
    if (in.empty()) return true;
    // Validate permissions and retain the native allocation before looking up
    // its alias. The tracker holds its own reader epoch through publication.
    auto pin = checked_pin(*static_cast<State*>(context), address, in.size(), rw, false);
    if (!pin) return false;
    auto* host = pin->writable_bytes().data();
    if (memory::WriteTracker::try_alias_store(host, in)) return true;
    memory::WriteTracker::prepare_host_write(host, in.size());
    std::memcpy(host, in.data(), in.size());
    return true;
}
bool write_word(void* context, std::uint64_t address, std::span<const std::byte, 4> in) noexcept {
    if (address % alignof(std::uint32_t) == 0) {
        std::uint32_t value;
        std::memcpy(&value, in.data(), 4);
        return static_cast<State*>(context)->with_word(address, 4, rw, true, [&] {
            __atomic_store_n(reinterpret_cast<std::uint32_t*>(address), value, __ATOMIC_RELEASE);
        });
    }
    auto pin = checked_pin(*static_cast<State*>(context), address, 4, rw);
    if (!pin) return false;
    if (address % alignof(std::uint32_t)) {
        std::memcpy(pin->writable_bytes().data(), in.data(), 4);
    } else {
        std::uint32_t value;
        std::memcpy(&value, in.data(), 4);
        __atomic_store_n(reinterpret_cast<std::uint32_t*>(address), value, __ATOMIC_RELEASE);
    }
    return true;
}
bool read_word(void* context, std::uint64_t address, std::span<std::byte, 4> out) noexcept {
    if (address % alignof(std::uint32_t)) return false;
    return static_cast<State*>(context)->with_word(address, 4, Permission::read, false, [&] {
        const auto value = __atomic_load_n(reinterpret_cast<const std::uint32_t*>(address), __ATOMIC_ACQUIRE);
        std::memcpy(out.data(), &value, 4);
    });
}
bool read_double(void* context, std::uint64_t address, std::uint64_t& out) noexcept {
    if (address % alignof(std::uint64_t)) return false;
    return static_cast<State*>(context)->with_word(address, 8, Permission::read, false, [&] {
        out = __atomic_load_n(reinterpret_cast<const std::uint64_t*>(address), __ATOMIC_ACQUIRE);
    });
}
ProcessWideKey::ExchangeResult exchange_word(void* context, std::uint64_t address,
    std::uint32_t& expected, std::uint32_t desired) noexcept {
    using Result = ProcessWideKey::ExchangeResult;
    if (address % alignof(std::uint32_t)) return Result::invalid_address;
    Result result = Result::mismatch;
    if (!static_cast<State*>(context)->with_word(address, 4, rw, true, [&] {
            result = __atomic_compare_exchange_n(reinterpret_cast<std::uint32_t*>(address), &expected,
                desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE) ? Result::exchanged : Result::mismatch;
        }))
        return Result::invalid_address;
    return result;
}

struct ServicePins {
    std::shared_ptr<State> state;
    std::mutex mutex;
    std::vector<GuestMemoryPin> pins;
    static bool map(void* context, std::uint64_t address, std::uint64_t size) noexcept {
        try {
            auto& self = *static_cast<ServicePins*>(context);
            auto pin = checked_pin(*self.state, address, size, rw);
            if (!pin) return false;
            std::lock_guard lock(self.mutex);
            self.pins.push_back(std::move(pin));
            return true;
        } catch (...) { return false; }
    }
    static void unmap(void* context, std::uint64_t address, std::uint64_t size) noexcept {
        auto& self = *static_cast<ServicePins*>(context);
        std::lock_guard lock(self.mutex);
        const auto found = std::find_if(self.pins.begin(), self.pins.end(), [&](const auto& p) {
            return same_range(p->range(), {address, size});
        });
        if (found != self.pins.end()) self.pins.erase(found);
    }
};

class NativeGuestMemory final : public GuestMemory {
public:
    NativeGuestMemory(std::shared_ptr<State> state, std::shared_ptr<HandleTable> handles)
        : state_(std::move(state)), handles_(std::move(handles)) {}
    std::shared_ptr<ProcessRecord> process() const noexcept override { return state_->process; }
    LaunchResult<GuestMemoryPin> allocate(std::uint64_t bytes, MemoryType type, Permission access) override;
    LaunchStatus release(AddressRange range) override;
    LaunchResult<QueryDisposition> dispatch(std::uint16_t svc, QueryRegisters& registers) override;
    LaunchResult<GuestMemoryPin> pin(AddressRange range, Permission access) override {
        try { return state_->pin(range, access); }
        catch (const std::bad_alloc&) { return failure(LaunchError::resource_limit, "pin", "Pin allocation failed."); }
        catch (const std::length_error&) { return failure(LaunchError::resource_limit, "pin", "Pin limit exceeded."); }
    }
    bool copy_observation(std::uint64_t address, std::span<std::byte> out) const noexcept override {
        if (out.empty() || out.size() > 8 || address > UINT64_MAX - out.size()) return false;
        return state_->with_word(address, out.size(), Permission::read, false, [&] {
            std::memcpy(out.data(), reinterpret_cast<const void*>(address), out.size());
        });
    }
    GuestCopyOut copy_out() noexcept override {
        return {write_word, read_word, state_.get(), exchange_word, read_double};
    }
    GuestBuffers ipc_buffers() noexcept override { return {read_bytes, write_bytes, state_.get()}; }
    GuestMemoryMapper service_mapper() noexcept override {
        try {
            auto context = std::make_shared<ServicePins>();
            context->state = state_;
            return {ServicePins::map, ServicePins::unmap, context.get(), context};
        } catch (...) { return {}; }
    }
private:
    LaunchResult<QueryDisposition> heap(QueryRegisters& r);
    LaunchResult<QueryDisposition> protect(QueryRegisters& r);
    LaunchResult<QueryDisposition> attribute(QueryRegisters& r);
    LaunchResult<QueryDisposition> stack(QueryRegisters& r, bool unmap);
    LaunchResult<QueryDisposition> shared(QueryRegisters& r, bool unmap);
    std::shared_ptr<State> state_;
    // Kept outside State: service mapper pins may outlive this dispatcher and
    // must not retain the handle table containing their own service.
    std::shared_ptr<HandleTable> handles_;
};

LaunchResult<GuestMemoryPin> NativeGuestMemory::allocate(std::uint64_t bytes,
    MemoryType type, Permission access) {
    try {
        auto& s = *state_;
        std::lock_guard lock(s.mutex);
        if (s.poisoned) return host_failure("allocate");
        if ((type != MemoryType::normal && type != MemoryType::stack &&
             type != MemoryType::thread_local_storage) ||
            (access != Permission::read && access != rw) || !bytes)
            return failure(LaunchError::invalid_input, "allocate", "Unsupported bootstrap allocation type, access or size.");
        if (!native_range({ProcessRecord::aslr_start, bytes}))
            return unsupported("allocate", "Bootstrap allocations require complete native pages.");
        // Thread-local pages are kernel pages mapped into the process; they are
        // not charged to the process's physical-memory limit nor counted as used
        // memory. The thread block the host publishes for a host-started process
        // stands in for memory the SDK would take from its own heap.
        const bool charged = type != MemoryType::thread_local_storage;
        if (charged && bytes > s.accounting.resource_limit_bytes - s.accounting.resource_limit_used)
            return failure(LaunchError::resource_limit, "allocate", "Process memory budget exceeded.");
        std::shared_ptr<Region> region;
        AddressRange range{0, bytes};
        if (type == MemoryType::stack) {
            region = s.stack;
            range.address = region->address();
            for (const auto& b : s.blocks) {
                if (b.range.address + b.range.size <= range.address) continue;
                if (b.range.address >= range.address + bytes) break;
                range.address = b.range.address + b.range.size;
            }
            if (!pm::contains(range_of(*region), range))
                return failure(LaunchError::resource_limit, "allocate", "Stack reservation is exhausted.");
        } else if (s.mappable) {
            // Host allocations take free native pages of the mappable space.
            const auto page = platform::native_page_size();
            region = s.mappable;
            range.address = region->address();
            for (const auto& b : s.blocks) {
                if (b.range.address + b.range.size <= range.address) continue;
                if (b.range.address >= range.address + bytes) break;
                range.address = (b.range.address + b.range.size + page - 1) & ~(page - 1);
            }
            if (!pm::contains(range_of(*region), range))
                return failure(LaunchError::resource_limit, "allocate", "Mappable reservation is exhausted.");
        } else {
            auto reserved = Region::reserve(bytes);
            if (!std::holds_alternative<Region>(reserved)) return host_failure("allocate");
            region = std::make_shared<Region>(std::move(std::get<Region>(reserved)));
            range.address = region->address();
            if (!pm::contains(s.process->configuration().allocation_region, range))
                return unsupported("allocate", "Native allocation lies outside the admitted address space.");
        }
        s.allocations.reserve(s.allocations.size() + 1);
        auto next = pm::replace(s.blocks, range,
            MemoryBlock{range, type, 0, static_cast<std::uint8_t>(access)}, pm::merge_disable_mapping);
        auto accounting = s.accounting;
        if (charged) {
            (type == MemoryType::stack ? accounting.initial_stack_bytes :
                                        accounting.other_accounted_mapping_bytes) += bytes;
            accounting.resource_limit_used += bytes;
        }
        auto pin = s.make_pin(range, access);
        const auto offset = range.address - region->address();
        const auto status = s.transaction(std::move(next), accounting, "allocate",
            [&] { return region->commit(offset, bytes, access); },
            [&] { return region->decommit(offset, bytes); });
        if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
        s.allocations.push_back({range, type, std::move(region)});
        return pin;
    } catch (const std::bad_alloc&) {
        return failure(LaunchError::resource_limit, "allocate", "Bootstrap allocation failed.");
    } catch (const std::length_error&) {
        return failure(LaunchError::resource_limit, "allocate", "Bootstrap allocation limit exceeded.");
    }
}

LaunchStatus NativeGuestMemory::release(AddressRange range) {
    try {
        auto& s = *state_;
        std::lock_guard lock(s.mutex);
        if (s.poisoned) return host_failure("release");
        const auto found = std::find_if(s.allocations.begin(), s.allocations.end(),
            [&](const auto& a) { return same_range(a.range, range); });
        if (found == s.allocations.end())
            return failure(LaunchError::invalid_input, "release", "Retirement requires an exact bootstrap allocation.");
        if (s.pinned(range)) return unsupported("release", "The allocation still has live pins.");
        std::vector<MemoryBlock> retired;
        retired.reserve(s.blocks.size());
        auto cursor = range.address;
        const auto end = range.address + range.size;
        for (const auto& block : s.blocks) {
            if (!pm::overlaps(block.range, range)) continue;
            const auto begin = std::max(block.range.address, range.address);
            const auto block_end = std::min(block.range.address + block.range.size, end);
            if (begin != cursor || block.type != found->type || block.attributes ||
                block.ipc_references || block.device_references)
                return unsupported("release", "The allocation has active aliases or incompatible state.");
            auto segment = block;
            segment.range = {begin, block_end - begin};
            retired.push_back(segment);
            cursor = block_end;
        }
        if (cursor != end)
            return unsupported("release", "The allocation has active aliases or incompatible state.");
        auto saved = Region::alias_copy(range.address, range.size, Permission::none);
        if (!std::holds_alternative<Region>(saved)) return host_failure("release");
        auto& backup = std::get<Region>(saved);
        auto accounting = s.accounting;
        if (found->type != MemoryType::thread_local_storage) {
            (found->type == MemoryType::stack ? accounting.initial_stack_bytes :
                                             accounting.other_accounted_mapping_bytes) -= range.size;
            accounting.resource_limit_used -= range.size;
        }
        auto next = pm::replace(s.blocks, range, std::nullopt, 0, pm::merge_disable_mapping);
        const auto region = found->region;
        const auto offset = range.address - region->address();
        auto status = s.transaction(std::move(next), accounting, "release",
            [&] { return region->decommit(offset, range.size); },
            [&] {
                NativeStatus restored;
                for (const auto& segment : retired) {
                    const auto segment_offset = segment.range.address - range.address;
                    restored = region->alias(offset + segment_offset, segment.range.size,
                        backup.address() + segment_offset,
                        static_cast<Permission>(segment.permissions));
                    if (!restored.ok()) return restored;
                }
                return restored;
            });
        if (std::holds_alternative<std::monostate>(status)) s.allocations.erase(found);
        return status;
    } catch (const std::bad_alloc&) {
        return failure(LaunchError::resource_limit, "release", "Retirement metadata allocation failed.");
    }
}

LaunchResult<QueryDisposition> NativeGuestMemory::heap(QueryRegisters& r) {
    auto& s = *state_;
    const auto size = r[1], old = s.accounting.committed_heap;
    if (const auto result = pm::validate_heap_size(size)) return finish(r, result, size);
    const auto maximum_heap = ProcessRecord::heap_capacity -
        s.accounting.code_bytes - s.accounting.initial_stack_bytes;
    if (size > s.heap->size() || size > maximum_heap)
        return finish(r, pm::out_of_memory, size);
    if (size > s.accounting.resource_limit_bytes - (s.accounting.resource_limit_used - old))
        return finish(r, pm::limit_reached, size);
    if (size == old) return finish(r, 0, s.heap->address());
    const AddressRange delta{s.heap->address() + std::min(size, old),
                             size > old ? size - old : old - size};
    if (size < old) {
        const auto block = pm::uniform(s.blocks, delta);
        if (!block || block->type != MemoryType::normal || block->permissions != 3 ||
            block->attributes || block->ipc_references || block->device_references)
            return finish(r, pm::invalid_memory, size);
    }
    if (s.pinned(delta)) return unsupported("SetHeapSize", "Heap resize conflicts with a retained range.");
    auto next = pm::replace(s.blocks, delta, size > old ?
        std::optional<MemoryBlock>{{delta, MemoryType::normal, 0, 3}} : std::nullopt,
        old == 0 ? pm::merge_disable_mapping : 0, size == 0 ? pm::merge_disable_mapping : 0);
    auto accounting = s.accounting;
    accounting.committed_heap = size;
    accounting.resource_limit_used = accounting.resource_limit_used - old + size;
    std::optional<Region> backup;
    if (size < old) {
        auto saved = Region::alias_copy(delta.address, delta.size, rw);
        if (!std::holds_alternative<Region>(saved)) return host_failure("SetHeapSize");
        backup.emplace(std::move(std::get<Region>(saved)));
    }
    const auto offset = delta.address - s.heap->address();
    auto status = s.transaction(std::move(next), accounting, "SetHeapSize", [&] {
        return size > old ? s.heap->commit(offset, delta.size, rw) : s.heap->decommit(offset, delta.size);
    }, [&] {
        return size > old ? s.heap->decommit(offset, delta.size) :
            s.heap->alias(offset, delta.size, backup->address(), rw);
    });
    if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
    return finish(r, 0, s.heap->address());
}

LaunchResult<QueryDisposition> NativeGuestMemory::protect(QueryRegisters& r) {
    auto& s = *state_;
    const AddressRange range{r[0], r[1]};
    const auto permission = static_cast<std::uint32_t>(r[2]);
    if (const auto result = pm::validate_range(range)) return finish(r, result);
    if (permission != 0 && permission != 1 && permission != 3)
        return finish(r, pm::invalid_permission);
    const auto block = pm::uniform(s.blocks, range);
    if (!block || (block->type != MemoryType::normal && block->type != MemoryType::code_data) ||
        block->attributes || block->ipc_references || block->device_references)
        return finish(r, pm::invalid_memory);
    if (block->permissions == permission) return finish(r, 0);
    if (s.pinned(range)) return unsupported("SetMemoryPermission", "Protection conflicts with a retained range.");
    auto changed = *block;
    changed.permissions = static_cast<std::uint8_t>(permission);
    auto next = pm::replace(s.blocks, range, changed);
    // Native pages wholly inside the range take the new permission. A native
    // page the range only partly covers keeps the union of its guest pages'
    // permissions, so a reduction there is enforced only in the guest map.
    struct Change { AddressRange page; Permission to, from; };
    std::vector<Change> changes;
    const auto pages = native_pages(range);
    const auto page_size = platform::native_page_size();
    // The source of a copied stack keeps its data on the host until unmap
    // writes the stack back, so it stays readable and writable there.
    const auto union_of = [&](std::span<const MemoryBlock> blocks, AddressRange page) {
        std::uint8_t bits = 0;
        for (const auto& b : blocks) if (b.type != MemoryType::free && pm::overlaps(b.range, page)) bits |= b.permissions;
        for (const auto& a : s.aliases)
            if (pm::overlaps({a.source, a.size}, page) &&
                (!native_range({a.source, a.size}) || !native_range({a.destination, a.size})))
                bits |= 3;
        return bits;
    };
    for (auto at = pages.address; at < pages.address + pages.size; at += page_size) {
        const AddressRange page{at, page_size};
        const auto to = pm::contains(range, page) ? static_cast<std::uint8_t>(permission) : union_of(next, page);
        const auto from = pm::contains(range, page) ? block->permissions : union_of(s.blocks, page);
        if (to == from) continue;
        if ((to & 4) && (to & 2))
            return unsupported("SetMemoryPermission", "A native page would need to be writable and executable.");
        changes.push_back({page, static_cast<Permission>(to), static_cast<Permission>(from)});
    }
    auto status = s.transaction(std::move(next), s.accounting,
        "SetMemoryPermission", [&] {
            platform::ProcessMemoryStatus result;
            for (const auto& change : changes) {
                result = platform::protect_process_memory(change.page.address, change.page.size, change.to);
                if (!result.ok()) return result;
            }
            return result;
        }, [&] {
            platform::ProcessMemoryStatus result;
            for (const auto& change : changes) {
                const auto undone = platform::protect_process_memory(change.page.address, change.page.size, change.from);
                if (!undone.ok()) result = undone;
            }
            return result;
        });
    if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
    return finish(r, 0);
}

// SetMemoryAttribute(address, size, mask, value) for the uncached (8) and
// permission-locked (16) attributes. Both live in the guest memory map: the
// host has no uncached guest memory, and a permission-locked block refuses
// later permission changes because SetMemoryPermission admits no attributes.
LaunchResult<QueryDisposition> NativeGuestMemory::attribute(QueryRegisters& r) {
    auto& s = *state_;
    const AddressRange range{r[0], r[1]};
    const auto mask = static_cast<std::uint32_t>(r[2]);
    const auto value = static_cast<std::uint32_t>(r[3]);
    constexpr std::uint32_t uncached = 8, permission_locked = 16;
    if (const auto result = pm::validate_range(range)) return finish(r, result);
    if ((mask & ~(uncached | permission_locked)) || (value & ~mask)) return finish(r, pm::invalid_combination);
    const auto block = pm::uniform(s.blocks, range);
    if (!block || (block->type != MemoryType::normal && block->type != MemoryType::code_data) ||
        (block->attributes & ~(uncached | permission_locked)) || block->ipc_references || block->device_references)
        return finish(r, pm::invalid_memory);
    // A permission lock can only be set, never cleared.
    if ((mask & permission_locked) && !(value & permission_locked)) return finish(r, pm::invalid_combination);
    auto changed = *block;
    changed.attributes = static_cast<std::uint8_t>((block->attributes & ~mask) | value);
    if (changed.attributes == block->attributes) return finish(r, 0);
    auto status = s.transaction(pm::replace(s.blocks, range, changed), s.accounting, "SetMemoryAttribute",
        [] { return platform::ProcessMemoryStatus{}; }, [] { return platform::ProcessMemoryStatus{}; });
    if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
    return finish(r, 0);
}

LaunchResult<QueryDisposition> NativeGuestMemory::stack(QueryRegisters& r, bool unmap) {
    auto& s = *state_;
    const AddressRange destination{r[0], r[2]}, source{r[1], r[2]};
    const auto* operation = unmap ? "UnmapMemory" : "MapMemory";
    if ((destination.address | source.address) % guest_page_size) return finish(r, pm::invalid_address);
    if (const auto result = pm::validate_range(destination)) return finish(r, result);
    if (const auto result = pm::validate_range(source)) return finish(r, result);
    if (!pm::valid_range(source)) return finish(r, pm::invalid_memory);
    if (!pm::contains(range_of(*s.stack), destination))
        return finish(r, pm::invalid_region);
    const auto offset = destination.address - s.stack->address();
    // A native page larger than the guest page cannot alias a stack mapped at
    // guest-page granularity. Such a stack is a copy of its source instead,
    // written back on unmap, on native pages it owns. The source stays
    // natively accessible; the guest memory map still reports it locked.
    const bool copied = !native_range(source) || !native_range(destination);
    const auto pages = native_pages(destination);
    if (!unmap) {
        const auto block = pm::uniform(s.blocks, source);
        if (!block || (block->type != MemoryType::normal && block->type != MemoryType::code_data) ||
            block->permissions != 3 || block->attributes || block->ipc_references || block->device_references ||
            !pm::free(s.blocks, destination)) return finish(r, pm::invalid_memory);
        if (s.pinned(source) || s.pinned(destination))
            return unsupported(operation, "Stack alias conflicts with a retained range.");
        if (s.aliases.size() >= 4096) return unsupported(operation, "Stack alias host limit exceeded.");
        s.aliases.reserve(s.aliases.size() + 1);
        auto locked = *block; locked.permissions = 0; locked.attributes = 1;
        auto next = pm::replace(s.blocks, source, locked, pm::merge_disable_locked);
        next = pm::replace(next, destination, MemoryBlock{destination, MemoryType::stack, 0, 3},
                           pm::merge_disable_mapping);
        if (copied) {
            if (!pm::contains(range_of(*s.stack), pages)) return finish(r, pm::invalid_memory);
            if (s.envelope_used(pages))
                return unsupported(operation, "A copied stack shares a native page with a sub-page shared mapping.");
            const auto page = platform::native_page_size();
            std::vector<std::uint64_t> fresh;
            for (auto at = pages.address; at < pages.address + pages.size; at += page)
                if (!native_page_used(s.blocks, at, destination)) fresh.push_back(at);
            auto status = s.transaction(std::move(next), s.accounting, operation, [&] {
                platform::ProcessMemoryStatus result;
                for (std::size_t i = 0; i < fresh.size(); ++i) {
                    result = s.stack->commit(fresh[i] - s.stack->address(), page, rw);
                    if (!result.ok()) {
                        while (i--) s.restored(s.stack->decommit(fresh[i] - s.stack->address(), page));
                        return result;
                    }
                }
                // A page already in use may hold a read-only copy; the stack
                // needs it writable.
                result = s.stack->protect(pages.address - s.stack->address(), pages.size, rw);
                if (!result.ok()) return result;
                std::memcpy(reinterpret_cast<void*>(destination.address),
                            reinterpret_cast<const void*>(source.address), destination.size);
                return result;
            }, [&] {
                platform::ProcessMemoryStatus result;
                for (const auto at : fresh) {
                    const auto undone = s.stack->decommit(at - s.stack->address(), page);
                    if (!undone.ok()) result = undone;
                }
                return result;
            });
            if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
            s.aliases.push_back({destination.address, source.address, source.size, block->type});
            return finish(r, 0);
        }
        auto status = s.transaction(std::move(next), s.accounting, operation, [&] {
            auto result = s.stack->alias(offset, destination.size, source.address, rw);
            if (!result.ok()) return result;
            result = platform::protect_process_memory(source.address, source.size, Permission::none);
            if (!result.ok()) s.restored(s.stack->decommit(offset, destination.size));
            return result;
        }, [&] {
            s.restored(platform::protect_process_memory(source.address, source.size, rw));
            return s.stack->decommit(offset, destination.size);
        });
        if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
        s.aliases.push_back({destination.address, source.address, source.size, block->type});
    } else {
        const auto source_block = pm::uniform(s.blocks, source);
        if (!source_block ||
            (source_block->type != MemoryType::normal && source_block->type != MemoryType::code_data) ||
            source_block->permissions != 0 || source_block->attributes != 1 ||
            source_block->ipc_references || source_block->device_references)
            return finish(r, pm::invalid_memory);
        const auto destination_block = pm::uniform(s.blocks, destination);
        if (!destination_block || destination_block->type != MemoryType::stack ||
            destination_block->attributes || destination_block->ipc_references ||
            destination_block->device_references)
            return finish(r, pm::invalid_memory);
        // Registered native aliases establish page identity even for a request
        // whose valid partial-pair shape the host profile cannot yet execute.
        auto cursor = destination.address;
        const auto end = destination.address + destination.size;
        while (cursor < end) {
            const auto alias = std::find_if(s.aliases.begin(), s.aliases.end(), [&](const auto& a) {
                return cursor >= a.destination && cursor - a.destination < a.size;
            });
            if (alias == s.aliases.end() ||
                alias->source + (cursor - alias->destination) != source.address + (cursor - destination.address))
                return finish(r, pm::invalid_region);
            cursor = std::min(end, alias->destination + alias->size);
        }
        auto plan = plan_stack_alias_unmap(s.blocks, s.aliases, destination.address, source.address, source.size);
        if (const auto* error = std::get_if<StackAliasFailure>(&plan)) {
            if (error->code == StackAliasError::AllocationFailure || error->code == StackAliasError::AdmissionLimit)
                return failure(LaunchError::resource_limit, operation, "Stack alias metadata allocation failed.");
            return unsupported(operation, "Only an unchanged exact registered stack pair can be unmapped.");
        }
        if (s.pinned(source) || s.pinned(destination))
            return unsupported(operation, "Stack alias conflicts with a retained range.");
        auto& next = std::get<StackAliasUnmapPlan>(plan);
        // Normalize the pure planner's explicit free destination to our sparse map.
        auto blocks = pm::replace(next.blocks, destination, std::nullopt);
        if (copied) {
            const auto page = platform::native_page_size();
            std::vector<std::uint64_t> released;
            for (auto at = pages.address; at < pages.address + pages.size; at += page)
                if (!native_page_used(s.blocks, at, destination)) released.push_back(at);
            std::vector<std::byte> original(source.size);
            std::memcpy(original.data(), reinterpret_cast<const void*>(source.address), source.size);
            auto status = s.transaction(std::move(blocks), s.accounting, operation, [&] {
                std::memcpy(reinterpret_cast<void*>(source.address),
                            reinterpret_cast<const void*>(destination.address), destination.size);
                std::memset(reinterpret_cast<void*>(destination.address), 0, destination.size);
                platform::ProcessMemoryStatus result;
                for (const auto at : released) {
                    result = s.stack->decommit(at - s.stack->address(), page);
                    if (!result.ok()) break;
                }
                if (!result.ok())
                    std::memcpy(reinterpret_cast<void*>(source.address), original.data(), source.size);
                return result;
            }, [&] {
                std::memcpy(reinterpret_cast<void*>(source.address), original.data(), source.size);
                return platform::ProcessMemoryStatus{};
            });
            if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
            s.aliases.erase(s.aliases.begin() + static_cast<std::ptrdiff_t>(next.alias_index));
            return finish(r, 0);
        }
        auto status = s.transaction(std::move(blocks), s.accounting, operation, [&] {
            auto result = platform::protect_process_memory(source.address, source.size, rw);
            if (!result.ok()) return result;
            result = s.stack->decommit(offset, destination.size);
            if (!result.ok()) s.restored(platform::protect_process_memory(source.address, source.size, Permission::none));
            return result;
        }, [&] {
            s.restored(s.stack->alias(offset, destination.size, source.address, rw));
            return platform::protect_process_memory(source.address, source.size, Permission::none);
        });
        if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
        s.aliases.erase(s.aliases.begin() + static_cast<std::ptrdiff_t>(next.alias_index));
    }
    return finish(r, 0);
}

LaunchResult<QueryDisposition> NativeGuestMemory::shared(QueryRegisters& r, bool unmap) {
    auto& s = *state_;
    const AddressRange range{r[1], r[2]};
    const auto permission = static_cast<std::uint32_t>(r[3]);
    const auto* operation = unmap ? "UnmapSharedMemory" : "MapSharedMemory";
    if (const auto result = pm::validate_range(range)) return finish(r, result);
    if (!unmap && permission != 1 && permission != 3) return finish(r, pm::invalid_permission);
    const auto object = handles_->lookup(static_cast<Handle>(r[0]));
    const auto mutable_object = std::dynamic_pointer_cast<SharedMemory>(object);
    const auto immutable = std::dynamic_pointer_cast<ReadOnlySharedMemory>(object);
    if (!mutable_object && !immutable) return finish(r, pm::invalid_handle);
    if (!pm::contains(s.process->configuration().allocation_region, range) ||
        pm::overlaps(range_of(*s.heap), range) || pm::overlaps(range_of(*s.alias), range))
        return finish(r, pm::invalid_region);
    const auto size = mutable_object ? mutable_object->size() : immutable->bytes().size();
    if (range.size != size) return finish(r, pm::invalid_size);
    if (!unmap && immutable && permission != 1) return finish(r, pm::invalid_permission);
    if (unmap) {
        const auto block = pm::uniform(s.blocks, range);
        if (!block || block->type != MemoryType::shared || block->attributes ||
            block->ipc_references || block->device_references)
            return finish(r, pm::invalid_memory);
        const auto found = std::find_if(s.shared.begin(), s.shared.end(), [&](const auto& m) {
            return same_range(m.range, range) && m.object == object;
        });
        if (found == s.shared.end()) return finish(r, pm::invalid_region);
        if (!native_range(range) && immutable) {
            // Native pages no other mapping uses are released with the copy.
            if (s.pinned(range)) return unsupported(operation, "Shared mapping conflicts with a retained range.");
            const auto pages = native_pages(range);
            const auto page = platform::native_page_size();
            const auto region = found->region;
            std::vector<std::uint64_t> released;
            for (auto at = pages.address; at < pages.address + pages.size; at += page)
                if (!native_page_used(s.blocks, at, range)) released.push_back(at);
            auto status = s.transaction(pm::replace(s.blocks, range, std::nullopt, 0, pm::merge_disable_mapping),
                s.accounting, operation,
                [&] {
                    platform::ProcessMemoryStatus result;
                    for (const auto at : released) {
                        result = region->decommit(at - region->address(), page);
                        if (!result.ok()) return result;
                    }
                    return result;
                },
                [&] { return platform::ProcessMemoryStatus{}; });
            if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
            s.shared.erase(found);
            return finish(r, 0);
        }
        if (found->envelope.size) {
            // The envelope's native pages hold only this mapping; they are
            // released together and, on failure, alias the object again.
            if (s.pinned(found->envelope)) return unsupported(operation, "Shared mapping conflicts with a retained range.");
            const auto envelope = found->envelope;
            const auto region = found->region;
            const auto access = found->access;
            const auto offset = envelope.address - region->address();
            const auto source = reinterpret_cast<std::uint64_t>(mutable_object->data()) - (range.address - envelope.address);
            auto status = s.transaction(pm::replace(s.blocks, range, std::nullopt, 0, pm::merge_disable_mapping), s.accounting, operation,
                [&] { return region->decommit(offset, envelope.size); },
                [&] { return region->alias(offset, envelope.size, source, access); });
            if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
            s.shared.erase(found);
            return finish(r, 0);
        }
        if (!native_range(range)) return unsupported(operation, "Subpage native shared mappings are unsupported.");
        if (s.pinned(range)) return unsupported(operation, "Shared mapping conflicts with a retained range.");
        auto saved = Region::alias_copy(range.address, range.size, found->access);
        if (!std::holds_alternative<Region>(saved)) return host_failure(operation);
        auto& backup = std::get<Region>(saved);
        const auto region = found->region;
        const auto offset = range.address - region->address();
        auto status = s.transaction(pm::replace(s.blocks, range, std::nullopt, 0, pm::merge_disable_mapping), s.accounting, operation,
            [&] { return region->decommit(offset, range.size); },
            [&] { return region->alias(offset, range.size, backup.address(), found->access); });
        if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
        s.shared.erase(found);
    } else {
        if (!pm::free(s.blocks, range)) return finish(r, pm::invalid_memory);
        if (!native_range(range) && immutable) {
            // A read-only object smaller than a native page is a copy; no
            // writer exists, so the copy stays exact. A native page it shares
            // with a writable mapping stays writable on the host.
            const auto pages = native_pages(range);
            const auto region = s.region(pages);
            if (!region || (region != s.mappable && region != s.stack) || s.pinned(pages))
                return unsupported(operation, "A read-only shared copy needs native pages of the mappable space.");
            if (s.envelope_used(pages))
                return unsupported(operation, "A read-only shared copy shares a native page with a sub-page shared mapping.");
            const auto page = platform::native_page_size();
            std::vector<std::uint64_t> fresh;
            bool writable = false;
            for (auto at = pages.address; at < pages.address + pages.size; at += page)
                if (!native_page_used(s.blocks, at, range, &writable)) fresh.push_back(at);
            const auto offset = pages.address - region->address();
            s.shared.reserve(s.shared.size() + 1);
            auto status = s.transaction(pm::replace(s.blocks, range,
                MemoryBlock{range, MemoryType::shared, 0, static_cast<std::uint8_t>(Permission::read)},
                pm::merge_disable_mapping), s.accounting, operation,
                [&] {
                    platform::ProcessMemoryStatus result;
                    for (const auto at : fresh) {
                        result = region->commit(at - region->address(), page, rw);
                        if (!result.ok()) return result;
                    }
                    result = region->protect(offset, pages.size, rw);
                    if (!result.ok()) return result;
                    std::memcpy(reinterpret_cast<void*>(range.address), immutable->bytes().data(), size);
                    return writable ? result : region->protect(offset, pages.size, Permission::read);
                },
                [&] {
                    platform::ProcessMemoryStatus result;
                    for (const auto at : fresh) {
                        const auto undone = region->decommit(at - region->address(), page);
                        if (!undone.ok()) result = undone;
                    }
                    return result;
                });
            if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
            s.shared.push_back({range, object, region, Permission::read});
            return finish(r, 0);
        }
        const auto page = platform::native_page_size();
        const bool mapped_elsewhere = std::any_of(s.shared.begin(), s.shared.end(),
            [&](const auto& m) { return m.object == object; });
        if (!native_range(range) && mutable_object) {
            // A host page larger than the guest page aliases only bytes at the
            // same offset within a page. The object's bytes move to the
            // range's offset and the native pages around the range alias the
            // region holding them whole. Those pages then hold the object's
            // neighbouring bytes, so no other guest block may use them; the
            // guest-free parts of the envelope read the region's zero padding
            // on the host rather than faulting.
            const auto pages = native_pages(range);
            if (s.pinned(pages)) return unsupported(operation, "Shared mapping conflicts with a retained range.");
            for (auto at = pages.address; at < pages.address + pages.size; at += page)
                if (native_page_used(s.blocks, at, range) || s.envelope_used({at, page}))
                    return unsupported(operation, "A sub-page shared mapping shares a native page with another mapping.");
            const auto phase = static_cast<std::size_t>(range.address - pages.address);
            if (!mutable_object->spans(phase, page)) {
                if (mapped_elsewhere)
                    return unsupported(operation, "Shared object is mapped at another native page offset.");
                if (!mutable_object->rephase(phase, page)) return host_failure(operation);
            }
            auto region = s.region(pages);
            if (!region) {
                auto reserved = Region::reserve(pages.size, pages.address);
                if (!std::holds_alternative<Region>(reserved)) return host_failure(operation);
                region = std::make_shared<Region>(std::move(std::get<Region>(reserved)));
            }
            s.shared.reserve(s.shared.size() + 1);
            const auto access = static_cast<Permission>(permission);
            const auto offset = pages.address - region->address();
            const auto source = reinterpret_cast<std::uint64_t>(mutable_object->data()) - phase;
            auto status = s.transaction(pm::replace(s.blocks, range,
                MemoryBlock{range, MemoryType::shared, 0, static_cast<std::uint8_t>(access)},
                pm::merge_disable_mapping), s.accounting, operation,
                [&] { return region->alias(offset, pages.size, source, access); },
                [&] { return region->decommit(offset, pages.size); });
            if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
            s.shared.push_back({range, object, std::move(region), access, pages});
            return finish(r, 0);
        }
        if (!native_range(range)) return unsupported(operation, "Subpage native shared mappings are unsupported.");
        if (s.pinned(range)) return unsupported(operation, "Shared mapping conflicts with a retained range.");
        if (s.envelope_used(range))
            return unsupported(operation, "A shared mapping shares a native page with a sub-page shared mapping.");
        if (mutable_object && !mutable_object->spans(0, page)) {
            // An earlier sub-page mapping moved the bytes; move them back to
            // native page alignment when no mapping still uses them.
            if (mapped_elsewhere)
                return unsupported(operation, "Shared object is mapped at another native page offset.");
            if (!mutable_object->rephase(0, page)) return host_failure(operation);
        }
        if (mutable_object && !native_source(reinterpret_cast<std::uint64_t>(mutable_object->data()), size))
            return unsupported(operation, "Shared object backing is not aligned to native pages.");
        auto region = s.region(range);
        if (!region) {
            auto reserved = Region::reserve(size, range.address);
            if (!std::holds_alternative<Region>(reserved)) return host_failure(operation);
            region = std::make_shared<Region>(std::move(std::get<Region>(reserved)));
        }
        s.shared.reserve(s.shared.size() + 1);
        const auto access = static_cast<Permission>(permission);
        const auto offset = range.address - region->address();
        // Immutable objects have no writer and expose no alternate writable view.
        // Each copy preserves their constant contents; object identity, not the
        // temporary host allocation, remains the unmap identity.
        std::optional<Region> immutable_backing;
        if (immutable) {
            auto reserved = Region::reserve(size);
            if (!std::holds_alternative<Region>(reserved)) return host_failure(operation);
            immutable_backing.emplace(std::move(std::get<Region>(reserved)));
            if (!immutable_backing->commit(0, size, rw).ok()) return host_failure(operation);
            std::memcpy(reinterpret_cast<void*>(immutable_backing->address()), immutable->bytes().data(), size);
        }
        const auto source = mutable_object ? reinterpret_cast<std::uint64_t>(mutable_object->data()) : immutable_backing->address();
        auto status = s.transaction(pm::replace(s.blocks, range,
            MemoryBlock{range, MemoryType::shared, 0, static_cast<std::uint8_t>(access)},
            pm::merge_disable_mapping), s.accounting, operation,
            [&] { return region->alias(offset, size, source, access); },
            [&] { return region->decommit(offset, size); });
        if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
        s.shared.push_back({range, object, std::move(region), access});
    }
    return finish(r, 0);
}

LaunchResult<QueryDisposition> NativeGuestMemory::dispatch(std::uint16_t svc, QueryRegisters& r) {
    try {
        std::lock_guard lock(state_->mutex);
        if (state_->poisoned) return host_failure("memory dispatch");
        switch (svc) {
        case 1: return heap(r);
        case 2: return protect(r);
        case 4: return stack(r, false);
        case 5: return stack(r, true);
        case 0x13: return shared(r, false);
        case 0x14: return shared(r, true);
        case 3: return attribute(r);
        // CreateTransferMemory records the range and permission; the startup
        // queries own that object. The owner's range is not locked meanwhile.
        default: return QueryDisposition::unsupported_operation;
        }
    } catch (const std::bad_alloc&) {
        return failure(LaunchError::resource_limit, "memory dispatch", "Transaction allocation failed.");
    } catch (const std::length_error&) {
        return failure(LaunchError::resource_limit, "memory dispatch", "Transaction container limit exceeded.");
    }
}
} // namespace

LaunchResult<std::shared_ptr<GuestMemory>> create_guest_memory(const LaunchProfile& profile,
    ProcessMemoryOptions options, std::shared_ptr<ModuleSet> modules,
    std::shared_ptr<HandleTable> handles) {
    try {
        if (!handles || !options.resource_limit_bytes || options.resource_limit_bytes > ProcessRecord::heap_capacity)
            return failure(LaunchError::invalid_input, "create_guest_memory", "A bounded memory budget and handle table are required.");
        if (!profile.is_64_bit || profile.process.address_bits != 39 ||
            profile.process.baseline != Baseline::horizon_20_5_0 ||
            profile.process.private_system_resource_bytes || !profile.process.allowed_core_mask)
            return unsupported("create_guest_memory", "The process profile is not supported by native memory.");
        if ((!modules || !modules->module_count()) && profile.origin != ProfileOrigin::synthetic_test)
            return failure(LaunchError::invalid_input, "create_guest_memory", "Ordinary processes require retained sealed modules.");
        if (profile.origin != ProfileOrigin::synthetic_test && !atomic_record_publication<ProcessRecord>())
            return unsupported("create_guest_memory", "Ordinary processes require atomic ProcessRecord publication from the integrator patch.");
        auto s = std::make_shared<State>();
        s->modules = std::move(modules);
        auto config = profile.process;
        if (config.heap_size > ProcessRecord::heap_capacity || config.base_alias_size > ProcessRecord::base_alias_capacity ||
            config.stack_size > ProcessRecord::stack_capacity || config.alias_extra_enabled)
            return unsupported("create_guest_memory", "The reservation profile is unsupported.");
        const auto reserve = [&](std::uint64_t size, std::uint64_t& address, std::shared_ptr<Region>& owner) {
            auto result = Region::reserve(size, address);
            if (!std::holds_alternative<Region>(result)) return false;
            owner = std::make_shared<Region>(std::move(std::get<Region>(result)));
            address = owner->address();
            return pm::contains(config.allocation_region, range_of(*owner));
        };
        if (profile.origin != ProfileOrigin::synthetic_test) {
            // The guest picks addresses for its own mappings anywhere free in
            // its address space, but host libraries also reserve address space
            // there. The process address space is therefore one host arena:
            // free mappable space, then the alias, stack and heap regions. The
            // modules stay where the host loaded them.
            const auto page = platform::native_page_size();
            constexpr std::uint64_t mappable_size = 1ull << 30;
            const auto arena_size = mappable_size + config.base_alias_size + config.stack_size + config.heap_size;
            bool placed = false;
            for (int attempt = 0; attempt < 4 && !placed; ++attempt) {
                std::uint64_t arena = 0;
                {
                    auto probe = Region::reserve((arena_size + page - 1) & ~(page - 1));
                    if (!std::holds_alternative<Region>(probe)) return host_failure("create_guest_memory arena");
                    arena = std::get<Region>(probe).address();
                }
                config.allocation_region = {arena, arena_size};
                auto cursor = arena;
                std::uint64_t mappable_address = cursor;
                cursor += mappable_size;
                config.alias_address = cursor;
                cursor += config.base_alias_size;
                config.stack_address = cursor;
                cursor += config.stack_size;
                config.heap_address = cursor;
                placed = reserve(mappable_size, mappable_address, s->mappable) &&
                    reserve(config.base_alias_size, config.alias_address, s->alias) &&
                    reserve(config.stack_size, config.stack_address, s->stack) &&
                    reserve(config.heap_size, config.heap_address, s->heap);
                if (!placed) s->mappable.reset(), s->alias.reset(), s->stack.reset(), s->heap.reset();
            }
            if (!placed) return host_failure("create_guest_memory reservations");
        } else if (!reserve(config.heap_size, config.heap_address, s->heap) ||
            !reserve(config.base_alias_size, config.alias_address, s->alias) ||
            !reserve(config.stack_size, config.stack_address, s->stack))
            return host_failure("create_guest_memory reservations");
        s->accounting.resource_limit_bytes = options.resource_limit_bytes;
        if (s->modules) {
            for (std::size_t index = 0; index < s->modules->module_count(); ++index) {
                const auto& mapping = s->modules->mapping(index);
                if (!mapping.valid() || !mapping.sealed() ||
                    mapping.base_address() < ProcessRecord::aslr_start ||
                    mapping.mapping_size() > ProcessRecord::query_end - mapping.base_address())
                    return unsupported("create_guest_memory modules", "Module backing is unsealed or outside the process address space.");
                const auto status = register_module(*s, mapping, s->modules->appended_code(index),
                    options.resource_limit_bytes, profile.origin != ProfileOrigin::synthetic_test);
                if (const auto* f = std::get_if<LaunchFailure>(&status)) return *f;
            }
        }
        s->accounting.resource_limit_used = s->accounting.code_bytes;
        std::sort(s->blocks.begin(), s->blocks.end(), [](const auto& a, const auto& b) {
            return a.range.address < b.range.address;
        });
        auto built = ProcessRecord::create(config, s->accounting, s->blocks);
        if (!built.ok()) return failure(LaunchError::invalid_input, "create_guest_memory", "Native reservation metadata was rejected.");
        s->process = std::move(built.process);
        return std::shared_ptr<GuestMemory>(std::make_shared<NativeGuestMemory>(std::move(s), std::move(handles)));
    } catch (const std::bad_alloc&) {
        return failure(LaunchError::resource_limit, "create_guest_memory", "Native memory owner allocation failed.");
    } catch (const std::length_error&) {
        return failure(LaunchError::resource_limit, "create_guest_memory", "Native memory metadata limit exceeded.");
    }
}
} // namespace armigo::runtime
