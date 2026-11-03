// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/startup_queries.h>

#include <armigo/horizon/process_wide_key.h>
#include <armigo/horizon/event.h>
#include "process_memory.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>
#include <new>
#include <stdexcept>
#include <utility>

namespace armigo::horizon {
namespace {

constexpr std::uint32_t invalid_handle = 0xe401;
constexpr std::uint32_t invalid_copy = 0xe601;
constexpr std::uint32_t invalid_combination = 0xe801;
constexpr std::uint32_t invalid_info = 0xf001;
constexpr std::uint32_t invalid_address = 0xcc01;
constexpr std::uint32_t out_of_handles = 0xbe01;
constexpr std::uint64_t result_limit = ProcessRecord::query_end;
constexpr std::uint64_t high_word = 0xffffffff00000000ull;

bool region_valid(AddressRange region) {
    return region.address >= ProcessRecord::aslr_start &&
           region.address < ProcessRecord::query_end &&
           region.size != 0 && region.size <= ProcessRecord::query_end - region.address &&
           (region.address | region.size) % guest_page_size == 0;
}

bool overlaps(AddressRange left, AddressRange right) {
    return left.address < right.address + right.size &&
           right.address < left.address + left.size;
}

bool used_bytes(const MemoryAccounting& accounting, std::uint64_t& used) {
    used = 0;
    for (auto bytes : {accounting.committed_heap, accounting.other_accounted_mapping_bytes,
                       accounting.code_bytes, accounting.initial_stack_bytes}) {
        if (bytes > std::numeric_limits<std::uint64_t>::max() - used) return false;
        used += bytes;
    }
    return true;
}

bool accounting_valid(const MemoryAccounting& accounting, const ProcessConfiguration& config) {
    std::uint64_t used;
    return used_bytes(accounting, used) &&
           accounting.committed_heap <= config.heap_size &&
           accounting.initial_stack_bytes <= config.stack_size &&
           used <= ProcessRecord::heap_capacity &&
           used <= accounting.resource_limit_used &&
           accounting.resource_limit_used <= accounting.resource_limit_bytes;
}

bool block_valid(const MemoryBlock& block) {
    const auto& range = block.range;
    if (range.size == 0 || range.address >= ProcessRecord::query_end ||
        range.size > ProcessRecord::query_end - range.address ||
        (range.address | range.size) % guest_page_size != 0) return false;
    switch (block.type) {
    case MemoryType::free:
        return block.attributes == 0 && block.permissions == 0 &&
               block.ipc_references == 0 && block.device_references == 0;
    case MemoryType::reserved:
        return block.permissions == 0;
    case MemoryType::code:
    case MemoryType::code_data:
    case MemoryType::normal:
    case MemoryType::shared:
    case MemoryType::stack:
    case MemoryType::thread_local_storage:
        return range.address >= ProcessRecord::aslr_start;
    }
    return false;
}

MemoryInfo lookup(std::span<const MemoryBlock> blocks, std::uint64_t address) {
    if (address >= ProcessRecord::query_end) {
        return {{ProcessRecord::query_end, std::uint64_t{0} - ProcessRecord::query_end},
                MemoryType::reserved};
    }
    const auto next = std::upper_bound(blocks.begin(), blocks.end(), address,
        [](auto at, const auto& block) { return at < block.range.address; });
    std::uint64_t begin = 0;
    if (next != blocks.begin()) {
        const auto& previous = *(next - 1);
        const auto end = previous.range.address + previous.range.size;
        if (address < end) return previous;
        begin = end;
    }
    const auto end = next == blocks.end() ? ProcessRecord::query_end : next->range.address;
    return {{begin, end - begin}, MemoryType::free};
}

void finish(QueryRegisters& registers, std::uint32_t result, std::uint64_t value) {
    registers[0] = result;
    registers[1] = value;
    std::fill(registers.begin() + 2, registers.end(), 0);
}

bool supported_info(std::uint32_t type) {
    switch (type) {
    case 0: case 2: case 3: case 4: case 5: case 8: case 11: case 12: case 13:
    case 14: case 15: case 16: case 20: case 21: case 22: case 28:
        return true;
    default:
        return false;
    }
}

// These are unrecognized by the selected kernel's GetInfo switch. Other
// recognized types remain unsupported Armigo capabilities, not guest errors.
bool rejected_info(std::uint32_t type) {
    return type == 19 || (type >= 29 && type <= 33) || type > 34;
}

} // namespace

std::array<std::byte, 40> encode_memory_info(const MemoryInfo& info) noexcept {
    std::array<std::byte, 40> result{};
    const auto put = [&](std::size_t offset, std::uint64_t value, unsigned size) {
        for (unsigned byte = 0; byte < size; ++byte) {
            result[offset + byte] = static_cast<std::byte>((value >> (byte * 8)) & 255);
        }
    };
    put(0, info.range.address, 8);
    put(8, info.range.size, 8);
    put(16, static_cast<std::uint8_t>(info.type), 4);
    put(20, info.attributes, 4);
    put(24, info.permissions & 7u, 4);
    put(28, info.ipc_references, 4);
    put(32, info.device_references, 4);
    return result;
}

ProcessBuild ProcessRecord::create(const ProcessConfiguration& configuration,
                                  const MemoryAccounting& accounting,
                                  std::span<const MemoryBlock> blocks) {
    if (configuration.baseline != Baseline::horizon_20_5_0 ||
        configuration.address_bits != 39 || configuration.private_system_resource_bytes != 0 ||
        configuration.allowed_core_mask == 0) {
        return {ProcessStateError::unsupported_profile, {}};
    }
    if (configuration.max_memory_blocks == 0 || configuration.max_memory_blocks > (1u << 20)) {
        return {ProcessStateError::resource_limit, {}};
    }
    if (configuration.heap_size > heap_capacity || configuration.base_alias_size > base_alias_capacity ||
        configuration.stack_size > stack_capacity || !region_valid(configuration.allocation_region)) {
        return {ProcessStateError::invalid_layout, {}};
    }
    const AddressRange heap{configuration.heap_address, configuration.heap_size};
    const AddressRange alias{configuration.alias_address,
        configuration.base_alias_size + (configuration.alias_extra_enabled ? base_alias_capacity : 0)};
    const AddressRange stack{configuration.stack_address, configuration.stack_size};
    const auto inside_allocation = [&](AddressRange range) {
        const auto allocation = configuration.allocation_region;
        return range.address >= allocation.address && range.address - allocation.address < allocation.size &&
               range.size <= allocation.size - (range.address - allocation.address);
    };
    if (!region_valid(heap) || !region_valid(alias) || !region_valid(stack) ||
        !inside_allocation(heap) || !inside_allocation(alias) || !inside_allocation(stack) ||
        overlaps(heap, alias) || overlaps(heap, stack) || overlaps(alias, stack)) {
        return {ProcessStateError::invalid_layout, {}};
    }
    if (!accounting_valid(accounting, configuration)) return {ProcessStateError::invalid_accounting, {}};
    try {
        auto process = std::shared_ptr<ProcessRecord>(new ProcessRecord(configuration));
        process->accounting_ = accounting;
        const auto updated = process->replace_memory_map(blocks);
        if (updated != ProcessStateError::none) return {updated, {}};
        return {ProcessStateError::none, std::move(process)};
    } catch (const std::bad_alloc&) {
        return {ProcessStateError::resource_limit, {}};
    } catch (const std::length_error&) {
        return {ProcessStateError::resource_limit, {}};
    }
}

ProcessStateError ProcessRecord::set_accounting(const MemoryAccounting& accounting) {
    if (!accounting_valid(accounting, configuration_)) return ProcessStateError::invalid_accounting;
    std::lock_guard lock(mutex_);
    accounting_ = accounting;
    return ProcessStateError::none;
}

ProcessStateError ProcessRecord::replace_memory_map(std::span<const MemoryBlock> blocks) {
    if (blocks.size() > configuration_.max_memory_blocks) return ProcessStateError::resource_limit;
    try {
        std::vector<MemoryBlock> replacement;
        if (!blocks.empty()) replacement.assign(blocks.begin(), blocks.end());
        std::sort(replacement.begin(), replacement.end(),
            [](const auto& left, const auto& right) {
                return left.range.address < right.range.address;
            });
        std::uint64_t previous_end = 0;
        for (const auto& block : replacement) {
            if (!block_valid(block) || block.range.address < previous_end) {
                return ProcessStateError::invalid_memory_map;
            }
            previous_end = block.range.address + block.range.size;
        }
        process_memory::coalesce(replacement);
        {
            std::lock_guard lock(mutex_);
            blocks_.swap(replacement);
        }
        return ProcessStateError::none;
    } catch (const std::bad_alloc&) {
        return ProcessStateError::resource_limit;
    } catch (const std::length_error&) {
        return ProcessStateError::resource_limit;
    }
}

ProcessStateError ProcessRecord::commit_memory_state(std::span<const MemoryBlock> blocks,
    const MemoryAccounting& accounting, bool (*apply)(void*) noexcept, void* context) {
    if (apply == nullptr) return ProcessStateError::backend_failure;
    auto prepared = create(configuration_, accounting, blocks);
    if (!prepared.ok()) return prepared.error;
    std::lock_guard lock(mutex_);
    if (!apply(context)) return ProcessStateError::backend_failure;
    blocks_.swap(prepared.process->blocks_);
    accounting_ = accounting;
    return ProcessStateError::none;
}

MemoryInfo ProcessRecord::query_memory(std::uint64_t address) const {
    std::lock_guard lock(mutex_);
    return lookup(blocks_, address);
}

bool ProcessRecord::writable(std::uint64_t address, std::size_t size) const {
    if (address >= query_end || size > query_end - address) return false;
    std::lock_guard lock(mutex_);
    auto remaining = static_cast<std::uint64_t>(size);
    while (remaining != 0) {
        const auto block = lookup(blocks_, address);
        if ((block.permissions & 2u) == 0) return false;
        const auto chunk = std::min(remaining, block.range.size - (address - block.range.address));
        address += chunk;
        remaining -= chunk;
    }
    return true;
}

std::uint64_t ProcessRecord::used_memory() const {
    std::lock_guard lock(mutex_);
    std::uint64_t used = 0;
    (void)used_bytes(accounting_, used);
    return used;
}

std::uint64_t ProcessRecord::total_memory() const {
    std::lock_guard lock(mutex_);
    std::uint64_t used = 0;
    (void)used_bytes(accounting_, used);
    const auto free = accounting_.resource_limit_bytes - accounting_.resource_limit_used;
    // Accounting validation guarantees this sum cannot wrap.
    return std::min(heap_capacity, used + free);
}

std::uint64_t ProcessRecord::alias_extra_size() const noexcept {
    return configuration_.alias_extra_enabled ? base_alias_capacity : 0;
}

std::vector<MemoryBlock> ProcessRecord::memory_map() const {
    const std::lock_guard lock(mutex_);
    return {blocks_.begin(), blocks_.end()};
}

AddressRange ProcessRecord::heap_region() const noexcept {
    return {configuration_.heap_address, configuration_.heap_size};
}

AddressRange ProcessRecord::alias_region() const noexcept {
    return {configuration_.alias_address, configuration_.base_alias_size + alias_extra_size()};
}

AddressRange ProcessRecord::stack_region() const noexcept {
    return {configuration_.stack_address, configuration_.stack_size};
}

ThreadRecord::ThreadRecord(std::shared_ptr<ProcessRecord> process, std::uint64_t id,
                           std::uint32_t priority)
    : process_(std::move(process)), id_(id), priority_(priority) {
    if (!process_ || id == 0 || priority > 63) {
        throw std::invalid_argument("Invalid guest thread query record.");
    }
    affinity_ = process_->configuration().allowed_core_mask;
}

bool ThreadRecord::set_priority(std::uint32_t priority) noexcept {
    if (priority > 63) return false;
    priority_.store(priority, std::memory_order_relaxed);
    return true;
}

std::uint64_t ThreadRecord::affinity_mask() const noexcept {
    const std::lock_guard lock(scheduling_mutex_);
    return affinity_;
}

void ThreadRecord::set_affinity_mask(std::uint64_t mask) noexcept {
    const std::lock_guard lock(scheduling_mutex_);
    affinity_ = mask;
    if (current_processor_ && (mask & (std::uint64_t{1} << *current_processor_)) == 0)
        current_processor_.reset();
}

std::optional<std::uint32_t> ThreadRecord::current_processor() const noexcept {
    const std::lock_guard lock(scheduling_mutex_);
    return current_processor_;
}

bool ThreadRecord::bind_current_processor(std::uint32_t processor) noexcept {
    if (processor >= 64) return false;
    const std::lock_guard lock(scheduling_mutex_);
    const auto bit = std::uint64_t{1} << processor;
    if ((affinity_ & process_->configuration().allowed_core_mask & bit) == 0) return false;
    current_processor_ = processor;
    return true;
}

void ThreadRecord::unbind_current_processor() noexcept {
    const std::lock_guard lock(scheduling_mutex_);
    current_processor_.reset();
}

void ThreadRecord::note_call(std::uint16_t svc, bool deferred, std::uint64_t x0, std::uint64_t x1,
                             std::uint64_t x2) noexcept {
    note_svc_.store(svc, std::memory_order_relaxed);
    note_deferred_.store(deferred, std::memory_order_relaxed);
    note_arguments_[0].store(x0, std::memory_order_relaxed);
    note_arguments_[1].store(x1, std::memory_order_relaxed);
    note_arguments_[2].store(x2, std::memory_order_relaxed);
    phase_.store(CallPhase::call, std::memory_order_relaxed);
}

void ThreadRecord::note_guest(std::uint64_t grant_address, std::uint32_t grant_tag,
                              std::uint32_t grants) noexcept {
    note_grant_address_.store(grant_address, std::memory_order_relaxed);
    note_grant_tag_.store(grant_tag, std::memory_order_relaxed);
    note_grants_.store(grants, std::memory_order_relaxed);
    phase_.store(CallPhase::guest, std::memory_order_relaxed);
}

void ThreadRecord::note_retained(std::uint64_t grant_address, std::uint32_t grant_tag,
                                 std::uint32_t grants) noexcept {
    note_grant_address_.store(grant_address, std::memory_order_relaxed);
    note_grant_tag_.store(grant_tag, std::memory_order_relaxed);
    note_grants_.store(grants, std::memory_order_relaxed);
    phase_.store(CallPhase::retained, std::memory_order_relaxed);
}

ThreadRecord::CallNote ThreadRecord::call_note() const noexcept {
    CallNote note;
    note.phase = phase_.load(std::memory_order_relaxed);
    note.svc = note_svc_.load(std::memory_order_relaxed);
    note.deferred = note_deferred_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i != 3; ++i) note.arguments[i] = note_arguments_[i].load(std::memory_order_relaxed);
    note.grant_address = note_grant_address_.load(std::memory_order_relaxed);
    note.grant_tag = note_grant_tag_.load(std::memory_order_relaxed);
    note.grants = note_grants_.load(std::memory_order_relaxed);
    return note;
}

void ThreadRecord::mark_exited() noexcept {
    {
        const std::lock_guard lock(exit_mutex_);
        exited_ = true;
        for (const auto& entry : exit_waiters_) entry.waiter->wake(entry.index);
    }
    exit_changed_.notify_all();
}

bool ThreadRecord::exited() const noexcept {
    const std::lock_guard lock(exit_mutex_);
    return exited_;
}

bool ThreadRecord::wait_for_exit(std::int64_t timeout_nanoseconds) const {
    return wait_for_exit(timeout_nanoseconds, {}) == ThreadWaitOutcome::exited;
}

ThreadWaitOutcome ThreadRecord::wait_for_exit(std::int64_t timeout_nanoseconds,
                                              std::stop_token stop) const {
    std::unique_lock lock(exit_mutex_);
    if (exited_) return ThreadWaitOutcome::exited;
    if (stop.stop_requested()) return ThreadWaitOutcome::cancelled;
    if (timeout_nanoseconds == 0) return ThreadWaitOutcome::timed_out;
    bool cancelled = false;
    bool completed = false;
    lock.unlock();
    std::stop_callback cancellation(stop, [&] {
        const std::lock_guard callback_lock(exit_mutex_);
        if (exited_ || completed) return;
        cancelled = true;
        exit_changed_.notify_all();
    });
    lock.lock();
    const auto ready = [this, &cancelled] { return exited_ || cancelled; };
    bool finished = true;
    if (timeout_nanoseconds < 0)
        exit_changed_.wait(lock, ready);
    else
        finished = exit_changed_.wait_for(lock, std::chrono::nanoseconds(timeout_nanoseconds),
                                          ready);
    const auto result = !finished ? ThreadWaitOutcome::timed_out
                                 : cancelled ? ThreadWaitOutcome::cancelled
                                             : ThreadWaitOutcome::exited;
    completed = true;
    lock.unlock();
    return result;
}

bool StartupQueries::read_guest_word(std::uint64_t address, std::uint32_t& value) const noexcept {
    if (copy_out_.read_word == nullptr) return false;
    std::array<std::byte, 4> raw{};
    if (!copy_out_.read_word(copy_out_.context, address, raw)) return false;
    value = std::uint32_t{std::to_integer<unsigned>(raw[0])} |
            (std::uint32_t{std::to_integer<unsigned>(raw[1])} << 8) |
            (std::uint32_t{std::to_integer<unsigned>(raw[2])} << 16) |
            (std::uint32_t{std::to_integer<unsigned>(raw[3])} << 24);
    return true;
}

bool StartupQueries::write_guest_word(std::uint64_t address, std::uint32_t value) const noexcept {
    if (copy_out_.write_word == nullptr) return false;
    std::array<std::byte, 4> raw{};
    for (unsigned index = 0; index != 4; ++index)
        raw[index] = static_cast<std::byte>((value >> (8 * index)) & 255);
    return copy_out_.write_word(copy_out_.context, address, raw);
}

ProcessWideKey::ExchangeResult StartupQueries::compare_exchange_guest_word(
    std::uint64_t address, std::uint32_t& expected, std::uint32_t desired) const noexcept {
    if (copy_out_.compare_exchange_word == nullptr)
        return ProcessWideKey::ExchangeResult::invalid_address;
    return copy_out_.compare_exchange_word(copy_out_.context, address, expected, desired);
}

bool StartupQueries::read_guest_word_thunk(void* context, std::uint64_t address,
                                           std::uint32_t& value) noexcept {
    const auto* queries = static_cast<const StartupQueries*>(context);
    return queries != nullptr && queries->read_guest_word(address, value);
}

bool StartupQueries::write_guest_word_thunk(void* context, std::uint64_t address,
                                            std::uint32_t value) noexcept {
    const auto* queries = static_cast<const StartupQueries*>(context);
    return queries != nullptr && queries->write_guest_word(address, value);
}

ProcessWideKey::ExchangeResult StartupQueries::compare_exchange_word_thunk(
    void* context, std::uint64_t address, std::uint32_t& expected,
    std::uint32_t desired) noexcept {
    const auto* queries = static_cast<const StartupQueries*>(context);
    return queries != nullptr ? queries->compare_exchange_guest_word(address, expected, desired)
                              : ProcessWideKey::ExchangeResult::invalid_address;
}

ProcessWideKey::GuestWords StartupQueries::guest_words() const noexcept {
    return {read_guest_word_thunk, write_guest_word_thunk, const_cast<StartupQueries*>(this),
            compare_exchange_word_thunk};
}

StartupQueries::StartupQueries(std::shared_ptr<ProcessRecord> process, HandleTable& handles,
                               std::shared_ptr<ThreadRecord> thread, GuestCopyOut copy_out,
                               std::shared_ptr<Arbitration> arbitration,
                               std::shared_ptr<ProcessWideKey> process_wide_keys,
                               std::stop_token cancellation)
    : process_(std::move(process)), handles_(&handles), thread_(std::move(thread)),
      copy_out_(copy_out),
      arbitration_(arbitration ? std::move(arbitration) : std::make_shared<Arbitration>()),
      keys_(process_wide_keys ? std::move(process_wide_keys)
                              : std::make_shared<ProcessWideKey>(arbitration_)), cancellation_(cancellation) {
    if (!process_ || !thread_ || thread_->process() != process_) {
        throw std::invalid_argument("Invalid current process or thread query context.");
    }
}

std::size_t describe_break(std::uint64_t reason, std::uint64_t address, std::uint64_t size,
                           char* buffer, std::size_t capacity) noexcept {
    if (buffer == nullptr || capacity == 0) return 0;
    const int written = std::snprintf(buffer, capacity,
                                      "The guest called svcBreak: reason=%llu address=%llx size=%llu.",
                                      static_cast<unsigned long long>(reason),
                                      static_cast<unsigned long long>(address),
                                      static_cast<unsigned long long>(size));
    if (written <= 0) {
        buffer[0] = '\0';
        return 0;
    }
    const auto length = static_cast<std::size_t>(written);
    return length < capacity ? length : capacity - 1;
}

QueryDisposition StartupQueries::dispatch(std::uint16_t supervisor_call,
                                          QueryRegisters& registers) const {
    unsupported_detail_ = nullptr;
    switch (supervisor_call) {
    case 0x29:
        return get_info(registers);
    case 0x02:
        // svcSetMemoryPermissions(address, size, permission). nn::os::detail::SetMemoryPermissionImpl
        // reaches the wrapper at sdk+0x1755ac with (address, size, permission), and nn::ro's
        // ProtectRelro uses it to make the RELRO range read-only.
        return set_memory_permission(registers);
    case 0x06:
        if (copy_out_.write_word == nullptr) return QueryDisposition::unsupported_operation;
        query_memory(registers);
        return QueryDisposition::handled;
    case 0x0c:
        query_thread(registers, false);
        return QueryDisposition::handled;
    case 0x0d:
        return set_thread_priority(registers);
    case 0x0e:
        return get_thread_core_mask(registers);
    case 0x0f:
        return set_thread_core_mask(registers);
    case 0x10: {
        // This SVC returns the runtime-assigned virtual processor directly, without a
        // Result prefix. Process permissions alone do not establish a current processor.
        const auto processor = thread_->current_processor();
        if (!processor) return QueryDisposition::unsupported_operation;
        registers[0] = *processor;
        std::fill(registers.begin() + 1, registers.end(), 0);
        return QueryDisposition::handled;
    }
    case 0x15:
        return create_transfer_memory(registers);
    case 0x12: {
        // svcClearEvent(handle): clears an event whether or not it was signalled.
        // Any other object, or no object, is an invalid handle.
        const auto event = handles_->lookup_as<Event>(static_cast<Handle>(registers[0]));
        if (event) (void)event->reset();
        finish(registers, event ? 0 : 0xe401, 0);
        return QueryDisposition::handled;
    }
    case 0x17: {
        // svcResetSignal(handle): clears a signalled readable event.
        const auto result = reset_event(*handles_, static_cast<Handle>(registers[0]));
        if (result.disposition != EventDisposition::handled) return QueryDisposition::unsupported_operation;
        finish(registers, result.result, 0);
        return QueryDisposition::handled;
    }
    case 0x16: {
        // svcCloseHandle(handle): the table drops its reference; the object
        // lives on while another owner holds it.
        const bool closed = handles_->close(static_cast<Handle>(registers[0]));
        finish(registers, closed ? 0 : invalid_handle, 0);
        return QueryDisposition::handled;
    }
    case 0x18:
        return wait_synchronization(registers);
    case 0x19:
        return cancel_synchronization(registers);
    case 0x1a:
        return arbitrate_lock(registers);
    case 0x1b:
        return arbitrate_unlock(registers);
    case 0x1c:
        return arbitrate(registers);
    case 0x1d:
        return signal(registers);
    case 0x34:
        return wait_for_address(registers);
    case 0x35:
        return signal_to_address(registers);
    case 0x25:
        query_thread(registers, true);
        return QueryDisposition::handled;
    default:
        return QueryDisposition::unsupported_operation;
    }
}


QueryDisposition StartupQueries::wait_synchronization(QueryRegisters& registers) const {
    const auto original_index = registers[1];
    const auto count = static_cast<std::uint32_t>(registers[2]);
    if (count > 64) {
        finish(registers, 0xee01, original_index);
        return QueryDisposition::handled;
    }
    std::array<Handle, 64> handles{};
    if (count != 0) {
        const auto bytes = std::uint64_t{count} * sizeof(Handle);
        if (original_index >= ProcessRecord::query_end ||
            bytes > ProcessRecord::query_end - original_index) {
            finish(registers, invalid_copy, original_index);
            return QueryDisposition::handled;
        }
        if (copy_out_.read_bytes != nullptr) {
            std::array<std::byte, 64 * sizeof(Handle)> copied{};
            if (!copy_out_.read_bytes(copy_out_.context, original_index,
                                      std::span(copied).first(static_cast<std::size_t>(bytes)))) {
                finish(registers, invalid_copy, original_index);
                return QueryDisposition::handled;
            }
            for (std::uint32_t index = 0; index != count; ++index) {
                for (unsigned byte = 0; byte != sizeof(Handle); ++byte)
                    handles[index] |= std::to_integer<std::uint32_t>(
                        copied[index * sizeof(Handle) + byte]) << (byte * 8);
            }
        } else {
            // Handle arrays have no guest alignment requirement. The synchronization
            // word backend is a bounded fallback only when its alignment contract holds.
            if (copy_out_.read_word == nullptr || (original_index & 3) != 0) {
                unsupported_detail_ = "The wait's handle array needs guest memory access.";
                return QueryDisposition::unsupported_operation;
            }
            for (std::uint32_t index = 0; index != count; ++index) {
                if (!read_guest_word(original_index + std::uint64_t{index} * sizeof(Handle),
                                     handles[index])) {
                    finish(registers, invalid_copy, original_index);
                    return QueryDisposition::handled;
                }
            }
        }
    }
    const auto result = wait_event(*handles_, std::span<const Handle>(handles.data(), count),
        static_cast<std::int64_t>(registers[3]), *thread_, cancellation_);
    if (result.disposition == EventDisposition::cancelled) return QueryDisposition::cancelled;
    if (result.disposition == EventDisposition::unsupported) {
        unsupported_detail_ = result.detail ? result.detail : "The wait is not implemented.";
        return QueryDisposition::unsupported_operation;
    }
    const auto index = result.index ? (original_index & high_word) | *result.index : original_index;
    finish(registers, result.result, index);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::cancel_synchronization(QueryRegisters& registers) const {
    const auto result = armigo::horizon::cancel_synchronization(
        *handles_, static_cast<Handle>(registers[0]), *thread_);
    finish(registers, result.result, 0);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::set_memory_permission(QueryRegisters& registers) const {
    // svcSetMemoryPermissions(address, size, permission): the guest may name any page-aligned
    // subrange of memory it owns, so the blocks it covers are split and only the covered part keeps
    // the new permission. A partial range is normal: ProtectRelro protects one part of a segment.
    const auto address = registers[0];
    const auto size = registers[1];
    const auto permission = registers[2];
    constexpr std::uint64_t page = 4096;
    if (size == 0 || (size & (page - 1)) != 0 || (address & (page - 1)) != 0 || permission > 3 ||
        address > result_limit || size > result_limit - address) {
        finish(registers, invalid_address, address);
        return QueryDisposition::handled;
    }
    const AddressRange range{address, size};
    const auto blocks = process_->memory_map();
    const auto memory = process_memory::uniform(blocks, range);
    if (!memory) {
        finish(registers, invalid_address, address);
        return QueryDisposition::handled;
    }
    auto changed = *memory;
    changed.permissions = static_cast<std::uint8_t>(permission);
    const auto replacement = process_memory::replace(blocks, range, changed);
    if (process_->replace_memory_map(replacement) != ProcessStateError::none) {
        finish(registers, invalid_address, address);
        return QueryDisposition::handled;
    }
    finish(registers, 0, 0);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::create_transfer_memory(QueryRegisters& registers) const {
    // svcCreateTransferMemory(out_handle, address, size, permission). The kernel
    // returns the new handle in x1 and the SDK wrapper stores it through the out
    // pointer, so only x0 and x1 carry results here.
    const auto address = registers[1];
    const auto size = registers[2];
    const auto permission = registers[3];
    constexpr std::uint64_t page = 4096;
    if (size == 0 || (size & (page - 1)) != 0 || (address & (page - 1)) != 0 ||
        permission > 3 || address > result_limit || size > result_limit - address) {
        finish(registers, invalid_address, address);
        return QueryDisposition::handled;
    }
    // Internal metadata boundaries do not divide one compatible owned range.
    // Uniform validation still rejects holes and every public state mismatch.
    if (!process_memory::uniform(process_->memory_map(), {address, size})) {
        finish(registers, invalid_address, address);
        return QueryDisposition::handled;
    }
    auto record = std::make_shared<TransferMemoryRecord>(AddressRange{address, size},
                                                         static_cast<std::uint32_t>(permission));
    const auto handle = handles_->insert(std::move(record));
    if (!handle) {
        finish(registers, out_of_handles, 0);
        return QueryDisposition::handled;
    }
    finish(registers, 0, 0);
    registers[1] = *handle;
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::arbitrate_lock(QueryRegisters& registers) const {
    // Ordinary lock waiters and signalled CV waiters share one ownership handoff queue.
    // Checking the owner's tag and enrolling are serialized against every kernel unlock.
    const auto owner = static_cast<std::uint32_t>(registers[0]);
    const auto address = registers[1];
    const auto tag = static_cast<std::uint32_t>(registers[2]);
    if (address < 0x200000 || address >= ProcessRecord::query_end) {
        finish(registers, 0xd401, 0);
        return QueryDisposition::handled;
    }
    if ((address & 3) != 0) {
        finish(registers, invalid_address, 0);
        return QueryDisposition::handled;
    }
    if (copy_out_.read_word == nullptr) return QueryDisposition::unsupported_operation;
    auto words = guest_words();
    words.validate_owner = [](void* context, std::uint32_t handle) noexcept {
        const auto* queries = static_cast<const StartupQueries*>(context);
        return queries->handles_->lookup_as<ThreadRecord>(handle) != nullptr;
    };
    const auto outcome = keys_->lock(words, address, owner, tag, cancellation_,
                                     thread_ ? thread_->priority() : 32);
    if (outcome == ProcessWideKey::LockOutcome::cancelled) return QueryDisposition::cancelled;
    finish(registers, outcome == ProcessWideKey::LockOutcome::invalid_address
                          ? 0xd401 : outcome == ProcessWideKey::LockOutcome::invalid_owner
                              ? invalid_handle : 0, 0);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::arbitrate_unlock(QueryRegisters& registers) const {
    // svcArbitrateUnlock(mutex_word): hand the tag to one ordinary or signalled CV waiter,
    // preserving the waiter marker when more handoffs remain; otherwise clear the word.
    const auto address = registers[0];
    if (address < 0x200000 || address >= ProcessRecord::query_end) {
        finish(registers, 0xd401, 0);
        return QueryDisposition::handled;
    }
    if ((address & 3) != 0) {
        finish(registers, invalid_address, 0);
        return QueryDisposition::handled;
    }
    if (copy_out_.write_word == nullptr) return QueryDisposition::unsupported_operation;
    if (!keys_->unlock(guest_words(), address)) {
        finish(registers, 0xd401, 0);
        return QueryDisposition::handled;
    }
    finish(registers, 0, 0);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::arbitrate(QueryRegisters& registers) const {
    // svcWaitProcessWideKeyAtomic(mutex_word, condition_word, tag, timeout_nanoseconds):
    // the kernel releases the mutex while the thread waits, arms the condition word so a
    // signaller knows to call svcSignalProcessWideKey, and blocks until signalled. A timed-out
    // wait reports the kernel timeout result and does not take the mutex back.
    constexpr std::uint32_t timed_out = 0xea01;
    const auto mutex = registers[0];
    if (mutex < 0x200000 || mutex >= ProcessRecord::query_end) {
        finish(registers, 0xd401, 0);
        return QueryDisposition::handled;
    }
    if ((mutex & 3) != 0) {
        finish(registers, invalid_address, 0);
        return QueryDisposition::handled;
    }
    if (copy_out_.write_word == nullptr) return QueryDisposition::unsupported_operation;
    const auto key = registers[1] & ~std::uint64_t{3};
    const auto tag = static_cast<std::uint32_t>(registers[2]);
    const auto timeout = static_cast<std::int64_t>(registers[3]);
    switch (keys_->wait(guest_words(), mutex, key, tag, timeout, cancellation_,
                        thread_ ? thread_->priority() : 32)) {
    case ProcessWideKey::WaitOutcome::signaled:
        finish(registers, 0, 0);
        break;
    case ProcessWideKey::WaitOutcome::timed_out:
        finish(registers, timed_out, 0);
        break;
    case ProcessWideKey::WaitOutcome::invalid_address:
        finish(registers, 0xd401, 0);
        break;
    case ProcessWideKey::WaitOutcome::invalid_state:
        finish(registers, 0xfa01, 0);
        break;
    case ProcessWideKey::WaitOutcome::cancelled:
        return QueryDisposition::cancelled;
    }
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::signal(QueryRegisters& registers) const {
    // svcSignalProcessWideKey(condition_word, count): a count of zero or less releases every
    // waiter of the key. The kernel clears the condition word once no waiter is left. The
    // handoff is an atomic compare-exchange; without that backend the SVC is reported as an
    // unsupported operation and no queue, register or guest word changes.
    if (copy_out_.write_word == nullptr || copy_out_.compare_exchange_word == nullptr)
        return QueryDisposition::unsupported_operation;
    const auto key = registers[0] & ~std::uint64_t{3};
    const auto count = static_cast<std::int32_t>(static_cast<std::uint32_t>(registers[1]));
    auto words = guest_words();
    words.validate_owner = [](void* context, std::uint32_t handle) noexcept {
        const auto* queries = static_cast<const StartupQueries*>(context);
        return queries->handles_->lookup_as<ThreadRecord>(handle) != nullptr;
    };
    if (keys_->signal(words, key, count) !=
        ProcessWideKey::SignalOutcome::signaled) {
        return QueryDisposition::unsupported_operation;
    }
    finish(registers, 0, 0);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::wait_for_address(QueryRegisters& registers) const {
    // The comparison (and conditional decrement) must share the queue lock with signals.
    constexpr std::uint32_t invalid_state = 0xfa01;
    constexpr std::uint32_t timed_out = 0xea01;
    const auto address = registers[0];
    const auto type = static_cast<std::uint32_t>(registers[1]);
    if (address < 0x200000 || address >= ProcessRecord::query_end) {
        finish(registers, 0xd401, address);
        return QueryDisposition::handled;
    }
    if ((address & (type == 3 ? 7 : 3)) != 0) {
        finish(registers, invalid_address, address);
        return QueryDisposition::handled;
    }
    if (type > 3) {
        finish(registers, invalid_info, address);
        return QueryDisposition::handled;
    }
    if ((type == 3 && copy_out_.read_double_word == nullptr) ||
        (type != 3 && copy_out_.read_word == nullptr) ||
        (type == 1 && copy_out_.compare_exchange_word == nullptr))
        return QueryDisposition::unsupported_operation;
    struct Check {
        const StartupQueries* queries;
        std::uint64_t address;
        std::uint64_t value;
        std::uint32_t type;
    } check{this, address, registers[2], type};
    const auto outcome = arbitration_->wait_transaction(
        address, static_cast<std::int64_t>(registers[3]), [](void* raw) noexcept {
            const auto& check = *static_cast<Check*>(raw);
            if (check.type == 3) {
                std::uint64_t observed = 0;
                const auto& copy = check.queries->copy_out_;
                if (!copy.read_double_word(copy.context, check.address, observed))
                    return ArbitrateOutcome::unreadable;
                return observed == check.value ? ArbitrateOutcome::succeeded
                                               : ArbitrateOutcome::value_mismatch;
            }
            std::uint32_t observed = 0;
            if (!check.queries->read_guest_word(check.address, observed))
                return ArbitrateOutcome::unreadable;
            const auto value = static_cast<std::int32_t>(static_cast<std::uint32_t>(check.value));
            for (;;) {
                const auto signed_word = static_cast<std::int32_t>(observed);
                if (check.type == 2 ? signed_word != value : signed_word >= value)
                    return ArbitrateOutcome::value_mismatch;
                if (check.type != 1) return ArbitrateOutcome::succeeded;
                // Unsigned subtraction preserves the guest's modulo-32-bit arithmetic.
                const auto result = check.queries->compare_exchange_guest_word(
                    check.address, observed, observed - 1u);
                if (result == ProcessWideKey::ExchangeResult::exchanged)
                    return ArbitrateOutcome::succeeded;
                if (result == ProcessWideKey::ExchangeResult::invalid_address)
                    return ArbitrateOutcome::unreadable;
                // A guest CAS raced us. Compare its new value before retrying the decrement.
            }
        }, &check, cancellation_, thread_ ? thread_->priority() : 32);
    if (outcome == ArbitrateOutcome::cancelled) return QueryDisposition::cancelled;
    finish(registers, outcome == ArbitrateOutcome::unreadable ? 0xd401 :
                      outcome == ArbitrateOutcome::value_mismatch ? invalid_state :
                      outcome == ArbitrateOutcome::timed_out ? timed_out : 0, address);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::signal_to_address(QueryRegisters& registers) const {
    constexpr std::uint32_t invalid_state = 0xfa01;
    const auto address = registers[0];
    const auto type = static_cast<std::uint32_t>(registers[1]);
    const auto value = static_cast<std::uint32_t>(registers[2]);
    const auto count = static_cast<std::int32_t>(static_cast<std::uint32_t>(registers[3]));
    if (address < 0x200000 || address >= ProcessRecord::query_end) {
        finish(registers, 0xd401, address);
        return QueryDisposition::handled;
    }
    if ((address & 3) != 0) {
        finish(registers, invalid_address, address);
        return QueryDisposition::handled;
    }
    if (type > 2) {
        finish(registers, invalid_info, address);
        return QueryDisposition::handled;
    }
    // Plain signals only operate on the wait tree; they do not access the guest word.
    if (type != 0 && (copy_out_.read_word == nullptr ||
                      copy_out_.compare_exchange_word == nullptr))
        return QueryDisposition::unsupported_operation;
    struct Check {
        const StartupQueries* queries;
        std::uint64_t address;
        std::uint32_t type;
        std::uint32_t value;
        std::int32_t count;
    } check{this, address, type, value, count};
    const auto outcome = arbitration_->signal_transaction(
        address, count <= 0 ? std::numeric_limits<std::size_t>::max()
                           : static_cast<std::size_t>(count),
        [](void* raw, std::size_t waiting) noexcept {
            const auto& check = *static_cast<Check*>(raw);
            if (check.type == 0) return ArbitrateOutcome::succeeded;
            auto desired = check.value + 1u;
            if (check.type == 2 && waiting != 0) {
                // Horizon 20.5.0 subtracts one when all present waiters are selected,
                // including zero/negative counts. A partial selection only compares.
                desired = check.count <= 0 || waiting <= static_cast<std::size_t>(check.count)
                    ? check.value - 1u : check.value;
            }
            if (desired == check.value) {
                std::uint32_t observed = 0;
                if (!check.queries->read_guest_word(check.address, observed))
                    return ArbitrateOutcome::unreadable;
                return observed == check.value ? ArbitrateOutcome::succeeded
                                               : ArbitrateOutcome::value_mismatch;
            }
            auto expected = check.value;
            const auto result = check.queries->compare_exchange_guest_word(
                check.address, expected, desired);
            switch (result) {
            case ProcessWideKey::ExchangeResult::exchanged:
                return ArbitrateOutcome::succeeded;
            case ProcessWideKey::ExchangeResult::mismatch:
                return ArbitrateOutcome::value_mismatch;
            case ProcessWideKey::ExchangeResult::invalid_address:
                return ArbitrateOutcome::unreadable;
            }
            return ArbitrateOutcome::unreadable;
        }, &check);
    finish(registers, outcome == ArbitrateOutcome::unreadable ? 0xd401 :
                      outcome == ArbitrateOutcome::value_mismatch ? invalid_state : 0, address);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::get_info(QueryRegisters& registers) const {
    const auto original = registers[1];
    const auto type = static_cast<std::uint32_t>(original);
    const auto handle = static_cast<Handle>(registers[2]);
    const auto sub_id = registers[3];
    if (rejected_info(type)) {
        finish(registers, invalid_info, original);
        return QueryDisposition::handled;
    }
    if (!supported_info(type)) return QueryDisposition::unsupported_operation;
    if (type == 20) {
        // svcGetInfo(UserExceptionContextAddress, handle, 0) resolves a process
        // handle and reports that process's user exception context address. The
        // SDK passes the current-process pseudo-handle.
        if (sub_id != 0) {
            finish(registers, invalid_combination, original);
            return QueryDisposition::handled;
        }
        const auto current = handle == current_process_handle || handle == 0;
        const auto process = current ? process_ : handles_->lookup_as<ProcessRecord>(handle);
        if (!process) {
            finish(registers, invalid_handle, original);
            return QueryDisposition::handled;
        }
        const auto address = process->user_exception_context();
        if (address == 0) return QueryDisposition::unsupported_operation;
        finish(registers, 0, address);
        return QueryDisposition::handled;
    }
    if (type == 11) {
        const auto result = handle != 0 ? invalid_handle :
                            sub_id >= 4 ? invalid_combination : 0;
        finish(registers, result, result != 0 ? original :
            process_->configuration().entropy[static_cast<std::size_t>(sub_id)]);
        return QueryDisposition::handled;
    }
    if (sub_id != 0) {
        finish(registers, invalid_combination, original);
        return QueryDisposition::handled;
    }
    // Info types that describe the calling process take the current-process pseudo-handle,
    // and the firmware also accepts a zero handle for them (svcGetInfo(&out, 8, 0, 0) is how
    // the SDK asks whether a debugger is attached).
    // A zero handle describes the calling process only for a plain, supported info type;
    // a type with extra bits set still has to name a real process.
    const auto current = handle == current_process_handle ||
                         (handle == 0 && original == type && supported_info(type));
    const auto process = current ? process_ : handles_->lookup_as<ProcessRecord>(handle);
    if (!process) {
        finish(registers, invalid_handle, original);
        return QueryDisposition::handled;
    }
    std::uint64_t value = 0;
    switch (type) {
    case 0: value = process->configuration().allowed_core_mask; break;
    case 2: value = process->alias_region().address; break;
    case 3: value = process->alias_region().size; break;
    case 4: value = process->heap_region().address; break;
    case 5: value = process->heap_region().size; break;
    // IsCurrentProcessBeingDebugged: this runtime attaches no debugger to the guest.
    case 8: value = 0; break;
    case 12: value = process->configuration().allocation_region.address; break;
    case 13: value = process->configuration().allocation_region.size; break;
    case 14: value = process->stack_region().address; break;
    case 15: value = process->stack_region().size; break;
    case 16: value = 0; break; // Only the explicit no-private-resource profile is supported.
    case 21: value = process->total_memory(); break;
    case 22: value = process->used_memory(); break;
    case 28: value = process->alias_extra_size(); break;
    default: return QueryDisposition::unsupported_operation;
    }
    finish(registers, 0, value);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::set_thread_priority(QueryRegisters& registers) const {
    // svcSetThreadPriority(handle, priority): x0 names the thread and w1 is the
    // guest priority. The guest keeps its own scheduling policy, so the request
    // only updates the recorded priority.
    const auto handle = static_cast<Handle>(registers[0]);
    const auto thread = handle == current_thread_handle ? thread_
                                                        : handles_->lookup_as<ThreadRecord>(handle);
    if (!thread) {
        finish(registers, invalid_handle, registers[0]);
        return QueryDisposition::handled;
    }
    if (!thread->set_priority(static_cast<std::uint32_t>(registers[1])))
        return QueryDisposition::unsupported_operation;
    finish(registers, 0, 0);
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::get_thread_core_mask(QueryRegisters& registers) const {
    // svcGetThreadCoreMask: the SDK wrapper passes the two guest output pointers
    // in x0/x1 and the thread handle in x2; the kernel returns the ideal core in
    // x1 and the affinity mask in x2.
    const auto handle = static_cast<Handle>(registers[2]);
    const auto thread = handle == current_thread_handle ? thread_ :
                        handles_->lookup_as<ThreadRecord>(handle);
    if (!thread) {
        registers[0] = invalid_handle;
        registers[1] = 0;
        registers[2] = 0;
        return QueryDisposition::handled;
    }
    registers[0] = 0;
    registers[1] = 0;  // the runtime does not pin an ideal core
    registers[2] = thread->affinity_mask();
    return QueryDisposition::handled;
}

QueryDisposition StartupQueries::set_thread_core_mask(QueryRegisters& registers) const {
    // svcSetThreadCoreMask(handle, ideal_core, affinity_mask): the guest keeps its
    // own scheduling policy; store the mask the process is allowed to use.
    const auto handle = static_cast<Handle>(registers[0]);
    const auto thread = handle == current_thread_handle ? thread_ :
                        handles_->lookup_as<ThreadRecord>(handle);
    if (!thread) {
        finish(registers, invalid_handle, registers[0]);
        return QueryDisposition::handled;
    }
    const auto allowed = process_->configuration().allowed_core_mask;
    const auto requested = registers[2] & allowed;
    if (requested == 0) {
        finish(registers, invalid_handle, registers[0]);
        return QueryDisposition::handled;
    }
    thread->set_affinity_mask(requested);
    finish(registers, 0, 0);
    return QueryDisposition::handled;
}

void StartupQueries::query_thread(QueryRegisters& registers, bool identifier) const {
    const auto original = registers[1];
    const auto handle = static_cast<Handle>(original);
    const auto thread = handle == current_thread_handle ? thread_ :
                        handles_->lookup_as<ThreadRecord>(handle);
    if (!thread) {
        finish(registers, invalid_handle, original);
        return;
    }
    finish(registers, 0, identifier ? thread->id() : (original & high_word) | thread->priority());
}

void StartupQueries::query_memory(QueryRegisters& registers) const {
    const auto destination = registers[0];
    const auto page_info = registers[1] & high_word;
    const auto info = encode_memory_info(process_->query_memory(registers[2]));
    bool copied = true;
    for (std::size_t offset = 0; copied && offset < info.size(); offset += 4) {
        if (destination > std::numeric_limits<std::uint64_t>::max() - offset) {
            copied = false;
            break;
        }
        const auto address = destination + offset;
        if (!process_->writable(address, 4)) {
            copied = false;
            break;
        }
        copied = copy_out_.write_word(copy_out_.context, address,
            std::span<const std::byte, 4>{info.data() + offset, 4});
    }
    finish(registers, copied ? 0 : invalid_copy, page_info);
}

} // namespace armigo::horizon
