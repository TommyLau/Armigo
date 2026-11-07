// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/common/guest_platform.h>

#include <armigo/horizon/arbitration.h>
#include <armigo/horizon/event.h>
#include <armigo/horizon/handle_table.h>
#include <armigo/horizon/process_wide_key.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <vector>

namespace armigo::horizon {

inline constexpr Handle current_process_handle = 0xffff8001u;
inline constexpr Handle current_thread_handle = 0xffff8000u;
inline constexpr std::uint64_t guest_page_size = 4096;
// Owner-approved, unverified Switch 2 allocation granule. Query 33 and physical
// map/unmap checks share this value; a console capture may establish 64 KiB.
inline constexpr std::uint64_t switch2_physical_mapping_granule = guest_page_size;

enum class Baseline { horizon_20_5_0, horizon_23_0_1 };

struct AddressRange {
    std::uint64_t address = 0;
    std::uint64_t size = 0;
};

// Public memory types used by the supported startup mappings.
enum class MemoryType : std::uint8_t {
    free = 0,
    code = 3,
    code_data = 4,
    normal = 5,
    shared = 6,
    stack = 11,
    thread_local_storage = 12,
    reserved = 16,
};

struct MemoryBlock {
    AddressRange range;
    MemoryType type = MemoryType::free;
    std::uint8_t attributes = 0;
    std::uint8_t permissions = 0;
    std::uint16_t ipc_references = 0;
    std::uint16_t device_references = 0;
    // Internal node-boundary barriers; never exposed in the MemoryInfo wire record.
    std::uint8_t merge_disable = 0;
};

// Range edits carry only the changed metadata; private processes stage them in
// their shared interval authority before applying native mapping changes.
struct MemoryMapUpdate {
    AddressRange range;
    std::optional<MemoryBlock> replacement;
    std::uint8_t set_merge = 0;
    std::uint8_t clear_merge = 0;
};

// A public MemoryInfo value, not the host layout of the wire structure.
// encode_memory_info writes its exact forty-byte little-endian representation.
using MemoryInfo = MemoryBlock;
std::array<std::byte, 40> encode_memory_info(const MemoryInfo& info) noexcept;

// Supported process profiles use 64-bit, 39-bit address space. The Switch 1
// profile has no private system resource; the selected 23.0.1 reference also
// admits an application-owned resource pool. Placement and resource budgets
// are supplied by the runtime, never selected by a game name.
struct ProcessConfiguration {
    GuestPlatform platform = GuestPlatform::switch1;
    Baseline baseline = Baseline::horizon_20_5_0;
    unsigned address_bits = 39;
    std::uint64_t heap_address = 0;
    std::uint64_t alias_address = 0;
    std::uint64_t stack_address = 0;
    bool alias_extra_enabled = false;
    std::uint64_t allowed_core_mask = 0;
    std::uint64_t private_system_resource_bytes = 0;
    std::array<std::uint64_t, 4> entropy{};
    std::size_t max_memory_blocks = 65536;
    // Runtime-owned reservations. Defaults match the selected device profile;
    // a native host may supply compact extents from its actual allocation.
    std::uint64_t heap_size = std::uint64_t{8} << 30;
    std::uint64_t base_alias_size = std::uint64_t{64} << 30;
    std::uint64_t stack_size = std::uint64_t{2} << 30;
    AddressRange allocation_region{std::uint64_t{128} << 20,
                                   (std::uint64_t{1} << 39) - (std::uint64_t{128} << 20)};
    // Actual native reservation; empty means remote-region queries are unsupported.
    AddressRange remote_region{};
};

struct MemoryAccounting {
    std::uint64_t committed_heap = 0;
    std::uint64_t other_accounted_mapping_bytes = 0;
    std::uint64_t code_bytes = 0;
    std::uint64_t initial_stack_bytes = 0;
    std::uint64_t resource_limit_bytes = 0;
    // May include other processes sharing the same resource limit.
    std::uint64_t resource_limit_used = 0;
};

enum class ProcessStateError {
    none,
    unsupported_profile,
    invalid_layout,
    invalid_accounting,
    invalid_memory_map,
    resource_limit,
    backend_failure,
};

class ProcessRecord;
class MemoryIntervalMap;
struct ProcessBuild {
    ProcessStateError error = ProcessStateError::none;
    std::shared_ptr<ProcessRecord> process;
    bool ok() const noexcept { return error == ProcessStateError::none && process != nullptr; }
};

// Query metadata and accounting only. This object does not reserve or map
// host memory. The runtime must keep its metadata consistent with its actual
// backing and must register the live guest stack before executing the SDK.
class ProcessRecord final : public KernelObject {
public:
    ~ProcessRecord() override;
    static constexpr std::uint64_t query_end = std::uint64_t{1} << 39;
    static constexpr std::uint64_t aslr_start = 128ull * 1024 * 1024;
    static constexpr std::uint64_t heap_capacity = 8ull * 1024 * 1024 * 1024;
    // Unverified Switch 2 application-pool assumption from practical startup
    // evidence; replace with owner-console GetInfo 16/21/22 evidence when available.
    static constexpr std::uint64_t switch2_application_capacity = 9809ull * 1024 * 1024;
    static constexpr std::uint64_t application_capacity(GuestPlatform platform) noexcept {
        return platform == GuestPlatform::switch2 ? switch2_application_capacity : heap_capacity;
    }
    static constexpr std::uint64_t base_alias_capacity = 64ull * 1024 * 1024 * 1024;
    static constexpr std::uint64_t stack_capacity = 2ull * 1024 * 1024 * 1024;

    static ProcessBuild create(const ProcessConfiguration& configuration,
                               const MemoryAccounting& accounting,
                               std::span<const MemoryBlock> blocks = {});
    ProcessStateError set_accounting(const MemoryAccounting& accounting);
    // Legacy metadata replacement only; private profiles require bounded edits.
    ProcessStateError replace_memory_map(std::span<const MemoryBlock> blocks);
    // Prepares and validates both query views before calling apply under the
    // record lock. apply must not reenter this record and must restore partial
    // native work before returning false. Success publishes without allocation.
    ProcessStateError commit_memory_state(std::span<const MemoryBlock> blocks,
        const MemoryAccounting& accounting, bool (*apply)(void*) noexcept, void* context);
    // Explicit diagnostic snapshot; private mutation must use bounded updates.
    std::vector<MemoryBlock> memory_map() const;

    MemoryInfo query_memory(std::uint64_t address) const;
    bool writable(std::uint64_t address, std::size_t size) const;
    std::uint64_t used_memory() const;
    std::uint64_t total_memory() const;
    std::uint64_t used_system_resource() const;
    std::shared_ptr<MemoryIntervalMap> memory_intervals() const noexcept { return private_memory_; }
    ProcessStateError commit_memory_updates(std::span<const MemoryMapUpdate> updates,
        const MemoryAccounting& accounting, bool (*apply)(void*) noexcept, void* context);
    const ProcessConfiguration& configuration() const noexcept { return configuration_; }
    std::uint64_t alias_extra_size() const noexcept;
    AddressRange heap_region() const noexcept;
    AddressRange alias_region() const noexcept;
    AddressRange stack_region() const noexcept;
    // The guest address the runtime reserves for the process's user exception
    // context. The firmware's svcGetInfo(UserExceptionContextAddress) reports
    // this process-level address, not a per-thread one.
    std::uint64_t user_exception_context() const noexcept {
        return user_exception_context_.load(std::memory_order_relaxed);
    }
    void set_user_exception_context(std::uint64_t address) noexcept {
        user_exception_context_.store(address, std::memory_order_relaxed);
    }
    // Host analogue of the process's scheduled count (KProcess+0x248). The kernel advances
    // it whenever a core switches away from one of the process's threads and on every yield
    // it does not skip. The runtime advances it when a guest thread may block, returns from
    // a blocking call, starts a timed sleep or yields to the host.
    std::uint64_t scheduled_count() const noexcept {
        return scheduled_count_.load(std::memory_order_relaxed);
    }
    std::uint64_t note_scheduling() noexcept {
        return scheduled_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

private:
    explicit ProcessRecord(const ProcessConfiguration& configuration);
    ProcessStateError initialize_memory_map(std::span<const MemoryBlock> blocks);
    const ProcessConfiguration configuration_;
    mutable std::mutex mutex_;
    MemoryAccounting accounting_;
    std::vector<MemoryBlock> blocks_;
    std::shared_ptr<MemoryIntervalMap> private_memory_;
    std::atomic<std::uint64_t> user_exception_context_{0};
    std::atomic<std::uint64_t> scheduled_count_{0};
};

struct ThreadSchedulingState {
    std::int32_t ideal_core = -1;
    std::uint64_t affinity_mask = 0;
    std::optional<std::uint32_t> current_processor;
};

// A live kernel thread identity. Running-state and host scheduling are not
// inferred by these read-only queries. Priority updates remain the caller's
// responsibility; the queried value is the guest priority.
enum class ThreadWaitOutcome { exited, timed_out, cancelled };

class ThreadRecord final : public KernelObject {
public:
    ThreadRecord(std::shared_ptr<ProcessRecord> process, std::uint64_t id,
                 std::uint32_t priority);
    const std::shared_ptr<ProcessRecord>& process() const noexcept { return process_; }
    std::uint64_t id() const noexcept { return id_; }
    std::uint32_t priority() const noexcept { return priority_.load(std::memory_order_relaxed); }
    // The handle the creating process received for this thread; zero until the
    // runtime inserts it. A Switch 2 thread finds it in its thread-local region.
    Handle creation_handle() const noexcept { return creation_handle_.load(std::memory_order_acquire); }
    void set_creation_handle(Handle handle) noexcept { creation_handle_.store(handle, std::memory_order_release); }
    bool set_priority(std::uint32_t priority) noexcept;
    // Atomically publish/query the runtime's ideal, affinity and virtual binding.
    ThreadSchedulingState scheduling_state() const noexcept;
    bool configure_scheduling(std::int32_t ideal, std::uint64_t mask,
                              std::optional<std::uint32_t> processor) noexcept;
    std::uint64_t affinity_mask() const noexcept;
    // A mask change invalidates a processor binding it no longer admits. The runtime
    // must establish a new binding before this thread resumes guest execution.
    void set_affinity_mask(std::uint64_t mask) noexcept;
    std::optional<std::uint32_t> current_processor() const noexcept;
    // Records the runtime's virtual processor assignment, not host CPU affinity or
    // scheduling exclusivity. Threads sharing a virtual processor can execute
    // concurrently on this host backend. Failure preserves the binding.
    bool bind_current_processor(std::uint32_t processor) noexcept;
    void unbind_current_processor() noexcept;
    // Marks the thread as finished and wakes every waiter. Idempotent; the thread object
    // outlives its handle while a wait pins it.
    void mark_exited() noexcept;
    bool exited() const noexcept;
    // Waits for the thread to finish. Zero polls, a negative timeout waits without a
    // deadline, and a positive timeout is the guest duration in nanoseconds. Returns true
    // when the thread has exited.
    bool wait_for_exit(std::int64_t timeout_nanoseconds) const;
    // Exit and cancellation race under the same mutex; whichever is observed first wins.
    ThreadWaitOutcome wait_for_exit(std::int64_t timeout_nanoseconds,
                                    std::stop_token stop) const;
    // Guest cancellation affects only an active WaitSynchronization. Otherwise a
    // single pending cancellation survives until a later blocking synchronization wait.
    void cancel_synchronization() noexcept;
    bool synchronization_wait_active() const noexcept;
    // The process's scheduled count when this thread's last yield found no other thread
    // ready (KThread+0x4b8). A thread that has never yielded alone holds no count.
    std::uint64_t lone_yield_count() const noexcept {
        return lone_yield_count_.load(std::memory_order_relaxed);
    }
    void set_lone_yield_count(std::uint64_t count) noexcept {
        lone_yield_count_.store(count, std::memory_order_relaxed);
    }

    // Diagnostic note of the supervisor call this thread is in, read only when a
    // shutdown misses its deadline. Fields are separate relaxed atomics, so a
    // snapshot taken while the thread runs can mix two calls. Nothing waits on it.
    enum class CallPhase : std::uint8_t { guest, call, retained };
    struct CallNote {
        CallPhase phase = CallPhase::guest;
        std::uint16_t svc = 0;
        bool deferred = false;
        std::array<std::uint64_t, 3> arguments{};
        std::uint64_t grant_address = 0;
        std::uint32_t grant_tag = 0;
        std::uint32_t grants = 0;
    };
    void note_call(std::uint16_t svc, bool deferred, std::uint64_t x0, std::uint64_t x1,
                   std::uint64_t x2) noexcept;
    void note_guest(std::uint64_t grant_address, std::uint32_t grant_tag, std::uint32_t grants) noexcept;
    void note_retained(std::uint64_t grant_address, std::uint32_t grant_tag,
                       std::uint32_t grants) noexcept;
    CallNote call_note() const noexcept;

private:
    friend class SynchronizationWait;
    // Lock order is guest_wait_mutex_ followed by the waited object's state mutex.
    // The callback runs under guest_wait_mutex_ and reports whether it consumed the
    // cancellation; a completed/signaled wait leaves cancellation pending instead.
    mutable std::mutex guest_wait_mutex_;
    bool guest_cancellation_pending_ = false;
    bool (*guest_wait_cancel_)(void*) noexcept = nullptr;
    void* guest_wait_context_ = nullptr;
    std::shared_ptr<ProcessRecord> process_;
    std::uint64_t id_;
    std::atomic<Handle> creation_handle_{0};
    std::atomic<std::uint32_t> priority_;
    mutable std::mutex scheduling_mutex_;
    std::int32_t ideal_core_ = -1;
    std::uint64_t affinity_ = 0;
    std::optional<std::uint32_t> current_processor_;
    mutable std::mutex exit_mutex_;
    mutable std::condition_variable exit_changed_;
    bool exited_ = false;
    // Waits on several objects that include this thread (exit_mutex_).
    std::vector<MultiWaiterEntry> exit_waiters_;
    std::atomic<std::uint64_t> lone_yield_count_{UINT64_MAX};
    std::atomic<CallPhase> phase_{CallPhase::guest};
    std::atomic<std::uint16_t> note_svc_{0};
    std::atomic<bool> note_deferred_{false};
    std::array<std::atomic<std::uint64_t>, 3> note_arguments_{};
    std::atomic<std::uint64_t> note_grant_address_{0};
    std::atomic<std::uint32_t> note_grant_tag_{0};
    std::atomic<std::uint32_t> note_grants_{0};
};

// x0..x17 are the integer registers changed by the supported syscall return
// path. Guest x18 and all higher registers are outside this bank and preserved.
using QueryRegisters = std::array<std::uint64_t, 18>;

// Copy exactly one four-byte word to validated guest backing. Calls are made
// in increasing destination order. Returning false stops the copy; earlier
// completed words remain written. Never dereference an unchecked host pointer.
using GuestWordWriter = bool (*)(void*, std::uint64_t, std::span<const std::byte, 4>) noexcept;
using GuestWordReader = bool (*)(void*, std::uint64_t, std::span<std::byte, 4>) noexcept;
// One aligned atomic eight-byte acquire load from validated native guest backing.
using GuestDoubleWordReader = bool (*)(void*, std::uint64_t, std::uint64_t&) noexcept;
// One atomic four-byte compare-exchange with the same result contract as the
// process-wide key handoff.
using GuestWordCompareExchange = ProcessWideKey::GuestWordCompareExchange;
struct GuestCopyOut {
    GuestWordWriter write_word = nullptr;
    // Synchronization words require aligned atomic acquire loads and release stores on
    // the same backing used by guest instructions, with stable mappings during each call.
    GuestWordReader read_word = nullptr;
    void* context = nullptr;
    // Strong atomic acq_rel compare-exchange (acquire on mismatch), interoperating with native
    // guest exclusive/atomic instructions. An independent host lock plus read/write does
    // not meet this contract. Required for conditional address updates and CV handoff.
    GuestWordCompareExchange compare_exchange_word = nullptr;
    // Required for WaitForAddress type 3; two four-byte reads can tear.
    GuestDoubleWordReader read_double_word = nullptr;
    // Checked ordinary guest copy, including unaligned addresses. Read exactly the
    // requested bytes without overread; keep backing stable for the complete copy.
    bool (*read_bytes)(void*, std::uint64_t, std::span<std::byte>) noexcept = nullptr;
};

// Writes a one-line description of a guest break into the caller's buffer and
// returns its length, excluding the terminator. Horizon's svcBreak carries the
// reason, the address of the payload and the payload size; the payload itself is
// guest memory the caller reads separately.
std::size_t describe_break(std::uint64_t reason, std::uint64_t address, std::uint64_t size,
                           char* buffer, std::size_t capacity) noexcept;

enum class QueryDisposition {
    handled,
    cancelled,
    unsupported_operation,
};

// Transfer memory is a block of the calling process's own address space exposed
// through a handle. The owner keeps the range but the object carries the
// permission other mappings will use, which is what the guest hands to another
// process later. No second mapping exists until one is requested.
class TransferMemoryRecord final : public KernelObject {
public:
    TransferMemoryRecord(AddressRange range, std::uint32_t permission)
        : range_(range), permission_(permission) {}
    AddressRange range() const noexcept { return range_; }
    std::uint32_t permission() const noexcept { return permission_; }

private:
    AddressRange range_;
    std::uint32_t permission_;
};

// Implements only the current startup queries. A handled call may return a
// genuine guest error in W0. Unsupported operations leave every register and
// guest byte unchanged, allowing the caller to report a missing capability.
// QueryMemory is unsupported when no copy-out backend is installed; a missing
// implementation is not presented as a guest output-pointer fault.
//
// Handles belongs to the current process and must outlive this dispatcher.
// Records are pinned during lookup. The copy backend must outlive calls and
// provide its own mapping/lifetime synchronization.
class StartupQueries {
public:
    StartupQueries(std::shared_ptr<ProcessRecord> process, HandleTable& handles,
                   std::shared_ptr<ThreadRecord> thread, GuestCopyOut copy_out = {},
                   std::shared_ptr<Arbitration> arbitration = {},
                   std::shared_ptr<ProcessWideKey> process_wide_keys = {},
                   std::stop_token cancellation = {});
    // The arbitration queue a process uses. Every thread of one process must share it, because
    // a signal from one guest thread has to release a wait in another.
    const std::shared_ptr<Arbitration>& arbitration() const noexcept { return arbitration_; }
    // The condition-variable table every thread of the process shares. A second
    // StartupQueries for another thread must be constructed with this object, because a
    // signal from one guest thread has to reach a wait in another.
    const std::shared_ptr<ProcessWideKey>& process_wide_keys() const noexcept { return keys_; }
    std::stop_token cancellation_token() const noexcept { return cancellation_; }
    const std::shared_ptr<ThreadRecord>& thread() const noexcept { return thread_; }
    QueryDisposition dispatch(std::uint16_t supervisor_call, QueryRegisters& registers) const;
    // Why the last unsupported query was unsupported, when the query says.
    const char* unsupported_detail() const noexcept { return unsupported_detail_; }

private:
    mutable const char* unsupported_detail_ = nullptr;
    QueryDisposition get_info(QueryRegisters& registers) const;
    void query_memory(QueryRegisters& registers) const;
    void query_thread(QueryRegisters& registers, bool identifier) const;
    QueryDisposition set_thread_priority(QueryRegisters& registers) const;
    QueryDisposition get_thread_core_mask(QueryRegisters& registers) const;
    QueryDisposition set_thread_core_mask(QueryRegisters& registers) const;
    QueryDisposition create_transfer_memory(QueryRegisters& registers) const;
    QueryDisposition wait_synchronization(QueryRegisters& registers) const;
    QueryDisposition cancel_synchronization(QueryRegisters& registers) const;
    QueryDisposition arbitrate(QueryRegisters& registers) const;
    // svcArbitrateLock and svcArbitrateUnlock: the word holds the current owner's tag
    // and a waiter marker. Ordinary and CV waiters use the same handoff queue.
    QueryDisposition arbitrate_lock(QueryRegisters& registers) const;
    QueryDisposition arbitrate_unlock(QueryRegisters& registers) const;
    QueryDisposition set_memory_permission(QueryRegisters& registers) const;
    QueryDisposition signal(QueryRegisters& registers) const;
    // svcWaitForAddress and svcSignalToAddress: the address arbiter's condition-word forms.
    QueryDisposition wait_for_address(QueryRegisters& registers) const;
    QueryDisposition signal_to_address(QueryRegisters& registers) const;
    // Four-byte guest word access for the arbitration and process-wide key paths.
    bool read_guest_word(std::uint64_t address, std::uint32_t& value) const noexcept;
    bool write_guest_word(std::uint64_t address, std::uint32_t value) const noexcept;
    ProcessWideKey::ExchangeResult compare_exchange_guest_word(std::uint64_t address,
                                                               std::uint32_t& expected,
                                                               std::uint32_t desired) const noexcept;
    static bool read_guest_word_thunk(void* context, std::uint64_t address,
                                      std::uint32_t& value) noexcept;
    static bool write_guest_word_thunk(void* context, std::uint64_t address,
                                       std::uint32_t value) noexcept;
    static ProcessWideKey::ExchangeResult compare_exchange_word_thunk(void* context,
                                                                      std::uint64_t address,
                                                                      std::uint32_t& expected,
                                                                      std::uint32_t desired) noexcept;
    ProcessWideKey::GuestWords guest_words() const noexcept;
    std::shared_ptr<ProcessRecord> process_;
    HandleTable* handles_;
    std::shared_ptr<ThreadRecord> thread_;
    GuestCopyOut copy_out_;
    std::shared_ptr<Arbitration> arbitration_;
    std::shared_ptr<ProcessWideKey> keys_;
    std::stop_token cancellation_;
};

} // namespace armigo::horizon
