// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Prepared gate-site dispatch for guest system registers and supervisor calls.
//
// An offline preparer replaces selected guest instructions with a direct `bl`
// into the host gate (see gate.h). The gate identifies each site only by its
// return address. A GuestSiteTable maps every such return address to the
// original instruction word it replaced, validated once against a fixed list
// of supported operations:
//
//   * BR/BLR to an explicitly declared prepared target set;
//   * SVC #imm16, forwarded to a caller-supplied supervisor handler;
//   * MRS/MSR TPIDR_EL0 and MRS TPIDRRO_EL0, served from an explicit
//     GuestThreadContext rather than from the host's hardware registers;
//   * MRS CNTPCT_EL0 / CNTVCT_EL0 (and optionally their self-synchronized
//     forms) and MRS CNTFRQ_EL0, served from an injected counter;
//   * MOVZ/MOVK, for code compiled ahead of time whose immediate is chosen
//     at launch: the table's word carries the immediate to apply.
//
// GuestDispatchScope installs a GateScope whose handler performs the lookup
// and emulates the site. Everything else is refused: unsupported words are
// rejected when the table is built, and unknown sites, missing handlers and
// unavailable or regressing counters fail the gate entry with a structured
// record. Nothing here writes the host's TPIDR_EL0, TPIDRRO_EL0 or x18.
//
// This is a dispatch primitive for prepared code. It does not run arbitrary
// guest code, implement any kernel service, or define a guest TLS layout.

#pragma once

#include <armigo/aarch64/instruction.h>
#include <armigo/execution/call_hook.h>
#include <armigo/execution/gate.h>
#include <armigo/execution/guest_register_cell.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace armigo::execution {

// ---------------------------------------------------------------------------
// Site table
// ---------------------------------------------------------------------------

enum class GuestSiteKind : std::uint8_t {
    supervisor_call,                // SVC #imm16
    read_thread_pointer,            // MRS Xt, TPIDR_EL0
    write_thread_pointer,           // MSR TPIDR_EL0, Xt
    read_read_only_thread_pointer,  // MRS Xt, TPIDRRO_EL0
    read_counter,                   // MRS Xt, CNTPCT_EL0 / CNTVCT_EL0 (/ CNTPCTSS / CNTVCTSS)
    read_counter_frequency,         // MRS Xt, CNTFRQ_EL0
    read_cache_type,                // MRS Xt, CTR_EL0
    clean_invalidate_data_cache,    // DC CIVAC, Xt
    invalidate_instruction_cache,   // IC IVAU, Xt
    checked_indirect,               // BR/BLR with an explicit allowed target set
    move_wide,                      // MOVZ/MOVK Rd, #imm16{, LSL #shift}
};

// One guest cache-maintenance operation on a guest virtual address. The runtime
// owns the host mapping, so it decides what the operation means on the host.
enum class CacheMaintenanceOperation : std::uint8_t {
    clean_invalidate_data = 0,
    invalidate_instruction = 1,
};

// Applies one cache-maintenance operation. Returns false when the runtime cannot
// honor it for that address; the gate then reports the site as unavailable
// instead of pretending the operation ran.
using GuestCacheMaintenanceHandler = bool (*)(void* context,
                                              CacheMaintenanceOperation operation,
                                              std::uint64_t address) noexcept;

struct GuestCacheMaintenance {
    GuestCacheMaintenanceHandler apply = nullptr;
    void* context = nullptr;
};

const char* describe(GuestSiteKind kind) noexcept;

// One prepared site as produced by a preparer: the gate return address (the
// address after the site's `bl`) and the guest instruction word it replaced.
struct GuestGateSite {
    std::uint64_t return_address = 0;
    std::uint32_t instruction = 0;
    // Nonempty only for BR/BLR. Absolute prepared target addresses at dispatch;
    // declarations do not prove target preparation or code ownership.
    std::vector<std::uint64_t> allowed_indirect_targets{};
};

// Classes of sites a table admits. A site of a disabled class is rejected
// when the table is built, never ignored.
struct GuestSiteTableOptions {
    bool supervisor_calls = true;
    bool thread_registers = true;
    bool counters = true;
    // CNTPCTSS_EL0 / CNTVCTSS_EL0 (FEAT_ECV). Requires `counters`.
    bool self_synchronized_counters = false;
    // CTR_EL0 reads and DC/IC cache maintenance naming a guest address.
    bool cache_maintenance = true;
    // MOVZ/MOVK sites whose immediate the table word supplies; only the
    // launch-time sites of embedded code use them.
    bool move_wide = false;
    // Upper bound on the number of sites; must be nonzero.
    std::size_t max_sites = std::size_t{1} << 20;
};

enum class GuestSiteTableErrorKind : std::uint8_t {
    none,
    invalid_options,              // inconsistent or empty option set
    too_many_sites,               // more sites than options.max_sites
    null_return_address,          // return address 0 (or below one instruction)
    misaligned_return_address,    // return address not 4-byte aligned
    duplicate_return_address,     // two sites share a return address
    unsupported_instruction,      // not a supported system or indirect operation
    unsupported_system_register,  // MRS/MSR of a register not served here
    unsupported_register_write,   // MSR to a register EL0 cannot write
    link_register_operand,        // MRS/MSR with Rt = x30, the gate's return address
    disabled_site_class,          // supported operation of a disabled class
    missing_indirect_targets,
    invalid_indirect_target,       // zero or unaligned target
    unsupported_indirect_register, // x18, x30 or x31 cannot carry the native target
    unexpected_indirect_targets,   // targets supplied for a non-indirect site
};

const char* describe(GuestSiteTableErrorKind kind) noexcept;

struct GuestSiteTableError {
    GuestSiteTableErrorKind kind = GuestSiteTableErrorKind::none;
    // Index into the input span of the first offending site; for
    // duplicate_return_address, the later of the two. Zero for option errors.
    std::size_t site_index = 0;
    std::uint64_t return_address = 0;
    std::uint32_t instruction = 0;
};

// A validated site with its decoded operands.
struct GuestSite {
    std::uint64_t return_address = 0;
    std::uint32_t instruction = 0;
    GuestSiteKind kind = GuestSiteKind::supervisor_call;
    // Rt for MRS/MSR: 0..29 or 31. 31 is the zero register (reads as zero,
    // writes are discarded) and 18 is the virtual guest x18. x30 is refused
    // because the gate's x30 is the site's return address. Zero for SVC.
    // Rn for checked indirects; x18, x30 and x31 are refused there.
    // Rd for MOVZ/MOVK, with the MRS/MSR rules (31 is the zero register).
    std::uint8_t reg = 0;
    // SVC or MOVZ/MOVK imm16; zero otherwise.
    std::uint16_t immediate = 0;
    aarch64::SystemRegister system_register = aarch64::SystemRegister::other;
    // Owned, sorted and unique. Only checked_indirect sites carry targets.
    std::vector<std::uint64_t> allowed_indirect_targets{};
};

class GuestSiteTable;

struct GuestSiteTableBuild {
    std::shared_ptr<const GuestSiteTable> table;  // null on error
    GuestSiteTableError error;
    bool ok() const noexcept {
        return table != nullptr && error.kind == GuestSiteTableErrorKind::none;
    }
};

// Immutable, validated map from gate return address to prepared site.
//
// The table copies everything it needs from its input; it keeps no reference
// to the caller's span or to any string. It is never modified after build and
// may be shared by any number of threads and scopes.
class GuestSiteTable {
public:
    static GuestSiteTableBuild build(std::span<const GuestGateSite> sites,
                                     const GuestSiteTableOptions& options = {});

    // Exact-match lookup; nullptr when no site has this return address.
    const GuestSite* find(std::uint64_t return_address) const noexcept;

    // All sites, sorted by return address.
    std::span<const GuestSite> sites() const noexcept { return sites_; }
    const GuestSiteTableOptions& options() const noexcept { return options_; }
    bool has_checked_indirect_sites() const noexcept { return checked_indirects_; }

    GuestSiteTable(const GuestSiteTable&) = delete;
    GuestSiteTable& operator=(const GuestSiteTable&) = delete;

private:
    GuestSiteTable(std::vector<GuestSite> sites, const GuestSiteTableOptions& options,
                   bool checked_indirects)
        : sites_(std::move(sites)), options_(options), checked_indirects_(checked_indirects) {}

    std::vector<GuestSite> sites_;
    GuestSiteTableOptions options_;
    bool checked_indirects_ = false;
};

// ---------------------------------------------------------------------------
// Thread context
// ---------------------------------------------------------------------------

// Injected guest counter. `read_ticks` must be callable from any thread that
// uses a context holding this counter and must not throw. `frequency` is the
// value MRS CNTFRQ_EL0 returns, in ticks per second; the architectural field
// is 32 bits wide.
using GuestTickSource = std::uint64_t (*)(void* context) noexcept;

struct GuestCounter {
    GuestTickSource read_ticks = nullptr;
    void* context = nullptr;
    std::uint64_t frequency = 0;
};

enum class GuestCounterError : std::uint8_t {
    none,
    missing_tick_source,
    zero_frequency,
    frequency_out_of_range,  // above 0xFFFFFFFF
};

GuestCounterError validate(const GuestCounter& counter) noexcept;

// Caller-owned guest register state for one guest thread.
//
// Owns one GuestRegisterCell holding the guest's x18, TPIDR_EL0 and
// TPIDRRO_EL0, and the guest counter. These are plain values; nothing is
// mirrored into host hardware registers. The cell lives inside the context,
// which is neither copyable nor movable, so its address is stable for the
// context's lifetime.
//
// While a GuestDispatchScope binds the context, gate entries served by that
// scope use the cell's x18 as their virtual guest x18 (gate.h), so the value
// follows the guest thread when the context is later bound on another host
// thread instead of staying with the host thread.
//
// A context may be bound by any number of GuestDispatchScopes, all on one
// host thread at a time; a scope on another thread is refused. Destroying a
// context that is still bound aborts the process.
//
// The value accessors below are for the context's owner and follow ordinary
// C++ data-race rules: while the context is bound, use them only on the
// binding thread.
class GuestThreadContext {
public:
    GuestThreadContext() noexcept = default;
    ~GuestThreadContext();

    GuestThreadContext(const GuestThreadContext&) = delete;
    GuestThreadContext& operator=(const GuestThreadContext&) = delete;
    GuestThreadContext(GuestThreadContext&&) = delete;
    GuestThreadContext& operator=(GuestThreadContext&&) = delete;

    std::uint64_t thread_pointer() const noexcept { return cell_.thread_pointer; }
    void set_thread_pointer(std::uint64_t value) noexcept { cell_.thread_pointer = value; }

    std::uint64_t read_only_thread_pointer() const noexcept {
        return cell_.read_only_thread_pointer;
    }
    void set_read_only_thread_pointer(std::uint64_t value) noexcept {
        cell_.read_only_thread_pointer = value;
    }

    // The guest's x18 as stored in this context. It is the virtual guest x18
    // of gate entries only while a scope binding this context is selected.
    std::uint64_t guest_x18() const noexcept { return cell_.x18; }
    void set_guest_x18(std::uint64_t value) noexcept { cell_.x18 = value; }

    // Direct access to the shared register cell for prepared native code.
    //
    // Returns the cell only when called on the host thread that currently
    // holds a successful GuestDispatchScope binding of this context, and null
    // otherwise (unbound or bound on another thread). An existing successful
    // binding can still expose the cell after a later scope fails setup;
    // this accessor does not establish that the innermost scope succeeded.
    // The pointer is valid only for as long as that binding stays
    // active on this thread: the caller must stop using it, and must end any
    // host publication of it such as a native thread slot binding, before
    // the last binding GuestDispatchScope is destroyed. Writes through the
    // pointer are the same writes gate handlers observe, and vice versa;
    // a failed gate entry restores its x18 word. A failed supervisor entry
    // also restores this context's thread-pointer words and counter
    // configuration. Other callback side effects are not transactional.
    GuestRegisterCell* bound_register_cell() noexcept;

    // Installs a counter after validating it; an invalid counter leaves the
    // previous one in place. The last value delivered to guest code is kept,
    // so a replacement source that runs behind is reported, not hidden.
    GuestCounterError set_counter(const GuestCounter& counter) noexcept;
    void clear_counter() noexcept { counter_.reset(); }
    const std::optional<GuestCounter>& counter() const noexcept { return counter_; }

    // The last counter value an MRS site delivered; counter reads through
    // this context never go backwards.
    std::optional<std::uint64_t> last_counter_value() const noexcept { return last_counter_; }

    // The cache-type register value an MRS CTR_EL0 site delivers. Zero means no
    // value was supplied and the site fails explicitly.
    void set_cache_type(std::uint64_t value) noexcept { cache_type_ = value; }
    std::uint64_t cache_type() const noexcept { return cache_type_; }

    // Installs the runtime's cache-maintenance handler. Without one, a cache site
    // fails explicitly instead of being skipped.
    void set_cache_maintenance(GuestCacheMaintenance handler) noexcept {
        cache_maintenance_ = handler;
    }
    const GuestCacheMaintenance& cache_maintenance() const noexcept {
        return cache_maintenance_;
    }

    bool bound() const noexcept { return owner_.load(std::memory_order_acquire) != nullptr; }

private:
    friend class GuestDispatchScope;

    GuestRegisterCell cell_;
    std::optional<GuestCounter> counter_;
    std::optional<std::uint64_t> last_counter_;
    std::uint64_t cache_type_ = 0;
    GuestCacheMaintenance cache_maintenance_{};
    std::atomic<const void*> owner_{nullptr};
    std::uint32_t bindings_ = 0;  // Modified only by the owning thread.
};

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Supervisor-call handler. Receives the SVC immediate, the gate call (whose
// registers it reads and writes under the gate's rules) and the thread's
// context. Returning GateStatus::failed fails the gate entry; any TPIDR_EL0 /
// TPIDRRO_EL0 change the handler made is then rolled back. Exceptions are
// rolled back the same way and reported by the gate as handler_threw.
// The gate clears host exclusive reservations before resuming guest code.
using GuestSupervisorHandler = GateStatus (*)(std::uint16_t immediate, GateCall& call,
                                              GuestThreadContext& thread, void* context);

struct GuestSupervisor {
    GuestSupervisorHandler handler = nullptr;
    void* context = nullptr;
};

enum class GuestDispatchFailureKind : std::uint8_t {
    none,
    missing_table,               // the scope was given no table
    context_bound_elsewhere,     // the thread context is bound on another thread
    unknown_site,                // no site has the entry's return address
    missing_supervisor_handler,  // an SVC site without a supervisor handler
    supervisor_failed,           // the supervisor handler returned failed
    supervisor_threw,            // the supervisor handler threw
    invalid_supervisor_frame,    // the handler changed protected frame fields
    counter_unavailable,         // a counter site without a counter
    counter_regressed,           // the tick source went backwards
    cache_type_unavailable,      // a CTR_EL0 site without a value
    cache_maintenance_unavailable, // a cache site without a handler or with a refused address
    indirect_target_rejected,    // the runtime register is outside the prepared set
    indirect_requires_trap,      // checked indirects cannot resume after failure
    call_hook_requires_trap,     // hook failures cannot resume into the original callee
    call_hook_site_conflict,     // a return address is both a normal and hook site
    missing_call_hook_handler,   // a hook table was supplied without a callback
    call_hook_failed,            // the call-hook callback returned failed
    call_hook_threw,             // the call-hook callback threw
    invalid_call_hook_frame,     // the callback changed x30, sp or the host x18 slot
};

const char* describe(GuestDispatchFailureKind kind) noexcept;

struct GuestDispatchFailure {
    GuestDispatchFailureKind kind = GuestDispatchFailureKind::none;
    const char* message = nullptr;  // Static text; never owned.
    std::uint64_t return_address = 0;
    std::uint32_t instruction = 0;  // Zero when the site is unknown.
    std::uint32_t depth = 0;
    std::uint64_t indirect_target = 0; // Actual rejected target; zero for other failures.
};

// Lexically scoped guest dispatch on the constructing thread.
//
// Installs a GateScope (with the given failure policy) whose handler serves
// every gate entry on this thread while it is the innermost scope, including
// nested entries made from a supervisor handler. It inherits every GateScope
// contract: strict LIFO destruction on the owning thread, trap on misuse, and
// either trap or restoration of the captured frame and x18 on failure.
//
// A successfully bound scope selects its context's cell x18 as the thread's
// virtual guest x18 (GateScope storage selection) for as long as it is the
// innermost scope; its destruction restores the previous selection. A scope
// that could not be set up (no table, or a context bound on another thread)
// selects no storage of its own and never the context's cell; it is still
// installed so that guest entries cannot fall through to an outer scope, and
// each entry then fails with setup_error().
//
// A supervisor may return GateStatus::exit_thread when call.can_exit_thread()
// is true. Accepted exit commits context changes and returns through the host
// callback normally; gate assembly then terminates the innermost guest entry.
//
// Tables containing checked indirect sites require the trap failure policy.
// A requested resume_unchanged policy fails setup and installs a trapping gate,
// so a rejected target can never resume into the stub's native branch.
// A nonempty call-hook table has the same rule: its callback runs before the
// original callee, and a failed callback can never resume into that callee.
// Hook and ordinary site return addresses must be disjoint.
// The table is shared ownership; the context and the supervisor's context
// and call-hook context are referenced and must outlive the scope.
class GuestDispatchScope {
public:
    GuestDispatchScope(std::shared_ptr<const GuestSiteTable> table, GuestThreadContext& thread,
                       GuestSupervisor supervisor = {},
                       GateFailurePolicy policy = GateFailurePolicy::trap,
                       std::shared_ptr<const CallHookTable> hooks = {},
                       CallHook hook = {}) noexcept;
    ~GuestDispatchScope();

    GuestDispatchScope(const GuestDispatchScope&) = delete;
    GuestDispatchScope& operator=(const GuestDispatchScope&) = delete;

    GuestDispatchFailureKind setup_error() const noexcept { return setup_error_; }
    const GuestSiteTable* table() const noexcept { return table_.get(); }
    GateScope& gate_scope() noexcept { return gate_; }
    const GateScope& gate_scope() const noexcept { return gate_; }

    // Successfully emulated sites.
    std::uint64_t dispatched() const noexcept { return dispatched_; }

    // Failures this scope's handler reported. The first is kept in full;
    // later ones only increment the count. Gate-level failure records (for
    // example under the resume_unchanged policy) live on gate_scope().
    std::uint64_t failure_count() const noexcept { return failure_count_; }
    const GuestDispatchFailure* first_failure() const noexcept {
        return failure_count_ == 0 ? nullptr : &first_failure_;
    }
    void clear_failures() noexcept;

private:
    static GateStatus handle(GateCall& call, void* raw);
    GateStatus fail(GateCall& call, GuestDispatchFailureKind kind, const char* message,
                    std::uint32_t instruction) noexcept;
    GateStatus dispatch_supervisor(GateCall& call, const GuestSite& site);
    GateStatus dispatch_call_hook(GateCall& call, const CallHookSite& site);
    GateStatus dispatch_counter(GateCall& call, const GuestSite& site) noexcept;
    GateStatus dispatch_cache(GateCall& call, const GuestSite& site) noexcept;
    GuestDispatchFailureKind bind(GateFailurePolicy policy) noexcept;

    std::shared_ptr<const GuestSiteTable> table_;
    std::shared_ptr<const CallHookTable> hooks_;
    GuestThreadContext* thread_;
    GuestSupervisor supervisor_;
    CallHook hook_;
    bool bound_ = false;
    GuestDispatchFailureKind setup_error_;
    std::uint64_t dispatched_ = 0;
    std::uint64_t failure_count_ = 0;
    GuestDispatchFailure first_failure_{};
    // Declared last: installed after setup and removed before unbinding.
    GateScope gate_;
};

}  // namespace armigo::execution
