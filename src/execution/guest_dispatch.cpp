// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Prepared gate-site dispatch: site validation, the guest thread context and
// the GateScope handler that emulates supported sites.

#include <armigo/execution/guest_dispatch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include <unistd.h>

namespace armigo::execution {
namespace {

using aarch64::Opcode;
using aarch64::SystemRegister;

constexpr std::uint8_t kZeroRegister = aarch64::register_field_zero_or_sp;
constexpr std::uint64_t kCounterFrequencyLimit = 0xFFFFFFFFull;

// Identity of the calling host thread for context binding. The address of a
// thread_local object is unique among live threads.
thread_local const char t_thread_identity = 0;

const void* current_thread_identity() noexcept {
    return &t_thread_identity;
}

[[noreturn]] void fatal(const char* message) noexcept {
    char text[192];
    const int length =
        std::snprintf(text, sizeof text, "armigo guest dispatch: fatal %s\n", message);
    if (length > 0) {
        const auto count = static_cast<std::size_t>(length) < sizeof text
                               ? static_cast<std::size_t>(length)
                               : sizeof text - 1;
        (void)::write(STDERR_FILENO, text, count);
    }
    std::abort();
}

struct Classification {
    GuestSiteTableErrorKind error = GuestSiteTableErrorKind::none;
    GuestSite site;
};

bool is_self_synchronized_counter(SystemRegister reg) noexcept {
    return reg == SystemRegister::cntpctss_el0 || reg == SystemRegister::cntvctss_el0;
}

// Maps one prepared site onto a supported operation, or explains why it is
// not one. Indirect branches additionally require explicit target declarations.
Classification classify(const GuestGateSite& input, const GuestSiteTableOptions& options) {
    Classification result;
    result.site.return_address = input.return_address;
    result.site.instruction = input.instruction;

    const aarch64::Instruction decoded = aarch64::decode(input.instruction);
    if (decoded.opcode == Opcode::br || decoded.opcode == Opcode::blr) {
        result.site.kind = GuestSiteKind::checked_indirect;
        result.site.reg = decoded.reg;
        if (decoded.reg == kGateHostReservedRegister || decoded.reg >= aarch64::link_register) {
            result.error = GuestSiteTableErrorKind::unsupported_indirect_register;
        } else if (input.allowed_indirect_targets.empty()) {
            result.error = GuestSiteTableErrorKind::missing_indirect_targets;
        } else if (std::any_of(input.allowed_indirect_targets.begin(),
                               input.allowed_indirect_targets.end(), [](auto target) {
                                   return target == 0 || target % 4 != 0;
                               })) {
            result.error = GuestSiteTableErrorKind::invalid_indirect_target;
        } else {
            auto& targets = result.site.allowed_indirect_targets;
            targets = input.allowed_indirect_targets;
            std::sort(targets.begin(), targets.end());
            targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        }
        return result;
    }
    if (!input.allowed_indirect_targets.empty()) {
        result.error = GuestSiteTableErrorKind::unexpected_indirect_targets;
        return result;
    }
    if (decoded.opcode == Opcode::cache_maintenance) {
        // DC/IC operations name a guest address in Rt and leave the register
        // unmodified, so only the address operand matters here.
        result.site.reg = decoded.reg;
        switch (decoded.cache_operation) {
        case aarch64::CacheOperation::clean_invalidate_data:
            result.site.kind = GuestSiteKind::clean_invalidate_data_cache;
            break;
        case aarch64::CacheOperation::invalidate_instruction:
            result.site.kind = GuestSiteKind::invalidate_instruction_cache;
            break;
        default:
            result.error = GuestSiteTableErrorKind::unsupported_instruction;
            return result;
        }
        if (decoded.reg == aarch64::link_register) {
            result.error = GuestSiteTableErrorKind::link_register_operand;
        } else if (!options.cache_maintenance) {
            result.error = GuestSiteTableErrorKind::disabled_site_class;
        }
        return result;
    }
    if (decoded.opcode == Opcode::svc) {
        result.site.kind = GuestSiteKind::supervisor_call;
        result.site.immediate = decoded.immediate;
        if (!options.supervisor_calls) {
            result.error = GuestSiteTableErrorKind::disabled_site_class;
        }
        return result;
    }
    // MOVZ (opc 10) and MOVK (opc 11): move wide immediate, bits 28:23 100101.
    if ((input.instruction & 0x1f800000u) == 0x12800000u && (input.instruction >> 29 & 3) >= 2) {
        result.site.kind = GuestSiteKind::move_wide;
        result.site.reg = static_cast<std::uint8_t>(input.instruction & 31);
        result.site.immediate = static_cast<std::uint16_t>(input.instruction >> 5);
        if (!(input.instruction >> 31) && (input.instruction >> 21 & 3) > 1) {
            result.error = GuestSiteTableErrorKind::unsupported_instruction;
        } else if (result.site.reg == aarch64::link_register) {
            result.error = GuestSiteTableErrorKind::link_register_operand;
        } else if (!options.move_wide) {
            result.error = GuestSiteTableErrorKind::disabled_site_class;
        }
        return result;
    }
    if (decoded.opcode != Opcode::mrs && decoded.opcode != Opcode::msr_register) {
        result.error = GuestSiteTableErrorKind::unsupported_instruction;
        return result;
    }

    const SystemRegister reg = aarch64::identify_system_register(decoded.system_register);
    const bool read = decoded.opcode == Opcode::mrs;
    result.site.reg = decoded.reg;
    result.site.system_register = reg;

    bool thread_class = false;
    bool counter_class = false;
    switch (reg) {
    case SystemRegister::tpidr_el0:
        result.site.kind = read ? GuestSiteKind::read_thread_pointer
                                : GuestSiteKind::write_thread_pointer;
        thread_class = true;
        break;
    case SystemRegister::tpidrro_el0:
        result.site.kind = GuestSiteKind::read_read_only_thread_pointer;
        thread_class = true;
        break;
    case SystemRegister::cntpct_el0:
    case SystemRegister::cntvct_el0:
    case SystemRegister::cntpctss_el0:
    case SystemRegister::cntvctss_el0:
        result.site.kind = GuestSiteKind::read_counter;
        counter_class = true;
        break;
    case SystemRegister::cntfrq_el0:
        result.site.kind = GuestSiteKind::read_counter_frequency;
        counter_class = true;
        break;
    case SystemRegister::ctr_el0:
        // The cache geometry the guest sees is the console's, not the host's.
        if (!read) {
            result.error = GuestSiteTableErrorKind::unsupported_register_write;
            return result;
        }
        result.site.kind = GuestSiteKind::read_cache_type;
        if (!options.cache_maintenance) {
            result.error = GuestSiteTableErrorKind::disabled_site_class;
        }
        return result;
    default:
        result.error = GuestSiteTableErrorKind::unsupported_system_register;
        return result;
    }

    // TPIDRRO_EL0, the counters and CNTFRQ_EL0 are read-only at EL0; an MSR
    // to them is undefined there and has nothing to emulate.
    if (!read && reg != SystemRegister::tpidr_el0) {
        result.error = GuestSiteTableErrorKind::unsupported_register_write;
        return result;
    }
    // The gate's x30 is the site's return address, not the guest's x30.
    if (decoded.reg == aarch64::link_register) {
        result.error = GuestSiteTableErrorKind::link_register_operand;
        return result;
    }
    if ((thread_class && !options.thread_registers) || (counter_class && !options.counters) ||
        (is_self_synchronized_counter(reg) && !options.self_synchronized_counters)) {
        result.error = GuestSiteTableErrorKind::disabled_site_class;
    }
    return result;
}

bool options_consistent(const GuestSiteTableOptions& options) noexcept {
    if (options.max_sites == 0) {
        return false;
    }
    if (!options.supervisor_calls && !options.thread_registers && !options.counters) {
        return false;
    }
    return !options.self_synchronized_counters || options.counters;
}

// Rt access with the A64 register-31 and virtual-x18 rules.
std::uint64_t read_register(const GateCall& call, std::uint8_t reg) noexcept {
    if (reg == kZeroRegister) {
        return 0;
    }
    if (reg == kGateHostReservedRegister) {
        return call.guest_x18();
    }
    return call.x(reg);
}

void write_register(GateCall& call, std::uint8_t reg, std::uint64_t value) noexcept {
    if (reg == kZeroRegister) {
        return;
    }
    if (reg == kGateHostReservedRegister) {
        call.set_guest_x18(value);
        return;
    }
    (void)call.set_x(reg, value);
}

}  // namespace

// ---------------------------------------------------------------------------
// Descriptions
// ---------------------------------------------------------------------------

const char* describe(GuestSiteKind kind) noexcept {
    switch (kind) {
    case GuestSiteKind::checked_indirect:
        return "checked indirect branch";
    case GuestSiteKind::supervisor_call:
        return "supervisor call";
    case GuestSiteKind::read_thread_pointer:
        return "TPIDR_EL0 read";
    case GuestSiteKind::write_thread_pointer:
        return "TPIDR_EL0 write";
    case GuestSiteKind::read_read_only_thread_pointer:
        return "TPIDRRO_EL0 read";
    case GuestSiteKind::read_counter:
        return "counter read";
    case GuestSiteKind::read_counter_frequency:
        return "counter frequency read";
    case GuestSiteKind::read_cache_type:
        return "cache type read";
    case GuestSiteKind::clean_invalidate_data_cache:
        return "data cache clean and invalidate";
    case GuestSiteKind::invalidate_instruction_cache:
        return "instruction cache invalidate";
    case GuestSiteKind::move_wide:
        return "move wide immediate";
    }
    return "unknown site kind";
}

const char* describe(GuestSiteTableErrorKind kind) noexcept {
    switch (kind) {
    case GuestSiteTableErrorKind::none:
        return "no error";
    case GuestSiteTableErrorKind::invalid_options:
        return "inconsistent site table options";
    case GuestSiteTableErrorKind::too_many_sites:
        return "more sites than the configured maximum";
    case GuestSiteTableErrorKind::null_return_address:
        return "site return address cannot follow a gate call";
    case GuestSiteTableErrorKind::misaligned_return_address:
        return "site return address is not 4-byte aligned";
    case GuestSiteTableErrorKind::duplicate_return_address:
        return "two sites share a return address";
    case GuestSiteTableErrorKind::unsupported_instruction:
        return "instruction is not a supported gate-site operation";
    case GuestSiteTableErrorKind::unsupported_system_register:
        return "system register is not served by guest dispatch";
    case GuestSiteTableErrorKind::unsupported_register_write:
        return "system register is not writable at EL0";
    case GuestSiteTableErrorKind::link_register_operand:
        return "site operand is x30, which holds the gate return address";
    case GuestSiteTableErrorKind::disabled_site_class:
        return "site class is disabled by the table options";
    case GuestSiteTableErrorKind::missing_indirect_targets:
        return "indirect site requires an explicit prepared target set";
    case GuestSiteTableErrorKind::invalid_indirect_target:
        return "indirect target must be nonzero and instruction aligned";
    case GuestSiteTableErrorKind::unsupported_indirect_register:
        return "indirect target register cannot be x18, x30 or x31";
    case GuestSiteTableErrorKind::unexpected_indirect_targets:
        return "target declarations require a BR or BLR instruction";
    }
    return "unknown error";
}

const char* describe(GuestDispatchFailureKind kind) noexcept {
    switch (kind) {
    case GuestDispatchFailureKind::none:
        return "no failure";
    case GuestDispatchFailureKind::missing_table:
        return "guest dispatch scope has no site table";
    case GuestDispatchFailureKind::context_bound_elsewhere:
        return "guest thread context is bound on another thread";
    case GuestDispatchFailureKind::unknown_site:
        return "gate return address is not a prepared site";
    case GuestDispatchFailureKind::missing_supervisor_handler:
        return "supervisor call without a supervisor handler";
    case GuestDispatchFailureKind::supervisor_failed:
        return "supervisor handler reported failure";
    case GuestDispatchFailureKind::supervisor_threw:
        return "supervisor handler threw an exception";
    case GuestDispatchFailureKind::invalid_supervisor_frame:
        return "supervisor handler changed protected gate frame fields";
    case GuestDispatchFailureKind::counter_unavailable:
        return "counter site without a guest counter";
    case GuestDispatchFailureKind::counter_regressed:
        return "guest counter went backwards";
    case GuestDispatchFailureKind::cache_type_unavailable:
        return "cache type site without a guest cache type";
    case GuestDispatchFailureKind::cache_maintenance_unavailable:
        return "cache maintenance site without a runtime handler";
    case GuestDispatchFailureKind::indirect_target_rejected:
        return "indirect target is outside the prepared target set";
    case GuestDispatchFailureKind::indirect_requires_trap:
        return "checked indirect sites require the trap failure policy";
    case GuestDispatchFailureKind::call_hook_requires_trap:
        return "call-hook sites require the trap failure policy";
    case GuestDispatchFailureKind::call_hook_site_conflict:
        return "a gate return address is both a normal site and a call hook";
    case GuestDispatchFailureKind::missing_call_hook_handler:
        return "call-hook site without a call-hook handler";
    case GuestDispatchFailureKind::call_hook_failed:
        return "call-hook handler reported failure";
    case GuestDispatchFailureKind::call_hook_threw:
        return "call-hook handler threw an exception";
    case GuestDispatchFailureKind::invalid_call_hook_frame:
        return "call-hook handler changed protected gate frame fields";
    }
    return "unknown failure";
}

// ---------------------------------------------------------------------------
// GuestSiteTable
// ---------------------------------------------------------------------------

GuestSiteTableBuild GuestSiteTable::build(std::span<const GuestGateSite> sites,
                                          const GuestSiteTableOptions& options) {
    GuestSiteTableBuild result;
    if (!options_consistent(options)) {
        result.error.kind = GuestSiteTableErrorKind::invalid_options;
        return result;
    }
    if (sites.size() > options.max_sites) {
        result.error.kind = GuestSiteTableErrorKind::too_many_sites;
        result.error.site_index = options.max_sites;
        return result;
    }

    // Pairs of (site, input index) so errors can name the caller's entry.
    std::vector<std::pair<GuestSite, std::size_t>> prepared;
    prepared.reserve(sites.size());
    for (std::size_t index = 0; index < sites.size(); ++index) {
        const GuestGateSite& input = sites[index];
        GuestSiteTableError error{GuestSiteTableErrorKind::none, index, input.return_address,
                                  input.instruction};
        // A gate return address follows a 4-byte `bl`, so it is at least 4.
        if (input.return_address < 4) {
            error.kind = GuestSiteTableErrorKind::null_return_address;
        } else if ((input.return_address & 3u) != 0) {
            error.kind = GuestSiteTableErrorKind::misaligned_return_address;
        } else {
            const Classification classification = classify(input, options);
            error.kind = classification.error;
            if (error.kind == GuestSiteTableErrorKind::none) {
                prepared.emplace_back(classification.site, index);
            }
        }
        if (error.kind != GuestSiteTableErrorKind::none) {
            result.error = error;
            break;
        }
    }

    std::stable_sort(prepared.begin(), prepared.end(), [](const auto& left, const auto& right) {
        return left.first.return_address < right.first.return_address;
    });
    // After a stable sort, duplicates are adjacent and in input order; report
    // the later input entry of the earliest-indexed duplicate pair.
    std::size_t duplicate = sites.size();
    for (std::size_t i = 1; i < prepared.size(); ++i) {
        if (prepared[i].first.return_address == prepared[i - 1].first.return_address) {
            duplicate = std::min(duplicate, prepared[i].second);
        }
    }
    if (duplicate != sites.size()) {
        result.error = GuestSiteTableError{GuestSiteTableErrorKind::duplicate_return_address,
                                           duplicate, sites[duplicate].return_address,
                                           sites[duplicate].instruction};
        return result;
    }
    if (result.error.kind != GuestSiteTableErrorKind::none) {
        return result;
    }

    std::vector<GuestSite> table_sites;
    table_sites.reserve(prepared.size());
    for (const auto& entry : prepared) {
        table_sites.push_back(entry.first);
    }
    const bool checked_indirects = std::any_of(table_sites.begin(), table_sites.end(),
        [](const GuestSite& site) { return site.kind == GuestSiteKind::checked_indirect; });
    result.table = std::shared_ptr<const GuestSiteTable>(
        new GuestSiteTable(std::move(table_sites), options, checked_indirects));
    return result;
}

const GuestSite* GuestSiteTable::find(std::uint64_t return_address) const noexcept {
    const auto it = std::lower_bound(
        sites_.begin(), sites_.end(), return_address,
        [](const GuestSite& site, std::uint64_t address) { return site.return_address < address; });
    if (it == sites_.end() || it->return_address != return_address) {
        return nullptr;
    }
    return &*it;
}

// ---------------------------------------------------------------------------
// GuestThreadContext
// ---------------------------------------------------------------------------

GuestCounterError validate(const GuestCounter& counter) noexcept {
    if (counter.read_ticks == nullptr) {
        return GuestCounterError::missing_tick_source;
    }
    if (counter.frequency == 0) {
        return GuestCounterError::zero_frequency;
    }
    if (counter.frequency > kCounterFrequencyLimit) {
        return GuestCounterError::frequency_out_of_range;
    }
    return GuestCounterError::none;
}

GuestThreadContext::~GuestThreadContext() {
    if (bound()) {
        fatal("GuestThreadContext destroyed while a GuestDispatchScope still binds it");
    }
}

GuestRegisterCell* GuestThreadContext::bound_register_cell() noexcept {
    return owner_.load(std::memory_order_acquire) == current_thread_identity() && bindings_ != 0
               ? &cell_
               : nullptr;
}

GuestCounterError GuestThreadContext::set_counter(const GuestCounter& counter) noexcept {
    const GuestCounterError error = validate(counter);
    if (error == GuestCounterError::none) {
        counter_ = counter;
    }
    return error;
}

// ---------------------------------------------------------------------------
// GuestDispatchScope
// ---------------------------------------------------------------------------

GuestDispatchScope::GuestDispatchScope(std::shared_ptr<const GuestSiteTable> table,
                                       GuestThreadContext& thread, GuestSupervisor supervisor,
                                       GateFailurePolicy policy,
                                       std::shared_ptr<const CallHookTable> hooks,
                                       CallHook hook) noexcept
    : table_(std::move(table)),
      hooks_(std::move(hooks)),
      thread_(&thread),
      supervisor_(supervisor),
      hook_(hook),
      setup_error_(bind(policy)),
      // Only a successful binding selects the context's x18; a failed setup
      // keeps the thread's current selection and never touches the cell.
      gate_(&GuestDispatchScope::handle, this,
            (table_ && table_->has_checked_indirect_sites()) ||
                    (hooks_ && !hooks_->sites().empty())
                ? GateFailurePolicy::trap : policy,
            bound_ ? &thread_->cell_.x18 : nullptr) {}

GuestDispatchScope::~GuestDispatchScope() {
    // gate_ is destroyed after this body and traps on every GateScope misuse
    // (another thread, out of order, or from inside its own callback). Leave
    // the binding alone on a foreign thread so the trap is the only effect.
    if (bound_ && thread_->owner_.load(std::memory_order_acquire) == current_thread_identity()) {
        if (--thread_->bindings_ == 0) {
            thread_->owner_.store(nullptr, std::memory_order_release);
        }
        bound_ = false;
    }
}

GuestDispatchFailureKind GuestDispatchScope::bind(GateFailurePolicy policy) noexcept {
    if (!table_) {
        return GuestDispatchFailureKind::missing_table;
    }
    if (table_->has_checked_indirect_sites() && policy != GateFailurePolicy::trap) {
        return GuestDispatchFailureKind::indirect_requires_trap;
    }
    if (hooks_ && !hooks_->sites().empty()) {
        if (policy != GateFailurePolicy::trap) {
            return GuestDispatchFailureKind::call_hook_requires_trap;
        }
        if (hook_.handler == nullptr) {
            return GuestDispatchFailureKind::missing_call_hook_handler;
        }
        if (std::any_of(hooks_->sites().begin(), hooks_->sites().end(), [&](const auto& site) {
                return table_->find(site.return_address) != nullptr;
            })) {
            return GuestDispatchFailureKind::call_hook_site_conflict;
        }
    }
    const void* const self = current_thread_identity();
    const void* expected = nullptr;
    if (!thread_->owner_.compare_exchange_strong(expected, self, std::memory_order_acq_rel) &&
        expected != self) {
        return GuestDispatchFailureKind::context_bound_elsewhere;
    }
    ++thread_->bindings_;
    bound_ = true;
    return GuestDispatchFailureKind::none;
}

void GuestDispatchScope::clear_failures() noexcept {
    failure_count_ = 0;
    first_failure_ = GuestDispatchFailure{};
}

GateStatus GuestDispatchScope::fail(GateCall& call, GuestDispatchFailureKind kind,
                                    const char* message, std::uint32_t instruction) noexcept {
    if (failure_count_ == 0) {
        first_failure_ = GuestDispatchFailure{kind, message, call.return_address(), instruction,
                                              call.depth()};
    }
    ++failure_count_;
    call.set_failure_message(message);
    return GateStatus::failed;
}

GateStatus GuestDispatchScope::handle(GateCall& call, void* raw) {
    auto& scope = *static_cast<GuestDispatchScope*>(raw);
    if (scope.setup_error_ != GuestDispatchFailureKind::none) {
        return scope.fail(call, scope.setup_error_, describe(scope.setup_error_), 0);
    }

    if (scope.hooks_) {
        if (const auto* hook = scope.hooks_->find(call.return_address())) {
            return scope.dispatch_call_hook(call, *hook);
        }
    }

    const GuestSite* const site = scope.table_->find(call.return_address());
    if (site == nullptr) {
        return scope.fail(call, GuestDispatchFailureKind::unknown_site,
                          describe(GuestDispatchFailureKind::unknown_site), 0);
    }

    GuestThreadContext& thread = *scope.thread_;
    switch (site->kind) {
    case GuestSiteKind::checked_indirect:
        if (!std::binary_search(site->allowed_indirect_targets.begin(),
                                site->allowed_indirect_targets.end(), call.x(site->reg))) {
            const bool first = scope.failure_count_ == 0;
            const auto status = scope.fail(call, GuestDispatchFailureKind::indirect_target_rejected,
                                           describe(GuestDispatchFailureKind::indirect_target_rejected),
                                           site->instruction);
            if (first) scope.first_failure_.indirect_target = call.x(site->reg);
            return status;
        }
        break;
    case GuestSiteKind::supervisor_call:
        return scope.dispatch_supervisor(call, *site);
    case GuestSiteKind::read_thread_pointer:
        write_register(call, site->reg, thread.cell_.thread_pointer);
        break;
    case GuestSiteKind::write_thread_pointer:
        thread.cell_.thread_pointer = read_register(call, site->reg);
        break;
    case GuestSiteKind::read_read_only_thread_pointer:
        write_register(call, site->reg, thread.cell_.read_only_thread_pointer);
        break;
    case GuestSiteKind::read_counter:
    case GuestSiteKind::read_counter_frequency:
        return scope.dispatch_counter(call, *site);
    case GuestSiteKind::read_cache_type:
        if (thread.cache_type() == 0) {
            return scope.fail(call, GuestDispatchFailureKind::cache_type_unavailable,
                              describe(GuestDispatchFailureKind::cache_type_unavailable),
                              site->instruction);
        }
        write_register(call, site->reg, thread.cache_type());
        break;
    case GuestSiteKind::clean_invalidate_data_cache:
    case GuestSiteKind::invalidate_instruction_cache:
        return scope.dispatch_cache(call, *site);
    case GuestSiteKind::move_wide: {
        // MOVZ clears Rd; MOVK keeps its other bits. A W destination's upper
        // half becomes zero.
        const unsigned shift = (site->instruction >> 21 & 3) * 16;
        std::uint64_t value = std::uint64_t{site->immediate} << shift;
        if ((site->instruction >> 29 & 3) == 3)
            value |= read_register(call, site->reg) & ~(std::uint64_t{0xffff} << shift);
        if (!(site->instruction >> 31)) value &= 0xffffffffu;
        write_register(call, site->reg, value);
        break;
    }
    }
    ++scope.dispatched_;
    return GateStatus::resume;
}

GateStatus GuestDispatchScope::dispatch_call_hook(GateCall& call, const CallHookSite& site) {
    if (hook_.handler == nullptr) {
        return fail(call, GuestDispatchFailureKind::missing_call_hook_handler,
                    describe(GuestDispatchFailureKind::missing_call_hook_handler), 0);
    }

    GuestThreadContext& thread = *thread_;
    struct Rollback {
        GuestThreadContext& thread;
        std::uint64_t thread_pointer;
        std::uint64_t read_only_thread_pointer;
        std::optional<GuestCounter> counter;
        bool committed = false;
        ~Rollback() {
            if (!committed) {
                thread.cell_.thread_pointer = thread_pointer;
                thread.cell_.read_only_thread_pointer = read_only_thread_pointer;
                thread.counter_ = counter;
            }
        }
    } rollback{thread, thread.cell_.thread_pointer, thread.cell_.read_only_thread_pointer,
               thread.counter_};
    const auto original_sp = call.sp();
    const auto gate_return = call.x(kGateLinkRegister);

    GateStatus status = GateStatus::failed;
    try {
        status = hook_.handler(site, call, thread, hook_.context);
    } catch (...) {
        (void)fail(call, GuestDispatchFailureKind::call_hook_threw,
                   describe(GuestDispatchFailureKind::call_hook_threw), 0);
        throw;
    }
    if (status == GateStatus::exit_thread && !call.can_exit_thread()) {
        return fail(call, GuestDispatchFailureKind::call_hook_failed,
                    "thread exit requires an active innermost guest entry", 0);
    }
    if (status != GateStatus::resume && status != GateStatus::exit_thread) {
        const char* const message = call.failure_message() != nullptr
                                        ? call.failure_message()
                                        : describe(GuestDispatchFailureKind::call_hook_failed);
        return fail(call, GuestDispatchFailureKind::call_hook_failed, message, 0);
    }
    if (call.frame().x[kGateHostReservedRegister] != 0 || call.sp() != original_sp ||
        call.x(kGateLinkRegister) != gate_return) {
        return fail(call, GuestDispatchFailureKind::invalid_call_hook_frame,
                    describe(GuestDispatchFailureKind::invalid_call_hook_frame), 0);
    }
    rollback.committed = true;
    ++dispatched_;
    return status;
}

GateStatus GuestDispatchScope::dispatch_cache(GateCall& call, const GuestSite& site) noexcept {
    GuestThreadContext& thread = *thread_;
    const auto operation = site.kind == GuestSiteKind::clean_invalidate_data_cache
                               ? CacheMaintenanceOperation::clean_invalidate_data
                               : CacheMaintenanceOperation::invalidate_instruction;
    const GuestCacheMaintenance handler = thread.cache_maintenance();
    if (handler.apply == nullptr) {
        return fail(call, GuestDispatchFailureKind::cache_maintenance_unavailable,
                    describe(GuestDispatchFailureKind::cache_maintenance_unavailable),
                    site.instruction);
    }
    // The operation names a guest virtual address in the site's register and
    // leaves that register unchanged.
    if (!handler.apply(handler.context, operation, read_register(call, site.reg))) {
        return fail(call, GuestDispatchFailureKind::cache_maintenance_unavailable,
                    describe(GuestDispatchFailureKind::cache_maintenance_unavailable),
                    site.instruction);
    }
    ++dispatched_;
    return GateStatus::resume;
}

GateStatus GuestDispatchScope::dispatch_counter(GateCall& call, const GuestSite& site) noexcept {
    GuestThreadContext& thread = *thread_;
    if (!thread.counter_) {
        return fail(call, GuestDispatchFailureKind::counter_unavailable,
                    describe(GuestDispatchFailureKind::counter_unavailable), site.instruction);
    }
    const GuestCounter& counter = *thread.counter_;
    if (site.kind == GuestSiteKind::read_counter_frequency) {
        write_register(call, site.reg, counter.frequency);
        ++dispatched_;
        return GateStatus::resume;
    }
    const std::uint64_t ticks = counter.read_ticks(counter.context);
    if (thread.last_counter_ && ticks < *thread.last_counter_) {
        return fail(call, GuestDispatchFailureKind::counter_regressed,
                    describe(GuestDispatchFailureKind::counter_regressed), site.instruction);
    }
    thread.last_counter_ = ticks;
    write_register(call, site.reg, ticks);
    ++dispatched_;
    return GateStatus::resume;
}

GateStatus GuestDispatchScope::dispatch_supervisor(GateCall& call, const GuestSite& site) {
    if (supervisor_.handler == nullptr) {
        return fail(call, GuestDispatchFailureKind::missing_supervisor_handler,
                    describe(GuestDispatchFailureKind::missing_supervisor_handler),
                    site.instruction);
    }

    // The gate restores the frame and the entry's virtual x18 (this
    // context's cell x18) when an entry fails; the context's thread pointers
    // and counter are restored here to match.
    GuestThreadContext& thread = *thread_;
    struct Rollback {
        GuestThreadContext& thread;
        std::uint64_t thread_pointer;
        std::uint64_t read_only_thread_pointer;
        std::optional<GuestCounter> counter;
        bool committed = false;
        ~Rollback() {
            if (!committed) {
                thread.cell_.thread_pointer = thread_pointer;
                thread.cell_.read_only_thread_pointer = read_only_thread_pointer;
                thread.counter_ = counter;
            }
        }
    } rollback{thread, thread.cell_.thread_pointer, thread.cell_.read_only_thread_pointer,
               thread.counter_};
    const auto original_sp = call.sp();

    GateStatus status = GateStatus::failed;
    try {
        status = supervisor_.handler(site.immediate, call, thread, supervisor_.context);
    } catch (...) {
        (void)fail(call, GuestDispatchFailureKind::supervisor_threw,
                   describe(GuestDispatchFailureKind::supervisor_threw), site.instruction);
        throw;
    }
    if (status == GateStatus::exit_thread && !call.can_exit_thread()) {
        return fail(call, GuestDispatchFailureKind::supervisor_failed,
                    "thread exit requires an active innermost guest entry", site.instruction);
    }
    if (status != GateStatus::resume && status != GateStatus::exit_thread) {
        const char* const message = call.failure_message() != nullptr
                                        ? call.failure_message()
                                        : describe(GuestDispatchFailureKind::supervisor_failed);
        return fail(call, GuestDispatchFailureKind::supervisor_failed, message, site.instruction);
    }
    if (call.frame().x[kGateHostReservedRegister] != 0 || call.sp() != original_sp) {
        return fail(call, GuestDispatchFailureKind::invalid_supervisor_frame,
                    describe(GuestDispatchFailureKind::invalid_supervisor_frame),
                    site.instruction);
    }
    rollback.committed = true;
    ++dispatched_;
    return status;
}

}  // namespace armigo::execution
