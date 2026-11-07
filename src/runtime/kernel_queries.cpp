// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/kernel_queries.h>
#include <armigo/common/diagnostics.h>
#include <cxxabi.h>
#include <algorithm>
#include <string>
#include <typeinfo>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <array>
#include <armigo/runtime/guest_clock.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <thread>

namespace armigo::runtime {
namespace {

execution::GateStatus finish_sleep(execution::GateCall& call) {
    for (unsigned index = 0; index < 18; ++index) call.set_x(index, 0);
    return execution::GateStatus::resume;
}

std::chrono::nanoseconds sleep_duration(std::int64_t nanoseconds) {
    // The baseline rounds nanoseconds up to 19.2 MHz ticks and adds two ticks.
    // Saturate when the corresponding host duration cannot be represented.
    const auto count = static_cast<std::uint64_t>(nanoseconds);
    const auto ticks = count / 625 * 12 + ((count % 625) * 12 + 624) / 625 + 2;
    const auto part = (ticks % 12) * 625 / 12;
    const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (ticks / 12 > (limit - part) / 625) return std::chrono::nanoseconds::max();
    return std::chrono::nanoseconds(ticks / 12 * 625 + part);
}

// A host yield that returns sooner than this ran no other thread.
constexpr auto lone_yield = std::chrono::microseconds(20);

// Calls in which the console's kernel may switch the calling thread off its core.
bool may_block(std::uint16_t immediate) noexcept {
    switch (immediate) {
    case 0x18: case 0x1a: case 0x1c: case 0x21: case 0x22: case 0x34: return true;
    default: return false;
    }
}

execution::GateStatus dispatch(std::uint16_t immediate, execution::GateCall& call,
                                execution::GuestThreadContext&, void* context) {
    auto& queries = *static_cast<horizon::StartupQueries*>(context);
    horizon::QueryRegisters registers;
    for (unsigned index = 0; index < registers.size(); ++index) registers[index] = call.x(index);
    const auto disposition = queries.dispatch(immediate, registers);
    if (disposition == horizon::QueryDisposition::cancelled)
        return call.can_exit_thread() ? execution::GateStatus::exit_thread : execution::GateStatus::failed;
    if (disposition != horizon::QueryDisposition::handled) {
        call.set_failure_message("The Horizon query capability is not implemented.");
        return execution::GateStatus::failed;
    }
    for (unsigned index = 0; index < registers.size(); ++index) call.set_x(index, registers[index]);
    return execution::GateStatus::resume;
}

} // namespace

execution::GuestSupervisor query_supervisor(horizon::StartupQueries& queries) noexcept {
    return {dispatch, &queries};
}

GuestSleep classify_guest_sleep(std::int64_t nanoseconds) noexcept {
    if (nanoseconds > 0) return GuestSleep::duration;
    // Horizon's yield types are 0 (without core migration), -1 (with core migration) and
    // -2 (to any thread). Every other negative value, including -3, is ignored.
    if (nanoseconds >= -2) return GuestSleep::yield;
    return GuestSleep::ignore;
}

execution::GateStatus StartupSupervisor::dispatch(std::uint16_t immediate,
                                                  execution::GateCall& call,
                                                  execution::GuestThreadContext&,
                                                  void* context) {
    auto& self = *static_cast<StartupSupervisor*>(context);
    // svcBreak(reason, address, size): the guest's own fatal path. It is reported
    // with the parameters the guest passed instead of a generic capability message.
    constexpr std::uint16_t break_immediate = 0x26;
    if (immediate == break_immediate) {
        auto length = horizon::describe_break(call.x(0), call.x(1), call.x(2), self.break_message_,
                                              sizeof self.break_message_);
        // A four-byte payload is the result code the guest aborted on.
        std::array<std::byte, 4> payload{};
        if (call.x(2) == 4 && self.caller_.read_memory &&
            self.caller_.read_memory(self.caller_.memory_context, call.x(1), payload) &&
            length < sizeof self.break_message_) {
            std::uint32_t result = 0;
            std::memcpy(&result, payload.data(), sizeof result);
            std::snprintf(self.break_message_ + length, sizeof self.break_message_ - length,
                          " Result 0x%x (module %u, description %u).", result, result & 0x1ffu,
                          (result >> 9) & 0x1fffu);
        }
        // ARMIGO_TRACE_SUPERVISOR=1 also prints the guest's return chain: x30 and
        // the return addresses of up to 16 AAPCS64 frame records from x29.
        static const bool trace = std::getenv("ARMIGO_TRACE_SUPERVISOR") != nullptr;
        if (trace && self.caller_.read_memory) {
            char chain[512];
            auto used = std::snprintf(chain, sizeof chain, "break x30=%llx frames",
                                      static_cast<unsigned long long>(call.x(30)));
            auto frame = call.x(29);
            for (unsigned depth = 0; depth != 16 && frame && !(frame & 15) && used > 0 &&
                                     static_cast<std::size_t>(used) < sizeof chain; ++depth) {
                std::array<std::byte, 16> record{};
                if (!self.caller_.read_memory(self.caller_.memory_context, frame, record)) break;
                std::uint64_t next = 0, link = 0;
                std::memcpy(&next, record.data(), sizeof next);
                std::memcpy(&link, record.data() + 8, sizeof link);
                used += std::snprintf(chain + used, sizeof chain - static_cast<std::size_t>(used), " %llx",
                                      static_cast<unsigned long long>(link));
                if (next <= frame) break;
                frame = next;
            }
            diag::opt_in("%s\n", chain);
        }
        call.set_failure_message(self.break_message_);
        return execution::GateStatus::failed;
    }
    // svcSleepThread(nanoseconds): the calling thread waits on the host for the duration the
    // guest asked for. The yield types 0, -1 and -2 use a host yield. Host scheduling
    // does not reproduce the firmware's migration policy. Other negative values do nothing.
    constexpr std::uint16_t sleep_immediate = 0x0b;
    if (immediate == sleep_immediate) {
        const auto nanoseconds = static_cast<std::int64_t>(call.x(0));
        switch (classify_guest_sleep(nanoseconds)) {
        case GuestSleep::duration: {
            self.queries_->thread()->process()->note_scheduling();
            if (self.clock_ != nullptr) {
                const auto sample = self.clock_->sample();
                if (std::holds_alternative<LaunchFailure>(sample)) {
                    call.set_failure_message("Process clock sample failed.");
                    return execution::GateStatus::failed;
                }
                const auto now = std::get<ClockSample>(sample).instant;
                const auto duration = sleep_duration(nanoseconds);
                const auto deadline = duration > ClockInstant::max() - now
                    ? ClockInstant::max() : now + duration;
                const auto waited = self.clock_->wait_until(deadline, self.queries_->cancellation_token());
                if (std::holds_alternative<LaunchFailure>(waited)) {
                    call.set_failure_message("Process clock wait failed.");
                    return execution::GateStatus::failed;
                }
                if (std::get<ClockWait>(waited) == ClockWait::cancelled)
                    return call.can_exit_thread() ? execution::GateStatus::exit_thread : execution::GateStatus::failed;
                return finish_sleep(call);
            }
            if (self.queries_->cancellation_token().stop_possible()) {
                call.set_failure_message("A cancellable sleep requires the process clock.");
                return execution::GateStatus::failed;
            }
            std::this_thread::sleep_for(sleep_duration(nanoseconds));
            return finish_sleep(call);
        }
        case GuestSleep::yield: {
            // Horizon's yields return at once while the thread's yield count equals the
            // process's scheduled count: nothing was scheduled since this thread last
            // yielded with no other thread ready. Otherwise the kernel counts the yield and,
            // when no other thread was ready, records the new count for the thread. The host
            // cannot see whether its yield ran another thread; a quick return counts as none.
            auto& thread = *self.queries_->thread();
            auto& process = *thread.process();
            if (thread.lone_yield_count() != process.scheduled_count()) {
                const auto start = std::chrono::steady_clock::now();
                std::this_thread::yield();
                const auto alone = std::chrono::steady_clock::now() - start < lone_yield;
                const auto count = process.note_scheduling();
                if (alone) thread.set_lone_yield_count(count);
            }
            return finish_sleep(call);
        }
        case GuestSleep::ignore:
            return finish_sleep(call);
        }
    }
    // A call that may block leaves and later re-enters a core, so it is a scheduling
    // event on both sides for other threads' yields.
    struct SchedulingPoint {
        horizon::ProcessRecord* process;
        explicit SchedulingPoint(horizon::ProcessRecord* record) : process(record) {
            if (process) process->note_scheduling();
        }
        ~SchedulingPoint() { if (process) process->note_scheduling(); }
    } scheduling_point(may_block(immediate) ? self.queries_->thread()->process().get() : nullptr);
    horizon::QueryRegisters registers;
    for (unsigned i = 0; i < registers.size(); ++i) registers[i] = call.x(i);
    if (self.memory_) {
        auto result = self.memory_->dispatch(immediate, registers);
        if (auto* failure = std::get_if<LaunchFailure>(&result)) {
            // Private interval capacity is a recoverable kernel resource error.
            // The memory transaction has not changed native mappings or accounting.
            if (failure->stage == LaunchStage::memory && failure->code == LaunchError::resource_limit &&
                failure->guest_result == 0xce01) {
                for (unsigned i = 0; i < registers.size(); ++i) call.set_x(i, 0);
                call.set_x(0, *failure->guest_result);
                return execution::GateStatus::resume;
            }
            self.memory_failure_ = std::move(*failure);
            call.set_failure_message(self.memory_failure_->detail.c_str());
            return execution::GateStatus::failed;
        }
        if (std::get<horizon::QueryDisposition>(result) == horizon::QueryDisposition::handled) {
            for (unsigned i = 0; i < registers.size(); ++i) call.set_x(i, registers[i]);
            return execution::GateStatus::resume;
        }
    }
    const auto disposition = self.queries_->dispatch(immediate, registers);
    if (disposition == horizon::QueryDisposition::cancelled)
        return call.can_exit_thread() ? execution::GateStatus::exit_thread : execution::GateStatus::failed;
    if (disposition != horizon::QueryDisposition::handled) {
        const auto result = self.ipc_->dispatch(immediate, registers, self.caller_);
        switch (result) {
        case horizon::IpcDisposition::handled: break;
        case horizon::IpcDisposition::cancelled:
            call.set_failure_message("The IPC request was cancelled by host shutdown.");
            return call.can_exit_thread() ? execution::GateStatus::exit_thread
                                          : execution::GateStatus::failed;
        case horizon::IpcDisposition::pending:
            call.set_failure_message("The IPC request is pending; its guest continuation cannot resume.");
            return execution::GateStatus::failed;
        case horizon::IpcDisposition::unsupported: {
            // Name the session and command, so the unsupported request is known
            // without a trace.
            self.ipc_failure_detail_ = "The requested Horizon startup capability is not implemented.";
            if (immediate != 0x21)
                self.ipc_failure_detail_ += std::string(" Supervisor call 0x") + [&] {
                    char hex[8];
                    std::snprintf(hex, sizeof hex, "%x", immediate);
                    return std::string(hex);
                }() + (self.queries_->unsupported_detail()
                           ? std::string(": ") + self.queries_->unsupported_detail()
                           : std::string("."));
            if (immediate == 0x29) {
                char query[160];
                std::snprintf(query, sizeof query, " GetInfo type=%#llx handle=%#llx sub-id=%#llx.",
                    static_cast<unsigned long long>(call.x(1)),
                    static_cast<unsigned long long>(call.x(2)),
                    static_cast<unsigned long long>(call.x(3)));
                self.ipc_failure_detail_ += query;
            }
            if (immediate == 0x21) {
                if (const auto object = self.ipc_->handles().lookup(static_cast<horizon::Handle>(call.x(0)))) {
                    int status = 0;
                    char* name = abi::__cxa_demangle(typeid(*object).name(), nullptr, nullptr, &status);
                    self.ipc_failure_detail_ += std::string(" Session ") + (status == 0 && name ? name : typeid(*object).name());
                    std::free(name);
                    horizon::IpcMessage message{};
                    const auto size = std::min(message.size(), self.caller_.message.size());
                    std::memcpy(message.data(), self.caller_.message.data(), size);
                    if (const auto cmif = horizon::locate_cmif(message)) {
                        std::uint32_t command = 0;
                        std::memcpy(&command, message.data() + *cmif + 8, sizeof command);
                        self.ipc_failure_detail_ += ", command " + std::to_string(command);
                    } else {
                        std::uint32_t type = 0;
                        std::memcpy(&type, message.data(), sizeof type);
                        self.ipc_failure_detail_ += ", message type " + std::to_string(type & 0xffffu);
                    }
                    self.ipc_failure_detail_ += ".";
                }
            }
            call.set_failure_message(self.ipc_failure_detail_.c_str());
            return execution::GateStatus::failed;
        }
        case horizon::IpcDisposition::invalid_backend:
            call.set_failure_message("The Horizon IPC backend supplied invalid state.");
            return execution::GateStatus::failed;
        }
    }
    for (unsigned i = 0; i < registers.size(); ++i) call.set_x(i, registers[i]);
    return execution::GateStatus::resume;
}

execution::GuestSupervisor StartupSupervisor::binding() noexcept {
    return {dispatch, this};
}

} // namespace armigo::runtime
