// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/guest_timeline.h>
#include <armigo/common/diagnostics.h>
#include <armigo/runtime/process_runtime.h>
#include <armigo/runtime/kernel_queries.h>
#include <armigo/romfs/romfs_virtual.h>
#include <armigo/romfs/nup_storage.h>
#include <armigo/runtime/addon_content.h>
#include <armigo/compat/title_settings.h>
#include <armigo/execution/guest_fault.h>
#include <armigo/memory/write_tracker.h>
#include <armigo/nvn/driver.h>
#include <armigo/services/file_system.h>
#include "service_registry.h"
#include "frame_capture.h"
#ifdef ARMIGO_RUNTIME_TESTING
#include "process_runtime_test_access.h"
#endif
#include "../horizon/thread_lifecycle.h"
#include "../platform/process_memory.h"

#include <libkern/OSCacheControl.h>

#include <algorithm>
#include <typeinfo>
#include <cstdio>
#include <array>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <future>
#include <thread>
#include <utility>

namespace armigo::runtime {
namespace {
using execution::GateCall;
using execution::GateStatus;
using execution::GuestThreadContext;
using execution::CacheMaintenanceOperation;
constexpr auto rw = memory::Permission::read | memory::Permission::write;

LaunchFailure fail(LaunchStage stage, LaunchError code, std::string operation,
                   std::string detail) {
    return {.stage = stage, .code = code, .operation = std::move(operation),
            .detail = std::move(detail)};
}

bool read_memory(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
    return static_cast<GuestMemory*>(raw)->ipc_buffers().read_memory(address, out);
}
bool write_memory(void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
    return static_cast<GuestMemory*>(raw)->ipc_buffers().write_memory(address, in);
}
bool read_byte(void* raw, std::uint64_t address, std::byte& out) noexcept {
    return read_memory(raw, address, {&out, 1});
}
horizon::GuestCopyOut checked_copy(GuestMemory* memory) {
    // Preserve C's native atomic adapters while adding ordinary unaligned copies
    // required by D's WaitSynchronization array validation.
    return {
        .write_word = [](void* raw, std::uint64_t address, std::span<const std::byte, 4> in) noexcept {
            const auto copy = static_cast<GuestMemory*>(raw)->copy_out();
            return copy.write_word && copy.write_word(copy.context, address, in);
        },
        .read_word = [](void* raw, std::uint64_t address, std::span<std::byte, 4> out) noexcept {
            const auto copy = static_cast<GuestMemory*>(raw)->copy_out();
            return copy.read_word && copy.read_word(copy.context, address, out);
        },
        .context = memory,
        .compare_exchange_word = [](void* raw, std::uint64_t address,
                                    std::uint32_t& expected, std::uint32_t desired) noexcept {
            const auto copy = static_cast<GuestMemory*>(raw)->copy_out();
            return copy.compare_exchange_word(copy.context, address, expected, desired);
        },
        .read_double_word = [](void* raw, std::uint64_t address, std::uint64_t& out) noexcept {
            const auto copy = static_cast<GuestMemory*>(raw)->copy_out();
            return copy.read_double_word && copy.read_double_word(copy.context, address, out);
        },
        .read_bytes = read_memory,
    };
}

LaunchResult<std::vector<std::byte>> read_metadata(const std::filesystem::path& path,
                                                std::size_t limit) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(std::filesystem::symlink_status(path, error)) || error)
        return fail(LaunchStage::inputs, LaunchError::invalid_input, "read-npdm",
                    "Detached NPDM must be a regular, non-symlink file.");
    const auto size = std::filesystem::file_size(path, error);
    if (error || !size || size > limit || size > std::uint64_t(std::numeric_limits<std::streamsize>::max()))
        return fail(LaunchStage::inputs, LaunchError::invalid_input, "read-npdm",
                    "Detached NPDM size is invalid or exceeds the input limit.");
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)) ||
        input.peek() != std::char_traits<char>::eof())
        return fail(LaunchStage::inputs, LaunchError::host_failure, "read-npdm",
                    "Detached NPDM could not be read at its observed size.");
    return bytes;
}

LaunchStatus validate_synthetic(const LaunchProfile& profile) {
    const auto& permissions = profile.permissions;
    const auto& main = profile.main_thread;
    if (profile.origin != ProfileOrigin::synthetic_test || !profile.is_64_bit ||
        profile.process.baseline != horizon::Baseline::horizon_23_0_1 ||
        profile.process.address_bits != 39 || !profile.process.allowed_core_mask ||
        (profile.process.allowed_core_mask & ~std::uint64_t{15}) ||
        profile.process.private_system_resource_bytes || profile.process.alias_extra_enabled)
        return fail(LaunchStage::profile, LaunchError::unsupported, "synthetic-profile",
                    "Synthetic input must use the supported 64-bit process capability subset.");
    if (!permissions.handle_table_capacity || permissions.handle_table_capacity > 0x400 ||
        permissions.highest_priority > permissions.lowest_priority || permissions.lowest_priority > 63 ||
        main.priority < permissions.highest_priority || main.priority > permissions.lowest_priority ||
        main.ideal_core < 0 || main.ideal_core > 3 ||
        !(profile.process.allowed_core_mask & (std::uint64_t{1} << main.ideal_core)) ||
        !main.stack_size || (main.stack_size & 0xfff) ||
        main.stack_size < execution::kGuestEntryDefaultMinimumStack ||
        !std::is_sorted(permissions.supervisor_calls.begin(), permissions.supervisor_calls.end()) ||
        std::adjacent_find(permissions.supervisor_calls.begin(), permissions.supervisor_calls.end()) !=
            permissions.supervisor_calls.end() ||
        (!permissions.supervisor_calls.empty() && permissions.supervisor_calls.back() > 0xbf))
        return fail(LaunchStage::profile, LaunchError::invalid_input, "synthetic-profile",
                    "Synthetic handles, priorities, cores, stack or SVC permissions are invalid.");
    if (permissions.program_type > 2 || permissions.debug_flags & ~3u)
        return fail(LaunchStage::profile, LaunchError::unsupported, "synthetic-profile",
                    "Synthetic program or debug capabilities are unsupported.");
    return std::monostate{};
}

// Publishes the process clock as the guest timeline subsystems below the
// runtime (the NVN driver) put their times on. A clock that cannot sample is
// not published.
void publish_guest_timeline(GuestClock* clock) noexcept {
    if (!clock) return;
    const auto sample = clock->sample();
    const auto* sampled = std::get_if<ClockSample>(&sample);
    if (!sampled) return;
    execution::publish_guest_timeline({
        .elapsed = [](void* context) noexcept -> std::optional<std::chrono::nanoseconds> {
            const auto now = static_cast<GuestClock*>(context)->sample();
            if (const auto* taken = std::get_if<ClockSample>(&now)) return taken->counter_time;
            return std::nullopt;
        },
        .context = clock,
        .origin = sampled->instant - sampled->counter_time});
}

class NativeProcessRuntime final : public ProcessRuntime {
#ifdef ARMIGO_RUNTIME_TESTING
    friend class armigo::runtime::ProcessRuntimeTestAccess;
    std::function<void(std::stop_token)> failed_ipc_boundary_;
    std::function<void(std::stop_token)> ipc_admission_boundary_;
#endif
public:
    NativeProcessRuntime(LaunchProfile profile, RuntimeDependencies dependencies,
                         PreparedLaunch launch, std::shared_ptr<horizon::HandleTable> handles,
                         std::shared_ptr<GuestMemory> memory,
                         services::OperationMode operation_mode)
        : profile_(std::move(profile)), dependencies_(std::move(dependencies)),
          launch_(std::move(launch)), handles_(std::move(handles)), memory_(std::move(memory)),
          arbitration_(std::make_shared<horizon::Arbitration>()),
          keys_(std::make_shared<horizon::ProcessWideKey>(arbitration_)),
          operation_mode_(operation_mode) {
        publish_guest_timeline(dependencies_.clock.get());
    }
    ~NativeProcessRuntime() override {
        // Unbounded destruction is intentional: elapsed time never permits
        // retirement beneath native execution. Shells retain timed-out owners.
        if (std::holds_alternative<LaunchFailure>(shutdown(std::nullopt))) std::terminate();
        release_nvn_handler();
        execution::withdraw_guest_timeline(dependencies_.clock.get());
    }

    LaunchStatus initialize(std::shared_ptr<romfs::Storage> romfs,
                            std::shared_ptr<const services::AddOnContentCatalog> addon_content,
                            const std::filesystem::path& shader_directory,
                            std::shared_ptr<PendingShaderPackage> shader_package,
                            std::optional<std::string> display_version = {});
    RunResult run(const RunLimits&, const ReportDestination&, std::stop_token) override;
    void request_stop(StopReason reason) noexcept override {
        {
            const std::lock_guard lock(mutex_);
            if (!outcome_) outcome_ = RunCancelled{reason};
        }
        stop_execution();
    }
    LaunchStatus shutdown(std::optional<ClockInstant> deadline) override;
    void retain_failed_construction(std::shared_ptr<ProcessRuntime> owner) {
        retained_construction_owner_ = std::move(owner);
    }

private:
    struct ThreadBinding;
    RunResult run_impl(const RunLimits&, const ReportDestination&, std::stop_token);
    LaunchResult<ThreadServiceBinding> bind(std::shared_ptr<horizon::ThreadRecord>,
                                           GuestThreadContext&, std::stop_token);
    GateStatus create_worker(GateCall& call);
    GateStatus start_worker(GateCall& call);
    void record_failure(LaunchFailure failure) {
        {
            const std::lock_guard lock(mutex_);
            if (failure.code == LaunchError::cancelled && stopping_ && outcome_) return;
            // A failure observed while completing a selected stop remains the
            // causal result; teardown errors have their own channel.
            if (!outcome_ || !std::holds_alternative<LaunchFailure>(*outcome_))
                outcome_ = std::move(failure);
        }
        changed_.notify_all();
    }
    // An NVN entry point the selected driver cannot serve ends the run with an
    // unsupported result; its caller never returns to the game.
    static void unsupported_nvn_call(void* context, const char* detail, std::uint64_t guest_return) noexcept {
        auto& self = *static_cast<NativeProcessRuntime*>(context);
        try {
            auto failure = fail(LaunchStage::graphics, LaunchError::unsupported, "nvn-entry", detail);
            failure.guest_pc = guest_return;
            self.record_failure(std::move(failure));
        } catch (...) {
        }
        self.stop_execution();
    }
    // The handler names this runtime, so it is installed by it and withdrawn
    // once its guest threads are gone, before the runtime can be destroyed.
    void install_nvn_handler() noexcept {
        nvn::set_unsupported_handler(&unsupported_nvn_call, this);
        nvn_handler_installed_ = true;
    }
    void release_nvn_handler() noexcept {
        if (!nvn_handler_installed_) return;
        nvn::set_unsupported_handler(nullptr, nullptr);
        nvn_handler_installed_ = false;
    }
    void record_teardown_failure(LaunchFailure failure) {
        failure.stage = LaunchStage::shutdown;
        const std::lock_guard lock(mutex_);
        if (!teardown_failure_) teardown_failure_ = std::move(failure);
    }
    GateStatus failure_at_gate(LaunchFailure failure, GateCall& call) {
        failure.guest_pc = call.return_address();
        record_failure(std::move(failure));
        return GateStatus::exit_thread;
    }
    void guest_exit() {
        if (const auto failure = threads_->failure()) record_failure(*failure);
        if (const auto failure = services_->failure()) record_failure(*failure);
        // ExitProcess observes completed VI releases at this boundary. A target
        // already reached wins; a release occurring after exit cannot rewrite it.
        const auto completed = services_->statistics().completed_frames;
        {
            const std::lock_guard lock(mutex_);
            if (!outcome_) {
                if (completed_frame_target_ && completed >= *completed_frame_target_)
                    outcome_ = RunSuccess{*completed_frame_target_};
                else
                    outcome_ = GuestExit{};
            }
        }
        stop_execution();
    }
    void stop_execution() noexcept {
        std::shared_ptr<ThreadRuntime> threads;
        std::shared_ptr<ServiceRegistry> services;
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
            threads = threads_;
            services = services_;
        }
        // Process termination ends every wait a thread is parked in, including
        // the ones a deferred stop cannot interrupt (see ThreadBinding). It is
        // requested first, so a wait cancelled by any stop below already sees it.
        termination_.request_stop();
        // Stop callbacks may acquire runtime state; invoke outside its lock.
        if (threads) threads->request_stop();
        if (services) services->request_stop();
        changed_.notify_all();
    }
    void force_stop() noexcept {
        forced_stop_.request_stop();
        changed_.notify_all();
    }
    bool forced() const noexcept { return forced_stop_.stop_requested(); }
    LaunchResult<std::shared_ptr<void>> acquire_graphics_access();
    LaunchStatus observe_frame(const platform::PresentedFrame& frame);
    // Names every thread still alive when a shutdown misses its deadline: its
    // supervisor call, whether the deferred supervisor runs it, and for a lock
    // wait the guest word and the thread that owns the lock.
    void describe_live_threads() noexcept {
        try {
            std::vector<std::shared_ptr<horizon::ThreadRecord>> live;
            {
                const std::lock_guard lock(mutex_);
                for (const auto& weak : started_)
                    if (const auto record = weak.lock(); record && !record->exited()) live.push_back(record);
            }
            const auto copy = memory_->copy_out();
            const auto owner_text = [&](std::uint64_t address) {
                std::array<std::byte, 4> bytes{};
                if (!copy.read_word || !copy.read_word(copy.context, address, bytes)) return std::string(" word=unreadable");
                std::uint32_t word = 0;
                for (unsigned i = 0; i != 4; ++i) word |= std::to_integer<std::uint32_t>(bytes[i]) << (8 * i);
                char text[160];
                const auto handle = word & ~std::uint32_t{0x40000000};
                const auto owner = handles_->lookup_as<horizon::ThreadRecord>(handle);
                std::snprintf(text, sizeof text, " word=%08x owner_handle=%08x owner=%s", word, handle,
                              !handle ? "none" : !owner ? "unknown" : owner->exited() ? "exited" : "live");
                if (owner) std::snprintf(text + std::strlen(text), sizeof text - std::strlen(text), " owner_thread=%llu",
                                         static_cast<unsigned long long>(owner->id()));
                return std::string(text);
            };
            armigo::diag::write(armigo::diag::Level::error,  "shutdown: %zu guest thread(s) still alive at the deadline\n", live.size());
            for (const auto& record : live) {
                const auto note = record->call_note();
                const char* phase = note.phase == horizon::ThreadRecord::CallPhase::guest ? "guest-code"
                    : note.phase == horizon::ThreadRecord::CallPhase::call ? "in-call" : "retained-under-grant";
                std::string line = "shutdown: thread " + std::to_string(record->id()) + " " + phase;
                char text[200];
                if (note.phase != horizon::ThreadRecord::CallPhase::guest || note.svc) {
                    std::snprintf(text, sizeof text, " svc=0x%02x supervisor=%s x0=%llx x1=%llx x2=%llx", note.svc,
                                  note.deferred ? "deferred" : "ordinary",
                                  static_cast<unsigned long long>(note.arguments[0]),
                                  static_cast<unsigned long long>(note.arguments[1]),
                                  static_cast<unsigned long long>(note.arguments[2]));
                    line += text;
                }
                if (note.phase == horizon::ThreadRecord::CallPhase::call && note.svc == 0x1a) line += owner_text(note.arguments[1]);
                if (note.grants) {
                    std::snprintf(text, sizeof text, " grants=%u first_grant=%llx tag=%08x", note.grants,
                                  static_cast<unsigned long long>(note.grant_address), note.grant_tag);
                    line += text + owner_text(note.grant_address);
                }
                armigo::diag::write(armigo::diag::Level::error,  "%s\n", line.c_str());
            }
        } catch (...) {
        }
    }
    bool all_started_exited() {
        const std::lock_guard lock(mutex_);
        return std::all_of(started_.begin(), started_.end(), [](const auto& weak) {
            const auto record = weak.lock();
            return !record || record->exited();
        });
    }

    const LaunchProfile profile_;
    RuntimeDependencies dependencies_;
    PreparedLaunch launch_;
    std::shared_ptr<horizon::HandleTable> handles_;
    std::shared_ptr<GuestMemory> memory_;
    std::shared_ptr<horizon::Arbitration> arbitration_;
    std::shared_ptr<horizon::ProcessWideKey> keys_;
    const services::OperationMode operation_mode_;
    std::shared_ptr<ServiceRegistry> services_;
    std::shared_ptr<ThreadRuntime> threads_;
    CreatedThread main_;
    GuestMemoryPin main_stack_;
    GuestMemoryPin exception_storage_;
    GuestMemoryPin host_arguments_;
    std::uint64_t process_id_ = 0;
    std::mutex mutex_;
    std::condition_variable changed_;
    // Requested once the process is being terminated (guest exit, run outcome,
    // host stop or shutdown). It never resets.
    std::stop_source termination_;
    // Requested when the grace period of a shutdown has passed. Process
    // termination ends every thread whatever user-mode lock it holds, so from
    // here on no observed mutex grant defers a thread's exit.
    std::stop_source forced_stop_;
    std::timed_mutex shutdown_mutex_;
    std::mutex graphics_start_mutex_;
    std::vector<std::weak_ptr<horizon::ThreadRecord>> started_;
    std::optional<RunOutcome> outcome_;
    std::optional<std::uint64_t> completed_frame_target_;
    std::optional<LaunchFailure> teardown_failure_;
    std::shared_ptr<ProcessRuntime> retained_construction_owner_;
    ReportDestination output_;
    RunStatistics final_statistics_;
    bool ran_ = false;
    bool stopping_ = false;
    bool quiescent_ = false;
    bool nvn_handler_installed_ = false;
};

struct NativeProcessRuntime::ThreadBinding {
    NativeProcessRuntime& runtime;
    std::shared_ptr<horizon::ThreadRecord> record;
    std::shared_ptr<GuestMemory> memory;
    GuestMemoryPin tls;
    // Host startup: the SDK's thread-local block, published at TLR+0x1f0.
    GuestMemoryPin storage;
    std::stop_token stop;
    horizon::StartupQueries queries;
    horizon::StartupQueries deferred_queries;
    horizon::IpcDispatcher ipc;
    StartupSupervisor supervisor;
    StartupSupervisor deferred_supervisor;
    // An observed kernel mutex grant must reach its release path. A previously
    // granted lock is tracked by address/tag, including releases in user code.
    std::map<std::uint64_t, std::uint32_t> grants;

    ThreadBinding(NativeProcessRuntime& owner, std::shared_ptr<horizon::ThreadRecord> thread,
                  GuestMemoryPin local, std::stop_token cancellation)
        : runtime(owner), record(thread), memory(owner.memory_), tls(std::move(local)), stop(cancellation),
          queries(memory->process(), *owner.handles_, thread, checked_copy(memory.get()),
                  owner.arbitration_, owner.keys_, cancellation),
          deferred_queries(memory->process(), *owner.handles_, std::move(thread), checked_copy(memory.get()),
                           owner.arbitration_, owner.keys_, owner.termination_.get_token()),
          ipc(*owner.handles_, [](void* raw, std::string_view port) {
              return static_cast<ServiceRegistry*>(raw)->connect_named_port(port);
          }, owner.services_.get()),
          supervisor(queries, ipc, {
              .process_id = owner.process_id_,
              .message = tls->writable_bytes().first(horizon::ipc_message_capacity),
              .read_byte = read_byte, .memory_context = memory.get(),
              .read_memory = read_memory, .write_memory = write_memory,
              .buffer_region_base = 0, .memory_mapper = memory->service_mapper(),
          }, owner.dependencies_.clock.get(), memory),
          deferred_supervisor(deferred_queries, ipc, {
              .process_id = owner.process_id_,
              .message = tls->writable_bytes().first(horizon::ipc_message_capacity),
              .read_byte = read_byte, .memory_context = memory.get(),
              .read_memory = read_memory, .write_memory = write_memory,
              .buffer_region_base = 0, .memory_mapper = memory->service_mapper(),
          }, owner.dependencies_.clock.get(), memory) {}
    ~ThreadBinding() {
        for (auto* pin : {&tls, &storage}) {
            if (!*pin) continue;
            const auto range = (*pin)->range();
            pin->reset();
            const auto retired = memory->release(range);
            if (const auto* failed = std::get_if<LaunchFailure>(&retired)) runtime.record_teardown_failure(*failed);
        }
    }
    bool owns_grant() {
        const auto copy = memory->copy_out();
        for (auto it = grants.begin(); it != grants.end();) {
            std::array<std::byte, 4> bytes{};
            std::uint32_t word = 0;
            if (!copy.read_word(copy.context, it->first, bytes)) {
                // Losing access is not proof that ownership was discharged.
                ++it;
                continue;
            }
            for (unsigned i = 0; i != 4; ++i) word |= std::to_integer<std::uint32_t>(bytes[i]) << (8 * i);
            // Only bit 30 is the waiter marker. D does not independently
            // validate the supplied tag, so bit 31 must survive comparison.
            constexpr std::uint32_t owner_mask = ~std::uint32_t{0x40000000};
            const auto owner = word & owner_mask;
            if (!owner || owner != (it->second & owner_mask)) it = grants.erase(it);
            else ++it;
        }
        return !grants.empty();
    }
    static GateStatus boundary(std::uint16_t, GateCall&, GuestThreadContext&, void* raw) {
        auto& self = *static_cast<ThreadBinding*>(raw);
        return self.exit_due() ? GateStatus::exit_thread : GateStatus::resume;
    }
    // A stopped thread leaves at a boundary unless it owns an observed mutex
    // grant; the forced stop ends it regardless.
    bool exit_due() {
        return stop.stop_requested() && (runtime.forced() || !owns_grant());
    }
    GateStatus complete(GateStatus status) {
        if (status == GateStatus::resume) {
            // Recheck after dispatch: a successful unlock may have released the
            // final grant that previously deferred cancellation.
            return exit_due() ? GateStatus::exit_thread : GateStatus::resume;
        }
        if (runtime.forced() || !owns_grant()) return status;
        // Admission may close while a real mutex grant is still owned. There
        // is no guest Result for that host stop and no safe continuation to
        // fabricate. Retain this invocation at its host boundary rather than
        // abandon the mutex or free its stack. A bounded join reports failure;
        // ordinary process supervision may terminate a permanently stuck owner.
        runtime.record_failure(fail(LaunchStage::threads, LaunchError::unsupported,
            "stop-with-mutex-grant",
            "A stopped operation cannot resume while its guest still owns an observed mutex grant; runtime resources remain retained."));
        if (!grants.empty())
            record->note_retained(grants.begin()->first, grants.begin()->second,
                                  static_cast<std::uint32_t>(grants.size()));
        while (!runtime.forced() && owns_grant()) {
            std::unique_lock lock(runtime.mutex_);
            runtime.changed_.wait_for(lock, std::chrono::milliseconds(10));
        }
        return GateStatus::exit_thread;
    }
    static GateStatus dispatch(std::uint16_t svc, GateCall& call, GuestThreadContext& context, void* raw) {
        auto& self = *static_cast<ThreadBinding*>(raw);
        try {
            const auto status = dispatch_impl(svc, call, context, raw);
            if (self.grants.empty()) self.record->note_guest(0, 0, 0);
            else self.record->note_guest(self.grants.begin()->first, self.grants.begin()->second,
                                         static_cast<std::uint32_t>(self.grants.size()));
            return status;
        }
        catch (...) {
            self.runtime.record_failure(fail(LaunchStage::threads, LaunchError::host_failure,
                "supervisor-call", "Runtime supervisor raised an exception."));
            return self.complete(GateStatus::failed);
        }
    }
    static GateStatus dispatch_impl(std::uint16_t svc, GateCall& call, GuestThreadContext& context, void* raw) {
        auto& self = *static_cast<ThreadBinding*>(raw);
        if (svc == 0x07) { self.runtime.guest_exit(); return GateStatus::exit_thread; }
        const bool owns = self.owns_grant();
        if (self.stop.stop_requested() && (!owns || self.runtime.forced())) return GateStatus::exit_thread;
        if (svc == 0x08) return self.complete(self.runtime.create_worker(call));
        if (svc == 0x09) return self.complete(self.runtime.start_worker(call));
        const auto mutex = svc == 0x1a ? call.x(1) : call.x(0);
        const auto tag = static_cast<std::uint32_t>(call.x(2));
        // A cancellation-raced grant is real ownership. Host cancellation is
        // deferred while the guest keeps running to its release, but a blocking
        // operation under the grant waits on the process-termination token, not
        // on the thread's stop: the owner may be waiting for a lock that no
        // surviving thread will ever release.
        // A CV wait that releases the only observed mutex can use D's stop token:
        // validation returns normally, or unlock precedes cancellable enrollment.
        const bool releases_only_grant = svc == 0x1c && self.grants.size() == 1 && self.grants.contains(mutex);
        auto& supervisor = owns && !releases_only_grant ? self.deferred_supervisor : self.supervisor;
        const auto binding = supervisor.binding();
        self.record->note_call(svc, &supervisor == &self.deferred_supervisor, call.x(0), call.x(1), call.x(2));
#ifdef ARMIGO_RUNTIME_TESTING
        if (svc == 0x1f || svc == 0x21) {
            std::function<void(std::stop_token)> boundary;
            {
                const std::lock_guard lock(self.runtime.mutex_);
                boundary = self.runtime.ipc_admission_boundary_;
            }
            if (boundary) boundary(self.stop);
        }
#endif
        // ARMIGO_TRACE_SUPERVISOR=1 prints every supervisor call with its
        // arguments and results, to follow the guest to its first failure.
        static const bool trace = std::getenv("ARMIGO_TRACE_SUPERVISOR") != nullptr;
        const std::array<std::uint64_t, 4> arguments{call.x(0), call.x(1), call.x(2), call.x(3)};
        if (trace && svc == 0x21) {
            horizon::IpcMessage message{};
            const auto bytes = self.tls->writable_bytes().first(horizon::ipc_message_capacity);
            std::memcpy(message.data(), bytes.data(), message.size());
            std::uint32_t command = 0xffffffff;
            if (const auto cmif = horizon::locate_cmif(message))
                std::memcpy(&command, message.data() + *cmif + 8, sizeof command);
            const auto object = self.runtime.handles_->lookup(static_cast<horizon::Handle>(arguments[0]));
            // The return addresses of the first frame records name the caller.
            std::array<std::uint64_t, 8> callers{};
            for (std::uint64_t frame = call.x(29), depth = 0; depth != callers.size() && frame && !(frame & 15); ++depth) {
                std::array<std::byte, 16> pair{};
                if (!self.memory->ipc_buffers().read_memory(frame, pair)) break;
                std::uint64_t next = 0;
                std::memcpy(&next, pair.data(), sizeof next);
                std::memcpy(&callers[depth], pair.data() + 8, sizeof callers[depth]);
                if (next <= frame) break;
                frame = next;
            }
            armigo::diag::opt_in( "ipc handle=%llx command=%u object=%s from=",
                         static_cast<unsigned long long>(arguments[0]), command,
                         object ? typeid(*object).name() : "-");
            for (std::size_t index = 0; index != callers.size(); ++index)
                armigo::diag::opt_in(index ? ",%llx" : "%llx", static_cast<unsigned long long>(callers[index]));
            armigo::diag::opt_in(" words=");
            for (std::size_t offset = 0; offset < 64; offset += 4) {
                std::uint32_t word;
                std::memcpy(&word, message.data() + offset, sizeof word);
                armigo::diag::opt_in( offset ? " %08x" : "%08x", word);
            }
            armigo::diag::newline();
        }
        const auto status = binding.handler(svc, call, context, binding.context);
        if (trace)
            armigo::diag::opt_in( "svc %02x in=%llx,%llx,%llx,%llx out=%llx,%llx status=%d\n", svc,
                         static_cast<unsigned long long>(arguments[0]), static_cast<unsigned long long>(arguments[1]),
                         static_cast<unsigned long long>(arguments[2]), static_cast<unsigned long long>(arguments[3]),
                         static_cast<unsigned long long>(call.x(0)), static_cast<unsigned long long>(call.x(1)),
                         static_cast<int>(status));
        // Failures latched by the operation are collected first, so a run that
        // ends by termination still reports them.
        if (const auto& failure = supervisor.memory_failure()) self.runtime.record_failure(*failure);
        if (const auto failure = self.runtime.services_->failure()) self.runtime.record_failure(*failure);
        // A wait (sleep, WaitSynchronization, ArbitrateLock, condition or
        // address wait) woken by process termination ends the thread, as the
        // firmware's termination does to a thread parked in any wait, whatever
        // lock it owns. Other operations refused by the stop keep the retained
        // boundary below.
        // The grant dies with the process; no surviving thread can be waiting on
        // it, and the thread has no Result to resume with.
        const bool wait = svc == 0x0b || svc == 0x18 || svc == 0x1a || svc == 0x1c || svc == 0x34;
        if (wait && status == GateStatus::exit_thread && self.runtime.termination_.stop_requested())
            return status;
        if (svc == 0x1b && status == GateStatus::resume && call.x(0) == 0) self.grants.erase(mutex);
        if ((svc == 0x1a || svc == 0x1c) && status == GateStatus::resume && call.x(0) == 0) {
            self.grants[mutex] = tag;
            static_cast<void>(self.owns_grant());
        }
        static_cast<void>(self.owns_grant());
        if (status == GateStatus::failed) {
#ifdef ARMIGO_RUNTIME_TESTING
            if (svc == 0x1f || svc == 0x21) {
                std::function<void(std::stop_token)> boundary;
                {
                    const std::lock_guard lock(self.runtime.mutex_);
                    boundary = self.runtime.failed_ipc_boundary_;
                }
                if (boundary) boundary(self.stop);
            }
#endif
            if (self.stop.stop_requested() && self.owns_grant())
                return self.runtime.forced() ? GateStatus::exit_thread : self.complete(status);
            auto failure = fail(svc == 0x1f || svc == 0x21 ? LaunchStage::services : LaunchStage::threads,
                                svc == 0x26 ? LaunchError::guest_fault : LaunchError::unsupported,
                                "supervisor-call", call.failure_message() ? call.failure_message() :
                                "Supervisor operation is unsupported.");
            failure.command = svc;
            return self.complete(self.runtime.failure_at_gate(std::move(failure), call));
        }
        return self.complete(status);
    }
};

namespace {
constexpr std::uint64_t guest_cache_type = 0x8444c004;
bool apply_cache_maintenance(void* raw, CacheMaintenanceOperation operation, std::uint64_t address) noexcept {
    const auto* process = static_cast<const horizon::ProcessRecord*>(raw);
    if (!process) return false;
    const auto region = process->query_memory(address);
    if (region.type == horizon::MemoryType::free || region.permissions == 0) return false;
    if (operation == CacheMaintenanceOperation::invalidate_instruction)
        sys_icache_invalidate(reinterpret_cast<void*>(address), 4);
    else
        sys_dcache_flush(reinterpret_cast<void*>(address & ~std::uint64_t{63}), 64);
    return true;
}
} // namespace

LaunchResult<ThreadServiceBinding> NativeProcessRuntime::bind(
    std::shared_ptr<horizon::ThreadRecord> record, GuestThreadContext& context, std::stop_token stop) {
    auto allocated = memory_->allocate(platform::native_page_size(), horizon::MemoryType::thread_local_storage, rw);
    if (const auto* failed = std::get_if<LaunchFailure>(&allocated)) return *failed;
    auto local = std::get<GuestMemoryPin>(std::move(allocated));
    const auto range = local->range();
    std::shared_ptr<ThreadBinding> binding;
    try { binding = std::make_shared<ThreadBinding>(*this, std::move(record), local, stop); }
    catch (...) {
        local.reset();
        static_cast<void>(memory_->release(range));
        throw;
    }
    // Firmware supplies a cleared thread-local region. The SDK owns its TLS
    // allocation and all private pointers it later installs in that region.
    const bool original = launch_.plan.entry_abi == LaunchEntryAbi::original_process;
    // The kernel publishes the thread's own handle at offset 0x110, both when it
    // starts the main thread and when CreateThread creates a worker; the SDK
    // reads it as the current thread's lock owner tag.
    {
        const std::uint32_t handle = binding->record->creation_handle();
        std::memcpy(reinterpret_cast<void*>(range.address + 0x110), &handle, sizeof handle);
    }
    // The guest reads the cache geometry of its own processor (64-byte data
    // and instruction lines) and maintains caches on addresses it maps.
    context.set_cache_type(guest_cache_type);
    context.set_cache_maintenance({apply_cache_maintenance, memory_->process().get()});
    if (launch_.plan.entry_abi == LaunchEntryAbi::host_startup) {
        // Without the guest loader, the host provides the thread-local block
        // the loader would publish; nnosInitialize installs it as TPIDR_EL0.
        auto block = memory_->allocate(platform::native_page_size(), horizon::MemoryType::thread_local_storage, rw);
        if (const auto* failed = std::get_if<LaunchFailure>(&block)) return *failed;
        binding->storage = std::get<GuestMemoryPin>(std::move(block));
        const std::uint64_t address = binding->storage->range().address;
        std::memcpy(reinterpret_cast<void*>(range.address + 0x1f0), &address, sizeof(address));
        context.set_thread_pointer(0);
        context.set_read_only_thread_pointer(range.address);
        context.set_guest_x18(0);
        return ThreadServiceBinding{.supervisor = {ThreadBinding::dispatch, binding.get()},
            .owner = binding, .scheduling_boundary = {ThreadBinding::boundary, binding.get()}};
    }
    context.set_thread_pointer(original ? 0 : range.address);
    context.set_read_only_thread_pointer(range.address);
    std::uint64_t initial_x18 = 0;
    if (original) {
        arc4random_buf(&initial_x18, sizeof(initial_x18));
        initial_x18 |= 1;
    }
    context.set_guest_x18(initial_x18);
    return ThreadServiceBinding{.supervisor = {ThreadBinding::dispatch, binding.get()},
        .owner = binding, .scheduling_boundary = {ThreadBinding::boundary, binding.get()}};
}

GateStatus NativeProcessRuntime::create_worker(GateCall& call) {
    // CreateThread snapshots the caller's floating-point state; StartThread
    // must not observe later changes made by the creator.
    const auto inherited_fpcr = call.fpcr();
    const auto inherited_fpsr = call.fpsr();
    const auto core = horizon::resolve_thread_creation_core(static_cast<std::int32_t>(call.x(5)),
        profile_.main_thread.ideal_core, profile_.process.allowed_core_mask);
    const auto priority = static_cast<std::uint32_t>(call.x(4));
    const auto reply = [&](std::uint32_t result, std::uint64_t handle) {
        for (unsigned i = 0; i < 18; ++i) call.set_x(i, 0);
        call.set_x(0, result); call.set_x(1, handle);
        return GateStatus::resume;
    };
    // Firmware validation order: core, then priority. Native backing admission
    // below is a host capability check, never a fabricated Horizon Result.
    if (core.result) return reply(core.result, call.x(1));
    if (priority > 63 || priority < profile_.permissions.highest_priority ||
        priority > profile_.permissions.lowest_priority) return reply(0xe001, call.x(1));
    const auto sp = call.x(3) & ~std::uint64_t{15};
    const auto block = memory_->process()->query_memory(sp ? sp - 1 : 0);
    if (!sp || sp <= block.range.address || sp - block.range.address > block.range.size)
        return failure_at_gate(fail(LaunchStage::threads, LaunchError::unsupported, "create-thread-stack",
                                    "Worker SP lacks admitted native stack backing."), call);
    auto pinned = memory_->pin({block.range.address, sp - block.range.address}, rw);
    if (const auto* failed = std::get_if<LaunchFailure>(&pinned)) return failure_at_gate(*failed, call);
    auto result = threads_->create({
        .entry = {.function = call.x(1), .x0 = call.x(2), .x1 = 0, .stack_pointer = sp,
                  .fpcr = inherited_fpcr, .fpsr = inherited_fpsr},
        .stack = std::get<GuestMemoryPin>(std::move(pinned)), .priority = priority,
        .ideal_core = core.ideal_core, .affinity_mask = core.affinity_mask});
    if (const auto* failed = std::get_if<LaunchFailure>(&result)) {
        if (failed->guest_result) return reply(*failed->guest_result, call.x(1));
        return failure_at_gate(*failed, call);
    }
    return reply(0, std::get<CreatedThread>(result).handle);
}

GateStatus NativeProcessRuntime::start_worker(GateCall& call) {
    const std::lock_guard graphics_guard(graphics_start_mutex_);
    const auto handle = static_cast<horizon::Handle>(call.x(0));
    auto record = handles_->lookup_as<horizon::ThreadRecord>(handle);
    // Reserve tracking before publishing a runnable worker.
    {
        const std::lock_guard lock(mutex_);
        std::erase_if(started_, [](const auto& weak) {
            const auto record = weak.lock();
            return !record || record->exited();
        });
        started_.reserve(started_.size() + 1);
    }
    auto status = threads_->start(handle);
    if (const auto* failed = std::get_if<LaunchFailure>(&status)) {
        if (!failed->guest_result) return failure_at_gate(*failed, call);
        for (unsigned i = 0; i < 18; ++i) call.set_x(i, 0);
        call.set_x(0, *failed->guest_result);
        return GateStatus::resume;
    }
    {
        const std::lock_guard lock(mutex_);
        started_.push_back(record);
    }
    for (unsigned i = 0; i < 18; ++i) call.set_x(i, 0);
    return GateStatus::resume;
}

LaunchResult<std::shared_ptr<void>> NativeProcessRuntime::acquire_graphics_access() {
    auto guard = std::make_shared<std::unique_lock<std::mutex>>(graphics_start_mutex_);
    const std::lock_guard lock(mutex_);
    std::size_t live = 0;
    for (const auto& weak : started_) {
        if (const auto record = weak.lock(); record && !record->exited()) ++live;
    }
    if (live != 1 || stopping_)
        return fail(LaunchStage::graphics, LaunchError::unsupported, "graphics-execution-access",
                    "Native rendering requires the invoking guest to be the only live worker.");
    return std::shared_ptr<void>(std::move(guard));
}

LaunchStatus NativeProcessRuntime::initialize(std::shared_ptr<romfs::Storage> romfs,
                                             std::shared_ptr<const services::AddOnContentCatalog> addon_content,
                                             const std::filesystem::path& shader_directory,
                                             std::shared_ptr<PendingShaderPackage> shader_package,
                                             std::optional<std::string> display_version) {
    static std::atomic<std::uint64_t> next_process{1};
    process_id_ = next_process.fetch_add(1, std::memory_order_relaxed);
    if (!process_id_ || process_id_ == std::numeric_limits<std::uint64_t>::max())
        return fail(LaunchStage::services, LaunchError::resource_limit, "process-identity",
                    "Runtime process identity space is exhausted.");
    ServiceRegistryDependencies service_dependencies;
    service_dependencies.profile = profile_;
    service_dependencies.process_id = process_id_;
    service_dependencies.operation_mode = operation_mode_;
    service_dependencies.handles = handles_;
    service_dependencies.memory = memory_;
    service_dependencies.clock = dependencies_.clock;
    service_dependencies.presentation_clock = dependencies_.presentation_clock;
    service_dependencies.presenter = dependencies_.presenter;
    service_dependencies.romfs = std::move(romfs);
    service_dependencies.addon_content = std::move(addon_content);
    service_dependencies.display_version = std::move(display_version);
    service_dependencies.shader_package_directory = shader_directory;
    service_dependencies.pending_shader_package = std::move(shader_package);
    service_dependencies.nvn_driver = launch_.nvn_driver;
    if (launch_.nvn_driver) install_nvn_handler();
    service_dependencies.nvn_pso_manifest = launch_.nvn_pso_manifest;
    service_dependencies.startup_user = launch_.startup_user;
    service_dependencies.observe_frame = [this](const platform::PresentedFrame& frame) { return observe_frame(frame); };
    // Presentation reads back exactly the frames observe_frame captures.
    service_dependencies.wants_readback = [this](std::uint64_t frame_number) {
        return captures_frame(output_, frame_number);
    };
    service_dependencies.acquire_graphics_access = [this] { return acquire_graphics_access(); };
    if (dependencies_.create_services && profile_.origin != ProfileOrigin::synthetic_test)
        return fail(LaunchStage::services, LaunchError::invalid_input, "create-services",
                    "A service factory override requires a synthetic profile.");
    auto create_services = std::move(dependencies_.create_services);
    auto registry = create_services ? create_services(std::move(service_dependencies))
                                    : create_service_registry(std::move(service_dependencies));
    if (const auto* failed = std::get_if<LaunchFailure>(&registry)) return *failed;
    services_ = std::get<std::shared_ptr<ServiceRegistry>>(std::move(registry));
    auto exception = memory_->allocate(platform::native_page_size(), horizon::MemoryType::normal, rw);
    if (const auto* failed = std::get_if<LaunchFailure>(&exception)) return *failed;
    exception_storage_ = std::get<GuestMemoryPin>(std::move(exception));
    memory_->process()->set_user_exception_context(exception_storage_->range().address);
    auto stack = memory_->allocate(profile_.main_thread.stack_size, horizon::MemoryType::stack, rw);
    if (const auto* failed = std::get_if<LaunchFailure>(&stack)) return *failed;
    main_stack_ = std::get<GuestMemoryPin>(std::move(stack));
    auto created = create_thread_runtime({.profile = profile_, .launch = launch_, .memory = memory_,
        .handles = handles_, .clock = dependencies_.clock,
        .bind_services = [this](auto record, auto& context, auto stop) {
            return bind(std::move(record), context, stop);
        }});
    if (const auto* failed = std::get_if<LaunchFailure>(&created)) return *failed;
    threads_ = std::get<std::shared_ptr<ThreadRuntime>>(std::move(created));
    const bool host_startup = launch_.plan.entry_abi == LaunchEntryAbi::host_startup;
    if (host_startup) {
        // argv for nn::os::SetHostArgv: one program name, then a terminator.
        auto arguments = memory_->allocate(platform::native_page_size(), horizon::MemoryType::normal, rw);
        if (const auto* failed = std::get_if<LaunchFailure>(&arguments)) return *failed;
        host_arguments_ = std::get<GuestMemoryPin>(std::move(arguments));
        const std::uint64_t base = host_arguments_->range().address;
        const std::uint64_t name = base + 16;
        std::memcpy(reinterpret_cast<void*>(base), &name, sizeof(name));
        std::memcpy(reinterpret_cast<void*>(name), "armigo", 7);
    }
    const auto range = main_stack_->range();
    const bool original = launch_.plan.entry_abi == LaunchEntryAbi::original_process;
    auto main = threads_->create({
        .entry = {.function = launch_.entry_address, .x0 = 0, .x1 = 0,
                  .stack_pointer = range.address + range.size, .fpcr = 0, .fpsr = 0},
        .stack = main_stack_, .priority = profile_.main_thread.priority,
        .ideal_core = profile_.main_thread.ideal_core,
        .affinity_mask = std::uint64_t{1} << profile_.main_thread.ideal_core,
        .original_process = original, .host_startup = host_startup,
        .host_arguments = host_startup ? host_arguments_->range().address : 0});
    if (const auto* failed = std::get_if<LaunchFailure>(&main)) return *failed;
    main_ = std::get<CreatedThread>(std::move(main));
    return std::monostate{};
}

LaunchStatus NativeProcessRuntime::observe_frame(const platform::PresentedFrame& frame) {
    if (!captures_frame(output_, frame.frame_number)) return std::monostate{};
    if (!frame.readback)
        return fail(LaunchStage::presentation, LaunchError::unsupported, "capture-frame",
                    "Capture requires a presenter-owned completed readback.");
    // Captures are external output. The PPM stores RGB from actual completed
    // pixels and does not manufacture a frame when readback is unavailable.
    const auto& pixels = *frame.readback;
    static const auto maximum_width = [] {
        std::pair<std::uint32_t, bool> result{0, true};
        if (const char* value = std::getenv("ARMIGO_CAPTURE_WIDTH")) {
            const char* end = value + std::strlen(value);
            const auto parsed = std::from_chars(value, end, result.first);
            result.second = parsed.ec == std::errc{} && parsed.ptr == end && result.first != 0;
        }
        return result;
    }();
    if (!maximum_width.second)
        return fail(LaunchStage::inputs, LaunchError::invalid_input, "capture-frame",
                    "ARMIGO_CAPTURE_WIDTH requires a positive width.");
    std::string bytes;
    try { bytes = encode_capture_ppm(pixels, maximum_width.first); }
    catch (const std::exception& error) {
        return fail(LaunchStage::presentation, LaunchError::host_failure, "capture-frame", error.what());
    }
    const auto path = *output_.capture_directory / ("frame-" + std::to_string(frame.frame_number) + ".ppm");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    file.close();
    if (!file) return fail(LaunchStage::presentation, LaunchError::host_failure, "capture-frame",
                           "Completed frame capture could not be written.");
    return std::monostate{};
}

RunResult NativeProcessRuntime::run(const RunLimits& limits, const ReportDestination& output,
                                   std::stop_token stop) {
    try {
        return run_impl(limits, output, stop);
    } catch (const std::exception& error) {
        record_failure(fail(LaunchStage::threads, LaunchError::host_failure, "run", error.what()));
    } catch (...) {
        record_failure(fail(LaunchStage::threads, LaunchError::host_failure, "run",
                            "Runtime orchestration raised an exception."));
    }
    const auto teardown = shutdown(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    const std::lock_guard lock(mutex_);
    RunResult result{.outcome = *outcome_, .statistics = final_statistics_, .quiescent = quiescent_};
    result.teardown_failure = teardown_failure_;
    if (!result.teardown_failure)
        if (const auto* failure = std::get_if<LaunchFailure>(&teardown)) result.teardown_failure = *failure;
    return result;
}

RunResult NativeProcessRuntime::run_impl(const RunLimits& limits, const ReportDestination& output,
                                        std::stop_token stop) {
    std::shared_ptr<ThreadRuntime> threads;
    std::shared_ptr<ServiceRegistry> services;
    CreatedThread main;
    {
        const std::lock_guard lock(mutex_);
        if (ran_ || quiescent_)
            return {.outcome = fail(LaunchStage::threads, LaunchError::invalid_input, "run",
                                    "A process runtime may run exactly once before shutdown."),
                    .statistics = final_statistics_, .quiescent = quiescent_};
        ran_ = true;
        completed_frame_target_ = limits.completed_frame_target;
        output_ = output;
        threads = threads_; services = services_;
        main = main_;
    }
    const auto clock = dependencies_.clock;
    auto first_sample = clock->sample();
    ClockInstant began{};
    std::optional<ClockInstant> deadline;
    if (const auto* failed = std::get_if<LaunchFailure>(&first_sample)) record_failure(*failed);
    else {
        began = std::get<ClockSample>(first_sample).instant;
        if ((limits.timeout && (limits.timeout->count() <= 0 || began > ClockInstant::max() - *limits.timeout)) ||
            (limits.completed_frame_target && !*limits.completed_frame_target))
            record_failure(fail(LaunchStage::inputs, LaunchError::invalid_input, "run-limits",
                                "Run limits must be positive and their deadline representable."));
        else if (limits.timeout) deadline = began + *limits.timeout;
    }
    if (output.capture_directory) {
        std::error_code error;
        std::filesystem::create_directories(*output.capture_directory, error);
        if (error) record_failure(fail(LaunchStage::presentation, LaunchError::host_failure, "capture-directory",
                                      "Capture directory could not be created."));
    }
    std::stop_callback cancellation(stop, [this] { request_stop(StopReason::user_request); });
    bool start = false;
    {
        const std::lock_guard lock(mutex_);
        start = !outcome_ && !stopping_;
    }
    if (start) {
        const std::lock_guard graphics_guard(graphics_start_mutex_);
        {
            const std::lock_guard lock(mutex_);
            started_.push_back(main.record);
        }
        const auto status = threads->start(main.handle);
        if (const auto* failed = std::get_if<LaunchFailure>(&status)) {
            if (failed->code != LaunchError::cancelled) record_failure(*failed);
        }
    }
    for (;;) {
        if (const auto failed = threads->failure()) record_failure(*failed);
        if (const auto failed = services->failure()) record_failure(*failed);
        const auto statistics = services->statistics();
        auto sampled = clock->sample();
        if (const auto* failed = std::get_if<LaunchFailure>(&sampled)) record_failure(*failed);
        const bool exited = start && all_started_exited();
        std::unique_lock lock(mutex_);
        if (!outcome_) {
            if (limits.completed_frame_target && statistics.completed_frames >= *limits.completed_frame_target)
                outcome_ = RunSuccess{*limits.completed_frame_target};
            else if (exited) outcome_ = GuestExit{};
            else if (deadline && std::get<ClockSample>(sampled).instant >= *deadline)
                outcome_ = RunTimeout{*limits.timeout};
        }
        if (outcome_) break;
        // Observe lifecycle notifications not exposed by the frozen ThreadRuntime
        // API. Actual guest waits remain single cancellation-aware waits in D.
        changed_.wait_for(lock, std::chrono::milliseconds(2));
    }
    stop_execution();
    // A bounded grace period is host teardown policy, not a guest timeout.
    const auto teardown = shutdown(std::chrono::steady_clock::now() + std::chrono::seconds(1));
    RunResult result;
    {
        const std::lock_guard lock(mutex_);
        result.outcome = *outcome_;
        result.statistics = final_statistics_;
        result.quiescent = quiescent_;
        result.teardown_failure = teardown_failure_;
    }
    if (!result.teardown_failure)
        if (const auto* failed = std::get_if<LaunchFailure>(&teardown)) result.teardown_failure = *failed;
    const auto observed = services->statistics();
    result.statistics.queued_frames = observed.queued_frames;
    result.statistics.completed_frames = observed.completed_frames;
    result.statistics.completed_draws = observed.completed_draws;
    const auto ended = clock->sample();
    if (const auto* sample = std::get_if<ClockSample>(&ended); sample && sample->instant >= began)
        result.statistics.elapsed = sample->instant - began;
    else if (const auto* failed = std::get_if<LaunchFailure>(&ended);
             failed && !std::holds_alternative<LaunchFailure>(result.outcome)) result.outcome = *failed;
    return result;
}

LaunchStatus NativeProcessRuntime::shutdown(std::optional<ClockInstant> deadline) {
    // Release a construction self-owner only after dropping every shutdown lock.
    std::shared_ptr<ProcessRuntime> released_owner;
    const auto remember_teardown = [this](LaunchFailure failure) -> LaunchStatus {
        failure.stage = LaunchStage::shutdown;
        record_teardown_failure(failure);
        return failure;
    };
    std::unique_lock serial(shutdown_mutex_, std::defer_lock);
    if (deadline ? !serial.try_lock_until(*deadline) : (serial.lock(), false))
        return remember_teardown(fail(LaunchStage::shutdown, LaunchError::deadline_expired, "shutdown",
                    "Another shutdown still owns the runtime."));
    {
        const std::lock_guard lock(mutex_);
        if (quiescent_) return std::monostate{};
        if (!outcome_) outcome_ = RunCancelled{};
    }
    stop_execution();
    // Stop admission first, then drain consumers before attempting guest joins.
    // A failed drain keeps all owners for retry, just like a failed join.
    try {
        if (services_) {
            const auto drained = services_->drain(deadline);
            if (const auto* failed = std::get_if<LaunchFailure>(&drained)) return remember_teardown(*failed);
        }
        if (threads_) {
            // Threads get half of the time left to leave on their own. A thread
            // still parked after that, typically under an observed mutex grant
            // whose waiter will never be released, is ended by the forced stop,
            // as the firmware's process termination ends it.
            const auto now = std::chrono::steady_clock::now();
            const auto grace = deadline ? now + (*deadline - now) / 2 : now + std::chrono::milliseconds(500);
            auto joined = threads_->join(grace);
            if (const auto* failed = std::get_if<LaunchFailure>(&joined);
                failed && failed->code == LaunchError::deadline_expired) {
                force_stop();
                // A thread that runs guest code without a supervisor call cannot
                // observe the stop: end it as process termination does, retrying
                // while a thread is in host code and will reach its boundary.
                while (threads_->preempt_guest_threads() != 0 &&
                       (!deadline || std::chrono::steady_clock::now() < *deadline))
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                joined = threads_->join(deadline);
            }
            if (const auto* failed = std::get_if<LaunchFailure>(&joined)) {
                if (failed->code == LaunchError::deadline_expired) describe_live_threads();
                return remember_teardown(*failed);
            }
            if (const auto failed = threads_->failure()) record_failure(*failed);
        }
        // Native submissions can still arrive while guest threads leave host
        // calls. Close and join those workers only after all producers joined,
        // and before mappings or process-exit registries can be destroyed.
        if (services_) {
            const auto finished = services_->finish_guest_shutdown(deadline);
            if (const auto* failed = std::get_if<LaunchFailure>(&finished)) return remember_teardown(*failed);
        }
    } catch (const std::exception& error) {
        return remember_teardown(fail(LaunchStage::shutdown, LaunchError::host_failure,
                                      "shutdown", error.what()));
    } catch (...) {
        return remember_teardown(fail(LaunchStage::shutdown, LaunchError::host_failure,
                                      "shutdown", "Runtime teardown raised an exception."));
    }
    // All workers and consumers are now quiescent. Drop callback owners before
    // mappings. No table or registry callback owns this runtime.
    const std::lock_guard lock(mutex_);
    if (services_) {
        const auto statistics = services_->statistics();
        final_statistics_.queued_frames = statistics.queued_frames;
        final_statistics_.completed_frames = statistics.completed_frames;
        final_statistics_.completed_draws = statistics.completed_draws;
    }
    threads_.reset();
    main_ = {};
    services_.reset();
    main_stack_.reset();
    exception_storage_.reset();
    host_arguments_.reset();
    memory_.reset();
    handles_.reset();
    release_nvn_handler();
    launch_ = {};
    keys_.reset(); arbitration_.reset();
    dependencies_.presenter.reset();
    dependencies_.presentation_clock.reset();
    quiescent_ = true;
    released_owner = std::move(retained_construction_owner_);
    return std::monostate{};
}
} // namespace

bool captures_frame(const ReportDestination& output, std::uint64_t frame_number) {
    if (!output.capture_directory) return false;
    struct Selection {
        std::uint64_t first = 0, last = std::numeric_limits<std::uint64_t>::max(), every = 1;
        bool valid = true;
        Selection() {
            if (const char* value = std::getenv("ARMIGO_CAPTURE_EVERY"))
                every = std::max<std::uint64_t>(1, std::strtoull(value, nullptr, 10));
            const char* range = std::getenv("ARMIGO_CAPTURE_RANGE");
            if (!range || !*range) return;
            const char* end = range + std::strlen(range);
            const char* dash = std::find(range, end, '-');
            if (dash == end) valid = false;
            else {
                const auto begin_result = std::from_chars(range, dash, first);
                const auto end_result = std::from_chars(dash + 1, end, last);
                valid = begin_result.ec == std::errc{} && begin_result.ptr == dash &&
                    end_result.ec == std::errc{} && end_result.ptr == end && first <= last;
            }
            if (!valid)
                armigo::diag::warning("Armigo capture: ARMIGO_CAPTURE_RANGE must be FIRST-LAST; frame capture disabled.\n");
        }
    };
    // Both the writer and readback observe one immutable selection.
    static const Selection selection;
    return selection.valid && frame_number >= selection.first && frame_number <= selection.last &&
        frame_number % selection.every == 0;
}

#ifdef ARMIGO_RUNTIME_TESTING
ProcessRuntimeTestState ProcessRuntimeTestAccess::inspect(ProcessRuntime& process) {
    auto& runtime = dynamic_cast<NativeProcessRuntime&>(process);
    const std::lock_guard lock(runtime.mutex_);
    return {runtime.handles_, runtime.memory_, runtime.services_, runtime.threads_,
            runtime.main_.handle, runtime.process_id_};
}

void ProcessRuntimeTestAccess::set_failed_ipc_boundary(
    ProcessRuntime& process, std::function<void(std::stop_token)> boundary) {
    auto& runtime = dynamic_cast<NativeProcessRuntime&>(process);
    const std::lock_guard lock(runtime.mutex_);
    runtime.failed_ipc_boundary_ = std::move(boundary);
}

void ProcessRuntimeTestAccess::set_ipc_admission_boundary(
    ProcessRuntime& process, std::function<void(std::stop_token)> boundary) {
    auto& runtime = dynamic_cast<NativeProcessRuntime&>(process);
    const std::lock_guard lock(runtime.mutex_);
    runtime.ipc_admission_boundary_ = std::move(boundary);
}

std::size_t ProcessRuntimeTestAccess::mutex_waiters(ProcessRuntime& process, std::uint64_t address) {
    auto& runtime = dynamic_cast<NativeProcessRuntime&>(process);
    std::shared_ptr<horizon::ProcessWideKey> keys;
    {
        const std::lock_guard lock(runtime.mutex_);
        keys = runtime.keys_;
    }
    return keys ? keys->mutex_waiters(address) : 0;
}

void ProcessRuntimeTestAccess::install_nvn_handler(ProcessRuntime& process) {
    dynamic_cast<NativeProcessRuntime&>(process).install_nvn_handler();
}

bool ProcessRuntimeTestAccess::run_success_selected(ProcessRuntime& process) {
    auto& runtime = dynamic_cast<NativeProcessRuntime&>(process);
    const std::lock_guard lock(runtime.mutex_);
    return runtime.outcome_ && std::holds_alternative<RunSuccess>(*runtime.outcome_);
}
#endif

LaunchResult<std::unique_ptr<ProcessRuntime>> ProcessRuntime::create(
    const LaunchDescription& description, RuntimeDependencies dependencies) {
    try {
        if (!description.prepared_directory.empty() && !description.embedded_directory.empty())
            return fail(LaunchStage::inputs, LaunchError::invalid_input, "preparation-mode",
                        "Prepared and embedded artifact directories are mutually exclusive.");
        if (description.operation_mode != services::OperationMode::Console &&
            description.operation_mode != services::OperationMode::Handheld)
            return fail(LaunchStage::inputs, LaunchError::invalid_input, "operation-mode",
                        "Operation mode must be console or handheld.");
        if (!dependencies.clock || !dependencies.presentation_clock || !dependencies.presenter ||
            dependencies.presentation_clock->clock() != dependencies.clock ||
            dependencies.presenter->mode() != description.presentation.mode)
            return fail(LaunchStage::inputs, LaunchError::invalid_input, "runtime-dependencies",
                        "A presenter matching the selected mode and one shared clock are required.");
        if (!dependencies.memory_options.resource_limit_bytes && !dependencies.use_platform_memory_budget)
            return fail(LaunchStage::memory, LaunchError::invalid_input, "runtime-budget",
                        "A positive process memory budget or platform-default selection is required.");
        auto input_limits = description.input_limits;
        input_limits.collect_all_subsdk = description.synthetic_profile.has_value();
        std::shared_ptr<loader::NupArchive> archive;
        std::shared_ptr<loader::NupSelection> nup_selection;
        loader::ExeFsDirectoryInput input;
        if (!description.nup_file.empty()) {
            if (!description.exefs_directory.empty() || !description.romfs_directory.empty())
                return fail(LaunchStage::inputs, LaunchError::invalid_input, "nup-input",
                            "NUP and extracted directory inputs are mutually exclusive.");
            auto opened = loader::NupArchive::open(description.nup_file);
            if (auto error = std::get_if<loader::NupFailure>(&opened))
                return fail(LaunchStage::inputs, LaunchError::invalid_input, error->operation, error->detail);
            archive = std::get<std::shared_ptr<loader::NupArchive>>(std::move(opened));
            auto selected = archive->select(description.container_title_id.value_or(0));
            if (auto error = std::get_if<loader::NupFailure>(&selected))
                return fail(LaunchStage::inputs, LaunchError::invalid_input, error->operation, error->detail);
            nup_selection = std::make_shared<loader::NupSelection>(std::get<loader::NupSelection>(std::move(selected)));
            auto collected = archive->read_exefs(*nup_selection, input_limits);
            if (auto error = std::get_if<loader::NupFailure>(&collected))
                return fail(LaunchStage::inputs, LaunchError::invalid_input, error->operation, error->detail);
            input = std::get<loader::ExeFsDirectoryInput>(std::move(collected));
        } else {
            auto collected = loader::read_exefs_directory(description.exefs_directory, input_limits);
            if (const auto* failed = std::get_if<loader::ExeFsDirectoryFailure>(&collected))
                return fail(LaunchStage::inputs, LaunchError::invalid_input, "collect-exefs", failed->detail);
            input = std::get<loader::ExeFsDirectoryInput>(std::move(collected));
        }
        if (description.npdm_file) {
            auto detached = read_metadata(*description.npdm_file, description.input_limits.max_file_bytes);
            if (const auto* failed = std::get_if<LaunchFailure>(&detached)) return *failed;
            auto bytes = std::get<std::vector<std::byte>>(std::move(detached));
            if (input.process_metadata && *input.process_metadata != bytes)
                return fail(LaunchStage::profile, LaunchError::invalid_input, "npdm-match",
                            "Detached NPDM differs from collected main.npdm.");
            input.process_metadata = std::move(bytes);
        }
        LaunchProfile profile;
        if (description.synthetic_profile) {
            if (input.process_metadata)
                return fail(LaunchStage::profile, LaunchError::invalid_input, "synthetic-profile",
                            "Synthetic profiles and NPDM input are mutually exclusive.");
            profile = *description.synthetic_profile;
            const auto valid = validate_synthetic(profile);
            if (const auto* failed = std::get_if<LaunchFailure>(&valid)) return *failed;
        } else {
            if (!input.process_metadata)
                return fail(LaunchStage::profile, LaunchError::invalid_input, "derive-profile",
                            "Ordinary launch requires NPDM metadata.");
            auto derived = derive_launch_profile(*input.process_metadata, description.npdm_trust);
            if (const auto* failed = std::get_if<LaunchFailure>(&derived)) return *failed;
            profile = std::get<LaunchProfile>(std::move(derived));
        }
        if (dependencies.use_platform_memory_budget)
            dependencies.memory_options.resource_limit_bytes =
                profile.process.platform == GuestPlatform::switch2
                    ? horizon::ProcessRecord::switch2_application_capacity : std::uint64_t{4} << 30;
        if (profile.origin != ProfileOrigin::synthetic_test) {
            // Native reservations must fit alongside the host's own mappings.
            // Physical quota and compact VA windows are independent. Keep the
            // existing Switch 2 window limit until a larger request is admitted;
            // the NPDM main-stack allocation itself remains unchanged.
            const auto page = platform::native_page_size();
            if (!page)
                return fail(LaunchStage::memory, LaunchError::host_failure,
                            "runtime-reservations", "The native page size is unavailable.");
            const auto window_limit = profile.process.platform == GuestPlatform::switch2
                ? std::min(dependencies.memory_options.resource_limit_bytes, std::uint64_t{4} << 30)
                : dependencies.memory_options.resource_limit_bytes;
            const auto extent = std::max(page, window_limit & ~(page - 1));
            // A Switch 2 title sizes its heap from the application quota, so its
            // heap window keeps the profile's heap region up to that quota.
            const auto heap_limit = profile.process.platform == GuestPlatform::switch2
                ? dependencies.memory_options.resource_limit_bytes : window_limit;
            profile.process.heap_size = std::min(profile.process.heap_size,
                                                 std::max(page, heap_limit & ~(page - 1)));
            // A Switch 2 process needs one contiguous host arena (1 GiB mappable,
            // alias, stack and heap) plus its 8 GiB remote window below the
            // 39-bit limit. The host leaves about 40 GiB free there, split by two
            // large system reservations into at most three gaps. The title's
            // resource heaps grow by mapping pages into the alias window, so a
            // 2 GiB window starves them. The window is address space, not
            // memory: the arena asks for the console's window, independent of
            // the memory budget, and falls back toward 2 GiB to what fits.
            const auto alias_limit = profile.process.platform == GuestPlatform::switch2
                ? switch2_alias_window : extent;
            profile.process.base_alias_size = std::min(profile.process.base_alias_size, alias_limit);
            profile.process.stack_size = std::min(profile.process.stack_size, extent);
        }
        if (nup_selection && nup_selection->title_id != profile.title_id)
            return fail(LaunchStage::inputs, LaunchError::invalid_input, "nup-title",
                        "The selected NUP title differs from the admitted process metadata.");
        arc4random_buf(profile.process.entropy.data(), sizeof(profile.process.entropy));
        auto preparation_options = dependencies.preparation_options;
        if (preparation_options.mode == PreparationMode::automatic) {
            preparation_options.mode = profile.origin == ProfileOrigin::synthetic_test
                ? PreparationMode::synthetic_test : PreparationMode::original_process;
        }
        if (description.container_title_id && *description.container_title_id != profile.title_id)
            return fail(LaunchStage::inputs, LaunchError::invalid_input, "compatibility-identity",
                        "Container title identity differs from admitted process metadata.");
        compat::PatchDescriptions patches;
        {
            auto found = compat::find_title_settings(profile.title_id);
            if (const auto* error = std::get_if<std::string>(&found))
                return fail(LaunchStage::inputs, LaunchError::invalid_input, "title-settings", *error);
            if (description.presentation.mode == platform::PresentationMode::windowed) {
                const auto cache = compat::title_cache_directory(profile.title_id);
                if (!cache.empty())
                    diag::open_last_run_log((cache / "last-run.log").string().c_str());
            }
            if (auto& settings = std::get<std::optional<compat::TitleSettings>>(found)) {
                armigo::diag::write(armigo::diag::Level::info,  "Armigo: title settings %s\n", settings->folder.c_str());
                patches = std::move(settings->patches);
            }
        }
        if (description.title_saves) {
            const auto saves = compat::title_save_directory(profile.title_id);
            const auto legacy = saves.parent_path().parent_path() / compat::title_id_text(profile.title_id);
            if (const auto error = compat::prepare_save_directory(saves, legacy))
                return fail(LaunchStage::inputs, LaunchError::host_failure, "title-saves", *error);
            services::fs::set_host_save_directory(saves);
        }
        // A guest fault ends its thread with a report instead of the process.
        // The write tracker handles its own faults first and chains to this
        // handler, so this one goes in before graphics installs the tracker.
        if (!execution::guest_fault_handler_installed() && memory::WriteTracker::installed())
            armigo::diag::write(armigo::diag::Level::warning,
                "Armigo runtime: the write tracker was installed first; guest faults end the process unreported\n");
        else
            execution::install_guest_fault_handler();
        // The shader package and the RomFS index do not depend on the guest
        // image, so they open while it loads.
        auto shader_package = start_shader_package_open(description.shader_package_directory);
        auto romfs_opening = std::async(std::launch::async, [&description, archive, nup_selection] {
            std::string error;
            std::shared_ptr<romfs::Storage> storage;
            if (archive) storage = romfs::NupStorage::open(archive, *nup_selection, error);
            else storage = romfs::DirectoryStorage::open(description.romfs_directory, error);
            return std::pair{std::move(storage), std::move(error)};
        });
        auto addon_opening = std::async(std::launch::async, [&description, title = profile.title_id] {
            return open_addon_content(title, description.addon_directories);
        });
        // NPDM-backed input without offline artifacts is prepared at load time.
        LaunchResult<PreparedLaunch> loaded = LaunchFailure{};
        if (!description.embedded_directory.empty()) {
            auto opened = open_embedded_modules(description.embedded_directory);
            if (const auto* failed = std::get_if<LaunchFailure>(&opened)) return *failed;
            const auto& files = std::get<EmbeddedNativeFiles>(opened);
            loaded = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(), patches,
                {true, nullptr, &files.artifacts(), files.backings()});
        } else if (description.prepared_directory.empty() && profile.origin == ProfileOrigin::npdm) {
            loaded = load_native_launch(input, profile, std::make_shared<platform::ThreadSlotKey>(), patches);
        } else {
            auto artifacts = read_prepared_modules(
                description.prepared_directory, input, preparation_options);
            if (const auto* failed = std::get_if<LaunchFailure>(&artifacts)) return *failed;
            auto planned = plan_launch(input, profile, std::get<PreparedModuleArtifacts>(artifacts));
            if (const auto* failed = std::get_if<LaunchFailure>(&planned)) return *failed;
            loaded = load_prepared_launch(input, std::get<PreparedModuleArtifacts>(artifacts), profile,
                std::get<LaunchPlan>(planned), std::make_shared<platform::ThreadSlotKey>(), {}, patches);
        }
        if (const auto* failed = std::get_if<LaunchFailure>(&loaded)) return *failed;
        auto launch = std::get<PreparedLaunch>(std::move(loaded));
        if (dependencies.presentation_clock->observes_presents()) {
            std::vector<PresentationModule> observed_modules;
            for (std::size_t i = 0; i < launch.modules->module_count(); ++i)
                observed_modules.push_back({launch.modules->module_name(i), launch.modules->mapping(i).base_address()});
            dependencies.presentation_clock->bind_observation_modules(observed_modules);
        }
        const auto admitted_frame_rate = launch.title_frame_rate;
        const auto refresh_period = dependencies.presentation_clock->configuration().refresh_period;
        auto handles = std::make_shared<horizon::HandleTable>(profile.permissions.handle_table_capacity);
        auto backed = create_guest_memory(profile, dependencies.memory_options, launch.modules, handles);
        if (const auto* failed = std::get_if<LaunchFailure>(&backed)) return *failed;
        auto [romfs, storage_error] = romfs_opening.get();
        if (!romfs) return fail(LaunchStage::inputs, LaunchError::invalid_input, "open-romfs", storage_error);
        auto addon_content = addon_opening.get();
        if (auto* failed = std::get_if<LaunchFailure>(&addon_content)) return std::move(*failed);
        // Allocate the retry ownership control block before staged construction.
        // Aliasing it after an initialization failure cannot itself allocate.
        struct ConstructionOwner { std::unique_ptr<NativeProcessRuntime> runtime; };
        auto owner = std::make_shared<ConstructionOwner>();
        owner->runtime = std::make_unique<NativeProcessRuntime>(std::move(profile), std::move(dependencies),
            std::move(launch), std::move(handles),
            std::get<std::shared_ptr<GuestMemory>>(std::move(backed)),
            description.operation_mode);
        auto& runtime = owner->runtime;
        LaunchStatus initialized;
        try {
            // A NUP carries no control data. The selected package's display
            // version metadata stands in for the control-data field; without it
            // the package version reads as major.minor.micro from bits 16, 8, 0.
            std::optional<std::string> display_version;
            if (nup_selection && !nup_selection->display_version.empty()) {
                display_version = nup_selection->display_version;
            } else if (nup_selection) {
                const auto version = nup_selection->version;
                display_version = std::to_string(version >> 16) + "." + std::to_string((version >> 8) & 0xff) + "." +
                                  std::to_string(version & 0xff);
            }
            initialized = runtime->initialize(
                std::move(romfs),
                std::get<std::shared_ptr<const services::AddOnContentCatalog>>(std::move(addon_content)),
                description.shader_package_directory, std::move(shader_package), std::move(display_version));
        } catch (const std::bad_alloc&) {
            initialized = fail(LaunchStage::memory, LaunchError::resource_limit, "create-runtime",
                               "Runtime allocation failed before guest execution.");
        } catch (const std::exception& error) {
            initialized = fail(LaunchStage::inputs, LaunchError::host_failure, "create-runtime", error.what());
        } catch (...) {
            initialized = fail(LaunchStage::inputs, LaunchError::host_failure, "create-runtime",
                               "Runtime initialization raised an exception.");
        }
        if (auto* failed = std::get_if<LaunchFailure>(&initialized)) {
            // Preserve the construction error, even if bounded rollback fails.
            std::shared_ptr<ProcessRuntime> retry(owner, runtime.get());
            runtime->retain_failed_construction(retry);
            failed->cleanup_owner = retry;
            try {
                if (std::holds_alternative<std::monostate>(runtime->shutdown(
                        std::chrono::steady_clock::now() + std::chrono::seconds(1))))
                    failed->cleanup_owner.reset();
            } catch (...) {
                // The preallocated owner remains reachable for a later retry.
            }
            return std::move(*failed);
        }
        // Both runners pass their actual presentation/VI clock here. Allow one
        // nanosecond for period rounding; this diagnostic changes no clock or code.
        if (admitted_frame_rate && refresh_period.count() > 1'000'000'000 / admitted_frame_rate + 1) {
            armigo::diag::write(armigo::diag::Level::warning,
                "Armigo: admitted title frame rate %u fps exceeds the configured presentation/VI clock (%.6f Hz). "
                "Check --refresh-hz for this title mode; continuing with the configured clock and patches.\n",
                admitted_frame_rate, 1e9 / static_cast<double>(refresh_period.count()));
        }
        return std::unique_ptr<ProcessRuntime>(std::move(runtime));
    } catch (const std::bad_alloc&) {
        return fail(LaunchStage::memory, LaunchError::resource_limit, "create-runtime",
                    "Runtime allocation failed before guest execution.");
    } catch (const std::exception& error) {
        return fail(LaunchStage::inputs, LaunchError::host_failure, "create-runtime", error.what());
    }
}
} // namespace armigo::runtime
