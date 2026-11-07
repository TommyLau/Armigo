// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <armigo/runtime/thread_runtime.h>
#include <armigo/execution/guest_fault.h>
#include <armigo/execution/guest_preempt.h>
#include <armigo/aarch64/instruction.h>
#include "../horizon/thread_lifecycle.h"
#include "../nvn/backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#ifdef __APPLE__
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <pthread/qos.h>
#include <typeinfo>
#endif

namespace armigo::runtime {
namespace {
using horizon::Handle;
using horizon::ThreadRecord;
using execution::GateStatus;
constexpr auto stack_access = memory::Permission::read | memory::Permission::write;

LaunchFailure fail(LaunchError code, const char* operation, const char* detail) {
    LaunchFailure value;
    value.stage = LaunchStage::threads;
    value.code = code;
    value.operation = operation;
    value.detail = detail;
    return value;
}

bool valid_range(horizon::AddressRange range) noexcept {
    return range.address && range.size &&
        range.size <= std::numeric_limits<std::uint64_t>::max() - range.address;
}
bool overlaps(horizon::AddressRange a, horizon::AddressRange b) noexcept {
    return a.address < b.address + b.size && b.address < a.address + a.size;
}
bool valid_stack(const GuestMemoryPin& pin) noexcept {
    if (!pin || !valid_range(pin->range()) ||
        !memory::has_permission(pin->permissions(), stack_access)) return false;
    const auto view = pin->writable_bytes();
    const auto read = pin->bytes();
    return view.size() == pin->range().size && read.size() == view.size() &&
        read.data() == view.data() &&
        reinterpret_cast<std::uintptr_t>(view.data()) == pin->range().address;
}
bool owned_entry(const ModuleSet& modules, std::uint64_t pc) noexcept {
    if (!pc || (pc & 3)) return false;
    for (std::size_t i = 0; i < modules.module_count(); ++i) {
        const auto& mapping = modules.mapping(i);
        if (mapping.sealed() && pc >= mapping.base_address()) {
            const auto offset = pc - mapping.base_address();
            if (offset < mapping.image_size() && mapping.executable_address(offset) == pc)
                return true;
        }
    }
    return false;
}

bool valid_process_landing(const ModuleSet& modules, std::uint64_t landing,
                           std::uint64_t entry) noexcept {
    if (!landing || (landing & 3) != 0 ||
        landing > std::numeric_limits<std::uint64_t>::max() - 4) return false;
    const auto branch = aarch64::encode_b(landing + 4, entry);
    if (!branch.ok()) return false;
    // MOV X16, XZR is the ORR alias 0xaa1f03f0. Requiring the exact veneer
    // keeps appendix transfer code bounded and makes the real entry's x16 zero.
    constexpr std::uint32_t clear_x16 = 0xaa1f03f0;
    for (std::size_t index = 0; index < modules.module_count(); ++index) {
        const auto appendix = modules.appended_code(index);
        if (!appendix || appendix->size < 8) continue;
        const auto& mapping = modules.mapping(index);
        const auto base = mapping.base_address();
        if (landing < base) continue;
        const auto offset = landing - base;
        if (offset < appendix->offset || offset - appendix->offset > appendix->size - 8 ||
            mapping.executable_address(offset) != landing ||
            mapping.executable_address(offset + 4) != landing + 4)
            continue;
        const auto bytes = mapping.readable_bytes(offset, 8);
        if (!bytes) return false;
        std::uint32_t words[2]{};
        std::memcpy(words, bytes->data(), sizeof(words));
        return words[0] == clear_x16 && words[1] == branch.word;
    }
    return false;
}

const PreparedEntryLanding* find_process_landing(
    const PreparedLaunch& launch, std::uint64_t root) noexcept {
    const auto found = std::lower_bound(launch.original_entry_landings.begin(),
        launch.original_entry_landings.end(), root,
        [](const auto& candidate, std::uint64_t value) {
            return candidate.logical_root < value;
        });
    return found != launch.original_entry_landings.end() &&
            found->logical_root == root ? &*found : nullptr;
}

class NativeThreadRuntime final : public ThreadRuntime {
    struct Worker {
        Handle handle = 0;
        std::shared_ptr<ThreadRecord> record;
        execution::GuestThreadEntryRequest entry;
        bool host_startup = false;
        std::uint64_t host_arguments = 0;
        GuestMemoryPin stack;
        execution::GuestThreadContext context;
        ThreadState state = ThreadState::created;
        std::optional<LaunchFailure> failure;
        std::thread host;
        execution::GuestPreemptSlot preempt;
    };
    struct Invocation {
        NativeThreadRuntime* runtime;
        Worker* worker;
        execution::GuestSupervisor supervisor;
        execution::GuestSupervisor scheduling_boundary;
    };
    static execution::GateStatus compatibility_call(const execution::CallHookSite& site,
        execution::GateCall& call, execution::GuestThreadContext&, void* raw) {
        auto& runtime = *static_cast<NativeThreadRuntime*>(raw);
        const auto& patch = runtime.dependencies_.launch.compatibility_patch;
        if (!patch) {
            call.set_failure_message("Compatibility hook has no retained module state.");
            return execution::GateStatus::failed;
        }
        return compat::handle_patch(*patch, call, runtime.dependencies_.memory->ipc_buffers(),
                                   site.call_return_address, {[](void*) noexcept {
                                       auto* backend = nvn::backends().replay;
                                       return backend ? backend->gpu_frame_work_ns() : std::uint64_t{0};
                                   }, nullptr});
    }
    static thread_local Invocation* current_;
    static thread_local NativeThreadRuntime* active_runtime_;
    static thread_local NativeThreadRuntime* cleanup_runtime_;
    struct CleanupScope {
        NativeThreadRuntime* previous;
        explicit CleanupScope(NativeThreadRuntime* runtime) : previous(cleanup_runtime_) {
            cleanup_runtime_ = runtime;
        }
        ~CleanupScope() { cleanup_runtime_ = previous; }
    };

public:
    explicit NativeThreadRuntime(ThreadRuntimeDependencies dependencies)
        : dependencies_(std::move(dependencies)),
          reaper_([this](std::stop_token stop) { reap_loop(stop); }) {
        // Executable pages of the guest modules (original text and the appended
        // gate copies), the only code an asynchronous stop may interrupt.
        if (const auto& modules = dependencies_.launch.modules)
            for (std::size_t index = 0; index != modules->module_count(); ++index) {
                const auto& mapping = modules->mapping(index);
                for (const auto& region : mapping.host_regions())
                    if ((static_cast<std::uint8_t>(region.permissions) &
                         static_cast<std::uint8_t>(memory::Permission::execute)) != 0 && region.size != 0)
                        guest_code_.push_back({mapping.base_address() + region.offset,
                                               mapping.base_address() + region.offset + region.size});
            }
        // The same ranges tell the guest fault handler which faults to report.
        guest_code_registration_.emplace(guest_code_);
        if (!guest_code_registration_->registered())
            armigo::diag::write(armigo::diag::Level::warning,
                "Armigo runtime: too many guest code ranges; guest faults end the process unreported\n");
    }
    ~NativeThreadRuntime() override {
        // A callback must not own the runtime: it would either create a cycle or
        // destroy its own worker. The public dependency contract forbids this.
        if (active_runtime_ == this) std::terminate();
        request_stop();
        static_cast<void>(join({}));
    }

    LaunchResult<CreatedThread> create(const ThreadLaunch& launch) override {
        try {
            const auto guard = std::lock_guard(mutex_);
            if (!accepting_) return fail(LaunchError::cancelled, "create_thread", "Thread runtime is closing.");
            if (!valid_stack(launch.stack) || (launch.entry.stack_pointer & 15) ||
                launch.entry.stack_pointer <= launch.stack->range().address ||
                launch.entry.stack_pointer - launch.stack->range().address > launch.stack->range().size)
                return fail(LaunchError::invalid_input, "create_thread", "Thread stack pin or exact SP is invalid.");
            // A gate must fit below SP even before the guest allocates a frame.
            if (launch.entry.stack_pointer - launch.stack->range().address < sizeof(execution::GateFrame))
                return fail(LaunchError::invalid_input, "create_thread", "Thread stack has no gate headroom.");
            if (!launch.entry.require_gate_scope ||
                (launch.entry.fpcr && (*launch.entry.fpcr & ~execution::kGuestEntryFpcrMask)))
                return fail(LaunchError::invalid_input, "create_thread", "Thread entry control state is invalid.");
            if (!launch.entry.function || (launch.entry.function & 3))
                return fail(LaunchError::invalid_input, "create_thread", "Thread entry must be nonzero and instruction aligned.");
            const bool original_plan =
                dependencies_.launch.plan.entry_abi == LaunchEntryAbi::original_process;
            const PreparedEntryLanding* trusted_landing = nullptr;
            if (original_plan) {
                if (launch.original_process &&
                    launch.entry.function != dependencies_.launch.entry_address)
                    return fail(LaunchError::invalid_input, "create_thread", "Initial process entry must use the admitted process root.");
                if (!launch.original_process && !original_process_created_)
                    return fail(LaunchError::invalid_input, "create_thread", "The initial process thread must be created before its workers.");
                if (launch.original_process && original_process_created_)
                    return fail(LaunchError::invalid_input, "create_thread", "Original process entry may be created only once.");
                trusted_landing = find_process_landing(dependencies_.launch,
                                                       launch.entry.function);
                if (!trusted_landing ||
                    !valid_process_landing(*dependencies_.launch.modules,
                        trusted_landing->landing, trusted_landing->logical_root))
                    return fail(LaunchError::permission_denied, "create_thread", "The logical root has no exact trusted process landing.");
            } else {
                if (launch.original_process || launch.entry.original_process)
                    return fail(LaunchError::invalid_input, "create_thread", "Raw process entry mode requires an original launch plan.");
                if (launch.host_startup &&
                    (dependencies_.launch.plan.entry_abi != LaunchEntryAbi::host_startup || host_startup_created_))
                    return fail(LaunchError::invalid_input, "create_thread", "Host startup runs once, on a host-startup plan.");
                if (launch.entry.transfer_function != 0)
                    return fail(LaunchError::invalid_input, "create_thread", "Synthetic thread entry cannot use a process transfer landing.");
            }
            if (!dependencies_.launch.modules->permits_guest_entry(launch.entry.function))
                return fail(LaunchError::permission_denied, "create_thread", "Entry is not an admitted prepared root owned by this launch.");
            const auto initial = horizon::resolve_thread_creation_core(launch.ideal_core,
                dependencies_.profile.main_thread.ideal_core, dependencies_.profile.process.allowed_core_mask);
            if (initial.result || !launch.affinity_mask ||
                (launch.affinity_mask & ~dependencies_.profile.process.allowed_core_mask) ||
                !(launch.affinity_mask & initial.affinity_mask))
                return fail(LaunchError::permission_denied, "create_thread", "Thread core state is outside admitted capabilities.");
            if (!allowed_priority(launch.priority))
                return fail(LaunchError::permission_denied, "create_thread", "Thread priority is outside admitted bounds.");
            // ThreadLaunch is a host launch description with explicit affinity.
            // A CreateThread SVC adapter must use resolve_thread_creation_core's
            // singleton mask; host synthetic launches may supply a broader mask.
            for (const auto& worker : workers_) {
                if (worker->stack && overlaps(worker->stack->range(), launch.stack->range()))
                    return fail(LaunchError::invalid_input, "create_thread", "Thread stacks must be exclusive until exit.");
            }
            // Re-pin with this process's authority; caller-provided keepalive
            // objects do not establish backing ownership or native permissions.
            auto retained = dependencies_.memory->pin(launch.stack->range(), stack_access);
            if (auto* error = std::get_if<LaunchFailure>(&retained)) return *error;
            auto stack = std::get<GuestMemoryPin>(std::move(retained));
            if (!valid_stack(stack) || stack->range().address != launch.stack->range().address ||
                stack->range().size != launch.stack->range().size)
                return fail(LaunchError::host_failure, "create_thread", "Memory authority returned an inconsistent stack pin.");
            if (next_id_ == 0) return fail(LaunchError::resource_limit, "create_thread", "Thread identity space exhausted.");
            auto worker = std::make_unique<Worker>();
            worker->record = std::make_shared<ThreadRecord>(dependencies_.memory->process(), next_id_, launch.priority);
            if (!worker->record->configure_scheduling(initial.ideal_core, launch.affinity_mask,
                                                       static_cast<std::uint32_t>(initial.ideal_core)))
                return fail(LaunchError::invalid_input, "create_thread", "Process and launch core permissions disagree.");
            worker->entry = launch.entry;
            worker->host_startup = launch.host_startup;
            worker->host_arguments = launch.host_arguments;
            worker->entry.original_process = original_plan;
            worker->entry.transfer_function = trusted_landing ? trusted_landing->landing : 0;
            worker->stack = std::move(stack);
            workers_.reserve(workers_.size() + 1);
            const auto handle = dependencies_.handles->insert(worker->record);
            if (!handle) return fail(LaunchError::resource_limit, "create_thread", "Thread handle table is full.");
            worker->handle = *handle;
            worker->record->set_creation_handle(*handle);
            if (launch.original_process) {
                worker->entry.x0 = 0;
                worker->entry.x1 = *handle;
            } else if (original_plan) {
                worker->entry.x1 = 0;
            }
            CreatedThread result{*handle, worker->record};
            workers_.push_back(std::move(worker));
            if (launch.original_process) original_process_created_ = true;
            if (launch.host_startup) host_startup_created_ = true;
            ++next_id_;
            return result;
        } catch (...) {
            return fail(LaunchError::host_failure, "create_thread", "Host could not allocate thread state.");
        }
    }

    LaunchStatus start(Handle handle) override {
        const auto guard = std::lock_guard(mutex_);
        if (!accepting_) return fail(LaunchError::cancelled, "start_thread", "Thread runtime is closing.");
        auto* worker = find(handle);
        if (!worker) return invalid_handle("start_thread");
        if (worker->state != ThreadState::created) {
            auto error = fail(LaunchError::invalid_input, "start_thread", "Thread may be started exactly once.");
            error.guest_result = 0xfa01;
            return error;
        }
        try {
            // Keep publication locked until std::thread construction succeeds.
            worker->host = std::thread([this, worker] { run(*worker); });
            worker->state = ThreadState::running;
            return std::monostate{};
        } catch (...) {
            return fail(LaunchError::host_failure, "start_thread", "Host worker creation failed; thread remains suspended.");
        }
    }

    std::optional<LaunchFailure> failure() const override {
        const auto guard = std::lock_guard(mutex_);
        return failure_;
    }
    LaunchResult<ThreadSnapshot> snapshot(Handle handle) const override {
        const auto guard = std::lock_guard(mutex_);
        const auto* worker = find(handle);
        if (!worker) return invalid_handle("thread_snapshot");
        const auto scheduling = worker->record->scheduling_state();
        return ThreadSnapshot{worker->state, worker->record->priority(), scheduling.ideal_core,
                              scheduling.affinity_mask, worker->failure};
    }
    LaunchStatus set_priority(Handle handle, std::uint32_t priority) override {
        const auto guard = std::lock_guard(mutex_);
        auto* worker = find(handle);
        if (!worker) return invalid_handle("set_thread_priority");
        if (!allowed_priority(priority)) {
            auto error = fail(LaunchError::permission_denied, "set_thread_priority", "Priority is outside admitted bounds.");
            error.guest_result = 0xe001;
            return error;
        }
        worker->record->set_priority(priority);
        return std::monostate{};
    }
    LaunchStatus set_core_mask(Handle handle, std::int32_t ideal, std::uint64_t mask) override {
        const auto guard = std::lock_guard(mutex_);
        auto* worker = find(handle);
        if (!worker) return invalid_handle("set_thread_core_mask");
        const auto previous = worker->record->scheduling_state();
        const auto next = horizon::resolve_thread_core_mask(ideal, mask, previous.ideal_core,
            dependencies_.profile.main_thread.ideal_core, dependencies_.profile.process.allowed_core_mask);
        if (next.result) {
            auto error = fail(LaunchError::permission_denied, "set_thread_core_mask", "Core request is not admitted.");
            error.guest_result = next.result;
            return error;
        }
        const auto selected = horizon::select_virtual_processor(next.ideal_core, next.affinity_mask,
                                                                 previous.current_processor);
        if (!selected || !worker->record->configure_scheduling(next.ideal_core, next.affinity_mask,
                worker->state == ThreadState::running || worker->state == ThreadState::created ? selected : std::nullopt))
            return fail(LaunchError::host_failure, "set_thread_core_mask", "Virtual processor assignment failed.");
        return std::monostate{};
    }
    LaunchStatus exit_current() override {
        if (!current_ || current_->runtime != this)
            return fail(LaunchError::invalid_input, "exit_thread", "Exit must originate from this runtime's supervisor boundary.");
        const auto guard = std::lock_guard(mutex_);
        current_->worker->state = ThreadState::exiting;
        return std::monostate{};
    }

    std::size_t preempt_guest_threads() noexcept override {
        if (active_runtime_ == this || cleanup_runtime_ == this || !stop_.stop_requested()) return 0;
        std::size_t live = 0;
        // Workers are only destroyed after join; the manager lock keeps the list
        // stable while the slots are inspected.
        const auto guard = std::lock_guard(mutex_);
        for (const auto& worker : workers_) {
            if (worker->state == ThreadState::exited || worker->state == ThreadState::created) continue;
            ++live;
            static_cast<void>(execution::preempt_guest_thread(worker->preempt, guest_code_));
        }
        return live;
    }

    void request_stop() noexcept override {
        {
            const auto guard = std::lock_guard(mutex_);
            accepting_ = false;
        }
        // Callbacks may acquire queue locks or call the runtime. Never request
        // stop while holding the manager lock.
        CleanupScope cleanup(this);
        stop_.request_stop();
        retire_suspended();
        changed_.notify_all();
    }

    LaunchStatus join(std::optional<ClockInstant> deadline) override {
        if (active_runtime_ == this || cleanup_runtime_ == this)
            return fail(LaunchError::invalid_input, "join_threads", "A worker or cleanup callback cannot join its own runtime.");
        std::unique_lock reaper_lock(reaper_mutex_, std::defer_lock);
        if (deadline) {
            if (!reaper_lock.try_lock_until(*deadline)) return join_timeout();
        } else reaper_lock.lock();
        // The reaper joins only workers already marked exited, so stopping it
        // never waits for live guest execution.
        stop_reaper_locked();
        std::unique_lock join_lock(join_mutex_, std::defer_lock);
        if (deadline) {
            if (!join_lock.try_lock_until(*deadline)) {
                if (!restart_reaper_locked()) return reaper_restart_failure();
                return join_timeout();
            }
        } else join_lock.lock();
        {
            const auto guard = std::lock_guard(mutex_);
            accepting_ = false;
        }
        CleanupScope cleanup(this);
        retire_suspended();
        std::unique_lock lock(mutex_);
        const auto finished = [&] {
            return std::all_of(workers_.begin(), workers_.end(), [](const auto& w) {
                return w->state == ThreadState::exited;
            });
        };
        if (deadline) {
            if (!changed_.wait_until(lock, *deadline, finished)) {
                lock.unlock();
                join_lock.unlock();
                if (!restart_reaper_locked()) return reaper_restart_failure();
                return join_timeout();
            }
        } else changed_.wait(lock, finished);
        lock.unlock();
        for (const auto& worker : workers_) if (worker->host.joinable()) worker->host.join();
        // Retire the binder's captures and all execution dependencies after
        // joining, outside locks. Keep records available for snapshots/handles.
        ThreadRuntimeDependencies retired;
        lock.lock();
        if (!joined_) {
            retired = std::move(dependencies_);
            // Profile is small value state needed by setters on exited records.
            dependencies_.profile = std::move(retired.profile);
            dependencies_.handles = retired.handles;
            joined_ = true;
        }
        lock.unlock();
        // Keep other joiners behind capture destruction. CleanupScope rejects
        // same-runtime recursive joins without holding the manager mutex.
        retired = {};
        join_lock.unlock();
        return std::monostate{};
    }

private:
    // Each started guest runs on its own host thread. Virtual processor state
    // serves queries and guest per-core lanes; it is not an execution permit.
    bool allowed_priority(std::uint32_t priority) const noexcept {
        return priority <= 63 && priority >= dependencies_.profile.permissions.highest_priority &&
               priority <= dependencies_.profile.permissions.lowest_priority;
    }
    static LaunchFailure invalid_handle(const char* operation) {
        auto error = fail(LaunchError::invalid_input, operation, "Handle does not name a thread owned by this runtime.");
        error.guest_result = 0xe401;
        return error;
    }
    static LaunchFailure join_timeout() {
        auto error = fail(LaunchError::deadline_expired, "join_threads", "Workers are still live; retain this runtime and retry shutdown.");
        error.stage = LaunchStage::shutdown;
        return error;
    }
    static LaunchFailure reaper_restart_failure() {
        auto error = fail(LaunchError::host_failure, "join_threads",
            "Timed shutdown could not restore autonomous worker retirement.");
        error.stage = LaunchStage::shutdown;
        return error;
    }
    Worker* find(Handle handle) const {
        if (handle == horizon::current_thread_handle)
            return current_ && current_->runtime == this ? current_->worker : nullptr;
        const auto record = dependencies_.handles->lookup_as<ThreadRecord>(handle);
        if (!record) return nullptr;
        for (const auto& worker : workers_) if (worker->record == record) return worker.get();
        return nullptr;
    }
    void record_failure(Worker& worker, LaunchFailure error) {
        error.thread_id = worker.record->id();
        const auto guard = std::lock_guard(mutex_);
        if (!worker.failure) worker.failure = error;
        if (!failure_) failure_ = std::move(error);
    }
    // "module+0x..." for an address inside a loaded module, else the address.
    std::string describe_code_address(std::uint64_t address) const {
        char text[96];
        if (const auto& modules = dependencies_.launch.modules)
            for (std::size_t index = 0; index != modules->module_count(); ++index) {
                const auto& mapping = modules->mapping(index);
                const std::uint64_t base = mapping.base_address();
                if (address < base || address - base >= mapping.mapping_size()) continue;
                const auto name = modules->module_name(index);
                std::snprintf(text, sizeof text, "%.*s+0x%llx", static_cast<int>(name.size()), name.data(),
                              static_cast<unsigned long long>(address - base));
                return text;
            }
        std::snprintf(text, sizeof text, "0x%llx", static_cast<unsigned long long>(address));
        return text;
    }
    // A guest fault ended the worker's invocation (guest_fault.h): report the
    // fault and stop the process, as any other thread failure does.
    void report_guest_fault(Worker& worker) {
        const auto fault = execution::take_guest_fault(worker.preempt);
        if (!fault) return;
        std::string detail;
        char text[160];
        const auto append = [&](const char* format, auto... values) {
            std::snprintf(text, sizeof text, format, values...);
            detail += text;
        };
        append("Guest %s at %s (pc 0x%llx), address 0x%llx, syndrome 0x%x, thread %llu.",
               execution::describe_guest_fault_signal(fault->signal), describe_code_address(fault->pc).c_str(),
               static_cast<unsigned long long>(fault->pc), static_cast<unsigned long long>(fault->address),
               fault->syndrome, static_cast<unsigned long long>(worker.record->id()));
        const auto note = worker.record->call_note();
        if (note.svc == 0) {
            detail += " No supervisor call yet.";
        } else {
            append(" Last supervisor call 0x%02x (x0 0x%llx, x1 0x%llx, x2 0x%llx)", note.svc,
                   static_cast<unsigned long long>(note.arguments[0]),
                   static_cast<unsigned long long>(note.arguments[1]),
                   static_cast<unsigned long long>(note.arguments[2]));
            // SendSyncRequest takes the session in x0, the user-buffer forms in x2.
            const auto session = note.svc == 0x21 ? note.arguments[0]
                               : note.svc == 0x22 || note.svc == 0x23 ? note.arguments[2] : 0;
            const auto object = session && dependencies_.handles
                ? dependencies_.handles->lookup(static_cast<Handle>(session)) : nullptr;
            if (object) {
                const auto& value = *object;
                const char* mangled = typeid(value).name();
                int status = 0;
                char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
                append(" on %s", status == 0 && demangled ? demangled : mangled);
                std::free(demangled);
            }
            detail += ".";
        }
        detail += " Registers:";
        for (std::size_t index = 0; index != 29; ++index)
            append(" x%zu=0x%llx", index, static_cast<unsigned long long>(fault->x[index]));
        append(" fp=0x%llx lr=%s sp=0x%llx pstate=0x%llx fpcr=0x%llx fpsr=0x%llx",
               static_cast<unsigned long long>(fault->x[29]), describe_code_address(fault->x[30]).c_str(),
               static_cast<unsigned long long>(fault->sp), static_cast<unsigned long long>(fault->pstate),
               static_cast<unsigned long long>(fault->fpcr), static_cast<unsigned long long>(fault->fpsr));
        armigo::diag::write(armigo::diag::Level::error, "Armigo runtime: %s\n", detail.c_str());
        auto error = fail(LaunchError::guest_fault, "guest_fault", detail.c_str());
        error.guest_pc = fault->pc;
        record_failure(worker, std::move(error));
    }
    void retire_suspended() noexcept {
        // Creation is closed before this call. The background reaper may still
        // remove unrelated exited entries between iterations.
        for (;;) {
            GuestMemoryPin retired;
            Worker* worker = nullptr;
            {
                const auto guard = std::lock_guard(mutex_);
                const auto found = std::find_if(workers_.begin(), workers_.end(), [](const auto& candidate) {
                    return candidate->state == ThreadState::created;
                });
                if (found == workers_.end()) break;
                worker = found->get();
                worker->state = ThreadState::exiting;
                retired = std::move(worker->stack);
            }
            retired.reset();
            finish(*worker);
        }
    }
    void finish(Worker& worker) noexcept {
        {
            const auto guard = std::lock_guard(mutex_);
            worker.record->unbind_current_processor();
            worker.record->mark_exited();
            worker.state = ThreadState::exited;
        }
        changed_.notify_all();
    }
    void reap_closed_workers() {
        CleanupScope cleanup(this);
        auto reap_guard = std::unique_lock(join_mutex_);
        for (;;) {
            std::unique_ptr<Worker> retired;
            {
                const auto guard = std::lock_guard(mutex_);
                if (!dependencies_.handles) break;
                const auto found = std::find_if(
                    workers_.begin(), workers_.end(), [&](const auto& worker) {
                        if (worker->state != ThreadState::exited) return false;
                        const auto published =
                            dependencies_.handles->lookup_as<ThreadRecord>(worker->handle);
                        return published != worker->record && worker->record.use_count() == 1;
                    });
                if (found == workers_.end()) break;
                retired = std::move(*found);
                workers_.erase(found);
            }
            if (retired->host.joinable()) retired->host.join();
            reap_guard.unlock();
            retired.reset();
            reap_guard.lock();
        }
    }
    void reap_loop(std::stop_token stop) noexcept {
        while (!stop.stop_requested()) {
            {
                auto lock = std::unique_lock(mutex_);
                changed_.wait_for(lock, std::chrono::milliseconds(10));
            }
            if (!stop.stop_requested()) reap_closed_workers();
        }
    }
    void stop_reaper_locked() noexcept {
        reaper_.request_stop();
        changed_.notify_all();
        if (reaper_.joinable()) reaper_.join();
    }
    bool restart_reaper_locked() noexcept {
        try {
            reaper_ = std::jthread([this](std::stop_token stop) { reap_loop(stop); });
            return true;
        } catch (...) {
            return false;
        }
    }
    bool scheduling_call(std::uint16_t svc, execution::GateCall& call) {
        if (svc < 0x0c || svc > 0x10) return false;
        const auto guard = std::lock_guard(mutex_);
        if (svc == 0x10) {
            const auto processor = current_->worker->record->current_processor();
            if (!processor) return false;
            for (unsigned i = 0; i < 18; ++i) call.set_x(i, 0);
            call.set_x(0, *processor);
            return true;
        }
        const auto handle_register = svc == 0x0c ? 1u : svc == 0x0e ? 2u : 0u;
        auto* worker = find(static_cast<Handle>(call.x(handle_register)));
        const auto incoming_x1 = call.x(1);
        const auto incoming_x2 = call.x(2);
        std::uint32_t result = worker ? 0 : 0xe401;
        std::uint64_t output_x1 = svc == 0x0c ? incoming_x1 : 0;
        std::uint64_t output_x2 = 0;
        if (svc == 0x0e) {
            output_x1 = incoming_x1;
            output_x2 = incoming_x2;
        }
        if (worker) {
            if (svc == 0x0c) {
                output_x1 = (incoming_x1 & 0xffffffff00000000ull) | worker->record->priority();
            } else if (svc == 0x0d) {
                const auto priority = static_cast<std::uint32_t>(incoming_x1);
                if (!allowed_priority(priority)) result = 0xe001;
                else worker->record->set_priority(priority);
            } else {
                const auto old = worker->record->scheduling_state();
                if (svc == 0x0e) {
                    output_x1 = (incoming_x1 & 0xffffffff00000000ull) |
                                static_cast<std::uint32_t>(old.ideal_core);
                    output_x2 = old.affinity_mask;
                } else {
                    const auto next = horizon::resolve_thread_core_mask(
                        static_cast<std::int32_t>(incoming_x1), incoming_x2, old.ideal_core,
                        dependencies_.profile.main_thread.ideal_core,
                        dependencies_.profile.process.allowed_core_mask);
                    result = next.result;
                    if (!result) {
                        const auto processor = horizon::select_virtual_processor(
                            next.ideal_core, next.affinity_mask, old.current_processor);
                        if (!processor || !worker->record->configure_scheduling(
                                next.ideal_core, next.affinity_mask,
                                worker->state == ThreadState::running || worker->state == ThreadState::created ? processor : std::nullopt))
                            return false;
                    }
                }
            }
        }
        for (unsigned i = 0; i < 18; ++i) call.set_x(i, 0);
        call.set_x(0, result);
        call.set_x(1, output_x1);
        call.set_x(2, output_x2);
        return true;
    }
    static GateStatus supervise(std::uint16_t svc, execution::GateCall& call,
                                execution::GuestThreadContext& context, void* raw) {
        auto& invocation = *static_cast<Invocation*>(raw);
        auto& runtime = *invocation.runtime;
        struct Scope {
            Invocation* previous;
            explicit Scope(Invocation* value) : previous(current_) { current_ = value; }
            ~Scope() { current_ = previous; }
        } scope(&invocation);
        GateStatus result = GateStatus::failed;
        try {
            const auto& permissions = runtime.dependencies_.profile.permissions.supervisor_calls;
            if (std::find(permissions.begin(), permissions.end(), svc) == permissions.end()) {
                runtime.record_failure(*invocation.worker,
                    fail(LaunchError::permission_denied, "thread_supervisor", "Supervisor call is not admitted by the process profile."));
                if (call.can_exit_thread()) {
                    static_cast<void>(runtime.exit_current());
                    return GateStatus::exit_thread;
                }
                return GateStatus::failed;
            }
            if (svc >= 0x0c && svc <= 0x10) {
                if (invocation.scheduling_boundary.handler) {
                    result = invocation.scheduling_boundary.handler(
                        svc, call, context, invocation.scheduling_boundary.context);
                    if (result != GateStatus::resume) {
                        if (result == GateStatus::failed) {
                            auto error = fail(LaunchError::unsupported, "thread_supervisor",
                                call.failure_message() ? call.failure_message() :
                                    "The scheduling boundary refused the supervisor operation.");
                            error.command = svc;
                            error.guest_pc = call.return_address();
                            runtime.record_failure(*invocation.worker, std::move(error));
                        }
                        if (call.can_exit_thread()) {
                            static_cast<void>(runtime.exit_current());
                            return GateStatus::exit_thread;
                        }
                        return result;
                    }
                } else if (runtime.stop_.stop_requested()) {
                    if (call.can_exit_thread()) {
                        static_cast<void>(runtime.exit_current());
                        return GateStatus::exit_thread;
                    }
                    return GateStatus::failed;
                }
                if (runtime.scheduling_call(svc, call)) return GateStatus::resume;
            }
            if (svc == 0x0a) {
                if (call.can_exit_thread()) {
                    static_cast<void>(runtime.exit_current());
                    return GateStatus::exit_thread;
                }
            } else if (invocation.supervisor.handler) {
                // The service binder owns cancellation safety. A granted mutex
                // must reach its release path even if cancellation raced it.
                result = invocation.supervisor.handler(svc, call, context, invocation.supervisor.context);
            }
            if (result == GateStatus::failed) {
                auto error = fail(LaunchError::unsupported, "thread_supervisor",
                    call.failure_message() ? call.failure_message() : "Supervisor operation is unsupported.");
                error.command = svc;
                error.guest_pc = call.return_address();
                runtime.record_failure(*invocation.worker, std::move(error));
            }
        } catch (...) {
            runtime.record_failure(*invocation.worker,
                fail(LaunchError::host_failure, "thread_supervisor", "Supervisor callback threw an exception."));
            result = GateStatus::failed;
        }
        if (result != GateStatus::resume && call.can_exit_thread()) {
            static_cast<void>(runtime.exit_current());
            return GateStatus::exit_thread;
        }
        return result;
    }
    // The host loader owns initialization: call each SDK startup function on
    // this thread's stack. Each must return; nnMain's return ends the process.
    void run_startup(Worker& worker) noexcept {
        for (const auto& call : dependencies_.launch.startup_calls) {
            if (stop_.stop_requested()) return;
            auto request = worker.entry;
            request.function = call.function;
            request.x0 = call.argument == StartupCall::Argument::main_thread_handle ? worker.handle
                       : call.argument == StartupCall::Argument::host_arguments ? worker.host_arguments
                       : call.x0;
            request.x1 = 0;
            request.returns = true;
            const auto entered = execution::invoke_guest_thread(worker.stack->writable_bytes(), request);
            if (entered.exited()) {
                report_guest_fault(worker);
                return;
            }
            if (!entered.returned()) {
                auto error = fail(LaunchError::host_failure, "startup_call", execution::describe(entered.error));
                error.detail += " (" + call.phase + ")";
                record_failure(worker, std::move(error));
                return;
            }
        }
    }
    void run(Worker& worker) noexcept {
#ifdef __APPLE__
        // Host scheduling only (Horizon priorities keep their semantics): guest
        // threads run at the interactive class, like the graphics threads they
        // feed, so they are not placed behind them. ARMIGO_GUEST_QOS=0 keeps the
        // inherited class.
        static const bool interactive = [] {
            const char* value = std::getenv("ARMIGO_GUEST_QOS");
            return !(value && *value == '0');
        }();
        static std::atomic<bool> warned{false};
        if (interactive && pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) != 0 &&
            !warned.exchange(true))
            armigo::diag::write(armigo::diag::Level::error,  "Armigo runtime: guest thread QoS class change failed; keeping the inherited class\n");
#endif
        struct ActiveWorker {
            explicit ActiveWorker(NativeThreadRuntime* runtime) { active_runtime_ = runtime; }
            ~ActiveWorker() { active_runtime_ = nullptr; }
        } active(this);
        // Pair with start's publication before touching any worker state.
        { const auto guard = std::lock_guard(mutex_); }
        const execution::GuestPreemptBinding preempt_binding(worker.preempt);
        try {
            if (!stop_.stop_requested()) {
                auto result = dependencies_.bind_services(worker.record, worker.context, stop_.get_token());
                if (auto* error = std::get_if<LaunchFailure>(&result)) record_failure(worker, *error);
                else {
                    auto binding = std::get<ThreadServiceBinding>(std::move(result));
                    if (!stop_.stop_requested()) {
                        if (worker.context.set_counter(dependencies_.clock->counter()) != execution::GuestCounterError::none)
                            record_failure(worker, fail(LaunchError::host_failure, "bind_thread_clock", "Clock counter is invalid."));
                        else {
                            Invocation invocation{
                                this, &worker, binding.supervisor, binding.scheduling_boundary};
                            execution::GuestDispatchScope dispatch(dependencies_.launch.sites, worker.context,
                                                                  {supervise, &invocation}, execution::GateFailurePolicy::trap,
                                                                  dependencies_.launch.call_hooks, {compatibility_call, this});
                            if (dispatch.setup_error() != execution::GuestDispatchFailureKind::none)
                                record_failure(worker, fail(LaunchError::host_failure, "bind_thread_dispatch", "Guest dispatch binding failed."));
                            else {
                                platform::ThreadSlotBinding slot(*dependencies_.launch.thread_slot,
                                                                 worker.context.bound_register_cell());
                                if (!slot.native_access())
                                    record_failure(worker, fail(LaunchError::host_failure, "bind_thread_slot", "Native thread slot binding failed."));
                                else if (worker.host_startup) {
                                    run_startup(worker);
                                } else if (!stop_.stop_requested()) {
                                    const auto entered = execution::invoke_guest_thread(worker.stack->writable_bytes(), worker.entry);
                                    if (entered.exited())
                                        report_guest_fault(worker);
                                    else
                                        record_failure(worker, fail(LaunchError::host_failure, "enter_thread", execution::describe(entered.error)));
                                }
                            }
                        }
                    }
                    // Dispatch and slot have ended before per-thread adapters.
                    binding.owner.reset();
                }
            }
        } catch (...) {
            record_failure(worker, fail(LaunchError::host_failure, "run_thread", "Thread service binding or execution setup threw."));
        }
        worker.context.set_thread_pointer(0);
        worker.context.set_read_only_thread_pointer(0);
        worker.context.set_guest_x18(0);
        worker.context.clear_counter();
        worker.context.set_cache_maintenance({});
        GuestMemoryPin retired;
        {
            const auto guard = std::lock_guard(mutex_);
            worker.state = ThreadState::exiting;
            retired = std::move(worker.stack);
        }
        retired.reset();
        finish(worker);
    }

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::timed_mutex join_mutex_;
    ThreadRuntimeDependencies dependencies_;
    std::vector<execution::GuestCodeRange> guest_code_;
    std::optional<execution::GuestCodeRegistration> guest_code_registration_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::stop_source stop_;
    std::optional<LaunchFailure> failure_;
    std::uint64_t next_id_ = 1;
    bool accepting_ = true;
    bool joined_ = false;
    bool original_process_created_ = false;
    bool host_startup_created_ = false;
    std::timed_mutex reaper_mutex_;
    std::jthread reaper_;
};
thread_local NativeThreadRuntime::Invocation* NativeThreadRuntime::current_ = nullptr;
thread_local NativeThreadRuntime* NativeThreadRuntime::active_runtime_ = nullptr;
thread_local NativeThreadRuntime* NativeThreadRuntime::cleanup_runtime_ = nullptr;

} // namespace

LaunchResult<std::shared_ptr<ThreadRuntime>> create_thread_runtime(ThreadRuntimeDependencies dependencies) {
    if (!dependencies.memory || !dependencies.memory->process() || !dependencies.handles ||
        !dependencies.clock || !dependencies.bind_services || !dependencies.launch.modules ||
        !dependencies.launch.sites || !dependencies.launch.thread_slot || !dependencies.launch.thread_slot->valid())
        return fail(LaunchError::invalid_input, "create_thread_runtime", "Thread runtime dependencies are incomplete.");
    const auto& profile = dependencies.profile;
    if (!profile.is_64_bit || !profile.process.allowed_core_mask ||
        profile.permissions.highest_priority > profile.permissions.lowest_priority ||
        profile.permissions.lowest_priority > 63 ||
        dependencies.handles->capacity() != profile.permissions.handle_table_capacity ||
        dependencies.memory->process()->configuration().allowed_core_mask != profile.process.allowed_core_mask)
        return fail(LaunchError::invalid_input, "create_thread_runtime", "Thread process capabilities are inconsistent.");
    const auto abi = dependencies.launch.plan.entry_abi;
    if (abi == LaunchEntryAbi::original_process) {
        if (dependencies.launch.plan.relocation_owner != LaunchOwnership::guest_bootstrap ||
            dependencies.launch.plan.initialization_owner != LaunchOwnership::guest_bootstrap)
            return fail(LaunchError::unsupported, "create_thread_runtime", "Original process entry requires guest bootstrap relocation and initialization ownership.");
        if (dependencies.launch.original_entry_landings.empty() ||
            !std::is_sorted(dependencies.launch.original_entry_landings.begin(),
                dependencies.launch.original_entry_landings.end(),
                [](const auto& left, const auto& right) {
                    return left.logical_root < right.logical_root;
                }))
            return fail(LaunchError::invalid_input, "create_thread_runtime", "Original entry landing map is empty or unsorted.");
        for (std::size_t index = 0;
             index < dependencies.launch.original_entry_landings.size(); ++index) {
            const auto& entry = dependencies.launch.original_entry_landings[index];
            if ((index != 0 && dependencies.launch.original_entry_landings[index - 1].logical_root ==
                    entry.logical_root) ||
                !dependencies.launch.modules->permits_guest_entry(entry.logical_root) ||
                dependencies.launch.modules->permits_guest_entry(entry.landing) ||
                !valid_process_landing(*dependencies.launch.modules,
                                       entry.landing, entry.logical_root))
                return fail(LaunchError::permission_denied, "create_thread_runtime", "Original entry landing map is not exact trusted appendix code.");
        }
        const auto* main_landing = find_process_landing(
            dependencies.launch, dependencies.launch.entry_address);
        if (!main_landing ||
            main_landing->landing != dependencies.launch.process_entry_address)
            return fail(LaunchError::invalid_input, "create_thread_runtime", "Process entry compatibility landing does not match the trusted map.");
    } else if (abi == LaunchEntryAbi::host_startup) {
        if (dependencies.launch.plan.relocation_owner != LaunchOwnership::host_loader ||
            dependencies.launch.plan.initialization_owner != LaunchOwnership::host_loader ||
            dependencies.launch.startup_calls.empty() ||
            dependencies.launch.process_entry_address != 0 ||
            !dependencies.launch.original_entry_landings.empty())
            return fail(LaunchError::invalid_input, "create_thread_runtime", "Host startup requires host-loader ownership and a startup sequence.");
        for (const auto& call : dependencies.launch.startup_calls)
            if (!dependencies.launch.modules->permits_guest_entry(call.function))
                return fail(LaunchError::permission_denied, "create_thread_runtime", "Startup call is not owned executable code.");
    } else if (abi != LaunchEntryAbi::synthetic_exact_sp) {
        return fail(LaunchError::unsupported, "create_thread_runtime", "Thread runtime entry ABI is unsupported.");
    } else if (dependencies.launch.process_entry_address != 0 ||
               !dependencies.launch.original_entry_landings.empty()) {
        return fail(LaunchError::invalid_input, "create_thread_runtime", "Synthetic launch contains original-process landings.");
    }
    if (!dependencies.launch.modules->permits_guest_entry(dependencies.launch.entry_address))
        return fail(LaunchError::permission_denied, "create_thread_runtime", "Prepared entry is not owned executable code.");
    for (const auto& site : dependencies.launch.sites->sites()) {
        if (site.return_address < 4 || !owned_entry(*dependencies.launch.modules, site.return_address - 4))
            return fail(LaunchError::permission_denied, "create_thread_runtime", "Prepared gate site is outside owned executable code.");
        for (const auto target : site.allowed_indirect_targets)
            if (!owned_entry(*dependencies.launch.modules, target))
                return fail(LaunchError::permission_denied, "create_thread_runtime", "Prepared indirect target is outside owned executable code.");
    }
    try {
        return std::shared_ptr<ThreadRuntime>(std::make_shared<NativeThreadRuntime>(std::move(dependencies)));
    } catch (...) {
        return fail(LaunchError::host_failure, "create_thread_runtime", "Host could not allocate thread runtime.");
    }
}

} // namespace armigo::runtime
