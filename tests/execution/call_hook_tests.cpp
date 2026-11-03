// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/call_hook.h>
#include <armigo/execution/gate_code.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/platform/host_module.h>

#include <armigo/aarch64/instruction.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <mach-o/dyld.h>
#include <signal.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {
using namespace armigo;
using namespace armigo::execution;

struct Observation {
    std::uint64_t target_lr = 0;
    std::uint64_t target_sp = 0;
    std::uint64_t target_order = 0;
    std::uint64_t continuation_order = 0;
    std::uint64_t callback_sp = 0;
    std::uint64_t callback_gate_lr = 0;
    std::uint64_t callback_guest_lr = 0;
};
static_assert(offsetof(Observation, continuation_order) == 24);

extern "C" {
extern const std::byte armigo_test_call_hook_begin[], armigo_test_call_hook_end[];
extern const std::byte armigo_test_call_hook_site[], armigo_test_call_hook_target[];
extern const std::byte armigo_test_call_hook_stack_words[];
}

bool check(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}

std::size_t offset(const std::byte* label) {
    return reinterpret_cast<std::uintptr_t>(label) -
           reinterpret_cast<std::uintptr_t>(armigo_test_call_hook_begin);
}

bool validates_table() {
    std::array sites = {
        CallHookSite{0x3008, 0x1004, 0x2000},
        CallHookSite{0x2008, 0x1104, 0x2100},
    };
    auto built = CallHookTable::build(sites);
    if (!check(built.ok(), "A valid call-hook table was rejected.")) return false;
    sites[1].target_address = 0x9000;
    const auto* found = built.table->find(0x2008);
    if (!check(found && found->call_return_address == 0x1104 &&
                       found->target_address == 0x2100 &&
                       built.table->sites().front().return_address == 0x2008,
               "The call-hook table did not own and sort its declarations.")) return false;

    const auto rejects = [](CallHookSite site, CallHookTableErrorKind expected) {
        const auto result = CallHookTable::build(std::span(&site, 1));
        return check(!result.ok() && result.error.kind == expected,
                     "An invalid call-hook declaration had the wrong result.");
    };
    if (!rejects({0, 0x1004, 0x2000}, CallHookTableErrorKind::null_return_address) ||
        !rejects({0x1001, 0x1004, 0x2000}, CallHookTableErrorKind::misaligned_return_address) ||
        !rejects({0x1008, 0, 0x2000}, CallHookTableErrorKind::invalid_call_return_address) ||
        !rejects({0x1008, 0x1004, 0}, CallHookTableErrorKind::invalid_target_address) ||
        !rejects({0x1008, 0x1004, 0x2001}, CallHookTableErrorKind::invalid_target_address)) {
        return false;
    }
    const std::array duplicates = {
        CallHookSite{0x1008, 0x1004, 0x2000},
        CallHookSite{0x1008, 0x1104, 0x2100},
    };
    const auto duplicate = CallHookTable::build(duplicates);
    if (!check(!duplicate.ok() &&
                   duplicate.error.kind == CallHookTableErrorKind::duplicate_return_address &&
                   duplicate.error.site_index == 1,
               "A duplicate call-hook identity was accepted.")) return false;
    const auto limited = CallHookTable::build(sites, 1);
    return check(!limited.ok() && limited.error.kind == CallHookTableErrorKind::too_many_sites,
                 "The call-hook table bound was not enforced.");
}

bool validates_patch() {
    constexpr std::uint64_t address = 0x1000;
    constexpr std::uint64_t target = 0x1800;
    constexpr std::uint64_t trampoline = 0x2000;
    constexpr CallHookCodeRegion gate{0x3000, 0x200};
    const auto original = aarch64::encode_bl(address, target);
    if (!check(original.ok(), "The test BL could not be encoded.")) return false;
    const auto made = make_call_hook_patch(original.word, address, trampoline, gate);
    if (!check(made.ok(), "A valid call-hook patch was rejected.")) return false;
    const auto& patch = *made.patch;
    std::array<std::uint32_t, 2> stack_words{};
    std::memcpy(stack_words.data(), armigo_test_call_hook_stack_words, sizeof(stack_words));
    if (!check(aarch64::decode(patch.replacement_instruction).opcode == aarch64::Opcode::bl &&
                   aarch64::referenced_address(aarch64::decode(patch.replacement_instruction),
                                               address) == trampoline &&
                   aarch64::referenced_address(aarch64::decode(patch.trampoline[1]),
                                               trampoline + 4) == gate.address &&
                   aarch64::referenced_address(aarch64::decode(patch.trampoline[3]),
                                               trampoline + 12) == target &&
                   patch.trampoline[0] == stack_words[0] &&
                   patch.trampoline[2] == stack_words[1] &&
                   patch.site.return_address == trampoline + 8 &&
                   patch.site.call_return_address == address + 4 &&
                   patch.site.target_address == target,
               "The call-hook patch does not preserve its branch and stack contract.")) return false;

    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    const std::array invalid = {
        make_call_hook_patch(original.word, address + 1, trampoline, gate),
        make_call_hook_patch(original.word, address, trampoline + 2, gate),
        make_call_hook_patch(original.word, address, trampoline, {gate.address + 4, gate.size}),
        make_call_hook_patch(original.word, address, trampoline, {gate.address, 0}),
        make_call_hook_patch(original.word, maximum - 3, trampoline, gate),
        make_call_hook_patch(original.word, address, maximum - 3, gate),
        make_call_hook_patch(original.word, address, address, gate),
        make_call_hook_patch(original.word, address, address + 4, gate),
        make_call_hook_patch(original.word, address, trampoline, {address, 8}),
        make_call_hook_patch(original.word, address, trampoline, {trampoline, 16}),
        make_call_hook_patch(aarch64::encode_b(address, target).word, address, trampoline, gate),
        make_call_hook_patch(aarch64::encode_bl(address, address + 0x1000).word,
                             address, 0x10000000, gate),
        make_call_hook_patch(original.word, address, trampoline, {0x10000000, 8}),
    };
    for (const auto& result : invalid) {
        if (!check(!result.ok() && result.error != CallHookPatchError::none,
                   "An invalid call-hook patch was accepted.")) return false;
    }
    return true;
}

GateStatus unused_hook(const CallHookSite&, GateCall&, GuestThreadContext&, void*) {
    return GateStatus::resume;
}

bool validates_dispatch_setup() {
    GuestGateSite ordinary{0x1008, aarch64::encode_svc(1).word};
    const auto normal = GuestSiteTable::build(std::span(&ordinary, 1));
    CallHookSite hook_site{0x1008, 0x2004, 0x3000};
    const auto hooks = CallHookTable::build(std::span(&hook_site, 1));
    if (!check(normal.ok() && hooks.ok(), "Dispatch setup fixtures are invalid.")) return false;
    {
        GuestThreadContext thread;
        GuestDispatchScope conflict(normal.table, thread, {}, GateFailurePolicy::trap,
                                    hooks.table, {unused_hook, nullptr});
        if (!check(conflict.setup_error() == GuestDispatchFailureKind::call_hook_site_conflict &&
                       !thread.bound(),
                   "A normal/hook return-address conflict was accepted.")) return false;
    }
    const auto empty = GuestSiteTable::build({});
    {
        GuestThreadContext thread;
        GuestDispatchScope unsafe(empty.table, thread, {}, GateFailurePolicy::resume_unchanged,
                                  hooks.table, {unused_hook, nullptr});
        if (!check(unsafe.setup_error() == GuestDispatchFailureKind::call_hook_requires_trap &&
                       unsafe.gate_scope().policy() == GateFailurePolicy::trap && !thread.bound(),
                   "A call hook was allowed to resume after callback failure.")) return false;
    }
    {
        GuestThreadContext thread;
        GuestDispatchScope missing(empty.table, thread, {}, GateFailurePolicy::trap, hooks.table);
        if (!check(missing.setup_error() == GuestDispatchFailureKind::missing_call_hook_handler &&
                       !thread.bound(),
                   "A call-hook table without a handler was accepted.")) return false;
    }
    return true;
}

struct CallbackState {
    enum class Mode { resume, fail, throw_exception, change_link_register } mode = Mode::resume;
    Observation* observation = nullptr;
    std::thread::id thread;
    std::uint64_t expected_return = 0;
    std::uint64_t expected_target = 0;
    bool valid = true;
    unsigned calls = 0;
};

GateStatus observe_hook(const CallHookSite& site, GateCall& call,
                        GuestThreadContext& thread, void* raw) {
    auto& state = *static_cast<CallbackState*>(raw);
    ++state.calls;
    state.valid &= std::this_thread::get_id() == state.thread && call.x(0) == 10 &&
                   call.x(1) == reinterpret_cast<std::uintptr_t>(state.observation) &&
                   site.call_return_address == state.expected_return &&
                   site.target_address == state.expected_target &&
                   call.return_address() == site.return_address &&
                   call.x(kGateLinkRegister) == site.return_address;
    switch (state.mode) {
    case CallbackState::Mode::fail:
        call.set_failure_message("synthetic call hook failed");
        return GateStatus::failed;
    case CallbackState::Mode::throw_exception:
        throw std::runtime_error("synthetic call hook exception");
    case CallbackState::Mode::change_link_register:
        call.set_x(kGateLinkRegister, call.x(kGateLinkRegister) + 4);
        return GateStatus::resume;
    case CallbackState::Mode::resume:
        break;
    }
    state.observation->target_order = 1;
    state.observation->callback_sp = call.sp() + CallHookPatch::stack_adjustment;
    state.observation->callback_gate_lr = call.x(kGateLinkRegister);
    state.observation->callback_guest_lr = site.call_return_address;
    call.set_x(0, 40);
    call.set_guest_x18(0x22222222);
    thread.set_thread_pointer(0x33333333);
    return GateStatus::resume;
}

const GuestDispatchScope* trapped_scope = nullptr;
Observation* trapped_observation = nullptr;
GuestDispatchFailureKind expected_trap = GuestDispatchFailureKind::none;

void report_hook_trap(const GateFailure&) noexcept {
    const auto* failure = trapped_scope ? trapped_scope->first_failure() : nullptr;
    if (failure && failure->kind == expected_trap && trapped_observation &&
        trapped_observation->target_lr == 0 &&
        trapped_observation->target_order == 0 &&
        trapped_observation->continuation_order == 0) {
        std::fputs("Call hook trapped before original callee\n", stderr);
    }
    std::abort();
}

bool executes_native_hook(CallbackState::Mode mode = CallbackState::Mode::resume) {
    const auto source_size = offset(armigo_test_call_hook_end);
    const auto gate = make_local_gate_code();
    const auto gate_offset = (source_size + 7) & ~std::size_t{7};
    const auto stub_offset = gate_offset + gate.size();
    std::vector<std::byte> bytes(stub_offset + 16);
    std::memcpy(bytes.data(), armigo_test_call_hook_begin, source_size);
    std::memcpy(bytes.data() + gate_offset, gate.data(), gate.size());
    const auto site_offset = offset(armigo_test_call_hook_site);
    std::uint32_t original = 0;
    std::memcpy(&original, bytes.data() + site_offset, sizeof(original));
    const auto made = make_call_hook_patch(original, site_offset, stub_offset,
                                           {gate_offset, gate.size()});
    if (!check(made.ok(), "The native call-hook fixture could not be prepared.")) return false;
    const auto& patch = *made.patch;
    std::memcpy(bytes.data() + site_offset, &patch.replacement_instruction, 4);
    std::memcpy(bytes.data() + stub_offset, patch.trampoline.data(), 16);

    const std::array regions = {platform::ModuleRegion{
        0, bytes.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(bytes, regions);
    if (!check(mapped.ok(), "The native call-hook fixture could not be mapped.")) return false;
    const auto base = mapped.module->base_address();
    CallHookSite absolute = patch.site;
    absolute.return_address += base;
    absolute.call_return_address += base;
    absolute.target_address += base;
    const auto hooks = CallHookTable::build(std::span(&absolute, 1));
    const auto normal = GuestSiteTable::build({});
    if (!check(hooks.ok() && normal.ok() && mapped.module->seal().ok(),
               "The native call-hook fixture could not be registered and sealed.")) return false;

    Observation observation;
    CallbackState callback{mode, &observation, std::this_thread::get_id(),
                           base + site_offset + 4,
                           base + offset(armigo_test_call_hook_target)};
    GuestThreadContext thread;
    thread.set_guest_x18(0x11111111);
    GuestDispatchScope scope(normal.table, thread, {}, GateFailurePolicy::trap,
                             hooks.table, {observe_hook, &callback});
    if (!check(scope.setup_error() == GuestDispatchFailureKind::none,
               "The native call-hook dispatch scope did not bind.")) return false;
    const auto entry = mapped.module->executable_address(0);
    if (!entry) return false;
    if (mode != CallbackState::Mode::resume) {
        trapped_scope = &scope;
        trapped_observation = &observation;
        expected_trap = mode == CallbackState::Mode::fail
                            ? GuestDispatchFailureKind::call_hook_failed
                            : mode == CallbackState::Mode::throw_exception
                                  ? GuestDispatchFailureKind::call_hook_threw
                                  : GuestDispatchFailureKind::invalid_call_hook_frame;
        set_gate_trap_hook(report_hook_trap);
    }
    const auto result = reinterpret_cast<std::uint64_t (*)(std::uint64_t, Observation*)>(*entry)(
        10, &observation);
    if (mode != CallbackState::Mode::resume) return false;
    return check(result == 47 && callback.valid && callback.calls == 1 &&
                     observation.target_order == 1 && observation.continuation_order == 2 &&
                     observation.target_lr == callback.expected_return &&
                     observation.callback_guest_lr == callback.expected_return &&
                     observation.callback_gate_lr == absolute.return_address &&
                     observation.target_sp == observation.callback_sp &&
                     thread.guest_x18() == 0x22222222 &&
                     thread.thread_pointer() == 0x33333333 && scope.dispatched() == 1 &&
                     scope.failure_count() == 0,
                 "The native callback/callee order, registers, LR or SP were incorrect.");
}

bool rejects_failed_callbacks() {
    std::uint32_t size = 0;
    (void)_NSGetExecutablePath(nullptr, &size);
    std::vector<char> executable(size);
    if (_NSGetExecutablePath(executable.data(), &size) != 0) return false;
    const std::array modes = {
        std::pair{"--trap-failed", CallbackState::Mode::fail},
        std::pair{"--trap-threw", CallbackState::Mode::throw_exception},
        std::pair{"--trap-frame", CallbackState::Mode::change_link_register},
    };
    for (const auto& [argument, mode] : modes) {
        (void)mode;
        int pipes[2];
        if (pipe(pipes) != 0) return false;
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, pipes[1], STDERR_FILENO);
        posix_spawn_file_actions_addclose(&actions, pipes[0]);
        posix_spawn_file_actions_addclose(&actions, pipes[1]);
        char* args[] = {executable.data(), const_cast<char*>(argument), nullptr};
        pid_t child = 0;
        const int status = posix_spawn(&child, executable.data(), &actions, nullptr, args, environ);
        posix_spawn_file_actions_destroy(&actions);
        close(pipes[1]);
        if (status != 0) {
            close(pipes[0]);
            return false;
        }
        std::string text;
        std::array<char, 1024> buffer{};
        ssize_t count = 0;
        while ((count = read(pipes[0], buffer.data(), buffer.size())) > 0) {
            text.append(buffer.data(), static_cast<std::size_t>(count));
        }
        close(pipes[0]);
        int result = 0;
        if (waitpid(child, &result, 0) != child ||
            !check(WIFSIGNALED(result) && WTERMSIG(result) == SIGABRT &&
                       text.find("Call hook trapped before original callee") != std::string::npos,
                   "A failed call hook reached its original callee or had the wrong failure.")) {
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        const rlimit no_core{0, 0};
        (void)setrlimit(RLIMIT_CORE, &no_core);
        if (std::strcmp(argv[1], "--trap-failed") == 0)
            return executes_native_hook(CallbackState::Mode::fail) ? 0 : 1;
        if (std::strcmp(argv[1], "--trap-threw") == 0)
            return executes_native_hook(CallbackState::Mode::throw_exception) ? 0 : 1;
        if (std::strcmp(argv[1], "--trap-frame") == 0)
            return executes_native_hook(CallbackState::Mode::change_link_register) ? 0 : 1;
        return 2;
    }
    if (!validates_table() || !validates_patch() || !validates_dispatch_setup() ||
        !executes_native_hook() || !rejects_failed_callbacks()) return 1;
    std::puts("Direct call hooks preserved guest registers, LR, SP and original-callee execution.");
    return 0;
}
