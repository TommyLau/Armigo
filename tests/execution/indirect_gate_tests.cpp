// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/indirect_gate_patch.h>
#include <armigo/execution/gate_code.h>
#include <armigo/platform/host_module.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

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

struct Image {
    std::uint64_t x[31]{};
    std::uint64_t nzcv = 0;
    std::uint64_t sp_before = 0, sp_after = 0;
    std::uint64_t x18_before = 0, x18_after = 0;
    std::uint64_t continuation = 0, marker = 0;
    std::uint64_t q0[2]{}, q31[2]{};
    std::uint64_t sp_at_target = 0;
};
static_assert(offsetof(Image, nzcv) == 248 && offsetof(Image, marker) == 296 &&
              offsetof(Image, q0) == 304 && offsetof(Image, sp_at_target) == 336);

extern "C" {
void armigo_test_indirect_run(const Image*, Image*, std::uint64_t);
extern const std::byte armigo_test_indirect_begin[], armigo_test_indirect_end[];
extern const std::byte armigo_test_indirect_call8[], armigo_test_indirect_tail8[];
extern const std::byte armigo_test_indirect_call16[], armigo_test_indirect_tail16[];
extern const std::byte armigo_test_indirect_target[], armigo_test_indirect_stub_words[];
}

bool check(bool value, const char* text) {
    if (!value) std::fprintf(stderr, "%s\n", text);
    return value;
}

const GuestDispatchScope* rejected_scope = nullptr;
std::uint64_t rejected_target = 0;
void report_rejected_target(const GateFailure&) noexcept {
    const auto* detail = rejected_scope->first_failure();
    if (detail && detail->kind == GuestDispatchFailureKind::indirect_target_rejected &&
        detail->indirect_target == rejected_target)
        std::fputs("Rejected target retained in diagnostics\n", stderr);
    std::fputs("indirect target is outside the prepared target set\n", stderr);
    std::abort();
}

std::size_t offset(const std::byte* label) {
    return reinterpret_cast<std::uintptr_t>(label) -
           reinterpret_cast<std::uintptr_t>(armigo_test_indirect_begin);
}

bool validates_declarations() {
    using E = GuestSiteTableErrorKind;
    const auto word = aarch64::encode_blr(16).word;
    GuestGateSite site{0x1000, word, {0x3000, 0x2000, 0x3000}};
    const auto table = GuestSiteTable::build(std::span(&site, 1));
    if (!check(table.ok(), "An explicit indirect declaration was rejected.")) return false;
    site.allowed_indirect_targets.assign(1, 0x4000);
    const auto* saved = table.table->find(0x1000);
    if (!check(saved && saved->kind == GuestSiteKind::checked_indirect && saved->reg == 16 &&
               saved->allowed_indirect_targets == std::vector<std::uint64_t>({0x2000, 0x3000}) &&
               table.table->has_checked_indirect_sites(),
               "The immutable table did not own its canonical target list.")) return false;
    const auto rejects = [&](GuestGateSite input, E error) {
        const auto result = GuestSiteTable::build(std::span(&input, 1));
        return check(!result.ok() && result.error.kind == error,
                     "Invalid indirect declaration had the wrong result.");
    };
    if (!rejects({0x1000, word}, E::missing_indirect_targets) ||
        !rejects({0x1000, word, {0}}, E::invalid_indirect_target) ||
        !rejects({0x1000, word, {0x2001}}, E::invalid_indirect_target) ||
        !rejects({0x1000, aarch64::encode_svc(1).word, {0x2000}},
                  E::unexpected_indirect_targets) ||
        !rejects({0x1000, aarch64::encode_mrs(1,
            *aarch64::encoding_of(aarch64::SystemRegister::tpidr_el0)).word, {0x2000}},
                  E::unexpected_indirect_targets)) return false;
    for (const auto reg : {18u, 30u, 31u}) {
        for (const auto op : {0xd61f0000u, 0xd63f0000u}) {
            const GuestGateSite input{0x1000, op | (reg << 5), {0x2000}};
            if (!check(!GuestSiteTable::build(std::span(&input, 1)).ok(),
                       "A reserved indirect target register was accepted.")) return false;
        }
    }
    GuestThreadContext thread;
    GuestDispatchScope unsafe(table.table, thread, {}, GateFailurePolicy::resume_unchanged);
    return check(unsafe.setup_error() == GuestDispatchFailureKind::indirect_requires_trap &&
                 unsafe.gate_scope().policy() == GateFailurePolicy::trap && !thread.bound(),
                 "An indirect table could resume after a failed target check.");
}

bool validates_layout() {
    const std::array targets = {std::uint64_t{0x4000}};
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    for (const bool call : {false, true}) {
        const auto word = call ? aarch64::encode_blr(16).word : aarch64::encode_br(16).word;
        const auto plan = make_indirect_gate_site_patch(word, 0x100, 0x200, {0x1000, 512}, targets);
        if (!check(plan.ok(), "A valid indirect layout was rejected.")) return false;
        const auto& patch = *plan.patch;
        std::array<std::uint32_t, 2> oracle{};
        std::memcpy(oracle.data(), armigo_test_indirect_stub_words, sizeof(oracle));
        if (!check(aarch64::decode(patch.replacement_instruction).opcode ==
                       (call ? aarch64::Opcode::bl : aarch64::Opcode::b) &&
                   aarch64::referenced_address(aarch64::decode(patch.replacement_instruction),
                                              0x100) == 0x200 &&
                   aarch64::referenced_address(aarch64::decode(patch.trampoline[1]),
                                              0x204) == 0x1000 &&
                   patch.trampoline[0] == oracle[0] && patch.trampoline[2] == oracle[1] &&
                   patch.trampoline[3] == aarch64::encode_br(16).word &&
                   patch.dispatch_site.return_address == 0x208 &&
                   patch.dispatch_site.instruction == word,
                   "An indirect stub has incorrect branch or stack instructions.")) return false;
        const std::array invalid = {
            make_indirect_gate_site_patch(word, 0x101, 0x200, {0x1000, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x202, {0x1000, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x200, {0x1004, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x200, {0x1000, 0}, targets),
            make_indirect_gate_site_patch(word, maximum - 3, 0x200, {0x1000, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, maximum - 3, {0x1000, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x200, {maximum - 7, 16}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x104, {0x1000, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x200, {0x100, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x200, {0x200, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x10000000, {0x1000, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x200, {0x10000000, 512}, targets),
            make_indirect_gate_site_patch(word, 0x100, 0x200, {0x1000, 512}, {}),
            make_indirect_gate_site_patch(aarch64::encode_svc(1).word, 0x100, 0x200,
                                           {0x1000, 512}, targets),
        };
        for (const auto& result : invalid) {
            if (!check(!result.ok() && result.error != GatePatchError::none,
                       "An invalid indirect patch was accepted.")) return false;
        }
        if (!check(make_indirect_gate_site_patch(word, maximum - 0xfff, maximum - 15,
                   {maximum - 0x1fff, 16}, targets).ok(),
                   "A representable upper-bound layout was rejected.")) return false;
    }
    return true;
}

bool executes(bool reject_target = false) {
    const auto source_size = offset(armigo_test_indirect_end);
    const auto gate = make_local_gate_code();
    const auto gate_offset = (source_size + 7) & ~std::size_t{7};
    const auto first_stub = gate_offset + gate.size();
    const std::array labels = {armigo_test_indirect_call8, armigo_test_indirect_tail8,
                              armigo_test_indirect_call16, armigo_test_indirect_tail16};
    std::vector<std::byte> bytes(first_stub + labels.size() * 16);
    std::memcpy(bytes.data(), armigo_test_indirect_begin, source_size);
    std::memcpy(bytes.data() + gate_offset, gate.data(), gate.size());
    std::vector<GuestGateSite> sites;
    const std::array targets = {std::uint64_t(offset(armigo_test_indirect_target) +
                                             (reject_target ? 4 : 0))};
    for (std::size_t i = 0; i != labels.size(); ++i) {
        const auto at = offset(labels[i]);
        std::uint32_t word;
        std::memcpy(&word, bytes.data() + at, 4);
        const auto plan = make_indirect_gate_site_patch(word, at, first_stub + 16 * i,
                                                       {gate_offset, gate.size()}, targets);
        if (!check(plan.ok(), "Synthetic indirect instruction could not be prepared.")) return false;
        const auto& patch = *plan.patch;
        std::memcpy(bytes.data() + at, &patch.replacement_instruction, 4);
        std::memcpy(bytes.data() + patch.trampoline_address, patch.trampoline.data(), 16);
        sites.push_back(patch.dispatch_site);
    }
    const std::array regions = {platform::ModuleRegion{0, bytes.size(),
        memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(bytes, regions);
    if (!check(mapped.ok(), "Synthetic indirect code could not be mapped.")) return false;
    const auto base = mapped.module->base_address();
    for (auto& site : sites) {
        site.return_address += base;
        for (auto& target : site.allowed_indirect_targets) target += base;
    }
    const auto table = GuestSiteTable::build(sites);
    if (!check(table.ok() && mapped.module->seal().ok(),
               "Synthetic indirect code could not be registered and sealed.")) return false;
    GuestThreadContext thread;
    thread.set_guest_x18(0x18181818);
    GuestDispatchScope scope(table.table, thread);
    if (reject_target) {
        rejected_scope = &scope;
        rejected_target = base + offset(armigo_test_indirect_target);
        set_gate_trap_hook(report_rejected_target);
    }
    for (std::size_t i = 0; i != labels.size(); ++i) {
        Image input{}, output{};
        for (unsigned reg = 0; reg != 31; ++reg) input.x[reg] = 0x12340000ull + reg;
        input.x[8] = input.x[16] = base + offset(armigo_test_indirect_target);
        input.nzcv = (i + 1) << 28;
        input.q0[0] = 0x1020304050607080; input.q0[1] = 0x90a0b0c0d0e0f001;
        input.q31[0] = 0x1122334455667788; input.q31[1] = 0x99aabbccddeeff00;
        const auto entry = base + offset(labels[i]);
        armigo_test_indirect_run(&input, &output, entry);
        if (reject_target) return false;
        for (unsigned reg = 0; reg != 30; ++reg) {
            if (reg == 18) continue;
            const auto expected = reg == 0 ? reinterpret_cast<std::uintptr_t>(&output) :
                                  reg == 17 ? entry : input.x[reg];
            if (!check(output.x[reg] == expected, "A checked branch clobbered a guest register.")) {
                std::fprintf(stderr, "Form%zu register%u: got%llx expected%llx\n", i, reg,
                    static_cast<unsigned long long>(output.x[reg]),
                    static_cast<unsigned long long>(expected));
                return false;
            }
        }
        const bool call = i % 2 == 0;
        if (!check(output.x[30] == (call ? entry + 4 : output.continuation) &&
                   output.marker == (call ? 1 : 0) && output.nzcv == input.nzcv &&
                   output.sp_before == output.sp_after &&
                   output.sp_before == output.sp_at_target &&
                   output.x18_before == output.x18_after &&
                   std::memcmp(output.q0, input.q0, sizeof(input.q0)) == 0 &&
                   std::memcmp(output.q31, input.q31, sizeof(input.q31)) == 0,
                   "Checked call/tail return, stack, flags or SIMD state differs.")) return false;
    }
    return check(scope.dispatched() == labels.size() && scope.failure_count() == 0 &&
                 thread.guest_x18() == 0x18181818,
                 "Checked native branches did not complete exactly once each.");
}

bool rejects_runtime_target() {
    std::uint32_t size = 0;
    (void)_NSGetExecutablePath(nullptr, &size);
    std::vector<char> executable(size);
    if (_NSGetExecutablePath(executable.data(), &size) != 0) return false;
    int pipes[2];
    if (pipe(pipes) != 0) return false;
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipes[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipes[0]);
    posix_spawn_file_actions_addclose(&actions, pipes[1]);
    char mode[] = "--reject-target";
    char* args[] = {executable.data(), mode, nullptr};
    pid_t child = 0;
    const int status = posix_spawn(&child, executable.data(), &actions, nullptr, args, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipes[1]);
    if (status != 0) { close(pipes[0]); return false; }
    std::string text;
    std::array<char, 1024> buffer{};
    ssize_t count;
    while ((count = read(pipes[0], buffer.data(), buffer.size())) > 0)
        text.append(buffer.data(), static_cast<std::size_t>(count));
    close(pipes[0]);
    int result = 0;
    if (waitpid(child, &result, 0) != child) return false;
    return check(WIFSIGNALED(result) && WTERMSIG(result) == SIGABRT &&
                 text.find("indirect target is outside the prepared target set") != std::string::npos &&
                 text.find("Rejected target retained in diagnostics") != std::string::npos,
                 "An undeclared runtime target did not fail before native transfer.");
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--reject-target") == 0) {
        const rlimit no_core{0, 0};
        (void)setrlimit(RLIMIT_CORE, &no_core);
        return executes(true) ? 0 : 1;
    }
    if (!validates_declarations() || !validates_layout() || !executes() ||
        !rejects_runtime_target()) return 1;
    std::puts("Checked BR/BLR target validation, native returns and register preservation passed.");
    return 0;
}
