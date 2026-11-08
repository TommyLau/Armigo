// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/register_effects.h>
#include <armigo/execution/indirect_x18_patch.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
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
constexpr ScalarX18Policy contract{true, true, true};

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
void armigo_test_indirect_x18_run(const Image*, Image*, std::uint64_t);
extern const std::byte armigo_test_indirect_x18_begin[], armigo_test_indirect_x18_site_end[];
extern const std::byte armigo_test_indirect_x18_target[], armigo_test_indirect_x18_marker[];
extern const std::byte armigo_test_indirect_x18_end[];
extern const std::uint32_t armigo_test_indirect_x18_oracle[4];
}

bool expect(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}

std::size_t distance(const std::byte* begin, const std::byte* end) {
    return reinterpret_cast<std::uintptr_t>(end) - reinterpret_cast<std::uintptr_t>(begin);
}

bool refusals() {
    using E = IndirectX18PatchError;
    const auto word = aarch64::encode_br(18).word;
    const std::array targets = {IndirectX18Target{0x12340000, 0x4000}};
    const auto reject = [&](std::uint32_t instruction, std::uint64_t at, std::uint64_t stub,
                            std::uint32_t slot, ScalarX18Policy policy,
                            std::span<const IndirectX18Target> map, E expected) {
        const auto result = make_indirect_x18_patch(instruction, at, stub, slot, map, policy);
        return expect(!result.ok() && result.error == expected, "Incorrect indirect x18 refusal.");
    };
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (!reject(word, 0x101, 0x200, 0, contract, targets, E::invalid_layout) ||
        !reject(word, 0x100, 0x202, 0, contract, targets, E::invalid_layout) ||
        !reject(word, maximum - 3, 0x200, 0, contract, targets, E::invalid_layout) ||
        !reject(word, 0x100, maximum - 51, 0, contract, targets, E::invalid_layout) ||
        !reject(word, 0x100, 0x104, 0, contract, targets, E::overlapping_regions) ||
        !reject(word, 0x100, 0xcc, 0, contract, targets, E::overlapping_regions) ||
        !reject(word, 0x100, 0x200, 1, contract, targets, E::invalid_thread_slot) ||
        !reject(word, 0x100, 0x200, 32768, contract, targets, E::invalid_thread_slot) ||
        !reject(word, 0x100, 0x200, 0, {}, targets, E::missing_execution_contract) ||
        !reject(word, 0x100, 0x200, 0, {false, true, true}, targets, E::missing_execution_contract) ||
        !reject(word, 0x100, 0x200, 0, {true, false, true}, targets, E::missing_execution_contract) ||
        !reject(word, 0x100, 0x200, 0, {true, true, false}, targets, E::missing_execution_contract) ||
        !reject(0xffffffff, 0x100, 0x200, 0, contract, targets, E::unsupported_instruction) ||
        !reject(aarch64::encode_ret(18).word, 0x100, 0x200, 0, contract, targets, E::unsupported_instruction) ||
        !reject(aarch64::encode_br(0).word, 0x100, 0x200, 0, contract, targets, E::register_not_used) ||
        !reject(word, 0x100, 0x10000000, 0, contract, targets, E::unreachable_branch) ||
        !reject(word, 0x100, 0x200, 0, contract, {}, E::missing_targets)) return false;
    for (const auto target : {IndirectX18Target{0, 0x4000}, {0x1000, 0},
                              {0x1001, 0x4000}, {0x1000, 0x4001}}) {
        if (!reject(word, 0x100, 0x200, 0, contract, std::span(&target, 1), E::invalid_target))
            return false;
    }
    for (const auto address : {0x200ull, 0x234ull}) {
        const IndirectX18Target target{0x1000, address};
        if (!reject(word, 0x100, 0x200, 0, contract, std::span(&target, 1), E::overlapping_regions))
            return false;
    }
    const std::array duplicate = {IndirectX18Target{0x1000, 0x4000}, IndirectX18Target{0x1000, 0x5000}};
    const std::array identical = {IndirectX18Target{0x1000, 0x4000}, IndirectX18Target{0x1000, 0x4000}};
    const IndirectX18Target far{0x1000, 0x10000000};
    if (!reject(word, 0x100, 0x200, 0, contract, duplicate, E::duplicate_target) ||
        !reject(word, 0x100, 0x200, 0, contract, identical, E::duplicate_target) ||
        !reject(word, 0x100, 0x200, 0, contract, std::span(&far, 1), E::unreachable_branch)) return false;
    const std::array aliases = {IndirectX18Target{0x1000, 0x4000}, IndirectX18Target{0x2000, 0x4000}};
    const IndirectX18Target top{maximum - 3, maximum - 0xfff};
    return expect(make_indirect_x18_patch(word, 0x100, 0x200, 32760, aliases, contract).ok() &&
                  make_indirect_x18_patch(word, maximum - 0xfff, maximum - 55, 0,
                                         std::span(&top, 1), contract).ok(),
                  "Valid aliases, slot boundary or top-of-address-space layout were rejected.");
}

bool executes(bool call, bool reject_target = false) {
    const auto site_size = distance(armigo_test_indirect_x18_begin, armigo_test_indirect_x18_site_end);
    const auto target_size = distance(armigo_test_indirect_x18_target, armigo_test_indirect_x18_end);
    const auto marker_offset = distance(armigo_test_indirect_x18_target, armigo_test_indirect_x18_marker);
    constexpr std::size_t target_offset = 64, stub = 1024;
    std::vector<std::byte> bytes(stub + 4 * (6 + 8 * 3));
    std::memcpy(bytes.data(), armigo_test_indirect_x18_begin, site_size);
    for (unsigned index = 0; index != 3; ++index) {
        const auto at = target_offset + index * target_size;
        std::memcpy(bytes.data() + at, armigo_test_indirect_x18_target, target_size);
        const std::uint32_t marker = 0xd2800001u | ((index + 1) << 5);
        std::memcpy(bytes.data() + at + marker_offset, &marker, 4);
    }
    const std::array regions = {platform::ModuleRegion{0, bytes.size(),
        memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(bytes, regions);
    if (!expect(mapped.ok(), "Indirect x18 fixture mapping failed.")) return false;
    const auto base = mapped.module->base_address();
    // The first two pointer identities share low bits, exercising full-width equality.
    std::array targets = {IndirectX18Target{0x1234abcd00004000ull, base + target_offset},
        IndirectX18Target{0xfedcabcd00004000ull, base + target_offset + target_size},
        IndirectX18Target{base + target_offset + 2 * target_size, base + target_offset + 2 * target_size}};
    platform::ThreadSlotKey key;
    if (!expect(key.valid() && key.native_access().has_value(), "Native TSD probe failed.")) return false;
    GuestThreadContext thread;
    const auto table = GuestSiteTable::build({});
    if (!expect(table.ok(), "Empty context-dispatch table was rejected.")) return false;
    GuestDispatchScope dispatch(table.table, thread);
    if (!expect(dispatch.setup_error() == GuestDispatchFailureKind::none, "Guest context binding failed."))
        return false;
    platform::ThreadSlotBinding binding(key, thread.bound_register_cell());
    if (!expect(binding.native_access().has_value(), "Guest register-cell publication failed.")) return false;
    const auto word = call ? aarch64::encode_blr(18).word : aarch64::encode_br(18).word;
    const auto result = make_indirect_x18_patch(word, base, base + stub,
        key.native_access()->tsd_byte_offset, targets, contract);
    if (!expect(result.ok(), "Native indirect x18 fixture could not be prepared.")) return false;
    const auto& patch = *result.patch;
    const auto& oracle = armigo_test_indirect_x18_oracle;
    if (!expect(patch.expected_instruction == word && patch.trampoline.size() == 30 &&
                aarch64::decode(patch.replacement_instruction).opcode ==
                    (call ? aarch64::Opcode::bl : aarch64::Opcode::b) &&
                patch.trampoline[0] == oracle[0] && patch.trampoline[8] == oracle[2] &&
                patch.trampoline[10] == oracle[1] && patch.trampoline[28] == oracle[1] &&
                patch.trampoline[29] == oracle[3], "Incorrect fixed indirect dispatch instructions.")) return false;
    for (std::size_t index = 0; index + 1 < patch.trampoline.size(); ++index) {
        const auto effects = aarch64::decode_register_effects(patch.trampoline[index]);
        if (!expect(aarch64::proven_free_of(effects, 18),
                    "Generated dispatch touches hardware x18 or has unknown effects.")) return false;
    }
    auto writable = mapped.module->preparation_bytes();
    std::memcpy(writable.data(), &patch.replacement_instruction, 4);
    std::memcpy(writable.data() + stub, patch.trampoline.data(), patch.trampoline.size() * 4);
    if (!expect(mapped.module->seal().ok(), "Indirect x18 fixture sealing failed.")) return false;
    for (unsigned index = 0; index != 3; ++index) {
        for (unsigned flags = 0; flags != 16; ++flags) {
            const auto value = reject_target ? std::uint64_t{0} : targets[index].guest_address;
            thread.set_guest_x18(value);
            Image input{}, output{};
            for (unsigned reg = 0; reg != 31; ++reg) input.x[reg] = 0x9876543200000000ull + reg;
            input.nzcv = std::uint64_t{flags} << 28;
            input.q0[0] = 0x1020304050607080; input.q0[1] = 0x90a0b0c0d0e0f001;
            input.q31[0] = 0x1122334455667788; input.q31[1] = 0x99aabbccddeeff00;
            armigo_test_indirect_x18_run(&input, &output, base);
            if (reject_target) return false;
            for (unsigned reg = 0; reg != 30; ++reg) {
                if (reg == 18) continue;
                const auto expected = reg == 0 ? reinterpret_cast<std::uintptr_t>(&output) :
                                      reg == 17 ? base : input.x[reg];
                if (!expect(output.x[reg] == expected, "Indirect x18 dispatch clobbered a GPR.")) return false;
            }
            if (!expect(output.marker == index + 1 &&
                        output.x[30] == (call ? base + 4 : output.continuation) &&
                        output.nzcv == input.nzcv && output.sp_before == output.sp_after &&
                        output.sp_before == output.sp_at_target && output.x18_before == output.x18_after &&
                        thread.guest_x18() == value &&
                        std::memcmp(output.q0, input.q0, sizeof(input.q0)) == 0 &&
                        std::memcmp(output.q31, input.q31, sizeof(input.q31)) == 0,
                        "Indirect x18 target, link, flags, stack or virtual register differs.")) return false;
        }
    }
    return true;
}

// A table of `entries` words (8192 or 65536, a halfword's whole range) past
// which the trampoline lies: its size, the bound check's shift and the last
// entry's branch to the table's last word.
bool large_table(std::uint32_t entries, unsigned shift) {
    constexpr std::uint64_t at = 0x100, table = 0x1000, stub = 0x200000;
    const auto result = make_indirect_x18_table_patch(aarch64::encode_br(18).word, at, stub, 0, table, entries, contract);
    if (!expect(result.ok(), "A large table layout was rejected.")) return false;
    const auto& words = result.patch->trampoline;
    if (!expect(words.size() == indirect_x18_table_words(entries), "Large table trampoline size differs.")) return false;
    const auto last_at = 16 + 2 * std::size_t{entries - 1} + 1;
    const auto last = aarch64::decode(words[last_at]);
    return expect(words[9] == (0xd340fc01u | (shift << 16)), "Large table bound shift differs.") &&
           expect(last.opcode == aarch64::Opcode::b &&
                      stub + 4 * last_at + static_cast<std::uint64_t>(last.displacement) ==
                          table + 4 * std::uint64_t{entries - 1},
                  "Large table last entry branches elsewhere.");
}

bool table_refusals() {
    using E = IndirectX18PatchError;
    const auto word = aarch64::encode_br(18).word;
    const auto reject = [&](std::uint32_t instruction, std::uint64_t at, std::uint64_t stub,
                            std::uint32_t slot, ScalarX18Policy policy, std::uint64_t table,
                            std::uint32_t entries, E expected) {
        const auto result = make_indirect_x18_table_patch(instruction, at, stub, slot, table, entries, policy);
        return expect(!result.ok() && result.error == expected, "Incorrect table x18 refusal.");
    };
    return reject(word, 0x100, 0x2000, 0, contract, 0x1000, 0, E::missing_targets) &&
           reject(word, 0x100, 0x2000, 0, contract, 0x1000, 3, E::missing_targets) &&
           reject(word, 0x100, 0x2000, 0, contract, 0x1000, 131072, E::missing_targets) &&
           reject(word, 0x102, 0x2000, 0, contract, 0x1000, 4, E::invalid_layout) &&
           reject(word, 0x100, 0x2002, 0, contract, 0x1000, 4, E::invalid_layout) &&
           reject(word, 0x2000, 0x2000, 0, contract, 0x1000, 4, E::overlapping_regions) &&
           reject(word, 0x100, 0x2000, 0, contract, 0, 4, E::invalid_target) &&
           reject(word, 0x100, 0x2000, 0, contract, 0x1002, 4, E::invalid_target) &&
           reject(word, 0x100, 0x2000, 0, contract, 0x1ff8, 4, E::overlapping_regions) &&
           reject(word, 0x100, 0x2000, 0, contract, 0x2010, 4, E::overlapping_regions) &&
           reject(word, 0x100, 0x2000, 1, contract, 0x1000, 4, E::invalid_thread_slot) &&
           reject(word, 0x100, 0x2000, 0, {}, 0x1000, 4, E::missing_execution_contract) &&
           reject(aarch64::encode_blr(18).word, 0x100, 0x2000, 0, contract, 0x1000, 4,
                  E::unsupported_instruction) &&
           reject(aarch64::encode_br(0).word, 0x100, 0x2000, 0, contract, 0x1000, 4, E::register_not_used) &&
           reject(word, 0x100, 0x10000000, 0, contract, 0x1000, 4, E::unreachable_branch) &&
           expect(make_indirect_x18_table_patch(word, 0x100, 0x2000, 32760, 0x1000, 256, contract).ok() &&
                  make_indirect_x18_table_patch(word, 0x100, 0x2000, 0, 0x1000, 256, contract).patch->trampoline.size() ==
                      indirect_x18_table_words(256),
                  "A valid table layout was rejected.") &&
           large_table(8192, 15) && large_table(65536, 18);
}

// A switch table: BR X18 to one of four table words, each a branch to its own
// copy of the target code. reject: 1 targets the word past the table, 2 a
// misaligned table address, 3 the word before the table; each must trap
// before any transfer.
bool executes_table(int reject = 0) {
    const auto site_size = distance(armigo_test_indirect_x18_begin, armigo_test_indirect_x18_site_end);
    const auto target_size = distance(armigo_test_indirect_x18_target, armigo_test_indirect_x18_end);
    const auto marker_offset = distance(armigo_test_indirect_x18_target, armigo_test_indirect_x18_marker);
    constexpr std::uint32_t entries = 4;
    constexpr std::size_t table_offset = 64, case_offset = 128, stub = 2048;
    const auto words = indirect_x18_table_words(entries);
    std::vector<std::byte> bytes(stub + 4 * words);
    std::memcpy(bytes.data(), armigo_test_indirect_x18_begin, site_size);
    for (unsigned index = 0; index != entries; ++index) {
        const auto at = case_offset + index * target_size;
        std::memcpy(bytes.data() + at, armigo_test_indirect_x18_target, target_size);
        const std::uint32_t marker = 0xd2800001u | ((index + 1) << 5);
        std::memcpy(bytes.data() + at + marker_offset, &marker, 4);
        const auto entry = aarch64::encode_b(table_offset + 4 * index, at);
        if (!expect(entry.ok(), "Table entry branch could not be encoded.")) return false;
        std::memcpy(bytes.data() + table_offset + 4 * index, &entry.word, 4);
    }
    const std::array regions = {platform::ModuleRegion{0, bytes.size(),
        memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(bytes, regions);
    if (!expect(mapped.ok(), "Table x18 fixture mapping failed.")) return false;
    const auto base = mapped.module->base_address();
    platform::ThreadSlotKey key;
    if (!expect(key.valid() && key.native_access().has_value(), "Native TSD probe failed.")) return false;
    GuestThreadContext thread;
    const auto table = GuestSiteTable::build({});
    if (!expect(table.ok(), "Empty context-dispatch table was rejected.")) return false;
    GuestDispatchScope dispatch(table.table, thread);
    if (!expect(dispatch.setup_error() == GuestDispatchFailureKind::none, "Guest context binding failed."))
        return false;
    platform::ThreadSlotBinding binding(key, thread.bound_register_cell());
    if (!expect(binding.native_access().has_value(), "Guest register-cell publication failed.")) return false;
    const auto word = aarch64::encode_br(18).word;
    const auto result = make_indirect_x18_table_patch(word, base, base + stub,
        key.native_access()->tsd_byte_offset, base + table_offset, entries, contract);
    if (!expect(result.ok(), "Native table x18 fixture could not be prepared.")) return false;
    const auto& patch = *result.patch;
    if (!expect(patch.trampoline.size() == words &&
                aarch64::decode(patch.replacement_instruction).opcode == aarch64::Opcode::b &&
                patch.trampoline.back() == aarch64::encode_brk(indirect_x18_table_trap).word,
                "Incorrect table dispatch shape.")) return false;
    for (std::size_t index = 0; index + 1 < patch.trampoline.size(); ++index) {
        const auto effects = aarch64::decode_register_effects(patch.trampoline[index]);
        if (!expect(aarch64::proven_free_of(effects, 18),
                    "Generated table dispatch touches hardware x18 or has unknown effects.")) return false;
    }
    auto writable = mapped.module->preparation_bytes();
    std::memcpy(writable.data(), &patch.replacement_instruction, 4);
    std::memcpy(writable.data() + stub, patch.trampoline.data(), patch.trampoline.size() * 4);
    if (!expect(mapped.module->seal().ok(), "Table x18 fixture sealing failed.")) return false;
    for (unsigned index = 0; index != entries; ++index) {
        for (unsigned flags = 0; flags != 16; ++flags) {
            const auto value = reject == 1 ? base + table_offset + 4 * entries :
                               reject == 2 ? base + table_offset + 2 :
                               reject == 3 ? base + table_offset - 4 :
                                             base + table_offset + 4 * index;
            thread.set_guest_x18(value);
            Image input{}, output{};
            for (unsigned reg = 0; reg != 31; ++reg) input.x[reg] = 0x9876543200000000ull + reg;
            input.nzcv = std::uint64_t{flags} << 28;
            input.q0[0] = 0x1020304050607080; input.q0[1] = 0x90a0b0c0d0e0f001;
            input.q31[0] = 0x1122334455667788; input.q31[1] = 0x99aabbccddeeff00;
            armigo_test_indirect_x18_run(&input, &output, base);
            if (reject) return false;
            for (unsigned reg = 0; reg != 30; ++reg) {
                if (reg == 18) continue;
                const auto expected = reg == 0 ? reinterpret_cast<std::uintptr_t>(&output) :
                                      reg == 17 ? base : input.x[reg];
                if (!expect(output.x[reg] == expected, "Table x18 dispatch clobbered a GPR.")) return false;
            }
            if (!expect(output.marker == index + 1 && output.x[30] == output.continuation &&
                        output.nzcv == input.nzcv && output.sp_before == output.sp_after &&
                        output.sp_before == output.sp_at_target && output.x18_before == output.x18_after &&
                        thread.guest_x18() == value &&
                        std::memcmp(output.q0, input.q0, sizeof(input.q0)) == 0 &&
                        std::memcmp(output.q31, input.q31, sizeof(input.q31)) == 0,
                        "Table x18 target, link, flags, stack or virtual register differs.")) return false;
        }
    }
    return true;
}

bool traps(const char* mode) {
    std::uint32_t size = 0;
    (void)_NSGetExecutablePath(nullptr, &size);
    std::vector<char> executable(size);
    if (_NSGetExecutablePath(executable.data(), &size) != 0) return false;
    std::string argument = mode;
    char* args[] = {executable.data(), argument.data(), nullptr};
    pid_t child = 0;
    if (posix_spawn(&child, executable.data(), nullptr, nullptr, args, environ) != 0) return false;
    int status = 0;
    return expect(waitpid(child, &status, 0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGTRAP,
                  "A target outside the table did not trap before transfer.");
}

bool rejects_runtime_target() {
    std::uint32_t size = 0;
    (void)_NSGetExecutablePath(nullptr, &size);
    std::vector<char> executable(size);
    if (_NSGetExecutablePath(executable.data(), &size) != 0) return false;
    char mode[] = "--reject-target";
    char* args[] = {executable.data(), mode, nullptr};
    pid_t child = 0;
    if (posix_spawn(&child, executable.data(), nullptr, nullptr, args, environ) != 0) return false;
    int status = 0;
    return expect(waitpid(child, &status, 0) == child && WIFSIGNALED(status) && WTERMSIG(status) == SIGTRAP,
                  "An unmapped virtual target did not trap before transfer.");
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--reject-target") == 0) {
        const rlimit no_core{0, 0};
        (void)setrlimit(RLIMIT_CORE, &no_core);
        return executes(false, true) ? 0 : 1;
    }
    for (const auto [mode, reject] : {std::pair{"--reject-past-table", 1}, std::pair{"--reject-misaligned", 2},
                                      std::pair{"--reject-before-table", 3}}) {
        if (argc == 2 && std::strcmp(argv[1], mode) == 0) {
            const rlimit no_core{0, 0};
            (void)setrlimit(RLIMIT_CORE, &no_core);
            return executes_table(reject) ? 0 : 1;
        }
    }
    if (!refusals() || !executes(false) || !executes(true) || !rejects_runtime_target()) return 1;
    if (!table_refusals() || !executes_table() || !traps("--reject-past-table") || !traps("--reject-misaligned") ||
        !traps("--reject-before-table"))
        return 1;
    std::puts("Virtual BR/BLR X18 preserved native state across mapped target copies and table words, and "
              "rejected unknown and out-of-table targets.");
    return 0;
}
