// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/register_substitution.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/scalar_x18_patch.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
using namespace armigo;
using namespace armigo::execution;
constexpr ScalarX18Policy contract{true, true, true, true};

struct Image {
    std::uint64_t x[31]{};
    std::uint64_t nzcv = 0, sp = 0, hardware_x18_before = 0;
    std::array<std::array<std::uint64_t, 2>, 32> q{};
    std::uint64_t fpcr = 0, fpsr = 0;
};
static_assert(offsetof(Image, nzcv) == 248 && offsetof(Image, sp) == 256 &&
              offsetof(Image, q) == 272 && offsetof(Image, fpcr) == 784 && sizeof(Image) == 800);

struct Form {
    std::uint32_t kind; // 0: value; 1: base; 2: post-index register; 3: SP arithmetic.
    std::uint32_t original;
    std::uint32_t reference;
};
static_assert(sizeof(Form) == 12);
struct Refusal {
    std::uint32_t exclusive;
    std::uint32_t word;
};
static_assert(sizeof(Refusal) == 8);
extern "C" {
void armigo_test_x18_forms_run(Image*, std::uint64_t);
void armigo_test_x18_forms_run_on(Image*, std::uint64_t, std::byte* stack_top);
extern const Form armigo_test_x18_forms[], armigo_test_x18_forms_end[];
extern const Refusal armigo_test_x18_refusals[], armigo_test_x18_refusals_end[];
extern const Form armigo_test_x18_swap_forms[], armigo_test_x18_swap_forms_end[];
}

bool expect(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}

bool execute(Form form, std::uint32_t slot, GuestThreadContext& context) {
    const auto original = aarch64::decode_register_effects(form.original);
    const auto renamed = aarch64::substitute_register(form.original, 18, 28,
                                                      aarch64::UnchangedWord::reject);
    if (!expect(original.decoded() && renamed.ok() && renamed.word == form.reference,
                "Assembler oracle disagrees with decoding or checked substitution.")) return false;
    constexpr std::uint64_t stub = 64;
    const auto plan = make_scalar_x18_patch(form.original, 0, stub, slot, contract);
    if (!expect(plan.ok(), "An authored x18 form was not prepared.")) return false;
    const auto& patch = *plan.patch;
    if (!expect(patch.expected_instruction == form.original && patch.trampoline_word_count <= 12,
                "Prepared form has an invalid identity or size.")) return false;
    for (std::size_t i = 0; i < patch.trampoline_word_count; ++i) {
        if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(patch.trampoline[i]), 18),
                    "Prepared form uses hardware x18 or has incomplete effects.")) return false;
    }
    // The two functions execute through the same wrapper call site, with the
    // same SP and LR. Only the reference uses X28 instead of virtual X18.
    std::vector<std::byte> bytes(128);
    const std::uint32_t ret = aarch64::encode_ret().word;
    std::memcpy(bytes.data(), &patch.replacement_instruction, 4);
    std::memcpy(bytes.data() + 4, &ret, 4);
    std::memcpy(bytes.data() + 16, &form.reference, 4);
    std::memcpy(bytes.data() + 20, &ret, 4);
    std::memcpy(bytes.data() + stub, patch.trampoline.data(), patch.trampoline_word_count * 4);
    const std::array regions = {platform::ModuleRegion{
        0, bytes.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(bytes, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "Form mapping or sealing failed.")) return false;
    const auto prepared_entry = mapped.module->executable_address(0);
    const auto reference_entry = mapped.module->executable_address(16);
    if (!expect(prepared_entry && reference_entry, "Form executable entry is absent.")) return false;
    alignas(16) std::array<std::byte, 256> memory{};
    std::array<std::byte, 256> initial_memory{}, reference_memory{};
    for (std::size_t i = 0; i < memory.size(); ++i) initial_memory[i] = std::byte((37 * i + 91) & 255);
    for (const auto input : {0xfedcba9880000081ull, 0x0000000100000103ull, 0ull}) {
        Image initial{};
        for (unsigned reg = 0; reg < 31; ++reg) initial.x[reg] = 0x9876543210000000ull + reg;
        initial.x[2] = input == 0 ? 0 : input == 0x0000000100000103ull ? 32 : UINT64_MAX - 31;
        initial.x[3] = reinterpret_cast<std::uintptr_t>(memory.data() + 64);
        for (unsigned reg = 0; reg < 32; ++reg) {
            initial.q[reg] = {0xc129e1808000ff81ull + reg, 0xc0298000ff800081ull + reg};
        }
        initial.nzcv = 0xa0000000;
        const auto virtual_value = form.kind == 1 ? initial.x[3] : input;
        Image reference = initial;
        reference.x[28] = virtual_value;
        memory = initial_memory;
        armigo_test_x18_forms_run(&reference, *reference_entry);
        reference_memory = memory;
        Image actual = initial;
        memory = initial_memory;
        context.set_guest_x18(virtual_value);
        armigo_test_x18_forms_run(&actual, *prepared_entry);
        if (!expect(memory == reference_memory && actual.q == reference.q &&
                        actual.nzcv == reference.nzcv && actual.sp == reference.sp &&
                        actual.fpcr == reference.fpcr && actual.fpsr == reference.fpsr &&
                        actual.x[18] == actual.hardware_x18_before &&
                        actual.x[28] == initial.x[28] && context.guest_x18() == reference.x[28],
                    "Prepared instruction differs from native reference state or memory.")) return false;
        for (unsigned reg = 0; reg < 31; ++reg) {
            if (reg == 28) continue;
            if (!expect(actual.x[reg] == reference.x[reg],
                        "Prepared instruction changed a guest-visible general register.")) return false;
        }
        if (form.kind == 3) {
            const bool wide = (form.original >> 31) != 0;
            const bool subtract = (form.original & (1u << 30)) != 0;
            const auto immediate = std::uint64_t{(form.original >> 10) & 4095u} <<
                                   ((form.original & (1u << 22)) ? 12 : 0);
            auto expected = subtract ? actual.sp - immediate : actual.sp + immediate;
            if (!wide) expected = static_cast<std::uint32_t>(expected);
            if (!expect(context.guest_x18() == expected,
                        "SP arithmetic used the spilled SP or lost an immediate boundary.")) return false;
        }
    }
    return true;
}

// One compare-and-swap form (kind 1: a CASP pair at x18:x19, renamed to
// x28:x29 in the reference), with a compare value that matches memory and one
// that differs. The base is x4. Guest x18 and x19 are compared with the
// reference's x28 and x29. Both run on a guarded stack: only the declared
// scratch below entry SP (32 bytes for a pair, 16 otherwise) may change.
bool execute_swap(Form form, std::uint32_t slot, GuestThreadContext& context) {
    const bool pair = form.kind == 1;
    const auto effects = aarch64::decode_register_effects(form.original);
    if (!expect(effects.decoded() &&
                    effects.operation == (pair ? aarch64::Operation::casp : aarch64::Operation::cas),
                "A swap form does not decode as compare and swap.")) return false;
    constexpr std::uint64_t stub = 64;
    auto narrow = contract;
    narrow.pair_stack_scratch_available = false;
    const auto without_pair_scratch = make_scalar_x18_patch(form.original, 0, stub, slot, narrow);
    if (!expect(pair ? without_pair_scratch.error == ScalarX18PatchError::missing_execution_contract
                     : without_pair_scratch.ok(),
                "Only a CASP pair at x18 requires 32 bytes of stack scratch.")) return false;
    const auto plan = make_scalar_x18_patch(form.original, 0, stub, slot, contract);
    if (!expect(plan.ok(), "An authored swap form was not prepared.")) return false;
    const auto& patch = *plan.patch;
    for (std::size_t i = 0; i < patch.trampoline_word_count; ++i) {
        if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(patch.trampoline[i]), 18),
                    "Prepared swap form uses hardware x18 or has incomplete effects.")) return false;
    }
    std::vector<std::byte> bytes(128);
    const std::uint32_t ret = aarch64::encode_ret().word;
    std::memcpy(bytes.data(), &patch.replacement_instruction, 4);
    std::memcpy(bytes.data() + 4, &ret, 4);
    std::memcpy(bytes.data() + 16, &form.reference, 4);
    std::memcpy(bytes.data() + 20, &ret, 4);
    std::memcpy(bytes.data() + stub, patch.trampoline.data(), patch.trampoline_word_count * 4);
    const std::array regions = {platform::ModuleRegion{
        0, bytes.size(), memory::Permission::read | memory::Permission::execute}};
    auto mapped = platform::HostModule::create(bytes, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "Swap form mapping or sealing failed.")) return false;
    const auto prepared_entry = mapped.module->executable_address(0);
    const auto reference_entry = mapped.module->executable_address(16);
    if (!expect(prepared_entry && reference_entry, "Swap form executable entry is absent.")) return false;
    const unsigned rs = (form.original >> 16) & 31u;
    const unsigned bytes_per = pair ? ((form.original >> 30) & 1u ? 8u : 4u) : 1u << (form.original >> 30);
    alignas(16) std::array<std::byte, 256> memory{};
    std::array<std::byte, 256> initial_memory{}, reference_memory{};
    for (std::size_t i = 0; i < memory.size(); ++i) initial_memory[i] = std::byte((37 * i + 91) & 255);
    const auto element = [&](unsigned index) {
        std::uint64_t value = 0;
        std::memcpy(&value, initial_memory.data() + 64 + index * bytes_per, bytes_per);
        return value;
    };
    // The runner's 256-byte frame sits at the top of the stack; entry SP is
    // just below it.
    alignas(16) std::array<std::byte, 1024> stack{};
    constexpr std::size_t entry_sp = stack.size() - 256;
    const std::size_t scratch = pair ? 32 : 16;
    constexpr auto guard = std::byte{0xa5};
    const auto run = [&](Image& image, std::uint64_t function) {
        stack.fill(guard);
        armigo_test_x18_forms_run_on(&image, function, stack.data() + stack.size());
    };
    const auto untouched = [&](std::size_t end) {
        return std::all_of(stack.begin(), stack.begin() + static_cast<std::ptrdiff_t>(end),
                           [&](std::byte value) { return value == guard; });
    };
    for (const bool match : {true, false}) {
        Image initial{};
        for (unsigned reg = 0; reg < 31; ++reg) initial.x[reg] = 0x9876543210000000ull + reg;
        initial.x[4] = reinterpret_cast<std::uintptr_t>(memory.data() + 64);
        for (unsigned reg = 0; reg < 32; ++reg)
            initial.q[reg] = {0xc129e1808000ff81ull + reg, 0xc0298000ff800081ull + reg};
        initial.nzcv = 0x60000000;
        // The compare register (pair) holds memory's value(s), plus one when
        // the compare must fail. Guest x18 lives in the cell.
        std::uint64_t guest_x18 = 0x5a5a5a5a00000018ull;
        const auto set = [&](unsigned reg, std::uint64_t value) {
            if (reg == 18) guest_x18 = value;
            else initial.x[reg] = value;
        };
        set(rs, element(0) + (match ? 0 : 1));
        if (pair) set(rs + 1, element(1));
        Image reference = initial;
        reference.x[28] = guest_x18;
        if (pair) reference.x[29] = initial.x[19];
        memory = initial_memory;
        run(reference, *reference_entry);
        if (!expect(untouched(entry_sp), "The native reference wrote below its stack pointer.")) return false;
        reference_memory = memory;
        Image actual = initial;
        memory = initial_memory;
        context.set_guest_x18(guest_x18);
        run(actual, *prepared_entry);
        // The spill holds the first scratch at the bottom of the declared
        // area, so it shows the area was used, and nothing below it changed.
        if (!expect(untouched(entry_sp - scratch) &&
                        !untouched(entry_sp - scratch + 8),
                    "The prepared form used stack outside its declared scratch.")) return false;
        // A match stores Rt; when Rt is the compare register itself the
        // stored value equals memory, so only the reference decides.
        const unsigned rt = form.original & 31u;
        if (!expect(memory == reference_memory && (rs == rt || (memory != initial_memory) == match),
                    "Prepared compare and swap differs from the reference in memory or outcome.")) return false;
        if (!expect(actual.q == reference.q && actual.nzcv == reference.nzcv && actual.sp == reference.sp &&
                        actual.fpcr == reference.fpcr && actual.fpsr == reference.fpsr &&
                        actual.x[18] == actual.hardware_x18_before && context.guest_x18() == reference.x[28] &&
                        actual.x[28] == initial.x[28] && actual.x[29] == initial.x[29] &&
                        actual.x[19] == (pair ? reference.x[29] : reference.x[19]),
                    "Prepared compare and swap differs from the reference state.")) return false;
        for (unsigned reg = 0; reg < 31; ++reg) {
            if (reg == 18 || reg == 19 || reg == 28 || reg == 29) continue;
            if (!expect(actual.x[reg] == reference.x[reg],
                        "Prepared compare and swap changed a guest-visible general register.")) return false;
        }
    }
    return true;
}

bool stack_arithmetic(std::uint32_t slot, GuestThreadContext& context) {
    // All ADD/SUB immediate variants, zero and maximum imm12, both shifts and
    // W/X views. Flag-setting forms must get carry/overflow from guest SP.
    for (unsigned wide = 0; wide != 2; ++wide) {
        for (unsigned subtract = 0; subtract != 2; ++subtract) {
            for (unsigned flags = 0; flags != 2; ++flags) {
                for (unsigned shift = 0; shift != 2; ++shift) {
                    for (unsigned immediate : {0u, 1u, 4095u}) {
                        const auto base = 0x11000000u | (wide << 31) | (subtract << 30) |
                            (flags << 29) | (shift << 22) | (immediate << 10) | (31u << 5);
                        if (!execute({3, base | 18u, base | 28u}, slot, context)) return false;
                    }
                }
            }
        }
    }
    return true;
}
} // namespace

int main() {
    const auto refusal_count = (reinterpret_cast<std::uintptr_t>(armigo_test_x18_refusals_end) -
                               reinterpret_cast<std::uintptr_t>(armigo_test_x18_refusals)) / sizeof(Refusal);
    for (std::size_t i = 0; i < refusal_count; ++i) {
        const auto form = armigo_test_x18_refusals[i];
        const auto plan = make_scalar_x18_patch(form.word, 0, 64, 0, contract);
        const auto expected = form.exclusive ? ScalarX18PatchError::exclusive_instruction
                                             : ScalarX18PatchError::stack_pointer_operand;
        if (!expect(aarch64::decode_register_effects(form.word).decoded() &&
                        !plan.ok() && plan.error == expected,
                    "A decoded form bypassed its unsupported stack or exclusive contract.")) return 1;
    }
    platform::ThreadSlotKey key;
    if (!expect(key.valid() && key.native_access().has_value(), "Native slot probe failed.")) return 1;
    const auto table = GuestSiteTable::build({});
    if (!expect(table.ok(), "Empty dispatch table was rejected.")) return 1;
    GuestThreadContext context;
    GuestDispatchScope dispatch(table.table, context);
    if (!expect(dispatch.setup_error() == GuestDispatchFailureKind::none,
                "Guest context binding failed.")) return 1;
    platform::ThreadSlotBinding binding(key, context.bound_register_cell());
    if (!expect(binding.native_access().has_value(), "Native cell publication failed.")) return 1;
    const auto slot = key.native_access()->tsd_byte_offset;
    const auto count = (reinterpret_cast<std::uintptr_t>(armigo_test_x18_forms_end) -
                        reinterpret_cast<std::uintptr_t>(armigo_test_x18_forms)) / sizeof(Form);
    for (std::size_t i = 0; i < count; ++i) {
        if (!execute(armigo_test_x18_forms[i], slot, context)) {
            std::fprintf(stderr, "Synthetic native form %zu failed.\n", i);
            return 1;
        }
    }
    if (!stack_arithmetic(slot, context)) return 1;
    const auto swaps = (reinterpret_cast<std::uintptr_t>(armigo_test_x18_swap_forms_end) -
                        reinterpret_cast<std::uintptr_t>(armigo_test_x18_swap_forms)) / sizeof(Form);
    for (std::size_t i = 0; i < swaps; ++i) {
        if (!execute_swap(armigo_test_x18_swap_forms[i], slot, context)) {
            std::fprintf(stderr, "Synthetic swap form %zu failed.\n", i);
            return 1;
        }
    }
    std::printf("%zu assembler forms, %zu compare-and-swap forms and 48 SP arithmetic forms matched native "
                "execution.\n", count, swaps);
    return 0;
}
