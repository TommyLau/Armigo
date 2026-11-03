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
#include <armigo/execution/conditional_x18_patch.h>
#include <armigo/execution/embedded_patch.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

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
void armigo_test_conditional_x18_run(const Image*, Image*, std::uint64_t);
extern const std::byte armigo_test_conditional_x18_begin[], armigo_test_conditional_x18_end[];
extern const std::byte armigo_test_conditional_x18_taken[];
extern const std::uint32_t armigo_test_conditional_x18_stack_words[2];
}

bool expect(bool value, const char* message) {
    if (!value) std::fprintf(stderr, "%s\n", message);
    return value;
}

std::size_t offset(const std::byte* label) {
    return reinterpret_cast<std::uintptr_t>(label) -
           reinterpret_cast<std::uintptr_t>(armigo_test_conditional_x18_begin);
}

bool refusals() {
    using E = ScalarX18PatchError;
    const auto word = aarch64::encode_cbz(false, true, 18, 0x100, 0x140).word;
    const auto reject = [](std::uint32_t word, std::uint64_t at, std::uint64_t stub,
                           std::uint32_t slot, ScalarX18Policy policy, E expected) {
        const auto result = make_conditional_x18_patch(word, at, stub, slot, policy);
        return expect(!result.ok() && result.error == expected, "Incorrect conditional x18 refusal.");
    };
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (!reject(word, 0x101, 0x200, 0, contract, E::invalid_layout) ||
        !reject(word, 0x100, 0x202, 0, contract, E::invalid_layout) ||
        !reject(word, maximum - 3, 0x200, 0, contract, E::invalid_layout) ||
        !reject(word, 0x100, maximum - 31, 0, contract, E::invalid_layout) ||
        !reject(word, 0x100, 0x104, 0, contract, E::overlapping_regions) ||
        !reject(word, 0x100, 0xe0, 0, contract, E::overlapping_regions) ||
        !reject(word, 0x100, 0x140, 0, contract, E::overlapping_regions) ||
        !reject(word, 0x100, 0x120, 0, contract, E::overlapping_regions) ||
        !reject(word, 0x100, 0x200, 1, contract, E::invalid_thread_slot) ||
        !reject(word, 0x100, 0x200, 32768, contract, E::invalid_thread_slot) ||
        !reject(word, 0x100, 0x200, 0, {}, E::missing_execution_contract) ||
        !reject(word, 0x100, 0x200, 0, {false, true, true}, E::missing_execution_contract) ||
        !reject(word, 0x100, 0x200, 0, {true, false, true}, E::missing_execution_contract) ||
        !reject(word, 0x100, 0x200, 0, {true, true, false}, E::missing_execution_contract) ||
        !reject(0xffffffff, 0x100, 0x200, 0, contract, E::unsupported_instruction) ||
        !reject(aarch64::encode_br(18).word, 0x100, 0x200, 0, contract, E::unsupported_instruction) ||
        !reject(aarch64::encode_b_cond(aarch64::Condition::eq, 0x100, 0x140).word,
                0x100, 0x200, 0, contract, E::unsupported_instruction) ||
        !reject(aarch64::encode_cbz(false, true, 0, 0x100, 0x140).word,
                0x100, 0x200, 0, contract, E::register_not_used) ||
        !reject(word, 0x100, 0x10000000, 0, contract, E::unreachable_branch) ||
        // The entry reaches, but the fallthrough branch cannot return.
        !reject(word, 0x10000000, 0x17fffffc, 0, contract, E::unreachable_branch) ||
        // The entry and fallthrough reach, but the original taken target does not.
        !reject(aarch64::encode_cbz(false, true, 18, 0x100, 0x200).word,
                0x10000000, 0x08000000, 0, contract, E::unreachable_branch) ||
        !reject(aarch64::encode_tbz(false, 18, 0, 0x100, 0xf0).word,
                0, 0x200, 0, contract, E::invalid_layout) ||
        !reject(word, maximum - 7, maximum - 0xff, 0, contract, E::invalid_layout)) return false;
    const auto boundary = aarch64::encode_tbz(false, 18, 63, 0x100, 0xc0).word;
    if (!expect(make_conditional_x18_patch(boundary, 0x10000000, 0x08000000,
                                          32760, contract).ok(),
                "Valid displacement and native slot boundaries were rejected.")) return false;
    return expect(make_conditional_x18_patch(boundary, maximum - 0xfff, maximum - 35,
                                             0, contract).ok(),
                  "A fully representable top-of-address-space trampoline was rejected.");
}

struct Form {
    bool compare;
    bool nonzero;
    bool wide;
    unsigned bit;
};

bool execute_form(Form form, std::uint32_t slot, GuestThreadContext& thread, bool embedded) {
    const auto source_size = offset(armigo_test_conditional_x18_end);
    const auto taken_offset = offset(armigo_test_conditional_x18_taken);
    const auto stub = (source_size + 3) & ~std::size_t{3};
    constexpr std::size_t key = 0x4000;
    std::vector<std::byte> bytes(embedded ? key + 0x4000 : stub + 36);
    std::memcpy(bytes.data(), armigo_test_conditional_x18_begin, source_size);
    const auto encoded = form.compare
        ? aarch64::encode_cbz(form.nonzero, form.wide, 18, 0, taken_offset)
        : aarch64::encode_tbz(form.nonzero, 18, form.bit, 0, taken_offset);
    if (!expect(encoded.ok(), "The authored conditional branch could not be encoded.")) return false;
    const auto plan = make_conditional_x18_patch(encoded.word, 0, stub, slot, contract);
    if (!expect(plan.ok(), "The authored conditional branch could not be prepared.")) return false;
    const auto& patch = *plan.patch;
    const auto embedded_plan = embedded ? make_embedded_conditional_x18_patch(
        encoded.word, 0, stub, key, contract) : std::optional<EmbeddedPatch>{};
    if (embedded && !expect(embedded_plan.has_value(), "Embedded conditional preparation failed.")) return false;
    const auto condition = aarch64::decode(patch.trampoline[4]);
    if (!expect(patch.expected_instruction == encoded.word &&
                patch.trampoline[0] == armigo_test_conditional_x18_stack_words[0] &&
                patch.trampoline[5] == armigo_test_conditional_x18_stack_words[1] &&
                patch.trampoline[7] == armigo_test_conditional_x18_stack_words[1] &&
                aarch64::referenced_address(aarch64::decode(patch.replacement_instruction), 0) == stub &&
                aarch64::referenced_address(condition, stub + 16) == stub + 28 &&
                aarch64::referenced_address(aarch64::decode(patch.trampoline[6]), stub + 24) == 4 &&
                aarch64::referenced_address(aarch64::decode(patch.trampoline[8]), stub + 32) == taken_offset,
                "Conditional branch patch has an incorrect destination or stack instruction.")) return false;
    for (const auto word : patch.trampoline) {
        if (!expect(aarch64::proven_free_of(aarch64::decode_register_effects(word), 18),
                    "Generated conditional code uses hardware x18 or has unknown effects.")) return false;
    }
    std::memcpy(bytes.data(), &patch.replacement_instruction, 4);
    if (embedded) {
        std::memcpy(bytes.data() + stub, embedded_plan->trampoline.data(),
                    embedded_plan->trampoline.size() * 4);
        const std::uint64_t key_value = slot;
        std::memcpy(bytes.data() + key, &key_value, 8);
    } else {
        std::memcpy(bytes.data() + stub, patch.trampoline.data(), 36);
    }
    std::vector<platform::ModuleRegion> regions{{0, embedded ? key : bytes.size(),
        memory::Permission::read | memory::Permission::execute}};
    if (embedded) regions.push_back({key, 0x4000, memory::Permission::read | memory::Permission::write});
    auto mapped = platform::HostModule::create(bytes, regions);
    if (!expect(mapped.ok() && mapped.module->seal().ok(), "Conditional fixture mapping failed."))
        return false;
    const auto entry = mapped.module->base_address();
    const unsigned bit = form.compare ? 0 : form.bit;
    const std::array values = {std::uint64_t{0}, ~std::uint64_t{0}, std::uint64_t{1} << bit,
        ~(std::uint64_t{1} << bit), std::uint64_t{1} << 32, std::uint64_t{1} << 63,
        std::uint64_t{1}, std::uint64_t{0xabcdef0000000000}};
    bool saw_taken = false, saw_fallthrough = false;
    for (std::size_t index = 0; index != values.size(); ++index) {
        const auto value = values[index];
        thread.set_guest_x18(value);
        Image input{}, output{};
        for (unsigned reg = 0; reg != 31; ++reg) input.x[reg] = 0x98760000ull + reg;
        input.nzcv = (index & 15) << 28;
        input.q0[0] = 0x1020304050607080; input.q0[1] = 0x90a0b0c0d0e0f001;
        input.q31[0] = 0x1122334455667788; input.q31[1] = 0x99aabbccddeeff00;
        armigo_test_conditional_x18_run(&input, &output, entry);
        const bool nonzero = form.compare
            ? (form.wide ? value != 0 : static_cast<std::uint32_t>(value) != 0)
            : ((value >> form.bit) & 1) != 0;
        const bool taken = nonzero == form.nonzero;
        saw_taken |= taken;
        saw_fallthrough |= !taken;
        for (unsigned reg = 0; reg != 30; ++reg) {
            if (reg == 18) continue;
            const auto expected = reg == 0 ? reinterpret_cast<std::uintptr_t>(&output) :
                                  reg == 17 ? entry : input.x[reg];
            if (!expect(output.x[reg] == expected, "A conditional trampoline clobbered a register."))
                return false;
        }
        if (!expect((output.marker != 0) == taken && output.x[30] == output.continuation &&
                    output.nzcv == input.nzcv && output.sp_before == output.sp_after &&
                    output.sp_before == output.sp_at_target && output.x18_before == output.x18_after &&
                    thread.guest_x18() == value &&
                    std::memcmp(output.q0, input.q0, sizeof(input.q0)) == 0 &&
                    std::memcmp(output.q31, input.q31, sizeof(input.q31)) == 0,
                    "Conditional result, link, flags, stack or virtual x18 differs.")) return false;
    }
    return expect(saw_taken && saw_fallthrough, "A conditional form did not exercise both outcomes.");
}
} // namespace

int main() {
    if (!refusals()) return 1;
    platform::ThreadSlotKey key;
    if (!expect(key.valid() && key.native_access().has_value(), "Native TSD probe failed.")) return 1;
    GuestThreadContext thread;
    const auto table = GuestSiteTable::build({});
    if (!expect(table.ok(), "Empty context-dispatch table was rejected.")) return 1;
    GuestDispatchScope dispatch(table.table, thread);
    if (!expect(dispatch.setup_error() == GuestDispatchFailureKind::none, "Guest context binding failed."))
        return 1;
    platform::ThreadSlotBinding binding(key, thread.bound_register_cell());
    if (!expect(binding.native_access().has_value(), "Guest register-cell publication failed.")) return 1;
    const auto slot = key.native_access()->tsd_byte_offset;
    for (const bool embedded : {false, true}) {
        for (const bool nonzero : {false, true}) {
            for (const bool wide : {false, true}) {
                if (!execute_form({true, nonzero, wide, 0}, slot, thread, embedded)) return 1;
            }
            for (const unsigned bit : {0u, 31u, 32u, 63u}) {
                if (!execute_form({false, nonzero, bit >= 32, bit}, slot, thread, embedded)) return 1;
            }
        }
    }
    std::puts("Twelve ordinary and embedded x18 conditional forms preserved native state across both outcomes.");
    return 0;
}
