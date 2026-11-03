// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/instruction.h>
#include <armigo/aarch64/register_substitution.h>
#include <armigo/execution/gate_code.h>
#include <armigo/execution/prepared_image.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace armigo;
using namespace armigo::execution;

constexpr std::uint64_t planning_base = 0x1'0000'0000ull;

bool check(bool value, const char* message) {
    if (!value)
        std::fprintf(stderr, "%s\n", message);
    return value;
}

ScalarX18Policy contract() {
    return {.exclusive_monitor_clear = true,
            .stack_scratch_available = true,
            .runtime_storage_disjoint = true};
}

std::vector<CoveredInstruction> coverage() {
    const auto call = aarch64::encode_bl(planning_base + 4, planning_base + 0x80);
    const auto conditional = aarch64::encode_cbz(false, true, aarch64::platform_register,
                                                 planning_base + 12, planning_base + 0x30);
    const auto address =
        aarch64::encode_adr(aarch64::platform_register, planning_base + 28, planning_base + 0x83);
    const auto page =
        aarch64::encode_adrp(aarch64::platform_register, planning_base + 32, planning_base);
    const auto nzcv = *aarch64::encoding_of(aarch64::SystemRegister::nzcv);
    const auto fpcr = *aarch64::encoding_of(aarch64::SystemRegister::fpcr);
    return {
        {0, 0xd503201fu, std::nullopt, {}},            // NOP, native-safe.
        {4, call.word, PreparedLocation{1, 0x40}, {}}, // Cross-module BL.
        {8, 0x91000652u, std::nullopt, {}},            // ADD X18,X18,#1.
        {12, conditional.word, std::nullopt, {}},      // CBZ X18.
        {16, aarch64::encode_br(18).word, std::nullopt, {{{1, 0x40}, {1, 0x40}}}}, // BR X18.
        {20, aarch64::encode_br(5).word, std::nullopt, {{{0, 0x30}, {0, 0x30}}}},  // Checked BR X5.
        {24, aarch64::encode_svc(0x42).word, std::nullopt, {}},                    // Gate SVC.
        {28, address.word, std::nullopt, {}},                      // ADR X18 to unaligned data.
        {32, page.word, std::nullopt, {}},                         // ADRP X18 to the module page.
        {36, aarch64::encode_mrs(0, nzcv).word, std::nullopt, {}}, // Native MRS NZCV.
        {40, aarch64::encode_msr(nzcv, 0).word, std::nullopt, {}}, // Native MSR NZCV.
        {44, aarch64::encode_mrs(1, fpcr).word, std::nullopt, {}}, // Native MRS FPCR.
        {0x30, 0xd503201fu, std::nullopt, {}},
    };
}

PreparedImageResult make_plan() {
    const auto covered = coverage();
    return prepare_image(0x100, covered,
                         {.module_index = 0, .module_count = 2, .scalar_x18_policy = contract()});
}

bool same_plan(const PreparedImage& left, const PreparedImage& right) {
    if (left.module_index != right.module_index ||
        left.original_image_size != right.original_image_size ||
        left.appendix_offset != right.appendix_offset || left.appendix != right.appendix ||
        left.coverage.size() != right.coverage.size() ||
        left.replacements.size() != right.replacements.size() ||
        left.gate_sites.size() != right.gate_sites.size() ||
        left.fixups.size() != right.fixups.size())
        return false;
    for (std::size_t i = 0; i < left.coverage.size(); ++i) {
        if (std::memcmp(&left.coverage[i], &right.coverage[i], sizeof(PreparedWordPatch)) != 0)
            return false;
    }
    for (std::size_t i = 0; i < left.replacements.size(); ++i) {
        if (std::memcmp(&left.replacements[i], &right.replacements[i], sizeof(PreparedWordPatch)) !=
            0)
            return false;
    }
    for (std::size_t i = 0; i < left.fixups.size(); ++i) {
        const auto& a = left.fixups[i];
        const auto& b = right.fixups[i];
        if (a.kind != b.kind || a.image_offset != b.image_offset ||
            a.target.module_index != b.target.module_index ||
            a.target.image_offset != b.target.image_offset)
            return false;
    }
    for (std::size_t i = 0; i < left.gate_sites.size(); ++i) {
        const auto& a = left.gate_sites[i];
        const auto& b = right.gate_sites[i];
        if (a.return_address_offset != b.return_address_offset || a.instruction != b.instruction ||
            a.allowed_indirect_targets.size() != b.allowed_indirect_targets.size())
            return false;
        for (std::size_t j = 0; j < a.allowed_indirect_targets.size(); ++j) {
            if (a.allowed_indirect_targets[j].module_index !=
                    b.allowed_indirect_targets[j].module_index ||
                a.allowed_indirect_targets[j].image_offset !=
                    b.allowed_indirect_targets[j].image_offset)
                return false;
        }
    }
    return true;
}

std::vector<std::byte> source_image(const PreparedImage& prepared) {
    std::vector<std::byte> image(prepared.image_size());
    for (const auto& site : prepared.coverage) {
        std::memcpy(image.data() + site.image_offset, &site.expected_word,
                    sizeof(site.expected_word));
    }
    return image;
}

bool deterministic_plan_covers_required_forms() {
    const auto first = make_plan();
    const auto second = make_plan();
    if (!check(first.ok() && second.ok(), "A valid covered image was not prepared."))
        return false;
    const auto& prepared = *first.image;
    if (!check(same_plan(prepared, *second.image),
               "Identical covered input did not produce identical output."))
        return false;
    if (!check(prepared.appendix_offset == 0x4000 && !prepared.appendix.empty(),
               "The appendix did not start at the aligned original-image end."))
        return false;
    if (!check(prepared.coverage.size() == 13 && prepared.replacements.size() == 8,
               "Native-safe coverage or prepared replacement counts are wrong."))
        return false;
    if (!check(prepared.gate_sites.size() == 2,
               "System and non-x18 indirect gate sites were not retained."))
        return false;
    if (!check(prepared.replacements.front().image_offset == 4,
               "The native-safe word was unexpectedly replaced."))
        return false;

    std::array<unsigned, 4> fixup_counts{};
    for (const auto& fixup : prepared.fixups) {
        ++fixup_counts[static_cast<unsigned>(fixup.kind)];
    }
    return check(
        fixup_counts[static_cast<unsigned>(PreparedFixupKind::thread_slot_load)] == 5 &&
            fixup_counts[static_cast<unsigned>(PreparedFixupKind::gate_dispatcher_address)] == 1 &&
            fixup_counts[static_cast<unsigned>(PreparedFixupKind::module_address_mov_wide)] == 2 &&
            fixup_counts[static_cast<unsigned>(PreparedFixupKind::branch26_module_target)] == 2,
        "The plan did not retain all required typed fixups.");
}

bool applies_only_after_complete_validation() {
    const auto result = make_plan();
    if (!result.ok())
        return false;
    const auto& prepared = *result.image;
    constexpr std::array bases = {0x2'0000'0000ull, 0x2'0400'0000ull};
    const PreparedImageBindings bindings{bases, 64, 0x1234'5678ull};

    auto image = source_image(prepared);
    const auto applied = apply_prepared_image(prepared, image, bindings);
    if (!check(applied.ok(), "A valid prepared image was not applied."))
        return false;
    std::uint32_t native_word = 0;
    std::memcpy(&native_word, image.data(), sizeof(native_word));
    if (!check(native_word == 0xd503201fu,
               "A native-safe covered instruction changed during application."))
        return false;
    std::uint32_t cross_call = 0;
    std::memcpy(&cross_call, image.data() + 4, sizeof(cross_call));
    const auto destination = aarch64::referenced_address(aarch64::decode(cross_call), bases[0] + 4);
    if (!check(destination && *destination == bases[1] + 0x40,
               "The cross-module branch fixup did not use actual module bases."))
        return false;
    bool found_adrp = false;
    for (std::uint64_t offset = prepared.appendix_offset; offset + 4 <= prepared.image_size();
         offset += 4) {
        std::uint32_t word = 0;
        std::memcpy(&word, image.data() + offset, sizeof(word));
        const auto decoded = aarch64::decode(word);
        if (decoded.opcode != aarch64::Opcode::adrp)
            continue;
        const auto target = aarch64::referenced_address(decoded, bases[0] + offset);
        if (!check(target && *target == bases[0],
                   "A relocated ADRP trampoline did not retain its module page target."))
            return false;
        found_adrp = true;
    }
    if (!check(found_adrp, "The prepared image did not retain its relocated ADRP."))
        return false;
    std::vector<std::uint64_t> materialized_addresses;
    for (const auto& fixup : prepared.fixups) {
        const auto* at = image.data() + fixup.image_offset;
        if (fixup.kind == PreparedFixupKind::thread_slot_load) {
            std::uint32_t word = 0;
            std::memcpy(&word, at, sizeof(word));
            if (!check(((word >> 10) & 0xfffu) == 8,
                       "A thread-slot fixup did not encode the shared byte offset."))
                return false;
        } else if (fixup.kind == PreparedFixupKind::gate_dispatcher_address) {
            std::uintptr_t pointer = 0;
            std::memcpy(&pointer, at, sizeof(pointer));
            if (!check(pointer == bindings.gate_dispatcher_address,
                       "The gate dispatcher fixup did not bind the supplied address."))
                return false;
        } else if (fixup.kind == PreparedFixupKind::module_address_mov_wide) {
            std::uint64_t value = 0;
            for (unsigned half = 0; half != 4; ++half) {
                std::uint32_t word = 0;
                std::memcpy(&word, at + half * 4, sizeof(word));
                value |= static_cast<std::uint64_t>((word >> 5) & 0xffffu) << (half * 16);
            }
            materialized_addresses.push_back(value);
        } else if (fixup.kind == PreparedFixupKind::branch26_module_target) {
            std::uint32_t word = 0;
            std::memcpy(&word, at, sizeof(word));
            const auto target =
                aarch64::referenced_address(aarch64::decode(word), bases[0] + fixup.image_offset);
            if (!check(target &&
                           *target == bases[fixup.target.module_index] + fixup.target.image_offset,
                       "A branch26 fixup did not reach its module-relative target."))
                return false;
        }
    }

    // These destinations come from the authored BR X18 target and ADR at
    // offset 28, independently of the planner's generated fixup records.
    std::ranges::sort(materialized_addresses);
    if (!check(materialized_addresses ==
                   std::vector<std::uint64_t>{bases[0] + 0x83, bases[1] + 0x40},
               "MOV-wide addresses differ from the independently authored destinations."))
        return false;

    auto stale = source_image(prepared);
    stale[0] = std::byte{0xff};
    const auto before_stale = stale;
    const auto stale_result = apply_prepared_image(prepared, stale, bindings);
    if (!check(stale_result.error == PreparedImageError::preimage_mismatch && stale == before_stale,
               "A stale preimage caused partial mutation."))
        return false;

    auto malformed_plan = prepared;
    malformed_plan.fixups.front().image_offset += 2;
    auto malformed = source_image(prepared);
    const auto before_malformed = malformed;
    const auto malformed_result = apply_prepared_image(malformed_plan, malformed, bindings);
    if (!check(malformed_result.error == PreparedImageError::invalid_fixup &&
                   malformed == before_malformed,
               "A malformed fixup caused partial mutation."))
        return false;

    auto unknown_plan = prepared;
    unknown_plan.fixups.front().kind = static_cast<PreparedFixupKind>(0xff);
    auto unknown = source_image(prepared);
    const auto before_unknown = unknown;
    const auto unknown_result = apply_prepared_image(unknown_plan, unknown, bindings);
    if (!check(unknown_result.error == PreparedImageError::invalid_fixup &&
                   unknown == before_unknown,
               "An unknown fixup kind caused partial mutation."))
        return false;

    auto misaligned = source_image(prepared);
    const auto before_misaligned = misaligned;
    constexpr std::array misaligned_bases = {0x2'0000'1000ull, 0x2'0400'0000ull};
    const auto misaligned_result =
        apply_prepared_image(prepared, misaligned, {misaligned_bases, 64, 0x1234'5678ull});
    if (!check(misaligned_result.error == PreparedImageError::invalid_binding &&
                   misaligned == before_misaligned,
               "A non-16-KiB module base caused partial mutation."))
        return false;

    auto overflowing = source_image(prepared);
    const auto before_overflowing = overflowing;
    constexpr std::array overflowing_bases = {0x2'0000'0000ull, std::uint64_t(-1) - 0x10};
    const auto overflowing_result =
        apply_prepared_image(prepared, overflowing, {overflowing_bases, 64, 0x1234'5678ull});
    if (!check(overflowing_result.error == PreparedImageError::invalid_binding &&
                   overflowing == before_overflowing,
               "An overflowing module-relative target caused partial mutation."))
        return false;

    auto unreachable = source_image(prepared);
    const auto before_unreachable = unreachable;
    constexpr std::array distant_bases = {0x2'0000'0000ull, 0x4'0000'0000ull};
    const auto unreachable_result =
        apply_prepared_image(prepared, unreachable, {distant_bases, 64, 0x1234'5678ull});
    return check(unreachable_result.error == PreparedImageError::unreachable_target &&
                     unreachable == before_unreachable,
                 "An unreachable cross-module branch caused partial mutation.");
}

bool supports_native_only_images_and_unbound_gate_templates() {
    const std::array native = {CoveredInstruction{0, 0xd503201fu, std::nullopt, {}}};
    const auto plan = prepare_image(
        4, native, {.module_index = 0, .module_count = 1, .scalar_x18_policy = contract()});
    if (!check(plan.ok() && plan.image->appendix.empty() && plan.image->image_size() == 4 &&
                   plan.image->replacements.empty(),
               "A native-only covered image gained injected storage."))
        return false;
    auto image = source_image(*plan.image);
    constexpr std::array bases = {0x3'0000'0000ull};
    if (!check(apply_prepared_image(*plan.image, image, {bases, 0, 0}).ok(),
               "A native-only image unnecessarily required a gate binding."))
        return false;

    auto gate = make_local_gate_template();
    std::uintptr_t binding = 1;
    std::memcpy(&binding, gate.code.data() + gate.dispatcher_offset, sizeof(binding));
    if (!check(binding == 0, "The persistent gate template contains a host pointer."))
        return false;
    const auto original = gate.code;
    if (!check(!bind_local_gate_dispatcher(gate.code, gate.dispatcher_offset + 1, 0x1234) &&
                   gate.code == original,
               "An invalid gate binding partially changed the template."))
        return false;
    return check(bind_local_gate_dispatcher(gate.code, gate.dispatcher_offset, 0x1234),
                 "A valid explicit gate dispatcher binding was rejected.");
}

bool rejects_invalid_inputs() {
    const auto covered = coverage();
    if (!check(prepare_image(0x100, covered, {.module_index = 0, .module_count = 2}).error ==
                   PreparedImageError::missing_execution_contract,
               "Default-false execution policy was accepted."))
        return false;

    auto unordered = covered;
    std::swap(unordered[0], unordered[1]);
    if (!check(
            prepare_image(0x100, unordered,
                          {.module_index = 0, .module_count = 2, .scalar_x18_policy = contract()})
                    .error == PreparedImageError::invalid_coverage,
            "Unordered covered words were accepted."))
        return false;

    const std::array exclusive = {
        CoveredInstruction{0, 0xc85f7c12u, std::nullopt, {}}}; // LDXR X18,[X0].
    if (!check(
            prepare_image(0x100, exclusive,
                          {.module_index = 0, .module_count = 1, .scalar_x18_policy = contract()})
                    .error == PreparedImageError::exclusive_window,
            "An unclosed x18 exclusive instruction was admitted."))
        return false;

    const std::array trap = {
        CoveredInstruction{0, aarch64::encode_brk(0x42).word, std::nullopt, {}}};
    return check(
        prepare_image(0x100, trap,
                      {.module_index = 0, .module_count = 1, .scalar_x18_policy = contract()})
                .error == PreparedImageError::unsupported_instruction,
        "A source BRK was admitted without a native fault boundary.");
}

std::vector<CoveredInstruction> exclusive_coverage(std::initializer_list<std::uint32_t> words) {
    std::vector<CoveredInstruction> covered;
    for (const auto word : words) covered.push_back({covered.size() * 4, word, {}, {}});
    return covered;
}

PreparedImageResult plan_exclusives(const std::vector<CoveredInstruction>& words,
                                  ExclusiveWindowLimits limits = {}) {
    return prepare_image(0x1000, words,
        {.module_index = 0, .module_count = 1,
         .scalar_x18_policy = {false, true, true}, .exclusive_limits = limits});
}

bool exclusive_window_contract() {
    constexpr std::uint32_t load = 0xc85ffc20u; // LDAXR X0,[X1].
    constexpr std::uint32_t store = 0xc803fc20u; // STLXR W3,X0,[X1].
    constexpr std::uint32_t clear = 0xd5033f5fu; // CLREX.
    constexpr std::uint32_t rewrite = 0x91000652u; // ADD X18,X18,#1.
    constexpr std::uint32_t nop = 0xd503201fu;
    const auto ret = aarch64::encode_ret().word;
    bool ok = true;

    // All single widths, pair widths, and acquire/release choices stay native.
    for (unsigned pair = 0; pair < 2; ++pair) {
        for (unsigned size = pair ? 2 : 0; size < 4; ++size) {
            for (unsigned ordered = 0; ordered < 2; ++ordered) {
                const auto common = 0x08000000u | (size << 30) | (pair << 21) |
                    (ordered << 15) | ((pair ? 2u : 31u) << 10) | (1u << 5);
                const auto words = exclusive_coverage({common | (1u << 22) | (31u << 16),
                                                       common | (3u << 16), ret});
                const auto result = plan_exclusives(words);
                ok &= check(result.ok() && result.image->replacements.empty() &&
                    result.image->appendix.empty() && result.exclusive_proof.windows_found == 1 &&
                    result.exclusive_proof.windows_proved == 1 &&
                    result.exclusive_proof.monitor_clear == std::vector<bool>{false, false, true},
                    "A native exclusive family changed or its window was not proved.");
            }
        }
    }
    auto words = exclusive_coverage({rewrite, load, clear, rewrite, ret});
    auto result = plan_exclusives(words);
    ok &= check(result.ok() && result.image->replacements.size() == 2 &&
        result.exclusive_proof.monitor_clear == std::vector<bool>{true, false, false, true, true},
        "Rewrites outside a CLREX-terminated window need a per-site proof, not a blanket flag.");
    result = plan_exclusives(exclusive_coverage({store, clear, ret}));
    ok &= check(result.ok() && result.image->replacements.empty(),
                "An unpaired native store-exclusive/CLREX must retain architectural behavior.");

    const auto refused = [&](const std::vector<CoveredInstruction>& sites,
                             ExclusiveWindowError reason, std::uint64_t at,
                             ExclusiveWindowLimits limits = {}) {
        const auto rejected = plan_exclusives(sites, limits);
        return check(!rejected.ok() && !rejected.exclusive_proof.diagnostics.empty() &&
            rejected.exclusive_proof.diagnostics.front().reason == reason &&
            rejected.exclusive_proof.diagnostics.front().window_offset == 0 &&
            rejected.exclusive_proof.diagnostics.front().instruction_offset == at,
            "An unsafe exclusive window did not identify its origin, site and refusal reason.");
    };
    const auto tls = aarch64::encode_mrs(0, *aarch64::encoding_of(aarch64::SystemRegister::tpidr_el0)).word;
    for (const auto unsafe : {aarch64::encode_svc(1).word, tls, aarch64::encode_br(5).word})
        ok &= refused(exclusive_coverage({load, unsafe, store, ret}),
                      ExclusiveWindowError::prepared_rewrite, 4);

    // Without an entry patch slot, naming x18 still fails closed.
    for (const auto x18_load : {0xc85f7c32u, 0xc85f7e40u, 0xc87f4820u})
        ok &= refused(exclusive_coverage({x18_load, clear, ret}), ExclusiveWindowError::entry_transfer, 0);
    for (const auto x18_store : {0xc8037c32u, 0xc8037e40u, 0xc8127c20u, 0xc8234820u})
        ok &= refused(exclusive_coverage({load, x18_store, ret}), ExclusiveWindowError::entry_transfer, 0);

    // A diamond terminates at either store-exclusive or CLREX. Nested loads
    // begin their own windows, without prematurely ending the outer one.
    words = exclusive_coverage({load, aarch64::encode_cbz(false, true, 4, 4, 20).word,
        load, store, aarch64::encode_b(16, 24).word, clear, rewrite, ret});
    result = plan_exclusives(words);
    ok &= check(result.ok() && result.exclusive_proof.windows_found == 2 &&
        result.exclusive_proof.windows_proved == 2 && result.image->replacements.size() == 1,
        "Branching/nested windows must close on every explored path.");
    words[5].word = rewrite;
    ok &= refused(words, ExclusiveWindowError::unclosed_exit, 28);

    // A backward edge can bring an earlier rewrite inside a later window.
    words = exclusive_coverage({rewrite, load, aarch64::encode_b(8, 0).word});
    result = plan_exclusives(words);
    ok &= check(!result.ok() && result.exclusive_proof.diagnostics.front().window_offset == 4 &&
        result.exclusive_proof.diagnostics.front().reason == ExclusiveWindowError::entry_transfer,
        "A live backedge cannot host an entry transfer.");
    words = exclusive_coverage({load, aarch64::encode_cbz(false, true, 4, 4, 4).word, clear, ret});
    ok &= check(plan_exclusives(words).ok(), "A finite CFG with a native backedge must converge.");

    ok &= refused(exclusive_coverage({load, ret}), ExclusiveWindowError::unclosed_exit, 4);
    ok &= refused(exclusive_coverage({load, aarch64::encode_bl(4, 12).word, ret, clear, ret}),
                  ExclusiveWindowError::unclosed_exit, 4);
    ok &= refused(exclusive_coverage({load, aarch64::encode_b(4, 0x80).word}),
                  ExclusiveWindowError::uncovered_successor, 0x80);
    words = exclusive_coverage({load, aarch64::encode_b(4, 8).word, clear, ret});
    const std::array<std::uint64_t, 1> entries{8};
    const auto boundary = prove_exclusive_windows(words, {}, entries, 0);
    ok &= check(!boundary.ok() && boundary.diagnostics.front().reason == ExclusiveWindowError::function_boundary,
                "An active transfer into a distinct known function must fail closed.");
    const auto planned_boundary = prepare_image(0x1000, words,
        {.module_index = 0, .module_count = 1, .scalar_x18_policy = contract(), .function_entries = entries});
    ok &= check(!planned_boundary.ok() && planned_boundary.exclusive_proof.diagnostics.front().reason ==
        ExclusiveWindowError::function_boundary, "Standalone preparation must honor supplied function entries.");
    words[1].cross_module_target = PreparedLocation{1, 0};
    ok &= refused(words, ExclusiveWindowError::function_boundary, 4);

    // Numeric offsets in a remote module cannot split this module's window.
    // Test both call and tail-transfer maps, including a nonzero current module.
    for (const auto indirect : {aarch64::encode_blr(9).word, aarch64::encode_br(9).word}) {
        for (const std::uint32_t current_module : {0u, 1u}) {
            words = exclusive_coverage({load, nop, clear, indirect, ret});
            const auto remote_module = 1u - current_module;
            words[3].indirect_targets = {{{remote_module, 4}, {remote_module, 4}}};
            auto remote = prepare_image(0x1000, words,
                {.module_index = current_module, .module_count = 2,
                 .scalar_x18_policy = contract()});
            ok &= check(remote.ok() && remote.exclusive_proof.windows_proved == 1,
                        "A remote indirect target offset became a local function boundary.");
            words[3].indirect_targets[0].guest.module_index = current_module;
            const auto aliased = prove_exclusive_windows(words, {}, {}, current_module);
            ok &= check(aliased.ok(), "Boundary ownership followed the guest alias instead of the prepared target.");
            words[3].indirect_targets = {{{current_module, 4}, {current_module, 4}}};
            auto local = prepare_image(0x1000, words,
                {.module_index = current_module, .module_count = 2,
                 .scalar_x18_policy = contract()});
            ok &= check(!local.ok() && !local.exclusive_proof.diagnostics.empty() &&
                local.exclusive_proof.diagnostics.front().reason == ExclusiveWindowError::function_boundary &&
                local.exclusive_proof.diagnostics.front().instruction_offset == 4,
                "A genuine local indirect target no longer establishes a function boundary.");
        }
    }

    words = exclusive_coverage({load, nop, clear, ret});
    ExclusiveWindowLimits limits;
    limits.max_words_per_window = 2;
    ok &= refused(words, ExclusiveWindowError::word_limit, 8, limits);
    limits.max_words_per_window = 3;
    ok &= check(plan_exclusives(words, limits).ok(), "The inclusive word bound rejected an exact fit.");
    limits = {};
    limits.max_total_steps = 1;
    ok &= refused(words, ExclusiveWindowError::step_limit, 4, limits);
    limits = {};
    limits.max_windows = 0;
    ok &= refused(words, ExclusiveWindowError::window_limit, 0, limits);
    const auto unknown = prove_exclusive_windows(exclusive_coverage({0xffffffffu, clear, ret}), {}, {}, 0);
    ok &= check(!unknown.ok() && unknown.diagnostics.front().reason == ExclusiveWindowError::unsupported_instruction,
                "Unknown effects must not establish monitor clearance even outside a known window.");
    // Rs is reserved in a load-exclusive, so this resembles LDXR but is invalid.
    ok &= check(plan_exclusives(exclusive_coverage({load & ~(31u << 16), clear, ret})).error ==
        PreparedImageError::unsupported_instruction, "An invalid exclusive encoding must never execute natively.");
    return ok;
}

bool exclusive_x18_windows() {
    constexpr std::uint32_t nop = 0xd503201fu;
    constexpr std::uint32_t clear = 0xd5033f5fu;
    constexpr std::uint32_t load = 0xc85ffc20u; // LDAXR X0,[X1].
    constexpr std::uint32_t kill = 0xd2800009u; // MOV X9,#0.
    constexpr std::uint32_t ret = 0xd65f03c0u;
    bool ok = true;
    const auto verify = [&](const std::vector<CoveredInstruction>& words, std::size_t windows) {
        const auto result = plan_exclusives(words);
        if (!check(result.ok() && result.exclusive_proof.windows_proved == windows &&
                   result.exclusive_proof.x18_regions.size() == 1,
                   "A bounded x18 window with dead scratch and safe boundaries was refused.")) return false;
        const auto& region = result.exclusive_proof.x18_regions.front();
        bool valid = region.scratch_register == 9;
        for (const auto i : region.instructions) {
            const auto& site = words[i];
            const auto renamed = aarch64::substitute_register(site.word, 18, 9, aarch64::UnchangedWord::accept);
            const auto patch = std::find_if(result.image->replacements.begin(), result.image->replacements.end(),
                [&](const auto& item) { return item.image_offset == site.image_offset; });
            const auto actual = patch == result.image->replacements.end() ? site.word : patch->replacement_word;
            const auto fields = aarch64::register_fields(aarch64::decode_register_effects(site.word), 18);
            valid &= renamed.ok() && actual == renamed.word && fields && ((actual ^ site.word) & ~*fields) == 0;
        }
        for (std::size_t at = 0; at < result.image->appendix.size(); at += 4) {
            std::uint32_t word;
            std::memcpy(&word, result.image->appendix.data() + at, 4);
            valid &= aarch64::proven_free_of(aarch64::decode_register_effects(word), 18);
        }
        return check(valid, "Window patches changed more than x18 fields or exposed physical x18.");
    };
    // Every explicit exclusive operand role, including second pair transfers.
    for (const auto x18_load : {0xc85f7c32u, 0xc85f7e40u, 0xc87f4820u})
        ok &= verify(exclusive_coverage({nop, x18_load, clear, kill, ret}), 1);
    for (const auto x18_store : {0xc8037c32u, 0xc8037e40u, 0xc8127c20u, 0xc8234820u})
        ok &= verify(exclusive_coverage({nop, load, x18_store, kill, ret}), 1);

    // A conditional x18 use and a nested load share one register on both exits.
    auto words = exclusive_coverage({nop, 0xc85ffc32u,
        aarch64::encode_cbz(false, true, 18, 8, 32).word, load, 0x91000652u,
        0xc803fc32u, kill, aarch64::encode_b(28, 40).word, clear, kill, ret});
    ok &= verify(words, 2);
    auto result = plan_exclusives(words);
    ok &= check(result.ok() && result.exclusive_proof.x18_regions.front().entries == std::vector<std::size_t>{1} &&
                result.exclusive_proof.x18_regions.front().exits == std::vector<std::size_t>{6, 9},
                "Nested windows inserted a transfer at a live nested load or missed a branch exit.");

    const auto refused = [&](const std::vector<CoveredInstruction>& sites, ExclusiveWindowError reason) {
        const auto result = plan_exclusives(sites);
        return check(!result.ok() && !result.image && !result.exclusive_proof.diagnostics.empty() &&
                     result.exclusive_proof.diagnostics.front().reason == reason,
                     "An unsafe x18 window did not produce its structured refusal.");
    };
    // An exit branch can also be another region's incoming branch. Publish
    // the first region before loading the second, without overwriting either
    // wrapper during branch retargeting.
    words = exclusive_coverage({nop, 0xc85ffc32u, clear, aarch64::encode_b(12, 24).word,
                                ret, nop, 0xc85ffc32u, clear, kill, ret});
    result = plan_exclusives(words);
    if (check(result.ok() && result.exclusive_proof.x18_regions.size() == 2,
              "Two disjoint x18 regions did not compose through an exit branch.")) {
        const auto patched = [&](std::uint64_t offset) {
            const auto it = std::find_if(result.image->replacements.begin(), result.image->replacements.end(),
                [&](const auto& patch) { return patch.image_offset == offset; });
            return it == result.image->replacements.end() ? 0u : it->replacement_word;
        };
        const auto appended = [&](std::uint64_t offset) {
            std::uint32_t word = 0;
            if (offset >= result.image->appendix_offset && offset + 4 <= result.image->image_size())
                std::memcpy(&word, result.image->appendix.data() + offset - result.image->appendix_offset, 4);
            return word;
        };
        const auto first_exit = aarch64::referenced_address(aarch64::decode(patched(12)), 12);
        const auto second_entry = aarch64::referenced_address(aarch64::decode(patched(20)), 20);
        const auto transfer = make_scalar_x18_patch(0xaa0903f2u, 12, 0x10000, 0, contract());
        bool composed = first_exit && second_entry && transfer.ok();
        if (composed) {
            const auto body = *first_exit + transfer.patch->trampoline_word_count * 4;
            const auto target = aarch64::referenced_address(aarch64::decode(appended(body)), body);
            composed = target == *second_entry + 4;
        }
        ok &= check(composed, "An entry redirect overwrote an earlier region's exit transfer.");
    } else ok = false;

    // All otherwise available registers are read after closure, before any kill.
    words = exclusive_coverage({nop, 0xc85ffc32u, clear});
    for (unsigned reg = 0; reg < 30; ++reg) {
        if (reg != 18) words.push_back({words.size() * 4, 0xaa0003ffu | (reg << 16), {}, {}});
    }
    words.push_back({words.size() * 4, ret, {}, {}});
    ok &= refused(words, ExclusiveWindowError::no_scratch_register);
    ok &= refused(exclusive_coverage({nop, 0xc85ffc32u, clear,
        aarch64::encode_bl(12, 24).word, ret, ret, ret}), ExclusiveWindowError::no_scratch_register);
    ok &= refused(exclusive_coverage({nop, 0xc85ffc32u, ret}), ExclusiveWindowError::unclosed_exit);
    ok &= refused(exclusive_coverage({0xc85ffc32u, clear, kill, ret}), ExclusiveWindowError::entry_transfer);
    ok &= refused(exclusive_coverage({0x91000652u, 0xc85ffc32u, clear, kill, ret}),
                  ExclusiveWindowError::entry_transfer);
    // A second root can reach the exit without ever loading the scratch value.
    words = exclusive_coverage({nop, 0xc85ffc32u, clear, kill, ret, aarch64::encode_b(20, 12).word});
    ok &= refused(words, ExclusiveWindowError::exit_transfer);
    // An outside branch into a non-load interior cannot establish the mapping.
    words = exclusive_coverage({nop, 0xc85ffc32u, nop, clear, kill, ret,
                                aarch64::encode_b(24, 8).word});
    ok &= refused(words, ExclusiveWindowError::entry_transfer);
    // A call or system gate continuation is an incoming edge even when a
    // native branch already reaches that interior word from the opening load.
    for (const auto boundary : {aarch64::encode_bl(12, 32).word, aarch64::encode_svc(1).word}) {
        words = exclusive_coverage({nop, 0xc85ffc32u, aarch64::encode_b(8, 16).word,
                                    boundary, 0x91000652u, clear, kill, ret, ret});
        ok &= refused(words, ExclusiveWindowError::entry_transfer);
    }
    auto limits = ExclusiveWindowLimits{};
    limits.max_total_steps = 2;
    words = exclusive_coverage({nop, 0xc85ffc32u, clear, kill, ret});
    result = plan_exclusives(words, limits);
    ok &= check(!result.ok() && result.exclusive_proof.steps == 2 &&
                result.exclusive_proof.diagnostics.front().reason == ExclusiveWindowError::step_limit,
                "Scratch liveness exceeded the shared proof budget.");
    // Killing a scratch on only one conditional exit path is insufficient.
    words = exclusive_coverage({nop, 0xc85ffc32u, clear,
        aarch64::encode_cbz(false, true, 2, 12, 24).word, kill, ret, 0xaa0903ffu, ret});
    ok &= refused(words, ExclusiveWindowError::no_scratch_register);
    // Short external retry branches use a register-free stub when the image
    // appendix is beyond their immediate range. Check both destinations.
    for (const auto retry : {aarch64::encode_cbz(true, false, 3, 16, 4).word,
                            aarch64::encode_tbz(true, 3, 0, 16, 4).word,
                            aarch64::encode_b_cond(aarch64::Condition::ne, 16, 4).word}) {
        words = exclusive_coverage({nop, 0xc85ffc32u, clear, kill, retry, ret});
        result = prepare_image(0x200000, words, {.module_index = 0, .module_count = 1,
            .scalar_x18_policy = contract()});
        if (!check(result.ok(), "An otherwise valid short retry branch could not reach its entry transfer.")) {
            ok = false;
            continue;
        }
        const auto patched = [&](std::uint64_t offset) {
            const auto it = std::find_if(result.image->replacements.begin(), result.image->replacements.end(),
                [&](const auto& patch) { return patch.image_offset == offset; });
            return it == result.image->replacements.end() ? 0u : it->replacement_word;
        };
        const auto stub = aarch64::referenced_address(aarch64::decode(patched(16)), 16);
        const auto entry = aarch64::referenced_address(aarch64::decode(patched(0)), 0);
        bool valid = stub && entry && *stub >= result.image->appendix_offset && *stub + 12 <= result.image->image_size();
        if (valid) {
            std::array<std::uint32_t, 3> emitted;
            std::memcpy(emitted.data(), result.image->appendix.data() + *stub - result.image->appendix_offset, 12);
            const auto original = aarch64::decode(retry);
            const auto condition = aarch64::decode(emitted[0]);
            valid = condition.opcode == original.opcode && condition.reg == original.reg &&
                condition.condition == original.condition && condition.test_bit == original.test_bit &&
                condition.wide == original.wide && aarch64::referenced_address(condition, *stub) == *stub + 8 &&
                aarch64::referenced_address(aarch64::decode(emitted[1]), *stub + 4) == 20 &&
                aarch64::referenced_address(aarch64::decode(emitted[2]), *stub + 8) == *entry + 4;
        }
        ok &= check(valid, "A far retry stub changed its condition or either branch destination.");
    }
    // Root metadata must never bypass an entry transfer at a load address.
    words = exclusive_coverage({nop, 0xc85ffc32u, clear, kill, ret});
    const std::array<std::uint64_t, 1> roots{4};
    result = prepare_image(0x1000, words, {.module_index = 0, .module_count = 1,
        .scalar_x18_policy = contract(), .function_entries = roots});
    ok &= check(!result.ok() && result.exclusive_proof.diagnostics.front().reason == ExclusiveWindowError::entry_transfer,
                "A function entry bypassed its x18 window transfer.");
    result = prepare_image(0x10000000, words, {.module_index = 0, .module_count = 1,
        .scalar_x18_policy = contract()});
    ok &= check(!result.ok() && result.exclusive_proof.diagnostics.front().reason == ExclusiveWindowError::entry_transfer,
                "An out-of-range boundary transfer did not fail structurally.");
    return ok;
}

} // namespace

int main() {
    if (!deterministic_plan_covers_required_forms() || !applies_only_after_complete_validation() ||
        !supports_native_only_images_and_unbound_gate_templates() || !rejects_invalid_inputs() ||
        !exclusive_window_contract() || !exclusive_x18_windows())
        return 1;
    std::puts("Deterministic prepared-image planning and typed load fixups passed.");
    return 0;
}
