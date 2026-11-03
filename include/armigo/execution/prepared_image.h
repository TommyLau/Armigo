// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/scalar_x18_patch.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace armigo::execution {

// A location in the ordered prepared-module manifest. The type deliberately
// lives outside the runtime namespace so offline tools need no launch objects.
struct PreparedLocation {
    std::uint32_t module_index = 0;
    std::uint64_t image_offset = 0;
};

struct PreparedIndirectTarget {
    // Value held by the guest target register and native prepared destination.
    PreparedLocation guest;
    PreparedLocation prepared;
};

// One instruction already selected and validated as covered by the caller.
// Instructions must be ordered by image_offset with no duplicates. The
// preparer copies all records and retains no reference to the input span.
struct CoveredInstruction {
    std::uint64_t image_offset = 0;
    std::uint32_t word = 0;
    // Required only for a direct B/BL whose target is in another module.
    std::optional<PreparedLocation> cross_module_target;
    // Required only for BR/BLR and interpreted as a complete finite map.
    std::vector<PreparedIndirectTarget> indirect_targets;
};

enum class ExclusiveWindowError : std::uint8_t {
    none,
    invalid_coverage,
    x18_operand,
    prepared_rewrite,
    unclosed_exit,
    function_boundary,
    uncovered_successor,
    unsupported_instruction,
    window_limit,
    word_limit,
    step_limit,
    no_scratch_register,
    entry_transfer,
    exit_transfer,
    substitution_failed,
};

struct ExclusiveWindowLimits {
    std::size_t max_windows = 65'536;
    std::size_t max_words_per_window = 4096;
    std::size_t max_total_steps = 4'000'000;
};

struct ExclusiveWindowDiagnostic {
    ExclusiveWindowError reason = ExclusiveWindowError::none;
    // Absent for preflight failures and isolated invalid exclusive operands.
    // Traversal diagnostics identify the opening load of their window.
    std::optional<std::uint64_t> window_offset;
    std::uint64_t instruction_offset = 0;
};

// Indices refer to the supplied coverage. Overlapping windows share one
// register and one transfer at each external entry and closed exit.
struct ExclusiveX18EntryBranch {
    std::size_t instruction_index = 0;
    std::size_t entry_index = 0;
};

struct ExclusiveX18Region {
    std::uint8_t scratch_register = 0;
    std::vector<std::size_t> instructions;
    std::vector<std::size_t> entries;
    std::vector<std::size_t> exits;
    std::vector<ExclusiveX18EntryBranch> entry_branches;
};

struct ExclusiveWindowProof {
    std::size_t windows_found = 0;
    std::size_t windows_proved = 0;
    std::size_t steps = 0;
    // Indexed exactly like the input coverage. Only usable when ok() is true;
    // a false entry may execute with a reservation from an opening load.
    std::vector<bool> monitor_clear;
    // At most one refusal per opening load, plus isolated invalid operands.
    std::vector<ExclusiveWindowDiagnostic> diagnostics;
    std::vector<ExclusiveX18Region> x18_regions;
    bool ok() const noexcept { return diagnostics.empty(); }
};

// Proves windows in the complete reached coverage of one module. Traversal
// follows both conditional edges, includes nested loads, and stops only at
// store-exclusive or CLREX. Live exits/calls, uncovered edges and transfers to
// another known function entry fail closed. Bounds count distinct words per
// opening load and total visits, not dynamic loop iterations. Function entries
// include metadata roots and direct call targets; their order is unrestricted.
// The entry contract supplies no pre-existing guest reservation. A successful
// proof ensures no function can export one through a call or return. Module
// identity distinguishes local indirect entries from remote target offsets.
// For x18 regions, external_entries lists all ingress from other modules.
// Such ingress cannot bypass a local transfer, including at a closed exit.
// Scratch liveness shares the total-step budget with reservation traversal.
ExclusiveWindowProof prove_exclusive_windows(
    std::span<const CoveredInstruction> covered,
    const ExclusiveWindowLimits& limits,
    std::span<const std::uint64_t> function_entries,
    std::uint32_t module_index,
    std::span<const std::uint64_t> external_entries = {});

const char* exclusive_window_error_name(ExclusiveWindowError error) noexcept;

struct PreparedWordPatch {
    std::uint64_t image_offset = 0;
    std::uint32_t expected_word = 0;
    std::uint32_t replacement_word = 0;
};

struct PreparedGateSite {
    std::uint64_t return_address_offset = 0;
    std::uint32_t instruction = 0;
    std::vector<PreparedLocation> allowed_indirect_targets;
};

enum class PreparedFixupKind : std::uint8_t {
    thread_slot_load,
    gate_dispatcher_address,
    module_address_mov_wide,
    branch26_module_target,
};

// image_offset names bytes in the complete prepared image, including its
// appendix. target is used only by the two module-target fixup kinds.
struct PreparedFixup {
    PreparedFixupKind kind = PreparedFixupKind::thread_slot_load;
    std::uint64_t image_offset = 0;
    PreparedLocation target{};
};

struct PreparedImage {
    std::uint32_t module_index = 0;
    std::uint64_t original_image_size = 0;
    std::uint64_t appendix_offset = 0;
    // Every caller-declared covered word, including words left intact.
    std::vector<PreparedWordPatch> coverage;
    // Only words changed at load time. expected_word matches coverage.
    std::vector<PreparedWordPatch> replacements;
    std::vector<std::byte> appendix;
    std::vector<PreparedGateSite> gate_sites;
    std::vector<PreparedFixup> fixups;

    std::uint64_t image_size() const noexcept {
        return appendix.empty() ? original_image_size : appendix_offset + appendix.size();
    }
};

struct PreparedImageOptions {
    std::uint32_t module_index = 0;
    std::uint32_t module_count = 0;
    ScalarX18Policy scalar_x18_policy{};
    GuestSiteTableOptions gate_options{};
    ExclusiveWindowLimits exclusive_limits{};
    // Optional metadata/ABI function entries in this module. Direct-call
    // targets are inferred as well. Whole-program analysis preserves these
    // through image planning; standalone callers supply all additional roots.
    std::span<const std::uint64_t> function_entries{};
    // All ingress targets from other modules, including direct/indirect tail
    // branches. They cannot be retargeted by this module's local window plan.
    std::span<const std::uint64_t> external_entries{};
};

enum class PreparedImageError : std::uint8_t {
    none,
    invalid_options,
    missing_execution_contract,
    invalid_image_size,
    invalid_coverage,
    invalid_target,
    unexpected_targets,
    missing_targets,
    exclusive_instruction,
    unsupported_instruction,
    patch_failed,
    layout_overflow,
    overlapping_regions,
    invalid_fixup,
    invalid_binding,
    preimage_mismatch,
    unreachable_target,
    allocation_failure,
    exclusive_window,
};

struct PreparedImageResult {
    PreparedImageError error = PreparedImageError::none;
    std::size_t instruction_index = 0;
    std::optional<PreparedImage> image;
    ExclusiveWindowProof exclusive_proof;
    bool ok() const noexcept { return image.has_value(); }
};

// Builds a deterministic image plan at a canonical nonzero planning address.
// It consumes only the supplied ordered coverage and explicit target maps; it
// does not scan image bytes or discover executable roots. Native-safe words remain
// unchanged. The appendix begins at the 16 KiB-aligned original image end.
// Exclusive windows are proved over the supplied coverage before patching.
// scalar_x18_policy.exclusive_monitor_clear is ignored: each site's value is
// derived from that proof. The other two policy premises remain caller-owned.
PreparedImageResult prepare_image(std::uint64_t original_image_size,
                                  std::span<const CoveredInstruction> covered,
                                  const PreparedImageOptions& options = {});

struct PreparedImageBindings {
    // One actual base for every module in the same manifest namespace.
    std::span<const std::uint64_t> module_bases;
    std::uint32_t thread_slot_byte_offset = 0;
    std::uintptr_t gate_dispatcher_address = 0;
};

struct PreparedImageApplyResult {
    PreparedImageError error = PreparedImageError::none;
    std::size_t record_index = 0;
    bool ok() const noexcept { return error == PreparedImageError::none; }
};

// Validates the complete structure, exact appendix layout, every covered
// preimage and every process-specific binding before mutating bytes. It then
// installs the owned appendix and replacements and applies only typed fixups;
// no instruction classification or patch builder runs at load time.
PreparedImageApplyResult apply_prepared_image(const PreparedImage& prepared,
                                              std::span<std::byte> image,
                                              const PreparedImageBindings& bindings) noexcept;

} // namespace armigo::execution
