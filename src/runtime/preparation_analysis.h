// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/prepared_image.h>
#include <armigo/linker/dynamic_module.h>
#include <armigo/loader/nso_module.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace armigo::runtime::preparation {

struct Target {
    std::uint32_t module_index = 0;
    std::uint64_t image_offset = 0;
    bool operator==(const Target&) const = default;
};

struct RelocatedPointer {
    std::uint64_t image_offset = 0;
    Target target;
};

struct AnalysisModule {
    const loader::NsoModuleImage* image = nullptr;
    const linker::DynamicModule* metadata = nullptr;
    std::span<const std::byte> linked_image;
    std::uint64_t synthetic_base = 0;
    // Sorted slot proofs derived from provider-relative LinkPlan writes only.
    // A numeric address in synthetic space is not relocation provenance.
    std::span<const RelocatedPointer> relocated_pointers;
    // Explicit executable roots supplied by the input ABI. These do not need
    // dynamic symbols, but retain the same text and analysis requirements.
    std::span<const std::uint64_t> entry_roots{};
    // Permit only exact stores of a proven relocation value into its declared
    // writable slot. General pointer escapes remain unsupported.
    bool guest_relocations = false;
};

struct CodeWord {
    std::uint64_t offset = 0;
    std::uint32_t word = 0;
    bool exclusive_monitor_clear = false;
};

struct IndirectSite {
    std::uint64_t offset = 0;
    std::vector<Target> targets;
};

struct ModuleAnalysis {
    std::vector<CodeWord> code;
    std::vector<IndirectSite> indirect_sites;
    std::vector<std::uint64_t> function_entries;
    std::vector<std::uint64_t> external_entries;
};

struct AnalysisLimits {
    std::size_t max_modules = 64;
    std::size_t max_roots = 1'000'000;
    std::size_t max_code_words = 1'000'000;
    std::size_t max_indirect_sites = 65'536;
    std::size_t max_unwind_records = 1'000'000;
    std::size_t max_address_steps = 32'000'000;
    execution::ExclusiveWindowLimits exclusive_windows{};
    std::size_t max_return_steps = 1'000'000;
    std::size_t max_return_states = 65'536;
    std::size_t max_return_stack_slots = 256;
};

struct ExclusiveDiagnostic {
    std::uint32_t module_index = 0;
    execution::ExclusiveWindowDiagnostic window;
};

enum class ReturnError : std::uint8_t {
    unsupported_return_register,
    unproven_link_write,
    unproven_return,
    unproven_stack_pointer,
    stack_pointer_escape,
    work_limit,
    state_limit,
    stack_slot_limit,
};

struct ReturnDiagnostic {
    std::uint32_t module_index = 0;
    std::uint64_t instruction_offset = 0;
    ReturnError reason = ReturnError::unproven_return;
};

struct AnalysisResult {
    std::vector<ModuleAnalysis> modules;
    std::vector<Target> roots;
    std::string error;
    // Retained on failure; an incomplete discovery/proof never establishes
    // monitor clearance. Refusals carry module-relative window/site locations.
    std::size_t exclusive_windows_found = 0;
    std::size_t exclusive_windows_proved = 0;
    bool exclusive_proof_complete = false;
    std::vector<ExclusiveDiagnostic> exclusive_diagnostics;
    std::vector<ReturnDiagnostic> return_diagnostics;
    bool ok() const noexcept { return error.empty(); }
};

// Reads linked synthetic-address images without changing them. Roots come from
// explicit ABI entries, executable function symbols, initialization metadata,
// and the supported version-one unwind-header search table. Unwind records
// provide roots only; their instruction programs and code ranges are not
// interpreted. Reached unsupported instructions, ambiguous data/code, and
// unproven indirect transfers reject the entire result. Unreached bytes remain
// untouched and are not classified as data.
AnalysisResult analyze_modules(std::span<const AnalysisModule> modules,
                               const AnalysisLimits& limits = {});

} // namespace armigo::runtime::preparation
