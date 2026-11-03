// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Metal Shading Language emission for validated shader IR programs.
#pragma once

#include <armigo/shader/diagnostics.h>
#include <armigo/shader/ir.h>

#include <cstdint>
#include <string>
#include <vector>

namespace armigo::shader {

enum class MslBindingKind : std::uint8_t {
    parameters,     // Constant parameter block; one 4-byte word per slot.
    input,          // device const array of the bound scalar type.
    output,         // device array of the bound scalar type.
    element_count,  // constant uint; invocations at or past it do nothing.
};

const char* to_string(MslBindingKind kind) noexcept;

struct MslBinding {
    MslBindingKind kind;
    // Input or output binding number; zero for the other kinds.
    std::uint32_t ordinal;
    // Metal buffer argument table index.
    std::uint32_t buffer_index;
    // Element type for input and output bindings; u32 for element_count and
    // unused for the parameter block.
    ScalarType type;

    friend bool operator==(const MslBinding&, const MslBinding&) = default;
};

struct MslKernel {
    // Emitted symbol, prefixed to avoid Metal standard-library collisions.
    // Consumers must use this name rather than Program::entry_point.
    std::string entry_point;
    std::string source;
    // Buffer bindings in ascending buffer index order.
    std::vector<MslBinding> bindings;
    // Size in bytes of the parameter block (4 bytes per parameter slot).
    std::uint32_t parameter_block_size = 0;
};

struct MslEmitResult {
    // Empty when emission succeeded; then `kernel` holds the result.
    std::vector<Diagnostic> diagnostics;
    MslKernel kernel;

    bool ok() const noexcept { return diagnostics.empty(); }
};

// Validates `program` and emits a Metal 4 compute kernel. Emission is
// deterministic: equal programs produce byte-identical source.
//
// Floating-point contract of the generated code:
//   * Each IR instruction becomes one statement with one operator, in IR order
//     and with the IR operand order, so source-level evaluation order is fixed.
//   * The source selects safe math mode and disables contraction with
//     `#pragma METAL fp`, so the compiler may not reassociate, fuse into FMA,
//     or assume the absence of NaN, infinity, or signed zero.
//   * Constants are emitted from their exact bit patterns.
// The host must compile the source as Metal 4.0 (for example `-std=metal4.0`
// or MTLLanguageVersion4_0). Emission is this straight-line scalar compute
// IR only: one statement per instruction, with no control flow or guest decode.
MslEmitResult emit_msl(const Program& program);

}  // namespace armigo::shader
