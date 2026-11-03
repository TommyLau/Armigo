// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Host reference evaluator for shader IR programs.
//
// The evaluator defines the expected results of a program independently of any
// GPU backend. It interprets each instruction with IEEE-754 binary32
// arithmetic in round-to-nearest-even mode and never reassociates or fuses
// operations. Subnormal handling in arithmetic is selected explicitly; see
// SubnormalMode.
#pragma once

#include <armigo/shader/diagnostics.h>
#include <armigo/shader/ir.h>

#include <cstdint>
#include <span>
#include <vector>

namespace armigo::shader {

// Treatment of subnormal binary32 values by fadd and fmul. Constants, loads,
// and stores always copy bit patterns unchanged.
enum class SubnormalMode : std::uint8_t {
    // IEEE-754 gradual underflow: subnormal operands and results are kept.
    preserve,
    // Subnormal operands are replaced by a zero of the same sign before the
    // operation, and a subnormal rounded result is replaced by a zero of the
    // same sign. This models the Metal backend, which compiles with denormals
    // disabled.
    flush_to_zero,
};

struct ReferenceOptions {
    SubnormalMode subnormals = SubnormalMode::preserve;
};

struct ReferenceResult {
    // Empty when evaluation succeeded. Invalid programs report their
    // validation diagnostics; malformed arguments report argument diagnostics.
    std::vector<Diagnostic> diagnostics;
    // One array of raw 32-bit words per declared output, each element_count
    // long.
    std::vector<std::vector<std::uint32_t>> outputs;

    bool ok() const noexcept { return diagnostics.empty(); }
};

// Evaluates `program` for invocations [0, element_count).
//
// `parameters` holds one raw word per declared parameter slot. `inputs` holds
// one array per declared input binding; each must contain at least
// element_count words.
ReferenceResult evaluate_reference(
    const Program& program, std::span<const std::uint32_t> parameters,
    std::span<const std::vector<std::uint32_t>> inputs,
    std::uint32_t element_count, const ReferenceOptions& options = {});

}  // namespace armigo::shader
