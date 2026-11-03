// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>
#include <string>

namespace armigo::shader {

// Multiply binary32 bit patterns, rounding toward zero. Input and output
// subnormals become signed zero; finite overflow becomes signed max-finite.
// NaN inputs and zero times infinity produce canonical quiet NaN 0x7fc00000.
// Uses integer arithmetic only, independently of the host floating environment.
[[nodiscard]] std::uint32_t multiply_rtz_ftz(std::uint32_t lhs_bits,
                                           std::uint32_t rhs_bits) noexcept;

// Emits inline uint armigo_mul_rtz_ftz(uint, uint), using Metal uint/ulong
// arithmetic with the identical bit-pattern contract. Add it after the caller's
// Metal language prelude; it declares no resources or entry point.
[[nodiscard]] std::string emit_rtz_ftz_multiply_msl();

}  // namespace armigo::shader
