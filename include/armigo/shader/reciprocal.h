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

// Correctly rounded binary32 1/x using nearest-even and gradual underflow.
// Preserves the sign of zero/infinity; canonicalizes NaNs to 0x7fc00000.
// Integer arithmetic only. This is a mathematical software contract, not a
// claim about the approximation accuracy of a hardware reciprocal instruction.
[[nodiscard]] std::uint32_t reciprocal_rne(std::uint32_t input_bits) noexcept;

// Emits standalone inline uint armigo_reciprocal_rne(uint). Place after the
// caller's Metal prelude; the helper uses uint/ulong arithmetic exclusively.
[[nodiscard]] std::string emit_reciprocal_msl();

}  // namespace armigo::shader
