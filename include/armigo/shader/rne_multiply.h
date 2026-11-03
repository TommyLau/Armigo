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

// Binary32 product, nearest-even, preserving input and output subnormals.
// Signed zero/infinity follow multiplication; NaNs and zero times infinity
// canonicalize to 0x7fc00000. Integer arithmetic, independent of host FP modes.
[[nodiscard]] std::uint32_t multiply_rne(std::uint32_t lhs_bits,
                                       std::uint32_t rhs_bits) noexcept;

// Emits inline uint armigo_multiply_rne(uint, uint) and its integer rounding
// helper. Place after the caller's Metal prelude. No resources or entry point.
[[nodiscard]] std::string emit_rne_multiply_msl();

}  // namespace armigo::shader
