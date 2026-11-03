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

// Binary32 saturation to the unit interval in pure integer arithmetic,
// independent of host floating-point modes and of any flush-to-zero state.
// Armigo's bounded bit policy for the saturation modifier: any input with the
// sign bit set, including negative zero, and every NaN maps to positive zero;
// nonnegative values above one, including positive infinity, map to one; all
// other nonnegative values, including positive subnormals, keep their bits.
// Inputs reaching this node are already canonicalized by Armigo's arithmetic.
// The universal negative-zero normalization is an explicit bounded-profile
// choice, not a hardware-verified claim.
[[nodiscard]] std::uint32_t saturate_float32(std::uint32_t bits) noexcept;

// Emits inline uint armigo_saturate_float32(uint) with the identical integer
// mapping. Place after the caller's Metal prelude. No resources or entry point.
[[nodiscard]] std::string emit_saturate_msl();

}  // namespace armigo::shader
