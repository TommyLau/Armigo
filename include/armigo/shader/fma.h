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

// Binary32 a*b+c: signed input FTZ, one nearest-even rounding, then signed
// output FTZ. A rounded carry into minimum normal is retained. NaNs and invalid
// operations return canonical 0x7fc00000. Integer arithmetic only.
[[nodiscard]] std::uint32_t fma_rne_ftz(std::uint32_t a_bits,
                                      std::uint32_t b_bits,
                                      std::uint32_t c_bits) noexcept;

// Emits inline uint armigo_fma_rne_ftz(uint, uint, uint) and private integer
// support functions. Place after the caller's Metal prelude.
[[nodiscard]] std::string emit_fma_msl();

}  // namespace armigo::shader
