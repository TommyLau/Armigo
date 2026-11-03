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

#include <cstddef>
#include <cstdint>
#include <string>

namespace armigo::runtime {

inline constexpr std::uint32_t prepared_modules_format_version = 1;
inline constexpr std::size_t default_prepared_artifact_limit = 256ull * 1024 * 1024;
inline constexpr std::size_t prepared_fixup_limit = 1'000'000;

// The installed preparer's algorithm revision and pointer-free native gate
// template digest. A caller cannot select an older implementation by supplying
// an identity string. Empty PreparationOptions identity selects this build.
[[nodiscard]] std::string installed_preparation_identity();

} // namespace armigo::runtime
