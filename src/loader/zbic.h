// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>

namespace armigo::loader::detail {

// Rewrites BIC entropy headers into standard Zstandard headers and decodes
// into the exact caller-owned segment extent. No dictionary or concatenated frames.
std::optional<std::string> decompress_zbic(std::span<const std::byte> input,
                                         std::span<std::byte> output);

} // namespace armigo::loader::detail
