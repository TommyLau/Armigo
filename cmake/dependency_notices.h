// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <string_view>

namespace armigo::pack {
inline constexpr std::string_view opus_notice = R"opus(@ARMIGO_OPUS_NOTICE@)opus";
inline constexpr std::string_view zstd_notice = R"zstd(@ARMIGO_ZSTD_NOTICE@)zstd";
} // namespace armigo::pack
