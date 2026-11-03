// ============================================================================
// Armigo Runtime - Process Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <string_view>

namespace armigo::runtime {

// The shader translation chain a trusted offline package must have been built
// with (the one launches accept); empty when this build has no Metal compiler
// tools and accepts no package.
[[nodiscard]] std::string_view expected_shader_chain_id() noexcept;

} // namespace armigo::runtime
