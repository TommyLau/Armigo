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

#include <armigo/shader/graphics_control_flow.h>
#include <optional>

namespace armigo::shader::graphics::detail {
// Reuses the legacy builder's SPH declaration and header admission rules.
[[nodiscard]] std::optional<Error> validate_graph_interface(const GraphSpec& spec);
}  // namespace armigo::shader::graphics::detail
