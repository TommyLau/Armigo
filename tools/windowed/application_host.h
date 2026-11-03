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

#include "runner_shell.h"

namespace armigo::tools {

// Real WindowedHost backed by the platform Application (#310). Create and use
// it on the application main thread only.
[[nodiscard]] WindowedHostCreation create_application_host();

} // namespace armigo::tools
