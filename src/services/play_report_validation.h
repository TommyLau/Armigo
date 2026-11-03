// ============================================================================
// Armigo Services - Offline Play Report Validation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <span>

namespace armigo::services::prepo::detail {

// Allocation failures propagate to the session's host diagnostic path rather
// than being reported to the guest as malformed report data.
bool valid_report(std::span<const std::byte> report);

} // namespace armigo::services::prepo::detail
