// ============================================================================
// Armigo Platform - Host Display Capabilities
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/display_capabilities.h>
#include <algorithm>
#include <cmath>

namespace armigo::platform {

double DisplayCapabilities::guest_refresh_hz() const noexcept {
    // The guest's display policy is a nominal 60 Hz mode (firmware admits
    // only matching 60 Hz modes), so the rate the guest's vsync event and its
    // swap interval refer to is the largest even division of the panel's refresh
    // that is not below 60 Hz: 60 on a 60 or 120 Hz panel, 72 on 144 Hz. The host
    // may show each guest frame on more than one panel refresh; that is a
    // presentation detail and must not change the guest's cadence.
    if (guest_override_hz > 0.0) return guest_override_hz;
    double panel = current.refresh_hz;
    if (max_refresh_hz > panel) panel = max_refresh_hz;
    if (!(panel > 0.0) || !(panel < 10000.0)) return 60.0;
    const auto divisions = std::max(1.0, std::ceil(panel / 60.0 - 0.02));
    return panel / divisions;
}

std::int64_t DisplayCapabilities::refresh_period_ns() const noexcept {
    const auto rate = guest_refresh_hz();
    return static_cast<std::int64_t>(1e9 / rate);
}

} // namespace armigo::platform
