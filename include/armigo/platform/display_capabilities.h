// ============================================================================
// Armigo Platform - Host Display Capabilities
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
//
// What the host display can do, in portable terms. The display service reports
// a refresh rate to applications and paces its vsync event and presentations
// against the same rate, so both read it from here instead of assuming one
// panel.

#pragma once

#include <cstdint>
#include <vector>

namespace armigo::platform {

// One display mode the panel offers, as the host describes it.
struct DisplayModeCapability {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Refresh rate the mode was described with; zero when the host does not
    // report one for it.
    double refresh_hz = 0.0;
    // True when the host runs the mode with a variable refresh rate.
    bool variable_refresh = false;
};

struct DisplayCapabilities {
    // Modes the connected panel offers, current first when the host reports it.
    std::vector<DisplayModeCapability> modes;
    // The mode the display is in now. Width and height are zero when the host
    // could not describe the display (a headless session, for example).
    DisplayModeCapability current;
    // Highest refresh rate any mode offers, zero when unknown.
    double max_refresh_hz = 0.0;
    // Guest rate pinned by the measurement override, zero when unset.
    double guest_override_hz = 0.0;

    [[nodiscard]] bool available() const noexcept { return current.width && current.height; }
    // The rate the guest's display policy runs at: the largest even division of
    // the panel's refresh that is not below the nominal 60 Hz a guest is given
    // (60 on a 60 or 120 Hz panel). Never the bare panel rate.
    [[nodiscard]] double guest_refresh_hz() const noexcept;
    // Refresh period of that guest rate in nanoseconds; falls back to 60 Hz.
    [[nodiscard]] std::int64_t refresh_period_ns() const noexcept;
};

// Queries the host's main display. Never throws; reports an unavailable result
// when the platform cannot answer.
[[nodiscard]] DisplayCapabilities query_display_capabilities() noexcept;

// macOS: query one display by its native numeric identifier. Framework types
// stay in the adapter; the application supplies the launch window's display.
[[nodiscard]] DisplayCapabilities query_display_capabilities(std::uint32_t display_id) noexcept;

} // namespace armigo::platform
