// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/platform/host_input.h>

#include <array>
#include <cstddef>

namespace armigo::platform {

enum class TouchControl : std::size_t {
    left_stick,
    right_stick,
    a,
    b,
    x,
    y,
    up,
    down,
    left,
    right,
    l,
    zl,
    r,
    zr,
    minus,
    plus,
    left_stick_click,
    right_stick_click,
};

inline constexpr std::size_t touch_control_count =
    static_cast<std::size_t>(TouchControl::right_stick_click) + 1;

struct TouchInsets {
    double left = 0;
    double top = 0;
    double right = 0;
    double bottom = 0;
};

struct TouchControlLayout {
    double x = 0;
    double y = 0;
    double radius = 0;
    double width = 0;
    double height = 0;
};

using TouchLayout = std::array<TouchControlLayout, touch_control_count>;
using TouchStick = std::array<float, 2>;

TouchLayout make_touch_layout(double width, double height, const TouchInsets& insets) noexcept;
int touch_control_at(const TouchLayout& layout, double x, double y) noexcept;
TouchStick clamped_touch_stick(const TouchControlLayout& layout, double x, double y) noexcept;
HostInputDeviceState touch_input_state(
    const std::array<unsigned, touch_control_count>& held,
    const std::array<TouchStick, 2>& sticks) noexcept;

} // namespace armigo::platform
