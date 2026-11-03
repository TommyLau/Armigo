// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/touch_controls.h>

#include <algorithm>
#include <cmath>

namespace armigo::platform {
namespace {

constexpr std::size_t index(TouchControl control) noexcept {
    return static_cast<std::size_t>(control);
}

TouchControlLayout round_control(double x, double y, double radius) noexcept {
    return {x, y, radius, 0, 0};
}

TouchControlLayout pill_control(double x, double y, double width, double height) noexcept {
    return {x, y, 0, width, height};
}

} // namespace

TouchLayout make_touch_layout(double width, double height, const TouchInsets& insets) noexcept {
    const double left = std::max(0.0, insets.left);
    const double top = std::max(0.0, insets.top);
    const double safe_width = std::max(0.0, width - left - std::max(0.0, insets.right));
    const double safe_height = std::max(0.0, height - top - std::max(0.0, insets.bottom));
    const double unit = std::clamp(height / 390.0, 0.0, 1.0);
    const double right = left + safe_width;
    const double bottom = top + safe_height;
    const double stick_y = bottom - 80 * unit;
    const double upper_y = stick_y - 135 * unit;
    const double left_x = left + 95 * unit;
    const double right_x = right - 95 * unit;
    const double middle = left + safe_width / 2;

    TouchLayout layout{};
    layout[index(TouchControl::left_stick)] = round_control(left_x, stick_y, 58 * unit);
    layout[index(TouchControl::right_stick)] = round_control(right_x, stick_y, 58 * unit);
    layout[index(TouchControl::a)] = round_control(right_x + 46 * unit, upper_y, 22 * unit);
    layout[index(TouchControl::b)] = round_control(right_x, upper_y + 46 * unit, 22 * unit);
    layout[index(TouchControl::x)] = round_control(right_x, upper_y - 46 * unit, 22 * unit);
    layout[index(TouchControl::y)] = round_control(right_x - 46 * unit, upper_y, 22 * unit);
    layout[index(TouchControl::up)] = round_control(left_x, upper_y - 42 * unit, 19 * unit);
    layout[index(TouchControl::down)] = round_control(left_x, upper_y + 42 * unit, 19 * unit);
    layout[index(TouchControl::left)] = round_control(left_x - 42 * unit, upper_y, 19 * unit);
    layout[index(TouchControl::right)] = round_control(left_x + 42 * unit, upper_y, 19 * unit);
    layout[index(TouchControl::l)] = pill_control(left + 50 * unit, top + 62 * unit, 64 * unit, 30 * unit);
    layout[index(TouchControl::zl)] = pill_control(left + 50 * unit, top + 24 * unit, 64 * unit, 30 * unit);
    layout[index(TouchControl::r)] = pill_control(right - 50 * unit, top + 62 * unit, 64 * unit, 30 * unit);
    layout[index(TouchControl::zr)] = pill_control(right - 50 * unit, top + 24 * unit, 64 * unit, 30 * unit);
    layout[index(TouchControl::minus)] = round_control(middle - 34 * unit, top + 24 * unit, 17 * unit);
    layout[index(TouchControl::plus)] = round_control(middle + 34 * unit, top + 24 * unit, 17 * unit);
    layout[index(TouchControl::left_stick_click)] =
        round_control(left_x + 92 * unit, stick_y - 30 * unit, 17 * unit);
    layout[index(TouchControl::right_stick_click)] =
        round_control(right_x - 92 * unit, stick_y - 30 * unit, 17 * unit);
    return layout;
}

int touch_control_at(const TouchLayout& layout, double x, double y) noexcept {
    const double scale = layout[index(TouchControl::left_stick)].radius / 58.0;
    const double margin = 10 * scale;
    for (std::size_t i = 0; i < layout.size(); ++i) {
        const auto& control = layout[i];
        const double dx = x - control.x;
        const double dy = y - control.y;
        const bool hit = i <= index(TouchControl::right_stick)
            ? std::hypot(dx, dy) < control.radius * 1.5
            : control.width > 0
                ? std::abs(dx) < control.width / 2 + margin && std::abs(dy) < control.height / 2 + margin
                : std::hypot(dx, dy) < control.radius + margin;
        if (hit) return static_cast<int>(i);
    }
    return -1;
}

TouchStick clamped_touch_stick(const TouchControlLayout& layout, double x, double y) noexcept {
    if (layout.radius <= 0) return {};
    double dx = (x - layout.x) / layout.radius;
    double dy = (layout.y - y) / layout.radius;
    const double magnitude = std::hypot(dx, dy);
    if (magnitude > 1) {
        dx /= magnitude;
        dy /= magnitude;
    }
    return {normalized_stick_component(static_cast<float>(dx)),
            normalized_stick_component(static_cast<float>(dy))};
}

HostInputDeviceState touch_input_state(
    const std::array<unsigned, touch_control_count>& held,
    const std::array<TouchStick, 2>& sticks) noexcept {
    HostInputDeviceState state;
    state.connected = true;
    state.supported = true;
    constexpr std::array<std::pair<TouchControl, HostButton>, 14> buttons{{
        {TouchControl::a, HostButton::East}, {TouchControl::b, HostButton::South},
        {TouchControl::x, HostButton::North}, {TouchControl::y, HostButton::West},
        {TouchControl::up, HostButton::Up}, {TouchControl::down, HostButton::Down},
        {TouchControl::left, HostButton::Left}, {TouchControl::right, HostButton::Right},
        {TouchControl::l, HostButton::LeftShoulder}, {TouchControl::r, HostButton::RightShoulder},
        {TouchControl::minus, HostButton::Options}, {TouchControl::plus, HostButton::Menu},
        {TouchControl::left_stick_click, HostButton::LeftStick},
        {TouchControl::right_stick_click, HostButton::RightStick},
    }};
    for (const auto& [control, button] : buttons) {
        if (held[index(control)] != 0) state.buttons |= button_mask(button);
    }
    state.left_stick = {normalized_stick_component(sticks[0][0]), normalized_stick_component(sticks[0][1])};
    state.right_stick = {normalized_stick_component(sticks[1][0]), normalized_stick_component(sticks[1][1])};
    state.left_trigger = held[index(TouchControl::zl)] != 0 ? 1.0f : 0.0f;
    state.right_trigger = held[index(TouchControl::zr)] != 0 ? 1.0f : 0.0f;
    return state;
}

} // namespace armigo::platform
