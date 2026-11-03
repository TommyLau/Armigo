// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/host_input.h>

#include <array>
#include <cmath>

namespace armigo::platform {
namespace {

// Opposing keys cancel exactly; a pressed diagonal is scaled to unit length
// so no direction moves faster than a cardinal one.
std::array<float, 2> stick_deflection(bool up, bool down, bool left, bool right) noexcept {
    float x = (right ? 1.0f : 0.0f) - (left ? 1.0f : 0.0f);
    float y = (up ? 1.0f : 0.0f) - (down ? 1.0f : 0.0f);
    if (x != 0.0f && y != 0.0f) {
        constexpr float kInvSqrt2 = 0.70710678118654752440f;
        x *= kInvSqrt2;
        y *= kInvSqrt2;
    }
    return {x, y};
}

}  // namespace

HostInputDeviceState keyboard_input(std::span<const HostKey> keys) noexcept {
    HostInputDeviceState state;
    state.connected = true;
    state.supported = true;

    bool left_up = false, left_down = false, left_left = false, left_right = false;
    bool right_up = false, right_down = false, right_left = false, right_right = false;
    for (const HostKey key : keys) {
        switch (key) {
        case HostKey::W: left_up = true; break;
        case HostKey::A: left_left = true; break;
        case HostKey::S: left_down = true; break;
        case HostKey::D: left_right = true; break;
        case HostKey::I: right_up = true; break;
        case HostKey::J: right_left = true; break;
        case HostKey::K: right_down = true; break;
        case HostKey::L: right_right = true; break;
        case HostKey::Up: state.buttons |= button_mask(HostButton::Up); break;
        case HostKey::Down: state.buttons |= button_mask(HostButton::Down); break;
        case HostKey::Left: state.buttons |= button_mask(HostButton::Left); break;
        case HostKey::Right: state.buttons |= button_mask(HostButton::Right); break;
        case HostKey::E: state.buttons |= button_mask(HostButton::East); break;
        case HostKey::Shift: state.buttons |= button_mask(HostButton::South); break;
        case HostKey::Space: state.buttons |= button_mask(HostButton::North); break;
        case HostKey::U: state.buttons |= button_mask(HostButton::West); break;
        case HostKey::Q: state.left_trigger = 1.0f; break;
        case HostKey::O: state.right_trigger = 1.0f; break;
        case HostKey::F: state.buttons |= button_mask(HostButton::LeftShoulder); break;
        case HostKey::R: state.buttons |= button_mask(HostButton::RightShoulder); break;
        case HostKey::C: state.buttons |= button_mask(HostButton::LeftStick); break;
        case HostKey::V: state.buttons |= button_mask(HostButton::RightStick); break;
        case HostKey::Enter: state.buttons |= button_mask(HostButton::Menu); break;
        case HostKey::Tab: state.buttons |= button_mask(HostButton::Options); break;
        }
    }

    state.left_stick = stick_deflection(left_up, left_down, left_left, left_right);
    state.right_stick = stick_deflection(right_up, right_down, right_left, right_right);
    return state;
}

float normalized_stick_component(float value) noexcept {
    if (!std::isfinite(value)) return 0.0f;
    if (value > 1.0f) return 1.0f;
    if (value < -1.0f) return -1.0f;
    return value;
}

float normalized_trigger_value(float value) noexcept {
    if (!std::isfinite(value)) return 0.0f;
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

}  // namespace armigo::platform
