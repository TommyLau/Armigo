// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Normalized host input state for polled host devices.
//
// Armigo's host input layer describes physical host devices in plain physical
// terms. Face buttons are named by their position on the controller: South is
// the bottom face button, East the right, West the left, and North the top.
// Positions follow the host platform's reports, including system remapping;
// no printed label, platform brand, or HID convention is assumed. Stick axes
// are finite floats in [-1, 1] with right and up positive; analog triggers are
// finite floats in [0, 1].
//
// This layer only describes host devices. It knows nothing about Horizon, the
// hid service, or shared memory, and it defines no merge policy between
// devices: the controller and keyboard states of one snapshot stay separate.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace armigo::platform {

// Physical button positions of a host controller. Each value is one bit flag
// of HostInputDeviceState::buttons. A position names what the host platform
// reports: a system-level user remapping may change which physical input
// lands on a position, and this layer assumes no printed label, platform
// brand, or HID convention.
enum class HostButton : std::uint64_t {
    South = std::uint64_t{1} << 0,         // bottom face button
    East = std::uint64_t{1} << 1,          // right face button
    West = std::uint64_t{1} << 2,          // left face button
    North = std::uint64_t{1} << 3,         // top face button
    LeftShoulder = std::uint64_t{1} << 4,
    RightShoulder = std::uint64_t{1} << 5,
    LeftStick = std::uint64_t{1} << 6,     // left stick click
    RightStick = std::uint64_t{1} << 7,    // right stick click
    Menu = std::uint64_t{1} << 8,          // menu / plus button
    Options = std::uint64_t{1} << 9,       // options / minus button
    Up = std::uint64_t{1} << 10,           // D-pad up
    Down = std::uint64_t{1} << 11,         // D-pad down
    Left = std::uint64_t{1} << 12,         // D-pad left
    Right = std::uint64_t{1} << 13,        // D-pad right
};

// Bit mask of one HostButton flag, for composing or testing
// HostInputDeviceState::buttons.
inline constexpr std::uint64_t button_mask(HostButton button) noexcept {
    return static_cast<std::uint64_t>(button);
}

// True when `buttons` carries the flag `button`.
inline constexpr bool has_button(std::uint64_t buttons, HostButton button) noexcept {
    return (buttons & button_mask(button)) != 0;
}

// Physical keys of the host keyboard that Armigo maps. The enum is closed:
// a poller translates raw key codes into these values before mapping.
enum class HostKey {
    W, A, S, D,
    Up, Down, Left, Right,  // arrow keys
    E, Shift, Space, Q,
    Enter, Tab,
    F, R, C, V,
    I, J, K, L,  // right stick: up, left, down, right
    U, O,        // West (Y), right trigger (ZR)
};

// One host input device: a controller or a keyboard. A default-constructed
// state is disconnected, unsupported, and idle. `connected` means the device
// is present; `supported` means the device exposes the inputs Armigo maps.
// A connected but unsupported device reports no inputs, never a phantom
// working state.
struct HostInputDeviceState {
    bool connected = false;
    bool supported = false;
    std::uint64_t buttons = 0;
    // Element 0 is the x axis (right positive), element 1 the y axis
    // (up positive). Both components are finite and within [-1, 1].
    std::array<float, 2> left_stick{};
    std::array<float, 2> right_stick{};
    // Set where an input script supplies the stick: its values are exact HID
    // coordinates, not a physical stick's, so they are not shaped.
    bool exact_left_stick = false;
    bool exact_right_stick = false;
    float left_trigger = 0.0f;   // finite, within [0, 1]
    float right_trigger = 0.0f;  // finite, within [0, 1]
    // Motion sensors, when the device has them, in the device's frame (x
    // right, y forward toward the shoulder buttons, z up out of the face):
    // acceleration including gravity in G, about (0, 0, -1) at rest face up,
    // and rotation rate in radians per second. All components are finite.
    bool has_motion = false;
    bool has_rotation_rate = false;  // a gyroscope; without one rotation_rate stays zero
    std::array<float, 3> acceleration{0.0f, 0.0f, -1.0f};
    std::array<float, 3> rotation_rate{};
};

// One polled input snapshot. The controller and keyboard stay separate
// states; nothing in this layer merges them.
struct HostInputSnapshot {
    HostInputDeviceState controller;
    HostInputDeviceState keyboard;
};

// Purely maps a set of pressed keyboard keys into the keyboard device state,
// with a Switch controller's buttons by position. W/A/S/D drive the left
// stick and I/J/K/L the right stick (up/left/down/right): opposing keys
// cancel, and a pressed diagonal is normalized to unit length. Arrow keys are
// the D-pad. E is East (A), Shift is South (B), Space is North (X), U is West
// (Y). Q fully presses the left trigger (ZL) and O the right trigger (ZR).
// F and R are the left and right shoulders (L, R); C and V click the left and
// right sticks; Enter is Menu (+) and Tab is Options (-). Duplicate keys are
// idempotent. F1 is not a mapped key. The result is always connected and
// supported: it describes the keyboard device itself, and a poller that found
// no keyboard returns a default state instead of calling this.
HostInputDeviceState keyboard_input(std::span<const HostKey> keys) noexcept;

// Normalizes one raw stick axis sample into the documented range: a
// non-finite input (NaN or infinity) becomes 0, and a finite input outside
// [-1, 1] is clamped into it.
float normalized_stick_component(float value) noexcept;

// Normalizes one raw analog trigger sample into the documented range: a
// non-finite input becomes 0, a value below 0 becomes 0, and a value above 1
// is clamped to 1.
float normalized_trigger_value(float value) noexcept;

}  // namespace armigo::platform
