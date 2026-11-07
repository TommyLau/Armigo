// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/host_controller.h>

#include <cmath>
#include <cstdio>
#include <span>
#include <limits>

namespace {

using armigo::platform::button_mask;
using armigo::platform::HostButton;
using armigo::platform::HostInputSnapshot;

int failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)

HostInputSnapshot connected() {
    HostInputSnapshot snapshot;
    snapshot.controller.connected = snapshot.controller.supported = true;
    snapshot.keyboard.connected = snapshot.keyboard.supported = true;
    return snapshot;
}

// Face buttons map by position: South is B, East is A, West is Y, North is X.
void test_buttons_by_position() {
    auto snapshot = connected();
    snapshot.controller.buttons = button_mask(HostButton::South) | button_mask(HostButton::North) |
        button_mask(HostButton::Menu) | button_mask(HostButton::Up);
    snapshot.keyboard.buttons = button_mask(HostButton::East) | button_mask(HostButton::LeftShoulder);
    const auto input = armigo::runtime::full_key_input(snapshot);
    // B 1, X 4, Plus 0x400, Up 0x2000, A 1, L 0x40.
    CHECK(input.buttons == (0x2u | 0x4u | 0x400u | 0x2000u | 0x1u | 0x40u));
}

// Triggers past half press ZL and ZR.
void test_triggers() {
    auto snapshot = connected();
    snapshot.controller.left_trigger = 0.6f;
    snapshot.keyboard.right_trigger = 0.4f;
    CHECK(armigo::runtime::full_key_input(snapshot).buttons == 0x100u);
}

// Each stick takes the device that deflects it further; past half it sets
// the stick's direction bits.
void test_sticks() {
    auto snapshot = connected();
    snapshot.keyboard.left_stick = {0.0f, 1.0f};
    snapshot.controller.left_stick = {0.2f, 0.0f};
    snapshot.controller.right_stick = {-1.0f, 0.0f};
    const auto input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.left_x == 0 && input.left_y == 32767);
    CHECK(input.right_x == -32767 && input.right_y == 0);
    CHECK(input.buttons == ((1u << 17) | (1u << 20)));
}

// Controller sticks: a 12% radial dead zone keeps a resting stick centred,
// the range past it is rescaled from zero, and a diagonal is clamped to the
// unit circle. Keyboard diagonals are unchanged.
void test_controller_stick_shaping() {
    auto snapshot = connected();
    snapshot.controller.left_stick = {0.1f, 0.05f};   // inside the dead zone
    auto input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.left_x == 0 && input.left_y == 0 && (input.buttons & 0xf0000u) == 0);
    snapshot.controller.left_stick = {0.13f, 0.0f};   // just past it: small, from zero
    input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.left_x > 0 && input.left_x < 1000 && input.left_y == 0);
    snapshot.controller.left_stick = {0.56f, 0.0f};   // halfway through the live range
    input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.left_x > 16000 && input.left_x < 16800);
    snapshot.controller.left_stick = {1.0f, 0.0f};
    input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.left_x == 32767);
    snapshot.controller.left_stick = {1.0f, 1.0f};    // a corner: clamped to the circle
    input = armigo::runtime::full_key_input(snapshot);
    const double length = std::hypot(double(input.left_x), double(input.left_y));
    CHECK(length > 32700 && length <= 32768 && input.left_x == input.left_y);
    snapshot.controller.left_stick = {std::numeric_limits<float>::quiet_NaN(), 0.5f};
    input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.left_x == 0 && input.left_y > 0);
    // An input script's stick is exact.
    snapshot.controller.left_stick = {-0.5f, 1.0f};
    snapshot.controller.exact_left_stick = true;
    input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.left_x == -16384 && input.left_y == 32767);
    snapshot.controller.exact_left_stick = false;
    // The keyboard's diagonal is not shaped.
    using armigo::platform::HostKey;
    using armigo::platform::keyboard_input;
    const HostKey diagonal[] = {HostKey::W, HostKey::D};
    auto keys = connected();
    keys.keyboard = keyboard_input(std::span<const HostKey>(diagonal, 2));
    const auto from_keys = armigo::runtime::full_key_input(keys);
    const auto expected = armigo::runtime::full_key_input([&] {
        auto only = connected();
        only.keyboard = keys.keyboard;
        return only;
    }());
    CHECK(from_keys.left_x == expected.left_x && from_keys.left_y == expected.left_y && from_keys.left_x > 0);
    // A controller stick just past the dead zone still takes the right stick
    // from the mouse camera.
    armigo::runtime::MouseCamera camera;
    camera.moved(8, 0);
    snapshot = connected();
    snapshot.controller.right_stick = {0.0f, 0.13f};
    input = armigo::runtime::full_key_input(snapshot);
    const auto held = input;
    camera.apply(input);
    CHECK(held.right_y > 0 && input.right_x == held.right_x && input.right_y == held.right_y);
}

// I/J/K/L and a controller share the right stick: the further deflection
// wins, and a tie keeps the controller. U is Y and O is ZR. The mouse camera
// yields once that stick passes the 12% dead zone, and supplies it otherwise.
// Mouse buttons OR with the keyboard.
void test_keyboard_right_stick_mouse_and_face() {
    using armigo::platform::HostKey;
    using armigo::platform::keyboard_input;
    auto snapshot = connected();
    const HostKey up[] = {HostKey::I};
    snapshot.keyboard = keyboard_input(std::span<const HostKey>(up, 1));
    snapshot.controller.right_stick = {0.25f, 0.0f};
    auto input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.right_x == 0 && input.right_y == 32767);
    CHECK((input.buttons & (1u << 21)) != 0);  // right stick up

    const HostKey duplicated[] = {HostKey::I, HostKey::I};
    snapshot.keyboard = keyboard_input(std::span<const HostKey>(duplicated, 2));
    snapshot.controller.right_stick = {-1.0f, 0.0f};
    input = armigo::runtime::full_key_input(snapshot);
    // Equal deflection: the controller is applied first and kept. A repeated
    // key does not deflect any further.
    CHECK(input.right_x == -32767 && input.right_y == 0);

    armigo::runtime::MouseCamera camera;
    camera.moved(8, 0);
    const auto held = input;
    camera.apply(input);
    CHECK(input.right_x == held.right_x && input.right_y == held.right_y);

    armigo::services::hid::HostFullKeyInput centred;
    camera.apply(centred);
    CHECK(centred.right_x > 10000 && centred.right_y == 0);

    const HostKey face[] = {HostKey::U, HostKey::O, HostKey::U};
    snapshot.keyboard = keyboard_input(std::span<const HostKey>(face, 3));
    snapshot.controller.right_stick = {};
    input = armigo::runtime::full_key_input(snapshot);
    CHECK((input.buttons & (1u << 3)) != 0);   // Y
    CHECK((input.buttons & (1u << 9)) != 0);   // ZR
    camera.button(0, true);  // mouse Y, already pressed on the keyboard
    camera.button(1, true);  // mouse ZR
    camera.apply(input);
    CHECK((input.buttons & (1u << 3)) != 0);
    CHECK((input.buttons & (1u << 9)) != 0);
    // U and O leave the right stick centred, under the dead zone, so the
    // mouse motion sampled above supplies it.
    CHECK(input.right_x > 10000 && input.right_y == 0);
}

// A disconnected or unsupported device contributes nothing.
void test_unusable_devices() {
    HostInputSnapshot snapshot;
    snapshot.controller.connected = true;
    snapshot.controller.buttons = button_mask(HostButton::South);
    snapshot.controller.left_stick = {1.0f, 0.0f};
    const auto input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.buttons == 0 && input.left_x == 0);
}

// Without focus the guest sees a released controller; what is still held when
// focus returns stays released until it is let go, then works again.
void test_focus() {
    using armigo::services::hid::HostFullKeyInput;
    armigo::runtime::FocusFilter filter;
    HostFullKeyInput running;
    running.buttons = (1u << 0) | (1u << 17);  // A, left stick up
    running.left_y = 32767;
    running.right_x = 20000;

    auto out = filter.apply(running, true);
    CHECK(out.buttons == running.buttons && out.left_y == 32767 && out.right_x == 20000);

    out = filter.apply(running, false);
    CHECK(out.buttons == 0 && out.left_x == 0 && out.left_y == 0 && out.right_x == 0 && out.right_y == 0);

    // Focus returns with the same input still reading as held.
    out = filter.apply(running, true);
    CHECK(out.buttons == 0 && out.left_y == 0 && out.right_x == 0);
    HostFullKeyInput b_pressed;
    b_pressed.buttons = (1u << 0) | (1u << 1) | (1u << 17);  // B is new
    b_pressed.left_y = 32767;
    out = filter.apply(b_pressed, true);
    CHECK(out.buttons == (1u << 1) && out.left_y == 0);

    // Released once, everything works again.
    out = filter.apply(HostFullKeyInput{}, true);
    CHECK(out.buttons == 0);
    out = filter.apply(running, true);
    CHECK(out.buttons == running.buttons && out.left_y == 32767 && out.right_x == 20000);

    // A small stick rest inside the dead zone is not held input.
    armigo::runtime::FocusFilter rest;
    HostFullKeyInput drift;
    drift.left_x = 1000;
    rest.apply(drift, false);
    out = rest.apply(drift, true);
    CHECK(out.left_x == 1000);
}

// The focus flag starts set so runs without a window keep their input.
void test_focus_flag() {
    CHECK(armigo::runtime::host_input_focus());
    armigo::runtime::set_host_input_focus(false);
    CHECK(!armigo::runtime::host_input_focus());
    armigo::runtime::set_host_input_focus(true);
    CHECK(armigo::runtime::host_input_focus());
}


// A controller's motion reaches the guest in the six-axis sensor's units:
// acceleration unchanged in G, rotation from radians to revolutions per
// second. A keyboard or an unusable controller has none, and a window
// without focus reports none.
void test_motion() {
    auto snapshot = connected();
    snapshot.controller.has_motion = true;
    snapshot.controller.acceleration = {0.5f, 0.0f, -0.75f};
    snapshot.controller.rotation_rate = {2.0f * 3.14159265f, 0.0f, -3.14159265f};
    const auto input = armigo::runtime::full_key_input(snapshot);
    CHECK(input.has_motion);
    CHECK(input.acceleration[0] == 0.5f && input.acceleration[2] == -0.75f);
    CHECK(std::fabs(input.angular_velocity[0] - 1.0f) < 1e-5f && std::fabs(input.angular_velocity[2] + 0.5f) < 1e-5f);

    auto keyboard_only = connected();
    keyboard_only.keyboard.has_motion = true;  // a keyboard's motion is not a controller's
    CHECK(!armigo::runtime::full_key_input(keyboard_only).has_motion);

    auto unusable = snapshot;
    unusable.controller.supported = false;
    CHECK(!armigo::runtime::full_key_input(unusable).has_motion);

    armigo::runtime::FocusFilter filter;
    CHECK(!filter.apply(input, false).has_motion);
}
} // namespace

int main() {
    test_buttons_by_position();
    test_triggers();
    test_sticks();
    test_controller_stick_shaping();
    test_keyboard_right_stick_mouse_and_face();
    test_unusable_devices();
    test_focus();
    test_focus_flag();
    test_motion();
    std::printf("Host controller failures: %d\n", failures);
    return failures ? 1 : 0;
}
