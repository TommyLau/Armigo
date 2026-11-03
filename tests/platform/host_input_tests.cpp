// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/host_input.h>
#include <armigo/platform/touch_controls.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>

namespace {
using namespace armigo::platform;

int g_failed = 0;
int g_total = 0;

#define CHECK(condition)                                                                \
    do {                                                                                \
        ++g_total;                                                                      \
        if (!(condition)) {                                                             \
            ++g_failed;                                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);            \
        }                                                                               \
    } while (false)

constexpr float kInvSqrt2 = 0.70710678118654752440f;

bool same(float a, float b) {
    return std::fabs(a - b) <= 1e-6f;
}

bool same_stick(const std::array<float, 2>& stick, float x, float y) {
    return same(stick[0], x) && same(stick[1], y);
}

HostInputDeviceState keys(const HostKey* pressed, std::size_t count) {
    return keyboard_input(std::span<const HostKey>(pressed, count));
}

void test_default_state() {
    const HostInputDeviceState state;
    CHECK(!state.connected);
    CHECK(!state.supported);
    CHECK(state.buttons == 0);
    CHECK(same_stick(state.left_stick, 0.0f, 0.0f));
    CHECK(same_stick(state.right_stick, 0.0f, 0.0f));
    CHECK(state.left_trigger == 0.0f);
    CHECK(state.right_trigger == 0.0f);

    const HostInputSnapshot snapshot;
    CHECK(!snapshot.controller.connected);
    CHECK(!snapshot.keyboard.connected);
    CHECK(snapshot.controller.buttons == 0);
    CHECK(snapshot.keyboard.buttons == 0);
}

void test_button_flags_are_distinct_bits() {
    const HostButton all[] = {
        HostButton::South, HostButton::East, HostButton::West, HostButton::North,
        HostButton::LeftShoulder, HostButton::RightShoulder,
        HostButton::LeftStick, HostButton::RightStick,
        HostButton::Menu, HostButton::Options,
        HostButton::Up, HostButton::Down, HostButton::Left, HostButton::Right,
    };
    std::uint64_t expected = 0;
    std::uint64_t seen = 0;
    for (const HostButton button : all) {
        const std::uint64_t mask = button_mask(button);
        CHECK(mask != 0);
        CHECK((mask & (mask - 1)) == 0);  // exactly one bit
        CHECK((seen & mask) == 0);        // no two buttons share a bit
        CHECK(has_button(seen | mask, button));
        expected |= mask;
        seen |= mask;
    }
    CHECK(seen == expected);
    CHECK(!has_button(seen, static_cast<HostButton>(0)));
    CHECK(!has_button(0, HostButton::South));
}

void test_keyboard_dpad() {
    const HostKey up[] = {HostKey::Up};
    const HostInputDeviceState state = keys(up, 1);
    CHECK(state.connected);
    CHECK(state.supported);
    CHECK(state.buttons == button_mask(HostButton::Up));
    CHECK(same_stick(state.left_stick, 0.0f, 0.0f));

    const HostKey down[] = {HostKey::Down};
    CHECK(keys(down, 1).buttons == button_mask(HostButton::Down));
    const HostKey left[] = {HostKey::Left};
    CHECK(keys(left, 1).buttons == button_mask(HostButton::Left));
    const HostKey right[] = {HostKey::Right};
    CHECK(keys(right, 1).buttons == button_mask(HostButton::Right));

    const HostKey corner[] = {HostKey::Up, HostKey::Left};
    CHECK(keys(corner, 2).buttons ==
          (button_mask(HostButton::Up) | button_mask(HostButton::Left)));
}

void test_keyboard_stick_cardinals() {
    const HostKey w[] = {HostKey::W};
    CHECK(same_stick(keys(w, 1).left_stick, 0.0f, 1.0f));
    const HostKey a[] = {HostKey::A};
    CHECK(same_stick(keys(a, 1).left_stick, -1.0f, 0.0f));
    const HostKey s[] = {HostKey::S};
    CHECK(same_stick(keys(s, 1).left_stick, 0.0f, -1.0f));
    const HostKey d[] = {HostKey::D};
    CHECK(same_stick(keys(d, 1).left_stick, 1.0f, 0.0f));
}

void test_keyboard_stick_diagonals() {
    const HostKey north_east[] = {HostKey::W, HostKey::D};
    const HostInputDeviceState ne = keys(north_east, 2);
    CHECK(same_stick(ne.left_stick, kInvSqrt2, kInvSqrt2));
    const HostKey north_west[] = {HostKey::W, HostKey::A};
    CHECK(same_stick(keys(north_west, 2).left_stick, -kInvSqrt2, kInvSqrt2));
    const HostKey south_east[] = {HostKey::S, HostKey::D};
    CHECK(same_stick(keys(south_east, 2).left_stick, kInvSqrt2, -kInvSqrt2));
    const HostKey south_west[] = {HostKey::S, HostKey::A};
    CHECK(same_stick(keys(south_west, 2).left_stick, -kInvSqrt2, -kInvSqrt2));
}

void test_keyboard_stick_opposing_cancel() {
    const HostKey north_south[] = {HostKey::W, HostKey::S};
    CHECK(same_stick(keys(north_south, 2).left_stick, 0.0f, 0.0f));
    const HostKey east_west[] = {HostKey::A, HostKey::D};
    CHECK(same_stick(keys(east_west, 2).left_stick, 0.0f, 0.0f));
    // Every direction pressed at once cancels on both axes.
    const HostKey all[] = {HostKey::W, HostKey::A, HostKey::S, HostKey::D};
    CHECK(same_stick(keys(all, 4).left_stick, 0.0f, 0.0f));
    // Duplicated keys are idempotent and cancel the same way.
    const HostKey duplicate[] = {HostKey::A, HostKey::D, HostKey::A};
    CHECK(same_stick(keys(duplicate, 3).left_stick, 0.0f, 0.0f));
}

void test_keyboard_action_buttons() {
    // A Switch controller's buttons by position: E is A (East), Shift is B
    // (South), Space is X (North).
    const struct {
        HostKey key;
        HostButton button;
    } mappings[] = {
        {HostKey::E, HostButton::East},
        {HostKey::Shift, HostButton::South},
        {HostKey::Space, HostButton::North},
        {HostKey::F, HostButton::LeftShoulder},
        {HostKey::R, HostButton::RightShoulder},
        {HostKey::C, HostButton::LeftStick},
        {HostKey::V, HostButton::RightStick},
        {HostKey::Enter, HostButton::Menu},
        {HostKey::Tab, HostButton::Options},
    };
    for (const auto& mapping : mappings) {
        const HostKey pressed[] = {mapping.key};
        const HostInputDeviceState state = keys(pressed, 1);
        CHECK(state.connected);
        CHECK(state.supported);
        CHECK(state.buttons == button_mask(mapping.button));
        CHECK(same_stick(state.left_stick, 0.0f, 0.0f));
        CHECK(state.left_trigger == 0.0f);
        CHECK(state.right_trigger == 0.0f);
    }

    const HostKey all[] = {HostKey::E, HostKey::Shift, HostKey::Space, HostKey::F, HostKey::R,
                           HostKey::C, HostKey::V, HostKey::Enter, HostKey::Tab};
    CHECK(keys(all, 9).buttons ==
          (button_mask(HostButton::East) | button_mask(HostButton::South) |
           button_mask(HostButton::North) | button_mask(HostButton::LeftShoulder) |
           button_mask(HostButton::RightShoulder) | button_mask(HostButton::LeftStick) |
           button_mask(HostButton::RightStick) | button_mask(HostButton::Menu) |
           button_mask(HostButton::Options)));
}

void test_keyboard_triggers() {
    // Q fully presses the left trigger (ZL); O fully presses the right (ZR).
    const HostKey left[] = {HostKey::Q};
    CHECK(keys(left, 1).left_trigger == 1.0f && keys(left, 1).right_trigger == 0.0f);
    CHECK(keys(left, 1).buttons == 0);
    const HostKey right[] = {HostKey::O};
    CHECK(keys(right, 1).right_trigger == 1.0f && keys(right, 1).left_trigger == 0.0f);
    CHECK(keys(right, 1).buttons == 0);
    const HostKey both[] = {HostKey::Q, HostKey::O, HostKey::O};
    const HostInputDeviceState state = keys(both, 3);
    CHECK(state.left_trigger == 1.0f && state.right_trigger == 1.0f);
}

void test_keyboard_right_stick() {
    const HostKey i[] = {HostKey::I};
    CHECK(same_stick(keys(i, 1).right_stick, 0.0f, 1.0f));
    CHECK(same_stick(keys(i, 1).left_stick, 0.0f, 0.0f));
    const HostKey j[] = {HostKey::J};
    CHECK(same_stick(keys(j, 1).right_stick, -1.0f, 0.0f));
    const HostKey k[] = {HostKey::K};
    CHECK(same_stick(keys(k, 1).right_stick, 0.0f, -1.0f));
    const HostKey l[] = {HostKey::L};
    CHECK(same_stick(keys(l, 1).right_stick, 1.0f, 0.0f));

    const HostKey north_east[] = {HostKey::I, HostKey::L};
    CHECK(same_stick(keys(north_east, 2).right_stick, kInvSqrt2, kInvSqrt2));
    const HostKey south_west[] = {HostKey::K, HostKey::J};
    CHECK(same_stick(keys(south_west, 2).right_stick, -kInvSqrt2, -kInvSqrt2));

    const HostKey vertical[] = {HostKey::I, HostKey::K};
    CHECK(same_stick(keys(vertical, 2).right_stick, 0.0f, 0.0f));
    const HostKey horizontal[] = {HostKey::J, HostKey::L};
    CHECK(same_stick(keys(horizontal, 2).right_stick, 0.0f, 0.0f));
    const HostKey all[] = {HostKey::I, HostKey::J, HostKey::K, HostKey::L};
    CHECK(same_stick(keys(all, 4).right_stick, 0.0f, 0.0f));
    // Duplicated keys are idempotent and cancel the same way.
    const HostKey duplicate[] = {HostKey::J, HostKey::L, HostKey::J};
    CHECK(same_stick(keys(duplicate, 3).right_stick, 0.0f, 0.0f));

    // The two sticks are independent.
    const HostKey both[] = {HostKey::W, HostKey::D, HostKey::I};
    const HostInputDeviceState mixed = keys(both, 3);
    CHECK(same_stick(mixed.left_stick, kInvSqrt2, kInvSqrt2));
    CHECK(same_stick(mixed.right_stick, 0.0f, 1.0f));
}

void test_keyboard_y_and_duplicate_buttons() {
    const HostKey y[] = {HostKey::U, HostKey::U};
    const HostInputDeviceState state = keys(y, 2);
    CHECK(state.buttons == button_mask(HostButton::West));
    CHECK(state.right_trigger == 0.0f);
    CHECK(same_stick(state.right_stick, 0.0f, 0.0f));
}

void test_keyboard_mixed_keys() {
    // Stick keys and action keys combine without interfering.
    const HostKey held[] = {HostKey::W, HostKey::D, HostKey::Shift, HostKey::F};
    const HostInputDeviceState state = keys(held, 4);
    CHECK(same_stick(state.left_stick, kInvSqrt2, kInvSqrt2));
    CHECK(state.buttons ==
          (button_mask(HostButton::South) | button_mask(HostButton::LeftShoulder)));
}

void test_keyboard_idle() {
    const std::span<const HostKey> none;
    const HostInputDeviceState state = keyboard_input(none);
    // The mapping describes the keyboard device itself, so an idle keyboard is
    // still connected and supported; a poller with no keyboard returns a
    // default state instead of calling the mapping.
    CHECK(state.connected);
    CHECK(state.supported);
    CHECK(state.buttons == 0);
    CHECK(same_stick(state.left_stick, 0.0f, 0.0f));
    CHECK(state.left_trigger == 0.0f);
}

void test_stick_component_normalization() {
    CHECK(normalized_stick_component(0.0f) == 0.0f);
    CHECK(normalized_stick_component(0.5f) == 0.5f);
    CHECK(normalized_stick_component(-0.5f) == -0.5f);
    CHECK(normalized_stick_component(1.0f) == 1.0f);
    CHECK(normalized_stick_component(-1.0f) == -1.0f);
    CHECK(normalized_stick_component(1.5f) == 1.0f);
    CHECK(normalized_stick_component(-1.5f) == -1.0f);
    CHECK(normalized_stick_component(std::numeric_limits<float>::max()) == 1.0f);
    CHECK(normalized_stick_component(-std::numeric_limits<float>::max()) == -1.0f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(normalized_stick_component(nan) == 0.0f);
    CHECK(normalized_stick_component(inf) == 0.0f);
    CHECK(normalized_stick_component(-inf) == 0.0f);
}

void test_trigger_normalization() {
    CHECK(normalized_trigger_value(0.0f) == 0.0f);
    CHECK(normalized_trigger_value(0.25f) == 0.25f);
    CHECK(normalized_trigger_value(1.0f) == 1.0f);
    CHECK(normalized_trigger_value(1.5f) == 1.0f);
    CHECK(normalized_trigger_value(-0.25f) == 0.0f);
    CHECK(normalized_trigger_value(std::numeric_limits<float>::max()) == 1.0f);
    CHECK(normalized_trigger_value(-std::numeric_limits<float>::max()) == 0.0f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(normalized_trigger_value(nan) == 0.0f);
    CHECK(normalized_trigger_value(inf) == 0.0f);
    CHECK(normalized_trigger_value(-inf) == 0.0f);
}

void test_touch_layout() {
    const auto ipad = make_touch_layout(1133, 744, {});
    const auto& left = ipad[static_cast<std::size_t>(TouchControl::left_stick)];
    CHECK(left.x == 95 && left.y == 664 && left.radius == 58);
    const auto& a = ipad[static_cast<std::size_t>(TouchControl::a)];
    CHECK(a.x == 1084 && a.y == 529 && a.radius == 22);
    CHECK(touch_control_at(ipad, a.x, a.y) == static_cast<int>(TouchControl::a));
    CHECK(touch_control_at(ipad, 566, 400) == -1);

    const auto phone = make_touch_layout(844, 390, {60, 0, 60, 21});
    const auto& phone_left = phone[static_cast<std::size_t>(TouchControl::left_stick)];
    CHECK(phone_left.x == 155 && phone_left.y == 289);
    const auto& zl = phone[static_cast<std::size_t>(TouchControl::zl)];
    CHECK(zl.x == 110 && zl.y == 24 && zl.width == 64 && zl.height == 30);
    CHECK(touch_control_at(phone, zl.x, zl.y) == static_cast<int>(TouchControl::zl));

    const auto small = make_touch_layout(667, 320, {});
    const auto& small_stick = small[static_cast<std::size_t>(TouchControl::left_stick)];
    CHECK(small_stick.radius > 47 && small_stick.radius < 48);
    for (const auto& control : small) {
        CHECK(control.x >= 0 && control.x <= 667);
        CHECK(control.y >= 0 && control.y <= 320);
    }
}

void test_touch_input_state() {
    const auto layout = make_touch_layout(1133, 744, {});
    const auto& stick = layout[static_cast<std::size_t>(TouchControl::left_stick)];
    const auto right = clamped_touch_stick(stick, stick.x + 1000, stick.y);
    CHECK(same_stick(right, 1.0f, 0.0f));
    const auto diagonal = clamped_touch_stick(stick, stick.x + 58, stick.y - 58);
    CHECK(same_stick(diagonal, kInvSqrt2, kInvSqrt2));

    std::array<unsigned, touch_control_count> held{};
    held[static_cast<std::size_t>(TouchControl::a)] = 1;
    held[static_cast<std::size_t>(TouchControl::b)] = 2;
    held[static_cast<std::size_t>(TouchControl::zl)] = 1;
    held[static_cast<std::size_t>(TouchControl::plus)] = 1;
    const auto state = touch_input_state(held, {right, diagonal});
    CHECK(state.connected && state.supported);
    CHECK(state.buttons == (button_mask(HostButton::East) | button_mask(HostButton::South) |
                            button_mask(HostButton::Menu)));
    CHECK(state.left_trigger == 1.0f && state.right_trigger == 0.0f);
    CHECK(same_stick(state.left_stick, 1.0f, 0.0f));
    CHECK(same_stick(state.right_stick, kInvSqrt2, kInvSqrt2));
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep reports if a defect crashes the process
    test_default_state();
    test_button_flags_are_distinct_bits();
    test_keyboard_dpad();
    test_keyboard_stick_cardinals();
    test_keyboard_stick_diagonals();
    test_keyboard_stick_opposing_cancel();
    test_keyboard_action_buttons();
    test_keyboard_triggers();
    test_keyboard_right_stick();
    test_keyboard_y_and_duplicate_buttons();
    test_keyboard_mixed_keys();
    test_keyboard_idle();
    test_stick_component_normalization();
    test_trigger_normalization();
    test_touch_layout();
    test_touch_input_state();
    std::printf("%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
