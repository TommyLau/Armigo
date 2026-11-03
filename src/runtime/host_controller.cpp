// ============================================================================
// Armigo Runtime - Host Controller Input
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/runtime/host_controller.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <utility>

namespace armigo::runtime {
namespace {

using platform::HostButton;

// NpadButton bits.
enum : std::uint64_t {
    npad_a = 1ull << 0, npad_b = 1ull << 1, npad_x = 1ull << 2, npad_y = 1ull << 3,
    npad_stick_l = 1ull << 4, npad_stick_r = 1ull << 5, npad_l = 1ull << 6, npad_r = 1ull << 7,
    npad_zl = 1ull << 8, npad_zr = 1ull << 9, npad_plus = 1ull << 10, npad_minus = 1ull << 11,
    npad_left = 1ull << 12, npad_up = 1ull << 13, npad_right = 1ull << 14, npad_down = 1ull << 15,
    npad_stick_l_left = 1ull << 16, npad_stick_l_up = 1ull << 17, npad_stick_l_right = 1ull << 18,
    npad_stick_l_down = 1ull << 19, npad_stick_r_left = 1ull << 20, npad_stick_r_up = 1ull << 21,
    npad_stick_r_right = 1ull << 22, npad_stick_r_down = 1ull << 23,
};

constexpr std::array<std::pair<HostButton, std::uint64_t>, 14> button_map{{
    {HostButton::South, npad_b}, {HostButton::East, npad_a}, {HostButton::West, npad_y},
    {HostButton::North, npad_x}, {HostButton::LeftShoulder, npad_l}, {HostButton::RightShoulder, npad_r},
    {HostButton::LeftStick, npad_stick_l}, {HostButton::RightStick, npad_stick_r},
    {HostButton::Menu, npad_plus}, {HostButton::Options, npad_minus}, {HostButton::Up, npad_up},
    {HostButton::Down, npad_down}, {HostButton::Left, npad_left}, {HostButton::Right, npad_right},
}};

// The stick of the device that deflects it further.
std::array<float, 2> stronger(const std::array<float, 2>& a, const std::array<float, 2>& b) {
    return a[0] * a[0] + a[1] * a[1] >= b[0] * b[0] + b[1] * b[1] ? a : b;
}

std::int32_t axis(float value) {
    return static_cast<std::int32_t>(std::lround(std::clamp(value, -1.0f, 1.0f) * 32767.0f));
}

// Direction bits of a stick past half deflection.
std::uint64_t direction_bits(const std::array<float, 2>& stick, std::uint64_t left, std::uint64_t up,
                             std::uint64_t right, std::uint64_t down) {
    std::uint64_t bits = 0;
    if (stick[0] < -0.5f) bits |= left;
    if (stick[0] > 0.5f) bits |= right;
    if (stick[1] > 0.5f) bits |= up;
    if (stick[1] < -0.5f) bits |= down;
    return bits;
}

} // namespace

services::hid::HostFullKeyInput full_key_input(const platform::HostInputSnapshot& snapshot) noexcept {
    const auto& pad = snapshot.controller;
    const auto& keys = snapshot.keyboard;
    const auto usable = [](const platform::HostInputDeviceState& device) {
        return device.connected && device.supported;
    };
    const std::uint64_t host = (usable(pad) ? pad.buttons : 0) | (usable(keys) ? keys.buttons : 0);
    services::hid::HostFullKeyInput input;
    for (const auto& [button, npad] : button_map)
        if (host & platform::button_mask(button)) input.buttons |= npad;
    const float left_trigger = std::max(usable(pad) ? pad.left_trigger : 0.0f, usable(keys) ? keys.left_trigger : 0.0f);
    const float right_trigger = std::max(usable(pad) ? pad.right_trigger : 0.0f, usable(keys) ? keys.right_trigger : 0.0f);
    if (left_trigger > 0.5f) input.buttons |= npad_zl;
    if (right_trigger > 0.5f) input.buttons |= npad_zr;
    const std::array<float, 2> idle{};
    const auto left = stronger(usable(pad) ? pad.left_stick : idle, usable(keys) ? keys.left_stick : idle);
    const auto right = stronger(usable(pad) ? pad.right_stick : idle, usable(keys) ? keys.right_stick : idle);
    input.left_x = axis(left[0]);
    input.left_y = axis(left[1]);
    input.right_x = axis(right[0]);
    input.right_y = axis(right[1]);
    input.buttons |= direction_bits(left, npad_stick_l_left, npad_stick_l_up, npad_stick_l_right, npad_stick_l_down);
    input.buttons |= direction_bits(right, npad_stick_r_left, npad_stick_r_up, npad_stick_r_right, npad_stick_r_down);
    // The controller's motion, if it has sensors: the frames match; the
    // six-axis sensor counts rotation in revolutions per second.
    if (usable(pad) && pad.has_motion) {
        constexpr float per_revolution = 1.0f / (2.0f * std::numbers::pi_v<float>);
        input.has_motion = true;
        input.has_gyroscope = pad.has_rotation_rate;
        input.acceleration = pad.acceleration;
        for (std::size_t axis = 0; axis < 3; ++axis) input.angular_velocity[axis] = pad.rotation_rate[axis] * per_revolution;
    }
    return input;
}

namespace {
std::uint64_t now_ns() {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
constexpr double camera_tau_ns = 50e6, camera_floor = 200, camera_gain = 2000, stick_max = 32767;
constexpr std::int32_t pad_deadzone = 3932;
}  // namespace

void MouseCamera::decay_locked(std::uint64_t now) {
    if (updated_ && now > updated_) {
        const double keep = std::exp(-double(now - updated_) / camera_tau_ns);
        x_ *= keep;
        y_ *= keep;
        if (std::hypot(x_, y_) < camera_floor) x_ = y_ = 0;
    }
    updated_ = now;
}

void MouseCamera::moved(double dx, double dy) {
    if (!std::isfinite(dx) || !std::isfinite(dy)) return;
    const std::lock_guard lock(mutex_);
    decay_locked(now_ns());
    // Pointer y grows downward; stick up is positive.
    x_ = std::clamp(x_ + dx * camera_gain, -stick_max, stick_max);
    y_ = std::clamp(y_ - dy * camera_gain, -stick_max, stick_max);
}

void MouseCamera::button(unsigned index, bool down) {
    const std::lock_guard lock(mutex_);
    if (index < 3) buttons_ = down ? (buttons_ | (1u << index)) : (buttons_ & ~(1u << index));
}

void MouseCamera::apply(services::hid::HostFullKeyInput& input) {
    const std::lock_guard lock(mutex_);
    decay_locked(now_ns());
    if (buttons_ & 1) input.buttons |= npad_y;
    if (buttons_ & 2) input.buttons |= npad_zr;
    if (std::hypot(double(input.right_x), double(input.right_y)) >= pad_deadzone) return;
    input.right_x = std::int32_t(std::lround(x_));
    input.right_y = std::int32_t(std::lround(y_));
    const std::array<float, 2> stick{float(x_ / stick_max), float(y_ / stick_max)};
    input.buttons |= direction_bits(stick, npad_stick_r_left, npad_stick_r_up, npad_stick_r_right, npad_stick_r_down);
}

MouseCamera& mouse_camera() {
    static MouseCamera camera;
    return camera;
}

namespace {
std::atomic<bool> host_focus{true};
constexpr std::uint64_t left_stick_bits = npad_stick_l_left | npad_stick_l_up | npad_stick_l_right | npad_stick_l_down;
constexpr std::uint64_t right_stick_bits = npad_stick_r_left | npad_stick_r_up | npad_stick_r_right | npad_stick_r_down;

bool deflected(std::int32_t x, std::int32_t y) {
    return std::hypot(double(x), double(y)) >= pad_deadzone;
}
}  // namespace

void set_host_input_focus(bool focused) noexcept {
    host_focus.store(focused, std::memory_order_relaxed);
}

bool host_input_focus() noexcept {
    return host_focus.load(std::memory_order_relaxed);
}

services::hid::HostFullKeyInput FocusFilter::apply(services::hid::HostFullKeyInput input, bool focused) noexcept {
    if (!focused) {
        focused_ = false;
        return {};
    }
    if (!focused_) {
        // Focus returned: what reads as held now was held across the loss.
        focused_ = true;
        held_buttons_ = input.buttons & ~(left_stick_bits | right_stick_bits);
        held_left_ = deflected(input.left_x, input.left_y);
        held_right_ = deflected(input.right_x, input.right_y);
    }
    held_buttons_ &= input.buttons;
    input.buttons &= ~held_buttons_;
    if (held_left_ && (held_left_ = deflected(input.left_x, input.left_y))) {
        input.left_x = input.left_y = 0;
        input.buttons &= ~left_stick_bits;
    }
    if (held_right_ && (held_right_ = deflected(input.right_x, input.right_y))) {
        input.right_x = input.right_y = 0;
        input.buttons &= ~right_stick_bits;
    }
    return input;
}

HostControllerPoller::HostControllerPoller(std::function<platform::HostInputSnapshot()> poll)
    : thread_([poll = std::move(poll)](std::stop_token stop) {
          FocusFilter focus;
          auto next = std::chrono::steady_clock::now();
          while (!stop.stop_requested()) {
              auto input = full_key_input(poll());
              mouse_camera().apply(input);
              input = focus.apply(input, host_input_focus());
              services::hid::set_host_full_key_input(input);
              next += std::chrono::milliseconds(5);
              std::this_thread::sleep_until(next);
          }
      }) {}

} // namespace armigo::runtime
