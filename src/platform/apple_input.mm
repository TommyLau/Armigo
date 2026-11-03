// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <armigo/platform/apple_input.h>

#import <GameController/GameController.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <mutex>

namespace armigo::platform {
namespace {

std::mutex touch_mutex;
HostInputDeviceState touch_state;

// The physical keys Armigo maps, as GameController key codes (GCKeyCodes.h).
// Everything else a keyboard may report is ignored.
struct KeyMapping {
    GCKeyCode code;
    HostKey key;
};

const KeyMapping kKeyMappings[] = {
    {GCKeyCodeKeyW, HostKey::W},
    {GCKeyCodeKeyA, HostKey::A},
    {GCKeyCodeKeyS, HostKey::S},
    {GCKeyCodeKeyD, HostKey::D},
    {GCKeyCodeUpArrow, HostKey::Up},
    {GCKeyCodeDownArrow, HostKey::Down},
    {GCKeyCodeLeftArrow, HostKey::Left},
    {GCKeyCodeRightArrow, HostKey::Right},
    {GCKeyCodeKeyE, HostKey::E},
    {GCKeyCodeLeftShift, HostKey::Shift},
    {GCKeyCodeRightShift, HostKey::Shift},
    {GCKeyCodeSpacebar, HostKey::Space},
    {GCKeyCodeKeyQ, HostKey::Q},
    {GCKeyCodeReturnOrEnter, HostKey::Enter},
    {GCKeyCodeKeypadEnter, HostKey::Enter},
    {GCKeyCodeTab, HostKey::Tab},
    {GCKeyCodeKeyF, HostKey::F},
    {GCKeyCodeKeyR, HostKey::R},
    {GCKeyCodeKeyC, HostKey::C},
    {GCKeyCodeKeyV, HostKey::V},
    {GCKeyCodeKeyI, HostKey::I},
    {GCKeyCodeKeyJ, HostKey::J},
    {GCKeyCodeKeyK, HostKey::K},
    {GCKeyCodeKeyL, HostKey::L},
    {GCKeyCodeKeyU, HostKey::U},
    {GCKeyCodeKeyO, HostKey::O},
};

HostInputDeviceState poll_keyboard() noexcept {
    // The coalesced keyboard merges every attached keyboard; nil means none.
    GCKeyboard *keyboard = GCKeyboard.coalescedKeyboard;
    if (keyboard == nil) return HostInputDeviceState{};
    GCKeyboardInput *input = keyboard.keyboardInput;
    if (input == nil) {
        // Attached, but the keyboard profile is unavailable: report connected
        // and unsupported instead of a phantom working device.
        HostInputDeviceState state;
        state.connected = true;
        return state;
    }
    std::array<HostKey, std::size(kKeyMappings)> pressed{};
    std::size_t count = 0;
    // One capture per profile: every key below reads the same instant, so the
    // state is one coherent sample instead of sequential live reads.
    GCKeyboardInput *captured = [input capture];
    if (captured == nil) {
        // A refused capture reports unsupported rather than reading a
        // half-live profile as a phantom working device.
        HostInputDeviceState state;
        state.connected = true;
        return state;
    }
    if (captured.anyKeyPressed) {
        for (const KeyMapping &mapping : kKeyMappings) {
            GCDeviceButtonInput *button = [captured buttonForKeyCode:mapping.code];
            if (button.pressed) pressed[count++] = mapping.key;
        }
    }
    return keyboard_input(std::span<const HostKey>(pressed.data(), count));
}

HostInputDeviceState poll_controller() noexcept {
    NSArray<GCController *> *controllers = [GCController controllers];
    if (controllers.count == 0) return HostInputDeviceState{};
    GCController *controller = controllers.firstObject;
    HostInputDeviceState state;
    state.connected = true;
    GCExtendedGamepad *profile = controller.extendedGamepad;
    if (profile == nil) return state;  // connected, but the required profile is missing

    // One capture per profile: every element below reads the same instant, so
    // the state is one coherent sample instead of sequential live reads. This
    // capture is independent of the keyboard's; the poll is not atomic across
    // distinct devices.
    GCExtendedGamepad *captured = [profile capture];
    if (captured == nil) return state;  // refused capture: unsupported, not a phantom state

    // Extended-gamepad face buttons are physical positions: buttonA is the
    // bottom button, buttonB the right, buttonX the left, buttonY the top, on
    // every controller layout. They map directly onto HostButton positions.
    // A position is what the system reports: a system-level user remapping can
    // change which physical input lands on a position, and this layer maps
    // whatever position is reported without assuming a printed brand label.
    const auto pressed_mask = [](GCControllerButtonInput *button, HostButton flag) {
        return button.pressed ? button_mask(flag) : 0;
    };
    state.buttons |= pressed_mask(captured.buttonA, HostButton::South);
    state.buttons |= pressed_mask(captured.buttonB, HostButton::East);
    state.buttons |= pressed_mask(captured.buttonX, HostButton::West);
    state.buttons |= pressed_mask(captured.buttonY, HostButton::North);
    state.buttons |= pressed_mask(captured.leftShoulder, HostButton::LeftShoulder);
    state.buttons |= pressed_mask(captured.rightShoulder, HostButton::RightShoulder);
    state.buttons |= pressed_mask(captured.dpad.up, HostButton::Up);
    state.buttons |= pressed_mask(captured.dpad.down, HostButton::Down);
    state.buttons |= pressed_mask(captured.dpad.left, HostButton::Left);
    state.buttons |= pressed_mask(captured.dpad.right, HostButton::Right);
    state.buttons |= pressed_mask(captured.buttonMenu, HostButton::Menu);
    // Optional elements are nil on controllers that do not provide them.
    if (captured.buttonOptions != nil) {
        state.buttons |= pressed_mask(captured.buttonOptions, HostButton::Options);
    }
    if (captured.leftThumbstickButton != nil) {
        state.buttons |= pressed_mask(captured.leftThumbstickButton, HostButton::LeftStick);
    }
    if (captured.rightThumbstickButton != nil) {
        state.buttons |= pressed_mask(captured.rightThumbstickButton, HostButton::RightStick);
    }

    const auto pad_stick = [](GCControllerDirectionPad *pad) {
        return std::array<float, 2>{normalized_stick_component(pad.xAxis.value),
                                    normalized_stick_component(pad.yAxis.value)};
    };
    state.left_stick = pad_stick(captured.leftThumbstick);
    state.right_stick = pad_stick(captured.rightThumbstick);
    state.left_trigger = normalized_trigger_value(captured.leftTrigger.value);
    state.right_trigger = normalized_trigger_value(captured.rightTrigger.value);

    // Motion sensors, read live (a profile capture does not include them).
    // Some controllers start their sensors only on request. The controller's
    // frame and units (G, radians per second) are the ones HostInputDeviceState
    // uses.
    GCMotion *motion = controller.motion;
    if (motion != nil) {
        if (motion.sensorsRequireManualActivation && !motion.sensorsActive) motion.sensorsActive = YES;
        if (!motion.sensorsRequireManualActivation || motion.sensorsActive) {
            const auto finite = [](double value) { return std::isfinite(value) ? static_cast<float>(value) : 0.0f; };
            const GCAcceleration acceleration = motion.acceleration;
            state.has_motion = true;
            state.acceleration = {finite(acceleration.x), finite(acceleration.y), finite(acceleration.z)};
            if (motion.hasRotationRate) {
                state.has_rotation_rate = true;
                const GCRotationRate rate = motion.rotationRate;
                state.rotation_rate = {finite(rate.x), finite(rate.y), finite(rate.z)};
            }
        }
    }
    state.supported = true;
    return state;
}

}  // namespace

void set_apple_touch_input(const HostInputDeviceState& state) noexcept {
    const std::lock_guard lock(touch_mutex);
    touch_state = state;
}

HostInputSnapshot poll_apple_input() noexcept {
    HostInputSnapshot snapshot;
    // Callers may have no application lifecycle or run loop and so no outer
    // autorelease pool; the poll drains its framework temporaries itself.
    @autoreleasepool {
        snapshot.controller = poll_controller();
        snapshot.keyboard = poll_keyboard();
    }
    if (!snapshot.controller.connected || !snapshot.controller.supported) {
        const std::lock_guard lock(touch_mutex);
        if (touch_state.connected) snapshot.controller = touch_state;
    }
    return snapshot;
}

}  // namespace armigo::platform
