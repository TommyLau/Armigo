// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Polls Apple host input devices through the GameController framework.
//
// poll_apple_input returns one snapshot of the first connected game controller
// and of the coalesced keyboard:
//
// - The controller state describes the first object of
//   GCController.controllers. With no controller attached it is the default
//   (disconnected) state. A connected controller that does not offer the
//   extended-gamepad profile is reported as connected but not supported, with
//   no inputs, instead of a phantom working state.
// - The keyboard state describes GCKeyboard.coalescedKeyboard, which merges
//   every attached keyboard; with no keyboard it is the default state.
// - Each available profile (the extended gamepad, the keyboard) is captured
//   with -[GCPhysicalInputProfile capture] before any element is read, so one
//   device state holds one coherent sample rather than sequential live reads.
//   The captures are independent: the snapshot is coherent per device, not
//   atomic across distinct devices.
// - Face buttons, the D-pad, stick clicks, the menu and options buttons map by
//   physical position — the position the system reports, which a system-level
//   user remapping may change. No printed label, platform brand, or HID
//   convention is assumed. Sticks and triggers pass through the normalization
//   of host_input.h.
//
// The call retains framework objects only for its duration, drains its own
// autorelease pool, observes no notifications, and requires no run loop,
// window, discovery prompt, or application lifecycle — a caller needs no
// outer pool either. Physical-controller state is not retained between calls.
// The iOS touch view publishes a thread-safe virtual controller state; polling
// uses it only when no usable physical extended gamepad is present.
//
// This header is plain C++; all Apple types stay inside the implementation.

#pragma once

#include <armigo/platform/host_input.h>

namespace armigo::platform {

// Polls the first connected game controller and the coalesced keyboard once
// and returns their normalized states.
HostInputSnapshot poll_apple_input() noexcept;

// Publishes the iOS on-screen controller for the polling thread. A default
// state disconnects it. Physical extended gamepads take precedence.
void set_apple_touch_input(const HostInputDeviceState& state) noexcept;

}  // namespace armigo::platform
