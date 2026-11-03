// ============================================================================
// Armigo Runtime - Host Controller Input
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/platform/host_input.h>
#include <armigo/services/hid.h>

#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

namespace armigo::runtime {

// Maps one host input snapshot to player one's controller as a Switch Pro
// Controller. Buttons map by position: South is B, East is A, West is Y,
// North is X; shoulders are L and R, a trigger pressed past half is ZL or ZR,
// stick clicks, Menu (+), Options (-) and the D-pad map to theirs. The
// controller and the keyboard combine: buttons of both count, and each stick
// takes whichever device deflects it further. A stick past half deflection
// also sets its direction bits, as the console's HID service does.
services::hid::HostFullKeyInput full_key_input(const platform::HostInputSnapshot& snapshot) noexcept;

// Mouse camera: pointer motion is integrated into a right-stick deflection
// that decays exponentially (50 ms) and snaps to centre below 200, so the
// camera stops when the mouse stops. The left button is Y, the right ZR.
// The keyboard and controller right sticks are already combined by
// full_key_input (the further deflection wins). That stick, past 12%
// deflection, overrides the mouse. Thread-safe.
class MouseCamera {
public:
    void moved(double dx, double dy);
    void button(unsigned index, bool down);
    // Applies the camera and buttons to one controller sample.
    void apply(services::hid::HostFullKeyInput& input);

private:
    void decay_locked(std::uint64_t now);
    std::mutex mutex_;
    double x_ = 0, y_ = 0;
    std::uint64_t updated_ = 0;
    unsigned buttons_ = 0;
};
MouseCamera& mouse_camera();

// Whether the window that owns host input has focus (key status). Starts
// focused, so runs without a window are unaffected. Thread-safe.
void set_host_input_focus(bool focused) noexcept;
bool host_input_focus() noexcept;

// Applied to every controller sample. While the window does not have focus
// the guest sees a released controller: no buttons, centred sticks. Input
// still held when focus returns stays released until it is let go once,
// since a key or stick released while another application had focus can
// still read as held.
class FocusFilter {
public:
    services::hid::HostFullKeyInput apply(services::hid::HostFullKeyInput input, bool focused) noexcept;

private:
    bool focused_ = true;
    std::uint64_t held_buttons_ = 0;
    bool held_left_ = false, held_right_ = false;
};

// Polls host input every 5 ms on its own thread and hands each sample to the
// HID service until destroyed.
class HostControllerPoller {
public:
    explicit HostControllerPoller(std::function<platform::HostInputSnapshot()> poll);

private:
    std::jthread thread_;
};

} // namespace armigo::runtime
