// ============================================================================
// Armigo Compatibility - Title Compatibility Semantics
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstdint>

namespace armigo::horizon {
class GuestBuffers;
}

namespace armigo::titles {

enum class FrameRateVariant {
    IntervalRatio60,
    FractionalCounter30,
    FractionalCountdown30,
    FractionalRepeat30,
    // The title divides its base rate by its own frame rate into the ratio its
    // clock steps by each frame, whatever time the frame took. A hook on that
    // division gives it a divisor from the measured frame period instead.
    RatioDivision,
};

// Every address is an offset supplied by the compatibility manifest. No title
// address or version-specific layout is compiled into this module.
struct FrameRateLayout {
    std::uint64_t framework_pointer = 0;
    std::uint64_t framework_period = 0;
    std::uint64_t override_flag = 0;
    std::uint64_t ratio = 0;
    std::uint64_t interval = 0;
    std::uint64_t base_interval = 0;
    // Optional; zero names none. The module offset of a float the title adds
    // to per-frame counters (for example controller hold times that drive
    // button repeat). It receives the measured period in 60 Hz frames, so
    // those counters advance with time rather than with presented frames.
    std::uint64_t frame_step = 0;
    FrameRateVariant variant = FrameRateVariant::IntervalRatio60;
    // FractionalCounter30 interprets the selected object register as a state
    // object, not an interval object. It advances a float in nominal 30 Hz
    // units only in active_state. It uses the native float seconds conversion,
    // caps at 50 ms, and imposes no minimum positive period. The countdown
    // variant subtracts the same step and saturates at positive zero.
    std::uint32_t object_register = 0;
    std::uint64_t state = 0;
    std::uint32_t active_state = 0;
    std::uint64_t counter = 0;
    // The repeat variant rearms an expired countdown on the next callback
    // while this guest-owned byte is nonzero. Initial delay and release
    // remain guest-owned; repeat_frames uses the same authored 30 Hz units.
    std::uint64_t repeat_flag = 0;
    std::uint32_t repeat_frames = 0;
};

// A zero-initialized guest float records progress through a repeating schedule.
// The timing source is the measured 60 Hz frame step published by this module.
struct FramePhaseLayout {
    std::uint64_t frame_step = 0;
    std::uint64_t counter = 0;
    std::uint32_t phase_count = 0;
};

struct FramePhaseDispatch {
    std::uint32_t first = 0;
    std::uint32_t count = 0;
    bool operator==(const FramePhaseDispatch&) const = default;
};

// Advance in nominal 30 Hz units and return every crossed phase, in order,
// wrapping at phase_count. The guest remains responsible for executing each
// phase's work. A failed access never publishes a dispatch; zero step is a no-op.
bool advance_frame_phases(const FramePhaseLayout& layout, std::uint64_t module_base,
                          std::uint64_t object, const horizon::GuestBuffers& buffers,
                          FramePhaseDispatch& dispatch) noexcept;

// IntervalRatio60 publishes the measured whole-frame ratio before the guest's original
// updateInterval call. False reports an address overflow or guest-memory
// access failure; an override, a zero interval/base, a null framework, or a
// nonpositive period is a successful no-op for the ratio. A layout's frame step
// is written whenever a positive period exists, including under the override. The optional maximum-rate hint is
// nonzero only after the measured update succeeds; errors and no-ops clear it.
// RatioDivision: the divisor that turns the title's base rate into the
// measured frame step: 60 / step, where step is the period in 60 Hz frames
// with the same bounds as a published frame step (20-300 Hz). The ratio then
// becomes step * base / 60, so the title's time follows real time.
float ratio_division_divisor(double period_seconds) noexcept;

bool update_frame_rate(const FrameRateLayout& layout, std::uint64_t module_base,
                       std::uint64_t vfr_object, std::uint32_t interval,
                       const horizon::GuestBuffers& buffers,
                       std::uint32_t* maximum_rate = nullptr,
                       std::uint64_t counter_frequency = 19'200'000) noexcept;

}  // namespace armigo::titles
