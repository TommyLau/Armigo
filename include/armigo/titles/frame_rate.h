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

// IntervalRatio60 publishes the measured whole-frame ratio before the guest's original
// updateInterval call. False reports an address overflow or guest-memory
// access failure; an override, a zero interval/base, a null framework, or a
// nonpositive period is a successful no-op for the ratio. A layout's frame step
// is written whenever a positive period exists, including under the override. The optional maximum-rate hint is
// nonzero only after the measured update succeeds; errors and no-ops clear it.
bool update_frame_rate(const FrameRateLayout& layout, std::uint64_t module_base,
                       std::uint64_t vfr_object, std::uint32_t interval,
                       const horizon::GuestBuffers& buffers,
                       std::uint32_t* maximum_rate = nullptr) noexcept;

}  // namespace armigo::titles
