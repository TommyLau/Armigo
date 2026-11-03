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

namespace armigo::titles::title_01007ef00011e000 {

// Every member is an offset supplied by the compatibility manifest. No title
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
};

// Publishes the measured whole-frame ratio before the guest's original
// updateInterval call. False reports an address overflow or guest-memory
// access failure; an override, a zero interval/base, a null framework, or a
// nonpositive period is a successful no-op for the ratio. A layout's frame step
// is written whenever a positive period exists, including under the override. The optional maximum-rate hint is
// nonzero only after the measured update succeeds; errors and no-ops clear it.
bool update_frame_rate(const FrameRateLayout& layout, std::uint64_t module_base,
                       std::uint64_t vfr_object, std::uint32_t interval,
                       const horizon::GuestBuffers& buffers,
                       std::uint32_t* maximum_rate = nullptr) noexcept;

}  // namespace armigo::titles::title_01007ef00011e000
