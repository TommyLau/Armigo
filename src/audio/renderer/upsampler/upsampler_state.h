// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <array>

#include "renderer/common/types.h"
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {
// Upsampling state used by the AudioRenderer across calls.
struct UpsamplerState {
    static constexpr u16 HistorySize = 20;

    // Source data to target data ratio. E.g 48'000/32'000 = 1.5
    FixedPoint<16, 16> ratio;
    // Sample history
    std::array<FixedPoint<24, 8>, HistorySize> history;
    // Size of the sinc coefficient window
    u16 window_size;
    // Read index for the history
    u16 history_output_index;
    // Write index for the history
    u16 history_input_index;
    // Start offset within the history, fixed to 0
    u16 history_start_index;
    // Ebd offset within the history, fixed to HistorySize
    u16 history_end_index;
    // Is this state initialized?
    bool initialized;
    // Index of the current sample.
    // E.g 16K -> 48K has a ratio of 3, so this will be 0-2.
    // See the Upsample command in the AudioRenderer for more information.
    u8 sample_index;
};

} // namespace armigo::audio::renderer
