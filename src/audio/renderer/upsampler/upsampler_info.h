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

#include "renderer/common/common.h"
#include "renderer/upsampler/upsampler_state.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class UpsamplerManager;

// Manages information needed to upsample a mix buffer.
struct UpsamplerInfo {
    // States used by the AudioRenderer across calls.
    std::array<UpsamplerState, MaxChannels> states{};
    // Pointer to the manager
    UpsamplerManager* manager{};
    // Pointer to the samples to be upsampled
    CpuAddr samples_pos{};
    // Target number of samples to upsample to
    u32 sample_count{};
    // Number of channels to upsample
    u32 input_count{};
    // Is this upsampler enabled?
    bool enabled{};
    // Mix buffer indexes to be upsampled
    std::array<s16, MaxChannels> inputs{};
};

} // namespace armigo::audio::renderer
