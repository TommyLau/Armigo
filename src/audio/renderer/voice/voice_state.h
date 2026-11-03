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
#include "renderer/common/types.h"
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {
// Holds a state for a voice. One is kept host-side, and one is used by the AudioRenderer,
// host-side is updated on the next iteration.
struct VoiceState {
    // State of the voice's biquad filter.
    struct BiquadFilterState {
        s64 s0;
        s64 s1;
        s64 s2;
        s64 s3;
    };

    // Context for ADPCM decoding.
    struct AdpcmContext {
        u16 header;
        s16 yn0;
        s16 yn1;
    };

    // Number of samples played
    u64 played_sample_count;
    // Current offset from the starting offset
    u32 offset;
    // Currently active wavebuffer index
    u32 wave_buffer_index;
    // Array of which wavebuffers are currently valid

    std::array<bool, MaxWaveBuffers> wave_buffer_valid;
    // Number of wavebuffers consumed, given back to the game
    u32 wave_buffers_consumed;
    // History of samples, used for rate conversion

    std::array<s16, MaxWaveBuffers * 2> sample_history;
    // Current read fraction, used for resampling
    FixedPoint<49, 15> fraction;
    // Current adpcm context
    AdpcmContext adpcm_context;
    // Current biquad states, used when filtering

    std::array<std::array<BiquadFilterState, MaxBiquadFilters>, MaxBiquadFilters> biquad_states;
    // Previous samples
    std::array<s32, MaxMixBuffers> previous_samples;
    // Unused
    u32 external_context_size;
    // Unused
    bool external_context_enabled;
    // Was this voice dropped?
    bool voice_dropped;
    // Number of times the wavebuffer has looped
    s32 loop_count;
};

} // namespace armigo::audio::renderer
