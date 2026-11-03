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
#include <span>

#include "renderer/common/common.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Represents a mixing node, can be connected to a previous and next destination forming a chain
// that a certain mix buffer will pass through to output.
class SplitterDestinationData {
public:
    // Biquad filter parameter with float coefficients (SDK REV15+).
    // Defined here to avoid circular dependency with VoiceInfo.
    struct BiquadFilterParameter2 {
        /* 0x00 */ bool enabled;
        /* 0x01 */ u8 reserved1;
        /* 0x02 */ u8 reserved2;
        /* 0x03 */ u8 reserved3;
        /* 0x04 */ std::array<f32, 3> numerator;   // b0, b1, b2
        /* 0x10 */ std::array<f32, 2> denominator; // a1, a2 (a0 = 1)
    };
    static_assert(sizeof(BiquadFilterParameter2) == 0x18,
                  "BiquadFilterParameter2 has the wrong size!");

    // Legacy biquad filter parameter with fixed-point coefficients (SDK REV12+ for splitters).
    // Matches the old voice biquad format.
    struct BiquadFilterParameterLegacy {
        /* 0x00 */ bool enabled;
        /* 0x02 */ std::array<s16, 3> b; // numerator
        /* 0x08 */ std::array<s16, 2> a; // denominator (a0 = 1)
    };
    static_assert(sizeof(BiquadFilterParameterLegacy) == 0xC,
                  "BiquadFilterParameterLegacy has the wrong size!");

    struct InParameter {
        /* 0x00 */ u32 magic; // 'SNDD'
        /* 0x04 */ s32 id;
        /* 0x08 */ std::array<f32, MaxMixBuffers> mix_volumes;
        /* 0x68 */ u32 mix_id;
        /* 0x6C */ bool in_use;
        /* 0x6D */ bool reset_prev_volume;
    };
    static_assert(sizeof(InParameter) == 0x70,
                  "SplitterDestinationData::InParameter has the wrong size!");

    struct InParameterVersion2a {
        /* 0x00 */ u32 magic; // 'SNDD'
        /* 0x04 */ s32 id;
        /* 0x08 */ std::array<f32, MaxMixBuffers> mix_volumes;
        /* 0x68 */ u32 mix_id;
        /* 0x6C */ std::array<SplitterDestinationData::BiquadFilterParameterLegacy, MaxBiquadFilters>
            biquad_filters;
        /* 0x84 */ bool in_use;
        /* 0x85 */ bool reset_prev_volume; // only effective if supported
        /* 0x86 */ u8 reserved[10];
    };
    static_assert(sizeof(InParameterVersion2a) == 0x90,
                  "SplitterDestinationData::InParameterVersion2a has the wrong size!");

    struct InParameterVersion2b {
        /* 0x00 */ u32 magic; // 'SNDD'
        /* 0x04 */ s32 id;
        /* 0x08 */ std::array<f32, MaxMixBuffers> mix_volumes;
        /* 0x68 */ u32 mix_id;
        /* 0x6C */ std::array<SplitterDestinationData::BiquadFilterParameter2, MaxBiquadFilters>
            biquad_filters;
        /* 0x9C */ bool in_use;
        /* 0x9D */ bool reset_prev_volume;
        /* 0x9E */ u8 reserved[10];
    };
    static_assert(sizeof(InParameterVersion2b) == 0xA8,
                  "SplitterDestinationData::InParameterVersion2b has the wrong size!");

    SplitterDestinationData(s32 id);

    // Reset the mix volumes for this destination.
    void clear_mix_volume();

    // Get the id of this destination.
    s32 get_id() const;

    // Check if this destination is correctly configured.
    bool is_configured() const;

    // Get the mix id for this destination.
    s32 get_mix_id() const;

    // Get the current mix volume of a given index in this destination.
    f32 get_mix_volume(u32 index) const;

    // Get the current mix volumes for all mix buffers in this destination.
    std::span<f32> get_mix_volume();

    // Get the previous mix volume of a given index in this destination.
    f32 get_mix_volume_prev(u32 index) const;

    // Get the previous mix volumes for all mix buffers.
    std::span<f32> get_mix_volume_prev();

    // Update this destination.
    void update(const InParameter& params);

    // Mark this destination as needing its volumes updated.
    void mark_as_need_to_update_internal_state();

    // Copy current volumes to previous if an update is required.
    void update_internal_state();

    // Get the next destination in the mix chain.
    SplitterDestinationData* get_next() const;

    // Set the next destination in the mix chain.
    void set_next(SplitterDestinationData* next);

    // Get biquad filter parameters for this destination (REV15+ or mapped from REV12).
    std::span<BiquadFilterParameter2> get_biquad_filters();

    // Get const biquad filter parameters for this destination (REV15+ or mapped from REV12).
    std::span<const BiquadFilterParameter2> get_biquad_filters() const;

private:
    // Id of this destination
    const s32 id;
    // Mix id this destination represents
    s32 destination_id{UnusedMixId};
    // Current mix volumes
    std::array<f32, MaxMixBuffers> mix_volumes{0.0f};
    // Previous mix volumes
    std::array<f32, MaxMixBuffers> prev_mix_volumes{0.0f};
    // Biquad filter parameters (REV15+ or mapped from REV12)
    std::array<BiquadFilterParameter2, MaxBiquadFilters> biquad_filters{};
    // Next destination in the mix chain
    SplitterDestinationData* next{};
    // Is this destination in use?
    bool in_use{};
    // Does this destination need its volumes updated?
    bool need_update{};
};

} // namespace armigo::audio::renderer
