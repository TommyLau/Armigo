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
#include "renderer/effect/effect_info_base.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Auxiliary Buffer used for Aux commands.
// Send and return buffers are available (names from the game's perspective).
// Send is read by the host, containing a buffer of samples to be used for whatever purpose.
// Return is written by the host, writing a mix buffer back to the game.
// This allows the game to use pre-processed samples skipping the other render processing,
// and to examine or modify what the audio renderer has generated.
class AuxInfo : public EffectInfoBase {
public:
    struct ParameterVersion1 {
        /* 0x00 */ std::array<s8, MaxMixBuffers> inputs;
        /* 0x18 */ std::array<s8, MaxMixBuffers> outputs;
        /* 0x30 */ u32 mix_buffer_count;
        /* 0x34 */ u32 sample_rate;
        /* 0x38 */ u32 count_max;
        /* 0x3C */ u32 mix_buffer_count_max;
        /* 0x40 */ CpuAddr send_buffer_info_address;
        /* 0x48 */ CpuAddr send_buffer_address;
        /* 0x50 */ CpuAddr return_buffer_info_address;
        /* 0x58 */ CpuAddr return_buffer_address;
        /* 0x60 */ u32 mix_buffer_sample_size;
        /* 0x64 */ u32 sample_count;
        /* 0x68 */ u32 mix_buffer_sample_count;
    };
    static_assert(sizeof(ParameterVersion1) <= sizeof(EffectInfoBase::InParameterVersion1),
                  "AuxInfo::ParameterVersion1 has the wrong size!");

    struct ParameterVersion2 {
        /* 0x00 */ std::array<s8, MaxMixBuffers> inputs;
        /* 0x18 */ std::array<s8, MaxMixBuffers> outputs;
        /* 0x30 */ u32 mix_buffer_count;
        /* 0x34 */ u32 sample_rate;
        /* 0x38 */ u32 count_max;
        /* 0x3C */ u32 mix_buffer_count_max;
        /* 0x40 */ CpuAddr send_buffer_info_address;
        /* 0x48 */ CpuAddr send_buffer_address;
        /* 0x50 */ CpuAddr return_buffer_info_address;
        /* 0x58 */ CpuAddr return_buffer_address;
        /* 0x60 */ u32 mix_buffer_sample_size;
        /* 0x64 */ u32 sample_count;
        /* 0x68 */ u32 mix_buffer_sample_count;
    };
    static_assert(sizeof(ParameterVersion2) <= sizeof(EffectInfoBase::InParameterVersion2),
                  "AuxInfo::ParameterVersion2 has the wrong size!");

    struct AuxInfoDsp {
        /* 0x00 */ u32 read_offset;
        /* 0x04 */ u32 write_offset;
        /* 0x08 */ u32 lost_sample_count;
        /* 0x0C */ u32 total_sample_count;
        /* 0x10 */ char unk10[0x30];
    };
    static_assert(sizeof(AuxInfoDsp) == 0x40, "AuxInfo::AuxInfoDsp has the wrong size!");

    struct AuxBufferInfo {
        /* 0x00 */ AuxInfoDsp cpu_info;
        /* 0x40 */ AuxInfoDsp dsp_info;
    };
    static_assert(sizeof(AuxBufferInfo) == 0x80, "AuxInfo::AuxBufferInfo has the wrong size!");

    // Update the info with new parameters, version 1.
    void update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion1& in_params,
                const PoolMapper& pool_mapper) override;

    // Update the info with new parameters, version 2.
    void update(BehaviorInfo::ErrorInfo& error_info, const InParameterVersion2& in_params,
                const PoolMapper& pool_mapper) override;

    // Update the info after command generation. Usually only changes its state.
    void update_for_command_generation() override;

    // Initialize a new result state. Version 2 only, unused.
    void initialize_result_state(EffectResultState& result_state) override;

    // Update the host-side state with the ADSP-side state. Version 2 only, unused.
    void update_result_state(EffectResultState& cpu_state, EffectResultState& dsp_state) override;

    // Get a workbuffer assigned to this effect with the given index.
    CpuAddr get_workbuffer(s32 index) override;
};

} // namespace armigo::audio::renderer
