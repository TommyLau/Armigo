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
#include <bitset>

#include "renderer/common/common.h"
#include "renderer/common/wave_buffer.h"
#include "renderer/behavior/behavior_info.h"
#include "renderer/memory/address_info.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class PoolMapper;
class VoiceContext;
struct VoiceState;

// Represents one voice. Voices are essentially noises, and they can be further mixed and have
// effects applied to them, but voices are the basis of all sounds.
class VoiceInfo {
public:
    enum class ServerPlayState {
        Started,
        Stopped,
        RequestStop,
        Paused,
    };

    struct Flags {
        u8 IsVoicePlayedSampleCountResetAtLoopPointSupported : 1;
        u8 IsVoicePitchAndSrcSkippedSupported : 1;
    };

    // A wavebuffer contains information on the data source buffers.
    struct WaveBuffer {
        void copy(WaveBufferVersion1& other) {
            other.buffer = buffer_address.get_reference(true);
            other.buffer_size = buffer_address.get_size();
            other.start_offset = start_offset;
            other.end_offset = end_offset;
            other.loop = loop;
            other.stream_ended = stream_ended;

            if (context_address.get_cpu_addr()) {
                other.context = context_address.get_reference(true);
                other.context_size = context_address.get_size();
            } else {
                other.context = CpuAddr(0);
                other.context_size = 0;
            }
        }

        void copy(WaveBufferVersion2& other) {
            other.buffer = buffer_address.get_reference(true);
            other.buffer_size = buffer_address.get_size();
            other.start_offset = start_offset;
            other.end_offset = end_offset;
            other.loop_start_offset = loop_start_offset;
            other.loop_end_offset = loop_end_offset;
            other.loop = loop;
            other.loop_count = loop_count;
            other.stream_ended = stream_ended;

            if (context_address.get_cpu_addr()) {
                other.context = context_address.get_reference(true);
                other.context_size = context_address.get_size();
            } else {
                other.context = CpuAddr(0);
                other.context_size = 0;
            }
        }

        void initialize() {
            buffer_address.setup(0, 0);
            context_address.setup(0, 0);
            start_offset = 0;
            end_offset = 0;
            loop = false;
            stream_ended = false;
            sent_to_DSP = true;
            loop_start_offset = 0;
            loop_end_offset = 0;
            loop_count = 0;
        }
        // Game memory address of the wavebuffer data
        AddressInfo buffer_address{0, 0};
        // Context for decoding, used for ADPCM
        AddressInfo context_address{0, 0};
        // Starting offset for the wavebuffer
        u32 start_offset{};
        // Ending offset the wavebuffer
        u32 end_offset{};
        // Should this wavebuffer loop?
        bool loop{};
        // Has this wavebuffer ended?
        bool stream_ended{};
        // Has this wavebuffer been sent to the AudioRenderer?
        bool sent_to_DSP{true};
        // Starting offset when looping, can differ from start_offset
        u32 loop_start_offset{};
        // Ending offset when looping, can differ from end_offset
        u32 loop_end_offset{};
        // Number of times to loop this wavebuffer
        s32 loop_count{};
    };

    struct WaveBufferInternal {
        /* 0x00 */ CpuAddr address;
        /* 0x08 */ u64 size;
        /* 0x10 */ s32 start_offset;
        /* 0x14 */ s32 end_offset;
        /* 0x18 */ bool loop;
        /* 0x19 */ bool stream_ended;
        /* 0x1A */ bool sent_to_DSP;
        /* 0x1C */ s32 loop_count;
        /* 0x20 */ CpuAddr context_address;
        /* 0x28 */ u64 context_size;
        /* 0x30 */ u32 loop_start;
        /* 0x34 */ u32 loop_end;
    };
    static_assert(sizeof(WaveBufferInternal) == 0x38,
                  "VoiceInfo::WaveBufferInternal has the wrong size!");

    struct BiquadFilterParameter {
        /* 0x00 */ bool enabled;
        /* 0x02 */ std::array<s16, 3> b;
        /* 0x08 */ std::array<s16, 2> a;
    };
    static_assert(sizeof(BiquadFilterParameter) == 0xC,
                  "VoiceInfo::BiquadFilterParameter has the wrong size!");

    struct BiquadFilterParameter2 {
        /* 0x00 */ bool enabled;
        /* 0x01 */ u8 reserved1;
        /* 0x02 */ u8 reserved2;
        /* 0x03 */ u8 reserved3;
        /* 0x04 */ std::array<f32, 3> numerator;   // b0, b1, b2
        /* 0x10 */ std::array<f32, 2> denominator; // a1, a2 (a0 = 1)
    };
    static_assert(sizeof(BiquadFilterParameter2) == 0x18,
                  "VoiceInfo::BiquadFilterParameter2 has the wrong size!");

    struct InParameter {
        /* 0x000 */ u32 id;
        /* 0x004 */ u32 node_id;
        /* 0x008 */ bool is_new;
        /* 0x009 */ bool in_use;
        /* 0x00A */ PlayState play_state;
        /* 0x00B */ SampleFormat sample_format;
        /* 0x00C */ u32 sample_rate;
        /* 0x010 */ s32 priority;
        /* 0x014 */ s32 sort_order;
        /* 0x018 */ u32 channel_count;
        /* 0x01C */ f32 pitch;
        /* 0x020 */ f32 volume;
        /* 0x024 */ std::array<BiquadFilterParameter, MaxBiquadFilters> biquads;
        /* 0x03C */ u32 wave_buffer_count;
        /* 0x040 */ u16 wave_buffer_index;
        /* 0x042 */ char unk042[0x6];
        /* 0x048 */ CpuAddr src_data_address;
        /* 0x050 */ u64 src_data_size;
        /* 0x058 */ u32 mix_id;
        /* 0x05C */ u32 splitter_id;
        /* 0x060 */ std::array<WaveBufferInternal, MaxWaveBuffers> wave_buffer_internal;
        /* 0x140 */ std::array<u32, MaxChannels> channel_resource_ids;
        /* 0x158 */ bool clear_voice_drop;
        /* 0x159 */ u8 flush_buffer_count;
        /* 0x15A */ char unk15A[0x2];
        /* 0x15C */ Flags flags;
        /* 0x15D */ char unk15D[0x1];
        /* 0x15E */ SrcQuality src_quality;
        /* 0x15F */ char unk15F[0x11];
    };
    static_assert(sizeof(InParameter) == 0x170, "VoiceInfo::InParameter has the wrong size!");

    struct InParameter2 {
        /* 0x000 */ u32 id;
        /* 0x004 */ u32 node_id;
        /* 0x008 */ bool is_new;
        /* 0x009 */ bool in_use;
        /* 0x00A */ PlayState play_state;
        /* 0x00B */ SampleFormat sample_format;
        /* 0x00C */ u32 sample_rate;
        /* 0x010 */ s32 priority;
        /* 0x014 */ s32 sort_order;
        /* 0x018 */ u32 channel_count;
        /* 0x01C */ f32 pitch;
        /* 0x020 */ f32 volume;
        /* 0x024 */ std::array<BiquadFilterParameter2, MaxBiquadFilters> biquads;
        /* 0x054 */ u32 wave_buffer_count;
        /* 0x058 */ u32 wave_buffer_index;
        /* 0x05C */ u32 reserved1;
        /* 0x060 */ CpuAddr src_data_address;
        /* 0x068 */ u64 src_data_size;
        /* 0x070 */ u32 mix_id;
        /* 0x074 */ u32 splitter_id;
        /* 0x078 */ std::array<WaveBufferInternal, MaxWaveBuffers> wave_buffer_internal;
        /* 0x158 */ std::array<s32, MaxChannels> channel_resource_ids;
        /* 0x170 */ bool clear_voice_drop;
        /* 0x171 */ u8 flush_buffer_count;
        /* 0x172 */ u16 reserved2;
        /* 0x174 */ Flags flags;
        /* 0x175 */ u8 reserved3;
        /* 0x176 */ SrcQuality src_quality;
        /* 0x177 */ u8 reserved4;
        /* 0x178 */ u32 external_context;
        /* 0x17C */ u32 external_context_size;
        /* 0x180 */ u32 reserved5;
        /* 0x184 */ u32 reserved6;
    };
    static_assert(sizeof(InParameter2) == 0x188, "VoiceInfo::InParameter2 has the wrong size!");

    struct OutStatus {
        /* 0x00 */ u64 played_sample_count;
        /* 0x08 */ u32 wave_buffers_consumed;
        /* 0x0C */ bool voice_dropped;
    };
    static_assert(sizeof(OutStatus) == 0x10, "OutStatus::InParameter has the wrong size!");

    VoiceInfo();

    // Initialize this voice.
    void initialize();

    // Does this voice need an update?
    bool should_update_parameters(const InParameter& params) const;

    // Update the parameters of this voice.
    void update_parameters(BehaviorInfo::ErrorInfo& error_info, const InParameter& params,
                          const PoolMapper& pool_mapper, const BehaviorInfo& behavior);

    // Update the current play state.
    void update_play_state(PlayState state);

    // Update the current sample rate conversion quality.
    void update_src_quality(SrcQuality quality);

    // Update all wavebuffers.
    void update_wave_buffers(std::span<std::array<BehaviorInfo::ErrorInfo, 2>> error_infos,
                           u32 error_count, const InParameter& params,
                           std::span<VoiceState*> voice_states, const PoolMapper& pool_mapper,
                           const BehaviorInfo& behavior);

    // Update a wavebuffer.
    void update_wave_buffer(std::span<BehaviorInfo::ErrorInfo> error_info, WaveBuffer& wave_buffer,
                          const WaveBufferInternal& wave_buffer_internal,
                          SampleFormat sample_format, bool valid, const PoolMapper& pool_mapper,
                          const BehaviorInfo& behavior);

    // Check if the input wavebuffer needs an update.
    bool should_update_wave_buffer(const WaveBufferInternal& wave_buffer_internal) const;

    // Write the number of played samples, number of consumed wavebuffers and if this voice was
    // dropped, to the given out_status.
    void write_out_status(OutStatus& out_status, const InParameter& in_params,
                        std::span<VoiceState*> voice_states);

    // Check if this voice should be skipped for command generation.
    // Checks various things such as usage state, whether data is mapped etc.
    bool should_skip() const;

    // Check if this voice has any mixing connections.
    bool has_any_connection() const;

    // Flush flush_count wavebuffers, marking them as consumed.
    void flush_wave_buffers(u32 flush_count, std::span<VoiceState*> voice_states, s8 channel_count);

    // Update this voice's parameters on command generation,
    // updating voice states and flushing if needed.
    bool update_parameters_for_command_generation(std::span<VoiceState*> voice_states);

    // Update this voice on command generation.
    bool update_for_command_generation(VoiceContext& voice_context);

    // Reset the AudioRenderer-side voice states, and the channel resources for this voice.
    void reset_resources(VoiceContext& voice_context) const;

    // Is this voice in use?
    bool in_use{};
    // Is this voice new?
    bool is_new{};
    // Was this voice last playing? Used for depopping
    bool was_playing{};
    // Sample format of the wavebuffers in this voice
    SampleFormat sample_format{};
    // Sample rate of the wavebuffers in this voice
    u32 sample_rate{};
    // Number of channels in this voice
    s8 channel_count{};
    // Id of this voice
    u32 id{};
    // Node id of this voice
    u32 node_id{};
    // Mix id this voice is mixed to
    u32 mix_id{};
    // Play state of this voice
    ServerPlayState current_play_state{ServerPlayState::Stopped};
    // Last play state of this voice
    ServerPlayState last_play_state{ServerPlayState::Started};
    // Priority of this voice, lower is higher
    s32 priority{};
    // Sort order of this voice, used when same priority
    s32 sort_order{};
    // Pitch of this voice (for sample rate conversion)
    f32 pitch{};
    // Current volume of this voice
    f32 volume{};
    // Previous volume of this voice
    f32 prev_volume{};
    // Biquad filters for generating filter commands on this voice
    std::array<BiquadFilterParameter, MaxBiquadFilters> biquads{};
    // Float biquad filters for REV15+ (native float coefficients)
    std::array<BiquadFilterParameter2, MaxBiquadFilters> biquads_float{};
    // Use float biquad coefficients (REV15+)
    bool use_float_biquads{};
    // Number of active wavebuffers
    u32 wave_buffer_count{};
    // Current playing wavebuffer index
    u16 wave_buffer_index{};
    // Flags controlling decode behavior
    u16 flags{};
    // Game memory for ADPCM coefficients
    AddressInfo data_address{0, 0};
    // Wavebuffers
    std::array<WaveBuffer, MaxWaveBuffers> wavebuffers{};
    // Channel resources for this voice
    std::array<u32, MaxChannels> channel_resource_ids{};
    // Splitter id this voice is connected with
    s32 splitter_id{UnusedSplitterId};
    // Sample rate conversion quality
    SrcQuality src_quality{SrcQuality::Medium};
    // Was this voice dropped due to limited time?
    bool voice_dropped{};
    // Is this voice's coefficient (data_address) unmapped?
    bool data_unmapped{};
    // Is this voice's buffers (wavebuffer data and ADPCM context) unmapped?
    bool buffer_unmapped{};
    // Initialisation state of the biquads
    std::array<bool, MaxBiquadFilters> biquad_initialized{};
    // Number of wavebuffers to flush
    u8 flush_buffer_count{};
};

} // namespace armigo::audio::renderer
