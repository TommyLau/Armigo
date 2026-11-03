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

#include <span>

#include "renderer/common/types.h"
#include "renderer/common/result.h"

namespace armigo::audio::renderer {
class GuestMemory;
class BehaviorInfo;
class VoiceContext;
class MixContext;
class SinkContext;
class SplitterContext;
class EffectContext;
class MemoryPoolInfo;
class PerformanceManager;

class InfoUpdater {
    struct UpdateDataHeader {
        explicit UpdateDataHeader(u32 revision_) : revision{revision_} {}

        /* 0x00 */ u32 revision;
        /* 0x04 */ u32 behaviour_size{};
        /* 0x08 */ u32 memory_pool_size{};
        /* 0x0C */ u32 voices_size{};
        /* 0x10 */ u32 voice_resources_size{};
        /* 0x14 */ u32 effects_size{};
        /* 0x18 */ u32 mix_size{};
        /* 0x1C */ u32 sinks_size{};
        /* 0x20 */ u32 performance_buffer_size{};
        /* 0x24 */ u32 splitter_size{};
        /* 0x28 */ u32 render_info_size{};
        /* 0x2C */ char unk2C[0x10];
        /* 0x3C */ u32 size{sizeof(UpdateDataHeader)};
    };
    static_assert(sizeof(UpdateDataHeader) == 0x40, "UpdateDataHeader has the wrong size!");

public:
    explicit InfoUpdater(std::span<const u8> input, std::span<u8> output,
                         GuestMemory* memory, BehaviorInfo& behaviour);

    // Update the voice channel resources.
    Result update_voice_channel_resources(VoiceContext& voice_context);

    // Update voices.
    Result update_voices(VoiceContext& voice_context, std::span<MemoryPoolInfo> memory_pools,
                        u32 memory_pool_count);

    // Update effects.
    Result update_effects(EffectContext& effect_context, bool renderer_active,
                         std::span<MemoryPoolInfo> memory_pools, u32 memory_pool_count);

    // Update mixes.
    Result update_mixes(MixContext& mix_context, u32 mix_buffer_count, EffectContext& effect_context,
                       SplitterContext& splitter_context);

    // Update sinks.
    Result update_sinks(SinkContext& sink_context, std::span<MemoryPoolInfo> memory_pools,
                       u32 memory_pool_count);

    // Update memory pools.
    Result update_memory_pools(std::span<MemoryPoolInfo> memory_pools, u32 memory_pool_count);

    // Update the performance buffer.
    Result update_performance_buffer(std::span<u8> output, u64 output_size,
                                   PerformanceManager* performance_manager);

    // Update behaviour.
    Result update_behavior_info(BehaviorInfo& behaviour);

    // Update errors.
    Result update_error_info(const BehaviorInfo& behaviour);

    // Update splitter.
    Result update_splitter_info(SplitterContext& splitter_context);

    // Update renderer info.
    Result update_renderer_info(u64 elapsed_frames);

    // Check that the input.output sizes match their expected values.
    Result check_consumed_size();

    // Whether the guest's input and output buffers hold `size` more bytes. The
    // update data is guest-controlled, so every section is bounds-checked
    // before it is read or written.
    bool input_has(u64 size) const;
    bool output_has(u64 size) const;

private:
    // Update effects version 1.
    Result update_effects_version1(EffectContext& effect_context, bool renderer_active,
                                 std::span<MemoryPoolInfo> memory_pools, u32 memory_pool_count);

    // Update effects version 2.
    Result update_effects_version2(EffectContext& effect_context, bool renderer_active,
                                 std::span<MemoryPoolInfo> memory_pools, u32 memory_pool_count);

    // Input buffer
    u8 const* input;
    // Input buffer start
    std::span<const u8> input_origin;
    // Output buffer start
    u8* output;
    // Output buffer start
    std::span<u8> output_origin;
    // Input header
    const UpdateDataHeader* in_header;
    // Output header
    UpdateDataHeader* out_header;
    // Expected input size, see CheckConsumedSize
    u64 expected_input_size;
    // Expected output size, see CheckConsumedSize
    u64 expected_output_size;
    // Guest memory the pool mappers pin into
    GuestMemory* memory;
    // Behaviour
    BehaviorInfo& behaviour;
};

} // namespace armigo::audio::renderer
