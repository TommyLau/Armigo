// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/dsp/command_list_processor.h"
#include "renderer/common/common.h"
#include "renderer/command/mix/depop_for_mix_buffers.h"

namespace armigo::audio::renderer {
// Apply depopping. Add the depopped sample to each incoming new sample, decaying it each time
// according to decay.
static s32 apply_depop_mix(std::span<s32> output, const s32 depop_sample,
                         FixedPoint<49, 15>& decay_, const u32 sample_count) {
    auto sample{std::abs(depop_sample)};
    auto decay{decay_.to_raw()};

    if (depop_sample <= 0) {
        for (u32 i = 0; i < sample_count; i++) {
            sample = static_cast<s32>((static_cast<s64>(sample) * decay) >> 15);
            output[i] -= sample;
        }
        return -sample;
    } else {
        for (u32 i = 0; i < sample_count; i++) {
            sample = static_cast<s32>((static_cast<s64>(sample) * decay) >> 15);
            output[i] += sample;
        }
        return sample;
    }
}

void DepopForMixBuffersCommand::process(const CommandListProcessor& processor) {
    auto end_index{(std::min)(processor.buffer_count, input + count)};
    std::span<s32> depop_buff{reinterpret_cast<s32*>(depop_buffer), end_index};

    for (u32 index = input; index < end_index; index++) {
        const auto depop_sample{depop_buff[index]};
        if (depop_sample != 0) {
            auto input_buffer{processor.mix_buffer(index * processor.sample_count,
                                                            processor.sample_count)};
            depop_buff[index] =
                apply_depop_mix(input_buffer, depop_sample, decay, processor.sample_count);
        }
    }
}

bool DepopForMixBuffersCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
