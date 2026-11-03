// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <algorithm>

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/sink/device.h"
#include <armigo/audio/pcm_output.h>

#include <limits>

namespace armigo::audio::renderer {

void DeviceSinkCommand::process(const CommandListProcessor& processor) {
    constexpr s32 min = (std::numeric_limits<s16>::min)();
    constexpr s32 max = (std::numeric_limits<s16>::max)();
    auto output{processor.get_output()};
    if (output == nullptr || input_count == 0 || input_count > MaxChannels) {
        return;
    }

    // The device sink always carries one 48 kHz frame of TargetSampleCount
    // samples; an upsampler precedes it when the renderer runs at another rate.
    constexpr u32 frames{TargetSampleCount};
    std::array<s16, TargetSampleCount * MaxChannels> samples{};
    for (u32 channel = 0; channel < input_count; channel++) {
        const auto offset{static_cast<u64>(inputs[channel]) * frames};
        if (inputs[channel] < 0 || offset + frames > sample_buffer.size()) {
            continue;
        }

        for (u32 index = 0; index < frames; index++) {
            samples[index * input_count + channel] =
                static_cast<s16>(std::clamp(sample_buffer[offset + index], min, max));
        }
    }

    // The host output accepts mono, stereo and 5.1; any other layout keeps its
    // front pair.
    if (input_count == 1 || input_count == 2 || input_count == 6) {
        output->write(samples.data(), frames, input_count);
        return;
    }
    std::array<s16, TargetSampleCount * 2> stereo{};
    for (u32 index = 0; index < frames; index++) {
        stereo[index * 2] = samples[index * input_count];
        stereo[index * 2 + 1] = samples[index * input_count + 1];
    }
    output->write(stereo.data(), frames, 2);
}

bool DeviceSinkCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
