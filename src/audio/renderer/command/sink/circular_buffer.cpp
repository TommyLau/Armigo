// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <vector>

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/sink/circular_buffer.h"
#include "renderer/common/guest_memory.h"

namespace armigo::audio::renderer {

void CircularBufferSinkCommand::process(const CommandListProcessor& processor) {
    constexpr s32 min{(std::numeric_limits<s16>::min)()};
    constexpr s32 max{(std::numeric_limits<s16>::max)()};

    std::array<s16, TargetSampleCount * MaxChannels> output{};
    for (u32 channel = 0; channel < input_count; channel++) {
        auto input{processor.mix_buffer(inputs[channel] * processor.sample_count,
                                                 processor.sample_count)};
        for (u32 sample_index = 0; sample_index < processor.sample_count; sample_index++) {
            output[sample_index] = static_cast<s16>(std::clamp(input[sample_index], min, max));
        }

        processor.memory->write_block(address + pos, output.data(),
                                           processor.sample_count * sizeof(s16));
        pos += static_cast<u32>(processor.sample_count * sizeof(s16));
        if (pos >= size) {
            pos = 0;
        }
    }
}

bool CircularBufferSinkCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
