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
#include "renderer/command/resample/downmix_6ch_to_2ch.h"

namespace armigo::audio::renderer {

void DownMix6chTo2chCommand::process(const CommandListProcessor& processor) {
    auto in_front_left{
        processor.mix_buffer(inputs[0] * processor.sample_count, processor.sample_count)};
    auto in_front_right{
        processor.mix_buffer(inputs[1] * processor.sample_count, processor.sample_count)};
    auto in_center{
        processor.mix_buffer(inputs[2] * processor.sample_count, processor.sample_count)};
    auto in_lfe{
        processor.mix_buffer(inputs[3] * processor.sample_count, processor.sample_count)};
    auto in_back_left{
        processor.mix_buffer(inputs[4] * processor.sample_count, processor.sample_count)};
    auto in_back_right{
        processor.mix_buffer(inputs[5] * processor.sample_count, processor.sample_count)};

    auto out_front_left{
        processor.mix_buffer(outputs[0] * processor.sample_count, processor.sample_count)};
    auto out_front_right{
        processor.mix_buffer(outputs[1] * processor.sample_count, processor.sample_count)};
    auto out_center{
        processor.mix_buffer(outputs[2] * processor.sample_count, processor.sample_count)};
    auto out_lfe{
        processor.mix_buffer(outputs[3] * processor.sample_count, processor.sample_count)};
    auto out_back_left{
        processor.mix_buffer(outputs[4] * processor.sample_count, processor.sample_count)};
    auto out_back_right{
        processor.mix_buffer(outputs[5] * processor.sample_count, processor.sample_count)};

    for (u32 i = 0; i < processor.sample_count; i++) {
        const auto left_sample{(in_front_left[i] * down_mix_coeff[0] +
                                in_center[i] * down_mix_coeff[1] + in_lfe[i] * down_mix_coeff[2] +
                                in_back_left[i] * down_mix_coeff[3])
                                   .to_int()};

        const auto right_sample{(in_front_right[i] * down_mix_coeff[0] +
                                 in_center[i] * down_mix_coeff[1] + in_lfe[i] * down_mix_coeff[2] +
                                 in_back_right[i] * down_mix_coeff[3])
                                    .to_int()};

        out_front_left[i] = left_sample;
        out_front_right[i] = right_sample;
    }

    std::memset(out_center.data(), 0, out_center.size_bytes());
    std::memset(out_lfe.data(), 0, out_lfe.size_bytes());
    std::memset(out_back_left.data(), 0, out_back_left.size_bytes());
    std::memset(out_back_right.data(), 0, out_back_right.size_bytes());
}

bool DownMix6chTo2chCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
