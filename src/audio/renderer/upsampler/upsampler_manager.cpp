// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/upsampler/upsampler_manager.h"

namespace armigo::audio::renderer {

UpsamplerManager::UpsamplerManager(const u32 count_, std::span<UpsamplerInfo> infos_,
                                   std::span<s32> workbuffer_)
    : count{count_}, upsampler_infos{infos_}, workbuffer{workbuffer_} {}

UpsamplerInfo* UpsamplerManager::allocate() {
    std::scoped_lock l{lock};

    if (count == 0) {
        return nullptr;
    }

    u32 free_index{0};
    for (auto& upsampler : upsampler_infos) {
        if (!upsampler.enabled) {
            break;
        }
        free_index++;
    }

    if (free_index >= count) {
        return nullptr;
    }

    auto& upsampler{upsampler_infos[free_index]};
    upsampler.manager = this;
    upsampler.sample_count = TargetSampleCount;
    upsampler.samples_pos = CpuAddr(&workbuffer[upsampler.sample_count * MaxChannels]);
    upsampler.enabled = true;
    return &upsampler;
}

void UpsamplerManager::free(UpsamplerInfo* info) {
    std::scoped_lock l{lock};
    info->enabled = false;
}

} // namespace armigo::audio::renderer
