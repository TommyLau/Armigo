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

#include <mutex>
#include <span>

#include "renderer/upsampler/upsampler_info.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Manages and has utility functions for upsampler infos.
class UpsamplerManager {
public:
    UpsamplerManager(u32 count, std::span<UpsamplerInfo> infos, std::span<s32> workbuffer);

    // Allocate a new UpsamplerInfo.
    UpsamplerInfo* allocate();

    // Free the given upsampler.
    void free(UpsamplerInfo* info);

private:
    // Maximum number of upsamplers in the buffer
    const u32 count;
    // Upsamplers buffer
    std::span<UpsamplerInfo> upsampler_infos;
    // Workbuffer for upsampling samples
    std::span<s32> workbuffer;
    // Lock for allocate/free
    std::mutex lock{};
};

} // namespace armigo::audio::renderer
