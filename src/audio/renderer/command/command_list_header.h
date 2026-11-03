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

#include "renderer/common/common.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {

struct CommandListHeader {
    u64 buffer_size;
    u32 command_count;
    std::span<s32> samples_buffer;
    s16 buffer_count;
    u32 sample_count;
    u32 sample_rate;
};

} // namespace armigo::audio::renderer
