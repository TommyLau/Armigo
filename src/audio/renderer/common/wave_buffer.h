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

#include "renderer/common/types.h"

namespace armigo::audio::renderer {

struct WaveBufferVersion1 {
    CpuAddr buffer;
    u64 buffer_size;
    u32 start_offset;
    u32 end_offset;
    bool loop;
    bool stream_ended;
    CpuAddr context;
    u64 context_size;
};

struct WaveBufferVersion2 {
    CpuAddr buffer;
    CpuAddr context;
    u64 buffer_size;
    u64 context_size;
    u32 start_offset;
    u32 end_offset;
    u32 loop_start_offset;
    u32 loop_end_offset;
    s32 loop_count;
    bool loop;
    bool stream_ended;
};

} // namespace armigo::audio::renderer
