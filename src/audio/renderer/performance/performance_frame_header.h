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

struct PerformanceFrameHeaderVersion1 {
    /* 0x00 */ u32 magic; // "PERF"
    /* 0x04 */ u32 entry_count;
    /* 0x08 */ u32 detail_count;
    /* 0x0C */ u32 next_offset;
    /* 0x10 */ u32 total_processing_time;
    /* 0x14 */ u32 frame_index;
};
static_assert(sizeof(PerformanceFrameHeaderVersion1) == 0x18,
              "PerformanceFrameHeaderVersion1 has the wrong size!");

struct PerformanceFrameHeaderVersion2 {
    /* 0x00 */ u32 magic; // "PERF"
    /* 0x04 */ u32 entry_count;
    /* 0x08 */ u32 detail_count;
    /* 0x0C */ u32 next_offset;
    /* 0x10 */ u32 total_processing_time;
    /* 0x14 */ u32 voices_dropped;
    /* 0x18 */ u64 start_time;
    /* 0x20 */ u32 frame_index;
    /* 0x24 */ bool render_time_exceeded;
    /* 0x25 */ char unk25[0xB];
};
static_assert(sizeof(PerformanceFrameHeaderVersion2) == 0x30,
              "PerformanceFrameHeaderVersion2 has the wrong size!");

} // namespace armigo::audio::renderer
