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

enum class PerformanceEntryType : u8 {
    Invalid,
    Voice,
    SubMix,
    FinalMix,
    Sink,
};

struct PerformanceEntryVersion1 {
    /* 0x00 */ u32 node_id;
    /* 0x04 */ u32 start_time;
    /* 0x08 */ u32 processed_time;
    /* 0x0C */ PerformanceEntryType entry_type;
};
static_assert(sizeof(PerformanceEntryVersion1) == 0x10,
              "PerformanceEntryVersion1 has the wrong size!");

struct PerformanceEntryVersion2 {
    /* 0x00 */ u32 node_id;
    /* 0x04 */ u32 start_time;
    /* 0x08 */ u32 processed_time;
    /* 0x0C */ PerformanceEntryType entry_type;
    /* 0x0D */ char unk0D[0xB];
};
static_assert(sizeof(PerformanceEntryVersion2) == 0x18,
              "PerformanceEntryVersion2 has the wrong size!");

} // namespace armigo::audio::renderer
