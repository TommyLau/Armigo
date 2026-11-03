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

#include "renderer/performance/performance_entry.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {

enum class PerformanceDetailType : u8 {
    Invalid,
    Unk1,
    Unk2,
    Unk3,
    Unk4,
    Unk5,
    Unk6,
    Unk7,
    Unk8,
    Unk9,
    Unk10,
    Unk11,
    Unk12,
    Unk13,
};

struct PerformanceDetailVersion1 {
    /* 0x00 */ u32 node_id;
    /* 0x04 */ u32 start_time;
    /* 0x08 */ u32 processed_time;
    /* 0x0C */ PerformanceDetailType detail_type;
    /* 0x0D */ PerformanceEntryType entry_type;
};
static_assert(sizeof(PerformanceDetailVersion1) == 0x10,
              "PerformanceDetailVersion1 has the wrong size!");

struct PerformanceDetailVersion2 {
    /* 0x00 */ u32 node_id;
    /* 0x04 */ u32 start_time;
    /* 0x08 */ u32 processed_time;
    /* 0x0C */ PerformanceDetailType detail_type;
    /* 0x0D */ PerformanceEntryType entry_type;
    /* 0x10 */ u32 unk_10;
    /* 0x14 */ char unk14[0x4];
};
static_assert(sizeof(PerformanceDetailVersion2) == 0x18,
              "PerformanceDetailVersion2 has the wrong size!");

} // namespace armigo::audio::renderer
