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

#include "renderer/common/common.h"

namespace armigo::audio::renderer {

struct PerformanceEntryAddresses {
    CpuAddr translated_address;
    CpuAddr entry_start_time_offset;
    CpuAddr header_entry_count_offset;
    CpuAddr entry_processed_time_offset;
};

} // namespace armigo::audio::renderer
