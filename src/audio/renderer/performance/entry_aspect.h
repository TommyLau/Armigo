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

#include "renderer/performance/performance_entry_addresses.h"
#include "renderer/performance/performance_manager.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandGenerator;

// Holds entry information about performance metrics, filled in by the AudioRenderer during
// Performance commands.
class EntryAspect {
public:
    EntryAspect(CommandGenerator& command_generator, PerformanceEntryType type, s32 node_id);

    // Command generator the command will be generated into
    CommandGenerator& command_generator;
    // Addresses to be filled by the AudioRenderer
    PerformanceEntryAddresses performance_entry_address{};
    // Is this detail aspect initialized?
    bool initialized{};
    // Node id of this aspect
    s32 node_id;
};

} // namespace armigo::audio::renderer
