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

#include <string>

#include "renderer/command/icommand.h"
#include "renderer/performance/performance_entry_addresses.h"
#include "renderer/performance/performance_manager.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
class CommandListProcessor;
}

namespace armigo::audio::renderer {

// AudioRenderer command for writing AudioRenderer performance metrics back to the sysmodule.
struct PerformanceCommand : ICommand {
    // Print this command's information to a string.

    // Process this command.
    void process(const CommandListProcessor& processor) override;

    // Verify this command's data is valid.
    bool verify(const CommandListProcessor& processor) override;

    // State of the performance
    PerformanceState state;
    // Pointers to be written
    PerformanceEntryAddresses entry_address;
};

} // namespace armigo::audio::renderer
