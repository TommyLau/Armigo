// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/command/command_buffer.h"
#include "renderer/command/command_generator.h"
#include "renderer/performance/detail_aspect.h"

namespace armigo::audio::renderer {

DetailAspect::DetailAspect(CommandGenerator& command_generator_,
                           const PerformanceEntryType entry_type, const s32 node_id_,
                           const PerformanceDetailType detail_type)
    : command_generator{command_generator_}, node_id{node_id_} {
    auto perf_manager{command_generator.get_performance_manager()};
    if (perf_manager != nullptr && perf_manager->is_initialized() &&
        perf_manager->is_detail_target(node_id) &&
        perf_manager->get_next_entry(performance_entry_address, detail_type, entry_type, node_id)) {
        command_generator.generate_performance_command(node_id, PerformanceState::Start,
                                                     performance_entry_address);

        initialized = true;
    }
}

} // namespace armigo::audio::renderer
