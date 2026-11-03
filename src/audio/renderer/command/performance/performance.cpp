// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/dsp/command_list_processor.h"
#include "renderer/command/performance/performance.h"
#include "renderer/common/support.h"

namespace armigo::audio::renderer {

void PerformanceCommand::process(const CommandListProcessor& processor) {
    auto base{entry_address.translated_address};
    if (state == PerformanceState::Start) {
        auto start_time_ptr{reinterpret_cast<u32*>(base + entry_address.entry_start_time_offset)};
        *start_time_ptr =
            static_cast<u32>(host_time_us() -
                             processor.start_time - processor.current_processing_time);
    } else if (state == PerformanceState::Stop) {
        auto processed_time_ptr{
            reinterpret_cast<u32*>(base + entry_address.entry_processed_time_offset)};
        auto entry_count_ptr{
            reinterpret_cast<u32*>(base + entry_address.header_entry_count_offset)};

        *processed_time_ptr =
            static_cast<u32>(host_time_us() -
                             processor.start_time - processor.current_processing_time);
        (*entry_count_ptr)++;
    }
}

bool PerformanceCommand::verify([[maybe_unused]] const CommandListProcessor& processor) {
    return true;
}

} // namespace armigo::audio::renderer
