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

#include "renderer/sink/sink_info_base.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Manages output sinks.
class SinkContext {
public:
    // Initialize the sink context.
    void initialize(std::span<SinkInfoBase> sink_infos, u32 sink_count);

    // Get a given index's info.
    SinkInfoBase* get_info(u32 index);

    // Get the current number of sinks.
    u32 get_count() const;

private:
    // Buffer of sink infos
    std::span<SinkInfoBase> sink_infos{};
    // Number of sinks in the buffer
    u32 sink_count{};
};

} // namespace armigo::audio::renderer
