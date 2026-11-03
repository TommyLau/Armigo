// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "renderer/sink/sink_context.h"

namespace armigo::audio::renderer {

void SinkContext::initialize(std::span<SinkInfoBase> sink_infos_, const u32 sink_count_) {
    sink_infos = sink_infos_;
    sink_count = sink_count_;
}

SinkInfoBase* SinkContext::get_info(const u32 index) {
    return &sink_infos[index];
}

u32 SinkContext::get_count() const {
    return sink_count;
}

} // namespace armigo::audio::renderer
