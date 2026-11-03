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

#include "renderer/sink/sink_info_base.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Info for a circular buffer sink.
class CircularBufferSinkInfo : public SinkInfoBase {
public:
    CircularBufferSinkInfo();

    // Clean up for info, resetting it to a default state.
    void clean_up() override;

    // Update the info according to parameters, and write the current state to out_status.
    void update(BehaviorInfo::ErrorInfo& error_info, OutStatus& out_status,
                const InParameter& in_params, const PoolMapper& pool_mapper) override;

    // Update the circular buffer on command generation, incrementing its current offsets.
    void update_for_command_generation() override;
};
static_assert(sizeof(CircularBufferSinkInfo) <= sizeof(SinkInfoBase),
              "CircularBufferSinkInfo is too large!");

} // namespace armigo::audio::renderer
