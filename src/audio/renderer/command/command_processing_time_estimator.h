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

// Command processing time estimation is not implemented: every command is
// estimated at zero DSP cycles. The estimates only feed voice dropping, which
// therefore never triggers (the estimated total never exceeds the time limit).
class CommandProcessingTimeEstimator {
public:
    template <typename Command>
    u32 estimate(const Command&) const noexcept {
        return 0;
    }
};

}  // namespace armigo::audio::renderer
