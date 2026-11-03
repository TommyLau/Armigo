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

#include "renderer/common/common.h"
#include "renderer/common/types.h"
#include "renderer/common/fixed_point.h"

namespace armigo::audio::renderer {
// Resample an input buffer into an output buffer, according to the sample_rate_ratio.
void resample(std::span<s32> output, std::span<const s16> input,
              const FixedPoint<49, 15>& sample_rate_ratio,
              FixedPoint<49, 15>& fraction, u32 samples_to_write, SrcQuality src_quality);

} // namespace armigo::audio::renderer
