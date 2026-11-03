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

#include "renderer/effect/aux_.h"
#include "renderer/effect/biquad_filter.h"
#include "renderer/effect/buffer_mixer.h"
#include "renderer/effect/capture.h"
#include "renderer/effect/compressor.h"
#include "renderer/effect/delay.h"
#include "renderer/effect/i3dl2.h"
#include "renderer/effect/light_limiter.h"
#include "renderer/effect/reverb.h"
#include "renderer/common/types.h"

namespace armigo::audio::renderer {
// Reset an effect, and create a new one of the given type.
static void reset_effect(EffectInfoBase* effect, const EffectInfoBase::Type type) {
    *effect = {};

    switch (type) {
    case EffectInfoBase::Type::Invalid:
        std::construct_at<EffectInfoBase>(effect);
        effect->set_type(EffectInfoBase::Type::Invalid);
        break;
    case EffectInfoBase::Type::Mix:
        std::construct_at<BufferMixerInfo>(reinterpret_cast<BufferMixerInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::Mix);
        break;
    case EffectInfoBase::Type::Aux:
        std::construct_at<AuxInfo>(reinterpret_cast<AuxInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::Aux);
        break;
    case EffectInfoBase::Type::Delay:
        std::construct_at<DelayInfo>(reinterpret_cast<DelayInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::Delay);
        break;
    case EffectInfoBase::Type::Reverb:
        std::construct_at<ReverbInfo>(reinterpret_cast<ReverbInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::Reverb);
        break;
    case EffectInfoBase::Type::I3dl2Reverb:
        std::construct_at<I3dl2ReverbInfo>(reinterpret_cast<I3dl2ReverbInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::I3dl2Reverb);
        break;
    case EffectInfoBase::Type::BiquadFilter:
        std::construct_at<BiquadFilterInfo>(reinterpret_cast<BiquadFilterInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::BiquadFilter);
        break;
    case EffectInfoBase::Type::LightLimiter:
        std::construct_at<LightLimiterInfo>(reinterpret_cast<LightLimiterInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::LightLimiter);
        break;
    case EffectInfoBase::Type::Capture:
        std::construct_at<CaptureInfo>(reinterpret_cast<CaptureInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::Capture);
        break;
    case EffectInfoBase::Type::Compressor:
        std::construct_at<CompressorInfo>(reinterpret_cast<CompressorInfo*>(effect));
        effect->set_type(EffectInfoBase::Type::Compressor);
        break;
    }
}

} // namespace armigo::audio::renderer
