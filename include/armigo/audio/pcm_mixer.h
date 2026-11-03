// ============================================================================
// Armigo Audio - Portable PCM Frame Mixing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace armigo::audio {

// A portable CPU mixer for interleaved PCM frames. This is host-side signal
// processing only: it does not speak any Horizon audio wire protocol, touch a
// host audio device, resample, loop, or apply effects. It is the admission
// component future audio output sinks build on.

// Input sample encodings accepted from voices. Both decode as little-endian
// byte sequences regardless of host endianness.
enum class PcmFormat {
    Signed16LittleEndian,
    Float32LittleEndian,
};

// One mixing input: a read-only view of interleaved PCM samples plus the
// per-channel gain to apply. A mono voice (one channel) feeds every output
// channel; a stereo voice keeps its left and right samples on their own
// output channels.
struct PcmVoiceView {
    PcmFormat format;
    std::span<const std::byte> samples;
    std::uint32_t channels = 1;
    float left_gain = 1.0f;
    float right_gain = 1.0f;
};

// Structured failure causes. Every failure leaves the destination unchanged.
enum class PcmMixError {
    None,
    InvalidOutput,
    VoiceLimit,
    InvalidFormat,
    InvalidChannels,
    InputSize,
    NonFinite,
    Overflow,
    AllocationFailure,
};

// The outcome of one mixing request. `detail` points at a static diagnostic
// string when `error` is not None and is empty on success.
struct PcmMixResult {
    PcmMixError error = PcmMixError::None;
    std::string_view detail;

    bool ok() const noexcept { return error == PcmMixError::None; }
};

// Host admission limits for one mixing request. These are Armigo policies
// that bound scratch memory and request size, not Horizon firmware limits.
inline constexpr std::size_t kMaxPcmVoices = 256;
inline constexpr std::size_t kMaxPcmStereoFrames = 4096;

// Mix every voice, in span order, into `interleaved_output` as unclipped
// interleaved stereo floats. The output holds exactly `size() / 2` frames;
// its length must be nonzero and even and at most `2 * kMaxPcmStereoFrames`
// floats. Each voice must carry exactly `frames * channels` samples.
//
// Signed 16-bit samples decode value-exactly, scaling by 1/32768; float
// 32-bit samples decode bit-exactly; every sample and gain must be finite.
// Samples accumulate per output channel in double precision in voice order
// under the host's default floating-point environment: traps stay masked
// and rounding follows the default mode. This component performs no global
// FP-environment changes and no flush-to-zero, so subnormal inputs and
// accumulations follow the host default (preserved with Armigo's default
// build flags). Reproducibility comes from the fixed evaluation order, not
// from an unconditional bit-identical guarantee, and no DSP fidelity beyond
// this contract is claimed. A result outside the finite float range fails
// with Overflow. The output is never clipped past +/-1; clipping is a
// future sink policy.
//
// The function reads no memory outside the given spans, assumes no alignment,
// and is safe when `interleaved_output` storage aliases any voice's samples:
// all input reads complete into internal scratch before any output byte is
// written. On any failure the destination bytes are unchanged. With no
// voices, a valid destination is filled with zeros.
PcmMixResult mix_pcm_stereo(std::span<const PcmVoiceView> voices,
                            std::span<float> interleaved_output) noexcept;

}  // namespace armigo::audio
