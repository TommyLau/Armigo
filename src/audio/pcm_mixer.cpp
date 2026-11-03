// ============================================================================
// Armigo Audio - Portable PCM Frame Mixing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/audio/pcm_mixer.h>

#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

namespace armigo::audio {
namespace {

constexpr std::size_t kSigned16Bytes = 2;
constexpr std::size_t kFloat32Bytes = 4;
constexpr double kSigned16Scale = 32768.0;

// Assemble little-endian bytes into the sample value. Reading bytes one at a
// time keeps the decode free of alignment and host-endianness assumptions.
double decode_signed16(const std::byte* bytes) {
    const auto raw = static_cast<std::uint16_t>(static_cast<unsigned>(bytes[0])
        | (static_cast<unsigned>(bytes[1]) << 8));
    // C++20 defines narrow conversion to a signed type as two's-complement
    // truncation, so the wrap reproduces the encoder's signed value exactly.
    const auto signed_value = static_cast<std::int16_t>(raw);
    return static_cast<double>(signed_value) / kSigned16Scale;
}

float decode_float32(const std::byte* bytes) {
    const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8)
        | (static_cast<std::uint32_t>(bytes[2]) << 16)
        | (static_cast<std::uint32_t>(bytes[3]) << 24);
    return std::bit_cast<float>(bits);
}

bool valid_output_length(std::size_t floats) noexcept {
    return floats != 0 && floats % 2 == 0 && floats / 2 <= kMaxPcmStereoFrames;
}

}  // namespace

PcmMixResult mix_pcm_stereo(std::span<const PcmVoiceView> voices,
                            std::span<float> interleaved_output) noexcept {
    if (!valid_output_length(interleaved_output.size())) {
        return {PcmMixError::InvalidOutput,
                "output must hold an even number of floats within the stereo frame limit"};
    }
    const std::size_t frames = interleaved_output.size() / 2;
    const std::size_t output_channels = frames * 2;

    if (voices.size() > kMaxPcmVoices) {
        return {PcmMixError::VoiceLimit, "voice count exceeds the host admission limit"};
    }

    // Validate every voice before touching scratch memory or the destination,
    // so error causes have a fixed precedence independent of allocation luck.
    for (const PcmVoiceView& voice : voices) {
        if (voice.format != PcmFormat::Signed16LittleEndian
                && voice.format != PcmFormat::Float32LittleEndian) {
            return {PcmMixError::InvalidFormat, "voice format is not a supported PCM encoding"};
        }
        if (voice.channels != 1 && voice.channels != 2) {
            return {PcmMixError::InvalidChannels, "voice channel count must be 1 or 2"};
        }
        const std::size_t sample_bytes =
            voice.format == PcmFormat::Signed16LittleEndian ? kSigned16Bytes : kFloat32Bytes;
        const auto expected = static_cast<std::uint64_t>(frames)
            * static_cast<std::uint64_t>(voice.channels)
            * static_cast<std::uint64_t>(sample_bytes);
        if (voice.samples.size() != expected) {
            return {PcmMixError::InputSize,
                    "voice sample span must hold exactly frames * channels samples"};
        }
        if (!std::isfinite(voice.left_gain) || !std::isfinite(voice.right_gain)) {
            return {PcmMixError::NonFinite, "voice gains must be finite"};
        }
        if (voice.format == PcmFormat::Float32LittleEndian) {
            for (std::size_t offset = 0; offset < voice.samples.size();
                 offset += kFloat32Bytes) {
                if (!std::isfinite(decode_float32(voice.samples.data() + offset))) {
                    return {PcmMixError::NonFinite, "voice samples must be finite"};
                }
            }
        }
    }

    // Bounded scratch: at most 4096 frames of stereo doubles twice, reused
    // across voices. Reads stage into `staged` and accumulate into
    // `accumulator`; neither is reachable through the caller's spans, so the
    // destination cannot observe mixing state before the final copy.
    std::vector<double> accumulator;
    std::vector<double> staged;
    try {
        accumulator.assign(output_channels, 0.0);
        staged.assign(output_channels, 0.0);
    } catch (const std::bad_alloc&) {
        return {PcmMixError::AllocationFailure, "mixing scratch allocation failed"};
    } catch (...) {
        return {PcmMixError::AllocationFailure, "mixing scratch allocation failed"};
    }

    for (const PcmVoiceView& voice : voices) {
        const std::size_t staged_channels = frames * voice.channels;
        const std::size_t sample_bytes =
            voice.format == PcmFormat::Signed16LittleEndian ? kSigned16Bytes : kFloat32Bytes;
        for (std::size_t i = 0; i < staged_channels; ++i) {
            const std::byte* bytes = voice.samples.data() + i * sample_bytes;
            staged[i] = voice.format == PcmFormat::Signed16LittleEndian
                ? decode_signed16(bytes)
                : static_cast<double>(decode_float32(bytes));
        }
        if (voice.channels == 1) {
            for (std::size_t frame = 0; frame < frames; ++frame) {
                accumulator[frame * 2] += staged[frame] * voice.left_gain;
                accumulator[frame * 2 + 1] += staged[frame] * voice.right_gain;
            }
        } else {
            for (std::size_t frame = 0; frame < frames; ++frame) {
                accumulator[frame * 2] += staged[frame * 2] * voice.left_gain;
                accumulator[frame * 2 + 1] += staged[frame * 2 + 1] * voice.right_gain;
            }
        }
    }

    // Convert in double precision and reject results a finite float cannot
    // represent. Finite inputs cannot produce non-finite doubles, but the
    // check keeps the range contract explicit. All channels are checked
    // before the first write so a failure never half-fills the destination.
    const double float_max = static_cast<double>(std::numeric_limits<float>::max());
    for (std::size_t i = 0; i < output_channels; ++i) {
        if (!std::isfinite(accumulator[i]) || std::fabs(accumulator[i]) > float_max) {
            return {PcmMixError::Overflow,
                    "mixed accumulation exceeds the finite float output range"};
        }
    }
    for (std::size_t i = 0; i < output_channels; ++i) {
        interleaved_output[i] = static_cast<float>(accumulator[i]);
    }
    return {};
}

}  // namespace armigo::audio
