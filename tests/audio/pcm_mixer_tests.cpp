// ============================================================================
// Armigo Tests - Audio PCM Mixer Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/audio/pcm_mixer.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

using armigo::audio::kMaxPcmStereoFrames;
using armigo::audio::kMaxPcmVoices;
using armigo::audio::mix_pcm_stereo;
using armigo::audio::PcmFormat;
using armigo::audio::PcmMixError;
using armigo::audio::PcmMixResult;
using armigo::audio::PcmVoiceView;

unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); ++failures; } } while (0)

// Build little-endian sample bytes explicitly so the tests verify the
// little-endian contract independently of the host byte order.
std::vector<std::byte> pcm16_bytes(const std::vector<int>& values) {
    std::vector<std::byte> bytes;
    bytes.reserve(values.size() * 2);
    for (int value : values) {
        const auto raw = static_cast<unsigned>(static_cast<unsigned short>(value));
        bytes.push_back(static_cast<std::byte>(raw & 0xFFu));
        bytes.push_back(static_cast<std::byte>((raw >> 8) & 0xFFu));
    }
    return bytes;
}

std::vector<std::byte> f32_bytes(const std::vector<float>& values) {
    std::vector<std::byte> bytes;
    bytes.reserve(values.size() * 4);
    for (float value : values) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        bytes.push_back(static_cast<std::byte>(bits & 0xFFu));
        bytes.push_back(static_cast<std::byte>((bits >> 8) & 0xFFu));
        bytes.push_back(static_cast<std::byte>((bits >> 16) & 0xFFu));
        bytes.push_back(static_cast<std::byte>((bits >> 24) & 0xFFu));
    }
    return bytes;
}

PcmVoiceView mono16(const std::vector<std::byte>& bytes, float gain = 1.0f) {
    return {PcmFormat::Signed16LittleEndian, bytes, 1, gain, gain};
}

PcmVoiceView stereo16(const std::vector<std::byte>& bytes, float left = 1.0f,
                      float right = 1.0f) {
    return {PcmFormat::Signed16LittleEndian, bytes, 2, left, right};
}

PcmVoiceView mono32(const std::vector<std::byte>& bytes, float gain = 1.0f) {
    return {PcmFormat::Float32LittleEndian, bytes, 1, gain, gain};
}

PcmVoiceView bad_format_voice(const std::vector<std::byte>& bytes) {
    return {static_cast<PcmFormat>(42), bytes, 1, 1.0f, 1.0f};
}

bool same_bytes(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

void expect_failure(const char* label, PcmMixError expected, PcmMixResult result) {
    if (result.error != expected) {
        std::fprintf(stderr, "FAIL %s: error %d != expected %d (%s)\n", label,
                     static_cast<int>(result.error), static_cast<int>(expected),
                     result.detail.data());
        ++failures;
    }
}

// std::span does not bind to a braced voice list directly; the wrapped list
// outlives the call inside this expression.
PcmMixResult mix(std::initializer_list<PcmVoiceView> voices, std::span<float> output) {
    return mix_pcm_stereo(std::span<const PcmVoiceView>(voices.begin(), voices.size()),
                          output);
}

void zero_voices_fill_output_with_zeros() {
    std::vector<float> output(4, 7.5f);
    const PcmMixResult result = mix_pcm_stereo({}, output);
    CHECK(result.ok());
    CHECK(result.detail.empty());
    CHECK((output == std::vector<float>(4, 0.0f)));
}

void pcm16_decodes_little_endian_extrema() {
    // 128 as LE bytes (80 00) must decode to +128/32768; a big-endian
    // misread would return -1.0 from the same bytes.
    const std::vector<std::byte> samples = pcm16_bytes({-32768, -2, 0, 128, 32767});
    std::vector<float> output(10, 0.0f);
    const PcmMixResult result = mix({mono16(samples)}, output);
    CHECK(result.ok());
    static const float expected[10] = {
        -1.0f, -1.0f,
        -2.0f / 32768.0f, -2.0f / 32768.0f,
        0.0f, 0.0f,
        128.0f / 32768.0f, 128.0f / 32768.0f,
        32767.0f / 32768.0f, 32767.0f / 32768.0f,
    };
    CHECK((output == std::vector<float>(expected, expected + 10)));
}

void mono_duplicates_and_stereo_preserves_channels() {
    const std::vector<std::byte> mono = pcm16_bytes({16384});
    std::vector<float> output(2, 0.0f);
    CHECK(mix({mono16(mono)}, output).ok());
    CHECK((output == std::vector<float>{0.5f, 0.5f}));

    const std::vector<std::byte> stereo = pcm16_bytes({32767, -32768});
    std::vector<float> gained(2, 0.0f);
    CHECK(mix({stereo16(stereo, 0.5f, 0.25f)}, gained).ok());
    CHECK((gained == std::vector<float>{32767.0f / 65536.0f, -0.25f}));

    // Asymmetric mono gains put one sample on both channels.
    std::vector<float> panned(2, 0.0f);
    const PcmVoiceView panned_voice{PcmFormat::Signed16LittleEndian, mono, 1, 0.5f, -0.25f};
    CHECK(mix({panned_voice}, panned).ok());
    CHECK((panned == std::vector<float>{0.25f, -0.125f}));
}

void multiple_voices_accumulate_in_order() {
    const std::vector<std::byte> first = pcm16_bytes({16384});            // 0.5
    const std::vector<std::byte> second = pcm16_bytes({8192});            // 0.25
    const std::vector<std::byte> third = pcm16_bytes({4096, -4096});      // 0.125, -0.125
    std::vector<float> output(2, 0.0f);
    const PcmMixResult result = mix(
        {mono16(first), mono16(second, 2.0f), stereo16(third)}, output);
    CHECK(result.ok());
    // 0.5 + 0.25*2 + 0.125 = 1.125; 0.5 + 0.5 - 0.125 = 0.875.
    CHECK((output == std::vector<float>{1.125f, 0.875f}));
}

void output_is_unclipped_float() {
    const std::vector<std::byte> full_scale = pcm16_bytes({32767, -32768});
    std::vector<float> output(2, 0.0f);
    CHECK(mix({stereo16(full_scale), stereo16(full_scale), stereo16(full_scale)},
              output)
              .ok());
    // 3 * 0.99996948... stays above 1.0 and is not clipped to it.
    const float high = 3.0f * (32767.0f / 32768.0f);
    CHECK((output == std::vector<float>{high, -3.0f}));
    CHECK(output[0] > 1.0f);
}

void float32_decodes_little_endian_and_mixes() {
    const std::vector<std::byte> samples = f32_bytes({1.0f, -0.5f, 0.25f, -1.0f});
    std::vector<float> output(4, 0.0f);
    const PcmVoiceView voice{PcmFormat::Float32LittleEndian, samples, 2, 1.0f, 2.0f};
    const PcmMixResult result = mix({voice}, output);
    CHECK(result.ok());
    CHECK((output == std::vector<float>{1.0f, -1.0f, 0.25f, -2.0f}));

    // The smallest positive subnormal is finite input, not an error.
    const std::vector<std::byte> tiny =
        f32_bytes({std::numeric_limits<float>::denorm_min()});
    std::vector<float> tiny_out(2, 0.0f);
    CHECK(mix({mono32(tiny)}, tiny_out).ok());
    CHECK(tiny_out[0] == std::numeric_limits<float>::denorm_min());
    CHECK(tiny_out[1] == std::numeric_limits<float>::denorm_min());
}

void nonfinite_inputs_are_rejected() {
    std::vector<float> output(2, 5.0f);
    const std::vector<float> untouched = output;

    const std::vector<std::byte> infinities =
        f32_bytes({std::numeric_limits<float>::infinity()});
    expect_failure("inf sample", PcmMixError::NonFinite, mix({mono32(infinities)}, output));

    const auto quiet_nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<std::byte> nans = f32_bytes({quiet_nan});
    expect_failure("nan sample", PcmMixError::NonFinite, mix({mono32(nans)}, output));

    const std::vector<std::byte> negative_infinity =
        f32_bytes({-std::numeric_limits<float>::infinity()});
    expect_failure("-inf sample", PcmMixError::NonFinite,
                   mix({mono32(negative_infinity)}, output));

    const std::vector<std::byte> finite = pcm16_bytes({16384});
    const PcmVoiceView nan_gain{PcmFormat::Signed16LittleEndian, finite, 1, quiet_nan, 1.0f};
    expect_failure("nan gain", PcmMixError::NonFinite, mix({nan_gain}, output));
    const PcmVoiceView inf_gain{PcmFormat::Signed16LittleEndian, finite, 1, 1.0f,
                                std::numeric_limits<float>::infinity()};
    expect_failure("inf gain", PcmMixError::NonFinite, mix({inf_gain}, output));

    CHECK(same_bytes(output, untouched));
}

void overflow_beyond_float_range_is_rejected() {
    std::vector<float> output(2, 5.0f);
    const std::vector<float> untouched = output;
    const float float_max = std::numeric_limits<float>::max();
    const std::vector<std::byte> unit = f32_bytes({1.0f});

    // One voice at the representable boundary succeeds.
    const PcmVoiceView at_limit{PcmFormat::Float32LittleEndian, unit, 1, float_max, 1.0f};
    std::vector<float> boundary(2, 0.0f);
    CHECK(mix({at_limit}, boundary).ok());
    CHECK(boundary[0] == float_max);

    // Two such voices exceed the finite float range.
    const PcmVoiceView beyond{PcmFormat::Float32LittleEndian, unit, 1, float_max, float_max};
    expect_failure("overflow", PcmMixError::Overflow, mix({at_limit, beyond}, output));
    CHECK(same_bytes(output, untouched));
}

void voice_size_must_match_frames_and_channels() {
    std::vector<float> ok_output(4, 5.0f);      // 2 frames
    std::vector<float> stereo_output(4, 5.0f);  // 2 frames
    const std::vector<float> untouched = stereo_output;
    const std::vector<std::byte> exact_stereo = pcm16_bytes({1, 2, 3, 4});

    CHECK(mix({stereo16(exact_stereo)}, ok_output).ok());

    const std::vector<std::byte> short_samples(exact_stereo.begin(), exact_stereo.end() - 2);
    expect_failure("short voice", PcmMixError::InputSize,
                   mix({stereo16(short_samples)}, stereo_output));

    std::vector<std::byte> long_samples = exact_stereo;
    long_samples.push_back(std::byte{0});
    expect_failure("long voice", PcmMixError::InputSize,
                   mix({stereo16(long_samples)}, stereo_output));

    // The same byte count is wrong for a mono voice at two frames.
    expect_failure("channel mismatch", PcmMixError::InputSize,
                   mix({mono16(exact_stereo)}, stereo_output));
    CHECK(same_bytes(stereo_output, untouched));
}

void invalid_format_and_channel_counts_are_rejected() {
    std::vector<float> output(2, 5.0f);
    const std::vector<float> untouched = output;
    const std::vector<std::byte> samples = pcm16_bytes({0});

    const PcmVoiceView bad_format = bad_format_voice(samples);
    expect_failure("bad format", PcmMixError::InvalidFormat, mix({bad_format}, output));
    // Format validation precedes the channel count check.
    const PcmVoiceView format_before_channels{static_cast<PcmFormat>(42), samples, 7,
                                              1.0f, 1.0f};
    expect_failure("format before channels", PcmMixError::InvalidFormat,
                   mix({format_before_channels}, output));

    const PcmVoiceView zero_channels{PcmFormat::Signed16LittleEndian, samples, 0, 1.0f, 1.0f};
    expect_failure("zero channels", PcmMixError::InvalidChannels,
                   mix({zero_channels}, output));
    const std::vector<std::byte> three_channel_samples = f32_bytes({0.0f, 0.0f, 0.0f});
    const PcmVoiceView three_channels{PcmFormat::Float32LittleEndian, three_channel_samples,
                                      3, 1.0f, 1.0f};
    expect_failure("three channels", PcmMixError::InvalidChannels,
                   mix({three_channels}, output));
    // Channel validation precedes the size check.
    const PcmVoiceView channels_before_size{PcmFormat::Signed16LittleEndian, {}, 5, 1.0f,
                                            1.0f};
    expect_failure("channels before size", PcmMixError::InvalidChannels,
                   mix({channels_before_size}, output));
    CHECK(same_bytes(output, untouched));
}

void admission_limits_are_enforced() {
    std::vector<float> output(2, 5.0f);
    const std::vector<float> untouched = output;
    const std::vector<std::byte> silence = pcm16_bytes({0});

    // The count check precedes per-voice validation.
    std::vector<PcmVoiceView> too_many(kMaxPcmVoices + 1, mono16(silence));
    too_many[5] = bad_format_voice(silence);
    expect_failure("voice limit", PcmMixError::VoiceLimit,
                   mix_pcm_stereo(too_many, output));
    CHECK(same_bytes(output, untouched));

    std::vector<PcmVoiceView> at_limit(kMaxPcmVoices, mono16(silence));
    CHECK(mix_pcm_stereo(at_limit, output).ok());
    CHECK((output == std::vector<float>{0.0f, 0.0f}));
}

void output_validation_rejects_odd_and_oversized() {
    std::vector<float> odd(3, 0.0f);
    expect_failure("odd output", PcmMixError::InvalidOutput, mix_pcm_stereo({}, odd));

    std::vector<float> oversized(2 * (kMaxPcmStereoFrames + 1), 0.0f);
    expect_failure("too many frames", PcmMixError::InvalidOutput,
                   mix_pcm_stereo({}, oversized));

    std::vector<float> at_limit(2 * kMaxPcmStereoFrames, 1.0f);
    CHECK(mix_pcm_stereo({}, at_limit).ok());
    std::vector<float> below_limit(2 * (kMaxPcmStereoFrames - 1), 1.0f);
    CHECK(mix_pcm_stereo({}, below_limit).ok());

    // Output validation precedes the voice count check.
    const std::vector<std::byte> silence = pcm16_bytes({0});
    std::vector<PcmVoiceView> too_many(kMaxPcmVoices + 1, mono16(silence));
    expect_failure("output before voices", PcmMixError::InvalidOutput,
                   mix_pcm_stereo(too_many, odd));
}

void empty_output_is_invalid() {
    std::vector<float> empty;
    expect_failure("empty output", PcmMixError::InvalidOutput, mix_pcm_stereo({}, empty));
    const std::vector<std::byte> nothing;
    // Output validation precedes voice checks.
    expect_failure("empty output with voice", PcmMixError::InvalidOutput,
                   mix({mono16(nothing)}, empty));
    const std::vector<std::byte> stray = pcm16_bytes({1});
    expect_failure("sample without frames", PcmMixError::InvalidOutput,
                   mix({mono16(stray)}, empty));
}

void unaligned_sample_storage_decodes_correctly() {
    // Voice samples start at odd byte offsets inside the storage, so no
    // sample is aligned for its 16-bit or 32-bit type.
    std::vector<std::byte> samples16 = pcm16_bytes({16384, 16384, -16384, -16384});
    std::vector<std::byte> odd16{std::byte{0xAB}};
    odd16.insert(odd16.end(), samples16.begin(), samples16.end());
    const auto shifted16 = std::span<const std::byte>(odd16).subspan(1);
    std::vector<float> pcm_out(4, 0.0f);
    const PcmVoiceView pcm_voice{PcmFormat::Signed16LittleEndian, shifted16, 2, 2.0f, 2.0f};
    CHECK(mix({pcm_voice}, pcm_out).ok());
    CHECK((pcm_out == std::vector<float>{1.0f, 1.0f, -1.0f, -1.0f}));

    const std::vector<float> values{1.0f, -1.0f, 1.0f, -1.0f};
    const std::vector<std::byte> raw32 = f32_bytes(values);
    for (std::size_t offset = 1; offset <= 3; ++offset) {
        std::vector<std::byte> odd32(offset, std::byte{0xAB});
        odd32.insert(odd32.end(), raw32.begin(), raw32.end());
        const auto shifted = std::span<const std::byte>(odd32).subspan(offset);
        std::vector<float> out(4, 0.0f);
        const PcmVoiceView voice{PcmFormat::Float32LittleEndian, shifted, 2, 1.0f, 0.5f};
        CHECK(mix({voice}, out).ok());
        CHECK((out == std::vector<float>{1.0f, -0.5f, 1.0f, -0.5f}));
    }
}

void aliased_input_with_failing_mix_preserves_destination() {
    // The destination overlaps the first voice's samples, and a later voice
    // fails validation: the overlapped bytes must survive byte-for-byte.
    std::vector<float> output(4, 0.5f);  // each float: bytes 00 00 00 3F
    const std::vector<float> untouched = output;
    const auto* storage = reinterpret_cast<const std::byte*>(output.data());
    const std::span<const std::byte> aliased_samples(storage, 2 * 2);
    const PcmVoiceView aliased{PcmFormat::Signed16LittleEndian, aliased_samples, 1, 2.0f,
                               3.0f};
    const auto quiet_nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<std::byte> finite = pcm16_bytes({1, 1});
    const PcmVoiceView nan_gain{PcmFormat::Signed16LittleEndian, finite, 1, quiet_nan, 1.0f};
    expect_failure("alias with nonfinite gain", PcmMixError::NonFinite,
                   mix({aliased, nan_gain}, output));
    CHECK(same_bytes(output, untouched));
}

void late_channel_overflow_mutates_nothing() {
    // Only the second-to-last output channel exceeds the float range; every
    // earlier channel mixes cleanly. A conversion loop that wrote while
    // checking would corrupt those channels before detecting the failure.
    std::vector<float> output(8, 12345.0f);  // 4 frames
    const std::vector<float> untouched = output;
    const float float_max = std::numeric_limits<float>::max();
    const std::vector<std::byte> full_left =
        f32_bytes({float_max, 1.0f, float_max, 1.0f, float_max, 1.0f, float_max, 1.0f});
    const std::vector<std::byte> last_frame_left =
        f32_bytes({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    const PcmVoiceView base{PcmFormat::Float32LittleEndian, full_left, 2, 1.0f, 1.0f};
    const PcmVoiceView boost{PcmFormat::Float32LittleEndian, last_frame_left, 2, float_max,
                             1.0f};
    expect_failure("late channel overflow", PcmMixError::Overflow,
                   mix({base, boost}, output));
    CHECK(same_bytes(output, untouched));
}

void accumulation_follows_voice_order() {
    // 2^60 + 2^-10 absorbs the small term in double precision, while
    // 2^60 + (-2^60) leaves it intact: the voice order changes the result,
    // and the mixer must follow the span order deterministically.
    const float big = 0x1p60f;
    const float tiny = 0x1p-10f;
    const std::vector<std::byte> big_bytes = f32_bytes({big});
    const std::vector<std::byte> tiny_bytes = f32_bytes({tiny});
    const std::vector<std::byte> neg_big_bytes = f32_bytes({-big});

    std::vector<float> absorbed(2, 7.0f);
    CHECK(mix({mono32(big_bytes), mono32(tiny_bytes), mono32(neg_big_bytes)},
              absorbed)
              .ok());
    CHECK(absorbed[0] == 0.0f);

    std::vector<float> preserved(2, 7.0f);
    CHECK(mix({mono32(big_bytes), mono32(neg_big_bytes), mono32(tiny_bytes)},
              preserved)
              .ok());
    CHECK(preserved[0] == tiny);
    CHECK(same_bytes(absorbed, preserved) == false);
}

void failures_leave_destination_unchanged() {
    std::vector<float> output(6, 12345.0f);
    const std::vector<float> untouched = output;
    const std::vector<std::byte> samples = pcm16_bytes({16384, -16384});

    expect_failure("size", PcmMixError::InputSize, mix({stereo16(samples)}, output));

    // The output holds three frames, so each mono voice must carry three
    // samples for the size check to pass and the deeper causes to surface.
    const std::vector<std::byte> inf =
        f32_bytes({std::numeric_limits<float>::infinity(), 1.0f, 1.0f});
    expect_failure("nonfinite", PcmMixError::NonFinite, mix({mono32(inf)}, output));

    const float float_max = std::numeric_limits<float>::max();
    const std::vector<std::byte> unit = f32_bytes({1.0f, 1.0f, 1.0f});
    const PcmVoiceView over{PcmFormat::Float32LittleEndian, unit, 1, float_max, 1.0f};
    const PcmVoiceView over2{PcmFormat::Float32LittleEndian, unit, 1, float_max, 1.0f};
    expect_failure("overflow", PcmMixError::Overflow, mix({over, over2}, output));
    const PcmMixResult failed = mix({over, over2}, output);
    CHECK(!failed.ok());
    CHECK(!failed.detail.empty());
    CHECK(same_bytes(output, untouched));
}

void aliased_output_over_pcm16_voice_is_read_first() {
    // The mono voice's bytes live inside the destination floats. The result
    // must match a mix of the pre-call bytes, proving inputs were fully read
    // before the first output write.
    std::vector<float> output(8, 0.5f);  // each float: bytes 00 00 00 3F
    const auto* storage = reinterpret_cast<const std::byte*>(output.data());
    const std::span<const std::byte> aliased_samples(storage, 4 * 2);
    int expected_samples[4];
    for (std::size_t i = 0; i < 4; ++i) {
        const unsigned lo = static_cast<unsigned>(storage[i * 2]);
        const unsigned hi = static_cast<unsigned>(storage[i * 2 + 1]);
        expected_samples[i] = static_cast<signed short>(lo | (hi << 8));
    }
    const PcmVoiceView voice{PcmFormat::Signed16LittleEndian, aliased_samples, 1, 2.0f, 3.0f};
    const PcmMixResult result = mix({voice}, output);
    CHECK(result.ok());
    for (std::size_t frame = 0; frame < 4; ++frame) {
        const double sample = static_cast<double>(expected_samples[frame]) / 32768.0;
        CHECK(output[frame * 2] == static_cast<float>(sample * 2.0));
        CHECK(output[frame * 2 + 1] == static_cast<float>(sample * 3.0));
    }
}

void aliased_stereo_float_voice_covers_whole_output() {
    // Every output float is also a voice sample; late frames must mix the
    // original values, not values already written by earlier frames.
    std::vector<float> output(6);
    for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = static_cast<float>(i) * 0.25f - 0.5f;
    }
    const std::vector<float> original = output;
    const auto* storage = reinterpret_cast<const std::byte*>(output.data());
    const std::span<const std::byte> aliased_samples(storage, output.size() * 4);
    const PcmVoiceView voice{PcmFormat::Float32LittleEndian, aliased_samples, 2, 2.0f, -1.5f};
    const PcmMixResult result = mix({voice}, output);
    CHECK(result.ok());
    for (std::size_t frame = 0; frame < 3; ++frame) {
        CHECK(output[frame * 2] == original[frame * 2] * 2.0f);
        CHECK(output[frame * 2 + 1] == original[frame * 2 + 1] * -1.5f);
    }
}

void mixing_is_deterministic() {
    const std::vector<std::byte> a = pcm16_bytes({32767, -32768, 12345, -12345});
    const std::vector<std::byte> b = pcm16_bytes({-1, 1});
    std::vector<float> first(4, 0.0f);
    std::vector<float> second(4, 9.0f);
    CHECK(mix({stereo16(a), mono16(b, 0.5f)}, first).ok());
    CHECK(mix({stereo16(a), mono16(b, 0.5f)}, second).ok());
    CHECK(same_bytes(first, second));
}

}  // namespace

int main() {
    zero_voices_fill_output_with_zeros();
    pcm16_decodes_little_endian_extrema();
    mono_duplicates_and_stereo_preserves_channels();
    multiple_voices_accumulate_in_order();
    output_is_unclipped_float();
    float32_decodes_little_endian_and_mixes();
    nonfinite_inputs_are_rejected();
    overflow_beyond_float_range_is_rejected();
    voice_size_must_match_frames_and_channels();
    invalid_format_and_channel_counts_are_rejected();
    admission_limits_are_enforced();
    output_validation_rejects_odd_and_oversized();
    empty_output_is_invalid();
    unaligned_sample_storage_decodes_correctly();
    aliased_input_with_failing_mix_preserves_destination();
    late_channel_overflow_mutates_nothing();
    accumulation_follows_voice_order();
    failures_leave_destination_unchanged();
    aliased_output_over_pcm16_voice_is_read_first();
    aliased_stereo_float_voice_covers_whole_output();
    mixing_is_deterministic();
    if (failures == 0) {
        std::printf("pcm_mixer_tests: all checks passed\n");
        return 0;
    }
    std::printf("pcm_mixer_tests: %u check(s) failed\n", failures);
    return 1;
}
