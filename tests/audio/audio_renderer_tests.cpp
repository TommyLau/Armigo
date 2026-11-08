// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic audio renderer checks: DSP-ADPCM decoding of a hand-built frame,
// and a complete RequestUpdate -> render path that plays a PCM16 voice through
// the final mix into the device sink. Every input is written byte by byte at
// the guest ABI offsets; expected values are worked out by hand from the
// format definitions, never by the code under test.

#include <armigo/audio/audio_renderer.h>
#include <armigo/audio/pcm_output.h>

#include "renderer/command/data_source/decode.h"
#include "renderer/common/feature_support.h"
#include "renderer/common/guest_memory.h"
#include "renderer/voice/voice_state.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace {

unsigned failures = 0;
#define CHECK(condition)                                                    \
    do {                                                                    \
        if (!(condition)) {                                                 \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);    \
            ++failures;                                                     \
        }                                                                   \
    } while (0)

template <typename T>
void put(std::vector<std::byte>& bytes, std::size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof value);
}

template <typename T>
T get(const std::vector<std::byte>& bytes, std::size_t offset) {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof value);
    return value;
}

constexpr std::uint32_t revision(unsigned number) {
    return 0x00564552u + ((0x30u + number) << 24);  // "REV" followed by '0' + number
}
// A Switch 2 library's tag: "REv" followed by '0' + number.
constexpr std::uint32_t tagged_revision(unsigned number, char letter) {
    return 0x00004552u + (std::uint32_t(std::uint8_t(letter)) << 16) + ((0x30u + number) << 24);
}

// Pins only ranges inside one host allocation standing in for guest memory.
class TestPins final : public armigo::audio::GuestMemoryPins {
public:
    TestPins(const void* base, std::size_t size)
        : base_(reinterpret_cast<std::uint64_t>(base)), size_(size) {}
    bool pin(std::uint64_t address, std::uint64_t size) override {
        if (address < base_ || size > size_ || address - base_ > size_ - size) return false;
        ++pinned;
        return true;
    }
    void unpin(std::uint64_t, std::uint64_t) override { ++unpinned; }
    unsigned pinned = 0;
    unsigned unpinned = 0;

private:
    std::uint64_t base_;
    std::size_t size_;
};

// Records everything the device sink writes.
class CapturePcmOutput final : public armigo::audio::PcmOutput {
public:
    bool set_active(bool value) override {
        active = value;
        return true;
    }
    std::size_t write(const std::int16_t* samples, std::size_t frames, unsigned channels) override {
        if (!active) return 0;
        last_channels = channels;
        frames_written += frames;
        last_frame.assign(samples, samples + frames * channels);
        return frames;
    }
    bool set_volume(float) override { return true; }
    armigo::audio::PcmCounters counters() const override { return {frames_written, frames_written, 0, 0}; }

    bool active = false;
    unsigned last_channels = 0;
    std::uint64_t frames_written = 0;
    std::vector<std::int16_t> last_frame;
};

// Page-aligned host memory standing in for a guest memory pool.
struct PoolMemory {
    explicit PoolMemory(std::size_t bytes) : size(bytes) {
        data = static_cast<std::byte*>(std::aligned_alloc(0x1000, bytes));
        std::memset(data, 0, bytes);
    }
    ~PoolMemory() { std::free(data); }
    std::uint64_t address(std::size_t offset = 0) const {
        return reinterpret_cast<std::uint64_t>(data) + offset;
    }
    std::byte* data;
    std::size_t size;
};

// ---------------------------------------------------------------------------
// DSP-ADPCM: a frame is one header byte (predictor index in bits 4..6, scale
// in bits 0..3) and seven bytes holding fourteen signed nibbles, high nibble
// first. Each sample is
//   clamp16(((nibble << scale) << 11) + 1024 + c0 * yn1 + c1 * yn2) >> 11)
// with yn1/yn2 the previous two output samples.
std::vector<std::int16_t> decode_adpcm(armigo::audio::renderer::GuestMemory& memory,
                                       std::uint64_t frame, std::uint64_t coefficients,
                                       std::uint32_t samples) {
    using namespace armigo::audio::renderer;
    VoiceState state{};
    state.wave_buffer_valid[0] = true;
    std::array<WaveBufferVersion2, MaxWaveBuffers> wave_buffers{};
    wave_buffers[0].buffer = frame;
    wave_buffers[0].buffer_size = 8 * ((samples + 13) / 14);
    wave_buffers[0].start_offset = 0;
    wave_buffers[0].end_offset = samples;
    wave_buffers[0].stream_ended = false;  // an ended stream resets the played count
    std::vector<s32> output(samples, 0x7777);
    DecodeFromWaveBuffersArgs args{
        .sample_format = SampleFormat::Adpcm,
        .output = output,
        .voice_state = &state,
        .wave_buffers = wave_buffers,
        .channel = 0,
        .channel_count = 1,
        .src_quality = SrcQuality::Medium,
        .pitch = 1.0f,
        .source_sample_rate = 48000,
        .target_sample_rate = 48000,
        .sample_count = samples,
        .data_address = coefficients,
        .data_size = 32,
        .IsVoicePlayedSampleCountResetAtLoopPointSupported = false,
        .IsVoicePitchAndSrcSkippedSupported = true,
    };
    decode_from_wave_buffers(memory, args);
    CHECK(state.played_sample_count == samples);
    CHECK(state.wave_buffers_consumed == 1);
    return {output.begin(), output.end()};
}

void adpcm_tests() {
    PoolMemory pool(0x1000);
    auto pins = std::make_shared<TestPins>(pool.data, pool.size);
    armigo::audio::renderer::GuestMemory memory(pins);
    CHECK(memory.map(pool.address(), pool.size));

    // Zero predictor, scale 4: each sample is the nibble times 16. Nibbles
    // 7, -8, 1, -1, 0, 2, -2, 3, -3, 4, -4, 5, -5, 6.
    auto* frame = reinterpret_cast<std::uint8_t*>(pool.data);
    const std::uint8_t first[8] = {0x04, 0x78, 0x1f, 0x02, 0xe3, 0xd4, 0xc5, 0xb6};
    std::memcpy(frame, first, sizeof first);
    auto* coefficients = reinterpret_cast<std::int16_t*>(pool.data + 0x100);
    std::memset(coefficients, 0, 32);
    const std::int16_t zero_predictor[14] = {112, -128, 16, -16, 0, 32, -32, 48, -48, 64, -64, 80, -80, 96};
    auto decoded = decode_adpcm(memory, pool.address(), pool.address(0x100), 14);
    for (int index = 0; index < 14; ++index) CHECK(decoded[index] == zero_predictor[index]);

    // Predictor 1 = (2048, 0): each sample adds the nibble to the previous
    // one. Scale 0 and all nibbles 1 count up 1..14.
    coefficients[2] = 2048;
    coefficients[3] = 0;
    const std::uint8_t counting[8] = {0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11};
    std::memcpy(frame + 0x200, counting, sizeof counting);
    decoded = decode_adpcm(memory, pool.address(0x200), pool.address(0x100), 14);
    for (int index = 0; index < 14; ++index) CHECK(decoded[index] == index + 1);

    // Predictor 2 = (4096, -2048): yn = 2 * yn1 - yn2 + nibble. With scale 0
    // and nibbles 1, 0, 0, ... the output ramps 1, 2, 3, ... after the first.
    coefficients[4] = 4096;
    coefficients[5] = -2048;
    const std::uint8_t ramp[8] = {0x20, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::memcpy(frame + 0x300, ramp, sizeof ramp);
    decoded = decode_adpcm(memory, pool.address(0x300), pool.address(0x100), 14);
    for (int index = 0; index < 14; ++index) CHECK(decoded[index] == index + 1);

    // Saturation: predictor 1 with scale 11 and nibble 7 climbs past int16.
    const std::uint8_t loud[8] = {0x1b, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77};
    std::memcpy(frame + 0x400, loud, sizeof loud);
    decoded = decode_adpcm(memory, pool.address(0x400), pool.address(0x100), 14);
    CHECK(decoded[0] == 7 * 2048);
    CHECK(decoded[1] == 14 * 2048);
    CHECK(decoded[2] == 32767);
    CHECK(decoded[13] == 32767);

    // A frame outside every pinned range reads as silence instead of being
    // dereferenced, and the rejection is counted.
    armigo::audio::renderer::GuestMemory empty(pins);
    const auto before = empty.rejected_accesses();
    std::int32_t value = 1;
    CHECK(!empty.read_block(pool.address(), &value, sizeof value));
    CHECK(value == 0);
    CHECK(empty.rejected_accesses() == before + 1);
    CHECK(!memory.contains(pool.address(pool.size - 4), 8));
    CHECK(memory.contains(pool.address(pool.size - 8), 8));
}

// ---------------------------------------------------------------------------
// One renderer: 48 kHz, 240 samples, two mix buffers, one voice, one device
// sink, no effects, sub-mixes or splitters.
struct UpdateLayout {
    unsigned revision_number;
    std::uint32_t voice_stride;   // 0x170, or 0x188 from revision 15
    bool dirty_mixes;             // revision 7 and later prefix the mixes
    std::uint32_t effect_status;  // 0x10, or 0x90 from revision 9
    char open_tag = 'V';          // 'v' opens as a Switch 2 library
    char update_tag = 'V';        // the tag the updates carry
    char rejected_tag = 0;        // an update tag the renderer must refuse first
};

constexpr std::uint32_t pool_count = 4;  // effects + voices * 4 wave buffers
constexpr std::uint32_t mix_in_size = 0x930;

std::vector<std::byte> build_update(const UpdateLayout& layout, const PoolMemory& pool,
                                    std::uint32_t samples_offset, std::uint32_t samples, char tag = 0) {
    const auto update_revision = tagged_revision(layout.revision_number, tag ? tag : layout.update_tag);
    const std::uint32_t behaviour = 0x10;
    const std::uint32_t pools = pool_count * 0x20;
    const std::uint32_t resources = 0x70;
    const std::uint32_t voices = layout.voice_stride;
    const std::uint32_t mixes = (layout.dirty_mixes ? 0x20 : 0) + mix_in_size;
    const std::uint32_t sinks = 0x140;
    const std::uint32_t performance = 0x10;
    const std::uint32_t total = 0x40 + behaviour + pools + resources + voices + mixes + sinks + performance;
    std::vector<std::byte> input(total);
    // Update header.
    put<std::uint32_t>(input, 0x00, update_revision);
    put<std::uint32_t>(input, 0x04, behaviour);
    put<std::uint32_t>(input, 0x08, pools);
    put<std::uint32_t>(input, 0x0C, voices);
    put<std::uint32_t>(input, 0x10, resources);
    put<std::uint32_t>(input, 0x14, 0);
    put<std::uint32_t>(input, 0x18, mixes);
    put<std::uint32_t>(input, 0x1C, sinks);
    put<std::uint32_t>(input, 0x20, performance);
    put<std::uint32_t>(input, 0x24, 0);
    put<std::uint32_t>(input, 0x3C, total);
    std::size_t at = 0x40;
    // Behaviour: revision, flags (no forced mapping).
    put<std::uint32_t>(input, at, update_revision);
    at += behaviour;
    // Memory pool 0 requests attachment of the whole pool; the rest stay invalid.
    put<std::uint64_t>(input, at + 0x00, pool.address());
    put<std::uint64_t>(input, at + 0x08, pool.size);
    put<std::uint32_t>(input, at + 0x10, 4);  // RequestAttach
    at += pools;
    // Voice channel resource 0: mix volume 1.0 into buffer 0, 0.5 into buffer 1.
    put<std::uint32_t>(input, at + 0x00, 0);
    put<float>(input, at + 0x04, 1.0f);
    put<float>(input, at + 0x08, 0.5f);
    put<std::uint8_t>(input, at + 0x64, 1);
    at += resources;
    // Voice 0: new, in use, started, PCM16 mono at 48 kHz, pitch and volume
    // 1.0, final mix, no splitter, one wave buffer, pitch/SRC skipped.
    const std::size_t voice = at;
    put<std::uint32_t>(input, voice + 0x000, 0);           // id
    put<std::uint32_t>(input, voice + 0x004, 0x10010000);  // node id
    put<std::uint8_t>(input, voice + 0x008, 1);            // is_new
    put<std::uint8_t>(input, voice + 0x009, 1);            // in_use
    put<std::uint8_t>(input, voice + 0x00A, 0);            // Started
    put<std::uint8_t>(input, voice + 0x00B, 2);            // PcmInt16
    put<std::uint32_t>(input, voice + 0x00C, 48000);
    put<std::int32_t>(input, voice + 0x010, 32);           // priority
    put<std::int32_t>(input, voice + 0x014, 0);            // sort order
    put<std::uint32_t>(input, voice + 0x018, 1);           // channels
    put<float>(input, voice + 0x01C, 1.0f);                // pitch
    put<float>(input, voice + 0x020, 1.0f);                // volume
    const bool v2 = layout.voice_stride == 0x188;
    const std::size_t wave_count = v2 ? 0x054 : 0x03C;
    const std::size_t mix_id = v2 ? 0x070 : 0x058;
    const std::size_t splitter = v2 ? 0x074 : 0x05C;
    const std::size_t wave = v2 ? 0x078 : 0x060;
    const std::size_t resource_ids = v2 ? 0x158 : 0x140;
    const std::size_t flags = v2 ? 0x174 : 0x15C;
    put<std::uint32_t>(input, voice + wave_count, 1);
    put<std::uint32_t>(input, voice + mix_id, 0);                // final mix
    put<std::uint32_t>(input, voice + splitter, 0xFFFFFFFFu);   // no splitter
    put<std::uint64_t>(input, voice + wave + 0x00, pool.address(samples_offset));
    put<std::uint64_t>(input, voice + wave + 0x08, std::uint64_t{samples} * 2);
    put<std::int32_t>(input, voice + wave + 0x10, 0);
    put<std::int32_t>(input, voice + wave + 0x14, static_cast<std::int32_t>(samples));
    // Unused wave buffer slots are marked as already sent, as the guest does.
    for (std::size_t slot = 1; slot < 4; ++slot)
        put<std::uint8_t>(input, voice + wave + slot * 0x38 + 0x1A, 1);
    put<std::uint32_t>(input, voice + resource_ids, 0);
    put<std::uint8_t>(input, voice + flags, 0b10);              // pitch and SRC skipped
    at += voices;
    // Final mix: optional dirty-mix prefix, then mix 0 with two buffers.
    if (layout.dirty_mixes) {
        put<std::int32_t>(input, at + 0x04, 1);
        at += 0x20;
    }
    put<float>(input, at + 0x000, 1.0f);
    put<std::uint32_t>(input, at + 0x004, 48000);
    put<std::uint32_t>(input, at + 0x008, 2);
    put<std::uint8_t>(input, at + 0x00C, 1);
    put<std::uint8_t>(input, at + 0x00D, 1);
    put<std::int32_t>(input, at + 0x010, 0);
    put<std::int32_t>(input, at + 0x018, 0x20000000);
    put<std::int32_t>(input, at + 0x924, 0x7FFFFFFF);  // no destination mix
    put<std::int32_t>(input, at + 0x928, -1);          // no destination splitter
    at += mix_in_size;
    // Device sink: two inputs from mix buffers 0 and 1, no downmix.
    put<std::uint8_t>(input, at + 0x000, 1);
    put<std::uint8_t>(input, at + 0x001, 1);
    put<std::uint32_t>(input, at + 0x004, 0x30000000);
    std::memcpy(input.data() + at + 0x020, "MainAudioOut", 12);
    put<std::uint32_t>(input, at + 0x120, 2);
    put<std::int8_t>(input, at + 0x124, 0);
    put<std::int8_t>(input, at + 0x125, 1);
    at += sinks;
    at += performance;
    CHECK(at == total);
    return input;
}

void render_path_test(const UpdateLayout& layout) {
    constexpr std::uint32_t samples = 960;
    constexpr std::uint32_t samples_offset = 0x1000;
    PoolMemory pool(0x2000);
    auto* source = reinterpret_cast<std::int16_t*>(pool.data + samples_offset);
    for (std::uint32_t index = 0; index < samples; ++index)
        source[index] = static_cast<std::int16_t>((index * 37) % 4001 - 2000);

    armigo::audio::AudioRendererParameter parameter{};
    parameter.sample_rate = 48000;
    parameter.sample_count = 240;
    parameter.mixes = 2;
    parameter.voices = 1;
    parameter.sinks = 1;
    parameter.revision = tagged_revision(layout.revision_number, layout.open_tag);
    std::uint64_t work_buffer_size = 0;
    CHECK(armigo::audio::AudioRenderer::work_buffer_size(parameter, work_buffer_size) == 0);
    CHECK(work_buffer_size != 0 && work_buffer_size % 0x1000 == 0);

    auto pins = std::make_shared<TestPins>(pool.data, pool.size);
    auto output = std::make_shared<CapturePcmOutput>();
    unsigned frames_signalled = 0;
    armigo::audio::AudioRenderer renderer;
    armigo::audio::AudioRenderer::Options options;
    options.memory = pins;
    options.output = output;
    options.frame_rendered = [&] { ++frames_signalled; };
    options.render_thread = false;
    // A transfer memory smaller than the required work buffer is refused.
    CHECK(renderer.initialize(parameter, work_buffer_size - 0x1000, options) == (153u | (4u << 9)));
    CHECK(renderer.initialize(parameter, work_buffer_size, options) == 0);

    const std::uint32_t expected_output = 0x40 + pool_count * 0x10 + 0x10 + 0x20 + 0x10 + 0xB0 + 0x10;
    std::vector<std::byte> status(expected_output);
    std::vector<std::byte> performance;
    if (layout.rejected_tag) {
        auto refused = build_update(layout, pool, samples_offset, samples, layout.rejected_tag);
        CHECK(renderer.update(refused, performance, status) == (153u | (41u << 9)));
    }
    auto input = build_update(layout, pool, samples_offset, samples);
    CHECK(renderer.update(input, performance, status) == 0);
    // The renderer reports its own revision with the tag of the library.
    CHECK(get<std::uint32_t>(status, 0x00) ==
          (layout.open_tag == 'v' ? tagged_revision(16, 'v') : revision(15)));
    CHECK(get<std::uint32_t>(status, 0x04) == 0xB0);
    CHECK(get<std::uint32_t>(status, 0x08) == pool_count * 0x10);
    CHECK(get<std::uint32_t>(status, 0x0C) == 0x10);
    CHECK(get<std::uint32_t>(status, 0x14) == 0);
    CHECK(get<std::uint32_t>(status, 0x1C) == 0x20);
    CHECK(get<std::uint32_t>(status, 0x20) == 0x10);
    CHECK(get<std::uint32_t>(status, 0x28) == 0x10);
    CHECK(get<std::uint32_t>(status, 0x3C) == expected_output);
    CHECK(get<std::uint32_t>(status, 0x40) == 5);  // pool 0 attached
    CHECK(get<std::uint32_t>(status, 0x50) == 0);  // untouched pools report nothing
    CHECK(pins->pinned == 1);
    const std::size_t errors = 0x40 + pool_count * 0x10 + 0x10 + 0x20 + 0x10;
    CHECK(get<std::uint32_t>(status, errors + 0xA0) == 0);  // no errors reported

    // A wrong total size is rejected as invalid update info.
    auto truncated = input;
    truncated.push_back(std::byte{0});
    std::vector<std::byte> rejected(expected_output);
    CHECK(renderer.update(truncated, performance, rejected) == (153u | (41u << 9)));

    // Nothing renders while stopped.
    renderer.render_frame();
    CHECK(frames_signalled == 0 && output->frames_written == 0);

    renderer.start();
    CHECK(renderer.active());
    for (unsigned frame = 0; frame < 3; ++frame) renderer.render_frame();
    CHECK(frames_signalled == 3);
    CHECK(output->frames_written == 3 * 240);
    CHECK(output->last_channels == 2);
    // From the second frame on, the voice and mix volume ramps have settled:
    // the left channel carries the samples at 1.0, the right at 0.5 (the mix
    // multiply rounds toward negative infinity on odd samples).
    bool exact = output->last_frame.size() == 480;
    for (std::uint32_t index = 0; exact && index < 240; ++index) {
        const std::int32_t sample = source[480 + index];
        exact = output->last_frame[index * 2] == sample &&
                output->last_frame[index * 2 + 1] == (sample >> 1);
        if (!exact)
            std::fprintf(stderr, "frame sample %u: got %d/%d, expected %d/%d\n", index,
                         output->last_frame[index * 2], output->last_frame[index * 2 + 1], sample,
                         sample >> 1);
    }
    CHECK(exact);

    // The voice status reports the played samples after the next update.
    input = build_update(layout, pool, samples_offset, samples);
    put<std::uint8_t>(input, 0x40 + 0x10 + pool_count * 0x20 + 0x70 + 0x008, 0);  // no longer new
    put<std::uint32_t>(input, 0x40 + 0x10 + 0x10, 0);  // pool already attached: no request
    std::vector<std::byte> status2(expected_output);
    CHECK(renderer.update(input, performance, status2) == 0);
    const std::size_t voice_status = 0x40 + pool_count * 0x10;
    // Generation publishes the DSP state of the frames processed before it:
    // two of the three rendered frames.
    CHECK(get<std::uint64_t>(status2, voice_status + 0x00) == 2 * 240);
    CHECK(get<std::uint32_t>(status2, voice_status + 0x08) == 0);
    CHECK(get<std::uint32_t>(status2, 0x40 + pool_count * 0x10 + 0x10 + 0x20 + 0x10 + 0xB0) == 3);

    renderer.stop();
    CHECK(!renderer.active());
    CHECK(!output->active);
    renderer.finalize();
    CHECK(pins->pinned == 2);  // the rejected update re-attached the pool
    CHECK(pins->unpinned == pins->pinned);
}

// In auto execution the renderer renders on its own thread, one frame per
// 5 ms while started and none while stopped.
void render_thread_test() {
    PoolMemory pool(0x1000);
    armigo::audio::AudioRendererParameter parameter{};
    parameter.sample_rate = 48000;
    parameter.sample_count = 240;
    parameter.mixes = 2;
    parameter.voices = 1;
    parameter.sinks = 1;
    parameter.revision = revision(15);
    std::uint64_t work_buffer_size = 0;
    CHECK(armigo::audio::AudioRenderer::work_buffer_size(parameter, work_buffer_size) == 0);
    std::atomic<unsigned> frames{0};
    armigo::audio::AudioRenderer renderer;
    armigo::audio::AudioRenderer::Options options;
    options.memory = std::make_shared<TestPins>(pool.data, pool.size);
    options.output = armigo::audio::make_muted_pcm_output();
    options.frame_rendered = [&] { frames.fetch_add(1); };
    CHECK(renderer.initialize(parameter, work_buffer_size, options) == 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(frames.load() == 0);
    renderer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    renderer.stop();
    const auto rendered = frames.load();
    CHECK(rendered >= 5 && rendered <= 25);
    CHECK(renderer.rendered_frames() == rendered);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(frames.load() == rendered);
    CHECK(options.output->counters().submitted_frames == 0);  // no device sink was configured
    renderer.finalize();
}

}  // namespace

// Switch 2 libraries tag revisions 'REv' and accept one revision more; tags of
// neither platform carry no number.
void revision_tests() {
    namespace rn = armigo::audio::renderer;
    constexpr std::uint32_t native16 = 0x00764552u + ((0x30u + 16) << 24);  // "REv@"
    constexpr std::uint32_t native15 = 0x00764552u + ((0x30u + 15) << 24);
    static_assert(rn::is_native_revision(native16) && !rn::is_native_revision(revision(15)));
    static_assert(rn::get_revision_num(native16) == 16 && rn::get_revision_num(revision(15)) == 15);
    static_assert(rn::get_revision_num(native15) == rn::get_revision_num(revision(15)));
    static_assert(rn::get_revision_num(0x00584552u + (0x35u << 24)) == ~0u);  // "REX5"
    static_assert(rn::check_valid_revision(native16) && !rn::check_valid_revision(revision(16)));
    static_assert(rn::encode_revision(16, true) == native16 && rn::encode_revision(15, false) == revision(15));
    static_assert(!rn::check_feature_supported(rn::SupportTags::Splitter, 0x00584552u + (0x35u << 24)));
    static_assert(rn::check_feature_supported(rn::SupportTags::VoiceInParameterV2, native16));
}

int main() {
    revision_tests();
    adpcm_tests();
    render_path_test({.revision_number = 8, .voice_stride = 0x170, .dirty_mixes = true, .effect_status = 0x10});
    render_path_test({.revision_number = 15, .voice_stride = 0x188, .dirty_mixes = true, .effect_status = 0x90});
    render_path_test({.revision_number = 5, .voice_stride = 0x170, .dirty_mixes = false, .effect_status = 0x10});
    // A Switch 1 renderer needs the exact tag; a Switch 2 renderer accepts its
    // revision under either tag.
    render_path_test({.revision_number = 15, .voice_stride = 0x188, .dirty_mixes = true, .effect_status = 0x90,
                      .rejected_tag = 'v'});
    render_path_test({.revision_number = 15, .voice_stride = 0x188, .dirty_mixes = true, .effect_status = 0x90,
                      .open_tag = 'v', .update_tag = 'V'});
    render_path_test({.revision_number = 15, .voice_stride = 0x188, .dirty_mixes = true, .effect_status = 0x90,
                      .open_tag = 'v', .update_tag = 'v'});
    render_thread_test();
    std::printf("audio renderer failures: %u\n", failures);
    return failures ? 1 : 0;
}
