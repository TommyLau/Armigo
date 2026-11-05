// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Host PCM sink with the AudioToolbox calls replaced by the mocks below
// (the sink is built with ARMIGO_PCM_SINK_MOCK_AUDIO_UNIT for this test). The
// test plays the AudioUnit render thread by invoking the callback the sink
// registers. No audio device is opened. Expected samples are written by hand
// from the documented channel mapping and gain rules.

#include <armigo/platform/pcm_sink.h>

#include <AudioToolbox/AudioToolbox.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
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

struct MockUnit {
    std::string calls;  // N new, S set property, I initialize, A start, O stop, U uninitialize, D dispose
    OSStatus fail_new = 0, fail_initialize = 0, fail_start = 0, fail_dispose = 0;
    AudioStreamBasicDescription format{};
    AURenderCallbackStruct callback{};
    bool alive = false, initialized = false, started = false;
} mock;
int dummy_unit;

AudioUnit unit_handle() { return reinterpret_cast<AudioUnit>(&dummy_unit); }

}  // namespace

extern "C" {
OSStatus armigo_pcm_mock_new(const AudioComponentDescription* description, AudioUnit* out) {
    mock.calls += 'N';
    if (description->componentType != kAudioUnitType_Output ||
        description->componentSubType != kAudioUnitSubType_DefaultOutput ||
        description->componentManufacturer != kAudioUnitManufacturer_Apple)
        std::abort();
    if (mock.fail_new) {
        *out = nullptr;
        return mock.fail_new;
    }
    mock.alive = true;
    *out = unit_handle();
    return noErr;
}
OSStatus armigo_pcm_mock_set_property(AudioUnit unit, AudioUnitPropertyID id, AudioUnitScope scope,
                                      AudioUnitElement element, const void* data, UInt32 size) {
    mock.calls += 'S';
    if (unit != unit_handle() || !mock.alive || scope != kAudioUnitScope_Input || element != 0)
        std::abort();
    if (id == kAudioUnitProperty_StreamFormat && size == sizeof mock.format)
        std::memcpy(&mock.format, data, size);
    else if (id == kAudioUnitProperty_SetRenderCallback && size == sizeof mock.callback)
        std::memcpy(&mock.callback, data, size);
    else
        std::abort();
    return noErr;
}
OSStatus armigo_pcm_mock_initialize(AudioUnit) {
    mock.calls += 'I';
    if (mock.fail_initialize) return mock.fail_initialize;
    mock.initialized = true;
    return noErr;
}
OSStatus armigo_pcm_mock_uninitialize(AudioUnit) {
    mock.calls += 'U';
    if (mock.started) return -10863;
    mock.initialized = false;
    return noErr;
}
OSStatus armigo_pcm_mock_start(AudioUnit) {
    mock.calls += 'A';
    if (!mock.initialized) std::abort();
    if (mock.fail_start) return mock.fail_start;
    mock.started = true;
    return noErr;
}
OSStatus armigo_pcm_mock_stop(AudioUnit) {
    mock.calls += 'O';
    mock.started = false;
    return noErr;
}
OSStatus armigo_pcm_mock_dispose(AudioUnit) {
    mock.calls += 'D';
    if (mock.fail_dispose) return mock.fail_dispose;
    mock.alive = mock.initialized = mock.started = false;
    return noErr;
}
}

namespace {

using armigo::platform::PcmSink;
using armigo::platform::PcmSinkCounters;

// One device pull of `frames` interleaved stereo frames, as the AudioUnit does.
std::vector<std::int16_t> pull(std::uint32_t frames, AudioUnitRenderActionFlags* flags_out = nullptr) {
    std::vector<std::int16_t> buffer(frames * 2, 0x5555);
    AudioBufferList list{};
    list.mNumberBuffers = 1;
    list.mBuffers[0].mNumberChannels = 2;
    list.mBuffers[0].mDataByteSize = frames * 4;
    list.mBuffers[0].mData = buffer.data();
    AudioUnitRenderActionFlags flags = 0;
    const AudioTimeStamp timestamp{};
    CHECK(mock.callback.inputProc != nullptr);
    if (mock.callback.inputProc)
        mock.callback.inputProc(mock.callback.inputProcRefCon, &flags, &timestamp, 0, frames, &list);
    if (flags_out) *flags_out = flags;
    return buffer;
}

PcmSinkCounters counters(const PcmSink* sink) {
    PcmSinkCounters value;
    armigo::platform::pcm_sink_get_counters(sink, &value);
    return value;
}

void lifecycle_and_format() {
    mock = {};
    PcmSink* sink = armigo::platform::pcm_sink_open(true);
    CHECK(sink != nullptr);
    CHECK(mock.calls == "NSSI");
    CHECK(mock.format.mSampleRate == 48000.0);
    CHECK(mock.format.mFormatID == kAudioFormatLinearPCM);
    CHECK(mock.format.mChannelsPerFrame == 2 && mock.format.mBitsPerChannel == 16);
    CHECK(mock.format.mBytesPerFrame == 4 && mock.format.mFramesPerPacket == 1);
    CHECK((mock.format.mFormatFlags & kLinearPCMFormatFlagIsSignedInteger) != 0);
    // Writes while stopped are refused and uncounted.
    const std::int16_t stereo[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(armigo::platform::pcm_sink_write(sink, stereo, 4, 2) == 0);
    CHECK(counters(sink).submitted_frames == 0);
    CHECK(armigo::platform::pcm_sink_set_active(sink, true) == 0);
    CHECK(mock.calls == "NSSIA");
    CHECK(armigo::platform::pcm_sink_set_active(sink, false) == 0);
    armigo::platform::pcm_sink_close(sink);
    CHECK(mock.calls == "NSSIAOUD");
    CHECK(!mock.alive);
}

void ring_and_channel_mapping() {
    mock = {};
    PcmSink* sink = armigo::platform::pcm_sink_open(true);
    CHECK(armigo::platform::pcm_sink_set_active(sink, true) == 0);

    // Stereo is queued unchanged; the device pull drains it and zero-fills
    // the frames that are missing, counting them as underruns.
    const std::int16_t stereo[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(armigo::platform::pcm_sink_write(sink, stereo, 4, 2) == 4);
    AudioUnitRenderActionFlags flags = 0;
    auto out = pull(6, &flags);
    const std::int16_t expected_stereo[12] = {1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0};
    CHECK(std::memcmp(out.data(), expected_stereo, sizeof expected_stereo) == 0);
    CHECK((flags & kAudioUnitRenderAction_OutputIsSilence) == 0);
    auto value = counters(sink);
    CHECK(value.submitted_frames == 4 && value.consumed_frames == 4 && value.underrun_frames == 2);

    // An empty pull is flagged as silence.
    out = pull(2, &flags);
    CHECK((flags & kAudioUnitRenderAction_OutputIsSilence) != 0);
    CHECK(out[0] == 0 && out[3] == 0);

    // Mono becomes dual mono.
    const std::int16_t mono[2] = {100, -200};
    CHECK(armigo::platform::pcm_sink_write(sink, mono, 2, 1) == 2);
    out = pull(2);
    CHECK(out[0] == 100 && out[1] == 100 && out[2] == -200 && out[3] == -200);

    // 5.1 (FL, FR, C, LFE, RL, RR) folds to Lo/Ro in Q15 without the LFE:
    // L = (23170 FL + 16384 C + 16384 RL) >> 15 and R symmetric, saturated.
    const std::int16_t surround[12] = {10000, -10000, 8000, 30000, 4000, -4000,
                                       32767, 0, 32767, 0, 32767, 0};
    CHECK(armigo::platform::pcm_sink_write(sink, surround, 2, 6) == 2);
    out = pull(2);
    CHECK(out[0] == 13070 && out[1] == -5071);
    CHECK(out[2] == 32767 && out[3] == 16383);

    // Unsupported channel counts and empty writes are refused.
    CHECK(armigo::platform::pcm_sink_write(sink, stereo, 1, 4) == 0);
    CHECK(armigo::platform::pcm_sink_write(sink, nullptr, 1, 2) == 0);

    // The ring holds 32768 frames; frames beyond are dropped and counted.
    std::vector<std::int16_t> block(2 * 40000, 7);
    const auto before = counters(sink);
    CHECK(armigo::platform::pcm_sink_write(sink, block.data(), 40000, 2) == 32768);
    value = counters(sink);
    CHECK(value.dropped_frames - before.dropped_frames == 40000 - 32768);
    CHECK(armigo::platform::pcm_sink_write(sink, block.data(), 1, 2) == 0);
    // Wrap-around: drain part of the ring, refill across the end, read back.
    out = pull(1000);
    CHECK(out[0] == 7 && out[1999] == 7);
    const std::int16_t marker[4] = {11, 12, 13, 14};
    CHECK(armigo::platform::pcm_sink_write(sink, marker, 2, 2) == 2);
    out = pull(31770);
    CHECK(out[2 * 31767] == 7 && out[2 * 31768] == 11 && out[2 * 31769 + 1] == 14);

    // Stopping discards what is still queued and counts it as dropped.
    CHECK(armigo::platform::pcm_sink_write(sink, stereo, 4, 2) == 4);
    const auto queued = counters(sink);
    CHECK(armigo::platform::pcm_sink_set_active(sink, false) == 0);
    CHECK(counters(sink).dropped_frames == queued.dropped_frames + 4);
    // Restarting starts empty.
    CHECK(armigo::platform::pcm_sink_set_active(sink, true) == 0);
    out = pull(1, &flags);
    CHECK((flags & kAudioUnitRenderAction_OutputIsSilence) != 0);
    armigo::platform::pcm_sink_close(sink);
}

void gain() {
    mock = {};
    PcmSink* sink = armigo::platform::pcm_sink_open(true);
    CHECK(armigo::platform::pcm_sink_set_active(sink, true) == 0);
    // Frames queued before the change come out at the new gain; the exact
    // product truncates toward zero and saturates to int16.
    const std::int16_t frames[4] = {32767, -1, 20000, -20000};
    CHECK(armigo::platform::pcm_sink_write(sink, frames, 2, 2) == 2);
    CHECK(armigo::platform::pcm_sink_set_volume(sink, 0.5f) == 0);
    auto out = pull(1);
    CHECK(out[0] == 16383 && out[1] == 0);
    CHECK(armigo::platform::pcm_sink_set_volume(sink, 2.0f) == 0);
    out = pull(1);
    CHECK(out[0] == 32767 && out[1] == -32768);
    // Non-finite gains are refused and keep the current gain; finite values
    // outside [0, 128] clamp.
    CHECK(armigo::platform::pcm_sink_set_volume(sink, std::numeric_limits<float>::quiet_NaN()) == -1);
    CHECK(armigo::platform::pcm_sink_set_volume(sink, std::numeric_limits<float>::infinity()) == -1);
    CHECK(armigo::platform::pcm_sink_write(sink, frames, 1, 2) == 1);
    out = pull(1);
    CHECK(out[0] == 32767 && out[1] == -2);
    CHECK(armigo::platform::pcm_sink_set_volume(sink, -3.0f) == 0);
    CHECK(armigo::platform::pcm_sink_write(sink, frames, 1, 2) == 1);
    out = pull(1);
    CHECK(out[0] == 0 && out[1] == 0);
    // A zero gain still consumes the frames: no underrun is counted.
    const auto value = counters(sink);
    CHECK(value.consumed_frames == 4 && value.underrun_frames == 0);
    armigo::platform::pcm_sink_close(sink);
}

void muted_and_failures() {
    // The muted sink never touches AudioToolbox and consumes accepted frames.
    mock = {};
    PcmSink* muted = armigo::platform::pcm_sink_open(false);
    CHECK(muted != nullptr && mock.calls.empty());
    const std::int16_t stereo[4] = {1, 2, 3, 4};
    CHECK(armigo::platform::pcm_sink_write(muted, stereo, 2, 2) == 0);
    CHECK(armigo::platform::pcm_sink_set_active(muted, true) == 0);
    CHECK(armigo::platform::pcm_sink_write(muted, stereo, 2, 2) == 2);
    const auto value = counters(muted);
    CHECK(value.submitted_frames == 2 && value.consumed_frames == 2);
    armigo::platform::pcm_sink_close(muted);
    CHECK(mock.calls.empty());

    // A device that cannot be initialized fails the open explicitly and the
    // half-built unit is disposed.
    mock = {};
    mock.fail_initialize = -1;
    CHECK(armigo::platform::pcm_sink_open(true) == nullptr);
    CHECK(mock.calls == "NSSID");

    // A start failure leaves the sink stopped.
    mock = {};
    mock.fail_start = -2;
    PcmSink* sink = armigo::platform::pcm_sink_open(true);
    CHECK(armigo::platform::pcm_sink_set_active(sink, true) == -1);
    CHECK(armigo::platform::pcm_sink_write(sink, stereo, 2, 2) == 0);
    mock.fail_start = 0;
    armigo::platform::pcm_sink_close(sink);

    // The portable output wrapper follows the same rules.
    mock = {};
    auto output = armigo::platform::make_pcm_sink_output(true);
    CHECK(output != nullptr);
    if (output) {
        CHECK(output->set_active(true));
        CHECK(output->write(stereo, 2, 2) == 2);
        CHECK(output->set_volume(1.0f));
        CHECK(!output->set_volume(std::numeric_limits<float>::quiet_NaN()));
        CHECK(output->counters().submitted_frames == 2);
        output.reset();
        CHECK(mock.calls == "NSSIAOUD");
    }
}

void audio_setting() {
    using armigo::platform::PcmOutputKind;
    using armigo::platform::pcm_output_kind_from_setting;
    CHECK(pcm_output_kind_from_setting(nullptr, false) == PcmOutputKind::muted);
    CHECK(pcm_output_kind_from_setting(nullptr, true) == PcmOutputKind::audible);
    CHECK(pcm_output_kind_from_setting("", false) == PcmOutputKind::muted);
    CHECK(pcm_output_kind_from_setting("", true) == PcmOutputKind::audible);
    CHECK(pcm_output_kind_from_setting("0", true) == PcmOutputKind::muted);
    CHECK(pcm_output_kind_from_setting("0", false) == PcmOutputKind::muted);
    CHECK(pcm_output_kind_from_setting("1", false) == PcmOutputKind::audible);
    CHECK(pcm_output_kind_from_setting("1", true) == PcmOutputKind::audible);
    CHECK(pcm_output_kind_from_setting("silent", false) == PcmOutputKind::silent);
    CHECK(pcm_output_kind_from_setting("silent", true) == PcmOutputKind::silent);
    // Any other value stays on the historical audible path, including a
    // different spelling of the silent setting.
    CHECK(pcm_output_kind_from_setting("Silent", false) == PcmOutputKind::audible);
    CHECK(pcm_output_kind_from_setting("2", true) == PcmOutputKind::audible);
}

void silent_render() {
    using armigo::platform::PcmOutputKind;
    mock = {};
    PcmSink* sink = armigo::platform::pcm_sink_open(PcmOutputKind::silent);
    CHECK(sink != nullptr);
    CHECK(mock.calls == "NSSI");
    const std::int16_t loud[8] = {32767, -32768, 1000, -1000, 5, 6, 7, 8};
    // Stopped writes are refused. An active silent sink queues, it does not
    // consume on the spot the way the muted sink does.
    CHECK(armigo::platform::pcm_sink_write(sink, loud, 4, 2) == 0);
    CHECK(armigo::platform::pcm_sink_set_active(sink, true) == 0);
    CHECK(mock.calls == "NSSIA");
    CHECK(armigo::platform::pcm_sink_write(sink, loud, 4, 2) == 4);
    auto value = counters(sink);
    CHECK(value.submitted_frames == 4 && value.consumed_frames == 0);
    // The guest volume is stored and does not raise the device buffer.
    CHECK(armigo::platform::pcm_sink_set_volume(sink, 128.0f) == 0);
    AudioUnitRenderActionFlags flags = 0;
    auto out = pull(6, &flags);
    for (std::int16_t sample : out) CHECK(sample == 0);
    CHECK((flags & kAudioUnitRenderAction_OutputIsSilence) == 0);
    value = counters(sink);
    CHECK(value.submitted_frames == 4 && value.consumed_frames == 4);
    CHECK(value.underrun_frames == 2 && value.dropped_frames == 0);
    CHECK(armigo::platform::pcm_sink_output_peak(sink) == 0);
    out = pull(2, &flags);
    CHECK((flags & kAudioUnitRenderAction_OutputIsSilence) != 0);
    CHECK(out[0] == 0 && out[3] == 0);
    CHECK(counters(sink).underrun_frames == 4);
    CHECK(armigo::platform::pcm_sink_output_peak(sink) == 0);
    armigo::platform::pcm_sink_close(sink);
    CHECK(mock.calls == "NSSIAOUD");

    // Same writes and the same pull: silent at full guest volume matches an
    // audible sink whose gain is zero, sample for sample and counter for counter.
    mock = {};
    PcmSink* audible = armigo::platform::pcm_sink_open(PcmOutputKind::audible);
    CHECK(armigo::platform::pcm_sink_set_active(audible, true) == 0);
    CHECK(armigo::platform::pcm_sink_write(audible, loud, 4, 2) == 4);
    CHECK(armigo::platform::pcm_sink_set_volume(audible, 0.0f) == 0);
    flags = 0;
    auto zero = pull(6, &flags);
    for (std::int16_t sample : zero) CHECK(sample == 0);
    CHECK((flags & kAudioUnitRenderAction_OutputIsSilence) == 0);
    const auto audible_counters = counters(audible);
    CHECK(audible_counters.submitted_frames == 4 && audible_counters.consumed_frames == 4);
    CHECK(audible_counters.underrun_frames == 2 && audible_counters.dropped_frames == 0);
    CHECK(armigo::platform::pcm_sink_output_peak(audible) == 0);
    armigo::platform::pcm_sink_close(audible);

    // The audible path still renders the queued samples once the gain is unity.
    mock = {};
    audible = armigo::platform::pcm_sink_open(true);
    CHECK(armigo::platform::pcm_sink_set_active(audible, true) == 0);
    CHECK(armigo::platform::pcm_sink_write(audible, loud, 1, 2) == 1);
    out = pull(1);
    CHECK(out[0] == 32767 && out[1] == -32768);
    armigo::platform::pcm_sink_close(audible);
}

void silent_dump_keeps_source() {
    const auto path = std::filesystem::temp_directory_path() / "armigo-pcm-silent-dump.wav";
    std::filesystem::remove(path);
    CHECK(setenv("ARMIGO_AUDIO_DUMP", path.c_str(), 1) == 0);
    mock = {};
    auto output = armigo::platform::make_pcm_sink_output(armigo::platform::PcmOutputKind::silent);
    unsetenv("ARMIGO_AUDIO_DUMP");
    CHECK(output != nullptr);
    if (!output) return;
    const std::int16_t stereo[4] = {1000, -2000, 3000, -4000};
    CHECK(output->set_active(true));
    CHECK(output->write(stereo, 2, 2) == 2);
    CHECK(output->set_volume(64.0f));
    auto rendered = pull(2);
    CHECK(rendered[0] == 0 && rendered[1] == 0 && rendered[2] == 0 && rendered[3] == 0);
    CHECK(output->counters().submitted_frames == 2);
    output.reset();
    std::FILE* file = std::fopen(path.c_str(), "rb");
    CHECK(file != nullptr);
    if (file) {
        CHECK(std::fseek(file, 0, SEEK_END) == 0);
        CHECK(std::ftell(file) == 44 + 8);
        CHECK(std::fseek(file, 44, SEEK_SET) == 0);
        std::int16_t samples[4] = {};
        CHECK(std::fread(samples, sizeof(std::int16_t), 4, file) == 4);
        CHECK(samples[0] == 1000 && samples[1] == -2000);
        CHECK(samples[2] == 3000 && samples[3] == -4000);
        std::fclose(file);
    }
    std::filesystem::remove(path);
    CHECK(mock.calls == "NSSIAOUD");
}

}  // namespace

int main() {
    lifecycle_and_format();
    ring_and_channel_mapping();
    gain();
    muted_and_failures();
    audio_setting();
    silent_render();
    silent_dump_keeps_source();
    std::printf("pcm sink failures: %u\n", failures);
    return failures ? 1 : 0;
}
