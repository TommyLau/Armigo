// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/audio/pcm_output.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace armigo::platform {

// Host-owned PCM output: fixed 48000 Hz stereo played by the system's default
// output device. No guest pointers or guest calls may enter the device
// callback. Framework types stay inside the implementation.
struct PcmSink;

// muted opens no device and consumes accepted frames immediately. audible and
// silent both open the default output device and drain a ring from its render
// callback. silent zero-fills the device buffer after that consumption, so the
// clock, the buffering and the counters match audible output while the samples
// that leave the callback are zero. A later volume change does not raise them.
enum class PcmOutputKind { muted, audible, silent };

// ARMIGO_AUDIO. A null or empty setting keeps the presentation default:
// audible when `windowed` is true, muted otherwise. "0" is muted, "silent" is
// the zero-level device path, and any other value is audible.
inline PcmOutputKind pcm_output_kind_from_setting(const char* setting, bool windowed) {
    if (setting == nullptr || setting[0] == '\0')
        return windowed ? PcmOutputKind::audible : PcmOutputKind::muted;
    const std::string_view value(setting);
    if (value == "0") return PcmOutputKind::muted;
    if (value == "silent") return PcmOutputKind::silent;
    return PcmOutputKind::audible;
}

struct PcmSinkCounters {
    std::uint64_t submitted_frames = 0;
    std::uint64_t consumed_frames = 0;
    std::uint64_t dropped_frames = 0;
    std::uint64_t underrun_frames = 0;
};

// Returns null on allocation or device failure. Initially stopped. Not
// realtime-safe. The bool overload is muted (false) or audible (true).
PcmSink* pcm_sink_open(PcmOutputKind kind);
inline PcmSink* pcm_sink_open(bool audible) {
    return pcm_sink_open(audible ? PcmOutputKind::audible : PcmOutputKind::muted);
}
// Caller serializes lifecycle and writes. Stop discards queued PCM; restart
// starts empty. Returns 0 on success, -1 on device failure. Not realtime-safe.
int pcm_sink_set_active(PcmSink* sink, bool active);
// Copies interleaved PCM16 with 1, 2 or 6 source channels and returns the
// accepted frames. Single producer; never blocks on the consumer; excess
// frames are counted as dropped. A muted active sink consumes accepted frames
// without opening a device.
std::size_t pcm_sink_write(PcmSink* sink, const std::int16_t* samples, std::size_t frames,
                           unsigned channels);
// Per-sink output gain applied when queued PCM is consumed (including frames
// already queued). Finite values clamp to [0, 128]; non-finite values are
// rejected without changing the gain. Initially 1.0. Returns 0 or -1.
int pcm_sink_set_volume(PcmSink* sink, float volume);
void pcm_sink_get_counters(const PcmSink* sink, PcmSinkCounters* out);
// Maximum absolute sample written into a device buffer while silent output is
// active. Stays 0 when that buffer is zero, and stays 0 for muted and audible
// sinks, which do not record a peak.
int pcm_sink_output_peak(const PcmSink* sink);
// Stops callbacks before releasing storage. No concurrent producer or API use.
void pcm_sink_close(PcmSink* sink);
// Restarts the device of every active audible sink after the system stopped it
// (an audio session interruption). Any thread; not realtime-safe.
void pcm_sink_resume_devices();

// The sink behind the portable audio output interface. Returns null when the
// device cannot be opened; the caller reports that and chooses a fallback.
// The bool overload is muted (false) or audible (true).
std::shared_ptr<audio::PcmOutput> make_pcm_sink_output(PcmOutputKind kind);
inline std::shared_ptr<audio::PcmOutput> make_pcm_sink_output(bool audible) {
    return make_pcm_sink_output(audible ? PcmOutputKind::audible : PcmOutputKind::muted);
}

}  // namespace armigo::platform
