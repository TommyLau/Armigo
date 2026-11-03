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

namespace armigo::platform {

// Host-owned PCM output: fixed 48000 Hz stereo played by the system's default
// output device. No guest pointers or guest calls may enter the device
// callback. Framework types stay inside the implementation.
struct PcmSink;

struct PcmSinkCounters {
    std::uint64_t submitted_frames = 0;
    std::uint64_t consumed_frames = 0;
    std::uint64_t dropped_frames = 0;
    std::uint64_t underrun_frames = 0;
};

// audible = false is an explicit muted sink. Returns null on allocation or
// device failure. Initially stopped. Not realtime-safe.
PcmSink* pcm_sink_open(bool audible);
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
// Stops callbacks before releasing storage. No concurrent producer or API use.
void pcm_sink_close(PcmSink* sink);
// Restarts the device of every active audible sink after the system stopped it
// (an audio session interruption). Any thread; not realtime-safe.
void pcm_sink_resume_devices();

// The sink behind the portable audio output interface. Returns null when the
// device cannot be opened; the caller reports that and chooses a fallback.
std::shared_ptr<audio::PcmOutput> make_pcm_sink_output(bool audible);

}  // namespace armigo::platform
