// ============================================================================
// Armigo Audio - Host PCM Output
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
#include <memory>

namespace armigo::audio {

struct PcmCounters {
    std::uint64_t submitted_frames = 0;
    std::uint64_t consumed_frames = 0;
    std::uint64_t dropped_frames = 0;
    std::uint64_t underrun_frames = 0;
};

// Host output for rendered PCM: 48000 Hz frames that the platform plays. The
// audio renderer depends only on this interface; the platform layer supplies
// the device-backed implementation. Calls come from one producer at a time.
class PcmOutput {
public:
    virtual ~PcmOutput() = default;
    // Starting accepts frames; stopping discards queued frames. False on a
    // device failure, which leaves the output stopped.
    virtual bool set_active(bool active) = 0;
    // Copies interleaved PCM16 with 1, 2 or 6 channels and returns the frames
    // accepted. Never blocks; frames that do not fit are dropped and counted.
    virtual std::size_t write(const std::int16_t* samples, std::size_t frames,
                              unsigned channels) = 0;
    // Output gain applied when frames are played; finite values clamp to [0, 128].
    virtual bool set_volume(float volume) = 0;
    virtual PcmCounters counters() const = 0;
};

// An explicit muted output: active writes are accepted and counted as
// consumed immediately, and nothing is played. Headless runs and tests use it.
std::shared_ptr<PcmOutput> make_muted_pcm_output();

}  // namespace armigo::audio
