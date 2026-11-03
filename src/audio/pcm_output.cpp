// ============================================================================
// Armigo Audio - Host PCM Output
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/audio/pcm_output.h>

#include <atomic>
#include <cmath>

namespace armigo::audio {
namespace {

// The explicit muted output: no device, no ring. Writes while stopped are
// rejected and uncounted; active writes are consumed on the spot.
class MutedPcmOutput final : public PcmOutput {
public:
    bool set_active(bool active) override {
        active_.store(active, std::memory_order_relaxed);
        return true;
    }
    std::size_t write(const std::int16_t* samples, std::size_t frames, unsigned channels) override {
        if (samples == nullptr || frames == 0) return 0;
        if (channels != 1 && channels != 2 && channels != 6) return 0;
        if (!active_.load(std::memory_order_relaxed)) return 0;
        submitted_.fetch_add(frames, std::memory_order_relaxed);
        consumed_.fetch_add(frames, std::memory_order_relaxed);
        return frames;
    }
    bool set_volume(float volume) override { return std::isfinite(volume); }
    PcmCounters counters() const override {
        return {submitted_.load(std::memory_order_relaxed), consumed_.load(std::memory_order_relaxed), 0, 0};
    }

private:
    std::atomic<bool> active_{false};
    std::atomic<std::uint64_t> submitted_{0};
    std::atomic<std::uint64_t> consumed_{0};
};

}  // namespace

std::shared_ptr<PcmOutput> make_muted_pcm_output() { return std::make_shared<MutedPcmOutput>(); }

}  // namespace armigo::audio
