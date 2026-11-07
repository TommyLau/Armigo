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

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace armigo::platform::metal::nvn_driver {

// Submission retirement and presentation markers share the backend fence
// mutex. Prefix differences combine every submission in a frame, including
// feedback that arrived before its presentation marker.
class GpuFrameWork {
public:
    void reset(std::uint64_t completed_fence = 0) noexcept {
        *this = {};
        completed_ = requested_ = previous_fence_ = completed_fence;
        // Only stream origin is a known boundary. Late collection may start
        // halfway through a frame; its first real boundary establishes the
        // baseline, never a sample containing just the remaining work.
        recovering_ = completed_fence != 0;
    }

    void retire(std::uint64_t fence, std::optional<std::uint64_t> work_ns) noexcept {
        if (fence <= completed_) return;
        if (fence != completed_ + 1) ++invalid_;
        completed_ = fence;
        if (work_ns) sum_ += *work_ns;
        else ++invalid_;
        while (count_ && pending_[head_].fence <= completed_) {
            finish(pending_[head_]);
            head_ = (head_ + 1) % pending_.size();
            --count_;
        }
    }

    void frame_boundary(std::uint64_t fence) noexcept {
        ++frame_;
        if (fence < requested_ || fence < completed_) {
            head_ = count_ = 0;
            recovering_ = true;
            sample_frame_ = 0;
            requested_ = completed_;
            return;
        }
        requested_ = fence;
        if (fence == completed_) {
            finish({fence, frame_});
            return;
        }
        if (count_ == pending_.size()) {
            // Timing loss must not publish a partial frame as a cheap frame.
            head_ = count_ = 0;
            recovering_ = true;
            sample_frame_ = 0;
        }
        pending_[(head_ + count_) % pending_.size()] = {fence, frame_};
        ++count_;
    }

    std::optional<std::uint64_t> latest() const noexcept {
        if (sample_frame_ && frame_ - sample_frame_ < 16) return sample_ns_;
        return {};
    }

private:
    struct Boundary { std::uint64_t fence = 0, frame = 0; };
    void finish(Boundary boundary) noexcept {
        if (!recovering_ && boundary.fence == completed_ && boundary.fence > previous_fence_ &&
            invalid_ == previous_invalid_) {
            sample_frame_ = boundary.frame;
            sample_ns_ = sum_ - previous_sum_;
        }
        previous_fence_ = boundary.fence;
        previous_sum_ = sum_;
        previous_invalid_ = invalid_;
        recovering_ = boundary.fence != completed_;
    }

    std::array<Boundary, 64> pending_{};
    std::size_t head_ = 0, count_ = 0;
    std::uint64_t completed_ = 0, requested_ = 0, frame_ = 0;
    std::uint64_t sum_ = 0, invalid_ = 0;
    std::uint64_t previous_fence_ = 0, previous_sum_ = 0, previous_invalid_ = 0;
    std::uint64_t sample_frame_ = 0, sample_ns_ = 0;
    bool recovering_ = false;
};

} // namespace armigo::platform::metal::nvn_driver
