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

#include <cstdint>

namespace armigo::platform::metal::nvn_driver {

// A submission is committed in segments at render-pass boundaries once `draws`
// draws were encoded since the last cut, while fewer than `inflight` segments
// are on the GPU. Cutting after every pass that drew keeps the GPU fed while
// the rest of the submission encodes; for title 01007EF00011E000 it ran 27%
// faster uncapped than 16-draw segments.
struct SegmentPolicy {
    std::uint64_t draws = 1;
    std::uint64_t inflight = 16;
    bool admit(std::uint64_t since_cut, std::uint64_t outstanding) const {
        return draws && since_cut >= draws && (!inflight || outstanding < inflight);
    }
};

// The owner serializes every access. A callback only finishes a Metal segment;
// guest retirement additionally requires the worker to seal the submission.
struct SubmissionSegments {
    std::uint64_t submitted = 0, completed = 0;
    bool sealed = false, retired = false, failed = false;
    void submit() { ++submitted; }
    void complete(bool error) { ++completed; failed |= error; }
    void seal() { sealed = true; }
    bool drained() const { return completed == submitted; }
    bool retire() {
        if (!sealed || !drained() || retired) return false;
        retired = true;
        return true;
    }
};

enum class SegmentKind : std::uint8_t { cut, drain, final };

} // namespace armigo::platform::metal::nvn_driver
