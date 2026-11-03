// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/host1x/host1x.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>

namespace armigo::host1x {

// ARMIGO_HOST1X_TRACE=1 makes the engines print each picture they decode or
// compose, stamped with the seconds since the first trace.
inline bool tracing() {
    static const bool enabled = [] {
        const char* value = std::getenv("ARMIGO_HOST1X_TRACE");
        return value && value[0] == '1';
    }();
    return enabled;
}
inline double trace_seconds() {
    static const auto start = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

std::unique_ptr<CommandPusher> make_video_decoder(Hub& hub, std::uint32_t fd, std::uint32_t syncpoint);
std::unique_ptr<CommandPusher> make_video_compositor(Hub& hub, std::uint32_t fd, std::uint32_t syncpoint);

} // namespace armigo::host1x
