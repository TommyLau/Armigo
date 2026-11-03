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

#include <cstddef>
#include <string>

namespace armigo::platform {

struct MetalProbeResult {
    bool supported = false;
    bool self_test_ran = false;
    bool self_test_passed = false;
    std::size_t verified_buffer_bytes = 0;
    std::size_t verified_pixels = 0;
    std::string device_name;
    std::string error;
};

// Exercises only synthetic platform operations; this does not execute a guest.
[[nodiscard]] MetalProbeResult probe_metal(bool run_self_test);

} // namespace armigo::platform
