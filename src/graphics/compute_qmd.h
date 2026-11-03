// ============================================================================
// Armigo Graphics - GPU Engine Front End
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Queue metadata fields are extracted into the bounded ComputeLaunch contract.

#pragma once

#include <armigo/graphics/engines.h>
#include <array>
#include <cstddef>
#include <span>

namespace armigo::graphics {

inline constexpr std::size_t compute_qmd_bytes = 0x100;

// The launch fields the compute front end and Metal backend consume, decoded
// from the 256-byte queue-meta-data descriptor the guest publishes before
// method 0xaf. Grid dimensions select dispatched threadgroups; block
// dimensions select the threads of one threadgroup.
[[nodiscard]] ComputeLaunch decode_compute_qmd(
    std::span<const std::byte, compute_qmd_bytes> descriptor) noexcept;

}  // namespace armigo::graphics
