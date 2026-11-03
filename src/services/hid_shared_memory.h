// ============================================================================
// Armigo Services - Human Interface Device Shared Memory
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
#include <span>

namespace armigo::services::hid::detail {

// The resource must contain at least 0x40000 bytes, aligned to eight bytes.
// Initialization precedes guest mapping; subsequent writers must be serialized.
void initialize_shared_memory(std::span<std::byte> memory);

// Publishes player one's neutral FullKey state.
void publish_headless_full_key(std::span<std::byte> memory, bool connected);

// One FullKey sample: NpadButton bits and stick positions in [-32767, 32767],
// up and right positive.
struct FullKeyInput {
    std::uint64_t buttons = 0;
    std::int32_t left_x = 0, left_y = 0, right_x = 0, right_y = 0;
};

// Publishes one input sample for player one's FullKey controller while it is
// connected; returns false and writes nothing otherwise.
bool publish_full_key_input(std::span<std::byte> memory, const FullKeyInput& input);

// One six-axis sensor sample in the controller's frame (x right, y forward,
// z up): acceleration in G, angular velocity in revolutions per second, the
// accumulated angle in revolutions and the orientation as three rows of a
// rotation matrix. delta_time is the sampling interval in nanoseconds.
struct SixAxisSample {
    std::int64_t delta_time = 0;
    std::array<float, 3> acceleration{};
    std::array<float, 3> angular_velocity{};
    std::array<float, 3> angle{};
    std::array<std::array<float, 3>, 3> direction{};
    bool connected = false;
};

// Publishes one sample to player one's FullKey six-axis ring while the
// controller is connected; returns false and writes nothing otherwise. Each
// entry's sampling number is the previous entry's plus one, which the SDK's
// reader requires of neighbouring entries.
bool publish_full_key_six_axis(std::span<std::byte> memory, const SixAxisSample& sample);

}  // namespace armigo::services::hid::detail
