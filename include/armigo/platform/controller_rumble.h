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

namespace armigo::platform {

// Drives the host game controller's haptics from player one's two motor
// levels, each in [0, 1] (out-of-range and non-finite values are clamped; 0
// stops). A controller with haptics on its left and right handles plays each
// level on its own side; one with a single actuator plays the stronger level;
// one without haptics is ignored. The latest pair wins: updates are coalesced
// and applied on a queue of their own. Thread-safe; never blocks.
void set_controller_rumble(float left, float right) noexcept;

// The combined strength of a Switch motor's two bands (amplitudes in [0, 1]):
// a value using either band alone keeps its amplitude; the frequencies are
// dropped, as the host's actuators take only a strength.
float rumble_level(float amplitude_low, float amplitude_high) noexcept;

} // namespace armigo::platform
