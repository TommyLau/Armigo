// ============================================================================
// Armigo Audio - Horizon Audio Renderer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "renderer/common/types.h"

namespace armigo::audio::renderer {

// A Horizon result code: module in bits 0..8, description in bits 9..21.
struct Result {
    u32 raw = 0;

    constexpr bool is_success() const noexcept { return raw == 0; }
    constexpr bool is_error() const noexcept { return raw != 0; }
    constexpr bool is_failure() const noexcept { return raw != 0; }
    constexpr u32 get_module() const noexcept { return raw & 0x1ff; }
    constexpr u32 get_description() const noexcept { return (raw >> 9) & 0x1fff; }
    friend constexpr bool operator==(Result, Result) = default;
};
static_assert(sizeof(Result) == 4);

constexpr Result make_result(u32 module, u32 description) noexcept {
    return Result{module | (description << 9)};
}

inline constexpr u32 audio_result_module = 153;

inline constexpr Result ResultSuccess{0};
inline constexpr Result ResultNotFound = make_result(audio_result_module, 1);
inline constexpr Result ResultOperationFailed = make_result(audio_result_module, 2);
inline constexpr Result ResultInvalidSampleRate = make_result(audio_result_module, 3);
inline constexpr Result ResultInsufficientBuffer = make_result(audio_result_module, 4);
inline constexpr Result ResultOutOfSessions = make_result(audio_result_module, 5);
inline constexpr Result ResultBufferCountReached = make_result(audio_result_module, 8);
inline constexpr Result ResultInvalidChannelCount = make_result(audio_result_module, 10);
inline constexpr Result ResultInvalidUpdateInfo = make_result(audio_result_module, 41);
inline constexpr Result ResultInvalidAddressInfo = make_result(audio_result_module, 42);
inline constexpr Result ResultNotSupported = make_result(audio_result_module, 513);
inline constexpr Result ResultInvalidArgument = make_result(audio_result_module, 900);
inline constexpr Result ResultInvalidHandle = make_result(audio_result_module, 1536);
inline constexpr Result ResultInvalidRevision = make_result(audio_result_module, 1537);

}  // namespace armigo::audio::renderer
