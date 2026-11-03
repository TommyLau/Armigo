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

#include <cstddef>
#include <cstdint>

namespace armigo::audio::renderer {

// Fixed-width aliases used throughout the renderer, matching the guest ABI
// field widths of the renderer's parameter and status structures.
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s8 = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;
using f32 = float;
using f64 = double;

// A guest (or renderer-internal) address. Guest code runs natively, so a guest
// address that lies in an attached memory pool is also a host address; the
// renderer only dereferences it through GuestMemory, which validates it first.
using CpuAddr = std::uintptr_t;
using VAddr = std::uint64_t;

}  // namespace armigo::audio::renderer
