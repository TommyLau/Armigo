// ============================================================================
// Armigo Graphics - DMA Copy Engine
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// DMA methods decode source and destination surfaces into bounded CPU copies.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <variant>

namespace armigo::graphics {

// Method indices of the DMA copy class (0xb0b5) a launch reads. Each block
// size register is followed by width, height, depth, layer and origin.
namespace dma_method {
inline constexpr std::uint32_t semaphore_a = 0x90;
inline constexpr std::uint32_t semaphore_b = 0x91;
inline constexpr std::uint32_t semaphore_payload = 0x92;
inline constexpr std::uint32_t launch = 0xc0;
inline constexpr std::uint32_t offset_in_upper = 0x100;
inline constexpr std::uint32_t offset_in_lower = 0x101;
inline constexpr std::uint32_t offset_out_upper = 0x102;
inline constexpr std::uint32_t offset_out_lower = 0x103;
inline constexpr std::uint32_t pitch_in = 0x104;
inline constexpr std::uint32_t pitch_out = 0x105;
inline constexpr std::uint32_t line_length_in = 0x106;
inline constexpr std::uint32_t line_count = 0x107;
inline constexpr std::uint32_t remap_const_a = 0x1c0;
inline constexpr std::uint32_t remap_const_b = 0x1c1;
inline constexpr std::uint32_t remap_components = 0x1c2;
inline constexpr std::uint32_t dst_block_size = 0x1c3;
inline constexpr std::uint32_t src_block_size = 0x1ca;
inline constexpr std::uint32_t count = 0x1d0;
} // namespace dma_method

// The guest bytes one launch reads and writes. A destination that the copy
// does not cover completely (pitch padding, partial GOBs) is read first and
// written back whole, so the bytes between copied pixels keep their values.
struct DmaCopyPlan {
    std::uint64_t source = 0;
    std::uint64_t source_size = 0;
    std::uint64_t destination = 0;
    std::uint64_t destination_size = 0;
    bool destination_covered = false;
    // Launch semaphore release: 0 none, 1 one word, 2 four words.
    std::uint32_t semaphore = 0;
};

struct DmaCopyError {
    std::string_view detail;
};

// Plans a launch from the class registers (at least dma_method::count words).
// Launches outside the implemented subset return an explicit error: layered
// or 3D surfaces, 1D copies between block-linear storage (page kinds are not
// tracked), and 1D remaps other than a constant fill.
[[nodiscard]] std::variant<DmaCopyPlan, DmaCopyError> plan_dma_copy(
    std::span<const std::uint32_t> registers, std::uint32_t launch) noexcept;

// Executes a planned copy. source holds plan.source_size bytes read at
// plan.source; destination holds plan.destination_size bytes for
// plan.destination (their current values unless the plan covers them).
void execute_dma_copy(std::span<const std::uint32_t> registers, std::uint32_t launch,
                      const DmaCopyPlan& plan, std::span<const std::byte> source,
                      std::span<std::byte> destination) noexcept;

} // namespace armigo::graphics
