// ============================================================================
// Armigo Graphics - GPU Engine Front End
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Queue metadata decoding extracts every consumed bit range, including each
// constant buffer's address, size, and valid flag, into ComputeLaunch.

#include "compute_qmd.h"

namespace armigo::graphics {
namespace {

// Words are little-endian 32-bit units in the fixed 64-word QMD descriptor.
std::uint32_t word(std::span<const std::byte, compute_qmd_bytes> descriptor,
                   std::size_t index) noexcept {
    const auto base = descriptor.data() + index * sizeof(std::uint32_t);
    return std::uint32_t{std::to_integer<std::uint8_t>(base[0])} |
           (std::uint32_t{std::to_integer<std::uint8_t>(base[1])} << 8) |
           (std::uint32_t{std::to_integer<std::uint8_t>(base[2])} << 16) |
           (std::uint32_t{std::to_integer<std::uint8_t>(base[3])} << 24);
}

bool bit(std::span<const std::byte, compute_qmd_bytes> descriptor, unsigned index) noexcept {
    return (word(descriptor, index >> 5) & (1u << (index & 31))) != 0;
}

std::uint32_t bit_range(std::span<const std::byte, compute_qmd_bytes> descriptor,
                        unsigned upper, unsigned lower) noexcept {
    const std::uint32_t mask =
        upper - lower + 1 == 32 ? 0xffffffffu : (0xffffffffu >> (32 - (upper - lower + 1)));
    return (word(descriptor, lower >> 5) >> (lower & 31)) & mask;
}

}  // namespace

ComputeLaunch decode_compute_qmd(
    std::span<const std::byte, compute_qmd_bytes> descriptor) noexcept {
    ComputeLaunch launch{};
    launch.program_offset = bit_range(descriptor, 287, 256);
    launch.sampler_index = bit(descriptor, 382);
    launch.grid = {bit_range(descriptor, 415, 384), bit_range(descriptor, 431, 416),
                   bit_range(descriptor, 447, 432)};
    launch.block = {bit_range(descriptor, 607, 592), bit_range(descriptor, 623, 608),
                    bit_range(descriptor, 639, 624)};
    launch.shared_memory_size = bit_range(descriptor, 561, 544);
    launch.local_memory_size =
        bit_range(descriptor, 1463, 1440) + bit_range(descriptor, 1495, 1472);
    launch.local_crs_size = bit_range(descriptor, 1527, 1504);
    launch.barrier_count = bit_range(descriptor, 1471, 1467);
    launch.register_count = bit_range(descriptor, 1503, 1496);
    for (std::size_t index = 0; index < launch.constant_buffers.size(); ++index) {
        auto& buffer = launch.constant_buffers[index];
        // Bit 640 + i, address words at 928 + 64 i (32-bit lower, 8-bit upper)
        // and the 17-bit size at 975 + 64 i follow the guest QMD layout.
        buffer.enabled = bit(descriptor, unsigned(640 + index));
        buffer.address =
            std::uint64_t{bit_range(descriptor, unsigned(967 + index * 64),
                                    unsigned(960 + index * 64))}
                << 32 |
            bit_range(descriptor, unsigned(959 + index * 64), unsigned(928 + index * 64));
        buffer.size = bit_range(descriptor, unsigned(991 + index * 64),
                                unsigned(975 + index * 64));
    }
    return launch;
}

}  // namespace armigo::graphics
