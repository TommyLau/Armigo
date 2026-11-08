// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/startup_queries.h>

namespace armigo::horizon::process_memory {

inline constexpr std::uint32_t invalid_size = 0xca01;
inline constexpr std::uint32_t invalid_address = 0xcc01;
inline constexpr std::uint32_t out_of_memory = 0xd001;
inline constexpr std::uint32_t invalid_memory = 0xd401;
inline constexpr std::uint32_t invalid_permission = 0xd801;
inline constexpr std::uint32_t invalid_region = 0xdc01;
inline constexpr std::uint32_t invalid_handle = 0xe401;
inline constexpr std::uint32_t invalid_combination = 0xe801;
inline constexpr std::uint32_t limit_reached = 0x10801;
inline constexpr std::uint32_t out_of_handles = 0xd201;
inline constexpr std::uint8_t locked_attribute = 1;
inline constexpr std::uint8_t merge_disable_left_mask = 0x0f;
inline constexpr std::uint8_t merge_disable_right_mask = 0x20;
inline constexpr std::uint8_t merge_disable_mapping = 1;
inline constexpr std::uint8_t merge_disable_locked = 8;

// Splitting transfers left barriers to the left fragment and right barriers to
// the right fragment. Interior fragments inherit neither boundary.
MemoryBlock fragment(const MemoryBlock& block, AddressRange range) noexcept;

// Merge adjacent blocks only when every public state field and reference count
// agrees and neither side retains a boundary barrier.
void coalesce(std::vector<MemoryBlock>& blocks) noexcept;

bool valid_range(AddressRange range) noexcept;
bool contains(AddressRange outer, AddressRange inner) noexcept;
bool overlaps(AddressRange left, AddressRange right) noexcept;
// Firmware argument checks only; placement, state and native feasibility follow.
std::uint32_t validate_range(AddressRange range) noexcept;
std::uint32_t validate_heap_size(std::uint64_t size) noexcept;
// CreateTransferMemory's argument checks in kernel order: address, size and
// overflow (validate_range), then the owner permission (none, read or
// read-write). Process placement and the owner's state follow.
std::uint32_t validate_transfer(AddressRange range, std::uint64_t permission) noexcept;
// Whether a uniform range may be locked for transfer: a data state (Normal or
// CodeData) mapped user read-write, with no attributes, IPC or device references.
bool transferable(const MemoryBlock& block) noexcept;

// A sparse canonical map. Replacement removes just the selected interval;
// nullopt leaves that interval free. Callers validate coverage/state first.
// Set barriers only at the range start; clear them on every updated node.
// Existing interior boundaries remain until their barriers admit coalescing.
std::vector<MemoryBlock> replace(std::span<const MemoryBlock> blocks,
    AddressRange range, std::optional<MemoryBlock> replacement,
    std::uint8_t set_merge = 0, std::uint8_t clear_merge = 0);
bool free(std::span<const MemoryBlock> blocks, AddressRange range) noexcept;
bool covered(std::span<const MemoryBlock> blocks, AddressRange range,
    std::uint8_t permissions) noexcept;
// Uniformity includes state, attributes and references, not just permission;
// node-boundary barriers do not change the range's memory semantics.
std::optional<MemoryBlock> uniform(std::span<const MemoryBlock> blocks,
    AddressRange range) noexcept;

} // namespace armigo::horizon::process_memory
