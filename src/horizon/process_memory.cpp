// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "process_memory.h"

#include <algorithm>
#include <limits>

namespace armigo::horizon::process_memory {
namespace {
bool same(const MemoryBlock& a, const MemoryBlock& b) noexcept {
    return a.type == b.type && a.permissions == b.permissions &&
        a.attributes == b.attributes && a.ipc_references == b.ipc_references &&
        a.device_references == b.device_references;
}
}

bool valid_range(AddressRange r) noexcept {
    return r.size && r.address < ProcessRecord::query_end &&
        r.size <= ProcessRecord::query_end - r.address;
}
bool contains(AddressRange a, AddressRange b) noexcept {
    return valid_range(a) && valid_range(b) && b.address >= a.address &&
        b.address - a.address < a.size && b.size <= a.size - (b.address - a.address);
}
bool overlaps(AddressRange a, AddressRange b) noexcept {
    return valid_range(a) && valid_range(b) &&
        a.address < b.address + b.size && b.address < a.address + a.size;
}
std::uint32_t validate_range(AddressRange r) noexcept {
    if (r.address % guest_page_size) return invalid_address;
    if (!r.size || r.size % guest_page_size) return invalid_size;
    if (r.size > std::numeric_limits<std::uint64_t>::max() - r.address)
        return invalid_memory;
    return 0;
}
std::uint32_t validate_transfer(AddressRange range, std::uint64_t permission) noexcept {
    if (const auto result = validate_range(range)) return result;
    return permission == 0 || permission == 1 || permission == 3 ? 0 : invalid_permission;
}
bool transferable(const MemoryBlock& block) noexcept {
    return (block.type == MemoryType::normal || block.type == MemoryType::code_data) &&
           block.permissions == 3 && !block.attributes && !block.ipc_references && !block.device_references;
}
std::uint32_t validate_heap_size(std::uint64_t size) noexcept {
    return size >= ProcessRecord::heap_capacity || size % (2u << 20) ? invalid_size : 0;
}
MemoryBlock fragment(const MemoryBlock& block, AddressRange range) noexcept {
    auto result = block;
    result.range = range;
    result.merge_disable = 0;
    if (range.address == block.range.address)
        result.merge_disable |= block.merge_disable & merge_disable_left_mask;
    if (range.address + range.size == block.range.address + block.range.size)
        result.merge_disable |= block.merge_disable & merge_disable_right_mask;
    return result;
}
void coalesce(std::vector<MemoryBlock>& blocks) noexcept {
    std::size_t count = 0;
    for (const auto& block : blocks) {
        if (count && blocks[count - 1].range.address + blocks[count - 1].range.size ==
            block.range.address && same(blocks[count - 1], block) &&
            !(blocks[count - 1].merge_disable & merge_disable_right_mask) &&
            !(block.merge_disable & merge_disable_left_mask)) {
            blocks[count - 1].range.size += block.range.size;
            blocks[count - 1].merge_disable |= block.merge_disable;
        } else {
            blocks[count++] = block;
        }
    }
    blocks.resize(count);
}
std::vector<MemoryBlock> replace(std::span<const MemoryBlock> blocks,
    AddressRange range, std::optional<MemoryBlock> replacement,
    std::uint8_t set_merge, std::uint8_t clear_merge) {
    std::vector<MemoryBlock> result;
    result.reserve(blocks.size() + 5);
    const auto replace_block = [&](const MemoryBlock& block) {
        if (!overlaps(block.range, range)) { result.push_back(block); return; }
        const auto end = block.range.address + block.range.size;
        const auto begin = std::max(block.range.address, range.address);
        const auto changed_end = std::min(end, range.address + range.size);
        if (block.range.address < begin)
            result.push_back(fragment(block, {block.range.address, begin - block.range.address}));
        auto changed = replacement.value_or(MemoryBlock{});
        changed.range = {begin, changed_end - begin};
        changed.merge_disable = fragment(block, changed.range).merge_disable & ~clear_merge;
        if (begin == range.address) changed.merge_disable |= set_merge;
        result.push_back(changed);
        if (changed_end < end)
            result.push_back(fragment(block, {changed_end, end - changed_end}));
    };
    // Materialize implicit free nodes while updating so a retained free-node
    // barrier has the same containing extent as an explicit metadata tree.
    std::uint64_t cursor = 0;
    for (const auto& block : blocks) {
        if (cursor < block.range.address)
            replace_block(MemoryBlock{{cursor, block.range.address - cursor}});
        replace_block(block);
        cursor = block.range.address + block.range.size;
    }
    if (cursor < ProcessRecord::query_end)
        replace_block(MemoryBlock{{cursor, ProcessRecord::query_end - cursor}});
    coalesce(result);
    std::erase_if(result, [](const auto& block) {
        return block.type == MemoryType::free && !block.merge_disable;
    });
    return result;
}
bool free(std::span<const MemoryBlock> blocks, AddressRange range) noexcept {
    if (!valid_range(range)) return false;
    return std::none_of(blocks.begin(), blocks.end(), [&](const auto& b) {
        return b.type != MemoryType::free && overlaps(b.range, range);
    });
}
bool covered(std::span<const MemoryBlock> blocks, AddressRange range,
             std::uint8_t permissions) noexcept {
    if (!valid_range(range)) return false;
    auto cursor = range.address;
    // Blocks are ordered by address and do not overlap: start at the first block that ends
    // after the range begins instead of walking the whole map.
    const auto first = std::partition_point(blocks.begin(), blocks.end(), [&](const MemoryBlock& b) {
        return b.range.address + b.range.size <= range.address;
    });
    for (auto it = first; it != blocks.end(); ++it) {
        const auto& b = *it;
        if (b.range.address >= range.address + range.size) break;
        if (!overlaps(b.range, range)) continue;
        if (b.range.address > cursor || b.type == MemoryType::free ||
            (b.permissions & permissions) != permissions) return false;
        cursor = std::min(range.address + range.size, b.range.address + b.range.size);
        if (cursor == range.address + range.size) return true;
    }
    return false;
}
std::optional<MemoryBlock> uniform(std::span<const MemoryBlock> blocks,
                                  AddressRange range) noexcept {
    if (!covered(blocks, range, 0)) return std::nullopt;
    std::optional<MemoryBlock> value;
    for (const auto& b : blocks) {
        if (!overlaps(b.range, range)) continue;
        if (value && !same(*value, b)) return std::nullopt;
        value = b;
    }
    value->range = range;
    return value;
}
} // namespace armigo::horizon::process_memory
