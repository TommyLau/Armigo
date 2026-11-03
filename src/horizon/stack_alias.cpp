// ============================================================================
// Armigo Horizon - Compatibility Metadata
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/stack_alias.h>
#include "process_memory.h"
#include <algorithm>
#include <array>
#include <new>
#include <stdexcept>

namespace armigo::horizon {
namespace {
constexpr std::size_t block_limit = 65536, alias_limit = 4096;
bool range_valid(std::uint64_t address, std::uint64_t size) {
    return size != 0 && ((address | size) & 4095) == 0 &&
        address < ProcessRecord::query_end && size <= ProcessRecord::query_end - address;
}
bool block_valid(const MemoryBlock& block) {
    if (!range_valid(block.range.address, block.range.size)) return false;
    switch (block.type) {
    case MemoryType::free:
        return block.attributes == 0 && block.permissions == 0 &&
            block.ipc_references == 0 && block.device_references == 0;
    case MemoryType::reserved: return block.permissions == 0;
    case MemoryType::code: case MemoryType::code_data: case MemoryType::normal:
    case MemoryType::shared: case MemoryType::stack: case MemoryType::thread_local_storage:
        return block.range.address >= ProcessRecord::aslr_start;
    }
    return false;
}
StackAliasFailure failure(StackAliasError code, std::string_view message) {
    return {code, message};
}
bool covered(std::span<const MemoryBlock> blocks, std::uint64_t address,
             std::uint64_t size, MemoryType type, std::uint8_t attributes,
             std::uint8_t permissions) {
    std::uint64_t total = 0;
    for (const auto& block : blocks) {
        const auto begin = std::max(address, block.range.address);
        const auto end = std::min(address + size, block.range.address + block.range.size);
        if (begin >= end) continue;
        if (block.type != type || block.attributes != attributes ||
            block.permissions != permissions || block.ipc_references != 0 ||
            block.device_references != 0) return false;
        total += end - begin;
    }
    return total == size;
}
}  // namespace

StackAliasUnmapResult plan_stack_alias_unmap(
    std::span<const MemoryBlock> blocks, std::span<const StackAliasRecord> aliases,
    std::uint64_t destination, std::uint64_t source, std::uint64_t size) {
    if (blocks.size() > block_limit || aliases.size() > alias_limit)
        return failure(StackAliasError::AdmissionLimit, "stack alias input limit exceeded");
    if (!range_valid(source, size) || !range_valid(destination, size) ||
        (source < destination + size && destination < source + size))
        return failure(StackAliasError::InvalidRange, "stack alias ranges are invalid");
    std::uint64_t previous_end = 0;
    for (const auto& block : blocks) {
        if (!block_valid(block) || block.range.address < previous_end)
            return failure(StackAliasError::InvalidMemoryMap, "stack alias memory map is invalid");
        previous_end = block.range.address + block.range.size;
    }
    std::size_t selected = aliases.size();
    for (std::size_t i = 0; i < aliases.size(); ++i) {
        const auto& alias = aliases[i];
        if (alias.destination != destination || alias.source != source || alias.size != size) continue;
        if (selected != aliases.size())
            return failure(StackAliasError::UnregisteredAlias, "stack alias registration is ambiguous");
        selected = i;
    }
    if (selected == aliases.size())
        return failure(StackAliasError::UnregisteredAlias, "stack alias pair is not registered");
    const auto type = aliases[selected].source_type;
    if ((type != MemoryType::normal && type != MemoryType::code_data) ||
        !covered(blocks, source, size, type, 1, 0))
        return failure(StackAliasError::InvalidSource, "stack alias source is not owned locked memory");
    if (!covered(blocks, destination, size, MemoryType::stack, 0, 3))
        return failure(StackAliasError::InvalidDestination, "stack alias destination is not owned stack memory");

    try {
        StackAliasUnmapPlan plan;
        plan.alias_index = selected;
        plan.blocks.reserve(std::min(block_limit, blocks.size() + 4));
        struct Change { std::uint64_t address; MemoryType type; std::uint8_t permissions; };
        std::array<Change, 2> changes{{{source, type, 3}, {destination, MemoryType::free, 0}}};
        if (changes[1].address < changes[0].address) std::swap(changes[0], changes[1]);
        const auto append = [&](MemoryBlock block) {
            if (plan.blocks.size() == block_limit) return false;
            plan.blocks.push_back(block);
            return true;
        };
        for (const auto& block : blocks) {
            auto cursor = block.range.address;
            const auto block_end = cursor + block.range.size;
            for (const auto& change : changes) {
                const auto begin = std::max(block.range.address, change.address);
                const auto end = std::min(block_end, change.address + size);
                if (begin >= end) continue;
                if (cursor < begin) {
                    auto prefix = process_memory::fragment(block, {cursor, begin - cursor});
                    if (!append(prefix))
                        return failure(StackAliasError::AdmissionLimit, "stack alias output limit exceeded");
                }
                auto changed = process_memory::fragment(block, {begin, end - begin});
                changed.type = change.type;
                changed.attributes = 0;
                changed.permissions = change.permissions;
                changed.merge_disable &= ~(change.type == MemoryType::free
                    ? process_memory::merge_disable_mapping : process_memory::merge_disable_locked);
                if (!append(changed))
                    return failure(StackAliasError::AdmissionLimit, "stack alias output limit exceeded");
                cursor = end;
            }
            if (cursor < block_end) {
                auto suffix = process_memory::fragment(block, {cursor, block_end - cursor});
                if (!append(suffix))
                    return failure(StackAliasError::AdmissionLimit, "stack alias output limit exceeded");
            }
        }
        return plan;
    } catch (const std::bad_alloc&) {
        return failure(StackAliasError::AllocationFailure, "stack alias plan allocation failed");
    } catch (const std::length_error&) {
        return failure(StackAliasError::AllocationFailure, "stack alias plan allocation failed");
    }
}
}  // namespace armigo::horizon
