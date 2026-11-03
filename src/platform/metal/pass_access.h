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

#include <armigo/graphics/resource_memory.h>

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace armigo::platform::metal {

// Summarizes the accesses of one render pass without weakening its existing
// hazard rules. Equal ranges of the same attachment class need only remember
// whether any access wrote. Other ranges and classes remain independent.
class PassAccessSummary {
    struct Key {
        graphics::BackingRange bytes;
        bool attachment = false;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            std::size_t result = 0;
            const auto append = [&](std::uint64_t value) {
                result ^= std::hash<std::uint64_t>{}(value) +
                          std::size_t{0x9e3779b9} + (result << 6) + (result >> 2);
            };
            append(key.bytes.backing.owner);
            append(key.bytes.backing.allocation);
            append(key.bytes.backing.generation);
            append(key.bytes.offset);
            append(key.bytes.size);
            append(key.attachment);
            return result;
        }
    };
    struct Record {
        Key key;
        bool writes = false;
    };

public:
    void record(const graphics::BackingRange& bytes, graphics::ResourceAccess access,
                bool attachment) {
        const Key key{bytes, attachment};
        const bool write = access != graphics::ResourceAccess::Read;
        const auto [entry, inserted] = index_.try_emplace(key, records_.size());
        if (!inserted) {
            records_[entry->second].writes |= write;
            return;
        }
        try {
            records_.push_back({key, write});
            by_backing_[backing_key(bytes)].push_back(records_.size() - 1);
        } catch (...) {
            if (records_.size() > entry->second) records_.pop_back();
            index_.erase(entry);
            throw;
        }
    }

    [[nodiscard]] bool conflicts(const graphics::BackingRange& bytes,
                                 graphics::ResourceAccess access,
                                 bool attachment) const noexcept {
        const bool write = access != graphics::ResourceAccess::Read;
        // Only records of the same backing can overlap.
        const auto bucket = by_backing_.find(backing_key(bytes));
        if (bucket == by_backing_.end()) return false;
        for (const auto index : bucket->second) {
            const auto& prior = records_[index];
            if ((!prior.writes && !write) || (prior.key.attachment && attachment))
                continue;
            const auto& previous = prior.key.bytes;
            if (previous.backing == bytes.backing &&
                previous.offset < bytes.offset + bytes.size &&
                bytes.offset < previous.offset + previous.size)
                return true;
        }
        return false;
    }

    void clear() noexcept {
        records_.clear();
        index_.clear();
        by_backing_.clear();
    }

    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

private:
    static std::uint64_t backing_key(const graphics::BackingRange& bytes) noexcept {
        return bytes.backing.owner * 0x9e3779b97f4a7c15ull ^ bytes.backing.allocation * 0xc2b2ae3d27d4eb4full ^
               bytes.backing.generation;
    }
    std::vector<Record> records_;
    std::unordered_map<Key, std::size_t, KeyHash> index_;
    // Record indices per backing (hashed identity; the overlap test still
    // compares the exact backing).
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> by_backing_;
};

} // namespace armigo::platform::metal
