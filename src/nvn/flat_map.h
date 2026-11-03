// ============================================================================
// Armigo NVN - Flat Hash Map
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// An open-addressing hash table for nonzero 64-bit keys (command handles and
// record addresses): entries live in one flat array probed linearly, so a
// lookup touches one or two cache lines and an insert allocates only when the
// table grows. Key zero marks an empty slot and is never stored. Erasing
// shifts the following entries back, so no tombstones accumulate. Entry
// pointers stay valid until the next insert.

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace armigo::nvn {

template <class Value> class FlatMap64 {
public:
    Value* find(std::uint64_t key) noexcept {
        if (!key || slots_.empty()) return nullptr;
        for (auto index = home(key);; index = next(index)) {
            auto& slot = slots_[index];
            if (slot.key == key) return &slot.value;
            if (!slot.key) return nullptr;
        }
    }
    const Value* find(std::uint64_t key) const noexcept {
        return const_cast<FlatMap64*>(this)->find(key);
    }

    // Inserts the value when the key is absent. Returns the entry and whether
    // it was inserted; a zero key is refused.
    std::pair<Value*, bool> emplace(std::uint64_t key, Value value) {
        if (!key) return {nullptr, false};
        if (auto* existing = find(key)) return {existing, false};
        if ((count_ + 1) * 4 > slots_.size() * 3) grow(slots_.empty() ? 16 : slots_.size() * 2);
        auto index = home(key);
        while (slots_[index].key) index = next(index);
        slots_[index].key = key;
        slots_[index].value = std::move(value);
        ++count_;
        return {&slots_[index].value, true};
    }

    void insert_or_assign(std::uint64_t key, Value value) {
        auto [entry, inserted] = emplace(key, Value{});
        if (entry) *entry = std::move(value);
    }

    bool erase(std::uint64_t key) noexcept {
        if (!key || slots_.empty()) return false;
        auto index = home(key);
        for (;; index = next(index)) {
            if (slots_[index].key == key) break;
            if (!slots_[index].key) return false;
        }
        // Backward-shift deletion: move each following entry of the probe run
        // into the hole when its home position does not lie in (hole, entry].
        for (auto hole = index, probe = next(index);; probe = next(probe)) {
            auto& slot = slots_[probe];
            if (!slot.key) {
                slots_[hole] = {};
                break;
            }
            const auto wanted = home(slot.key);
            const bool stays = hole <= probe ? (hole < wanted && wanted <= probe)
                                             : (hole < wanted || wanted <= probe);
            if (stays) continue;
            slots_[hole] = std::move(slot);
            hole = probe;
        }
        --count_;
        return true;
    }

    void reserve(std::size_t count) {
        std::size_t capacity = slots_.empty() ? 16 : slots_.size();
        while (count * 4 > capacity * 3) capacity *= 2;
        if (capacity > slots_.size()) grow(capacity);
    }

    std::size_t size() const noexcept { return count_; }

private:
    struct Slot {
        std::uint64_t key = 0;
        Value value{};
    };

    std::size_t home(std::uint64_t key) const noexcept {
        key ^= key >> 33;
        key *= 0xff51afd7ed558ccdull;
        key ^= key >> 33;
        key *= 0xc4ceb9fe1a85ec53ull;
        key ^= key >> 33;
        return std::size_t(key) & (slots_.size() - 1);
    }
    std::size_t next(std::size_t index) const noexcept { return (index + 1) & (slots_.size() - 1); }

    void grow(std::size_t capacity) {
        auto old = std::move(slots_);
        slots_.clear();
        slots_.resize(capacity);
        count_ = 0;
        for (auto& slot : old) {
            if (!slot.key) continue;
            auto index = home(slot.key);
            while (slots_[index].key) index = next(index);
            slots_[index] = std::move(slot);
            ++count_;
        }
    }

    std::vector<Slot> slots_;
    std::size_t count_ = 0;
};

} // namespace armigo::nvn
