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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

// Fixed byte keys, parallel key/value arrays and linear probing. Values have
// stable addresses so an encoder can keep its current pipeline across growth.
// Callers canonicalize padding and unused fields before inserting a key.
template <class Key, class Value> class DrawTable {
    static_assert(std::is_trivially_copyable_v<Key>);
    using Bytes = std::array<std::byte, sizeof(Key)>;
    std::vector<Bytes> keys_;
    std::vector<std::unique_ptr<Value>> values_;
    std::size_t count_ = 0;

    static std::uint64_t hash(const void* key) {
        auto* bytes = static_cast<const std::byte*>(key);
        std::size_t remaining = sizeof(Key);
        std::uint64_t value = 0x9e3779b97f4a7c15ull ^ remaining;
        for (; remaining >= 8; bytes += 8, remaining -= 8) {
            std::uint64_t word;
            std::memcpy(&word, bytes, 8);
            value = (value ^ word) * 0xff51afd7ed558ccdull;
            value ^= value >> 32;
        }
        if (remaining) {
            std::uint64_t word = 0;
            std::memcpy(&word, bytes, remaining);
            value = (value ^ word) * 0xff51afd7ed558ccdull;
        }
        value ^= value >> 33;
        value *= 0xc4ceb9fe1a85ec53ull;
        return value ^ (value >> 33);
    }

    Value* put(const void* key, std::unique_ptr<Value> value) {
        const auto mask = values_.size() - 1;
        for (auto i = std::size_t(hash(key)) & mask;; i = (i + 1) & mask) {
            if (!values_[i]) {
                std::memcpy(keys_[i].data(), key, sizeof(Key));
                values_[i] = std::move(value);
                ++count_;
                return values_[i].get();
            }
            if (std::memcmp(keys_[i].data(), key, sizeof(Key)) == 0)
                return values_[i].get();
        }
    }

public:
    Value* find(const Key& key) const {
        if (!count_) return nullptr;
        const auto mask = values_.size() - 1;
        for (auto i = std::size_t(hash(&key)) & mask;; i = (i + 1) & mask) {
            if (!values_[i]) return nullptr;
            if (std::memcmp(keys_[i].data(), &key, sizeof(Key)) == 0)
                return values_[i].get();
        }
    }

    Value* insert(const Key& key, std::unique_ptr<Value> value) {
        if ((count_ + 1) * 4 > values_.size() * 3) {
            auto old_keys = std::move(keys_);
            auto old_values = std::move(values_);
            const auto capacity = old_values.empty() ? 64 : old_values.size() * 2;
            keys_.resize(capacity);
            values_.resize(capacity);
            count_ = 0;
            for (std::size_t i = 0; i < old_values.size(); ++i)
                if (old_values[i]) put(old_keys[i].data(), std::move(old_values[i]));
        }
        return put(&key, std::move(value));
    }
    std::size_t size() const { return count_; }
};

// Validity belongs to a submission, not a resource or pipeline cache. A new
// pass re-emits encoder state and publishes writes after its first real draw.
struct DrawInvalidation {
    bool pipeline = true;
    bool vertices = true;
    std::uint32_t fixed = 0x3ff;
    std::array<bool, 6> arguments{true, true, true, true, true, true};
    // A stage whose only stale bindings are uniform buffer addresses (an
    // in-stream update moved them to a new version): its textures, samplers
    // and storage buffers resolve to what they did at its last bind.
    std::array<bool, 6> uniforms{};
    bool attachments_written = false;
    std::uint64_t registration = 0, cpu_writes = 0;

    void begin_pass() {
        pipeline = vertices = true;
        fixed = 0x3ff;
        arguments.fill(true);
        attachments_written = false;
    }
    bool sync_resources(std::uint64_t next_registration, std::uint64_t next_writes) {
        if (registration == next_registration && cpu_writes == next_writes) return false;
        registration = next_registration;
        cpu_writes = next_writes;
        vertices = true;
        attachments_written = false;
        arguments.fill(true);
        return true;
    }
};

// Buffer hazards are separate from attachment notifications. A read-only draw
// must clear a consumed obligation instead of retaining a previous writer's bit.
// Inside a render pass only vertex-stage writes can be ordered before later
// work, so the order remembers whether the writes came only from that stage.
// Only writes create an obligation; reads never need ordering here.
struct DrawWriteOrder {
    enum class Writer : std::uint8_t { none, vertex, any };
    Writer pending = Writer::none;
    Writer consume() { return std::exchange(pending, Writer::none); }
    void draw(bool vertex_writes, bool fragment_writes) {
        pending = fragment_writes ? Writer::any : vertex_writes ? Writer::vertex : Writer::none;
    }
    // A write by other work (dispatch, transfer, copy): any stage.
    void write() { pending = Writer::any; }
    void clear() { pending = Writer::none; }
};

} // namespace armigo::platform::metal::nvn_driver
