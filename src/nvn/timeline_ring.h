// ============================================================================
// Armigo NVN - Frame Timeline Trace
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

namespace armigo::nvn::detail {

struct TimelineRecord {
    std::uint64_t time, key, value;
    std::uint32_t event;
};

// Admission is separate from queue reservation: a full queue or failed name
// lookup can still be an in-flight attempt when shutdown snapshots losses.
// Closing the gate captures those attempts without waiting on their threads.
class TimelineAdmission {
public:
    bool begin() noexcept {
        auto state = state_.load(std::memory_order_relaxed);
        while (!(state & closed_bit)) {
            if (state_.compare_exchange_weak(state, state + 1, std::memory_order_acquire)) return true;
        }
        return false;
    }
    void end() noexcept { state_.fetch_sub(1, std::memory_order_release); }
    std::uint64_t close() noexcept {
        return state_.fetch_or(closed_bit, std::memory_order_acq_rel) & ~closed_bit;
    }
private:
    static constexpr std::uint64_t closed_bit = std::uint64_t{1} << 63;
    std::atomic<std::uint64_t> state_{0};
};

// Bounded multiple-producer, single-consumer queue. A sequence number owns
// each slot until the consumer releases it; a full queue drops rather than
// waiting for the writer. Reservation order preserves each producer's order.
class TimelineRing {
public:
    explicit TimelineRing(std::size_t capacity)
        : capacity_(capacity), slots_(std::make_unique<Slot[]>(capacity)) {
        for (std::size_t i = 0; i < capacity; ++i)
            slots_[i].sequence.store(i, std::memory_order_relaxed);
    }

    struct NoPause { void operator()() const noexcept {} };
    // The empty hook compiles out; tests pause a reserved producer here to
    // exercise publication and shutdown without timing-dependent sleeps.
    template <class BeforePublish = NoPause>
    bool push(TimelineRecord record, BeforePublish before_publish = {}) noexcept {
        auto position = next_.load(std::memory_order_relaxed);
        for (;;) {
            if (position & closed_bit) return false;
            auto& slot = slots_[position % capacity_];
            const auto sequence = slot.sequence.load(std::memory_order_acquire);
            if (sequence == position) {
                if (next_.compare_exchange_weak(position, position + 1, std::memory_order_relaxed)) {
                    before_publish();
                    slot.record = record;
                    slot.sequence.store(position + 1, std::memory_order_release);
                    return true;
                }
            } else if (sequence < position) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return false;
            } else {
                position = next_.load(std::memory_order_relaxed);
            }
        }
    }

    // Only the writer calls pop or finish.
    bool pop(TimelineRecord& record) noexcept {
        auto& slot = slots_[read_ % capacity_];
        if (slot.sequence.load(std::memory_order_acquire) != read_ + 1) return false;
        record = slot.record;
        slot.sequence.store(read_ + capacity_, std::memory_order_release);
        ++read_;
        return true;
    }

    void close() noexcept { next_.fetch_or(closed_bit, std::memory_order_acq_rel); }
    bool closed() const noexcept { return next_.load(std::memory_order_acquire) & closed_bit; }
    std::uint64_t dropped() const noexcept { return dropped_.load(std::memory_order_relaxed); }

    // Exit may race a producer that reserved but has not published a slot.
    // Skip and count that record without reading its payload or waiting for
    // that thread. Storage must outlive such producers, including after exit
    // draining; the process-wide recorder deliberately retains its storage.
    template <class Emit> std::uint64_t finish(Emit&& emit) {
        const auto end = next_.load(std::memory_order_acquire) & ~closed_bit;
        std::uint64_t incomplete = 0;
        while (read_ < end) {
            auto& slot = slots_[read_ % capacity_];
            if (slot.sequence.load(std::memory_order_acquire) == read_ + 1) emit(slot.record);
            else ++incomplete;
            ++read_;
        }
        return incomplete;
    }

private:
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    static constexpr std::uint64_t closed_bit = std::uint64_t{1} << 63;
    struct Slot {
        std::atomic<std::uint64_t> sequence{0};
        TimelineRecord record{};
    };
    const std::size_t capacity_;
    std::unique_ptr<Slot[]> slots_;
    alignas(128) std::atomic<std::uint64_t> next_{0};
    alignas(128) std::atomic<std::uint64_t> dropped_{0};
    alignas(128) std::uint64_t read_ = 0;
};

// Event names can be transient or composed from a prefix and a wait tag.
// Intern them in fixed storage before publishing their numeric IDs. A busy
// slot is skipped, never waited on; duplicate names in that rare race are
// harmless and have identical TSV output.
class TimelineSymbols {
public:
    static constexpr std::size_t capacity = 512, name_bytes = 64;
    static constexpr std::uint32_t invalid = UINT32_MAX;

    std::uint32_t intern(const char* name, const char* suffix = nullptr) noexcept {
        if (!name) name = "?";
        if (!suffix) suffix = "";
        const char* parts[] = {name, suffix};
        std::size_t lengths[2]{}, length = 0;
        std::uint64_t hash = 14695981039346656037ull;
        for (unsigned part = 0; part < 2; ++part) {
            while (length < name_bytes && parts[part][lengths[part]]) {
                hash = (hash ^ static_cast<unsigned char>(parts[part][lengths[part]++])) * 1099511628211ull;
                ++length;
            }
            if (length == name_bytes) return fail();
        }
        if (hash < 2) hash += 2;
        for (std::size_t probe = 0; probe < capacity; ++probe) {
            const auto index = (hash + probe) % capacity;
            auto& slot = slots_[index];
            auto state = slot.hash.load(std::memory_order_acquire);
            if (state == hash && std::strncmp(slot.name, name, lengths[0]) == 0 &&
                std::strcmp(slot.name + lengths[0], suffix) == 0)
                return static_cast<std::uint32_t>(index);
            if (state == 0 && slot.hash.compare_exchange_strong(state, 1, std::memory_order_acquire)) {
                std::memcpy(slot.name, name, lengths[0]);
                std::memcpy(slot.name + lengths[0], suffix, lengths[1] + 1);
                slot.hash.store(hash, std::memory_order_release);
                return static_cast<std::uint32_t>(index);
            }
        }
        return fail();
    }
    const char* name(std::uint32_t id) const noexcept { return slots_[id].name; }
    std::uint64_t dropped() const noexcept { return dropped_.load(std::memory_order_relaxed); }

private:
    std::uint32_t fail() noexcept {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return invalid;
    }
    struct Slot {
        std::atomic<std::uint64_t> hash{0};
        char name[name_bytes]{};
    };
    Slot slots_[capacity];
    std::atomic<std::uint64_t> dropped_{0};
};

} // namespace armigo::nvn::detail
