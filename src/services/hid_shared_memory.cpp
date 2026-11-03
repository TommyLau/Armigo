// ============================================================================
// Armigo Services - Human Interface Device Shared Memory
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "hid_shared_memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>

namespace armigo::services::hid::detail {
namespace {

constexpr std::size_t shared_memory_size = 0x40000;
constexpr std::size_t npad_base = 0x9a00;
constexpr std::size_t npad_stride = 0x5000;
constexpr std::size_t npad_count = 10;
constexpr std::size_t full_key_ring = 0x28;
constexpr std::size_t common_entry_size = 0x30;
constexpr std::uint64_t ring_capacity = 17;
// Seven common-state rings, six six-axis rings (FullKey, Handheld, JoyDual
// left and right, JoyLeft, JoyRight) and the GameCube trigger ring.
constexpr std::array<std::size_t, 14> npad_ring_offsets{
    0x28, 0x378, 0x6c8, 0xa18, 0xd68, 0x10b8, 0x1408,
    0x1758, 0x1e60, 0x2568, 0x2c70, 0x3378, 0x3a80, 0x4228,
};
constexpr std::size_t full_key_six_axis_ring = 0x1758;
constexpr std::size_t six_axis_entry_size = 0x68;

static_assert(std::endian::native == std::endian::little);

void validate(std::span<std::byte> memory) {
    assert(memory.size() >= shared_memory_size);
    assert(reinterpret_cast<std::uintptr_t>(memory.data()) % alignof(std::uint64_t) == 0);
    (void)memory;
}

std::uint64_t load64(std::span<std::byte> memory, std::size_t offset) {
    return __atomic_load_n(reinterpret_cast<const std::uint64_t*>(memory.data() + offset),
                           __ATOMIC_ACQUIRE);
}

void store64(std::span<std::byte> memory, std::size_t offset, std::uint64_t value) {
    __atomic_store_n(reinterpret_cast<std::uint64_t*>(memory.data() + offset), value,
                     __ATOMIC_RELEASE);
}

std::uint32_t load32(std::span<std::byte> memory, std::size_t offset) {
    return __atomic_load_n(reinterpret_cast<const std::uint32_t*>(memory.data() + offset),
                           __ATOMIC_ACQUIRE);
}

void store32(std::span<std::byte> memory, std::size_t offset, std::uint32_t value) {
    __atomic_store_n(reinterpret_cast<std::uint32_t*>(memory.data() + offset), value,
                     __ATOMIC_RELEASE);
}

void publish_common_state(std::span<std::byte> memory, std::size_t ring, bool connected,
                          const FullKeyInput& input = {}) {
    const auto previous_index = load64(memory, ring + 0x10);
    const auto count = load64(memory, ring + 0x18);
    assert(previous_index < ring_capacity);
    const auto previous_entry = ring + 0x20 + common_entry_size * previous_index;
    const auto sample = count == 0 ? 1 : load64(memory, previous_entry + 8) + 1;
    const auto index = (previous_index + 1) % ring_capacity;
    const auto entry = ring + 0x20 + common_entry_size * index;

    // A reader retries an odd or changed latch. Complete the payload before
    // making its even latch, newest index, and available count visible.
    __atomic_store_n(reinterpret_cast<std::uint64_t*>(memory.data() + entry),
                     (sample << 1) | 1, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    std::array<std::byte, 0x28> state{};
    std::memcpy(state.data(), &sample, sizeof(sample));
    std::memcpy(state.data() + 0x08, &input.buttons, sizeof(input.buttons));
    const std::array<std::int32_t, 4> sticks{input.left_x, input.left_y, input.right_x, input.right_y};
    std::memcpy(state.data() + 0x10, sticks.data(), sizeof(sticks));
    const std::uint32_t attributes = connected ? 1 : 0;
    std::memcpy(state.data() + 0x20, &attributes, sizeof(attributes));
    std::memcpy(memory.data() + entry + 8, state.data(), state.size());
    store64(memory, entry, sample << 1);
    store64(memory, ring + 0x10, index);
    if (count < ring_capacity - 1)
        store64(memory, ring + 0x18, count + 1);
}

}  // namespace

void initialize_shared_memory(std::span<std::byte> memory) {
    validate(memory);
    std::fill_n(memory.begin(), shared_memory_size, std::byte{});
    for (std::size_t index = 0; index < npad_count; ++index) {
        const auto base = npad_base + npad_stride * index;
        for (const auto offset : npad_ring_offsets)
            store64(memory, base + offset + 8, ring_capacity);
    }
}

void publish_headless_full_key(std::span<std::byte> memory, bool connected) {
    validate(memory);
    const bool previously_connected = (load32(memory, npad_base) & 1) != 0;
    if (!connected && !previously_connected)
        return;

    publish_common_state(memory, npad_base + full_key_ring, connected);
    store32(memory, npad_base, connected ? 1 : 0);
}

bool publish_full_key_input(std::span<std::byte> memory, const FullKeyInput& input) {
    validate(memory);
    if ((load32(memory, npad_base) & 1) == 0) return false;
    publish_common_state(memory, npad_base + full_key_ring, true, input);
    return true;
}

bool publish_full_key_six_axis(std::span<std::byte> memory, const SixAxisSample& sample) {
    validate(memory);
    if ((load32(memory, npad_base) & 1) == 0) return false;
    const auto ring = npad_base + full_key_six_axis_ring;
    const auto previous_index = load64(memory, ring + 0x10);
    const auto count = load64(memory, ring + 0x18);
    assert(previous_index < ring_capacity);
    const auto previous_entry = ring + 0x20 + six_axis_entry_size * previous_index;
    // The state's sampling number follows its delta time.
    const auto number = count == 0 ? 1 : load64(memory, previous_entry + 0x10) + 1;
    const auto index = (previous_index + 1) % ring_capacity;
    const auto entry = ring + 0x20 + six_axis_entry_size * index;

    // Same publication order as the common-state rings: odd latch, payload,
    // even latch, newest index, available count.
    __atomic_store_n(reinterpret_cast<std::uint64_t*>(memory.data() + entry),
                     (number << 1) | 1, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    std::array<std::byte, 0x60> state{};
    std::memcpy(state.data(), &sample.delta_time, 8);
    std::memcpy(state.data() + 0x08, &number, 8);
    std::memcpy(state.data() + 0x10, sample.acceleration.data(), 12);
    std::memcpy(state.data() + 0x1c, sample.angular_velocity.data(), 12);
    std::memcpy(state.data() + 0x28, sample.angle.data(), 12);
    for (std::size_t row = 0; row < 3; ++row)
        std::memcpy(state.data() + 0x34 + 12 * row, sample.direction[row].data(), 12);
    const std::uint32_t attribute = sample.connected ? 1 : 0;  // IsConnected
    std::memcpy(state.data() + 0x58, &attribute, 4);
    std::memcpy(memory.data() + entry + 8, state.data(), state.size());
    store64(memory, entry, number << 1);
    store64(memory, ring + 0x10, index);
    if (count < ring_capacity - 1)
        store64(memory, ring + 0x18, count + 1);
    return true;
}

}  // namespace armigo::services::hid::detail
