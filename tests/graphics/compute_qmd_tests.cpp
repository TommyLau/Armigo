// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic fixtures for the queue-meta-data decoder.
// Fixture bit positions follow the published QMD layout.

#include "compute_qmd.h"
#include <cstdint>
#include <cstdio>

namespace {

using armigo::graphics::ComputeLaunch;
using armigo::graphics::decode_compute_qmd;
using armigo::graphics::compute_qmd_bytes;

unsigned failures = 0;

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

using Descriptor = std::array<std::byte, compute_qmd_bytes>;
using Dimensions = std::array<std::uint32_t, 3>;

void set_bit(Descriptor& descriptor, unsigned index, bool value = true) {
    auto& target = reinterpret_cast<std::uint32_t*>(descriptor.data())[index >> 5];
    const std::uint32_t mask = 1u << (index & 31);
    if (value) target |= mask;
    else target &= ~mask;
}

void set_range(Descriptor& descriptor, unsigned lower, unsigned width, std::uint32_t value) {
    auto& target = reinterpret_cast<std::uint32_t*>(descriptor.data())[lower >> 5];
    const unsigned shift = lower & 31;
    const std::uint32_t mask = width == 32 ? 0xffffffffu : ((1u << width) - 1u) << shift;
    target = (target & ~mask) | ((value << shift) & mask);
    // Bits crossing into the next word (shift + width > 32) are not exercised
    // by these fixtures; every checked field starts word aligned or fits.
    if (shift != 0) CHECK(shift + width <= 32);
}

Descriptor all_zero() { return {}; }

// Full synthetic launch covering grid, block, shared/local memory, program
// offset and every enabled constant-buffer slot.
Descriptor full_launch() {
    Descriptor descriptor{};
    set_range(descriptor, 256, 32, 0x1u << 14);  // program offset
    set_range(descriptor, 384, 32, 0x13);        // grid x
    set_range(descriptor, 416, 16, 0x2);         // grid y
    set_range(descriptor, 432, 16, 0x3);         // grid z
    set_range(descriptor, 544, 18, 0x4000);      // shared memory size words
    set_range(descriptor, 592, 16, 0x40);        // block x
    set_range(descriptor, 608, 16, 0x8);         // block y
    set_range(descriptor, 624, 16, 0x4);         // block z
    set_bit(descriptor, 640 + 0);
    set_bit(descriptor, 640 + 5);
    set_bit(descriptor, 382);                    // linked sampler index
    set_range(descriptor, 1440, 24, 0x1234);     // local memory low
    set_range(descriptor, 1472, 24, 0x5678);     // local memory high
    set_range(descriptor, 1504, 24, 0x20c);      // local crs
    set_range(descriptor, 1467, 5, 0b10110);     // barrier count
    set_range(descriptor, 1496, 8, 0x80);        // register count
    return descriptor;
}

void set_constant_buffer(Descriptor& descriptor, unsigned index, std::uint64_t address,
                         std::uint32_t size, bool enabled) {
    set_bit(descriptor, 640 + index, enabled);
    set_range(descriptor, 928 + index * 64, 32, std::uint32_t(address));
    set_range(descriptor, 960 + index * 64, 8, std::uint32_t(address >> 32));
    set_range(descriptor, 975 + index * 64, 17, size & 0x1'ffffu);
}

void run_all_zero() {
    const auto launch = decode_compute_qmd(all_zero());
    CHECK(launch.grid == Dimensions{});
    CHECK(launch.block == Dimensions{});
    CHECK(launch.shared_memory_size == 0);
    CHECK(launch.local_memory_size == 0);
    CHECK(launch.local_crs_size == 0);
    CHECK(launch.barrier_count == 0);
    CHECK(launch.register_count == 0);
    CHECK(launch.program_offset == 0);
    CHECK(!launch.sampler_index);
    for (const auto& buffer : launch.constant_buffers) {
        CHECK(!buffer.enabled);
        CHECK(buffer.address == 0);
        CHECK(buffer.size == 0);
    }
}

void run_full_launch() {
    auto descriptor = full_launch();
    // The single 32-bit write covered grid x and the low half of grid y; the
    // explicit ranges above made them 0x13, 0x2 and 0x3.
    set_constant_buffer(descriptor, 0, 0x1'0000'1000ull, 0x1'ffff, true);
    set_constant_buffer(descriptor, 5, 0xad'beef'0000ull, 0x40, true);
    const auto launch = decode_compute_qmd(descriptor);
    CHECK((launch.grid == Dimensions{0x13, 0x2, 0x3}));
    CHECK((launch.block == Dimensions{0x40, 0x8, 0x4}));
    CHECK(launch.shared_memory_size == 0x4000);
    CHECK(launch.local_memory_size == 0x1234 + 0x5678);
    CHECK(launch.local_crs_size == 0x20c);
    CHECK(launch.barrier_count == 0b10110);
    CHECK(launch.register_count == 0x80);
    CHECK(launch.program_offset == 0x4000);
    CHECK(launch.sampler_index);

    CHECK(launch.constant_buffers[0].enabled);
    CHECK(launch.constant_buffers[0].address == 0x1'0000'1000ull);
    CHECK(launch.constant_buffers[0].size == 0x1'ffff);
    CHECK(!launch.constant_buffers[1].enabled);
    CHECK(launch.constant_buffers[5].enabled);
    CHECK(launch.constant_buffers[5].address == 0xad'beef'0000ull);
    CHECK(launch.constant_buffers[5].size == 0x40);
    CHECK(!launch.constant_buffers[7].enabled);
}

void run_buffer_boundaries() {
    // The sizes and addresses are exact bit extractions: the
    // maximum 17-bit size and the maximum 8-bit address upper byte must
    // survive unchanged, so the largest representable address is 40 bits.
    Descriptor descriptor{};
    set_constant_buffer(descriptor, 7, 0xff'ffff'ffffull, 0x1'ffff, true);
    const auto launch = decode_compute_qmd(descriptor);
    CHECK(launch.constant_buffers[7].enabled);
    CHECK(launch.constant_buffers[7].address == 0xff'ffff'ffffull);
    CHECK(launch.constant_buffers[7].size == 0x1'ffff);
}

}  // namespace

int main() {
    run_all_zero();
    run_full_launch();
    run_buffer_boundaries();
    if (failures == 0) std::printf("compute_qmd tests passed\n");
    else std::fprintf(stderr, "%u compute_qmd checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
