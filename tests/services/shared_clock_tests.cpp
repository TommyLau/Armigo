// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/services/shared_clock.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <limits>

namespace {
using namespace armigo::services::shared_clock;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)

struct alignas(4096) Page { std::array<std::byte, 4096> bytes; };

// Independent wire decoder: no implementation layout constants or native
// struct copies are used to identify payloads and signed integer fields.
std::uint64_t integer(std::span<const std::byte> bytes, std::size_t position,
                      std::size_t count = 8) {
    std::uint64_t value = 0;
    for (std::size_t i = count; i != 0; --i)
        value = value * 256 + std::to_integer<unsigned char>(bytes[position + i - 1]);
    return value;
}

std::int64_t signed_integer(std::span<const std::byte> bytes, std::size_t position) {
    return std::bit_cast<std::int64_t>(integer(bytes, position));
}

std::span<const std::byte> selected(const Page& page, std::size_t header,
                                  std::size_t first_slot, std::size_t payload_size) {
    const auto generation = integer(page.bytes, header, 4);
    return std::span<const std::byte>(page.bytes).subspan(
        first_slot + (generation % 2) * payload_size, payload_size);
}

SourceId source(unsigned first) {
    SourceId id;
    for (std::size_t i = 0; i < id.size(); ++i) id[i] = static_cast<std::byte>(first + i);
    return id;
}

bool has_source(std::span<const std::byte> bytes, std::size_t offset, const SourceId& id) {
    return std::equal(id.begin(), id.end(), bytes.begin() + offset);
}

Configuration configuration() {
    Configuration config;
    config.counter_nanoseconds = 5'250'000'000;
    config.steady_offset_nanoseconds = -2'000'000'000;
    config.steady_source_id = source(1);
    config.fallback_wall_seconds = 1'234;
    return config;
}

void check_context(std::span<const std::byte> bytes, std::int64_t offset,
                   std::int64_t point, const SourceId& id) {
    CHECK(signed_integer(bytes, 0) == offset);
    CHECK(signed_integer(bytes, 8) == point);
    CHECK(has_source(bytes, 16, id));
}

void test_all_blocks_and_counter_progress() {
    Page page;
    page.bytes.fill(std::byte{0xa5});
    const auto config = configuration();
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    CHECK(integer(page.bytes, 0, 4) == 2);
    CHECK(integer(page.bytes, 0x38, 4) == 2);
    CHECK(integer(page.bytes, 0x80, 4) == 2);
    CHECK(integer(page.bytes, 0xc8, 4) == 2);
    CHECK(integer(page.bytes, 0xd0, 4) == 1);

    const auto steady = selected(page, 0, 8, 24);
    CHECK(signed_integer(steady, 0) == -2'000'000'000);
    CHECK(has_source(steady, 8, source(1)));
    CHECK(steady[0] == std::byte{0} && steady[1] == std::byte{0x6c} &&
          steady[2] == std::byte{0xca} && steady[3] == std::byte{0x88});
    const auto local = selected(page, 0x38, 0x40, 32);
    check_context(local, 1231, 3, source(1));
    check_context(selected(page, 0x80, 0x88, 32), 0, 0, SourceId{});
    CHECK(!has_source(selected(page, 0x80, 0x88, 32), 16, source(1)));
    CHECK(selected(page, 0xc8, 0xcc, 1)[0] == std::byte{0});
    const auto continuous = selected(page, 0xd0, 0xd8, 56);
    CHECK(continuous.data() == page.bytes.data() + 0x110);
    CHECK(signed_integer(continuous, 0) == 5'250'000'000);
    // diff_scale is Q32, shift_amount is its exponent, lower is zero and upper is the clamp bound,
    // so the SDK's correction evaluates to the identity transform.
    CHECK(signed_integer(continuous, 8) == (std::int64_t{1} << 32));
    CHECK(signed_integer(continuous, 16) == 32);
    CHECK(signed_integer(continuous, 24) == 0);
    CHECK(signed_integer(continuous, 32) > 0);
    {
        // Same arithmetic the SDK performs: (now - rtc_offset) * diff_scale >> shift + lower.
        const auto rtc_offset = signed_integer(continuous, 0);
        const auto diff_scale = signed_integer(continuous, 8);
        const auto shift = signed_integer(continuous, 16);
        const auto lower = signed_integer(continuous, 24);
        const std::int64_t now = rtc_offset + 1'000'000'000;
        CHECK((((now - rtc_offset) * diff_scale) >> shift) + lower == 1'000'000'000);
    }
    CHECK(has_source(continuous, 40, source(1)));

    // Check inactive slots, padding and all unused page bytes independently.
    std::array<bool, 4096> occupied{};
    const auto mark = [&](std::size_t start, std::size_t count) {
        std::fill_n(occupied.begin() + start, count, true);
    };
    for (const auto header : {0, 0x38, 0x80, 0xc8, 0xd0}) mark(header, 4);
    mark(8, 24); mark(0x40, 32); mark(0x88, 32); mark(0xcc, 1); mark(0x110, 56);
    for (std::size_t i = 0; i < occupied.size(); ++i)
        if (!occupied[i]) CHECK(page.bytes[i] == std::byte{0});
    const auto snapshot = page.bytes;
    const auto future_counter_ns = config.counter_nanoseconds + 10'000'000'000;
    const auto future_steady_seconds = (future_counter_ns + signed_integer(steady, 0)) / 1'000'000'000;
    CHECK(future_steady_seconds + signed_integer(local, 0) == 1244);
    CHECK(page.bytes == snapshot);
}

void test_saved_context_and_fallback() {
    Page page;
    auto config = configuration();
    config.local_context = SystemClockContext{-700, 83, source(1)};
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    check_context(selected(page, 0x38, 0x40, 32), -700, 83, source(1));
    config.local_context->source_id = source(70);
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    check_context(selected(page, 0x38, 0x40, 32), 1231, 3, source(1));
    config.local_context.reset();
    config.counter_nanoseconds = 1'000'000'000;
    config.steady_offset_nanoseconds = -3'500'000'000;
    config.fallback_wall_seconds = -500;
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    check_context(selected(page, 0x38, 0x40, 32), -498, -2, source(1));
}

void test_network_context_and_correction() {
    Page page;
    auto config = configuration();
    config.network_context = SystemClockContext{900, -12, source(80)};
    config.automatic_correction_enabled = true;
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    check_context(selected(page, 0x80, 0x88, 32), 900, -12, source(80));
    check_context(selected(page, 0x38, 0x40, 32), 1231, 3, source(1));
    CHECK(selected(page, 0xc8, 0xcc, 1)[0] == std::byte{1});
    config.network_context->source_id = source(1);
    config.automatic_correction_enabled = false;
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    check_context(selected(page, 0x38, 0x40, 32), 1231, 3, source(1));
    config.automatic_correction_enabled = true;
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    check_context(selected(page, 0x38, 0x40, 32), 900, -12, source(1));
    check_context(selected(page, 0x80, 0x88, 32), 900, -12, source(1));
    config.network_context.reset();
    CHECK(initialize(page.bytes, config) == InitializeResult::initialized);
    check_context(selected(page, 0x80, 0x88, 32), 0, 0, SourceId{});
    check_context(selected(page, 0x38, 0x40, 32), 1231, 3, source(1));
}

void test_invalid_inputs_preserve_storage() {
    Page page;
    page.bytes.fill(std::byte{0x5a});
    const auto snapshot = page.bytes;
    auto config = configuration();
    CHECK(initialize(std::span<std::byte>(page.bytes).first(4095), config) == InitializeResult::invalid_backing);
    CHECK(initialize({}, config) == InitializeResult::invalid_backing);
    alignas(8) std::array<std::byte, 4097> unaligned;
    unaligned.fill(std::byte{0x5a});
    CHECK(initialize(std::span<std::byte>(unaligned).subspan(1), config) == InitializeResult::invalid_backing);
    CHECK(std::all_of(unaligned.begin(), unaligned.end(), [](std::byte b) { return b == std::byte{0x5a}; }));
    config.steady_source_id = {};
    CHECK(initialize(page.bytes, config) == InitializeResult::invalid_source);
    config = configuration();
    config.counter_nanoseconds = std::numeric_limits<std::int64_t>::min();
    CHECK(initialize(page.bytes, config) == InitializeResult::time_out_of_range);
    config.counter_nanoseconds = std::numeric_limits<std::int64_t>::max();
    config.steady_offset_nanoseconds = 1;
    CHECK(initialize(page.bytes, config) == InitializeResult::time_out_of_range);
    config = configuration();
    config.fallback_wall_seconds = std::numeric_limits<std::int64_t>::min();
    CHECK(initialize(page.bytes, config) == InitializeResult::time_out_of_range);
    config.counter_nanoseconds = 0;
    config.fallback_wall_seconds = std::numeric_limits<std::int64_t>::max();
    CHECK(initialize(page.bytes, config) == InitializeResult::time_out_of_range);
    CHECK(page.bytes == snapshot);
}
}  // namespace

int main() {
    test_all_blocks_and_counter_progress();
    test_saved_context_and_fallback();
    test_network_context_and_correction();
    test_invalid_inputs_preserve_storage();
    if (failures) return 1;
    std::puts("shared-clock checks passed");
    return 0;
}
