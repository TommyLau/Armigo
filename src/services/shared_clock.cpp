// ============================================================================
// Armigo Services - Shared Clock Initialization
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/services/shared_clock.h"

#include <algorithm>
#include <limits>

namespace armigo::services::shared_clock {
namespace {

constexpr std::int64_t nanoseconds_per_second = 1'000'000'000;

bool add(std::int64_t a, std::int64_t b, std::int64_t& result) {
    if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b))
        return false;
    result = a + b;
    return true;
}

bool subtract(std::int64_t a, std::int64_t b, std::int64_t& result) {
    if ((b > 0 && a < std::numeric_limits<std::int64_t>::min() + b) ||
        (b < 0 && a > std::numeric_limits<std::int64_t>::max() + b))
        return false;
    result = a - b;
    return true;
}

void put(std::span<std::byte> destination, std::size_t offset,
         std::uint64_t value, std::size_t count = 8) {
    for (std::size_t i = 0; i < count; ++i)
        destination[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 0xff);
}

std::array<std::byte, 32> encode_context(const SystemClockContext& context) {
    std::array<std::byte, 32> bytes{};
    put(bytes, 0, static_cast<std::uint64_t>(context.offset_seconds));
    put(bytes, 8, static_cast<std::uint64_t>(context.steady_seconds));
    std::copy(context.source_id.begin(), context.source_id.end(), bytes.begin() + 16);
    return bytes;
}

void publish(std::span<std::byte> page, std::size_t block, std::size_t slot_offset,
             std::span<const std::byte> payload, std::uint32_t generation) {
    const auto slot = block + slot_offset + (generation & 1U) * payload.size();
    std::copy(payload.begin(), payload.end(), page.begin() + slot);
    // Construction is exclusively before publication to readers. The wire
    // generation is written last, but these byte stores are not a concurrent
    // writer protocol: the owner must synchronize the completed page handoff.
    put(page, block, generation, 4);
}

}  // namespace

InitializeResult initialize(std::span<std::byte> backing,
                            const Configuration& configuration) noexcept {
    if (backing.size() != page_size || backing.data() == nullptr ||
        reinterpret_cast<std::uintptr_t>(backing.data()) % backing_alignment != 0)
        return InitializeResult::invalid_backing;
    if (std::all_of(configuration.steady_source_id.begin(), configuration.steady_source_id.end(),
                    [](std::byte value) { return value == std::byte{0}; }))
        return InitializeResult::invalid_source;

    std::int64_t steady_nanoseconds = 0;
    if (!add(configuration.counter_nanoseconds, configuration.steady_offset_nanoseconds,
             steady_nanoseconds))
        return InitializeResult::time_out_of_range;
    const auto steady_seconds = steady_nanoseconds / nanoseconds_per_second;
    SystemClockContext local;
    if (configuration.local_context &&
        configuration.local_context->source_id == configuration.steady_source_id) {
        local = *configuration.local_context;
    } else {
        if (!subtract(configuration.fallback_wall_seconds, steady_seconds, local.offset_seconds))
            return InitializeResult::time_out_of_range;
        local.steady_seconds = steady_seconds;
        local.source_id = configuration.steady_source_id;
    }
    const auto network = configuration.network_context.value_or(SystemClockContext{});
    // The user clock starts with correction disabled. Enabling correction at
    // startup copies a valid network context into the local system clock.
    if (configuration.automatic_correction_enabled &&
        network.source_id == configuration.steady_source_id)
        local = network;

    std::array<std::byte, 24> steady{};
    put(steady, 0, static_cast<std::uint64_t>(configuration.steady_offset_nanoseconds));
    std::copy(configuration.steady_source_id.begin(), configuration.steady_source_id.end(),
              steady.begin() + 8);
    // ContinuousAdjustmentTimePoint = {rtc_offset, diff_scale, shift_amount, lower, upper, source_id}.
    // The SDK consumes it as corrected = (now - rtc_offset) * diff_scale >> shift_amount + lower,
    // clamped by upper (sdk+0x443958). Publishing the identity transform keeps the corrected time equal
    // to the steady clock; the previous revision wrote a nanosecond offset into lower and upper, which
    // made every corrected time unbounded.
    std::array<std::byte, 56> continuous{};
    put(continuous, 0, static_cast<std::uint64_t>(configuration.counter_nanoseconds));
    put(continuous, 8, std::uint64_t{1} << 32);
    put(continuous, 16, 32);
    put(continuous, 24, 0);
    put(continuous, 32, 0x7fffffffffffffffull);
    std::copy(configuration.steady_source_id.begin(), configuration.steady_source_id.end(),
              continuous.begin() + 40);
    const auto local_bytes = encode_context(local);
    const auto network_bytes = encode_context(network);
    const std::array automatic{static_cast<std::byte>(configuration.automatic_correction_enabled)};

    std::fill(backing.begin(), backing.end(), std::byte{0});
    // Mirror provisional zero publication in the first four firmware blocks,
    // followed by initialized state. The continuous block is first published
    // during steady-clock setup. Inactive slots remain zero in this startup
    // snapshot; future updates and their publication barriers are out of scope.
    put(backing, 0x000, 1, 4);
    put(backing, 0x038, 1, 4);
    put(backing, 0x080, 1, 4);
    put(backing, 0x0c8, 1, 4);
    publish(backing, 0x000, 8, steady, 2);
    publish(backing, 0x038, 8, local_bytes, 2);
    publish(backing, 0x080, 8, network_bytes, 2);
    publish(backing, 0x0c8, 4, automatic, 2);
    publish(backing, 0x0d0, 8, continuous, 1);
    return InitializeResult::initialized;
}

}  // namespace armigo::services::shared_clock
