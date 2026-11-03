// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_encode_ring.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

namespace {

using armigo::platform::metal::nvn_driver::EncodeRing;

int failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)

constexpr std::uint16_t record_op = 7, stop_op = 8;

// Payload bytes of record n are a function of n, so the consumer can check
// every byte it replays.
std::uint8_t pattern(std::uint64_t n, std::size_t i) {
    return std::uint8_t((n * 131 + i * 7) ^ (n >> 3));
}

std::size_t payload_size(std::uint64_t n) {
    // Mostly small records, now and then one near the payload limit, so
    // records straddle the end of the ring at many offsets.
    if (n % 997 == 0) return EncodeRing::max_payload - (n % 64);
    return (n * 2654435761u >> 7) % 600;
}

// Records of every size cross the end of the ring many times; held records
// are published later in a batch; the consumer sees each record once, in
// order, with its argument and payload intact.
void test_ordered_records_across_wraps() {
    auto ring = std::make_unique<EncodeRing>();
    constexpr std::uint64_t count = 400'000;
    std::uint64_t seen = 0, bad = 0;
    std::atomic<bool> idle{false};
    std::thread consumer([&] {
        ring->consume([&](const EncodeRing::Header& header, const void* payload) {
            if (header.op == stop_op) return false;
            if (header.op != record_op || header.argument != std::uint32_t(seen)) { ++bad; ++seen; return true; }
            const auto size = payload_size(seen);
            const auto* bytes = static_cast<const std::uint8_t*>(payload);
            if (header.size < sizeof(EncodeRing::Header) + size) ++bad;
            for (std::size_t i = 0; i < size; ++i)
                if (bytes[i] != pattern(seen, i)) { ++bad; break; }
            ++seen;
            return true;
        }, [&] { return idle.load(); }, 20'000);
    });
    for (std::uint64_t n = 0; n < count; ++n) {
        const auto size = payload_size(n);
        auto* bytes = static_cast<std::uint8_t*>(ring->put(record_op, std::uint32_t(n), size));
        for (std::size_t i = 0; i < size; ++i) bytes[i] = pattern(n, i);
        if (n % 5 != 4) ring->publish();
        // The consumer alternates between spinning and sleeping.
        if (n % 50'000 == 0) idle.store(!idle.load());
    }
    ring->drain();
    CHECK(seen == count);
    ring->put(stop_op, 0, 0);
    ring->publish();
    consumer.join();
    CHECK(seen == count);
    CHECK(bad == 0);
    CHECK(ring->full_waits() > 0);
}

// A sleeping consumer wakes for each publish; drain returns only after the
// consumer has replayed every record put so far.
void test_drain_and_sleep() {
    auto ring = std::make_unique<EncodeRing>();
    std::atomic<std::uint64_t> seen{0};
    std::thread consumer([&] {
        ring->consume([&](const EncodeRing::Header& header, const void*) {
            if (header.op == stop_op) return false;
            seen.fetch_add(1);
            return true;
        }, [] { return true; }, 0);
    });
    for (int round = 0; round < 200; ++round) {
        ring->put(record_op, 0, 16);
        ring->put(record_op, 0, 0);
        if (round % 3) std::this_thread::sleep_for(std::chrono::microseconds(50));
        ring->drain();
        CHECK(seen.load() == std::uint64_t(2 * (round + 1)));
    }
    ring->put(stop_op, 0, 0);
    ring->publish();
    consumer.join();
}

} // namespace

int main() {
    test_ordered_records_across_wraps();
    test_drain_and_sleep();
    std::printf("NVN encode ring failures: %d\n", failures);
    return failures ? 1 : 0;
}
