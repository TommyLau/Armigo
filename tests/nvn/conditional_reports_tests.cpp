// ============================================================================
// Armigo Tests - NVN Conditional Report Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "conditional_reports.h"

#include <cstdint>
#include <cstdio>
#include <map>

namespace {

using armigo::nvn::ConditionalReports;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

// Every address is readable. Bytes that were never stored read as 0, which is
// what an untouched counter record contains.
struct Memory {
    std::map<std::uint64_t, std::uint8_t> bytes;
    void set(std::uint64_t address, std::uint64_t value) {
        for (unsigned index = 0; index != 8; ++index)
            bytes[address + index] = static_cast<std::uint8_t>(value >> (8 * index));
    }
    ConditionalReports::Reader reader() {
        return [this](std::uint64_t address, std::uint64_t& value) {
            value = 0;
            for (unsigned index = 0; index != 8; ++index) {
                const auto found = bytes.find(address + index);
                if (found != bytes.end())
                    value |= static_cast<std::uint64_t>(found->second) << (8 * index);
            }
            return true;
        };
    }
};

bool known_value(ConditionalReports& reports, std::uint64_t address, std::uint64_t& out) {
    return reports.value(address, out);
}

// RENDER_IF_NOT_EQUAL skips only when both counters are known and equal.
bool skips_if_not_equal(ConditionalReports& reports, std::uint64_t address) {
    std::uint64_t left = 0, right = 0;
    const bool known = reports.value(address, left) && reports.value(address + 16, right);
    return known && left == right;
}

void check_timestamp_skips_equal_zero() {
    Memory memory;
    memory.set(0x1000, 1);
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    reports.note_report(0x1000, ConditionalReports::Kind::timestamp);
    std::uint64_t value = 1;
    CHECK(known_value(reports, 0x1000, value) && value == 0);
    CHECK(known_value(reports, 0x1010, value) && value == 0);
    CHECK(skips_if_not_equal(reports, 0x1000));
}

void check_samples_until_complete() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t address = 0x2000;
    reports.note_report(address, ConditionalReports::Kind::samples_passed);
    std::uint64_t value = 0;
    CHECK(!known_value(reports, address, value));
    reports.commit(1);
    CHECK(!known_value(reports, address, value));
    memory.set(address, 42);
    reports.set_completed(1);
    CHECK(known_value(reports, address, value) && value == 42);
    // The neighbour is a query operand with no report of its own, so it is
    // the bytes currently in memory.
    memory.set(address + 16, 7);
    CHECK(known_value(reports, address + 16, value) && value == 7);
}

void check_explicit_writes() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t address = 0x3000;
    const std::uint64_t pattern = 0x1122334455667788ull;
    reports.note_report(address, ConditionalReports::Kind::samples_passed);
    reports.write(address, 8, &pattern, 0);
    std::uint64_t value = 0;
    // The report is still pending. All 8 command bytes are known, including
    // while that writer has been stamped and the watermark has not reached it.
    CHECK(known_value(reports, address, value) && value == pattern);
    reports.commit(4);
    CHECK(known_value(reports, address, value) && value == pattern);
    memory.set(address, 99);
    reports.set_completed(4);
    // The write raced the report, so the landed memory value is not used.
    CHECK(!known_value(reports, address, value));

    constexpr std::uint64_t partial = 0x3100;
    reports.note_report(partial, ConditionalReports::Kind::samples_passed);
    const std::uint32_t low = 0xaabbccdd;
    reports.write(partial, 4, &low, 0);
    CHECK(!known_value(reports, partial, value));
    reports.commit(5);
    reports.set_completed(5);
    memory.set(partial, 0x99);
    // The other four bytes were never command-known, and the raced report
    // stays unknown after the fence completes.
    CHECK(!known_value(reports, partial, value));

    constexpr std::uint64_t timestamp = 0x3200;
    reports.note_report(timestamp, ConditionalReports::Kind::timestamp);
    const std::uint8_t byte = 0xab;
    reports.write(timestamp, 1, &byte, 0);
    // A timestamp counter is already known 0, so the unwritten bytes stay 0.
    CHECK(known_value(reports, timestamp, value) && value == 0xab);

    constexpr std::uint64_t settled = 0x3300;
    reports.note_report(settled, ConditionalReports::Kind::samples_passed);
    reports.commit(6);
    reports.set_completed(6);
    memory.set(settled, 0x1122334455667788ull);
    CHECK(known_value(reports, settled, value) && value == 0x1122334455667788ull);
    reports.write(settled, 1, &byte, 0);
    CHECK(known_value(reports, settled, value) && value == 0x11223344556677abull);
}

void check_copy() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t source = 0x4000, destination = 0x4100;
    const std::uint64_t pattern = 0x0fedcba987654321ull;
    reports.note_report(source, ConditionalReports::Kind::timestamp);
    reports.write(source, 8, &pattern, 0);
    reports.note_report(destination, ConditionalReports::Kind::timestamp);
    reports.copy(source, destination, 8);
    std::uint64_t value = 0;
    CHECK(known_value(reports, destination, value) && value == pattern);
}

void check_unbounded_store() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t query = 0x5000;
    reports.note_report(query, ConditionalReports::Kind::samples_passed);
    reports.commit(1);
    reports.set_completed(1);
    memory.set(query, 42);
    memory.set(query + 16, 7);
    std::uint64_t value = 0;
    CHECK(known_value(reports, query, value) && value == 42);
    CHECK(known_value(reports, query + 16, value) && value == 7);
    reports.gpu_write_unbounded();
    CHECK(!known_value(reports, query, value));
    CHECK(!known_value(reports, query + 16, value));
    reports.commit(2);
    reports.set_completed(2);
    // The reported query stays unknown after the store's fence. The neighbour
    // was never reported, so it is readable again.
    CHECK(!known_value(reports, query, value));
    CHECK(known_value(reports, query + 16, value) && value == 7);
    const std::uint64_t pattern = 0x5555aaa511110000ull;
    reports.write(query, 8, &pattern, 0);
    CHECK(known_value(reports, query, value) && value == pattern);

    constexpr std::uint64_t fresh = 0x5100;
    reports.note_report(fresh, ConditionalReports::Kind::samples_passed);
    reports.commit(3);
    reports.set_completed(3);
    reports.gpu_write_unbounded();
    CHECK(!known_value(reports, fresh, value));
    reports.note_report(fresh, ConditionalReports::Kind::timestamp);
    CHECK(known_value(reports, fresh, value) && value == 0);
}

void check_completed_timestamp_copy_after_unbounded() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t source = 0x6000, destination = 0x6010;
    reports.note_report(source, ConditionalReports::Kind::timestamp);
    reports.commit(1);
    reports.set_completed(1);
    memory.set(source, 0x1111111111111111ull);
    std::uint64_t value = 0;
    CHECK(known_value(reports, source, value) && value == 0x1111111111111111ull);
    reports.gpu_write_unbounded();
    // The timestamp itself is not a query operand, so its value still comes
    // from memory. A copy must not forward those bytes.
    CHECK(known_value(reports, source, value) && value == 0x1111111111111111ull);
    reports.note_report(destination, ConditionalReports::Kind::timestamp);
    reports.copy(source, destination, 8);
    CHECK(!known_value(reports, destination, value));
    reports.note_report(source, ConditionalReports::Kind::timestamp);
    CHECK(known_value(reports, source, value) && value == 0);
}

void check_fence_order() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t address = 0x7000;
    memory.set(address, 0x1111111111111111ull);
    reports.note_report(address, ConditionalReports::Kind::timestamp);
    reports.commit(1);
    std::uint64_t value = 0;
    CHECK(known_value(reports, address, value) && value == 0);
    reports.set_completed(1);
    CHECK(known_value(reports, address, value) && value == 0x1111111111111111ull);
    const std::uint64_t pattern = 0x2222222222222222ull;
    reports.write(address, 8, &pattern, 0);
    reports.commit(2);
    CHECK(known_value(reports, address, value) && value == pattern);
    reports.set_completed(2);
    CHECK(known_value(reports, address, value) && value == 0x1111111111111111ull);
}

void check_other_and_forget() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t address = 0x8000;
    reports.note_report(address, ConditionalReports::Kind::other);
    reports.commit(1);
    reports.set_completed(1);
    memory.set(address, 5);
    std::uint64_t value = 0;
    CHECK(!known_value(reports, address, value));

    reports.note_report(address + 32, ConditionalReports::Kind::timestamp);
    CHECK(known_value(reports, address + 32, value) && value == 0);
    reports.forget();
    CHECK(known_value(reports, address + 32, value) && value == 0);
    memory.set(address + 32, 5);
    CHECK(known_value(reports, address + 32, value) && value == 5);

    reports.set_completed(10);
    reports.forget();
    reports.note_report(address, ConditionalReports::Kind::timestamp);
    reports.commit(10);
    memory.set(address, 99);
    // forget() leaves the watermark at 10, so this writer has already landed.
    CHECK(known_value(reports, address, value) && value == 99);
    reports.note_report(address, ConditionalReports::Kind::timestamp);
    reports.commit(11);
    CHECK(known_value(reports, address, value) && value == 0);
}

void check_many_queries_then_timestamp() {
    Memory memory;
    memory.set(0x9000, 1);
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t base = 0x7100000000ull;
    for (std::uint64_t index = 0; index != 9000; ++index)
        reports.note_report(base + index * 16, ConditionalReports::Kind::samples_passed);
    reports.note_report(0x9000, ConditionalReports::Kind::timestamp);
    std::uint64_t value = 1;
    CHECK(known_value(reports, 0x9000, value) && value == 0);
    CHECK(!known_value(reports, base, value));
    CHECK(!known_value(reports, base + 8999 * 16, value));
    reports.note_report(0x9100, ConditionalReports::Kind::timestamp);
    CHECK(known_value(reports, 0x9100, value) && value == 0);
}

void check_wide_write_page_index() {
    ConditionalReports reports;
    Memory memory;
    reports.set_reader(memory.reader());
    constexpr std::uint64_t start = 0x7200100000ull;
    constexpr std::uint64_t span = 0x800000;
    const std::uint64_t addresses[] = {start + span, start - 3, start + 0x700000,
                                       start - 16, start + 32, start + 0x100000};
    for (auto address : addresses)
        reports.note_report(address, ConditionalReports::Kind::timestamp);
    const std::uint8_t fill = 0xab;
    reports.write(start, span, &fill, 1);
    std::uint64_t value = 1;
    CHECK(known_value(reports, addresses[0], value) && value == 0);
    CHECK(known_value(reports, addresses[1], value) && value == 0xababababab000000ull);
    CHECK(known_value(reports, addresses[2], value) && value == 0xababababababababull);
    CHECK(known_value(reports, addresses[3], value) && value == 0);
    CHECK(known_value(reports, addresses[4], value) && value == 0xababababababababull);
    CHECK(known_value(reports, addresses[5], value) && value == 0xababababababababull);
    reports.forget();
    reports.note_report(start + 32, ConditionalReports::Kind::timestamp);
    reports.write(start, span, &fill, 1);
    CHECK(known_value(reports, start + 32, value) && value == 0xababababababababull);
}

void check_missing_reader() {
    ConditionalReports reports;
    reports.note_report(0x100, ConditionalReports::Kind::timestamp);
    std::uint64_t value = 1;
    CHECK(known_value(reports, 0x100, value) && value == 0);
    reports.forget();
    CHECK(!known_value(reports, 0x100, value));
    reports.note_report(0x100, ConditionalReports::Kind::timestamp);
    reports.commit(1);
    reports.set_completed(1);
    CHECK(!known_value(reports, 0x100, value));
}

// A failed submission's reports never land. Once a later submission moves
// the watermark past it, those counters stay unknown instead of reading the
// memory the failed report would have written.
void check_failed_sequence() {
    Memory memory;
    ConditionalReports reports;
    reports.set_reader(memory.reader());
    memory.set(0x300, 7);
    reports.note_report(0x300, ConditionalReports::Kind::samples_passed);
    reports.commit(1);
    reports.note_report(0x400, ConditionalReports::Kind::timestamp);
    reports.commit(2);
    reports.set_failed(1);
    reports.set_completed(2);
    std::uint64_t value = 0;
    CHECK(!known_value(reports, 0x300, value));
    CHECK(known_value(reports, 0x400, value) && value == 0);
    // A fresh report of a later submission is known again once it lands.
    reports.note_report(0x300, ConditionalReports::Kind::samples_passed);
    reports.commit(3);
    reports.set_completed(3);
    CHECK(known_value(reports, 0x300, value) && value == 7);
}

} // namespace

int main() {
    check_timestamp_skips_equal_zero();
    check_samples_until_complete();
    check_explicit_writes();
    check_copy();
    check_unbounded_store();
    check_completed_timestamp_copy_after_unbounded();
    check_fence_order();
    check_other_and_forget();
    check_many_queries_then_timestamp();
    check_wide_write_page_index();
    check_missing_reader();
    check_failed_sequence();
    std::printf("NVN conditional report failures: %u\n", failures);
    return failures ? 1 : 0;
}
