// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/play_report.h>
#include <armigo/horizon/event.h>
#include <armigo/horizon/startup_queries.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace armigo::horizon;
using armigo::services::prepo::PlayReportSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (false)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index)
        message[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}
std::uint32_t get(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < 4; ++index)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + index])}
                 << (8 * index);
    return value;
}

IpcMessage control(std::uint32_t command) {
    IpcMessage message{};
    put(message, 0, 5);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 24, command);
    return message;
}

// Independently authored room and MessagePack {"event": 1} report. High
// address bits exercise descriptors without relying on a host pointer value.
constexpr std::uint64_t room_address = 0x2100001000;
constexpr std::uint64_t report_address = 0x2100002000;
struct Memory {
    std::array<std::byte, 64> room{std::byte{'t'}, std::byte{'e'}, std::byte{'s'},
                                  std::byte{'t'}};
    std::vector<std::byte> report{std::byte{0x81}, std::byte{0xa5}, std::byte{'e'},
        std::byte{'v'}, std::byte{'e'}, std::byte{'n'}, std::byte{'t'}, std::byte{1}};
    unsigned reads = 0;
    unsigned writes = 0;

    static bool read(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<Memory*>(context);
        ++self.reads;
        for (const auto base : {room_address, report_address}) {
            const auto bytes = base == room_address ? std::span<const std::byte>{self.room}
                                                   : std::span<const std::byte>{self.report};
            if (address >= base && address - base <= bytes.size() &&
                out.size() <= bytes.size() - (address - base)) {
                std::copy_n(bytes.begin() + (address - base), out.size(), out.begin());
                return true;
            }
        }
        return false;
    }
    static bool write(void* context, std::uint64_t, std::span<const std::byte>) noexcept {
        ++static_cast<Memory*>(context)->writes;
        return false;
    }
};

IpcMessage report_request(std::uint32_t command, bool user, bool domain = false) {
    IpcMessage message{};
    put(message, 0, 6 | (1u << 16) | (1u << 20)); // One X and one A buffer.
    const auto opt_in = command == 10107 ? 8u : 0u;
    const auto data_size = (user ? 40u : 24u) + opt_in;
    put(message, 4, 0x80000000u | ((16 + (domain ? 16 : 0) + data_size) / 4));
    put(message, 8, 1); // Send PID, no handles.
    put(message, 12, 999); // The service must use the dispatcher's sender PID.
    put(message, 20, (5u << 16) | (2u << 6) | (1u << 12));
    put(message, 24, static_cast<std::uint32_t>(room_address));
    put(message, 28, 8);
    put(message, 32, static_cast<std::uint32_t>(report_address));
    put(message, 36, (2u << 2) | (1u << 28));
    const auto cmif = domain ? 64u : 48u;
    if (domain) {
        put(message, 48, (data_size << 16) | 1);
        put(message, 52, 1);
    }
    put(message, cmif, 0x49434653);
    put(message, cmif + 4, 1);
    put(message, cmif + 8, command);
    if (user) put(message, cmif + 16 + opt_in, 1);
    put(message, cmif + (user ? 32 : 16) + opt_in, 999);
    return message;
}

struct Fixture {
    Memory memory;
    HandleTable handles{4};
    std::shared_ptr<PlayReportSession> session = std::make_shared<PlayReportSession>(71);
    Handle handle = handles.insert(session).value();
    IpcDispatcher dispatcher{handles, nullptr};
    IpcMessage message{};
    QueryRegisters registers{};

    IpcDisposition send(std::uint64_t sender = 71) {
        registers.fill(0x123456789abcdef0);
        registers[0] = handle;
        IpcCallContext caller{sender, message, nullptr, nullptr};
        caller.read_memory = &Memory::read;
        caller.write_memory = &Memory::write;
        caller.memory_context = &memory;
        return dispatcher.dispatch(0x21, registers, caller);
    }
    void reject() {
        const auto before = message;
        CHECK(send() == IpcDisposition::unsupported);
        CHECK(message == before);
        CHECK(session->diagnostic() != nullptr);
    }
    void result(std::uint32_t expected, bool domain = false) {
        CHECK(send() == IpcDisposition::handled);
        const auto cmif = domain ? 32u : 16u;
        CHECK(get(message, cmif) == 0x4f434653);
        CHECK(get(message, cmif + 4) == 0);
        CHECK(get(message, cmif + 8) == expected);
        CHECK(get(message, cmif + 12) == 0x2749f93d);
        CHECK(std::all_of(registers.begin(), registers.end(), [](auto value) { return value == 0; }));
        CHECK(session->diagnostic() == nullptr);
        CHECK(memory.writes == 0);
    }
};

void test_session_controls_and_lifetime() {
    Fixture fixture;
    CHECK(fixture.handles.lookup_as<ClientSession>(fixture.handle) == fixture.session);
    for (const auto type : {5u, 7u}) {
        fixture.message = control(3);
        put(fixture.message, 0, type);
        CHECK(fixture.send() == IpcDisposition::handled);
        CHECK(get(fixture.message, 16) == 0x4f434653);
        CHECK(get(fixture.message, 24) == 0);
        CHECK(get(fixture.message, 32) == 0x80);
        CHECK(fixture.session->diagnostic() == nullptr);
    }
    fixture.message = control(0);
    CHECK(fixture.send() == IpcDisposition::handled);
    CHECK(get(fixture.message, 32) == 1);
    fixture.message = control(0);
    fixture.reject();
    fixture.message = control(2);
    fixture.reject();

    fixture.message = control(3);
    const auto before = fixture.message;
    CHECK(fixture.send(72) == IpcDisposition::unsupported);
    CHECK(fixture.message == before);
    CHECK(fixture.session->diagnostic() != nullptr);

    std::weak_ptr<PlayReportSession> lifetime = fixture.session;
    fixture.session.reset();
    CHECK(!lifetime.expired());
    CHECK(fixture.handles.close(fixture.handle));
    CHECK(lifetime.expired());
}

void test_save_and_malformed_buffers() {
    for (const auto domain : {false, true}) {
        for (const auto type : {4u, 6u}) {
            Fixture fixture;
            if (domain) {
                fixture.message = control(0);
                CHECK(fixture.send() == IpcDisposition::handled);
            }
            const auto room_before = fixture.memory.room;
            const auto report_before = fixture.memory.report;
            for (unsigned repetition = 0; repetition < 2; ++repetition) {
                fixture.message = report_request(10105, true, domain);
                put(fixture.message, 0, 0x00110000 | type);
                fixture.result(0, domain);
            }
            CHECK(fixture.memory.room == room_before);
            CHECK(fixture.memory.report == report_before);
            CHECK(fixture.handles.size() == 1);

            // An empty room descriptor is not dropped or confused with the report.
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 20, get(fixture.message, 20) & 0xffff);
            fixture.result(0x281, domain);
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 20, (get(fixture.message, 20) & 0xffff) | (33u << 16));
            fixture.result(0x281, domain);

            fixture.memory.room[4] = std::byte{'x'};
            fixture.message = report_request(10105, true, domain);
            fixture.result(0x281, domain);
            fixture.memory.room[4] = std::byte{0};

            fixture.message = report_request(10105, true, domain);
            put(fixture.message, (domain ? 64 : 48) + 16, 0);
            fixture.result(0x281, domain);
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, (domain ? 64 : 48) + 16, 0);
            put(fixture.message, (domain ? 64 : 48) + 28, 1);
            fixture.result(0, domain); // All 128 UID bits participate.

            for (const auto bad_name : {std::byte{0}, std::byte{'T'}, std::byte{'-'}}) {
                fixture.memory.room[0] = bad_name;
                fixture.message = report_request(10105, true, domain);
                fixture.result(0xa81, domain);
            }
            fixture.memory.room = room_before;

            fixture.memory.room[0] = std::byte{'T'};
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 28, 0);
            fixture.result(0xa81, domain); // Room validation precedes report validation.
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, (domain ? 64 : 48) + 16, 0);
            fixture.result(0x281, domain); // UID validation precedes logical room validation.
            fixture.memory.room = room_before;
            std::fill_n(fixture.memory.room.begin(), 31, std::byte{'a'});
            fixture.memory.room[31] = std::byte{0};
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 20, (get(fixture.message, 20) & 0xffff) | (32u << 16));
            fixture.result(0, domain);
            fixture.memory.room = room_before;
            fixture.memory.room[2] = std::byte{0};
            fixture.memory.room[3] = std::byte{0xff};
            fixture.message = report_request(10105, true, domain);
            fixture.result(0, domain); // Validation stops at the first NUL.
            fixture.memory.room = room_before;

            for (const auto length : {0u, 65537u}) {
                fixture.message = report_request(10105, true, domain);
                put(fixture.message, 28, length);
                fixture.result(0x1281, domain);
            }
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 36, get(fixture.message, 36) | (1u << 24));
            fixture.result(0x1281, domain); // Preserve A descriptor's upper size bits.

            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 32, 0x3000);
            fixture.reject(); // Unreadable host backing is explicit, never success.
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 24, 0x3000);
            fixture.reject();

            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 0, type | (1u << 16)); // Missing A descriptor.
            fixture.reject();
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 8, 0); // Missing PID.
            fixture.reject();
            fixture.message = report_request(10105, true, domain);
            put(fixture.message, 4, 0x800003ff); // Extent exceeds TLS message.
            fixture.reject();
            fixture.message = report_request(10105, true, domain);
            if (domain) put(fixture.message, 48, 0x00200001);
            else put(fixture.message, 4, 0x8000000c);
            fixture.reject(); // Truncated UID/PID input is not a service success.
            if (domain) {
                fixture.message = report_request(10105, true, true);
                put(fixture.message, 52, 2);
                fixture.reject();
            }
            for (const auto command : {10100u, 10104u, 10200u, 10400u, 99999u}) {
                fixture.message = report_request(command, true, domain);
                fixture.reject();
            }
        }
    }
}

using Bytes = std::vector<std::byte>;
Bytes bytes(std::initializer_list<unsigned> values) {
    Bytes result;
    for (const auto value : values) result.push_back(static_cast<std::byte>(value));
    return result;
}
Bytes root_value(Bytes value) {
    Bytes result = bytes({0x81, 0xa1, 'x'});
    result.insert(result.end(), value.begin(), value.end());
    return result;
}
void check_report(Bytes report, bool valid) {
    Fixture fixture;
    fixture.memory.report = std::move(report);
    fixture.message = report_request(10105, true);
    put(fixture.message, 28, static_cast<std::uint32_t>(fixture.memory.report.size()));
    const auto before = fixture.memory.report;
    fixture.result(valid ? 0 : 0x1281);
    CHECK(fixture.memory.report == before);
}

void test_report_contract() {
    for (const auto& value : {
        bytes({0}), bytes({0xff}), bytes({0xc2}), bytes({0xc3}),
        bytes({0xcc, 0xff}), bytes({0xd3, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}),
        bytes({0xa0}), bytes({0xa3, 0xe2, 0x82, 0xac}), bytes({0xa1, 0}),
        bytes({0xc4, 2, 0xff, 0}), bytes({0xca, 0x3f, 0x80, 0, 0}),
        bytes({0xcb, 0x3f, 0xf0, 0, 0, 0, 0, 0, 0}),
        bytes({0xd7, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
        bytes({0xd4, 65, 1}), bytes({0xc7, 10, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
        bytes({0x90}), bytes({0x92, 1, 0xff}),
        bytes({0x92, 0xca, 0, 0, 0, 0, 0xcb, 0, 0, 0, 0, 0, 0, 0, 0}),
        bytes({0x80}), bytes({0x81, 0xa1, 'a', 1}),
        bytes({0x92, 0x81, 0xa1, 'a', 1, 0x81, 0xa1, 'a', 0xff})})
        check_report(root_value(value), true);

    for (const auto& value : {
        bytes({0xc0}), bytes({0xc1}), bytes({0xa1}), bytes({0xa2, 0xc0, 0x80}),
        bytes({0xa3, 0xed, 0xa0, 0x80}), bytes({0xa4, 0xf4, 0x90, 0x80, 0x80}),
        bytes({0xca, 0x7f, 0x80, 0, 0}), bytes({0xcb, 0x7f, 0xf8, 0, 0, 0, 0, 0, 0}),
        bytes({0xd4, 0, 1}), bytes({0xd4, 1, 0}),
        bytes({0x91, 0x90}), bytes({0x92, 0xc2, 0}), bytes({0x91, 0xc0}),
        bytes({0x91, 0xc4, 0}), bytes({0x91, 0xd4, 65, 0}),
        bytes({0x81, 0xa1, 'a', 0xc4, 0}), bytes({0x81, 0xa1, 'a', 0xd4, 65, 0}),
        bytes({0x92, 0x81, 0xa1, 'a', 1, 0x81, 0xa1, 'b', 1}),
        bytes({0x82, 0xa1, 'a', 1, 0xa1, 'A', 2}),
        bytes({0xdd, 0xff, 0xff, 0xff, 0xff})})
        check_report(root_value(value), false);

    check_report(bytes({0x80}), false);
    check_report(bytes({0x90}), false);
    check_report(bytes({0x81, 0xa1, '1', 1}), false);
    check_report(bytes({0x81, 0xa0, 1}), false);
    check_report(bytes({0x82, 0xa1, 'a', 1, 0xa1, 'A', 2}), false);
    check_report(bytes({0x81, 0xa1, 'a', 1, 0}), false);

    for (const auto length : {63u, 64u}) {
        auto report = bytes({0x81, 0xd9, length});
        report.insert(report.end(), length, std::byte{'k'});
        report.push_back(std::byte{1});
        check_report(report, length == 63);
    }
    for (const auto count : {100u, 101u}) {
        auto report = bytes({0xde, 0, count});
        for (unsigned index = 0; index < count; ++index) {
            const auto key = "k" + std::to_string(index);
            report.push_back(static_cast<std::byte>(0xa0 + key.size()));
            for (const auto byte : key) report.push_back(static_cast<std::byte>(byte));
            report.push_back(std::byte{1});
        }
        check_report(report, count == 100);
    }
    auto nested = bytes({0x80});
    for (unsigned depth = 1; depth <= 4; ++depth) {
        check_report(root_value(nested), depth <= 3);
        nested = root_value(nested);
    }
    auto maximum = root_value(bytes({0xc6, 0, 0, 0xff, 0xf8}));
    maximum.resize(65536);
    check_report(maximum, true);
    maximum.push_back(std::byte{0});
    check_report(maximum, false);

    // Every proper prefix of a complete one-value report must fail, including
    // prefixes inside multi-byte lengths, extension bodies and container data.
    for (const auto& value : {
        bytes({0xdb, 0, 0, 0, 3, 'a', 'b', 'c'}),
        bytes({0xc5, 0, 3, 1, 2, 3}),
        bytes({0xc8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
        bytes({0xdf, 0, 0, 0, 1, 0xa1, 'a', 0xdc, 0, 2, 1, 2})}) {
        const auto complete = root_value(value);
        check_report(complete, true);
        for (std::size_t length = 0; length < complete.size(); ++length)
            check_report(Bytes(complete.begin(), complete.begin() + length), false);
    }
    check_report(root_value(bytes({0x92,
        0x81, 0xa1, 'a', 0x91, 1,
        0x81, 0xa1, 'a', 0x92, 1, 2})), false); // Nested array counts form part of the schema.
    check_report(root_value(bytes({0x92,
        0x81, 0xa1, 'a', 1,
        0x81, 0xa1, 'A', 1})), false); // Schema key matching is case-sensitive.
}

} // namespace

void test_save_with_opt_in_flag() {
    for (const auto domain : {false, true}) {
        Fixture fixture;
        if (domain) {
            fixture.message = control(0);
            CHECK(fixture.send() == IpcDisposition::handled);
        }
        for (const auto flag : {0u, 1u}) {
            fixture.message = report_request(10107, true, domain);
            put(fixture.message, (domain ? 64 : 48) + 16, flag);
            fixture.result(0, domain);
        }
        // The user ID follows the padded flag; a zero ID is rejected.
        fixture.message = report_request(10107, true, domain);
        put(fixture.message, (domain ? 64 : 48) + 24, 0);
        fixture.result(0x281, domain);
        // A 10105-sized payload is too short for 10107.
        fixture.message = report_request(10105, true, domain);
        put(fixture.message, (domain ? 64 : 48) + 8, 10107);
        fixture.reject();
    }
}

// SendReportWithUser (10500): the report comes in a copied transfer memory and
// a domain session gets an async context object back, which it may release.
IpcMessage send_request(Handle memory, std::uint32_t memory_size, std::uint32_t report_size) {
    IpcMessage message{};
    put(message, 0, 6 | (1u << 16)); // One X buffer.
    put(message, 4, 0x80000000u | 22);
    put(message, 8, 3); // Send PID and copy one handle.
    put(message, 12, 999);
    put(message, 20, memory);
    put(message, 24, (5u << 16) | (2u << 6) | (1u << 12));
    put(message, 28, static_cast<std::uint32_t>(room_address));
    put(message, 32, (0x38u << 16) | 1);
    put(message, 36, 1);
    put(message, 48, 0x49434653);
    put(message, 52, 1);
    put(message, 56, 10500);
    put(message, 68, memory_size);
    put(message, 72, report_size);
    put(message, 80, 1); // User ID.
    put(message, 96, 999);
    return message;
}

void test_send_report_with_user() {
    Fixture fixture;
    const auto memory = fixture.handles.insert(
        std::make_shared<TransferMemoryRecord>(AddressRange{report_address, 0x1000}, 0)).value();
    const auto size = static_cast<std::uint32_t>(fixture.memory.report.size());
    // A session that is not a domain cannot receive the async context.
    fixture.message = send_request(memory, 0x1000, size);
    put(fixture.message, 32, 0);
    fixture.reject();
    fixture.message = control(0);
    CHECK(fixture.send() == IpcDisposition::handled);
    for (const std::uint32_t object : {2u, 3u}) {
        fixture.message = send_request(memory, 0x1000, size);
        CHECK(fixture.send() == IpcDisposition::handled);
        CHECK(get(fixture.message, 16) == 1);
        CHECK(get(fixture.message, 32) == 0x4f434653 && get(fixture.message, 40) == 0);
        CHECK(get(fixture.message, 44) == 0x2749f93d && get(fixture.message, 48) == object);
        CHECK(fixture.session->diagnostic() == nullptr && fixture.memory.writes == 0);
    }
    // Validation results, in the save commands' order.
    fixture.message = send_request(memory, 0x1000, size);
    put(fixture.message, 80, 0);
    fixture.result(0x281, true);
    fixture.message = send_request(memory, 0x2000, size);
    fixture.result(0x281, true);
    fixture.message = send_request(memory, 0x1000, 0);
    fixture.result(0x1281, true);
    fixture.message = send_request(memory, 0x1000, 0x1001);
    fixture.result(0x1281, true);
    fixture.memory.report[0] = std::byte{0xc1}; // Not a MessagePack map.
    fixture.message = send_request(memory, 0x1000, size);
    fixture.result(0x1281, true);
    fixture.memory.report[0] = std::byte{0x81};
    fixture.memory.room[0] = std::byte{'T'};
    fixture.message = send_request(memory, 0x1000, size);
    fixture.result(0xa81, true);
    fixture.memory.room[0] = std::byte{'t'};
    // No copied handle, or one that is not a transfer memory.
    fixture.message = send_request(memory, 0x1000, size);
    put(fixture.message, 8, 1);
    fixture.reject();
    fixture.message = send_request(fixture.handle, 0x1000, size);
    fixture.reject();
    // An async context has completed: its event is signaled, Cancel and
    // GetResult succeed, and other commands stay unsupported.
    const auto context = [&](std::uint32_t object, std::uint32_t command) {
        fixture.message = IpcMessage{};
        put(fixture.message, 0, 4);
        put(fixture.message, 4, 12);
        put(fixture.message, 16, (16u << 16) | 1);
        put(fixture.message, 20, object);
        put(fixture.message, 32, 0x49434653);
        put(fixture.message, 36, 1);
        put(fixture.message, 40, command);
    };
    context(3, 0);
    CHECK(fixture.send() == IpcDisposition::handled);
    // One copied handle; the dispatcher moves the domain and CMIF headers past it.
    CHECK((get(fixture.message, 8) >> 1 & 15) == 1 && get(fixture.message, 48) == 0x4f434653);
    CHECK(get(fixture.message, 56) == 0);
    const auto event = fixture.handles.lookup_as<Event>(static_cast<Handle>(get(fixture.message, 20)));
    CHECK(event && event->is_signaled());
    for (const std::uint32_t command : {1u, 2u}) {
        context(3, command);
        CHECK(fixture.send() == IpcDisposition::handled);
        CHECK(get(fixture.message, 32) == 0x4f434653 && get(fixture.message, 40) == 0);
        CHECK(get(fixture.message, 8) == 0 || (get(fixture.message, 4) & 0x80000000u) == 0);
    }
    context(3, 3);
    fixture.reject();
    context(9, 2);
    fixture.reject();
    const std::pair<std::uint32_t, bool> closes[] = {{2, true}, {2, false}, {7, false}, {1, false}};
    // A released context takes no more commands.
    for (const auto& [object, open] : closes) {
        fixture.message = IpcMessage{};
        put(fixture.message, 0, 4);
        put(fixture.message, 4, 8);
        put(fixture.message, 16, 2);
        put(fixture.message, 20, object);
        if (open) {
            CHECK(fixture.send() == IpcDisposition::handled);
            CHECK(get(fixture.message, 16) == 0x4f434653);
            CHECK(fixture.session->diagnostic() == nullptr);
        } else {
            fixture.reject();
        }
    }
    context(2, 2);
    fixture.reject();
}

int main() {
    test_session_controls_and_lifetime();
    test_save_and_malformed_buffers();
    test_save_with_opt_in_flag();
    test_report_contract();
    test_send_report_with_user();
    std::printf("Play-report service failures: %u\n", failures);
    return failures ? 1 : 0;
}
