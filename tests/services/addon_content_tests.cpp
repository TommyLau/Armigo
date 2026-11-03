// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/addon_content.h>

#include <array>
#include <span>
#include <memory>
#include <cstring>
#include <cstdio>

namespace {
using namespace armigo::horizon;
using armigo::services::aoc::AddOnContentSession;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)
void put(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
IpcMessage control(unsigned type, unsigned command) {
    IpcMessage bytes{};
    put(bytes, 0, type); put(bytes, 4, 8); put(bytes, 16, 0x49434653);
    put(bytes, 20, 1); put(bytes, 24, command); put(bytes, 28, 0xaabbccdd);
    return bytes;
}
IpcMessage method(std::uint32_t command) {
    IpcMessage bytes{};
    put(bytes, 0, 6);
    put(bytes, 4, 12);
    put(bytes, 16, 0x00100001);
    put(bytes, 20, 1);
    put(bytes, 32, 0x49434653);
    put(bytes, 36, 1);
    put(bytes, 40, command);
    return bytes;
}
// The layout the SDK sends for a method with a process id: a special header with the pid
// moves the domain header to byte 32 and the CMIF to byte 48.
IpcMessage pid_method(std::uint32_t command) {
    IpcMessage bytes{};
    put(bytes, 0, 6);
    put(bytes, 4, 0x80000010);
    put(bytes, 8, 1);
    put(bytes, 12, 37);
    put(bytes, 32, 0x00180001);
    put(bytes, 36, 1);
    put(bytes, 48, 0x49434653);
    put(bytes, 52, 1);
    put(bytes, 56, command);
    return bytes;
}

void test_methods() {
    HandleTable table(2);
    auto session = std::make_shared<AddOnContentSession>(37, table);
    // A buffered ListAddOnContent (3) reports an empty installed list: the count output is
    // zero and the caller's buffer stays untouched.
    std::array<std::byte, 16> buffer{};
    GuestBuffers buffers;
    buffers.add({0, static_cast<std::uint32_t>(buffer.size()), false, true});
    auto reply = session->request_with_buffers(method(3), 37, buffers);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(reply.size == 52);
    CHECK(reply.bytes[32] == std::byte{0x53} && reply.bytes[35] == std::byte{0x4f});
    CHECK(reply.bytes[36] == std::byte{1});
    CHECK(reply.bytes[48] == std::byte{0} && reply.bytes[49] == std::byte{0});
    CHECK(buffer[0] == std::byte{0});
    // A non-domain type-6 request carries its descriptors before the CMIF; the reply has no
    // descriptors, so its CMIF header is at byte16 and the count output at byte32.
    {
        IpcMessage raw{};
        put(raw, 0, 0x01000006);
        put(raw, 4, 0x8000000c);
        put(raw, 8, 1);
        put(raw, 12, 0);
        put(raw, 16, 0);
        put(raw, 20, 8);
        put(raw, 24, 0);
        put(raw, 28, 0);
        put(raw, 32, 0x49434653);
        put(raw, 36, 1);
        put(raw, 40, 3);
        reply = session->request_with_buffers(raw, 37, buffers);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(reply.size == 36 && reply.bytes[4] == std::byte{7});
        CHECK(reply.bytes[16] == std::byte{0x53} && reply.bytes[19] == std::byte{0x4f});
        CHECK(reply.bytes[32] == std::byte{0} && reply.bytes[33] == std::byte{0});
    }
    // CountAddOnContent (2) reports zero without a buffer.
    reply = session->request(method(2), 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 52);
    CHECK(reply.bytes[48] == std::byte{0});
    // Another method stays explicit.
    reply = session->request(method(7), 37);
    CHECK(reply.disposition == IpcDisposition::unsupported);
}

struct Memory {
    std::array<std::byte, 64> bytes{};
    static bool read(void* context, std::uint64_t address, std::span<std::byte> value) noexcept {
        auto& self = *static_cast<Memory*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        std::memcpy(value.data(), self.bytes.data() + address, value.size());
        return true;
    }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> value) noexcept {
        auto& self = *static_cast<Memory*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        std::memcpy(self.bytes.data() + address, value.data(), value.size());
        return true;
    }
    std::uint32_t word(std::size_t offset) const {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof value);
        return value;
    }
};
std::uint32_t word(const IpcMessage& bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}

// A synthetic catalog: the session counts and lists its items by index, reports
// the base id, prepares an installed item and accepts its mount notice.
void test_catalog() {
    HandleTable table(2);
    auto catalog = std::make_shared<armigo::services::AddOnContentCatalog>();
    catalog->base_id = 0x0100000000011000ull;
    catalog->items.push_back({catalog->base_id + 1, nullptr});
    catalog->items.push_back({catalog->base_id + 3, nullptr});
    auto session = std::make_shared<AddOnContentSession>(37, table, catalog);
    auto reply = session->request(method(2), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 48) == 2);
    const auto list = [&](std::uint32_t offset, std::uint32_t count, Memory& memory, std::uint32_t capacity = 16) {
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0, capacity, false, true});
        auto message = method(3);
        put(message, 48, offset);
        put(message, 52, count);
        return session->request_with_buffers(message, 37, buffers);
    };
    Memory all;
    reply = list(0, 4, all);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 48) == 2);
    CHECK(all.word(0) == 1 && all.word(4) == 3 && all.word(8) == 0);
    Memory tail;
    reply = list(1, 4, tail);
    CHECK(word(reply.bytes, 48) == 1 && tail.word(0) == 3 && tail.word(4) == 0);
    Memory first;
    reply = list(0, 1, first);
    CHECK(word(reply.bytes, 48) == 1 && first.word(0) == 1 && first.word(4) == 0);
    Memory past;
    reply = list(5, 4, past);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 48) == 0 && past.word(0) == 0);
    // A buffer with room for one index (and a partial word) bounds the list before the count does.
    Memory narrow;
    reply = list(0, 4, narrow, 7);
    CHECK(word(reply.bytes, 48) == 1 && narrow.word(0) == 1 && narrow.word(4) == 0);
    // The pid-carrying layout reads its arguments after the CMIF at byte 48 and replies in the domain layout.
    {
        Memory pid;
        GuestBuffers buffers(Memory::read, Memory::write, &pid);
        buffers.add({0, 16, false, true});
        auto message = pid_method(3);
        put(message, 64, 1);
        put(message, 68, 4);
        reply = session->request_with_buffers(message, 37, buffers);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 52);
        CHECK(word(reply.bytes, 32) == 0x4f434653 && word(reply.bytes, 48) == 1 && pid.word(0) == 3);
        message = pid_method(7);
        put(message, 64, 1);
        CHECK(session->request(message, 37).disposition == IpcDisposition::handled);
        put(message, 64, 2);
        CHECK(session->request(message, 37).disposition == IpcDisposition::unsupported);
    }
    reply = session->request(method(5), 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 56);
    CHECK(word(reply.bytes, 48) == 0x00011000 && word(reply.bytes, 52) == 0x01000000);
    const auto with_word = [](std::uint32_t command, std::uint64_t value) {
        auto message = method(command);
        put(message, 48, static_cast<std::uint32_t>(value));
        put(message, 52, static_cast<std::uint32_t>(value >> 32));
        return message;
    };
    CHECK(session->request(with_word(7, 3), 37).disposition == IpcDisposition::handled);
    CHECK(session->request(with_word(7, 2), 37).disposition == IpcDisposition::unsupported);
    CHECK(session->request(with_word(7, 0), 37).disposition == IpcDisposition::unsupported);
    CHECK(session->request(with_word(11, catalog->base_id + 1), 37).disposition == IpcDisposition::handled);
    CHECK(session->request(with_word(11, catalog->base_id + 2), 37).disposition == IpcDisposition::unsupported);
    // Without a catalog there is no base id and nothing to mount.
    auto bare = std::make_shared<AddOnContentSession>(37, table);
    CHECK(bare->request(method(5), 37).disposition == IpcDisposition::unsupported);
    CHECK(bare->request(with_word(11, catalog->base_id + 1), 37).disposition == IpcDisposition::unsupported);
    // The mount status check takes no input and succeeds with or without items.
    for (const auto& checked : {session, bare}) {
        const auto status = checked->request(method(50), 37);
        CHECK(status.disposition == IpcDisposition::handled && status.size == 48);
        CHECK(word(status.bytes, 32) == 0x4f434653 && word(status.bytes, 40) == 0);
    }
}

void test_controls() {
    HandleTable table(2);
    auto session = std::make_shared<AddOnContentSession>(37, table);
    const auto handle = table.insert(session).value();
    IpcDispatcher dispatcher(table, nullptr);
    IpcMessage message{};
    IpcCallContext caller{37, message};
    QueryRegisters registers{};
    for (unsigned type : {5, 7}) {
        const auto reply = session->request(control(type, 3), 37);
        IpcMessage expected{};
        put(expected, 4, 9); put(expected, 16, 0x4f434653); put(expected, 28, 0xec6be3ff);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 44 && reply.bytes == expected);
        CHECK(table.size() == 1 && session->domain_object_count() == 0);
    }
    message = control(7, 0); registers[0] = handle;
    CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::handled && registers[0] == 0);
    CHECK(message[32] == std::byte{1} && session->domain_object_count() == 1 && table.size() == 1);
    const auto reject = [&](IpcMessage input) {
        message = input; registers.fill(0x1234); registers[0] = handle;
        const auto saved = registers;
        CHECK(dispatcher.dispatch(0x21, registers, caller) == IpcDisposition::unsupported);
        CHECK(message == input && registers == saved && table.size() == 1);
        CHECK(session->domain_object_count() == 1 && session->diagnostic() != nullptr);
    };
    reject(control(5, 0)); reject(control(7, 4));
    auto input = control(7, 3); put(input, 4, 7); reject(input);
    input = control(7, 3); put(input, 4, 0x80000008); reject(input);
    // A type 6 packet without a CMIF header is not a domain method.
    input.fill(std::byte{0});
    put(input, 0, 6);
    reject(input);
    caller.process_id = 38; reject(control(7, 3));
    caller.process_id = 37;
    std::weak_ptr<AddOnContentSession> weak = session;
    session.reset(); CHECK(!weak.expired()); CHECK(table.close(handle) && weak.expired());
}
}  // namespace
int main() { test_controls(); test_methods(); test_catalog(); if (failures) return 1; std::puts("add-on content checks passed"); return 0; }
