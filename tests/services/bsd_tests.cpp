// ============================================================================
// Armigo Tests - Horizon Service Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/bsd.h>
#include <armigo/horizon/event.h>
#include <armigo/horizon/startup_queries.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <span>

namespace {
using namespace armigo::horizon;
using armigo::services::bsd::BsdSession;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)

void put(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
std::uint32_t word(const IpcMessage& bytes, unsigned offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}
IpcMessage control(std::uint32_t command) {
    IpcMessage bytes{};
    put(bytes, 0, 5);
    put(bytes, 4, 8);
    put(bytes, 16, 0x49434653);
    put(bytes, 24, command);
    return bytes;
}
// RegisterClient on domain object 1: special header with the PID and one copied
// handle, domain header at 32, CMIF at 48, config at 64, size at 104.
IpcMessage register_client(Handle memory) {
    IpcMessage bytes{};
    bytes.fill(std::byte{0xa5});
    put(bytes, 0, 6);
    put(bytes, 4, 0x80000014);
    put(bytes, 8, 1 | (1 << 1));
    put(bytes, 12, 0);
    put(bytes, 16, 0);
    put(bytes, 20, memory);
    put(bytes, 32, 0x00380001);
    put(bytes, 36, 1);
    put(bytes, 48, 0x49434653);
    put(bytes, 52, 1);
    put(bytes, 56, 0);
    put(bytes, 60, 0);
    for (unsigned i = 0; i < 32; i += 4) put(bytes, 64 + i, 0x01020304 + i);
    put(bytes, 104, 0x5e0000);
    put(bytes, 108, 0);
    return bytes;
}
// StartMonitoring on a plain session: PID special header, CMIF at 32.
IpcMessage start_monitoring() {
    IpcMessage bytes{};
    put(bytes, 0, 6);
    put(bytes, 4, 0x8000000a);
    put(bytes, 8, 1);
    put(bytes, 32, 0x49434653);
    put(bytes, 36, 1);
    put(bytes, 40, 1);
    return bytes;
}

void test_initialization() {
    HandleTable handles(16);
    BsdSession client(37, handles);
    auto reply = client.request(control(3), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == 0);
    reply = client.request(control(0), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == 1);
    CHECK(client.request(control(0), 37).disposition == IpcDisposition::unsupported);
    reply = client.request(control(2), 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1);
    const auto clone = std::dynamic_pointer_cast<BsdSession>(reply.transferred[0]);
    CHECK(clone != nullptr);

    // Only transfer memory registers.
    const auto event = handles.insert(std::make_shared<Event>(false));
    CHECK(event.has_value() && clone->request(register_client(*event), 37).disposition == IpcDisposition::unsupported);
    // The client closes its handle to the work memory after registering; the
    // service keeps the object.
    auto memory = std::make_shared<TransferMemoryRecord>(AddressRange{0x10000, 0x5e0000}, 0);
    std::weak_ptr<TransferMemoryRecord> weak = memory;
    const auto handle = handles.insert(memory);
    memory.reset();
    CHECK(handle.has_value());
    reply = clone->request(register_client(*handle), 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 52 && word(reply.bytes, 4) == 11);
    CHECK(word(reply.bytes, 32) == 0x4f434653 && word(reply.bytes, 40) == 0 && word(reply.bytes, 48) == 0);
    CHECK(client.registered() && clone->registered());
    CHECK(handles.close(*handle));
    CHECK(!weak.expired() && client.work_memory() != nullptr);
    CHECK(std::to_integer<unsigned>(client.configuration()[0]) == 0x04);
    CHECK(client.request(register_client(*handle), 37).disposition == IpcDisposition::unsupported);

    // Closing the domain root at finalization is a plain success.
    IpcMessage close{};
    put(close, 0, 4);
    put(close, 4, 0x10);
    put(close, 16, 0x00000002);
    put(close, 20, 1);
    reply = client.request(close, 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 32);

    BsdSession monitor(37, handles);
    reply = monitor.request(start_monitoring(), 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 40 && word(reply.bytes, 24) == 0);
    CHECK(monitor.request(start_monitoring(), 38).disposition == IpcDisposition::unsupported);
}
// A domain request on object 1 (domain header at 32, CMIF at 48, inputs at 64);
// `pid` adds the PID special header.
IpcMessage domain_request(std::uint32_t command, std::initializer_list<std::uint32_t> inputs, bool pid = false) {
    IpcMessage bytes{};
    put(bytes, 0, 6);
    put(bytes, 4, pid ? 0x80000014 : 0x14);
    if (pid) put(bytes, 8, 1);
    put(bytes, 32, 0x00200001);
    put(bytes, 36, 1);
    put(bytes, 48, 0x49434653);
    put(bytes, 52, 1);
    put(bytes, 56, command);
    unsigned offset = 64;
    for (const auto value : inputs) put(bytes, offset, value), offset += 4;
    return bytes;
}

struct Memory {
    std::array<std::byte, 0x100> bytes{};
    static bool read(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<Memory*>(context);
        if (address + out.size() > self.bytes.size()) return false;
        std::memcpy(out.data(), self.bytes.data() + address, out.size());
        return true;
    }
    static bool write(void* context, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& self = *static_cast<Memory*>(context);
        if (address + in.size() > self.bytes.size()) return false;
        std::memcpy(self.bytes.data() + address, in.data(), in.size());
        return true;
    }
    std::uint64_t get64(std::size_t at) const { std::uint64_t v; std::memcpy(&v, bytes.data() + at, 8); return v; }
    void put64(std::size_t at, std::uint64_t v) { std::memcpy(bytes.data() + at, &v, 8); }
};

// Switch 2 registration and the event descriptors gRPC wakes its pollers with.
void test_event_descriptors() {
    HandleTable handles(16);
    BsdSession client(37, handles);
    CHECK(client.request(control(0), 37).disposition == IpcDisposition::handled);
    CHECK(client.request(domain_request(36, {1, 2, 3, 4}), 37).disposition == IpcDisposition::unsupported);
    auto reply = client.request(domain_request(36, {1, 2, 3, 4}, true), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 48) == 0 && client.registered());

    // flags 6 never wait; flags 1 reads one at a time.
    reply = client.request(domain_request(31, {6, 0, 0, 0}), 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 48) == 0 && word(reply.bytes, 52) == 0);
    reply = client.request(domain_request(31, {1 | 6, 0, 2, 0}), 37);
    CHECK(word(reply.bytes, 48) == 1);

    Memory memory;
    const auto with = [&](std::initializer_list<GuestBuffer> list) {
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        for (const auto& buffer : list) buffers.add(buffer);
        return buffers;
    };
    const auto read_fd = [&](std::uint32_t fd) {
        return client.request_with_buffers(domain_request(25, {fd}), 37, with({{0x80, 8, false, true}}));
    };
    const auto write_fd = [&](std::uint32_t fd, std::uint64_t value) {
        memory.put64(0x40, value);
        return client.request_with_buffers(domain_request(24, {fd}), 37, with({{0x40, 8, true, false}}));
    };
    const auto poll = [&](std::initializer_list<std::pair<std::int32_t, std::int16_t>> fds, std::int32_t timeout) {
        unsigned at = 0;
        for (const auto& [fd, events] : fds) {
            std::uint64_t entry = static_cast<std::uint32_t>(fd) | std::uint64_t(static_cast<std::uint16_t>(events)) << 32;
            memory.put64(at, entry);
            at += 8;
        }
        return client.request_with_buffers(
            domain_request(6, {static_cast<std::uint32_t>(fds.size()), static_cast<std::uint32_t>(timeout)}), 37,
            with({{0, at, true, false}, {0xa0, at, false, true}}));
    };
    const auto revents = [&](unsigned index) { return static_cast<std::uint16_t>(memory.get64(0xa0 + 8 * index) >> 48); };

    // Empty and non-blocking: EAGAIN; poll reports only POLLOUT.
    reply = read_fd(0);
    CHECK(static_cast<std::int32_t>(word(reply.bytes, 48)) == -1 && word(reply.bytes, 52) == 35);
    reply = poll({{0, 0x1 | 0x4}}, 0);
    CHECK(word(reply.bytes, 48) == 1 && revents(0) == 0x4);
    reply = poll({{0, 0x1}}, 0);
    CHECK(word(reply.bytes, 48) == 0 && revents(0) == 0);
    // The SDK's poll sends an empty pointer descriptor and the send and receive
    // descriptors before the raw data, which moves the domain header to 48 and
    // the CMIF header to 64.
    {
        IpcMessage bytes{};
        put(bytes, 0, 0x01110006);
        put(bytes, 4, 0x00000c0f);
        put(bytes, 16, 8);
        put(bytes, 20, 0);
        put(bytes, 28, 8);
        put(bytes, 32, 0xa0);
        put(bytes, 48, 0x00180001);
        put(bytes, 52, 1);
        put(bytes, 64, 0x49434653);
        put(bytes, 68, 1);
        put(bytes, 72, 6);
        put(bytes, 80, 1);
        put(bytes, 84, 0);
        memory.put64(0, std::uint64_t{0x4} << 32);
        reply = client.request_with_buffers(bytes, 37, with({{0, 8, true, false}, {0xa0, 8, false, true}}));
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 48) == 1 && revents(0) == 0x4);
    }
    // A write wakes a waiting poll and the read takes the counter.
    reply = write_fd(0, 5);
    CHECK(word(reply.bytes, 48) == 8);
    write_fd(0, 2);
    reply = poll({{0, 0x1}, {-1, 0x1}, {9, 0x1}}, -1);
    CHECK(word(reply.bytes, 48) == 2 && revents(0) == 0x1 && revents(1) == 0 && revents(2) == 0x20);
    reply = read_fd(0);
    CHECK(word(reply.bytes, 48) == 8 && memory.get64(0x80) == 7);
    // Semaphore descriptors take one at a time.
    reply = read_fd(1);
    CHECK(word(reply.bytes, 48) == 8 && memory.get64(0x80) == 1);
    reply = read_fd(1);
    CHECK(word(reply.bytes, 48) == 8 && memory.get64(0x80) == 1);
    reply = read_fd(1);
    CHECK(word(reply.bytes, 52) == 35);
    // An endless poll on nothing ready returns 0 after the offline limit.
    const auto start = std::chrono::steady_clock::now();
    reply = poll({{0, 0x1}}, -1);
    const auto waited = std::chrono::steady_clock::now() - start;
    CHECK(word(reply.bytes, 48) == 0 && waited >= std::chrono::milliseconds(90) && waited < std::chrono::seconds(2));
    // Close releases the descriptor; closing or reading it again fails.
    reply = client.request(domain_request(26, {0}), 37);
    CHECK(word(reply.bytes, 48) == 0);
    reply = client.request(domain_request(26, {0}), 37);
    CHECK(static_cast<std::int32_t>(word(reply.bytes, 48)) == -1 && word(reply.bytes, 52) == 9);
    CHECK(word(read_fd(0).bytes, 52) == 9);
    // With no network, sockets fail as with the network down.
    reply = client.request(domain_request(2, {2, 1, 0}), 37);
    CHECK(static_cast<std::int32_t>(word(reply.bytes, 48)) == -1 && word(reply.bytes, 52) == 50);
    // A blocking descriptor never waits here: an empty read stops explicitly.
    reply = client.request(domain_request(31, {0, 0, 0, 0}), 37);
    CHECK(word(reply.bytes, 48) == 0);
    CHECK(read_fd(0).disposition == IpcDisposition::unsupported && client.diagnostic());
}
} // namespace

int main() {
    test_initialization();
    test_event_descriptors();
    if (failures) return 1;
    std::puts("socket checks passed");
    return 0;
}
