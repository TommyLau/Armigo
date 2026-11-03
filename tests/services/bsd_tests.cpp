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

#include <cstdio>

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
} // namespace

int main() {
    test_initialization();
    if (failures) return 1;
    std::puts("socket checks passed");
    return 0;
}
