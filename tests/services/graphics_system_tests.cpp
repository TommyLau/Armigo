// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/graphics_system.h>
#include <armigo/horizon/startup_queries.h>

#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <vector>

namespace {
using namespace armigo::horizon;
using armigo::services::gsv::DisplaySession;
using armigo::services::gsv::GpuSession;
using armigo::services::gsv::GraphicsSystemDependencies;
using armigo::services::gsv::GraphicsSystemSession;
using armigo::services::gsv::SyncpointSession;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (false)

constexpr std::uint64_t pid = 71;

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
std::uint32_t get(const IpcReply& reply, std::size_t offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i)
        value |= std::uint32_t{std::to_integer<unsigned char>(reply.bytes[offset + i])} << (8 * i);
    return value;
}

// A TIPC request: the type is 16 plus the command, then the raw data words.
// Copied handles add a special header before the data.
IpcMessage tipc(std::uint32_t command, std::initializer_list<std::uint32_t> data,
                std::initializer_list<Handle> copied = {}) {
    IpcMessage message{};
    put(message, 0, 16 + command);
    put(message, 4, static_cast<std::uint32_t>(data.size()) | (copied.size() ? 0x80000000u : 0));
    std::size_t offset = 8;
    if (copied.size()) {
        put(message, 8, static_cast<std::uint32_t>(copied.size()) << 1);
        offset = 12;
        for (const auto handle : copied) { put(message, offset, handle); offset += 4; }
    }
    for (const auto word : data) { put(message, offset, word); offset += 4; }
    return message;
}

bool handled(const IpcReply& reply, std::uint32_t command) {
    return reply.disposition == IpcDisposition::handled && get(reply, 0) == 16 + command;
}

// A handle reply carries the handle at word 3 and the result at word 4.
template <class T>
std::shared_ptr<T> handed_out(HandleTable& handles, const IpcReply& reply, std::uint32_t command) {
    if (!handled(reply, command) || get(reply, 4) != 0x80000001 || get(reply, 16) != 0) return nullptr;
    return handles.lookup_as<T>(static_cast<Handle>(get(reply, 12)));
}

class OtherSession final : public ClientSession {
public:
    IpcReply request(const IpcMessage&, std::uint64_t) override { return {IpcDisposition::handled}; }
};

// Guest memory for transfer memories: a flat buffer at a fixed address.
struct Memory {
    static constexpr std::uint64_t base = 0x100000;
    std::vector<std::byte> bytes = std::vector<std::byte>(0x4000);
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& self = *static_cast<Memory*>(raw);
        if (address < base || address - base + out.size() > self.bytes.size()) return false;
        std::copy_n(self.bytes.begin() + static_cast<std::ptrdiff_t>(address - base), out.size(), out.begin());
        return true;
    }
    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> in) noexcept {
        auto& self = *static_cast<Memory*>(raw);
        if (address < base || address - base + in.size() > self.bytes.size()) return false;
        std::copy(in.begin(), in.end(), self.bytes.begin() + static_cast<std::ptrdiff_t>(address - base));
        return true;
    }
    std::uint32_t word(std::uint64_t address) const {
        std::uint32_t value = 0;
        for (unsigned i = 0; i < 4; ++i)
            value |= std::uint32_t{std::to_integer<unsigned char>(bytes[address - base + i])} << (8 * i);
        return value;
    }
};

void test_session_basics() {
    HandleTable handles{16};
    GraphicsSystemSession session(pid, handles);
    auto reply = session.request(tipc(0, {0x11223344, 0x55667788}), pid);
    CHECK(handled(reply, 0) && get(reply, 4) == 3 && get(reply, 8) == 0);
    CHECK(get(reply, 12) == 0x11223344 && get(reply, 16) == 0x55667788);
    // A provisional handler succeeds without data.
    reply = session.request(tipc(4, {1, 0, 7}), pid);
    CHECK(handled(reply, 4) && get(reply, 4) == 1 && get(reply, 8) == 0 && reply.size == 12);
    // Zero replies carry 32 zero words.
    reply = session.request(tipc(113, {}), pid);
    CHECK(handled(reply, 113) && get(reply, 4) == 33 && reply.size == 12 + 128);
    reply = session.request(tipc(115, {0xf0, 0x1}), pid);
    CHECK(handled(reply, 115) && get(reply, 12) == 0xf0 && get(reply, 16) == 1);
    reply = session.request(tipc(126, {}), pid);
    CHECK(handled(reply, 126) && get(reply, 4) == 2 && get(reply, 12) == 0);
    CHECK(handled(session.request(tipc(1501, {}), pid), 1501));
    // The client handle registers once.
    CHECK(handled(session.request(tipc(11, {}, {current_process_handle}), pid), 11));
    CHECK(session.request(tipc(11, {}, {current_process_handle}), pid).disposition == IpcDisposition::unsupported);
    // Unknown commands, other processes and non-TIPC messages stay unsupported.
    CHECK(session.request(tipc(9999, {}), pid).disposition == IpcDisposition::unsupported);
    CHECK(session.diagnostic() != nullptr);
    CHECK(session.request(tipc(0, {1, 2}), pid + 1).disposition == IpcDisposition::unsupported);
    IpcMessage cmif{};
    put(cmif, 0, 4);
    CHECK(session.request(cmif, pid).disposition == IpcDisposition::unsupported);
    // Without a driver factory, the driver session stays unsupported.
    CHECK(session.request(tipc(1402, {}), pid).disposition == IpcDisposition::unsupported);
    CHECK(session.close_server().disposition == IpcDisposition::handled);
    CHECK(session.request(tipc(0, {1, 2}), pid).transport_result == session_closed_result);
}

void test_driver_session() {
    HandleTable handles{16};
    GraphicsSystemDependencies dependencies;
    auto other = std::make_shared<OtherSession>();
    dependencies.open_driver = [&] { return other; };
    GraphicsSystemSession session(pid, handles, dependencies);
    CHECK(handed_out<OtherSession>(handles, session.request(tipc(1402, {}), pid), 1402) == other);
}

void test_syncpoint_session() {
    HandleTable handles{16};
    GraphicsSystemSession session(pid, handles);
    const auto syncpoint = handed_out<SyncpointSession>(handles, session.request(tipc(202, {}), pid), 202);
    CHECK(syncpoint != nullptr);
    if (!syncpoint) return;
    auto reply = syncpoint->request(tipc(1, {}), pid);
    CHECK(handled(reply, 1) && get(reply, 12) == 32);
    const auto event = handed_out<Event>(handles, syncpoint->request(tipc(22, {3}), pid), 22);
    CHECK(event != nullptr && !event->is_signaled());
    if (!event) return;
    // A value the counter already reached signals at once; a later one waits.
    CHECK(handled(syncpoint->request(tipc(201, {3, ~0u, 0, 0, 0}), pid), 201) && event->is_signaled());
    CHECK(handled(syncpoint->request(tipc(201, {3, ~0u, 5, 0, 1}), pid), 201) && !event->is_signaled());
    // A cancel always signals.
    CHECK(handled(syncpoint->request(tipc(202, {3}), pid), 202) && event->is_signaled());
    reply = syncpoint->request(tipc(103, {~0u}), pid);
    CHECK(handled(reply, 103) && get(reply, 4) == 4 && get(reply, 12) == 0);
    CHECK(syncpoint->request(tipc(7, {}), pid).disposition == IpcDisposition::unsupported);
}

void test_syncpoint_memory() {
    HandleTable handles{16};
    Memory memory;
    const GuestBuffers buffers(Memory::read, Memory::write, &memory);
    GraphicsSystemSession session(pid, handles);
    const auto writable = handles.insert(std::make_shared<TransferMemoryRecord>(AddressRange{Memory::base, 0x1000}, 3));
    const auto readable = handles.insert(std::make_shared<TransferMemoryRecord>(AddressRange{Memory::base + 0x1000, 0x2000}, 1));
    CHECK(writable && readable);
    if (!writable || !readable) return;
    // A read-only memory below 0x2000 bytes is refused.
    const auto small = handles.insert(std::make_shared<TransferMemoryRecord>(AddressRange{Memory::base + 0x3000, 0x1000}, 1));
    CHECK(session.request_with_buffers(tipc(204, {1, 0x1ff, 0x1000, 0}, {*writable, *small}), pid, buffers).disposition ==
          IpcDisposition::unsupported);
    const auto reply = session.request_with_buffers(tipc(204, {1, 0x1ff, 0x1000, 0}, {*writable, *readable}), pid, buffers);
    CHECK(handled(reply, 204) && get(reply, 4) == 0x80000001 && get(reply, 8) == 4 && get(reply, 20) == 0);
    CHECK(handles.lookup_as<Event>(static_cast<Handle>(get(reply, 12))) != nullptr);
    CHECK(handles.lookup_as<Event>(static_cast<Handle>(get(reply, 16))) != nullptr);
    const auto header = Memory::base + 0x1000;
    // Version 1, base id 0, the count capped at the value page, then the offsets.
    CHECK(memory.word(header) == 1 && memory.word(header + 4) == 0 && memory.word(header + 8) == 0x1f8);
    CHECK(memory.word(header + 12) == 0x40 && memory.word(header + 16) == 0 && memory.word(header + 20) == 0x1040);
    CHECK(memory.word(header + 24) == 0x100 && memory.word(header + 28) == 0x1100);
}

void test_gpu_session() {
    HandleTable handles{16};
    GraphicsSystemSession session(pid, handles);
    const auto gpu = handed_out<GpuSession>(handles, session.request(tipc(497, {4, 0}), pid), 497);
    CHECK(gpu != nullptr);
    if (!gpu) return;
    // The sub-session is the same object.
    CHECK(handed_out<GpuSession>(handles, gpu->request(tipc(22, {}), pid), 22) == gpu);
    auto reply = gpu->request(tipc(11, {}), pid);
    CHECK(handled(reply, 11) && get(reply, 12) == 0 && get(reply, 16) == 31);
    const auto event = handed_out<Event>(handles, gpu->request(tipc(12, {5}), pid), 12);
    CHECK(event != nullptr && !event->is_signaled());
    CHECK(handled(gpu->request(tipc(21, {5}), pid), 21) && event && event->is_signaled());
    reply = gpu->request(tipc(41, {1, 0, 0, 0, 0, 0}), pid);
    CHECK(handled(reply, 41) && get(reply, 12) == 1);
    reply = gpu->request(tipc(41, {1, 0, 0, 0, 0, 0}), pid);
    CHECK(handled(reply, 41) && get(reply, 12) == 2);
    CHECK(handled(gpu->request(tipc(42, {1}), pid), 42));
    CHECK(gpu->request(tipc(42, {1}), pid).disposition == IpcDisposition::unsupported);
    reply = gpu->request(tipc(203, {0}), pid);
    CHECK(handled(reply, 203) && get(reply, 12) == 1 && get(reply, 20) == 0x8000000);
    reply = gpu->request(tipc(202, {0}), pid);
    CHECK(handled(reply, 202) && get(reply, 12) == ~0u && get(reply, 20) == 0);
}

void test_display_session() {
    HandleTable handles{16};
    GraphicsSystemSession session(pid, handles);
    const auto display = handed_out<DisplaySession>(handles, session.request(tipc(305, {0x40000, 3}), pid), 305);
    CHECK(display != nullptr);
    if (!display) return;
    auto reply = display->request(tipc(5, {}), pid);
    CHECK(handled(reply, 5) && get(reply, 12) == 7);
    const auto slot = handed_out<Event>(handles, display->request(tipc(31, {0}), pid), 31);
    CHECK(slot != nullptr && !slot->is_signaled());
    // The duplicate returns the same slot event.
    CHECK(handed_out<Event>(handles, display->request(tipc(32, {0}, {Handle{0x1234}}), pid), 32) == slot);
    CHECK(display->request(tipc(31, {8}), pid).disposition == IpcDisposition::unsupported);
    // A change notification fires once.
    CHECK(handled(display->request(tipc(263, {0x61666564, 0x746c75, 0}), pid), 263) && slot && slot->is_signaled());
    CHECK(handled(display->request(tipc(262, {0x61666564, 0x746c75, 0}), pid), 262));
    reply = display->request(tipc(106, {0x61666564, 0x746c75}), pid);
    CHECK(handled(reply, 106) && get(reply, 8) == 0 && get(reply, 12) == 4);
    reply = display->request(tipc(106, {1, 2}), pid);
    CHECK(handled(reply, 106) && get(reply, 8) == 0xe72);
    reply = display->request(tipc(212, {1, 0}), pid);
    CHECK(handled(reply, 212) && get(reply, 12) == 0x7fff);
    CHECK(handled(display->request(tipc(201, {1, 0, 2, 3}), pid), 201));
    CHECK(display->request(tipc(229, {}), pid).disposition == IpcDisposition::unsupported);
    CHECK(display->close_server().disposition == IpcDisposition::handled);
}
} // namespace

int main() {
    test_session_basics();
    test_driver_session();
    test_syncpoint_session();
    test_syncpoint_memory();
    test_gpu_session();
    test_display_session();
    if (failures) std::fprintf(stderr, "%u failures\n", failures);
    return failures ? 1 : 0;
}
