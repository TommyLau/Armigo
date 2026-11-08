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

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <vector>

#include "../host1x/flat_av1.h"

namespace {
using namespace armigo::horizon;
using armigo::services::gsv::DisplaySession;
using armigo::services::gsv::GpuSession;
using armigo::services::gsv::GraphicsSystemDependencies;
using armigo::services::gsv::GraphicsSystemSession;
using armigo::services::gsv::SyncpointSession;
using armigo::services::gsv::VideoSession;
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
    reply = session.request(tipc(903, {}), pid);
    CHECK(handled(reply, 903) && get(reply, 8) == 0 && reply.size == 12);
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
    CHECK(handles.lookup_as<WritableEvent>(static_cast<Handle>(get(reply, 12))) != nullptr);
    CHECK(handles.lookup_as<Event>(static_cast<Handle>(get(reply, 16))) != nullptr);
    const auto header = Memory::base + 0x1000;
    // Version 1, base id 0, the count capped at the value page, then the offsets.
    CHECK(memory.word(header) == 1 && memory.word(header + 4) == 0 && memory.word(header + 8) == 0x1f8);
    CHECK(memory.word(header + 12) == 0x40 && memory.word(header + 16) == 0 && memory.word(header + 20) == 0x1040);
    CHECK(memory.word(header + 24) == 0x100 && memory.word(header + 28) == 0x1100);
    // The client holds the request event's writable end and the notify event's
    // readable end; only the readable end is waitable.
    const auto request = handles.lookup_as<WritableEvent>(static_cast<Handle>(get(reply, 12)));
    const auto notify = handles.lookup_as<Event>(static_cast<Handle>(get(reply, 16)));
    CHECK(request && notify && !handles.lookup_as<WritableEvent>(static_cast<Handle>(get(reply, 16))));
    if (request) request->signal();
}

// Counter values are published into the read-only memory. An armed wait is
// acknowledged at once the first time, then only after its counter moves.
void test_syncpoint_acknowledge() {
    Memory memory;
    const auto rw = Memory::base, ro = Memory::base + 0x1000;
    armigo::services::gsv::SyncpointCounters counters;
    const auto id = counters.allocate();
    CHECK(id == 64);
    auto notify = std::make_shared<Event>(false);
    counters.attach({rw, ro, 0x1f8,
                     [&](std::uint64_t address, std::span<std::byte> out) { return Memory::read(&memory, address, out); },
                     [&](std::uint64_t address, std::span<const std::byte> in) {
                         return Memory::write(&memory, address, in);
                     },
                     notify});
    const auto set = [&](std::uint64_t address, std::uint32_t value) {
        const std::array<std::byte, 4> bytes{static_cast<std::byte>(value), static_cast<std::byte>(value >> 8),
                                             static_cast<std::byte>(value >> 16), static_cast<std::byte>(value >> 24)};
        Memory::write(&memory, address, bytes);
    };
    // Syncpoint 64 is bit 0 of the second bit word; the summary names that word.
    set(rw + 0x108, 1);
    set(rw, 2);
    counters.serve_requests();
    CHECK(notify->is_signaled() && memory.word(ro + 0x1108) == 1 && memory.word(ro + 0x1040) == 2);
    notify->reset();
    set(rw + 0x108, 0);
    counters.serve_requests();
    CHECK(!notify->is_signaled() && memory.word(ro + 0x1108) == 1);
    counters.increment(id);
    CHECK(memory.word(ro + 0x40 + 64 * 8) == 1 && memory.word(ro + 0x40 + 64 * 8 + 4) == 0);
    CHECK(notify->is_signaled() && memory.word(ro + 0x1108) == 0);
    // Without an armed wait, an increment only publishes the value.
    notify->reset();
    counters.increment(id);
    CHECK(memory.word(ro + 0x40 + 64 * 8) == 2 && !notify->is_signaled());
    // A syncpoint nobody advances is acknowledged once, then waits.
    set(rw + 0x100, 4);
    counters.serve_requests();
    CHECK(notify->is_signaled() && memory.word(ro + 0x1100) == 4);
    notify->reset();
    set(rw + 0x100, 0);
    counters.serve_requests();
    CHECK(!notify->is_signaled() && memory.word(ro + 0x1100) == 4);
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

// A TIPC request with one send (A) descriptor before its raw data; the test
// supplies the buffer itself.
IpcMessage tipc_send(std::uint32_t command, std::initializer_list<std::uint32_t> data) {
    IpcMessage message{};
    put(message, 0, (16 + command) | 1u << 20);
    put(message, 4, static_cast<std::uint32_t>(data.size()));
    std::size_t offset = 20;
    for (const auto word : data) { put(message, offset, word); offset += 4; }
    return message;
}

// The decoder image pool, native memory and output images, then the video
// decoder service: it decodes the synthetic AV1 clip into the images in their
// hand-out order, advances its syncpoint for each and reports the outputs.
void test_video_decoder() {
    HandleTable handles{32};
    constexpr std::uint64_t pool_base = 0x200000;
    std::vector<std::byte> pool(0x8000);
    GraphicsSystemDependencies dependencies;
    dependencies.write_memory = [&](std::uint64_t address, std::span<const std::byte> bytes) {
        if (address < pool_base || address - pool_base + bytes.size() > pool.size()) return false;
        std::copy(bytes.begin(), bytes.end(), pool.begin() + static_cast<std::ptrdiff_t>(address - pool_base));
        return true;
    };
    GraphicsSystemSession session(pid, handles, dependencies);
    // A 64x64 NV12 image: pitch 64, luma and chroma planes of 128 rows each.
    const std::uint32_t extent = 63 | 63 << 16;
    auto reply = session.request(tipc(991, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}), pid);
    CHECK(handled(reply, 991) && get(reply, 12) == 0x8000 && get(reply, 20) == 0x2000);
    reply = session.request(tipc(111, {0, 0x8000, 0}), pid);
    const auto memory = get(reply, 12);
    CHECK(handled(reply, 111) && memory != 0);
    CHECK(handled(session.request(tipc(114, {memory, static_cast<std::uint32_t>(pool_base), 0, 0x8000, 0, 2, 0, 0, 0}),
                                  pid), 114));
    reply = session.request(tipc(911, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, memory}), pid);
    const auto first = get(reply, 12);
    CHECK(handled(reply, 911));
    reply = session.request(tipc(961, {first + 1}), pid);
    CHECK(handled(reply, 961) && get(reply, 12) == 10 && get(reply, 20) == 0x4000 && get(reply, 28) == 64 &&
          get(reply, 36) == 0x6000 && get(reply, 44) == 64);
    const auto video = handed_out<VideoSession>(handles, session.request(tipc(906, {0, 0, 3}), pid), 906);
    CHECK(video != nullptr);
    if (!video) return;
    const auto decoder = handed_out<VideoSession>(handles, video->request(tipc(1, {}), pid), 1);
    const auto syncpoint = handed_out<SyncpointSession>(handles, session.request(tipc(202, {}), pid), 202);
    CHECK(decoder != nullptr && syncpoint != nullptr);
    if (!decoder || !syncpoint) return;

    // Guest memory the request buffers point into: the codec name, then units.
    std::vector<std::byte> input(0x1000);
    const auto read = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& bytes = *static_cast<std::vector<std::byte>*>(raw);
        if (address + out.size() > bytes.size()) return false;
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(address), out.size(), out.begin());
        return true;
    };
    const auto with_input = [&](std::span<const std::uint8_t> data) {
        std::copy(data.begin(), data.end(), reinterpret_cast<std::uint8_t*>(input.data()));
        GuestBuffers buffers(read, nullptr, &input);
        buffers.add({0, static_cast<std::uint32_t>(data.size()), true, false});
        return buffers;
    };
    const std::string codec = "video/av1";
    std::vector<std::uint8_t> name(codec.begin(), codec.end());
    name.push_back(0);
    CHECK(handled(decoder->request_with_buffers(tipc_send(101, {64, 64, 24, 0}), pid, with_input(name)), 101));
    const auto ready = handed_out<Event>(handles, decoder->request(tipc(3, {}), pid), 3);
    CHECK(handled(decoder->request(tipc(6, {}), pid), 6) && ready && ready->is_signaled());
    // Each registered image's sync handle names the decoder's syncpoint and
    // the counter value of the frame that fills it.
    reply = decoder->request(tipc(311, {first}), pid);
    const auto counter = ~get(reply, 12);
    CHECK(handled(reply, 311) && get(reply, 16) == 0 && get(reply, 20) == 1);
    reply = decoder->request(tipc(311, {first + 1}), pid);
    CHECK(handled(reply, 311) && ~get(reply, 12) == counter && get(reply, 20) == 2);
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 8) == 0x1c286f);

    std::string failure;
    if (!armigo::host1x::make_av1_decoder(failure)) {
        std::puts("No hardware AV1 decoder: decoding through the session is not checked");
        return;
    }
    const auto units = armigo::test::flat_av1_temporal_units();
    for (std::size_t index = 0; index < units.size(); ++index) {
        reply = decoder->request_with_buffers(
            tipc_send(109, {static_cast<std::uint32_t>(41667 * index), 0, index + 1 == units.size() ? 1u : 0u}), pid,
            with_input(units[index]));
        CHECK(handled(reply, 109) && get(reply, 12) == index + 1);
    }
    // Two images took the first two frames; the third waits for a release.
    reply = syncpoint->request(tipc(103, {~counter}), pid);
    CHECK(handled(reply, 103) && get(reply, 12) == 2);
    const int luma = std::to_integer<int>(pool[32 * 64 + 32]);
    CHECK(luma > 118 && luma < 132);
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first && get(reply, 16) == 64 && get(reply, 20) == 64 &&
          get(reply, 44) == 0);
    reply = decoder->request(tipc(314, {first, 0, ~0u, 0, 0, 0}), pid);
    CHECK(handled(reply, 314) && get(reply, 20) == 3);
    reply = syncpoint->request(tipc(103, {~counter}), pid);
    CHECK(get(reply, 12) == 3);
    // A slot waiting for the next value signals when the decoder reaches it.
    const auto slot = handed_out<Event>(handles, syncpoint->request(tipc(22, {5}), pid), 22);
    CHECK(slot && handled(syncpoint->request(tipc(201, {5, ~counter, 4, 0, 0}), pid), 201) && !slot->is_signaled());
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first + 1);
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first && get(reply, 44) == 1);
    CHECK(handled(decoder->request(tipc(311, {first + 1}), pid), 311) && slot && !slot->is_signaled());
    // An end of stream whose input holds no picture still ends the stream: an
    // empty frame of the configured size takes the next image.
    reply = decoder->request_with_buffers(tipc_send(109, {0, 0, 1}), pid, with_input({}));
    CHECK(handled(reply, 109) && get(reply, 8) == 0);
    CHECK(slot && slot->is_signaled());
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first + 1 && get(reply, 16) == 64 && get(reply, 20) == 64 &&
          get(reply, 44) == 1);
    // An end-of-stream unit may come without a buffer, with time -1: the
    // stream still ends with an empty frame.
    CHECK(handled(decoder->request(tipc(311, {first}), pid), 311));
    reply = decoder->request(tipc(109, {~0u, ~0u, 1}), pid);
    CHECK(handled(reply, 109) && get(reply, 8) == 0);
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first && get(reply, 44) == 1);
    // An end-of-stream unit that does not decode ends the stream without a
    // picture; any other unit that does not decode is refused with its reason.
    const std::vector<std::uint8_t> junk(64, 0xff);
    CHECK(decoder->request_with_buffers(tipc_send(109, {0, 0, 0}), pid, with_input(junk)).disposition ==
          IpcDisposition::unsupported);
    CHECK(decoder->diagnostic() != nullptr);
    CHECK(handled(decoder->request(tipc(311, {first + 1}), pid), 311));
    reply = decoder->request_with_buffers(tipc_send(109, {~0u, ~0u, 1}), pid, with_input(junk));
    CHECK(handled(reply, 109) && get(reply, 8) == 0);
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first + 1 && get(reply, 44) == 1);
}

// Stop and finalize complete every fence handed out: images the client
// released after the stream's end read as reached, while a flush keeps them.
void test_video_teardown_fences() {
    HandleTable handles{32};
    constexpr std::uint64_t pool_base = 0x200000;
    std::vector<std::byte> pool(0x8000);
    GraphicsSystemDependencies dependencies;
    dependencies.write_memory = [&](std::uint64_t address, std::span<const std::byte> bytes) {
        if (address < pool_base || address - pool_base + bytes.size() > pool.size()) return false;
        std::copy(bytes.begin(), bytes.end(), pool.begin() + static_cast<std::ptrdiff_t>(address - pool_base));
        return true;
    };
    GraphicsSystemSession session(pid, handles, dependencies);
    // A 64x64 NV12 image: pitch 64, luma and chroma planes of 128 rows each.
    const std::uint32_t extent = 63 | 63 << 16;
    auto reply = session.request(tipc(991, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}), pid);
    CHECK(handled(reply, 991) && get(reply, 12) == 0x8000 && get(reply, 20) == 0x2000);
    reply = session.request(tipc(111, {0, 0x8000, 0}), pid);
    const auto memory = get(reply, 12);
    CHECK(handled(reply, 111) && memory != 0);
    CHECK(handled(session.request(tipc(114, {memory, static_cast<std::uint32_t>(pool_base), 0, 0x8000, 0, 2, 0, 0, 0}),
                                  pid), 114));
    reply = session.request(tipc(911, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, memory}), pid);
    const auto first = get(reply, 12);
    CHECK(handled(reply, 911));
    reply = session.request(tipc(961, {first + 1}), pid);
    CHECK(handled(reply, 961) && get(reply, 12) == 10 && get(reply, 20) == 0x4000 && get(reply, 28) == 64 &&
          get(reply, 36) == 0x6000 && get(reply, 44) == 64);
    const auto video = handed_out<VideoSession>(handles, session.request(tipc(906, {0, 0, 3}), pid), 906);
    CHECK(video != nullptr);
    if (!video) return;
    const auto decoder = handed_out<VideoSession>(handles, video->request(tipc(1, {}), pid), 1);
    const auto syncpoint = handed_out<SyncpointSession>(handles, session.request(tipc(202, {}), pid), 202);
    CHECK(decoder != nullptr && syncpoint != nullptr);
    if (!decoder || !syncpoint) return;

    // Guest memory the request buffers point into: the codec name, then units.
    std::vector<std::byte> input(0x1000);
    const auto read = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& bytes = *static_cast<std::vector<std::byte>*>(raw);
        if (address + out.size() > bytes.size()) return false;
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(address), out.size(), out.begin());
        return true;
    };
    const auto with_input = [&](std::span<const std::uint8_t> data) {
        std::copy(data.begin(), data.end(), reinterpret_cast<std::uint8_t*>(input.data()));
        GuestBuffers buffers(read, nullptr, &input);
        buffers.add({0, static_cast<std::uint32_t>(data.size()), true, false});
        return buffers;
    };
    const std::string codec = "video/av1";
    std::vector<std::uint8_t> name(codec.begin(), codec.end());
    name.push_back(0);
    CHECK(handled(decoder->request_with_buffers(tipc_send(101, {64, 64, 24, 0}), pid, with_input(name)), 101));
    const auto ready = handed_out<Event>(handles, decoder->request(tipc(3, {}), pid), 3);
    CHECK(handled(decoder->request(tipc(6, {}), pid), 6) && ready && ready->is_signaled());
    std::string failure;
    if (!armigo::host1x::make_av1_decoder(failure)) {
        std::puts("No hardware AV1 decoder: teardown fences are not checked");
        return;
    }
    reply = decoder->request(tipc(311, {first}), pid);
    const auto counter = ~get(reply, 12);
    CHECK(handled(reply, 311) && get(reply, 20) == 1);
    CHECK(handled(decoder->request(tipc(311, {first + 1}), pid), 311));
    const auto key = armigo::test::flat_av1_temporal_units()[0];
    CHECK(handled(decoder->request_with_buffers(tipc_send(109, {0, 0, 0}), pid, with_input(key)), 109));
    CHECK(handled(decoder->request_with_buffers(tipc_send(109, {16667, 0, 1}), pid, with_input(key)), 109));
    const auto counted = [&] { return get(syncpoint->request(tipc(103, {~counter}), pid), 12); };
    CHECK(counted() == 2);
    CHECK(handled(decoder->request(tipc(213, {}), pid), 213));
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 44) == 1);
    // Both images go back after the stream's end: values 3 and 4 that no
    // picture will reach.
    reply = decoder->request(tipc(314, {first, 0, ~0u, 0, 0, 0}), pid);
    CHECK(handled(reply, 314) && get(reply, 20) == 3);
    reply = decoder->request(tipc(314, {first + 1, 0, ~0u, 0, 0, 0}), pid);
    CHECK(handled(reply, 314) && get(reply, 20) == 4);
    CHECK(handled(decoder->request(tipc(8, {}), pid), 8) && counted() == 2);
    CHECK(handled(decoder->request(tipc(207, {}), pid), 207) && counted() == 4);
    CHECK(handled(decoder->request(tipc(115, {}), pid), 115) && counted() == 4);
}

// Movie pacing: a decoded frame that the next waiting frame has already
// overtaken on the host clock is dropped when an image frees up, unless the
// client stopped acquiring for more than a second.
void test_video_pacing() {
    std::int64_t now_us = 1'000'000;
    HandleTable handles{32};
    constexpr std::uint64_t pool_base = 0x200000;
    std::vector<std::byte> pool(0x8000);
    GraphicsSystemDependencies dependencies;
    dependencies.clock = [&] {
        return std::chrono::steady_clock::time_point(std::chrono::microseconds(now_us));
    };
    dependencies.write_memory = [&](std::uint64_t address, std::span<const std::byte> bytes) {
        if (address < pool_base || address - pool_base + bytes.size() > pool.size()) return false;
        std::copy(bytes.begin(), bytes.end(), pool.begin() + static_cast<std::ptrdiff_t>(address - pool_base));
        return true;
    };
    GraphicsSystemSession session(pid, handles, dependencies);
    // A 64x64 NV12 image: pitch 64, luma and chroma planes of 128 rows each.
    const std::uint32_t extent = 63 | 63 << 16;
    auto reply = session.request(tipc(991, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}), pid);
    CHECK(handled(reply, 991) && get(reply, 12) == 0x8000 && get(reply, 20) == 0x2000);
    reply = session.request(tipc(111, {0, 0x8000, 0}), pid);
    const auto memory = get(reply, 12);
    CHECK(handled(reply, 111) && memory != 0);
    CHECK(handled(session.request(tipc(114, {memory, static_cast<std::uint32_t>(pool_base), 0, 0x8000, 0, 2, 0, 0, 0}),
                                  pid), 114));
    reply = session.request(tipc(911, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, memory}), pid);
    const auto first = get(reply, 12);
    CHECK(handled(reply, 911));
    reply = session.request(tipc(961, {first + 1}), pid);
    CHECK(handled(reply, 961) && get(reply, 12) == 10 && get(reply, 20) == 0x4000 && get(reply, 28) == 64 &&
          get(reply, 36) == 0x6000 && get(reply, 44) == 64);
    const auto video = handed_out<VideoSession>(handles, session.request(tipc(906, {0, 0, 3}), pid), 906);
    CHECK(video != nullptr);
    if (!video) return;
    const auto decoder = handed_out<VideoSession>(handles, video->request(tipc(1, {}), pid), 1);
    const auto syncpoint = handed_out<SyncpointSession>(handles, session.request(tipc(202, {}), pid), 202);
    CHECK(decoder != nullptr && syncpoint != nullptr);
    if (!decoder || !syncpoint) return;

    // Guest memory the request buffers point into: the codec name, then units.
    std::vector<std::byte> input(0x1000);
    const auto read = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& bytes = *static_cast<std::vector<std::byte>*>(raw);
        if (address + out.size() > bytes.size()) return false;
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(address), out.size(), out.begin());
        return true;
    };
    const auto with_input = [&](std::span<const std::uint8_t> data) {
        std::copy(data.begin(), data.end(), reinterpret_cast<std::uint8_t*>(input.data()));
        GuestBuffers buffers(read, nullptr, &input);
        buffers.add({0, static_cast<std::uint32_t>(data.size()), true, false});
        return buffers;
    };
    const std::string codec = "video/av1";
    std::vector<std::uint8_t> name(codec.begin(), codec.end());
    name.push_back(0);
    CHECK(handled(decoder->request_with_buffers(tipc_send(101, {64, 64, 24, 0}), pid, with_input(name)), 101));
    const auto ready = handed_out<Event>(handles, decoder->request(tipc(3, {}), pid), 3);
    CHECK(handled(decoder->request(tipc(6, {}), pid), 6) && ready && ready->is_signaled());
    std::string failure;
    if (!armigo::host1x::make_av1_decoder(failure)) {
        std::puts("No hardware AV1 decoder: movie pacing is not checked");
        return;
    }
    CHECK(handled(decoder->request(tipc(311, {first}), pid), 311));
    CHECK(handled(decoder->request(tipc(311, {first + 1}), pid), 311));
    // The key frame again and again, 60 frames a second: two images take the
    // first two frames, three wait.
    const auto key = armigo::test::flat_av1_temporal_units()[0];
    const auto submit = [&](std::uint32_t pts_us) {
        return handled(decoder->request_with_buffers(tipc_send(109, {pts_us, 0, 0}), pid, with_input(key)), 109);
    };
    for (const std::uint32_t pts_us : {0u, 16667u, 33333u, 50000u, 66667u}) CHECK(submit(pts_us));
    const auto shown_pts = [&](IpcReply reply) { return std::uint64_t{get(reply, 36)} | std::uint64_t{get(reply, 40)} << 32; };
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first && shown_pts(reply) == 0);
    // 60 ms later the frame at 50 ms is due: the one at 33 ms is dropped, and
    // the one at 67 ms, not yet due, keeps the frame at 50 ms.
    now_us += 60'000;
    CHECK(handled(decoder->request(tipc(314, {first, 0, ~0u, 0, 0, 0}), pid), 314));
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first + 1 && shown_pts(reply) == 16'667'000);
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first && shown_pts(reply) == 50'000'000);
    // A client that acquired nothing for two seconds was paused: nothing is
    // dropped, and the clock restarts at its next acquire.
    for (const std::uint32_t pts_us : {83333u, 100000u}) CHECK(submit(pts_us));
    now_us += 2'000'000;
    CHECK(handled(decoder->request(tipc(314, {first + 1, 0, ~0u, 0, 0, 0}), pid), 314));
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first + 1 && shown_pts(reply) == 66'667'000);
    // A flush restarts the clock too: after it nothing is dropped until the
    // client acquires again.
    CHECK(handled(decoder->request(tipc(8, {}), pid), 8));
    for (const std::uint32_t pts_us : {0u, 16667u, 33333u}) CHECK(submit(pts_us));
    now_us += 50'000;
    CHECK(handled(decoder->request(tipc(314, {first, 0, ~0u, 0, 0, 0}), pid), 314));
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 12) == first && shown_pts(reply) == 0);
}

// Word counts, image layout, unknown ids, a refused codec, an empty output
// queue, a dropped frame and a 64-bit syncpoint wait.
void test_video_decoder_edges() {
    // 906 accepts exactly three raw words. 1002 accepts none and hands out a session.
    {
        HandleTable handles{16};
        GraphicsSystemSession session(pid, handles);
        CHECK(session.request(tipc(906, {}), pid).disposition == IpcDisposition::unsupported);
        CHECK(session.request(tipc(906, {0, 0, 3, 1}), pid).disposition == IpcDisposition::unsupported);
        CHECK(handed_out<VideoSession>(handles, session.request(tipc(1002, {}), pid), 1002) != nullptr);
    }
    // Code 2 is BGRA: each 64x64 image is 256 by 128 bytes and has no chroma plane.
    {
        HandleTable handles{16};
        GraphicsSystemSession session(pid, handles);
        constexpr std::uint32_t extent = 63 | 63 << 16;
        auto reply = session.request(tipc(991, {0, 2, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}), pid);
        CHECK(handled(reply, 991) && get(reply, 12) == 256 * 128 && get(reply, 16) == 0 && get(reply, 20) == 0x2000);
        reply = session.request(tipc(111, {0, 256 * 128, 0}), pid);
        const auto memory = get(reply, 12);
        CHECK(handled(reply, 111) && memory != 0);
        reply = session.request(tipc(911, {0, 2, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, memory}), pid);
        const auto image = get(reply, 12);
        CHECK(handled(reply, 911) && image != 0);
        // Kind 9, pitch 256. The chroma words stay zero.
        reply = session.request(tipc(961, {image}), pid);
        CHECK(handled(reply, 961) && get(reply, 8) == 0 && get(reply, 12) == 9 && get(reply, 20) == 0 &&
              get(reply, 28) == 256 && get(reply, 36) == 0 && get(reply, 44) == 0);
    }
    // Unknown ids are refused. Destroying an image or a memory makes a later use fail.
    {
        HandleTable handles{16};
        GraphicsSystemSession session(pid, handles);
        constexpr std::uint32_t extent = 63 | 63 << 16;
        CHECK(session.request(tipc(911, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}), pid)
                  .disposition == IpcDisposition::unsupported);
        CHECK(session.request(tipc(114, {0, 0, 0, 0x1000, 0, 0, 0, 0, 0}), pid).disposition ==
              IpcDisposition::unsupported);
        auto reply = session.request(tipc(111, {0, 0x1000, 0}), pid);
        const auto memory = get(reply, 12);
        CHECK(handled(reply, 111) && memory != 0);
        reply = session.request(tipc(911, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, memory}), pid);
        const auto image = get(reply, 12);
        CHECK(handled(reply, 911) && image != 0);
        reply = session.request(tipc(912, {image}), pid);
        CHECK(handled(reply, 912) && get(reply, 8) == 0 && reply.size == 12);
        CHECK(session.request(tipc(961, {image}), pid).disposition == IpcDisposition::unsupported);
        reply = session.request(tipc(112, {memory}), pid);
        CHECK(handled(reply, 112) && get(reply, 8) == 0 && reply.size == 12);
        CHECK(session.request(tipc(114, {memory, 0, 0, 0x1000, 0, 0, 0, 0, 0}), pid).disposition ==
              IpcDisposition::unsupported);
    }
    // A service configured for video/vp9 has no decoder, so a submit is refused.
    {
        HandleTable handles{16};
        GraphicsSystemSession session(pid, handles);
        const auto video = handed_out<VideoSession>(handles, session.request(tipc(1002, {}), pid), 1002);
        const auto decoder = video ? handed_out<VideoSession>(handles, video->request(tipc(1, {}), pid), 1) : nullptr;
        CHECK(decoder != nullptr);
        if (decoder) {
            std::vector<std::byte> input(64);
            const auto read = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
                auto& bytes = *static_cast<std::vector<std::byte>*>(raw);
                if (address + out.size() > bytes.size()) return false;
                std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(address), out.size(), out.begin());
                return true;
            };
            const auto with_input = [&](std::span<const std::uint8_t> data) {
                std::copy(data.begin(), data.end(), reinterpret_cast<std::uint8_t*>(input.data()));
                GuestBuffers buffers(read, nullptr, &input);
                buffers.add({0, static_cast<std::uint32_t>(data.size()), true, false});
                return buffers;
            };
            const std::string codec = "video/vp9";
            std::vector<std::uint8_t> name(codec.begin(), codec.end());
            name.push_back(0);
            CHECK(handled(decoder->request_with_buffers(tipc_send(101, {64, 64, 24, 0}), pid, with_input(name)), 101));
            CHECK(decoder->request_with_buffers(tipc_send(109, {0, 0, 0}), pid, with_input(name)).disposition ==
                  IpcDisposition::unsupported);
            const char* text = decoder->diagnostic();
            CHECK(text != nullptr && std::string_view(text).find("video/vp9") != std::string_view::npos);
        }
    }
    // Nothing has been decoded, so the output query is the empty-queue result and carries no data.
    {
        HandleTable handles{16};
        GraphicsSystemSession session(pid, handles);
        const auto video = handed_out<VideoSession>(handles, session.request(tipc(1002, {}), pid), 1002);
        const auto decoder = video ? handed_out<VideoSession>(handles, video->request(tipc(1, {}), pid), 1) : nullptr;
        CHECK(decoder != nullptr);
        if (decoder) {
            const auto reply = decoder->request(tipc(213, {}), pid);
            CHECK(handled(reply, 213) && get(reply, 4) == 1 && get(reply, 8) == 0x1c286f && reply.size == 12);
        }
    }

    std::string failure;
    if (!armigo::host1x::make_av1_decoder(failure)) {
        std::puts("No hardware AV1 decoder: dropped frames and 64-bit syncpoint waits are not checked");
        return;
    }
    // Two images, the first unregistered: the first frame fills the second image.
    // The next frame waits with no free image; stop drops it. A later register
    // leaves the counter where it is. Mode 1 then waits for that counter to reach 2.
    HandleTable handles{32};
    constexpr std::uint64_t pool_base = 0x200000;
    std::vector<std::byte> pool(0x8000);
    GraphicsSystemDependencies dependencies;
    dependencies.write_memory = [&](std::uint64_t address, std::span<const std::byte> bytes) {
        if (address < pool_base || address - pool_base + bytes.size() > pool.size()) return false;
        std::copy(bytes.begin(), bytes.end(), pool.begin() + static_cast<std::ptrdiff_t>(address - pool_base));
        return true;
    };
    GraphicsSystemSession session(pid, handles, dependencies);
    constexpr std::uint32_t extent = 63 | 63 << 16;
    auto reply = session.request(tipc(991, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2}), pid);
    CHECK(handled(reply, 991));
    reply = session.request(tipc(111, {0, 0x8000, 0}), pid);
    const auto memory = get(reply, 12);
    CHECK(handled(reply, 111) && memory != 0);
    CHECK(handled(session.request(tipc(114, {memory, static_cast<std::uint32_t>(pool_base), 0, 0x8000, 0, 2, 0, 0, 0}),
                                  pid),
                  114));
    reply = session.request(tipc(911, {0, 1, extent, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, memory}), pid);
    const auto first = get(reply, 12);
    CHECK(handled(reply, 911) && first != 0);
    const auto video = handed_out<VideoSession>(handles, session.request(tipc(906, {0, 0, 3}), pid), 906);
    const auto decoder = video ? handed_out<VideoSession>(handles, video->request(tipc(1, {}), pid), 1) : nullptr;
    const auto syncpoint = handed_out<SyncpointSession>(handles, session.request(tipc(202, {}), pid), 202);
    CHECK(decoder != nullptr && syncpoint != nullptr);
    if (!decoder || !syncpoint) return;

    std::vector<std::byte> input(0x1000);
    const auto read = [](void* raw, std::uint64_t address, std::span<std::byte> out) noexcept {
        auto& bytes = *static_cast<std::vector<std::byte>*>(raw);
        if (address + out.size() > bytes.size()) return false;
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(address), out.size(), out.begin());
        return true;
    };
    const auto with_input = [&](std::span<const std::uint8_t> data) {
        std::copy(data.begin(), data.end(), reinterpret_cast<std::uint8_t*>(input.data()));
        GuestBuffers buffers(read, nullptr, &input);
        buffers.add({0, static_cast<std::uint32_t>(data.size()), true, false});
        return buffers;
    };
    const std::string codec = "video/av1";
    std::vector<std::uint8_t> name(codec.begin(), codec.end());
    name.push_back(0);
    CHECK(handled(decoder->request_with_buffers(tipc_send(101, {64, 64, 24, 0}), pid, with_input(name)), 101));
    CHECK(handled(decoder->request(tipc(6, {}), pid), 6));
    // 311, 103 and 201 name the syncpoint by its bitwise complement.
    reply = decoder->request(tipc(311, {first}), pid);
    const auto sync_bits = get(reply, 12);
    CHECK(handled(reply, 311) && get(reply, 20) == 1);
    reply = decoder->request(tipc(311, {first + 1}), pid);
    CHECK(handled(reply, 311) && get(reply, 12) == sync_bits && get(reply, 20) == 2);
    CHECK(handled(decoder->request(tipc(212, {first}), pid), 212));

    const auto units = armigo::test::flat_av1_temporal_units();
    CHECK(units.size() == 3);
    if (units.size() < 3) return;
    const auto submit = [&](std::size_t index) {
        const auto end = index + 1 == units.size() ? 1u : 0u;
        return decoder->request_with_buffers(
            tipc_send(109, {static_cast<std::uint32_t>(41667 * index), 0, end}), pid, with_input(units[index]));
    };
    reply = submit(0);
    CHECK(handled(reply, 109) && get(reply, 12) == 1);
    reply = decoder->request(tipc(213, {}), pid);
    CHECK(handled(reply, 213) && get(reply, 8) == 0 && get(reply, 12) == first + 1);
    reply = syncpoint->request(tipc(103, {sync_bits}), pid);
    CHECK(handled(reply, 103) && get(reply, 12) == 1 && get(reply, 16) == 0);

    const auto waiting = handed_out<Event>(handles, syncpoint->request(tipc(22, {1}), pid), 22);
    const auto cancelled = handed_out<Event>(handles, syncpoint->request(tipc(22, {2}), pid), 22);
    CHECK(waiting != nullptr && cancelled != nullptr);
    if (!waiting || !cancelled) return;
    // Value 2 in mode 1 is still ahead of counter 1. Cancel signals its own slot.
    CHECK(handled(syncpoint->request(tipc(201, {1, sync_bits, 2, 0, 1}), pid), 201) && !waiting->is_signaled());
    CHECK(handled(syncpoint->request(tipc(201, {2, sync_bits, 4, 0, 1}), pid), 201) && !cancelled->is_signaled());
    CHECK(handled(syncpoint->request(tipc(202, {2}), pid), 202) && cancelled->is_signaled() &&
          !waiting->is_signaled());

    reply = submit(1);
    CHECK(handled(reply, 109) && get(reply, 12) == 2);
    reply = syncpoint->request(tipc(103, {sync_bits}), pid);
    CHECK(handled(reply, 103) && get(reply, 12) == 1);
    CHECK(handled(decoder->request(tipc(207, {}), pid), 207));
    reply = syncpoint->request(tipc(103, {sync_bits}), pid);
    CHECK(handled(reply, 103) && get(reply, 12) == 2 && waiting->is_signaled());
    CHECK(handled(decoder->request(tipc(311, {first}), pid), 311));
    // The last unit fills the image registered after the stop: the counter reaches 3.
    reply = submit(2);
    CHECK(handled(reply, 109));
    reply = syncpoint->request(tipc(103, {sync_bits}), pid);
    CHECK(handled(reply, 103) && get(reply, 12) == 3);

    // With no free image, twelve decoded frames wait; the next unit is refused
    // as busy (nn::codec 3603) and decodes after an image takes a frame.
    CHECK(handled(decoder->request(tipc(6, {}), pid), 6));
    std::size_t accepted = 0;
    for (std::size_t round = 0; round < 4; ++round)
        for (std::size_t index = 0; index < units.size(); ++index) {
            reply = decoder->request_with_buffers(
                tipc_send(109, {static_cast<std::uint32_t>(41667 * index), 0, 0}), pid, with_input(units[index]));
            if (handled(reply, 109) && get(reply, 8) == 0) ++accepted;
        }
    CHECK(accepted == 12);
    reply = decoder->request_with_buffers(tipc_send(109, {0, 0, 0}), pid, with_input(units[0]));
    CHECK(handled(reply, 109) && get(reply, 8) == 0x1c266f && reply.size == 12);
    CHECK(handled(decoder->request(tipc(311, {first + 1}), pid), 311));
    reply = decoder->request_with_buffers(tipc_send(109, {0, 0, 0}), pid, with_input(units[0]));
    CHECK(handled(reply, 109) && get(reply, 8) == 0);
}

int main() {
    test_session_basics();
    test_driver_session();
    test_syncpoint_session();
    test_syncpoint_memory();
    test_syncpoint_acknowledge();
    test_gpu_session();
    test_display_session();
    test_video_decoder();
    test_video_pacing();
    test_video_teardown_fences();
    test_video_decoder_edges();
    if (failures) std::fprintf(stderr, "%u failures\n", failures);
    return failures ? 1 : 0;
}
