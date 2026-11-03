// ============================================================================
// Armigo Tests - Horizon Service Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/audio_out.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {
using namespace armigo::horizon;
using armigo::services::audout::AudioOutSession;
using armigo::services::audout::ManagerSession;
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
// A non-domain method whose CMIF header lands at `cmif` (descriptors before it
// are given by the header words; the session reads only the CMIF).
IpcMessage method(std::uint32_t first, std::uint32_t second, unsigned cmif, std::uint32_t command) {
    IpcMessage bytes{};
    bytes.fill(std::byte{0xa5});
    put(bytes, 0, first);
    put(bytes, 4, second);
    put(bytes, cmif, 0x49434653);
    put(bytes, cmif + 4, 1);
    put(bytes, cmif + 8, command);
    put(bytes, cmif + 12, 0);
    return bytes;
}

struct Memory {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x4000);
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
    void put64(std::size_t at, std::uint64_t value) { std::memcpy(bytes.data() + at, &value, 8); }
    std::uint64_t get64(std::size_t at) const { std::uint64_t v; std::memcpy(&v, bytes.data() + at, 8); return v; }
};

// OpenAudioOut: A name in at 0x100, B normalized name out at 0x200; CMIF at 48
// (header 8, special header and PID 12, one copy handle 4, A and B 24).
IpcReply open(ManagerSession& manager, Memory& memory, const char* name, std::uint32_t rate, std::uint32_t channels) {
    std::memset(memory.bytes.data() + 0x100, 0, 0x200);
    std::memcpy(memory.bytes.data() + 0x100, name, std::strlen(name));
    auto message = method(0x01100006, 0x8000000c, 48, 1);
    put(message, 64, rate);
    put(message, 68, channels);
    GuestBuffers buffers(Memory::read, Memory::write, &memory);
    buffers.add({0x100, 256, true, true});
    buffers.add({0x200, 256, true, true});
    return manager.request_with_buffers(message, 37, buffers);
}

void test_open() {
    ManagerSession manager(37);
    Memory memory;
    auto reply = open(manager, memory, "", 48000, 6);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1 && !reply.transferred_as_copy);
    CHECK(word(reply.bytes, 16) == 0x4f434653 && word(reply.bytes, 24) == 0);
    CHECK(word(reply.bytes, 32) == 48000 && word(reply.bytes, 36) == 6 && word(reply.bytes, 40) == 2 &&
          word(reply.bytes, 44) == 1);
    CHECK(std::memcmp(memory.bytes.data() + 0x200, "DeviceOut", 10) == 0);
    reply = open(manager, memory, "DeviceOut", 0, 0);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 36) == 2);
    CHECK(word(open(manager, memory, "Other", 48000, 2).bytes, 24) == 0x299);
    CHECK(word(open(manager, memory, "", 44100, 2).bytes, 24) == 0x699);
    CHECK(word(open(manager, memory, "", 48000, 4).bytes, 24) == 0x1499);
    CHECK(open(manager, memory, "", 48000, 4).transferred_count == 0);
    CHECK(manager.request_with_buffers(method(6, 8, 16, 1), 38, GuestBuffers{}).disposition ==
          IpcDisposition::unsupported);
}

// Two 10 ms stereo buffers play in order and come back by tag once their time
// has passed; an empty poll writes a zero tag.
void test_playback() {
    AudioOutSession session(37, 2, nullptr);
    Memory memory;
    // QueryPointerBufferSize reports no pointer buffer.
    IpcMessage control{};
    put(control, 0, 7);
    put(control, 4, 8);
    put(control, 16, 0x49434653);
    put(control, 24, 3);
    auto reply = session.request(control, 37);
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 32) == 0);

    reply = session.request(method(6, 8, 16, 4), 37);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1 && reply.transferred_as_copy);
    const auto event = std::dynamic_pointer_cast<Event>(reply.transferred[0]);
    CHECK(event && !event->is_signaled());

    const auto append = [&](std::uint64_t record, std::uint64_t samples, std::uint64_t bytes, std::uint64_t tag) {
        memory.put64(record + 8, samples);
        memory.put64(record + 16, bytes);
        memory.put64(record + 24, bytes);
        auto message = method(0x00110006, 0xa, 32, 7);
        put(message, 48, static_cast<std::uint32_t>(tag));
        put(message, 52, static_cast<std::uint32_t>(tag >> 32));
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({record, 40, true, true});
        return session.request_with_buffers(message, 37, buffers);
    };
    CHECK(word(append(0x40, 0x1000, 480 * 4, 0x1111).bytes, 24) == 0);
    CHECK(word(append(0x80, 0x1800, 480 * 4, 0x2222).bytes, 24) == 0);
    CHECK(word(append(0xc0, 0x1800, 0, 0x3333).bytes, 24) == 0x499);  // empty buffer
    CHECK(append(0xc0, 0x1800, 6, 0x3333).disposition == IpcDisposition::unsupported);  // not whole frames
    auto contains = method(6, 10, 16, 6);
    put(contains, 32, 0x2222);
    put(contains, 36, 0);
    CHECK(word(session.request(contains, 37).bytes, 32) == 1);

    const auto released = [&](std::uint64_t at, std::uint32_t capacity) {
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({at, capacity, true, true});
        memory.put64(at, 0xdead);
        return session.request_with_buffers(method(0x01000006, 0xc09, 32, 8), 37, buffers);
    };
    reply = released(0x300, 16);
    CHECK(word(reply.bytes, 32) == 0 && memory.get64(0x300) == 0);  // nothing played yet: stopped

    CHECK(word(session.request(method(6, 8, 16, 1), 37).bytes, 24) == 0);       // Start
    CHECK(word(session.request(method(6, 8, 16, 1), 37).bytes, 24) == 0x499);   // already started
    CHECK(word(session.request(method(6, 8, 16, 0), 37).bytes, 32) == 0);       // started
    std::vector<std::uint64_t> tags;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (tags.size() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        reply = released(0x300, 16);
        for (std::uint32_t i = 0; i != word(reply.bytes, 32); ++i) tags.push_back(memory.get64(0x300 + 8 * i));
    }
    CHECK(tags == (std::vector<std::uint64_t>{0x1111, 0x2222}));
    CHECK(word(session.request(contains, 37).bytes, 32) == 0);
    CHECK(word(session.request(method(6, 8, 16, 2), 37).bytes, 24) == 0);       // Stop
    CHECK(word(session.request(method(6, 8, 16, 0), 37).bytes, 32) == 1);       // stopped

    // The queue holds at most 32 buffers while stopped.
    for (std::uint64_t i = 0; i != AudioOutSession::max_buffers; ++i)
        CHECK(word(append(0x40, 0x1000, 4, 0x100 + i).bytes, 24) == 0);
    CHECK(word(append(0x40, 0x1000, 4, 0x999).bytes, 24) == 0x1099);
    CHECK(session.request(method(6, 8, 16, 99), 37).disposition == IpcDisposition::unsupported);
}

// Queued buffers are released at the rate they play: 400 buffers of 1 ms take
// 400 ms however late each wait of the worker wakes (a schedule that restarted
// from each wake-up would lose that latency on every buffer).
void test_real_time_releases() {
    AudioOutSession session(37, 2, nullptr);
    Memory memory;
    const auto append = [&](std::uint64_t tag) {
        memory.put64(0x40 + 8, 0x1000);
        memory.put64(0x40 + 16, 48 * 4);
        memory.put64(0x40 + 24, 48 * 4);
        auto message = method(0x00110006, 0xa, 32, 7);
        put(message, 48, static_cast<std::uint32_t>(tag));
        put(message, 52, 0);
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0x40, 40, true, true});
        return session.request_with_buffers(message, 37, buffers);
    };
    constexpr std::uint64_t count = 400;
    std::uint64_t appended = 0, released_count = 0;
    while (appended != AudioOutSession::max_buffers) CHECK(word(append(++appended).bytes, 24) == 0);
    const auto start = std::chrono::steady_clock::now();
    CHECK(word(session.request(method(6, 8, 16, 1), 37).bytes, 24) == 0);  // Start
    const auto deadline = start + std::chrono::seconds(2);
    while (released_count < count && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0x300, 256, true, true});
        const auto reply = session.request_with_buffers(method(0x01000006, 0xc09, 32, 8), 37, buffers);
        released_count += word(reply.bytes, 32);
        // Keep the queue full, as a game does.
        while (appended < count && appended - released_count < AudioOutSession::max_buffers)
            CHECK(word(append(++appended).bytes, 24) == 0);
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    CHECK(released_count == count);
    CHECK(elapsed > 0.395 && elapsed < 0.415);
}
} // namespace

int main() {
    test_open();
    test_playback();
    test_real_time_releases();
    if (failures) return 1;
    std::puts("audio out checks passed");
    return 0;
}
