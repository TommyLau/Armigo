// ============================================================================
// Armigo Tests - Switch 2 Audio Device Manager
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// auddmg:u hands out the virtual output device manager, which lists one
// stereo 48 kHz device, names it as the default, answers format queries,
// returns its property block and a state event; stream attachment and other
// requests stay unsupported with a diagnostic.

#include <armigo/services/audio_device_manager.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace {
using armigo::horizon::GuestBuffers;
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::horizon::IpcReply;
using namespace armigo::services::auddmg;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

std::uint32_t word(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}

// A request whose CMIF header is at 16 (no descriptors the session reads).
IpcMessage request(std::uint32_t command, std::initializer_list<std::uint32_t> words = {}) {
    IpcMessage message{};
    message.fill(std::byte{0});
    put(message, 0, 4);
    put(message, 4, 0x9);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    std::size_t offset = 32;
    for (const auto value : words) put(message, offset, value), offset += 4;
    return message;
}

struct Memory {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x1000);
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
    std::string_view text(std::size_t at) const { return reinterpret_cast<const char*>(bytes.data() + at); }
    std::uint32_t get32(std::size_t at) const { std::uint32_t v; std::memcpy(&v, bytes.data() + at, 4); return v; }
    std::uint64_t get64(std::size_t at) const { std::uint64_t v; std::memcpy(&v, bytes.data() + at, 8); return v; }
};
} // namespace

int main() {
    constexpr std::uint64_t pid = 0x81;
    DeviceManagerSession service(pid);
    // Command 0: 16 bytes in, the virtual device manager moved out.
    auto opened = service.request(request(0, {1, 2, 3, 4}), pid);
    CHECK(opened.disposition == IpcDisposition::handled && opened.transferred_count == 1 &&
          !opened.transferred_as_copy);
    CHECK(service.open_argument() && (*service.open_argument())[4] == std::byte{2});
    const auto manager = std::dynamic_pointer_cast<VirtualDeviceManagerSession>(opened.transferred[0]);
    CHECK(manager != nullptr);
    CHECK(service.request(request(1), pid).disposition == IpcDisposition::unsupported && service.diagnostic());
    CHECK(service.request(request(0), pid + 1).disposition == IpcDisposition::unsupported);
    if (!manager) return 1;

    Memory memory;
    // 0 (flag, functionality): output (1) lists one device, its name in the
    // out buffer; input (2) lists none and writes nothing; others are refused.
    {
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0x100, 0x100, true, true});
        const auto reply = manager->request_with_buffers(request(0, {0, 1}), pid, buffers);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 24) == 0 && word(reply.bytes, 32) == 1);
        CHECK(memory.text(0x100) == "AudioVirtualOutput0");
    }
    {
        std::memset(memory.bytes.data() + 0x100, 0, 0x100);
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0x100, 0x100, true, true});
        const auto reply = manager->request_with_buffers(request(0, {0, 2}), pid, buffers);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 24) == 0 && word(reply.bytes, 32) == 0);
        CHECK(memory.text(0x100).empty());
        CHECK(manager->request_with_buffers(request(0, {0, 3}), pid, buffers).disposition == IpcDisposition::unsupported);
        CHECK(manager->request_with_buffers(request(0, {1}), pid, buffers).disposition == IpcDisposition::unsupported);
    }
    // 100: the default device's name.
    {
        std::memset(memory.bytes.data() + 0x100, 0, 0x100);
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0x100, 0x100, false, true});
        const auto reply = manager->request_with_buffers(request(100), pid, buffers);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 24) == 0);
        CHECK(memory.text(0x100) == "AudioVirtualOutput0");
    }
    // 300: 16-bit or float stereo 48 kHz for functionality 1 only.
    CHECK(word(manager->request(request(300, {2, 2, 48000, 1}), pid).bytes, 24) == 0);
    CHECK(word(manager->request(request(300, {5, 2, 48000, 1}), pid).bytes, 24) == 0);
    CHECK(word(manager->request(request(300, {2, 6, 48000, 1}), pid).bytes, 24) == result_not_supported);
    CHECK(word(manager->request(request(300, {2, 2, 32000, 1}), pid).bytes, 24) == result_not_supported);
    CHECK(word(manager->request(request(300, {2, 2, 48000, 2}), pid).bytes, 24) == result_not_supported);
    // 401: the property block for the name given.
    {
        std::memcpy(memory.bytes.data() + 0x100, "AudioVirtualOutput0", 20);
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0x100, 0x100, true, true});
        buffers.add({0x400, static_cast<std::uint32_t>(device_properties_size), true, true});
        const auto reply = manager->request_with_buffers(request(401), pid, buffers);
        CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 24) == 0);
        CHECK(memory.text(0x400) == "AudioVirtualOutput0");
        CHECK(memory.get32(0x500) == 3 && memory.get32(0x504) == 2 && memory.get32(0x508) == 2 &&
              memory.get32(0x50c) == 48000);
        CHECK(memory.text(0x510) == "AudioTvOutput");
        CHECK(memory.get64(0x618) == 10'000'000 && memory.get32(0x620) == 1);
        GuestBuffers short_block(Memory::read, Memory::write, &memory);
        short_block.add({0x100, 0x100, true, true});
        short_block.add({0x400, 0x100, true, true});
        CHECK(manager->request_with_buffers(request(401), pid, short_block).disposition == IpcDisposition::unsupported);
    }
    // 500: a copied state event, the same each time.
    const auto first = manager->request(request(500), pid);
    const auto second = manager->request(request(500), pid);
    CHECK(first.disposition == IpcDisposition::handled && first.transferred_count == 1 && first.transferred_as_copy);
    CHECK(first.transferred[0] && first.transferred[0] == second.transferred[0]);
    // Stream attachment needs the sound mixer service; other commands and
    // other processes are unsupported.
    CHECK(manager->request(request(700, {1, 1}), pid).disposition == IpcDisposition::unsupported &&
          manager->diagnostic());
    CHECK(manager->request(request(800, {1}), pid).disposition == IpcDisposition::unsupported);
    CHECK(manager->request(request(2), pid).disposition == IpcDisposition::unsupported);
    CHECK(manager->request(request(500), pid + 1).disposition == IpcDisposition::unsupported);
    std::printf("Audio device manager failures: %u\n", failures);
    return failures ? 1 : 0;
}
