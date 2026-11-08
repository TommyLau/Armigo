// ============================================================================
// Armigo Tests - Name Service Directory
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// nsd:u: the production environment and name resolution with it.

#include <armigo/services/nsd.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace {
using namespace armigo::horizon;
using armigo::services::nsd::NsdSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) { std::memcpy(message.data() + offset, &value, 4); }
std::uint32_t word(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}
IpcMessage request(std::uint32_t command, std::uint32_t input = 0) {
    IpcMessage message{};
    put(message, 0, 4);
    put(message, 4, 0x10);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    put(message, 32, input);
    return message;
}
struct Memory {
    std::array<std::byte, 0x400> bytes{};
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
    std::string_view text(std::size_t at) const { return reinterpret_cast<const char*>(bytes.data() + at); }
};
} // namespace

int main() {
    constexpr std::uint64_t pid = 0x81;
    NsdSession service(pid);
    Memory memory;
    GuestBuffers out(Memory::read, Memory::write, &memory);
    out.add({0x300, 8, false, true});
    CHECK(service.request_with_buffers(request(11), pid, out).disposition == IpcDisposition::handled);
    CHECK(memory.text(0x300) == "lp1");
    CHECK(word(service.request(request(100), pid).bytes, 32) == 1);
    CHECK(service.request(request(15, 1), pid).disposition == IpcDisposition::handled &&
          service.environment_change_disabled() == true);

    const auto resolve = [&](std::uint32_t command, std::string_view name) {
        std::memset(memory.bytes.data(), 0, 0x200);
        std::memcpy(memory.bytes.data(), name.data(), name.size());
        GuestBuffers buffers(Memory::read, Memory::write, &memory);
        buffers.add({0, 0x100, true, false});
        buffers.add({0x100, 0x100, false, true});
        return service.request_with_buffers(request(command), pid, buffers);
    };
    auto reply = resolve(20, "api-%.npln.srv.nintendo.net");
    CHECK(reply.disposition == IpcDisposition::handled && word(reply.bytes, 24) == 0);
    CHECK(memory.text(0x100) == "api-lp1.npln.srv.nintendo.net");
    reply = resolve(21, "plain.example");
    CHECK(word(reply.bytes, 24) == 0 && word(reply.bytes, 32) == 0 && memory.text(0x100) == "plain.example");
    // A result that does not fit is the module's overflow result.
    reply = resolve(21, std::string(0xfe, '%'));
    CHECK(word(reply.bytes, 24) == 0 && word(reply.bytes, 32) == (141u | 2u << 9));
    CHECK(service.request(request(5), pid).disposition == IpcDisposition::unsupported && service.diagnostic());
    std::printf("Name service failures: %u\n", failures);
    return failures ? 1 : 0;
}
