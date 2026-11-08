// ============================================================================
// Armigo Tests - SSL
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// ssl: the interface version, contexts with options and PKI imports, as a
// plain session and as a CMIF domain; connections stay unsupported offline.

#include <armigo/services/ssl.h>

#include <cstdio>
#include <cstring>
#include <memory>

namespace {
using namespace armigo::horizon;
using armigo::services::ssl::Context;
using armigo::services::ssl::SslSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) { std::memcpy(message.data() + offset, &value, 4); }
std::uint32_t word(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}

// A request with its CMIF header at `cmif` (a domain header 16 bytes before
// it when `object` is nonzero) and inputs after the CMIF header.
IpcMessage request(std::uint32_t command, std::initializer_list<std::uint32_t> inputs, std::uint32_t object = 0) {
    IpcMessage message{};
    put(message, 0, 4);
    put(message, 4, 0x14);
    const std::size_t cmif = object ? 32 : 16;
    if (object) {
        put(message, 16, 1 | 0x20u << 16);
        put(message, 20, object);
    }
    put(message, cmif, 0x49434653);
    put(message, cmif + 4, 1);
    put(message, cmif + 8, command);
    std::size_t offset = cmif + 16;
    for (const auto value : inputs) put(message, offset, value), offset += 4;
    return message;
}

struct Memory {
    static bool read(void*, std::uint64_t, std::span<std::byte>) noexcept { return true; }
    static bool write(void*, std::uint64_t, std::span<const std::byte>) noexcept { return true; }
};
} // namespace

int main() {
    constexpr std::uint64_t pid = 0x81;
    SslSession service(pid);
    CHECK(service.request(request(5, {0x40001}), pid).disposition == IpcDisposition::handled);
    CHECK(service.interface_version() == 0x40001u);
    auto created = service.request(request(0, {2, 0, 0, 0}), pid);
    CHECK(created.disposition == IpcDisposition::handled && created.transferred_count == 1);
    const auto context = std::dynamic_pointer_cast<Context>(created.transferred[0]);
    CHECK(context != nullptr && context->version() == 2);
    if (!context) return 1;
    CHECK(context->request(request(0, {1, 7}), pid).disposition == IpcDisposition::handled);
    auto option = context->request(request(1, {1}), pid);
    CHECK(option.disposition == IpcDisposition::handled && word(option.bytes, 32) == 7);
    CHECK(word(context->request(request(1, {9}), pid).bytes, 32) == 0);
    CHECK(word(context->request(request(3, {}), pid).bytes, 32) == 0);
    GuestBuffers buffers(Memory::read, Memory::write, nullptr);
    buffers.add({0x1000, 0x100, true, false});
    CHECK(word(context->request_with_buffers(request(4, {0}), pid, buffers).bytes, 32) == 1);
    CHECK(word(context->request_with_buffers(request(5, {}), pid, buffers).bytes, 32) == 2);
    CHECK(context->request(request(4, {0}), pid).disposition == IpcDisposition::unsupported);
    CHECK(context->request(request(2, {}), pid).disposition == IpcDisposition::unsupported && context->diagnostic());
    CHECK(service.request(request(1, {}), pid).disposition == IpcDisposition::unsupported && service.diagnostic());
    CHECK(service.request(request(5, {1}), pid + 1).disposition == IpcDisposition::unsupported);

    // As a domain: the service is object 1 and contexts are new objects.
    SslSession domain(pid);
    auto convert = request(0, {});
    put(convert, 0, 7);
    CHECK(word(domain.request(convert, pid).bytes, 32) == 1);
    created = domain.request(request(0, {2, 0, 0, 0}, 1), pid);
    CHECK(created.disposition == IpcDisposition::handled && created.transferred_count == 0 &&
          word(created.bytes, 16) == 1 && word(created.bytes, 48) == 2);
    CHECK(domain.request(request(0, {3, 5}, 2), pid).disposition == IpcDisposition::handled);
    CHECK(word(domain.request(request(1, {3}, 2), pid).bytes, 48) == 5);
    IpcMessage close{};
    put(close, 0, 4);
    put(close, 4, 8);
    put(close, 16, 2);
    put(close, 20, 2);
    CHECK(domain.request(close, pid).disposition == IpcDisposition::handled);
    CHECK(domain.request(request(1, {3}, 2), pid).disposition == IpcDisposition::unsupported);
    std::printf("SSL failures: %u\n", failures);
    return failures ? 1 : 0;
}
