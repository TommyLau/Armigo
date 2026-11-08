// ============================================================================
// Armigo Tests - Random-Number Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/csrng.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {
using armigo::horizon::GuestBuffers;
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

struct Memory {
    std::uint64_t base = 0x20000;
    std::vector<std::byte> bytes = std::vector<std::byte>(0x1000);
};

bool read_memory(void* context, std::uint64_t address, std::span<std::byte> out) noexcept {
    auto& memory = *static_cast<Memory*>(context);
    if (address < memory.base || address - memory.base + out.size() > memory.bytes.size()) return false;
    std::memcpy(out.data(), memory.bytes.data() + (address - memory.base), out.size());
    return true;
}

bool write_memory(void* context, std::uint64_t address, std::span<const std::byte> in) noexcept {
    auto& memory = *static_cast<Memory*>(context);
    if (address < memory.base || address - memory.base + in.size() > memory.bytes.size()) return false;
    std::memcpy(memory.bytes.data() + (address - memory.base), in.data(), in.size());
    return true;
}

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

std::uint32_t get(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}

IpcMessage request(std::uint32_t type, std::uint32_t command) {
    IpcMessage message{};
    message.fill(std::byte{0});
    put(message, 0, type);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    return message;
}
} // namespace

int main() {
    constexpr std::uint64_t pid = 0x51;
    armigo::services::csrng::Session session(pid);
    // QueryPointerBufferSize answers no pointer buffer.
    auto pointer = session.request(request(5, 3), pid);
    CHECK(pointer.disposition == IpcDisposition::handled && get(pointer.bytes, 32) == 0);
    // GenerateRandomBytes fills the whole output buffer, including a partial
    // final chunk, and nothing past it.
    Memory memory;
    GuestBuffers buffers(read_memory, write_memory, &memory);
    buffers.add({.address = memory.base + 16, .size = 600, .input = false, .output = true});
    const auto generated = session.request_with_buffers(request(4, 0), pid, buffers);
    CHECK(generated.disposition == IpcDisposition::handled && session.diagnostic() == nullptr);
    CHECK(get(generated.bytes, 16) == 0x4f434653 && get(generated.bytes, 24) == 0);
    const auto first = std::span(memory.bytes).subspan(16, 600);
    CHECK(std::count(first.begin(), first.end(), std::byte{0}) < 32);
    CHECK(std::all_of(memory.bytes.begin(), memory.bytes.begin() + 16, [](std::byte b) { return b == std::byte{0}; }));
    CHECK(std::all_of(memory.bytes.begin() + 616, memory.bytes.end(), [](std::byte b) { return b == std::byte{0}; }));
    // Without a buffer, for another command or another process: unsupported.
    CHECK(session.request(request(4, 0), pid).disposition == IpcDisposition::unsupported && session.diagnostic());
    CHECK(session.request_with_buffers(request(4, 1), pid, buffers).disposition == IpcDisposition::unsupported);
    CHECK(session.request_with_buffers(request(4, 0), pid + 1, buffers).disposition == IpcDisposition::unsupported);
    std::printf("Random-number failures: %u\n", failures);
    return failures ? 1 : 0;
}
