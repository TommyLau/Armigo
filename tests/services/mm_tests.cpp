// ============================================================================
// Armigo Tests - Multimedia Clock Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/mm.h>

#include <array>
#include <cstdio>
#include <span>

namespace {
using namespace armigo::horizon;
using armigo::services::mm::Session;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "Failed line %d: %s\n", __LINE__, #c); ++failures; } } while (false)
constexpr std::uint64_t pid = 9;

void put(IpcMessage& bytes, unsigned offset, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}
std::uint32_t word(const IpcMessage& bytes, unsigned offset) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= std::uint32_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}
IpcMessage command(std::uint32_t id, std::span<const std::uint32_t> inputs = {}) {
    IpcMessage bytes{};
    put(bytes, 0, 4);
    put(bytes, 4, 8 + static_cast<std::uint32_t>(inputs.size()));
    put(bytes, 16, 0x49434653);
    put(bytes, 20, 1);
    put(bytes, 24, id);
    for (std::size_t i = 0; i < inputs.size(); ++i) put(bytes, 32 + 4 * static_cast<unsigned>(i), inputs[i]);
    return bytes;
}
} // namespace

int main() {
    Session session(pid);
    // Two requests get distinct ids; each records its own range.
    const std::array<std::uint32_t, 3> nvdec{5, 0, 1};
    auto first = session.request(command(4, nvdec), pid);
    auto second = session.request(command(4, nvdec), pid);
    CHECK(first.disposition == IpcDisposition::handled && second.disposition == IpcDisposition::handled);
    const auto a = word(first.bytes, 32), b = word(second.bytes, 32);
    CHECK(a != b);
    const std::array<std::uint32_t, 3> range{a, 192000000, 0xffffffffu};
    CHECK(session.request(command(6, range), pid).disposition == IpcDisposition::handled);
    const std::array<std::uint32_t, 1> get_a{a}, get_b{b}, unknown{77};
    CHECK(word(session.request(command(7, get_a), pid).bytes, 32) == 192000000);
    CHECK(word(session.request(command(7, get_b), pid).bytes, 32) == 0);
    CHECK(word(session.request(command(7, unknown), pid).bytes, 32) == 0);
    CHECK(session.request(command(5, get_a), pid).disposition == IpcDisposition::handled);
    CHECK(word(session.request(command(7, get_a), pid).bytes, 32) == 0);
    // Old module-keyed commands, missing inputs and other processes are refused.
    CHECK(session.request(command(0, nvdec), pid).disposition == IpcDisposition::unsupported);
    CHECK(session.request(command(4), pid).disposition == IpcDisposition::unsupported);
    CHECK(session.request(command(7, get_b), pid + 1).disposition == IpcDisposition::unsupported);
    if (failures) return 1;
    std::puts("Multimedia clock service checks passed");
    return 0;
}
