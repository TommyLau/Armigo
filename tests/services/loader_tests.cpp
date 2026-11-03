// ============================================================================
// Armigo Tests - Runtime-Loader Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/loader.h>

#include <cstdio>
#include <cstring>

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::services::ro::LoaderSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}
}  // namespace

int main() {
    LoaderSession session(41);
    IpcMessage message{};
    const auto domain = [&](std::uint32_t command) {
        message.fill(std::byte{0});
        put(message, 0, 6);
        put(message, 4, 12);
        put(message, 16, 0x00200001);
        put(message, 20, 1);
        put(message, 48, 0x49434653);
        put(message, 52, 1);
        put(message, 56, command);
    };
    domain(2);
    put(message, 64, 41);        // process id
    put(message, 72, 0x70000000);  // nrr address
    put(message, 84, 0x1000);      // nrr size
    CHECK(session.request(message, 41).disposition == IpcDisposition::handled);
    CHECK(session.module_count() == 1);

    // The same range registers once and the entry is replaced, not duplicated.
    domain(2);
    put(message, 64, 41);
    put(message, 72, 0x70000000);
    put(message, 84, 0x1000);
    CHECK(session.request(message, 41).disposition == IpcDisposition::handled);
    CHECK(session.module_count() == 1);

    // A registration for another process, or with an empty range, is refused.
    domain(2);
    put(message, 64, 42);
    put(message, 72, 0x70000000);
    put(message, 84, 0x1000);
    CHECK(session.request(message, 41).disposition == IpcDisposition::unsupported);
    domain(2);
    put(message, 64, 41);
    put(message, 72, 0x70000000);
    put(message, 84, 0);
    CHECK(session.request(message, 41).disposition == IpcDisposition::unsupported);

    // Unregistration removes a known range and refuses an unknown one.
    domain(3);
    put(message, 64, 0x70000000);
    CHECK(session.request(message, 41).disposition == IpcDisposition::handled);
    CHECK(session.module_count() == 0);
    domain(3);
    put(message, 64, 0x70000000);
    CHECK(session.request(message, 41).disposition == IpcDisposition::unsupported);

    // Commands that would load code into another process stay explicit.
    domain(0);
    CHECK(session.request(message, 41).disposition == IpcDisposition::unsupported);
    CHECK(session.diagnostic() != nullptr);
    // A request from another process is refused outright.
    domain(3);
    put(message, 64, 0x70000000);
    CHECK(session.request(message, 42).disposition == IpcDisposition::unsupported);

    if (failures == 0) std::fprintf(stderr, "Runtime-loader service failures: 0\n");
    return failures == 0 ? 0 : 1;
}
