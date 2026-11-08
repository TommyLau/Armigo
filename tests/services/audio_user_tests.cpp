// ============================================================================
// Armigo Tests - Switch 2 Audio User Client
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/audio_out.h>
#include <armigo/services/audio_user.h>

#include <cstdio>
#include <cstring>
#include <memory>

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::services::audio_user::UserClientSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}

IpcMessage request(std::uint32_t command, std::uint32_t argument) {
    IpcMessage message{};
    message.fill(std::byte{0});
    put(message, 0, 4);
    put(message, 4, 0x9);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command);
    put(message, 32, argument);
    return message;
}
} // namespace

int main() {
    constexpr std::uint64_t pid = 0x81;
    UserClientSession session(pid);
    // Command 2: the audio renderer manager, moved to the caller.
    auto renderer = session.request(request(2, 7), pid);
    CHECK(renderer.disposition == IpcDisposition::handled && renderer.transferred_count == 1 &&
          !renderer.transferred_as_copy);
    CHECK(std::dynamic_pointer_cast<armigo::services::audren::ManagerSession>(renderer.transferred[0]) != nullptr);
    CHECK(session.manager_argument(2) == 7);
    // Command 1: the audio output manager.
    auto output = session.request(request(1, 3), pid);
    CHECK(output.disposition == IpcDisposition::handled && output.transferred_count == 1);
    CHECK(std::dynamic_pointer_cast<armigo::services::audout::ManagerSession>(output.transferred[0]) != nullptr);
    CHECK(session.manager_argument(1) == 3);
    // Anything else, and another process, stay unsupported with a diagnostic.
    CHECK(session.request(request(3, 0), pid).disposition == IpcDisposition::unsupported && session.diagnostic());
    CHECK(session.request(request(2, 0), pid + 1).disposition == IpcDisposition::unsupported);
    std::printf("Audio user client failures: %u\n", failures);
    return failures ? 1 : 0;
}
