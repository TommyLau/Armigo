// ============================================================================
// Armigo Tests - NFC / amiibo service protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/services/nfp.h>

#include <cstdio>
#include <cstring>
#include <memory>

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::services::nfp::ManagerSession;
using armigo::services::nfp::UserSession;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); ++failures; } } while (0)

void put(IpcMessage& message, std::size_t offset, std::uint32_t value) {
    std::memcpy(message.data() + offset, &value, 4);
}
std::uint32_t get(const IpcMessage& message, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, message.data() + offset, 4);
    return value;
}
// A CMIF control request: type 7, the CMIF header at byte16, no domain header.
IpcMessage control(std::uint32_t command_id) {
    IpcMessage message{};
    put(message, 0, 7);
    put(message, 4, 8);
    put(message, 16, 0x49434653);
    put(message, 20, 1);
    put(message, 24, command_id);
    return message;
}
// A domain request for one object: domain header at byte16, CMIF at byte32.
IpcMessage domain(std::uint32_t object, std::uint32_t command_id) {
    IpcMessage message{};
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, object);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, command_id);
    return message;
}
}  // namespace

int main() {
    ManagerSession manager(17);
    // CMIF control: pointer-buffer size, then domain conversion.
    auto reply = manager.request(control(3), 17);
    CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 32) == 0);
    // Type-7 command 0 converts the manager session to a domain.
    reply = manager.request(control(0), 17);
    CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 32) == 1);
    CHECK(manager.converted());
    CHECK(manager.request(control(0), 17).disposition == IpcDisposition::unsupported);
    // A clone keeps the domain state on a second session.
    reply = manager.request(control(2), 17);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1);
    CHECK(std::dynamic_pointer_cast<ManagerSession>(reply.transferred[0]) != nullptr);

    // CreateUserInterface (manager command 0 in a domain) returns an IUser
    // object id and the object's commands are routed through the manager.
    reply = manager.request(domain(1, 0), 17);
    CHECK(reply.disposition == IpcDisposition::handled && reply.size == 60);
    CHECK(get(reply.bytes, 48) == 2 && manager.user_count() == 1);
    CHECK(manager.request(domain(1, 1), 17).disposition == IpcDisposition::unsupported);
    CHECK(manager.request(domain(9, 0), 17).disposition == IpcDisposition::unsupported);

    // A plain command on the manager session is the root IUser's command.
    UserSession direct(17);
    IpcMessage plain{};
    put(plain, 0, 6);
    put(plain, 4, 8);
    put(plain, 16, 0x49434653);
    put(plain, 20, 1);
    put(plain, 24, 0);  // Initialize
    reply = manager.request(plain, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    auto list = plain;
    put(list, 24, 2);  // ListDevices
    reply = manager.request(list, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 24) == UserSession::device_not_found);
    reply = direct.request(domain(2, 2), 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 40) == UserSession::device_not_found);
    // IUser commands on the domain object created by CreateUserInterface.
    reply = manager.request(domain(2, 0), 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    reply = manager.request(domain(2, 2), 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 40) == UserSession::device_not_found);
    reply = manager.request(domain(2, 23), 17);
    CHECK(reply.disposition == IpcDisposition::handled && reply.transferred_count == 1);
    CHECK(std::dynamic_pointer_cast<armigo::horizon::Event>(reply.transferred[0]) != nullptr);

    // A domain object release (type 6, nonzero object count, no CMIF header) is
    // answered with a plain success.
    {
        IpcMessage release{};
        put(release, 0, 6);
        put(release, 4, 8);
        put(release, 16, 2);
        put(release, 20, 1);
        auto reply = manager.request(release, 17);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 32);
        CHECK(get(reply.bytes, 16) == 0x4f434653);
        reply = direct.request(release, 17);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 32);
    }

    // Another process and an unimplemented IUser command stay explicit.
    CHECK(manager.request(domain(2, 0), 18).disposition == IpcDisposition::unsupported);
    reply = manager.request(domain(2, 3), 17);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    CHECK(manager.diagnostic() != nullptr);

    std::printf("NFP service failures: %u\n", failures);
    return failures ? 1 : 0;
}
