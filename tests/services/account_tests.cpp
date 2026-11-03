// ============================================================================
// Armigo Tests - Account Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/account.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <span>

namespace {
using armigo::horizon::IpcDisposition;
using armigo::horizon::IpcMessage;
using armigo::services::acc::AccountSession;

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
std::uint64_t get64(const IpcMessage& message, std::size_t offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, message.data() + offset, 8);
    return value;
}

// A domain request with the standard header at byte16 and the CMIF header at byte32.
IpcMessage command(std::uint32_t id, std::uint32_t command_id) {
    IpcMessage message{};
    put(message, 0, 6);
    put(message, 4, 12);
    put(message, 16, 0x00100001);
    put(message, 20, id);
    put(message, 32, 0x49434653);
    put(message, 36, 1);
    put(message, 40, command_id);
    return message;
}

struct Buffer {
    std::array<std::byte, 64> bytes{};
    static bool read(void* context, std::uint64_t address, std::span<std::byte> value) noexcept {
        auto& self = *static_cast<Buffer*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        std::memcpy(value.data(), self.bytes.data() + address, value.size());
        return true;
    }
    static bool write(void* context, std::uint64_t address,
                      std::span<const std::byte> value) noexcept {
        auto& self = *static_cast<Buffer*>(context);
        if (address + value.size() > self.bytes.size()) return false;
        std::memcpy(self.bytes.data() + address, value.data(), value.size());
        return true;
    }
};
}  // namespace

int main() {
    AccountSession session(17);
    // Control command 3 reports the session pointer capacity; command 0 converts the domain.
    IpcMessage control{};
    put(control, 0, 7);
    put(control, 4, 8);
    put(control, 16, 0x49434653);
    put(control, 24, 3);
    auto reply = session.request(control, 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 32) == 0x400);
    put(control, 24, 0);
    reply = session.request(control, 17);
    CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 32) == 1);

    // A domain request whose descriptor area starts with a stale SFCI header:
    // the real CMIF header is at byte32 and must win over the stale sequence.
    IpcMessage stale{};
    put(stale, 0, 6);
    put(stale, 4, 0x00000c0d);  // inline descriptor flag
    put(stale, 16, 0x00100001); // domain header at 16
    put(stale, 20, 1);
    put(stale, 32, 0x49434653);
    put(stale, 36, 1);
    put(stale, 40, 2);          // ListAllUsers
    put(stale, 48, 0x49434653); // stale SFCI in the descriptor area
    put(stale, 52, 1);
    put(stale, 56, 0x40);
    put(stale, 60, 0);
    Buffer stale_buffer;
    armigo::horizon::GuestBuffers stale_buffers(Buffer::read, Buffer::write, &stale_buffer);
    stale_buffers.add({0, 0x40, false, true});
    reply = session.request_with_buffers(stale, 17, stale_buffers);
    CHECK(reply.disposition == IpcDisposition::handled);
    bool uuid_written = false;
    for (const auto byte : stale_buffer.bytes) uuid_written = uuid_written || byte != std::byte{0};
    CHECK(uuid_written);

    // GetUserCount: one local profile.
    reply = session.request(command(1, 0), 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 32) == 0x4f434653 && get(reply.bytes, 36) == 1);
    CHECK(get(reply.bytes, 48) == 1);

    // GetLastOpenedUser and TrySelectUserWithoutInteraction return the local uuid.
    const auto& local = AccountSession::local_user_id();
    for (const auto id : {4u, 52u}) {
        reply = session.request(command(1, id), 17);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(get(reply.bytes, 32) == 0x4f434653 && get(reply.bytes, 36) == 1);
        bool same = true;
        for (std::size_t index = 0; index < local.size(); ++index)
            same = same && reply.bytes[48 + index] == local[index];
        CHECK(same);
    }

    // Account switch lock is unlocked; the launch-info call succeeds with no output.
    for (const auto id : {150u, 160u}) {
        reply = session.request(command(1, id), 17);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(get(reply.bytes, 32) == 0x4f434653 && get(reply.bytes, 36) == 1);
        if (id == 150) CHECK(get(reply.bytes, 48) == 0);
        else CHECK(reply.size == 48);
    }

    // ListOpenContextStoredUsers writes the uuid into the caller's buffer.
    {
        Buffer buffer;
        armigo::horizon::GuestBuffers buffers(Buffer::read, Buffer::write, &buffer);
        buffers.add({0, 16, false, true});
        reply = session.request_with_buffers(command(1, 131), 17, buffers);
        CHECK(reply.disposition == IpcDisposition::handled);
        bool same = true;
        for (std::size_t index = 0; index < local.size(); ++index)
            same = same && buffer.bytes[index] == local[index];
        CHECK(same);
        // Without the buffer the command is refused instead of inventing data.
        reply = session.request(command(1, 131), 17);
        CHECK(reply.disposition == IpcDisposition::unsupported);
    }

    // LoadOpenContext hands out one domain object; its commands stay explicit.
    reply = session.request(command(1, 130), 17);
    CHECK(reply.disposition == IpcDisposition::handled);
    CHECK(get(reply.bytes, 16) == 1);
    const auto manager = get(reply.bytes, 48);
    CHECK(manager == 2);
    reply = session.request(command(manager, 0), 17);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    // GetBaasAccountManagerForApplication(uid) hands out another such object for
    // the local profile; an unknown user or a missing user id is refused.
    {
        const auto& local = AccountSession::local_user_id();
        auto baas = command(1, 101);
        put(baas, 4, 16);
        put(baas, 16, 0x00200001);
        std::memcpy(baas.data() + 48, local.data(), local.size());
        reply = session.request(baas, 17);
        CHECK(reply.disposition == IpcDisposition::handled);
        CHECK(get(reply.bytes, 16) == 1 && get(reply.bytes, 48) == manager + 1);
        baas[63] = std::byte{0x7f};
        CHECK(session.request(baas, 17).disposition == IpcDisposition::unsupported);
        CHECK(session.request(command(1, 101), 17).disposition == IpcDisposition::unsupported);

        // GetUserExistence(uid) is true only for the local profile.
        auto existence = command(1, 1);
        put(existence, 4, 16);
        put(existence, 16, 0x00200001);
        std::memcpy(existence.data() + 48, local.data(), local.size());
        reply = session.request(existence, 17);
        CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 48) == 1);
        existence[50] = std::byte{0x33};
        reply = session.request(existence, 17);
        CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 48) == 0);

        // GetProfile(uid) hands out a profile object whose base record names
        // the local profile; Get also clears the caller's user data.
        auto profile_request = command(1, 5);
        put(profile_request, 4, 16);
        put(profile_request, 16, 0x00200001);
        std::memcpy(profile_request.data() + 48, local.data(), local.size());
        reply = session.request(profile_request, 17);
        CHECK(reply.disposition == IpcDisposition::handled && get(reply.bytes, 16) == 1);
        const auto profile = get(reply.bytes, 48);
        reply = session.request(command(profile, 1), 17);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 32 + 16 + 0x38);
        CHECK(std::memcmp(reply.bytes.data() + 48, local.data(), local.size()) == 0);
        CHECK(std::memcmp(reply.bytes.data() + 48 + 24, "Armigo", 7) == 0);
        struct UserData {
            std::array<std::byte, 0x80> bytes{};
            static bool read(void*, std::uint64_t, std::span<std::byte>) noexcept { return false; }
            static bool write(void* context, std::uint64_t address,
                              std::span<const std::byte> value) noexcept {
                auto& self = *static_cast<UserData*>(context);
                if (address != 0 || value.size() > self.bytes.size()) return false;
                std::memcpy(self.bytes.data(), value.data(), value.size());
                return true;
            }
        } user_data;
        user_data.bytes.fill(std::byte{0xa5});
        armigo::horizon::GuestBuffers out(UserData::read, UserData::write, &user_data);
        out.add({0, 0x80, false, true});
        reply = session.request_with_buffers(command(profile, 0), 17, out);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 32 + 16 + 0x38);
        CHECK(std::all_of(user_data.bytes.begin(), user_data.bytes.end(),
                          [](std::byte value) { return value == std::byte{}; }));
        CHECK(session.request(command(profile, 0), 17).disposition == IpcDisposition::unsupported);
        CHECK(session.request(command(profile, 10), 17).disposition == IpcDisposition::unsupported);
        profile_request[63] = std::byte{0x7f};
        CHECK(session.request(profile_request, 17).disposition == IpcDisposition::unsupported);

        // Closing the profile releases it; a second close names no object.
        IpcMessage close{};
        put(close, 0, 4);
        put(close, 4, 8);
        put(close, 16, 2);
        put(close, 20, profile);
        reply = session.request(close, 17);
        CHECK(reply.disposition == IpcDisposition::handled && reply.size == 32);
        CHECK(get(reply.bytes, 16) == 0x4f434653 && get(reply.bytes, 24) == 0);
        CHECK(session.request(command(profile, 1), 17).disposition == IpcDisposition::unsupported);
        CHECK(session.request(close, 17).disposition == IpcDisposition::unsupported);
        put(close, 20, 1);
        CHECK(session.request(close, 17).disposition == IpcDisposition::unsupported);
    }
    // A session for another process refuses every request.
    reply = session.request(command(1, 0), 18);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    // An unimplemented command stays explicit.
    reply = session.request(command(1, 999), 17);
    CHECK(reply.disposition == IpcDisposition::unsupported);
    if (failures == 0) std::fprintf(stderr, "Account service failures: 0\n");
    return failures == 0 ? 0 : 1;
}
