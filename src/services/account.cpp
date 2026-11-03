// ============================================================================
// Armigo Services - Account Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/account.h>

#include <algorithm>
#include <cstdio>
#include <string_view>

namespace armigo::services::acc {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint16_t session_pointer_capacity = 0x400;
// "Armigo" followed by a local profile number. This is a host-local identity for the
// single profile this runtime hosts, not a console account id.
constexpr std::array<std::byte, 16> local_uuid{
    std::byte{0x41}, std::byte{0x72}, std::byte{0x6d}, std::byte{0x69},
    std::byte{0x67}, std::byte{0x6f}, std::byte{0x00}, std::byte{0x01},
    std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
    std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x01}};

std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index < width; ++index)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + index])}
                 << (8 * index);
    return value;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned width = 4) {
    for (unsigned index = 0; index < width; ++index)
        message[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}


IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 32, value, width);
    return reply;
}

// A CMIF reply whose command data starts after the CMIF header. A domain request keeps its
// context header at byte32 and its CMIF at byte48; the response places the CMIF at byte32, the
// same shape the application service uses for its domain commands.
IpcReply command_reply(std::size_t cmif, std::size_t output_bytes = 0) {
    const auto response = cmif >= 32 ? std::size_t{32} : cmif;
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = response + 16 + output_bytes;
    write(reply.bytes, 4, (reply.size - 8) / 4);
    write(reply.bytes, response, sfco);
    write(reply.bytes, response + 4, 1);
    return reply;
}

void write_uuid(IpcReply& reply, std::size_t offset, const std::array<std::byte, 16>& uuid) {
    for (std::size_t index = 0; index < uuid.size(); ++index)
        reply.bytes[offset + index] = uuid[index];
}
}  // namespace

AccountSession::AccountSession(std::uint64_t process_id) : process_id_(process_id) {}

const std::array<std::byte, 16>& AccountSession::local_user_id() noexcept { return local_uuid; }

IpcReply AccountSession::unsupported(const char* detail) {
    diagnostic_ = detail;
    return {};
}

IpcReply AccountSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return handle(message, sender_pid, nullptr);
}

IpcReply AccountSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                              const horizon::GuestBuffers& buffers) {
    return handle(message, sender_pid, &buffers);
}

IpcReply AccountSession::handle(const IpcMessage& message, std::uint64_t sender_pid,
                                const horizon::GuestBuffers* buffers) {
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Account session belongs to another process");
    const auto type = read(message, 0, 2);
    if (type == 5 || type == 7) {
        if (read(message, 16) != sfci) return unsupported("Account control magic is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_reply(session_pointer_capacity, 2);
        if (command == 0) return control_reply(1, 4);
        return unsupported("Account control command is unsupported");
    }
    if (type != 4 && type != 6) return unsupported("Account request type is unsupported");
    if ((read(message, 16) & 0xff) == 2 && read(message, 32) != sfci) {
        // A domain close releases one object the service handed out; its reply
        // has no CMIF payload. The service object itself is never closed this way.
        const auto object = read(message, 20);
        const auto found = std::find(objects_.begin(), objects_.end(), object);
        if (object == 1 || found == objects_.end()) return unsupported("Account close names no object");
        objects_.erase(found);
        std::erase(profiles_, object);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, 6);
        write(reply.bytes, 16, sfco);
        return reply;
    }
    const auto located = horizon::locate_cmif(message);
    if (!located) return unsupported("Account request shape is unsupported");
    const auto cmif = *located;
    // The domain header precedes the CMIF header; its second word is the object id.
    const auto domain = cmif >= 32 ? cmif - 16 : cmif;
    const auto object = read(message, domain + 4);
    if (std::find(profiles_.begin(), profiles_.end(), object) != profiles_.end()) {
        // IProfile: the local profile's base record (uid, last edit time,
        // nickname) and, for Get, its user data, which this runtime leaves zero.
        const auto profile_command = read(message, cmif + 8);
        if (profile_command != 0 && profile_command != 1)
            return unsupported("Account profile command is not implemented");
        if (profile_command == 0) {
            if (buffers == nullptr || buffers->size() != 1 || (*buffers)[0].size < 0x80)
                return unsupported("Account profile user data needs its output buffer");
            const std::array<std::byte, 0x80> user_data{};
            if (!buffers->write(0, user_data)) return unsupported("Account profile buffer is not writable");
        }
        auto reply = command_reply(cmif, 0x38);
        const auto base = (cmif >= 32 ? std::size_t{32} : cmif) + 16;
        write_uuid(reply, base, local_uuid);
        constexpr std::string_view nickname = "Armigo";
        for (std::size_t index = 0; index < nickname.size(); ++index)
            reply.bytes[base + 24 + index] = static_cast<std::byte>(nickname[index]);
        return reply;
    }
    if (object != 1) {
        // The open-context manager object exists but its online-account commands are not
        // implemented yet. Report the command instead of answering with an invented value.
        return unsupported("Account manager command is not implemented");
    }
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 1: {
        // GetUserExistence(uid): only the local profile exists.
        if (std::size_t{read(message, 4) & 0x3ffu} * 4 + 16 < cmif + 32)
            return unsupported("Account user existence needs the user id");
        bool exists = true;
        for (std::size_t index = 0; index < local_uuid.size(); ++index)
            exists = exists && message[cmif + 16 + index] == local_uuid[index];
        auto reply = command_reply(cmif, 4);
        write(reply.bytes, (cmif >= 32 ? std::size_t{32} : cmif) + 16, exists ? 1 : 0);
        return reply;
    }
    case 5: {
        // GetProfile(uid): a profile object for the user, which must be the local profile.
        if (std::size_t{read(message, 4) & 0x3ffu} * 4 + 16 < cmif + 32)
            return unsupported("Account profile request needs the user id");
        for (std::size_t index = 0; index < local_uuid.size(); ++index)
            if (message[cmif + 16 + index] != local_uuid[index])
                return unsupported("Account profile names a user that does not exist");
        std::uint32_t id = 1;
        for (const auto current : objects_) id = current > id ? current : id;
        id += 1;
        objects_.push_back(id);
        profiles_.push_back(id);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 60;
        write(reply.bytes, 4, 13);
        write(reply.bytes, 16, 1);
        write(reply.bytes, 32, sfco);
        write(reply.bytes, 36, 1);
        write(reply.bytes, 48, id);
        return reply;
    }
    case 101:
    case 130: {
        // GetBaasAccountManagerForApplication(uid) (101) and LoadOpenContext(uid) (130) hand
        // the caller an online-account manager object for the user. The runtime hosts no
        // online account, so the object is created for the local profile; its commands stay
        // explicit until the run reaches them. The one user that exists is the local profile.
        if (command == 101) {
            if (std::size_t{read(message, 4) & 0x3ffu} * 4 + 16 < cmif + 32)
                return unsupported("Account manager request needs the user id");
            for (std::size_t index = 0; index < local_uuid.size(); ++index)
                if (message[cmif + 16 + index] != local_uuid[index])
                    return unsupported("Account manager names a user that does not exist");
        }
        std::uint32_t id = 1;
        for (const auto current : objects_) id = current > id ? current : id;
        id += 1;
        objects_.push_back(id);
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 60;
        write(reply.bytes, 4, 13);
        write(reply.bytes, 16, 1);
        write(reply.bytes, 32, sfco);
        write(reply.bytes, 36, 1);
        write(reply.bytes, 48, id);
        return reply;
    }
    case 0: {
        // GetUserCount: the runtime hosts one local profile.
        auto reply = command_reply(cmif, 4);
        write(reply.bytes, (cmif >= 32 ? std::size_t{32} : cmif) + 16, 1);
        return reply;
    }
    case 2:
    case 3:
    case 131: {
        // ListAllUsers / ListOpenUsers / ListOpenContextStoredUsers: the caller supplies the
        // uid array buffer and asks for the count separately. The single local profile is
        // written when the buffer has room for one uuid.
        if (buffers == nullptr || buffers->empty())
            return unsupported("Account user list needs the caller's uid buffer");
        if ((*buffers)[0].size < 16) return unsupported("Account user-list buffer is too small");
        std::array<std::byte, 16> uuid{};
        for (std::size_t index = 0; index < uuid.size(); ++index) uuid[index] = local_uuid[index];
        if (!buffers->write(0, uuid)) return unsupported("Account user-list buffer is not writable");
        return command_reply(cmif);
    }
    case 4: {
        // GetLastOpenedUser: the one local profile is the last opened user.
        auto reply = command_reply(cmif, 16);
        write_uuid(reply, (cmif >= 32 ? std::size_t{32} : cmif) + 16, local_uuid);
        return reply;
    }
    case 52: {
        // TrySelectUserWithoutInteraction: the caller does not choose; with one profile the
        // runtime returns it.
        auto reply = command_reply(cmif, 16);
        write_uuid(reply, (cmif >= 32 ? std::size_t{32} : cmif) + 16, local_uuid);
        return reply;
    }
    case 150: {
        // IsUserAccountSwitchLocked: the runtime has no NACP control metadata, so the
        // application's account is not locked. The value is the NACP policy, not a game
        // identity, and a default launch configuration leaves it unlocked.
        auto reply = command_reply(cmif, 4);
        write(reply.bytes, (cmif >= 32 ? std::size_t{32} : cmif) + 16, 0);
        return reply;
    }
    case 160: {
        // InitializeApplicationInfoV2: the account service records the launch information.
        // The runtime hosts one local profile and has no per-title account state yet, so the
        // call succeeds without further work.
        return command_reply(cmif);
    }
    default:
        break;
    }
    return unsupported("Account command is not implemented");
}

const char* AccountSession::diagnostic() const { return diagnostic_; }

}  // namespace armigo::services::acc
