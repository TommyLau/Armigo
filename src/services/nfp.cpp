// ============================================================================
// Armigo Services - NFC / amiibo service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/nfp.h>

namespace armigo::services::nfp {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint32_t control_interface = 0xec6be3ff;

std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned index = 0; index != width; ++index)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + index])}
                 << (8 * index);
    return value;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned width = 4) {
    for (unsigned index = 0; index != width; ++index)
        message[offset + index] = static_cast<std::byte>((value >> (8 * index)) & 255);
}

// A domain request's reply carries the domain out header at byte16 and the CMIF
// header at byte32, like the other domain services; a plain request keeps its
// CMIF header at the same offset as the request.
std::size_t response_offset(std::size_t cmif) noexcept {
    return cmif >= 32 ? 32 : cmif;
}

IpcReply success_reply(std::size_t cmif) {
    const auto response = response_offset(cmif);
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = response + 16;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, response, sfco);
    write(reply.bytes, response + 4, 1);
    return reply;
}

IpcReply value_reply(std::size_t cmif, std::uint32_t value) {
    const auto response = response_offset(cmif);
    IpcReply reply = success_reply(cmif);
    reply.size = response + 20;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, response + 16, value);
    return reply;
}

IpcReply error_reply(std::size_t cmif, std::uint32_t result) {
    const auto response = response_offset(cmif);
    IpcReply reply = success_reply(cmif);
    write(reply.bytes, response + 8, result);
    return reply;
}

IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 28, control_interface);
    write(reply.bytes, 32, value, width);
    return reply;
}


// A domain object release arrives as a type-4/6 message whose domain header has
// a nonzero object count and no CMIF header; the reply is a plain success.
bool release_request(const IpcMessage& message) {
    if (read(message, 0, 2) != 4 && read(message, 0, 2) != 6) return false;
    const auto header = read(message, 16);
    const auto objects = header & 0xffff;
    return objects >= 1 && objects <= 8 && (header >> 16) == 0;
}

IpcReply release_reply() {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 32;
    write(reply.bytes, 4, 6);
    write(reply.bytes, 16, sfco);
    return reply;
}

bool is_control(const IpcMessage& message) {
    const auto type = read(message, 0, 2);
    return (type == 5 || type == 7) && read(message, 16) == sfci;
}
}  // namespace

UserSession::UserSession(std::uint64_t process_id) : process_id_(process_id) {}

bool UserSession::initialized() const noexcept {
    const std::lock_guard lock(mutex_);
    return initialized_;
}

IpcReply UserSession::unsupported(const char* detail) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = detail;
    return {};
}

IpcReply UserSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return handle(message, sender_pid);
}

IpcReply UserSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) {
    (void)buffers;
    return handle(message, sender_pid);
}

IpcReply UserSession::handle(const IpcMessage& message, std::uint64_t sender_pid) {
    {
        const std::lock_guard lock(mutex_);
        diagnostic_ = nullptr;
    }
    if (sender_pid != process_id_) return unsupported("NFP IUser belongs to another process");
    if (is_control(message)) {
        // Domain objects are not converted: their commands come as domain
        // requests. A control command here would be a protocol error.
        return unsupported("NFP IUser control request is unsupported");
    }
    if (release_request(message)) return release_reply();
    const auto type = read(message, 0, 2);
    if (type != 4 && type != 6) return unsupported("NFP IUser request type is unsupported");
    const auto located = horizon::locate_cmif(message);
    if (!located) return unsupported("NFP IUser request shape is unsupported");
    const auto cmif = *located;
    const auto command = read(message, cmif + 8);
    switch (command) {
    case 0: {  // Initialize
        const std::lock_guard lock(mutex_);
        initialized_ = true;
        return success_reply(cmif);
    }
    case 1: {  // Finalize
        const std::lock_guard lock(mutex_);
        initialized_ = false;
        return success_reply(cmif);
    }
    case 2:  // ListDevices: both outputs stay untouched on the unavailable-reader path
        return error_reply(cmif, device_not_found);
    case 19:  // GetState: no mounted device
        return value_reply(cmif, 0);
    case 20:  // GetDeviceState: no mounted device
        return value_reply(cmif, 0);
    case 21:  // GetNpadId: no mounted device
        return error_reply(cmif, device_not_found);
    case 23: {  // AttachAvailabilityChangeEvent
        std::shared_ptr<horizon::Event> event;
        {
            const std::lock_guard lock(mutex_);
            if (!availability_event_) availability_event_ = std::make_shared<horizon::Event>(false);
            event = availability_event_;
        }
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = 32;
        write(reply.bytes, 4, (reply.size - 8) / 4);
        write(reply.bytes, 16, sfco);
        reply.transferred[0] = std::move(event);
        reply.transferred_count = 1;
        reply.transferred_as_copy = true;
        return reply;
    }
    default:
        break;
    }
    return unsupported("NFP IUser command is not implemented");
}

ManagerSession::ManagerSession(std::uint64_t process_id) : process_id_(process_id) {}

bool ManagerSession::converted() const noexcept {
    const std::lock_guard lock(mutex_);
    return converted_;
}

std::size_t ManagerSession::user_count() const noexcept {
    const std::lock_guard lock(mutex_);
    return users_.size();
}

IpcReply ManagerSession::unsupported(const char* detail) {
    const std::lock_guard lock(mutex_);
    diagnostic_ = detail;
    return {};
}

IpcReply ManagerSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return handle(message, sender_pid, nullptr);
}

IpcReply ManagerSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                              const horizon::GuestBuffers& buffers) {
    return handle(message, sender_pid, &buffers);
}

IpcReply ManagerSession::control(const IpcMessage& message) {
    const auto command = read(message, 24);
    switch (command) {
    case 3:  // QueryPointerBufferSize
        return control_reply(0, 2);
    case 0:  // ConvertCurrentObjectToDomain
        if (converted_) return unsupported("NFP domain conversion was already performed");
        converted_ = true;
        return control_reply(1, 4);
    case 2:   // CloneCurrentObject
    case 4: {  // CloneCurrentObjectEx
        // The clone is a second session on the same port; it shares the domain
        // state, so the IUser objects it created stay reachable.
        if (command == 4) clone_tag_ = read(message, 32);
        auto clone = std::make_shared<ManagerSession>(process_id_);
        clone->converted_ = converted_;
        clone->users_ = users_;
        clone->next_object_id_ = next_object_id_;
        IpcReply reply = control_reply(0, 4);
        reply.transferred[0] = std::move(clone);
        reply.transferred_count = 1;
        return reply;
    }
    default:
        break;
    }
    return unsupported("NFP manager control command is unsupported");
}

IpcReply ManagerSession::handle(const IpcMessage& message, std::uint64_t sender_pid,
                                const horizon::GuestBuffers* buffers) {
    {
        const std::lock_guard lock(mutex_);
        diagnostic_ = nullptr;
    }
    if (sender_pid != process_id_) return unsupported("NFP manager session belongs to another process");
    if (is_control(message)) return control(message);
    if (release_request(message)) return release_reply();
    const auto type = read(message, 0, 2);
    if (type != 4 && type != 6) return unsupported("NFP manager request type is unsupported");
    const auto located = horizon::locate_cmif(message);
    if (!located) return unsupported("NFP manager request shape is unsupported");
    const auto cmif = *located;
    const auto command = read(message, cmif + 8);
    // A plain CMIF command (no domain header) on the manager session is the
    // root IUser's command: the generated client uses the session it already
    // holds for ListDevices and the state queries. CreateUserInterface remains
    // available as a domain command that returns a second IUser object.
    if (cmif < 32) {
        std::shared_ptr<UserSession> user;
        {
            const std::lock_guard lock(mutex_);
            if (!root_user_) root_user_ = std::make_shared<UserSession>(process_id_);
            user = root_user_;
        }
        auto reply = user->request(message, sender_pid);
        if (reply.disposition != IpcDisposition::handled) {
            const std::lock_guard lock(mutex_);
            diagnostic_ = user->diagnostic() != nullptr ? user->diagnostic()
                                                        : "NFP IUser command is not implemented";
        }
        return reply;
    }
    const auto domain = cmif - 16;
    const auto header = read(message, domain);
    if ((header & 0xffff) != 1 || (header >> 16) < 16)
        return unsupported("NFP manager domain header is unsupported");
    const auto object = read(message, domain + 4);
    if (object == 1) {
        if (command != 0) return unsupported("NFP manager command is not implemented");
        // CreateUserInterface: add an IUser object and return its domain object id.
        const auto id = next_object_id_ + 1;
        users_.push_back(std::make_shared<UserSession>(process_id_));
        ++next_object_id_;
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
    if (object < 2 || object > next_object_id_)
        return unsupported("NFP manager request names an unknown IUser object");
    const auto& user = users_[object - 2];
    if (!user) return unsupported("NFP manager request names an unknown IUser object");
    (void)buffers;
    auto reply = user->request(message, sender_pid);
    if (reply.disposition != IpcDisposition::handled) {
        const std::lock_guard lock(mutex_);
        diagnostic_ = user->diagnostic() != nullptr ? user->diagnostic()
                                                    : "NFP IUser command is not implemented";
    }
    return reply;
}

}  // namespace armigo::services::nfp
