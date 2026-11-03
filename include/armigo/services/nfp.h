// ============================================================================
// Armigo Services - NFC / amiibo service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace armigo::services::nfp {

// The IUser interface (manager command 0 CreateUserInterface returns one). The
// runtime has no NFC reader, so Initialize/Finalize succeed, state queries
// report no mounted device and ListDevices returns the firmware's
// device-not-found Result without touching the caller's outputs.
class UserSession final : public horizon::ClientSession {
public:
    explicit UserSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;

    bool initialized() const noexcept;
    const char* diagnostic() const noexcept { return diagnostic_; }

    // NFP ResultDeviceNotFound, the unavailable-reader Result the game's caller
    // explicitly handles.
    static constexpr std::uint32_t device_not_found = 0x8073;

private:
    horizon::IpcReply handle(const horizon::IpcMessage& message, std::uint64_t sender_pid);
    horizon::IpcReply unsupported(const char* detail);
    std::uint64_t process_id_ = 0;
    mutable std::mutex mutex_;
    bool initialized_ = false;
    std::shared_ptr<horizon::Event> availability_event_;
    const char* diagnostic_ = nullptr;
};

// `nfp:user` is the IUserManager. It is a CMIF domain: type-5/7 control
// commands 0-4 convert/copy/clone the session and report the pointer buffer
// size; command 0 then creates an IUser object and later requests carry that
// object's id in the domain header.
class ManagerSession final : public horizon::ClientSession {
public:
    explicit ManagerSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;

    bool converted() const noexcept;
    std::size_t user_count() const noexcept;
    const char* diagnostic() const noexcept { return diagnostic_; }

private:
    horizon::IpcReply handle(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                             const horizon::GuestBuffers* buffers);
    horizon::IpcReply control(const horizon::IpcMessage& message);
    horizon::IpcReply unsupported(const char* detail);
    std::uint64_t process_id_ = 0;
    mutable std::mutex mutex_;
    bool converted_ = false;
    std::uint32_t next_object_id_ = 1;
    std::uint32_t clone_tag_ = 0;
    std::shared_ptr<UserSession> root_user_;
    std::vector<std::shared_ptr<UserSession>> users_;
    const char* diagnostic_ = nullptr;
};

}  // namespace armigo::services::nfp
