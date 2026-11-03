// ============================================================================
// Armigo Services - Account Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>

#include <array>
#include <cstdint>
#include <vector>

namespace armigo::services::acc {

// The `acc:u0` application account service. The runtime hosts one local profile;
// the service reports that user for the account queries the application makes.
// The profile uuid is a host-local identity, not a console account id.
class AccountSession final : public horizon::ClientSession {
public:
    explicit AccountSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    const char* diagnostic() const;
    // The uuid this runtime reports for its single local profile.
    static const std::array<std::byte, 16>& local_user_id() noexcept;

private:
    horizon::IpcReply handle(const horizon::IpcMessage& message, std::uint64_t sender_pid,
                             const horizon::GuestBuffers* buffers);
    horizon::IpcReply unsupported(const char* detail);

    std::uint64_t process_id_;
    // Domain objects: id 1 is the account service itself; LoadOpenContext and
    // GetBaasAccountManagerForApplication add online-account manager objects and
    // GetProfile adds profile objects (also listed in profiles_).
    std::vector<std::uint32_t> objects_{1};
    std::vector<std::uint32_t> profiles_;
    const char* diagnostic_ = nullptr;
};

}  // namespace armigo::services::acc
