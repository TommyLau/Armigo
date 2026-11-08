// ============================================================================
// Armigo Services - Switch 2 Audio User Client
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The Switch 2 "aud:u" session (nn::audio::CreateUserClientSessionByHipc).
// The SDK reaches the audio interfaces it shares with Switch 1 through it:
// command 1 hands out the audio output manager, command 2 the audio renderer
// manager, each taking one word the SDK supplies. Both follow the practical
// reference; the word's meaning is not established and is kept. Any other
// request is unsupported with an explicit diagnostic.

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/services/audio_renderer.h>

#include <cstdint>
#include <mutex>

namespace armigo::services::audio_user {

class UserClientSession final : public horizon::ClientSession {
public:
    UserClientSession(std::uint64_t process_id, audren::OutputFactory output_factory = {});
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;
    // The word of the latest manager request, by command (1 or 2).
    std::uint32_t manager_argument(unsigned command) const;

private:
    horizon::IpcReply unsupported(const char* detail);
    const std::uint64_t process_id_;
    const audren::OutputFactory output_factory_;
    mutable std::mutex mutex_;
    std::uint32_t arguments_[2]{};
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::audio_user
