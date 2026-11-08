// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>

#include <cstdint>
#include <mutex>

namespace armigo::services::csrng {

// Cryptographically secure random bytes (`csrng`), non-domain. Command 0,
// GenerateRandomBytes, fills the caller's output buffer from the host's
// entropy source and returns no data. QueryPointerBufferSize answers 0. Any
// other command, domain conversion or a request without an output buffer is an
// explicit unsupported diagnostic.
class Session final : public horizon::ClientSession {
public:
    explicit Session(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply dispatch(const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers*);
    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::csrng
