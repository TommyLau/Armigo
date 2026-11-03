// ============================================================================
// Armigo Services - Offline Play Report Service
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>

namespace armigo::services::prepo {

// Process-bound application play reports. Accepted reports are discarded by
// the offline host; no report bytes or user identifiers survive the request.
class PlayReportSession final : public horizon::ClientSession {
public:
    explicit PlayReportSession(std::uint64_t process_id) : process_id_(process_id) {}

    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                          const horizon::GuestBuffers&) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply unsupported(const char* detail);

    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    bool domain_ = false;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::prepo
