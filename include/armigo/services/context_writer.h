// ============================================================================
// Armigo Services - Initial Error Context Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>

namespace armigo::services::err {

// A real process-bound ectx:aw endpoint with only framework acquisition
// controls. No pointer buffer is provisioned. Context registration, commit,
// report generation and all service methods remain explicitly unsupported.
class ContextWriterSession final : public horizon::ClientSession {
public:
    explicit ContextWriterSession(std::uint64_t process_id) : process_id_(process_id) {}
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;

private:
    const std::uint64_t process_id_;
    // Domain ID1 names this session's root service object.
    bool domain_ = false;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::err
