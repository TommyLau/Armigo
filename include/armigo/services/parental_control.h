// ============================================================================
// Armigo Services - Initial Parental Control Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>

namespace armigo::services::pctl {

// Startup endpoint for a launch-authorized application client. Supports only
// framework controls, domain CreateServiceWithoutInitialize1 and child
// Initialize1. Permission queries, parental settings and policy enforcement
// remain unsupported; initialization does not grant any such permission.
class PctlSession final : public horizon::ClientSession {
public:
    explicit PctlSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;

private:
    struct Object {
        std::uint64_t process_id;
        bool initialized = false;
        std::shared_ptr<Object> parent;
    };
    const std::uint64_t process_id_;
    const std::shared_ptr<Object> root_;
    // Index+1 is the session-scoped domain ID of an owned object.
    std::vector<std::shared_ptr<Object>> objects_;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::pctl
