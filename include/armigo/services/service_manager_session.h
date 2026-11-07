// ============================================================================
// Armigo Services - Initial Service Manager Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "armigo/horizon/ipc.h"
#include "armigo/services/service_manager.h"

#include <mutex>

namespace armigo::services::sm {

// Kernel-facing adapter for one configured process. The connector publishes
// real client sessions in that process's handle table before returning handles.
// Members serialize access; callbacks run under the lock and must not reenter.
class ServiceManagerSession final : public horizon::ClientSession {
public:
    ServiceManagerSession(ProcessConfig process, ConnectService connect,
        bool close_on_hipc = false);
    horizon::IpcReply close_server() override;

    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;

    // Call only upon an actual matching service-registration notification.
    // A handled return belongs to the original blocked request. The caller
    // retains its suspended context and delivers the reply exactly once; this
    // function never resumes a guest or restarts an already completed call.
    // A different name or a notification without a pending request produces
    // unsupported without calling the connector or changing the transaction.
    horizon::IpcReply service_registered(const ServiceName& name);

    // Static diagnostic text for the most recent unsupported operation, or
    // null. Malformed packets latch the session unusable: exact kernel close
    // results are not implemented and must not be invented by this adapter.
    const char* diagnostic() const;

private:
    horizon::IpcReply translate(const DispatchResult& result);

    mutable std::mutex mutex_;
    const std::uint64_t process_id_;
    const bool close_on_hipc_;
    std::optional<Session> session_;
    std::optional<ServiceName> pending_name_;
    const char* diagnostic_ = nullptr;
    bool malformed_ = false;
};

}  // namespace armigo::services::sm
