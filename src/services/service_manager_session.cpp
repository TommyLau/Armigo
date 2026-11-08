// ============================================================================
// Armigo Services - Initial Service Manager Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/services/service_manager_session.h"

#include <algorithm>
#include <utility>

namespace armigo::services::sm {

ServiceManagerSession::ServiceManagerSession(ProcessConfig process, ConnectService connect)
    : process_id_(process.process_id), session_(std::in_place, std::move(process), std::move(connect)) {}

horizon::IpcReply ServiceManagerSession::close_server() {
    const std::lock_guard lock(mutex_);
    if (!session_) return {.disposition = horizon::IpcDisposition::handled,
                           .transport_result = horizon::session_closed_result};
    session_.reset();
    pending_name_.reset();
    diagnostic_ = "Service-manager server session is closed";
    return {.disposition = horizon::IpcDisposition::handled};
}

horizon::IpcReply ServiceManagerSession::translate(const DispatchResult& result) {
    horizon::IpcReply reply;
    diagnostic_ = result.detail;
    switch (result.disposition) {
    case Disposition::handled:
        reply.disposition = horizon::IpcDisposition::handled;
        reply.size = result.reply_size;
        std::copy_n(result.reply.begin(), result.reply_size, reply.bytes.begin());
        pending_name_.reset();
        break;
    case Disposition::pending:
        reply.disposition = horizon::IpcDisposition::pending;
        break;
    case Disposition::malformed:
        malformed_ = true;
        diagnostic_ = "Malformed service-manager session requires closure; kernel close result unsupported";
        break;
    case Disposition::cancelled:
        reply.disposition = horizon::IpcDisposition::cancelled;
        pending_name_.reset();
        break;
    case Disposition::unsupported:
        break;
    }
    return reply;
}

horizon::IpcReply ServiceManagerSession::request(const horizon::IpcMessage& message,
                                                std::uint64_t sender_pid) {
    const std::lock_guard lock(mutex_);
    if (!session_) return {.disposition = horizon::IpcDisposition::handled,
                           .transport_result = horizon::session_closed_result};
    if (malformed_) return {};
    if (sender_pid != process_id_) {
        diagnostic_ = "Service-manager session cannot serve a different configured process";
        return {};
    }
    const auto result = session_->dispatch(message, sender_pid);
    if (result.disposition == Disposition::pending) {
        ServiceName name{};
        std::copy_n(message.begin() + 8, name.size(), name.begin());
        pending_name_ = name;
    }
    return translate(result);
}

horizon::IpcReply ServiceManagerSession::service_registered(const ServiceName& name) {
    const std::lock_guard lock(mutex_);
    if (!session_) return {.disposition = horizon::IpcDisposition::handled,
                           .transport_result = horizon::session_closed_result};
    if (malformed_) return {};
    if (!pending_name_ || *pending_name_ != name) {
        diagnostic_ = "Service registration does not match a pending service-manager request";
        return {};
    }
    return translate(session_->retry_pending());
}

const char* ServiceManagerSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

}  // namespace armigo::services::sm
