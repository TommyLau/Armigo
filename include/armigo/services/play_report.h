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

#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>
#include <armigo/horizon/startup_queries.h>

#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace armigo::services::prepo {

// Process-bound application play reports. Accepted reports are discarded by
// the offline host; no report bytes or user identifiers survive the request.
class PlayReportSession final : public horizon::ClientSession {
public:
    explicit PlayReportSession(std::uint64_t process_id) : process_id_(process_id) {}

    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                          const horizon::GuestBuffers&) override;
    // SendReportWithUser (10500) passes its report in a copied transfer memory.
    bool receives_copied_objects() const noexcept override { return true; }
    horizon::IpcReply request_with_handles(
        const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers&,
        std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects) override;
    const char* diagnostic() const;

private:
    horizon::IpcReply unsupported(const char* detail);
    horizon::IpcReply send_report(const horizon::IpcMessage&, const horizon::GuestBuffers&,
                                  std::span<const std::shared_ptr<horizon::KernelObject>> copied_objects);
    horizon::IpcReply context_request(std::uint32_t object, std::uint32_t command);

    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    bool domain_ = false;
    // Domain objects 2, 3, ...: the async contexts SendReportWithUser returned,
    // each with its completion event, null once released. Nothing is ever
    // transmitted, so every context is complete from the start.
    std::vector<std::shared_ptr<horizon::Event>> contexts_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::prepo
