// ============================================================================
// Armigo Services - Initial Log Manager Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>

#include <mutex>
#include <optional>
#include <vector>

namespace armigo::services::lm {

// One firmware-20.5.0 LM client session. Supports CMIF controls 3 and 0,
// followed by domain ILogService::OpenLogger (command 0), and the logger's
// command 0, which firmware 20.5.0 completes without consuming its input.
// HIPC types 5/7 serve controls and types 4/6 serve domain requests. No SDK
// replacement or non-domain handle factory is implied.
class LogManagerSession final : public horizon::ClientSession {
public:
    explicit LogManagerSession(std::uint64_t process_id);

    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;

    // Host observations of owned domain objects; IDs are scoped to this session.
    // A root ID, missing ID, or unconverted session has no logger PID.
    std::optional<std::uint64_t> logger_process_id(std::uint32_t object_id) const;
    std::size_t domain_object_count() const;
    // Static text for the most recent unsupported request, otherwise null.
    const char* diagnostic() const;

private:
    enum class ObjectKind { root, logger };
    struct DomainObject {
        ObjectKind kind;
        std::uint64_t process_id;
        // A logger owns a reference to the root service that created it.
        std::shared_ptr<DomainObject> parent;
    };

    horizon::IpcReply unsupported(const char* message);
    horizon::IpcReply control(const horizon::IpcMessage& message);
    horizon::IpcReply domain(const horizon::IpcMessage& message,
                             std::uint64_t sender_pid);
    horizon::IpcReply open_logger(const horizon::IpcMessage& message,
                                 std::uint64_t sender_pid);
    horizon::IpcReply log(const horizon::IpcMessage& message,
                          std::uint64_t sender_pid);

    mutable std::mutex mutex_;
    const std::uint64_t process_id_;
    const std::shared_ptr<DomainObject> root_;
    // IDs are one-based indices into owned objects. No guest/kernel handle is
    // invented. Closing the session releases the registry and its references.
    std::vector<std::shared_ptr<DomainObject>> objects_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::lm
