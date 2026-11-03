// ============================================================================
// Armigo Services - Initial Time Service Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/horizon/read_only_shared_memory.h>

#include <mutex>
#include <utility>

namespace armigo::services::time {

// Firmware 20.5.0 time:u startup: controls 3/0 and domain commands 3/20.
// The provided page must already contain coherent initialized clock state.
// The caller's handle table must outlive the session and every request.
class TimeServiceSession final : public horizon::ClientSession {
public:
    // Throws invalid_argument for a null object or a payload other than 4096
    // bytes. The shared-memory object copies supplied data before this call.
    TimeServiceSession(std::uint64_t process_id, horizon::HandleTable& handles,
                       std::shared_ptr<horizon::ReadOnlySharedMemory> shared_memory);

    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;

    std::size_t domain_object_count() const;
    bool is_time_zone_object(std::uint32_t object_id) const;
    const char* diagnostic() const;

private:
    struct DomainObject {
        virtual ~DomainObject() = default;
    };
    struct RootObject final : DomainObject {
        explicit RootObject(std::shared_ptr<horizon::ReadOnlySharedMemory> value)
            : shared_memory(std::move(value)) {}
        const std::shared_ptr<horizon::ReadOnlySharedMemory> shared_memory;
    };
    struct TimeZoneObject final : DomainObject {
        explicit TimeZoneObject(std::shared_ptr<RootObject> value) : parent(std::move(value)) {}
        // time:u's child has no mutation authority. Child methods remain
        // unsupported until their individual contracts are implemented.
        const std::shared_ptr<RootObject> parent;
    };

    horizon::IpcReply unsupported(const char* detail);
    horizon::IpcReply control(const horizon::IpcMessage& message);
    horizon::IpcReply domain_request(const horizon::IpcMessage& message);
    horizon::IpcReply time_zone_command(std::uint32_t command, const horizon::IpcMessage& message);

    mutable std::mutex mutex_;
    const std::uint64_t process_id_;
    horizon::HandleTable& handles_;
    const std::shared_ptr<RootObject> root_;
    // Opaque session-local IDs are one-based indices into actual owned objects.
    std::vector<std::shared_ptr<DomainObject>> objects_;
    const char* diagnostic_ = nullptr;
};

}  // namespace armigo::services::time
