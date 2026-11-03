// ============================================================================
// Armigo Services - Initial Performance Manager Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/services/operation_mode.h>

namespace armigo::services::apm {

// Initial APM startup only. Buffer capacity reports owned server storage;
// zero is the minimal host configuration, not a firmware-wide constant.
// The manager reports the launch performance mode; opened sessions record
// requested configurations. Other commands and buffer descriptors remain
// explicit unsupported operations. This service does not change Apple clocks.
//
// The original SDK opens the performance session through the plain HIPC type6
// command0 path before any domain conversion, so the root also serves that
// non-domain variant: the reply transfers the opened session as a new caller
// handle through the HIPC move-handle array.
class PerformanceSession final : public horizon::ClientSession {
public:
    enum class Role {
        root,     // the `apm` service session returned by SM
        opened,   // a session already opened by the root
    };
    explicit PerformanceSession(std::uint64_t process_id,
                                std::uint16_t pointer_buffer_capacity = 0,
                                Role role = Role::root,
                                OperationMode operation_mode = OperationMode::Console);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    std::size_t domain_object_count() const;
    const char* diagnostic() const;

    // Opened sessions report the last configuration the caller requested.
    // Armigo records it; changing Apple hardware clocks is out of scope.
    struct RequestedConfiguration {
        std::uint32_t mode = 0;
        std::uint32_t configuration = 0;
        bool set = false;
    };
    RequestedConfiguration requested_configuration() const;

private:
    enum class Kind { root, session };
    struct Object {
        Kind kind;
        std::shared_ptr<Object> parent;
    };
    horizon::IpcReply unsupported(const char*);
    horizon::IpcReply open_session(const horizon::IpcMessage& message);
    const std::uint64_t process_id_;
    const std::uint16_t capacity_;
    const Role role_;
    const OperationMode operation_mode_;
    std::vector<std::byte> pointer_buffer_;
    std::shared_ptr<Object> root_;
    std::vector<std::shared_ptr<Object>> objects_;
    mutable std::mutex mutex_;
    RequestedConfiguration requested_{};
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::apm
