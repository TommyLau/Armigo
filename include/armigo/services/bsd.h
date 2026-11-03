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

#include <armigo/horizon/handle_table.h>
#include <armigo/horizon/ipc.h>

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

namespace armigo::services::bsd {

// The socket service bsd:u, as far as the SDK's socket initialization needs
// it. Controls: QueryPointerBufferSize (no pointer buffer), ConvertToDomain and
// CloneCurrentObject; clones share one state. RegisterClient (0) keeps the
// library configuration and a reference to the transfer memory the client
// hands over (so the work area outlives the client's own handle) and answers
// status 0; StartMonitoring (1) on the monitor session accepts the client.
// Socket operations are explicit unsupported diagnostics: no network exists.
class BsdSession final : public horizon::ClientSession {
public:
    BsdSession(std::uint64_t process_id, horizon::HandleTable& handles);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;
    bool registered() const;
    std::shared_ptr<horizon::KernelObject> work_memory() const;
    std::array<std::byte, 32> configuration() const;

private:
    struct State {
        std::mutex mutex;
        bool domain = false;
        bool registered = false;
        bool monitoring = false;
        std::array<std::byte, 32> configuration{};
        std::uint64_t work_memory_size = 0;
        std::shared_ptr<horizon::KernelObject> work_memory;
        const char* diagnostic = nullptr;
    };
    BsdSession(std::uint64_t process_id, horizon::HandleTable& handles, std::shared_ptr<State> state);

    const std::uint64_t process_id_;
    horizon::HandleTable& handles_;
    const std::shared_ptr<State> state_;
};

} // namespace armigo::services::bsd
