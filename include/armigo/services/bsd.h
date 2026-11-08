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
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace armigo::services::bsd {

// The socket service bsd:u, as far as the SDK's socket initialization needs
// it. Controls: QueryPointerBufferSize (no pointer buffer), ConvertToDomain and
// CloneCurrentObject; clones share one state. RegisterClient (0) keeps the
// library configuration and a reference to the transfer memory the client
// hands over (so the work area outlives the client's own handle) and answers
// status 0; Switch 2 titles register with command 36 instead (16 bytes and the
// process id, errno 0 back). StartMonitoring (1) on the monitor session
// accepts the client.
//
// Event descriptors are local to the service and work with no network:
// EventFd (31) creates one (flags bit 0: reads take one at a time; bits 1 and
// 2: reads never wait), Write (24) adds an 8-byte value, Read (25) takes the
// counter or one, Poll (6) reports POLLIN while the counter is nonzero and
// POLLOUT always, and Close (26) releases one. A poll that would wait does so
// without the session lock for at most its timeout, and an endless poll
// returns 0 after 100 ms so a waiting thread still sees a shutdown (the
// practical reference does the same offline). Descriptors the service never
// made report POLLNVAL. No network exists (nifm reports no connection), so
// Socket (2) and SocketExempt (3) fail with ENETDOWN. A read that would wait
// on an empty blocking descriptor and every other operation are explicit
// unsupported diagnostics.
class BsdSession final : public horizon::ClientSession {
public:
    BsdSession(std::uint64_t process_id, horizon::HandleTable& handles);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;
    bool registered() const;
    std::shared_ptr<horizon::KernelObject> work_memory() const;
    std::array<std::byte, 32> configuration() const;

private:
    struct EventFd {
        std::uint64_t counter = 0;
        std::uint32_t flags = 0;
    };
    struct EventFds {
        std::mutex mutex;
        std::condition_variable changed;
        std::vector<std::optional<EventFd>> table;
    };
    struct State {
        std::mutex mutex;
        EventFds event_fds;
        bool domain = false;
        bool registered = false;
        bool monitoring = false;
        std::array<std::byte, 32> configuration{};
        std::uint64_t work_memory_size = 0;
        std::shared_ptr<horizon::KernelObject> work_memory;
        const char* diagnostic = nullptr;
    };
    BsdSession(std::uint64_t process_id, horizon::HandleTable& handles, std::shared_ptr<State> state);
    horizon::IpcReply dispatch(const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers*);

    const std::uint64_t process_id_;
    horizon::HandleTable& handles_;
    const std::shared_ptr<State> state_;
};

} // namespace armigo::services::bsd
