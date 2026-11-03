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

#include <armigo/horizon/event.h>
#include <armigo/horizon/ipc.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

namespace armigo::services::nifm {

// Network interface management for an application (`nifm:u`) on a console
// with no network: the runtime has no network connection.
//
// IStaticService (the port's object): CreateGeneralServiceOld (4) and
// CreateGeneralService (5, reserved u64 and the sender PID) hand out an
// IGeneralService. IGeneralService: GetClientId (1) writes the client's id,
// unique within the process, to its output buffer; CreateRequest (4,
// requirement preset) hands out an IRequest. IRequest: GetRequestState (0),
// GetSystemEventReadableHandles (2) copies the request's two readable events,
// Submit (4) evaluates the request at once: with no network profile it fails
// with no candidate (0x8986e), the state returns to free and the first event
// is signaled. GetResult (1) answers that result as the reply's result, and
// GetAppletInfo (21, theme color) names the network-connection applet (15,
// mode 0) with its argument records for that result. Every other method is an
// explicit unsupported diagnostic.
//
// A session serves one object until ConvertToDomain; a converted session
// serves the port object as domain object 1 and hands out further objects as
// domain objects, which a domain close releases. An unconverted session hands
// out further objects as new sessions.
struct Object {
    enum class Kind { static_service, general_service, request };
    Kind kind = Kind::static_service;
    std::uint32_t client_id = 0;
    std::int32_t requirement_preset = 0;
    std::uint32_t state = 1;  // request: free
    bool submitted = false;
    std::uint32_t result = 0;  // request: the last evaluation's result
    std::shared_ptr<horizon::Event> events[2];
};

struct ProcessState {
    explicit ProcessState(std::uint64_t id) : process_id(id) {}
    const std::uint64_t process_id;
    std::mutex mutex;
    std::uint32_t next_client_id = 1;
};

class Session final : public horizon::ClientSession {
public:
    // The port session of `nifm:u` for one process.
    explicit Session(std::uint64_t process_id);
    Session(std::shared_ptr<ProcessState> process, std::shared_ptr<Object> object);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;
    // The object an unconverted session serves, or domain object `id`.
    std::shared_ptr<const Object> object(std::uint32_t id = 1) const;
    std::size_t domain_object_count() const;

private:
    horizon::IpcReply handle(const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers*);
    const std::shared_ptr<ProcessState> process_;
    std::shared_ptr<Object> object_;
    bool domain_ = false;
    std::map<std::uint32_t, std::shared_ptr<Object>> objects_;
    std::uint32_t next_object_ = 2;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::nifm
