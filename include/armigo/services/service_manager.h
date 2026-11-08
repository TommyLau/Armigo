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

#include "armigo/horizon/handle_table.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace armigo::services::sm {

using ServiceName = std::array<std::byte, 8>;

// Trusted process metadata, distinct from RegisterClient's session state.
// The ACL is the supplied client/server service-access-control byte stream.
struct ProcessConfig {
    std::uint64_t process_id = 0;
    bool registered = false;
    std::vector<std::byte> service_access_control;
};

enum class LookupKind { connected, failed, pending, unsupported, cancelled };

struct LookupResult {
    LookupKind kind = LookupKind::unsupported;
    horizon::Handle handle = 0;
    std::uint32_t result = 0;
};

// Called only after name and client ACL validation. A connected handle must
// name a real client session in the namespace chosen by transport. Transport
// owns caller-local publication, any needed translation, and failure cleanup.
// Failed/pending/unsupported/cancelled outcomes must not carry an owned handle.
using ConnectService = std::function<LookupResult(std::uint64_t, const ServiceName&)>;

enum class Disposition { handled, unsupported, malformed, pending, cancelled };

struct DispatchResult {
    Disposition disposition = Disposition::unsupported;
    std::array<std::byte, 20> reply{};
    std::size_t reply_size = 0;
    // Internal server result, never an invented guest service reply.
    std::uint32_t dispatcher_result = 0;
    const char* detail = nullptr;
};

// Initial firmware TIPC RegisterClient/GetService session for ordinary
// configured processes. Caller serializes dispatch/retry/destruction and
// backend calls; no scheduler or service registry is owned here.
class Session final {
public:
    Session(ProcessConfig process, ConnectService connect);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // trusted_received_pid comes from kernel receive context, only when the
    // special descriptor requests a PID. Raw request PID bytes are ignored.
    // Malformed means close the server session without sending a reply.
    DispatchResult dispatch(std::span<const std::byte> request,
                            std::optional<std::uint64_t> trusted_received_pid = std::nullopt);

    // After service registration, retry the saved lookup. Pending produces
    // no reply and retains the request. A new dispatch while pending is an
    // unsupported caller sequencing error, not another guest transaction.
    DispatchResult retry_pending();
    bool has_pending_request() const noexcept { return pending_name_.has_value(); }

private:
    DispatchResult get_service(const ServiceName& name);

    ProcessConfig process_;
    ConnectService connect_;
    bool client_registered_ = false;
    std::optional<ServiceName> pending_name_;
};

}  // namespace armigo::services::sm
