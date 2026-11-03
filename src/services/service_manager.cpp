// ============================================================================
// Armigo Services - Initial Service Manager Protocol
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/services/service_manager.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace armigo::services::sm {
namespace {

constexpr std::uint64_t header_mask = 0x800003ffffffffffULL;
constexpr std::uint64_t register_header = 0x8000000000000010ULL;
constexpr std::uint64_t get_header = 0x0000000200000011ULL;
constexpr std::uint32_t not_registered = 0x415;
constexpr std::uint32_t invalid_name = 0xc15;
constexpr std::uint32_t permission_denied = 0x1015;
constexpr std::uint32_t deferred_result = 0xc823;

std::uint64_t read_le(std::span<const std::byte> bytes, std::size_t offset,
                      std::size_t count) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < count; ++i)
        value |= std::uint64_t{std::to_integer<unsigned char>(bytes[offset + i])} << (8 * i);
    return value;
}

void write_le(std::span<std::byte> bytes, std::size_t offset, std::size_t count,
              std::uint64_t value) {
    for (std::size_t i = 0; i < count; ++i)
        bytes[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 0xff);
}

DispatchResult unsupported(const char* detail) {
    DispatchResult out;
    out.detail = detail;
    return out;
}

DispatchResult malformed() {
    DispatchResult out;
    out.disposition = Disposition::malformed;
    out.dispatcher_result = 0x1e23;
    out.detail = "Malformed initial service-manager TIPC request; close the session";
    return out;
}

DispatchResult service_reply(std::uint32_t result, horizon::Handle handle = 0) {
    DispatchResult out;
    out.disposition = Disposition::handled;
    out.reply_size = 20;
    write_le(out.reply, 0, 8, 0x8000000100000011ULL);
    write_le(out.reply, 8, 4, 0x20);
    write_le(out.reply, 12, 4, handle);
    write_le(out.reply, 16, 4, result);
    return out;
}

bool valid_name(const ServiceName& name) {
    if (name.front() == std::byte{0}) return false;
    const auto terminator = std::find(name.begin(), name.end(), std::byte{0});
    return std::all_of(terminator, name.end(), [](std::byte b) { return b == std::byte{0}; });
}

bool permits_client(std::span<const std::byte> acl, const ServiceName& name) {
    while (!acl.empty()) {
        const auto control = std::to_integer<unsigned char>(acl.front());
        const std::size_t length = (control & 7U) + 1U;
        if (acl.size() < length + 1) return false;
        const auto entry = acl.subspan(1, length);
        if ((control & 0x80U) == 0) {
            if (entry.back() == std::byte{'*'}) {
                if (std::equal(entry.begin(), entry.end() - 1, name.begin())) return true;
            } else {
                ServiceName exact{};
                std::copy(entry.begin(), entry.end(), exact.begin());
                if (exact == name) return true;
            }
        }
        acl = acl.subspan(length + 1);
    }
    return false;
}

}  // namespace

Session::Session(ProcessConfig process, ConnectService connect)
    : process_(std::move(process)), connect_(std::move(connect)) {}

DispatchResult Session::dispatch(std::span<const std::byte> request,
                                 std::optional<std::uint64_t> trusted_received_pid) {
    if (pending_name_) return unsupported("A service-manager request is already pending");
    if (request.size() < 8) return malformed();
    const auto header = read_le(request, 0, 8);
    const auto type = header & 0xffff;
    if (type == 0x10) {
        if (request.size() < 20 || (header & header_mask) != register_header ||
            read_le(request, 8, 4) != 1)
            return malformed();
        if (!trusted_received_pid || *trusted_received_pid != process_.process_id ||
            *trusted_received_pid == std::numeric_limits<std::uint64_t>::max())
            return unsupported("RegisterClient requires matching trusted kernel process identity");
        client_registered_ = true;
        DispatchResult out;
        out.disposition = Disposition::handled;
        out.reply_size = 12;
        write_le(out.reply, 0, 8, 0x0000000100000010ULL);
        return out;
    }
    if (type == 0x11) {
        if (request.size() < 16 || (header & header_mask) != get_header) return malformed();
        ServiceName name{};
        std::copy_n(request.begin() + 8, name.size(), name.begin());
        return get_service(name);
    }
    return unsupported("Service-manager command is outside the initial TIPC implementation");
}

DispatchResult Session::get_service(const ServiceName& name) {
    if (!client_registered_) return service_reply(not_registered);
    if (!valid_name(name)) return service_reply(invalid_name);
    if (!process_.registered) return service_reply(not_registered);
    if (!permits_client(process_.service_access_control, name))
        return service_reply(permission_denied);
    if (!connect_) return unsupported("No service connection backend is configured");
    const auto connected = connect_(process_.process_id, name);
    switch (connected.kind) {
    case LookupKind::connected:
        if (connected.handle == 0 || connected.result != 0)
            return unsupported("Service backend returned an invalid connected outcome");
        return service_reply(0, connected.handle);
    case LookupKind::failed:
        if (connected.handle != 0 || connected.result == 0 ||
            (connected.result & 0x3fffffU) == deferred_result)
            return unsupported("Service backend returned an invalid failure outcome");
        return service_reply((connected.result & 0x3fffffU) == 0xe01 ? 0x615 : connected.result);
    case LookupKind::pending: {
        if (connected.handle != 0 || connected.result != 0)
            return unsupported("Service backend returned an invalid pending outcome");
        pending_name_ = name;
        DispatchResult out;
        out.disposition = Disposition::pending;
        return out;
    }
    case LookupKind::cancelled: {
        if (connected.handle != 0 || connected.result != 0)
            return unsupported("Service backend returned an invalid cancellation outcome");
        DispatchResult out;
        out.disposition = Disposition::cancelled;
        out.detail = "Service connection was cancelled by host shutdown";
        return out;
    }
    case LookupKind::unsupported:
        return unsupported("Service connection backend does not implement this request");
    }
    return unsupported("Invalid service connection backend disposition");
}

DispatchResult Session::retry_pending() {
    if (!pending_name_) return unsupported("No service-manager request is pending");
    const auto name = *pending_name_;
    // Keep the pending transaction intact if the backend throws or reports
    // unsupported behavior; neither outcome completes the guest request.
    auto out = get_service(name);
    if (out.disposition == Disposition::handled || out.disposition == Disposition::cancelled)
        pending_name_.reset();
    return out;
}

}  // namespace armigo::services::sm
