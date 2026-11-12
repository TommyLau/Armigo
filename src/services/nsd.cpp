// ============================================================================
// Armigo Services - Name Service Directory
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/cmif.h>
#include <armigo/services/nsd.h>

#include <array>
#include <string>
#include <vector>

namespace armigo::services::nsd {
namespace {
using namespace horizon;

using namespace horizon::cmif;
constexpr std::string_view environment = "lp1";
constexpr std::uint32_t environment_type_lp = 1;
constexpr std::size_t fqdn_size = 0x100;
// Result of the nsd module (141): the resolved name does not fit.
constexpr std::uint32_t result_overflow = 141u | 2u << 9;



IpcReply reply_with(std::uint32_t result, std::size_t data = 0) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 32 + data;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, 16, sfco);
    write(reply.bytes, 24, result);
    return reply;
}
} // namespace

NsdSession::NsdSession(std::uint64_t process_id) : process_id_(process_id) {}

const char* NsdSession::diagnostic() const {
    const std::lock_guard lock(mutex_);
    return diagnostic_;
}

std::optional<bool> NsdSession::environment_change_disabled() const {
    const std::lock_guard lock(mutex_);
    return change_disabled_;
}

IpcReply NsdSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    return dispatch(message, sender_pid, nullptr);
}

IpcReply NsdSession::request_with_buffers(const IpcMessage& message, std::uint64_t sender_pid,
                                          const GuestBuffers& buffers) {
    return dispatch(message, sender_pid, &buffers);
}

IpcReply NsdSession::dispatch(const IpcMessage& message, std::uint64_t sender_pid, const GuestBuffers* buffers) {
    const std::lock_guard lock(mutex_);
    const auto unsupported = [&](const char* detail) {
        diagnostic_ = detail;
        return IpcReply{};
    };
    diagnostic_ = nullptr;
    if (sender_pid != process_id_) return unsupported("Name service session belongs to another process");
    const auto type = read(message, 0) & 0xffffu;
    if (type == 5 || type == 7) {
        // QueryPointerBufferSize only: this endpoint has no pointer buffer.
        if (read(message, 16) != sfci || read(message, 24) != 3)
            return unsupported("Name service control request is unsupported");
        auto reply = reply_with(0, 4);
        write(reply.bytes, 32, 0);
        return reply;
    }
    const auto cmif = locate_cmif(message);
    if (!cmif || read(message, *cmif) != sfci) return unsupported("Name service request shape is unsupported");
    const auto command = read(message, *cmif + 8);
    switch (command) {
    case 11: {  // GetEnvironmentIdentifier -> 8-byte identifier
        if (!buffers || buffers->empty() || (*buffers)[buffers->size() - 1].size < 8)
            return unsupported("Name service environment needs its buffer");
        std::array<std::byte, 8> identifier{};
        for (std::size_t i = 0; i < environment.size(); ++i) identifier[i] = std::byte(environment[i]);
        if (!buffers->write(buffers->size() - 1, identifier))
            return unsupported("Name service environment buffer is not writable");
        return reply_with(0);
    }
    case 15:  // SetChangeEnvironmentIdentifierDisabled(bool)
        change_disabled_ = (read(message, *cmif + 16) & 0xff) != 0;
        return reply_with(0);
    case 20:    // Resolve(name) -> resolved name
    case 21: {  // ResolveEx(name) -> result, resolved name
        if (!buffers || buffers->size() < 2) return unsupported("Name service resolution needs its buffers");
        const auto& input = (*buffers)[0];
        std::vector<std::byte> bytes(std::min<std::size_t>(input.size, fqdn_size));
        if (!buffers->read(0, bytes)) return unsupported("Name service name is not readable");
        std::string name;
        for (const auto byte : bytes) {
            if (byte == std::byte{0}) break;
            name.push_back(static_cast<char>(byte));
        }
        std::string resolved;
        for (const char c : name) {
            if (c == '%') resolved += environment;
            else resolved.push_back(c);
        }
        const std::uint32_t result = resolved.size() >= fqdn_size ? result_overflow : 0;
        if (!result) {
            const auto out = buffers->size() - 1;
            if ((*buffers)[out].size < fqdn_size) return unsupported("Name service result buffer is too small");
            std::array<std::byte, fqdn_size> written{};
            for (std::size_t i = 0; i < resolved.size(); ++i) written[i] = std::byte(resolved[i]);
            if (!buffers->write(out, written)) return unsupported("Name service result buffer is not writable");
        }
        if (command == 20) return reply_with(result);
        auto reply = reply_with(0, 4);
        write(reply.bytes, 32, result);
        return reply;
    }
    case 100: {  // GetApplicationServerEnvironmentType -> Lp
        auto reply = reply_with(0, 4);
        write(reply.bytes, 32, environment_type_lp);
        return reply;
    }
    default:
        return unsupported("Name service command is not implemented");
    }
}

} // namespace armigo::services::nsd
