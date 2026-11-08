// ============================================================================
// Armigo Services - SSL
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The "ssl" service as far as a title's online library needs it with no
// network, following the practical reference. A context is a local object of
// the service, so creating one succeeds offline; only a connection needs the
// network. The service sets the interface version (5) and creates contexts
// (0); a context keeps its options (0, 1), counts no connections (3) and
// imports server and client PKI (4, 5) under increasing ids. Creating a
// connection and every other request are unsupported with an explicit
// diagnostic. The SDK may convert the service session to a CMIF domain; its
// contexts are then objects of that domain.

#pragma once

#include <armigo/horizon/ipc.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

namespace armigo::services::ssl {

class Context;

class SslSession final : public horizon::ClientSession {
public:
    explicit SslSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;
    std::optional<std::uint32_t> interface_version() const;

private:
    struct Domain {
        std::mutex mutex;
        bool converted = false;
        bool root_released = false;
        std::uint32_t next_object = 2;
        std::map<std::uint32_t, std::shared_ptr<Context>> objects;
        std::optional<std::uint32_t> interface_version;
        const char* diagnostic = nullptr;
    };
    SslSession(std::uint64_t process_id, std::shared_ptr<Domain> domain);
    horizon::IpcReply serve(const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers*);
    std::optional<horizon::IpcReply> command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                             std::shared_ptr<Context>& opened);
    horizon::IpcReply unsupported(const char* detail);
    const std::uint64_t process_id_;
    const std::shared_ptr<Domain> domain_;
};

// One SSL context: its options and imported PKI.
class Context final : public horizon::ClientSession {
public:
    explicit Context(std::uint64_t process_id, std::uint32_t version);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;
    std::uint32_t version() const noexcept { return version_; }
    // One context command whose CMIF header is at `cmif`; the reply's CMIF
    // header is at `out`. Null with `detail` set when unsupported.
    std::optional<horizon::IpcReply> command(const horizon::IpcMessage&, std::size_t cmif, std::size_t out,
                                             const horizon::GuestBuffers*, const char*& detail);

private:
    const std::uint64_t process_id_;
    const std::uint32_t version_;
    mutable std::mutex mutex_;
    std::map<std::uint32_t, std::int32_t> options_;
    std::uint64_t next_pki_ = 1;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::ssl
