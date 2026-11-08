// ============================================================================
// Armigo Services - Name Service Directory
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The "nsd:u" service, as far as a title's online library needs it with no
// network, following the practical reference. The environment is "lp1", the
// production environment (11 writes it, 100 answers its type, Lp); Resolve
// (20) and ResolveEx (21) replace the "%" of a name with that environment, as
// the console does, and write the 0x100-byte result, which must fit
// (ResolveEx returns the result as its second word too); 15 records whether
// the environment may change. Any other request is unsupported with an
// explicit diagnostic.

#pragma once

#include <armigo/horizon/ipc.h>

#include <cstdint>
#include <mutex>
#include <optional>

namespace armigo::services::nsd {

class NsdSession final : public horizon::ClientSession {
public:
    explicit NsdSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage&, std::uint64_t sender_pid,
                                           const horizon::GuestBuffers&) override;
    const char* diagnostic() const;
    std::optional<bool> environment_change_disabled() const;

private:
    horizon::IpcReply dispatch(const horizon::IpcMessage&, std::uint64_t sender_pid, const horizon::GuestBuffers*);
    const std::uint64_t process_id_;
    mutable std::mutex mutex_;
    std::optional<bool> change_disabled_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::nsd
