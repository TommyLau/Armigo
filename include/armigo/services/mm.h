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

#include <armigo/horizon/ipc.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace armigo::services::mm {

// Multimedia clock requests (`mm:u`), non-domain. A client registers a request
// for one module (CPU, GPU, EMC, NVDEC, ...) and asks for a clock range; the
// host has no clocks to change, so the range is recorded and Get reports the
// request's minimum. Initialize (4: u32 module, u32 priority, u32 event clear
// mode) returns the request id; Finalize (5: id) drops it; SetAndWait (6: id,
// u32 minimum, s32 maximum) records the range; Get (7: id) returns the minimum,
// or 0 for an unknown id. The older module-keyed commands (0..3) and domain
// conversion are explicit unsupported diagnostics.
class Session final : public horizon::ClientSession {
public:
    explicit Session(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage&, std::uint64_t sender_pid) override;
    const char* diagnostic() const;

private:
    struct Request {
        std::uint32_t module = 0;
        std::uint32_t id = 0;
        std::uint32_t minimum = 0;
        std::int32_t maximum = -1;
        bool auto_clear_event = false;
    };
    const std::uint64_t process_id_;
    std::vector<Request> requests_;
    std::uint32_t next_id_ = 1;
    mutable std::mutex mutex_;
    const char* diagnostic_ = nullptr;
};

} // namespace armigo::services::mm
