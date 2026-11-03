// ============================================================================
// Armigo Services - Initial Runtime-Loader (ro) Service
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
#include <map>
#include <mutex>

namespace armigo::services::ro {

// One module the application registered with the runtime loader. The record keeps the
// range the client handed over, because unregistering has to name the same range.
struct ModuleRange {
    std::uint64_t address = 0;
    std::uint64_t size = 0;
};

// The `ro` service. Its command set follows the firmware contract used by every title:
// registration of module ranges and process module info. The commands that would load
// code into another process are refused explicitly rather than reported as success.
class LoaderSession final : public horizon::ClientSession {
public:
    explicit LoaderSession(std::uint64_t process_id);
    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    std::size_t domain_object_count() const;
    const char* diagnostic() const;
    std::size_t module_count() const;

private:
    horizon::IpcReply unsupported(const char* detail);
    horizon::IpcReply register_module(const horizon::IpcMessage& message, std::size_t cmif);
    horizon::IpcReply unregister_module(const horizon::IpcMessage& message, std::size_t cmif);
    mutable std::mutex mutex_;
    const std::uint64_t process_id_;
    bool converted_ = false;
    std::map<std::uint64_t, ModuleRange> modules_;
    const char* diagnostic_ = nullptr;
};

}  // namespace armigo::services::ro
