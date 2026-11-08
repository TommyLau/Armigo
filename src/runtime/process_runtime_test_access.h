// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/runtime/process_runtime.h>
#include "service_registry.h"
#include <functional>

namespace armigo::runtime {

// Available only in BUILD_TESTING builds. These barriers let authored fixtures
// schedule competing host events without replacing real IPC or VI operations.
struct ProcessRuntimeTestState {
    std::shared_ptr<horizon::HandleTable> handles;
    std::shared_ptr<GuestMemory> memory;
    std::shared_ptr<ServiceRegistry> services;
    std::shared_ptr<ThreadRuntime> threads;
    horizon::Handle main_thread = 0;
    std::uint64_t process_id = 0;
};

class ProcessRuntimeTestAccess {
public:
    static ProcessRuntimeTestState inspect(ProcessRuntime& runtime);
    static void set_failed_ipc_boundary(ProcessRuntime& runtime, std::function<void(std::stop_token)> boundary);
    static void set_ipc_admission_boundary(ProcessRuntime& runtime, std::function<void(std::stop_token)> boundary);
    static bool run_success_selected(ProcessRuntime& runtime);
    static std::size_t mutex_waiters(ProcessRuntime& runtime, std::uint64_t address);
    // Installs the runtime's unsupported-NVN handler as an NVN launch does.
    static void install_nvn_handler(ProcessRuntime& runtime);
};

} // namespace armigo::runtime
