// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "host_limits.h"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach/mach.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif
#endif

namespace armigo::platform {

std::uint64_t native_page_size() noexcept {
#if defined(__APPLE__)
    const std::uint64_t page = vm_page_size;
    return page != 0 && (page & (page - 1)) == 0 ? page : 0;
#else
    return 0;
#endif
}

std::optional<HostLimits> query_host_limits() noexcept {
#if defined(__APPLE__)
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const auto result = ::task_info(mach_task_self(), TASK_VM_INFO,
        reinterpret_cast<task_info_t>(&info), &count);
    const auto page = native_page_size();
    if (result != KERN_SUCCESS || count < TASK_VM_INFO_REV2_COUNT ||
        !page || info.max_address <= info.min_address) return {};
    HostLimits limits{info.min_address, info.max_address, page, {}};
#if TARGET_OS_IPHONE
    limits.available_memory_bytes = ::os_proc_available_memory();
#else
    if (count >= TASK_VM_INFO_REV4_COUNT && info.limit_bytes_remaining)
        limits.available_memory_bytes = info.limit_bytes_remaining;
#endif
    return limits;
#else
    return {};
#endif
}

} // namespace armigo::platform
