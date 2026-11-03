// ============================================================================
// Armigo Memory - Guest CPU Write Tracking
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#pragma once

#include <TargetConditionals.h>
#include <array>
#include <cstddef>
#include <cstdint>

#if TARGET_OS_IPHONE
// libSystem exports this syscall wrapper on iOS, but its SDK omits libproc.h.
// Keep the signal-path query on the same syscall as macOS: a MIG query would
// introduce a shared reply-port dependency inside a fault handler.
extern "C" int proc_pidinfo(int pid, int flavor, std::uint64_t argument, void* buffer, int size);
#else
#include <libproc.h>
#include <sys/proc_info.h>
#endif

namespace armigo::memory::detail {

// Fixed Darwin PROC_PIDREGIONINFO ABI. Only protection/address/size are used;
// the intervening statistics and policy fields remain opaque.
struct NativeRegionInfo {
    std::uint32_t protection;
    std::array<std::byte, 76> reserved;
    std::uint64_t address;
    std::uint64_t size;
};
inline constexpr int native_region_info_flavor = 7;
static_assert(sizeof(NativeRegionInfo) == 96 && offsetof(NativeRegionInfo, address) == 80 &&
              offsetof(NativeRegionInfo, size) == 88);
#if !TARGET_OS_IPHONE
static_assert(sizeof(NativeRegionInfo) == sizeof(proc_regioninfo));
static_assert(offsetof(NativeRegionInfo, protection) == offsetof(proc_regioninfo, pri_protection));
static_assert(offsetof(NativeRegionInfo, address) == offsetof(proc_regioninfo, pri_address));
static_assert(offsetof(NativeRegionInfo, size) == offsetof(proc_regioninfo, pri_size));
static_assert(native_region_info_flavor == PROC_PIDREGIONINFO);
#endif

} // namespace armigo::memory::detail
