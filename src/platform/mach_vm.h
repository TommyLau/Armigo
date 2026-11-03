// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <TargetConditionals.h>
#include <mach/mach.h>
#include <mach/vm_region.h>

#if TARGET_OS_IPHONE
// The arm64 iOS libSystem exports these 64-bit VM entry points, while its SDK
// intentionally omits mach_vm.h. Retain the same VM calls and width as macOS.
static_assert(sizeof(mach_vm_address_t) == 8 && sizeof(mach_vm_size_t) == 8);
extern "C" {
kern_return_t mach_vm_allocate(vm_map_t, mach_vm_address_t*, mach_vm_size_t, int);
kern_return_t mach_vm_deallocate(vm_map_t, mach_vm_address_t, mach_vm_size_t);
kern_return_t mach_vm_protect(vm_map_t, mach_vm_address_t, mach_vm_size_t, boolean_t, vm_prot_t);
kern_return_t mach_vm_remap(vm_map_t, mach_vm_address_t*, mach_vm_size_t, mach_vm_offset_t,
    int, vm_map_t, mach_vm_address_t, boolean_t, vm_prot_t*, vm_prot_t*, vm_inherit_t);
kern_return_t mach_vm_region(vm_map_read_t, mach_vm_address_t*, mach_vm_size_t*,
    vm_region_flavor_t, vm_region_info_t, mach_msg_type_number_t*, mach_port_t*);
kern_return_t mach_vm_read_overwrite(vm_map_read_t, mach_vm_address_t, mach_vm_size_t,
    mach_vm_address_t, mach_vm_size_t*);
}
#else
#include <mach/mach_vm.h>
#endif
