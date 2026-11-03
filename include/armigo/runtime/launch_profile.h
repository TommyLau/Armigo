// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/startup_queries.h>
#include <armigo/runtime/launch_result.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace armigo::runtime {

// Lane A validates META/ACI0/ACID bounds and requested capabilities against
// their restrictions before returning a profile. No SDK-private objects or
// title defaults. The admitted 20.5.0 subset is GetInfo, QueryMemory,
// GetThreadPriority and GetThreadId on a 64-bit, 39-bit process with no
// private system resource. NPDM authentication and additional
// profiles remain unsupported. Unknown
// required capabilities, missing metadata and unsupported address spaces fail.
enum class NpdmTrustPolicy {
    require_authenticated,
    // Explicit owner opt-in: structural/capability checks still mandatory.
    structurally_validated_extracted_input,
};
enum class ProfileOrigin { npdm, synthetic_test };
struct ProcessPermissions {
    // Admitted SVC immediates, sorted and unique; empty denies every SVC.
    std::vector<std::uint16_t> supervisor_calls;
    // The validated byte-stream representation consumed by sm::ProcessConfig.
    std::vector<std::byte> service_access_control;
    // Validated effective FS access data; lane F must enforce it or reject.
    std::vector<std::byte> file_system_access_control;
    std::uint32_t handle_table_capacity = 0;
    // Retained admitted capabilities for downstream process initialization.
    std::uint32_t intended_kernel_version = 0;
    std::uint32_t program_type = 0;
    std::uint32_t debug_flags = 0;
    // Inclusive numeric bounds; lower numbers mean higher guest priority.
    std::uint32_t highest_priority = 0;
    std::uint32_t lowest_priority = 0;
};
struct MainThreadProfile {
    std::uint32_t priority = 0;
    std::int32_t ideal_core = 0;
    std::uint64_t stack_size = 0;
};
struct LaunchProfile {
    ProfileOrigin origin = ProfileOrigin::npdm;
    // Program identity admitted from the NPDM ACI0/ACID restriction.
    std::uint64_t title_id = 0;
    // Reuse address_bits, allowed_core_mask and private_system_resource_bytes.
    // Placement, entropy and accounting are assigned later by lanes C/F;
    // the default ProcessConfiguration is not evidence of NPDM admission.
    horizon::ProcessConfiguration process;
    ProcessPermissions permissions;
    MainThreadProfile main_thread;
    bool is_64_bit = true;
};

// No retained span, no mappings, no execution. A requested authentication
// policy without a verifier is unsupported, never silently downgraded.
[[nodiscard]] LaunchResult<LaunchProfile> derive_launch_profile(
    std::span<const std::byte> npdm, NpdmTrustPolicy trust);

} // namespace armigo::runtime
