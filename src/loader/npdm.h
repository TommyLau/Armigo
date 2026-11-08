// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Structural NPDM (META) reader.
//
// Structural admission rules follow the Horizon loader;
// npdm.cpp implements the ordered validation stages.
//
// This unit validates magics, header fields, alignment and bounded section
// ranges. It does not interpret capabilities or authenticate the input.
// The returned view borrows the input image; runtime admission follows later.

#pragma once

#include <armigo/common/guest_platform.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace armigo::loader {

// Size of the fixed META header and the two access-info headers, in bytes.
inline constexpr std::size_t kNpdmMetaHeaderSize = 0x80;
inline constexpr std::size_t kNpdmAci0HeaderSize = 0x40;
inline constexpr std::size_t kNpdmAcidSize = 0x240;

// One byte range inside the NPDM image. Offsets are absolute image positions.
struct NpdmSection {
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
};

// The structural content of one NPDM image. Subsection ranges point into the
// input image and are only valid while that image is alive; nothing here owns
// bytes. Interpretation of the subsections is the caller's responsibility.
struct ParsedNpdm {
    std::span<const std::byte> image;
    GuestPlatform platform = GuestPlatform::switch1;

    // Fixed META header fields.
    std::uint8_t meta_flags = 0;
    std::uint8_t main_thread_priority = 0;
    std::uint8_t main_thread_ideal_core = 0;
    std::uint32_t system_resource_size = 0;
    std::uint32_t meta_version = 0;
    std::uint32_t main_thread_stack_size = 0;

    NpdmSection aci0;
    std::uint64_t aci0_program_id = 0;
    NpdmSection aci0_fs_access_header;
    NpdmSection aci0_service_access_control;
    NpdmSection aci0_kernel_access_control;

    NpdmSection acid;
    std::uint32_t acid_flags = 0;
    std::uint64_t acid_program_id_min = 0;
    std::uint64_t acid_program_id_max = 0;
    NpdmSection acid_fs_access_control;
    NpdmSection acid_service_access_control;
    NpdmSection acid_kernel_access_control;
};

// A structured structural failure. `operation` is a stable snake_case stage
// name; `detail` is one English sentence describing the cause. Both are report
// data only and carry no guest-visible meaning.
struct NpdmParseFailure {
    std::string operation;
    std::string detail;
};

// Validates the NPDM byte layout and returns the section table, or a failure
// naming the first structural problem found (bad magic, truncated header,
// invalid alignment, version, or out-of-bounds section).
[[nodiscard]] std::variant<NpdmParseFailure, ParsedNpdm> parse_npdm(
    std::span<const std::byte> image);

} // namespace armigo::loader
