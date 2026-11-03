// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Linear analysis of a guest text span for host-sensitive instructions.
//
// The analysis examines every aligned word with the bounded classifier in
// instruction.h. It reports classified sites a native preparer must handle
// and states, through limitations and statistics, what it could not examine.
// An empty site list therefore never means a span is safe to run natively:
// consult the limitations and the unclassified word count.

#pragma once

#include <armigo/aarch64/instruction.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace armigo::aarch64 {

enum class SiteFlag : std::uint32_t {
    // SVC: control must reach the host gate instead of the guest kernel.
    supervisor_call = 1u << 0,
    // HVC, SMC, HLT: undefined or trapping at EL0 on the host.
    privileged_exception = 1u << 1,
    // BRK: raises a debug exception whose handling the host must own.
    breakpoint = 1u << 2,
    // MRS/MSR TPIDR_EL0: the host owns this register's value.
    thread_register_read = 1u << 3,
    thread_register_write = 1u << 4,
    // MRS TPIDRRO_EL0: holds the host's thread pointer, not the guest's.
    read_only_thread_register_read = 1u << 5,
    // MSR to a register EL0 cannot write (TPIDRRO_EL0, counters, CNTFRQ_EL0,
    // CTR_EL0, DCZID_EL0): undefined at EL0.
    read_only_system_register_write = 1u << 6,
    // MRS CNTPCT/CNTVCT and their self-synchronized forms: host counter
    // values and frequency differ from the guest's.
    counter_read = 1u << 7,
    // MRS CNTFRQ_EL0: host counter frequency.
    counter_frequency_read = 1u << 8,
    // MRS CTR_EL0 or DCZID_EL0: cache geometry of the host, not the guest.
    host_defined_register_read = 1u << 9,
    // MRS/MSR of a register not identified by identify_system_register.
    unclassified_system_register = 1u << 10,
    // A classified instruction architecturally reads or writes x18.
    platform_register_operand = 1u << 11,
    // An unclassified word with Rd/Rt, Rn, Rt2/Ra or Rm/Rs equal to 18.
    // Reported only on request; see Limitation::possible_platform_register.
    possible_platform_register_operand = 1u << 12,
    // A PC-relative instruction; reported only on request.
    pc_relative = 1u << 13,
    // DC/IC cache maintenance naming a guest address: the host decides what the
    // operation means for that address.
    cache_maintenance = 1u << 14,
};

using SiteFlags = std::uint32_t;

constexpr SiteFlags operator|(SiteFlag left, SiteFlag right) noexcept {
    return static_cast<SiteFlags>(left) | static_cast<SiteFlags>(right);
}
constexpr bool has(SiteFlags flags, SiteFlag flag) noexcept {
    return (flags & static_cast<SiteFlags>(flag)) != 0;
}

struct Site {
    std::uint64_t offset = 0;   // byte offset within the analyzed span
    std::uint64_t address = 0;  // base address + offset
    Instruction instruction;
    SiteFlags flags = 0;
    SystemRegister system_register = SystemRegister::other;  // MRS/MSR only
};

struct AnalysisOptions {
    bool report_pc_relative = false;
    // Over-approximate x18 use in unclassified words. In real text this marks
    // a large fraction of all words; it is a search aid, not a decoder.
    bool report_possible_platform_register = false;
    std::size_t max_sites = 1'000'000;
};

enum class AnalysisError : std::uint8_t {
    none,
    misaligned_base,   // base address is not 4-byte aligned
    invalid_size,      // span size is not a multiple of 4
    address_overflow,  // base address + size leaves the address space
    resource_limit,    // more than max_sites sites
};

enum class Limitation : std::uint8_t {
    // Words outside the classifier were not examined. Their position
    // independence, register use and host safety are unknown.
    unclassified_words,
    // x18 use inside unclassified words is unknown. The opt-in field scan is
    // neither precise nor complete.
    possible_platform_register,
    // Operands implied by another register (for example LD64B's eight
    // consecutive registers) are not modeled for unclassified words.
    implicit_register_operands,
    // Every aligned word is treated as an instruction. Literal pools or jump
    // tables in the span produce spurious sites; no reachability is proven.
    embedded_data,
    // Newer PC-relative encodings (for example FEAT_CMPBR compare-and-branch)
    // are not classified and fall into the unclassified count.
    newer_extensions,
    // Register usage of supervisor calls is defined by the guest OS ABI and
    // is not included in register masks.
    supervisor_abi,
};

const char* describe(Limitation limitation) noexcept;

struct AnalysisStatistics {
    std::uint64_t words = 0;
    std::uint64_t classified = 0;
    std::uint64_t unclassified = 0;
    std::uint64_t pc_relative = 0;
    // Immediate, conditional, compare and test branches whose target lies
    // outside the span; normal for cross-module or data references only when
    // explained by the module layout.
    std::uint64_t branch_targets_outside_span = 0;
    // PC-relative instructions whose reference leaves the address space.
    std::uint64_t unrepresentable_references = 0;
    std::uint64_t platform_register_operands = 0;
    std::uint64_t possible_platform_register_operands = 0;
};

struct Analysis {
    AnalysisError error = AnalysisError::none;
    std::vector<Site> sites;
    AnalysisStatistics statistics;
    std::vector<Limitation> limitations;
    bool ok() const noexcept { return error == AnalysisError::none; }
};

// Analyzes little-endian instruction words loaded at base_address. On error
// no sites, statistics, or limitations are returned. The span is read only.
Analysis analyze_text(std::span<const std::byte> text, std::uint64_t base_address,
                      const AnalysisOptions& options = {});

} // namespace armigo::aarch64
