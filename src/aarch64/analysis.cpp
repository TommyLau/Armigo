// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/aarch64/analysis.h>

#include <limits>

namespace armigo::aarch64 {
namespace {

std::uint32_t load_word(std::span<const std::byte> text, std::size_t offset) noexcept {
    return std::to_integer<std::uint32_t>(text[offset]) |
           (std::to_integer<std::uint32_t>(text[offset + 1]) << 8) |
           (std::to_integer<std::uint32_t>(text[offset + 2]) << 16) |
           (std::to_integer<std::uint32_t>(text[offset + 3]) << 24);
}

bool is_branch(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::b:
    case Opcode::bl:
    case Opcode::b_cond:
    case Opcode::bc_cond:
    case Opcode::cbz:
    case Opcode::cbnz:
    case Opcode::tbz:
    case Opcode::tbnz:
        return true;
    default:
        return false;
    }
}

bool read_only_at_el0(SystemRegister reg) noexcept {
    switch (reg) {
    case SystemRegister::tpidrro_el0:
    case SystemRegister::ctr_el0:
    case SystemRegister::dczid_el0:
    case SystemRegister::cntfrq_el0:
    case SystemRegister::cntpct_el0:
    case SystemRegister::cntvct_el0:
    case SystemRegister::cntpctss_el0:
    case SystemRegister::cntvctss_el0:
        return true;
    default:
        return false;
    }
}

SiteFlags system_register_flags(Opcode opcode, SystemRegister reg) noexcept {
    const bool read = opcode == Opcode::mrs;
    if (!read && read_only_at_el0(reg)) {
        return static_cast<SiteFlags>(SiteFlag::read_only_system_register_write);
    }
    switch (reg) {
    case SystemRegister::other:
        return static_cast<SiteFlags>(SiteFlag::unclassified_system_register);
    case SystemRegister::tpidr_el0:
        return static_cast<SiteFlags>(read ? SiteFlag::thread_register_read
                                           : SiteFlag::thread_register_write);
    case SystemRegister::tpidrro_el0:
        return static_cast<SiteFlags>(SiteFlag::read_only_thread_register_read);
    case SystemRegister::cntfrq_el0:
        return static_cast<SiteFlags>(SiteFlag::counter_frequency_read);
    case SystemRegister::cntpct_el0:
    case SystemRegister::cntvct_el0:
    case SystemRegister::cntpctss_el0:
    case SystemRegister::cntvctss_el0:
        return static_cast<SiteFlags>(SiteFlag::counter_read);
    case SystemRegister::ctr_el0:
    case SystemRegister::dczid_el0:
        return static_cast<SiteFlags>(SiteFlag::host_defined_register_read);
    case SystemRegister::nzcv:
    case SystemRegister::fpcr:
    case SystemRegister::fpsr:
        break;  // EL0 state the host preserves for the guest
    }
    return 0;
}

SiteFlags opcode_flags(const Instruction& i, SystemRegister reg) noexcept {
    switch (i.opcode) {
    case Opcode::svc:
        return static_cast<SiteFlags>(SiteFlag::supervisor_call);
    case Opcode::hvc:
    case Opcode::smc:
    case Opcode::hlt:
        return static_cast<SiteFlags>(SiteFlag::privileged_exception);
    case Opcode::brk:
        return static_cast<SiteFlags>(SiteFlag::breakpoint);
    case Opcode::mrs:
    case Opcode::msr_register:
        return system_register_flags(i.opcode, reg);
    case Opcode::cache_maintenance:
        return static_cast<SiteFlags>(SiteFlag::cache_maintenance);
    default:
        return 0;
    }
}

bool has_platform_register_field(std::uint32_t word) noexcept {
    for (const unsigned low : {0u, 5u, 10u, 16u}) {
        if (((word >> low) & 31u) == platform_register) {
            return true;
        }
    }
    return false;
}

} // namespace

const char* describe(Limitation limitation) noexcept {
    switch (limitation) {
    case Limitation::unclassified_words:
        return "unclassified words were not examined; their position independence, register use "
               "and host safety are unknown";
    case Limitation::possible_platform_register:
        return "x18 use inside unclassified words is unknown; the register-field scan is neither "
               "precise nor complete";
    case Limitation::implicit_register_operands:
        return "registers implied by another operand are not modeled for unclassified words";
    case Limitation::embedded_data:
        return "every aligned word was treated as an instruction; data in the span can produce "
               "spurious sites and no reachability is proven";
    case Limitation::newer_extensions:
        return "newer PC-relative encodings such as FEAT_CMPBR are not classified";
    case Limitation::supervisor_abi:
        return "supervisor call register usage is defined by the guest OS ABI and not modeled";
    }
    return "unknown limitation";
}

Analysis analyze_text(std::span<const std::byte> text, std::uint64_t base_address,
                      const AnalysisOptions& options) {
    Analysis result;
    if (base_address % 4 != 0) {
        result.error = AnalysisError::misaligned_base;
        return result;
    }
    if (text.size() % 4 != 0) {
        result.error = AnalysisError::invalid_size;
        return result;
    }
    const std::uint64_t size = text.size();
    if (size != 0 && size - 1 > std::numeric_limits<std::uint64_t>::max() - base_address) {
        result.error = AnalysisError::address_overflow;
        return result;
    }

    auto& stats = result.statistics;
    bool any_supervisor_call = false;
    for (std::size_t offset = 0; offset < text.size(); offset += 4) {
        const std::uint64_t address = base_address + offset;
        const Instruction i = decode(load_word(text, offset));
        ++stats.words;

        SiteFlags flags = 0;
        SystemRegister reg = SystemRegister::other;
        if (!i.classified()) {
            ++stats.unclassified;
            if (has_platform_register_field(i.word)) {
                ++stats.possible_platform_register_operands;
                if (options.report_possible_platform_register) {
                    flags |= static_cast<SiteFlags>(SiteFlag::possible_platform_register_operand);
                }
            }
        } else {
            ++stats.classified;
            if (i.opcode == Opcode::mrs || i.opcode == Opcode::msr_register) {
                reg = identify_system_register(i.system_register);
            }
            flags |= opcode_flags(i, reg);
            any_supervisor_call = any_supervisor_call || i.opcode == Opcode::svc;
            if (((i.reads | i.writes) & register_bit(platform_register)) != 0) {
                ++stats.platform_register_operands;
                flags |= static_cast<SiteFlags>(SiteFlag::platform_register_operand);
            }
            if (is_pc_relative(i.opcode)) {
                ++stats.pc_relative;
                const auto reference = referenced_address(i, address);
                if (!reference) {
                    ++stats.unrepresentable_references;
                } else if (is_branch(i.opcode) &&
                           (*reference < base_address || *reference - base_address >= size)) {
                    ++stats.branch_targets_outside_span;
                }
                if (options.report_pc_relative) {
                    flags |= static_cast<SiteFlags>(SiteFlag::pc_relative);
                }
            }
        }
        if (flags == 0) {
            continue;
        }
        if (result.sites.size() >= options.max_sites) {
            return Analysis{.error = AnalysisError::resource_limit};
        }
        result.sites.push_back(Site{.offset = offset,
                                    .address = address,
                                    .instruction = i,
                                    .flags = flags,
                                    .system_register = reg});
    }

    // The word sweep itself never proves reachability or code/data separation.
    result.limitations.push_back(Limitation::embedded_data);
    if (stats.unclassified != 0) {
        result.limitations.push_back(Limitation::unclassified_words);
        result.limitations.push_back(Limitation::possible_platform_register);
        result.limitations.push_back(Limitation::implicit_register_operands);
        result.limitations.push_back(Limitation::newer_extensions);
    }
    if (any_supervisor_call) {
        result.limitations.push_back(Limitation::supervisor_abi);
    }
    return result;
}

} // namespace armigo::aarch64
