// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Text-span analysis over an independently authored synthetic word list.
// Every word below was checked against the system assembler's encoding.

#include <armigo/aarch64/analysis.h>

#include "test_support.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using namespace armigo::aarch64;

struct Word {
    std::uint32_t word;
    SiteFlags expected;  // with default options
};

constexpr SiteFlags none = 0;
constexpr SiteFlags flag(SiteFlag f) { return static_cast<SiteFlags>(f); }

const Word program[] = {
    {0xA9BF7BFDu, none},                                    // stp x29, x30, [sp, #-16]!
    {0xD4000021u, flag(SiteFlag::supervisor_call)},         // svc #1
    {0xD53BD052u, SiteFlag::thread_register_read | SiteFlag::platform_register_operand},  // mrs x18, tpidr_el0
    {0xD51BD041u, flag(SiteFlag::thread_register_write)},   // msr tpidr_el0, x1
    {0xD53BD062u, flag(SiteFlag::read_only_thread_register_read)},  // mrs x2, tpidrro_el0
    {0xD51BD063u, flag(SiteFlag::read_only_system_register_write)}, // msr tpidrro_el0, x3
    {0xD53BE024u, flag(SiteFlag::counter_read)},            // mrs x4, cntpct_el0
    {0xD53BE045u, flag(SiteFlag::counter_read)},            // mrs x5, cntvct_el0
    {0xD53BE0C7u, flag(SiteFlag::counter_read)},            // mrs x7, cntvctss_el0
    {0xD53BE003u, flag(SiteFlag::counter_frequency_read)},  // mrs x3, cntfrq_el0
    {0xD51BE003u, flag(SiteFlag::read_only_system_register_write)}, // msr cntfrq_el0, x3
    {0xD53B002Bu, flag(SiteFlag::host_defined_register_read)},      // mrs x11, ctr_el0
    {0xD53B00ECu, flag(SiteFlag::host_defined_register_read)},      // mrs x12, dczid_el0
    {0xD53B4208u, none},                                    // mrs x8, nzcv
    {0xD51B4409u, none},                                    // msr fpcr, x9
    {0xD538000Du, flag(SiteFlag::unclassified_system_register)},    // mrs x13, midr_el1
    {0xD4000002u, flag(SiteFlag::privileged_exception)},    // hvc #0
    {0xD4400000u, flag(SiteFlag::privileged_exception)},    // hlt #0
    {0xD4200300u, flag(SiteFlag::breakpoint)},              // brk #0x18
    {0xB4000052u, flag(SiteFlag::platform_register_operand)},       // cbz x18, .+8
    {0xD61F0240u, flag(SiteFlag::platform_register_operand)},       // br x18
    {0x10000012u, flag(SiteFlag::platform_register_operand)},       // adr x18, .
    {0x98000012u, flag(SiteFlag::platform_register_operand)},       // ldrsw x18, .
    {0xD8000012u, none},                                    // prfm #18, . (operation, not x18)
    {0x8B120000u, none},                                    // add x0, x0, x18 (unclassified)
    {0xD503201Fu, none},                                    // nop
    {0x17FFFFFFu, none},                                    // b .-4
    {0x94000100u, none},                                    // bl .+0x400 (outside the span)
    {0xD65F03C0u, none},                                    // ret
};
constexpr std::size_t program_words = std::size(program);

std::vector<std::byte> bytes_of(const Word* words, std::size_t count) {
    std::vector<std::byte> bytes;
    for (std::size_t index = 0; index < count; ++index) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            bytes.push_back(static_cast<std::byte>((words[index].word >> shift) & 0xFF));
        }
    }
    return bytes;
}

bool has_limitation(const Analysis& analysis, Limitation limitation) {
    return std::find(analysis.limitations.begin(), analysis.limitations.end(), limitation) !=
           analysis.limitations.end();
}

void reports_expected_sites() {
    const auto bytes = bytes_of(program, program_words);
    const std::uint64_t base = 0x80004000;
    const auto analysis = analyze_text(bytes, base);
    CHECK(analysis.ok());

    std::size_t site = 0;
    for (std::size_t index = 0; index < program_words; ++index) {
        if (program[index].expected == none) {
            continue;
        }
        if (!CHECK(site < analysis.sites.size())) {
            break;
        }
        const auto& s = analysis.sites[site++];
        CHECK_EQ(s.offset, index * 4);
        CHECK_EQ(s.address, base + index * 4);
        CHECK_EQ(s.instruction.word, program[index].word);
        if (!CHECK_EQ(s.flags, program[index].expected)) {
            std::fprintf(stderr, "  word 0x%08x\n", program[index].word);
        }
    }
    CHECK_EQ(site, analysis.sites.size());

    const auto& tls = analysis.sites[1];
    CHECK(tls.system_register == SystemRegister::tpidr_el0);
    CHECK(tls.instruction.opcode == Opcode::mrs);

    const auto& stats = analysis.statistics;
    CHECK_EQ(stats.words, program_words);
    CHECK_EQ(stats.unclassified, 3);  // stp, add, nop
    CHECK_EQ(stats.classified, program_words - 3);
    CHECK_EQ(stats.pc_relative, 6);   // cbz, adr, ldrsw, prfm, b, bl
    CHECK_EQ(stats.branch_targets_outside_span, 1);
    CHECK_EQ(stats.unrepresentable_references, 0);
    CHECK_EQ(stats.platform_register_operands, 5);
    CHECK_EQ(stats.possible_platform_register_operands, 1);  // add x0, x0, x18

    CHECK(has_limitation(analysis, Limitation::embedded_data));
    CHECK(has_limitation(analysis, Limitation::unclassified_words));
    CHECK(has_limitation(analysis, Limitation::possible_platform_register));
    CHECK(has_limitation(analysis, Limitation::implicit_register_operands));
    CHECK(has_limitation(analysis, Limitation::newer_extensions));
    CHECK(has_limitation(analysis, Limitation::supervisor_abi));
    for (const auto limitation : analysis.limitations) {
        CHECK(describe(limitation) != nullptr);
    }
}

void optional_reports() {
    const auto bytes = bytes_of(program, program_words);
    const std::uint64_t base = 0x80004000;
    const auto analysis = analyze_text(
        bytes, base, {.report_pc_relative = true, .report_possible_platform_register = true});
    CHECK(analysis.ok());
    std::size_t pc_relative = 0;
    std::size_t possible = 0;
    for (const auto& s : analysis.sites) {
        pc_relative += has(s.flags, SiteFlag::pc_relative);
        if (has(s.flags, SiteFlag::possible_platform_register_operand)) {
            ++possible;
            CHECK_EQ(s.instruction.word, 0x8B120000u);
            CHECK(!s.instruction.classified());
        }
    }
    CHECK_EQ(pc_relative, 6);
    CHECK_EQ(possible, 1);
    // The PRFM site is reported for PC-relativity alone.
    const auto prfm = std::find_if(analysis.sites.begin(), analysis.sites.end(),
                                   [](const Site& s) { return s.instruction.word == 0xD8000012u; });
    CHECK(prfm != analysis.sites.end() && prfm->flags == flag(SiteFlag::pc_relative));
}

void classified_only_span_still_reports_embedded_data() {
    const Word words[] = {{0xD65F03C0u, none}};
    const auto bytes = bytes_of(words, 1);
    const auto analysis = analyze_text(bytes, 0x1000);
    CHECK(analysis.ok());
    CHECK(analysis.sites.empty());
    CHECK(has_limitation(analysis, Limitation::embedded_data));
    CHECK(!has_limitation(analysis, Limitation::unclassified_words));
    CHECK(!has_limitation(analysis, Limitation::supervisor_abi));

    const auto empty = analyze_text({}, 0x1000);
    CHECK(empty.ok() && empty.sites.empty() && empty.statistics.words == 0);
}

void unrepresentable_reference() {
    const Word words[] = {{0x17FFFFFFu, none}};  // b .-4 at address zero
    const auto bytes = bytes_of(words, 1);
    const auto analysis = analyze_text(bytes, 0);
    CHECK(analysis.ok());
    CHECK_EQ(analysis.statistics.unrepresentable_references, 1);
    CHECK_EQ(analysis.statistics.branch_targets_outside_span, 0);
}

void rejects_invalid_spans() {
    const auto bytes = bytes_of(program, program_words);
    CHECK(analyze_text(bytes, 0x1002).error == AnalysisError::misaligned_base);
    CHECK(analyze_text(std::span(bytes).first(6), 0x1000).error == AnalysisError::invalid_size);
    CHECK(analyze_text(bytes, 0xFFFFFFFFFFFFFFF0ull).error == AnalysisError::address_overflow);
    const std::uint64_t last_fit = 0 - static_cast<std::uint64_t>(bytes.size());
    CHECK(analyze_text(bytes, last_fit).ok());

    const auto limited = analyze_text(bytes, 0x1000, {.max_sites = 3});
    CHECK(limited.error == AnalysisError::resource_limit);
    CHECK(limited.statistics.words == 0 && limited.statistics.classified == 0 &&
          limited.statistics.unclassified == 0 && limited.statistics.pc_relative == 0 &&
          limited.statistics.branch_targets_outside_span == 0 &&
          limited.statistics.unrepresentable_references == 0 &&
          limited.statistics.platform_register_operands == 0 &&
          limited.statistics.possible_platform_register_operands == 0);
    CHECK(limited.limitations.empty());
    CHECK(limited.sites.empty());
    std::size_t sites = 0;
    for (const auto& word : program) {
        sites += word.expected != none;
    }
    CHECK(analyze_text(bytes, 0x1000, {.max_sites = sites}).ok());
}

} // namespace

int main() {
    reports_expected_sites();
    optional_reports();
    classified_only_span_still_reports_embedded_data();
    unrepresentable_reference();
    rejects_invalid_spans();
    return armigo::aarch64::test::finish("aarch64.analysis");
}
