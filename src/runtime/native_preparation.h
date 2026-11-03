// ============================================================================
// Armigo Runtime - Native Module Preparation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/guest_dispatch.h>
#include <armigo/execution/reachable_code.h>
#include <armigo/execution/scalar_x18_patch.h>
#include <armigo/execution/thread_register_patch.h>
#include <armigo/loader/nso_module.h>
#include <armigo/runtime/module_set.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::runtime::native {

// Roles of one text word that preparation must handle. Words with no role run
// unchanged, including ordinary branches, calls and returns.
struct WordRole {
    bool uses_x18 = false;
    // Host-sensitive system or exception instruction routed through a gate.
    bool system = false;
    // Branch or exception class word this decoder does not identify.
    bool unknown_control = false;
    bool exclusive_load = false;
    bool exclusive_store = false;
    bool clear_exclusive = false;
};
[[nodiscard]] WordRole classify_word(std::uint32_t word) noexcept;

// Function roots of one linked image: defined functions, the unwind index,
// constructors, DT_INIT and linked function relocations. Roots only end
// exclusive-access windows; every text word is still examined.
struct ModuleRoots {
    std::vector<std::uint64_t> local;
    std::vector<execution::ExternalLeaf> external_leaves;
    std::vector<std::uint64_t> required_foreign_entries;
    std::size_t unwind_roots = 0;
};
[[nodiscard]] ModuleRoots module_roots(const ModulePreparationView& view, std::uint64_t text_begin,
                                       std::uint64_t text_size,
                                       const std::optional<loader::ModuleHeaderInfo>& header,
                                       std::size_t maximum_functions = 1000000);

// A word preparation replaced with a terminal trap because no safe rewrite
// exists. Executing it stops the guest; it is never silently run.
struct NativeStop {
    std::uint64_t address = 0;
    std::uint32_t word = 0;
    std::string reason;
};

// Linear preparation of one module's whole text at load time: x18 uses become
// virtual-register trampolines and system instructions become gate sites.
struct ModuleCodePreparation {
    std::uint32_t slot = 0;
    // Pack-time variant: runtime values are read from view.appended_data.
    // The caller supplies at least one separate, page-aligned key page.
    bool embedded = false;
    std::vector<std::byte> gate;
    std::uint64_t text_begin = 0;
    std::uint64_t text_size = 0;
    std::optional<loader::ModuleHeaderInfo> module_header;
    execution::ScalarX18Policy stack_contract{true, true, true};
    // Embedded only: image offsets of MOVZ/MOVK words whose immediate is
    // chosen at launch. Each becomes a gate site that holds the original word
    // (16 appendix bytes each, beyond scan_text's count); the launch gives the
    // site table the word to apply.
    std::vector<std::uint64_t> launch_words;

    std::vector<execution::GuestGateSite> sites;
    std::vector<std::uint64_t> prepared_entries;
    std::vector<std::uint64_t> required_foreign_entries;
    std::vector<NativeStop> stops;
    std::uint64_t prepared_begin = 0;
    std::uint64_t prepared_end = 0;
    std::size_t code_bytes = 0;
    std::size_t scalar_sites = 0;
    std::size_t thread_register_sites = 0;
    bool completed = false;
    std::string error;

    // Classifies the whole text on up to `threads` threads and returns the
    // appendix bytes prepare() can need for it, before page rounding. Keeps the
    // words that have a role, so prepare() visits only those; the text must
    // reach prepare() unchanged (relocation never touches executable text).
    // `image` is the whole module image, from which switch tables through
    // virtual x18 are sized as prepare() sizes them.
    [[nodiscard]] std::size_t scan_text(std::span<const std::byte> text, unsigned threads,
                                        std::span<const std::byte> image);
    // Rewrites view.image and fills view.appended_code from offset zero.
    void prepare(const ModulePreparationView& view);

private:
    // A text word with at least one WordRole flag, in text order.
    struct RoleWord {
        std::uint32_t index;
        std::uint32_t word;
        std::uint8_t flags;
    };
    std::vector<RoleWord> role_words_;
    std::uint64_t scanned_text_size_ = 0;
    bool scanned_ = false;
};

// Every linked executable owner must finish its own scan before guest entry.
[[nodiscard]] bool foreign_entries_prepared(std::span<const ModuleCodePreparation> modules) noexcept;

} // namespace armigo::runtime::native
