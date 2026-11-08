// ============================================================================
// Armigo NVN - API 231 Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// The entry points the Switch 2 NVN API 231 hands out. Many share their name
// with an API 55 entry point, but object storage, property numbers and some
// signatures differ, so the table is separate: an entry point serves the
// API 55 implementation only where that implementation was checked against
// the title's own driver (share), and its own implementation otherwise
// (install). Every other known name resolves to a stub that stops the run
// with an unsupported result naming the entry point.

#pragma once

#include "entry_names_231.h"
#include "entry_table.h"

#include <cstdint>
#include <optional>
#include <string_view>

namespace armigo::nvn::v231 {

enum class Entry : std::uint16_t {
#define ARMIGO_NVN231_NAME(name) name,
    ARMIGO_NVN231_ENTRY_POINTS(ARMIGO_NVN231_NAME)
#undef ARMIGO_NVN231_NAME
    count
};

inline constexpr std::size_t entry_count = static_cast<std::size_t>(Entry::count);

[[nodiscard]] std::string_view entry_name(Entry entry) noexcept;
[[nodiscard]] std::optional<Entry> find_entry(std::string_view name) noexcept;

// What the loader hands out for an entry point now: its own implementation,
// the shared API 55 implementation, or its unsupported stub.
[[nodiscard]] GenericFunction resolve(Entry entry) noexcept;

// The driver accepts the variable-rate shading commands and never applies a
// shading rate: every fragment shades at full rate.
inline constexpr bool applies_shading_rate = false;

// Whether the loader hands out an implementation (its own or a shared one)
// rather than the unsupported stub.
[[nodiscard]] bool implemented(Entry entry) noexcept;

void install_entry(Entry entry, GenericFunction function) noexcept;
template <typename Function>
void install(Entry entry, Function* function) noexcept {
    install_entry(entry, reinterpret_cast<GenericFunction>(function));
}

// Hands out whatever the API 55 table implements for `implementation` when the
// loader resolves `entry`, so a backend installed later is the one served.
void share(Entry entry, nvn::Entry implementation) noexcept;

// Installs everything implemented for API 231.40. Call after the API 55
// entries are registered and before the game resolves an entry point.
void register_entries() noexcept;

} // namespace armigo::nvn::v231
