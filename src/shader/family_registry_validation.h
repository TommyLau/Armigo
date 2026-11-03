// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Shared compile-time validation for the private instruction-family tables.
// A table is valid when every entry is well-formed, families are unique, and
// no two match patterns can admit the same instruction word. Entries without
// a match pattern only take part in the well-formedness and uniqueness
// checks.
#pragma once

#include <cstdint>
#include <span>

namespace armigo::shader::detail {

// Two patterns intersect when a single word can satisfy both: every bit
// position fixed by both patterns must require the same value.
// Masks that do not overlap always intersect, because some word satisfies
// each constraint independently.
constexpr bool patterns_intersect(std::uint16_t a_mask, std::uint16_t a_value,
                                  std::uint16_t b_mask, std::uint16_t b_value) {
    return ((a_value ^ b_value) & a_mask & b_mask) == 0;
}

// `well_formed` reports per-entry validity outside this function's scope,
// such as a non-null entry function or an admitted family.
template <typename Entry, typename WellFormed>
constexpr bool family_registry_valid(std::span<const Entry> entries,
                                     WellFormed well_formed) {
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const Entry& entry = entries[i];
        if (!well_formed(entry)) {
            return false;
        }
        if constexpr (requires { entry.mask; entry.value; }) {
            // A value bit outside the mask is unreachable by construction
            // and a zero mask would match every word the legacy classifier
            // rejected, so neither may produce the required explicit
            // unknown-opcode error.
            if (entry.mask == 0 || (entry.value & ~entry.mask) != 0) {
                return false;
            }
        }
        for (std::size_t j = i + 1; j < entries.size(); ++j) {
            const Entry& other = entries[j];
            if (entry.family == other.family) {
                return false;
            }
            if constexpr (requires { entry.mask; entry.value; }) {
                if (patterns_intersect(entry.mask, entry.value, other.mask,
                                       other.value)) {
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace armigo::shader::detail
