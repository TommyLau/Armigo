// ============================================================================
// Armigo NVN - API 231 Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "entry_table_231.h"
#include "host_stack.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <utility>

namespace armigo::nvn::v231 {
namespace {

constexpr std::array<std::string_view, entry_count> names{
#define ARMIGO_NVN231_NAME(name) std::string_view{#name},
    ARMIGO_NVN231_ENTRY_POINTS(ARMIGO_NVN231_NAME)
#undef ARMIGO_NVN231_NAME
};

// The resolver searches names by byte value; so does find_entry.
static_assert(std::is_sorted(names.begin(), names.end()), "API 231 names must stay sorted");

// An API 55 entry index plus one; zero shares nothing.
constexpr std::uint16_t no_share = 0;

struct Slot {
    std::atomic<GenericFunction> target{};
    std::atomic<std::uint16_t> shared{no_share};
};
std::array<Slot, entry_count> slots{};

[[gnu::noinline]] void report_unimplemented(std::size_t index, std::uint64_t guest_return) noexcept {
    on_host_stack([&] {
        char detail[192];
        std::snprintf(detail, sizeof detail, "Armigo NVN 231.40: unimplemented entry point %.*s",
                      static_cast<int>(names[index].size()), names[index].data());
        report_unsupported(detail, guest_return);
    });
}

// Each stub knows its entry point. Without a runtime handler (synthetic
// tests) the call reports and returns zero in the integer result register.
template <std::size_t Index>
std::uint64_t unimplemented() noexcept {
    report_unimplemented(Index, reinterpret_cast<std::uint64_t>(__builtin_return_address(0)));
    return 0;
}

template <std::size_t... Index>
constexpr std::array<GenericFunction, entry_count> make_stubs(std::index_sequence<Index...>) {
    return {reinterpret_cast<GenericFunction>(&unimplemented<Index>)...};
}
const std::array<GenericFunction, entry_count> stubs = make_stubs(std::make_index_sequence<entry_count>{});

bool selected() noexcept {
    const auto version = selected_api_version();
    return version && version->major == 231;
}

} // namespace

std::string_view entry_name(Entry entry) noexcept { return names[static_cast<std::size_t>(entry)]; }

std::optional<Entry> find_entry(std::string_view name) noexcept {
    const auto found = std::lower_bound(names.begin(), names.end(), name);
    if (found == names.end() || *found != name) return std::nullopt;
    return static_cast<Entry>(found - names.begin());
}

GenericFunction resolve(Entry entry) noexcept {
    const auto index = static_cast<std::size_t>(entry);
    if (const auto target = slots[index].target.load(std::memory_order_acquire)) return target;
    if (const auto shared = slots[index].shared.load(std::memory_order_acquire); shared != no_share)
        if (const auto target = implemented_entry(static_cast<nvn::Entry>(shared - 1))) return target;
    return stubs[index];
}

bool implemented(Entry entry) noexcept { return resolve(entry) != stubs[static_cast<std::size_t>(entry)]; }

void install_entry(Entry entry, GenericFunction function) noexcept {
    slots[static_cast<std::size_t>(entry)].target.store(function, std::memory_order_release);
}

void share(Entry entry, nvn::Entry implementation) noexcept {
    slots[static_cast<std::size_t>(entry)].shared.store(
        static_cast<std::uint16_t>(static_cast<std::size_t>(implementation) + 1), std::memory_order_release);
}

} // namespace armigo::nvn::v231

namespace armigo::nvn {

extern "C" GenericFunction armigo_nvn231_bootstrap_loader(const char* name) {
    if (!name || !v231::selected()) return nullptr;
    const auto entry = v231::find_entry(name);
    if (!entry) {
        armigo::diag::write(armigo::diag::Level::warning, "Armigo NVN 231.40: unknown entry point %s\n", name);
        return nullptr;
    }
    return v231::resolve(*entry);
}

} // namespace armigo::nvn
