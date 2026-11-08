// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "backend.h"
#include "entry_table.h"
#include "host_stack.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace armigo::nvn {

struct alignas(16) EntrySlot {
    std::atomic<std::uint64_t> calls{};
    std::atomic<GenericFunction> target{};
};

static_assert(sizeof(EntrySlot) == 16);
static_assert(offsetof(EntrySlot, calls) == 0);
static_assert(offsetof(EntrySlot, target) == 8);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::atomic<GenericFunction>::is_always_lock_free);

extern "C" {
alignas(16) EntrySlot armigo_nvn_entry_slots[entry_count]{};
void armigo_nvn_count_thunks();
}

namespace {

constexpr std::size_t count_thunk_stride = 64;
static_assert(entry_count == 541, "abi.S must be regenerated when the entry table changes");

constexpr std::array<std::string_view, entry_count> names{
#define ARMIGO_NVN_TYPED(name, result, parameters) std::string_view{#name},
#define ARMIGO_NVN_UNTYPED(name) std::string_view{#name},
    ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
    ARMIGO_NVN_UNTYPED_ENTRY_POINTS(ARMIGO_NVN_UNTYPED)
#undef ARMIGO_NVN_TYPED
#undef ARMIGO_NVN_UNTYPED
};

std::array<std::atomic<bool>, entry_count> reported{};

// Typed fallbacks initialize the correct result registers or indirect result
// storage. Untyped entries retain an integer-zero diagnostic fallback until
// their full signatures are admitted.
template <std::size_t Index>
void report_unimplemented() noexcept {
    if (!reported[Index].exchange(true, std::memory_order_relaxed))
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: unimplemented entry point %.*s\n",
                     static_cast<int>(names[Index].size()), names[Index].data());
}

template <Entry E, typename Signature = typename EntrySignature<E>::type>
struct TypedFallback;
template <Entry E, typename Result, typename... Parameters>
struct TypedFallback<E, Result (*)(Parameters...)> {
    static Result call(Parameters...) noexcept {
        report_unimplemented<static_cast<std::size_t>(E)>();
        if constexpr (!std::is_void_v<Result>) return Result{};
    }
};
template <Entry E>
std::uint64_t untyped_fallback() noexcept {
    report_unimplemented<static_cast<std::size_t>(E)>();
    return 0;
}

const std::array<GenericFunction, entry_count> stubs{
#define ARMIGO_NVN_TYPED(name, result, parameters) reinterpret_cast<GenericFunction>(&TypedFallback<Entry::name>::call),
#define ARMIGO_NVN_UNTYPED(name) reinterpret_cast<GenericFunction>(&untyped_fallback<Entry::name>),
    ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
    ARMIGO_NVN_UNTYPED_ENTRY_POINTS(ARMIGO_NVN_UNTYPED)
#undef ARMIGO_NVN_TYPED
#undef ARMIGO_NVN_UNTYPED
};
// The selected API version as major << 16 | minor; zero selects no driver.
std::atomic<std::uint32_t> selected{0};

GenericFunction target_for(std::size_t index) noexcept {
    auto target = armigo_nvn_entry_slots[index].target.load(std::memory_order_acquire);
    if (target) return target;

    auto expected = static_cast<GenericFunction>(nullptr);
    if (armigo_nvn_entry_slots[index].target.compare_exchange_strong(
            expected, stubs[index], std::memory_order_release, std::memory_order_acquire))
        return stubs[index];
    return expected;
}

bool counting_requested() noexcept {
    const char* value = std::getenv("ARMIGO_NVN_COUNT");
    return value && value[0] == '1' && value[1] == '\0';
}

GenericFunction count_thunk(std::size_t index) noexcept {
    const auto base = reinterpret_cast<std::uintptr_t>(&armigo_nvn_count_thunks);
    return reinterpret_cast<GenericFunction>(base + index * count_thunk_stride);
}

// Entry indices sorted by name for lookup.
const std::array<std::uint16_t, entry_count>& sorted_entries() noexcept {
    static const auto sorted = [] {
        std::array<std::uint16_t, entry_count> order{};
        for (std::size_t i = 0; i < entry_count; ++i) order[i] = static_cast<std::uint16_t>(i);
        std::sort(order.begin(), order.end(), [](auto a, auto b) { return names[a] < names[b]; });
        return order;
    }();
    return sorted;
}

struct KnownPackage {
    std::string_view package;
    ApiVersion version;
};
// NVN driver packages and the API version each reports. The API 55 packages
// share their entry points; the device reports the selected version.
constexpr std::array known_packages{
    KnownPackage{"20_5_22", {55, 15}},
    KnownPackage{"16_2_4", {55, 15}},
    KnownPackage{"13_3_2", {55, 12}},
    // The Switch 2 package: API 231, served by the 231 table.
    KnownPackage{"22_2_8", {231, 40}},
};

} // namespace

std::string_view entry_name(Entry entry) noexcept {
    return names[static_cast<std::size_t>(entry)];
}

std::optional<Entry> find_entry(std::string_view name) noexcept {
    const auto& sorted = sorted_entries();
    const auto found = std::lower_bound(sorted.begin(), sorted.end(), name,
                                        [](std::uint16_t index, std::string_view key) { return names[index] < key; });
    if (found == sorted.end() || names[*found] != name) return std::nullopt;
    return static_cast<Entry>(*found);
}

void install_entry(Entry entry, GenericFunction function) noexcept {
    const auto index = static_cast<std::size_t>(entry);
    armigo_nvn_entry_slots[index].target.store(function ? function : stubs[index],
                                                std::memory_order_release);
}

GenericFunction installed_entry(Entry entry) noexcept {
    return armigo_nvn_entry_slots[static_cast<std::size_t>(entry)].target.load(std::memory_order_acquire);
}

GenericFunction implemented_entry(Entry entry) noexcept {
    const auto index = static_cast<std::size_t>(entry);
    const auto target = armigo_nvn_entry_slots[index].target.load(std::memory_order_acquire);
    return target == stubs[index] ? nullptr : target;
}

BootstrapLoader bootstrap_loader_for(ApiVersion version) noexcept {
    return version.major == 231 ? &armigo_nvn231_bootstrap_loader : &armigo_nvn_bootstrap_loader;
}

std::optional<std::string_view> driver_package(std::span<const std::byte> image) noexcept {
    static constexpr std::string_view prefix = "NintendoSDK_NVN-";
    static constexpr std::string_view suffix = "-Release";
    if (image.empty()) return std::nullopt;
    const std::string_view text(reinterpret_cast<const char*>(image.data()), image.size());
    std::optional<std::string_view> package;
    std::size_t search = 0;
    while (true) {
        const auto at = text.find(prefix, search);
        if (at == std::string_view::npos) return package;
        const auto start = at + prefix.size();
        const auto end = text.find(suffix, start);
        if (end == std::string_view::npos || end == start || end - start > 32) return std::nullopt;
        const auto candidate = text.substr(start, end - start);
        if (package && *package != candidate) return std::nullopt;
        package = candidate;
        search = end + suffix.size();
    }
}

std::optional<ApiVersion> known_api_version(std::string_view package) noexcept {
    for (const auto& known : known_packages)
        if (known.package == package) return known.version;
    return std::nullopt;
}

void select_api_version(ApiVersion version) noexcept {
    const bool known = std::any_of(known_packages.begin(), known_packages.end(),
                                   [&](const KnownPackage& entry) { return entry.version == version; });
    selected.store(known ? static_cast<std::uint32_t>(version.major) << 16 | static_cast<std::uint16_t>(version.minor) : 0,
                   std::memory_order_release);
}

namespace {
std::atomic<bool> policy_cpu_cached_texture_writes{false};
std::atomic<TitlePolicy::Timestamps> policy_timestamps{TitlePolicy::Timestamps::report};
std::atomic<TitlePolicy::Barriers> policy_barriers{TitlePolicy::Barriers::full};
std::atomic<bool> policy_hoist_implicit_samples{false};
} // namespace

void set_title_policy(const TitlePolicy& policy) noexcept {
    policy_cpu_cached_texture_writes.store(policy.cpu_cached_texture_writes, std::memory_order_release);
    policy_timestamps.store(policy.timestamps, std::memory_order_release);
    policy_barriers.store(policy.barriers, std::memory_order_release);
    policy_hoist_implicit_samples.store(policy.hoist_implicit_samples, std::memory_order_release);
}

TitlePolicy title_policy() noexcept {
    TitlePolicy policy;
    policy.cpu_cached_texture_writes = policy_cpu_cached_texture_writes.load(std::memory_order_acquire);
    policy.timestamps = policy_timestamps.load(std::memory_order_acquire);
    policy.barriers = policy_barriers.load(std::memory_order_acquire);
    policy.hoist_implicit_samples = policy_hoist_implicit_samples.load(std::memory_order_acquire);
    return policy;
}

std::optional<ApiVersion> selected_api_version() noexcept {
    const auto packed = selected.load(std::memory_order_acquire);
    if (!packed) return std::nullopt;
    return ApiVersion{static_cast<int>(packed >> 16), static_cast<int>(packed & 0xffff)};
}

extern "C" GenericFunction armigo_nvn_bootstrap_loader(const char* name) {
    if (!name || (selected.load(std::memory_order_acquire) >> 16) != 55) return nullptr;
    const auto entry = find_entry(name);
    if (!entry) {
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN: unknown entry point %s\n", name);
        return nullptr;
    }
    const auto index = static_cast<std::size_t>(*entry);
    const auto function = target_for(index);
    return counting_requested() ? count_thunk(index) : function;
}

namespace {
Backends installed_backends;

struct UnsupportedReport {
    UnsupportedHandler handler;
    void* context;
};
std::atomic<UnsupportedReport> unsupported_report{};
} // namespace

void set_unsupported_handler(UnsupportedHandler handler, void* context) noexcept {
    unsupported_report.store({handler, handler ? context : nullptr}, std::memory_order_release);
}

void report_unsupported(const char* detail, std::uint64_t guest_return) noexcept {
    armigo::diag::write(armigo::diag::Level::error, "%s\n", detail);
    const auto report = unsupported_report.load(std::memory_order_acquire);
    if (!report.handler) return;
    report.handler(report.context, detail, guest_return);
    // The run is stopping with this result. Returning would hand the game a
    // value no driver produced, so the thread stays here until process exit.
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
}

void set_backends(Backends backends) noexcept { installed_backends = backends; }

namespace {
std::atomic<HostStackProvider> host_stack_provider{nullptr};
} // namespace

void set_host_stack_provider(HostStackProvider provider) noexcept {
    host_stack_provider.store(provider, std::memory_order_release);
}

std::uintptr_t host_stack_top() noexcept {
    const auto provider = host_stack_provider.load(std::memory_order_acquire);
    if (!provider) return 0;
    const std::uintptr_t top = provider();
    return (top & 15) == 0 ? top : 0;
}
const Backends& backends() noexcept { return installed_backends; }

std::size_t call_counts(std::span<CallCount> out) noexcept {
    const auto written = std::min(out.size(), entry_count);
    for (std::size_t index = 0; index < written; ++index) {
        out[index] = {
            names[index], armigo_nvn_entry_slots[index].calls.load(std::memory_order_relaxed)};
    }
    return entry_count;
}

std::size_t sample_call_count_frame(CallCountFrameState& state,
                                    std::span<CallCount> scratch,
                                    std::size_t top_entry_count,
                                    CallCountFrameHook hook,
                                    void* context) noexcept {
    const auto limit = std::min(scratch.size(), top_entry_count);
    std::size_t used = 0;
    state.total_calls = 0;
    const auto better = [](const CallCount& left, const CallCount& right) {
        return left.calls > right.calls || (left.calls == right.calls && left.name < right.name);
    };

    for (std::size_t index = 0; index < entry_count; ++index) {
        const auto current = armigo_nvn_entry_slots[index].calls.load(std::memory_order_relaxed);
        const auto delta = current - state.previous[index];
        state.previous[index] = current;
        state.total_calls += delta;
        if (delta == 0 || limit == 0) continue;

        const CallCount candidate{names[index], delta};
        if (used == limit && !better(candidate, scratch[used - 1])) continue;

        std::size_t position = used < limit ? used++ : used - 1;
        while (position > 0 && better(candidate, scratch[position - 1])) {
            scratch[position] = scratch[position - 1];
            --position;
        }
        scratch[position] = candidate;
    }

    const auto top = scratch.first(used);
    if (hook) hook(top, context);
    return used;
}

void print_call_count_frame(std::span<const CallCount> top_entries, void* context) noexcept {
    auto* output = context ? static_cast<std::FILE*>(context) : stderr;
    std::uint64_t displayed_calls = 0;
    for (const auto& entry : top_entries) displayed_calls += entry.calls;
    std::fprintf(output, "Armigo NVN: frame top entries (%zu, %llu displayed calls)\n",
                 top_entries.size(), static_cast<unsigned long long>(displayed_calls));
    for (const auto& entry : top_entries)
        std::fprintf(output, "  %.*s: %llu\n", static_cast<int>(entry.name.size()), entry.name.data(),
                     static_cast<unsigned long long>(entry.calls));
}

} // namespace armigo::nvn
