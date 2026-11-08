// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Every NVN entry point has an Entry value and, when its signature is known,
// a function type. Driver subsystems install their implementations with
// install<Entry::name>(function); the compiler checks the signature. Entry
// points without an implementation resolve to a per-entry stub that reports
// the name once and returns zero.
//
// Host functions take guest arguments under the platform ABI: parameters
// narrower than 32 bits (booleans) are declared as 32-bit and only their low
// bits are read, because guest callers do not extend them.

#pragma once

#include <armigo/nvn/driver.h>
#include <armigo/nvn/entry_points.h>

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

namespace armigo::nvn {

using namespace api;

enum class Entry : std::uint16_t {
#define ARMIGO_NVN_TYPED(name, result, parameters) name,
#define ARMIGO_NVN_UNTYPED(name) name,
    ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
    ARMIGO_NVN_UNTYPED_ENTRY_POINTS(ARMIGO_NVN_UNTYPED)
#undef ARMIGO_NVN_TYPED
#undef ARMIGO_NVN_UNTYPED
    count
};

inline constexpr std::size_t entry_count = static_cast<std::size_t>(Entry::count);

template <Entry E>
struct EntrySignature;

#define ARMIGO_NVN_TYPED(name, result, parameters)       \
    template <>                                          \
    struct EntrySignature<Entry::name> {                 \
        using type = result(*) parameters;               \
    };
ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
#undef ARMIGO_NVN_TYPED

// Untyped entry points take any signature until their parameters are known.
#define ARMIGO_NVN_UNTYPED(name)                         \
    template <>                                          \
    struct EntrySignature<Entry::name> {                 \
        using type = GenericFunction;                    \
    };
ARMIGO_NVN_UNTYPED_ENTRY_POINTS(ARMIGO_NVN_UNTYPED)
#undef ARMIGO_NVN_UNTYPED

// Describes the two AArch64 argument-register banks used by an entry. Apple
// arm64 and AAPCS64 agree until one bank is exhausted. Integer inputs narrower
// than 32 bits are also unsafe because only Apple requires caller extension.
template <typename Function>
struct EntryAbi;

template <typename Result, typename... Parameters>
struct EntryAbi<Result (*)(Parameters...)> {
private:
    template <typename T>
    static constexpr bool floating_parameter =
        std::is_same_v<std::remove_cv_t<T>, float> || std::is_same_v<std::remove_cv_t<T>, double>;

    template <typename T>
    static constexpr bool narrow_integer =
        (std::is_integral_v<std::remove_cv_t<T>> || std::is_enum_v<std::remove_cv_t<T>>) &&
        sizeof(std::remove_cv_t<T>) < sizeof(std::uint32_t);

    template <typename T>
    static constexpr bool overaligned_integer_parameter =
        !floating_parameter<T> && alignof(std::remove_cv_t<T>) > alignof(std::uint64_t);

    template <typename T>
    static constexpr bool empty_record_parameter =
        std::is_class_v<std::remove_cv_t<T>> && std::is_empty_v<std::remove_cv_t<T>>;

public:
    static constexpr std::size_t integer_parameter_count =
        (std::size_t{0} + ... + (floating_parameter<Parameters> ? 0u : 1u));
    static constexpr std::size_t floating_parameter_count =
        (std::size_t{0} + ... + (floating_parameter<Parameters> ? 1u : 0u));
    static constexpr bool has_narrow_integer_parameter = (false || ... || narrow_integer<Parameters>);
    static constexpr bool has_overaligned_integer_parameter =
        (false || ... || overaligned_integer_parameter<Parameters>);
    static constexpr bool has_empty_record_parameter =
        (false || ... || empty_record_parameter<Parameters>);
    static constexpr bool has_narrow_integer_result = [] {
        if constexpr (std::is_void_v<Result>)
            return false;
        else
            return narrow_integer<Result>;
    }();
    static constexpr bool needs_stack_shim =
        integer_parameter_count > 8 || floating_parameter_count > 8;
};

template <Entry E>
using EntryAbiFor = EntryAbi<typename EntrySignature<E>::type>;

// The current typed contract needs neither caller-extension nor stack-layout
// adaptation. A newly typed entry that violates either assertion must install
// an assembly shim instead of being handed out directly.
#define ARMIGO_NVN_TYPED(name, result, parameters)                                      \
    static_assert(!EntryAbiFor<Entry::name>::has_narrow_integer_parameter,              \
                  #name " has an ABI-unsafe narrow integer parameter");                 \
    static_assert(!EntryAbiFor<Entry::name>::has_overaligned_integer_parameter,         \
                  #name " requires aligned-register adaptation");                       \
    static_assert(!EntryAbiFor<Entry::name>::has_empty_record_parameter,                \
                  #name " requires empty-record adaptation");                           \
    static_assert(!EntryAbiFor<Entry::name>::needs_stack_shim,                          \
                  #name " requires an AAPCS64-to-Apple-arm64 stack shim");
ARMIGO_NVN_ENTRY_POINTS(ARMIGO_NVN_TYPED)
#undef ARMIGO_NVN_TYPED

[[nodiscard]] std::string_view entry_name(Entry entry) noexcept;
[[nodiscard]] std::optional<Entry> find_entry(std::string_view name) noexcept;

void install_entry(Entry entry, GenericFunction function) noexcept;
// The implementation currently installed for an entry point (its reporting
// stub when none is), never a counting thunk: wrappers call through it.
[[nodiscard]] GenericFunction installed_entry(Entry entry) noexcept;
// The installed implementation, or null while the entry point has none.
[[nodiscard]] GenericFunction implemented_entry(Entry entry) noexcept;

// Writes detail as an error and hands it to the runtime's unsupported handler.
// Returns only when no handler is installed.
void report_unsupported(const char* detail, std::uint64_t guest_return) noexcept;

template <Entry E>
void install(typename EntrySignature<E>::type function) noexcept {
    install_entry(E, reinterpret_cast<GenericFunction>(function));
}

// Registers the recording lane and its independently usable state/program subsets.
void register_command_entries() noexcept;
void register_state_entries() noexcept;
void register_program_entries() noexcept;

// Booleans as received from the guest: only the low 8 bits are defined.
constexpr bool guest_bool(std::uint32_t value) noexcept { return (value & 0xffu) != 0; }

struct CallCountFrameState {
    std::array<std::uint64_t, entry_count> previous{};
    std::uint64_t total_calls = 0;
};

using CallCountFrameHook = void (*)(std::span<const CallCount> top_entries, void* context) noexcept;

// Samples one frame, updates state, keeps the largest deltas in scratch, and
// invokes hook once with those entries in descending call-count order.
[[nodiscard]] std::size_t sample_call_count_frame(CallCountFrameState& state,
                                                  std::span<CallCount> scratch,
                                                  std::size_t top_entry_count,
                                                  CallCountFrameHook hook,
                                                  void* context) noexcept;

// A CallCountFrameHook that writes a compact summary to the FILE* in context,
// or stderr when context is null.
void print_call_count_frame(std::span<const CallCount> top_entries, void* context) noexcept;

} // namespace armigo::nvn
