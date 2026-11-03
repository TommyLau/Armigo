// ============================================================================
// Armigo NVN - Graphics Driver
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Armigo's NVN driver replaces the game's own NVN driver; every other SDK
// function runs unchanged. The game reaches the driver through
// nvnBootstrapLoader, whose entry in the SDK module is redirected in memory
// at load time to bootstrap_loader below. Every entry point it hands out is a
// host function the guest calls directly (no gate): guest x18 lives in
// Armigo's per-thread cell and guest TLS reads are rewritten, so host code
// runs with its own registers and TLS on the guest thread and stack.
//
// The driver is enabled per supported title by its private compatibility
// description, and only for an NVN API version this implementation knows;
// otherwise the game's own driver runs on the command-stream renderer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace armigo::nvn {

struct ApiVersion {
    std::int32_t major = 0;
    std::int32_t minor = 0;
    bool operator==(const ApiVersion&) const = default;
};

// The NVN driver package a module ships, from its SDK middleware marker
// ("NintendoSDK_NVN-<version>-Release"), e.g. "20_5_22".
[[nodiscard]] std::optional<std::string_view> driver_package(std::span<const std::byte> image) noexcept;

// The API version a driver package reports, when this implementation knows it.
[[nodiscard]] std::optional<ApiVersion> known_api_version(std::string_view package) noexcept;

using GenericFunction = void (*)();

// Host entry for the game's nvnBootstrapLoader: returns the driver's entry
// point for an NVN API name, a stub that reports an unimplemented entry point
// for a known name, or null for an unknown name.
extern "C" GenericFunction armigo_nvn_bootstrap_loader(const char* name);

// Selects the API version the entry points describe. Called once before the
// game runs; the default is no driver (bootstrap returns null).
void select_api_version(ApiVersion version) noexcept;

// The selected API version, when a known one is selected.
[[nodiscard]] std::optional<ApiVersion> selected_api_version() noexcept;

// Calls of each entry point since start (counting enabled by ARMIGO_NVN_COUNT).
struct CallCount {
    std::string_view name;
    std::uint64_t calls;
};
[[nodiscard]] std::size_t call_counts(std::span<CallCount> out) noexcept;

} // namespace armigo::nvn
