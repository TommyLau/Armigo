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

// The API version a driver package reports, when this implementation knows it:
// API 55 for the Switch 1 packages, API 231 for the Switch 2 package whose
// entry points the 231 table describes.
[[nodiscard]] std::optional<ApiVersion> known_api_version(std::string_view package) noexcept;

using GenericFunction = void (*)();

// Host entry for the game's nvnBootstrapLoader: returns the driver's entry
// point for an NVN API name, a stub that reports an unimplemented entry point
// for a known name, or null for an unknown name.
extern "C" GenericFunction armigo_nvn_bootstrap_loader(const char* name);

// The same for API 231 driver packages, whose entry points and objects differ
// from API 55. A stub stops the run with an unsupported result (see
// set_unsupported_handler) instead of returning zero.
extern "C" GenericFunction armigo_nvn231_bootstrap_loader(const char* name);

// The host entry an SDK's nvnBootstrapLoader is redirected to for an admitted
// API version.
using BootstrapLoader = GenericFunction (*)(const char* name);
[[nodiscard]] BootstrapLoader bootstrap_loader_for(ApiVersion version) noexcept;

// The runtime's response to a guest call the selected driver cannot serve:
// record detail, with the guest's return address, as the run's unsupported
// result and stop the process. The calling thread does not return to the
// game once a handler is installed; without one the call only reports.
using UnsupportedHandler = void (*)(void* context, const char* detail, std::uint64_t guest_return) noexcept;
void set_unsupported_handler(UnsupportedHandler handler, void* context) noexcept;

// Selects the API version the entry points describe. Called once before the
// game runs; the default is no driver (bootstrap returns null).
void select_api_version(ApiVersion version) noexcept;

// The selected API version, when a known one is selected.
[[nodiscard]] std::optional<ApiVersion> selected_api_version() noexcept;

// Driver behaviour a title's compatibility description selects; the defaults
// are the driver's own. Set once before the game runs.
struct TitlePolicy {
    // CPU writes into a CPU-cached pool reach its textures at their next use
    // without a flush call, as for uncached pools: for titles whose cache
    // maintenance runs natively and never calls the driver.
    bool cpu_cached_texture_writes = false;
    // What a timestamp report records: the GPU time of the report itself,
    // which ends the render pass (report), or the GPU start (TIMESTAMP_TOP)
    // or end (other timestamps) of the command-buffer segment it was encoded
    // in, without ending the pass (segment).
    enum class Timestamps : std::uint8_t { report, segment } timestamps = Timestamps::report;
    // How render passes are ordered: every pass after all earlier GPU work
    // (full), or a pass that directly follows another render pass only after
    // its fragment work, unless a guest barrier, a storage write in that pass
    // or other GPU work came between (game). ARMIGO_NVN_BARRIERS=full forces
    // full.
    enum class Barriers : std::uint8_t { full, game } barriers = Barriers::full;
    // Run-time shader translation also moves implicit-LOD and bias samples
    // out of their predicate guards (the project documentation).
    bool hoist_implicit_samples = false;
};
void set_title_policy(const TitlePolicy& policy) noexcept;
[[nodiscard]] TitlePolicy title_policy() noexcept;

// Calls of each entry point since start (counting enabled by ARMIGO_NVN_COUNT).
struct CallCount {
    std::string_view name;
    std::uint64_t calls;
};
[[nodiscard]] std::size_t call_counts(std::span<CallCount> out) noexcept;

} // namespace armigo::nvn
