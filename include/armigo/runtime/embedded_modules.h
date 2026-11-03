// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/execution/prepared_image.h>
#include <armigo/platform/host_module.h>
#include <armigo/runtime/launch_result.h>
#include <filesystem>
#include <memory>
#include <utility>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace armigo::runtime {

struct EmbeddedNativeStop {
    std::uint64_t image_offset = 0;
    std::uint32_t instruction = 0;
    std::string reason;
};
struct EmbeddedNativeHook {
    std::uint64_t return_offset = 0;
    std::uint64_t call_return_offset = 0;
    execution::PreparedLocation target;
};
struct EmbeddedNativeModule {
    std::string name;
    std::array<std::byte, 32> source_sha256{};
    // Complete pack-time image: prepared text, original unrelocated non-code
    // bytes, zero padding/key page, and prepared appendix. No runtime bindings.
    std::vector<std::byte> image;
    std::uint64_t original_image_size = 0;
    std::uint64_t key_page_offset = 0;
    std::uint64_t appended_code_offset = 0;
    std::uint64_t text_offset = 0;
    std::uint64_t text_size = 0;
    bool host_executes = true;
    std::vector<execution::PreparedGateSite> sites;
    std::vector<std::uint64_t> entry_offsets;
    std::vector<execution::PreparedLocation> foreign_entries;
    std::vector<EmbeddedNativeStop> stops;
    std::vector<EmbeddedNativeHook> hooks;
    std::uint64_t code_bytes = 0;
    std::uint64_t scalar_sites = 0;
    std::uint64_t thread_register_sites = 0;
};
struct EmbeddedNativeBacking {
    std::span<const platform::EmbeddedModulePiece> pieces;
};
struct EmbeddedNativeArtifacts {
    std::uint32_t version = 1;
    std::uint64_t title_id = 0;
    std::string preparation_identity;
    // Resolved instruction patches, call hooks, and selected NVN module.
    std::array<std::byte, 32> code_settings_sha256{};
    // Native load order; every PreparedLocation uses this namespace.
    std::vector<EmbeddedNativeModule> modules;
};

// A file-backed macOS adapter. Its shared owner retains all mappings and
// borrowed backing spans; loaded ModuleSets retain their remaps independently.
class EmbeddedNativeFiles {
public:
    struct Storage;
    explicit EmbeddedNativeFiles(std::shared_ptr<Storage> storage) : storage_(std::move(storage)) {}
    [[nodiscard]] const EmbeddedNativeArtifacts& artifacts() const noexcept;
    [[nodiscard]] std::span<const EmbeddedNativeBacking> backings() const noexcept;
private:
    std::shared_ptr<Storage> storage_;
};
[[nodiscard]] LaunchStatus write_embedded_modules(const std::filesystem::path& directory,
                                                 const EmbeddedNativeArtifacts& artifacts);
[[nodiscard]] LaunchResult<EmbeddedNativeFiles> open_embedded_modules(const std::filesystem::path& directory);

// The linked provider retains these immutable spans for the process lifetime.
// The adapter requires the manifest and code in the main immutable __TEXT
// segment and retains private read-only data snapshots. It validates checksums,
// layout and zero key pages before publishing NativeLaunchOptions data.
struct LinkedEmbeddedBundle {
    std::span<const std::byte> manifest;
    std::span<const EmbeddedNativeBacking> backings;
};
[[nodiscard]] LaunchResult<EmbeddedNativeFiles> open_linked_embedded_modules(
    const LinkedEmbeddedBundle& bundle);

// Bumped when native scanning, emission, key ABI, or capture rules change;
// includes the embedded gate digest to reject a different linked template.
[[nodiscard]] std::string installed_embedded_preparation_identity();

} // namespace armigo::runtime

// Defined by the generated embedded.cpp. An unpacked iOS player has no provider
// and reports an explicit missing-code error when asked to load a title.
extern "C" const armigo::runtime::LinkedEmbeddedBundle* armigo_linked_embedded_bundle()
    __attribute__((weak_import));
