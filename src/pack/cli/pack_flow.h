// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "bundle.h"

#include <armigo/compat/patches.h>

#include <filesystem>
#include <string>
#include <vector>

namespace armigo::pack {

// What one pack input kind can do. The pack flow branches on these flags.
struct InputCapabilities {
    // XCI, NSP and folder packs accept updates and add-on content. A .nup pack rejects both.
    bool updates_and_addons = false;
    // XCI, NSP and folder packs can target iOS. A .nup pack is macOS only.
    bool ios = false;
    // XCI, NSP and folder packs prepare and extract firmware resources. A .nup pack does not embed them.
    bool embeds_firmware = false;
    // A .nup pack translates native shader programs. An XCI, NSP or folder pack translates shader containers.
    bool native_shaders = false;
    // A macOS XCI, NSP or folder pack requires the title's pipeline list. A .nup pack treats that list as optional.
    bool pipeline_list_required = false;
};

[[nodiscard]] inline InputCapabilities game_input_capabilities() {
    return InputCapabilities{
        .updates_and_addons = true,
        .ios = true,
        .embeds_firmware = true,
        .native_shaders = false,
        .pipeline_list_required = true,
    };
}

[[nodiscard]] inline InputCapabilities nup_input_capabilities() {
    return InputCapabilities{
        .updates_and_addons = false,
        .ios = false,
        .embeds_firmware = false,
        .native_shaders = true,
        .pipeline_list_required = false,
    };
}

// Manifests for one pipeline archive build and the matching warm-up, in list order.
// An XCI, NSP or folder pack contributes only the title list, as one path, even when that path is empty.
// A .nup pack contributes the title list when it is set, then the packaged compute list when that file is present
// under `shaders`.
[[nodiscard]] std::vector<std::filesystem::path> select_pack_manifests(
    const compat::PatchDescription& description, const std::filesystem::path& shaders,
    InputCapabilities capabilities);

// The lists the player's warm-up reads once a macOS app is published at
// `app`: the same selection, with a packaged list found in the published app's
// shaders rather than where the app was assembled.
[[nodiscard]] std::vector<std::filesystem::path> warm_up_manifests(
    const compat::PatchDescription& description, const std::filesystem::path& app, InputCapabilities capabilities);

// The compile cache used when ARMIGO_SHADER_COMPILE_CACHE is unset:
// ~/Library/Caches/Armigo/<title id>/compile-cache. An empty path leaves the cache to the package library.
[[nodiscard]] std::filesystem::path default_shader_compile_cache(const std::string& title_id);

// Bundle fields both pack kinds set from the matched title. Frame rates are a macOS launch feature;
// `apply_frame_rates` is false for an iOS pack, which keeps the player's ordinary display rate.
struct CommonBundleFields {
    std::string title_id;
    std::string build_id;
    std::string commit;
    std::filesystem::path player;
    std::filesystem::path title_folder;
    std::string platform;
    bool apply_frame_rates = false;
};

void fill_common_bundle_fields(BundleSpec& spec, const CommonBundleFields& fields,
                               const compat::PatchDescription& description);

} // namespace armigo::pack
