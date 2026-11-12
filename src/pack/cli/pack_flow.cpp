// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pack_flow.h"

#include "pso_manifest.h"

#include <cstdlib>
#include <system_error>

namespace armigo::pack {

std::vector<std::filesystem::path> select_pack_manifests(const compat::PatchDescription& description,
                                                         const std::filesystem::path& shaders,
                                                         InputCapabilities capabilities) {
    if (!capabilities.native_shaders) {
        return {description.pso_manifest};
    }
    std::error_code manifest_error;
    const std::filesystem::path compute_list =
        shaders / "native" / armigo::platform::metal::nvn_driver::compute_pipeline_manifest_file;
    const bool compute_present = std::filesystem::is_regular_file(compute_list, manifest_error);
    const auto selected = armigo::platform::metal::nvn_driver::select_pipeline_manifests(
        description.pso_manifest, compute_list.string(), compute_present);
    std::vector<std::filesystem::path> manifests;
    manifests.reserve(selected.size());
    for (const std::string& path : selected) {
        manifests.emplace_back(path);
    }
    return manifests;
}

std::vector<std::filesystem::path> warm_up_manifests(const compat::PatchDescription& description,
                                                     const std::filesystem::path& app,
                                                     InputCapabilities capabilities) {
    return select_pack_manifests(description, app / "Contents" / "Resources" / "Armigo" / "shaders", capabilities);
}

std::filesystem::path default_shader_compile_cache(const std::string& title_id) {
    if (std::getenv("ARMIGO_SHADER_COMPILE_CACHE") != nullptr) {
        return {};
    }
    const char* home = std::getenv("HOME");
    if (home == nullptr) {
        return {};
    }
    return std::filesystem::path(home) / "Library" / "Caches" / "Armigo" / title_id / "compile-cache";
}

void fill_common_bundle_fields(BundleSpec& spec, const CommonBundleFields& fields,
                               const compat::PatchDescription& description) {
    spec.title_id = fields.title_id;
    spec.build_id = fields.build_id;
    spec.commit = fields.commit;
    spec.player = fields.player;
    spec.title_folder = fields.title_folder;
    spec.platform = fields.platform;
    if (fields.apply_frame_rates && description.frame_rate) {
        spec.default_frame_rate = description.frame_rate->default_rate;
        spec.display_refresh_rate = description.frame_rate->display_refresh_rate;
        spec.supported_frame_rates = description.frame_rate->supported_rates;
        spec.experimental_frame_rates = description.frame_rate->experimental_rates;
    }
}

} // namespace armigo::pack
