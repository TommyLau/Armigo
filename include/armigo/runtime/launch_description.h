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

#include <armigo/loader/exefs_directory.h>
#include <armigo/platform/presenter.h>
#include <armigo/runtime/launch_profile.h>
#include <armigo/services/operation_mode.h>
#include <chrono>
#include <filesystem>
#include <optional>
#include <vector>

namespace armigo::runtime {

// Lane I supplies explicit paths; lane F collects with read_exefs_directory
// and opens RomFS via the existing romfs::Storage implementation. Inputs stay
// unchanged through collection/preparation admission. These are host inputs,
// not Horizon wire structures. Collection transports bytes; it does not parse
// process metadata or choose a load order.
// No container extraction, key handling, private environment fallback, missing
// NPDM fallback or SDK version selection is supported by this interface.
struct LaunchDescription {
    // Title settings (titles/<TitleID>/, including the NVN driver opt-in) are
    // found by the admitted title id (compat::find_title_settings).
    // ARMIGO_NVN=driver|stream overrides opt-in; known API admission still applies.
    // Saves go to the title's save directory (compat::title_save_directory)
    // unless the caller set a host save directory: the windowed player.
    bool title_saves = false;
    std::optional<std::uint64_t> container_title_id;
    std::filesystem::path exefs_directory;
    std::filesystem::path romfs_directory;
    // Extracted add-on content: one RomFS tree per item, each directory named
    // by the item's data id (see runtime/addon_content.h). Empty installs none.
    std::vector<std::filesystem::path> addon_directories;
    // Absent uses collected main.npdm. If both sources exist their bytes must
    // match; an override must never silently widen the executable's rights.
    std::optional<std::filesystem::path> npdm_file;
    std::filesystem::path prepared_directory;
    // Native images prepared at pack time, mutually exclusive with prepared_directory.
    std::filesystem::path embedded_directory;
    std::filesystem::path shader_package_directory;
    platform::PresenterOptions presentation;
    services::OperationMode operation_mode = services::OperationMode::Console;
    loader::ExeFsReadLimits input_limits;
    NpdmTrustPolicy npdm_trust = NpdmTrustPolicy::require_authenticated;
    // Only independently authored test workloads may use this. Mutually
    // exclusive with NPDM input; validate capabilities exactly as ordinary input.
    std::optional<LaunchProfile> synthetic_profile;
};
struct RunLimits {
    // A value must be positive. Absence means no bound of this kind.
    std::optional<std::uint64_t> completed_frame_target;
    std::optional<std::chrono::nanoseconds> timeout;
};
struct ReportDestination {
    // Explicit external output only; absence disables capture/report writing.
    std::optional<std::filesystem::path> capture_directory;
    std::optional<std::filesystem::path> report_file;
};

} // namespace armigo::runtime
