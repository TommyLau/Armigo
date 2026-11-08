// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/nvn/driver.h>
#include <armigo/runtime/module_set.h>
#include <armigo/execution/prepared_image.h>

namespace armigo::nvn {

struct DriverSelection {
    std::optional<ApiVersion> version;
    const char* error = nullptr;
};
// An explicit driver override bypasses title opt-in, never version admission.
[[nodiscard]] DriverSelection select_driver(bool title_enabled,
    std::span<const std::byte> sdk_image, std::string_view override_value = {}) noexcept;

struct RedirectResult {
    const char* error = nullptr;
    std::uint64_t image_offset = 0;
    [[nodiscard]] bool ok() const noexcept { return error == nullptr; }
};
// Call after apply_prepared_image, before sealing. The complete preparation plan
// and executable text range are required to reject overlap before any mutation.
// The caller selects the admitted API version and installs driver entries before
// running guests. No borrowed data is retained and no game file is changed.
// An embedded stub offset selects a 12-byte appendix stub that loads the host
// target from appended_data's NVN key word instead of embedding its address.
// The host target is `loader` (bootstrap_loader_for the selected version), the
// API 55 loader when null.
[[nodiscard]] RedirectResult redirect_bootstrap_loader(
    const runtime::ModulePreparationView& view,
    const execution::PreparedImage& prepared,
    std::uint64_t text_offset, std::uint64_t text_size,
    std::optional<std::size_t> embedded_stub_offset = {},
    BootstrapLoader loader = nullptr) noexcept;

void register_null_entries() noexcept;

} // namespace armigo::nvn
