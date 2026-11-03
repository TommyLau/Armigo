// ============================================================================
// Armigo Platform - NVN Metal Driver Installation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <memory>
#include <string>

namespace armigo::platform {
class MetalShaderPackage;
}
namespace armigo::services::vi {
class DisplaySession;
}

namespace armigo::platform::metal::nvn_driver {

// Installs the Metal NVN driver over the null device the launch selected:
// resources, command replay, programs from the package and presentation.
// Returns an empty string on success, otherwise why nothing was installed.
//
// pso_manifest names the title's pipeline warm-up seed manifest; empty for none.
// title_cache is the title's cache directory (<Caches>/Armigo/<TitleID>): the
// run-time shader stage cache is its shaders/ and the pipeline state its pso/
// unless the environment names other places. Without a package every stage is
// prepared at run time. A packed app's pipeline archive (Armigo/pipelines in
// its resources) is looked up before the one in the cache.
[[nodiscard]] std::string install_driver(std::shared_ptr<MetalShaderPackage> package,
                                         const std::string& pso_manifest = {},
                                         const std::string& title_cache = {});

// Native windows the guest builds present through this display's layer.
void bind_native_windows(std::weak_ptr<services::vi::DisplaySession> display);

} // namespace armigo::platform::metal::nvn_driver
