// ============================================================================
// Armigo NVN - Presentation Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "objects.h"
#include <armigo/services/vi_producer.h>
#include <functional>
#include <optional>

namespace armigo::nvn {

// Export pins a particular texture serial until VI releases the registration.
// The returned reader must read completed GPU content, not stale pool memory.
using TextureExporter =
    std::function<std::optional<services::vi::NativeTexture>(
        const TextureState &)>;
// Install before initializing windows; existing registrations retain their
// reader.
void set_texture_exporter(TextureExporter exporter);
void register_present_entries();

} // namespace armigo::nvn
