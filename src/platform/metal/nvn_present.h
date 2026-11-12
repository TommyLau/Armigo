// ============================================================================
// Armigo Platform - NVN Texture Presentation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include "../../nvn/present.h"
#import <Metal/Metal.h>

namespace armigo::platform::metal::nvn_driver {

// The resolver normally calls Resources::texture(state, nullptr). Each present
// resolves the current allocation and retains it through presentation. The
// lookup returns ready contents; resource-backed lookups use the overload below.
class Resources;
class MetalReplayBackend;

using PresentationTextureLookup =
    std::function<id<MTLTexture>(const armigo::nvn::TextureState &)>;
armigo::nvn::TextureExporter
make_texture_exporter(id<MTLDevice> device, PresentationTextureLookup lookup);
// Publishes lookup uploads and returns a pinned allocation with an ordered consumer context.
armigo::nvn::TextureExporter make_texture_exporter(id<MTLDevice> device, Resources& resources);
// Window templates capture each frame on the replay worker before a later
// submission can replace or overwrite its native allocation.
armigo::nvn::TextureExporter make_texture_exporter(id<MTLDevice> device, Resources& resources,
                                                  MetalReplayBackend& replay);

} // namespace armigo::platform::metal::nvn_driver
