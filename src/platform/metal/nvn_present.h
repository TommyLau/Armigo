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

// The resolver normally calls Resources::texture(state, nullptr). Export keeps
// that exact Metal object alive even if its guest object is copied or
// finalized.
using PresentationTextureLookup =
    std::function<id<MTLTexture>(const armigo::nvn::TextureState &)>;
armigo::nvn::TextureExporter
make_texture_exporter(id<MTLDevice> device, PresentationTextureLookup lookup);

} // namespace armigo::platform::metal::nvn_driver
