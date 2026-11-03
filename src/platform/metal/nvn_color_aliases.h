// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#import <Metal/Metal.h>
#include <array>
#include <bit>
#include <cstdint>

namespace armigo::platform::metal::nvn_driver {

struct RenderColorSubresource {
    id<MTLTexture> texture;
    NSUInteger level, slice, plane;

    bool operator==(const RenderColorSubresource&) const = default;
};

inline RenderColorSubresource render_color_subresource(MTLRenderPassColorAttachmentDescriptor* slot) {
    RenderColorSubresource result{slot.texture, slot.level, slot.slice, slot.depthPlane};
    while (result.texture.parentTexture) {
        result.level += result.texture.parentRelativeLevel;
        result.slice += result.texture.parentRelativeSlice;
        result.texture = result.texture.parentTexture;
    }
    if (result.texture.textureType == MTLTextureType3D) {
        result.plane += result.slice;
        result.slice = 0;
    }
    return result;
}

// The same subresource attached to several color slots: on the guest GPU the
// slots are one memory, so a slot a draw does not write leaves it untouched.
// Metal gives each attachment its own tile storage, and an unwritten duplicate
// stores its loaded contents back over the written slot. Each group keeps one
// slot attached, the lowest one the pass's opening draw writes, else the
// lowest; the others are detached, so their pipeline format is invalid and
// writes to them are discarded. A later draw that writes only detached slots
// of a group reopens the pass with one of them kept.
struct RenderColorAliases {
    // groups[i]: every slot sharing slot i's subresource, set on the lowest.
    std::array<std::uint8_t, 8> groups{};
    bool any = false;

    explicit RenderColorAliases(MTL4RenderPassDescriptor* pass = nil) {
        if (!pass) return;
        std::array<RenderColorSubresource, 8> resources{};
        for (unsigned i = 0; i < 8; ++i)
            resources[i] = render_color_subresource(pass.colorAttachments[i]);
        unsigned visited = 0;
        for (unsigned i = 0; i < 8; ++i) {
            if (!resources[i].texture || (visited & (1u << i))) continue;
            unsigned members = 1u << i;
            for (unsigned j = i + 1; j < 8; ++j)
                if (resources[i] == resources[j]) members |= 1u << j;
            visited |= members;
            if (std::popcount(members) > 1) {
                groups[i] = static_cast<std::uint8_t>(members);
                any = true;
            }
        }
    }

    // The slots to attach for a pass whose opening draw writes `writers`.
    std::uint8_t attachments(std::uint8_t writers) const {
        unsigned attached = 0xff;
        for (unsigned members : groups) {
            if (!members) continue;
            unsigned keep = members & writers;
            if (!keep) keep = members;
            attached = (attached & ~members) | (keep & (0u - keep));
        }
        return static_cast<std::uint8_t>(attached);
    }

    // Whether a draw writing `writers` writes a group only through detached slots.
    bool needs_reopen(std::uint8_t writers, std::uint8_t attached) const {
        for (unsigned members : groups)
            if ((writers & members & ~attached) && !(writers & members & attached)) return true;
        return false;
    }
};

} // namespace armigo::platform::metal::nvn_driver
