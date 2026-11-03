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

#include <cstddef>

// Pure rules for which attachment work a render pass can skip. They hold no
// Metal state, so they are tested without a device.
namespace armigo::platform::metal::nvn_driver {

// Clears recorded for earlier render targets stay pending until a pass opens.
// A clear folds into the pass's load actions only when the pass attaches the
// cleared view; the others land as clear passes before it. Clears land in
// their recorded order, so everything up to and including the last clear the
// pass cannot fold lands, and only the clears after it fold. Returns how many
// leading clears land; foldable(index) says whether the pass folds clear index.
template <class Foldable>
inline std::size_t clears_to_land(std::size_t count, Foldable&& foldable) {
    for (std::size_t index = count; index-- > 0;)
        if (!foldable(index)) return index + 1;
    return 0;
}

// What the draws of one pass do with the depth and stencil attachment.
struct DepthStencilUse {
    bool depth_test, depth_write, stencil_test, fragment_depth_export;
};

inline bool uses_depth_stencil(const DepthStencilUse& use) {
    return use.depth_test || use.depth_write || use.stencil_test || use.fragment_depth_export;
}

// A pass opens without its depth and stencil attachment (no load, no store)
// when the draw that opens it does not use them, nothing is clearing them, a
// colour attachment remains, and no earlier pass of the same kind needed them.
inline bool may_omit_depth_stencil(bool enabled, unsigned colour_writers, bool pending_clear,
                                   const DepthStencilUse& use, bool kind_needs_depth) {
    return enabled && colour_writers && !pending_clear && !uses_depth_stencil(use) && !kind_needs_depth;
}

} // namespace armigo::platform::metal::nvn_driver
