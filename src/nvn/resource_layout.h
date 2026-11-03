// ============================================================================
// Armigo NVN - Driver Backends
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "backend.h"
#include <armigo/graphics/texture_format.h>
#include <armigo/graphics/texture_layout.h>
#include <optional>

namespace armigo::nvn {

// Common storage interpretation for API sizing and host resource transfers.
std::optional<graphics::TextureFormat> resource_format(std::int32_t format);
std::optional<graphics::TextureLayoutDescription> resource_layout(const TextureDescription& description);
std::size_t resource_alignment(const TextureDescription& description);
// Capture strided input before coherence operations can overwrite aliased pool bytes.
bool snapshot_texel_input(const TextureState& texture, const TexelTransfer& transfer,
                          const std::byte* input, std::vector<std::byte>& bytes,
                          TexelTransfer& packed);
// Access pool bytes only; callers synchronize outstanding GPU writes first.
bool transfer_pool_texels(const TextureState& texture, const TexelTransfer& transfer,
                          std::byte* bytes, bool write);
// The block-linear addressing of every whole level of every layer of the
// texture's storage, layer-major then level (all depth slices of a 3D level
// at once), with guest_base relative to the storage start: the bytes a
// whole-level transfer_pool_texels read copies, in its order. False for pitch
// storage and for any level such a read would refuse.
bool pool_texture_addressing(const TextureState& texture,
                             std::vector<graphics::BlockLinearLevelAddressing>& levels);

} // namespace armigo::nvn
