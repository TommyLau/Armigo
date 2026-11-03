// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/host1x/host1x.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace armigo::host1x {

// A host VP9 decoder: takes complete VP9 pictures (with uncompressed and
// compressed headers) in decode order and returns the pictures it shows, in
// output order, as 8-bit 4:2:0 frames.
class Vp9Decoder {
public:
    virtual ~Vp9Decoder() = default;
    virtual bool decode(std::span<const std::uint8_t> picture, std::uint32_t width, std::uint32_t height,
                        std::vector<std::shared_ptr<const DecodedFrame>>& shown, std::string& failure) = 0;
};

// The platform's VP9 decoder, or null with `failure` set.
std::unique_ptr<Vp9Decoder> make_vp9_decoder(std::string& failure);

} // namespace armigo::host1x
