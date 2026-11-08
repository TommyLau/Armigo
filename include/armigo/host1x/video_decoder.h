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
#include <optional>
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

// An AV1 decoder of one stream. Each call takes one temporal unit (OBUs with
// size fields, as in an MP4 sample) and appends the frames it shows; the first
// unit must carry the sequence header.
class Av1Decoder {
public:
    virtual ~Av1Decoder() = default;
    virtual bool decode(std::span<const std::uint8_t> temporal_unit,
                        std::vector<std::shared_ptr<const DecodedFrame>>& shown, std::string& failure) = 0;
};
// Null, with `failure` naming the reason, when the host has no hardware AV1
// decoder.
std::unique_ptr<Av1Decoder> make_av1_decoder(std::string& failure);

// The AV1 codec configuration record (av1C) for the sequence header OBU among
// `obus`, which follows it as its configuration OBU; nothing when there is no
// well-formed sequence header.
std::optional<std::vector<std::uint8_t>> av1_configuration_record(std::span<const std::uint8_t> obus);

} // namespace armigo::host1x
