// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/shader_input/shader_input.h>

#include <optional>
#include <vector>

namespace armigo::shader_input {

// Resource policy, independent of container format. BAHS candidate limits also
// count rejected and repeated entries, bounding validation and deduplication.
struct ContainerLimits {
    std::uint32_t max_variations = 65536;
    std::size_t max_stages = 262144;
    std::size_t max_candidates = 262144;
    std::size_t max_scan_bytes = 64u * 1024u * 1024u;
};

enum class StageSlot : std::uint8_t {
    Vertex, TessellationControl, TessellationEvaluation, Geometry, Fragment,
    Compute,
};

struct ContainerError {
    enum class Code : std::uint8_t {
        InvalidSpan, Truncated, Magic, MissingResourceBlock, Range,
        VariationLimit, StageLimit, ScanLimit, CandidateLimit,
        ControlSizeMismatch, CodeSizeMismatch, SlotStageMismatch,
        ShaderInvalid,
    };
    Code code = Code::InvalidSpan;
    std::uint64_t offset = 0;
    std::uint64_t value = 0;
    std::uint64_t bound = 0;
    std::optional<ParseError> shader_error;

    [[nodiscard]] const char* message() const noexcept;
};

// Every offset is relative to the supplied container span, including a BAHS
// entry's resolved control pointer. Every shader span borrows that same input;
// keep its storage alive, unmoved and unmodified while any result is in use.
struct ContainerStage {
    ParsedShader shader;
    std::size_t descriptor_offset = 0;
    std::size_t control_offset = 0;
    std::size_t code_offset = 0;
    // Only BNSH carries verified variation/slot membership. BAHS candidates
    // cannot establish a vertex/fragment pairing and leave these absent.
    std::optional<std::uint32_t> variation;
    std::optional<StageSlot> slot;
};

struct BnshContainer {
    std::uint32_t variation_count = 0;
    // Variation-major, then slot-major order; empty variations remain counted.
    std::vector<ContainerStage> stages;
};

struct BahsContainer {
    std::size_t scanned_positions = 0;
    std::size_t candidate_count = 0;
    std::size_t rejected_candidates = 0;
    std::size_t duplicate_candidates = 0;
    // Entry order, deduplicated by resolved control offset and code offset.
    std::vector<ContainerStage> stages;
};

using BnshResult = std::variant<BnshContainer, ContainerError>;
using BahsResult = std::variant<BahsContainer, ContainerError>;

// Inputs begin at the container magic. All range checks use the supplied span
// as the container boundary; callers extracting an embedded container must
// supply its known extent. Allocation failure propagates std::bad_alloc.
[[nodiscard]] BnshResult parse_bnsh(ByteSpan input,
                                  ContainerLimits limits = {});
[[nodiscard]] BahsResult parse_bahs(ByteSpan input,
                                  ContainerLimits limits = {});

}  // namespace armigo::shader_input
