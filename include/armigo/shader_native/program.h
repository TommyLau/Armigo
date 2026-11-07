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
#include <armigo/shader_native/input.h>
#include <array>
#include <variant>

namespace armigo::shader_native {
enum class Stage : std::uint8_t { vertex = 0, fragment = 1, compute = 5 };
enum class ProgramError { truncated_control, control_family, unsupported_stage, stage_slot,
                          payload_size, program_bounds, instruction_alignment, nonzero_padding };
struct ProgramFailure {
    ProgramError code;
    std::uint8_t selector{};
    std::string message;
};
struct ComputeMetadata {
    // Getter-verified words; their launch meaning is not yet established.
    std::array<std::uint32_t, 3> stage_words{};
};
struct ComputeLaunchFacts {
    std::array<std::uint32_t, 3> local_size{};
    std::uint32_t shared_memory_bytes{};
    std::uint32_t local_memory_bytes{};
};
struct Program {
    Stage stage{};
    Bytes control, payload, image, header, instructions, padding, constants, trailing_padding;
    std::optional<ComputeMetadata> compute_metadata;
    // No launch facts are synthesized from opaque control words.
    std::optional<ComputeLaunchFacts> launch;
    std::string image_digest, control_digest, payload_digest, stage_identity;
};
using ProgramResult = std::variant<Program, ProgramFailure>;
// Architecture/control schema and exact complete bytes define the stage identity.
// Container ancestry and selected package/build provenance remain separate.
ProgramResult parse_program(const Record& record);
} // namespace armigo::shader_native
