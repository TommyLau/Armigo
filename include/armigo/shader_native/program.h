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
enum class Stage : std::uint8_t { vertex = 0, fragment = 1, compute = 5, task = 6, mesh = 7 };
enum class ProgramError { truncated_control, control_family, unsupported_stage, stage_slot,
                          payload_size, program_bounds, instruction_alignment, nonzero_padding };
struct ProgramFailure {
    ProgramError code;
    std::uint8_t selector{};
    std::string message;
};
// A compute stage's launch facts are static in its control: the thread
// group size at +0x58 (three words) and the shared memory bytes at +0x64.
struct ComputeLaunchFacts {
    std::array<std::uint32_t, 3> local_size{};
    std::uint32_t shared_memory_bytes{};
};
// A task or mesh stage's launch facts are static in its control too: the
// workgroup's threads (+0x58, 16 bits) and the payload bytes (+0x5a, 16 bits:
// what a task writes for its mesh workgroups, what a mesh stage reads). A mesh
// stage adds whether a task launches it (+0x5e), the vertices a primitive
// has (+0x5f: 1 point, 2 line, 3 triangle), its most vertices and primitives
// (+0x60, +0x62) and, when +0x68 bit 7 is set, the code offset of its
// per-primitive output header (+0x64).
struct MeshLaunchFacts {
    std::uint32_t threads{};
    std::uint32_t payload_bytes{};
    bool task{};
    std::uint8_t primitive_vertices{};
    std::uint32_t max_vertices{}, max_primitives{};
    std::optional<std::uint32_t> primitive_header;
};
struct Program {
    Stage stage{};
    Bytes control, payload, image, header, instructions, padding, constants, trailing_padding;
    std::optional<ComputeLaunchFacts> launch;
    std::optional<MeshLaunchFacts> mesh_launch;
    // control_digest covers the complete record control; the stage identity
    // uses canonical_control_digest (see parse_program).
    std::string image_digest, control_digest, canonical_control_digest, payload_digest, stage_identity;
    // Fragment control +0x5b requests API early tests; a header depth output
    // keeps the shader late even when that request is set.
    bool early_fragment_tests{};
};
using ProgramResult = std::variant<Program, ProgramFailure>;
// Architecture/control schema, the canonical control (reader span with the
// container extents zeroed) and the complete code define the stage identity.
// Container ancestry and selected package/build provenance remain separate.
// A runtime record's control must be exactly runtime_control_bytes long and
// its code exactly the extent the control states at +0x44.
inline constexpr std::size_t runtime_control_bytes = 0xe0;
ProgramResult parse_program(const Record& record);
} // namespace armigo::shader_native
