// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The stage buffer of task and mesh stages. ISBERD and ISBEWR address it in
// linear bytes; a translated stage keeps it in its workgroup memory, all zero
// when the stage starts:
//
//   +0              the workgroup's number (ISBERD mode 0x78)
//   +0x10           the input: the payload a mesh stage's task stage wrote
//                   (ISBERD mode 0x38); its word 0 is the mesh workgroup's
//                   number among its task's launches
//   output          a task stage's payload (ISBEWR and ISBERD mode 0xb8): word
//                   1 is the number of mesh workgroups to launch; or a mesh
//                   stage's attributes (ISBEWR mode 0xf8): every block of 32
//                   vertices, one 128-byte row per output component the
//                   header's output map enables (a word per vertex), then
//                   every block of 32 primitives, a row per component of the
//                   per-primitive header's map
//   primitives      a mesh stage's primitive bytes: the count at +3 (ISBEWR
//                   mode 0x88), index bytes from +4 (mode 0x80), the control's
//                   vertices per primitive each, reserved for whole blocks of
//                   32 primitives
//
// output is 0x10 plus the input rounded up to 16 bytes: the payload, and at
// least the word the hardware writes for a mesh stage under a task stage. A
// task stage's buffer holds at least two payload words; a payload shorter
// than 8 bytes holds no launch count (word 1) and launches none.

#pragma once
#include <armigo/shader_native/program.h>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace armigo::shader_native {
// An output component an SPH output map enables: point size, a position
// component or a component of generic attribute 0-31.
struct StageComponent {
    enum class Kind : std::uint8_t { point_size, position, generic } kind{};
    std::uint8_t attribute{}, component{};
    friend bool operator==(const StageComponent&, const StageComponent&) = default;
};
// In stage-buffer order: point size (byte 0x35 bit 3), position x-w (bits
// 4-7), then each generic attribute's components (from byte 0x36, a nibble an
// attribute, bit c its component c).
std::vector<StageComponent> output_components(std::span<const std::byte> header);

inline constexpr std::uint32_t stage_block = 32;
// Metal's limits for a mesh or object function: threadgroup memory, an object
// payload, and a mesh's vertices and primitives.
inline constexpr std::uint32_t metal_threadgroup_bytes = 0x8000;
inline constexpr std::uint32_t metal_payload_bytes = 0x4000;
inline constexpr std::uint32_t metal_mesh_vertices = 256, metal_mesh_primitives = 512;

struct StageMemory {
    std::uint32_t workgroup = 0, input = 0x10, output = 0, primitives = 0, bytes = 0;
    // Mesh stages: the vertex and per-primitive output components.
    std::vector<StageComponent> vertex, primitive;
};
// The layout of a task or mesh program's stage buffer, or why it cannot be
// kept in Metal workgroup memory.
std::variant<StageMemory, std::string> stage_memory(const Program& program);
} // namespace armigo::shader_native
