// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Task and mesh stages as Metal object and mesh functions. The shared
// emitter writes such a stage as a compute kernel whose workgroup memory is
// the stage buffer (stage_memory.h). Its body becomes an inline function that
// takes that memory, and the entry function owns it:
//
//   1. every word zero, then the workgroup's number at +0: the draw's first
//      workgroup (word 0 of the draw-parameter record) plus the threadgroup's
//      position; for a mesh stage under a task stage, the payload copied to
//      the input and the mesh workgroup's number among its task's launches,
//      also input word 0
//   2. the guest body, every thread, an early exit returning to the entry
//   3. a task stage's payload to the object payload, and its word 1 as the
//      mesh grid; or a mesh stage's primitive count, vertices, per-primitive
//      attributes and index bytes to the Metal mesh
//
// A task workgroup asking for more than Metal's 1,024 mesh workgroups
// launches none and records the request: status word 0 holds the largest
// such count and word 1 the largest program tag (the stage identity's first
// four bytes), in the buffer index after the draw parameters, which the
// driver binds zeroed and reports from. A mesh whose declared outputs exceed
// Metal's 32 KB is refused.
//
// Vertex outputs are the components the header's output map enables, named
// as a native vertex stage names them. The position is invariant, as a
// vertex stage's, and takes the vertex stage's depth convention. A point
// mesh outputs its point size, 1.0 when the output map has none. With its
// paired fragment stage's generic reads, a mesh outputs exactly the generic
// components that stage reads, defaulting those its output maps lack.

#pragma once
#include "stage_memory.h"
#include <armigo/shader_native/frontend.h>
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace armigo::shader_native {
// Metal's mesh workgroups per object threadgroup and mesh output bytes.
inline constexpr std::uint32_t metal_mesh_grid = 1024;
inline constexpr std::uint32_t metal_mesh_output_bytes = 32768;
// Rewrites artifact (the stage's emitted kernel) into its object or mesh
// function. Returns why it cannot, or nothing.
std::optional<std::string> wrap_workgroup_stage(ir::MslArtifact& artifact, const Program& program, const StageMemory& memory,
                                                std::optional<ir::DepthConvention> depth,
                                                const std::optional<std::array<std::uint8_t, 32>>& fragment_reads = std::nullopt);
} // namespace armigo::shader_native
