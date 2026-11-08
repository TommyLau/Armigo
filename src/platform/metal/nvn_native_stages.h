// ============================================================================
// Armigo Platform - NVN API 231 Native Shader Stages
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/platform/metal_draw.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader_native/program.h>
#include "pipeline/phases.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace armigo::platform::metal {

// A task or mesh stage's launch facts from its control. With them the
// reflection names the stage "task" or "mesh", writes them as its
// "mesh_launch" record and no compute launch; a mesh stage is written as the
// vertex-like stage its outputs make it.
struct NativeMeshLaunch {
    bool task = false;
    std::uint32_t threads = 0, payload_bytes = 0;
    bool launched_by_task = false;
    std::uint8_t primitive_vertices = 0;
    std::uint32_t max_vertices = 0, max_primitives = 0;
};

// Launch facts translate() records for a task or mesh stage. Empty for every
// other stage, and when a task or mesh stage's control has no launch facts.
[[nodiscard]] std::optional<NativeMeshLaunch> native_mesh_launch(const shader_native::Program& program);

// The reflection a package stage holds (armigo-shader-stage-v3), written for
// an artifact of the native front end exactly as the ported translator writes
// it for its own. identity holds the program bytes the stage came from.
[[nodiscard]] std::string native_stage_reflection(const shader::pipeline::MslArtifact& artifact,
                                                  const shader::graphics_msl::EmitOptions& emission,
                                                  shader::pipeline::Stage stage,
                                                  std::span<const std::byte> identity,
                                                  const NativeMeshLaunch* launch = nullptr);

// Records a refused API 231 stage for offline study (ARMIGO_SHADER_MISS_DIR):
// DIR/231-<selector>-<SHA-256 of control and code>/{control.bin,code.bin},
// published whole by a rename. Returns the directory name, or empty after
// reporting why the record failed; bytes already recorded are not rewritten.
std::string record_refused_stage(const std::filesystem::path& directory, unsigned selector,
                                 std::span<const std::byte> control, std::span<const std::byte> code);

} // namespace armigo::platform::metal

namespace armigo::nvn {

// Resolves API 231 programs as the console driver takes them: SetShaders
// validates and records each native stage and succeeds for well-formed input
// (malformed input fails it). The native front end translates a stage when a
// draw or dispatch first uses it; a stage it refuses is reported once and the
// draws of programs holding it are refused. Call after set_program_package.
void set_native_program_resolver() noexcept;

// A mesh draw's mesh stage for its fragment stage. A mesh stage whose output
// maps lack generic components the fragment stage reads is translated again
// for it (once per set of reads), outputting exactly what it reads with the
// defaults of unwritten inputs; otherwise the mesh stage itself. Null, with
// why in error, when that translation is refused.
[[nodiscard]] std::shared_ptr<const platform::CompiledGraphicsStage> mesh_stage_for(
    const std::shared_ptr<const platform::CompiledGraphicsStage>& mesh, const platform::CompiledGraphicsStage& fragment,
    std::string& error);

struct NativePackageUse {
    std::size_t hits = 0;
    std::size_t misses = 0;
};

// The package directory from --shaders. The first call is kept. Before a
// recorded stage is translated, its stage identity is looked up in the
// package's native section; a hit uses the packaged library and a miss falls
// back to translation. One summary line is written when the process exits
// after a package was configured.
void set_native_shader_package(std::filesystem::path directory) noexcept;
[[nodiscard]] NativePackageUse native_shader_package_counts() noexcept;

} // namespace armigo::nvn
