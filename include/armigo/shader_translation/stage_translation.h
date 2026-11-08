// ============================================================================
// Armigo Shader - Stage Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/shader/graphics_msl.h>
#include <armigo/shader_input/shader_input.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace armigo::shader::translation {

// The options of one stage's translation: the offline translator's flags.
struct StageOptions {
    // Vertex attribute formats, the vertex depth convention and the fragment
    // position convention.
    graphics_msl::EmitOptions emission;
    // The constant buffer that holds bound texture handles (3D register 0x982).
    std::optional<std::uint8_t> texture_handle_cbuf;
    // Answer the texture-descriptor facts the program's texture encodings prove.
    bool texture_descriptors_derived = false;
    // Compute launch specialization: local size and static shared memory.
    std::optional<std::array<std::uint32_t, 3>> compute_local_size;
    std::optional<std::uint32_t> compute_shared_memory_bytes;
    // Also hoist implicit-LOD and bias samples out of their predicate guards,
    // as a title opts in to (the project documentation, texture metadata).
    bool hoist_implicit_samples = false;
};

// A translated stage: the files a package stage holds besides its Metal library.
struct TranslatedStage {
    shader_input::Stage stage = shader_input::Stage::Vertex;
    std::string source;              // source.metal
    std::string reflection;          // reflection.json
    std::vector<std::byte> program;  // program.bin: the stage's identity bytes
};

// Translates one stage, given its shader control and code, with the ported
// pipeline. Throws std::runtime_error naming the failing phase, the text the
// offline translator reports.
[[nodiscard]] TranslatedStage translate_stage(std::span<const std::byte> control,
                                              std::span<const std::byte> code,
                                              const StageOptions& options);

} // namespace armigo::shader::translation
