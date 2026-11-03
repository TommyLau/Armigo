// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Conservative structurally reachable forward-program decode for the bounded
// Maxwell frontend. It reuses the production instruction decoders behind the
// strict decode() entry point, admits guarded instructions by preserving all
// four predicate-guard bits, and admits direct forward branches through the
// validated decode_forward_branch helper. The result is a frontend program
// view for later flow lowering; it does not claim dynamic reachability or
// guest control-flow execution.
#pragma once
#include <armigo/shader/maxwell.h>
#include <armigo/shader/maxwell_flow.h>

#include <armigo/shader_input/shader_input.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace armigo::shader::maxwell {

// One structurally reachable node: a decoded production instruction with its
// preserved predicate guard (including a statically resolved SSY marker),
// or a validated direct forward branch, including resolved SYNC.
using ForwardInstruction = std::variant<Instruction, ForwardBranch>;

struct ForwardProgram {
    shader_input::Stage stage = shader_input::Stage::Vertex;
    std::array<std::byte, shader_input::format::sph_size> sph{};
    // Size of the instruction program region in bytes; every node pc is an
    // offset into that region.
    std::uint32_t program_size = 0;
    // Reachable nodes sorted by pc (source_pc for branches), each unique.
    std::vector<ForwardInstruction> instructions;
};

using ForwardDecodeResult = std::variant<ForwardProgram, DecodeError>;

// Parses the supplied shader containers and decodes every instruction slot
// that is structurally reachable from the entry slot at pc 8 through
// admitted direct forward branches and fallthrough, stopping each path at
// the first unconditional PT EXIT. Instruction guards are preserved, never
// forced to PT. Every unsupported opcode or modifier, unsupported flow test,
// malformed branch target, reachable backward edge, reachable program end
// without a PT EXIT, and limit overflow is a structured DecodeError; an
// unknown word never silently becomes a NOP. No returned object borrows
// either input span. The existing strict decode() is unchanged and remains
// PT-only. SSY pushes an ordered reconvergence token; SYNC pops the nearest
// SSY and resolves to ForwardBranch. Incoming complete stacks must agree at
// every join. Missing tokens and incompatible stacks are UnsupportedControlFlow
// with ReconvergenceStack; malformed targets use ReconvergenceTarget.
// PBK/BRK, other addressing forms, and non-always SYNC flow tests are rejected.
[[nodiscard]] ForwardDecodeResult decode_forward_program(
    shader_input::ByteSpan control, shader_input::ByteSpan code,
    DecodeLimits limits = {});

}  // namespace armigo::shader::maxwell
