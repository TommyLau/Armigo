// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Internal declaration of the guarded single-word decoder entry point shared
// by maxwell.cpp and the forward-program walk. Not part of the public shader
// interface.
#pragma once
#include <armigo/shader/maxwell.h>

#include <cstdint>
#include <variant>

namespace armigo::shader::maxwell {

// Decodes one instruction word with the production Maxwell decoders while
// admitting any 4-bit instruction predicate guard: the returned instruction
// preserves the raw guard verbatim as Predicate{index, inverted}, including
// PT (raw 7) and its inversion (raw 15). Everything else matches the strict
// decode() path: the same opcode families, field validation, modifier
// admission, and the Branch rejection. The guarded forward-program walk
// calls this only after decode_forward_branch reports not_branch, so a
// branch-shaped word still fails explicitly instead of decoding as an
// instruction.
[[nodiscard]] std::variant<Instruction, DecodeError> decode_guarded_instruction(
    std::uint64_t word, std::uint32_t pc);

}  // namespace armigo::shader::maxwell
