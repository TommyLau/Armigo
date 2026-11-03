// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../maxwell_families.h"

#include "../maxwell_decode_helpers.h"

namespace armigo::shader::maxwell::detail {

// The dependency-barrier unit: the opcode pattern admits the instruction and
// every operand bit is admitted because the visitor reads no fields. A word
// outside the pattern never reaches this
// decoder, so nothing that would decode as a different instruction is
// accepted here.
DecodeOutcome decode_dependency_barrier(std::uint64_t word, std::uint32_t pc,
                                        Predicate predicate) {
    return Instruction{pc, word, OpcodeFamily::DependencyBarrier, predicate,
                       DependencyBarrier{field(word, 0, 48)}};
}

}  // namespace armigo::shader::maxwell::detail
