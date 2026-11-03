// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../graphics_families.h"
namespace armigo::shader::graphics::detail {
void lower_word_and_immediate(const maxwell::Instruction& instruction,FamilyLower& lower) {
    const auto* value=std::get_if<maxwell::WordAndImmediate>(&instruction.payload);
    if(!value)lower.fail(Error::Code::InvalidInstruction);
    const auto source=lower.read(value->source);
    lower.write(value->destination,lower.word_and(source,value->mask));
}
}  // namespace armigo::shader::graphics::detail
