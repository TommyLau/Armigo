// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/opcode_inventory.h>

namespace armigo::shader_native {
const ProgramAudit& OpcodeInventory::inspect(const Program& program) {
    if (program.instructions.size() % 16) throw InputError("Native inventory instruction span is misaligned");
    auto [entry, inserted] = programs_.try_emplace(std::pair{program.stage, program.image_digest});
    if (!inserted) return entry->second;
    auto& audit = entry->second;
    for (std::size_t pc = 0; pc < program.instructions.size(); pc += 16) {
        const auto instruction = Instruction::read(program.instructions.subspan(pc, 16));
        auto& count = counts_[instruction.bits(0, 12)];
        ++count.slots; ++audit.slots;
        auto result = decode_instruction(instruction, pc, program.instructions.size());
        if (auto failure = std::get_if<DecodeFailure>(&result)) {
            ++count.refused; ++audit.refused; ++failures_[static_cast<unsigned>(failure->code)];
            if (!audit.first_failure) audit.first_failure = *failure;
        } else { ++count.decoded; ++audit.decoded; }
    }
    return audit;
}
} // namespace armigo::shader_native
