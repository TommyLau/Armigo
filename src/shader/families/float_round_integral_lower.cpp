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

#include <variant>

namespace armigo::shader::graphics::detail {

// F2F F32-to-F32 lowers to the registered FloatRoundIntegral operation on
// the raw source word; the flush, the sign handling and the NaN policy are
// the operation's own. The saturation modifier applies to the rounded value
// through the checked saturation node after rounding. The source is read
// before the destination is written, so aliasing
// is safe.
void lower_float_round_integral(const maxwell::Instruction& instruction,
                                FamilyLower& lower) {
    const auto* value = std::get_if<maxwell::FloatRoundIntegral>(&instruction.payload);
    if (value == nullptr) {
        lower.fail(Error::Code::InvalidInstruction);
    }
    auto source = value->source;
    ScalarId input{0};
    if (const auto* reg = std::get_if<maxwell::RegisterOperand>(&source)) {
        input = lower.read(*reg);
    } else {
        input = lower.constant(std::get<maxwell::ConstantBufferOperand>(source));
    }
    const auto rounded = lower.operation(PureKind::FloatRoundIntegral, {input},
                                         static_cast<std::uint32_t>(value->rounding));
    lower.write(value->destination,
                value->saturate ? lower.saturate(rounded) : rounded);
}

}  // namespace armigo::shader::graphics::detail
