// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include "emitter_context.h"
#include "msl_context.h"

#include <array>

namespace armigo::shader::pipeline {
#define ARMIGO_EMITTER(name, function) Status function(EmitterContext&);
#include "emitter_registry.h"
#undef ARMIGO_EMITTER
#define ARMIGO_MSL(name, function) Status function(MslContext&, OperationId);
#include "msl_registry.h"
#undef ARMIGO_MSL

struct EmitterBinding { InstructionName instruction; InstructionEmitter emit; };
struct MslBinding { Opcode opcode; MslGenerator emit; };
inline constexpr std::size_t emitter_binding_count = 0
#define ARMIGO_EMITTER(name, function) + 1
#include "emitter_registry.h"
#undef ARMIGO_EMITTER
;
inline constexpr std::size_t msl_binding_count = 0
#define ARMIGO_MSL(name, function) + 1
#include "msl_registry.h"
#undef ARMIGO_MSL
;
inline constexpr std::array<EmitterBinding, emitter_binding_count> emitter_bindings{{
#define ARMIGO_EMITTER(name, function) {InstructionName::name, &function},
#include "emitter_registry.h"
#undef ARMIGO_EMITTER
}};
inline constexpr std::array<MslBinding, msl_binding_count> msl_bindings{{
#define ARMIGO_MSL(name, function) {Opcode::name, &function},
#include "msl_registry.h"
#undef ARMIGO_MSL
}};
template <std::size_t N> constexpr bool valid_bindings(const std::array<EmitterBinding, N>& entries) {
    std::array<bool, instruction_name_count> seen{};
    for (auto entry : entries) {
        const auto index = static_cast<std::size_t>(entry.instruction);
        if (index == 0 || index >= seen.size() || seen[index] || entry.emit == nullptr) return false;
        seen[index] = true;
    }
    return true;
}
template <std::size_t N> constexpr bool valid_bindings(const std::array<MslBinding, N>& entries) {
    std::array<bool, static_cast<std::size_t>(Opcode::count)> seen{};
    for (auto entry : entries) {
        const auto index = static_cast<std::size_t>(entry.opcode);
        if (index >= seen.size() || seen[index] || entry.emit == nullptr) return false;
        seen[index] = true;
    }
    return true;
}
static_assert(valid_bindings(emitter_bindings), "Duplicate or invalid instruction emitter registration");
static_assert(valid_bindings(msl_bindings), "Duplicate or invalid MSL generator registration");
inline constexpr auto instruction_emitters = [] {
    std::array<InstructionEmitter, instruction_name_count> table{};
    for (auto entry : emitter_bindings) table[static_cast<std::size_t>(entry.instruction)] = entry.emit;
    return table;
}();
inline constexpr auto msl_generators = [] {
    std::array<MslGenerator, static_cast<std::size_t>(Opcode::count)> table{};
    for (auto entry : msl_bindings) table[static_cast<std::size_t>(entry.opcode)] = entry.emit;
    return table;
}();
[[nodiscard]] Status dispatch_instruction(EmitterContext&);
[[nodiscard]] Status dispatch_msl(MslContext&, OperationId);
} // namespace armigo::shader::pipeline
