// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "internal.h"

namespace armigo::shader::pipeline::passes {
Status legalize_nontexture(Program& program, FactReader& facts, const MetalProfile& profile) {
    if (auto status = validate(program, IrForm::ssa, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return status;
    if (profile.identity.empty() || profile.version == 0 || profile.buffer_slots == 0 ||
        profile.texture_slots == 0 || profile.sampler_slots == 0)
        return diagnose(Phase::legalize, DiagnosticCode::unsupported_capability, {}, "Nontexture legalization requires an explicit Metal profile");
    detail::Budget budget(program);
    try {
        for (const auto& op : program.operations) {
            budget.spend(1 + op.inputs.size() + op.outputs.size(), op.source);
            auto unsupported = [&](std::string message, DiagnosticCode code = DiagnosticCode::unsupported_capability) {
                return diagnose(Phase::legalize, code, op.source, std::move(message));
            };
            if (const auto* comparison = std::get_if<ComparisonInfo>(&op.info);
                comparison && comparison->relation > Compare::always)
                return unsupported("Comparison relation is outside the defined contract", DiagnosticCode::unsupported_modifier);
            auto width_supported = [&](ScalarType type) {
                return (type != ScalarType::f64 || profile.supports_fp64) && (type != ScalarType::f16 || profile.supports_fp16);
            };
            if (!width_supported(op.numeric.scalar)) return unsupported("Floating width is unavailable; precision substitution is forbidden");
            if (const auto* conversion = std::get_if<ConversionInfo>(&op.info)) {
                if (conversion->source > ScalarType::boolean || conversion->destination > ScalarType::boolean)
                    return unsupported("Conversion scalar is outside the defined contract", DiagnosticCode::unsupported_modifier);
                if (!width_supported(conversion->source) || !width_supported(conversion->destination))
                    return unsupported("Conversion width is unavailable; conversion boundaries must be preserved");
            }
            if (const auto* subgroup = std::get_if<SubgroupInfo>(&op.info)) {
                if (subgroup->shuffle > ShuffleKind::butterfly || subgroup->width != 32)
                    return unsupported("Subgroup operation requires the defined 32-lane contract", DiagnosticCode::unsupported_modifier);
                if (!profile.supports_simd_shuffle)
                    return unsupported("Subgroup operations require Metal SIMD capability");
                // Preserve both shuffle results and the guest mask. The numeric
                // generator must implement the checked lane/validity operation.
            }
            if ((op.opcode == Opcode::derivative_x || op.opcode == Opcode::derivative_y || op.opcode == Opcode::discard) &&
                program.stage != Stage::fragment)
                return unsupported("Derivative and discard operations require fragment stage");
            if ((op.opcode == Opcode::barrier || op.opcode == Opcode::memory_barrier || op.opcode == Opcode::atomic) &&
                program.stage != Stage::compute)
                return unsupported("Memory synchronization adapter requires compute stage");
            if (const auto* memory = std::get_if<MemoryInfo>(&op.info); memory && memory->space == AddressSpace::shared && program.stage != Stage::compute)
                return unsupported("Shared memory requires compute stage");
            if ((op.opcode == Opcode::divide || op.opcode == Opcode::remainder) && op.numeric.domain != Domain::floating) {
                const auto* divisor = std::get_if<LiteralBits>(&op.inputs[1]);
                const auto width = scalar_bits(op.numeric.scalar);
                const auto mask = width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1;
                if (divisor && (divisor->bits & mask) == 0)
                    return unsupported("Integer division by zero has no defined legalization", DiagnosticCode::unsupported_modifier);
                const auto* dividend = std::get_if<LiteralBits>(&op.inputs[0]);
                if (op.numeric.domain == Domain::signed_integer && dividend && divisor &&
                    (dividend->bits & mask) == (std::uint64_t{1} << (width - 1)) && (divisor->bits & mask) == mask)
                    return unsupported("Signed division overflow has no defined legalization", DiagnosticCode::unsupported_modifier);
            }
            // IO, byte-addressed memory and scalar selects are already explicit.
            // Do not infer SDK buffer layouts, cancel perspective W, force host
            // precision policy, or introduce compute-only CAS helper functions.
        }
    } catch (const Diagnostic& error) { return error; }
    auto dependencies = merge_dependencies(program.dependencies, facts.dependencies(), program.limits, Phase::legalize);
    if (const auto* error = std::get_if<Diagnostic>(&dependencies)) return *error;
    program.dependencies = std::get<std::vector<Dependency>>(std::move(dependencies));
    return std::monostate{};
}
} // namespace armigo::shader::pipeline::passes
