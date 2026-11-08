// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once
#include "phases.h"
#include <string_view>

namespace armigo::shader::pipeline {
struct MatrixPlan;
// The stage emitter walks structured control flow. Per-op generators emit all
// results and effects, returning failure for unsupported numerical/texture modes.
class MslContext {
public:
    virtual ~MslContext() = default;
    [[nodiscard]] virtual const StructuredProgram& program() const = 0;
    [[nodiscard]] virtual const ResourcePlan& resources() const = 0;
    [[nodiscard]] virtual bool fragment_output_types_from_render_targets() const { return false; }
    [[nodiscard]] virtual std::uint32_t guest_memory_tag() const { return 0; }
    // Where warp matrix operations keep their operands; null outside a
    // compute stage with matrix multiplies.
    [[nodiscard]] virtual const MatrixPlan* matrix_plan() const { return nullptr; }
    [[nodiscard]] virtual const Operation& operation(OperationId) const = 0;
    [[nodiscard]] virtual Result<std::string> operand(const Operand&, ScalarType) = 0;
    [[nodiscard]] virtual Result<std::string> resource(ResourceId) = 0;
    [[nodiscard]] virtual Status assign(ValueId, std::string_view expression, ScalarType) = 0;
    [[nodiscard]] virtual Status statement(std::string_view) = 0;
    [[nodiscard]] virtual Status require_helper(std::string_view name) = 0;
    [[nodiscard]] virtual Diagnostic error(OperationId, DiagnosticCode,
                                           std::string message, std::string modifier = {}) const = 0;
};
using MslGenerator = Status (*)(MslContext&, OperationId);
static_assert(std::is_abstract_v<MslContext>);
} // namespace armigo::shader::pipeline
