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
#include "passes.h"

namespace armigo::shader::pipeline::passes::detail {
class Budget {
public:
    explicit Budget(const Program& p) : remaining_(p.limits.analysis_steps) {}
    void spend(std::uint64_t count = 1, SourceLocation source = {}) {
        if (count > remaining_) throw diagnose(Phase::legalize, DiagnosticCode::limit_exceeded,
            source, "Shader pass analysis work limit exceeded");
        remaining_ -= count;
    }
private:
    std::uint64_t remaining_;
};
[[nodiscard]] bool neutral(const Operation&);
[[nodiscard]] std::optional<Operand> simplify(const Operation&, ValueType);
} // namespace armigo::shader::pipeline::passes::detail
