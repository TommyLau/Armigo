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
#include <armigo/shader_native/instruction.h>
#include <armigo/shader_native/program.h>
#include <map>

namespace armigo::shader_native {
struct ProgramAudit {
    std::uint64_t slots{}, decoded{}, refused{};
    std::optional<DecodeFailure> first_failure;
};
struct OpcodeCount { std::uint64_t slots{}, decoded{}, refused{}; };
class OpcodeInventory {
public:
    // Repeated metadata variants share a decode census only when stage and
    // exact image agree. Every allocated slot is counted, reachable or not.
    const ProgramAudit& inspect(const Program& program);
    const std::array<OpcodeCount, 4096>& counts() const { return counts_; }
    const std::array<std::uint64_t, 5>& failures() const { return failures_; }
    std::size_t programs() const { return programs_.size(); }
private:
    std::map<std::pair<Stage, std::string>, ProgramAudit> programs_;
    std::array<OpcodeCount, 4096> counts_{};
    std::array<std::uint64_t, 5> failures_{};
};
} // namespace armigo::shader_native
