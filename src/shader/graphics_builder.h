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

#include <armigo/shader/graphics.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace armigo::shader::graphics {

struct GraphSpec;

// Structured failure thrown by every Builder primitive. Direct Builder
// drivers observe the same Error code, pc, and detail that the legacy
// lower() path reports; nothing flattens or replaces it.
struct BuilderFailure {
    Error error;
};

// Declaration of the legacy graphics Builder, shared so the control-flow
// lowering reuses the same checked primitives, numeric instruction lowering,
// and the structured BuilderFailure declared above. Every member definition
// and the header-decoding helpers stay private to graphics.cpp.
struct Builder {
    Program program;
    Limits limits;
    std::array<std::optional<ScalarId>, 255> registers{};
    std::array<std::optional<ScalarId>, 160> stored{};
    // P0..P6; PT is not stored. Forward lowering snapshots this file per
    // block exactly as it snapshots registers.
    std::array<std::optional<PredicateId>, 7> predicates{};
    ScalarId zero{0};
    std::uint32_t pc = 0;

    explicit Builder(Limits requested);

    std::uint32_t append(Operation operation);
    ScalarId scalar(Operation operation);
    // Appends a registered pure operation after checking its kind,
    // immediate, operand types and operand order against the registry and
    // the expected result type; any mismatch is an invalid instruction.
    std::uint32_t pure(const Pure& operation, ValueType expected);
    ScalarId read(maxwell::RegisterOperand reg);
    void write(maxwell::RegisterOperand reg, ScalarId value);
    bool declared(unsigned component, bool output) const;
    ScalarId attribute(unsigned component);
    ScalarId constant(maxwell::ConstantBufferOperand operand);
    ScalarId source(const maxwell::FloatSource& operand, bool expect_constant);
    template<class T> const T& payload(const maxwell::Instruction& instruction);
    void validate_header();
    void validate_graph_interface(const GraphSpec& spec);
    void exports();
    // Fresh-Builder only: validates stage and SPH through the existing header
    // rules and installs the zero literal. Reuse of a Builder that already
    // holds nodes, exports, or blocks fails as an invalid instruction.
    void begin_graph(shader_input::Stage stage, const std::array<std::byte, 80>& sph);
    // Collects exports from the current register/stored snapshot and moves
    // them out, clearing the program accumulator so a later collection only
    // reports that snapshot's outputs. Nodes and builder state are retained.
    std::vector<Export> collect_exports();
    bool instruction(const maxwell::Instruction& instruction);
    Program run(const maxwell::DecodedProgram& decoded);
};

}  // namespace armigo::shader::graphics
