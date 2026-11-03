// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo derives each vertex attribute class from the IR because an offline
// package is translated before the guest vertex layout exists: Maxwell
// attribute registers carry float bits for float/unorm/snorm guest formats and
// raw extended integer bits otherwise, so a code use analysis decides whether
// the declaration is float4, uint4 or int4.

#include "detail.h"

#include <map>
#include <unordered_set>

namespace armigo::shader::pipeline {
namespace {

// The interpretation a classified load carries, mirroring the emitter's
// interpretation() helper without borrowing an emitter-lane header.
[[nodiscard]] NumericMode class_interpretation(ScalarType scalar) {
    const auto domain = scalar_domain(scalar);
    return {domain, scalar,
        domain == Domain::floating ? Rounding::nearest_even : Rounding::not_applicable,
        false, false, false, false};
}

// Evidence accumulated over every use of one generic input location.
struct Evidence {
    bool floating{};
    bool signed_integer{};
    bool other_integer{};
};

class Classifier {
public:
    explicit Classifier(Program& program) : p(program), users(p.values.size()) {
        for (std::uint32_t i = 0; i < p.operations.size(); ++i) {
            for (const auto& input : p.operations[i].inputs)
                if (const auto* value = std::get_if<ValueId>(&input))
                    if (value->value < users.size()) users[value->value].operations.push_back(i);
        }
        for (const auto& block : p.blocks) {
            for (const auto& phi : block.phis)
                for (const auto& input : phi.inputs)
                    if (input.value.value < users.size()) users[input.value.value].phis.push_back(&phi);
            // A switch selector consumes the raw selector bits.
            if (const auto* branch = std::get_if<Switch>(&block.terminator))
                if (const auto* value = std::get_if<ValueId>(&branch->selector); value && value->value < users.size())
                    users[value->value].selector = true;
        }
    }

    // Classifies every directly addressed generic vertex input read in this
    // program, rewriting each load to its class interpretation. Fails with a
    // structured diagnostic when a location mixes interpretations or the work
    // budget is exhausted.
    Status run() {
        for (auto& op : p.operations) {
            if (op.opcode != Opcode::load_io) continue;
            const auto* info = std::get_if<IoInfo>(&op.info);
            if (info == nullptr || info->semantic != IoSemantic::generic ||
                info->dynamic_byte_offset || op.outputs.size() != 1 ||
                !std::holds_alternative<ValueId>(op.outputs.front())) continue;
            const auto value = std::get<ValueId>(op.outputs.front());
            if (!value.valid() || value.value >= p.values.size()) continue;
            auto& state = locations[info->location];
            state.sources.push_back(value);
            state.loads.push_back(&op);
        }
        for (auto& [location, state] : locations) {
            Evidence evidence;
            for (const auto source : state.sources)
                if (!trace(source, evidence))
                    return diagnose(Phase::legalize, DiagnosticCode::limit_exceeded, source_of(state),
                        "Vertex input class analysis exceeded its work budget");
            const bool integer = evidence.signed_integer || evidence.other_integer;
            if (evidence.floating && integer)
                return diagnose(Phase::legalize, DiagnosticCode::unsupported_capability, source_of(state),
                    "Generic vertex input " + std::to_string(location) +
                    " mixes floating and integer interpretations of its attribute register");
            // Sint is proven only when signedness is the whole integer story;
            // mixed or unsigned uses keep the raw uint word. Without integer
            // evidence the declaration stays the float default: the host
            // vertex descriptor converts float/unorm/snorm guest formats, so
            // an unproven input never claims raw integer bits.
            state.cls = !evidence.floating && integer ?
                (evidence.signed_integer && !evidence.other_integer ? VertexAttributeClass::sint_class :
                                                                     VertexAttributeClass::uint_class) :
                VertexAttributeClass::float_class;
            const auto scalar = vertex_class_scalar(*state.cls);
            for (auto* load : state.loads) {
                std::get<IoInfo>(load->info).type.scalar = scalar;
                load->numeric = class_interpretation(scalar);
            }
        }
        return std::monostate{};
    }

private:
    struct Uses {
        std::vector<std::uint32_t> operations;
        std::vector<const Phi*> phis;
        bool selector{};
    };
    struct LocationState {
        std::vector<ValueId> sources;
        std::vector<Operation*> loads;
        std::optional<VertexAttributeClass> cls;
    };

    [[nodiscard]] SourceLocation source_of(const LocationState& state) const {
        return state.loads.empty() ? SourceLocation{} : state.loads.front()->source;
    }

    // Follows one SSA value to its users, recording interpretation evidence.
    bool trace(ValueId value, Evidence& evidence) {
        if (!visited.insert(value.value).second) return true;
        if (++steps > p.limits.analysis_steps) return false;
        for (const auto index : users[value.value].operations)
            if (!consume(p.operations[index], evidence)) return false;
        for (const auto* phi : users[value.value].phis)
            if (!trace(phi->output, evidence)) return false;
        if (users[value.value].selector) evidence.other_integer = true;
        return true;
    }

    // One consuming operation: records its interpretation of the read word and
    // continues through bit-preserving operations.
    bool consume(Operation& op, Evidence& evidence) {
        if (++steps > p.limits.analysis_steps) return false;
        // Copies, bitcasts and raw-word selects preserve the register word.
        const bool transparent = op.opcode == Opcode::copy || op.opcode == Opcode::bitcast ||
            (op.opcode == Opcode::select && op.numeric.domain == Domain::bits);
        if (transparent) {
            for (const auto& output : op.outputs)
                if (const auto* value = std::get_if<ValueId>(&output))
                    if (!trace(*value, evidence)) return false;
            return true;
        }
        if (const auto* conversion = std::get_if<ConversionInfo>(&op.info)) {
            // The attribute feeds the conversion source; its result no longer
            // carries the register word.
            switch (scalar_domain(conversion->source)) {
            case Domain::floating: evidence.floating = true; break;
            case Domain::signed_integer: evidence.signed_integer = true; break;
            default: evidence.other_integer = true; break;
            }
            return true;
        }
        switch (op.numeric.domain) {
        case Domain::floating: evidence.floating = true; break;
        case Domain::signed_integer: evidence.signed_integer = true; break;
        case Domain::unsigned_integer: case Domain::bits: evidence.other_integer = true; break;
        case Domain::boolean: break;
        }
        return true;
    }

    Program& p;
    std::vector<Uses> users;
    std::map<std::uint16_t, LocationState> locations;
    std::unordered_set<std::uint32_t> visited;
    std::uint64_t steps{};
};

} // namespace

namespace resources {

// Runs after SSA construction and cleanup passes, before resource resolution.
Status classify_vertex_inputs(Program& program) {
    if (program.stage != Stage::vertex) return std::monostate{};
    return Classifier{program}.run();
}
} // namespace resources
} // namespace armigo::shader::pipeline
