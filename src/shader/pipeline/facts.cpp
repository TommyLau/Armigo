// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "facts.h"

namespace armigo::shader::pipeline {
namespace {
bool value_matches(FactKind kind, const FactValue& value) {
    switch (kind) {
#define ARMIGO_MATCH(name) case FactKind::name: return std::holds_alternative<FactTraits<FactKind::name>::value_type>(value)
    ARMIGO_MATCH(local_memory_bytes);
    ARMIGO_MATCH(compute_local_size);
    ARMIGO_MATCH(compute_shared_memory_bytes);
    // Deprecated fact identities remain type-valid for legacy serialized users.
    ARMIGO_MATCH(constant_buffer_bound_mask);
    ARMIGO_MATCH(constant_buffer_size);
    ARMIGO_MATCH(constant_buffer_word);
    ARMIGO_MATCH(texture_handle_cbuf);
    ARMIGO_MATCH(texture_descriptor);
    ARMIGO_MATCH(texture_array_length);
    ARMIGO_MATCH(vertex_attribute_format);
    ARMIGO_MATCH(fragment_output_type);
    ARMIGO_MATCH(input_interpolation);
    ARMIGO_MATCH(topology);
    ARMIGO_MATCH(depth_convention);
    ARMIGO_MATCH(origin_convention);
    ARMIGO_MATCH(negate_y);
    ARMIGO_MATCH(alpha_test);
    ARMIGO_MATCH(early_z);
    ARMIGO_MATCH(alpha_to_coverage);
    ARMIGO_MATCH(dual_source_blend);
    ARMIGO_MATCH(draw_parameters);
#undef ARMIGO_MATCH
    case FactKind::count: return false;
    }
    return false;
}
}
Result<std::vector<Dependency>> merge_dependencies(std::span<const Dependency> first,
    std::span<const Dependency> second, Limits limits, Phase phase) {
    std::vector<Dependency> result;
    auto insert = [&](const Dependency& item) -> Status {
        auto fail = [&](DiagnosticCode code, std::string message) -> Status {
            auto error = diagnose(phase, code, {}, std::move(message));
            error.missing_fact = item.key;
            return error;
        };
        if (!valid_fact_key(item.key) || !value_matches(item.key.kind, item.value) || item.origin.identity.empty())
            return fail(DiagnosticCode::invalid_fact, "Invalid dependency at phase handoff");
        auto position = std::lower_bound(result.begin(), result.end(), item.key,
            [](const Dependency& dependency, FactKey key) { return dependency.key < key; });
        if (position != result.end() && position->key == item.key) {
            if (*position != item) return fail(DiagnosticCode::conflicting_fact, "Conflicting dependency at phase handoff");
            return std::monostate{};
        }
        if (result.size() >= limits.dependencies) return fail(DiagnosticCode::limit_exceeded, "Dependency union exceeds limit");
        result.insert(position, item);
        return std::monostate{};
    };
    for (const auto span : {first, second}) for (const auto& dependency : span) {
        auto status = insert(dependency);
        if (auto* error = std::get_if<Diagnostic>(&status)) return *error;
    }
    return result;
}
Result<Knowledge<FactValue>> FactReader::consume(FactKey key, Phase phase, SourceLocation location) {
    auto fail = [&](DiagnosticCode code, std::string message) -> Result<Knowledge<FactValue>> {
        auto error = diagnose(phase, code, location, std::move(message));
        error.missing_fact = key;
        return error;
    };
    if (!valid_fact_key(key)) return fail(DiagnosticCode::invalid_fact, "Invalid fact key or byte address");
    auto state = source_.lookup(key);
    const auto* known = std::get_if<Known<FactValue>>(&state);
    const auto position = std::lower_bound(dependencies_.begin(), dependencies_.end(), key,
        [](const Dependency& dependency, FactKey query) { return dependency.key < query; });
    if (position != dependencies_.end() && position->key == key) {
        if (!known || position->value != known->value || position->origin != known->origin) {
            return fail(DiagnosticCode::conflicting_fact, "Fact source changed during translation");
        }
        return state;
    }
    if (!known) return state;
    if (!value_matches(key.kind, known->value) || known->origin.identity.empty()) {
        return fail(DiagnosticCode::invalid_fact, "Fact type or evidence identity is invalid");
    }
    if (dependencies_.size() >= limits_.dependencies) {
        return fail(DiagnosticCode::limit_exceeded, "Fact dependency limit exceeded");
    }
    dependencies_.insert(position, {key, known->value, known->origin});
    return state;
}
} // namespace armigo::shader::pipeline
