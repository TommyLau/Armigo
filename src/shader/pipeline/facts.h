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
#include "diagnostic.h"

#include <algorithm>
#include <span>
#include <utility>

namespace armigo::shader::pipeline {
// One immutable source per stage/translation. No permissive virtual defaults.
class FactSource {
public:
    virtual ~FactSource() = default;
    [[nodiscard]] virtual Knowledge<FactValue> lookup(FactKey key) const = 0;
};

// Phase handoffs must carry the canonical union, even with a fresh reader.
[[nodiscard]] Result<std::vector<Dependency>> merge_dependencies(
    std::span<const Dependency>, std::span<const Dependency>, Limits, Phase);

class FactReader {
public:
    explicit FactReader(const FactSource& source, Limits limits = {}) : source_(source), limits_(limits) {}
    FactReader(const FactReader&) = delete;
    FactReader& operator=(const FactReader&) = delete;

    // Consuming a known fact records it. Unknown remains unknown, with no dependency value.
    template <FactKind K>
    [[nodiscard]] Result<Knowledge<typename FactTraits<K>::value_type>> read(
        FactQuery<K> query, Phase phase, SourceLocation location = {}) {
        using T = typename FactTraits<K>::value_type;
        auto result = consume(query.key(), phase, location);
        if (auto* error = std::get_if<Diagnostic>(&result)) return *error;
        const auto& state = std::get<Knowledge<FactValue>>(result);
        if (auto* unknown = std::get_if<Unknown>(&state)) return Knowledge<T>{*unknown};
        const auto& known = std::get<Known<FactValue>>(state);
        if (const auto* value = std::get_if<T>(&known.value)) {
            return Knowledge<T>{Known<T>{*value, known.origin}};
        }
        auto error = diagnose(phase, DiagnosticCode::invalid_fact, location, "Fact value has the wrong type");
        error.missing_fact = query.key();
        return error;
    }
    template <FactKind K>
    [[nodiscard]] Result<typename FactTraits<K>::value_type> require(
        FactQuery<K> query, Phase phase, SourceLocation location = {}) {
        using T = typename FactTraits<K>::value_type;
        auto result = read(query, phase, location);
        if (auto* error = std::get_if<Diagnostic>(&result)) return *error;
        const auto& state = std::get<Knowledge<T>>(result);
        if (auto* known = std::get_if<Known<T>>(&state)) return known->value;
        auto error = diagnose(phase, DiagnosticCode::missing_fact, location, std::get<Unknown>(state).reason);
        error.missing_fact = query.key();
        return error;
    }
    [[nodiscard]] std::span<const Dependency> dependencies() const { return dependencies_; }
private:
    [[nodiscard]] Result<Knowledge<FactValue>> consume(FactKey key, Phase phase, SourceLocation location);
    const FactSource& source_;
    Limits limits_;
    std::vector<Dependency> dependencies_;
};
} // namespace armigo::shader::pipeline
