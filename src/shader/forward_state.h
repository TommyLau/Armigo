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
#include <cstddef>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace armigo::shader::graphics::detail {

using InitExpr = std::uint32_t;
using PredicateFacts = std::map<std::uint32_t, bool>;
inline constexpr InitExpr init_false = 0;
inline constexpr InitExpr init_true = 1;

template<class Id> struct ForwardCell {
    Id carrier{0};
    InitExpr validity = init_false;
};

struct ForwardState {
    std::array<ForwardCell<ScalarId>, 255> registers{};
    std::array<ForwardCell<ScalarId>, 160> stored{};
    std::array<ForwardCell<PredicateId>, 7> predicates{};
    PredicateFacts facts;
    InitExpr scope = init_true;
};

// Hash-consed boolean DAG used only for admission, never runtime execution.
// A join selects validity with its predecessor's reachability condition;
// unqualified OR of predecessor validities would admit undefined reads.
class InitExpressions {
public:
    static constexpr std::size_t max_expressions = 262144;
    static constexpr std::size_t max_proof_steps = 4096;
    static constexpr unsigned max_proof_depth = 128;

    InitExpressions();
    std::uint32_t pc = 0;
    InitExpr literal(PredicateId predicate, bool value);
    InitExpr conjunction(InitExpr left, InitExpr right);
    InitExpr disjunction(InitExpr left, InitExpr right);
    bool proves(InitExpr scope, InitExpr validity, const PredicateFacts& facts);

private:
    enum class Kind { false_, true_, literal, and_, or_ };
    struct Expression { Kind kind; InitExpr left; InitExpr right; };
    InitExpr intern(Kind kind, InitExpr left, InitExpr right);
    bool complements(InitExpr left, InitExpr right) const;
    InitExpr restrict(InitExpr value, const PredicateFacts& facts,
                      std::map<InitExpr, InitExpr>& memo,
                      std::size_t& steps, unsigned depth);
    bool implies(InitExpr left, InitExpr right,
                 std::map<std::pair<InitExpr, InitExpr>, bool>& memo,
                 std::size_t& steps, unsigned depth) const;
    std::vector<Expression> expressions_;
    std::map<std::tuple<Kind, InitExpr, InitExpr>, InitExpr> identities_;
};

}  // namespace armigo::shader::graphics::detail
