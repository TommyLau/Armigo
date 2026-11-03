// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "forward_state.h"
#include "graphics_builder.h"

#include <algorithm>

namespace armigo::shader::graphics::detail {

InitExpressions::InitExpressions()
    : expressions_{{Kind::false_, 0, 0}, {Kind::true_, 0, 0}} {}

InitExpr InitExpressions::intern(Kind kind, InitExpr left, InitExpr right) {
    const auto key = std::tuple{kind, left, right};
    if (const auto found = identities_.find(key); found != identities_.end()) {
        return found->second;
    }
    if (expressions_.size() >= max_expressions) {
        throw BuilderFailure{{Error::Code::Limit, pc,
                              static_cast<std::uint32_t>(max_expressions)}};
    }
    const auto id = static_cast<InitExpr>(expressions_.size());
    expressions_.push_back({kind, left, right});
    identities_.emplace(key, id);
    return id;
}

InitExpr InitExpressions::literal(PredicateId predicate, bool value) {
    return intern(Kind::literal, predicate.index, value ? 1 : 0);
}

bool InitExpressions::complements(InitExpr left, InitExpr right) const {
    const auto a = expressions_[left], b = expressions_[right];
    return a.kind == Kind::literal && b.kind == Kind::literal &&
           a.left == b.left && a.right != b.right;
}

InitExpr InitExpressions::conjunction(InitExpr left, InitExpr right) {
    if (left == init_false || right == init_false || complements(left, right)) {
        return init_false;
    }
    if (left == init_true || left == right) return right;
    if (right == init_true) return left;
    const auto a = expressions_[left], b = expressions_[right];
    if ((a.kind == Kind::and_ && (a.left == right || a.right == right)) ||
        (a.kind == Kind::or_ && (a.left == right || a.right == right))) {
        return a.kind == Kind::and_ ? left : right;
    }
    if ((b.kind == Kind::and_ && (b.left == left || b.right == left)) ||
        (b.kind == Kind::or_ && (b.left == left || b.right == left))) {
        return b.kind == Kind::and_ ? right : left;
    }
    if (left > right) std::swap(left, right);
    return intern(Kind::and_, left, right);
}

InitExpr InitExpressions::disjunction(InitExpr left, InitExpr right) {
    if (left == init_true || right == init_true || complements(left, right)) {
        return init_true;
    }
    if (left == init_false || left == right) return right;
    if (right == init_false) return left;
    const auto a = expressions_[left], b = expressions_[right];
    if ((a.kind == Kind::and_ || a.kind == Kind::or_) &&
        (a.left == right || a.right == right)) {
        return a.kind == Kind::and_ ? right : left;
    }
    if ((b.kind == Kind::and_ || b.kind == Kind::or_) &&
        (b.left == left || b.right == left)) {
        return b.kind == Kind::and_ ? left : right;
    }
    // Reconverging a diamond restores the enclosing scope without expanding
    // products or enumerating truth assignments.
    if (a.kind == Kind::and_ && b.kind == Kind::and_) {
        for (unsigned i = 0; i < 2; ++i) {
            for (unsigned j = 0; j < 2; ++j) {
                const auto common_a = i == 0 ? a.left : a.right;
                const auto common_b = j == 0 ? b.left : b.right;
                const auto other_a = i == 0 ? a.right : a.left;
                const auto other_b = j == 0 ? b.right : b.left;
                if (common_a == common_b && complements(other_a, other_b)) {
                    return common_a;
                }
            }
        }
    }
    if (left > right) std::swap(left, right);
    return intern(Kind::or_, left, right);
}

InitExpr InitExpressions::restrict(InitExpr value, const PredicateFacts& facts,
                                   std::map<InitExpr, InitExpr>& memo,
                                   std::size_t& steps, unsigned depth) {
    if (value <= init_true) return value;
    if (const auto found = memo.find(value); found != memo.end()) return found->second;
    if (++steps > max_proof_steps || depth >= max_proof_depth) return value;
    const auto expression = expressions_[value];
    InitExpr result = value;
    if (expression.kind == Kind::literal) {
        if (const auto fact = facts.find(expression.left); fact != facts.end()) {
            result = fact->second == (expression.right != 0) ? init_true : init_false;
        }
    } else {
        const auto left = restrict(expression.left, facts, memo, steps, depth + 1);
        const auto right = restrict(expression.right, facts, memo, steps, depth + 1);
        result = expression.kind == Kind::and_ ? conjunction(left, right)
                                              : disjunction(left, right);
    }
    memo.emplace(value, result);
    return result;
}

bool InitExpressions::implies(
    InitExpr left, InitExpr right,
    std::map<std::pair<InitExpr, InitExpr>, bool>& memo,
    std::size_t& steps, unsigned depth) const {
    if (left == init_false || right == init_true || left == right) return true;
    if (const auto found = memo.find({left, right}); found != memo.end()) {
        return found->second;
    }
    if (++steps > max_proof_steps || depth >= max_proof_depth) return false;
    const auto a = expressions_[left], b = expressions_[right];
    const auto prove = [&](InitExpr x, InitExpr y) {
        return implies(x, y, memo, steps, depth + 1);
    };
    bool result = false;
    if (b.kind == Kind::and_) result = prove(left, b.left) && prove(left, b.right);
    if (!result && a.kind == Kind::or_) result = prove(a.left, right) && prove(a.right, right);
    if (!result && a.kind == Kind::and_) result = prove(a.left, right) || prove(a.right, right);
    if (!result && b.kind == Kind::or_) result = prove(left, b.left) || prove(left, b.right);
    memo.emplace(std::pair{left, right}, result);
    return result;
}

bool InitExpressions::proves(InitExpr scope, InitExpr validity,
                             const PredicateFacts& facts) {
    if (validity == init_true || scope == init_false || scope == validity) return true;
    if (validity == init_false) return false;
    std::size_t steps = 0;
    std::map<InitExpr, InitExpr> restricted;
    const auto left = restrict(scope, facts, restricted, steps, 0);
    const auto right = restrict(validity, facts, restricted, steps, 0);
    std::map<std::pair<InitExpr, InitExpr>, bool> implications;
    return implies(left, right, implications, steps, 0);
}

}  // namespace armigo::shader::graphics::detail
