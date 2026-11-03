// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pipeline/phases.h"
#include "zero_reads.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;
int failures{};

void check(bool condition, std::string_view message) {
    if (!condition) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}

template<class T> T take(Result<T> result) {
    if (const auto* failure = std::get_if<Diagnostic>(&result)) {
        throw std::runtime_error(failure->message);
    }
    return std::get<T>(std::move(result));
}

struct Facts final : FactSource {
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::depth_convention) {
            return Known<FactValue>{DepthConvention::zero_to_one,
                {OriginKind::explicit_option, "synthetic-complementary-guards"}};
        }
        return Unknown{"No other facts in the complementary-guard fixture"};
    }
};

MetalProfile profile() {
    return {"synthetic-metal4", 1, 31, 32, 16, true, false, true, false, false, true};
}

constexpr std::uint64_t pattern(std::string_view bits) {
    std::uint64_t result{};
    for (std::size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] == '1') result |= std::uint64_t{1} << (63 - i);
    }
    return result;
}

std::uint64_t guarded(std::uint64_t word, unsigned predicate, bool inverted = false) {
    return (word & ~(15ull << 16)) | (std::uint64_t{predicate} << 16) |
           (std::uint64_t{inverted} << 19);
}

std::uint64_t mov(unsigned reg, unsigned value, bool inverted) {
    return guarded(pattern("000000010000") | reg | (std::uint64_t{value} << 20), 0, inverted);
}

// ISETP.EQ/NE.AND P0, PT, RZ, RZ, PT. Both truth values are exercised.
std::uint64_t set_predicate(bool truth, unsigned destination = 0) {
    return pattern("010110110110") | (std::uint64_t{destination} << 3) | 7ull |
           (255ull << 8) | (7ull << 16) | (255ull << 20) | (7ull << 39) |
           (std::uint64_t{truth ? 2u : 5u} << 49);
}

std::uint64_t store(unsigned reg, unsigned offset) {
    return pattern("1110111111110x") | reg | (255ull << 8) | (7ull << 16) |
           (std::uint64_t{offset} << 20);
}

StageImage image(std::vector<std::uint64_t> words) {
    // All four position components are observable, including unconditional
    // reads of R1 and R2. Scheduling words are inserted by bundle position.
    words.insert(words.end(), {store(1, 0x70), store(2, 0x74), store(255, 0x78), store(255, 0x7c),
                              pattern("111000110000") | (7ull << 16) | 15});
    StageImage result;
    result.stage = Stage::vertex;
    const auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) result.bytes.push_back(std::byte((word >> (8 * i)) & 255));
    };
    for (const auto word : words) {
        if (result.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    return result;
}

std::vector<std::uint64_t> pair(bool truth = true, bool inverted = false) {
    return {set_predicate(truth), mov(1, 11, inverted), mov(2, 12, inverted),
            mov(1, 21, !inverted), mov(2, 22, !inverted)};
}

std::size_t branches(const Program& program) {
    return std::count_if(program.blocks.begin(), program.blocks.end(),
        [](const BasicBlock& block) { return std::holds_alternative<Branch>(block.terminator); });
}

RegisterProgram registers(std::vector<std::uint64_t> words, FactReader& facts) {
    take(facts.require(FactQuery<FactKind::depth_convention>{}, Phase::decode));
    auto decoded = take(decode(image(std::move(words)), facts, {}));
    return take(emit_ir(decoded, facts, {}));
}

// A small executable oracle for these SSA fixtures checks the actual values
// and order of effects, in addition to downstream acceptance. Unexpected
// operations fail explicitly instead of acquiring substitute semantics.
std::vector<std::uint64_t> outputs(const Program& program) {
    std::vector<std::optional<std::uint64_t>> values(program.values.size());
    const auto value = [&](const Operand& operand) -> std::uint64_t {
        if (const auto* literal = std::get_if<LiteralBits>(&operand)) return literal->bits;
        return values.at(std::get<ValueId>(operand).value).value();
    };
    std::vector<std::uint64_t> stores;
    auto block_id = program.functions.at(program.entry.value).entry;
    BlockId predecessor;
    for (std::size_t step = 0; step <= program.blocks.size(); ++step) {
        const auto& block = program.blocks.at(block_id.value);
        std::vector<std::pair<ValueId, std::uint64_t>> phi_values;
        for (const auto& phi : block.phis) {
            const auto input = std::find_if(phi.inputs.begin(), phi.inputs.end(),
                [&](const PhiInput& item) { return item.predecessor == predecessor; });
            if (input == phi.inputs.end()) throw std::runtime_error("Missing executed phi edge");
            phi_values.emplace_back(phi.output, values.at(input->value.value).value());
        }
        for (const auto& [id, bits] : phi_values) values.at(id.value) = bits;
        for (const auto id : block.operations) {
            const auto& op = program.operations.at(id.value);
            if (op.opcode == Opcode::store_io) {
                const auto& io = std::get<IoInfo>(op.info);
                for (const auto index : io.data) stores.push_back(value(op.inputs.at(index.value)));
                continue;
            }
            std::uint64_t result{};
            switch (op.opcode) {
            case Opcode::copy: case Opcode::bitcast: result = value(op.inputs.at(0)); break;
            case Opcode::logical_not: result = !value(op.inputs.at(0)); break;
            case Opcode::logical_and: result = value(op.inputs.at(0)) && value(op.inputs.at(1)); break;
            case Opcode::compare: {
                const auto comparison = std::get<ComparisonInfo>(op.info).relation;
                if (comparison != Compare::equal && comparison != Compare::not_equal)
                    throw std::runtime_error("Unexpected fixture comparison");
                const bool equal = value(op.inputs.at(0)) == value(op.inputs.at(1));
                result = comparison == Compare::equal ? equal : !equal;
                break;
            }
            default: throw std::runtime_error("Unexpected fixture opcode");
            }
            if (op.outputs.size() != 1) throw std::runtime_error("Unexpected fixture result count");
            values.at(std::get<ValueId>(op.outputs.front()).value) = result;
        }
        predecessor = block_id;
        if (const auto* branch = std::get_if<Branch>(&block.terminator)) {
            block_id = value(branch->condition) ? branch->when_true : branch->when_false;
        } else if (const auto* jump = std::get_if<Jump>(&block.terminator)) {
            block_id = jump->target;
        } else if (std::holds_alternative<Return>(block.terminator)) {
            return stores;
        } else throw std::runtime_error("Unexpected fixture terminator");
    }
    throw std::runtime_error("Fixture did not return");
}

void accepted(std::vector<std::uint64_t> words, std::size_t expected_branches,
              std::array<unsigned, 2> expected) {
    Facts source;
    FactReader facts(source);
    auto emitted = registers(std::move(words), facts);
    check(branches(emitted.ir) == expected_branches, "Complementary runs pair once, preserving later runs");
    auto ssa = take(construct_ssa(std::move(emitted), {}));
    check(outputs(ssa.ir) == std::vector<std::uint64_t>{expected[0], expected[1], 0, 0},
          "SSA execution preserves the selected definitions and unconditional output order");
    auto legal = take(legalize(std::move(ssa), facts, profile()));
    auto structured = take(structure(std::move(legal), {}));
    auto plan = take(plan_resources(structured, profile()));
    auto artifact = take(emit_msl(structured, plan, {}));
    check(!artifact.source.empty(), "Complementary synthetic vertex shader reaches MSL");
}

void rejected(std::vector<std::uint64_t> words, std::size_t expected_branches, std::string_view reg) {
    Facts source;
    FactReader facts(source);
    auto emitted = registers(std::move(words), facts);
    check(branches(emitted.ir) == expected_branches, "Unsafe guard boundaries keep separate regions");
    auto ssa = construct_ssa(std::move(emitted), {});
    const auto* failure = std::get_if<Diagnostic>(&ssa);
    const auto* program = std::get_if<SsaProgram>(&ssa);
    // No repair may justify the read; it falls back to an explicit zero.
    check(program && test::entry_zero_reads(program->ir, ValueType::word32) > 0,
          "Expected live undefined " + std::string(reg) + " to read zero: " +
              (failure ? failure->message : "no zero definition"));
}

void test_pairs_and_alternation() {
    for (const bool truth : {false, true}) {
        for (const bool inverted : {false, true}) {
            const bool first_arm = truth != inverted;
            accepted(pair(truth, inverted), 1, first_arm ? std::array{11u, 12u} : std::array{21u, 22u});
            auto words = pair(truth, inverted);
            words.insert(words.end(), {mov(1, 31, inverted), mov(2, 32, inverted)});
            accepted(words, 2, first_arm ? std::array{31u, 32u} : std::array{21u, 22u});
            words.insert(words.end(), {mov(1, 41, !inverted), mov(2, 42, !inverted)});
            accepted(words, 2, first_arm ? std::array{31u, 32u} : std::array{41u, 42u});

            // A writer after a safe complementary prefix remains separate.
            // Its predicate update cannot alter either preceding definition.
            words = pair(truth, inverted);
            words.push_back(guarded(set_predicate(!truth), 0, !inverted));
            accepted(words, 2, first_arm ? std::array{11u, 12u} : std::array{21u, 22u});
        }
    }
}

void test_missing_definitions_and_rewrites() {
    auto partial = pair();
    partial.pop_back();
    rejected(partial, 1, "R2");

    auto rewritten = pair(false);
    rewritten.insert(rewritten.begin() + 3, set_predicate(true));
    rejected(rewritten, 2, "Guest register R");

    // A writer at the start of the opposite run prevents pairing. On the
    // initially false path the inverted writer suppresses both MOV runs.
    rewritten = pair(false);
    rewritten.insert(rewritten.begin() + 3, guarded(set_predicate(true), 0, true));
    rejected(rewritten, 3, "Guest register R");

    // R2P has a dynamic predicate mask without decoded predicate destinations.
    // Initialize all predicates so the expected undefined read is a GPR.
    rewritten = pair(false);
    const auto r2p = pattern("0011100x11110") | (3ull << 8) | (7ull << 16) | (1ull << 20);
    rewritten.insert(rewritten.begin() + 3, r2p);
    for (unsigned pred = 1; pred < 7; ++pred) rewritten.insert(rewritten.begin(), set_predicate(false, pred));
    rewritten.insert(rewritten.begin(), guarded(mov(3, 1, false), 7));
    rejected(rewritten, 2, "Guest register R");
    *std::find(rewritten.begin(), rewritten.end(), r2p) = guarded(r2p, 0, true);
    rejected(rewritten, 3, "Guest register R");
}
} // namespace

int main() {
    try {
        test_pairs_and_alternation();
        test_missing_definitions_and_rewrites();
    } catch (const std::exception& error) {
        check(false, error.what());
    }
    return failures == 0 ? 0 : 1;
}
