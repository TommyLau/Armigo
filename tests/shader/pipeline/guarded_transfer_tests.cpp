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
                {OriginKind::explicit_option, "synthetic-guarded-transfers"}};
        }
        if (key.kind == FactKind::fragment_output_type) {
            return Known<FactValue>{IoType{ScalarType::f32, 4},
                {OriginKind::explicit_option, "synthetic-guarded-transfers"}};
        }
        return Unknown{"No other facts in the guarded-transfer fixture"};
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

// Logical instruction indices omit the scheduling word in every bundle.
constexpr Pc pc(std::size_t index) { return (index / 3) * 32 + (index % 3) * 8 + 8; }

std::uint64_t transfer(std::string_view bits, unsigned predicate = 7, bool inverted = false,
                       unsigned condition = 15) {
    return guarded(pattern(bits) | condition, predicate, inverted);
}

std::uint64_t target(std::string_view bits, std::size_t from, std::size_t to,
                     unsigned predicate = 7, bool inverted = false, unsigned condition = 15) {
    const auto displacement = (static_cast<std::int64_t>(pc(to)) -
                               static_cast<std::int64_t>(pc(from)) - 8) & 0xffffff;
    return transfer(bits, predicate, inverted, condition) | (std::uint64_t(displacement) << 20);
}

void vertex_outputs(std::vector<std::uint64_t>& words) {
    words.insert(words.end(), {store(0, 0x70), store(1, 0x74), store(255, 0x78), store(255, 0x7c),
                              transfer("111000110000")});
}

StageImage image(const std::vector<std::uint64_t>& words, Stage stage = Stage::vertex) {
    StageImage result;
    result.stage = stage;
    if (stage == Stage::fragment) {
        result.sph[0x48] = std::byte{15};
        result.sph[0x4c] = std::byte{3}; // Also export sample mask and depth.
    }
    const auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) result.bytes.push_back(std::byte((word >> (8 * i)) & 255));
    };
    for (const auto word : words) {
        if (result.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    return result;
}

std::size_t branches(const Program& program) {
    return std::count_if(program.blocks.begin(), program.blocks.end(),
        [](const BasicBlock& block) { return std::holds_alternative<Branch>(block.terminator); });
}

// A small executable oracle for these SSA fixtures checks the actual values
// and order of effects, in addition to downstream acceptance. Unexpected
// operations fail explicitly instead of acquiring substitute semantics.
std::vector<std::uint64_t> outputs(const Program& program, unsigned* discards = nullptr) {
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
            if (op.opcode == Opcode::discard) {
                if (discards) ++*discards;
                continue;
            }
            std::uint64_t result{};
            switch (op.opcode) {
            case Opcode::copy: case Opcode::bitcast: result = value(op.inputs.at(0)); break;
            case Opcode::logical_not: result = !value(op.inputs.at(0)); break;
            case Opcode::logical_and: result = value(op.inputs.at(0)) && value(op.inputs.at(1)); break;
            case Opcode::compare: {
                const auto comparison = std::get<ComparisonInfo>(op.info).relation;
                if (comparison == Compare::less) {
                    result = static_cast<std::int32_t>(value(op.inputs.at(0))) <
                             static_cast<std::int32_t>(value(op.inputs.at(1)));
                    break;
                }
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

void accepted(const StageImage& input, std::size_t expected_branches,
              const std::vector<std::uint64_t>& expected, unsigned expected_discards = 0) {
    Facts source;
    FactReader facts(source);
    if (input.stage == Stage::vertex)
        take(facts.require(FactQuery<FactKind::depth_convention>{}, Phase::decode));
    const auto decoded = take(decode(input, facts, {}));
    std::vector<InstructionPredicate> predicates;
    for (const auto& block : decoded.blocks)
        for (const auto& instruction : block.instructions) predicates.push_back(instruction.predicate);
    auto emitted = take(emit_ir(decoded, facts, {}));
    check(branches(emitted.ir) == expected_branches, "A transfer shares its stable outer guard");
    std::size_t index{};
    for (const auto& block : decoded.blocks) {
        for (const auto& instruction : block.instructions) {
            const auto original = predicates[index++];
            check(instruction.predicate.index == original.index &&
                  instruction.predicate.inverted == original.inverted &&
                  instruction.predicate.enabled == original.enabled,
                  "The emitter-local PT override never changes decoded predicates");
        }
    }
    auto ssa = take(construct_ssa(std::move(emitted), {}));
    unsigned discards{};
    check(outputs(ssa.ir, &discards) == expected, "SSA execution preserves selected values and exports");
    check(discards == expected_discards, "Discard executes only on its guarded path");
    const auto legal = take(legalize(std::move(ssa), facts, profile()));
    auto structured = take(structure(legal, {}));
    const auto plan = take(plan_resources(structured, profile()));
    const auto artifact = take(emit_msl(structured, plan, {}));
    check(!artifact.source.empty(), "Guarded transfer reaches MSL through all pipeline phases");
    if (input.stage == Stage::fragment && !expected.empty()) {
        check(artifact.source.find("[[color(0)]]") != std::string::npos &&
              artifact.source.find("[[sample_mask]]") != std::string::npos &&
              artifact.source.find("[[depth(any)]]") != std::string::npos,
              "Guarded fragment EXIT retains color, sample mask and depth exports");
    }
    if (expected_discards) check(artifact.source.find("discard_fragment()") != std::string::npos,
                                "KIL emits a Metal discard");
}

std::vector<std::uint64_t> sync_words(bool truth, bool inverted, bool rewrite = false,
                                      std::string_view push = "111000101001",
                                      std::string_view pop = "1111000011111") {
    // An outer token is consumed at the inner token's target, matching the
    // nested reconvergence shape. The first two pushes carry NoPred P0 bits.
    std::vector<std::uint64_t> words{0, 0, set_predicate(truth), mov(0, 11, inverted)};
    if (rewrite) words.push_back(guarded(set_predicate(!truth), 0, inverted));
    words.push_back(mov(1, 12, inverted));
    words.push_back(transfer(pop, 0, inverted));
    words.push_back(guarded(mov(0, 21, false), 7));
    words.push_back(guarded(mov(1, 22, false), 7));
    words.push_back(transfer(pop));
    while (words.size() < 15) words.push_back(transfer("0101000010110"));
    const auto inner_target = words.size();
    words.push_back(transfer("1111000011111"));
    const auto outer_target = words.size();
    vertex_outputs(words);
    words[0] = target("111000101001", 0, outer_target, 0);
    words[1] = target(push, 1, inner_target, 0);
    return words;
}

void test_token_transfers() {
    for (const auto& [push, pop] : std::array{
            std::pair{"111000101001", "1111000011111"},
            std::pair{"111000101010", "111000110100"},
            std::pair{"111000101011", "111000110101"}}) {
        for (const bool truth : {false, true}) {
            for (const bool inverted : {false, true}) {
                accepted(image(sync_words(truth, inverted, false, push, pop)), 1,
                    truth != inverted ? std::vector<std::uint64_t>{11, 12, 0, 0} :
                                        std::vector<std::uint64_t>{21, 22, 0, 0});
            }
        }
    }
    Facts source;
    FactReader facts(source);
    const auto decoded = take(decode(image(sync_words(true, false, true)), facts, {}));
    auto emitted = take(emit_ir(decoded, facts, {}));
    check(branches(emitted.ir) == 3, "A write to P0 splits the run before guarded SYNC");
    const auto ssa = take(construct_ssa(std::move(emitted), {}));
    check(outputs(ssa.ir) == std::vector<std::uint64_t>{21, 22, 0, 0},
          "The predicate rewrite disables SYNC and reaches both fallback definitions");
}

void test_branch_and_condition() {
    for (const bool truth : {false, true}) {
        for (const bool inverted : {false, true}) {
            for (const unsigned condition : {0u, 15u}) {
                std::vector<std::uint64_t> words{set_predicate(truth), mov(0, 11, inverted),
                    mov(1, 12, inverted), target("111000100100", 3, 6, 0, inverted, condition),
                    guarded(mov(0, 21, false), 7), guarded(mov(1, 22, false), 7)};
                vertex_outputs(words);
                accepted(image(words), 1, truth != inverted && condition == 15 ?
                    std::vector<std::uint64_t>{11, 12, 0, 0} : std::vector<std::uint64_t>{21, 22, 0, 0});
            }
        }
    }
    // The complementary arm may itself end in a transfer within this block.
    for (const bool truth : {false, true}) {
        std::vector<std::uint64_t> words{set_predicate(truth), mov(0, 11, false), mov(1, 12, false),
            mov(0, 21, true), mov(1, 22, true), target("111000100100", 5, 8, 0, true),
            guarded(mov(0, 31, false), 7), guarded(mov(1, 32, false), 7)};
        vertex_outputs(words);
        accepted(image(words), 1, truth ? std::vector<std::uint64_t>{31, 32, 0, 0} :
                                         std::vector<std::uint64_t>{21, 22, 0, 0});
        // A complementary arm can also consist of just the terminal transfer.
        words = {set_predicate(truth), guarded(mov(0, 21, false), 7),
            guarded(mov(1, 22, false), 7), mov(0, 11, false), mov(1, 12, false),
            target("111000100100", 5, 8, 0, true), guarded(mov(0, 31, false), 7),
            guarded(mov(1, 32, false), 7)};
        vertex_outputs(words);
        accepted(image(words), 1, truth ? std::vector<std::uint64_t>{31, 32, 0, 0} :
                                         std::vector<std::uint64_t>{21, 22, 0, 0});
    }
}

void test_fragment_exit_and_kill() {
    for (const bool truth : {false, true}) {
        for (const bool inverted : {false, true}) {
            std::vector<std::uint64_t> words{set_predicate(truth)};
            for (unsigned reg = 0; reg < 6; ++reg) words.push_back(mov(reg, 11 + reg, inverted));
            words.push_back(transfer("111000110000", 0, inverted));
            for (unsigned reg = 0; reg < 6; ++reg) words.push_back(guarded(mov(reg, 21 + reg, false), 7));
            words.push_back(transfer("111000110000"));
            accepted(image(words, Stage::fragment), 1, truth != inverted ?
                std::vector<std::uint64_t>{11, 12, 13, 14, 15, 16} :
                std::vector<std::uint64_t>{21, 22, 23, 24, 25, 26});
            words[7] = target("111000100100", 7, 14, 0, inverted);
            accepted(image(words, Stage::fragment), 1, truth != inverted ?
                std::vector<std::uint64_t>{11, 12, 13, 14, 15, 16} :
                std::vector<std::uint64_t>{21, 22, 23, 24, 25, 26});
            // KIL preserves the existing discard effect and continues to the
            // decoded fallthrough. Both paths then initialize all exports.
            words[7] = transfer("111000110011", 0, inverted);
            accepted(image(words, Stage::fragment), 1, {21, 22, 23, 24, 25, 26}, truth != inverted);
        }
    }
}

void test_dynamic_condition_and_multiple_targets() {
    for (const bool truth : {false, true}) {
        for (const bool zero : {false, true}) {
            // @P0 LOP32I.PASS_B.CC RZ, RZ, imm initializes Z inside the
            // guarded body. BRA.Z must still test it after satisfying P0.
            const auto flags = pattern("000001") | 255ull | (255ull << 8) |
                (std::uint64_t{zero ? 0u : 1u} << 20) | (1ull << 52) | (3ull << 53);
            std::vector<std::uint64_t> words{set_predicate(truth), guarded(flags, 0),
                mov(0, 11, false), mov(1, 12, false), target("111000100100", 4, 7, 0, false, 10),
                guarded(mov(0, 21, false), 7), guarded(mov(1, 22, false), 7)};
            vertex_outputs(words);
            accepted(image(words), 2, truth && zero ? std::vector<std::uint64_t>{11, 12, 0, 0} :
                                                     std::vector<std::uint64_t>{21, 22, 0, 0});
        }
        for (const bool nested : {false, true}) {
            // P1 selects one or two tokens. Both the fused pop and the
            // fallthrough pop must dispatch using their original selectors.
            std::vector<std::uint64_t> words{set_predicate(nested, 1), set_predicate(truth),
                target("111000101001", 2, 13), target("111000100100", 3, 5, 1),
                target("111000100100", 4, 6), target("111000101001", 5, 12),
                mov(0, 11, false), mov(1, 12, false), transfer("1111000011111", 0),
                guarded(mov(0, 21, false), 7), guarded(mov(1, 22, false), 7),
                transfer("1111000011111"), transfer("1111000011111")};
            vertex_outputs(words);
            accepted(image(words), 4, truth ? std::vector<std::uint64_t>{11, 12, 0, 0} :
                                             std::vector<std::uint64_t>{21, 22, 0, 0});
        }
        // A final guarded EXIT has no decoded fallthrough. Its false edge
        // retains the existing empty-return behavior at the end of a stream.
        std::vector<std::uint64_t> words{set_predicate(truth)};
        for (unsigned reg = 0; reg < 6; ++reg) words.push_back(mov(reg, 11 + reg, false));
        words.push_back(transfer("111000110000", 0));
        accepted(image(words, Stage::fragment), 1, truth ?
            std::vector<std::uint64_t>{11, 12, 13, 14, 15, 16} : std::vector<std::uint64_t>{});
    }
}
} // namespace

int main() {
    try {
        test_token_transfers();
        test_branch_and_condition();
        test_fragment_exit_and_kill();
        test_dynamic_condition_and_multiple_targets();
    } catch (const std::exception& error) {
        check(false, error.what());
    }
    return failures == 0 ? 0 : 1;
}
