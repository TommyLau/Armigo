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
#include <bit>
#include <cstdint>
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
                {OriginKind::explicit_option, "synthetic-correlated-guards"}};
        }
        return Unknown{"No other facts in the correlated-guard fixture"};
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

std::uint64_t mov32i(unsigned destination, std::uint32_t value, unsigned predicate = 7,
                     bool inverted = false) {
    return guarded(pattern("000000010000") | destination | (std::uint64_t{value} << 20),
                   predicate, inverted);
}

std::uint64_t mufu(unsigned destination, unsigned source, unsigned predicate,
                   bool inverted = false) {
    return guarded(pattern("0101000010000") | destination | (std::uint64_t{source} << 8) |
                       (4ull << 20),
                   predicate, inverted);
}

std::uint64_t fmul(unsigned destination, unsigned source_a, unsigned source_b,
                   unsigned predicate, bool inverted = false) {
    return guarded(pattern("0101110001101") | destination | (std::uint64_t{source_a} << 8) |
                       (std::uint64_t{source_b} << 20),
                   predicate, inverted);
}

std::uint64_t iadd32i(unsigned destination, unsigned source, std::uint32_t immediate) {
    return guarded(pattern("0001110") | destination | (std::uint64_t{source} << 8) |
                       (std::uint64_t{immediate} << 20),
                   7);
}

// ISETP.EQ/NE.AND Pn, PT, RZ, RZ, PT.
std::uint64_t set_predicate(bool truth, unsigned destination) {
    return pattern("010110110110") | (std::uint64_t{destination} << 3) | 7ull |
           (255ull << 8) | (7ull << 16) | (255ull << 20) | (7ull << 39) |
           (std::uint64_t{truth ? 2u : 5u} << 49);
}

std::uint64_t set_predicate_less(unsigned destination, unsigned left, unsigned right) {
    return pattern("010110110110") | (std::uint64_t{destination} << 3) | 7ull |
           (std::uint64_t{left} << 8) | (7ull << 16) | (std::uint64_t{right} << 20) |
           (7ull << 39) | (1ull << 49);
}

// PSETP.AND.AND Pn, PT, source, PT, PT; invert the first input
// to materialize false from PT or the complement of another predicate.
std::uint64_t predicate_copy(unsigned destination, unsigned source, bool inverted = false) {
    return guarded(pattern("0101000010010") | (std::uint64_t{destination} << 3) | 7ull |
        (std::uint64_t{source} << 12) | (std::uint64_t{inverted} << 15) |
        (7ull << 29) | (7ull << 39), 7);
}

std::uint64_t store(unsigned source, unsigned offset, unsigned predicate = 7,
                    bool inverted = false) {
    return guarded(pattern("1110111111110x") | source | (255ull << 8) |
                       (std::uint64_t{offset} << 20),
                   predicate, inverted);
}

// Logical instruction indices omit the scheduling word in every bundle.
constexpr Pc pc(std::size_t index) { return (index / 3) * 32 + (index % 3) * 8 + 8; }

std::uint64_t branch(std::size_t from, std::size_t to, unsigned predicate) {
    const auto displacement = (static_cast<std::int64_t>(pc(to)) -
                               static_cast<std::int64_t>(pc(from)) - 8) & 0xffffff;
    return guarded(pattern("111000100100") | 15 | (std::uint64_t(displacement) << 20),
                   predicate);
}

StageImage image(const std::vector<std::uint64_t>& words) {
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

constexpr std::size_t fixture_prefix_size = 14;
constexpr std::size_t correlated_fmul_index = fixture_prefix_size + 5;
constexpr std::size_t correlated_store_index = fixture_prefix_size + 6;

std::vector<std::uint64_t> fixture(bool truth, bool inverted, bool loop) {
    constexpr std::uint32_t one = 0x3f800000;
    constexpr std::uint32_t two = 0x40000000;
    constexpr std::uint32_t four = 0x40800000;
    constexpr std::uint32_t eight = 0x41000000;
    std::vector<std::uint64_t> words{
        mov32i(0, one), mov32i(4, four), mov32i(11, two), mov32i(14, eight),
        mov32i(16, four), mov32i(20, 0), mov32i(21, 2), set_predicate(true, 0),
        set_predicate(true, 1), set_predicate(true, 2), set_predicate(truth, 3),
        set_predicate(true, 4), set_predicate(false, 5), store(0, 0x70)};

    const auto region = words.size();
    words.insert(words.end(), {
        mufu(23, 11, 3, inverted),
        mufu(2, 4, 2),
        mov32i(8, 0x42a00000, 0),
        mufu(1, 14, 4),
        mov32i(5, 0x42c80000, 1),
        fmul(11, 23, 16, 3, inverted),
        store(11, 0x70, 3, inverted),
    });
    if (loop) {
        words.push_back(iadd32i(20, 20, 1));
        words.push_back(set_predicate_less(5, 20, 21));
        words.push_back(branch(words.size(), region, 5));
    }
    words.insert(words.end(), {store(255, 0x74), store(255, 0x78), store(255, 0x7c),
                              guarded(pattern("111000110000") | 15, 7)});
    return words;
}

RegisterProgram registers(std::vector<std::uint64_t> words, FactReader& facts) {
    take(facts.require(FactQuery<FactKind::depth_convention>{}, Phase::decode));
    auto decoded = take(decode(image(words), facts, {}));
    return take(emit_ir(decoded, facts, {}));
}

// This executable oracle makes the guarded store observable before later
// phases erase SSA phis. The selected reciprocal and multiply inputs produce
// exact binary32 results, so host evaluation does not introduce tolerance.
std::vector<std::uint64_t> outputs(const Program& program) {
    std::vector<std::optional<std::uint64_t>> values(program.values.size());
    const auto value = [&](const Operand& operand) -> std::uint64_t {
        if (const auto* literal = std::get_if<LiteralBits>(&operand)) return literal->bits;
        return values.at(std::get<ValueId>(operand).value).value();
    };
    std::vector<std::uint64_t> stores;
    auto block_id = program.functions.at(program.entry.value).entry;
    BlockId predecessor;
    for (std::size_t step = 0; step < 200; ++step) {
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
                if (comparison == Compare::less) {
                    result = static_cast<std::uint32_t>(value(op.inputs.at(0))) <
                             static_cast<std::uint32_t>(value(op.inputs.at(1)));
                    break;
                }
                if (comparison != Compare::equal && comparison != Compare::not_equal)
                    throw std::runtime_error("Unexpected fixture comparison");
                const bool equal = value(op.inputs.at(0)) == value(op.inputs.at(1));
                result = comparison == Compare::equal ? equal : !equal;
                break;
            }
            case Opcode::reciprocal: {
                const auto input = std::bit_cast<float>(static_cast<std::uint32_t>(value(op.inputs.at(0))));
                result = std::bit_cast<std::uint32_t>(1.0f / input);
                break;
            }
            case Opcode::multiply: {
                const auto a = std::bit_cast<float>(static_cast<std::uint32_t>(value(op.inputs.at(0))));
                const auto b = std::bit_cast<float>(static_cast<std::uint32_t>(value(op.inputs.at(1))));
                result = std::bit_cast<std::uint32_t>(a * b);
                break;
            }
            case Opcode::add:
                result = static_cast<std::uint32_t>(value(op.inputs.at(0))) +
                         static_cast<std::uint32_t>(value(op.inputs.at(1)));
                break;
            default: throw std::runtime_error("Unexpected fixture opcode");
            }
            if (op.outputs.size() != 1) throw std::runtime_error("Unexpected fixture result count");
            values.at(std::get<ValueId>(op.outputs.front()).value) = result;
        }
        predecessor = block_id;
        if (const auto* branch_op = std::get_if<Branch>(&block.terminator)) {
            block_id = value(branch_op->condition) ? branch_op->when_true : branch_op->when_false;
        } else if (const auto* jump = std::get_if<Jump>(&block.terminator)) {
            block_id = jump->target;
        } else if (std::holds_alternative<Return>(block.terminator)) {
            return stores;
        } else {
            throw std::runtime_error("Unexpected fixture terminator");
        }
    }
    throw std::runtime_error("Fixture did not return");
}

void accepted(bool truth, bool inverted, bool loop) {
    Facts source;
    FactReader facts(source);
    auto emitted = registers(fixture(truth, inverted, loop), facts);
    auto ssa = take(construct_ssa(std::move(emitted), {}));
    const bool active = truth != inverted;
    std::vector<std::uint64_t> expected{0x3f800000};
    if (active) {
        expected.push_back(0x40000000);
        if (loop) expected.push_back(0x40000000);
    }
    expected.insert(expected.end(), {0, 0, 0});
    check(outputs(ssa.ir) == expected,
          "The correlated definition reaches its guarded output store only on the selected path");
    auto legal = take(legalize(std::move(ssa), facts, profile()));
    auto structured = take(structure(std::move(legal), {}));
    auto plan = take(plan_resources(structured, profile()));
    auto artifact = take(emit_msl(structured, plan, {}));
    check(!artifact.source.empty() && artifact.source.find("[[position]]") != std::string::npos,
          "The correlated synthetic vertex shader reaches an observable MSL position output");
}

void rejected(std::vector<std::uint64_t> words, std::string_view message) {
    Facts source;
    FactReader facts(source);
    auto emitted = registers(std::move(words), facts);
    const auto ssa = construct_ssa(std::move(emitted), {});
    const auto* failure = std::get_if<Diagnostic>(&ssa);
    const auto* program = std::get_if<SsaProgram>(&ssa);
    // The guarded definition must not be used; R23 falls back to zero.
    check(program && test::entry_zero_reads(program->ir, ValueType::word32) > 0,
          std::string(message) + ": " + (failure ? failure->message : "no zero definition"));
}

void test_interleaved_correlated_guards() {
    for (const bool truth : {false, true}) {
        for (const bool inverted : {false, true}) accepted(truth, inverted, false);
    }
}

void test_loop_carried_region() {
    for (const bool truth : {false, true}) {
        for (const bool inverted : {false, true}) accepted(truth, inverted, true);
    }
}

void test_predicate_copies_and_reconvergence() {
    for (const bool truth : {false, true}) {
        Facts source;
        FactReader facts(source);
        // P1 records which SSY/SYNC arm supplied the data, then controls
        // the fallback definition after reconvergence.
        const auto target = (pc(7) - pc(2) - 8) & 0xffffff;
        std::vector<std::uint64_t> words{
            set_predicate(truth, 0), predicate_copy(1, 7),
            guarded(pattern("111000101001") | (target << 20), 7),
            guarded(pattern("1111000011111") | 15, 0, true),
            mov32i(10, 0x3f800000), predicate_copy(1, 7, true),
            guarded(pattern("1111000011111") | 15, 7),
            mov32i(10, 0x40000000, 1), store(10, 0x70),
            store(255, 0x74), store(255, 0x78), store(255, 0x7c),
            guarded(pattern("111000110000") | 15, 7)};
        auto ssa = take(construct_ssa(registers(std::move(words), facts), {}));
        check(outputs(ssa.ir) == std::vector<std::uint64_t>{truth ? 0x3f800000u : 0x40000000u, 0, 0, 0},
              "Predicate constants transferred through reconvergence select the actual definition");
        auto legal = take(legalize(std::move(ssa), facts, profile()));
        auto structured = take(structure(std::move(legal), {}));
        auto plan = take(plan_resources(structured, profile()));
        check(!take(emit_msl(structured, plan, {})).source.empty(),
              "PSETP and SSY/SYNC tail initialization reaches MSL");

        Facts copy_source;
        FactReader copy_facts(copy_source);
        words = {set_predicate(truth, 0), mov32i(10, 0x3f800000, 0),
            predicate_copy(1, 0), store(10, 0x70, 1),
            store(255, 0x74), store(255, 0x78), store(255, 0x7c),
            guarded(pattern("111000110000") | 15, 7)};
        auto copied = take(construct_ssa(registers(std::move(words), copy_facts), {}));
        auto expected = std::vector<std::uint64_t>{0, 0, 0};
        if (truth) expected.insert(expected.begin(), 0x3f800000);
        check(outputs(copied.ir) == expected, "PSETP copies preserve the source guard's definition");
    }
}

void test_noncorrelated_guards_rejected() {
    auto rewritten = fixture(false, false, false);
    rewritten.insert(rewritten.begin() + correlated_fmul_index, set_predicate(true, 3));
    rejected(std::move(rewritten), "A rewritten predicate cannot justify the earlier guarded definition");

    auto rewritten_inverted = fixture(true, true, false);
    rewritten_inverted.insert(rewritten_inverted.begin() + correlated_fmul_index,
                              set_predicate(false, 3));
    rejected(std::move(rewritten_inverted),
             "An inverted rewritten predicate cannot justify the earlier guarded definition");

    auto different = fixture(false, false, false);
    different[correlated_fmul_index] = fmul(11, 23, 16, 2);
    different[correlated_store_index] = store(11, 0x70, 2);
    rejected(std::move(different), "A different predicate cannot justify the guarded definition");
}
} // namespace

int main() {
    try {
        test_interleaved_correlated_guards();
        test_loop_carried_region();
        test_predicate_copies_and_reconvergence();
        test_noncorrelated_guards_rejected();
    } catch (const std::exception& error) {
        check(false, error.what());
    }
    return failures == 0 ? 0 : 1;
}
