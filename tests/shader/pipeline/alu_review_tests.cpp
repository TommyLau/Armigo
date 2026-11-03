// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// All regression words below are independently encoded synthetic instructions.

#include "pipeline/phases.h"

#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
std::filesystem::path fixture_directory;
void check(bool value, const std::string& label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
template<class T> T take(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) throw std::runtime_error(error->message);
    return std::get<T>(std::move(result));
}

// No constant-buffer contents are supplied as facts: the operands remain
// runtime loads through decoding, emission, SSA, legalization and MSL.
struct Facts final : FactSource {
    Knowledge<FactValue> lookup(FactKey key) const override {
        const EvidenceOrigin origin{OriginKind::explicit_option, "synthetic-alu-review"};
        if (key.kind == FactKind::constant_buffer_bound_mask) return Known<FactValue>{std::uint32_t{1}, origin};
        if (key.kind == FactKind::constant_buffer_size) return Known<FactValue>{std::uint32_t{64}, origin};
        if (key.kind == FactKind::fragment_output_type) return Known<FactValue>{IoType{ScalarType::f32, 4}, origin};
        return Unknown{"Runtime input is not an offline constant"};
    }
};

StageImage image(std::uint64_t instruction) {
    StageImage result;
    result.stage = Stage::fragment;
    auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) result.bytes.push_back(std::byte((word >> (8 * i)) & 255));
    };
    auto emit = [&](std::uint64_t word) {
        if (result.bytes.size() % 32 == 0) append(0);
        append(word);
    };
    for (unsigned reg = 0; reg != 4; ++reg) {
        // LDC.32 Rn,c[0][n*4], with RZ as the dynamic address register.
        emit(0xef9400000007ff00ull | reg | (std::uint64_t{reg * 4} << 20));
    }
    emit(instruction);
    emit(0xeff000000807ff00ull); // AST color component zero from R0.
    emit(0xe30000000007000full); // EXIT, unconditionally.
    return result;
}

struct Translation {
    StructuredProgram program;
    MslArtifact artifact;
};
std::string quote(const std::filesystem::path& path) {
    std::string result = "'";
    for (char c : path.string()) result += c == '\'' ? "'\\''" : std::string(1, c);
    return result + "'";
}
Translation translate(std::uint64_t instruction) {
    Facts source;
    FactReader facts(source);
    const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
    auto decoded = take(decode(image(instruction), facts, {}));
    auto registers = take(emit_ir(decoded, facts, {}));
    auto ssa = take(construct_ssa(std::move(registers), {}));
    auto legal = take(legalize(std::move(ssa), facts, profile));
    auto structured = take(structure(std::move(legal), {}));
    auto plan = take(plan_resources(structured, profile));
    auto artifact = take(emit_msl(structured, plan, {}));
    if (!fixture_directory.empty()) {
        const auto stem = std::to_string(instruction);
        const auto path = fixture_directory / (stem + ".metal");
        std::ofstream(path) << artifact.source;
        const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
                             quote(fixture_directory / "module-cache") + " -c " + quote(path) +
                             " -o " + quote(fixture_directory / (stem + ".air"));
        check(std::system(command.c_str()) == 0, "Encoded ALU fixture compiles as Metal 4");
    }
    return {std::move(structured), std::move(artifact)};
}

std::string raw(const Operand& operand) {
    if (auto* id = std::get_if<ValueId>(&operand)) return "value" + std::to_string(id->value);
    if (auto* literal = std::get_if<LiteralBits>(&operand)) return std::to_string(literal->bits) + "u";
    throw std::runtime_error("Unexpected register after SSA");
}
void contains(const Translation& result, const std::string& expected) {
    check(result.artifact.source.find(expected) != std::string::npos, "Exact generated MSL: " + expected);
}

void signed_clamp(std::uint64_t instruction, const std::string& label) {
    const auto result = translate(instruction);
    unsigned found{};
    for (std::size_t i = 0; i < result.program.ir.operations.size(); ++i) {
        const auto& op = result.program.ir.operations[i];
        if (op.opcode != Opcode::clamp) continue;
        ++found;
        check(op.numeric.domain == Domain::signed_integer, label + " retains signed clamp domain");
        const auto stem = "armigo_w" + std::to_string(i) + "_";
        for (unsigned slot = 0; slot != 3; ++slot) {
            contains(result, "const int " + stem + std::to_string(slot) + " = as_type<int>(" + raw(op.inputs[slot]) + ");");
        }
        contains(result, "min(max(" + stem + "0, " + stem + "1), " + stem + "2)");
    }
    check(found == 1, label + " emits exactly one signed clamp");
}

void bitfield(std::uint64_t instruction, Opcode opcode, const std::string& label) {
    const auto result = translate(instruction);
    unsigned found{};
    for (std::size_t i = 0; i < result.program.ir.operations.size(); ++i) {
        const auto& op = result.program.ir.operations[i];
        if (op.opcode != opcode) continue;
        const unsigned offset_index = opcode == Opcode::bit_extract ? 1 : 2;
        const auto* width = std::get_if<LiteralBits>(&op.inputs[offset_index + 1]);
        if (!width || width->bits != 32) continue;
        ++found;
        const auto stem = "armigo_w" + std::to_string(i) + "_";
        const auto offset = stem + (opcode == Opcode::bit_extract ? "1" : "0");
        const auto count = stem + (opcode == Opcode::bit_extract ? "2" : "1");
        contains(result, "const uint " + offset + " = min(0u, 32u);");
        contains(result, "const uint " + count + " = min(32u, 32u - " + offset + ");");
        contains(result, count + " == 0u ? 0u : ");
        contains(result, "0xffffffffu >> ((32u - " + count + ") & 31u)");
    }
    check(found == 1, label + " preserves the full-word width through decoder and emitter");
}

void same_type_flush() {
    // F2F.F32.F32.FTZ pass, R1 -> R0. Runtime input 0x00000001
    // must reach an explicit signed-zero flush, even without arithmetic.
    const auto result = translate(0x5ca8100000170a00ull);
    bool found{};
    for (const auto& op : result.program.ir.operations) {
        if (op.opcode != Opcode::convert) continue;
        const auto* conversion = std::get_if<ConversionInfo>(&op.info);
        if (!conversion || conversion->source != ScalarType::f32 || conversion->destination != ScalarType::f32) continue;
        found = true;
        check(op.numeric.flush_inputs && op.numeric.flush_output, "Same-size F2F retains both FTZ boundaries");
        const auto input = raw(op.inputs[0]);
        contains(result, "((" + input + " & 0x7f800000u) == 0u ? (" + input + " & 0x80000000u) : (" + input + "))");
    }
    check(found, "Same-size F2F retains a conversion boundary");
}

void conversion_rounding() {
    const auto rz = translate(0x5cb8018000170a00ull);
    bool found{};
    for (const auto& op : rz.program.ir.operations) {
        if (op.opcode != Opcode::convert) continue;
        found = true;
        check(op.numeric.rounding == Rounding::toward_zero, "I2F RZ survives the complete pipeline");
        contains(rz, "armigo_i2f(" + raw(op.inputs[0]) + ", false, 32u, true, false)");
    }
    check(found, "Runtime I2F reaches its rounding-aware exact helper");
    try {
        // Half narrowing with RZ has no admitted generator; it must fail
        // explicitly instead of substituting the target's nearest-even cast.
        static_cast<void>(translate(0x5ca8018000170900ull));
        check(false, "F32-to-F16 RZ is rejected explicitly");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("rounding") != std::string::npos,
              "F32-to-F16 RZ rejection identifies the unavailable rounding mode");
    }
}

void fmz_translation() {
    for (const auto instruction : {0x5c68200000270100ull, 0x59c0018000270100ull}) {
        const auto result = translate(instruction);
        bool found{};
        for (const auto& op : result.program.ir.operations) {
            if (op.opcode == Opcode::select) found = true;
        }
        check(found, "FMZ zero-product selection survives to generated MSL");
        // The zero test compares the operands directly: the GPU comparison
        // itself treats a subnormal operand as its signed zero.
        contains(result, " == as_type<float>(0u))");
    }
}

void dynamic_fields_and_funnels() {
    for (const auto instruction : {0x5c00000000270100ull, 0x5bf0018000270100ull,
                                    0x5bf8018000270100ull, 0x5cf8018000270100ull}) {
        const auto result = translate(instruction);
        bool found{};
        for (const auto& op : result.program.ir.operations) {
            if (op.opcode == Opcode::bit_extract || op.opcode == Opcode::bit_insert) {
                const auto index = op.opcode == Opcode::bit_extract ? 2u : 3u;
                if (!std::holds_alternative<ValueId>(op.inputs[index])) continue;
                found = true;
                contains(result, "min(" + raw(op.inputs[index]) + ", 32u - armigo_w");
            } else if (op.opcode == Opcode::funnel_shift_left || op.opcode == Opcode::funnel_shift_right) {
                found = true;
                contains(result, "((ulong(" + raw(op.inputs[1]) + ") << 32u) | ulong(" + raw(op.inputs[0]) + "))");
                contains(result, "(" + raw(op.inputs[2]) + " & 31u)");
            }
        }
        check(found, "Dynamic field/funnel operands survive through the real guest pipeline");
    }
    // A signed full-width extraction must leave the entire source word intact.
    const auto signed_field = translate(0x3801000200070100ull);
    contains(signed_field, "as_type<uint>(as_type<int>((");
}

constexpr std::uint64_t pattern(std::string_view bits) {
    std::uint64_t word{};
    for (unsigned index = 0; index < bits.size(); ++index)
        if (bits[index] == '1') word |= std::uint64_t{1} << (63 - index);
    return word;
}

void modified_ftz_inputs() {
    const auto add = pattern("0101110001011") | (1ull << 8) | (2ull << 20) | (7ull << 16) |
                     (1ull << 44) | (1ull << 46) | (1ull << 48) | (1ull << 50);
    const auto fma = pattern("010110011") | (1ull << 8) | (2ull << 20) | (3ull << 39) |
                     (7ull << 16) | (1ull << 48) | (1ull << 49) | (1ull << 53);
    for (const auto instruction : {add, fma}) {
        const auto result = translate(instruction);
        unsigned signs{}, arithmetic{};
        for (const auto& op : result.program.ir.operations) {
            if (op.opcode == Opcode::absolute || op.opcode == Opcode::negate) {
                ++signs;
                check(!op.numeric.flush_inputs && !op.numeric.flush_output &&
                          !op.numeric.saturate && !op.numeric.precise &&
                          op.numeric.rounding == Rounding::nearest_even,
                      "Source sign operations have no arithmetic modifiers");
            }
            if (op.opcode == Opcode::add || op.opcode == Opcode::fused_multiply_add) {
                ++arithmetic;
                check(op.numeric.flush_inputs && op.numeric.flush_output &&
                          op.numeric.rounding == Rounding::nearest_even,
                      "The consuming arithmetic retains guest FTZ and rounding");
                check(op.numeric.saturate == (instruction == add),
                      "Saturation remains on the consuming result");
            }
        }
        check(signs == 2 && arithmetic == 1, "Modified FTZ sources reach the real MSL pipeline");
    }
}

void half_packing() {
    for (unsigned selector = 0; selector != 4; ++selector) {
        const auto instruction = pattern("0101110100010") | (1ull << 8) | (2ull << 20) |
                                 (7ull << 16) | (std::uint64_t{selector} << 49);
        const auto result = translate(instruction);
        unsigned packs{}, adds{};
        for (const auto& op : result.program.ir.operations) {
            if (op.opcode == Opcode::add) {
                ++adds;
                check(op.numeric.scalar == ScalarType::f32, "Unpacked half lanes use explicit FP32 arithmetic");
            }
            if (op.opcode == Opcode::pack_half2) {
                ++packs;
                const auto* info = std::get_if<ConversionInfo>(&op.info);
                check(info && info->source == ScalarType::f32 && info->destination == ScalarType::f16,
                      "Every half result pack carries its conversion metadata");
            }
        }
        check(adds == (selector == 0 ? 2u : 1u) && packs == (selector == 1 ? 0u : 1u),
              "HADD2 output selectors preserve live lanes and merge packing");
    }
    // Compact immediate halves are the upper ten bits of each half word.
    const auto immediate = pattern("0111101x0") | (1ull << 8) | (7ull << 16) |
                           (0xf0ull << 20) | (0x100ull << 30);
    const auto result = translate(immediate);
    bool packed{};
    for (const auto& op : result.program.ir.operations) {
        if (op.opcode != Opcode::unpack_half2) continue;
        for (const auto& operand : op.inputs) {
            if (const auto* literal = std::get_if<LiteralBits>(&operand)) {
                packed |= literal->bits == 0x40003c00u;
            }
        }
    }
    check(packed, "Compact half immediates expand before explicit half unpacking");
    static_cast<void>(translate(pattern("0010110") | (1ull << 8) | (7ull << 16) |
                                (0x40003c00ull << 20)));
}

void directional_f2i() {
    const Rounding modes[] = {Rounding::nearest_even, Rounding::toward_negative,
                              Rounding::toward_positive, Rounding::toward_zero};
    const Opcode opcodes[] = {Opcode::round, Opcode::floor, Opcode::ceiling, Opcode::truncate};
    for (unsigned selector = 0; selector != 4; ++selector) {
        const auto instruction = pattern("0101110010110") | (1ull << 20) | (7ull << 16) |
            (2ull << 8) | (2ull << 10) | (1ull << 12) | (std::uint64_t{selector} << 39) |
            (1ull << 44) | (1ull << 45) | (1ull << 49);
        const auto result = translate(instruction);
        bool rounded{}, converted{};
        for (const auto& op : result.program.ir.operations) {
            if (op.opcode == opcodes[selector]) {
                rounded = true;
                check(op.numeric.rounding == modes[selector] && op.numeric.flush_inputs && op.numeric.flush_output,
                      "F2I integral operation retains decoded direction and FTZ");
            }
            if (op.opcode == Opcode::convert) {
                converted = true;
                check(op.numeric.rounding == Rounding::toward_zero,
                      "F2I casts the explicitly rounded value with truncation");
            }
        }
        check(rounded && converted, "All signed F2I directions reach MSL through the real decoder");
    }
}

void integer_source_signs() {
    const auto i2f = pattern("0101110010111") | (1ull << 20) | (7ull << 16) |
                     (2ull << 8) | (2ull << 10) | (1ull << 13) | (1ull << 45) | (1ull << 49);
    const auto i2i = pattern("0101110011100") | (1ull << 20) | (7ull << 16) |
                     (2ull << 8) | (2ull << 10) | (1ull << 12) | (1ull << 13) | (1ull << 49);
    for (const auto instruction : {i2f, i2i}) {
        const auto result = translate(instruction);
        bool found{};
        for (const auto& op : result.program.ir.operations) {
            if (op.opcode != Opcode::absolute && op.opcode != Opcode::negate) continue;
            found = true;
            check(op.numeric.domain == Domain::signed_integer && op.numeric.scalar == ScalarType::s32 &&
                      op.numeric.rounding == Rounding::not_applicable,
                  "Integer conversion source modifiers retain their interpreted domain");
        }
        check(found, "Decoded I2F and I2I integer source modifiers reach MSL");
    }
}

void lop3_routes() {
    for (const auto instruction : {
        pattern("001111") | (1ull << 8) | (7ull << 16) | (0xffull << 20) |
            (3ull << 39) | (0x96ull << 48) | (1ull << 56) | (1ull << 47),
        pattern("0000001x") | (1ull << 8) | (7ull << 16) | (3ull << 39) |
            (0x96ull << 48) | (1ull << 47)}) {
        const auto result = translate(instruction);
        bool logic{};
        for (const auto& op : result.program.ir.operations)
            logic |= op.opcode == Opcode::bit_xor || op.opcode == Opcode::bit_and;
        check(logic, "Immediate and constant LOP3 with CC reach real emission without old-Z reads");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2) {
        fixture_directory = argv[1];
        std::filesystem::create_directories(fixture_directory);
    }
    try {
        // Independently encoded review examples, with runtime register loads.
        signed_clamp(0x5ce4000000173500ull, "I2I.S16.S16.SAT");
        signed_clamp(0x5cb0100000171900ull, "F2I.S16.F32.FTZ");
        bitfield(0x3800000200070100ull, Opcode::bit_extract, "BFE.U32 width 32");
        bitfield(0x36f0010200070100ull, Opcode::bit_insert, "BFI width 32");
        same_type_flush();
        conversion_rounding();
        fmz_translation();
        dynamic_fields_and_funnels();
        modified_ftz_inputs();
        half_packing();
        directional_f2i();
        integer_source_signs();
        lop3_routes();
        // NOP emits nothing and the stage translates around it.
        const auto nop = translate(0x50b0000000070f00ull);
        check(nop.artifact.source.find("return") != std::string::npos, "NOP translates as no operation");
        // VOTE.VTG emits nothing and the stage translates around it.
        const auto vote = translate(0x50e0000000070000ull);
        check(vote.artifact.source.find("return") != std::string::npos, "VOTE.VTG translates as no operation");
        // EXIT under FCSM_TR (selector 28) falls through: the color store
        // after it still runs.
        const auto guarded_exit = translate(0xe30000000007001cull);
        check(guarded_exit.artifact.source.find("return") != std::string::npos,
              "A coverage-sample-mode EXIT translates and the stage still returns");
        // The other coverage-sample-mode selectors have no observed use and stay explicit.
        bool refused = false;
        try {
            (void)translate(0xe300000000070018ull);
        } catch (const std::exception&) {
            refused = true;
        }
        check(refused, "EXIT on coverage-sample-mode selector 24 is refused");
    } catch (const std::exception& error) {
        std::cerr << "ALU review translation failed: " << error.what() << '\n';
        return 1;
    }
    if (failures) return 1;
    std::cout << "Encoded ALU review MSL regressions passed\n";
    return 0;
}
