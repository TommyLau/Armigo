// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/frontend.h>
#include "fixture.h"
#include <armigo/shader_native/instruction.h>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
namespace {
using namespace sn::test;
using Outputs = std::map<std::tuple<ir::IoSemantic, unsigned, unsigned>, std::uint32_t>;
// Independent scalar execution of the emitted register IR: the oracle does
// not know native opcode selectors, register field positions or SPH offsets.
Outputs execute(const ir::Program& p) {
    std::array<std::uint32_t, 256> registers{}; std::array<std::uint32_t, 7> predicates{}; std::map<unsigned, std::uint32_t> values; Outputs output;
    auto read = [&](const ir::Operand& input) -> std::uint32_t {
        if (auto l = std::get_if<ir::LiteralBits>(&input)) return l->bits;
        if (auto r = std::get_if<ir::Register>(&input)) { return r->file == ir::RegisterFile::predicate ? predicates.at(r->index) : registers.at(r->index); }
        if (auto v = std::get_if<ir::ValueId>(&input)) return values.at(v->value);
        if (auto c = std::get_if<ir::ConstantWord>(&input)) {
            check(c->slot == 3 && (c->byte_offset == 16 || c->byte_offset == 20), "oracle constant address");
            return c->byte_offset == 16 ? 0x40000000 : 0x3e800000;
        }
        throw std::runtime_error("oracle unsupported operand");
    };
    auto block = p.functions.at(p.entry.value).entry;
    for (unsigned steps = 0; steps < 1000; ++steps) {
        const auto& b = p.blocks.at(block.value);
        for (auto id : b.operations) {
            auto& op = p.operations.at(id.value);
            if (op.opcode == ir::Opcode::store_io) {
                const auto& io = std::get<ir::IoInfo>(op.info);
                for (unsigned c = 0; c < 4; ++c) if (io.component_mask & (1u << c)) output[{io.semantic, io.location, c}] = read(op.inputs.at(0));
            } else {
                std::uint32_t value = 0;
                if (op.opcode == ir::Opcode::load_io) {
                    const auto& io = std::get<ir::IoInfo>(op.info);
                    check(io.semantic == ir::IoSemantic::vertex_id || io.semantic == ir::IoSemantic::instance_id || io.semantic == ir::IoSemantic::generic || io.semantic == ir::IoSemantic::fragment_coord, "oracle system input");
                    value = io.semantic == ir::IoSemantic::vertex_id ? 17 : io.semantic == ir::IoSemantic::instance_id ? 23 : io.semantic == ir::IoSemantic::fragment_coord ? 0x3f000000 : 0xc0200000;
                } else if (op.opcode == ir::Opcode::load) {
                    const auto& m = std::get<ir::MemoryInfo>(op.info);
                    check(m.space == ir::AddressSpace::constant_buffer && m.constant_buffer_slot == 3 && m.element_bytes == 4, "oracle memory contract");
                    const auto base = std::uint64_t(m.byte_offset) + (m.dynamic_byte_offset ? read(op.inputs.at(m.dynamic_byte_offset->value)) : 0);
                    for (unsigned component = 0; component < m.elements; ++component) {
                        const auto address = (base + component * m.element_bytes) & ~3ull;
                        value = address == 0 ? 0xdeadbeef : address == 16 ? 0x40000000 : address == 20 ? 0x3e800000 : 0;
                        values[std::get<ir::ValueId>(op.outputs.at(component)).value] = value;
                    }
                    continue;
                } else if (op.opcode == ir::Opcode::absolute) value = read(op.inputs[0]) & 0x7fffffff;
                else if (op.opcode == ir::Opcode::negate) value = read(op.inputs[0]) ^ 0x80000000;
                else if (op.opcode == ir::Opcode::copy) value = read(op.inputs.at(0));
                else if (op.opcode == ir::Opcode::logical_not) value = !read(op.inputs[0]);
                else if (op.opcode == ir::Opcode::logical_and) value = bool(read(op.inputs[0])) && bool(read(op.inputs[1]));
                else if (op.opcode == ir::Opcode::logical_or) value = bool(read(op.inputs[0])) || bool(read(op.inputs[1]));
                else if (op.opcode == ir::Opcode::logical_xor) value = bool(read(op.inputs[0])) != bool(read(op.inputs[1]));
                else if (op.opcode == ir::Opcode::bit_and) value = read(op.inputs[0]) & read(op.inputs[1]);
                else if (op.opcode == ir::Opcode::bit_xor) value = read(op.inputs[0]) ^ read(op.inputs[1]);
                else if (op.numeric.domain == ir::Domain::floating) {
                    auto fp = [&](unsigned index) {
                        auto bits = read(op.inputs.at(index));
                        if (op.numeric.flush_inputs && !(bits & 0x7f800000)) bits &= 0x80000000;
                        return std::bit_cast<float>(bits);
                    };
                    if (op.opcode == ir::Opcode::compare || op.opcode == ir::Opcode::is_nan) {
                        bool comparison = false;
                        if (op.opcode == ir::Opcode::is_nan) comparison = std::isnan(fp(0));
                        else {
                            auto info = std::get<ir::ComparisonInfo>(op.info); const auto a = fp(0), b = fp(1);
                            if (std::isnan(a) || std::isnan(b)) comparison = info.unordered;
                            else switch (info.relation) {
                            case ir::Compare::less: comparison = a < b; break;
                            case ir::Compare::equal: comparison = a == b; break;
                            case ir::Compare::less_equal: comparison = a <= b; break;
                            case ir::Compare::greater: comparison = a > b; break;
                            case ir::Compare::not_equal: comparison = a != b; break;
                            case ir::Compare::greater_equal: comparison = a >= b; break;
                            default: throw std::runtime_error("oracle comparison relation");
                            }
                        }
                        values[std::get<ir::ValueId>(op.outputs[0]).value] = comparison; continue;
                    }
                    float result = 0;
                    switch (op.opcode) {
                    case ir::Opcode::add: result = fp(0) + fp(1); break;
                    case ir::Opcode::multiply: result = fp(0) * fp(1); break;
                    case ir::Opcode::fused_multiply_add: result = std::fma(fp(0), fp(1), fp(2)); break;
                    case ir::Opcode::reciprocal: result = 1.0f / fp(0); break;
                    case ir::Opcode::reciprocal_sqrt: result = 1.0f / std::sqrt(fp(0)); break;
                    case ir::Opcode::square_root: result = std::sqrt(fp(0)); break;
                    default: throw std::runtime_error("oracle unsupported floating operation");
                    }
                    if (op.numeric.saturate) result = !(result > 0) ? 0 : result >= 1 ? 1 : result;
                    value = std::bit_cast<std::uint32_t>(result);
                    if (op.numeric.flush_output && !(value & 0x7f800000)) value &= 0x80000000;
                } else if (op.opcode == ir::Opcode::add) value = read(op.inputs[0]) + read(op.inputs[1]);
                else if (op.opcode == ir::Opcode::multiply) value = read(op.inputs[0]) * read(op.inputs[1]);
                else throw std::runtime_error("oracle unsupported operation");
                if (auto r = std::get_if<ir::Register>(&op.outputs.at(0))) {
                    if (r->file == ir::RegisterFile::predicate) predicates.at(r->index) = value;
                    else registers.at(r->index) = value;
                }
                else values[std::get<ir::ValueId>(op.outputs.at(0)).value] = value;
            }
        }
        if (auto jump = std::get_if<ir::Jump>(&b.terminator)) block = jump->target;
        else if (auto branch = std::get_if<ir::Branch>(&b.terminator)) block = read(branch->condition) ? branch->when_true : branch->when_false;
        else { check(std::holds_alternative<ir::Return>(b.terminator), "oracle unsupported terminator"); return output; }
    }
    throw std::runtime_error("oracle did not terminate");
}
ir::RegisterProgram lowered(const Fixture& f) {
    auto r = sn::emit_native_ir(f.program());
    if (auto e = std::get_if<ir::Diagnostic>(&r)) throw std::runtime_error(e->message);
    return std::get<ir::RegisterProgram>(std::move(r));
}
void refused(const Fixture& f) { auto r = sn::emit_native_ir(f.program()); check(std::holds_alternative<ir::Diagnostic>(r), "expected native frontend refusal"); }
void compile_fixture(const Fixture& f, const std::filesystem::path& directory, const char* name) {
    auto r = sn::translate_native(f.program(), {ir::DepthConvention::negative_one_to_one, true, f.stage == sn::Stage::fragment ? ir::FragmentPositionConvention::metal_position_w_matches_guest_position_w : ir::FragmentPositionConvention::unspecified});
    if (auto e = std::get_if<ir::Diagnostic>(&r)) throw std::runtime_error(e->message);
    auto& artifact = std::get<ir::MslArtifact>(r); check(!artifact.source.empty(), "MSL source");
    check(artifact.source.find("math_mode(safe)") != std::string::npos && artifact.source.find("contract(off)") != std::string::npos, "native-only safe non-contracting policy");
    check(artifact.reflection.profile.version == 2, "numeric policy changes native artifact version");
    check(!artifact.reflection.io.empty(), "reflection IO");
    if (!directory.empty()) std::ofstream(directory / (std::string(name) + ".metal")) << artifact.source;
}

sn::Instruction comparison(unsigned function, unsigned combine = 0, unsigned input = 7, unsigned first = 0, unsigned second = 1) {
    auto op = raw(0x20b); field(op, 24, 8, 2); field(op, 32, 8, 3);
    field(op, 76, 4, function); field(op, 80, 1, 1); field(op, 74, 2, combine); field(op, 87, 4, input);
    field(op, 81, 3, first); field(op, 84, 3, second); return op;
}
void predicate_tests(const std::filesystem::path& directory) {
    // For -1 < +2: ordered comparisons F,LT,EQ,LE,GT,NE,GE,NUM,NAN,
    // unordered comparisons, and T have these independently specified results.
    constexpr bool ordered[]{false,true,false,true,false,true,false,true,false,true,false,true,false,true,false,true};
    for (unsigned function = 0; function < 16; ++function) for (bool nan : {false,true}) {
        const bool expected = nan ? function >= 8 : ordered[function];
        for (unsigned combine = 0; combine < 3; ++combine) for (bool mask : {false,true}) {
            auto op = comparison(function, combine, mask ? 7 : 15);
            Fixture f({move(0,1),move(1,2),move(2,nan?0x7fc00001:0xbf800000),move(3,0x40000000),op,move(0,10,0),move(1,20,1),exit_instruction()}); put(f.code,0x48,3);
            auto values=execute(lowered(f).ir);
            auto merge=[&](bool c){return combine==0?c&&mask:combine==1?c||mask:c!=mask;};
            check(values.at({ir::IoSemantic::color,0,0})==(merge(expected)?10u:1u) && values.at({ir::IoSemantic::color,0,1})==(merge(!expected)?20u:2u),"ordered/unordered comparison and independent paired predicate combination");
        }
    }
    for (unsigned selector : {7u,8u}) {
        Fixture nan_ftz({move(0,0),move(1,0),move(2,0x7fc00001),move(3,0),comparison(selector),move(0,1,0),exit_instruction()});
        compile_fixture(nan_ftz,directory,selector==7?"num_ftz":"nan_ftz");
    }
    auto initialize=comparison(1,0,7,0,7), aliased=comparison(4,0,0);
    Fixture alias({move(0,1),move(1,2),move(2,0x3f800000),move(3,0x40000000),initialize,aliased,move(0,10,0),move(1,20,1),exit_instruction()}); put(alias.code,0x48,3);
    auto values=execute(lowered(alias).ir); check(values.at({ir::IoSemantic::color,0,0})==1 && values.at({ir::IoSemantic::color,0,1})==20,"predicate accumulator is read before either aliased output is written");
    compile_fixture(alias,directory,"predicate_alias");
    auto cb=raw(0xa02); field(cb,16,8,2); field(cb,54,5,3); field(cb,38,16,16);cb.words[2]=0xf00;
    auto branch=raw(0x947);field(branch,12,4,0);field(branch,87,4,7);field(branch,32,50,16);
    for (bool taken : {false,true}) {
        Fixture f({move(0,11),cb,move(3,taken?0x40400000:0x3f800000),comparison(1,0,7,0,7),branch,move(0,22),exit_instruction()});
        check(execute(lowered(f).ir).at({ir::IoSemantic::color,0,0})==(taken?11u:22u),"data-dependent native branch");
        compile_fixture(f,directory,taken?"branch_taken":"branch_fallthrough");
    }
    Fixture duplicate({move(2,0),move(3,0),comparison(1,0,7,0,0),exit_instruction()});refused(duplicate);
    Fixture invalid({move(2,0),move(3,0),comparison(1,3),exit_instruction()});refused(invalid);
}
sn::Instruction ipa(unsigned destination, unsigned address, bool constant = false, bool centroid = false) {
    auto op = raw(0x326); field(op,16,8,destination); field(op,32,8,255); field(op,64,8,address/4);
    field(op,81,3,7); field(op,78,1,constant); field(op,76,2,centroid?1:0); return op;
}
void interpolation_tests(const std::filesystem::path& directory) {
    for (unsigned mode : {1u,2u,3u}) {
        Fixture f({ipa(0,0x80,mode==1),exit_instruction()}); f.code[0x18]=std::byte(mode);
        if(mode==2) { refused(f); continue; }
        auto out=execute(lowered(f).ir);
        check(out.at({ir::IoSemantic::color,0,0})==(mode==2?0xbfa00000u:0xc0200000u),"flat, linear and perspective IPA expected bits");
        compile_fixture(f,directory,("interpolation_"+std::to_string(mode)).c_str());
        if(mode==2)check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(f.program())),"perspective IPA needs an explicit PositionW adapter");
    }
    Fixture linear_centroid({ipa(0,0x80,false,true),exit_instruction()});linear_centroid.code[0x18]=std::byte{3};
    compile_fixture(linear_centroid,directory,"linear_centroid");
    linear_centroid.code[0x18]=std::byte{2};refused(linear_centroid);
    Fixture absent({ipa(0,0x80),exit_instruction()});refused(absent);
    Fixture contradicted({ipa(0,0x80,true),exit_instruction()});contradicted.code[0x18]=std::byte{2};refused(contradicted);
    auto reciprocal=raw(0x308);field(reciprocal,16,8,2);field(reciprocal,32,8,1);field(reciprocal,74,6,4);
    auto multiply=raw(0x220);field(multiply,24,8,0);field(multiply,32,8,2);field(multiply,84,3,4);
    Fixture corrected({ipa(0,0x80),ipa(1,0x7c),reciprocal,multiply,exit_instruction()});
    corrected.code[0x18]=std::byte{2};corrected.code[0x17]=std::byte{0x80};
    refused(corrected);
}
void arithmetic_tests(const std::filesystem::path& directory) {
    auto result_bits = [&](const Fixture& f) { return execute(lowered(f).ir).at({ir::IoSemantic::color, 0, 0}); };
    for (unsigned opcode : {0x220u, 0x820u, 0xa20u, 0x221u, 0x421u, 0x621u, 0x821u, 0xa21u, 0x223u, 0x423u, 0x623u, 0x823u, 0xa23u}) {
        auto op = raw(opcode); field(op, 16, 8, 0); field(op, 24, 8, 1);
        const bool multiply = (opcode & 0x1ff) == 0x20, fma = (opcode & 0x1ff) == 0x23;
        const auto form = (opcode >> 9) & 7;
        if (multiply) field(op, 84, 3, 4);
        if (form == 1) field(op, 32, 8, 2);
        else if (form == 2 || form == 4) op.words[1] = fma && form == 2 ? 0x3e800000 : 0x40000000;
        else { field(op, 54, 5, 3); field(op, 38, 16, fma && form == 3 ? 20 : 16); }
        if (fma) field(op, 64, 8, form == 2 || form == 3 ? 2 : 3);
        field(op, 80, 1, 1);
        Fixture f({move(1, 0x3fc00000), move(2, 0x40000000), move(3, 0x3e800000), op, exit_instruction()});
        check(result_bits(f) == (fma ? 0x40500000u : multiply ? 0x40400000u : 0x40600000u), "native arithmetic source form expected bits");
        compile_fixture(f, directory, ("arithmetic_" + std::to_string(opcode)).c_str());
        field(op, 78, 2, 1); Fixture rounding({move(1, 0), move(2, 0), move(3, 0), op, exit_instruction()}); refused(rounding);
    }
    auto fma = raw(0x223); field(fma, 24, 8, 1); field(fma, 32, 8, 2); field(fma, 64, 8, 3);
    field(fma, 80, 1, 1);
    Fixture fused({move(1, 0x3f800001), move(2, 0x3f7ffffe), move(3, 0xbf800000), fma, exit_instruction()});
    check(result_bits(fused) == 0xa8800000u, "single-rounding fused multiply-add cancellation");
    compile_fixture(fused, directory, "fused_cancellation");
    auto mul = raw(0x220); field(mul, 24, 8, 1); field(mul, 32, 8, 2); field(mul, 84, 3, 4);
    Fixture denormal({move(1, 1), move(2, 0x40000000), mul, exit_instruction()}); refused(denormal);
    field(mul, 80, 1, 1); Fixture flushed({move(1, 1), move(2, 0x40000000), mul, exit_instruction()}); check(result_bits(flushed) == 0, "native FTZ mapping");
    compile_fixture(flushed, directory, "ftz_multiply");
    auto add = raw(0x221); field(add, 24, 8, 1); field(add, 32, 8, 2); field(add, 77, 1, 1);
    field(add, 80, 1, 1);
    Fixture saturated({move(1, 0x3fc00000), move(2, 0x40000000), add, exit_instruction()}); check(result_bits(saturated) == 0x3f800000, "native saturation mapping");
    compile_fixture(saturated, directory, "saturated_add");
    field(add, 77, 1, 0); field(add, 73, 1, 1); field(add, 72, 1, 1);
    Fixture signs({move(1, 0xbfc00000), move(2, 0x40000000), add, exit_instruction()}); check(result_bits(signs) == 0x3f000000, "absolute then negate source order");
    compile_fixture(signs, directory, "signed_add");
    auto sign_b = raw(0x221); field(sign_b, 24, 8, 1); field(sign_b, 32, 8, 2); field(sign_b, 62, 2, 3);
    field(sign_b, 80, 1, 1);
    Fixture signed_b({move(1, 0x3fc00000), move(2, 0xc0000000), sign_b, exit_instruction()});
    check(result_bits(signed_b) == 0xbf000000, "register B absolute and negate ordering");
    auto sign_c = raw(0x223); field(sign_c, 24, 8, 1); field(sign_c, 32, 8, 2); field(sign_c, 64, 8, 3); field(sign_c, 74, 2, 3);
    field(sign_c, 80, 1, 1);
    Fixture signed_c({move(1, 0x3fc00000), move(2, 0x40000000), move(3, 0xbe800000), sign_c, exit_instruction()});
    check(result_bits(signed_c) == 0x40300000, "upper source C absolute and negate ordering");
    auto sign_constant = raw(0xa21); field(sign_constant, 24, 8, 1); field(sign_constant, 54, 5, 3); field(sign_constant, 38, 16, 16); field(sign_constant, 63, 1, 1);
    field(sign_constant, 80, 1, 1);
    Fixture neg_constant({move(1, 0x3fc00000), sign_constant, exit_instruction()});
    check(result_bits(neg_constant) == 0xbf000000, "constant sign modifier preserves bank and byte offset");
    for (unsigned function : {4u, 5u, 8u}) {
        auto op = raw(0x308); field(op, 32, 8, 1); field(op, 74, 6, function);
        Fixture f({move(1, 0x40800000), op, exit_instruction()});
        refused(f);
        field(op, 73, 1, 1); Fixture half({move(1, 0), op, exit_instruction()}); refused(half);
    }
    auto ldc = raw(0xb82); field(ldc, 24, 8, 1); field(ldc, 54, 5, 3); field(ldc, 38, 16, 0xfffc); field(ldc, 73, 3, 4);
    Fixture indexed({move(1, 20), ldc, exit_instruction()}); check(result_bits(indexed) == 0x40000000, "signed constant displacement is added before host widening");
    compile_fixture(indexed, directory, "indexed_constant_negative");
    field(ldc, 38, 16, 17); Fixture wrapped({move(1, 0xffffffff), ldc, exit_instruction()});
    check(result_bits(wrapped) == 0x40000000, "constant address arithmetic wraps at guest width");
    compile_fixture(wrapped, directory, "indexed_constant_wrap");
    field(ldc, 38, 16, 0xfffc); field(ldc, 73, 3, 5);
    Fixture pair({move(1, 20), ldc, exit_instruction()}); put(pair.code, 0x48, 3);
    auto pair_output = execute(lowered(pair).ir);
    check(pair_output.at({ir::IoSemantic::color, 0, 0}) == 0x40000000 && pair_output.at({ir::IoSemantic::color, 0, 1}) == 0x3e800000, "paired load snapshots overlapping index register and preserves word order");
    compile_fixture(pair, directory, "constant_pair_overlap");
    field(ldc, 38, 16, 0); Fixture boundary({move(1, 0xfffffffc), ldc, exit_instruction()}); put(boundary.code, 0x48, 3);
    pair_output = execute(lowered(boundary).ir);
    check(pair_output.at({ir::IoSemantic::color, 0, 0}) == 0 && pair_output.at({ir::IoSemantic::color, 0, 1}) == 0, "paired host component address must not wrap into low backing");
    compile_fixture(boundary, directory, "constant_pair_boundary");
    field(ldc, 16, 8, 1); Fixture odd_pair({move(1, 0), ldc, exit_instruction()}); refused(odd_pair);
    field(ldc, 16, 8, 254); Fixture zero_pair({move(1, 0), ldc, exit_instruction()}); refused(zero_pair);
    field(ldc, 16, 8, 0);
    field(ldc, 78, 2, 2); Fixture slot_indexed({move(1, 0), ldc, exit_instruction()}); refused(slot_indexed);
    for (unsigned signedness : {0u, 1u}) {
        auto op = raw(0x824); field(op, 24, 8, 1); op.words[1] = 2; op.words[2] = 0x078e0003 | (signedness << 9);
        Fixture f({move(1, 0xffffffff), move(3, 5), op, exit_instruction()}); check(result_bits(f) == 3, "low-word integer multiply-add wraps modulo 2^32");
        compile_fixture(f, directory, ("integer_multiply_add_" + std::to_string(signedness)).c_str());
    }
}
}
int main(int argc, char** argv) {
    try {
        std::filesystem::path directory; if (argc == 2) { directory = argv[1]; std::filesystem::create_directories(directory); }
        auto branch = raw(0x947); field(branch, 32, 50, 16); field(branch, 87, 4, 7);
        Fixture fragment({move(0, 0x7fc12345), move(0, 0x12345678, 15), branch, move(0, 0x55555555), exit_instruction()});
        auto output = execute(lowered(fragment).ir); check(output.at({ir::IoSemantic::color, 0, 0}) == 0x7fc12345, "raw bits, false predicate and branch preserve original value");
        compile_fixture(fragment, directory, "fragment_move_branch");
        std::vector<sn::Instruction> moves; for (unsigned i = 0; i < 8; ++i) moves.push_back(move(i, 100 + i)); moves.push_back(exit_instruction());
        Fixture sparse(moves); put(sparse.code, 0x48, (5 << 4) | (8 << 12)); output = execute(lowered(sparse).ir);
        check(output.size() == 3 && output.at({ir::IoSemantic::color, 1, 0}) == 100 && output.at({ir::IoSemantic::color, 1, 2}) == 102 && output.at({ir::IoSemantic::color, 3, 3}) == 107, "sparse target export allocation");
        compile_fixture(sparse, directory, "fragment_sparse_exports");
        auto store = raw(0x322); field(store, 24, 8, 255); field(store, 40, 10, 0x70); field(store, 64, 8, 255); field(store, 74, 2, 3);
        Fixture vertex({move(0, 0x3f800000), move(1, 0x40000000), move(2, 0x3f000000), move(3, 0x3f800000), store, exit_instruction()}, sn::Stage::vertex);
        output = execute(lowered(vertex).ir); check(output.at({ir::IoSemantic::position, 0, 2}) == 0x3f000000, "native position store");
        compile_fixture(vertex, directory, "vertex_position");
        auto no_depth = sn::translate_native(vertex.program()); check(std::holds_alternative<ir::Diagnostic>(no_depth), "unknown clip convention remains refused");
        vertex.code[0x35] = std::byte{}; refused(vertex);
        Fixture broadcast({move(0, 0), exit_instruction()});
        put(broadcast.code, 0, (4 << 5) | 2 | (5 << 10)); refused(broadcast);
        Fixture undefined({exit_instruction()}); refused(undefined);
        Fixture unknown({move(0, 1), raw(0x3a9), exit_instruction()}); refused(unknown);
        Fixture unreachable({move(0, 2), exit_instruction(), raw(0x3a9)}); lowered(unreachable);
        sparse.code[0x4c] = std::byte{1}; refused(sparse);
        arithmetic_tests(directory);
        predicate_tests(directory);
        interpolation_tests(directory);
        auto load = raw(0x321); field(load, 24, 8, 255); field(load, 32, 8, 255); field(load, 40, 10, 0x2fc);
        Fixture system({load, move(1, 0), move(2, 0), move(3, 0x3f800000), store, exit_instruction()}, sn::Stage::vertex);
        system.code[0x2b] = std::byte{0x80};
        check(execute(lowered(system).ir).at({ir::IoSemantic::position, 0, 0}) == 17, "vertex system-value load");
        compile_fixture(system, directory, "vertex_id");
        auto point_store = raw(0x322); field(point_store, 24, 8, 255); field(point_store, 40, 10, 0x6c); field(point_store, 64, 8, 255);
        Fixture point({move(0, 0x40000000), point_store, move(0, 0), move(1, 0), move(2, 0), move(3, 0x3f800000), store, exit_instruction()}, sn::Stage::vertex);
        point.code[0x35] |= std::byte{8};
        check(execute(lowered(point).ir).at({ir::IoSemantic::point_size, 0, 0}) == 0x40000000, "point size output");
        compile_fixture(point, directory, "point_size");
        auto attr = raw(0x321); field(attr, 16, 8, 4); field(attr, 24, 8, 255); field(attr, 32, 8, 255); field(attr, 40, 10, 0x80);
        auto multiply = raw(0x820); field(multiply, 24, 8, 4); multiply.words[1] = 0x40000000; field(multiply, 73, 1, 1); field(multiply, 72, 1, 1); field(multiply, 84, 3, 4);
        auto varying = raw(0x322); field(varying, 24, 8, 255); field(varying, 40, 10, 0x80); field(varying, 64, 8, 255);
        field(multiply, 80, 1, 1);
        Fixture attributes({attr, multiply, varying, move(1, 0), move(2, 0), move(3, 0x3f800000), store, exit_instruction()}, sn::Stage::vertex);
        attributes.code[0x18] = std::byte{1}; attributes.code[0x36] = std::byte{1};
        check(execute(lowered(attributes).ir).at({ir::IoSemantic::generic, 0, 0}) == 0xc0a00000, "typed sign operations preserve floating input class and output bits");
        compile_fixture(attributes, directory, "vertex_attribute_sign");
        auto rate = raw(0x322); field(rate,24,8,255); field(rate,32,8,4); field(rate,40,10,0x5c); field(rate,64,8,255);
        for (unsigned index : {0u, 15u, 0xffffffffu}) {
            Fixture vprs({move(0,0),move(1,0),move(2,0),move(3,0x3f800000),move(4,index),rate,store,exit_instruction()},sn::Stage::vertex);
            refused(vprs);
            sn::FrontendOptions options; options.vertex_depth=ir::DepthConvention::negative_one_to_one;
            options.pixel_rate=sn::PixelRateState::enabled;
            check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(vprs.program(),options)),"enabled VPRS remains refused");
            options.pixel_rate=sn::PixelRateState::disabled;
            auto specialized=sn::translate_native(vprs.program(),options);
            if(auto e=std::get_if<ir::Diagnostic>(&specialized))throw std::runtime_error(e->message);
            auto& artifact=std::get<ir::MslArtifact>(specialized);
            check(artifact.reflection.profile.identity==sn::disabled_pixel_rate_profile,"VPRS state requirement is retained in the artifact profile");
            check(artifact.reflection.io.size()==1 && artifact.reflection.io[0].semantic==ir::IoSemantic::position,"disabled VPRS introduces no fabricated output binding");
            if(!directory.empty())std::ofstream(directory/("vprs_disabled_"+std::to_string(index)+".metal"))<<artifact.source;
        }
        sn::FrontendOptions disabled_unused; disabled_unused.vertex_depth=ir::DepthConvention::negative_one_to_one;disabled_unused.pixel_rate=sn::PixelRateState::disabled;
        auto unchanged=sn::translate_native(system.program(),disabled_unused);
        check(std::holds_alternative<ir::MslArtifact>(unchanged) && std::get<ir::MslArtifact>(unchanged).reflection.profile.identity=="armigo-native-metal4","unused specialization does not add a false state requirement");
        std::cout << "Native frontend contracts passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
