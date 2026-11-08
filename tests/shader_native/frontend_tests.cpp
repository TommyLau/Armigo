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
#include "stage_memory.h"
#include "builder.h"
#include "pipeline/handle_program.h"
#include "pipeline/phases.h"
#include "pipeline/passes/passes.h"
#include <armigo/shader_native/instruction.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <tuple>
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
// Bank 1 (the program's constants) reads `constants`.
Outputs execute(const ir::Program& p, const std::vector<std::uint32_t>& constants = {}) {
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
                    check(m.space == ir::AddressSpace::constant_buffer && (m.constant_buffer_slot == 3 || m.constant_buffer_slot == 1) &&
                          m.element_bytes == 4, "oracle memory contract");
                    const auto base = std::uint64_t(m.byte_offset) + (m.dynamic_byte_offset ? read(op.inputs.at(m.dynamic_byte_offset->value)) : 0);
                    for (unsigned component = 0; component < m.elements; ++component) {
                        const auto address = (base + component * m.element_bytes) & ~3ull;
                        if (m.constant_buffer_slot == 1) value = address / 4 < constants.size() ? constants[address / 4] : 0;
                        else value = address == 0 ? 0xdeadbeef : address == 16 ? 0x40000000 : address == 20 ? 0x3e800000 : 0;
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
        else if (auto multiway = std::get_if<ir::Switch>(&b.terminator)) {
            const auto selector = read(multiway->selector);
            const auto found = std::find_if(multiway->cases.begin(), multiway->cases.end(),
                                            [&](const ir::SwitchCase& c) { return c.bits == selector; });
            block = found != multiway->cases.end() ? found->target : multiway->fallback;
        }
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
    check(artifact.reflection.profile.version == 3, "numeric policy changes native artifact version");
    // Native vertex positions are invariant across programs.
    if (f.stage == sn::Stage::vertex)
        check(artifact.source.find("float4 position [[position, invariant]];") != std::string::npos, "native vertex position is invariant");
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
    // Instructions guarded by !PT never execute, whatever their opcode: the
    // MOV leaves R0 alone, an opcode the front end does not admit is not
    // refused, and the code stays one block.
    auto unadmitted=raw(0xfff); field(unadmitted,12,4,15);
    Fixture never({move(0,11,7),move(0,33,15),unadmitted,move(1,2,7),exit_instruction()}); put(never.code,0x48,3);
    const auto never_ir=lowered(never);
    check(execute(never_ir.ir).at({ir::IoSemantic::color,0,0})==11u && never_ir.ir.blocks.size()==1,"!PT instructions never execute and do not split blocks");
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
        auto out=execute(lowered(f).ir);
        check(out.at({ir::IoSemantic::color,0,0})==(mode==2?0xbfa00000u:0xc0200000u),"flat, linear and perspective IPA expected bits");
        compile_fixture(f,directory,("interpolation_"+std::to_string(mode)).c_str());
        if(mode==2)check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(f.program())),"perspective IPA needs an explicit PositionW adapter");
    }
    Fixture linear_centroid({ipa(0,0x80,false,true),exit_instruction()});linear_centroid.code[0x18]=std::byte{3};
    compile_fixture(linear_centroid,directory,"linear_centroid");
    linear_centroid.code[0x18]=std::byte{2};lowered(linear_centroid);
    Fixture absent({ipa(0,0x80),exit_instruction()});refused(absent);
    Fixture contradicted({ipa(0,0x80,true),exit_instruction()});contradicted.code[0x18]=std::byte{2};refused(contradicted);
    auto reciprocal=raw(0x308);field(reciprocal,16,8,2);field(reciprocal,32,8,1);field(reciprocal,74,6,4);
    auto multiply=raw(0x220);field(multiply,24,8,0);field(multiply,32,8,2);field(multiply,84,3,4);
    field(multiply,80,1,1);
    Fixture corrected({ipa(0,0x80),ipa(1,0x7c),reciprocal,multiply,exit_instruction()});
    corrected.code[0x18]=std::byte{2};corrected.code[0x17]=std::byte{0x80};
    // Plane value -1.25 (-2.5 times reconstructed 1/W 0.5) over RCP(PositionW 0.5) gives -2.5.
    check(execute(lowered(corrected).ir).at({ir::IoSemantic::color,0,0})==0xc0200000u,"perspective IPA reconstruction and shader correction");
    compile_fixture(corrected,directory,"perspective_correction");
    // At the centroid: the input and PositionW both read at the centroid
    // reconstruct with the pixel-centre W, so the shader's division gives the
    // same corrected value.
    Fixture centroid({ipa(0,0x80,false,true),ipa(1,0x7c,false,true),reciprocal,multiply,exit_instruction()});
    centroid.code[0x18]=std::byte{2};centroid.code[0x17]=std::byte{0x80};
    check(execute(lowered(centroid).ir).at({ir::IoSemantic::color,0,0})==0xc0200000u,"centroid perspective reconstruction");
    compile_fixture(centroid,directory,"perspective_centroid");
    // IPA indexed by a uniform register (0x526): byte address URa plus the
    // attribute's. With UR4 = 0x10 and attribute 0x88 (generic 0.z) the
    // export reads generic 1.z once SSA and cleanup fold the selection over
    // the enabled components; URZ reads the attribute itself; a register the
    // program loads selects at run time; the constant mode is refused.
    auto indexed_ipa = [](unsigned index, unsigned address, bool constant = false) {
        auto op = raw(0x526); field(op, 16, 8, 0); field(op, 24, 6, index); field(op, 32, 8, 255); field(op, 64, 8, address / 4);
        field(op, 78, 1, constant); field(op, 81, 3, 7); field(op, 91, 1, 1); return op;
    };
    auto umov = raw(0x882); field(umov, 16, 6, 4); field(umov, 32, 32, 0x10);
    auto surviving = [](const Fixture& f) {
        auto registers = sn::emit_native_ir(f.program());
        check(std::holds_alternative<ir::RegisterProgram>(registers), "indexed IPA lowers");
        auto ssa = ir::construct_ssa(std::get<ir::RegisterProgram>(std::move(registers)), {});
        check(std::holds_alternative<ir::SsaProgram>(ssa), "indexed IPA reaches SSA");
        auto& program = std::get<ir::SsaProgram>(ssa).ir;
        check(std::holds_alternative<std::monostate>(ir::passes::cleanup(program)), "indexed IPA cleanup");
        // The input load the color export reads.
        std::map<std::uint32_t, const ir::Operation*> definitions;
        const ir::Operation* exported = nullptr;
        for (const auto& block : program.blocks) for (auto id : block.operations) {
            const auto& op = program.operations[id.value];
            check(op.opcode != ir::Opcode::select, "a constant index leaves no selection");
            for (const auto& output : op.outputs) if (const auto* v = std::get_if<ir::ValueId>(&output)) definitions[v->value] = &op;
            if (op.opcode == ir::Opcode::store_io) exported = &op;
        }
        check(exported && std::holds_alternative<ir::ValueId>(exported->inputs.at(0)), "indexed IPA exports a value");
        const auto* load = definitions.at(std::get<ir::ValueId>(exported->inputs[0]).value);
        const auto* io = std::get_if<ir::IoInfo>(&load->info);
        check(load->opcode == ir::Opcode::load_io && io && io->semantic == ir::IoSemantic::generic, "indexed IPA reads a generic input");
        return std::vector<std::pair<unsigned, unsigned>>{{io->location, io->component_mask}};
    };
    Fixture indexed({umov, indexed_ipa(4, 0x88), exit_instruction()});
    indexed.code[0x18] = std::byte{0xff}; indexed.code[0x19] = std::byte{0xff};
    check(surviving(indexed) == std::vector<std::pair<unsigned, unsigned>>{{1, 4}}, "a constant index reads the component it addresses");
    Fixture direct({indexed_ipa(63, 0x84), exit_instruction()});
    direct.code[0x18] = std::byte{0xff};
    check(surviving(direct) == std::vector<std::pair<unsigned, unsigned>>{{0, 2}}, "URZ reads the attribute itself");
    auto uldc = raw(0xab9); field(uldc, 16, 6, 4); field(uldc, 73, 3, 4); field(uldc, 54, 5, 3); field(uldc, 38, 16, 0x10);
    Fixture dynamic({uldc, indexed_ipa(4, 0x80), exit_instruction()});
    dynamic.code[0x18] = std::byte{0xff}; dynamic.code[0x19] = std::byte{0x03};
    compile_fixture(dynamic, directory, "indexed_interpolation");
    Fixture constant_indexed({umov, indexed_ipa(4, 0x80, true), exit_instruction()});
    constant_indexed.code[0x18] = std::byte{0x55};
    refused(constant_indexed);
    // PositionZ and the point coordinate.
    Fixture depth({ipa(0,0x78),exit_instruction()}); depth.code[0x17]=std::byte{0x40};
    compile_fixture(depth,directory,"position_z");
    Fixture point({ipa(0,0x2e0),ipa(1,0x2e4),exit_instruction()}); put(point.code,0x48,3);
    check(sn::translate_native(point.program(),{ir::DepthConvention::negative_one_to_one,true,
        ir::FragmentPositionConvention::metal_position_w_matches_guest_position_w}).index()==0,"point coordinate input");
}
void global_memory_tests(const std::filesystem::path& directory) {
    // LDG.E.32 R0, [R2.64] with the common fields; each variant changes one.
    auto ldg = [](std::initializer_list<std::tuple<unsigned, unsigned, std::uint64_t>> changes, unsigned opcode = 0x381) {
        auto op = raw(opcode); field(op, 24, 8, 2); field(op, 72, 1, 1); field(op, 73, 3, 4); field(op, 77, 4, 5);
        field(op, 81, 3, 7); field(op, 84, 3, 1);
        if (opcode == 0x981) { field(op, 32, 6, 4); field(op, 91, 1, 1); }
        for (auto [first, count, value] : changes) field(op, first, count, value);
        return op;
    };
    auto program = [](sn::Instruction op) { return Fixture({move(2, 0x1000), move(3, 0x7d), op, exit_instruction()}); };
    // The address adds a uniform when a mutable local (a uniform register)
    // reaches either address word. A descriptor's uniform is not an address term.
    auto adds_uniform = [](const Fixture& fixture) {
        const auto p = lowered(fixture).ir;
        bool guest = false, adds = false;
        for (const auto& op : p.operations) {
            if (op.opcode != ir::Opcode::load && op.opcode != ir::Opcode::store && op.opcode != ir::Opcode::atomic) continue;
            const auto* info = std::get_if<ir::MemoryInfo>(&op.info);
            if (!info || info->space != ir::AddressSpace::guest || !info->dynamic_byte_offset || !info->global_address_high) continue;
            guest = true;
            std::vector<ir::Operand> pending{op.inputs.at(info->dynamic_byte_offset->value), op.inputs.at(info->global_address_high->value)};
            std::vector<char> seen(p.values.size());
            while (!pending.empty()) {
                const auto operand = pending.back(); pending.pop_back();
                const auto* id = std::get_if<ir::ValueId>(&operand);
                if (!id || id->value >= seen.size() || seen[id->value]) continue;
                seen[id->value] = 1;
                if (p.values[id->value].kind == ir::ValueKind::mutable_local) { adds = true; continue; }
                for (const auto& definition : p.operations) {
                    bool defines = false;
                    for (const auto& output : definition.outputs)
                        if (const auto* out = std::get_if<ir::ValueId>(&output); out && out->value == id->value) defines = true;
                    if (defines) pending.insert(pending.end(), definition.inputs.begin(), definition.inputs.end());
                }
            }
        }
        check(guest, "global memory operation");
        return adds;
    };
    auto inc_dec_loop = [](const Fixture& fixture) {
        const auto p = lowered(fixture).ir;
        bool cas = false, branch = false;
        for (const auto& op : p.operations) {
            if (op.opcode != ir::Opcode::atomic) continue;
            const auto* info = std::get_if<ir::MemoryInfo>(&op.info);
            if (info && info->atomic && *info->atomic == ir::AtomicKind::compare_exchange && info->space == ir::AddressSpace::guest) cas = true;
        }
        for (const auto& block : p.blocks) if (std::holds_alternative<ir::Branch>(block.terminator)) branch = true;
        check(cas && branch, "INC/DEC is a compare-and-swap loop");
    };
    compile_fixture(program(ldg({})), directory, "global_load");
    compile_fixture(program(ldg({}, 0x981)), directory, "global_uniform_load");
    check(adds_uniform(program(ldg({}, 0x981))), "a uniform global load adds the uniform register");
    compile_fixture(program(ldg({{76, 1, 1}}, 0x981)), directory, "global_descriptor_load");
    check(!adds_uniform(program(ldg({{76, 1, 1}}, 0x981))), "a descriptor-form load leaves the uniform register out of the address");
    refused(program(ldg({{72, 1, 0}})));                 // 32-bit register address
    refused(program(ldg({{24, 8, 3}})));                 // odd register pair
    refused(program(ldg({{73, 3, 7}})));                 // no type 7
    refused(program(ldg({{77, 4, 0xa}})));               // system-scope order
    refused(program(ldg({{81, 3, 0}})));                 // fault predicate written
    refused(program(ldg({{16, 8, 1}, {73, 3, 5}})));     // 64-bit load into an odd register
    refused(program(ldg({{64, 3, 1}}, 0x981)));          // predicated load
    refused(program(ldg({{72, 1, 0}}, 0x981)));          // 32-bit uniform base
    refused(program(ldg({{32, 6, 5}}, 0x981)));          // odd uniform pair
    // Descriptor form: an odd uniform index, a clear bit 72 and URZ are
    // admitted. The uniform register is not an address term.
    for (const auto [ur, bit72] : {std::pair{5u, 1u}, {4u, 0u}, {63u, 0u}}) {
        const auto op = ldg({{76, 1, 1}, {32, 6, ur}, {72, 1, bit72}}, 0x981);
        compile_fixture(program(op), directory, "global_descriptor_load_any_uniform");
        check(!adds_uniform(program(op)), "a descriptor-form load leaves the uniform register out of the address");
    }
    // Uniform store, atomic and reduction. Bit 76 (store) or 71 (atomic,
    // reduction) set is the descriptor form: the uniform register is not added.
    auto uniform_memory = [](unsigned opcode, bool descriptor) {
        auto op = raw(opcode);
        const bool store = opcode == 0x986, atom = opcode == 0x9a8;
        field(op, 24, 8, 2); field(op, 32, 8, 4); field(op, 64, 6, 4); field(op, 72, 1, 1); field(op, 84, 3, 1); field(op, 91, 1, 1);
        if (store) { field(op, 73, 3, 4); field(op, 76, 1, descriptor); field(op, 77, 4, 5); field(op, 90, 1, 1); }
        else {
            field(op, 77, 4, 7); field(op, 71, 1, descriptor);
            if (atom) { field(op, 16, 8, 0); field(op, 70, 1, 1); field(op, 81, 3, 7); field(op, 87, 4, 0); }
            else { field(op, 87, 3, 2); field(op, 90, 1, 1); }
        }
        return op;
    };
    compile_fixture(program(uniform_memory(0x986, false)), directory, "global_uniform_store");
    check(adds_uniform(program(uniform_memory(0x986, false))), "a uniform global store adds the uniform register");
    compile_fixture(program(uniform_memory(0x986, true)), directory, "global_descriptor_store");
    check(!adds_uniform(program(uniform_memory(0x986, true))), "a descriptor-form store leaves the uniform register out of the address");
    compile_fixture(program(uniform_memory(0x9a8, false)), directory, "global_uniform_atomic");
    check(adds_uniform(program(uniform_memory(0x9a8, false))), "a uniform global atomic adds the uniform register");
    compile_fixture(program(uniform_memory(0x9a8, true)), directory, "global_descriptor_atomic");
    check(!adds_uniform(program(uniform_memory(0x9a8, true))), "a descriptor-form atomic leaves the uniform register out of the address");
    compile_fixture(program(uniform_memory(0x98e, false)), directory, "global_uniform_reduction");
    check(adds_uniform(program(uniform_memory(0x98e, false))), "a uniform global reduction adds the uniform register");
    compile_fixture(program(uniform_memory(0x98e, true)), directory, "global_descriptor_reduction");
    check(!adds_uniform(program(uniform_memory(0x98e, true))), "a descriptor-form reduction leaves the uniform register out of the address");
    // The same descriptor admission for STG, ATOMG and RED: odd index, bit 72
    // clear, and URZ. The non-descriptor form still refuses both.
    for (const unsigned opcode : {0x986u, 0x9a8u, 0x98eu}) {
        for (const auto [ur, bit72] : {std::pair{5u, 1u}, {4u, 0u}, {63u, 0u}}) {
            auto op = uniform_memory(opcode, true);
            field(op, 64, 6, ur); field(op, 72, 1, bit72);
            compile_fixture(program(op), directory, "global_descriptor_any_uniform");
            check(!adds_uniform(program(op)), "a descriptor form leaves the uniform register out of the address");
        }
        auto odd = uniform_memory(opcode, false); field(odd, 64, 6, 5); refused(program(odd));
        auto narrow = uniform_memory(opcode, false); field(narrow, 72, 1, 0); refused(program(narrow));
    }
    auto red = raw(0x98e); field(red, 24, 8, 2); field(red, 32, 8, 0); field(red, 72, 1, 1);
    field(red, 77, 4, 7); field(red, 84, 3, 1); field(red, 87, 3, 2);
    compile_fixture(program(red), directory, "global_reduction");
    // F32 operations other than ADD stay refused. 32-bit INC and DEC translate.
    // 64-bit INC and DEC, and other sizes, stay refused.
    for (auto [type, operation] : {std::pair{3u, 1u}, std::pair{3u, 3u}, std::pair{3u, 4u}, std::pair{3u, 6u}, std::pair{2u, 0u}, std::pair{4u, 3u}}) {
        auto red = raw(0x98e); field(red, 24, 8, 2); field(red, 32, 8, 0); field(red, 72, 1, 1);
        field(red, 73, 3, type); field(red, 77, 4, 7); field(red, 84, 3, 1); field(red, 87, 3, operation);
        refused(program(red));
    }
    auto atomic = [](unsigned opcode, unsigned type, unsigned operation) {
        auto op = raw(opcode);
        const bool reduce = opcode == 0x98e;
        if (!reduce) field(op, 16, 8, 0);
        field(op, 24, 8, 2); field(op, 32, 8, 4); field(op, 72, 1, 1); field(op, 73, 3, type);
        field(op, 77, 4, 7); field(op, 84, 3, 1); field(op, 87, reduce ? 3 : 4, operation);
        if (!reduce) field(op, 81, 3, 7);
        return op;
    };
    for (const unsigned opcode : {0x3a8u, 0x98eu}) for (const unsigned operation : {3u, 4u}) {
        compile_fixture(program(atomic(opcode, 0, operation)), directory, opcode == 0x3a8 ? (operation == 3 ? "global_increment" : "global_decrement")
                                                                                            : (operation == 3 ? "global_reduction_increment" : "global_reduction_decrement"));
        inc_dec_loop(program(atomic(opcode, operation == 3 ? 0u : 1u, operation)));
        compile_fixture(program(atomic(opcode, 1, operation)), directory, opcode == 0x3a8 ? (operation == 3 ? "global_increment_signed" : "global_decrement_signed")
                                                                                            : (operation == 3 ? "global_reduction_increment_signed" : "global_reduction_decrement_signed"));
        for (const unsigned type : {2u, 5u}) refused(program(atomic(opcode, type, operation)));
    }
    auto float_add = raw(0x98e); field(float_add, 24, 8, 2); field(float_add, 32, 8, 0); field(float_add, 72, 1, 1);
    field(float_add, 73, 3, 3); field(float_add, 77, 4, 7); field(float_add, 84, 3, 1);
    compile_fixture(program(float_add), directory, "global_float_reduction");
}
void jump_table_tests() {
    // BRX R6 at instruction 1 with base 0; the table holds instruction 2's offset.
    auto brx = raw(0x949); field(brx, 24, 8, 6); field(brx, 32, 50, std::uint64_t(-32) & ((1ull << 50) - 1)); field(brx, 87, 4, 7);
    auto ldc = raw(0xb82); field(ldc, 16, 8, 6); field(ldc, 24, 8, 2); field(ldc, 54, 5, 1); field(ldc, 73, 3, 4);
    Fixture loaded({ldc, brx, exit_instruction()}, sn::Stage::fragment, {32});
    lowered(loaded);
    Fixture moved({move(6, 32), brx, exit_instruction()}, sn::Stage::fragment, {32});
    refused(moved);   // the register does not come from a bank-1 table load
    Fixture other_bank({[&] { auto l = ldc; field(l, 54, 5, 3); return l; }(), brx, exit_instruction()}, sn::Stage::fragment, {32});
    refused(other_bank);
    Fixture outside({ldc, brx, exit_instruction()}, sn::Stage::fragment, {48});
    refused(outside); // no entry is an instruction of the program
    // Texel offsets are refused on a cube sample, and unpacked operands on
    // a gather.
    auto texture = [](unsigned opcode, unsigned dimension, bool packed, bool offsets) {
        auto t = raw(opcode); field(t, 24, 8, 8); field(t, 32, 8, 10); field(t, 40, 14, 0x10); field(t, 54, 5, 3);
        field(t, 60, 1, packed); field(t, 61, 3, dimension); field(t, 64, 8, 255); field(t, 72, 4, 1); field(t, 76, 1, offsets);
        field(t, 81, 3, 7); return t;
    };
    refused(Fixture({texture(0xb60, 3, true, true), exit_instruction()}, sn::Stage::fragment));
    refused(Fixture({texture(0xb63, 1, false, false), exit_instruction()}, sn::Stage::fragment));
    // Gather offsets: mode 3, a cube, and per-texel offsets with comparison
    // or fewer than four components are refused.
    auto gather = [&](unsigned dimension, unsigned mode, unsigned components, bool compare) {
        auto t = texture(0xb63, dimension, true, false); field(t, 76, 2, mode); field(t, 72, 4, components); field(t, 78, 1, compare);
        field(t, 64, 8, 2); return t;
    };
    refused(Fixture({gather(1, 3, 15, false), exit_instruction()}, sn::Stage::fragment));
    refused(Fixture({gather(3, 1, 15, false), exit_instruction()}, sn::Stage::fragment));
    refused(Fixture({gather(1, 2, 15, true), exit_instruction()}, sn::Stage::fragment));
    refused(Fixture({gather(1, 2, 3, false), exit_instruction()}, sn::Stage::fragment));
    // BRXU UR6 takes the table an indexed ULDC (UR6 = c[1][UR2]) loads; an
    // ordinary instruction writing R6 between them does not hide it, a
    // uniform one writing UR6 does, and BRXU needs its form bit.
    auto brxu = raw(0x958); field(brxu, 24, 6, 6); field(brxu, 32, 50, std::uint64_t(-48) & ((1ull << 50) - 1));
    field(brxu, 87, 4, 7); field(brxu, 91, 1, 1);
    auto uldc = raw(0xabb); field(uldc, 16, 6, 6); field(uldc, 24, 6, 2); field(uldc, 54, 5, 1); field(uldc, 73, 3, 4); field(uldc, 91, 1, 1);
    auto umov = raw(0x882); field(umov, 16, 6, 6); field(umov, 32, 32, 48);
    Fixture uniform_loaded({uldc, move(6, 7), brxu, exit_instruction()}, sn::Stage::fragment, {48});
    lowered(uniform_loaded);
    Fixture uniform_moved({uldc, umov, brxu, exit_instruction()}, sn::Stage::fragment, {48});
    refused(uniform_moved);
    Fixture without_form({uldc, move(6, 7), [&] { auto b = brxu; field(b, 91, 1, 0); return b; }(), exit_instruction()},
                         sn::Stage::fragment, {48});
    refused(without_form);
    // The high half of a 64-bit ULDC (UR5:UR6) is the selector; a later
    // 32-bit write of UR6 hides the pair.
    auto pair = uldc; field(pair, 16, 6, 5); field(pair, 73, 3, 5);
    Fixture high_half({pair, move(7, 0), brxu, exit_instruction()}, sn::Stage::fragment, {0, 48});
    lowered(high_half);
    Fixture hidden_pair({pair, umov, brxu, exit_instruction()}, sn::Stage::fragment, {0, 48});
    refused(hidden_pair);
    // A guarded CALL over a backward BRA is a loop exit unless its next
    // offset is moved into a register just before it: then it is a call,
    // whose RET at 5 returns to the BRA (4), which leads to the EXIT (1).
    // Any other value before it leaves the loop exit, and the RET outside a
    // call is refused.
    auto to = [](unsigned from, unsigned target) {
        auto b = raw(0x947); field(b, 32, 50, (std::uint64_t(target) * 16 - std::uint64_t(from) * 16 - 16) & ((1ull << 50) - 1)); field(b, 87, 4, 7); return b;
    };
    auto guarded_call = raw(0x944); field(guarded_call, 12, 4, 0); field(guarded_call, 32, 50, 16); field(guarded_call, 86, 1, 1); field(guarded_call, 87, 4, 7);
    auto return_at_5 = raw(0x950); field(return_at_5, 24, 8, 10); field(return_at_5, 32, 50, std::uint64_t(-96) & ((1ull << 50) - 1));
    field(return_at_5, 86, 1, 1); field(return_at_5, 87, 4, 7);
    Fixture linked({to(0, 2), exit_instruction(), move(10, 0x40), guarded_call, to(4, 1), return_at_5});
    lowered(linked);
    Fixture unlinked({to(0, 2), exit_instruction(), move(10, 0x50), guarded_call, to(4, 1), return_at_5});
    refused(unlinked);
    // Instructions that never execute (!PT) leave table discovery alone. The
    // live LDC at 0 loads the selector from table A (offset 0: entries 48 and
    // 80, instructions 3 and 5, which export 11 and 22); table B at offset 8
    // holds 80. Slot 1 holds a live filler or a padding instruction, slot 8
    // a filler or a padding BRX whose own LDC (slot 7) names offset 4.
    auto ldc_at = [](unsigned reg, unsigned offset, unsigned guard) {
        auto l = raw(0xb82); field(l, 12, 4, guard); field(l, 16, 8, reg); field(l, 24, 8, 2); field(l, 38, 16, offset);
        field(l, 54, 5, 1); field(l, 73, 3, 4); return l;
    };
    auto brx_at = [](unsigned index, unsigned reg, unsigned guard) {
        auto b = raw(0x949); field(b, 12, 4, guard); field(b, 24, 8, reg);
        field(b, 32, 50, (0 - std::uint64_t(index * 16 + 16)) & ((1ull << 50) - 1)); field(b, 87, 4, 7); return b;
    };
    auto bra = raw(0x947); field(bra, 12, 4, 15); field(bra, 32, 50, 0x10); field(bra, 87, 4, 7);
    struct Outcome { std::uint32_t color; std::vector<std::uint32_t> cases; };
    auto table_run = [&](sn::Instruction slot1, sn::Instruction slot8) {
        Fixture f({ldc_at(6, 0, 7), slot1, brx_at(2, 6, 7), move(0, 11), exit_instruction(), move(0, 22), exit_instruction(),
                   ldc_at(8, 4, 7), slot8, exit_instruction()}, sn::Stage::fragment, {48, 80, 80});
        put(f.code, 0x48, 1);
        const auto program = lowered(f);
        Outcome outcome{execute(program.ir, {48, 80, 80}).at({ir::IoSemantic::color, 0, 0}), {}};
        for (const auto& block : program.ir.blocks)
            if (const auto* multiway = std::get_if<ir::Switch>(&block.terminator))
                for (const auto& c : multiway->cases) outcome.cases.push_back(c.bits);
        return outcome;
    };
    const auto filler = move(7, 0);
    const auto live = table_run(filler, filler);
    check(live.color == 11 && live.cases == std::vector<std::uint32_t>{48, 80}, "live jump table baseline");
    for (const auto& [slot1, slot8, name] : {
             std::tuple{ldc_at(6, 8, 15), filler, "a !PT LDC of the selector"},
             std::tuple{move(6, 80, 15), filler, "a !PT write of the selector"},
             std::tuple{bra, filler, "a !PT branch between the LDC and the BRX"},
             std::tuple{filler, brx_at(8, 8, 15), "a !PT BRX naming a table inside the live one"}}) {
        const auto padded = table_run(slot1, slot8);
        check(padded.color == live.color && padded.cases == live.cases, name);
    }
    // A guard that may hold (!P0) is not padding: that LDC stays the
    // selector's last writer, so the BRX takes table B.
    check(table_run(ldc_at(6, 8, 8), filler).cases == std::vector<std::uint32_t>{80}, "a !P0 LDC is not padding");
    // CALL at 1 into a RET R10 at 3 that returns to the EXIT at 2: R10 must
    // hold that return offset as a literal; a constant-buffer value is
    // refused once SSA shows where it comes from, and a RET outside any call
    // is refused.
    auto call = raw(0x944); field(call, 32, 50, 0x10); field(call, 86, 1, 1); field(call, 87, 4, 7);
    auto ret = raw(0x950); field(ret, 24, 8, 10); field(ret, 32, 50, std::uint64_t(-64) & ((1ull << 50) - 1)); field(ret, 86, 1, 1); field(ret, 87, 4, 7);
    auto unknown = raw(0xb82); field(unknown, 16, 8, 10); field(unknown, 24, 8, 255); field(unknown, 54, 5, 3); field(unknown, 73, 3, 4);
    Fixture known({move(10, 0x20), call, exit_instruction(), ret});
    lowered(known);
    Fixture top_level({move(10, 0x20), ret, exit_instruction()});
    refused(top_level);
    Fixture loaded_return({unknown, call, exit_instruction(), ret});
    check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(loaded_return.program(),
        {ir::DepthConvention::negative_one_to_one, true, ir::FragmentPositionConvention::metal_position_w_matches_guest_position_w})),
        "a return register that is not a call return literal is refused");
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
        // Toward minus: refused for products, the native nearest sum for FADD.
        field(op, 78, 2, 1); Fixture rounding({move(1, 0), move(2, 0), move(3, 0), op, exit_instruction()});
        if (multiply || fma) refused(rounding); else lowered(rounding);
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
        check(result_bits(f) == (function == 4 ? 0x3e800000u : function == 5 ? 0x3f000000u : 0x40000000u), "flushing multifunction results");
        compile_fixture(f, directory, ("multifunction_" + std::to_string(function)).c_str());
        field(op, 73, 1, 1); Fixture half({move(1, 0), op, exit_instruction()}); refused(half);
    }
    for (unsigned function : {6u, 7u, 9u}) {
        auto op = raw(0x308); field(op, 32, 8, 1); field(op, 74, 6, function);
        Fixture f({move(1, 0x40800000), op, exit_instruction()}); refused(f);
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
// Task and mesh stages keep their stage buffer in workgroup memory
// (stage_memory.h): each admitted ISBERD and ISBEWR form addresses its region,
// a count store of RZ is dropped, and other forms are refused.
sn::Instruction isbe(unsigned opcode, unsigned vector, unsigned ra, std::uint32_t offset, unsigned mode) {
    auto i = raw(opcode); field(i, opcode == 0x927 ? 32 : 16, 8, vector); field(i, 24, 8, ra); field(i, 40, 24, offset); field(i, 72, 8, mode);
    return i;
}
// 32 threads and a 64-byte payload; a mesh stage with a task stage, triangles,
// 32 vertices of a position (header 0x35 = 0xf0) and 32 primitives.
Fixture workgroup(std::vector<sn::Instruction> code, sn::Stage stage) {
    Fixture f(std::move(code), stage);
    put(f.control, 0x58, 32 | (0x40u << 16));
    if (stage == sn::Stage::mesh) { put(f.control, 0x5c, 0x03010000); put(f.control, 0x60, 32 | (32u << 16)); }
    return f;
}
// The literal workgroup-memory offset and element bytes of each shared access, in program order.
std::vector<std::pair<std::uint32_t, unsigned>> shared_accesses(const ir::Program& p) {
    std::vector<std::pair<std::uint32_t, unsigned>> out;
    for (const auto& op : p.operations) {
        if (op.opcode != ir::Opcode::load && op.opcode != ir::Opcode::store) continue;
        const auto& m = std::get<ir::MemoryInfo>(op.info);
        check(m.space == ir::AddressSpace::shared && m.dynamic_byte_offset, "stage buffer access is a shared access");
        const auto* literal = std::get_if<ir::LiteralBits>(&op.inputs.at(m.dynamic_byte_offset->value));
        out.push_back({literal ? std::uint32_t(literal->bits) : ~0u, m.element_bytes});
    }
    return out;
}
void stage_buffer_tests() {
    using Accesses = std::vector<std::pair<std::uint32_t, unsigned>>;
    {
        // A program's spans point into its fixture, which stays alive.
        const auto mesh = workgroup({exit_instruction()}, sn::Stage::mesh);
        const auto layout = std::get<sn::StageMemory>(sn::stage_memory(mesh.program()));
        check(layout.input == 0x10 && layout.output == 0x50 && layout.vertex.size() == 4 && layout.primitive.empty(), "mesh stage buffer regions");
        // One block of 32 vertices of 4 components, then 4 + 3 * 32 index bytes.
        check(layout.primitives == 0x50 + 4 * 128 && layout.bytes == layout.primitives + 100, "mesh primitive bytes follow the attribute blocks");
        const auto task = workgroup({exit_instruction()}, sn::Stage::task);
        const auto task_layout = std::get<sn::StageMemory>(sn::stage_memory(task.program()));
        check(task_layout.output == 0x50 && task_layout.bytes == 0x90, "task stage buffer holds its payload");
        auto large = workgroup({exit_instruction()}, sn::Stage::mesh);
        put(large.control, 0x60, 257 | (32u << 16));
        check(std::holds_alternative<std::string>(sn::stage_memory(large.program())), "a mesh beyond Metal's vertex limit is refused");
    }
    auto count = isbe(0x927, 3, 255, 3, 0x88); field(count, 12, 4, 0);  // @P0
    const Fixture mesh = workgroup({isbe(0x923, 1, 255, 0, 0x78), isbe(0x923, 2, 255, 8, 0x38), isbe(0x927, 1, 255, 0x10, 0xf8),
                                    isbe(0x927, 255, 255, 3, 0x88), count, isbe(0x927, 2, 255, 4, 0x80), isbe(0x927, 2, 7, 0, 0xf8),
                                    exit_instruction()}, sn::Stage::mesh);
    const auto accesses = shared_accesses(lowered(mesh).ir);
    check(accesses == Accesses{{0, 4}, {0x18, 4}, {0x60, 4}, {0x253, 1}, {0x254, 1}, {~0u, 4}}, "mesh stage buffer accesses");
    // A mesh stage's position takes the vertex stage's clip convention, a caller option.
    check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(mesh.program())), "a mesh stage without a clip convention is refused");
    sn::FrontendOptions clip; clip.vertex_depth = ir::DepthConvention::negative_one_to_one;
    const auto mesh_msl = sn::translate_native(mesh.program(), clip);
    check(std::holds_alternative<ir::MslArtifact>(mesh_msl) && std::get<ir::MslArtifact>(mesh_msl).entry_point == "armigo_mesh_main",
          "mesh stage translates to a mesh function");
    const Fixture task = workgroup({isbe(0x923, 1, 255, 0, 0x78), isbe(0x927, 1, 255, 4, 0xb8), isbe(0x923, 2, 255, 4, 0xb8), exit_instruction()},
                                   sn::Stage::task);
    check(shared_accesses(lowered(task).ir) == Accesses{{0, 4}, {0x54, 4}, {0x54, 4}}, "task stage buffer accesses");
    const auto task_msl = sn::translate_native(task.program());
    check(std::holds_alternative<ir::MslArtifact>(task_msl) && std::get<ir::MslArtifact>(task_msl).entry_point == "armigo_object_main",
          "task stage translates to an object function");
    refused(workgroup({isbe(0x923, 1, 255, 0, 0x38), exit_instruction()}, sn::Stage::task));   // a task stage has no input
    refused(workgroup({isbe(0x927, 1, 255, 0, 0xf8), exit_instruction()}, sn::Stage::task));   // nor attributes
    refused(workgroup({isbe(0x927, 1, 255, 0, 0xb8), exit_instruction()}, sn::Stage::mesh));   // a mesh stage has no payload
    refused(workgroup({isbe(0x923, 1, 4, 0, 0x78), exit_instruction()}, sn::Stage::mesh));     // workgroup read with an address
    refused(workgroup({isbe(0x923, 1, 255, 0, 0x40), exit_instruction()}, sn::Stage::mesh));   // unknown mode
    auto reserved = isbe(0x927, 1, 255, 0, 0xf8); field(reserved, 64, 8, 1);
    refused(workgroup({reserved, exit_instruction()}, sn::Stage::mesh));
    refused(Fixture({isbe(0x923, 1, 255, 0, 0x78), exit_instruction()}, sn::Stage::compute)); // no stage buffer
    // Word accesses drop the address's low two bits; byte accesses keep it.
    {
        std::vector<sn::Instruction> code;
        for (std::uint32_t low = 0; low < 4; ++low) code.push_back(isbe(0x927, 1, 255, 0x10 + low, 0xf8));
        code.push_back(isbe(0x927, 1, 255, 5, 0x80)); code.push_back(exit_instruction());
        const Fixture unaligned = workgroup(code, sn::Stage::mesh);
        check(shared_accesses(lowered(unaligned).ir) == Accesses{{0x60, 4}, {0x60, 4}, {0x60, 4}, {0x60, 4}, {0x255, 1}},
              "unaligned immediate word addresses");
    }
    // A count store of RZ rests on one warp: refused in larger workgroups.
    {
        auto wide = workgroup({isbe(0x927, 255, 255, 3, 0x88), exit_instruction()}, sn::Stage::mesh);
        put(wide.control, 0x58, 64 | (0x40u << 16)); refused(wide);
        auto counted = workgroup({isbe(0x927, 1, 255, 3, 0x88), exit_instruction()}, sn::Stage::mesh);
        put(counted.control, 0x58, 64 | (0x40u << 16)); lowered(counted);
    }
    // A task payload too short for the launch count still has its two words.
    for (const std::uint32_t payload : {0u, 1u, 4u, 5u, 8u}) {
        auto task = workgroup({exit_instruction()}, sn::Stage::task);
        put(task.control, 0x58, 32 | (payload << 16));
        const auto layout = std::get<sn::StageMemory>(sn::stage_memory(task.program()));
        check(layout.bytes >= layout.output + 8 && layout.bytes >= layout.output + ((payload + 3) & ~3u), "task payload words");
        check(std::holds_alternative<ir::MslArtifact>(sn::translate_native(task.program())), "a short task payload translates");
    }
    // Metal counts a mesh's declared outputs against 32 KB: 256 vertices of a
    // position and 28 generic components fit exactly; 29 do not, although the
    // stage buffer fits.
    for (const unsigned generics : {28u, 29u}) {
        auto wide = workgroup({exit_instruction()}, sn::Stage::mesh);
        put(wide.control, 0x58, 32); put(wide.control, 0x5c, 0x03000000); put(wide.control, 0x60, 256 | (1u << 16));
        wide.code[0x35] = std::byte{0x10};
        for (unsigned n = 0; n < generics; ++n) wide.code[0x36 + n / 2] |= std::byte(1u << (4 * (n % 2)));
        check(std::holds_alternative<sn::StageMemory>(sn::stage_memory(wide.program())), "the stage buffer fits");
        sn::FrontendOptions options; options.vertex_depth = ir::DepthConvention::zero_to_one;
        check(std::holds_alternative<ir::MslArtifact>(sn::translate_native(wide.program(), options)) == (generics == 28),
              "mesh output limit");
    }
    // Workgroup memory of their own would lie over the stage buffer.
    auto lds = raw(0x984); field(lds, 16, 8, 1); field(lds, 24, 8, 255); field(lds, 73, 3, 4);
    refused(workgroup({lds, exit_instruction()}, sn::Stage::mesh));
}
// A bindless TEX whose handle is the guest word at c3[0x10]:c3[0x14] +
// c3[0x18] * 32 + 0x20 (LDG): the same in every invocation, so the host
// computes it (pipeline/handle_program.h) and the texture is keyed at
// handle_program_slot. A handle read at a per-invocation address stays
// refused.
void handle_program_tests() {
    auto cword = [](unsigned d, unsigned offset) {
        auto i = raw(0xa02); field(i, 16, 8, d); field(i, 54, 5, 3); field(i, 38, 16, offset); field(i, 64, 32, 0xf00); return i;
    };
    auto imad = [](unsigned d, unsigned a, unsigned b, unsigned c) {
        auto i = raw(0x224); field(i, 16, 8, d); field(i, 24, 8, a); field(i, 32, 8, b); field(i, 64, 8, c);
        field(i, 81, 3, 7); field(i, 87, 4, 15); return i;
    };
    auto ldg = [](unsigned d, unsigned address, unsigned displacement) {
        auto i = raw(0x381); field(i, 16, 8, d); field(i, 24, 8, address); field(i, 40, 24, displacement); field(i, 72, 1, 1);
        field(i, 73, 3, 4); field(i, 77, 4, 5); field(i, 81, 3, 7); field(i, 84, 3, 1); return i;
    };
    // TEX.B R0, R8, R12 (handle register after the coordinates), 2D, red and green.
    auto tex = raw(0x361); field(tex, 16, 8, 0); field(tex, 24, 8, 8); field(tex, 32, 8, 12); field(tex, 59, 1, 1); field(tex, 60, 1, 1);
    field(tex, 61, 3, 1); field(tex, 64, 8, 255); field(tex, 72, 4, 3); field(tex, 81, 3, 7);
    sn::FrontendOptions options;
    options.fragment_position = ir::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    {
        Fixture uniform({cword(2, 0x10), cword(3, 0x14), cword(4, 0x18), move(5, 32), imad(2, 4, 5, 2), ldg(12, 2, 0x20),
                         move(8, 0x3f000000), move(9, 0x3f000000), tex, exit_instruction()});
        put(uniform.code, 0x48, 0x3);
        auto translated = sn::translate_native(uniform.program(), options);
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        const auto& artifact = std::get<ir::MslArtifact>(translated);
        check(artifact.reflection.handle_programs.size() == 1, "one handle program");
        bool keyed = false;
        for (const auto& resource : artifact.reflection.resources)
            if (const auto* texture = std::get_if<ir::TextureUse>(&resource.use))
                keyed |= texture->key == ir::TextureKey{ir::handle_program_slot, 0};
        check(keyed, "the texture is keyed at the handle program slot");
        std::uint64_t read_at = 0;
        const auto handle = ir::evaluate_handle_program(artifact.reflection.handle_programs[0],
            [](std::uint32_t slot, std::uint32_t offset) -> std::uint32_t {
                if (slot != 3) return 0;
                return offset == 0x10 ? 0xfff00000u : offset == 0x14 ? 0x7du : offset == 0x18 ? 0x10u : 0u;
            },
            [&](std::uint64_t address) { read_at = address; return 0x00500123u; });
        check(handle == 0x00500123u, "the program reads the handle word");
        // 0xfff00000 + 16 * 32 + 0x20 without a carry into the high word, as IMAD's low word.
        check(read_at == ((0x7dull << 32) | (0xfff00000u + 0x200u + 0x20u)), "the program reads the handle at the computed address");
    }
    {
        auto lane = raw(0x919); field(lane, 16, 8, 2); field(lane, 72, 8, 0);
        Fixture varying({lane, move(3, 0x7d), ldg(12, 2, 0x20), move(8, 0x3f000000), move(9, 0x3f000000), tex, exit_instruction()});
        put(varying.code, 0x48, 0x3);
        check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(varying.program(), options)),
              "a handle read at a per-invocation address stays refused");
        // With pool tables bound (the project documentation) the same
        // handle reads the texture and sampler pools by its IDs, and the
        // marking fragment stage keeps early fragment tests.
        auto pooled = options;
        pooled.texture_pools = true;
        auto translated = sn::translate_native(varying.program(), pooled);
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        const auto& artifact = std::get<ir::MslArtifact>(translated);
        unsigned texture_pools = 0, sampler_pools = 0;
        for (const auto& resource : artifact.reflection.resources) {
            const auto* range = std::get_if<ir::MetalRange>(&resource.location);
            if (const auto* texture = std::get_if<ir::TextureUse>(&resource.use); texture && texture->pool)
                texture_pools += texture->key == ir::TextureKey{ir::texture_pool_slot, 0} && range && range->space == ir::MetalNamespace::buffer;
            if (const auto* sampler = std::get_if<ir::SamplerUse>(&resource.use); sampler && sampler->pool)
                sampler_pools += sampler->key == ir::TextureKey{ir::texture_pool_slot, 0} && range && range->space == ir::MetalNamespace::buffer;
        }
        check(texture_pools == 1 && sampler_pools == 1, "one texture pool and one sampler pool in the buffer namespace");
        check(artifact.source.find("armigo_pool_read(") != std::string::npos && artifact.source.find(" & 0xfffffu)") != std::string::npos &&
              artifact.source.find(") >> 20u, ") != std::string::npos, "the texture and sampler IDs index the pools");
        check(artifact.source.find("[[early_fragment_tests]] fragment ") != std::string::npos, "a marking stage keeps early fragment tests");
        auto kill = raw(0x95b); field(kill, 87, 4, 7);
        Fixture discarding({lane, move(3, 0x7d), ldg(12, 2, 0x20), move(8, 0x3f000000), move(9, 0x3f000000), tex, kill, exit_instruction()});
        put(discarding.code, 0x48, 0x3);
        translated = sn::translate_native(discarding.program(), pooled);
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        check(std::get<ir::MslArtifact>(translated).source.find("early_fragment_tests") == std::string::npos,
              "a discarding stage keeps its late depth test");
        // A guest store and a sample-mask write are the stage's own, so the
        // mark must not pull the depth test forward.
        auto stg = raw(0x386); field(stg, 24, 8, 2); field(stg, 32, 8, 13); field(stg, 40, 24, 0x20); field(stg, 72, 1, 1);
        field(stg, 73, 3, 4); field(stg, 77, 4, 5); field(stg, 84, 3, 1);
        Fixture stored({lane, move(3, 0x7d), ldg(12, 2, 0x20), move(8, 0x3f000000), move(9, 0x3f000000), move(13, 1), tex, stg, exit_instruction()});
        put(stored.code, 0x48, 0x3);
        translated = sn::translate_native(stored.program(), pooled);
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        check(std::get<ir::MslArtifact>(translated).source.find("early_fragment_tests") == std::string::npos,
              "a guest store keeps the late depth test");
        Fixture masked({lane, move(3, 0x7d), ldg(12, 2, 0x20), move(8, 0x3f000000), move(9, 0x3f000000), move(4, 1), tex, exit_instruction()});
        put(masked.code, 0x48, 0x3);
        masked.code[0x4c] = std::byte{1};
        translated = sn::translate_native(masked.program(), pooled);
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        const auto& masked_source = std::get<ir::MslArtifact>(translated).source;
        check(masked_source.find("sample_mask") != std::string::npos && masked_source.find("early_fragment_tests") == std::string::npos,
              "a sample-mask write keeps the late depth test");
        auto bad = tex; field(bad, 61, 3, 6);
        Fixture selector({move(8, 0x3f000000), move(9, 0x3f000000), move(12, 1), bad, exit_instruction()});
        put(selector.code, 0x48, 0x3);
        const auto selector_result = sn::translate_native(selector.program(), pooled);
        const auto* selector_error = std::get_if<ir::Diagnostic>(&selector_result);
        check(selector_error && selector_error->message.find("texture dimension selector is not admitted") != std::string::npos,
              "an unadmitted texture dimension stays refused with pools");
    }
    // A stage that writes guest memory takes no handle program: a store to
    // the handle's word before the load (or a reduction anywhere) could
    // change it after the draw starts, so the handle stays refused.
    {
        auto stg = raw(0x386); field(stg, 24, 8, 2); field(stg, 32, 8, 13); field(stg, 40, 24, 0x20); field(stg, 72, 1, 1);
        field(stg, 73, 3, 4); field(stg, 77, 4, 5); field(stg, 84, 3, 1);
        auto red = raw(0x98e); field(red, 24, 8, 2); field(red, 32, 8, 13); field(red, 72, 1, 1);
        field(red, 77, 4, 7); field(red, 84, 3, 1); field(red, 87, 3, 2);
        for (const auto& write : {stg, red}) {
            Fixture written({cword(2, 0x10), cword(3, 0x14), move(13, 0x00100005), write, ldg(12, 2, 0x20),
                             move(8, 0x3f000000), move(9, 0x3f000000), tex, exit_instruction()});
            put(written.code, 0x48, 0x3);
            check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(written.program(), options)),
                  "a stage writing guest memory takes no handle program");
        }
    }
    // The pass charges the analysis budget and keeps at most the resource
    // limit of programs; equal programs share the first one's key.
    {
        auto tex_at = [&](unsigned handle) { auto t = tex; field(t, 32, 8, handle); field(t, 16, 8, 0); return t; };
        Fixture three({cword(2, 0x10), cword(3, 0x14), ldg(12, 2, 0x20), ldg(14, 2, 0x24), ldg(16, 2, 0x20),
                       move(8, 0x3f000000), move(9, 0x3f000000), tex_at(12), tex_at(14), tex_at(16), exit_instruction()});
        put(three.code, 0x48, 0x3);
        auto programs = [&](std::uint64_t steps, std::uint32_t resources) -> std::optional<std::vector<std::vector<std::uint32_t>>> {
            auto registers = sn::emit_native_ir(three.program());
            check(std::holds_alternative<ir::RegisterProgram>(registers), "handle limit fixture lowers");
            auto ssa = ir::construct_ssa(std::get<ir::RegisterProgram>(std::move(registers)), {});
            check(std::holds_alternative<ir::SsaProgram>(ssa), "handle limit fixture reaches SSA");
            auto& ir_program = std::get<ir::SsaProgram>(ssa).ir;
            ir_program.limits.analysis_steps = steps; ir_program.limits.resources = resources;
            try {
                auto built = sn::detail::build_handle_programs(ir_program);
                std::vector<std::uint32_t> keys;
                for (const auto& o : ir_program.operations)
                    if (const auto* t = std::get_if<ir::TextureInfo>(&o.info))
                        if (const auto* key = std::get_if<ir::TextureKey>(&t->texture)) keys.push_back(key->handle_byte_offset);
                check(keys == std::vector<std::uint32_t>{0, 4, 0}, "equal handle programs share the first key");
                return built;
            } catch (const sn::detail::Failure& failure) {
                check(failure.diagnostic.code == ir::DiagnosticCode::limit_exceeded, "handle program limits are structured");
                return std::nullopt;
            }
        };
        const auto all = programs(ir::Limits{}.analysis_steps, ir::Limits{}.resources);
        check(all && all->size() == 2, "two distinct handle programs");
        check(!programs(ir::Limits{}.analysis_steps, 1), "the resource limit bounds handle programs");
        check(!programs(8, ir::Limits{}.resources), "the analysis budget bounds handle programs");
    }
    // The evaluator: every operation over two words, and malformed programs.
    using O = ir::HandleOp;
    auto run = [](std::vector<std::uint32_t> code) {
        return ir::evaluate_handle_program(code, [](std::uint32_t, std::uint32_t) { return 0u; }, [](std::uint64_t) { return 0u; });
    };
    auto binary = [&](O op, std::uint32_t a, std::uint32_t b) {
        return run({unsigned(O::immediate), a, unsigned(O::immediate), b, unsigned(op)});
    };
    check(binary(O::subtract, 5, 7) == 0xfffffffeu && binary(O::shift_right_signed, 0x80000000u, 36) == 0xf8000000u &&
          binary(O::less_signed, 0xffffffffu, 0) == 1u && binary(O::less, 0xffffffffu, 0) == 0u &&
          binary(O::minimum_signed, 0xffffffffu, 3) == 0xffffffffu && binary(O::shift_left, 1, 33) == 2u, "handle program arithmetic");
    check(run({unsigned(O::immediate), 1, unsigned(O::immediate), 7, unsigned(O::immediate), 9, unsigned(O::select)}) == 7u &&
          run({unsigned(O::immediate), 3, unsigned(O::keep), 0, unsigned(O::recall), 0, unsigned(O::multiply)}) == 9u, "handle program select and kept values");
    check(!run({unsigned(O::add)}) && !run({unsigned(O::immediate)}) && !run({unsigned(O::immediate), 1, unsigned(O::immediate), 2}) &&
          !run({unsigned(O::recall), 64}) && !run({999u}), "malformed handle programs are refused");
}
// SUST.P and SULD.P admit a 2D surface (dimension 3, two coordinates) and a
// 3D surface (dimension 5, Ra..Ra+2). A 2D array (dimension 4) and the other
// surface dimensions stay refused. RZ reads zero coordinates, and three
// coordinate registers may end on R254.
void surface_tests() {
    auto surface = [](unsigned opcode, unsigned dimension, unsigned coordinate, unsigned data) {
        if (opcode == 0xb9b) {
            auto i = raw(0xb9b);
            field(i, 24, 8, coordinate); field(i, 32, 8, data); field(i, 40, 14, 0x1e); field(i, 54, 5, 0);
            field(i, 61, 3, dimension); field(i, 72, 4, 0xf); field(i, 84, 3, 1);
            return i;
        }
        auto i = raw(0xb97);
        field(i, 16, 8, data); field(i, 24, 8, coordinate); field(i, 40, 14, 0x1e); field(i, 54, 5, 0);
        field(i, 61, 3, dimension); field(i, 72, 4, 0xf); field(i, 81, 4, 15);
        return i;
    };
    auto coordinates_of = [](const ir::Program& program, ir::Opcode opcode) {
        const ir::Operation* found = nullptr;
        for (const auto& operation : program.operations)
            if (operation.opcode == opcode) { check(!found, "one surface operation"); found = &operation; }
        check(found, "the surface lowers");
        return std::pair{found, std::get<ir::TextureInfo>(found->info)};
    };
    auto expect = [&](unsigned opcode, unsigned dimension, unsigned coordinate, ir::TextureDimension shape, unsigned axes) {
        Fixture fixture({surface(opcode, dimension, coordinate, 4), exit_instruction()}, sn::Stage::compute);
        const auto program = lowered(fixture).ir;
        const auto [operation, info] = coordinates_of(program, opcode == 0xb9b ? ir::Opcode::image_store : ir::Opcode::image_load);
        check(info.descriptor && info.descriptor->shape == ir::TextureShape{shape, false, false}, "the surface derives its image shape");
        check(info.coordinates.size() == axes, "the surface takes one coordinate per axis");
        for (unsigned axis = 0; axis < axes; ++axis) {
            const auto& operand = operation->inputs[info.coordinates[axis].value];
            if (coordinate == 255) {
                const auto* literal = std::get_if<ir::LiteralBits>(&operand);
                check(literal && literal->bits == 0, "RZ reads a zero coordinate");
            } else {
                const auto* reg = std::get_if<ir::Register>(&operand);
                check(reg && reg->file == ir::RegisterFile::gpr && reg->index == coordinate + axis, "coordinates are Ra onward");
            }
        }
        if (opcode == 0xb97 && coordinate != 255) {
            std::array<bool, 3> queried{};
            for (const auto& operation : program.operations) {
                if (operation.opcode != ir::Opcode::texture_query) continue;
                const auto& query = std::get<ir::TextureInfo>(operation.info);
                if (query.mode != ir::TextureMode::query_size) continue;
                check(query.query_component < axes && query.descriptor && query.descriptor->shape.dimension == shape,
                      "a surface load queries each axis of its image");
                queried[query.query_component] = true;
            }
            for (unsigned axis = 0; axis < axes; ++axis) check(queried[axis], "a surface load checks every axis");
        }
    };
    expect(0xb9b, 5, 0, ir::TextureDimension::d3, 3);
    expect(0xb97, 5, 0, ir::TextureDimension::d3, 3);
    expect(0xb9b, 5, 255, ir::TextureDimension::d3, 3);
    expect(0xb9b, 5, 252, ir::TextureDimension::d3, 3);
    expect(0xb9b, 3, 0, ir::TextureDimension::d2, 2);
    expect(0xb97, 3, 0, ir::TextureDimension::d2, 2);
    refused(Fixture({surface(0xb9b, 5, 253, 4), exit_instruction()}, sn::Stage::compute));
    for (const unsigned dimension : {0u, 1u, 2u, 4u, 6u, 7u}) {
        for (const unsigned opcode : {0xb9bu, 0xb97u}) {
            Fixture refused_dimension({surface(opcode, dimension, 0, 4), exit_instruction()}, sn::Stage::compute);
            auto result = sn::emit_native_ir(refused_dimension.program());
            const auto* error = std::get_if<ir::Diagnostic>(&result);
            check(error && error->message.find("other than 2D and 3D") != std::string::npos,
                  "a surface dimension other than 2D and 3D stays refused");
        }
    }
}
// Register HFMA2 (0x231), immediate HSETP2 (0x434), bound U32 surface
// reduction (0x59f) and half-precision gather (0xb63 with bit 79). Neighbouring
// operand forms and the reduction operations the image path cannot express stay refused.
void admitted_form_tests() {
    auto message = [](const Fixture& fixture) {
        auto result = sn::emit_native_ir(fixture.program());
        const auto* error = std::get_if<ir::Diagnostic>(&result);
        check(error, "expected native frontend refusal");
        return error->message;
    };
    auto hfma = [](unsigned opcode) {
        auto i = raw(opcode);
        field(i, 16, 8, 0); field(i, 24, 8, 8); field(i, 32, 8, 9); field(i, 60, 2, 2); field(i, 64, 8, 10);
        return i;
    };
    {
        const auto program = lowered(Fixture({hfma(0x231), exit_instruction()})).ir;
        bool multiply = false, factor = false, literal = false;
        for (const auto& operation : program.operations) {
            if (operation.opcode == ir::Opcode::fused_multiply_add) multiply = true;
            if (operation.opcode != ir::Opcode::unpack_half2) continue;
            if (const auto* reg = std::get_if<ir::Register>(&operation.inputs.at(0))) factor |= reg->index == 9;
            else literal = true;
        }
        check(multiply && factor && !literal, "register fused multiply-add widens its register factor");
    }
    check(message(Fixture({[&] { auto i = hfma(0x231); field(i, 40, 1, 1); return i; }(), exit_instruction()})).find("reserved bits") != std::string::npos,
          "an unread bit of register HFMA2 stays refused");
    check(message(Fixture({hfma(0x431), exit_instruction()})).find("operand form") != std::string::npos,
          "the other fused multiply-add operand forms stay refused");
    check(message(Fixture({[&] { auto i = hfma(0x231); field(i, 86, 1, 1); return i; }(), exit_instruction()})).find("mixed-sign") != std::string::npos,
          "exchanging a selected half factor stays refused");

    auto hmul = [](unsigned opcode) {
        auto i = raw(opcode);
        field(i, 16, 8, 0); field(i, 24, 8, 8);
        if (opcode == 0x832) field(i, 32, 32, 0x3c003c00u);
        else { field(i, 32, 8, 9); field(i, 62, 1, 1); }
        return i;
    };
    {
        const auto program = lowered(Fixture({hmul(0x232), exit_instruction()})).ir;
        bool multiply = false, factor = false, absolute = false;
        for (const auto& operation : program.operations) {
            if (operation.opcode == ir::Opcode::multiply) multiply = true;
            if (operation.opcode == ir::Opcode::absolute) absolute = true;
            if (operation.opcode != ir::Opcode::unpack_half2) continue;
            if (const auto* reg = std::get_if<ir::Register>(&operation.inputs.at(0))) factor |= reg->index == 9;
        }
        check(multiply && factor && absolute, "register half multiply widens and takes the absolute of its register factor");
    }
    {
        const auto program = lowered(Fixture({hmul(0x832), exit_instruction()})).ir;
        bool multiply = false, literal = false;
        for (const auto& operation : program.operations) {
            if (operation.opcode == ir::Opcode::multiply) multiply = true;
            if (operation.opcode == ir::Opcode::unpack_half2)
                if (const auto* bits = std::get_if<ir::LiteralBits>(&operation.inputs.at(0))) literal |= bits->bits == 0x3c003c00u;
        }
        check(multiply && literal, "immediate half multiply widens the packed immediate");
    }
    check(message(Fixture({[&] { auto i = hmul(0x232); field(i, 40, 1, 1); return i; }(), exit_instruction()})).find("reserved bits") != std::string::npos,
          "an unread bit of register HMUL2 stays refused");
    check(message(Fixture({[&] { auto i = hmul(0x232); field(i, 76, 1, 1); return i; }(), exit_instruction()})).find("denormal") != std::string::npos,
          "half multiply denormal mode stays refused");
    check(message(Fixture({[&] { auto i = hmul(0x232); field(i, 78, 1, 1); return i; }(), exit_instruction()})).find("half multiply mode") != std::string::npos,
          "a half multiply mode bit stays refused");
    for (const unsigned opcode : {0x432u, 0x632u, 0xa32u})
        check(message(Fixture({hmul(opcode), exit_instruction()})).find("operand form") != std::string::npos,
              "the other half multiply operand forms stay refused");

    // A 2D texel fetch may take its per-axis offsets from a register. The
    // same register offset on a sample stays a literal-offset rejection.
    auto texel = [](unsigned opcode, unsigned lod) {
        auto t = raw(opcode);
        field(t, 16, 8, 0); field(t, 24, 8, 8); field(t, 32, 8, 12);
        field(t, 40, 14, 0x10); field(t, 54, 5, 3);
        field(t, 60, 1, 1); field(t, 61, 3, 1); field(t, 64, 8, 255);
        field(t, 72, 4, 1); field(t, 76, 1, 1); field(t, 81, 3, 7); field(t, 87, 3, lod);
        return t;
    };
    auto offset_word = [] {
        auto i = raw(0xa02); field(i, 16, 8, 12); field(i, 54, 5, 3); field(i, 38, 16, 16); field(i, 64, 32, 0xf00); return i;
    };
    {
        const auto program = lowered(Fixture({offset_word(), texel(0xb66, 1), exit_instruction()})).ir;
        const ir::Operation* fetch = nullptr;
        bool extracted = false;
        for (const auto& operation : program.operations) {
            if (operation.opcode == ir::Opcode::texture_sample) fetch = &operation;
            if (operation.opcode != ir::Opcode::bit_extract) continue;
            const auto* reg = std::get_if<ir::Register>(&operation.inputs.at(0));
            const auto* width = std::get_if<ir::LiteralBits>(&operation.inputs.at(2));
            extracted |= reg && reg->index == 12 && width && width->bits == 4 &&
                operation.numeric.domain == ir::Domain::signed_integer;
        }
        check(fetch && extracted, "a register-offset fetch lowers");
        const auto& info = std::get<ir::TextureInfo>(fetch->info);
        check(info.mode == ir::TextureMode::fetch && info.offsets.size() == 2, "a 2D fetch carries one offset per axis");
        for (auto index : info.offsets) check(std::holds_alternative<ir::ValueId>(fetch->inputs.at(index.value)),
                                              "a register texel offset is not a literal");
        auto translated = sn::translate_native(Fixture({offset_word(), texel(0xb66, 1), exit_instruction()}).program());
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        check(std::get<ir::MslArtifact>(translated).source.find(".read(") != std::string::npos,
              "a register-offset fetch reaches a texel read");
    }
    {
        auto translated = sn::translate_native(Fixture({offset_word(), texel(0xb60, 0), exit_instruction()}).program());
        const auto* error = std::get_if<ir::Diagnostic>(&translated);
        check(error && error->message.find("literal") != std::string::npos,
              "a sample with a register offset stays refused");
    }
    // A per-texel gather may take its eight signed fields from registers.
    // One shared offset, and a sample offset, stay literal.
    auto gather_off = [](unsigned mode) {
        auto t = raw(0xb63);
        field(t, 16, 8, 0); field(t, 24, 8, 8); field(t, 32, 8, 12);
        field(t, 40, 14, 0x10); field(t, 54, 5, 3);
        field(t, 60, 1, 1); field(t, 61, 3, 1); field(t, 64, 8, 2);
        field(t, 72, 4, 15); field(t, 76, 2, mode); field(t, 81, 3, 7);
        return t;
    };
    auto second_offset = [] {
        auto i = raw(0xa02); field(i, 16, 8, 13); field(i, 54, 5, 3); field(i, 38, 16, 20); field(i, 64, 32, 0xf00); return i;
    };
    {
        const auto program = lowered(Fixture({offset_word(), second_offset(), gather_off(2), exit_instruction()}, sn::Stage::fragment)).ir;
        std::vector<std::pair<unsigned, std::uint64_t>> fields_seen;
        for (const auto& operation : program.operations) {
            if (operation.opcode != ir::Opcode::bit_extract) continue;
            const auto* reg = std::get_if<ir::Register>(&operation.inputs.at(0));
            const auto* start = std::get_if<ir::LiteralBits>(&operation.inputs.at(1));
            const auto* width = std::get_if<ir::LiteralBits>(&operation.inputs.at(2));
            if (!reg || !start || !width || width->bits != 6 || operation.numeric.domain != ir::Domain::signed_integer) continue;
            fields_seen.push_back({reg->index, start->bits});
        }
        const std::vector<std::pair<unsigned, std::uint64_t>> want{{12, 0}, {12, 8}, {12, 16}, {12, 24}, {13, 0}, {13, 8}, {13, 16}, {13, 24}};
        check(fields_seen == want, "a register per-texel gather sign-extends each byte in texel order");
        auto translated = sn::translate_native(Fixture({offset_word(), second_offset(), gather_off(2), exit_instruction()}, sn::Stage::fragment).program());
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        const auto& source = std::get<ir::MslArtifact>(translated).source;
        check(source.find(".gather(") != std::string::npos && source.find("int2(") != std::string::npos,
              "a register per-texel gather reaches four gathers");
        auto at = source.find("int2(");
        check(at != std::string::npos && at + 5 < source.size() && source[at + 5] != '-' &&
              (source[at + 5] < '0' || source[at + 5] > '9'),
              "a register per-texel offset is a run-time int2");
    }
    {
        auto translated = sn::translate_native(Fixture({offset_word(), gather_off(1), exit_instruction()}, sn::Stage::fragment).program());
        const auto* error = std::get_if<ir::Diagnostic>(&translated);
        check(error && error->message.find("literal") != std::string::npos,
              "a gather with one register offset stays refused");
    }
    {
        auto handle = raw(0x367);
        field(handle, 16, 8, 0); field(handle, 24, 8, 8); field(handle, 32, 8, 10);
        field(handle, 59, 1, 1); field(handle, 60, 1, 1); field(handle, 61, 3, 1);
        field(handle, 64, 8, 255); field(handle, 72, 4, 3); field(handle, 76, 1, 1);
        field(handle, 81, 3, 7); field(handle, 87, 3, 1);
        check(message(Fixture({handle, exit_instruction()})).find("bound form") != std::string::npos,
              "a register-handle texel fetch with an offset stays refused");
    }

    auto hset = [](unsigned function, unsigned combine, unsigned first, unsigned second) {
        auto i = raw(0x434);
        field(i, 24, 8, 8); field(i, 32, 32, 0x3c003c00u); field(i, 69, 2, combine);
        field(i, 76, 4, function); field(i, 81, 3, first); field(i, 84, 3, second); field(i, 87, 4, 7);
        return i;
    };
    auto has_greater = [](const ir::Program& program) {
        for (const auto& operation : program.operations)
            if (operation.opcode == ir::Opcode::compare && std::get<ir::ComparisonInfo>(operation.info).relation == ir::Compare::greater)
                return true;
        return false;
    };
    check(has_greater(lowered(Fixture({hset(4, 0, 0, 1), exit_instruction()})).ir), "immediate half compare uses the lane relation");
    {
        auto flushed = hset(4, 0, 0, 1); field(flushed, 80, 1, 1);
        bool threshold = false;
        for (const auto& operation : lowered(Fixture({flushed, exit_instruction()})).ir.operations)
            for (const auto& input : operation.inputs)
                if (const auto* bits = std::get_if<ir::LiteralBits>(&input)) threshold |= bits->bits == 0x38800000u;
        check(threshold, "half compare flush uses the smallest normal binary16");
    }
    check(message(Fixture({hset(4, 3, 0, 1), exit_instruction()})).find("combination mode") != std::string::npos,
          "half predicate combination mode 3 stays refused");
    check(message(Fixture({hset(4, 0, 1, 1), exit_instruction()})).find("alias") != std::string::npos,
          "aliased half predicate destinations stay refused");
    check(message(Fixture({[&] { auto i = hset(4, 0, 0, 1); field(i, 64, 1, 1); return i; }(), exit_instruction()})).find("reserved bits") != std::string::npos,
          "an unread bit of immediate HSETP2 stays refused");
    auto hset_reg = [](unsigned function) {
        auto i = raw(0x234);
        field(i, 24, 8, 8); field(i, 32, 8, 9); field(i, 60, 2, 2); field(i, 62, 1, 1);
        field(i, 69, 2, 0); field(i, 76, 4, function); field(i, 81, 3, 0); field(i, 84, 3, 1); field(i, 87, 4, 7);
        return i;
    };
    {
        const auto program = lowered(Fixture({hset_reg(4), exit_instruction()})).ir;
        bool greater = false, factor = false, absolute = false;
        for (const auto& operation : program.operations) {
            if (operation.opcode == ir::Opcode::compare && std::get<ir::ComparisonInfo>(operation.info).relation == ir::Compare::greater)
                greater = true;
            if (operation.opcode == ir::Opcode::absolute) absolute = true;
            if (operation.opcode != ir::Opcode::unpack_half2) continue;
            if (const auto* reg = std::get_if<ir::Register>(&operation.inputs.at(0))) factor |= reg->index == 9;
        }
        check(greater && factor && absolute, "register half compare widens its register source and takes its absolute");
    }
    check(message(Fixture({[&] { auto i = hset_reg(4); field(i, 48, 1, 1); return i; }(), exit_instruction()})).find("reserved bits") != std::string::npos,
          "a bit in 40-59 of register HSETP2 stays refused");
    check(message(Fixture({[&] { auto i = hset_reg(4); field(i, 66, 1, 1); return i; }(), exit_instruction()})).find("reserved bits") != std::string::npos,
          "a bit in 64-68 of register HSETP2 stays refused");

    auto reduction = [](unsigned dimension, unsigned kind, unsigned coordinate = 0, unsigned slot = 0) {
        auto i = raw(0x59f);
        field(i, 24, 8, coordinate); field(i, 32, 8, 1); field(i, 40, 14, 132); field(i, 54, 5, slot);
        field(i, 61, 3, dimension); field(i, 77, 3, 7); field(i, 84, 1, 1); field(i, 87, 3, kind);
        return i;
    };
    auto atomic_of = [](const ir::Program& program) {
        const ir::Operation* found = nullptr;
        for (const auto& operation : program.operations)
            if (operation.opcode == ir::Opcode::image_atomic) { check(!found, "one surface reduction"); found = &operation; }
        check(found, "the surface reduction lowers");
        return std::pair{found, std::get<ir::TextureInfo>(found->info)};
    };
    {
        // Keep the program: the operation pointer addresses its storage.
        const auto program = lowered(Fixture({reduction(3, 6), exit_instruction()}, sn::Stage::compute)).ir;
        const auto [operation, info] = atomic_of(program);
        const auto* key = std::get_if<ir::TextureKey>(&info.texture);
        check(info.atomic == ir::AtomicKind::bit_or && info.mode == ir::TextureMode::fetch && info.component_mask == 1, "U32 reduction is a one-component fetch atomic");
        check(key && key->cbuf_slot == 0 && key->handle_byte_offset == 132u * 4, "the reduction handle word is a byte offset");
        check(info.descriptor && info.descriptor->scalar == ir::ScalarType::u32 &&
              info.descriptor->shape == ir::TextureShape{ir::TextureDimension::d2, false, false}, "the reduction image is 2D U32");
        check(operation->inputs.size() == 3, "two coordinates and one datum");
        auto translated = sn::translate_native(Fixture({reduction(3, 6), exit_instruction()}, sn::Stage::compute).program());
        if (const auto* error = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error(error->message);
        const auto& source = std::get<ir::MslArtifact>(translated).source;
        check(source.find("atomic_fetch_or(") != std::string::npos &&
              source.find("texture2d<uint, access::read_write>") != std::string::npos &&
              source.find("uint4(") != std::string::npos && source.find("access::atomic") == std::string::npos,
              "a U32 reduction reaches a read-write image atomic");
    }
    for (const unsigned kind : {0u, 1u, 2u, 5u, 7u}) {
        const auto program = lowered(Fixture({reduction(3, kind), exit_instruction()}, sn::Stage::compute)).ir;
        const auto kind_of = atomic_of(program).second.atomic;
        const ir::AtomicKind want[]{ir::AtomicKind::add, ir::AtomicKind::minimum, ir::AtomicKind::maximum,
                                     ir::AtomicKind::add, ir::AtomicKind::add, ir::AtomicKind::bit_and, ir::AtomicKind::bit_or, ir::AtomicKind::bit_xor};
        check(kind_of == want[kind], "admitted surface reduction operations");
    }
    check(message(Fixture({reduction(1, 6), exit_instruction()}, sn::Stage::compute)).find("buffer is not admitted") != std::string::npos,
          "a buffer reduction stays refused");
    check(message(Fixture({reduction(0, 6), exit_instruction()}, sn::Stage::compute)).find("other than a buffer or 2D") != std::string::npos,
          "a reduction dimension other than a buffer or 2D stays refused");
    check(message(Fixture({reduction(3, 3), exit_instruction()}, sn::Stage::compute)).find("increment or decrement") != std::string::npos,
          "wrapping surface increment stays refused");
    check(message(Fixture({reduction(3, 4), exit_instruction()}, sn::Stage::compute)).find("increment or decrement") != std::string::npos,
          "wrapping surface decrement stays refused");
    check(message(Fixture({[&] { auto i = reduction(3, 6); field(i, 77, 3, 0); field(i, 84, 1, 0); return i; }(), exit_instruction()}, sn::Stage::compute)).find("reduction mode") != std::string::npos,
          "a surface reduction outside the bound U32 form stays refused");
    check(message(Fixture({reduction(3, 6, 0, 18), exit_instruction()}, sn::Stage::compute)).find("bank exceeds") != std::string::npos,
          "a surface reduction bank above 17 stays refused");
    check(message(Fixture({reduction(3, 6, 254), exit_instruction()}, sn::Stage::compute)).find("crosses RZ") != std::string::npos,
          "a surface reduction coordinate on RZ stays refused");

    auto gather = [](unsigned opcode, unsigned mask, unsigned mode, bool halves) {
        auto t = raw(opcode);
        field(t, 24, 8, 8); field(t, 32, 8, 10); field(t, 60, 1, 1); field(t, 61, 3, 1);
        field(t, 64, 8, 2); field(t, 72, 4, mask); field(t, 76, 2, mode); field(t, 81, 3, 7);
        if (halves) field(t, 79, 1, 1);
        if ((opcode >> 8) == 0xb) { field(t, 40, 14, 0x10); field(t, 54, 5, 3); }
        if ((opcode >> 8) == 0xf) { field(t, 40, 6, 4); field(t, 91, 1, 1); }
        if ((opcode >> 8) == 0x3) field(t, 59, 1, 1);
        return t;
    };
    {
        const auto program = lowered(Fixture({gather(0xb63, 15, 2, true), exit_instruction()}, sn::Stage::fragment)).ir;
        bool packed = false, gathered = false;
        for (const auto& operation : program.operations) {
            if (operation.opcode == ir::Opcode::pack_half2) packed = true;
            if (operation.opcode == ir::Opcode::texture_sample && std::get<ir::TextureInfo>(operation.info).mode == ir::TextureMode::gather)
                gathered = true;
        }
        check(packed && gathered, "a bound four-component gather packs binary16 results");
    }
    check(message(Fixture({gather(0xb63, 7, 0, true), exit_instruction()}, sn::Stage::fragment)).find("half-precision gather") != std::string::npos,
          "a partial half gather stays refused");
    check(message(Fixture({gather(0xb63, 15, 1, true), exit_instruction()}, sn::Stage::fragment)).find("half-precision gather") != std::string::npos,
          "a half gather with one offset register stays refused");
    check(message(Fixture({gather(0xf63, 15, 0, true), exit_instruction()}, sn::Stage::fragment)).find("half-precision gather") != std::string::npos,
          "a uniform half gather stays refused");
    check(message(Fixture({gather(0x364, 15, 0, true), exit_instruction()}, sn::Stage::fragment)).find("half-precision gather") != std::string::npos,
          "a register-handle half gather stays refused");
    check(message(Fixture({[&] { auto t = gather(0xb63, 15, 2, true); field(t, 80, 1, 1); return t; }(), exit_instruction()}, sn::Stage::fragment)).find("result format") != std::string::npos,
          "gather result format bit 80 stays refused");
    check(message(Fixture({[&] { auto t = gather(0xb63, 15, 0, true); field(t, 78, 1, 1); return t; }(), exit_instruction()}, sn::Stage::fragment)).find("half-precision gather") != std::string::npos,
          "a depth-compare half gather stays refused");
    check(message(Fixture({[&] { auto t = gather(0xb63, 15, 0, true); field(t, 64, 8, 255); return t; }(), exit_instruction()}, sn::Stage::fragment)).find("half-precision gather") != std::string::npos,
          "a half gather needs both destinations");
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
        // Owner decision: a register no definition reaches reads zero.
        Fixture undefined({exit_instruction()});
        check(execute(lowered(undefined).ir).at({ir::IoSemantic::color, 0, 0}) == 0, "undefined register reads zero");
        Fixture unknown({move(0, 1), raw(0xfff), exit_instruction()}); refused(unknown);
        Fixture unreachable({move(0, 2), exit_instruction(), raw(0x3a9)}); lowered(unreachable);
        // Depth and the sample mask follow four registers per enabled target,
        // however few components each enables: depth is R(4 * targets + 1).
        for (const std::uint32_t targets_mask : {0x10fu, 0x100u, 0x3fu, 0x30u})
            for (const unsigned extra : {2u, 3u}) {
                std::vector<sn::Instruction> words; for (unsigned i = 0; i < 16; ++i) words.push_back(move(i, 100 + i));
                words.push_back(exit_instruction());
                Fixture exported(words); put(exported.code, 0x48, targets_mask); exported.code[0x4c] = std::byte(extra);
                const auto out = execute(lowered(exported).ir);
                unsigned base = 0;
                for (unsigned target = 0; target < 8; ++target) {
                    const auto components = (targets_mask >> (target * 4)) & 15u;
                    if (!components) continue;
                    for (unsigned c = 0; c < 4; ++c)
                        if (components & (1u << c)) check(out.at({ir::IoSemantic::color, target, c}) == 100 + base + c, "sparse color with depth");
                    base += 4;
                }
                check(out.at({ir::IoSemantic::fragment_depth, 0, 0}) == 100 + base + 1, "depth follows four registers per target");
                if (extra & 1u) check(out.at({ir::IoSemantic::sample_mask, 0, 0}) == 100 + base, "the sample mask precedes depth");
                if (targets_mask == 0x10f && extra == 2) compile_fixture(exported, directory, "sparse_color_depth");
            }
        arithmetic_tests(directory);
        predicate_tests(directory);
        interpolation_tests(directory);
        global_memory_tests(directory);
        jump_table_tests();
        stage_buffer_tests();
        handle_program_tests();
        surface_tests();
        admitted_form_tests();
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
        // One attribute read as float (FMUL) and as integer (IADD3): exact
        // with shader-side fetch, which rebuilds the guest register word;
        // with host conversion the result would depend on the draw's format.
        auto add = raw(0x210); field(add, 16, 8, 1); field(add, 24, 8, 4); field(add, 32, 8, 4); field(add, 64, 8, 255);
        field(add, 77, 4, 15); field(add, 81, 3, 7); field(add, 84, 3, 7); field(add, 87, 4, 15);
        auto pair = varying; field(pair, 74, 2, 1);
        Fixture mixed({attr, multiply, add, pair, move(2, 0), move(3, 0x3f800000), store, exit_instruction()}, sn::Stage::vertex);
        mixed.code[0x18] = std::byte{1}; mixed.code[0x36] = std::byte{3};
        compile_fixture(mixed, directory, "vertex_attribute_mixed");
        sn::FrontendOptions host_conversion; host_conversion.vertex_depth = ir::DepthConvention::negative_one_to_one; host_conversion.vertex_fetch = false;
        check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(mixed.program(), host_conversion)),
              "a mixed attribute without shader-side fetch stays refused");
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
        // S2R of the shading rate (132) in a fragment stage: 1x1 (0x00010100)
        // only with the pixel rate proven disabled.
        {
            auto shading = raw(0x919); field(shading, 16, 8, 0); field(shading, 72, 8, 132);
            Fixture fragment_rate({shading, exit_instruction()});
            refused(fragment_rate);
            sn::FrontendOptions options;
            options.fragment_position = ir::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
            options.pixel_rate = sn::PixelRateState::enabled;
            check(std::holds_alternative<ir::Diagnostic>(sn::translate_native(fragment_rate.program(), options)), "an enabled pixel rate leaves the shading rate refused");
            options.pixel_rate = sn::PixelRateState::disabled;
            auto rated = sn::translate_native(fragment_rate.program(), options);
            if (auto e = std::get_if<ir::Diagnostic>(&rated)) throw std::runtime_error(e->message);
            check(std::get<ir::MslArtifact>(rated).reflection.profile.identity == sn::disabled_pixel_rate_profile, "the shading rate requires the disabled pixel rate");
            const auto& source = std::get<ir::MslArtifact>(rated).source;
            check(source.find("65792u") != std::string::npos || source.find("0x10100u") != std::string::npos ||
                  source.find("0x00010100u") != std::string::npos, "a disabled pixel rate reads 1x1");
        }
        sn::FrontendOptions disabled_unused; disabled_unused.vertex_depth=ir::DepthConvention::negative_one_to_one;disabled_unused.pixel_rate=sn::PixelRateState::disabled;
        auto unchanged=sn::translate_native(system.program(),disabled_unused);
        check(std::holds_alternative<ir::MslArtifact>(unchanged) && std::get<ir::MslArtifact>(unchanged).reflection.profile.identity=="armigo-native-metal4","unused specialization does not add a false state requirement");
        // A vertex that writes nothing is a void vertex for rasterizer-discard
        // draws, with or without a clip-depth option: it records no depth fact.
        Fixture empty_vertex({exit_instruction()}, sn::Stage::vertex);
        for (const bool depth : {false, true}) {
            sn::FrontendOptions options; if (depth) options.vertex_depth = ir::DepthConvention::negative_one_to_one;
            auto translated = sn::translate_native(empty_vertex.program(), options);
            if (auto e = std::get_if<ir::Diagnostic>(&translated)) throw std::runtime_error("empty vertex: " + e->message);
            const auto& artifact = std::get<ir::MslArtifact>(translated);
            check(artifact.reflection.requires_rasterizer_discard && artifact.source.find("vertex void ") != std::string::npos &&
                  std::none_of(artifact.reflection.dependencies.begin(), artifact.reflection.dependencies.end(),
                               [](const auto& d) { return d.key.kind == ir::FactKind::depth_convention; }),
                  "an empty vertex is a void vertex without a depth fact");
        }
        std::cout << "Native frontend contracts passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
