// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic tests for the FFMA register/register family. Every expected
// result word below is authored from the fused binary32 contract: flush
// subnormal inputs, round the exact product-sum nearest-even, flush a
// subnormal result, then saturate in a separate step. The established
// fma_rne_ftz primitive only cross-checks the authored words.

#include <armigo/shader/maxwell.h>
#include <armigo/shader/maxwell_forward_program.h>
#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/fma.h>
#include "test_support.h"
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>
namespace armigo::shader::test {
namespace {
namespace mx=maxwell; namespace gx=graphics; namespace gm=graphics_msl;
using Family=mx::OpcodeFamily; using Stage=shader_input::Stage;
using Value=mx::FloatFmaRegister;

// One synthetic FFMA (reg) word: destination 0-7, A 8-15, B 20-27, C 39-46,
// guard 16-19, CC 47, negate B 48, negate C 49, saturate 50, rounding 51-52,
// FMZ 53-54. The admitted profile fixes rounding and FMZ to their FTZ/RN
// encoding, so the word carries bit 53 set and nothing else there.
std::uint64_t word(unsigned d=3,unsigned a=1,unsigned b=2,unsigned c=4,
                   bool negate_b=false,bool negate_c=false,bool saturate=false) {
    return (std::uint64_t{0x5980}<<48)|(std::uint64_t{1}<<53)|(std::uint64_t{7}<<16)|
        d|(std::uint64_t{a}<<8)|(std::uint64_t{b}<<20)|(std::uint64_t{c}<<39)|
        (negate_b?std::uint64_t{1}<<48:std::uint64_t{0})|
        (negate_c?std::uint64_t{1}<<49:std::uint64_t{0})|
        (saturate?std::uint64_t{1}<<50:std::uint64_t{0});
}

void put(std::vector<std::uint8_t>& bytes,unsigned at,std::uint64_t value,unsigned count) {
    for(unsigned i=0;i<count;++i)bytes[at+i]=std::uint8_t(value>>(8*i));
}

// Full container decode of one instruction at pc 8 followed by a PT EXIT at
// pc 16; `guard` overrides the raw instruction-predicate field.
mx::DecodeResult decode(std::uint64_t value,Stage stage=Stage::Vertex,
                        unsigned guard=7) {
    std::vector<std::uint8_t> control(0x718),code(0xa0);
    put(control,0,0x98761234,4);put(control,4,1,4);put(control,8,14,4);
    put(control,0x20,control.size(),4);put(control,0x700,code.size(),4);
    put(control,0x704,code.size(),4);put(control,0x708,0x30,4);
    put(control,0x714,static_cast<unsigned>(stage),4);put(code,0,0x12345678,4);
    put(code,0x30,shader_input::sph_stage_type_for(stage)<<10,4);
    put(code,0x80,0x1122334455667788ull,8);
    put(code,0x88,(value&~(std::uint64_t{15}<<16))|(std::uint64_t{guard}<<16),8);
    put(code,0x90,(std::uint64_t{0xe300}<<48)|(std::uint64_t{7}<<16)|15,8);
    return mx::decode({control.data(),control.size()},{code.data(),code.size()});
}

mx::ForwardDecodeResult decode_forward(std::uint64_t value,Stage stage=Stage::Vertex,
                                       unsigned guard=7) {
    std::vector<std::uint8_t> control(0x718),code(0xa0);
    put(control,0,0x98761234,4);put(control,4,1,4);put(control,8,14,4);
    put(control,0x20,control.size(),4);put(control,0x700,code.size(),4);
    put(control,0x704,code.size(),4);put(control,0x708,0x30,4);
    put(control,0x714,static_cast<unsigned>(stage),4);put(code,0,0x12345678,4);
    put(code,0x30,shader_input::sph_stage_type_for(stage)<<10,4);
    put(code,0x80,0x1122334455667788ull,8);
    put(code,0x88,(value&~(std::uint64_t{15}<<16))|(std::uint64_t{guard}<<16),8);
    put(code,0x90,(std::uint64_t{0xe300}<<48)|(std::uint64_t{7}<<16)|15,8);
    return mx::decode_forward_program({control.data(),control.size()},
                                      {code.data(),code.size()});
}

void append(mx::DecodedProgram& p,Family family,mx::InstructionPayload payload) {
    const auto i=p.instructions.size();
    p.instructions.push_back({std::uint32_t((i/3)*32+(i%3+1)*8),0,family,{7,false},std::move(payload)});
}
void mov(mx::DecodedProgram& p,unsigned reg,std::uint32_t bits) {
    append(p,Family::Mov32Immediate,mx::Mov32Immediate{{std::uint8_t(reg)},15,bits});
}

// The payload exactly as the admitted decode profile produces it.
Value value(unsigned d=3,bool negate_b=false,bool negate_c=false,bool saturate=false) {
    return {{std::uint8_t(d)},{1},{2},{4},mx::RoundingMode::NearestEven,
        mx::FlushMode::FlushToZero,negate_b,negate_c,saturate};
}

// Straight-line program: MOV32I all three sources, the fused multiply-add,
// then an unconditional EXIT. Vertex programs store the destination to the
// generic attribute lanes; fragment programs expose register zero as the
// color output, so fragment callers pass destination zero.
mx::DecodedProgram program(Value v,std::uint32_t a=0x40000000,std::uint32_t b=0x40400000,
                           std::uint32_t c=0x40a00000,Stage stage=Stage::Vertex,
                           bool init_b=true,bool init_c=true) {
    mx::DecodedProgram p;p.stage=stage;
    const auto header=shader_input::sph_stage_type_for(stage)<<10;
    for(unsigned i=0;i<4;++i)p.sph[i]=std::byte(header>>(i*8));
    if(stage==Stage::Vertex)p.sph[0x35]=std::byte{0xf0};
    else p.sph[0x48]=std::byte{0xf};
    mov(p,1,a);if(init_b)mov(p,2,b);if(init_c)mov(p,4,c);
    if(stage==Stage::Fragment){mov(p,0,0);mov(p,3,0x3f800000);}
    append(p,Family::FloatFmaRegister,std::move(v));
    if(stage==Stage::Vertex)for(unsigned lane=28;lane<32;++lane)
        append(p,Family::AttributeStore,mx::AttributeAccess{
            {std::uint8_t(v.destination.is_zero()?1:v.destination.index)},{255},{255},
            std::uint16_t(lane*4),0,false});
    append(p,Family::Exit,mx::Exit{15});return p;
}

std::uint32_t evaluate(const mx::DecodedProgram& p) {
    const auto lowered=gx::lower(p);ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if(!std::holds_alternative<gx::Program>(lowered))return 0;
    const auto result=gx::evaluate(std::get<gx::Program>(lowered),{});
    const auto* evaluated=std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(evaluated);if(!evaluated)return 0;
    for(const auto& output:evaluated->outputs)if(output.component==0)return output.bits;
    ARMIGO_CHECK(false);return 0;
}

void error(const mx::DecodedProgram& p,gx::Error::Code expected) {
    const auto result=gx::lower(p);const auto* e=std::get_if<gx::Error>(&result);
    ARMIGO_CHECK(e);if(e)ARMIGO_CHECK(e->code==expected);
}

// Locates the single fused node (and optional saturate node) of a lowered
// program and reports their operand wiring through `checks`.
struct Nodes {
    std::optional<std::size_t> fused,saturate;
    bool negate_lhs=false;
    std::size_t lhs=0,rhs=0,addend=0;
};
Nodes fused_nodes(const mx::DecodedProgram& p) {
    const auto lowered=gx::lower(p);ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    Nodes found;
    if(!std::holds_alternative<gx::Program>(lowered))return found;
    const auto nodes=std::get<gx::Program>(lowered).nodes();
    for(std::size_t i=0;i<nodes.size();++i) {
        if(const auto* arithmetic=std::get_if<gx::FloatArithmetic>(&nodes[i].operation)) {
            if(arithmetic->operation!=gx::Arithmetic::FusedMultiplyAdd)continue;
            ARMIGO_CHECK(!found.fused.has_value());
            found.fused=i;found.negate_lhs=arithmetic->negate_lhs;
            ARMIGO_CHECK(arithmetic->addend.has_value());
            if(arithmetic->addend.has_value()) {
                found.lhs=arithmetic->lhs.index;found.rhs=arithmetic->rhs.index;
                found.addend=arithmetic->addend->index;
            }
        } else if(std::holds_alternative<gx::Saturate>(nodes[i].operation)) {
            found.saturate=i;
        }
    }
    return found;
}
}  // namespace

ARMIGO_TEST(ffma_register, decodes_fields_and_rz_sources) {
    for(const auto stage:{Stage::Vertex,Stage::Fragment}) {
        const auto encoded=word(7,9,11,13);
        const auto result=decode(encoded,stage);
        const auto* p=std::get_if<mx::DecodedProgram>(&result);ARMIGO_CHECK(p);if(!p)continue;
        const auto& i=p->instructions.front();ARMIGO_CHECK(i.family==Family::FloatFmaRegister);
        ARMIGO_CHECK_EQ(i.pc,8u);ARMIGO_CHECK_EQ(i.original_word,encoded);
        const auto* v=std::get_if<Value>(&i.payload);ARMIGO_CHECK(v);if(!v)continue;
        ARMIGO_CHECK(v->destination.index==7&&v->source_a.index==9);
        ARMIGO_CHECK(v->source_b.index==11&&v->source_c.index==13);
        ARMIGO_CHECK(v->rounding==mx::RoundingMode::NearestEven);
        ARMIGO_CHECK(v->flush==mx::FlushMode::FlushToZero);
        ARMIGO_CHECK(!v->negate_b&&!v->negate_c&&!v->saturate);
    }
    const auto result=decode(word(255,255,255,255));
    const auto* p=std::get_if<mx::DecodedProgram>(&result);ARMIGO_CHECK(p);
    if(p){const auto& v=std::get<Value>(p->instructions.front().payload);
        ARMIGO_CHECK(v.destination.is_zero()&&v.source_a.is_zero());
        ARMIGO_CHECK(v.source_b.is_zero()&&v.source_c.is_zero());}
}

ARMIGO_TEST(ffma_register, admits_every_modifier_combination) {
    for(unsigned flags=0;flags<8;++flags) {
        const auto encoded=word(3,1,2,4,flags&1,flags&2,flags&4);
        const auto result=decode(encoded);
        const auto* p=std::get_if<mx::DecodedProgram>(&result);
        ARMIGO_CHECK(p);if(!p)continue;
        const auto& v=std::get<Value>(p->instructions.front().payload);
        ARMIGO_CHECK(v.negate_b==((flags&1)!=0)&&v.negate_c==((flags&2)!=0)&&
                     v.saturate==((flags&4)!=0));
    }
}

ARMIGO_TEST(ffma_register, rejects_cc_rounding_flush_and_reserved_bits) {
    const auto cc=decode(word()|(std::uint64_t{1}<<47));
    const auto* e=std::get_if<mx::DecodeError>(&cc);ARMIGO_CHECK(e);
    if(e){ARMIGO_CHECK(e->code==mx::DecodeError::Code::UnsupportedModifier);
        ARMIGO_CHECK(e->field==mx::DecodeField::ConditionCode);ARMIGO_CHECK_EQ(e->pc,8u);}
    for(unsigned rounding:{1u,2u,3u}) {
        const auto result=decode(word()|(std::uint64_t{rounding}<<51));
        const auto* error=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(error);
        if(error)ARMIGO_CHECK(error->field==mx::DecodeField::Rounding);
    }
    for(unsigned flush:{0u,2u,3u}) {
        const auto result=decode((word()&~(std::uint64_t{3}<<53))|
                                 (std::uint64_t{flush}<<53));
        const auto* error=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(error);
        if(error){ARMIGO_CHECK(error->field==mx::DecodeField::FlushMode);
            ARMIGO_CHECK(error->expected==1);}
    }
    // The field gap between the B and C registers is reserved in this form.
    for(unsigned bit:{28u,33u,38u}) {
        const auto result=decode(word()|(std::uint64_t{1}<<bit));
        const auto* error=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(error);
        if(error){ARMIGO_CHECK(error->code==mx::DecodeError::Code::UnsupportedModifier);
            ARMIGO_CHECK(error->field==mx::DecodeField::ReservedBits);}
    }
    for(unsigned pred:{0u,15u}) {
        const auto result=decode(word(),Stage::Vertex,pred);
        const auto* error=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(error);
        if(error)ARMIGO_CHECK(error->code==mx::DecodeError::Code::UnsupportedPredicate);
    }
}

ARMIGO_TEST(ffma_register, window_leaves_neighbouring_ffma_windows_alone) {
    // Words across the window's unmasked low half still route here: the
    // whole 0x5980-0x59ff range reports this family (the modifier rules
    // reject the forged modifier bits, attributed to this family).
    const auto wide=decode((word()&~(std::uint64_t{0xffff}<<48))|(std::uint64_t{0x59ff}<<48));
    const auto* wide_error=std::get_if<mx::DecodeError>(&wide);ARMIGO_CHECK(wide_error);
    if(wide_error){ARMIGO_CHECK(wide_error->family==Family::FloatFmaRegister);
        ARMIGO_CHECK(wide_error->field==mx::DecodeField::Rounding);}
    // A DSET (reg) word one bit away must remain an explicit unknown opcode.
    const auto neighbour=decode((word()&~(std::uint64_t{0xffff}<<48))|(std::uint64_t{0x5900}<<48));
    const auto* unknown=std::get_if<mx::DecodeError>(&neighbour);ARMIGO_CHECK(unknown);
    if(unknown)ARMIGO_CHECK(unknown->code==mx::DecodeError::Code::UnknownOpcode);
    // Each neighbouring FFMA window still routes to its own family.
    const std::uint64_t common=(std::uint64_t{7}<<16)|(std::uint64_t{1}<<53);
    struct Neighbour { std::uint64_t high; Family family; };
    const std::array<Neighbour,5> neighbours{{
        {(std::uint64_t{0x4980}<<48)|common|3|(std::uint64_t{1}<<8)|
         (std::uint64_t{5}<<34)|(std::uint64_t{2}<<39),
         Family::FloatFmaConstantRegister},
        {(std::uint64_t{0x4981}<<48)|common|3|(std::uint64_t{1}<<8)|
         (std::uint64_t{5}<<34)|(std::uint64_t{2}<<39),
         Family::FloatFmaConstantNegated},
        {(std::uint64_t{0x5180}<<48)|common|3|(std::uint64_t{1}<<8)|
         (std::uint64_t{2}<<39)|(std::uint64_t{5}<<34)|(std::uint64_t{0x123}<<20),
         Family::FloatFmaRegisterConstant},
        {(std::uint64_t{0x3280}<<48)|common|3|(std::uint64_t{1}<<8)|
         (std::uint64_t{2}<<39)|std::uint64_t{0x12345},
         Family::FloatFmaImmediate},
        {(std::uint64_t{0x5c6c}<<48)|(std::uint64_t{7}<<16)|(std::uint64_t{1}<<44)|3|
         (std::uint64_t{1}<<8)|(std::uint64_t{2}<<20),
         Family::FloatMultiplySaturateRegister},
    }};
    for(const auto& n:neighbours) {
        const auto result=decode(n.high);
        const auto* p=std::get_if<mx::DecodedProgram>(&result);ARMIGO_CHECK(p);if(!p)continue;
        ARMIGO_CHECK(p->instructions.front().family==n.family);
    }
}

ARMIGO_TEST(ffma_register, lowers_fused_node_with_operand_order_and_negate_lhs) {
    auto nodes=fused_nodes(program(value()));
    ARMIGO_CHECK(nodes.fused.has_value());
    ARMIGO_CHECK(!nodes.saturate.has_value());
    if(nodes.fused)ARMIGO_CHECK(!nodes.negate_lhs);
    nodes=fused_nodes(program(value(3,true)));
    ARMIGO_CHECK(nodes.fused.has_value());
    if(nodes.fused)ARMIGO_CHECK(nodes.negate_lhs);
    nodes=fused_nodes(program(value(3,false,false,true)));
    ARMIGO_CHECK(nodes.fused.has_value()&&nodes.saturate.has_value());
    if(nodes.fused&&nodes.saturate) {
        ARMIGO_CHECK(*nodes.fused<*nodes.saturate);
        const auto lowered=gx::lower(program(value(3,false,false,true)));
        const auto& saturate=std::get<gx::Saturate>(
            std::get<gx::Program>(lowered).nodes()[*nodes.saturate].operation);
        ARMIGO_CHECK_EQ(saturate.operand.index,static_cast<std::uint32_t>(*nodes.fused));
    }
}

ARMIGO_TEST(ffma_register, forged_payload_and_modifiers_fail) {
    auto forged=program(value());
    for(auto& i:forged.instructions)if(i.family==Family::FloatFmaRegister)
        i.payload=mx::MovConstant{{3},{5,0},15};
    error(forged,gx::Error::Code::InvalidInstruction);
    auto v=value();v.rounding=mx::RoundingMode::Zero;
    error(program(v),gx::Error::Code::UnsupportedModifier);
    v=value();v.flush=mx::FlushMode::None;
    error(program(v),gx::Error::Code::UnsupportedModifier);
    // NegateC negates the addend before the fused operation: 2*3 - 5 = 1,
    // and an exact cancellation 2*2.5 - 5 is +0 (round to nearest), which
    // -fma(-a, b, c) would turn into -0.
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,true))),0x3f800000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,true),0x40000000,0x40200000,0x40a00000)),
                    0x00000000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(3,true,true),0x40000000,0x40200000,0x40a00000)),
                    0xc1200000u);
    error(program(value(),0x40000000,0x40400000,0x40a00000,Stage::Vertex,true,false),
          gx::Error::Code::UninitializedRegister);
    error(program(value(),0x40000000,0x40400000,0x40a00000,Stage::Vertex,false,true),
          gx::Error::Code::UninitializedRegister);
}

ARMIGO_TEST(ffma_register, evaluates_fused_result_where_unfused_differs) {
    // fma(1+2^-23, 1-2^-24, -1) is exact: 2^-24 - 2^-47, one ulp pair below
    // 2^-24. The unfused chain rounds the product to 1.0 and yields +0, so
    // only the fused contract produces this word.
    ARMIGO_CHECK_EQ(evaluate(program(value(),0x3f800001u,0x3f7fffffu,0xbf800000u)),
                    0x337ffffeu);
    // Operand order: lhs*B + addend, not any other pairing.
    ARMIGO_CHECK_EQ(evaluate(program(value(),0x40000000u,0x40400000u,0x40a00000u)),
                    0x41300000u);  // 2*3+5 = 11
    // The product's sign flip carries NegateB: (-1)*(-3)+5 = 8.
    ARMIGO_CHECK_EQ(evaluate(program(value(3,true),0xbf800000u,0x40400000u,0x40a00000u)),
                    0x41000000u);
    // Aliased addend register: the read precedes the destination write.
    ARMIGO_CHECK_EQ(evaluate(program(value(4),0x40000000u,0x40400000u,0x40a00000u)),
                    0x41300000u);
    // Register-zero sources read as zero and a register-zero destination
    // discards the result without clobbering a source.
    auto rz_addend=value();rz_addend.source_c={255};
    ARMIGO_CHECK_EQ(evaluate(program(rz_addend,0x40000000u,0x40400000u,0u,
                                     Stage::Vertex,true,false)),
                    0x40c00000u);  // 2*3+0
    auto rz=value();rz.source_b={255};
    ARMIGO_CHECK_EQ(evaluate(program(rz,0x40000000u,0x40400000u,0x40a00000u)),
                    0x40a00000u);  // 2*0+5
    ARMIGO_CHECK_EQ(evaluate(program(value(255),0x40000000u,0x40400000u,0x40a00000u)),
                    0x40000000u);  // discarded destination leaves source A intact
    // Cross-checks of the authored words against the established fused
    // primitive, with NegateB applied to either multiplicand.
    constexpr std::array values{0u,0x80000000u,1u,0x80000001u,0x3f800000u,
        0xbf800000u,0x7f800000u,0xff800000u,0x7fc12345u,0x7f811234u};
    for(auto a:values)for(auto b:values)for(auto c:values) {
        ARMIGO_CHECK_EQ(evaluate(program(value(),a,b,c)),
                        shader::fma_rne_ftz(a,b,c));
        ARMIGO_CHECK_EQ(evaluate(program(value(3,true),a,b,c)),
                        shader::fma_rne_ftz(a,b^0x80000000u,c));
    }
    ARMIGO_CHECK_EQ(evaluate(program(value(0),0x40000000u,0x40400000u,0x40a00000u,
                                     Stage::Fragment)),0x41300000u);
}

ARMIGO_TEST(ffma_register, saturate_follows_the_fused_sum) {
    // The fused sum saturates as a whole: 2*3-5 = 1 stays one, 2*3+5 = 11
    // clamps to one, and -1*3+5 = 2 clamps after the arithmetic, which no
    // operand saturation could produce.
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,false,true),0x40000000u,
                                     0x40400000u,0x40a00000u)),0x3f800000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,false,true),0x40000000u,
                                     0x40400000u,0x40a00000u)),0x3f800000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,false,true),0xbf800000u,
                                     0x40400000u,0x40a00000u)),0x3f800000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,false,true),0xbf800000u,
                                     0x40400000u,0x40000000u)),0x00000000u);
    // Signed-zero and NaN sums through the bounded saturation policy.
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,false,true),0x80000000u,
                                     0x3f800000u,0x00000000u)),0x00000000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(3,false,false,true),0x7fc00000u,
                                     0x3f800000u,0x00000000u)),0x00000000u);
}

ARMIGO_TEST(ffma_register, forward_decoder_admits_guarded_family) {
    for(const auto [guard,index,inverted]:{std::tuple{1u,1u,false},
                                           std::tuple{9u,1u,true},
                                           std::tuple{6u,6u,false}}) {
        const auto guarded=(word(3,1,2,4,true,false,true)&
                            ~(std::uint64_t{15}<<16))|(std::uint64_t{guard}<<16);
        const auto result=decode_forward(guarded,Stage::Vertex,guard);
        const auto* p=std::get_if<mx::ForwardProgram>(&result);ARMIGO_CHECK(p);if(!p)continue;
        ARMIGO_CHECK_EQ(p->instructions.size(),2u);
        const auto* node=std::get_if<mx::Instruction>(&p->instructions.front());
        ARMIGO_CHECK(node);if(!node)continue;
        ARMIGO_CHECK(node->family==Family::FloatFmaRegister);
        ARMIGO_CHECK_EQ(node->predicate.index,static_cast<std::uint8_t>(index));
        ARMIGO_CHECK_EQ(node->predicate.inverted,inverted);
        const auto& v=std::get<Value>(node->payload);
        ARMIGO_CHECK(v.negate_b&&v.saturate&&!v.negate_c);
        ARMIGO_CHECK_EQ(node->original_word,guarded);
    }
    // The strict path still rejects the same guarded word.
    const auto strict=decode(word(),Stage::Vertex,1);
    ARMIGO_CHECK(std::holds_alternative<mx::DecodeError>(strict));
}

ARMIGO_TEST(ffma_register, emits_fused_helper_and_saturation) {
    gm::EmitOptions options;options.vertex_depth=gm::VertexDepthConvention::NegativeOneToOne;
    const auto lowered=gx::lower(program(value(3,true,false,true)));
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if(!std::holds_alternative<gx::Program>(lowered))return;
    const auto emitted=gm::emit(std::get<gx::Program>(lowered),options);
    const auto* stage=std::get_if<gm::EmittedStage>(&emitted);ARMIGO_CHECK(stage);if(!stage)return;
    ARMIGO_CHECK(stage->source.find("= armigo_fma_rne_ftz((v")!=std::string::npos);
    const auto call=stage->source.find("= armigo_fma_rne_ftz(");
    if(call!=std::string::npos)
        ARMIGO_CHECK(stage->source.substr(call,stage->source.find('\n',call)-call)
                         .find("^ 0x80000000u")!=std::string::npos);
    ARMIGO_CHECK(stage->source.find("armigo_saturate_float32(v")!=std::string::npos);
    ARMIGO_CHECK(stage->constant_buffers.empty());
    // The plain profile emits the fused helper with a bare first operand and
    // no sign-flipped call site.
    const auto plain=gx::lower(program(value()));
    if(!std::holds_alternative<gx::Program>(plain))return;
    const auto plain_emitted=gm::emit(std::get<gx::Program>(plain),options);
    const auto* plain_stage=std::get_if<gm::EmittedStage>(&plain_emitted);
    ARMIGO_CHECK(plain_stage);if(!plain_stage)return;
    ARMIGO_CHECK(plain_stage->source.find("= armigo_fma_rne_ftz(v")!=std::string::npos);
    ARMIGO_CHECK(plain_stage->source.find("= armigo_fma_rne_ftz((v")==std::string::npos);
}
}  // namespace armigo::shader::test
