// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/maxwell.h>
#include <armigo/shader/graphics.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/fma.h>
#include "test_support.h"
#include <array>
#include <bit>
#include <vector>
namespace armigo::shader::test {
namespace {
namespace mx=maxwell; namespace gx=graphics; namespace gm=graphics_msl;
using Family=mx::OpcodeFamily; using Stage=shader_input::Stage;
using Value=mx::FloatFmaRegisterConstant;
std::uint64_t word(unsigned d=3,unsigned a=1,unsigned b=2,unsigned binding=5,unsigned offset=0) {
    return (std::uint64_t{0x5180}<<48)|(std::uint64_t{1}<<53)|(std::uint64_t{7}<<16)|
        d|(std::uint64_t{a}<<8)|(std::uint64_t{b}<<39)|
        (std::uint64_t{binding}<<34)|(std::uint64_t{offset}<<20);
}
void put(std::vector<std::uint8_t>& bytes,unsigned at,std::uint64_t value,unsigned count) {
    for(unsigned i=0;i<count;++i)bytes[at+i]=std::uint8_t(value>>(8*i));
}
mx::DecodeResult decode(std::uint64_t value,Stage stage=Stage::Vertex) {
    std::vector<std::uint8_t> control(0x718),code(0xa0);
    put(control,0,0x98761234,4);put(control,4,1,4);put(control,8,14,4);
    put(control,0x20,control.size(),4);put(control,0x700,code.size(),4);
    put(control,0x704,code.size(),4);put(control,0x708,0x30,4);
    put(control,0x714,static_cast<unsigned>(stage),4);put(code,0,0x12345678,4);
    put(code,0x30,shader_input::sph_stage_type_for(stage)<<10,4);
    put(code,0x80,0x1122334455667788ull,8);put(code,0x88,value,8);
    put(code,0x90,(std::uint64_t{0xe300}<<48)|(std::uint64_t{7}<<16)|15,8);
    return mx::decode({control.data(),control.size()},{code.data(),code.size()});
}
void append(mx::DecodedProgram& p,Family family,mx::InstructionPayload payload) {
    const auto i=p.instructions.size();
    p.instructions.push_back({std::uint32_t((i/3)*32+(i%3+1)*8),0,family,{7,false},std::move(payload)});
}
void mov(mx::DecodedProgram& p,unsigned reg,std::uint32_t bits) {
    append(p,Family::Mov32Immediate,mx::Mov32Immediate{{std::uint8_t(reg)},15,bits});
}
Value value(unsigned d=3) {
    return {{std::uint8_t(d)},{1},{2},{5,0},mx::RoundingMode::NearestEven,mx::FlushMode::FlushToZero};
}
mx::DecodedProgram program(Value v,std::uint32_t a=0x40000000,std::uint32_t b=0x40400000,
                           Stage stage=Stage::Vertex,bool init_b=true) {
    mx::DecodedProgram p;p.stage=stage;
    const auto header=shader_input::sph_stage_type_for(stage)<<10;
    for(unsigned i=0;i<4;++i)p.sph[i]=std::byte(header>>(i*8));
    if(stage==Stage::Vertex)p.sph[0x35]=std::byte{0xf0};
    else p.sph[0x48]=std::byte{0xf};
    mov(p,1,a);if(init_b)mov(p,2,b);
    if(stage==Stage::Fragment){mov(p,0,0);mov(p,3,0x3f800000);}
    append(p,Family::FloatFmaRegisterConstant,v);
    if(stage==Stage::Vertex)for(unsigned c=28;c<32;++c)
        append(p,Family::AttributeStore,mx::AttributeAccess{
            {std::uint8_t(v.destination.is_zero()?1:v.destination.index)},{255},{255},
            std::uint16_t(c*4),0,false});
    append(p,Family::Exit,mx::Exit{15});return p;
}
std::uint32_t evaluate(const mx::DecodedProgram& p,std::uint32_t c) {
    const auto lowered=gx::lower(p);ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if(!std::holds_alternative<gx::Program>(lowered))return 0;
    const std::array data{c};gx::Inputs inputs;inputs.constant_buffers[5]=data;
    const auto result=gx::evaluate(std::get<gx::Program>(lowered),inputs);
    const auto* evaluated=std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(evaluated);if(!evaluated)return 0;
    for(const auto& output:evaluated->outputs)if(output.component==0)return output.bits;
    ARMIGO_CHECK(false);return 0;
}
void error(const mx::DecodedProgram& p,gx::Error::Code expected) {
    const auto result=gx::lower(p);const auto* e=std::get_if<gx::Error>(&result);
    ARMIGO_CHECK(e);if(e)ARMIGO_CHECK(e->code==expected);
}
}  // namespace

ARMIGO_TEST(fma_rc, decodes_register_multiplicand_and_constant_addend) {
    for(const auto stage:{Stage::Vertex,Stage::Fragment}) {
        const auto encoded=word(7,9,11,6,0x123);
        const auto result=decode(encoded,stage);
        const auto* p=std::get_if<mx::DecodedProgram>(&result);ARMIGO_CHECK(p);if(!p)continue;
        const auto& i=p->instructions.front();ARMIGO_CHECK(i.family==Family::FloatFmaRegisterConstant);
        ARMIGO_CHECK_EQ(i.pc,8u);ARMIGO_CHECK_EQ(i.original_word,encoded);
        const auto* v=std::get_if<Value>(&i.payload);ARMIGO_CHECK(v);if(!v)continue;
        ARMIGO_CHECK(v->destination.index==7&&v->source_a.index==9&&v->source_b.index==11);
        ARMIGO_CHECK(v->source_c.binding==6&&v->source_c.byte_offset==0x48c);
        ARMIGO_CHECK(v->rounding==mx::RoundingMode::NearestEven&&v->flush==mx::FlushMode::FlushToZero);
    }
    const auto result=decode(word(255,255,255,17,0x3fff));
    const auto* p=std::get_if<mx::DecodedProgram>(&result);ARMIGO_CHECK(p);
    if(p){const auto& v=std::get<Value>(p->instructions.front().payload);
        ARMIGO_CHECK(v.destination.is_zero()&&v.source_a.is_zero()&&v.source_b.is_zero());
        ARMIGO_CHECK(v.source_c.binding==17&&v.source_c.byte_offset==65532);}
}
ARMIGO_TEST(fma_rc, rejects_modifiers_predicates_and_bindings) {
    const std::array fields{mx::DecodeField::ConditionCode,mx::DecodeField::NegateB,
        mx::DecodeField::NegateC,mx::DecodeField::Saturate,mx::DecodeField::Rounding,
        mx::DecodeField::Rounding,mx::DecodeField::FlushMode,mx::DecodeField::FlushMode};
    for(unsigned bit=47;bit<=54;++bit) {
        if(bit==48)continue;
        const auto result=decode(word()^(std::uint64_t{1}<<bit));
        const auto* e=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(e);
        if(e){ARMIGO_CHECK(e->code==mx::DecodeError::Code::UnsupportedModifier);
            ARMIGO_CHECK(e->field==fields[bit-47]);ARMIGO_CHECK_EQ(e->pc,8u);}
    }
    for(unsigned binding:{18u,31u}) {
        const auto result=decode(word(3,1,2,binding));
        const auto* e=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(e);
        if(e)ARMIGO_CHECK(e->field==mx::DecodeField::ConstantBinding&&e->expected==17);
    }
    for(unsigned pred:{0u,15u}) {
        const auto result=decode((word()&~(std::uint64_t{15}<<16))|(std::uint64_t{pred}<<16));
        const auto* e=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(e);
        if(e)ARMIGO_CHECK(e->code==mx::DecodeError::Code::UnsupportedPredicate);
    }
    const auto neighbour=decode((word()&~(std::uint64_t{0xffff}<<48))|(std::uint64_t{0x5100}<<48));
    ARMIGO_CHECK(std::holds_alternative<mx::DecodeError>(neighbour));
    const auto legacy=decode((word()&~(std::uint64_t{0xffff}<<48))|(std::uint64_t{0x49a0}<<48));
    const auto* p=std::get_if<mx::DecodedProgram>(&legacy);ARMIGO_CHECK(p);
    if(p)ARMIGO_CHECK(p->instructions.front().family==Family::FloatFmaConstantRegister);
}
ARMIGO_TEST(fma_rc, preserves_roles_aliasing_zero_and_fusion) {
    for(unsigned destination:{1u,2u,3u})
        ARMIGO_CHECK_EQ(evaluate(program(value(destination)),0x40800000),0x41200000u);
    auto rz=value();rz.source_a={255};
    ARMIGO_CHECK_EQ(evaluate(program(rz),0x40800000),0x40800000u);
    rz=value();rz.source_b={255};
    ARMIGO_CHECK_EQ(evaluate(program(rz),0x40800000),0x40800000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(255)),0x40800000),0x40000000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(),0x3f800001,0x3f7fffff),0xbf800000),0x337ffffeu);
    ARMIGO_CHECK_EQ(evaluate(program(value(),0x80000000,0x40400000),0x80000000),0x80000000u);
    ARMIGO_CHECK_EQ(evaluate(program(value(),0,0x3f800000),1),0u);
    ARMIGO_CHECK_EQ(evaluate(program(value(0),0x40000000,0x40400000,Stage::Fragment),0x40800000),0x41200000u);
}
ARMIGO_TEST(fma_rc, negate_b_preserves_the_established_fused_contract) {
    const auto decoded=decode(word()|(std::uint64_t{1}<<48));
    const auto* p=std::get_if<mx::DecodedProgram>(&decoded);ARMIGO_CHECK(p);
    if(p)ARMIGO_CHECK(std::get<Value>(p->instructions.front().payload).negate_b);
    auto v=value();v.negate_b=true;
    ARMIGO_CHECK_EQ(evaluate(program(v),0x40800000),0xc0000000u);
    constexpr std::array values{0u,0x80000000u,1u,0x80000001u,0x3f800000u,
        0xbf800000u,0x7f800000u,0xff800000u,0x7fc12345u,0x7f811234u};
    for(auto a:values)for(auto b:values)for(auto c:values) {
        const auto expected=shader::fma_rne_ftz(a,b^0x80000000u,c);
        ARMIGO_CHECK_EQ(evaluate(program(v,a,b),c),expected);
    }
}
ARMIGO_TEST(fma_rc, forged_payload_modes_and_operands_fail) {
    auto p=program(value());
    for(auto& i:p.instructions)if(i.family==Family::FloatFmaRegisterConstant)
        i.payload=mx::MovConstant{{3},{5,0},15};
    error(p,gx::Error::Code::InvalidInstruction);
    auto v=value();v.rounding=mx::RoundingMode::Zero;
    error(program(v),gx::Error::Code::UnsupportedModifier);
    v=value();v.flush=mx::FlushMode::None;error(program(v),gx::Error::Code::UnsupportedModifier);
    v=value();v.source_c.binding=18;error(program(v),gx::Error::Code::UnsupportedModifier);
    for(unsigned offset:{2u,65536u}) {
        v=value();v.source_c.byte_offset=offset;error(program(v),gx::Error::Code::UnsupportedModifier);
    }
    error(program(value(),0x40000000,0x40400000,Stage::Vertex,false),gx::Error::Code::UninitializedRegister);
    v=value();v.source_c.byte_offset=4;
    const auto lowered=gx::lower(program(v));ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if(std::holds_alternative<gx::Program>(lowered)) {
        const std::array data{0x40800000u};gx::Inputs inputs;inputs.constant_buffers[5]=data;
        const auto result=gx::evaluate(std::get<gx::Program>(lowered),inputs);
        const auto* e=std::get_if<gx::Error>(&result);ARMIGO_CHECK(e);
        if(e)ARMIGO_CHECK(e->code==gx::Error::Code::ConstantBounds);
    }
}
ARMIGO_TEST(fma_rc, reflects_addend_extent_and_emits_fused_helper) {
    auto v=value();v.source_c.byte_offset=8;v.negate_b=true;
    const auto lowered=gx::lower(program(v));ARMIGO_CHECK(std::holds_alternative<gx::Program>(lowered));
    if(!std::holds_alternative<gx::Program>(lowered))return;
    gm::EmitOptions options;options.vertex_depth=gm::VertexDepthConvention::NegativeOneToOne;
    const auto emitted=gm::emit(std::get<gx::Program>(lowered),options);
    const auto* stage=std::get_if<gm::EmittedStage>(&emitted);ARMIGO_CHECK(stage);if(!stage)return;
    ARMIGO_CHECK_EQ(stage->constant_buffers.size(),1u);
    if(!stage->constant_buffers.empty()) {
        ARMIGO_CHECK(stage->constant_buffers[0].guest_binding==5);
        ARMIGO_CHECK_EQ(stage->constant_buffers[0].minimum_size,12u);
    }
    ARMIGO_CHECK(stage->source.find("= armigo_fma_rne_ftz(")!=std::string::npos);
    ARMIGO_CHECK(stage->source.find("= armigo_fma_rne_ftz((v")!=std::string::npos);
    const auto call=stage->source.find("= armigo_fma_rne_ftz(");
    if(call!=std::string::npos)
        ARMIGO_CHECK(stage->source.substr(call,stage->source.find('\n',call)-call)
                         .find("^ 0x80000000u")!=std::string::npos);
}
}  // namespace armigo::shader::test
