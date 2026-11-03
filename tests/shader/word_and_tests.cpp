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
#include "test_support.h"
#include <array>
#include <vector>
namespace armigo::shader::test {
namespace {
namespace mx=maxwell;namespace gx=graphics;namespace gm=graphics_msl;
using Family=mx::OpcodeFamily;
std::uint64_t word(unsigned destination=3,unsigned source=1,std::uint32_t mask=0xa5c312ef) {
    return (std::uint64_t{0x0400}<<48)|(std::uint64_t{7}<<16)|
        (std::uint64_t{mask}<<20)|(std::uint64_t{source}<<8)|destination;
}
void put(std::vector<std::uint8_t>& bytes,unsigned at,std::uint64_t value,unsigned n) {
    for(unsigned i=0;i<n;++i)bytes[at+i]=std::uint8_t(value>>(8*i));
}
mx::DecodeResult decode(std::uint64_t instruction) {
    std::vector<std::uint8_t> control(0x718),code(0xa0);
    put(control,0,0x98761234,4);put(control,4,1,4);put(control,8,14,4);
    put(control,0x20,control.size(),4);put(control,0x700,code.size(),4);
    put(control,0x704,code.size(),4);put(control,0x708,0x30,4);put(control,0x714,0,4);
    put(code,0,0x12345678,4);put(code,0x30,shader_input::sph_stage_type_for(shader_input::Stage::Vertex)<<10,4);
    put(code,0x88,instruction,8);
    put(code,0x90,(std::uint64_t{0xe300}<<48)|(std::uint64_t{7}<<16)|15,8);
    return mx::decode({control.data(),control.size()},{code.data(),code.size()});
}
void append(mx::DecodedProgram& p,Family family,mx::InstructionPayload payload) {
    const auto i=p.instructions.size();
    p.instructions.push_back({std::uint32_t((i/3)*32+(i%3+1)*8),0,family,{7,false},std::move(payload)});
}
mx::DecodedProgram program(std::uint32_t input,std::uint32_t mask,unsigned destination=3,unsigned source=1) {
    mx::DecodedProgram p;p.stage=shader_input::Stage::Vertex;
    const auto h=shader_input::sph_stage_type_for(p.stage)<<10;
    for(unsigned i=0;i<4;++i)p.sph[i]=std::byte(h>>(8*i));
    p.sph[0x35]=std::byte{0xf0};
    append(p,Family::Mov32Immediate,mx::Mov32Immediate{{1},15,input});
    append(p,Family::WordAndImmediate,mx::WordAndImmediate{
        {std::uint8_t(destination)},{std::uint8_t(source)},mask});
    for(unsigned c=28;c<32;++c)
        append(p,Family::AttributeStore,mx::AttributeAccess{
            {std::uint8_t(destination==255?1:destination)},{255},{255},std::uint16_t(c*4),0,false});
    append(p,Family::Exit,mx::Exit{15});return p;
}
std::uint32_t evaluate(const mx::DecodedProgram& p) {
    const auto lowered=gx::lower(p);const auto* value=std::get_if<gx::Program>(&lowered);
    ARMIGO_CHECK(value);if(!value)return 0;
    const auto result=gx::evaluate(*value,{});const auto* output=std::get_if<gx::Evaluation>(&result);
    ARMIGO_CHECK(output);if(!output)return 0;
    return output->outputs.front().bits;
}
}  // namespace

ARMIGO_TEST(word_and, decodes_full_mask_and_register_roles) {
    for(auto mask:{0u,0xffffffffu,0xa5c312efu}) {
        const auto encoded=word(7,9,mask);const auto result=decode(encoded);
        const auto* p=std::get_if<mx::DecodedProgram>(&result);ARMIGO_CHECK(p);if(!p)continue;
        const auto& i=p->instructions.front();ARMIGO_CHECK(i.family==Family::WordAndImmediate);
        ARMIGO_CHECK_EQ(i.pc,8u);ARMIGO_CHECK_EQ(i.original_word,encoded);
        const auto& v=std::get<mx::WordAndImmediate>(i.payload);
        ARMIGO_CHECK(v.destination.index==7&&v.source.index==9);ARMIGO_CHECK_EQ(v.mask,mask);
    }
    const auto result=decode(word(255,255));const auto* p=std::get_if<mx::DecodedProgram>(&result);
    ARMIGO_CHECK(p);if(p){const auto& v=std::get<mx::WordAndImmediate>(p->instructions.front().payload);
        ARMIGO_CHECK(v.destination.is_zero()&&v.source.is_zero());}
}
ARMIGO_TEST(word_and, rejects_other_operations_and_active_controls) {
    for(unsigned shift:{52u,53u,54u,55u,56u,57u}) {
        const auto result=decode(word()|(std::uint64_t{1}<<shift));
        const auto* e=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(e);
        if(e)ARMIGO_CHECK(e->code==mx::DecodeError::Code::UnsupportedModifier&&e->pc==8);
    }
    for(unsigned pred:{0u,15u}) {
        const auto result=decode((word()&~(std::uint64_t{15}<<16))|(std::uint64_t{pred}<<16));
        const auto* e=std::get_if<mx::DecodeError>(&result);ARMIGO_CHECK(e);
        if(e)ARMIGO_CHECK(e->code==mx::DecodeError::Code::UnsupportedPredicate);
    }
}
ARMIGO_TEST(word_and, preserves_raw_bits_without_floating_point_processing) {
    ARMIGO_CHECK_EQ(evaluate(program(0x01234567,0xa5c312ef)),0x01030067u);
    for(auto bits:{0u,1u,0x80000000u,0x80000001u,0x7f811234u,0x7fc12345u,0xffffffffu}) {
        ARMIGO_CHECK_EQ(evaluate(program(bits,0xffffffff)),bits);
        ARMIGO_CHECK_EQ(evaluate(program(bits,0)),0u);
        ARMIGO_CHECK_EQ(evaluate(program(bits,0x7fffffff)),bits&0x7fffffffu);
    }
    ARMIGO_CHECK_EQ(evaluate(program(0xff811234,0xffffffff,1)),0xff811234u);
    ARMIGO_CHECK_EQ(evaluate(program(0x12345678,0xffffffff,3,255)),0u);
    ARMIGO_CHECK_EQ(evaluate(program(0x12345678,0xffffffff,255)),0x12345678u);
}
ARMIGO_TEST(word_and, forged_payload_and_undefined_source_fail) {
    auto p=program(0,0);
    for(auto& i:p.instructions)if(i.family==Family::WordAndImmediate)
        i.payload=mx::MovRegister{{3},{1},15};
    const auto wrong=gx::lower(p);const auto* error=std::get_if<gx::Error>(&wrong);
    ARMIGO_CHECK(error);if(error)ARMIGO_CHECK(error->code==gx::Error::Code::InvalidInstruction);
    const auto undefined=gx::lower(program(0,0,3,7));error=std::get_if<gx::Error>(&undefined);
    ARMIGO_CHECK(error);if(error)ARMIGO_CHECK(error->code==gx::Error::Code::UninitializedRegister);
}
ARMIGO_TEST(word_and, emits_uint_and_validates_scalar_dependencies) {
    auto lowered=gx::lower(program(0xff811234,0xff00ff00));
    auto* p=std::get_if<gx::Program>(&lowered);ARMIGO_CHECK(p);if(!p)return;
    gm::EmitOptions options;options.vertex_depth=gm::VertexDepthConvention::NegativeOneToOne;
    const auto emitted=gm::emit(*p,options);const auto* stage=std::get_if<gm::EmittedStage>(&emitted);
    ARMIGO_CHECK(stage);if(stage) {
        const auto at=stage->source.find(" & 0xff00ff00u;");
        ARMIGO_CHECK(at!=std::string::npos);
        if(at!=std::string::npos) {
            const auto start=stage->source.rfind('\n',at);
            const auto line=stage->source.substr(start==std::string::npos?0:start,at-start);
            ARMIGO_CHECK(line.find("const uint")!=std::string::npos);
            ARMIGO_CHECK(line.find("flush")==std::string::npos&&line.find("as_type")==std::string::npos);
        }
    }
    auto* nodes=const_cast<gx::Node*>(p->nodes().data());
    for(std::size_t i=0;i<p->nodes().size();++i)if(auto* op=std::get_if<gx::WordAnd>(&nodes[i].operation))
        op->operand=gx::ScalarId{0xffffffffu};
    const auto invalid=gm::emit(*p,options);ARMIGO_CHECK(std::holds_alternative<gm::EmitError>(invalid));
    const auto evaluated=gx::evaluate(*p,{});const auto* e=std::get_if<gx::Error>(&evaluated);
    ARMIGO_CHECK(e);if(e)ARMIGO_CHECK(e->code==gx::Error::Code::InvalidInstruction);
}
}  // namespace armigo::shader::test
