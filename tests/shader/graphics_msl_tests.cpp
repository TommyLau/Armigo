// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "test_support.h"

#include <armigo/shader/graphics_msl.h>

#include <bit>
#include <string>

namespace armigo::shader::test {
namespace {

namespace gm = graphics_msl;
namespace gx = graphics;
namespace mx = maxwell;
using Stage = shader_input::Stage;
using Family = mx::OpcodeFamily;

void put(mx::DecodedProgram& program, unsigned at, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        program.sph[at + i] = std::byte(value >> (i * 8));
    }
}
mx::DecodedProgram base(Stage stage) {
    mx::DecodedProgram result;
    result.stage = stage;
    put(result, 0, shader_input::sph_stage_type_for(stage) << 10);
    return result;
}
void attribute(mx::DecodedProgram& program, unsigned component,
               bool output = false, unsigned interpolation = 2) {
    unsigned at = 0, bit = 0, value = 1;
    if (component >= 28 && component < 32) {
        at = output ? 0x35 : 0x17;
        bit = component - 24;
    } else if (program.stage == Stage::Vertex) {
        at = (output ? 0x36 : 0x18) + (component - 32) / 8;
        bit = (component - 32) % 8;
    } else {
        at = 0x18 + (component - 32) / 4;
        bit = ((component - 32) % 4) * 2;
        value = interpolation;
    }
    program.sph[at] |= std::byte(value << bit);
}
void append(mx::DecodedProgram& program, Family family,
            mx::InstructionPayload payload) {
    const auto i = program.instructions.size();
    program.instructions.push_back({
        static_cast<std::uint32_t>((i / 3) * 32 + (i % 3 + 1) * 8),
        0, family, {7, false}, std::move(payload)});
}
void mov(mx::DecodedProgram& p, std::uint8_t r, std::uint32_t value) {
    append(p, Family::Mov32Immediate, mx::Mov32Immediate{{r}, 15, value});
}
void reg_move(mx::DecodedProgram& p, std::uint8_t destination,
              std::uint8_t source, std::uint8_t write_mask = 15) {
    append(p, Family::MovRegister,
           mx::MovRegister{{destination}, {source}, write_mask});
}
void load(mx::DecodedProgram& p, std::uint8_t r, unsigned component) {
    append(p, Family::AttributeLoad,
           mx::AttributeAccess{{r}, {255}, {255},
                               static_cast<std::uint16_t>(component * 4), 0,
                               false});
}
void store(mx::DecodedProgram& p, std::uint8_t r, unsigned component) {
    append(p, Family::AttributeStore,
           mx::AttributeAccess{{r}, {255}, {255},
                               static_cast<std::uint16_t>(component * 4), 0,
                               false});
}
void exit(mx::DecodedProgram& p) { append(p, Family::Exit, mx::Exit{15}); }

gx::Program lower(mx::DecodedProgram decoded) {
    const gx::LowerResult result = gx::lower(decoded);
    ARMIGO_CHECK(std::holds_alternative<gx::Program>(result));
    return std::get<gx::Program>(result);
}

gx::Program vertex_program(bool full_position = true, bool second_attribute = false) {
    auto p = base(Stage::Vertex);
    for (unsigned c = 32; c < 35; ++c) attribute(p, c);
    for (unsigned c = 28; c < (full_position ? 32u : 31u); ++c) attribute(p, c, true);
    attribute(p, 32, true); attribute(p, 33, true);
    load(p, 1, 32); load(p, 2, 33); load(p, 3, 34);
    if (second_attribute) {
        attribute(p, 36);
        load(p, 13, 36);
    }
    mov(p, 4, std::bit_cast<std::uint32_t>(0.5f));
    append(p, Family::FloatSetRegister,
           mx::FloatSetRegister{{5}, {1}, {255}, {7, false},
                                mx::OrderedComparison::LessThan, false, true,
                                mx::FlushMode::FlushToZero});
    append(p, Family::FloatAddRegister,
           mx::FloatAdd{{6}, {1}, mx::RegisterOperand{2},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero, false});
    append(p, Family::FloatAddConstant,
           mx::FloatAdd{{7}, {1}, mx::ConstantBufferOperand{3, 20},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero, false});
    append(p, Family::FloatMultiplyRegister,
           mx::FloatMultiply{{8}, {6}, mx::RegisterOperand{3},
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero});
    append(p, Family::FloatMultiplyRegister,
           mx::FloatMultiply{{9}, {6}, mx::RegisterOperand{3},
                             mx::RoundingMode::Zero,
                             mx::FlushMode::FlushToZero});
    append(p, Family::FloatFmaImmediate,
           mx::FloatFma{{10}, {1}, mx::ShortFloatImmediate{0x3f000000}, {4},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero});
    append(p, Family::FloatFmaConstantRegister,
           mx::FloatFma{{11}, {1}, mx::ConstantBufferOperand{3, 0}, {4},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero});
    append(p, Family::MultiFunction, mx::Reciprocal{{12}, {4}});
    store(p, 6, 28); store(p, 8, 29); store(p, 10, 30);
    if (full_position) store(p, 11, 31);
    store(p, 5, 32); store(p, 12, 33);
    exit(p);
    return lower(std::move(p));
}

gx::Program fragment_program() {
    auto p = base(Stage::Fragment);
    attribute(p, 31);
    attribute(p, 32, false, 2);
    put(p, 0x48, 0xf);
    append(p, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{4}, {255}, {255}, 32, 0x7ff});
    append(p, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{5}, {255}, {255}, 31, 0x7ff});
    append(p, Family::MultiFunction, mx::Reciprocal{{6}, {5}});
    append(p, Family::FloatMultiplyRegister,
           mx::FloatMultiply{{7}, {4}, mx::RegisterOperand{6},
                             mx::RoundingMode::Zero,
                             mx::FlushMode::FlushToZero});
    append(p, Family::TextureSample,
           mx::TextureSample2D{{0}, {2}, {4}, {6}, 32, true});
    exit(p);
    return lower(std::move(p));
}

const gm::EmittedStage* accepted(const gm::EmitResult& result) {
    const auto* stage = std::get_if<gm::EmittedStage>(&result);
    ARMIGO_CHECK(stage != nullptr);
    return stage;
}
void rejected(const gm::EmitResult& result, gm::EmitError::Code code) {
    const auto* error = std::get_if<gm::EmitError>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error) ARMIGO_CHECK(error->code == code);
}
bool contains(const std::string& text, const std::string& value) {
    return text.find(value) != std::string::npos;
}

}  // namespace

ARMIGO_TEST(graphics_msl, emits_vertex_ssa_and_reflection) {
    const gx::Program program = vertex_program();
    gm::EmitOptions options;
    options.vertex_attributes.push_back({0, 5, gm::VertexFormat::Float32x3});
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const gm::EmitResult result = gm::emit(program, options);
    const gm::EmittedStage* stage = accepted(result);
    if (!stage) return;
    ARMIGO_CHECK(stage->entry_point == "armigo_vertex_main");
    ARMIGO_CHECK(stage->constant_buffers.size() == 1);
    ARMIGO_CHECK(stage->constant_buffers[0].guest_binding == 3);
    ARMIGO_CHECK_EQ(stage->constant_buffers[0].minimum_size, 24u);
    ARMIGO_CHECK(stage->interface.position_output_mask == 0xf);
    ARMIGO_CHECK(stage->vertex_depth == gm::VertexDepthConvention::NegativeOneToOne);
    ARMIGO_CHECK(stage->interface.generic_inputs[0].component_mask == 7);
    ARMIGO_CHECK(stage->interface.generic_outputs[0].component_mask == 3);
    ARMIGO_CHECK(contains(stage->source, "float4 attr0 [[attribute(5)]]"));
    ARMIGO_CHECK(contains(stage->source, "out.generic0 = float4(0.0f)"));
    ARMIGO_CHECK(contains(stage->source, "armigo_add_rn_ftz"));
    ARMIGO_CHECK(contains(stage->source, "armigo_mul_rn_ftz"));
    ARMIGO_CHECK(contains(stage->source, "armigo_mul_rtz_ftz"));
    ARMIGO_CHECK(contains(stage->source, "armigo_fma_rne_ftz"));
    ARMIGO_CHECK(!contains(stage->source, "float result = fma("));
    ARMIGO_CHECK(contains(stage->source, "armigo_reciprocal_rne"));
    ARMIGO_CHECK(contains(stage->source, "#pragma METAL fp contract(off)"));
    ARMIGO_CHECK(contains(stage->source,
                          "armigo_depth_sum = out.position.z + out.position.w"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.z = armigo_depth_sum * 0.5f"));
    ARMIGO_CHECK(stage->source == std::get<gm::EmittedStage>(gm::emit(program, options)).source);
    options.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    const gm::EmitResult zero_to_one_result = gm::emit(program, options);
    const gm::EmittedStage* zero_to_one = accepted(zero_to_one_result);
    if (zero_to_one) {
        ARMIGO_CHECK(zero_to_one->vertex_depth == gm::VertexDepthConvention::ZeroToOne);
        ARMIGO_CHECK(!contains(zero_to_one->source, "armigo_depth_sum"));
    }
}

ARMIGO_TEST(graphics_msl, emits_fragment_interface_texture_and_perspective) {
    const gx::Program program = fragment_program();
    rejected(gm::emit(program), gm::EmitError::Code::PositionConventionRequired);
    gm::EmitOptions options;
    options.fragment_position =
        gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    const gm::EmitResult result = gm::emit(program, options);
    const gm::EmittedStage* stage = accepted(result);
    if (!stage) return;
    ARMIGO_CHECK(stage->entry_point == "armigo_fragment_main");
    ARMIGO_CHECK(stage->textures.size() == 1);
    ARMIGO_CHECK_EQ(stage->textures[0].handle_byte_offset, 32u);
    ARMIGO_CHECK(stage->textures[0].texture_index == 0 &&
                 stage->textures[0].sampler_index == 0);
    ARMIGO_CHECK(stage->interface.position_input_mask == 8);
    ARMIGO_CHECK(stage->interface.color_output_mask == 0xf);
    ARMIGO_CHECK(stage->interface.generic_inputs[0].interpolation ==
                 gx::Interpolation::Perspective);
    ARMIGO_CHECK(contains(stage->source, "float4 position [[position]]"));
    ARMIGO_CHECK(contains(stage->source, "armigo_multiply_rne"));
    ARMIGO_CHECK(contains(stage->source, ".sample(samp0, float2("));
    ARMIGO_CHECK(contains(stage->source, "texture2d<float> tex0 [[texture(0)]]"));
    ARMIGO_CHECK(!contains(stage->source, "v4 * v5"));
}

ARMIGO_TEST(graphics_msl, emits_short_immediate_add_constant_bits) {
    auto p = base(Stage::Vertex);
    attribute(p, 32);
    for (unsigned component = 28; component < 32; ++component) {
        attribute(p, component, true);
    }
    load(p, 1, 32);
    append(p, Family::FloatAddImmediate,
           mx::FloatAdd{{2}, {1}, mx::ShortFloatImmediate{0xbf800000u},
                        mx::RoundingMode::NearestEven,
                        mx::FlushMode::FlushToZero, false});
    for (unsigned component = 28; component < 32; ++component) {
        store(p, 2, component);
    }
    exit(p);
    gm::EmitOptions options;
    options.vertex_attributes.push_back({0, 5, gm::VertexFormat::Float32x3});
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const gm::EmitResult result = gm::emit(lower(std::move(p)), options);
    const gm::EmittedStage* stage = accepted(result);
    if (!stage) return;
    ARMIGO_CHECK(contains(stage->source, "const uint v2 = 0xbf800000u;"));
    ARMIGO_CHECK(contains(stage->source, "armigo_add_rn_ftz(v1, v2)"));
}

ARMIGO_TEST(graphics_msl, emits_short_immediate_multiply_literal_and_helpers) {
    auto p = base(Stage::Vertex);
    attribute(p, 32);
    for (unsigned component = 28; component < 32; ++component) {
        attribute(p, component, true);
    }
    attribute(p, 33, true);
    load(p, 1, 32);
    append(p, Family::FloatMultiplyImmediate,
           mx::FloatMultiply{{2}, {1}, mx::ShortFloatImmediate{0xbf801000u},
                             mx::RoundingMode::NearestEven,
                             mx::FlushMode::FlushToZero});
    for (unsigned component = 28; component < 32; ++component) {
        store(p, 2, component);
    }
    load(p, 3, 32);
    append(p, Family::FloatMultiplyImmediate,
           mx::FloatMultiply{{4}, {3}, mx::ShortFloatImmediate{0x40000000u},
                             mx::RoundingMode::Zero,
                             mx::FlushMode::FlushToZero});
    store(p, 4, 33);
    exit(p);
    gm::EmitOptions options;
    options.vertex_attributes.push_back({0, 5, gm::VertexFormat::Float32x3});
    options.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    const gm::EmitResult result = gm::emit(lower(std::move(p)), options);
    const gm::EmittedStage* stage = accepted(result);
    if (!stage) return;
    ARMIGO_CHECK(contains(stage->source, "const uint v2 = 0xbf801000u;"));
    ARMIGO_CHECK(
        contains(stage->source, "const uint v3 = armigo_mul_rn_ftz(v1, v2);"));
    ARMIGO_CHECK(contains(stage->source, "const uint v5 = 0x40000000u;"));
    ARMIGO_CHECK(
        contains(stage->source, "const uint v6 = armigo_mul_rtz_ftz(v4, v5);"));
}

ARMIGO_TEST(graphics_msl, rejects_missing_formats_interfaces_and_limits) {
    const gx::Program vertex = vertex_program();
    rejected(gm::emit(vertex), gm::EmitError::Code::DepthConventionRequired);
    gm::EmitOptions no_format;
    no_format.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    rejected(gm::emit(vertex, no_format), gm::EmitError::Code::MissingVertexFormat);
    no_format.vertex_depth = static_cast<gm::VertexDepthConvention>(99);
    rejected(gm::emit(vertex, no_format), gm::EmitError::Code::DepthConventionRequired);
    gm::EmitOptions duplicate;
    duplicate.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    duplicate.vertex_attributes = {{0, 0, gm::VertexFormat::Float32x3},
                                   {0, 1, gm::VertexFormat::Float32x3}};
    rejected(gm::emit(vertex, duplicate), gm::EmitError::Code::DuplicateVertexFormat);
    duplicate.vertex_attributes = {{0, 5, gm::VertexFormat::Float32x3},
                                   {1, 5, gm::VertexFormat::Float32x3}};
    rejected(gm::emit(vertex_program(true, true), duplicate),
             gm::EmitError::Code::DuplicateVertexFormat);
    gm::EmitOptions unused;
    unused.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    unused.vertex_attributes = {{0, 0, gm::VertexFormat::Float32x3},
                                {1, 1, gm::VertexFormat::Float32x3}};
    rejected(gm::emit(vertex, unused), gm::EmitError::Code::UnsupportedVertexFormat);
    gm::EmitOptions tiny;
    tiny.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    tiny.vertex_attributes = {{0, 0, gm::VertexFormat::Float32x3}};
    tiny.max_source_bytes = 32;
    rejected(gm::emit(vertex, tiny), gm::EmitError::Code::SourceLimit);
    rejected(gm::emit(vertex_program(false), tiny),
             gm::EmitError::Code::UnsupportedInterface);
    gm::EmitOptions no_textures;
    no_textures.fragment_position =
        gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
    no_textures.max_texture_resources = 0;
    rejected(gm::emit(fragment_program(), no_textures),
             gm::EmitError::Code::ResourceLimit);

    auto position_only = base(Stage::Fragment);
    attribute(position_only, 31);
    put(position_only, 0x48, 0xf);
    append(position_only, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{0}, {255}, {255}, 31, 0x7ff});
    mov(position_only, 1, 0); mov(position_only, 2, 0); mov(position_only, 3, 0);
    exit(position_only);
    rejected(gm::emit(lower(std::move(position_only))),
             gm::EmitError::Code::PositionConventionRequired);
}

ARMIGO_TEST(graphics_msl, register_move_reuses_source_ssa_bits) {
    auto p = base(Stage::Vertex);
    for (unsigned component = 28; component < 32; ++component) {
        attribute(p, component, true);
    }
    mov(p, 1, 0x7fc12345u);
    reg_move(p, 2, 1);
    reg_move(p, 3, 255);
    store(p, 2, 28);
    store(p, 2, 29);
    store(p, 3, 30);
    store(p, 3, 31);
    exit(p);
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    const gm::EmitResult result = gm::emit(lower(std::move(p)), options);
    const gm::EmittedStage* stage = accepted(result);
    if (!stage) return;
    ARMIGO_CHECK_EQ(stage->interface.position_output_mask, 0xf);
    ARMIGO_CHECK(contains(stage->source, "const uint v1 = 0x7fc12345u;"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.x = as_type<float>(v1);"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.y = as_type<float>(v1);"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.z = as_type<float>(v0);"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.w = as_type<float>(v0);"));
    ARMIGO_CHECK(!contains(stage->source, "v2"));
    ARMIGO_CHECK(!contains(stage->source, "v3"));
    ARMIGO_CHECK(!contains(stage->source, "armigo_canonical_nan(v1)"));
    ARMIGO_CHECK(!contains(stage->source, "armigo_add_rn_ftz(v1"));
}

ARMIGO_TEST(graphics_msl, registry_lowered_register_moves_keep_ssa_alias_chain) {
    auto p = base(Stage::Vertex);
    for (unsigned component = 28; component < 32; ++component) {
        attribute(p, component, true);
    }
    mov(p, 1, 0x7fc12345u);
    reg_move(p, 2, 1);
    reg_move(p, 3, 2);
    store(p, 3, 28);
    store(p, 3, 29);
    store(p, 3, 30);
    store(p, 3, 31);
    exit(p);
    gm::EmitOptions options;
    options.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
    const gm::EmitResult result = gm::emit(lower(std::move(p)), options);
    const gm::EmittedStage* stage = accepted(result);
    if (!stage) return;
    ARMIGO_CHECK_EQ(stage->interface.position_output_mask, 0xf);
    // Both registry-routed moves alias the literal SSA value; no move node or
    // arithmetic is emitted and the copied bits stay exact.
    ARMIGO_CHECK(contains(stage->source, "const uint v1 = 0x7fc12345u;"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.x = as_type<float>(v1);"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.y = as_type<float>(v1);"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.z = as_type<float>(v1);"));
    ARMIGO_CHECK(contains(stage->source,
                          "out.position.w = as_type<float>(v1);"));
    ARMIGO_CHECK(!contains(stage->source, "v2"));
    ARMIGO_CHECK(!contains(stage->source, "v3"));
    ARMIGO_CHECK(!contains(stage->source, "armigo_add_rn_ftz(v1"));
}

}  // namespace armigo::shader::test
