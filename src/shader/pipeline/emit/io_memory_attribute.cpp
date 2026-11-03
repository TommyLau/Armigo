// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Attribute and interpolation emitters: ALD, AST, IPA and OUT. Attribute
// offsets are guest byte addresses; every invalid range, stage combination or
// missing fact is a structured diagnostic, never a zero substitute.

#include "io_memory_shared.h"

#include <algorithm>
#include <string>

namespace armigo::shader::pipeline {
namespace {

using namespace io_memory;

// One guest attribute word resolved to its Armigo IO semantic. location is the
// semantic location (generic slot, clip index, or zero for built-in vectors)
// and component is the word index inside the attribute vector.
struct AttributeTarget {
    IoSemantic semantic{IoSemantic::generic};
    std::uint16_t location{};
    std::uint8_t component{};
    ScalarType scalar{ScalarType::f32};
};

[[nodiscard]] constexpr std::uint8_t interface_components(IoSemantic semantic) {
    switch (semantic) {
    case IoSemantic::generic: case IoSemantic::position: case IoSemantic::fragment_coord:
    case IoSemantic::color:
        return 4;
    case IoSemantic::point_coord:
        return 2;
    default:
        return 1;
    }
}

// Maxwell attribute byte offset to IO semantic, with per-stage validity.
// Fixed-function colour, fog, texcoord, tessellation and viewport-mask ranges
// are rejected because the Armigo emitter contract carries no allocation
// state for remapping them.
[[nodiscard]] std::string attribute_offset_label(std::uint32_t offset) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string text = "Attribute offset 0x";
    for (int shift = 28; shift >= 0; shift -= 4) {
        text.push_back(digits[(offset >> shift) & 0xf]);
    }
    return text;
}

[[nodiscard]] Result<AttributeTarget> attribute_target(
    Stage stage, bool is_output, std::uint32_t offset) {
    const bool vertex = stage == Stage::vertex;
    const bool fragment = stage == Stage::fragment;
    auto target = [](IoSemantic semantic, std::uint16_t location,
                     std::uint8_t component, ScalarType scalar) {
        return Result<AttributeTarget>{AttributeTarget{semantic, location, component, scalar}};
    };
    auto invalid = [&](std::string reason) {
        return Result<AttributeTarget>{diagnose(Phase::emit_ir,
            DiagnosticCode::unsupported_capability, {},
            attribute_offset_label(offset) + " " + reason)};
    };
    if (offset < user_attribute_base) {
        if (offset >= 0x060 && offset < 0x064) {
            // The frozen IoSemantic vocabulary has no primitive id member, so
            // this fragment input is rejected until the contract names it.
            return invalid("primitive id has no contract semantic");
        }
        if (offset >= 0x064 && offset < 0x068) {
            if (is_output ? !vertex : !fragment) {
                return invalid("layer is not available for this direction and stage");
            }
            return target(IoSemantic::layer, 0, static_cast<std::uint8_t>((offset - 0x64) >> 2), ScalarType::s32);
        }
        if (offset >= 0x068 && offset < 0x06c) {
            if (is_output ? !vertex : !fragment) {
                return invalid("viewport index is not available for this direction and stage");
            }
            return target(IoSemantic::viewport_index, 0,
                static_cast<std::uint8_t>((offset - 0x68) >> 2), ScalarType::s32);
        }
        if (offset >= 0x06c && offset < 0x070) {
            if (!is_output || !vertex) return invalid("point size is a vertex output only");
            return target(IoSemantic::point_size, 0, static_cast<std::uint8_t>(offset & 3), ScalarType::f32);
        }
        if (offset >= 0x070 && offset < 0x080) {
            if (is_output ? !vertex : !fragment) {
                return invalid("position is not available for this direction and stage");
            }
            const auto semantic = is_output ? IoSemantic::position : IoSemantic::fragment_coord;
            return target(semantic, 0, static_cast<std::uint8_t>((offset - 0x70) >> 2), ScalarType::f32);
        }
        return invalid("is not a supported attribute range");
    }
    if (offset < user_attribute_end) {
        const std::uint16_t location = static_cast<std::uint16_t>((offset - user_attribute_base) >> 4);
        const std::uint8_t component = static_cast<std::uint8_t>((offset >> 2) & 3);
        if (is_output) {
            // Fragment output space holds the colour render targets; the vertex
            // stage exports plain generic varyings.
            if (fragment) return target(IoSemantic::color, location, component, ScalarType::f32);
            if (vertex) return target(IoSemantic::generic, location, component, ScalarType::f32);
            return invalid("generic outputs require the vertex stage");
        }
        return target(IoSemantic::generic, location, component, ScalarType::f32);
    }
    if (offset >= 0x2c0 && offset < 0x2e0) {
        const std::uint16_t clip = static_cast<std::uint16_t>((offset - 0x2c0) >> 2);
        if (is_output ? !vertex : !fragment) {
            return invalid("clip distance is not available for this direction and stage");
        }
        return target(IoSemantic::clip_distance, clip, 0, ScalarType::f32);
    }
    if (offset >= 0x2e0 && offset < 0x2e8) {
        if (is_output || !fragment) return invalid("point coord is a fragment input only");
        return target(IoSemantic::point_coord, 0, static_cast<std::uint8_t>((offset - 0x2e0) >> 2), ScalarType::f32);
    }
    if (offset >= 0x2f8 && offset < 0x2fc) {
        if (is_output || !vertex) return invalid("instance id is a vertex input only");
        return target(IoSemantic::instance_id, 0, 0, ScalarType::u32);
    }
    if (offset >= 0x2fc && offset < 0x300) {
        if (is_output || !vertex) return invalid("vertex id is a vertex input only");
        return target(IoSemantic::vertex_id, 0, 0, ScalarType::u32);
    }
    if (offset >= 0x3fc && offset < 0x400) {
        if (is_output || !fragment) return invalid("front facing is a fragment input only");
        return target(IoSemantic::front_facing, 0, 0, ScalarType::boolean);
    }
    return invalid("is a fixed-function or unsupported attribute range");
}

// Vertex input scalar after the host fetch boundary: normalized formats arrive
// as float, integer formats keep their guest width.
[[nodiscard]] constexpr ScalarType vertex_format_scalar(VertexFormat format) {
    switch (format) {
    case VertexFormat::u8: return ScalarType::u8;
    case VertexFormat::s8: return ScalarType::s8;
    case VertexFormat::u16: return ScalarType::u16;
    case VertexFormat::s16: return ScalarType::s16;
    case VertexFormat::f16: return ScalarType::f16;
    case VertexFormat::u32: return ScalarType::u32;
    case VertexFormat::s32: return ScalarType::s32;
    default: return ScalarType::f32;
    }
}

// Fragment input interpolation is a per-location, per-component fact; an
// undeclared component is Unknown and therefore a missing-fact diagnostic.
[[nodiscard]] Result<Interpolation> input_interpolation(
    EmitterContext& context, std::uint16_t location, std::uint8_t component) {
    auto knowledge = context.facts().require(
        fact_query::interpolation(location, component), Phase::emit_ir,
        {context.instruction().pc, context.instruction().name});
    if (const auto* error = std::get_if<Diagnostic>(&knowledge)) return *error;
    return std::get<Interpolation>(knowledge);
}

// The perspective-correct W multiplier is fragment coord component three;
// the fragment-position convention decides at MSL time whether that equals
// the guest W. Emission keeps the multiply explicit in the IR.
[[nodiscard]] Result<Operand> fragment_w(EmitterContext& context) {
    IoInfo info;
    info.semantic = IoSemantic::fragment_coord;
    info.location = 0;
    info.component_mask = 1u << 3;
    info.type = {ScalarType::f32, 4};
    // Built-in inputs carry no interpolation qualifier; describe the read
    // exactly as an explicit IPA of the fragment coordinate does, so both
    // reads of W merge into one interface use.
    info.interpolation = Interpolation{InterpolationMode::flat, InterpolationLocation::center};
    const ValueType word_type = ValueType::word32;
    auto loaded = context.load_io(info, {}, interpretation(ScalarType::f32), {&word_type, 1});
    if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
    return Operand{std::get<std::vector<ValueId>>(loaded).front()};
}

[[nodiscard]] Result<Operand> io_word(EmitterContext& context, ValueId value,
                                     ScalarType scalar) {
    if (scalar != ScalarType::boolean) return Operand{value};
    return context.unary(Opcode::predicate_to_mask, Operand{value}, NumericMode{},
        ValueType::word32);
}

[[nodiscard]] Result<InterpolationLocation> ipa_location(EmitterContext& context) {
    auto selector = decoded_field(context.instruction(), DecodedField::interpolation_location);
    if (const auto* error = std::get_if<Diagnostic>(&selector)) return *error;
    switch (std::get<std::uint64_t>(selector)) {
    case 0: return InterpolationLocation::center;
    case 1: return InterpolationLocation::centroid;
    default:
        return fail(context, DiagnosticCode::unsupported_modifier,
            "IPA sample and offset locations are not supported");
    }
}

[[nodiscard]] Status multiply_into(EmitterContext& context, Operand& value,
                                   Operand multiplier, NumericMode mode) {
    auto product = context.binary(Opcode::multiply, value, multiplier, mode, ValueType::word32);
    if (const auto* error = std::get_if<Diagnostic>(&product)) return *error;
    value = std::get<Operand>(product);
    return std::monostate{};
}

[[nodiscard]] Result<Operand> read_source_gpr(EmitterContext& context, SourceRole role) {
    auto reg = source_register(context.instruction(), role);
    if (const auto* error = std::get_if<Diagnostic>(&reg)) return *error;
    return read_gpr(context, std::get<Register>(reg).index);
}

[[nodiscard]] const RegisterAccess* require_destination(
    EmitterContext& context) {
    return find_register(context.instruction(), RegisterRole::destination);
}

} // namespace

// ALD: attribute load, direct and physical addressing.
Status emit_attribute_load(EmitterContext& context) {
    const DecodedInstruction& inst = context.instruction();
    const Stage stage = context.decoded_program().stage;
    if (attribute_per_patch(inst.raw)) {
        return fail(context, DiagnosticCode::unsupported_capability,
            "ALD per-patch addressing requires tessellation stages");
    }
    const bool output_space = attribute_output_space(inst.raw);
    if (output_space && stage == Stage::fragment) {
        return fail(context, DiagnosticCode::unsupported_capability,
            "ALD cannot read fragment output attributes");
    }
    const unsigned words = attribute_words(inst.raw);
    const auto* destination = require_destination(context);
    if (destination == nullptr) return fail(context, DiagnosticCode::malformed_input,
        "ALD is missing its destination register access");
    if (destination->reg.file != RegisterFile::gpr) {
        return fail(context, DiagnosticCode::malformed_input, "ALD destination is not a GPR");
    }
    // Per-primitive vertex indexing (SrcB) exists for tessellation and
    // geometry stages only, which the pipeline contract does not admit, so
    // SrcB is never read here.

    if (attribute_physical(inst.raw)) {
        // Physical access targets the generic user attribute space with a
        // dynamic byte displacement from attribute zero.
        auto address = read_gpr(context, attribute_address_index(inst.raw));
        if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
        auto displacement = context.binary(Opcode::subtract, std::get<Operand>(address),
            word_literal(user_attribute_base), address_mode, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&displacement)) return *error;
        IoInfo info;
        info.semantic = IoSemantic::generic;
        // A physical vector access reads consecutive words starting at the
        // computed address; component zero is reloaded for each word.
        const auto mask = static_cast<std::uint8_t>(words >= 4 ? 0xf : (1u << words) - 1);
        info.component_mask = mask;
        info.type = {ScalarType::f32, 4};
        info.dynamic_byte_offset = InputIndex{0};
        info.output_storage = output_space;
        const std::vector<ValueType> results(words, ValueType::word32);
        std::vector<Operand> inputs{std::get<Operand>(displacement)};
        auto loaded = context.load_io(info, inputs, interpretation(ScalarType::f32), results);
        if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
        const auto& values = std::get<std::vector<ValueId>>(loaded);
        for (unsigned index = 0; index < words; ++index) {
            const auto dest = static_cast<std::uint16_t>(destination->reg.index + index);
            if (dest >= 255) break;
            if (auto status = context.write(Register{RegisterFile::gpr, dest},
                    index < values.size() ? Operand{values[index]} : Operand{values.back()});
                !std::holds_alternative<std::monostate>(status)) return status;
        }
        return std::monostate{};
    }

    const std::uint32_t base = attribute_byte_offset(inst.raw);
    for (unsigned index = 0; index < words; ++index) {
        const auto dest = static_cast<std::uint16_t>(destination->reg.index + index);
        if (dest >= 255) break;
        auto target = attribute_target(stage, output_space, base + index * 4);
        if (const auto* error = std::get_if<Diagnostic>(&target)) return *error;
        const auto& word = std::get<AttributeTarget>(target);

        IoInfo info;
        info.semantic = word.semantic;
        info.location = word.location;
        info.component_mask = static_cast<std::uint8_t>(1u << word.component);
        info.interpolation = Interpolation{InterpolationMode::flat, InterpolationLocation::center};
        info.output_storage = output_space;
        ScalarType scalar = word.scalar;
        NumericMode numeric = interpretation(scalar);
        if (word.semantic == IoSemantic::generic && !output_space && stage == Stage::vertex) {
            // A supplied format fact keeps the old pipeline-state path; without
            // one the attribute register word stays raw and legalize derives
            // the declared class from every use of the loaded value.
            auto knowledge = context.facts().read(
                FactQuery<FactKind::vertex_attribute_format>{word.location}, Phase::emit_ir,
                {inst.pc, inst.name});
            if (const auto* error = std::get_if<Diagnostic>(&knowledge)) return *error;
            if (const auto* known = std::get_if<Known<AttributeFormat>>(
                    &std::get<Knowledge<AttributeFormat>>(knowledge))) {
                scalar = vertex_format_scalar(known->value.format);
                numeric = interpretation(scalar);
            } else {
                scalar = ScalarType::u32;
                numeric = NumericMode{Domain::bits, ScalarType::u32};
            }
        } else if (word.semantic == IoSemantic::generic && !output_space) {
            auto interpolation = input_interpolation(context, word.location, word.component);
            if (const auto* error = std::get_if<Diagnostic>(&interpolation)) return *error;
            info.interpolation = std::get<Interpolation>(interpolation);
            scalar = ScalarType::f32;
        }
        info.type = {scalar, interface_components(word.semantic)};
        const ValueType result_type = storage_type(scalar);
        auto loaded = context.load_io(info, {}, numeric, {&result_type, 1});
        if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
        auto result = io_word(context, std::get<std::vector<ValueId>>(loaded).front(), scalar);
        if (const auto* error = std::get_if<Diagnostic>(&result)) return *error;
        if (auto status = context.write(Register{RegisterFile::gpr, dest},
                std::get<Operand>(result));
            !std::holds_alternative<std::monostate>(status)) return status;
    }
    return std::monostate{};
}

// AST: attribute store, direct and physical addressing.
Status emit_attribute_store(EmitterContext& context) {
    const DecodedInstruction& inst = context.instruction();
    const Stage stage = context.decoded_program().stage;
    if (attribute_per_patch(inst.raw)) {
        return fail(context, DiagnosticCode::unsupported_capability,
            "AST per-patch addressing requires tessellation stages");
    }
    const unsigned words = attribute_words(inst.raw);
    auto value_register = store_source_register(inst);
    if (const auto* error = std::get_if<Diagnostic>(&value_register)) return *error;
    const auto value_base = std::get<Register>(value_register).index;

    // Values at and beyond RZ do not exist; the guest encodes a discarded
    // store as RZ itself, which reads as literal zero.
    std::vector<Operand> values;
    for (unsigned index = 0; index < words; ++index) {
        const auto source = static_cast<std::uint16_t>(
            std::min<std::uint32_t>(value_base + index, 255));
        auto value = read_gpr(context, source);
        if (const auto* error = std::get_if<Diagnostic>(&value)) return *error;
        values.push_back(std::get<Operand>(value));
    }

    if (attribute_physical(inst.raw)) {
        auto address = read_gpr(context, attribute_address_index(inst.raw));
        if (const auto* error = std::get_if<Diagnostic>(&address)) return *error;
        auto displacement = context.binary(Opcode::subtract, std::get<Operand>(address),
            word_literal(user_attribute_base), address_mode, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&displacement)) return *error;
        IoInfo info;
        info.semantic = IoSemantic::generic;
        info.component_mask = static_cast<std::uint8_t>(words >= 4 ? 0xf : (1u << words) - 1);
        info.type = {ScalarType::f32, 4};
        info.dynamic_byte_offset = InputIndex{0};
        for (unsigned index = 0; index < words; ++index) info.data.push_back(InputIndex{static_cast<std::uint16_t>(1 + index)});
        std::vector<Operand> inputs{std::get<Operand>(displacement)};
        inputs.insert(inputs.end(), values.begin(), values.end());
        return context.store_io(info, inputs, interpretation(ScalarType::f32));
    }

    const std::uint32_t base = attribute_byte_offset(inst.raw);
    for (unsigned index = 0; index < words; ++index) {
        auto target = attribute_target(stage, true, base + index * 4);
        if (const auto* error = std::get_if<Diagnostic>(&target)) return *error;
        const auto& word = std::get<AttributeTarget>(target);

        IoInfo info;
        info.semantic = word.semantic;
        info.location = word.location;
        info.component_mask = static_cast<std::uint8_t>(1u << word.component);
        info.interpolation = Interpolation{InterpolationMode::flat, InterpolationLocation::center};
        ScalarType scalar = word.scalar;
        std::uint8_t components = interface_components(word.semantic);
        if (word.semantic == IoSemantic::color) {
            auto output_type = context.facts().require(
                FactQuery<FactKind::fragment_output_type>{word.location}, Phase::emit_ir,
                {inst.pc, inst.name});
            if (const auto* error = std::get_if<Diagnostic>(&output_type)) return *error;
            const auto type = std::get<IoType>(output_type);
            scalar = type.scalar;
            components = type.components;
        } else if (word.semantic == IoSemantic::generic && stage == Stage::vertex) {
            scalar = ScalarType::f32;
        } else if (word.semantic == IoSemantic::generic) {
            return fail(context, DiagnosticCode::unsupported_capability,
                "AST generic stores require the vertex stage");
        }
        info.type = {scalar, components};
        info.data.push_back(InputIndex{0});
        std::vector<Operand> store_inputs{values[index]};
        if (auto status = context.store_io(info, store_inputs, interpretation(scalar));
            !std::holds_alternative<std::monostate>(status)) return status;
    }
    return std::monostate{};
}

// IPA: interpolator load with explicit perspective correction, multiplier and
// guest saturation.
Status emit_interpolate(EmitterContext& context) {
    const DecodedInstruction& inst = context.instruction();
    const Stage stage = context.decoded_program().stage;
    const auto* destination = require_destination(context);
    if (destination == nullptr) return fail(context, DiagnosticCode::malformed_input,
        "IPA is missing its destination register access");
    const auto operation = ipa_operation(inst.raw);
    if (operation == IpaOperation::sc) {
        return fail(context, DiagnosticCode::unsupported_modifier,
            "IPA.SC interpolation semantics are not established");
    }
    const bool constant = operation == IpaOperation::constant;
    if (constant && ipa_indexed(inst.raw)) {
        return fail(context, DiagnosticCode::unsupported_modifier,
            "Indexed IPA.CONSTANT requires a flat interpolation-mode override");
    }
    auto location = ipa_location(context);
    if (const auto* error = std::get_if<Diagnostic>(&location)) return *error;
    const auto instruction_location = std::get<InterpolationLocation>(location);

    Operand value;
    if (ipa_indexed(inst.raw)) {
        // Indexed accesses address the generic space dynamically and always
        // receive the explicit guest W correction.
        auto displacement_source = read_source_gpr(context, SourceRole::a);
        if (const auto* error = std::get_if<Diagnostic>(&displacement_source)) return *error;
        auto displacement = context.binary(Opcode::subtract, std::get<Operand>(displacement_source),
            word_literal(user_attribute_base), address_mode, ValueType::word32);
        if (const auto* error = std::get_if<Diagnostic>(&displacement)) return *error;
        IoInfo info;
        info.semantic = IoSemantic::generic;
        info.component_mask = 1;
        info.type = {ScalarType::f32, 4};
        info.interpolation.location = instruction_location;
        info.interpolation_location_override = instruction_location;
        info.dynamic_byte_offset = InputIndex{0};
        std::vector<Operand> inputs{std::get<Operand>(displacement)};
        const ValueType word_type = ValueType::word32;
        auto loaded = context.load_io(info, inputs, interpretation(ScalarType::f32), {&word_type, 1});
        if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
        value = Operand{std::get<std::vector<ValueId>>(loaded).front()};
        auto w = fragment_w(context);
        if (const auto* error = std::get_if<Diagnostic>(&w)) return *error;
        if (auto status = multiply_into(context, value, std::get<Operand>(w), float_multiply);
            !std::holds_alternative<std::monostate>(status)) return status;
    } else {
        auto target = attribute_target(stage, false, ipa_byte_offset(inst.raw));
        if (const auto* error = std::get_if<Diagnostic>(&target)) return *error;
        const auto& word = std::get<AttributeTarget>(target);
        IoInfo info;
        info.semantic = word.semantic;
        info.location = word.location;
        info.component_mask = static_cast<std::uint8_t>(1u << word.component);
        info.interpolation = Interpolation{InterpolationMode::flat, InterpolationLocation::center};
        info.interpolation_location_override = instruction_location;
        if (word.semantic == IoSemantic::generic && stage == Stage::fragment) {
            auto interpolation = input_interpolation(context, word.location, word.component);
            if (const auto* error = std::get_if<Diagnostic>(&interpolation)) return *error;
            info.interpolation = std::get<Interpolation>(interpolation);
            info.interpolation.location = instruction_location;
            if (constant && info.interpolation.mode != InterpolationMode::flat) {
                return fail(context, DiagnosticCode::unsupported_modifier,
                    "IPA.CONSTANT requires a proven flat attribute under the current IO contract");
            }
        } else if (word.semantic == IoSemantic::fragment_coord && word.component == 1) {
            // PositionY applies a vertical flip when the host
            // negate-y state is set. The flip needs the support-buffer
            // viewport height, which the emitter contract does not expose.
            auto negate = context.facts().read(FactQuery<FactKind::negate_y>{}, Phase::emit_ir,
                {inst.pc, inst.name});
            if (const auto* error = std::get_if<Diagnostic>(&negate)) return *error;
            // read keeps the Knowledge wrapper; only a proven true rejects.
            const auto* knowledge = std::get_if<Known<bool>>(&std::get<Knowledge<bool>>(negate));
            if (knowledge != nullptr && knowledge->value) {
                return fail(context, DiagnosticCode::unsupported_capability,
                    "IPA PositionY with negated Y requires viewport-height support");
            }
        }
        info.type = {word.scalar, interface_components(word.semantic)};
        const ValueType result_type = storage_type(word.scalar);
        auto loaded = context.load_io(info, {}, interpretation(word.scalar), {&result_type, 1});
        if (const auto* error = std::get_if<Diagnostic>(&loaded)) return *error;
        auto word_value = io_word(context, std::get<std::vector<ValueId>>(loaded).front(), word.scalar);
        if (const auto* error = std::get_if<Diagnostic>(&word_value)) return *error;
        value = std::get<Operand>(word_value);
        if (word.semantic == IoSemantic::generic && stage == Stage::fragment &&
            info.interpolation.mode == InterpolationMode::perspective) {
            auto w = fragment_w(context);
            if (const auto* error = std::get_if<Diagnostic>(&w)) return *error;
            if (auto status = multiply_into(context, value, std::get<Operand>(w), float_multiply);
                !std::holds_alternative<std::monostate>(status)) return status;
        }
    }

    if (operation == IpaOperation::multiply) {
        auto multiplier = read_source_gpr(context, SourceRole::b);
        if (const auto* error = std::get_if<Diagnostic>(&multiplier)) return *error;
        if (auto status = multiply_into(context, value, std::get<Operand>(multiplier), float_multiply);
            !std::holds_alternative<std::monostate>(status)) return status;
    }
    if (ipa_saturate(inst.raw)) {
        // Guest [0,1] saturation including its NaN behavior travels as the
        // floating saturate flag; numeric lanes own the lowering.
        NumericMode saturated = float_multiply;
        saturated.saturate = true;
        if (auto status = multiply_into(context, value, float_literal(1.0f), saturated);
            !std::holds_alternative<std::monostate>(status)) return status;
    }
    return context.write(Register{RegisterFile::gpr, destination->reg.index}, value);
}

// OUT: primitive emit/cut requires geometry stages, which the contract does
// not admit.
Status emit_out_primitive(EmitterContext& context) {
    return fail(context, DiagnosticCode::unsupported_instruction,
        "OUT primitive emit/cut requires geometry stages");
}

} // namespace armigo::shader::pipeline
