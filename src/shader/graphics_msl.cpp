// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/graphics_msl.h>

#include <armigo/shader/fma.h>
#include <armigo/shader/reciprocal.h>
#include <armigo/shader/rne_multiply.h>
#include <armigo/shader/rtz_multiply.h>
#include <armigo/shader/saturate.h>

#include "graphics_operations.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace armigo::shader::graphics_msl {
namespace {

namespace gx = graphics;

struct Failure { EmitError error; };
[[noreturn]] void fail(EmitError::Code code, std::uint32_t node = 0,
                       std::uint32_t detail = 0) {
    throw Failure{{code, node, detail}};
}
void require(bool condition, EmitError::Code code, std::uint32_t node = 0,
             std::uint32_t detail = 0) {
    if (!condition) fail(code, node, detail);
}

std::string number(std::uint32_t value) { return std::to_string(value); }
std::string hex(std::uint32_t value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "0x00000000u";
    for (unsigned index = 0; index < 8; ++index) {
        result[2 + index] = digits[(value >> (28 - index * 4)) & 15];
    }
    return result;
}
std::string scalar(gx::ScalarId id) { return "v" + number(id.index); }
std::string predicate(gx::PredicateId id) { return "p" + number(id.index); }
std::string sample(gx::SampleId id) { return "s" + number(id.index); }
const char* lane(unsigned component) {
    constexpr std::array names{"x", "y", "z", "w"};
    return names[component];
}
const char* qualifier(gx::Interpolation mode) {
    switch (mode) {
        case gx::Interpolation::Constant: return ", flat";
        case gx::Interpolation::Perspective: return "";
        case gx::Interpolation::ScreenLinear: return ", center_no_perspective";
    }
    return "<invalid>";
}

class Emitter {
public:
    Emitter(const gx::Program& program, const EmitOptions& options)
        : program_(program), options_(options) {}

    EmittedStage run() {
        stage_.stage = program_.stage();
        require(stage_.stage == shader_input::Stage::Vertex ||
                    stage_.stage == shader_input::Stage::Fragment,
                EmitError::Code::UnsupportedStage);
        graph_mode_ = !program_.blocks().empty();
        inspect();
        assign_resources();
        emit_source();
        require(stage_.source.size() <= options_.max_source_bytes,
                EmitError::Code::SourceLimit, 0,
                static_cast<std::uint32_t>(stage_.source.size()));
        return std::move(stage_);
    }

private:
    void append(std::string_view text) {
        require(text.size() <= options_.max_source_bytes -
                                   std::min(options_.max_source_bytes,
                                            source_.size()),
                EmitError::Code::SourceLimit);
        source_.append(text);
    }

    void line(std::string text = {}) {
        append(text);
        append("\n");
    }

    void check_scalar(gx::ScalarId id, std::size_t index) {
        require(id.index < index && program_.nodes()[id.index].type() == gx::ValueType::Raw32,
                EmitError::Code::InvalidProgram,
                static_cast<std::uint32_t>(index), id.index);
    }
    void check_sample(gx::SampleId id, std::size_t index) {
        require(id.index < index && program_.nodes()[id.index].type() == gx::ValueType::Sample4,
                EmitError::Code::InvalidProgram,
                static_cast<std::uint32_t>(index), id.index);
    }
    void check_predicate(gx::PredicateId id, std::size_t index) {
        require(id.index < index &&
                    program_.nodes()[id.index].type() == gx::ValueType::Predicate,
                EmitError::Code::InvalidProgram,
                static_cast<std::uint32_t>(index), id.index);
    }

    void mark_generic(std::map<unsigned, std::uint8_t>& masks,
                      unsigned component) {
        const unsigned relative = component - 32;
        masks[relative / 4] |= static_cast<std::uint8_t>(1u << (relative % 4));
    }

    void inspect() {
        const auto nodes = program_.nodes();
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            std::visit([&](const auto& operation) {
                using T = std::decay_t<decltype(operation)>;
                if (graph_mode_) {
                    if constexpr (std::is_same_v<T, gx::ScalarPhi>) {
                        for (const auto& arm : operation.arms) {
                            check_scalar(arm.value, index);
                        }
                    } else if constexpr (std::is_same_v<T, gx::PredicatePhi>) {
                        for (const auto& arm : operation.arms) {
                            check_predicate(arm.value, index);
                        }
                    }
                }
                if constexpr (std::is_same_v<T, gx::WordNotEqual>) {
                    check_scalar(operation.lhs, index);
                    check_scalar(operation.rhs, index);
                } else if constexpr (std::is_same_v<T, gx::PredicateNot>) {
                    check_predicate(operation.operand, index);
                } else if constexpr (std::is_same_v<T, gx::Pure>) {
                    const auto* entry = gx::detail::find_operation(operation.kind);
                    require(entry != nullptr && entry->admits(operation.immediate),
                            EmitError::Code::InvalidProgram,
                            static_cast<std::uint32_t>(index));
                    for (std::size_t slot = 0; slot < operation.operands.size(); ++slot) {
                        const auto operand = operation.operands[slot];
                        if (slot >= entry->arity) {
                            require(operand == 0, EmitError::Code::InvalidProgram,
                                    static_cast<std::uint32_t>(index), operand);
                        } else if (entry->operands[slot] == gx::ValueType::Predicate) {
                            check_predicate(gx::PredicateId{operand}, index);
                        } else {
                            check_scalar(gx::ScalarId{operand}, index);
                        }
                    }
                    operations_used_.insert(operation.kind);
                } else if constexpr (std::is_same_v<T, gx::AttributeLoad>) {
                    if (operation.component >= 32 && operation.component < 160) {
                        mark_generic(generic_inputs_, operation.component);
                    } else {
                        require(stage_.stage == shader_input::Stage::Fragment &&
                                    operation.component >= 28 && operation.component < 32,
                                EmitError::Code::UnsupportedInterface,
                                static_cast<std::uint32_t>(index), operation.component);
                        stage_.interface.position_input_mask |=
                            static_cast<std::uint8_t>(1u << (operation.component - 28));
                    }
                } else if constexpr (std::is_same_v<T, gx::ConstantLoad>) {
                    require(operation.binding < 18 && operation.byte_offset % 4 == 0,
                            EmitError::Code::InvalidProgram,
                            static_cast<std::uint32_t>(index));
                    const std::uint32_t required = operation.byte_offset + 4;
                    cb_sizes_[operation.binding] =
                        std::max(cb_sizes_[operation.binding], required);
                } else if constexpr (std::is_same_v<T, gx::WordAnd>) {
                    check_scalar(operation.operand,index);
                } else if constexpr (std::is_same_v<T, gx::Saturate>) {
                    check_scalar(operation.operand, index);
                    saturate_ = true;
                } else if constexpr (std::is_same_v<T, gx::FloatCompare>) {
                    check_scalar(operation.lhs, index); check_scalar(operation.rhs, index);
                } else if constexpr (std::is_same_v<T, gx::FloatArithmetic>) {
                    check_scalar(operation.lhs, index); check_scalar(operation.rhs, index);
                    if (operation.addend) check_scalar(*operation.addend, index);
                } else if constexpr (std::is_same_v<T, gx::Reciprocal>) {
                    check_scalar(operation.operand, index);
                } else if constexpr (std::is_same_v<T, gx::Interpolate>) {
                    check_scalar(operation.attribute, index);
                    if (operation.multiplier) check_scalar(*operation.multiplier, index);
                    if (operation.position_w) {
                        check_scalar(*operation.position_w, index);
                        perspective_ = true;
                    }
                    require(operation.encoding_control == 0x7ff,
                            EmitError::Code::InvalidProgram,
                            static_cast<std::uint32_t>(index));
                    const auto* attribute = std::get_if<gx::AttributeLoad>(
                        &nodes[operation.attribute.index].operation);
                    require(attribute != nullptr, EmitError::Code::InvalidProgram,
                            static_cast<std::uint32_t>(index));
                    if (attribute->component >= 32) {
                        const unsigned slot = (attribute->component - 32) / 4;
                        const auto [it, inserted] = interpolation_.emplace(slot, operation.mode);
                        require(inserted || it->second == operation.mode,
                                EmitError::Code::UnsupportedInterface,
                                static_cast<std::uint32_t>(index), slot);
                    }
                } else if constexpr (std::is_same_v<T, gx::TextureSample>) {
                    check_scalar(operation.x, index); check_scalar(operation.y, index);
                    texture_offsets_.insert(operation.handle_byte_offset);
                } else if constexpr (std::is_same_v<T, gx::SampleComponent>) {
                    check_sample(operation.sample, index);
                    require(operation.component < 4, EmitError::Code::InvalidProgram,
                            static_cast<std::uint32_t>(index), operation.component);
                }
            }, nodes[index].operation);
        }
        if (graph_mode_) {
            for (const gx::BasicBlock& basic : program_.blocks()) {
                if (const auto* branch = std::get_if<gx::BranchIf>(&basic.terminator)) {
                    check_predicate(branch->condition, nodes.size());
                }
            }
        }
        for (const gx::Export& output : program_.exports()) {
            check_scalar(output.value, nodes.size());
            if (output.kind == gx::OutputKind::Position) {
                require(stage_.stage == shader_input::Stage::Vertex && output.component < 4,
                        EmitError::Code::UnsupportedInterface, 0, output.component);
                stage_.interface.position_output_mask |=
                    static_cast<std::uint8_t>(1u << output.component);
            } else if (output.kind == gx::OutputKind::Generic) {
                require(stage_.stage == shader_input::Stage::Vertex && output.component < 128,
                        EmitError::Code::UnsupportedInterface, 0, output.component);
                mark_generic(generic_outputs_, output.component + 32);
            } else {
                require(stage_.stage == shader_input::Stage::Fragment && output.component < 32,
                        EmitError::Code::UnsupportedInterface, 0, output.component);
                stage_.interface.color_output_mask |= 1u << output.component;
            }
        }
        if (stage_.stage == shader_input::Stage::Vertex) {
            require(stage_.interface.position_output_mask == 0xf,
                    EmitError::Code::UnsupportedInterface, 0,
                    stage_.interface.position_output_mask);
            require(options_.vertex_depth == VertexDepthConvention::ZeroToOne ||
                        options_.vertex_depth == VertexDepthConvention::NegativeOneToOne,
                    EmitError::Code::DepthConventionRequired);
            stage_.vertex_depth = options_.vertex_depth;
            validate_vertex_options();
        } else {
            require(stage_.interface.color_output_mask == 0xf,
                    EmitError::Code::UnsupportedInterface, 0,
                    stage_.interface.color_output_mask);
            if (perspective_ || (stage_.interface.position_input_mask & 8) != 0) {
                require(options_.fragment_position ==
                            FragmentPositionConvention::MetalPositionWMatchesGuestPositionW,
                        EmitError::Code::PositionConventionRequired);
            }
        }
        for (const auto& [slot, mask] : generic_inputs_) {
            stage_.interface.generic_inputs.push_back(
                {static_cast<std::uint8_t>(slot), mask,
                 input_interpolation(slot)});
        }
        for (const auto& [slot, mask] : generic_outputs_) {
            stage_.interface.generic_outputs.push_back(
                {static_cast<std::uint8_t>(slot), mask,
                 gx::Interpolation::ScreenLinear});
        }
    }

    // Interpolation mode for one generic input vector slot. An Interpolate
    // node still decides its slot. A graph fragment program may also read the
    // attribute directly, so its declared SPH mode decides instead of a fixed
    // default: byte 0x18+slot holds two-bit lane modes (0 undeclared, matching
    // the lowering and CPU evaluation conventions). One MSL qualifier covers
    // the whole float4, so a slot whose declared lanes disagree is an
    // unsupported combination, rejected explicitly rather than choosing one
    // component's mode for the others.
    gx::Interpolation input_interpolation(unsigned slot) {
        std::optional<gx::Interpolation> declared;
        if (stage_.stage == shader_input::Stage::Fragment && graph_mode_) {
            const unsigned modes = std::to_integer<unsigned>(
                program_.sph()[0x18 + slot]);
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned value = (modes >> (lane * 2)) & 3;
                if (value == 0) continue;
                const gx::Interpolation mode = static_cast<gx::Interpolation>(value);
                require(!declared || *declared == mode,
                        EmitError::Code::UnsupportedInterface, 0, slot);
                declared = mode;
            }
        }
        const auto explicit_mode = interpolation_.find(slot);
        if (explicit_mode != interpolation_.end()) {
            require(!declared || *declared == explicit_mode->second,
                    EmitError::Code::UnsupportedInterface, 0, slot);
            return explicit_mode->second;
        }
        return declared.value_or(gx::Interpolation::ScreenLinear);
    }

    void validate_vertex_options() {
        std::map<unsigned, VertexAttributeOption> supplied;
        std::set<unsigned> attribute_indices;
        for (const auto& option : options_.vertex_attributes) {
            require(option.generic_slot < 32 && option.attribute_index <= 30,
                    EmitError::Code::UnsupportedVertexFormat, 0,
                    option.generic_slot);
            require(supplied.emplace(option.generic_slot, option).second,
                    EmitError::Code::DuplicateVertexFormat, 0,
                    option.generic_slot);
            require(attribute_indices.insert(option.attribute_index).second,
                    EmitError::Code::DuplicateVertexFormat, 0,
                    option.attribute_index);
        }
        for (const auto& [slot, mask] : generic_inputs_) {
            const auto it = supplied.find(slot);
            require(it != supplied.end(), EmitError::Code::MissingVertexFormat,
                    0, slot);
            require(it->second.format == VertexFormat::Float32x3 &&
                        (mask & 0x8) == 0,
                    EmitError::Code::UnsupportedVertexFormat, 0, slot);
            vertex_options_[slot] = it->second;
        }
        require(supplied.size() == generic_inputs_.size(),
                EmitError::Code::UnsupportedVertexFormat);
    }

    void assign_resources() {
        std::uint32_t next_buffer = 0;
        for (unsigned binding = 0; binding < cb_sizes_.size(); ++binding) {
            if (cb_sizes_[binding] == 0) continue;
            require(next_buffer <= 17,
                    EmitError::Code::ResourceLimit);
            cb_indices_[binding] = next_buffer;
            stage_.constant_buffers.push_back(
                {static_cast<std::uint8_t>(binding), cb_sizes_[binding],
                 static_cast<std::uint8_t>(next_buffer++)});
        }
        std::uint32_t next_texture = 0;
        for (const auto offset : texture_offsets_) {
            require(next_texture < options_.max_texture_resources &&
                        next_texture <= 15,
                    EmitError::Code::ResourceLimit);
            texture_indices_[offset] = next_texture;
            stage_.textures.push_back({offset,
                static_cast<std::uint8_t>(next_texture),
                static_cast<std::uint8_t>(next_texture)});
            ++next_texture;
        }
    }

    void emit_prelude() {
        line("// Independently generated Armigo graphics shader metadata lowering.");
        line("#include <metal_stdlib>");
        line("using namespace metal;");
        line("#pragma METAL fp math_mode(safe)");
        line("#pragma METAL fp contract(off)");
        line();
        append(emit_rne_multiply_msl());
        append(emit_rtz_ftz_multiply_msl());
        append(emit_reciprocal_msl());
        append(emit_fma_msl());
        // The saturation helper ships only with a program that saturates, so
        // programs without a Saturate node keep their historical source bytes.
        if (saturate_) {
            append(emit_saturate_msl());
        }
        // Helpers of registered operations follow in registry order, and
        // only for kinds the program uses.
        for (const auto& entry : gx::detail::registered_operations()) {
            if (entry.helper != nullptr && operations_used_.count(entry.kind) != 0) {
                append(std::string(entry.helper()));
            }
        }
        line(R"msl(inline uint armigo_flush_ftz(uint x) {
    return (x & 0x7f800000u) == 0u ? x & 0x80000000u : x;
}
inline uint armigo_canonical_nan(uint x) {
    return (x & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : x;
}
inline uint armigo_add_rn_ftz(uint a, uint b) {
    float result = as_type<float>(armigo_flush_ftz(a)) + as_type<float>(armigo_flush_ftz(b));
    return armigo_canonical_nan(armigo_flush_ftz(as_type<uint>(result)));
}
inline uint armigo_mul_rn_ftz(uint a, uint b) {
    return armigo_canonical_nan(armigo_flush_ftz(
        armigo_multiply_rne(armigo_flush_ftz(a), armigo_flush_ftz(b))));
}
)msl");
    }

    void emit_structs() {
        if (stage_.stage == shader_input::Stage::Vertex) {
            line("struct ArmigoVertexIn {");
            for (const auto& [slot, option] : vertex_options_) {
                line("    float4 attr" + number(slot) + " [[attribute(" +
                     number(option.attribute_index) + ")]];");
            }
            line("};");
            line("struct ArmigoVertexOut {");
            line("    float4 position [[position]];");
            for (const auto& [slot, mask] : generic_outputs_) {
                static_cast<void>(mask);
                line("    float4 generic" + number(slot) + " [[user(locn" +
                     number(slot) + ")]];");
            }
            line("};");
        } else {
            line("struct ArmigoFragmentIn {");
            line("    float4 position [[position]];");
            for (const auto& [slot, mask] : generic_inputs_) {
                static_cast<void>(mask);
                line("    float4 generic" + number(slot) + " [[user(locn" +
                     number(slot) + ")" + qualifier(input_interpolation(slot)) +
                     "]];");
            }
            line("};");
            line("struct ArmigoFragmentOut {");
            line("    float4 color [[color(0)]];");
            line("};");
        }
        line();
    }

    void emit_signature() {
        if (stage_.stage == shader_input::Stage::Vertex) {
            stage_.entry_point = "armigo_vertex_main";
            line("vertex ArmigoVertexOut armigo_vertex_main(ArmigoVertexIn in [[stage_in]]");
        } else {
            stage_.entry_point = "armigo_fragment_main";
            line("fragment ArmigoFragmentOut armigo_fragment_main(ArmigoFragmentIn in [[stage_in]]");
        }
        for (const auto& binding : stage_.constant_buffers) {
            line("    , constant uint* cb" + number(binding.guest_binding) +
                 " [[buffer(" + number(binding.metal_buffer_index) + ")]]");
        }
        for (const auto& binding : stage_.textures) {
            line("    , texture2d<float> tex" + number(binding.texture_index) +
                 " [[texture(" + number(binding.texture_index) + ")]]");
            line("    , sampler samp" + number(binding.sampler_index) +
                 " [[sampler(" + number(binding.sampler_index) + ")]]");
        }
        line(") {");
        if (stage_.stage == shader_input::Stage::Vertex) {
            line("    ArmigoVertexOut out{};");
            line("    out.position = float4(0.0f);");
            for (const auto& [slot, mask] : generic_outputs_) {
                static_cast<void>(mask);
                line("    out.generic" + number(slot) + " = float4(0.0f);");
            }
        } else {
            line("    ArmigoFragmentOut out{};");
            line("    out.color = float4(0.0f);");
        }
    }

    // Expression text for one Raw32- or Sample4-valued operation, shared by
    // the straight-line and graph emitters so numeric helpers, operand text
    // and operation order stay identical between the two paths.
    std::string scalar_expression(std::size_t index, const gx::Node& node) {
        const std::string id = number(static_cast<std::uint32_t>(index));
        return std::visit([&](const auto& operation) -> std::string {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, gx::Literal>) {
                return hex(operation.bits);
            } else if constexpr (std::is_same_v<T, gx::AttributeLoad>) {
                if (operation.component >= 32) {
                    const unsigned relative = operation.component - 32;
                    const unsigned slot = relative / 4, component = relative % 4;
                    const std::string field = stage_.stage == shader_input::Stage::Vertex
                        ? "attr" : "generic";
                    return "as_type<uint>(in." + field + number(slot) + "." +
                           lane(component) + ")";
                }
                return "as_type<uint>(in.position." +
                       std::string(lane(operation.component - 28)) + ")";
            } else if constexpr (std::is_same_v<T, gx::ConstantLoad>) {
                return "cb" + number(operation.binding) +
                       "[" + number(operation.byte_offset / 4) + "]";
            } else if constexpr (std::is_same_v<T, gx::WordAnd>) {
                return scalar(operation.operand) + " & " + hex(operation.mask);
            } else if constexpr (std::is_same_v<T, gx::FloatCompare>) {
                const std::string lhs = operation.negate_lhs
                    ? "(armigo_flush_ftz(" + scalar(operation.lhs) + ") ^ 0x80000000u)"
                    : "armigo_flush_ftz(" + scalar(operation.lhs) + ")";
                const char* op = operation.comparison == maxwell::OrderedComparison::LessThan
                    ? " < " : " > ";
                return "(as_type<float>(" + lhs + ")" + op +
                       "as_type<float>(armigo_flush_ftz(" + scalar(operation.rhs) +
                       "))) ? 0x3f800000u : 0u";
            } else if constexpr (std::is_same_v<T, gx::FloatArithmetic>) {
                std::string lhs = scalar(operation.lhs);
                if (operation.negate_lhs) lhs = "(" + lhs + " ^ 0x80000000u)";
                switch (operation.operation) {
                    case gx::Arithmetic::Add:
                        return "armigo_add_rn_ftz(" + lhs + ", " +
                               scalar(operation.rhs) + ")";
                    case gx::Arithmetic::MultiplyNearest:
                        return "armigo_mul_rn_ftz(" + lhs + ", " +
                               scalar(operation.rhs) + ")";
                    case gx::Arithmetic::MultiplyZero:
                        return "armigo_mul_rtz_ftz(" + lhs + ", " +
                               scalar(operation.rhs) + ")";
                    case gx::Arithmetic::FusedMultiplyAdd:
                        require(operation.addend.has_value(), EmitError::Code::InvalidProgram,
                                static_cast<std::uint32_t>(index));
                        return "armigo_fma_rne_ftz(" + lhs + ", " + scalar(operation.rhs) +
                               ", " + scalar(*operation.addend) + ")";
                }
                return "<invalid>";
            } else if constexpr (std::is_same_v<T, gx::Reciprocal>) {
                return "armigo_reciprocal_rne(" + scalar(operation.operand) + ")";
            } else if constexpr (std::is_same_v<T, gx::Saturate>) {
                return "armigo_saturate_float32(" + scalar(operation.operand) + ")";
            } else if constexpr (std::is_same_v<T, gx::Interpolate>) {
                std::string expression = operation.position_w
                    ? "armigo_multiply_rne(" + scalar(operation.attribute) + ", " +
                          scalar(*operation.position_w) + ")"
                    : scalar(operation.attribute);
                if (operation.multiplier) expression = "armigo_multiply_rne(" + expression +
                    ", " + scalar(*operation.multiplier) + ")";
                return expression;
            } else if constexpr (std::is_same_v<T, gx::TextureSample>) {
                const unsigned resource = texture_indices_.at(operation.handle_byte_offset);
                return "tex" + number(resource) + ".sample(samp" + number(resource) +
                       ", float2(as_type<float>(" + scalar(operation.x) +
                       "), as_type<float>(" + scalar(operation.y) + ")))";
            } else if constexpr (std::is_same_v<T, gx::SampleComponent>) {
                return "as_type<uint>(" + sample(operation.sample) +
                       "[" + number(operation.component) + "])";
            } else if constexpr (std::is_same_v<T, gx::Pure>) {
                return pure_expression(index, operation);
            } else {
                return "<invalid>";
            }
        }, node.operation);
    }

    // Expression text for a registered operation: operand texts follow the
    // signature (uint for Raw32, bool for Predicate) and the entry renders
    // the expression of its result type.
    std::string pure_expression(std::size_t index, const gx::Pure& operation) {
        const auto* entry = gx::detail::find_operation(operation.kind);
        require(entry != nullptr, EmitError::Code::InvalidProgram,
                static_cast<std::uint32_t>(index));
        gx::detail::OperationText text;
        for (std::size_t slot = 0; slot < entry->arity; ++slot) {
            const auto operand = operation.operands[slot];
            text[slot] = entry->operands[slot] == gx::ValueType::Predicate
                ? predicate(gx::PredicateId{operand})
                : scalar(gx::ScalarId{operand});
        }
        return entry->emit(text, operation.immediate);
    }

    // Bool expression for one predicate-valued definition, shared by the
    // straight-line and graph emitters. Phis are assigned on edges instead.
    std::string predicate_expression(std::size_t index, const gx::Node& node) {
        return std::visit([&](const auto& operation) -> std::string {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, gx::PredicateLiteral>) {
                return operation.value ? "true" : "false";
            } else if constexpr (std::is_same_v<T, gx::WordNotEqual>) {
                return scalar(operation.lhs) + " != " + scalar(operation.rhs);
            } else if constexpr (std::is_same_v<T, gx::PredicateNot>) {
                return "!" + predicate(operation.operand);
            } else if constexpr (std::is_same_v<T, gx::Pure>) {
                return pure_expression(index, operation);
            } else {
                fail(EmitError::Code::InvalidProgram, static_cast<std::uint32_t>(index));
            }
        }, node.operation);
    }

    void emit_node(std::size_t index, const gx::Node& node) {
        const std::string id = number(static_cast<std::uint32_t>(index));
        // Straight-line programs carry predicates only as plain definitions;
        // a phi has no meaning without blocks.
        if (node.type() == gx::ValueType::Predicate) {
            require(!std::holds_alternative<gx::PredicatePhi>(node.operation),
                    EmitError::Code::InvalidProgram, static_cast<std::uint32_t>(index));
            line("    [[maybe_unused]] const bool p" + id + " = " +
                 predicate_expression(index, node) + ";");
            return;
        }
        if (std::holds_alternative<gx::TextureSample>(node.operation)) {
            line("    [[maybe_unused]] const float4 s" + id + " = " +
                 scalar_expression(index, node) + ";");
        } else {
            line("    [[maybe_unused]] const uint v" + id + " = " +
                 scalar_expression(index, node) + ";");
        }
    }

    // Control-flow emission: typed storage per SSA node, one activity guard
    // per block in forward order, and phi values assigned on the executed
    // predecessor edge so untaken arms are never computed or read.
    void emit_graph_body() {
        const auto nodes = program_.nodes();
        for (std::size_t index = 0; index < nodes.size(); ++index) {
            const std::string id = number(static_cast<std::uint32_t>(index));
            // Zero/false/empty-vector starts only silence uninitialized
            // warnings; the validator rejects every read its definition does
            // not dominate, so initialization never repairs an invalid program.
            if (nodes[index].type() == gx::ValueType::Predicate) {
                line("    [[maybe_unused]] bool p" + id + " = false;");
            } else if (nodes[index].type() == gx::ValueType::Sample4) {
                line("    [[maybe_unused]] float4 s" + id + " = float4(0.0f);");
            } else {
                line("    [[maybe_unused]] uint v" + id + " = 0u;");
            }
        }
        const std::size_t blocks = program_.blocks().size();
        for (std::size_t block = 0; block < blocks; ++block) {
            line("    bool a" + number(static_cast<std::uint32_t>(block)) +
                 (block == 0 ? " = true;" : " = false;"));
        }
        for (std::size_t block = 0; block < blocks; ++block) {
            const gx::BasicBlock& basic = program_.blocks()[block];
            line("    if (a" + number(static_cast<std::uint32_t>(block)) + ") {");
            for (std::uint32_t node = basic.begin; node < basic.end; ++node) {
                emit_graph_node(node, nodes[node]);
            }
            emit_graph_terminator(static_cast<std::uint32_t>(block), basic.terminator);
            line("    }");
        }
    }

    void emit_graph_node(std::uint32_t index, const gx::Node& node) {
        const std::string id = number(index);
        std::visit([&](const auto& operation) {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, gx::ScalarPhi> ||
                          std::is_same_v<T, gx::PredicatePhi>) {
                // Values arrive through predecessor edge assignments.
            } else if (node.type() == gx::ValueType::Predicate) {
                line("        p" + id + " = " + predicate_expression(index, node) + ";");
            } else if (node.type() == gx::ValueType::Sample4) {
                // Implicit-LOD samples execute at their emitted control-flow
                // location using the host implementation's derivative/helper
                // behaviour. Equivalence to Maxwell under divergence is unverified.
                line("        " + sample(gx::SampleId{index}) + " = " +
                     scalar_expression(index, node) + ";");
            } else {
                line("        " + scalar(gx::ScalarId{index}) + " = " +
                     scalar_expression(index, node) + ";");
            }
        }, node.operation);
    }

    void emit_graph_edge(std::uint32_t predecessor, gx::BlockId successor) {
        const auto nodes = program_.nodes();
        const gx::BasicBlock& basic = program_.blocks()[successor.index];
        for (std::uint32_t node = basic.begin; node < basic.end; ++node) {
            std::visit([&](const auto& operation) {
                using T = std::decay_t<decltype(operation)>;
                if constexpr (std::is_same_v<T, gx::ScalarPhi>) {
                    for (const auto& arm : operation.arms) {
                        if (arm.predecessor.index == predecessor) {
                            line("            v" + number(node) + " = v" +
                                 number(arm.value.index) + ";");
                        }
                    }
                } else if constexpr (std::is_same_v<T, gx::PredicatePhi>) {
                    for (const auto& arm : operation.arms) {
                        if (arm.predecessor.index == predecessor) {
                            line("            p" + number(node) + " = p" +
                                 number(arm.value.index) + ";");
                        }
                    }
                }
            }, nodes[node].operation);
        }
    }

    void emit_graph_terminator(std::uint32_t block, const gx::Terminator& terminator) {
        std::visit([&](const auto& step) {
            using T = std::decay_t<decltype(step)>;
            if constexpr (std::is_same_v<T, gx::Jump>) {
                emit_graph_edge(block, step.target);
                line("        a" + number(step.target.index) + " = true;");
            } else if constexpr (std::is_same_v<T, gx::BranchIf>) {
                line("        if (" + predicate(step.condition) + ") {");
                emit_graph_edge(block, step.true_target);
                line("            a" + number(step.true_target.index) + " = true;");
                line("        } else {");
                emit_graph_edge(block, step.false_target);
                line("            a" + number(step.false_target.index) + " = true;");
                line("        }");
            }
            // Return: the validator admits exactly one return block and it is
            // last, so the export epilogue emitted once after all guarded
            // regions is reached by every executed path.
        }, terminator);
    }

    void emit_exports() {
        for (const gx::Export& output : program_.exports()) {
            const std::string value = "as_type<float>(" + scalar(output.value) + ")";
            if (output.kind == gx::OutputKind::Position) {
                line("    out.position." + std::string(lane(output.component)) + " = " + value + ";");
            } else if (output.kind == gx::OutputKind::Generic) {
                line("    out.generic" + number(output.component / 4) + "." +
                     lane(output.component % 4) + " = " + value + ";");
            } else {
                line("    out.color." + std::string(lane(output.component)) + " = " + value + ";");
            }
        }
        if (stage_.stage == shader_input::Stage::Vertex &&
            stage_.vertex_depth == VertexDepthConvention::NegativeOneToOne) {
            line("    const float armigo_depth_sum = out.position.z + out.position.w;");
            line("    out.position.z = armigo_depth_sum * 0.5f;");
        }
        line("    return out;");
        line("}");
    }

    void emit_source() {
        emit_prelude();
        emit_structs();
        emit_signature();
        if (graph_mode_) {
            emit_graph_body();
        } else {
            for (std::size_t index = 0; index < program_.nodes().size(); ++index) {
                emit_node(index, program_.nodes()[index]);
            }
        }
        emit_exports();
        stage_.source = std::move(source_);
    }

    const gx::Program& program_;
    const EmitOptions& options_;
    EmittedStage stage_;
    std::string source_;
    bool graph_mode_ = false;
    std::array<std::uint32_t, 18> cb_sizes_{};
    std::array<std::uint32_t, 18> cb_indices_{};
    std::set<std::uint32_t> texture_offsets_;
    std::map<std::uint32_t, std::uint32_t> texture_indices_;
    std::map<unsigned, std::uint8_t> generic_inputs_;
    std::map<unsigned, std::uint8_t> generic_outputs_;
    std::map<unsigned, gx::Interpolation> interpolation_;
    std::map<unsigned, VertexAttributeOption> vertex_options_;
    bool perspective_ = false;
    bool saturate_ = false;
    std::set<gx::PureKind> operations_used_;
};

}  // namespace

const char* EmitError::message() const noexcept {
    switch (code) {
        case Code::UnsupportedStage: return "graphics MSL stage is unsupported";
        case Code::InvalidProgram: return "graphics SSA program is internally inconsistent";
        case Code::MissingVertexFormat: return "vertex input format is not explicitly supplied";
        case Code::DuplicateVertexFormat: return "vertex input format is duplicated";
        case Code::UnsupportedVertexFormat: return "vertex input format is unsupported";
        case Code::UnsupportedInterface: return "graphics stage interface is unsupported";
        case Code::DepthConventionRequired: return "vertex clip-depth convention requires explicit selection";
        case Code::PositionConventionRequired: return "fragment PositionW convention requires explicit acknowledgement";
        case Code::ResourceLimit: return "graphics MSL resources exceed the bounded limit";
        case Code::SourceLimit: return "graphics MSL source exceeds the bounded limit";
        case Code::Allocation: return "graphics MSL metadata allocation failed";
        case Code::GraphUnsupported: return "control-flow graph emission does not support this operation";
    }
    return "unknown graphics MSL error";
}

EmitResult emit(const gx::Program& program, const EmitOptions& options) {
    try {
        // Straight-line programs keep their exact historical source; graph
        // programs emit per-block activity guards with predecessor-selected
        // phi edge assignments and no eager evaluation of untaken arms.
        return Emitter{program, options}.run();
    } catch (const Failure& failure) {
        return failure.error;
    } catch (const std::bad_alloc&) {
        return EmitError{EmitError::Code::Allocation};
    } catch (const std::length_error&) {
        return EmitError{EmitError::Code::Allocation};
    }
}

}  // namespace armigo::shader::graphics_msl
