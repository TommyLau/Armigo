// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/graphics.h>

#include "graphics_builder.h"
#include "graphics_families.h"
#include "graphics_operations.h"
#include "graphics_validation.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>

#pragma STDC FP_CONTRACT OFF

namespace armigo::shader::graphics {
namespace {

namespace mw = maxwell;
using Stage = shader_input::Stage;
using Failure = BuilderFailure;
[[noreturn]] void fail(Error::Code code, std::uint32_t pc = 0,
                      std::uint32_t detail = 0) { throw Failure{{code, pc, detail}}; }
void require(bool condition, Error::Code code, std::uint32_t pc = 0,
             std::uint32_t detail = 0) {
    if (!condition) fail(code, pc, detail);
}
bool generic(unsigned component) { return component >= 32 && component < 160; }
bool position(unsigned component) { return component >= 28 && component < 32; }
std::uint32_t word(const std::array<std::byte, 80>& data, std::size_t offset) {
    std::uint32_t result = 0;
    for (unsigned i = 0; i < 4; ++i) {
        result |= std::to_integer<std::uint32_t>(data[offset + i]) << (8 * i);
    }
    return result;
}
unsigned byte(const std::array<std::byte, 80>& data, std::size_t offset) {
    return std::to_integer<unsigned>(data[offset]);
}


}  // namespace

Builder::Builder(Limits requested) : limits(requested) {}

std::uint32_t Builder::append(Operation operation) {
    require(program.nodes_.size() < limits.max_nodes &&
            program.nodes_.size() < std::numeric_limits<std::uint32_t>::max(),
            Error::Code::Limit, pc);
    const auto id = static_cast<std::uint32_t>(program.nodes_.size());
    program.nodes_.push_back({pc, std::move(operation)});
    return id;
}
ScalarId Builder::scalar(Operation operation) { return {append(std::move(operation))}; }
std::uint32_t Builder::pure(const Pure& operation, ValueType expected) {
    const auto* entry = detail::find_operation(operation.kind);
    require(entry != nullptr && entry->result == expected &&
            entry->admits(operation.immediate), Error::Code::InvalidInstruction, pc);
    const auto nodes = program.nodes();
    for (std::size_t slot = 0; slot < operation.operands.size(); ++slot) {
        const auto index = operation.operands[slot];
        if (slot >= entry->arity) {
            require(index == 0, Error::Code::InvalidInstruction, pc, index);
            continue;
        }
        require(index < nodes.size() && nodes[index].type() == entry->operands[slot],
                Error::Code::InvalidInstruction, pc, index);
    }
    return append(operation);
}
ScalarId Builder::read(mw::RegisterOperand reg) {
    if (reg.is_zero()) return zero;
    require(registers[reg.index].has_value(), Error::Code::UninitializedRegister,
            pc, reg.index);
    return *registers[reg.index];
}
void Builder::write(mw::RegisterOperand reg, ScalarId value) {
    if (!reg.is_zero()) registers[reg.index] = value;
}
bool Builder::declared(unsigned component, bool output) const {
    if (!position(component) && !generic(component)) return false;
    const auto& sph = program.sph_;
    if (program.stage_ == Stage::Vertex) {
        if (position(component)) {
            return (byte(sph, output ? 0x35 : 0x17) &
                    (1u << (component - 28 + 4))) != 0;
        }
        const unsigned index = component - 32;
        return (byte(sph, (output ? 0x36 : 0x18) + index / 8) &
                (1u << (index % 8))) != 0;
    }
    if (output) return false;
    if (position(component)) return (byte(sph, 0x17) &
                                     (1u << (component - 28 + 4))) != 0;
    const unsigned index = component - 32;
    return ((byte(sph, 0x18 + index / 4) >> ((index % 4) * 2)) & 3) != 0;
}
ScalarId Builder::attribute(unsigned component) {
    require(declared(component, false), Error::Code::UndeclaredAttribute, pc, component);
    return scalar(AttributeLoad{static_cast<std::uint16_t>(component)});
}
ScalarId Builder::constant(mw::ConstantBufferOperand operand) {
    require(operand.binding < 18 && operand.byte_offset % 4 == 0 &&
            operand.byte_offset <= 65532, Error::Code::UnsupportedModifier, pc);
    return scalar(ConstantLoad{operand.binding, operand.byte_offset});
}
ScalarId Builder::source(const mw::FloatSource& operand, bool expect_constant) {
    if (expect_constant) {
        const auto* value = std::get_if<mw::ConstantBufferOperand>(&operand);
        require(value != nullptr, Error::Code::InvalidInstruction, pc);
        return constant(*value);
    }
    const auto* value = std::get_if<mw::RegisterOperand>(&operand);
    require(value != nullptr, Error::Code::InvalidInstruction, pc);
    return read(*value);
}
template<class T> const T& Builder::payload(const mw::Instruction& instruction) {
    const auto* value = std::get_if<T>(&instruction.payload);
    require(value != nullptr, Error::Code::InvalidInstruction, pc);
    return *value;
}
void Builder::validate_header() {
    const auto& sph = program.sph_;
    require(((word(sph, 0) >> 10) & 15) == shader_input::sph_stage_type_for(program.stage_),
            Error::Code::HeaderMismatch);
    require((word(sph, 0) & ((1u << 15) | (1u << 16) | (1u << 27))) == 0,
            Error::Code::UnsupportedInterface);
    for (const auto offset : {4u, 8u, 12u}) {
        require((word(sph, offset) & 0x00ffffffu) == 0,
                Error::Code::UnsupportedInterface, 0, offset);
    }
    for (unsigned offset = 0x14; offset < 0x17; ++offset) {
        require(byte(sph, offset) == 0, Error::Code::UnsupportedInterface, 0, offset);
    }
    require((byte(sph, 0x17) & 15) == 0, Error::Code::UnsupportedInterface);
    if (program.stage_ == Stage::Vertex) {
        for (unsigned offset = 0x28; offset < 0x35; ++offset) {
            require(byte(sph, offset) == 0, Error::Code::UnsupportedInterface, 0, offset);
        }
        require((byte(sph, 0x35) & 15) == 0, Error::Code::UnsupportedInterface);
        for (unsigned offset = 0x46; offset < 80; ++offset) {
            require(byte(sph, offset) == 0, Error::Code::UnsupportedInterface, 0, offset);
        }
    } else {
        for (unsigned offset = 0x38; offset < 0x48; ++offset) {
            require(byte(sph, offset) == 0, Error::Code::UnsupportedInterface, 0, offset);
        }
        require(word(sph, 0x4c) == 0, Error::Code::UnsupportedInterface);
    }
}
void Builder::validate_graph_interface(const GraphSpec& spec) {
    program.stage_ = spec.stage;
    program.sph_ = spec.sph;
    validate_header();
    for (const auto& node : spec.nodes) {
        const auto at = node.pc;
        std::visit([&](const auto& operation) {
            using T = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<T, AttributeLoad>) {
                require(declared(operation.component, false), Error::Code::UndeclaredAttribute,
                        at, operation.component);
            } else if constexpr (std::is_same_v<T, ConstantLoad>) {
                require(operation.binding < 18 && operation.byte_offset % 4 == 0 &&
                        operation.byte_offset <= 65532, Error::Code::UnsupportedModifier, at);
            } else if constexpr (std::is_same_v<T, FloatCompare>) {
                require(operation.comparison == mw::OrderedComparison::LessThan ||
                        operation.comparison == mw::OrderedComparison::GreaterThan,
                        Error::Code::UnsupportedModifier, at);
            } else if constexpr (std::is_same_v<T, FloatArithmetic>) {
                require(operation.operation == Arithmetic::Add ||
                        operation.operation == Arithmetic::MultiplyNearest ||
                        operation.operation == Arithmetic::MultiplyZero ||
                        operation.operation == Arithmetic::FusedMultiplyAdd,
                        Error::Code::UnsupportedModifier, at);
                require(operation.addend.has_value() ==
                        (operation.operation == Arithmetic::FusedMultiplyAdd),
                        Error::Code::InvalidInstruction, at);
            } else if constexpr (std::is_same_v<T, Interpolate>) {
                require(program.stage_ == Stage::Fragment, Error::Code::UnsupportedStage, at);
                require(operation.attribute.index < spec.nodes.size(), Error::Code::InvalidInstruction, at);
                const auto* input = std::get_if<AttributeLoad>(
                    &spec.nodes[operation.attribute.index].operation);
                require(input != nullptr, Error::Code::InvalidInstruction, at);
                auto mode = Interpolation::ScreenLinear;
                if (generic(input->component)) {
                    const auto map = byte(program.sph_, 0x18 + (input->component - 32) / 4);
                    unsigned effective = 0;
                    for (unsigned lane = 0; lane < 4 && effective == 0; ++lane)
                        effective = (map >> (lane * 2)) & 3;
                    mode = static_cast<Interpolation>(effective);
                }
                require(operation.encoding_control == 0x7ff && operation.mode == mode &&
                        operation.position_w.has_value() == (mode == Interpolation::Perspective),
                        Error::Code::UnsupportedModifier, at);
                if (operation.position_w) {
                    require(operation.position_w->index < spec.nodes.size(),
                            Error::Code::InvalidInstruction, at);
                    const auto* position = std::get_if<AttributeLoad>(
                        &spec.nodes[operation.position_w->index].operation);
                    require(position != nullptr && position->component == 31,
                            Error::Code::InvalidInstruction, at);
                }
            } else if constexpr (std::is_same_v<T, TextureSample>) {
                require(program.stage_ == Stage::Fragment, Error::Code::UnsupportedStage, at);
                require(operation.handle_byte_offset % 4 == 0 && operation.handle_byte_offset <= 32764,
                        Error::Code::UnsupportedModifier, at);
            } else if constexpr (std::is_same_v<T, SampleComponent>) {
                require(operation.component < 4, Error::Code::InvalidInstruction, at,
                        operation.component);
            } else if constexpr (std::is_same_v<T, Saturate>) {
                // No interface inputs beyond the scalar operand; its type,
                // definedness, and dominance are graph-constructor checks.
            }
        }, node.operation);
    }
    std::array<bool, 160> vertex_outputs{};
    std::array<bool, 32> color_outputs{};
    for (const auto& output : spec.exports) {
        if (program.stage_ == Stage::Vertex) {
            unsigned component = 0;
            if (output.kind == OutputKind::Position && output.component < 4)
                component = 28 + output.component;
            else if (output.kind == OutputKind::Generic && output.component < 128)
                component = 32 + output.component;
            else fail(Error::Code::UnsupportedInterface, 0, output.component);
            require(declared(component, true), Error::Code::UnsupportedInterface, 0, component);
            require(!vertex_outputs[component], Error::Code::GraphFormat, 0, component);
            vertex_outputs[component] = true;
        } else {
            require(output.kind == OutputKind::Color && output.component < 32,
                    Error::Code::UnsupportedInterface, 0, output.component);
            require((word(program.sph_, 0x48) & (1u << output.component)) != 0,
                    Error::Code::UnsupportedInterface, 0, output.component);
            require(!color_outputs[output.component], Error::Code::GraphFormat, 0, output.component);
            color_outputs[output.component] = true;
        }
    }
    if (program.stage_ == Stage::Vertex) {
        for (unsigned component = 28; component < 160; ++component)
            require(!declared(component, true) || vertex_outputs[component],
                    Error::Code::MissingOutput, 0, component);
    } else {
        const auto mask = word(program.sph_, 0x48);
        for (unsigned component = 0; component < 32; ++component)
            require((mask & (1u << component)) == 0 || color_outputs[component],
                    Error::Code::MissingOutput, 0, component);
    }
}
void Builder::begin_graph(shader_input::Stage stage,
                          const std::array<std::byte, 80>& sph) {
    // A graph build must start from an empty program; splicing into a Builder
    // that already lowered instructions would corrupt node references.
    require(program.nodes_.empty() && program.exports_.empty() &&
                program.blocks_.empty(),
            Error::Code::InvalidInstruction, pc);
    require(stage == Stage::Vertex || stage == Stage::Fragment,
            Error::Code::UnsupportedStage);
    program.stage_ = stage;
    program.sph_ = sph;
    validate_header();
    zero = scalar(Literal{0});
}

void Builder::exports() {
    if (program.stage_ == Stage::Vertex) {
        for (unsigned component = 28; component < 160; ++component) {
            if (!declared(component, true)) continue;
            require(stored[component].has_value(), Error::Code::MissingOutput, pc, component);
            program.exports_.push_back({position(component) ? OutputKind::Position : OutputKind::Generic,
                static_cast<std::uint16_t>(component - (position(component) ? 28 : 32)),
                *stored[component]});
        }
    } else {
        const auto mask = word(program.sph_, 0x48);
        unsigned reg = 0;
        for (unsigned target = 0; target < 8; ++target) {
            if (((mask >> (target * 4)) & 15) == 0) continue;
            for (unsigned lane = 0; lane < 4; ++lane, ++reg) {
                if ((mask & (1u << (target * 4 + lane))) == 0) continue;
                program.exports_.push_back({OutputKind::Color,
                    static_cast<std::uint16_t>(target * 4 + lane),
                    read(mw::RegisterOperand{static_cast<std::uint8_t>(reg)})});
            }
        }
    }
}
bool Builder::instruction(const mw::Instruction& instruction) {
    require(instruction.predicate.index == 7 && !instruction.predicate.inverted,
            Error::Code::UnsupportedModifier, pc);
    using Family = mw::OpcodeFamily;
    switch (instruction.family) {
        case Family::Mov32Immediate: {
            const auto& value = payload<mw::Mov32Immediate>(instruction);
            require(value.write_mask == 15, Error::Code::UnsupportedModifier, pc);
            write(value.destination, scalar(Literal{value.immediate_bits}));
            break;
        }
        case Family::AttributeLoad:
        case Family::AttributeStore: {
            require(program.stage_ == Stage::Vertex, Error::Code::UnsupportedStage, pc);
            const auto& value = payload<mw::AttributeAccess>(instruction);
            require(value.index.is_zero() && value.vertex.is_zero() && !value.patch &&
                    value.width_code == 0 && value.byte_offset % 4 == 0,
                    Error::Code::UnsupportedModifier, pc);
            const unsigned component = value.byte_offset / 4;
            if (instruction.family == Family::AttributeLoad) {
                write(value.value, attribute(component));
            } else {
                require(declared(component, true), Error::Code::UndeclaredAttribute, pc, component);
                stored[component] = read(value.value);
            }
            break;
        }
        case Family::FloatSetRegister: {
            const auto& value = payload<mw::FloatSetRegister>(instruction);
            require(value.boolean_predicate.index == 7 && !value.boolean_predicate.inverted &&
                    value.float_boolean && value.flush == mw::FlushMode::FlushToZero &&
                    (value.comparison == mw::OrderedComparison::LessThan ||
                     value.comparison == mw::OrderedComparison::GreaterThan),
                    Error::Code::UnsupportedModifier, pc);
            const auto lhs = read(value.source_a), rhs = read(value.source_b);
            write(value.destination, scalar(FloatCompare{lhs, rhs, value.comparison, value.negate_a}));
            break;
        }
        case Family::FloatAddRegister:
        case Family::FloatAddConstant:
        case Family::FloatAddImmediate: {
            const auto& value = payload<mw::FloatAdd>(instruction);
            const bool cbuf = instruction.family == Family::FloatAddConstant;
            const bool immediate = instruction.family == Family::FloatAddImmediate;
            require(value.rounding == mw::RoundingMode::NearestEven &&
                    value.flush == mw::FlushMode::FlushToZero && (!cbuf || !value.negate_a),
                    Error::Code::UnsupportedModifier, pc);
            const auto lhs = read(value.source_a);
            ScalarId rhs{0};
            if (immediate) {
                const auto* literal = std::get_if<mw::ShortFloatImmediate>(&value.source_b);
                require(literal != nullptr, Error::Code::InvalidInstruction, pc);
                require((literal->raw_bits & 4095) == 0, Error::Code::UnsupportedModifier, pc);
                rhs = scalar(Literal{literal->raw_bits});
            } else {
                rhs = source(value.source_b, cbuf);
            }
            write(value.destination, scalar(FloatArithmetic{Arithmetic::Add, lhs, rhs,
                std::nullopt, value.negate_a}));
            break;
        }
        case Family::FloatMultiplyRegister:
        case Family::FloatMultiplyConstant:
        case Family::FloatMultiplyImmediate: {
            const auto& value = payload<mw::FloatMultiply>(instruction);
            require((value.rounding == mw::RoundingMode::NearestEven ||
                     value.rounding == mw::RoundingMode::Zero) &&
                    value.flush == mw::FlushMode::FlushToZero,
                    Error::Code::UnsupportedModifier, pc);
            const auto lhs = read(value.source_a);
            ScalarId rhs{0};
            if (instruction.family == Family::FloatMultiplyImmediate) {
                const auto* literal = std::get_if<mw::ShortFloatImmediate>(&value.source_b);
                require(literal != nullptr, Error::Code::InvalidInstruction, pc);
                require((literal->raw_bits & 4095) == 0, Error::Code::UnsupportedModifier, pc);
                rhs = scalar(Literal{literal->raw_bits});
            } else {
                rhs = source(value.source_b, instruction.family == Family::FloatMultiplyConstant);
            }
            write(value.destination, scalar(FloatArithmetic{
                value.rounding == mw::RoundingMode::Zero ? Arithmetic::MultiplyZero : Arithmetic::MultiplyNearest,
                lhs, rhs, std::nullopt, false}));
            break;
        }
        case Family::FloatFmaImmediate:
        case Family::FloatFmaConstantRegister: {
            const auto& value = payload<mw::FloatFma>(instruction);
            require(value.rounding == mw::RoundingMode::NearestEven &&
                    value.flush == mw::FlushMode::FlushToZero,
                    Error::Code::UnsupportedModifier, pc);
            const auto lhs = read(value.source_a), addend = read(value.source_c);
            ScalarId rhs{0};
            if (instruction.family == Family::FloatFmaImmediate) {
                const auto* immediate = std::get_if<mw::ShortFloatImmediate>(&value.source_b);
                require(immediate != nullptr, Error::Code::InvalidInstruction, pc);
                require((immediate->raw_bits & 4095) == 0, Error::Code::UnsupportedModifier, pc);
                rhs = scalar(Literal{immediate->raw_bits});
            } else {
                const auto* cbuf = std::get_if<mw::ConstantBufferOperand>(&value.source_b);
                require(cbuf != nullptr, Error::Code::InvalidInstruction, pc);
                rhs = constant(*cbuf);
            }
            write(value.destination, scalar(FloatArithmetic{Arithmetic::FusedMultiplyAdd,
                lhs, rhs, addend, false}));
            break;
        }
        case Family::MultiFunction: {
            const auto& value = payload<mw::Reciprocal>(instruction);
            const auto operand = read(value.source);
            write(value.destination, scalar(Reciprocal{operand}));
            break;
        }
        case Family::InterpolateAttribute: {
            require(program.stage_ == Stage::Fragment, Error::Code::UnsupportedStage, pc);
            const auto& value = payload<mw::InterpolateAttribute>(instruction);
            require(value.index.is_zero() && (value.multiply || value.multiplier.is_zero()) &&
                    value.encoding_control == 0x7ff, Error::Code::UnsupportedModifier, pc);
            const auto input = attribute(value.attribute_component);
            auto mode = Interpolation::ScreenLinear;
            std::optional<ScalarId> position_w;
            if (generic(value.attribute_component)) {
                // The first declared mode applies to the entire vector.
                const auto modes = byte(program.sph_, 0x18 + (value.attribute_component - 32) / 4);
                unsigned effective = 0;
                for (unsigned lane = 0; lane < 4 && effective == 0; ++lane) {
                    effective = (modes >> (lane * 2)) & 3;
                }
                mode = static_cast<Interpolation>(effective);
                if (mode == Interpolation::Perspective) position_w = attribute(31);
            }
            const auto multiplier = value.multiply
                ? std::optional<ScalarId>{read(value.multiplier)} : std::nullopt;
            write(value.destination, scalar(Interpolate{input, position_w, mode,
                value.encoding_control, multiplier}));
            break;
        }
        case Family::TextureSample: {
            require(program.stage_ == Stage::Fragment, Error::Code::UnsupportedStage, pc);
            const auto& value = payload<mw::TextureSample2D>(instruction);
            require(value.destination_a.index <= 252 && value.destination_b.index <= 252 &&
                    value.destination_a.index % 2 == 0 && value.destination_b.index % 2 == 0 &&
                    value.handle_byte_offset % 4 == 0 && value.handle_byte_offset <= 32764,
                    Error::Code::UnsupportedModifier, pc);
            const auto x = read(value.coordinate_x), y = read(value.coordinate_y);
            const SampleId sample{append(TextureSample{x, y, value.handle_byte_offset, value.no_dependency})};
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned base = lane < 2 ? value.destination_a.index : value.destination_b.index;
                const auto component = scalar(SampleComponent{sample, static_cast<std::uint8_t>(lane)});
                write(mw::RegisterOperand{static_cast<std::uint8_t>(base + lane % 2)}, component);
            }
            break;
        }
        case Family::Exit: {
            require(payload<mw::Exit>(instruction).flow_test == 15,
                    Error::Code::UnsupportedModifier, pc);
            exports();
            return true;
        }
        default: {
            // Legacy families all have explicit cases above; the private
            // registry dispatches the families that live in their own
            // units and unknown families fail as before.
            const detail::LowerEntry* registered =
                detail::find_registered_lowerer(instruction.family);
            if (registered == nullptr) {
                fail(Error::Code::InvalidInstruction, pc);
            }
            detail::FamilyLower lower{*this};
            registered->lower(instruction, lower);
            break;
        }
    }
    return false;
}
std::vector<Export> Builder::collect_exports() {
    exports();
    std::vector<Export> collected = std::move(program.exports_);
    program.exports_.clear();
    return collected;
}

Program Builder::run(const mw::DecodedProgram& decoded) {
    require(decoded.stage == Stage::Vertex || decoded.stage == Stage::Fragment,
            Error::Code::UnsupportedStage);
    require(decoded.instructions.size() <= limits.max_nodes, Error::Code::Limit);
    begin_graph(decoded.stage, decoded.sph);
    bool exited = false;
    for (std::size_t index = 0; index < decoded.instructions.size(); ++index) {
        require(!exited, Error::Code::InvalidInstruction, pc);
        const std::uint64_t expected = (index / 3) * 32 + (index % 3 + 1) * 8;
        require(expected == decoded.instructions[index].pc, Error::Code::InvalidInstruction,
                decoded.instructions[index].pc);
        pc = decoded.instructions[index].pc;
        exited = instruction(decoded.instructions[index]);
    }
    require(exited, Error::Code::MissingExit, pc);
    return std::move(program);
}

namespace detail {

std::optional<Error> validate_graph_interface(const GraphSpec& spec) {
    try {
        Builder builder(Limits{});
        builder.validate_graph_interface(spec);
        return std::nullopt;
    } catch (const Failure& failure) { return failure.error; }
}

FamilyLower::FamilyLower(Builder& builder) noexcept
    : builder_(&builder), context_(&builder) {}

FamilyLower::FamilyLower(const Builder& context, std::uint32_t pc,
                         ValidationOnly) noexcept
    : builder_(nullptr), context_(&context), pc_(pc) {}

std::uint32_t FamilyLower::pc() const noexcept {
    return builder_ != nullptr ? builder_->pc : pc_;
}

shader_input::Stage FamilyLower::stage() const noexcept {
    return context_->program.stage();
}

ScalarId FamilyLower::read(mw::RegisterOperand reg) {
    if (validating()) return {0};
    return builder_->read(reg);
}

void FamilyLower::write(mw::RegisterOperand reg, ScalarId value) {
    if (validating()) return;
    builder_->write(reg, value);
}

ScalarId FamilyLower::literal(std::uint32_t bits) {
    if (validating()) return {0};
    return builder_->scalar(Literal{bits});
}

ScalarId FamilyLower::arithmetic(FloatArithmetic arithmetic) {
    if (validating()) return {0};
    return builder_->scalar(std::move(arithmetic));
}
ScalarId FamilyLower::word_and(ScalarId operand, std::uint32_t mask) {
    if (validating()) return {0};
    return builder_->scalar(WordAnd{operand,mask});
}

ScalarId FamilyLower::saturate(ScalarId operand) {
    if (validating()) return {0};
    return builder_->scalar(Saturate{operand});
}

ScalarId FamilyLower::constant(mw::ConstantBufferOperand operand) {
    if (validating()) {
        // Builder::constant's admission without the load node.
        require(operand.binding < 18 && operand.byte_offset % 4 == 0 &&
                operand.byte_offset <= 65532, Error::Code::UnsupportedModifier, pc_);
        return {0};
    }
    return builder_->constant(operand);
}

namespace {
Pure pure_node(PureKind kind, std::initializer_list<OperationOperand> operands,
               std::uint32_t immediate, std::uint32_t pc) {
    Pure node{kind, {}, immediate};
    require(operands.size() <= node.operands.size(),
            Error::Code::InvalidInstruction, pc);
    std::size_t slot = 0;
    for (const auto& operand : operands) node.operands[slot++] = operand.index;
    return node;
}
}  // namespace

ScalarId FamilyLower::operation(PureKind kind,
                                std::initializer_list<OperationOperand> operands,
                                std::uint32_t immediate) {
    const auto* entry = find_operation(kind);
    require(entry != nullptr && operands.size() == entry->arity,
            Error::Code::InvalidInstruction, pc());
    if (validating()) {
        require(entry->result == ValueType::Raw32 && entry->admits(immediate),
                Error::Code::InvalidInstruction, pc_);
        return {0};
    }
    return {builder_->pure(pure_node(kind, operands, immediate, builder_->pc),
                           ValueType::Raw32)};
}

PredicateId FamilyLower::predicate_operation(
    PureKind kind, std::initializer_list<OperationOperand> operands,
    std::uint32_t immediate) {
    const auto* entry = find_operation(kind);
    require(entry != nullptr && operands.size() == entry->arity,
            Error::Code::InvalidInstruction, pc());
    if (validating()) {
        require(entry->result == ValueType::Predicate && entry->admits(immediate),
                Error::Code::InvalidInstruction, pc_);
        return {0};
    }
    return {builder_->pure(pure_node(kind, operands, immediate, builder_->pc),
                           ValueType::Predicate)};
}

PredicateId FamilyLower::read_predicate(std::uint8_t index) {
    require(index <= 7, Error::Code::InvalidInstruction, pc(), index);
    if (validating()) return {0};
    if (index == 7) return predicate_literal(true);
    require(builder_->predicates[index].has_value(),
            Error::Code::UninitializedRegister, builder_->pc, index);
    return *builder_->predicates[index];
}

void FamilyLower::write_predicate(std::uint8_t index, PredicateId value) {
    require(index <= 7, Error::Code::InvalidInstruction, pc(), index);
    if (validating()) return;
    const auto nodes = builder_->program.nodes();
    require(value.index < nodes.size() &&
            nodes[value.index].type() == ValueType::Predicate,
            Error::Code::InvalidInstruction, builder_->pc, value.index);
    if (index != 7) builder_->predicates[index] = value;
}

PredicateId FamilyLower::predicate_literal(bool value) {
    if (validating()) return {0};
    return {builder_->append(PredicateLiteral{value})};
}

PredicateId FamilyLower::negate_predicate(PredicateId operand) {
    if (validating()) return {0};
    const auto nodes = builder_->program.nodes();
    require(operand.index < nodes.size() &&
            nodes[operand.index].type() == ValueType::Predicate,
            Error::Code::InvalidInstruction, builder_->pc, operand.index);
    return {builder_->append(PredicateNot{operand})};
}

[[noreturn]] void FamilyLower::fail(Error::Code code, std::uint32_t detail) {
    throw Failure{{code, pc(), detail}};
}

}  // namespace detail

const char* Error::message() const noexcept {
    switch (code) {
        case Code::UnsupportedStage: return "unsupported graphics stage or stage-specific instruction";
        case Code::HeaderMismatch: return "shader header stage differs from decoded stage";
        case Code::UnsupportedInterface: return "shader header requests an unsupported interface";
        case Code::InvalidInstruction: return "decoded instruction metadata is inconsistent";
        case Code::UnsupportedModifier: return "decoded modifier or operand is unsupported";
        case Code::UninitializedRegister: return "register is read before being written";
        case Code::UndeclaredAttribute: return "attribute access is outside the declared interface";
        case Code::MissingOutput: return "declared vertex output has no value at exit";
        case Code::MissingExit: return "graphics program has no terminating exit";
        case Code::Limit: return "graphics SSA exceeds its node limit";
        case Code::Allocation: return "graphics metadata allocation failed";
        case Code::MissingAttribute: return "required input attribute is absent";
        case Code::DuplicateAttribute: return "input attribute is supplied more than once";
        case Code::ConstantBounds: return "constant buffer does not cover the requested word";
        case Code::MissingSampler: return "texture sample requires a sampler callback";
        case Code::SamplingFailed: return "texture sampler callback failed";
        case Code::FloatingEnvironment: return "host floating environment cannot provide the required binary32 contract";
        case Code::GraphFormat: return "control-flow graph block layout or reference is out of bounds";
        case Code::GraphShape: return "control-flow graph is unreachable, backward, or does not terminate";
        case Code::GraphTypes: return "control-flow graph operand has the wrong value type";
        case Code::GraphDataFlow: return "control-flow graph reads a value its paths do not define";
        case Code::GraphExecution: return "control-flow graph execution is not implemented yet";
    }
    return "unknown graphics error";
}

LowerResult lower(const mw::DecodedProgram& decoded, Limits limits) {
    try { return Builder{limits}.run(decoded); }
    catch (const Failure& failure) { return failure.error; }
    catch (const std::bad_alloc&) { return Error{Error::Code::Allocation}; }
    catch (const std::length_error&) { return Error{Error::Code::Allocation}; }
}

}  // namespace armigo::shader::graphics
