// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "forward_metadata.h"
#include "graphics_families.h"

#include "graphics_builder.h"

#include <variant>

namespace armigo::shader::graphics::detail {
namespace {

namespace mw = maxwell;
using Stage = shader_input::Stage;
using Failure = BuilderFailure;

// The error triple mirrors Builder::require: the pc is always the validated
// instruction's own pc, and details carry the offending operand where the
// Builder reports one.
[[noreturn]] void fail(Error::Code code, std::uint32_t pc,
                       std::uint32_t detail = 0) {
    throw Failure{{code, pc, detail}};
}

void require(bool condition, Error::Code code, std::uint32_t pc,
             std::uint32_t detail = 0) {
    if (!condition) fail(code, pc, detail);
}

// The shared constant-buffer admission of Builder::constant and the
// ConstantLoad interface check: binding below 18, word-aligned offset
// within the 64 KiB buffer.
void require_constant(const mw::ConstantBufferOperand& operand,
                      std::uint32_t pc) {
    require(operand.binding < 18 && operand.byte_offset % 4 == 0 &&
                operand.byte_offset <= 65532,
            Error::Code::UnsupportedModifier, pc);
}

bool is_pt(const mw::Predicate& predicate) {
    return predicate.index == 7 && !predicate.inverted;
}

bool generic(unsigned component) {
    return component >= 32 && component < 160;
}

// Effective interpolation mode of a declared generic input vector, derived
// from the SPH exactly as Builder::instruction does: the first nonzero
// 2-bit lane map of the vector's header byte.
unsigned declared_interpolation_mode(const Builder& context,
                                     unsigned component) {
    const unsigned char modes = std::to_integer<unsigned char>(
        context.program.sph()[0x18 + (component - 32) / 4]);
    unsigned effective = 0;
    for (unsigned lane = 0; lane < 4 && effective == 0; ++lane) {
        effective = (modes >> (lane * 2)) & 3;
    }
    return effective;
}

void validate(const mw::Instruction& instruction, const Builder& context) {
    const std::uint32_t pc = instruction.pc;
    // The forward walk preserves all four guard bits as index 0..7 plus the
    // inversion flag, so any larger index is inconsistent metadata. PT and
    // inverted PT are both admissible guards here.
    require(instruction.predicate.index < 8, Error::Code::InvalidInstruction,
            pc, instruction.predicate.index);
    using Family = mw::OpcodeFamily;
    switch (instruction.family) {
        case Family::Mov32Immediate: {
            const auto* value =
                std::get_if<mw::Mov32Immediate>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->write_mask == 15, Error::Code::UnsupportedModifier,
                    pc);
            break;
        }
        case Family::MovRegister: {
            const auto* value =
                std::get_if<mw::MovRegister>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->write_mask == 15 || value->write_mask == 1,
                    Error::Code::UnsupportedModifier, pc);
            break;
        }
        case Family::MovConstant: {
            const auto* value =
                std::get_if<mw::MovConstant>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->write_mask == 15 || value->write_mask == 1,
                    Error::Code::UnsupportedModifier, pc);
            require_constant(value->source, pc);
            break;
        }
        case Family::AttributeLoad:
        case Family::AttributeStore: {
            require(context.program.stage() == Stage::Vertex,
                    Error::Code::UnsupportedStage, pc);
            const auto* value =
                std::get_if<mw::AttributeAccess>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->index.is_zero() && value->vertex.is_zero() &&
                        !value->patch && value->width_code == 0 &&
                        value->byte_offset % 4 == 0,
                    Error::Code::UnsupportedModifier, pc);
            const unsigned component = value->byte_offset / 4;
            require(context.declared(component,
                                     instruction.family == Family::AttributeStore),
                    Error::Code::UndeclaredAttribute, pc, component);
            break;
        }
        case Family::FloatSetRegister: {
            const auto* value =
                std::get_if<mw::FloatSetRegister>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(is_pt(value->boolean_predicate) && value->float_boolean &&
                        value->flush == mw::FlushMode::FlushToZero &&
                        (value->comparison == mw::OrderedComparison::LessThan ||
                         value->comparison ==
                             mw::OrderedComparison::GreaterThan),
                    Error::Code::UnsupportedModifier, pc);
            break;
        }
        case Family::FloatAddRegister:
        case Family::FloatAddConstant:
        case Family::FloatAddImmediate: {
            const auto* value = std::get_if<mw::FloatAdd>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            const bool constant = instruction.family == Family::FloatAddConstant;
            require(value->rounding == mw::RoundingMode::NearestEven &&
                        value->flush == mw::FlushMode::FlushToZero &&
                        (!constant || !value->negate_a),
                    Error::Code::UnsupportedModifier, pc);
            if (instruction.family == Family::FloatAddImmediate) {
                const auto* literal =
                    std::get_if<mw::ShortFloatImmediate>(&value->source_b);
                require(literal != nullptr, Error::Code::InvalidInstruction, pc);
                require((literal->raw_bits & 4095) == 0,
                        Error::Code::UnsupportedModifier, pc);
            } else if (constant) {
                const auto* buffer =
                    std::get_if<mw::ConstantBufferOperand>(&value->source_b);
                require(buffer != nullptr, Error::Code::InvalidInstruction, pc);
                require_constant(*buffer, pc);
            } else {
                const auto* register_source =
                    std::get_if<mw::RegisterOperand>(&value->source_b);
                require(register_source != nullptr,
                        Error::Code::InvalidInstruction, pc);
            }
            break;
        }
        case Family::FloatMultiplyRegister:
        case Family::FloatMultiplyConstant:
        case Family::FloatMultiplyImmediate:
        case Family::FloatMultiplySaturateRegister: {
            const auto* value =
                std::get_if<mw::FloatMultiply>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            // The saturated register form decodes only the nearest-even
            // profile; the plain register form additionally admits the
            // zero-rounding multiply. Both require flush-to-zero.
            const bool saturated =
                instruction.family == Family::FloatMultiplySaturateRegister;
            require((value->rounding == mw::RoundingMode::NearestEven ||
                     (!saturated &&
                      value->rounding == mw::RoundingMode::Zero)) &&
                        value->flush == mw::FlushMode::FlushToZero,
                    Error::Code::UnsupportedModifier, pc);
            if (instruction.family == Family::FloatMultiplyImmediate) {
                const auto* literal =
                    std::get_if<mw::ShortFloatImmediate>(&value->source_b);
                require(literal != nullptr, Error::Code::InvalidInstruction, pc);
                require((literal->raw_bits & 4095) == 0,
                        Error::Code::UnsupportedModifier, pc);
            } else if (instruction.family == Family::FloatMultiplyConstant) {
                const auto* buffer =
                    std::get_if<mw::ConstantBufferOperand>(&value->source_b);
                require(buffer != nullptr, Error::Code::InvalidInstruction, pc);
                require_constant(*buffer, pc);
            } else {
                const auto* register_source =
                    std::get_if<mw::RegisterOperand>(&value->source_b);
                require(register_source != nullptr,
                        Error::Code::InvalidInstruction, pc);
            }
            break;
        }
        case Family::FloatFmaImmediate:
        case Family::FloatFmaConstantRegister: {
            const auto* value = std::get_if<mw::FloatFma>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->rounding == mw::RoundingMode::NearestEven &&
                        value->flush == mw::FlushMode::FlushToZero,
                    Error::Code::UnsupportedModifier, pc);
            if (instruction.family == Family::FloatFmaImmediate) {
                const auto* immediate =
                    std::get_if<mw::ShortFloatImmediate>(&value->source_b);
                require(immediate != nullptr,
                        Error::Code::InvalidInstruction, pc);
                require((immediate->raw_bits & 4095) == 0,
                        Error::Code::UnsupportedModifier, pc);
            } else {
                const auto* buffer =
                    std::get_if<mw::ConstantBufferOperand>(&value->source_b);
                require(buffer != nullptr, Error::Code::InvalidInstruction, pc);
                require_constant(*buffer, pc);
            }
            break;
        }
        case Family::FloatFmaRegisterConstant: {
            const auto* value =
                std::get_if<mw::FloatFmaRegisterConstant>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->rounding == mw::RoundingMode::NearestEven &&
                        value->flush == mw::FlushMode::FlushToZero,
                    Error::Code::UnsupportedModifier, pc);
            require_constant(value->source_c, pc);
            break;
        }
        case Family::FloatFmaConstantNegated: {
            const auto* value =
                std::get_if<mw::FloatFmaConstantNegated>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->rounding == mw::RoundingMode::NearestEven &&
                        value->flush == mw::FlushMode::FlushToZero,
                    Error::Code::UnsupportedModifier, pc);
            require_constant(value->multiplier, pc);
            break;
        }
        case Family::WordAndImmediate: {
            require(std::get_if<mw::WordAndImmediate>(&instruction.payload) !=
                        nullptr,
                    Error::Code::InvalidInstruction, pc);
            break;
        }
        case Family::MultiFunction: {
            require(std::get_if<mw::Reciprocal>(&instruction.payload) !=
                        nullptr,
                    Error::Code::InvalidInstruction, pc);
            break;
        }
        case Family::InterpolateAttribute: {
            require(context.program.stage() == Stage::Fragment,
                    Error::Code::UnsupportedStage, pc);
            const auto* value =
                std::get_if<mw::InterpolateAttribute>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->index.is_zero() &&
                        (value->multiply || value->multiplier.is_zero()) &&
                        value->encoding_control == 0x7ff,
                    Error::Code::UnsupportedModifier, pc);
            require(context.declared(value->attribute_component, false),
                    Error::Code::UndeclaredAttribute, pc,
                    value->attribute_component);
            if (generic(value->attribute_component) &&
                declared_interpolation_mode(context,
                                            value->attribute_component) ==
                    static_cast<unsigned>(Interpolation::Perspective)) {
                // A perspective vector additionally reads position.w, whose
                // declaredness the Builder checks when it loads component 31.
                require(context.declared(31, false),
                        Error::Code::UndeclaredAttribute, pc, 31);
            }
            break;
        }
        case Family::TextureSample: {
            require(context.program.stage() == Stage::Fragment,
                    Error::Code::UnsupportedStage, pc);
            const auto* value =
                std::get_if<mw::TextureSample2D>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->destination_a.index <= 252 &&
                        value->destination_b.index <= 252 &&
                        value->destination_a.index % 2 == 0 &&
                        value->destination_b.index % 2 == 0 &&
                        value->handle_byte_offset % 4 == 0 &&
                        value->handle_byte_offset <= 32764,
                    Error::Code::UnsupportedModifier, pc);
            break;
        }
        case Family::Exit: {
            const auto* value = std::get_if<mw::Exit>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->flow_test == 15, Error::Code::UnsupportedModifier,
                    pc);
            break;
        }
        case Family::IntegerSetPredicateRegister: {
            const auto* value =
                std::get_if<mw::IntegerSetPredicate>(&instruction.payload);
            require(value != nullptr, Error::Code::InvalidInstruction, pc);
            require(value->destination_a < 8 && value->destination_b < 8,
                    Error::Code::InvalidInstruction, pc);
            // The decoded profile combines not-equal with the PT source
            // predicate; the signed-compare flag is carried raw and always
            // admissible because raw-word equality ignores it.
            require(is_pt(value->source), Error::Code::UnsupportedModifier, pc);
            break;
        }
        case Family::PredicateSetPredicate:
            // Predicate-set-predicate decodes but no lowering implements it;
            // a forward program carrying one is an unsupported modifier even
            // under a false guard.
            fail(Error::Code::UnsupportedModifier, pc);
        case Family::Unknown:
        case Family::Branch:
            // Branches never reach instruction metadata, and an unknown
            // family has no admitted syntax; both fail exactly as the
            // Builder's unregistered-family path does.
            fail(Error::Code::InvalidInstruction, pc);
        default:
            // A registered family without a case above is admitted by its
            // own lowerer run in validation-only mode: the same payload,
            // modifier and constant-buffer checks, with no nodes, writes or
            // definedness requirements, so a guard-false instruction is
            // never an unvalidated no-op.
            if (const auto* entry = find_registered_lowerer(instruction.family)) {
                FamilyLower check(context, pc, FamilyLower::ValidationOnly{});
                entry->lower(instruction, check);
                break;
            }
            fail(Error::Code::InvalidInstruction, pc);
    }
}

}  // namespace

std::optional<Error> validate_forward_instruction_metadata(
    const maxwell::Instruction& instruction, const Builder& context) {
    try {
        validate(instruction, context);
        return std::nullopt;
    } catch (const Failure& failure) {
        return failure.error;
    }
}

}  // namespace armigo::shader::graphics::detail
