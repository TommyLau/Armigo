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

#include "maxwell_decode_helpers.h"
#include "forward_reconvergence.h"
#include "maxwell_families.h"
#include "maxwell_forward_decode.h"

#include <algorithm>
#include <new>
#include <stdexcept>

namespace armigo::shader::maxwell {
namespace {

// Decoder primitives moved to the private detail header; local call sites
// are unchanged.
using namespace detail;

OpcodeFamily classify(std::uint64_t word) {
    if (high_matches(word, 0xfff8, 0xefd8)) {
        return OpcodeFamily::AttributeLoad;
    }
    if (high_matches(word, 0xfff8, 0xeff0)) {
        return OpcodeFamily::AttributeStore;
    }
    if (high_matches(word, 0xfff0, 0xe300)) {
        return OpcodeFamily::Exit;
    }
    if (high_matches(word, 0xfff0, 0xe240)) {
        return OpcodeFamily::Branch;
    }
    if (high_matches(word, 0xfff8, 0x4c58)) {
        return OpcodeFamily::FloatAddConstant;
    }
    if (high_matches(word, 0xfff8, 0x5c58)) {
        return OpcodeFamily::FloatAddRegister;
    }
    if (high_matches(word, 0xfef8, 0x3858)) {
        return OpcodeFamily::FloatAddImmediate;
    }
    if (high_matches(word, 0xfff8, 0x4c68)) {
        return OpcodeFamily::FloatMultiplyConstant;
    }
    // The former register-FMUL saturation words were always rejected here as
    // an unsupported modifier; they are now the registered
    // FloatMultiplySaturateRegister family, so this classifier defers exactly
    // that window (and nothing admitted above) to the registry. The constant
    // and immediate saturation windows stay with their legacy families.
    if (high_matches(word, 0xfffc, 0x5c6c)) {
        return OpcodeFamily::Unknown;
    }
    if (high_matches(word, 0xfff8, 0x5c68)) {
        return OpcodeFamily::FloatMultiplyRegister;
    }
    if (high_matches(word, 0xfef8, 0x3868)) {
        return OpcodeFamily::FloatMultiplyImmediate;
    }
    if (high_matches(word, 0xff00, 0x5800)) {
        return OpcodeFamily::FloatSetRegister;
    }
    // The former cbuf-FFMA NegateB words were always rejected here as an
    // unsupported modifier; they are now the registered
    // FloatFmaConstantNegated family, so this classifier defers exactly that
    // window (and nothing admitted above) to the registry.
    if (high_matches(word, 0xff81, 0x4981)) {
        return OpcodeFamily::Unknown;
    }
    if (high_matches(word, 0xff80, 0x4980)) {
        return OpcodeFamily::FloatFmaConstantRegister;
    }
    if (high_matches(word, 0xfe80, 0x3280)) {
        return OpcodeFamily::FloatFmaImmediate;
    }
    if (high_matches(word, 0xfff8, 0x5080)) {
        return OpcodeFamily::MultiFunction;
    }
    if (high_matches(word, 0xff00, 0xe000)) {
        return OpcodeFamily::InterpolateAttribute;
    }
    if (high_matches(word, 0xf600, 0xd000)) {
        return OpcodeFamily::TextureSample;
    }
    if (high_matches(word, 0xfff0, 0x0100)) {
        return OpcodeFamily::Mov32Immediate;
    }
    // Families classified above always win; the private registry is consulted
    // only for words this classifier leaves Unknown, so it can never shadow a
    // legacy pattern.
    return OpcodeFamily::Unknown;
}

// Predicate admission shared by the strict decode() path and the guarded
// forward-program walk. RequirePT is the production straight-line behavior;
// AnyGuard admits all 16 raw guard values and preserves them verbatim.
enum class PredicateAdmission : std::uint8_t { RequirePT, AnyGuard };

std::variant<Instruction, DecodeError> decode_instruction_with_admission(
    std::uint64_t word, std::uint32_t pc, PredicateAdmission admission) {
    // SSY, SYNC, PBK and BRK are reconvergence control flow. The strict
    // straight-line path reports them as unsupported control flow, exactly
    // like BRA, so callers fall back to the forward decoder; the forward walk
    // resolves SSY/SYNC itself before reaching this function and rejects the
    // rest explicitly.
    if (detail::reconvergence_opcode(word) != detail::ReconvergenceOpcode::None) {
        return failure(DecodeError::Code::UnsupportedControlFlow, pc,
                       OpcodeFamily::Unknown);
    }
    OpcodeFamily family = classify(word);
    const detail::FamilyEntry* registered = nullptr;
    if (family == OpcodeFamily::Unknown) {
        registered = detail::find_registered_family(word);
        if (registered == nullptr) {
            return failure(DecodeError::Code::UnknownOpcode, pc, family);
        }
        family = registered->family;
    }

    const std::uint8_t predicate_raw =
        static_cast<std::uint8_t>(field(word, 16, 4));
    if (admission == PredicateAdmission::RequirePT && predicate_raw != 7) {
        return failure(DecodeError::Code::UnsupportedPredicate, pc, family,
                       DecodeField::InstructionPredicate, predicate_raw, 7);
    }
    if (family == OpcodeFamily::Branch) {
        return failure(DecodeError::Code::UnsupportedControlFlow, pc, family);
    }
    // PT is raw 7 and its inversion raw 15. The strict path has already
    // pinned the raw guard to 7, so both admissions share this extraction;
    // guarded instructions keep all four guard bits as Predicate.
    const Predicate predicate{
        static_cast<std::uint8_t>(predicate_raw & 7), predicate_raw >= 8};
    const std::uint64_t common = bit_mask(4, 16);
    switch (family) {
        case OpcodeFamily::Mov32Immediate: {
            const std::uint64_t allowed = high_mask(0xfff0) |
                                          bit_mask(8, 0) | bit_mask(4, 12) |
                                          common | bit_mask(32, 20);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            const auto mask = static_cast<std::uint8_t>(field(word, 12, 4));
            if (auto error = unsupported(pc, family, DecodeField::WriteMask,
                                         mask, 15)) {
                return *error;
            }
            return Instruction{pc, word, family, predicate,
                               Mov32Immediate{
                                   reg(word, 0), mask,
                                   static_cast<std::uint32_t>(
                                       field(word, 20, 32))}};
        }
        case OpcodeFamily::AttributeLoad:
        case OpcodeFamily::AttributeStore: {
            std::uint64_t allowed = high_mask(0xfff8) | bit_mask(8, 0) |
                                    bit_mask(8, 8) | common |
                                    bit_mask(11, 20) | bit_mask(1, 31) |
                                    bit_mask(8, 39) | bit_mask(2, 47);
            if (family == OpcodeFamily::AttributeLoad) {
                allowed |= bit_mask(1, 32);
            }
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            const auto index = reg(word, 8);
            const auto vertex = reg(word, 39);
            const auto offset = static_cast<std::uint16_t>(field(word, 20, 10));
            const auto width = static_cast<std::uint8_t>(field(word, 47, 2));
            if (auto error = unsupported(pc, family,
                                         DecodeField::IndexRegister,
                                         index.index, 255)) {
                return *error;
            }
            if (auto error = unsupported(pc, family,
                                         DecodeField::VertexRegister,
                                         vertex.index, 255)) {
                return *error;
            }
            if (auto error = unsupported(pc, family,
                                         DecodeField::AttributeWidth, width,
                                         0)) {
                return *error;
            }
            if (auto error = unsupported(pc, family, DecodeField::Patch,
                                         field(word, 31, 1), 0)) {
                return *error;
            }
            if (auto error = unsupported(pc, family, DecodeField::AddressMode,
                                         field(word, 30, 1), 0)) {
                return *error;
            }
            if (family == OpcodeFamily::AttributeLoad) {
                if (auto error = unsupported(pc, family,
                                             DecodeField::AddressMode,
                                             field(word, 32, 1), 0)) {
                    return *error;
                }
            }
            if (offset % 4 != 0) {
                return failure(DecodeError::Code::UnsupportedModifier, pc,
                               family, DecodeField::AttributeOffset, offset,
                               offset & ~std::uint16_t{3});
            }
            return Instruction{
                pc, word, family, predicate,
                AttributeAccess{reg(word, 0), index, vertex, offset, width,
                                false}};
        }
        case OpcodeFamily::FloatSetRegister: {
            const std::uint64_t allowed = high_mask(0xff00) |
                                          bit_mask(8, 0) | bit_mask(8, 8) |
                                          common | bit_mask(8, 20) |
                                          bit_mask(3, 39) | bit_mask(1, 42) |
                                          bit_mask(1, 43) | bit_mask(1, 44) |
                                          bit_mask(2, 45) | bit_mask(1, 47) |
                                          bit_mask(4, 48) | bit_mask(1, 52) |
                                          bit_mask(1, 53) | bit_mask(1, 54) |
                                          bit_mask(1, 55);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            const auto bool_pred =
                static_cast<std::uint8_t>(field(word, 39, 3));
            const auto comparison =
                static_cast<std::uint8_t>(field(word, 48, 4));
            for (const auto [decode_field, value, expected] :
                 {std::array<std::uint64_t, 3>{
                      static_cast<std::uint64_t>(DecodeField::BooleanPredicate),
                      bool_pred, 7},
                  {static_cast<std::uint64_t>(
                       DecodeField::BooleanPredicateInversion),
                   field(word, 42, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::AbsoluteB),
                   field(word, 44, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::BooleanOperation),
                   field(word, 45, 2), 0},
                  {static_cast<std::uint64_t>(DecodeField::ConditionCode),
                   field(word, 47, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::BooleanFloat),
                   field(word, 52, 1), 1},
                  {static_cast<std::uint64_t>(DecodeField::NegateB),
                   field(word, 53, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::AbsoluteA),
                   field(word, 54, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::FlushMode),
                   field(word, 55, 1), 1}}) {
                if (auto error = unsupported(
                        pc, family, static_cast<DecodeField>(decode_field),
                        value, expected)) {
                    return *error;
                }
            }
            if (comparison != 1 && comparison != 4) {
                return failure(DecodeError::Code::UnsupportedModifier, pc,
                               family, DecodeField::Comparison, comparison,
                               1);
            }
            return Instruction{
                pc, word, family, predicate,
                FloatSetRegister{
                    reg(word, 0), reg(word, 8), reg(word, 20),
                    Predicate{bool_pred, false},
                    static_cast<OrderedComparison>(comparison),
                    field(word, 43, 1) != 0, true,
                    FlushMode::FlushToZero}};
        }
        case OpcodeFamily::FloatAddRegister:
        case OpcodeFamily::FloatAddConstant:
        case OpcodeFamily::FloatAddImmediate: {
            const bool is_constant =
                family == OpcodeFamily::FloatAddConstant;
            const bool is_immediate =
                family == OpcodeFamily::FloatAddImmediate;
            const std::uint64_t opcode =
                high_mask(is_immediate ? 0xfef8 : 0xfff8);
            const std::uint64_t source =
                is_immediate ? bit_mask(19, 20) | bit_mask(1, 56)
                             : (is_constant ? bit_mask(19, 20)
                                            : bit_mask(8, 20));
            const std::uint64_t allowed =
                opcode | bit_mask(8, 0) | bit_mask(8, 8) | common | source |
                bit_mask(2, 39) | bit_mask(7, 44);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            for (const auto [decode_field, value, expected] :
                 {std::array<std::uint64_t, 3>{
                      static_cast<std::uint64_t>(DecodeField::Rounding),
                      field(word, 39, 2), 0},
                  {static_cast<std::uint64_t>(DecodeField::FlushMode),
                   field(word, 44, 1), 1},
                  {static_cast<std::uint64_t>(DecodeField::NegateB),
                   field(word, 45, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::AbsoluteA),
                   field(word, 46, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::ConditionCode),
                   field(word, 47, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::AbsoluteB),
                   field(word, 49, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::Saturate),
                   field(word, 50, 1), 0}}) {
                if (auto error = unsupported(
                        pc, family, static_cast<DecodeField>(decode_field),
                        value, expected)) {
                    return *error;
                }
            }
            if (is_constant && field(word, 48, 1) != 0) {
                return failure(DecodeError::Code::UnsupportedModifier, pc,
                               family, DecodeField::NegateA, 1, 0);
            }
            FloatSource source_b = reg(word, 20);
            if (is_immediate) {
                source_b = ShortFloatImmediate{
                    static_cast<std::uint32_t>(field(word, 20, 19) << 12) |
                    static_cast<std::uint32_t>(field(word, 56, 1) << 31)};
            } else if (is_constant) {
                auto operand = constant_buffer(word, pc, family);
                if (auto* error = std::get_if<DecodeError>(&operand)) {
                    return *error;
                }
                source_b = std::get<ConstantBufferOperand>(operand);
            }
            return Instruction{
                pc, word, family, predicate,
                FloatAdd{reg(word, 0), reg(word, 8), source_b,
                         RoundingMode::NearestEven, FlushMode::FlushToZero,
                         field(word, 48, 1) != 0}};
        }
        case OpcodeFamily::FloatMultiplyRegister:
        case OpcodeFamily::FloatMultiplyConstant:
        case OpcodeFamily::FloatMultiplyImmediate: {
            const bool is_constant =
                family == OpcodeFamily::FloatMultiplyConstant;
            const bool is_immediate =
                family == OpcodeFamily::FloatMultiplyImmediate;
            const std::uint64_t opcode =
                high_mask(is_immediate ? 0xfef8 : 0xfff8);
            const std::uint64_t source =
                is_immediate ? bit_mask(19, 20) | bit_mask(1, 56)
                             : (is_constant ? bit_mask(19, 20)
                                            : bit_mask(8, 20));
            const std::uint64_t allowed =
                opcode | bit_mask(8, 0) | bit_mask(8, 8) |
                common | source | bit_mask(7, 39) | bit_mask(1, 47) |
                bit_mask(1, 48) | bit_mask(1, 50);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            const auto rounding =
                static_cast<std::uint8_t>(field(word, 39, 2));
            if (rounding != 0 && rounding != 3) {
                return failure(DecodeError::Code::UnsupportedModifier, pc,
                               family, DecodeField::Rounding, rounding, 0);
            }
            for (const auto [decode_field, value, expected] :
                 {std::array<std::uint64_t, 3>{
                      static_cast<std::uint64_t>(DecodeField::Scale),
                      field(word, 41, 3), 0},
                  {static_cast<std::uint64_t>(DecodeField::FlushMode),
                   field(word, 44, 2), 1},
                  {static_cast<std::uint64_t>(DecodeField::ConditionCode),
                   field(word, 47, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::NegateB),
                   field(word, 48, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::Saturate),
                   field(word, 50, 1), 0}}) {
                if (auto error = unsupported(
                        pc, family, static_cast<DecodeField>(decode_field),
                        value, expected)) {
                    return *error;
                }
            }
            FloatSource source_b = reg(word, 20);
            if (is_immediate) {
                source_b = ShortFloatImmediate{
                    static_cast<std::uint32_t>(field(word, 20, 19) << 12) |
                    static_cast<std::uint32_t>(field(word, 56, 1) << 31)};
            } else if (is_constant) {
                auto operand = constant_buffer(word, pc, family);
                if (auto* error = std::get_if<DecodeError>(&operand)) {
                    return *error;
                }
                source_b = std::get<ConstantBufferOperand>(operand);
            }
            return Instruction{
                pc, word, family, predicate,
                FloatMultiply{reg(word, 0), reg(word, 8), source_b,
                              static_cast<RoundingMode>(rounding),
                              FlushMode::FlushToZero}};
        }
        case OpcodeFamily::FloatFmaImmediate:
        case OpcodeFamily::FloatFmaConstantRegister: {
            const bool is_immediate =
                family == OpcodeFamily::FloatFmaImmediate;
            const std::uint64_t opcode =
                high_mask(is_immediate ? 0xfe80 : 0xff80);
            const std::uint64_t source = is_immediate
                                             ? bit_mask(19, 20) |
                                                   bit_mask(1, 56)
                                             : bit_mask(19, 20);
            const std::uint64_t allowed =
                opcode | bit_mask(8, 0) | bit_mask(8, 8) | common |
                source | bit_mask(8, 39) | bit_mask(8, 47);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            for (const auto [decode_field, value, expected] :
                 {std::array<std::uint64_t, 3>{
                      static_cast<std::uint64_t>(DecodeField::ConditionCode),
                      field(word, 47, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::NegateB),
                   field(word, 48, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::NegateC),
                   field(word, 49, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::Saturate),
                   field(word, 50, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::Rounding),
                   field(word, 51, 2), 0},
                  {static_cast<std::uint64_t>(DecodeField::FlushMode),
                   field(word, 53, 2), 1}}) {
                if (auto error = unsupported(
                        pc, family, static_cast<DecodeField>(decode_field),
                        value, expected)) {
                    return *error;
                }
            }
            FmaSource source_b = ShortFloatImmediate{};
            if (is_immediate) {
                source_b = ShortFloatImmediate{
                    static_cast<std::uint32_t>(field(word, 20, 19) << 12) |
                    static_cast<std::uint32_t>(field(word, 56, 1) << 31)};
            } else {
                auto operand = constant_buffer(word, pc, family);
                if (auto* error = std::get_if<DecodeError>(&operand)) {
                    return *error;
                }
                source_b = std::get<ConstantBufferOperand>(operand);
            }
            return Instruction{
                pc, word, family, predicate,
                FloatFma{reg(word, 0), reg(word, 8), source_b,
                         reg(word, 39), RoundingMode::NearestEven,
                         FlushMode::FlushToZero}};
        }
        case OpcodeFamily::MultiFunction: {
            const std::uint64_t allowed =
                high_mask(0xfff8) | bit_mask(8, 0) | bit_mask(8, 8) |
                common | bit_mask(4, 20) | bit_mask(1, 46) |
                bit_mask(1, 48) | bit_mask(1, 50);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            for (const auto [decode_field, value, expected] :
                 {std::array<std::uint64_t, 3>{
                      static_cast<std::uint64_t>(DecodeField::Function),
                      field(word, 20, 4), 4},
                  {static_cast<std::uint64_t>(DecodeField::AbsoluteA),
                   field(word, 46, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::NegateA),
                   field(word, 48, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::Saturate),
                   field(word, 50, 1), 0}}) {
                if (auto error = unsupported(
                        pc, family, static_cast<DecodeField>(decode_field),
                        value, expected)) {
                    return *error;
                }
            }
            return Instruction{pc, word, family, predicate,
                               Reciprocal{reg(word, 0), reg(word, 8)}};
        }
        case OpcodeFamily::InterpolateAttribute: {
            const std::uint64_t allowed =
                high_mask(0xff00) | bit_mask(8, 0) | bit_mask(8, 8) |
                common | bit_mask(8, 20) | bit_mask(8, 30) |
                bit_mask(1, 38) | bit_mask(11, 39) | bit_mask(1, 51) |
                bit_mask(2, 52) | bit_mask(2, 54);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            const auto index = reg(word, 8);
            const auto multiplier = reg(word, 20);
            const auto mode = field(word, 54, 2);
            if (auto error = unsupported(pc, family, DecodeField::InterpolationMode,
                                         mode > 1 ? mode : 0, 0)) {
                return *error;
            }
            for (const auto [decode_field, value, expected] :
                 {std::array<std::uint64_t, 3>{
                      static_cast<std::uint64_t>(DecodeField::IndexRegister),
                      index.index, 255},
                  {static_cast<std::uint64_t>(
                       DecodeField::MultiplierRegister),
                   mode == 1 ? 255u : multiplier.index, 255},
                  {static_cast<std::uint64_t>(DecodeField::Indexed),
                   field(word, 38, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::Saturate),
                   field(word, 51, 1), 0},
                  {static_cast<std::uint64_t>(DecodeField::SampleMode),
                   field(word, 52, 2), 0},
                  {static_cast<std::uint64_t>(
                       DecodeField::InterpolationEncodingControl),
                   field(word, 39, 11), 0x7ff}}) {
                if (auto error = unsupported(
                        pc, family, static_cast<DecodeField>(decode_field),
                        value, expected)) {
                    return *error;
                }
            }
            return Instruction{
                pc, word, family, predicate,
                InterpolateAttribute{
                    reg(word, 0), index, multiplier,
                    static_cast<std::uint8_t>(field(word, 30, 8)),
                    static_cast<std::uint16_t>(field(word, 39, 11)), mode == 1}};
        }
        case OpcodeFamily::TextureSample: {
            const std::uint64_t allowed =
                high_mask(0xf600) | bit_mask(8, 0) | bit_mask(8, 8) |
                common | bit_mask(8, 20) | bit_mask(8, 28) |
                bit_mask(13, 36) | bit_mask(1, 49) | bit_mask(3, 50) |
                bit_mask(4, 53) | bit_mask(1, 59);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            const auto destination_a = reg(word, 0);
            const auto destination_b = reg(word, 28);
            for (const auto [decode_field, value, expected] :
                 {std::array<std::uint64_t, 3>{
                      static_cast<std::uint64_t>(
                          DecodeField::TexturePrecision),
                      field(word, 59, 1), 1},
                  {static_cast<std::uint64_t>(
                       DecodeField::TextureEncoding),
                   field(word, 53, 4), 1},
                  {static_cast<std::uint64_t>(DecodeField::TextureSwizzle),
                   field(word, 50, 3), 4}}) {
                if (auto error = unsupported(
                        pc, family, static_cast<DecodeField>(decode_field),
                        value, expected)) {
                    return *error;
                }
            }
            if (destination_a.index > 252 || destination_b.index > 252 ||
                (destination_a.index & 1) != 0 ||
                (destination_b.index & 1) != 0) {
                return failure(DecodeError::Code::UnsupportedModifier, pc,
                               family, DecodeField::TextureDestination,
                               static_cast<std::uint64_t>(destination_a.index) |
                                   (static_cast<std::uint64_t>(
                                        destination_b.index)
                                    << 8),
                               0);
            }
            return Instruction{
                pc, word, family, predicate,
                TextureSample2D{
                    destination_a, destination_b, reg(word, 8), reg(word, 20),
                    static_cast<std::uint32_t>(field(word, 36, 13) * 4),
                    field(word, 49, 1) != 0}};
        }
        case OpcodeFamily::Exit: {
            const std::uint64_t allowed =
                high_mask(0xfff0) | common | bit_mask(5, 0);
            if (auto error = reserved(pc, family, word, allowed)) {
                return *error;
            }
            const auto flow_test =
                static_cast<std::uint8_t>(field(word, 0, 5));
            if (flow_test != 15) {
                return failure(DecodeError::Code::UnsupportedControlFlow, pc,
                               family, DecodeField::FlowTest, flow_test, 15);
            }
            return Instruction{pc, word, family, predicate, Exit{flow_test}};
        }
        case OpcodeFamily::Unknown:
        case OpcodeFamily::Branch:
            break;

        default:
            // Only registry-provided families can reach this point; legacy
            // families all have explicit cases above.
            if (registered != nullptr) {
                return registered->decode(word, pc, predicate);
            }
            break;
    }
    return failure(DecodeError::Code::UnknownOpcode, pc, family);
}

std::variant<Instruction, DecodeError> decode_instruction(std::uint64_t word,
                                                          std::uint32_t pc) {
    return decode_instruction_with_admission(word, pc,
                                             PredicateAdmission::RequirePT);
}

}  // namespace

std::variant<Instruction, DecodeError> decode_guarded_instruction(
    std::uint64_t word, std::uint32_t pc) {
    return decode_instruction_with_admission(word, pc,
                                             PredicateAdmission::AnyGuard);
}

const char* DecodeError::message() const noexcept {
    switch (code) {
        case Code::ShaderParse:
            return "shader input failed structural validation";
        case Code::UnsupportedStage:
            return "shader stage is outside the bounded decoder scope";
        case Code::InstructionLimit:
            return "reachable instruction count exceeds the decoder limit";
        case Code::AllocationFailure:
            return "decoder metadata allocation failed";
        case Code::UnsupportedPredicate:
            return "reachable instruction is not predicated on PT";
        case Code::UnsupportedControlFlow:
            return "reachable instruction requires unsupported control flow";
        case Code::UnknownOpcode:
            return "reachable instruction opcode is unsupported";
        case Code::UnsupportedModifier:
            return "reachable instruction modifier or operand form is unsupported";
        case Code::MissingExit:
            return "straight-line program has no unconditional PT EXIT";
    }
    return "unknown Maxwell decode error";
}

DecodeResult decode(shader_input::ByteSpan control, shader_input::ByteSpan code,
                    DecodeLimits limits) {
    const shader_input::ParseResult parsed_result =
        shader_input::parse_shader({control, code, {}});
    if (const auto* parse_error =
            std::get_if<shader_input::ParseError>(&parsed_result)) {
        DecodeError error = failure(DecodeError::Code::ShaderParse, 0);
        error.parse_error = *parse_error;
        return error;
    }

    const shader_input::ParsedShader& parsed =
        std::get<shader_input::ParsedShader>(parsed_result);
    if (parsed.stage != shader_input::Stage::Vertex &&
        parsed.stage != shader_input::Stage::Fragment) {
        return failure(DecodeError::Code::UnsupportedStage, 0,
                       OpcodeFamily::Unknown, DecodeField::None,
                       static_cast<std::uint32_t>(parsed.stage), 0);
    }

    DecodedProgram result;
    result.stage = parsed.stage;
    std::copy_n(reinterpret_cast<const std::byte*>(parsed.sph.data),
                result.sph.size(), result.sph.begin());
    try {
        result.instructions.reserve(
            std::min<std::size_t>(parsed.instruction_slot_count(),
                                  limits.max_instructions));
        for (std::uint32_t index = 0;
             index < parsed.instruction_slot_count(); ++index) {
            const shader_input::InstructionSlot slot =
                parsed.instruction_slot(index);
            if (result.instructions.size() == limits.max_instructions) {
                return failure(DecodeError::Code::InstructionLimit, slot.pc,
                               OpcodeFamily::Unknown, DecodeField::None,
                               result.instructions.size() + 1,
                               limits.max_instructions);
            }
            const std::uint64_t word =
                read_word(parsed.code.data + slot.code_offset);
            auto decoded = decode_instruction(word, slot.pc);
            if (auto* error = std::get_if<DecodeError>(&decoded)) {
                return *error;
            }
            result.instructions.push_back(std::get<Instruction>(decoded));
            if (result.instructions.back().family == OpcodeFamily::Exit) {
                return result;
            }
        }
    } catch (const std::bad_alloc&) {
        return failure(DecodeError::Code::AllocationFailure, 0);
    } catch (const std::length_error&) {
        return failure(DecodeError::Code::AllocationFailure, 0);
    }
    return failure(DecodeError::Code::MissingExit,
                   parsed.program_size == 0
                       ? 0
                       : parsed.program_size -
                             shader_input::format::instruction_word_size);
}

}  // namespace armigo::shader::maxwell
