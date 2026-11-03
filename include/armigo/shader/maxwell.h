// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Bounded metadata decoder for the first straight-line Maxwell subset.
#pragma once
#include <armigo/shader/maxwell_operands.h>
#include <armigo/shader/families/mov_constant.h>
#include <armigo/shader/families/fma_register_constant.h>
#include <armigo/shader/families/fma_constant_negated.h>
#include <armigo/shader/families/float_min_max.h>
#include <armigo/shader/families/word_and_immediate.h>
#include <armigo/shader/families/predicate_logic.h>
#include <armigo/shader/families/ffma_register.h>
#include <armigo/shader/families/float_immediate32.h>
#include <armigo/shader/families/range_reduction.h>
#include <armigo/shader/families/select_by_predicate.h>
#include <armigo/shader/families/word_logic.h>
#include <armigo/shader/families/float_to_integer.h>
#include <armigo/shader/families/float_set_predicate.h>
#include <armigo/shader/families/integer_to_float.h>
#include <armigo/shader/families/float_round_integral.h>
#include <armigo/shader/maxwell_predicates.h>

#include <armigo/shader_input/shader_input.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace armigo::shader::maxwell {

struct DecodeLimits {
    std::size_t max_instructions = 262144;
};

enum class OpcodeFamily : std::uint8_t {
    Unknown,
    Branch,
    Mov32Immediate,
    MovRegister,
    AttributeLoad,
    AttributeStore,
    FloatSetRegister,
    FloatAddRegister,
    FloatAddConstant,
    FloatAddImmediate,
    FloatMultiplyRegister,
    FloatMultiplyConstant,
    FloatMultiplyImmediate,
    FloatFmaImmediate,
    FloatFmaConstantRegister,
    MultiFunction,
    InterpolateAttribute,
    TextureSample,
    Exit,
    MovConstant,
    FloatFmaRegisterConstant,
    WordAndImmediate,
    IntegerSetPredicateRegister,
    PredicateSetPredicate,
    FloatFmaConstantNegated,
    FloatMultiplySaturateRegister,
    FloatFmaRegister,
    FloatMultiplyImmediate32,
    FloatAddImmediate32,
    RangeReductionRegister,
    RangeReductionConstant,
    DependencyBarrier,
    SelectByPredicateRegister,
    SelectByPredicateConstant,
    SelectByPredicateImmediate,
    WordLogicRegister,
    WordLogicConstant,
    WordLogicImmediate,
    FloatToIntegerRegister,
    FloatToIntegerConstant,
    FloatMinMaxRegister,
    FloatMinMaxConstant,
    FloatMinMaxImmediate,
    FloatSetPredicateRegister,
    FloatSetPredicateConstant,
    FloatSetPredicateImmediate,
    ReconvergenceSetup,
    ReconvergenceSync,
    IntegerToFloatRegister,
    IntegerToFloatConstant,
    FloatRoundIntegralRegister,
    FloatRoundIntegralConstant,
};

enum class DecodeField : std::uint8_t {
    None,
    InstructionPredicate,
    ReservedBits,
    WriteMask,
    IndexRegister,
    VertexRegister,
    AttributeOffset,
    AttributeWidth,
    Patch,
    AddressMode,
    BooleanPredicate,
    BooleanPredicateInversion,
    Comparison,
    BooleanOperation,
    BooleanFloat,
    NegateA,
    NegateB,
    NegateC,
    AbsoluteA,
    AbsoluteB,
    ConditionCode,
    Rounding,
    FlushMode,
    Scale,
    Saturate,
    ConstantBinding,
    Function,
    MultiplierRegister,
    Indexed,
    SampleMode,
    InterpolationMode,
    InterpolationEncodingControl,
    TexturePrecision,
    TextureEncoding,
    TextureSwizzle,
    TextureDestination,
    FlowTest,
    LogicOperation,
    InvertA,
    InvertB,
    ExtendedMode,
    DestinationFormat,
    SourceFormat,
    SourceHalf,
    Signedness,
    SelectPredicate,
    ReconvergenceStack,
    ReconvergenceTarget,
    FloatFormat,
    IntegerFormat,
    IntegerSelector,
    SourceSize,
    DestinationSize,
    RoundingOperation,
};

struct DecodeError {
    enum class Code : std::uint8_t {
        ShaderParse,
        UnsupportedStage,
        InstructionLimit,
        AllocationFailure,
        UnsupportedPredicate,
        UnsupportedControlFlow,
        UnknownOpcode,
        UnsupportedModifier,
        MissingExit,
    };

    Code code = Code::ShaderParse;
    std::uint32_t pc = 0;
    OpcodeFamily family = OpcodeFamily::Unknown;
    DecodeField field = DecodeField::None;
    std::uint64_t value = 0;
    std::uint64_t expected = 0;
    std::optional<shader_input::ParseError> parse_error;

    [[nodiscard]] const char* message() const noexcept;
};


struct Mov32Immediate {
    RegisterOperand destination;
    std::uint8_t write_mask = 0;
    std::uint32_t immediate_bits = 0;
};

struct MovRegister {
    RegisterOperand destination;
    RegisterOperand source;
    std::uint8_t write_mask = 15;
};

struct AttributeAccess {
    RegisterOperand value;
    RegisterOperand index;
    RegisterOperand vertex;
    std::uint16_t byte_offset = 0;
    std::uint8_t width_code = 0;
    bool patch = false;
};

struct FloatSetRegister {
    RegisterOperand destination;
    RegisterOperand source_a;
    RegisterOperand source_b;
    Predicate boolean_predicate;
    OrderedComparison comparison = OrderedComparison::LessThan;
    bool negate_a = false;
    bool float_boolean = true;
    FlushMode flush = FlushMode::FlushToZero;
};

// The short immediate alternative is produced only by the FADD and FMUL
// immediate forms; the register and constant-buffer alternatives belong to
// the corresponding register and buffer forms.
using FloatSource =
    std::variant<RegisterOperand, ConstantBufferOperand, ShortFloatImmediate>;

struct FloatAdd {
    RegisterOperand destination;
    RegisterOperand source_a;
    FloatSource source_b;
    RoundingMode rounding = RoundingMode::NearestEven;
    FlushMode flush = FlushMode::FlushToZero;
    bool negate_a = false;
};

struct FloatMultiply {
    RegisterOperand destination;
    RegisterOperand source_a;
    FloatSource source_b;
    RoundingMode rounding = RoundingMode::NearestEven;
    FlushMode flush = FlushMode::FlushToZero;
};

using FmaSource = std::variant<ShortFloatImmediate, ConstantBufferOperand>;

struct FloatFma {
    RegisterOperand destination;
    RegisterOperand source_a;
    FmaSource source_b;
    RegisterOperand source_c;
    RoundingMode rounding = RoundingMode::NearestEven;
    FlushMode flush = FlushMode::FlushToZero;
};

struct Reciprocal {
    RegisterOperand destination;
    RegisterOperand source;
};

struct InterpolateAttribute {
    RegisterOperand destination;
    RegisterOperand index;
    RegisterOperand multiplier;
    std::uint8_t attribute_component = 0;
    // Reached-form control bits whose value is preserved without assigning
    // interpolation semantics beyond the validated direct forms.
    std::uint16_t encoding_control = 0;
    bool multiply = false;
};

struct TextureSample2D {
    RegisterOperand destination_a;
    RegisterOperand destination_b;
    RegisterOperand coordinate_x;
    RegisterOperand coordinate_y;
    std::uint32_t handle_byte_offset = 0;
    bool no_dependency = false;
};

struct Exit {
    std::uint8_t flow_test = 15;
};

// A forward-decoder marker; its token has already been resolved statically.
struct ReconvergenceSetup {
    std::uint32_t target_pc = 0;
};

using InstructionPayload =
    std::variant<Mov32Immediate, MovRegister, AttributeAccess, FloatSetRegister, FloatAdd,
                 FloatMultiply, FloatFma, Reciprocal, InterpolateAttribute,
                 TextureSample2D, Exit, MovConstant, FloatFmaRegisterConstant,
                 WordAndImmediate, IntegerSetPredicate, PredicateSet,
                 FloatFmaConstantNegated, FloatFmaRegister, FloatMultiplyImmediate32,
                 FloatAddImmediate32, RangeReduction, DependencyBarrier, SelectByPredicate,
                 WordLogic, FloatToInteger, FloatMinMax, FloatSetPredicate,
                 ReconvergenceSetup, IntegerToFloat, FloatRoundIntegral>;

struct Instruction {
    std::uint32_t pc = 0;
    std::uint64_t original_word = 0;
    OpcodeFamily family = OpcodeFamily::Unknown;
    Predicate predicate;
    InstructionPayload payload;
};

struct DecodedProgram {
    shader_input::Stage stage = shader_input::Stage::Vertex;
    std::array<std::byte, shader_input::format::sph_size> sph{};
    std::vector<Instruction> instructions;
};

using DecodeResult = std::variant<DecodedProgram, DecodeError>;

// Parses the supplied shader containers, copies all returned metadata, and
// decodes only entry-reachable straight-line instructions through the first
// unconditional PT EXIT. No returned object borrows either input span.
[[nodiscard]] DecodeResult decode(shader_input::ByteSpan control,
                                  shader_input::ByteSpan code,
                                  DecodeLimits limits = {});

}  // namespace armigo::shader::maxwell
