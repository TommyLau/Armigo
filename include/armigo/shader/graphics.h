// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/shader/maxwell.h>

#include <array>
#include <functional>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace armigo::shader::graphics {

struct Limits { std::size_t max_nodes = 1048576; };

struct Error {
    enum class Code {
        UnsupportedStage, HeaderMismatch, UnsupportedInterface, InvalidInstruction,
        UnsupportedModifier, UninitializedRegister, UndeclaredAttribute,
        MissingOutput, MissingExit, Limit, Allocation, MissingAttribute,
        DuplicateAttribute, ConstantBounds, MissingSampler, SamplingFailed,
        FloatingEnvironment,
        // Control-flow graph validation and phase-1 execution guards.
        GraphFormat, GraphShape, GraphTypes, GraphDataFlow, GraphExecution,
    };
    Code code;
    std::uint32_t pc = 0;
    std::uint32_t detail = 0;
    [[nodiscard]] const char* message() const noexcept;
};

// References are strongly separated: a sample tuple cannot be used as a scalar.
// Scalar values carry raw binary32/word bits, never host float storage.
struct ScalarId { std::uint32_t index; };
struct SampleId { std::uint32_t index; };
struct PredicateId { std::uint32_t index; };
struct BlockId { std::uint32_t index; };
enum class ValueType { Raw32, Sample4, Predicate };
enum class Arithmetic { Add, MultiplyNearest, MultiplyZero, FusedMultiplyAdd };
enum class Interpolation { Constant = 1, Perspective = 2, ScreenLinear = 3 };

struct Literal { std::uint32_t bits; };
struct AttributeLoad { std::uint16_t component; };
struct ConstantLoad { std::uint8_t binding; std::uint32_t byte_offset; };
struct FloatCompare {
    ScalarId lhs, rhs;
    maxwell::OrderedComparison comparison;
    bool negate_lhs;
};
struct FloatArithmetic {
    Arithmetic operation;
    ScalarId lhs, rhs;
    std::optional<ScalarId> addend;
    bool negate_lhs;
};
struct Reciprocal { ScalarId operand; };
// Unit-interval clamp of the already-computed source word through Armigo's
// bounded saturation policy (armigo::shader::saturate_float32). Raw bits in,
// Raw32 bits out; no host floating state is consulted.
struct Saturate { ScalarId operand; };
struct Interpolate {
    ScalarId attribute;
    std::optional<ScalarId> position_w;
    Interpolation mode;
    std::uint16_t encoding_control;
    std::optional<ScalarId> multiplier = {};
};
struct TextureSample {
    ScalarId x, y;
    std::uint32_t handle_byte_offset;
    bool no_dependency;
};
struct SampleComponent { SampleId sample; std::uint8_t component; };
struct WordAnd { ScalarId operand; std::uint32_t mask; };
// Control-flow operations. Predicate results are distinct from scalar words;
// phi inputs are labelled with the predecessor block that supplies them.
struct PredicateLiteral { bool value; };
struct WordNotEqual { ScalarId lhs, rhs; };
struct PredicateNot { PredicateId operand; };
struct ScalarPhiArm { BlockId predecessor; ScalarId value; };
struct PredicatePhiArm { BlockId predecessor; PredicateId value; };
struct ScalarPhi { std::vector<ScalarPhiArm> arms; };
struct PredicatePhi { std::vector<PredicatePhiArm> arms; };
// Registered pure operations. A kind is
// a bit-exact function of up to three node operands and a kind-specific
// immediate. Its operand and result types, CPU reference, and MSL text are
// one immutable registry entry; kinds are append-only.
enum class PureKind : std::uint16_t {
    FloatNegate,
    FloatAbsolute,
    Select,
    PredicateAnd,
    PredicateOr,
    PredicateXor,
    PredicateSelect,
    WordLogic,
    WordIsZero,
    FloatToInteger,
    FloatMinimum,
    FloatMaximum,
    FloatComparePredicate,
    IntegerToFloat,
    FloatRoundIntegral,
};
struct Pure {
    PureKind kind;
    // Node indices in signature order; slots beyond the kind's arity are 0.
    std::array<std::uint32_t, 3> operands{};
    std::uint32_t immediate = 0;
};
// Result type of a registered kind. An unregistered kind reports Raw32 and
// is rejected by every validator and backend.
[[nodiscard]] ValueType pure_result_type(PureKind kind) noexcept;
using Operation = std::variant<Literal, AttributeLoad, ConstantLoad, FloatCompare,
    FloatArithmetic, Reciprocal, Interpolate, TextureSample, SampleComponent, WordAnd,
    PredicateLiteral, WordNotEqual, PredicateNot, ScalarPhi, PredicatePhi, Saturate, Pure>;
struct Node {
    std::uint32_t pc;
    Operation operation;
    [[nodiscard]] ValueType type() const noexcept {
        if (std::holds_alternative<TextureSample>(operation)) return ValueType::Sample4;
        if (const auto* pure = std::get_if<Pure>(&operation)) {
            return pure_result_type(pure->kind);
        }
        if (std::holds_alternative<PredicateLiteral>(operation) ||
            std::holds_alternative<WordNotEqual>(operation) ||
            std::holds_alternative<PredicateNot>(operation) ||
            std::holds_alternative<PredicatePhi>(operation)) {
            return ValueType::Predicate;
        }
        return ValueType::Raw32;
    }
};
enum class OutputKind { Position, Generic, Color };
struct Export {
    OutputKind kind;
    // Position: 0..3; generic: vector*4+lane; color: target*4+lane.
    std::uint16_t component;
    ScalarId value;
};

// Block terminators. Graph targets are validated strictly forward.
struct Jump { BlockId target; };
struct BranchIf { PredicateId condition; BlockId true_target, false_target; };
struct Return { };
using Terminator = std::variant<Jump, BranchIf, Return>;
// Contiguous half-open [begin, end) node range owned by one basic block.
struct BasicBlock {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    Terminator terminator = Return{};
};

struct Builder;
class Program;
struct GraphSpec;
using GraphResult = std::variant<Program, Error>;
[[nodiscard]] GraphResult make_graph(const GraphSpec& spec, Limits limits);

struct Builder;
class Program {
public:
    Program(const Program&) = default;
    Program(Program&&) noexcept = default;
    Program& operator=(const Program&) = default;
    Program& operator=(Program&&) noexcept = default;
    [[nodiscard]] shader_input::Stage stage() const noexcept { return stage_; }
    [[nodiscard]] std::span<const Node> nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::span<const Export> exports() const noexcept { return exports_; }
    // Immutable control-flow metadata. Legacy straight-line programs carry no
    // blocks; graph programs always enter at block 0.
    [[nodiscard]] std::span<const BasicBlock> blocks() const noexcept { return blocks_; }
    [[nodiscard]] std::optional<BlockId> entry() const noexcept {
        return blocks_.empty() ? std::nullopt : std::optional<BlockId>{BlockId{0}};
    }
    [[nodiscard]] const std::array<std::byte, 80>& sph() const noexcept { return sph_; }
private:
    Program() = default;
    shader_input::Stage stage_ = shader_input::Stage::Vertex;
    std::array<std::byte, 80> sph_{};
    std::vector<Node> nodes_;
    std::vector<Export> exports_;
    std::vector<BasicBlock> blocks_;
    friend struct Builder;
    friend GraphResult make_graph(const GraphSpec&, Limits);
};

using LowerResult = std::variant<Program, Error>;
// Mutable decoded payloads are validated independently of decoder provenance.
// original_word is diagnostic only: the typed payload is the lowering input.
[[nodiscard]] LowerResult lower(const maxwell::DecodedProgram&, Limits = {});

struct AttributeInput {
    // Maxwell component numbering: position 28..31, generic 32..159.
    std::uint16_t component;
    std::uint32_t bits;
};
struct TextureRequest {
    std::uint32_t handle_byte_offset;
    std::uint32_t x_bits, y_bits;
    bool no_dependency;
};
using Sampler = std::function<bool(const TextureRequest&, std::array<std::uint32_t, 4>&)>;
struct Inputs {
    // Fragment generic values are already interpolated using their SPH mode.
    // Perspective IPA additionally multiplies by the explicit PositionW input.
    std::span<const AttributeInput> attributes;
    std::array<std::span<const std::uint32_t>, 18> constant_buffers{};
    Sampler sample_2d_implicit_lod;
};
struct Output { OutputKind kind; std::uint16_t component; std::uint32_t bits; };
struct Evaluation { std::vector<Output> outputs; };
using EvaluateResult = std::variant<Evaluation, Error>;
// One invocation, no actual texture derivatives/filtering or GPU execution.
// Saves/restores the host floating environment. The callback must not change
// floating control modes; exceptions and false returns become SamplingFailed.
[[nodiscard]] EvaluateResult evaluate(const Program&, const Inputs&);

}  // namespace armigo::shader::graphics
