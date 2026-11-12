// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once
#include "diagnostic.h"
#include "opcode.h"

#include <span>
#include <variant>
#include <vector>

namespace armigo::shader::pipeline {
struct LiteralBits {
    std::uint64_t bits{};
    ValueType type{ValueType::word32};
    constexpr auto operator<=>(const LiteralBits&) const = default;
};
struct ConstantWord {
    std::uint8_t slot{};
    std::uint32_t byte_offset{};
    constexpr auto operator<=>(const ConstantWord&) const = default;
};
struct Undefined { ValueType type{ValueType::word32}; };
// Function arguments are ValueIds listed in Function::parameters. Labels are BlockIds.
using Operand = std::variant<ValueId, Register, LiteralBits, ConstantWord, Undefined>;

struct ConversionInfo {
    ScalarType source{ScalarType::u32}, destination{ScalarType::u32};
};
struct ComparisonInfo {
    Compare relation{Compare::equal};
    bool unordered{}; // True also when either floating input is NaN.
};
enum class AtomicKind : std::uint8_t { add, minimum, maximum, increment, decrement, bit_and, bit_or, bit_xor, exchange, compare_exchange };
struct InputIndex {
    std::uint16_t value{};
    constexpr auto operator<=>(const InputIndex&) const = default;
};
// Runtime descriptor words belong to the same storage binding. The host binds
// its complete byte extent, padded to words, at the descriptor's base address.
struct StorageAddress {
    TextureKey descriptor;
    InputIndex base_low, base_high, byte_size;
};
// One more descriptor a global access may name at run time when its address
// has several descriptor origins for multi-target lowering.
struct StorageTarget {
    ResourceId resource;
    StorageAddress address;
};
// Stage-buffer instructions address header-sized task/mesh regions. Dynamic
// accesses outside their declared region are undefined by this contract.
enum class StageBufferPolicy : std::uint8_t { header_direct_v1 = 1 };
struct StageBufferAccess {
    StageBufferPolicy policy{StageBufferPolicy::header_direct_v1};
    std::uint32_t region_base{}, region_bytes{}, allocation_bytes{};
    [[nodiscard]] constexpr bool valid_for(std::uint32_t allocation) const noexcept {
        return policy == StageBufferPolicy::header_direct_v1 && allocation_bytes == allocation &&
            region_bytes && !(region_base & 3u) && !(region_bytes & 3u) &&
            region_base <= allocation && region_bytes <= allocation - region_base;
    }
};
// Native non-RMW LDG orders 5/7 use ordinary aligned scalar/vector loads.
// The decoded scope remains metadata; stores and atomic operations are separate.
enum class NativeLoadPolicy : std::uint8_t { aligned_v1 = 1 };
// Native word atomics address the actual workgroup allocation directly.
// Defined accesses stay within that allocation; no zero/drop substitute exists.
enum class NativeSharedAtomicPolicy : std::uint8_t { word_address_v1 = 1 };
struct MemoryInfo {
    AddressSpace space{AddressSpace::constant_buffer};
    Access access{Access::read};
    std::optional<std::uint8_t> constant_buffer_slot; // Only for constant_buffer.
    // Local/shared memory have no resource; storage always uses a resolved ResourceId.
    std::uint32_t byte_offset{};
    std::uint8_t element_bytes{4};
    std::uint8_t elements{1};
    std::uint8_t alignment{1};
    bool is_volatile{};
    MemoryOrder order{MemoryOrder::relaxed};
    MemoryScope scope{MemoryScope::invocation};
    std::optional<InputIndex> dynamic_byte_offset;
    std::optional<InputIndex> array_index;
    std::vector<InputIndex> data; // Store data; CAS is [expected, replacement].
    std::optional<AtomicKind> atomic;
    std::optional<ResourceId> resource;
    // Inclusive bound established by legalization for a nonliteral dynamic byte
    // offset. Emission validates static offset + this bound + access width against
    // the resolved binding extent; full_binding alone is not an offset proof.
    std::optional<std::uint32_t> dynamic_byte_offset_upper_bound;
    // Before resolution, storage offsets are guest addresses. Extended forms
    // retain Ra:Ra+1 and the signed displacement until full-width subtraction.
    std::optional<InputIndex> global_address_high;
    std::optional<StorageAddress> storage_address;
    // Further candidates after storage_address, in stable order. Emission
    // serves the access from the first descriptor whose range contains the
    // guest address; none reads zero or drops the store.
    std::vector<StorageTarget> storage_alternatives;
    // Translation proved an ordinary load's descriptor-relative byte offset.
    // This path follows the host storage binding without guest-pool lookup.
    std::optional<TextureKey> direct_storage;
    // Storage: legalization proved the descriptor-relative address word
    // aligned, so each 4-byte element is one host word.
    bool storage_word_aligned{};
    // Set only by stage-buffer instruction lowering, never ordinary LDS/STS.
    std::optional<StageBufferAccess> stage_buffer;
    std::optional<NativeLoadPolicy> native_load_policy;
    std::optional<NativeSharedAtomicPolicy> native_shared_atomic_policy;
};
struct DynamicHandle { InputIndex input; };
using HandleSource = std::variant<TextureKey, DynamicHandle>;
struct TextureInfo {
    HandleSource texture{TextureKey{}};
    std::optional<HandleSource> sampler;
    TextureMode mode{TextureMode::sample_implicit};
    // Facts must resolve these before legal IR; absence never implies 2D/float.
    std::optional<TextureDescriptor> descriptor;
    std::optional<ResourceId> resource, sampler_resource;
    std::vector<InputIndex> coordinates, gradients_x, gradients_y, offsets, data;
    std::optional<InputIndex> array_layer, array_index, level_or_bias, compare,
                              sample_index, lod_clamp;
    // A pool read (the project documentation): the handle words whose
    // texture ID (bits 0-19) and sampler ID (bits 20-31) index the pools.
    std::optional<InputIndex> pool_texture, pool_sampler;
    std::uint8_t component_mask{1};
    std::uint8_t gather_component{};
    // Scalar query selection, independent of component_mask: query_size uses
    // spatial dimensions followed by array length (cube has two size axes).
    // query_lod uses the host query pair; TMML reverses its two components.
    // query_levels/query_samples select component zero.
    std::uint8_t query_component{};
    bool coherent{};
    std::optional<AtomicKind> atomic;
    // Inclusive bound established by legalization for a nonliteral resource-array
    // index. This is distinct from a texture layer and from the binding count.
    std::optional<std::uint32_t> array_index_upper_bound;
};
struct IoInfo {
    IoSemantic semantic{IoSemantic::generic};
    std::uint16_t location{};
    std::uint8_t component_mask{1};
    IoType type;
    Interpolation interpolation;
    std::optional<InputIndex> dynamic_byte_offset, vertex_index, sample_index;
    std::vector<InputIndex> data;
    // IPA arithmetic is explicit; native interpolation must not cancel guest W.
    // For perspective interpolate, reciprocal_w is the explicit guest PositionW
    // (reciprocal clip W) factor applied after native interpolation. It is absent
    // for linear/flat interpolation; subsequent guest arithmetic stays separate.
    std::optional<InputIndex> reciprocal_w, offset_x, offset_y;
    // A load may read the stage output register file independently of opcode
    // access. IPA records its instruction-selected sampling location separately
    // so legalization can reconcile it with the interpolation fact.
    bool output_storage{};
    std::optional<InterpolationLocation> interpolation_location_override;
};
struct CallInfo { FunctionId target; };
enum class ShuffleKind : std::uint8_t { index, up, down, butterfly };
// A subgroup reduction names its kind: add, minimum, maximum, bit_or or
// bit_xor.
struct SubgroupInfo {
    ShuffleKind shuffle{ShuffleKind::index};
    std::uint8_t width{32};
    std::optional<AtomicKind> reduction;
};
using OperationInfo = std::variant<std::monostate, ConversionInfo, ComparisonInfo,
                                   MemoryInfo, TextureInfo, IoInfo, CallInfo, SubgroupInfo>;
static_assert(std::variant_size_v<OperationInfo> == static_cast<std::size_t>(InfoKind::subgroup) + 1);

struct Operation {
    SourceLocation source;
    Opcode opcode{Opcode::copy};
    Effects effects{Effects::none};
    NumericMode numeric;
    std::vector<Operand> inputs, outputs;
    OperationInfo info;
};
inline bool valid_native_shared_atomic(const Operation& op, const MemoryInfo& m, Stage stage) {
    return m.native_shared_atomic_policy == NativeSharedAtomicPolicy::word_address_v1 &&
        stage == Stage::compute && op.opcode == Opcode::atomic &&
        op.numeric == NumericMode{Domain::unsigned_integer, ScalarType::u32} &&
        m.space == AddressSpace::shared && m.access == Access::read_write &&
        m.atomic == AtomicKind::add && m.element_bytes == 4 && m.elements == 1 && m.alignment == 4 &&
        m.data.size() == 1 && !m.is_volatile && m.order == MemoryOrder::relaxed &&
        m.scope == MemoryScope::workgroup && !m.resource && !m.constant_buffer_slot && !m.array_index &&
        !m.global_address_high && !m.storage_address && m.storage_alternatives.empty() &&
        !m.direct_storage && !m.storage_word_aligned && !m.stage_buffer && !m.native_load_policy &&
        !m.dynamic_byte_offset_upper_bound && (!m.dynamic_byte_offset || m.byte_offset == 0);
}
struct PhiInput { BlockId predecessor; ValueId value; };
struct Phi {
    SourceLocation source;
    ValueId output;
    std::vector<PhiInput> inputs;
};
struct Jump { BlockId target; };
struct Branch { Operand condition; BlockId when_true, when_false; };
struct SwitchCase { std::uint32_t bits{}; BlockId target; };
struct Switch { Operand selector; std::vector<SwitchCase> cases; BlockId fallback; };
struct Return { std::vector<Operand> values; };
struct Unreachable {};
using Terminator = std::variant<Unreachable, Jump, Branch, Switch, Return>;
struct BasicBlock {
    FunctionId function;
    SourceLocation source;
    std::vector<BlockId> predecessors; // Distinct; exactly the reverse of terminator edges.
    std::vector<Phi> phis;
    std::vector<OperationId> operations;
    Terminator terminator;
};
struct Function {
    Pc entry_pc{};
    BlockId entry;
    std::vector<BlockId> blocks;
    std::vector<ValueId> parameters;
    std::vector<ValueType> results;
};
enum class ValueKind : std::uint8_t { temporary, mutable_local };
struct Value { ValueType type{ValueType::word32}; ValueKind kind{ValueKind::temporary}; };
struct Program {
    Stage stage{Stage::vertex};
    FunctionId entry;
    Limits limits;
    std::vector<Function> functions;
    std::vector<BasicBlock> blocks;
    std::vector<Operation> operations;
    std::vector<Value> values;
    std::vector<Dependency> dependencies;
    // API-mandated fragment test ordering is observable for shader side effects.
    bool api_mandated_early_fragment_tests{};
};
struct RegisterProgram { Program ir; };
struct SsaProgram { Program ir; };
enum class IrForm : std::uint8_t { registers, ssa, legal };
// Implemented contract checking, not a phase stub. No mutation or repair.
[[nodiscard]] Status validate(const Program&, IrForm, Phase);
[[nodiscard]] std::vector<BlockId> successors(const Terminator&);
// Includes conditional effects (implicit derivatives, atomic reads/writes).
[[nodiscard]] Effects required_effects(const Operation&);

static_assert(!std::is_same_v<RegisterProgram, SsaProgram>);
static_assert(std::is_move_constructible_v<Program>);
static_assert(std::is_copy_constructible_v<Program>);
} // namespace armigo::shader::pipeline
