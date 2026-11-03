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

#include <compare>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace armigo::shader::pipeline {

inline constexpr std::uint32_t contract_version = 1;
inline constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();
// IDs are indices within one owning program, never addresses or execution order.
template <class Tag> struct Id {
    std::uint32_t value{invalid_index};
    [[nodiscard]] constexpr bool valid() const { return value != invalid_index; }
    constexpr auto operator<=>(const Id&) const = default;
};
using ValueId = Id<struct ValueTag>;
using OperationId = Id<struct OperationTag>;
using BlockId = Id<struct BlockTag>;
using FunctionId = Id<struct FunctionTag>;
using TokenId = Id<struct TokenTag>;
using StackStateId = Id<struct StackStateTag>;
using RegionId = Id<struct RegionTag>;
using ResourceId = Id<struct ResourceTag>;
using Pc = std::uint32_t; // Byte offset from the beginning of StageImage::bytes.

// Stage values are append-only for serialized reflection consumers.
enum class Stage : std::uint8_t { vertex, fragment, compute };
enum class ValueType : std::uint8_t { word32, word64, predicate };
enum class ScalarType : std::uint8_t { u8, s8, u16, s16, u32, s32, u64, s64, f16, f32, f64, boolean };
enum class RegisterFile : std::uint8_t { gpr, predicate, condition_code };
enum class ConditionCode : std::uint16_t { zero, sign, carry, overflow };
[[nodiscard]] constexpr unsigned scalar_bits(ScalarType type) {
    switch (type) {
    case ScalarType::u8: case ScalarType::s8: return 8;
    case ScalarType::u16: case ScalarType::s16: case ScalarType::f16: return 16;
    case ScalarType::u64: case ScalarType::s64: case ScalarType::f64: return 64;
    case ScalarType::boolean: return 1;
    default: return 32;
    }
}
[[nodiscard]] constexpr ValueType storage_type(ScalarType type) {
    switch (type) {
    case ScalarType::u64: case ScalarType::s64: case ScalarType::f64: return ValueType::word64;
    case ScalarType::boolean: return ValueType::predicate;
    default: return ValueType::word32;
    }
}
struct Register {
    RegisterFile file{RegisterFile::gpr};
    std::uint16_t index{};
    constexpr auto operator<=>(const Register&) const = default;
};
// R255 and P7 are read-only constants. Writes are discarded only at emission.
inline constexpr Register zero_register{RegisterFile::gpr, 255};
inline constexpr Register true_predicate{RegisterFile::predicate, 7};
enum class Domain : std::uint8_t { bits, unsigned_integer, signed_integer, floating, boolean };
enum class Rounding : std::uint8_t {
    not_applicable, nearest_even, toward_zero, toward_positive, toward_negative, nearest_away
};
struct NumericMode {
    Domain domain{Domain::bits};
    ScalarType scalar{ScalarType::u32};
    Rounding rounding{Rounding::not_applicable};
    bool flush_inputs{};
    bool flush_output{};
    bool saturate{};
    bool precise{};
    constexpr auto operator<=>(const NumericMode&) const = default;
};

[[nodiscard]] constexpr Domain scalar_domain(ScalarType scalar) {
    switch (scalar) {
    case ScalarType::s8: case ScalarType::s16: case ScalarType::s32: case ScalarType::s64: return Domain::signed_integer;
    case ScalarType::f16: case ScalarType::f32: case ScalarType::f64: return Domain::floating;
    case ScalarType::boolean: return Domain::boolean;
    default: return Domain::unsigned_integer;
    }
}
[[nodiscard]] constexpr bool valid_numeric_mode(NumericMode mode) {
    if (mode.scalar > ScalarType::boolean || mode.rounding > Rounding::nearest_away) return false;
    if (mode.domain == Domain::bits) {
        if (mode.scalar != ScalarType::u32 && mode.scalar != ScalarType::u64) return false;
    } else if (mode.domain != scalar_domain(mode.scalar)) return false;
    if (mode.domain != Domain::floating && (mode.rounding != Rounding::not_applicable ||
        mode.flush_inputs || mode.flush_output || mode.saturate)) return false;
    return true;
}
static_assert(!valid_numeric_mode({Domain::bits, ScalarType::f32}));
static_assert(valid_numeric_mode({Domain::floating, ScalarType::f32, Rounding::nearest_even}));

enum class Access : std::uint8_t { read, write, read_write };
enum class AddressSpace : std::uint8_t { constant_buffer, local, storage, shared };
enum class MemoryOrder : std::uint8_t { relaxed, acquire, release, acquire_release, sequential };
enum class MemoryScope : std::uint8_t { invocation, subgroup, workgroup, device };
enum class TextureDimension : std::uint8_t { d1, d2, d3, cube, buffer };
struct TextureShape {
    TextureDimension dimension{TextureDimension::d2};
    bool array{};
    bool multisample{};
    constexpr auto operator<=>(const TextureShape&) const = default;
};
struct TextureKey {
    std::uint8_t cbuf_slot{};
    std::uint32_t handle_byte_offset{};
    constexpr auto operator<=>(const TextureKey&) const = default;
};
enum class TextureMode : std::uint8_t {
    sample_implicit, sample_level, sample_bias, sample_gradients,
    gather, fetch, query_lod, query_size, query_levels, query_samples
};
enum class IoSemantic : std::uint8_t {
    generic, position, point_size, clip_distance, vertex_id, instance_id,
    base_vertex, base_instance, draw_id, front_facing, point_coord,
    fragment_coord, fragment_depth, color, sample_id, sample_position,
    sample_mask, helper_invocation, layer, viewport_index,
    thread_id, workgroup_id, subgroup_lane_id,
    // Append-only tail: draw-state inputs the host specializes per pipeline.
    // S2R SR_YDIRECTION reads it; the value is the bit pattern of +1.0f, or
    // -1.0f when the guest YControl negates Y.
    y_direction,
    // Fragment-stage quad position 0..3 within the 2x2 pixel quad, the
    // fragment counterpart of the SIMD lane index. FSWZADD reads it to select
    // its per-lane operation; the host declares it as a builtin attribute.
    quad_lane_id
};
enum class InterpolationMode : std::uint8_t { flat, perspective, linear };
enum class InterpolationLocation : std::uint8_t { center, centroid, sample, offset };
struct Interpolation {
    InterpolationMode mode{InterpolationMode::perspective};
    InterpolationLocation location{InterpolationLocation::center};
    constexpr auto operator<=>(const Interpolation&) const = default;
};
struct IoType {
    ScalarType scalar{ScalarType::f32};
    std::uint8_t components{1};
    constexpr auto operator<=>(const IoType&) const = default;
};

struct Limits {
    std::uint32_t code_bytes{16U * 1024U * 1024U};
    std::uint32_t instructions{1U << 20};
    std::uint32_t functions{1024};
    std::uint32_t blocks{1U << 16};
    std::uint32_t edges{1U << 18};
    std::uint32_t operations{1U << 20};
    std::uint32_t values{1U << 21};
    std::uint32_t phi_inputs{1U << 22};
    std::uint32_t operands_per_operation{256};
    std::uint32_t token_depth{64};
    std::uint32_t token_states{1U << 18};
    std::uint32_t regions{1U << 18};
    std::uint32_t nesting_depth{256};
    std::uint32_t resources{4096};
    std::uint32_t dependencies{1U << 20};
    std::uint32_t source_bytes{64U * 1024U * 1024U};
    std::uint64_t analysis_steps{1ULL << 28};
};

static_assert(sizeof(ValueId) == sizeof(std::uint32_t));
static_assert(!std::is_convertible_v<ValueId, BlockId>);
static_assert(!ValueId{}.valid());
static_assert(std::is_trivially_copyable_v<NumericMode>);
// The fragment position convention is a host adapter selection the MSL back
// end must not assume. It defaults to unspecified; emitting a fragment stage
// that needs it while it is unspecified is a diagnostic, never a silent
// default. The selected value is part of the emitted stage's identity and is
// recorded in its reflection. (The vertex depth convention is a fact:
// FactKey depth_convention.)
enum class FragmentPositionConvention : std::uint8_t {
    unspecified,
    // Metal's position.w input equals the guest's PositionW (the adapter the
    // existing emitter documents), so perspective interpolation and PositionW
    // reads may use it.
    metal_position_w_matches_guest_position_w,
};

} // namespace armigo::shader::pipeline
