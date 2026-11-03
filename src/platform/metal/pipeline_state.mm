// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Pipeline cache identity includes the complete baked state. A strong hash
// selects a bucket and full key equality selects an entry, so collisions remain
// distinct. Viewport, scissor, blend constants, culling, winding, fill mode,
// depth bias, and stencil references stay dynamic on the Metal encoder. Depth,
// stencil, and sampler objects are cached by their mapped Metal properties.

#import <Metal/Metal.h>

#include "device_context.h"
#include "fragment_output.h"
#include "native_pipeline_state.h"
#include <attachments.h>
#include "texture_format.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace armigo::platform::metal {
namespace {

using graphics::BlendFactor;
using graphics::BlendOperation;
using graphics::ColorBlendState;
using graphics::CommandResult;
using graphics::CommandStatus;
using graphics::CompareFunction;
using graphics::CoverageState;
using graphics::DepthStencilState;
using graphics::PrimitiveClass;
using graphics::RendererError;
using graphics::RendererResult;
using graphics::RendererStatus;
using graphics::StencilFaceState;
using graphics::StencilOperation;
using graphics::TextureFormat;

// Every rejected key or descriptor produces this structured result; nothing is
// dropped, defaulted or substituted on the paths below.
RendererError unsupported(std::string text) {
    return RendererError{CommandResult{CommandStatus::Unsupported, std::string_view{text}}};
}

// ---------------------------------------------------------------------------
// Key hashing. FNV-1a over a byte image that names every baked field in a
// fixed order; full key equality (PipelineKey::operator==) decides lookups,
// so two keys that land in one bucket stay distinct entries.
// ---------------------------------------------------------------------------

constexpr std::uint64_t fnv_offset_basis = 1469598103934665603ull;
constexpr std::uint64_t fnv_prime = 1099511628211ull;

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= fnv_prime;
    }
}

void hash_word(std::uint64_t& hash, std::uint64_t word) {
    hash_bytes(hash, &word, sizeof(word));
}

void hash_text(std::uint64_t& hash, const std::string& text) {
    hash_word(hash, text.size());
    hash_bytes(hash, text.data(), text.size());
}

void hash_bool(std::uint64_t& hash, bool value) {
    hash_word(hash, value ? 1u : 0u);
}

void hash_enum(std::uint64_t& hash, std::uint32_t value) {
    hash_word(hash, value);
}

template <class T>
void hash_optional(std::uint64_t& hash, const std::optional<T>& value) {
    if (value) {
        hash_bool(hash, true);
        hash_enum(hash, static_cast<std::uint32_t>(*value));
    } else {
        hash_bool(hash, false);
    }
}

void hash_specialization(std::uint64_t& hash, const SpecializationConstant& constant) {
    hash_word(hash, constant.index);
    hash_text(hash, constant.type);
    hash_word(hash, constant.value.size());
    hash_bytes(hash, constant.value.data(), constant.value.size());
}

void hash_function_key(std::uint64_t& hash, const ShaderFunctionKey& key) {
    if (key.hash) {
        hash_word(hash, key.hash);
        return;
    }
    hash_text(hash, key.package);
    hash_text(hash, key.variant);
    hash_text(hash, key.artifact);
    hash_text(hash, key.entry_point);
    hash_word(hash, key.specialization.size());
    for (const auto& constant : key.specialization) hash_specialization(hash, constant);
}

} // namespace

std::uint64_t function_key_hash(const ShaderFunctionKey& key) {
    ShaderFunctionKey fields = key;
    fields.hash = 0;
    std::uint64_t hash = fnv_offset_basis;
    hash_function_key(hash, fields);
    return hash ? hash : 1;
}

namespace {

void hash_blend(std::uint64_t& hash, const ColorBlendState& blend) {
    hash_bool(hash, blend.enabled);
    hash_enum(hash, static_cast<std::uint32_t>(blend.rgb.operation));
    hash_enum(hash, static_cast<std::uint32_t>(blend.rgb.source));
    hash_enum(hash, static_cast<std::uint32_t>(blend.rgb.destination));
    hash_enum(hash, static_cast<std::uint32_t>(blend.alpha.operation));
    hash_enum(hash, static_cast<std::uint32_t>(blend.alpha.source));
    hash_enum(hash, static_cast<std::uint32_t>(blend.alpha.destination));
    hash_word(hash, blend.write_mask);
}

std::uint64_t hash_pipeline_key(const PipelineKey& key) {
    std::uint64_t hash = fnv_offset_basis;
    hash_word(hash, key.context.device);
    hash_text(hash, key.context.compiler_chain);
    hash_text(hash, key.context.capability_profile);
    hash_function_key(hash, key.vertex);
    hash_function_key(hash, key.fragment);
    hash_word(hash, key.attributes.size());
    for (const auto& attribute : key.attributes) {
        hash_word(hash, attribute.location);
        hash_word(hash, attribute.buffer_index);
        hash_word(hash, attribute.offset);
        hash_enum(hash, static_cast<std::uint32_t>(attribute.format));
    }
    hash_word(hash, key.buffers.size());
    for (const auto& buffer : key.buffers) {
        hash_word(hash, buffer.buffer_index);
        hash_word(hash, buffer.stride);
        hash_enum(hash, static_cast<std::uint32_t>(buffer.step));
        hash_word(hash, buffer.rate);
    }
    for (const auto& color : key.colors) {
        hash_optional(hash, color.format);
        hash_blend(hash, color.blend);
    }
    hash_optional(hash, key.depth_format);
    hash_optional(hash, key.stencil_format);
    hash_word(hash, key.coverage.samples);
    hash_word(hash, key.coverage.sample_mask);
    hash_bool(hash, key.coverage.multisample);
    hash_bool(hash, key.coverage.alpha_to_coverage);
    hash_bool(hash, key.coverage.alpha_to_one);
    hash_enum(hash, static_cast<std::uint32_t>(key.primitive));
    hash_bool(hash, key.rasterization_enabled);
    hash_bool(hash, key.shader_depth_bias);
    // The emulation variant id is a 32-bit tag (the upper word is reserved).
    // Serializing the tag only keeps the image compact; full key equality
    // remains the lookup authority, so colliding tags stay distinct entries.
    hash_enum(hash, static_cast<std::uint32_t>(key.shader_emulation & 0xFFFFFFFFu));
    return hash;
}

void hash_stencil_face(std::uint64_t& hash, const StencilFaceState& face) {
    hash_enum(hash, static_cast<std::uint32_t>(face.compare));
    hash_enum(hash, static_cast<std::uint32_t>(face.fail));
    hash_enum(hash, static_cast<std::uint32_t>(face.depth_fail));
    hash_enum(hash, static_cast<std::uint32_t>(face.pass));
    hash_word(hash, face.compare_mask);
    hash_word(hash, face.write_mask);
}

std::uint64_t hash_depth_stencil_key(const DepthStencilState& key) {
    std::uint64_t hash = fnv_offset_basis;
    hash_bool(hash, key.depth_test);
    hash_bool(hash, key.depth_write);
    hash_bool(hash, key.stencil_test);
    hash_enum(hash, static_cast<std::uint32_t>(key.depth_compare));
    hash_stencil_face(hash, key.front);
    hash_stencil_face(hash, key.back);
    return hash;
}

void hash_float(std::uint64_t& hash, float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    // Floating equality treats both signed zeros as the same configuration.
    if (value != 0.0f) std::memcpy(&bits, &value, sizeof(bits));
    hash_word(hash, bits);
}

std::uint64_t hash_sampler_key(const graphics::SamplerDescriptor& key) {
    // raw_words is excluded deliberately: it records the decoded TSC input,
    // not admitted sampler state (see pipeline_state.h).
    std::uint64_t hash = fnv_offset_basis;
    for (const auto wrap : key.wrap) hash_enum(hash, static_cast<std::uint32_t>(wrap));
    hash_enum(hash, static_cast<std::uint32_t>(key.magnification));
    hash_enum(hash, static_cast<std::uint32_t>(key.minification));
    hash_enum(hash, static_cast<std::uint32_t>(key.mip_filter));
    hash_enum(hash, static_cast<std::uint32_t>(key.reduction));
    hash_bool(hash, key.depth_compare);
    hash_enum(hash, static_cast<std::uint32_t>(key.compare_function));
    hash_bool(hash, key.srgb_border_conversion);
    for (const auto component : key.srgb_border_color) hash_enum(hash, component);
    hash_float(hash, key.lod_bias);
    hash_float(hash, key.min_lod);
    hash_float(hash, key.max_lod);
    hash_float(hash, key.max_anisotropy);
    hash_float(hash, key.trilinear_optimization);
    hash_bool(hash, key.float_coordinate_normalization);
    for (const float component : key.border_color) hash_float(hash, component);
    return hash;
}

std::uint64_t hash_sampler_properties(const MetalSamplerProperties& properties) {
    std::uint64_t hash = fnv_offset_basis;
    hash_enum(hash, static_cast<std::uint32_t>(properties.address_u));
    hash_enum(hash, static_cast<std::uint32_t>(properties.address_v));
    hash_enum(hash, static_cast<std::uint32_t>(properties.address_r));
    hash_enum(hash, static_cast<std::uint32_t>(properties.mag_filter));
    hash_enum(hash, static_cast<std::uint32_t>(properties.min_filter));
    hash_enum(hash, static_cast<std::uint32_t>(properties.mip_filter));
    hash_enum(hash, static_cast<std::uint32_t>(properties.reduction_mode));
    hash_enum(hash, static_cast<std::uint32_t>(properties.compare_function));
    hash_enum(hash, static_cast<std::uint32_t>(properties.border_color));
    hash_float(hash, properties.lod_bias);
    hash_float(hash, properties.lod_min_clamp);
    hash_float(hash, properties.lod_max_clamp);
    hash_float(hash, properties.max_anisotropy);
    hash_bool(hash, properties.normalized_coordinates);
    return hash;
}

// Sampler equality over every admitted field except raw_words.
bool sampler_state_equal(const graphics::SamplerDescriptor& left,
                         const graphics::SamplerDescriptor& right) noexcept {
    return left.wrap == right.wrap && left.magnification == right.magnification &&
           left.minification == right.minification && left.mip_filter == right.mip_filter &&
           left.reduction == right.reduction && left.depth_compare == right.depth_compare &&
           left.compare_function == right.compare_function &&
           left.srgb_border_conversion == right.srgb_border_conversion &&
           left.srgb_border_color == right.srgb_border_color &&
           left.lod_bias == right.lod_bias && left.min_lod == right.min_lod &&
           left.max_lod == right.max_lod && left.max_anisotropy == right.max_anisotropy &&
           left.trilinear_optimization == right.trilinear_optimization &&
           left.float_coordinate_normalization == right.float_coordinate_normalization &&
           left.border_color == right.border_color;
}

// ---------------------------------------------------------------------------
// Enum mappings. Unknown enumerators can only come from a corrupted key, but
// every switch still stops on them instead of choosing a host default.
// ---------------------------------------------------------------------------

std::optional<MTLCompareFunction> map_compare(CompareFunction compare) noexcept {
    switch (compare) {
        case CompareFunction::Never: return MTLCompareFunctionNever;
        case CompareFunction::Less: return MTLCompareFunctionLess;
        case CompareFunction::Equal: return MTLCompareFunctionEqual;
        case CompareFunction::LessEqual: return MTLCompareFunctionLessEqual;
        case CompareFunction::Greater: return MTLCompareFunctionGreater;
        case CompareFunction::NotEqual: return MTLCompareFunctionNotEqual;
        case CompareFunction::GreaterEqual: return MTLCompareFunctionGreaterEqual;
        case CompareFunction::Always: return MTLCompareFunctionAlways;
    }
    return std::nullopt;
}

std::optional<MTLStencilOperation> map_stencil(StencilOperation operation) noexcept {
    switch (operation) {
        case StencilOperation::Keep: return MTLStencilOperationKeep;
        case StencilOperation::Zero: return MTLStencilOperationZero;
        case StencilOperation::Replace: return MTLStencilOperationReplace;
        case StencilOperation::IncrementClamp: return MTLStencilOperationIncrementClamp;
        case StencilOperation::DecrementClamp: return MTLStencilOperationDecrementClamp;
        case StencilOperation::Invert: return MTLStencilOperationInvert;
        case StencilOperation::IncrementWrap: return MTLStencilOperationIncrementWrap;
        case StencilOperation::DecrementWrap: return MTLStencilOperationDecrementWrap;
    }
    return std::nullopt;
}

std::optional<MTLBlendOperation> map_blend_operation(BlendOperation operation) noexcept {
    switch (operation) {
        case BlendOperation::Add: return MTLBlendOperationAdd;
        case BlendOperation::Subtract: return MTLBlendOperationSubtract;
        case BlendOperation::ReverseSubtract: return MTLBlendOperationReverseSubtract;
        case BlendOperation::Min: return MTLBlendOperationMin;
        case BlendOperation::Max: return MTLBlendOperationMax;
    }
    return std::nullopt;
}

// Metal expresses every single-output factor. The Source1 family requires a
// dual-source fragment output ([[color(0), index(1)]]), which the render-state
// planner stops upstream; the cache repeats the stop rather than silently
// reading an unwritten second output.
std::optional<MTLBlendFactor> map_blend_factor(BlendFactor factor) noexcept {
    switch (factor) {
        case BlendFactor::Zero: return MTLBlendFactorZero;
        case BlendFactor::One: return MTLBlendFactorOne;
        case BlendFactor::SourceColor: return MTLBlendFactorSourceColor;
        case BlendFactor::OneMinusSourceColor: return MTLBlendFactorOneMinusSourceColor;
        case BlendFactor::SourceAlpha: return MTLBlendFactorSourceAlpha;
        case BlendFactor::OneMinusSourceAlpha: return MTLBlendFactorOneMinusSourceAlpha;
        case BlendFactor::DestinationColor: return MTLBlendFactorDestinationColor;
        case BlendFactor::OneMinusDestinationColor: return MTLBlendFactorOneMinusDestinationColor;
        case BlendFactor::DestinationAlpha: return MTLBlendFactorDestinationAlpha;
        case BlendFactor::OneMinusDestinationAlpha: return MTLBlendFactorOneMinusDestinationAlpha;
        case BlendFactor::ConstantColor: return MTLBlendFactorBlendColor;
        case BlendFactor::OneMinusConstantColor: return MTLBlendFactorOneMinusBlendColor;
        case BlendFactor::ConstantAlpha: return MTLBlendFactorBlendAlpha;
        case BlendFactor::OneMinusConstantAlpha: return MTLBlendFactorOneMinusBlendAlpha;
        case BlendFactor::SourceAlphaSaturate: return MTLBlendFactorSourceAlphaSaturated;
        case BlendFactor::Source1Color:
        case BlendFactor::OneMinusSource1Color:
        case BlendFactor::Source1Alpha:
        case BlendFactor::OneMinusSource1Alpha:
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<MTLPrimitiveTopologyClass> map_primitive(PrimitiveClass primitive) noexcept {
    switch (primitive) {
        case PrimitiveClass::Point: return MTLPrimitiveTopologyClassPoint;
        case PrimitiveClass::Line: return MTLPrimitiveTopologyClassLine;
        case PrimitiveClass::Triangle: return MTLPrimitiveTopologyClassTriangle;
    }
    return std::nullopt;
}

std::optional<MTLVertexFormat> map_vertex_format(graphics::VertexAttribFormat format) noexcept {
    using graphics::VertexAttribFormat;
    switch (format) {
        case VertexAttribFormat::R8Unorm: return MTLVertexFormatUCharNormalized;
        case VertexAttribFormat::R8Snorm: return MTLVertexFormatCharNormalized;
        case VertexAttribFormat::R8Uint: return MTLVertexFormatUChar;
        case VertexAttribFormat::R8Sint: return MTLVertexFormatChar;
        case VertexAttribFormat::R16Float: return MTLVertexFormatHalf;
        case VertexAttribFormat::R16Unorm: return MTLVertexFormatUShortNormalized;
        case VertexAttribFormat::R16Snorm: return MTLVertexFormatShortNormalized;
        case VertexAttribFormat::R16Uint: return MTLVertexFormatUShort;
        case VertexAttribFormat::R16Sint: return MTLVertexFormatShort;
        case VertexAttribFormat::R32Float: return MTLVertexFormatFloat;
        case VertexAttribFormat::R32Uint: return MTLVertexFormatUInt;
        case VertexAttribFormat::R32Sint: return MTLVertexFormatInt;
        case VertexAttribFormat::R8G8Unorm: return MTLVertexFormatUChar2Normalized;
        case VertexAttribFormat::R8G8Snorm: return MTLVertexFormatChar2Normalized;
        case VertexAttribFormat::R8G8Uint: return MTLVertexFormatUChar2;
        case VertexAttribFormat::R8G8Sint: return MTLVertexFormatChar2;
        case VertexAttribFormat::R16G16Float: return MTLVertexFormatHalf2;
        case VertexAttribFormat::R16G16Unorm: return MTLVertexFormatUShort2Normalized;
        case VertexAttribFormat::R16G16Snorm: return MTLVertexFormatShort2Normalized;
        case VertexAttribFormat::R16G16Uint: return MTLVertexFormatUShort2;
        case VertexAttribFormat::R16G16Sint: return MTLVertexFormatShort2;
        case VertexAttribFormat::R32G32Float: return MTLVertexFormatFloat2;
        case VertexAttribFormat::R32G32Uint: return MTLVertexFormatUInt2;
        case VertexAttribFormat::R32G32Sint: return MTLVertexFormatInt2;
        case VertexAttribFormat::R8G8B8Unorm: return MTLVertexFormatUChar3Normalized;
        case VertexAttribFormat::R8G8B8Snorm: return MTLVertexFormatChar3Normalized;
        case VertexAttribFormat::R8G8B8Uint: return MTLVertexFormatUChar3;
        case VertexAttribFormat::R8G8B8Sint: return MTLVertexFormatChar3;
        case VertexAttribFormat::R16G16B16Float: return MTLVertexFormatHalf3;
        case VertexAttribFormat::R16G16B16Unorm: return MTLVertexFormatUShort3Normalized;
        case VertexAttribFormat::R16G16B16Snorm: return MTLVertexFormatShort3Normalized;
        case VertexAttribFormat::R16G16B16Uint: return MTLVertexFormatUShort3;
        case VertexAttribFormat::R16G16B16Sint: return MTLVertexFormatShort3;
        case VertexAttribFormat::R32G32B32Float: return MTLVertexFormatFloat3;
        case VertexAttribFormat::R32G32B32Uint: return MTLVertexFormatUInt3;
        case VertexAttribFormat::R32G32B32Sint: return MTLVertexFormatInt3;
        case VertexAttribFormat::R8G8B8A8Unorm: return MTLVertexFormatUChar4Normalized;
        case VertexAttribFormat::R8G8B8A8Snorm: return MTLVertexFormatChar4Normalized;
        case VertexAttribFormat::R8G8B8A8Uint: return MTLVertexFormatUChar4;
        case VertexAttribFormat::R8G8B8A8Sint: return MTLVertexFormatChar4;
        case VertexAttribFormat::R16G16B16A16Float: return MTLVertexFormatHalf4;
        case VertexAttribFormat::R16G16B16A16Unorm: return MTLVertexFormatUShort4Normalized;
        case VertexAttribFormat::R16G16B16A16Snorm: return MTLVertexFormatShort4Normalized;
        case VertexAttribFormat::R16G16B16A16Uint: return MTLVertexFormatUShort4;
        case VertexAttribFormat::R16G16B16A16Sint: return MTLVertexFormatShort4;
        case VertexAttribFormat::R32G32B32A32Float: return MTLVertexFormatFloat4;
        case VertexAttribFormat::R32G32B32A32Uint: return MTLVertexFormatUInt4;
        case VertexAttribFormat::R32G32B32A32Sint: return MTLVertexFormatInt4;
        case VertexAttribFormat::A2B10G10R10Unorm: return MTLVertexFormatUInt1010102Normalized;
        case VertexAttribFormat::A2B10G10R10Snorm: return MTLVertexFormatInt1010102Normalized;
        case VertexAttribFormat::B10G11R11Float: return MTLVertexFormatFloatRG11B10;
    }
    return std::nullopt;
}

// Use the same numeric classification as package specialization.
using OutputScalarClass = graphics::TextureNumericClass;
OutputScalarClass output_scalar_class(TextureFormat format) noexcept {
    return graphics::texture_numeric_class(format);
}

// Storage class of one vertex fetch format, named after its Maxwell encoding.
graphics::VertexFormatClass vertex_format_class(graphics::VertexAttribFormat format) noexcept {
    using graphics::VertexAttribFormat;
    switch (format) {
        case VertexAttribFormat::R8Uint:
        case VertexAttribFormat::R16Uint:
        case VertexAttribFormat::R32Uint:
        case VertexAttribFormat::R8G8Uint:
        case VertexAttribFormat::R16G16Uint:
        case VertexAttribFormat::R32G32Uint:
        case VertexAttribFormat::R8G8B8Uint:
        case VertexAttribFormat::R16G16B16Uint:
        case VertexAttribFormat::R32G32B32Uint:
        case VertexAttribFormat::R8G8B8A8Uint:
        case VertexAttribFormat::R16G16B16A16Uint:
        case VertexAttribFormat::R32G32B32A32Uint:
            return graphics::VertexFormatClass::uint_class;
        case VertexAttribFormat::R8Sint:
        case VertexAttribFormat::R16Sint:
        case VertexAttribFormat::R32Sint:
        case VertexAttribFormat::R8G8Sint:
        case VertexAttribFormat::R16G16Sint:
        case VertexAttribFormat::R32G32Sint:
        case VertexAttribFormat::R8G8B8Sint:
        case VertexAttribFormat::R16G16B16Sint:
        case VertexAttribFormat::R32G32B32Sint:
        case VertexAttribFormat::R8G8B8A8Sint:
        case VertexAttribFormat::R16G16B16A16Sint:
        case VertexAttribFormat::R32G32B32A32Sint:
            return graphics::VertexFormatClass::sint_class;
        default:
            return graphics::VertexFormatClass::float_class;
    }
}

// One fetch's byte extent, used for the offset-inside-stride layout check.
std::uint32_t vertex_format_size_bytes(graphics::VertexAttribFormat format) noexcept {
    using graphics::VertexAttribFormat;
    switch (format) {
        case VertexAttribFormat::R8Unorm:
        case VertexAttribFormat::R8Snorm:
        case VertexAttribFormat::R8Uint:
        case VertexAttribFormat::R8Sint:
            return 1;
        case VertexAttribFormat::R16Float:
        case VertexAttribFormat::R16Unorm:
        case VertexAttribFormat::R16Snorm:
        case VertexAttribFormat::R16Uint:
        case VertexAttribFormat::R16Sint:
            return 2;
        case VertexAttribFormat::R32Float:
        case VertexAttribFormat::R32Uint:
        case VertexAttribFormat::R32Sint:
        case VertexAttribFormat::R8G8Unorm:
        case VertexAttribFormat::R8G8Snorm:
        case VertexAttribFormat::R8G8Uint:
        case VertexAttribFormat::R8G8Sint:
        case VertexAttribFormat::R8G8B8A8Unorm:
        case VertexAttribFormat::R8G8B8A8Snorm:
        case VertexAttribFormat::R8G8B8A8Uint:
        case VertexAttribFormat::R8G8B8A8Sint:
        case VertexAttribFormat::A2B10G10R10Unorm:
        case VertexAttribFormat::A2B10G10R10Snorm:
        case VertexAttribFormat::B10G11R11Float:
            return 4;
        case VertexAttribFormat::R8G8B8Unorm:
        case VertexAttribFormat::R8G8B8Snorm:
        case VertexAttribFormat::R8G8B8Uint:
        case VertexAttribFormat::R8G8B8Sint:
            return 3;
        case VertexAttribFormat::R16G16Float:
        case VertexAttribFormat::R16G16Unorm:
        case VertexAttribFormat::R16G16Snorm:
        case VertexAttribFormat::R16G16Uint:
        case VertexAttribFormat::R16G16Sint:
            return 4;
        case VertexAttribFormat::R16G16B16Float:
        case VertexAttribFormat::R16G16B16Unorm:
        case VertexAttribFormat::R16G16B16Snorm:
        case VertexAttribFormat::R16G16B16Uint:
        case VertexAttribFormat::R16G16B16Sint:
            return 6;
        case VertexAttribFormat::R32G32Float:
        case VertexAttribFormat::R32G32Uint:
        case VertexAttribFormat::R32G32Sint:
        case VertexAttribFormat::R16G16B16A16Float:
        case VertexAttribFormat::R16G16B16A16Unorm:
        case VertexAttribFormat::R16G16B16A16Snorm:
        case VertexAttribFormat::R16G16B16A16Uint:
        case VertexAttribFormat::R16G16B16A16Sint:
            return 8;
        case VertexAttribFormat::R32G32B32Float:
        case VertexAttribFormat::R32G32B32Uint:
        case VertexAttribFormat::R32G32B32Sint:
            return 12;
        case VertexAttribFormat::R32G32B32A32Float:
        case VertexAttribFormat::R32G32B32A32Uint:
        case VertexAttribFormat::R32G32B32A32Sint:
            return 16;
    }
    return 0;
}

}  // namespace

class PipelineCacheCharge {
public:
    PipelineCacheCharge(std::shared_ptr<std::atomic<std::uint64_t>> live_bytes,
                        std::uint64_t bytes)
        : live_bytes_(std::move(live_bytes)), bytes_(bytes) {
        live_bytes_->fetch_add(bytes_, std::memory_order_relaxed);
    }
    ~PipelineCacheCharge() {
        live_bytes_->fetch_sub(bytes_, std::memory_order_release);
    }

private:
    std::shared_ptr<std::atomic<std::uint64_t>> live_bytes_;
    std::uint64_t bytes_ = 0;
};

namespace {

class DeviceArgumentBufferSamplers final
    : public std::enable_shared_from_this<DeviceArgumentBufferSamplers> {
public:
    class ConfigurationLease final {
    public:
        ConfigurationLease(MetalSamplerProperties properties,
                           std::shared_ptr<DeviceArgumentBufferSamplers> owner)
            : properties_(properties), owner_(std::move(owner)) {
            owner_->active_configurations_.fetch_add(1, std::memory_order_relaxed);
        }
        ~ConfigurationLease() { owner_->release(); }

        const MetalSamplerProperties& properties() const noexcept { return properties_; }

    private:
        MetalSamplerProperties properties_;
        std::shared_ptr<DeviceArgumentBufferSamplers> owner_;
    };

    explicit DeviceArgumentBufferSamplers(std::uint64_t limit) : limit_(limit) {}

    std::shared_ptr<ConfigurationLease> acquire(
        const MetalSamplerProperties& properties) {
        std::lock_guard guard{mutex_};
        prune_expired();
        const auto hash = hash_sampler_properties(properties);
        if (const auto found = configurations_.find(hash);
            found != configurations_.end()) {
            for (const auto& candidate : found->second) {
                if (auto live = candidate.lock();
                    live && live->properties() == properties)
                    return live;
            }
        }
        if (active_configurations_.load(std::memory_order_acquire) >= limit_) return {};
        auto lease = std::make_shared<ConfigurationLease>(properties, shared_from_this());
        configurations_[hash].push_back(lease);
        return lease;
    }

private:
    void prune_expired() {
        for (auto bucket = configurations_.begin(); bucket != configurations_.end();) {
            std::erase_if(bucket->second,
                          [](const auto& candidate) { return candidate.expired(); });
            if (bucket->second.empty()) {
                bucket = configurations_.erase(bucket);
            } else {
                ++bucket;
            }
        }
    }

    void release() noexcept {
        active_configurations_.fetch_sub(1, std::memory_order_release);
    }

    std::uint64_t limit_ = 0;
    std::atomic<std::uint64_t> active_configurations_{0};
    std::mutex mutex_;
    std::unordered_map<std::uint64_t,
                       std::vector<std::weak_ptr<ConfigurationLease>>>
        configurations_;
};

std::shared_ptr<DeviceArgumentBufferSamplers> argument_buffer_samplers(
    id<MTLDevice> device) {
    static std::mutex registry_mutex;
    static std::unordered_map<std::uintptr_t,
                              std::weak_ptr<DeviceArgumentBufferSamplers>>
        registry;
    const auto identity = static_cast<std::uintptr_t>(device.registryID);
    std::lock_guard guard{registry_mutex};
    if (const auto found = registry.find(identity); found != registry.end()) {
        if (auto live = found->second.lock()) return live;
    }
    auto samplers = std::make_shared<DeviceArgumentBufferSamplers>(
        static_cast<std::uint64_t>(device.maxArgumentBufferSamplerCount));
    registry[identity] = samplers;
    return samplers;
}

struct CacheEntry {
    enum class Kind { Pipeline, DepthStencil, Sampler, ComputePipeline };
    Kind kind = Kind::Pipeline;
    std::uint64_t sequence = 0;
    std::uint64_t cost_bytes = 0;
    virtual ~CacheEntry() = default;
};
struct PipelineCacheEntry final : CacheEntry {
    ResidentPipeline resident;
};
struct ComputePipelineCacheEntry final : CacheEntry {
    ResidentComputePipeline resident;
};
std::uint64_t hash_compute_key(const ComputePipelineKey& key) {
    const auto& function = key.function;
    std::string image = key.context.compiler_chain + '\0' + key.context.capability_profile + '\0' +
        function.package + '\0' + function.variant + '\0' + function.artifact + '\0' +
        function.entry_point + '\0' + std::to_string(key.context.device);
    for (const auto& constant : function.specialization) {
        image += '\0' + std::to_string(constant.index) + constant.type;
        for (const auto byte : constant.value) image += static_cast<char>(byte);
    }
    return std::hash<std::string>{}(image);
}
struct NativeCacheEntry final : CacheEntry {
    graphics::RetainedResource resource;
};
struct NativeArgumentBufferSamplerLease final : MetalSamplerResource {
    std::shared_ptr<PipelineCacheCharge> charge;
    std::shared_ptr<DeviceArgumentBufferSamplers::ConfigurationLease> configuration;
    id<MTLSamplerState> state = nil;
    id<MTLSamplerState> sampler() const noexcept override { return state; }
};

// Fixed per-object estimates: Metal state objects expose no byte size, so the
// budget tracks the retained Armigo-side bytes (keys, metallib payloads) plus
// this documented constant per native object.
constexpr std::uint64_t kNativeObjectBytes = 512;

std::size_t serialized_key_bytes(const PipelineKey& key) {
    // Upper bound of the hash image; allocation count, not content length.
    return sizeof(PipelineKey) + key.context.compiler_chain.size() +
           key.context.capability_profile.size() + key.vertex.package.size() +
           key.vertex.variant.size() + key.vertex.artifact.size() +
           key.vertex.entry_point.size() + key.fragment.package.size() +
           key.fragment.variant.size() + key.fragment.artifact.size() +
           key.fragment.entry_point.size();
}

id<MTLLibrary> load_library(id<MTLDevice> device, const CompiledGraphicsStage& stage,
                            std::string& failure) {
    NSError* error = nil;
    dispatch_data_t data = dispatch_data_create(stage.metallib.data(), stage.metallib.size(),
                                                nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (!library) {
        failure = error ? std::string(error.localizedDescription.UTF8String)
                        : "unknown library load failure";
        return nil;
    }
    if (![[library functionNames]
             containsObject:[NSString stringWithUTF8String:stage.entry_point.c_str()]]) {
        failure = "compiled library lacks its entry point " + stage.entry_point;
        return nil;
    }
    return library;
}

}  // namespace

class MetalPipelineStateCache final : public PipelineStateCache {
public:
    static RendererResult<std::shared_ptr<PipelineStateCache>> create(
        std::shared_ptr<DeviceContext> device_context, std::uint64_t budget_bytes) {
        if (!device_context) return unsupported("pipeline cache requires a device context");
        if (budget_bytes == 0) return unsupported("pipeline cache requires a nonzero budget");
        id<MTLDevice> device = device_context->device();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory) {
            return unsupported("pipeline cache requires a unified-memory Metal 4 device");
        }
        NSError* error = nil;
        id<MTL4Compiler> compiler =
            [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        if (!compiler) {
            const char* detail = error ? error.localizedDescription.UTF8String
                                       : "unknown Metal 4 compiler failure";
            return unsupported(std::string("pipeline cache Metal 4 compiler creation failed: ") +
                               detail);
        }
        auto cache = std::shared_ptr<MetalPipelineStateCache>(
            new MetalPipelineStateCache(std::move(device_context), compiler, budget_bytes,
                                        argument_buffer_samplers(device)));
        return cache;
    }

    const PipelineContextKey& context() const noexcept override { return context_; }

    RendererResult<ResidentPipeline> acquire(const PipelineKey& key,
                                             const AdmittedGraphicsStages& stages) override {
        const auto hash = hash_pipeline_key(key);
        std::lock_guard guard{mutex_};
        // A cached pipeline was validated when built: an equal key with the
        // same stage objects needs no second validation.
        const auto cached = [&]() -> const ResidentPipeline* {
            if (const auto found = pipeline_buckets_.find(hash); found != pipeline_buckets_.end())
                for (const auto& entry : found->second)
                    if (entry->resident.key == key) return &entry->resident;
            return nullptr;
        }();
        if (cached && cached->stages.stages.vertex == stages.stages.vertex &&
            cached->stages.stages.fragment == stages.stages.fragment)
            return *cached;
        if (auto failure = validate(key, stages)) return unsupported(*failure);
        if (cached) return *cached;
        const auto cost = pipeline_cost(key, stages);
        auto charged = reserve_charge(cost);
        if (auto* error = std::get_if<RendererError>(&charged)) return std::move(*error);
        auto built = build_pipeline(
            key, stages, std::get<std::shared_ptr<PipelineCacheCharge>>(std::move(charged)));
        if (auto* error = std::get_if<RendererError>(&built)) return std::move(*error);

        auto entry = std::make_shared<PipelineCacheEntry>();
        entry->kind = CacheEntry::Kind::Pipeline;
        entry->sequence = next_sequence_++;
        entry->cost_bytes = cost;
        entry->resident = ResidentPipeline{key, std::get<graphics::RetainedResource>(built), stages};
        pipeline_buckets_[hash].push_back(entry);
        pipeline_order_.push_back(entry);
        resident_bytes_ += entry->cost_bytes;
        return entry->resident;
    }

    // One compute pipeline per compiled compute function is created
    // on first dispatch and reused for every later launch of it.
    RendererResult<ResidentComputePipeline> acquire_compute(
        const ComputePipelineKey& key, const AdmittedComputeStage& stage) override {
        if (key.context != context_) return unsupported("compute pipeline context contradicts the device identity");
        if (!stage.stage || stage.stage->stage != shader_input::Stage::Compute ||
            stage.stage->metallib.empty() || key.function != stage.key ||
            !key.function.specialization.empty())
            return unsupported("compute pipeline requires its admitted compute stage");
        const auto hash = hash_compute_key(key);
        std::lock_guard guard{mutex_};
        if (const auto found = compute_buckets_.find(hash); found != compute_buckets_.end()) {
            for (const auto& entry : found->second)
                if (entry->resident.key == key) return entry->resident;
        }
        auto cost = kNativeObjectBytes + std::uint64_t(stage.stage->metallib.size());
        auto charged = reserve_charge(cost);
        if (auto* error = std::get_if<RendererError>(&charged)) return std::move(*error);
        id<MTLDevice> device = device_context_->device();
        auto lease = std::make_shared<NativeComputePipelineLease>();
        lease->charge = std::get<std::shared_ptr<PipelineCacheCharge>>(std::move(charged));
        std::string library_failure;
        lease->library = load_library(device, *stage.stage, library_failure);
        if (!lease->library) return unsupported("compute pipeline library load failed: " + library_failure);
        auto* function = [MTL4LibraryFunctionDescriptor new];
        function.library = lease->library;
        function.name = [NSString stringWithUTF8String:key.function.entry_point.c_str()];
        auto* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        NSError* error = nil;
        lease->state = [compiler_ newComputePipelineStateWithDescriptor:descriptor
                                                    compilerTaskOptions:nil error:&error];
        if (!lease->state) {
            const char* detail = error ? error.localizedDescription.UTF8String
                                       : "unknown Metal 4 compute pipeline failure";
            return unsupported(std::string("compute pipeline creation failed: ") + detail);
        }
        auto entry = std::make_shared<ComputePipelineCacheEntry>();
        entry->kind = CacheEntry::Kind::ComputePipeline;
        entry->sequence = next_sequence_++;
        entry->cost_bytes = cost;
        entry->resident = ResidentComputePipeline{key, lease, stage};
        compute_buckets_[hash].push_back(entry);
        compute_order_.push_back(entry);
        resident_bytes_ += entry->cost_bytes;
        return entry->resident;
    }

    RendererResult<graphics::RetainedResource> depth_stencil(
        const DepthStencilState& key) override {
        if (auto failure = validate_depth_stencil(key)) return unsupported(*failure);
        const auto hash = hash_depth_stencil_key(key);
        std::lock_guard guard{mutex_};
        if (const auto found = depth_stencil_buckets_.find(hash);
            found != depth_stencil_buckets_.end()) {
            for (const auto& entry : found->second) {
                const auto* cached = static_cast<DepthStencilCacheEntry*>(entry.get());
                if (cached->state_key == key) return cached->resource;
            }
        }
        constexpr auto cost = sizeof(DepthStencilState) + kNativeObjectBytes;
        auto charged = reserve_charge(cost);
        if (auto* error = std::get_if<RendererError>(&charged)) return std::move(*error);
        auto built = build_depth_stencil(
            key, std::get<std::shared_ptr<PipelineCacheCharge>>(std::move(charged)));
        if (auto* error = std::get_if<RendererError>(&built)) return std::move(*error);
        auto entry = std::make_shared<DepthStencilCacheEntry>();
        entry->kind = CacheEntry::Kind::DepthStencil;
        entry->sequence = next_sequence_++;
        entry->cost_bytes = cost;
        entry->state_key = key;
        entry->resource = std::get<graphics::RetainedResource>(built);
        depth_stencil_buckets_[hash].push_back(entry);
        depth_stencil_order_.push_back(entry);
        resident_bytes_ += entry->cost_bytes;
        return entry->resource;
    }

    RendererResult<graphics::RetainedResource> sampler(
        const graphics::SamplerDescriptor& descriptor) override {
        if (auto failure = validate_sampler(descriptor)) return unsupported(*failure);
        const auto hash = hash_sampler_key(descriptor);
        std::lock_guard guard{mutex_};
        if (const auto found = sampler_buckets_.find(hash);
            found != sampler_buckets_.end()) {
            for (const auto& entry : found->second) {
                const auto* cached = static_cast<SamplerCacheEntry*>(entry.get());
                if (sampler_state_equal(cached->descriptor, descriptor)) return cached->resource;
            }
        }
        auto mapped = metal_sampler_properties(descriptor);
        if (auto* error = std::get_if<graphics::TextureError>(&mapped)) {
            return unsupported(error->message());
        }
        constexpr auto cost = sizeof(graphics::SamplerDescriptor) + kNativeObjectBytes;
        auto charged = reserve_charge(cost);
        if (auto* error = std::get_if<RendererError>(&charged)) return std::move(*error);
        const auto& properties = std::get<MetalSamplerProperties>(mapped);
        auto configuration = sampler_configurations_->acquire(properties);
        while (!configuration && !sampler_order_.empty()) {
            evict(sampler_order_);
            configuration = sampler_configurations_->acquire(properties);
        }
        if (!configuration) {
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "Metal maxArgumentBufferSamplerCount is exhausted by live sampler leases"}};
        }
        auto* descriptor_object = metal_sampler_descriptor(properties);
        id<MTLSamplerState> state = [device_context_->device() newSamplerStateWithDescriptor:
                                                               descriptor_object];
        if (!state) {
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "Metal rejected an admitted argument-buffer sampler state"}};
        }
        auto entry = std::make_shared<SamplerCacheEntry>();
        entry->kind = CacheEntry::Kind::Sampler;
        entry->sequence = next_sequence_++;
        entry->cost_bytes = cost;
        entry->descriptor = descriptor;
        auto lease = std::make_shared<NativeArgumentBufferSamplerLease>();
        lease->state = state;
        lease->charge = std::get<std::shared_ptr<PipelineCacheCharge>>(std::move(charged));
        lease->configuration = std::move(configuration);
        entry->resource = lease;
        sampler_buckets_[hash].push_back(entry);
        sampler_order_.push_back(entry);
        resident_bytes_ += entry->cost_bytes;
        return entry->resource;
    }

    RendererStatus trim(std::uint64_t budget_bytes) override {
        std::lock_guard guard{mutex_};
        trim_live_to(budget_bytes);
        if (live_bytes() > budget_bytes)
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "pipeline cache live resources exceed the trim budget"}};
        return graphics::renderer_success();
    }

private:
    MetalPipelineStateCache(std::shared_ptr<DeviceContext> device_context,
                            id<MTL4Compiler> compiler, std::uint64_t budget_bytes,
                            std::shared_ptr<DeviceArgumentBufferSamplers>
                                sampler_configurations)
        : device_context_(std::move(device_context)),
          context_(device_context_->identity()),
          compiler_(compiler),
          budget_bytes_(budget_bytes),
          sampler_configurations_(std::move(sampler_configurations)),
          live_bytes_(std::make_shared<std::atomic<std::uint64_t>>(0)) {}

    struct DepthStencilCacheEntry final : CacheEntry {
        DepthStencilState state_key;
        graphics::RetainedResource resource;
    };
    struct SamplerCacheEntry final : CacheEntry {
        graphics::SamplerDescriptor descriptor;
        graphics::RetainedResource resource;
    };

    static std::uint64_t pipeline_cost(const PipelineKey& key,
                                       const AdmittedGraphicsStages& stages) {
        std::uint64_t result = kNativeObjectBytes;
        const auto add = [&](std::size_t bytes) {
            const auto amount = static_cast<std::uint64_t>(bytes);
            result = amount > UINT64_MAX - result ? UINT64_MAX : result + amount;
        };
        add(serialized_key_bytes(key));
        add(stages.stages.vertex->metallib.size());
        add(stages.stages.fragment->metallib.size());
        return result;
    }

    std::uint64_t live_bytes() const noexcept {
        return live_bytes_->load(std::memory_order_acquire);
    }

    RendererResult<std::shared_ptr<PipelineCacheCharge>> reserve_charge(
        std::uint64_t cost) {
        if (cost > budget_bytes_)
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "pipeline cache resource exceeds the residency budget"}};
        trim_live_to(budget_bytes_ - cost);
        if (live_bytes() > budget_bytes_ - cost)
            return RendererError{CommandResult{
                CommandStatus::BudgetExceeded,
                "pipeline cache live resources prevent residency admission"}};
        return std::make_shared<PipelineCacheCharge>(live_bytes_, cost);
    }

    void trim_live_to(std::uint64_t target) {
        while (live_bytes() > target) {
            CacheEntry* oldest = nullptr;
            std::deque<std::shared_ptr<CacheEntry>>* queue = nullptr;
            const auto consider = [&](std::deque<std::shared_ptr<CacheEntry>>& candidates) {
                if (candidates.empty()) return;
                if (!oldest || candidates.front()->sequence < oldest->sequence) {
                    oldest = candidates.front().get();
                    queue = &candidates;
                }
            };
            consider(pipeline_order_);
            consider(compute_order_);
            consider(depth_stencil_order_);
            consider(sampler_order_);
            if (!queue) break;
            evict(*queue);
        }
    }

    void evict(std::deque<std::shared_ptr<CacheEntry>>& queue) {
        auto entry = queue.front();
        queue.pop_front();
        resident_bytes_ -= entry->cost_bytes;
        const auto remove_from = [&](auto& buckets, std::uint64_t hash) {
            auto found = buckets.find(hash);
            if (found == buckets.end()) return;
            auto& bucket = found->second;
            std::erase_if(bucket,
                          [&](const auto& candidate) { return candidate.get() == entry.get(); });
            if (bucket.empty()) buckets.erase(found);
        };
        switch (entry->kind) {
            case CacheEntry::Kind::Pipeline: {
                const auto* pipeline = static_cast<PipelineCacheEntry*>(entry.get());
                remove_from(pipeline_buckets_, hash_pipeline_key(pipeline->resident.key));
                break;
            }
            case CacheEntry::Kind::ComputePipeline: {
                const auto* pipeline = static_cast<ComputePipelineCacheEntry*>(entry.get());
                remove_from(compute_buckets_, hash_compute_key(pipeline->resident.key));
                break;
            }
            case CacheEntry::Kind::DepthStencil: {
                const auto* depth_stencil = static_cast<DepthStencilCacheEntry*>(entry.get());
                remove_from(depth_stencil_buckets_,
                            hash_depth_stencil_key(depth_stencil->state_key));
                break;
            }
            case CacheEntry::Kind::Sampler: {
                const auto* sampler = static_cast<SamplerCacheEntry*>(entry.get());
                remove_from(sampler_buckets_, hash_sampler_key(sampler->descriptor));
                break;
            }
        }
    }

    // -- validation ---------------------------------------------------------

    std::optional<std::string> validate(const PipelineKey& key,
                                        const AdmittedGraphicsStages& stages) const {
        if (key.context != context_) return "pipeline context contradicts the device identity";
        if (!stages.stages.vertex || !stages.stages.fragment)
            return "pipeline requires admitted vertex and fragment stages";
        // y-direction-v1: a consuming stage appends exactly one bool constant
        // (index 9) carrying the guest Y sign; the remainder must still match
        // admission exactly.
        const auto without_draw_specialization = [&](const std::vector<SpecializationConstant>& specialization,
                                                     const CompiledGraphicsStage& stage)
            -> std::optional<std::vector<SpecializationConstant>> {
            auto copy = specialization;
            if (stage.consumes_y_direction) {
                const auto found = std::find_if(copy.begin(), copy.end(),
                    [](const SpecializationConstant& constant) { return constant.index == 9; });
                if (found == copy.end() || found->type != "bool" || found->value.size() != 1 ||
                    found->value.front() > std::byte{1}) return std::nullopt;
                copy.erase(found);
            }
            if (stage.reads_point_coord) {
                const auto found = std::find_if(copy.begin(), copy.end(),
                    [](const SpecializationConstant& constant) { return constant.index == 11; });
                if (found == copy.end() || found->type != "bool" || found->value.size() != 1 ||
                    found->value.front() > std::byte{1}) return std::nullopt;
                copy.erase(found);
            }
            return copy;
        };
        const auto vertex_specialization = without_draw_specialization(key.vertex.specialization,
            *stages.stages.vertex);
        const auto fragment_specialization = without_draw_specialization(key.fragment.specialization,
            *stages.stages.fragment);
        const auto admitted_function = [&](const ShaderFunctionKey& key_stage,
                                           const ShaderFunctionKey& admitted_stage,
                                           const std::optional<std::vector<SpecializationConstant>>& stripped) {
            return key_stage.package == admitted_stage.package &&
                   key_stage.variant == admitted_stage.variant &&
                   key_stage.artifact == admitted_stage.artifact &&
                   key_stage.entry_point == admitted_stage.entry_point &&
                   stripped.has_value() && *stripped == admitted_stage.specialization;
        };
        if (!admitted_function(key.vertex, stages.vertex_key, vertex_specialization) ||
            !admitted_function(key.fragment, stages.fragment_key, fragment_specialization))
            return "pipeline key contradicts the admitted stage identities";
        if (stages.stages.vertex->stage != shader_input::Stage::Vertex ||
            stages.stages.fragment->stage != shader_input::Stage::Fragment)
            return "admitted stages contradict their pipeline roles";
        if (stages.stages.vertex->entry_point != key.vertex.entry_point ||
            stages.stages.fragment->entry_point != key.fragment.entry_point)
            return "pipeline key entry points contradict the admitted stages";
        // Fragment depth and sample-mask exports ride on the stage identity:
        // the admitted fragment key's variant digest carries both flags, so
        // exporting and non-exporting variants never share a pipeline. The
        // MSL depth output is [[depth(any)]] without [[early_fragment_tests]],
        // so Metal compares and writes the exported value after the fragment
        // stage; with no depth attachment Metal discards the export, and the
        // key's depth format already mirrors the render pass attachments.
        // Metal ANDs a sample-mask export into the rasterizer coverage, which
        // is well defined for single-sample pipelines too. Shader depth-bias
        // activation below stays exclusive with a guest depth export because
        // admission refuses a stage that claims both.
        if (stages.stages.vertex->fragment_output_types_from_render_targets ||
            *vertex_specialization != fragment_output_specialization(*stages.stages.vertex) ||
            *fragment_specialization != fragment_output_specialization(*stages.stages.fragment))
            return "specialization constants contradict the fragment output contract";

        if (key.shader_depth_bias &&
            (!key.depth_format || !stages.stages.fragment->emulates_depth_bias))
            return "shader depth bias requires a depth attachment and fragment capability";

        // Typed fragment outputs versus bound attachments. The legacy v1
        // schema synthesizes a float32x4 export at target 0.
        const auto& outputs = stages.stages.fragment->color_outputs;
        if (outputs.empty() && !stages.stages.fragment->fragment_output_types_from_render_targets) {
            const bool any_color = std::any_of(key.colors.begin(), key.colors.end(),
                [](const PipelineColorKey& color) { return color.format.has_value(); });
            if (any_color && key.colors[0].format) {
                if (output_scalar_class(*key.colors[0].format) != OutputScalarClass::Float)
                    return "legacy stage float32x4 export contradicts an integer target";
            }
        } else {
            for (const auto& output : outputs) {
                if (output.location >= key.colors.size())
                    return "fragment output location exceeds the target array";
                const auto& target = key.colors[output.location];
                // Maxwell discards exports to unbound logical targets.
                if (!target.format) continue;
                const auto expected = output_scalar_class(*target.format);
                const auto actual = static_cast<OutputScalarClass>(output.scalar);
                if (expected != actual)
                    return "fragment output scalar contradicts the attachment format";
            }
        }

        // Attachment formats: renderable, correct aspect, holes are holes.
        for (std::size_t i = 0; i < key.colors.size(); ++i) {
            if (!key.colors[i].format) continue;
            const auto& format = *key.colors[i].format;
            const auto info = graphics::texture_format_info(format);
            if (info.depth || info.stencil)
                return "color attachment names a depth or stencil format";
            if (graphics::is_block_compressed(info))
                return "color attachment names a block-compressed format";
            const auto mapped = metal_render_format(format, graphics::TextureAspect::Color);
            if (const auto* failure = std::get_if<CommandResult>(&mapped))
                return std::string("color attachment has no Metal render format: ") +
                       std::string(failure->detail);
        }
        if (key.stencil_format && !key.depth_format)
            return "stencil attachment is bound without a depth attachment";
        if (key.depth_format) {
            const auto mapped = metal_render_format(*key.depth_format,
                                                    graphics::TextureAspect::Depth);
            if (const auto* failure = std::get_if<CommandResult>(&mapped))
                return std::string("depth attachment has no Metal render format: ") +
                       std::string(failure->detail);
        }
        if (key.stencil_format) {
            const auto mapped = metal_render_format(*key.stencil_format,
                                                    graphics::TextureAspect::Stencil);
            if (const auto* failure = std::get_if<CommandResult>(&mapped))
                return std::string("stencil attachment has no Metal render format: ") +
                       std::string(failure->detail);
        }

        // Coverage: the sample count must be a device capability, and the
        // multisample flag must agree with it. Alpha to coverage is a
        // multisample-only feature.
        if (key.coverage.samples == 0)
            return "coverage sample count is zero";
        if (![device_context_->device() supportsTextureSampleCount:key.coverage.samples])
            return "coverage sample count is unsupported on this device";
        if (key.coverage.multisample != (key.coverage.samples > 1))
            return "coverage multisample flag contradicts the sample count";
        if (key.coverage.alpha_to_coverage && !key.coverage.multisample)
            return "alpha to coverage requires multisampling";
        // Alpha-to-one has no effect on integer targets on the guest GPU, but
        // Metal refuses it with any integer attachment. An all-integer pass
        // disables it at construction; a mixed pass cannot be expressed.
        if (key.coverage.alpha_to_one) {
            bool float_target = false, integer_target = false;
            for (const auto& color : key.colors) {
                if (!color.format) continue;
                (output_scalar_class(*color.format) == OutputScalarClass::Float ? float_target
                                                                                 : integer_target) = true;
            }
            if (float_target && integer_target)
                return "alpha to one with mixed float and integer attachments is unsupported";
        }

        // Blend: decodable factors and operations, and no blending on an
        // integer attachment (Metal refuses it; Maxwell cannot request it).
        for (std::size_t i = 0; i < key.colors.size(); ++i) {
            const auto& blend = key.colors[i].blend;
            if (!blend.enabled || !key.colors[i].format) continue;
            if (output_scalar_class(*key.colors[i].format) != OutputScalarClass::Float)
                return "blend is enabled on an integer attachment";
            if (!map_blend_operation(blend.rgb.operation) ||
                !map_blend_operation(blend.alpha.operation))
                return "blend operation is unsupported";
            if (!map_blend_factor(blend.rgb.source) || !map_blend_factor(blend.rgb.destination) ||
                !map_blend_factor(blend.alpha.source) ||
                !map_blend_factor(blend.alpha.destination))
                return "blend factor is unsupported (dual-source blending is not admitted)";
        }

        // Vertex layout: canonical order, complete buffer references, offsets
        // inside strides, and every shader input covered by a keyed attribute.
        std::uint32_t previous_location = 0;
        bool first_attribute = true;
        for (const auto& attribute : key.attributes) {
            if (!first_attribute && attribute.location <= previous_location)
                return "vertex attributes are not in canonical ascending location order";
            previous_location = attribute.location;
            first_attribute = false;
            if (attribute.location >= 31)
                return "vertex attribute location exceeds the Metal attribute array";
            if (attribute.buffer_index >= 31)
                return "vertex attribute buffer index exceeds the Metal layout array";
            const auto buffer = std::find_if(key.buffers.begin(), key.buffers.end(),
                [&](const VertexBufferLayoutKey& layout) {
                    return layout.buffer_index == attribute.buffer_index;
                });
            if (buffer == key.buffers.end())
                return "vertex attribute references an undeclared buffer layout";
            const auto mapped_format = map_vertex_format(attribute.format);
            if (!mapped_format)
                return "vertex attribute format is unsupported";
            // Maxwell, like Metal, fetches an attribute at vertex * stride + offset
            // even when offset + size exceeds the stride (overlapping records);
            // Metal's validation layer accepts it and the fetch is exact, so the
            // stride is not a bound here. Buffer extents are checked at binding.
        }
        std::uint32_t previous_buffer = 0;
        bool first_buffer = true;
        for (const auto& buffer : key.buffers) {
            if (!first_buffer && buffer.buffer_index <= previous_buffer)
                return "vertex buffer layouts are not in canonical ascending index order";
            previous_buffer = buffer.buffer_index;
            first_buffer = false;
            if (buffer.stride > 0xffff)
                return "vertex buffer stride exceeds the Metal stride maximum";
            if (buffer.step == graphics::VertexStep::PerInstance && buffer.rate == 0)
                return "per-instance vertex buffer rate is zero";
        }
        if (!validate_stage_inputs(key, *stages.stages.vertex))
            return "vertex layout does not cover the admitted shader inputs";
        return std::nullopt;
    }

    bool validate_stage_inputs(const PipelineKey& key, const CompiledGraphicsStage& vertex) const {
        // Reflection v2 class inputs: one keyed attribute per input whose
        // format's storage class matches the declared input class.
        for (const auto& input : vertex.vertex_class_inputs) {
            const auto attribute = std::find_if(key.attributes.begin(), key.attributes.end(),
                [&](const VertexAttributeKey& candidate) {
                    return candidate.location == input.location;
                });
            if (attribute == key.attributes.end()) return false;
            if (!map_vertex_format(attribute->format)) return false;
            if (vertex_format_class(attribute->format) != input.input_class) return false;
        }
        // Legacy v1 options: Float32x3 fetch at the declared attribute index.
        for (const auto& option : vertex.vertex_attributes) {
            if (option.format != shader::graphics_msl::VertexFormat::Float32x3) return false;
            const auto attribute = std::find_if(key.attributes.begin(), key.attributes.end(),
                [&](const VertexAttributeKey& candidate) {
                    return candidate.location == option.attribute_index;
                });
            if (attribute == key.attributes.end() ||
                attribute->format != graphics::VertexAttribFormat::R32G32B32Float)
                return false;
        }
        return true;
    }

    std::optional<std::string> validate_depth_stencil(const DepthStencilState& key) const {
        if (!map_compare(key.depth_compare))
            return "depth compare function is unsupported";
        for (const auto* face : {&key.front, &key.back}) {
            if (!map_compare(face->compare)) return "stencil compare function is unsupported";
            if (!map_stencil(face->fail) || !map_stencil(face->depth_fail) ||
                !map_stencil(face->pass))
                return "stencil operation is unsupported";
        }
        return std::nullopt;
    }

    static std::optional<std::string> validate_sampler(
        const graphics::SamplerDescriptor& descriptor) {
        for (const float value : {descriptor.lod_bias, descriptor.min_lod, descriptor.max_lod,
                                  descriptor.max_anisotropy, descriptor.trilinear_optimization})
            if (!std::isfinite(value)) return "sampler lod or anisotropy value is not finite";
        for (const float component : descriptor.border_color)
            if (!std::isfinite(component)) return "sampler border color is not finite";
        if (descriptor.max_lod < descriptor.min_lod)
            return "sampler maximum lod precedes its minimum";
        if (!(descriptor.max_anisotropy >= 1.0f) || descriptor.max_anisotropy > 16.0f ||
            descriptor.max_anisotropy != std::floor(descriptor.max_anisotropy))
            return "sampler anisotropy is outside the Metal range 1..16";
        if (descriptor.max_anisotropy > 1.0f &&
            std::popcount(static_cast<std::uint32_t>(descriptor.max_anisotropy)) != 1)
            return "sampler anisotropy is not a power of two";
        // The canonical disabled comparison state is Never; a comparison
        // request with any other record is contradictory, not ignorable.
        if (!descriptor.depth_compare && descriptor.compare_function !=
                                             graphics::SamplerCompareFunction::Never)
            return "comparison function is set while comparison sampling is disabled";
        if (descriptor.float_coordinate_normalization)
            return "float coordinate normalization polarity is unestablished";
        return std::nullopt;
    }

    // -- construction -------------------------------------------------------

    RendererResult<graphics::RetainedResource> build_pipeline(
        const PipelineKey& key, const AdmittedGraphicsStages& stages,
        std::shared_ptr<PipelineCacheCharge> charge) {
        id<MTLDevice> device = device_context_->device();
        auto lease = std::make_shared<NativePipelineLease>();
        lease->charge = std::move(charge);
        std::string library_failure;
        lease->vertex_library = load_library(device, *stages.stages.vertex, library_failure);
        lease->fragment_library = load_library(device, *stages.stages.fragment, library_failure);
        if (!lease->vertex_library || !lease->fragment_library) {
            return unsupported("pipeline library load failed: " + library_failure);
        }
        auto* vertex = [MTL4LibraryFunctionDescriptor new];
        vertex.library = lease->vertex_library;
        vertex.name = [NSString stringWithUTF8String:key.vertex.entry_point.c_str()];
        auto* fragment = [MTL4LibraryFunctionDescriptor new];
        fragment.library = lease->fragment_library;
        fragment.name = [NSString stringWithUTF8String:key.fragment.entry_point.c_str()];

        auto* descriptor = [MTL4RenderPipelineDescriptor new];
        // y-direction-v1: consuming stages read their armigo_y_negate
        // function constant (index 9) from the key's specialization entry.
        const auto y_direction_negate = [](const ShaderFunctionKey& stage_key) {
            const auto found = std::find_if(stage_key.specialization.begin(), stage_key.specialization.end(),
                [](const SpecializationConstant& constant) { return constant.index == 9; });
            return found != stage_key.specialization.end() && !found->value.empty() &&
                   found->value.front() != std::byte{0};
        };
        const auto specialized_function = [&](MTL4LibraryFunctionDescriptor* base,
                                              const CompiledGraphicsStage& stage,
                                              const ShaderFunctionKey& stage_key,
                                              MTLFunctionConstantValues* inherited)
            -> MTL4FunctionDescriptor* {
            auto* constants = inherited;
            if (stage.consumes_y_direction) {
                if (!constants) constants = [MTLFunctionConstantValues new];
                const bool negate = y_direction_negate(stage_key);
                [constants setConstantValue:&negate type:MTLDataTypeBool atIndex:9];
            }
            if (stage.reads_point_coord) {
                if (!constants) constants = [MTLFunctionConstantValues new];
                const auto found = std::find_if(stage_key.specialization.begin(), stage_key.specialization.end(),
                    [](const SpecializationConstant& constant) { return constant.index == 11; });
                bool lower_left = found != stage_key.specialization.end() && found->value.size() == 1 &&
                                  found->value.front() == std::byte{1};
                [constants setConstantValue:&lower_left type:MTLDataTypeBool atIndex:11];
            }
            if (!constants) return base;
            auto* specialized = [MTL4SpecializedFunctionDescriptor new];
            specialized.functionDescriptor = base;
            specialized.constantValues = constants;
            return specialized;
        };
        descriptor.vertexFunctionDescriptor = specialized_function(
            vertex, *stages.stages.vertex, key.vertex, nil);
        descriptor.fragmentFunctionDescriptor = specialized_function(
            fragment, *stages.stages.fragment, key.fragment,
            fragment_output_constants(*stages.stages.fragment, key.shader_depth_bias));
        // Metal refuses a point-size export with a triangle or line topology
        // class but accepts it with the class unspecified, which only layered
        // rendering requires; stages that write a point size never declare one.
        descriptor.inputPrimitiveTopology = stages.stages.vertex->writes_point_size
            ? MTLPrimitiveTopologyClassUnspecified : *map_primitive(key.primitive);
        descriptor.rasterizationEnabled = key.rasterization_enabled ? YES : NO;
        descriptor.rasterSampleCount = key.coverage.samples;
        descriptor.alphaToCoverageState = key.coverage.alpha_to_coverage
                                              ? MTL4AlphaToCoverageStateEnabled
                                              : MTL4AlphaToCoverageStateDisabled;
        const bool float_attachment = std::any_of(key.colors.begin(), key.colors.end(),
            [](const auto& color) {
                return color.format && output_scalar_class(*color.format) == OutputScalarClass::Float;
            });
        descriptor.alphaToOneState = key.coverage.alpha_to_one && float_attachment
                                         ? MTL4AlphaToOneStateEnabled
                                         : MTL4AlphaToOneStateDisabled;

        auto* vertex_descriptor = [MTLVertexDescriptor vertexDescriptor];
        for (const auto& attribute : key.attributes) {
            auto* host = vertex_descriptor.attributes[attribute.location];
            host.format = *map_vertex_format(attribute.format);
            host.bufferIndex = attribute.buffer_index;
            host.offset = attribute.offset;
        }
        for (const auto& layout : key.buffers) {
            auto* host = vertex_descriptor.layouts[layout.buffer_index];
            host.stride = layout.stride;
            switch (layout.step) {
                case graphics::VertexStep::PerVertex:
                    host.stepFunction = MTLVertexStepFunctionPerVertex;
                    host.stepRate = 1;
                    break;
                case graphics::VertexStep::PerInstance:
                    host.stepFunction = MTLVertexStepFunctionPerInstance;
                    host.stepRate = layout.rate;
                    break;
                case graphics::VertexStep::Constant:
                    // Metal fetches a constant layout once but still
                    // validates a nonzero, four-byte-aligned stride.
                    host.stepFunction = MTLVertexStepFunctionConstant;
                    host.stepRate = 0;
                    if (!host.stride) host.stride = 4;
                    break;
            }
        }
        descriptor.vertexDescriptor = vertex_descriptor;

        for (std::size_t i = 0; i < key.colors.size(); ++i) {
            auto* attachment = descriptor.colorAttachments[i];
            if (!key.colors[i].format) {
                attachment.pixelFormat = MTLPixelFormatInvalid;
                attachment.writeMask = MTLColorWriteMaskNone;
                continue;
            }
            attachment.pixelFormat =
                std::get<MTLPixelFormat>(metal_render_format(*key.colors[i].format,
                                                             graphics::TextureAspect::Color));
            const auto& blend = key.colors[i].blend;
            attachment.blendingState =
                blend.enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
            attachment.rgbBlendOperation = *map_blend_operation(blend.rgb.operation);
            attachment.sourceRGBBlendFactor = *map_blend_factor(blend.rgb.source);
            attachment.destinationRGBBlendFactor = *map_blend_factor(blend.rgb.destination);
            attachment.alphaBlendOperation = *map_blend_operation(blend.alpha.operation);
            attachment.sourceAlphaBlendFactor = *map_blend_factor(blend.alpha.source);
            attachment.destinationAlphaBlendFactor =
                *map_blend_factor(blend.alpha.destination);
            // The key carries Vulkan-order RGBA bits (R at bit 0); Metal's
            // color write mask places alpha at bit 0.
            auto mask = blend.write_mask;
            const auto& fragment_stage = *stages.stages.fragment;
            if (fragment_stage.fragment_output_types_from_render_targets ||
                !fragment_stage.color_outputs.empty()) {
                const auto output = std::find_if(fragment_stage.color_outputs.begin(),
                    fragment_stage.color_outputs.end(), [&](const auto& item) {
                        return item.location == i;
                    });
                mask &= output == fragment_stage.color_outputs.end() ? 0 : output->component_mask;
            }
            attachment.writeMask =
                static_cast<MTLColorWriteMask>(((mask & 1u) << 3) | ((mask & 2u) << 1) |
                                               ((mask & 4u) >> 1) | ((mask & 8u) >> 3));
        }
        // MTL4RenderPipelineDescriptor carries no depth/stencil attachment
        // formats; the render pass attachments supply them. The key still
        // records both formats, because admission must never silently alias
        // different attachment layouts into one pipeline.

        NSError* error = nil;
        id<MTLRenderPipelineState> state = [compiler_ newRenderPipelineStateWithDescriptor:
                                                                     descriptor
                                                       compilerTaskOptions:nil
                                                                     error:&error];
        if (!state) {
            const char* detail = error ? error.localizedDescription.UTF8String
                                       : "unknown Metal 4 pipeline failure";
            return unsupported(std::string("pipeline creation failed: ") + detail);
        }        lease->state = state;
        return lease;
    }

    RendererResult<graphics::RetainedResource> build_depth_stencil(
        const DepthStencilState& key, std::shared_ptr<PipelineCacheCharge> charge) {
        auto* descriptor = [MTLDepthStencilDescriptor new];
        descriptor.depthCompareFunction =
            key.depth_test ? *map_compare(key.depth_compare) : MTLCompareFunctionAlways;
        // Metal applies depth writes only when the compare passes, so a
        // disabled test also disables writes.
        descriptor.depthWriteEnabled = (key.depth_test && key.depth_write) ? YES : NO;
        const auto build_face = [&](const StencilFaceState& face) {
            auto* stencil = [MTLStencilDescriptor new];
            stencil.stencilCompareFunction = *map_compare(face.compare);
            stencil.stencilFailureOperation = *map_stencil(face.fail);
            stencil.depthFailureOperation = *map_stencil(face.depth_fail);
            stencil.depthStencilPassOperation = *map_stencil(face.pass);
            stencil.readMask = face.compare_mask;
            stencil.writeMask = face.write_mask;
            return stencil;
        };
        descriptor.frontFaceStencil = key.stencil_test ? build_face(key.front) : nil;
        descriptor.backFaceStencil = key.stencil_test ? build_face(key.back) : nil;
        id<MTLDepthStencilState> state =
            [device_context_->device() newDepthStencilStateWithDescriptor:descriptor];
        if (!state) return unsupported("depth/stencil state creation failed");
        auto lease = std::make_shared<NativeDepthStencilLease>();
        lease->state = state;
        lease->charge = std::move(charge);
        return lease;
    }

    std::shared_ptr<DeviceContext> device_context_;
    PipelineContextKey context_;
    id<MTL4Compiler> compiler_;
    std::uint64_t budget_bytes_ = 0;
    std::shared_ptr<DeviceArgumentBufferSamplers> sampler_configurations_;
    std::shared_ptr<std::atomic<std::uint64_t>> live_bytes_;
    std::uint64_t resident_bytes_ = 0;
    std::uint64_t next_sequence_ = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<PipelineCacheEntry>>>
        pipeline_buckets_;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<DepthStencilCacheEntry>>>
        depth_stencil_buckets_;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<SamplerCacheEntry>>>
        sampler_buckets_;
    std::unordered_map<std::uint64_t, std::vector<std::shared_ptr<ComputePipelineCacheEntry>>>
        compute_buckets_;
    std::deque<std::shared_ptr<CacheEntry>> pipeline_order_, depth_stencil_order_,
        sampler_order_, compute_order_;
};

}  // namespace armigo::platform::metal

namespace armigo::platform::metal {

graphics::RendererResult<std::shared_ptr<PipelineStateCache>> create_pipeline_state_cache(
    std::shared_ptr<DeviceContext> device_context, std::uint64_t budget_bytes) {
    return MetalPipelineStateCache::create(std::move(device_context), budget_bytes);
}

}  // namespace armigo::platform::metal
