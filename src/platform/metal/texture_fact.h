// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Reflected shader declarations determine null resource types. Compatible
// same-dimension targets may use layer-zero views in the shader's target, and
// comparison sampling may use the defined colour-to-depth storage aliases.

#pragma once

#include <armigo/platform/metal_draw.h>
#include <armigo/graphics/texture.h>
#include <armigo/graphics/texture_format.h>
#include "../../graphics/texture_residency.h"
#include "texture_format.h"
#include <optional>
#include <set>
#include <string>
#include <variant>

namespace armigo::platform::metal {

inline bool valid_custom_border_pairs(const CompiledGraphicsStage& stage) {
    if (!stage.custom_border) return true;
    if ((stage.stage != shader_input::Stage::Fragment &&
         stage.stage != shader_input::Stage::Vertex) ||
        stage.custom_border->pairs.empty() || stage.custom_border->pairs.size() > 4096)
        return false;
    std::set<std::uint32_t> textures;
    for (const auto& pair : stage.custom_border->pairs) {
        const auto find = [&](std::uint32_t id) -> const CompiledShaderResource* {
            for (const auto& resource : stage.resources)
                if (resource.resource_id == id) return &resource;
            return nullptr;
        };
        const auto* texture = find(pair.texture_resource);
        const auto* sampler = find(pair.sampler_resource);
        if (!texture || !sampler || !textures.insert(pair.texture_resource).second ||
            texture->kind != CompiledShaderResourceKind::SampledTexture ||
            sampler->kind != CompiledShaderResourceKind::Sampler ||
            texture->array_length != 1 || sampler->array_length != 1 ||
            texture->cbuf_slot != sampler->cbuf_slot ||
            texture->handle_byte_offset != sampler->handle_byte_offset ||
            !texture->texture_descriptor || texture->comparison || sampler->comparison ||
            !texture->normalized_coordinates || !sampler->normalized_coordinates)
            return false;
        const auto& d = *texture->texture_descriptor;
        if (d.dimension != CompiledShaderTextureDimension::d2 || d.array || d.multisample ||
            d.scalar != CompiledShaderTextureScalar::f32 ||
            d.component_format == TextureComponentFormat::depth ||
            !d.normalized_coordinates || d.comparison) return false;
    }
    return true;
}

inline graphics::TextureNumericClass null_texture_scalar(const DerivedTextureFact* fact) {
    if (fact && fact->scalar == TextureSampledScalar::u32) return graphics::TextureNumericClass::Uint;
    if (fact && fact->scalar == TextureSampledScalar::s32) return graphics::TextureNumericClass::Sint;
    return graphics::TextureNumericClass::Float;
}

// The texture type the shader's reflected target names. 3D has no array form,
// so that combination names no texture type at all.
inline std::optional<graphics::TextureType> texture_fact_target(
    TextureFactDimension dimension, bool array) {
    using graphics::TextureType;
    switch (dimension) {
    case TextureFactDimension::d1:
        return array ? TextureType::Texture1DArray : TextureType::Texture1D;
    case TextureFactDimension::d2:
        return array ? TextureType::Texture2DArray : TextureType::Texture2D;
    case TextureFactDimension::d3:
        return array ? std::nullopt : std::optional(TextureType::Texture3D);
    case TextureFactDimension::cube:
        return array ? std::optional(TextureType::CubemapArray)
                     : std::optional(TextureType::Cubemap);
    }
    return std::nullopt;
}

inline std::string texture_target_shape(TextureFactDimension dimension, bool array) {
    std::string label = dimension == TextureFactDimension::d1 ? "1D"
        : dimension == TextureFactDimension::d2 ? "2D"
        : dimension == TextureFactDimension::d3 ? "3D" : "cube";
    return array ? label + " array" : label;
}

// Armigo binds every texture in the shader's reflected target: an exact match
// returns the host texture, and a same-dimensions target (1D<->1D array,
// 2D<->2D array, or cube<->cube array) binds a
// layer-zero view of one layer, or six slices for cube targets
// layer-zero view. Any other target difference binds nothing.
struct TextureTargetPlan {
    bool exact = true;
    graphics::TextureType view_target = graphics::TextureType::Texture2D;
    std::uint32_t first_layer = 0;
    std::uint32_t layer_count = 1;
};

// Admits the shader's target fact against the live texture, or returns the
// target mismatch with both targets (dimension, arrayness, samples) named.
inline std::variant<TextureTargetPlan, std::string> texture_target_plan(
    const DerivedTextureFact& fact, const graphics::ResolvedTextureBinding& live) {
    using graphics::TextureType;
    TextureFactDimension dimension{};
    bool array = false;
    switch (live.texture.type) {
    case TextureType::Texture1D: dimension = TextureFactDimension::d1; break;
    case TextureType::Texture1DArray: dimension = TextureFactDimension::d1; array = true; break;
    case TextureType::Texture2D: case TextureType::Texture2DRect:
        dimension = TextureFactDimension::d2; break;
    case TextureType::Texture2DArray: dimension = TextureFactDimension::d2; array = true; break;
    case TextureType::Texture3D: dimension = TextureFactDimension::d3; break;
    case TextureType::Cubemap: dimension = TextureFactDimension::cube; break;
    case TextureType::CubemapArray: dimension = TextureFactDimension::cube; array = true; break;
    default: return "the live texture target is unsupported";
    }
    const auto mismatch = [&](const char* reason) {
        return std::string(reason) + ": the shader target (" +
               texture_target_shape(fact.dimension, fact.array) +
               (fact.multisample ? ", multisample" : "") + ") does not match the live texture target (" +
               texture_target_shape(dimension, array) + ", " +
               std::to_string(live.texture.samples) + " samples)";
    };
    if (fact.multisample != (live.texture.samples > 1) || live.texture.samples != 1)
        return mismatch("the multisample expectation does not match the live target");
    const auto expected = texture_fact_target(fact.dimension, fact.array);
    if (!expected)
        return mismatch("the reflected 3D-array target is not a bindable texture target");
    if (fact.dimension == dimension && fact.array == array) return TextureTargetPlan{};
    if (fact.dimension != dimension)
        return mismatch("the dimension expectation does not match the live target");
    TextureTargetPlan plan;
    plan.exact = false;
    plan.view_target = *expected;
    plan.layer_count = fact.dimension == TextureFactDimension::cube ? 6u : 1u;
    return plan;
}

inline bool null_texture_depth(const DerivedTextureFact* fact) {
    return fact && fact->component_format == TextureComponentFormat::depth;
}

// Diagnostic names for the reflected component contract of a fact refusal.
inline std::string component_format_name(TextureComponentFormat format) {
    switch (format) {
    case TextureComponentFormat::floating: return "float";
    case TextureComponentFormat::unorm: return "unorm";
    case TextureComponentFormat::snorm: return "snorm";
    case TextureComponentFormat::unsigned_integer: return "uint";
    case TextureComponentFormat::signed_integer: return "sint";
    case TextureComponentFormat::depth: return "depth";
    }
    return "unknown component format";
}

// Diagnostic names for the decoded table format, used by structured fact
// refusals so a stop names the guest encoding it saw.
inline std::string texture_format_name(graphics::TextureFormat format) {
    using graphics::TextureFormat;
    switch (format) {
    case TextureFormat::R8Unorm: return "R8Unorm";
    case TextureFormat::R8Snorm: return "R8Snorm";
    case TextureFormat::R8Uint: return "R8Uint";
    case TextureFormat::R8Sint: return "R8Sint";
    case TextureFormat::R16Unorm: return "R16Unorm";
    case TextureFormat::R16Snorm: return "R16Snorm";
    case TextureFormat::R16Uint: return "R16Uint";
    case TextureFormat::R16Sint: return "R16Sint";
    case TextureFormat::R16Float: return "R16Float";
    case TextureFormat::R32Uint: return "R32Uint";
    case TextureFormat::R32Sint: return "R32Sint";
    case TextureFormat::R32Float: return "R32Float";
    case TextureFormat::R8G8Unorm: return "R8G8Unorm";
    case TextureFormat::R8G8Snorm: return "R8G8Snorm";
    case TextureFormat::R8G8Uint: return "R8G8Uint";
    case TextureFormat::R8G8Sint: return "R8G8Sint";
    case TextureFormat::R16G16Unorm: return "R16G16Unorm";
    case TextureFormat::R16G16Snorm: return "R16G16Snorm";
    case TextureFormat::R16G16Uint: return "R16G16Uint";
    case TextureFormat::R16G16Sint: return "R16G16Sint";
    case TextureFormat::R16G16Float: return "R16G16Float";
    case TextureFormat::R32G32Uint: return "R32G32Uint";
    case TextureFormat::R32G32Sint: return "R32G32Sint";
    case TextureFormat::R32G32Float: return "R32G32Float";
    case TextureFormat::R32G32B32Uint: return "R32G32B32Uint";
    case TextureFormat::R32G32B32Sint: return "R32G32B32Sint";
    case TextureFormat::R32G32B32Float: return "R32G32B32Float";
    case TextureFormat::R4G4Unorm: return "R4G4Unorm";
    case TextureFormat::R4G4B4A4Unorm: return "R4G4B4A4Unorm";
    case TextureFormat::R5G5B5A1Unorm: return "R5G5B5A1Unorm";
    case TextureFormat::A5B5G5R1Unorm: return "A5B5G5R1Unorm";
    case TextureFormat::R5G6B5Unorm: return "R5G6B5Unorm";
    case TextureFormat::R10G10B10A2Unorm: return "R10G10B10A2Unorm";
    case TextureFormat::R10G10B10A2Uint: return "R10G10B10A2Uint";
    case TextureFormat::R11G11B10Float: return "R11G11B10Float";
    case TextureFormat::R9G9B9E5Float: return "R9G9B9E5Float";
    case TextureFormat::R8G8B8A8Unorm: return "R8G8B8A8Unorm";
    case TextureFormat::R8G8B8A8Snorm: return "R8G8B8A8Snorm";
    case TextureFormat::R8G8B8A8Uint: return "R8G8B8A8Uint";
    case TextureFormat::R8G8B8A8Sint: return "R8G8B8A8Sint";
    case TextureFormat::R8G8B8A8Srgb: return "R8G8B8A8Srgb";
    case TextureFormat::B8G8R8A8Unorm: return "B8G8R8A8Unorm";
    case TextureFormat::B8G8R8A8Srgb: return "B8G8R8A8Srgb";
    case TextureFormat::R16G16B16A16Unorm: return "R16G16B16A16Unorm";
    case TextureFormat::R16G16B16A16Snorm: return "R16G16B16A16Snorm";
    case TextureFormat::R16G16B16A16Uint: return "R16G16B16A16Uint";
    case TextureFormat::R16G16B16A16Sint: return "R16G16B16A16Sint";
    case TextureFormat::R16G16B16A16Float: return "R16G16B16A16Float";
    case TextureFormat::R32G32B32A32Uint: return "R32G32B32A32Uint";
    case TextureFormat::R32G32B32A32Sint: return "R32G32B32A32Sint";
    case TextureFormat::R32G32B32A32Float: return "R32G32B32A32Float";
    case TextureFormat::D16Unorm: return "D16Unorm";
    case TextureFormat::D32Float: return "D32Float";
    case TextureFormat::D24UnormS8Uint: return "D24UnormS8Uint";
    case TextureFormat::X8UintD24Unorm: return "X8UintD24Unorm";
    case TextureFormat::S8UintD24Unorm: return "S8UintD24Unorm";
    case TextureFormat::D32FloatS8Uint: return "D32FloatS8Uint";
    case TextureFormat::Bc1RgbaUnorm: return "Bc1RgbaUnorm";
    case TextureFormat::Bc1RgbaSrgb: return "Bc1RgbaSrgb";
    case TextureFormat::Bc2Unorm: return "Bc2Unorm";
    case TextureFormat::Bc2Srgb: return "Bc2Srgb";
    case TextureFormat::Bc3Unorm: return "Bc3Unorm";
    case TextureFormat::Bc3Srgb: return "Bc3Srgb";
    case TextureFormat::Bc4Unorm: return "Bc4Unorm";
    case TextureFormat::Bc4Snorm: return "Bc4Snorm";
    case TextureFormat::Bc5Unorm: return "Bc5Unorm";
    case TextureFormat::Bc5Snorm: return "Bc5Snorm";
    case TextureFormat::Bc7Unorm: return "Bc7Unorm";
    case TextureFormat::Bc7Srgb: return "Bc7Srgb";
    case TextureFormat::Bc6HUfloat: return "Bc6HUfloat";
    case TextureFormat::Bc6HSfloat: return "Bc6HSfloat";
    case TextureFormat::Etc2RgbUnorm: return "Etc2RgbUnorm";
    case TextureFormat::Etc2RgbSrgb: return "Etc2RgbSrgb";
    case TextureFormat::Etc2RgbPtaUnorm: return "Etc2RgbPtaUnorm";
    case TextureFormat::Etc2RgbPtaSrgb: return "Etc2RgbPtaSrgb";
    case TextureFormat::Etc2RgbaUnorm: return "Etc2RgbaUnorm";
    case TextureFormat::Etc2RgbaSrgb: return "Etc2RgbaSrgb";
    case TextureFormat::Astc4x4Unorm: return "Astc4x4Unorm";
    case TextureFormat::Astc4x4Srgb: return "Astc4x4Srgb";
    case TextureFormat::Astc5x4Unorm: return "Astc5x4Unorm";
    case TextureFormat::Astc5x4Srgb: return "Astc5x4Srgb";
    case TextureFormat::Astc5x5Unorm: return "Astc5x5Unorm";
    case TextureFormat::Astc5x5Srgb: return "Astc5x5Srgb";
    case TextureFormat::Astc6x5Unorm: return "Astc6x5Unorm";
    case TextureFormat::Astc6x5Srgb: return "Astc6x5Srgb";
    case TextureFormat::Astc6x6Unorm: return "Astc6x6Unorm";
    case TextureFormat::Astc6x6Srgb: return "Astc6x6Srgb";
    case TextureFormat::Astc8x5Unorm: return "Astc8x5Unorm";
    case TextureFormat::Astc8x5Srgb: return "Astc8x5Srgb";
    case TextureFormat::Astc8x6Unorm: return "Astc8x6Unorm";
    case TextureFormat::Astc8x6Srgb: return "Astc8x6Srgb";
    case TextureFormat::Astc8x8Unorm: return "Astc8x8Unorm";
    case TextureFormat::Astc8x8Srgb: return "Astc8x8Srgb";
    case TextureFormat::Astc10x5Unorm: return "Astc10x5Unorm";
    case TextureFormat::Astc10x5Srgb: return "Astc10x5Srgb";
    case TextureFormat::Astc10x6Unorm: return "Astc10x6Unorm";
    case TextureFormat::Astc10x6Srgb: return "Astc10x6Srgb";
    case TextureFormat::Astc10x8Unorm: return "Astc10x8Unorm";
    case TextureFormat::Astc10x8Srgb: return "Astc10x8Srgb";
    case TextureFormat::Astc10x10Unorm: return "Astc10x10Unorm";
    case TextureFormat::Astc10x10Srgb: return "Astc10x10Srgb";
    case TextureFormat::Astc12x10Unorm: return "Astc12x10Unorm";
    case TextureFormat::Astc12x10Srgb: return "Astc12x10Srgb";
    case TextureFormat::Astc12x12Unorm: return "Astc12x12Unorm";
    case TextureFormat::Astc12x12Srgb: return "Astc12x12Srgb";
    }
    return "unknown format";
}

inline TextureComponentFormat texture_component_format(graphics::TextureFormat format) {
    using graphics::TextureFormat;
    if (graphics::texture_format_info(format).depth) return TextureComponentFormat::depth;
    switch (graphics::texture_numeric_class(format)) {
    case graphics::TextureNumericClass::Uint: return TextureComponentFormat::unsigned_integer;
    case graphics::TextureNumericClass::Sint: return TextureComponentFormat::signed_integer;
    case graphics::TextureNumericClass::Float: break;
    }
    switch (format) {
    case TextureFormat::R8Snorm: case TextureFormat::R16Snorm:
    case TextureFormat::R8G8Snorm: case TextureFormat::R16G16Snorm:
    case TextureFormat::R8G8B8A8Snorm: case TextureFormat::R16G16B16A16Snorm:
    case TextureFormat::Bc4Snorm: case TextureFormat::Bc5Snorm:
        return TextureComponentFormat::snorm;
    case TextureFormat::R16Float: case TextureFormat::R32Float:
    case TextureFormat::R16G16Float: case TextureFormat::R32G32Float:
    case TextureFormat::R32G32B32Float: case TextureFormat::R11G11B10Float:
    case TextureFormat::R9G9B9E5Float: case TextureFormat::R16G16B16A16Float:
    case TextureFormat::R32G32B32A32Float:
    case TextureFormat::Bc6HUfloat: case TextureFormat::Bc6HSfloat:
        return TextureComponentFormat::floating;
    default: return TextureComponentFormat::unorm;
    }
}

// Floating is the translator's broad float-sampled contract. Explicit unorm
// and snorm contracts retain their narrower component-class requirements.
inline std::string texture_fact_mismatch(const DerivedTextureFact& fact,
                                         const graphics::ResolvedTextureBinding& live,
                                         bool uses_sampler = true) {
    const auto target = texture_target_plan(fact, live);
    if (const auto* mismatch = std::get_if<std::string>(&target)) return *mismatch;
    if (fact.normalized_coordinates != live.texture.normalized_coordinates)
        return "the normalized-coordinate expectation does not match the live descriptor";
    if (uses_sampler && fact.comparison != live.sampler.depth_compare)
        return "the comparison expectation (depth-compare) does not match the live sampler";
    // Null bindings have no storage component class or depth/stencil aspect.
    // The shader's reflected declaration determines the typed zero resource.
    if (live.is_null) return {};
    auto component = texture_component_format(live.texture.format);
    // R16Unorm and R32Float
    // TICs are depth-storage aliases for samplers, so a compare-sampled
    // colour-coded descriptor reads the depth component through the alias.
    // Colour formats without a depth equivalent keep the structured refusal.
    if (fact.comparison && graphics::depth_alias_storage(live.texture.format))
        component = TextureComponentFormat::depth;
    const auto scalar = component == TextureComponentFormat::unsigned_integer
        ? TextureSampledScalar::u32 : component == TextureComponentFormat::signed_integer
        ? TextureSampledScalar::s32 : TextureSampledScalar::f32;
    if (fact.scalar != scalar)
        return "the sampled scalar expectation does not match the live format";
    if (fact.comparison && (fact.component_format != TextureComponentFormat::depth ||
                            component != TextureComponentFormat::depth))
        return "comparison sampling requires a depth component format (the shader expects " +
               component_format_name(fact.component_format) + ", the live format " +
               texture_format_name(live.texture.format) + " has no depth alias)";
    if (fact.component_format == TextureComponentFormat::floating) {
        // A non-comparison float sample of a depth format reads the depth
        // value in the first component; Metal admits depth formats behind
        // texture2d<float>, so the depth aspect view is bound as is.
        if (component != TextureComponentFormat::floating &&
            component != TextureComponentFormat::unorm && component != TextureComponentFormat::snorm &&
            component != TextureComponentFormat::depth)
            return "the float-sampled component expectation does not match the live format";
    } else if (fact.component_format != component) {
        return "the component-format expectation does not match the live format";
    }
    if (component == TextureComponentFormat::depth &&
        std::holds_alternative<graphics::TextureError>(metal_sampled_format(live.texture)))
        return "depth sampling cannot use the live stencil aspect";
    return {};
}

} // namespace armigo::platform::metal
