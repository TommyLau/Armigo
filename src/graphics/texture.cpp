// ============================================================================
// Armigo Graphics - GPU Texture Descriptors
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// Texture descriptors decode TIC formats, component swizzles, sRGB selection,
// targets, mip ranges, layers, and pool views from the guest ABI. A null sampled
// backing remains valid only after descriptor memory itself passes validation.
// Sampler words decode coordinate normalization, filtering, trilinear options,
// border fields, and the five-bit fractional conversion table. Pool entries are
// keyed directly and freshness is proved by the mapping/coherence registry.

#include <armigo/graphics/texture.h>
#include <armigo/graphics/resource_memory_registry.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_map>

namespace armigo::graphics {
namespace {

constexpr std::size_t gob_width_bytes = 64;
constexpr std::size_t gob_height_rows = 8;
// Component-type and swizzle source encodings of TIC word 0; the legacy
// profile words keep their historical diagnostics when no table entry names
// their combination.
constexpr std::uint32_t rgba8_format_word = 0x08;
constexpr std::uint32_t rgb10a2_format_word = 0x09;
constexpr std::uint32_t bc1_format_word = 0x24;
constexpr std::uint32_t tic_unorm = 2;
constexpr std::size_t texture_buffer_selector_register = 0x982;
constexpr std::size_t sampler_binding_register = 0x48d;
constexpr std::size_t sampler_pool_register = 0x557;
constexpr std::size_t texture_pool_register = 0x55d;

// Maximum-anisotropy request codes 0-7 of TSC word 0 bits 20-22 to the
// guest sample-ratio table, and TSC word 1 bits 26-30 to the guest's
// 5-bit fractional conversion values.
constexpr std::array<float, 8> sampler_max_anisotropy_table{
    1.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 16.0f};
constexpr std::array<float, 32> sampler_five_bit_fraction_table{
    0.0f,
    0.055555556f,
    0.1f,
    0.13636364f,
    0.16666667f,
    0.1923077f,
    0.21428572f,
    0.23333333f,
    0.25f,
    0.2777778f,
    0.3f,
    0.3181818f,
    0.33333334f,
    0.34615386f,
    0.35714287f,
    0.36666667f,
    0.375f,
    0.3888889f,
    0.4f,
    0.4090909f,
    0.41666666f,
    0.42307693f,
    0.42857143f,
    0.43333334f,
    0.4375f,
    0.44444445f,
    0.45f,
    0.45454547f,
    0.45833334f,
    0.46153846f,
    0.4642857f,
    0.46666667f};

constexpr std::uint32_t bits(std::uint32_t word, unsigned shift,
                             unsigned width) noexcept {
    return (word >> shift) & ((std::uint32_t{1} << width) - 1);
}

std::array<std::uint32_t, 8> read_words(
    std::span<const std::byte> input) noexcept {
    std::array<std::uint32_t, 8> words{};
    for (std::size_t word = 0; word < words.size(); ++word) {
        for (unsigned byte = 0; byte < 4; ++byte) {
            words[word] |=
                std::uint32_t{std::to_integer<std::uint8_t>(
                    input[word * 4 + byte])}
                << (byte * 8);
        }
    }
    return words;
}

TextureError failure(TextureErrorCode code, TextureField field = TextureField::None,
                     std::uint32_t word = 0, std::uint64_t value = 0,
                     std::uint64_t expected = 0,
                     MemoryErrorCode memory_error = MemoryErrorCode::None) noexcept {
    return {code, field, word, value, expected, memory_error};
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t& result) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

// The descriptor's storage arithmetic runs in size_t (a distinct type from
// uint64_t on Darwin), so the same check exists in those units.
bool checked_add(std::size_t left, std::size_t right,
                 std::size_t& result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool align_up(std::size_t value, std::size_t alignment,
              std::size_t& result) noexcept {
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    const std::size_t increment = alignment - remainder;
    if (increment > std::numeric_limits<std::size_t>::max() - value) {
        return false;
    }
    result = value + increment;
    return true;
}

std::uint64_t pool_base(const RenderStop& stop, std::size_t first) noexcept {
    return (std::uint64_t{stop.registers[first]} << 32) |
           stop.registers[first + 1];
}

TextureError memory_failure(TextureField field, std::uint64_t address,
                            MemoryResult result) noexcept {
    if ([] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) {
        armigo::diag::opt_in( "gpu_texture_fault field=%u gpu=%#llx code=%u detail=%.*s\n",
            unsigned(field), static_cast<unsigned long long>(address), unsigned(result.code),
            int(result.detail.size()), result.detail.data());
    }
    return failure(TextureErrorCode::MemoryRead, field, 0, address, 0,
                   result.code);
}

// Pitch storage carries 1D and 2D shapes on this hardware; volume and cube
// targets use block-linear tiling.
bool is_2d_shape(TextureType type) noexcept {
    switch (type) {
        case TextureType::Texture1D:
        case TextureType::Texture1DArray:
        case TextureType::Texture2D:
        case TextureType::Texture2DArray:
        case TextureType::Texture2DRect:
            return true;
        default:
            return false;
    }
}

// Maps a structured layout-layer rejection onto the descriptor profile's
// error vocabulary.
TextureError layout_failure(const TextureLayoutError& error) noexcept {
    switch (error.code) {
        case TextureLayoutErrorCode::InvalidDimensions:
            return failure(TextureErrorCode::UnsupportedTextureState,
                           TextureField::BackingRange, 0, error.value,
                           error.expected);
        case TextureLayoutErrorCode::UnsupportedLayout:
            return failure(TextureErrorCode::UnsupportedLayout,
                           TextureField::BackingRange, 0, error.value,
                           error.expected);
        case TextureLayoutErrorCode::UnsupportedBytesPerBlock:
            return failure(TextureErrorCode::UnsupportedTextureFormat,
                           TextureField::Format, 0, error.value,
                           error.expected);
        case TextureLayoutErrorCode::ExtentOverflow:
        case TextureLayoutErrorCode::DestinationTooSmall:
        case TextureLayoutErrorCode::SourceTooSmall:
            return failure(TextureErrorCode::ExtentOverflow,
                           TextureField::BackingRange);
        case TextureLayoutErrorCode::ExtentLimit:
            return failure(TextureErrorCode::ExtentLimit,
                           TextureField::BackingRange, 0, error.value,
                           error.expected);
    }
    return failure(TextureErrorCode::ExtentOverflow, TextureField::BackingRange);
}

}  // namespace

const char* TextureError::message() const noexcept {
    switch (code) {
        case TextureErrorCode::DescriptorTooSmall:
            return "texture or sampler descriptor is shorter than 32 bytes";
        case TextureErrorCode::UnsupportedTextureFormat:
            return "texture format is not in the TIC format table";
        case TextureErrorCode::UnsupportedComponentType:
            return "texture component types do not select a table format";
        case TextureErrorCode::UnsupportedSwizzle:
            return "texture swizzle source is not a defined TIC component";
        case TextureErrorCode::UnsupportedLayout:
            return "texture layout is outside the reached descriptor profile";
        case TextureErrorCode::UnsupportedTextureType:
            return "texture dimension is outside the reached descriptor profile";
        case TextureErrorCode::UnsupportedTextureState:
            return "texture descriptor field is outside the reached profile";
        case TextureErrorCode::InvalidDimensions:
            return "texture dimensions are invalid";
        case TextureErrorCode::ExtentOverflow:
            return "texture backing extent arithmetic overflows";
        case TextureErrorCode::ExtentLimit:
            return "texture backing extent exceeds the 256 MiB policy";
        case TextureErrorCode::UnsupportedSamplerState:
            return "sampler descriptor field is outside the reached profile";
        case TextureErrorCode::InvalidLodRange:
            return "sampler minimum LOD exceeds maximum LOD";
        case TextureErrorCode::NotDrawStop:
            return "texture resolution requires a stopped 3D draw";
        case TextureErrorCode::StageOutOfRange:
            return "texture shader stage index is outside the render snapshot";
        case TextureErrorCode::ConstantBufferSlotOutOfRange:
            return "selected texture constant-buffer slot is out of range";
        case TextureErrorCode::ConstantBufferDisabled:
            return "selected texture constant buffer is disabled";
        case TextureErrorCode::HandleOutsideConstantBuffer:
            return "texture handle lies outside the selected constant buffer";
        case TextureErrorCode::AddressOverflow:
            return "texture resolver address arithmetic overflows";
        case TextureErrorCode::TextureIndexOutOfRange:
            return "texture descriptor index exceeds its pool limit";
        case TextureErrorCode::SamplerIndexOutOfRange:
            return "sampler descriptor index exceeds its pool limit";
        case TextureErrorCode::MemoryRead:
            return "texture resolver could not read required GPU memory";
    }
    return "unknown texture descriptor error";
}

TextureDescriptorResult decode_texture_descriptor(
    std::span<const std::byte> input) {
    if (input.size() < texture_descriptor_size) {
        return failure(TextureErrorCode::DescriptorTooSmall, TextureField::None,
                       0, input.size(), texture_descriptor_size);
    }
    const auto raw = read_words(input.first(texture_descriptor_size));
    // Bit 31 marks a packed descriptor whose components the shader must
    // unpack itself; the staging profile cannot reproduce that and rejects it.
    const std::uint32_t packed_format_bits = raw[0] & 0x8000007fu;
    if ((packed_format_bits & 0x80000000u) != 0) {
        return failure(TextureErrorCode::UnsupportedTextureFormat,
                       TextureField::Format, 0, packed_format_bits, 0);
    }
    const std::uint32_t format_word = raw[0] & 0x7f;
    std::uint32_t component_type_bits = 0;
    for (unsigned component = 0; component < 4; ++component) {
        component_type_bits |=
            bits(raw[0], 7 + component * 3, 3) << (7 + component * 3);
    }
    const bool srgb = bits(raw[4], 22, 1) != 0;
    // RGB10A2 has no sRGB descriptor form; the reached profile keeps that
    // explicit state error rather than a table miss.
    if (srgb && format_word == rgb10a2_format_word) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::TextureSrgb, 4, 1, 0);
    }
    TextureFormat format{};
    if (!lookup_texture_format(format_word | component_type_bits, srgb,
                               format)) {
        if (format_word == rgba8_format_word ||
            format_word == rgb10a2_format_word ||
            format_word == bc1_format_word) {
            // The originally reached formats report the first component whose
            // type is not UNORM, matching the narrow-profile diagnostics.
            for (unsigned component = 0; component < 4; ++component) {
                const auto type = bits(raw[0], 7 + component * 3, 3);
                if (type != tic_unorm) {
                    return failure(TextureErrorCode::UnsupportedComponentType,
                                   TextureField::ComponentType, 0, type,
                                   tic_unorm);
                }
            }
        }
        return failure(TextureErrorCode::UnsupportedTextureFormat,
                       TextureField::Format, 0,
                       format_word | component_type_bits |
                           (srgb ? tic_srgb_flag : 0),
                       0);
    }
    std::array<TextureSwizzle, 4> swizzle{};
    for (unsigned component = 0; component < 4; ++component) {
        switch (const auto source = bits(raw[0], 19 + component * 3, 3)) {
            case 0:
                swizzle[component] = TextureSwizzle::Zero;
                break;
            case 2:
                swizzle[component] = TextureSwizzle::Red;
                break;
            case 3:
                swizzle[component] = TextureSwizzle::Green;
                break;
            case 4:
                swizzle[component] = TextureSwizzle::Blue;
                break;
            case 5:
                swizzle[component] = TextureSwizzle::Alpha;
                break;
            case 6:
                swizzle[component] = TextureSwizzle::OneSInt;
                break;
            case 7:
                swizzle[component] = TextureSwizzle::OneFloat;
                break;
            default:
                return failure(TextureErrorCode::UnsupportedSwizzle,
                               TextureField::Swizzle, 0, source, 0);
        }
    }
    // Descriptor types Linear (2) and BlockLinear (3) carry the layouts this
    // profile serves; the buffer and color-key variants stay structured
    // rejections.
    const auto descriptor_type = bits(raw[2], 21, 3);
    if (descriptor_type != 2 && descriptor_type != 3) {
        return failure(TextureErrorCode::UnsupportedLayout,
                       TextureField::HeaderVersion, 2, descriptor_type, 3);
    }
    const bool pitch = descriptor_type == 2;
    if (bits(raw[2], 24, 1) != 0) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::LoadStoreHint, 2,
                       bits(raw[2], 24, 1), 0);
    }
    const auto block_width = bits(raw[3], 0, 3);
    const auto block_height = bits(raw[3], 3, 3);
    const auto block_depth = bits(raw[3], 6, 3);
    const auto level_bits = bits(raw[3], 28, 4);
    // TIC texture target: buffer textures have no staging path and the
    // remaining encodings are reserved.
    const auto target_word = bits(raw[4], 23, 4);
    if (target_word == 6 || target_word > 8) {
        return failure(TextureErrorCode::UnsupportedTextureType,
                       TextureField::TextureType, 4, target_word, 0);
    }
    TextureType type{};
    switch (target_word) {
        case 0:
            type = TextureType::Texture1D;
            break;
        case 1:
            type = TextureType::Texture2D;
            break;
        case 2:
            type = TextureType::Texture3D;
            break;
        case 3:
            type = TextureType::Cubemap;
            break;
        case 4:
            type = TextureType::Texture1DArray;
            break;
        case 5:
            type = TextureType::Texture2DArray;
            break;
        case 7:
            type = TextureType::Texture2DRect;
            break;
        default:
            type = TextureType::CubemapArray;
            break;
    }
    const bool is_3d = type == TextureType::Texture3D;
    if (pitch && !is_2d_shape(type)) {
        return failure(TextureErrorCode::UnsupportedLayout,
                       TextureField::TextureType, 4, target_word, 0);
    }
    // Pitch storage carries one level, and the 2D-rect target name promises
    // one; a larger encoded count has no layout this profile can serve.
    if (pitch && level_bits != 0) {
        return failure(TextureErrorCode::UnsupportedLayout,
                       TextureField::MipCount, 3, level_bits + 1, 1);
    }
    if (type == TextureType::Texture2DRect && level_bits != 0) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::MipCount, 3, level_bits + 1, 1);
    }
    // The depth field carries the 3D depth, the array layer count, or the
    // cubemap face count depending on the target; plain 1D/2D shapes ignore
    // it.
    const std::uint32_t depth_or_layers = bits(raw[5], 16, 14) + 1;
    std::uint32_t depth = 1;
    std::uint32_t layers = 1;
    switch (type) {
        case TextureType::Texture3D:
            depth = depth_or_layers;
            break;
        case TextureType::Texture2DArray:
        case TextureType::Texture1DArray:
            layers = depth_or_layers;
            break;
        case TextureType::CubemapArray:
            layers = depth_or_layers * 6;
            break;
        case TextureType::Cubemap:
            layers = 6;
            break;
        default:
            break;
    }
    std::uint32_t width = bits(raw[4], 0, 16) + 1;
    std::uint32_t height = bits(raw[5], 0, 16) + 1;
    // 1D and 1D-array shapes sample as height-one 2D images so views and
    // copies match 2D resources.
    if (type == TextureType::Texture1D ||
        type == TextureType::Texture1DArray) {
        height = 1;
    }
    if (bits(raw[3], 26, 1) != 0 || bits(raw[3], 27, 1) != 0) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       bits(raw[3], 26, 1) != 0
                           ? TextureField::HeaderOptimization
                           : TextureField::DepthTexture,
                       3, 1, 0);
    }
    if (bits(raw[5], 30, 1) != 0) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::Sparse, 5, 1, 0);
    }
    if (bits(raw[5], 31, 1) != 1 && type != TextureType::Texture2DRect) {
        // Non-normalized coordinates are the 2D-rect sampling state.
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::NormalizedCoordinates, 5, 0, 1);
    }
    const std::uint32_t base_layer = bits(raw[4], 16, 3) |
                                     (bits(raw[2], 16, 5) << 3) |
                                     (bits(raw[2], 29, 3) << 8);
    if (base_layer != 0) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::BaseLayer, 4, base_layer, 0);
    }
    const auto base_level = bits(raw[7], 0, 4);
    const auto max_level = bits(raw[7], 4, 4);
    if (pitch && (base_level != 0 || max_level != 0)) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::ResourceMipRange, 7,
                       base_level | (std::uint32_t{max_level} << 4), 0);
    }
    if (bits(raw[7], 8, 4) != 0) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::MultisampleMode, 7,
                       bits(raw[7], 8, 4), 0);
    }
    if (bits(raw[7], 12, 12) != 0) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::TextureMinLod, 7,
                       bits(raw[7], 12, 12), 0);
    }
    const TextureFormatInfo format_info = texture_format_info(format);
    if (!pitch) {
        if (block_width != 0) {
            return failure(TextureErrorCode::UnsupportedLayout,
                           TextureField::BlockWidth, 3, block_width, 0);
        }
        if (block_height > 5) {
            return failure(TextureErrorCode::UnsupportedLayout,
                           TextureField::BlockHeight, 3, block_height, 5);
        }
        if (block_depth > 5) {
            return failure(TextureErrorCode::UnsupportedLayout,
                           TextureField::BlockDepth, 3, block_depth, 5);
        }
        if (block_depth != 0 && !is_3d) {
            // The hardware requires a one-GOB block depth off 3D targets.
            return failure(TextureErrorCode::UnsupportedLayout,
                           TextureField::BlockDepth, 3, block_depth, 0);
        }
        // The GOB height policy counts rows: block-compressed formats count
        // block rows. An encoded stack larger than twice the base-level image rows requires shrinking; the profile rejects that encoding instead
        // of reading the wrong layout.
        const std::size_t profile_rows =
            (std::size_t{height} + format_info.block_height - 1) /
            format_info.block_height;
        if (block_height != 0 && profile_rows <= (4u << block_height)) {
            return failure(TextureErrorCode::UnsupportedLayout,
                           TextureField::BlockHeight, 3, block_height, 0);
        }
        if (bits(raw[3], 10, 3) != 0) {
            return failure(TextureErrorCode::UnsupportedLayout,
                           TextureField::TileWidthSpacing, 3,
                           bits(raw[3], 10, 3), 0);
        }
    }
    // Linear (pitch) storage keeps its stride in the low half of word 3,
    // 32-byte aligned by construction; the block-linear GOB fields share
    // those bits and only matter off the pitch path.
    const std::size_t stride = std::size_t{raw[3] & 0xffff} << 5;

    // Mip levels: the encoded count is clamped to the chain the target's
    // widest axis can carry, and the base..max range
    // selects the sub-chain this descriptor names. A range past the clamped
    // chain is a structured rejection instead of a shorter guess.
    const std::uint32_t levels_raw = level_bits + 1;
    std::uint32_t widest = width;
    if (type != TextureType::Texture1D &&
        type != TextureType::Texture1DArray) {
        widest = std::max(widest, height);
    }
    if (is_3d) {
        widest = std::max(widest, depth);
    }
    const std::uint32_t full_capacity =
        32u - static_cast<std::uint32_t>(std::countl_zero(widest));
    const std::uint32_t full_levels = std::min(levels_raw, full_capacity);
    if (base_level >= full_levels || max_level < base_level) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::ResourceMipRange, 7,
                       base_level | (std::uint32_t{max_level} << 4),
                       full_levels);
    }
    const std::uint32_t view_levels =
        std::min(max_level - base_level + 1, full_levels - base_level);

    // Guest storage geometry of the full chain and of the requested view,
    // from the same layout layer the conversion uses.
    const auto build_description =
        [&](std::uint32_t description_width, std::uint32_t description_height,
            std::uint32_t description_depth,
            std::uint32_t description_levels, std::uint32_t height_log2,
            std::uint32_t depth_log2) {
            TextureLayoutDescription description{};
            description.width = description_width;
            description.height = description_height;
            description.depth = description_depth;
            description.levels = description_levels;
            description.layers = layers;
            description.block_width = format_info.block_width;
            description.block_height = format_info.block_height;
            description.bytes_per_block = format_info.bytes_per_block;
            description.gob_block_height_log2 = height_log2;
            description.gob_block_depth_log2 = depth_log2;
            description.storage = pitch ? TextureStorageLayout::Pitch
                                        : TextureStorageLayout::BlockLinear;
            description.pitch_bytes = stride;
            return description;
        };
    auto full_result = calculate_texture_size(build_description(
        width, height, depth, full_levels, block_height, block_depth));
    if (const auto* layout_error =
            std::get_if<TextureLayoutError>(&full_result)) {
        return layout_failure(*layout_error);
    }
    const auto& full = std::get<TextureLayoutDescription>(full_result);
    std::uint64_t address =
        (std::uint64_t{bits(raw[2], 0, 16)} << 32) | raw[1];
    std::uint32_t view_width = width;
    std::uint32_t view_height = height;
    std::uint32_t view_depth = depth;
    std::uint32_t view_height_log2 = block_height;
    std::uint32_t view_depth_log2 = block_depth;
    if (base_level != 0) {
        // A sub-chain view starts at its base level's storage, advances the address
        // by the level offset, and retargets the extents and GOB blocks.
        const TextureLevelLayout& anchor = full.mips[base_level];
        if (anchor.offset >
            std::numeric_limits<std::uint64_t>::max() - address) {
            return failure(TextureErrorCode::AddressOverflow,
                           TextureField::BackingRange, 1, address,
                           anchor.offset);
        }
        address += anchor.offset;
        view_width = anchor.width;
        view_height = anchor.height;
        view_depth = anchor.depth;
        view_height_log2 = anchor.gob_block_height_log2;
        view_depth_log2 = anchor.gob_block_depth_log2;
    }
    auto view_result = calculate_texture_size(
        build_description(view_width, view_height, view_depth, view_levels,
                          view_height_log2, view_depth_log2));
    if (const auto* layout_error =
            std::get_if<TextureLayoutError>(&view_result)) {
        return layout_failure(*layout_error);
    }
    TextureLayoutDescription view =
        std::get<TextureLayoutDescription>(std::move(view_result));
    std::size_t final_stride = view.layer_stride;
    if (layers > 1) {
        // The guest stores the allocated mip chain for each layer, so every
        // layered view keeps the full chain's stride. This also covers prefix
        // views whose base level is zero but whose maximum level is truncated.
        final_stride = full.layer_stride;
    }
    // One layer's mip chain, plus the strides of the layers before it.
    std::size_t chain = 0;
    for (const TextureLevelLayout& mip : view.mips) {
        if (!checked_add(chain, mip.size, chain)) {
            return failure(TextureErrorCode::ExtentOverflow,
                           TextureField::BackingRange);
        }
    }
    std::size_t extent = chain;
    if (layers > 1) {
        std::size_t layered = 0;
        if (!checked_multiply(final_stride, layers - 1, layered) ||
            !checked_add(layered, chain, extent)) {
            return failure(TextureErrorCode::ExtentOverflow,
                           TextureField::BackingRange);
        }
    }
    if (extent == 0 || extent > max_texture_backing_size) {
        return failure(TextureErrorCode::ExtentLimit,
                       TextureField::BackingRange, 0, extent,
                       max_texture_backing_size);
    }
    std::uint64_t end = 0;
    if (!checked_add(address, extent, end)) {
        return failure(TextureErrorCode::ExtentOverflow,
                       TextureField::BackingRange, 1, address, extent);
    }
    static_cast<void>(end);

    // Level-0 storage shape of the view, cross-checked by read_linear_texture
    // against a forged descriptor.
    std::size_t row_bytes = 0;
    std::size_t storage_height = 0;
    if (pitch) {
        row_bytes = stride;
        storage_height = view.mips[0].block_rows;
    } else {
        std::size_t logical_row = 0;
        if (!checked_multiply(std::size_t{view.mips[0].block_columns},
                              format_info.bytes_per_block, logical_row) ||
            !align_up(logical_row, gob_width_bytes, row_bytes) ||
            !align_up(std::size_t{view.mips[0].block_rows},
                      gob_height_rows << view.mips[0].gob_block_height_log2,
                      storage_height)) {
            return failure(TextureErrorCode::ExtentOverflow,
                           TextureField::BackingRange);
        }
    }

    TextureDescriptor result{};
    result.raw_words = raw;
    result.address = address;
    result.backing_extent = extent;
    result.row_bytes = row_bytes;
    result.storage_height = storage_height;
    result.layer_stride = final_stride;
    result.width = view_width;
    result.height = view_height;
    result.depth = view_depth;
    result.levels = view_levels;
    result.layers = layers;
    result.base_level = base_level;
    result.samples = 1;
    result.block_height_log2 = static_cast<std::uint8_t>(view_height_log2);
    result.block_depth_log2 = static_cast<std::uint8_t>(view_depth_log2);
    result.format = format;
    result.layout =
        pitch ? TextureLayout::Pitch : TextureLayout::BlockLinear;
    result.type = type;
    result.swizzle = swizzle;
    result.srgb = srgb;
    result.normalized_coordinates = bits(raw[5], 31, 1) != 0;
    return result;
}

namespace {

// One TSC wrap axis: hardware codes 0-7 each name an address mode.
bool decode_wrap(SamplerWrap& out, std::uint32_t code) noexcept {
    switch (code) {
        case 0: out = SamplerWrap::Repeat; return true;
        case 1: out = SamplerWrap::MirroredRepeat; return true;
        case 2: out = SamplerWrap::ClampToEdge; return true;
        case 3: out = SamplerWrap::ClampToBorder; return true;
        case 4: out = SamplerWrap::Clamp; return true;
        case 5: out = SamplerWrap::MirrorClampToEdge; return true;
        case 6: out = SamplerWrap::MirrorClampToBorder; return true;
        case 7: out = SamplerWrap::MirrorClamp; return true;
        default: return false;
    }
}

bool decode_filter(SamplerFilter& out, std::uint32_t code) noexcept {
    switch (code) {
        case 1: out = SamplerFilter::Nearest; return true;
        case 2: out = SamplerFilter::Linear; return true;
        default: return false;  // Codes 0 and 3 are reserved encodings.
    }
}

bool decode_mip_filter(SamplerMipFilter& out, std::uint32_t code) noexcept {
    switch (code) {
        case 1: out = SamplerMipFilter::None; return true;
        case 2: out = SamplerMipFilter::Nearest; return true;
        case 3: out = SamplerMipFilter::Linear; return true;
        default: return false;  // Code 0 is a reserved encoding.
    }
}

bool decode_reduction(SamplerReduction& out, std::uint32_t code) noexcept {
    switch (code) {
        case 0: out = SamplerReduction::WeightedAverage; return true;
        case 1: out = SamplerReduction::Minimum; return true;
        case 2: out = SamplerReduction::Maximum; return true;
        default: return false;  // Code 3 is a reserved encoding.
    }
}

bool decode_compare_function(SamplerCompareFunction& out,
                             std::uint32_t code) noexcept {
    switch (code) {
        case 0: out = SamplerCompareFunction::Never; return true;
        case 1: out = SamplerCompareFunction::Less; return true;
        case 2: out = SamplerCompareFunction::Equal; return true;
        case 3: out = SamplerCompareFunction::LessOrEqual; return true;
        case 4: out = SamplerCompareFunction::Greater; return true;
        case 5: out = SamplerCompareFunction::NotEqual; return true;
        case 6: out = SamplerCompareFunction::GreaterOrEqual; return true;
        case 7: out = SamplerCompareFunction::Always; return true;
        default: return false;
    }
}

}  // namespace

SamplerDescriptorResult decode_sampler_descriptor(
    std::span<const std::byte> input) noexcept {
    if (input.size() < texture_descriptor_size) {
        return failure(TextureErrorCode::DescriptorTooSmall, TextureField::None,
                       0, input.size(), texture_descriptor_size);
    }
    const auto raw = read_words(input.first(texture_descriptor_size));
    SamplerDescriptor result{};
    result.raw_words = raw;
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (!decode_wrap(result.wrap[axis], bits(raw[0], axis * 3, 3))) {
            return failure(TextureErrorCode::UnsupportedSamplerState,
                           TextureField::WrapMode, 0,
                           bits(raw[0], axis * 3, 3), 0);
        }
    }
    result.depth_compare = bits(raw[0], 9, 1) != 0;
    if (!decode_compare_function(result.compare_function,
                                 bits(raw[0], 10, 3))) {
        return failure(TextureErrorCode::UnsupportedSamplerState,
                       TextureField::DepthCompare, 0, bits(raw[0], 10, 3), 0);
    }
    result.srgb_border_conversion = bits(raw[0], 13, 1) != 0;
    result.max_anisotropy =
        sampler_max_anisotropy_table[bits(raw[0], 20, 3)];
    if (!decode_filter(result.magnification, bits(raw[1], 0, 2))) {
        return failure(TextureErrorCode::UnsupportedSamplerState,
                       TextureField::MagnificationFilter, 1,
                       bits(raw[1], 0, 2), 0);
    }
    if (!decode_filter(result.minification, bits(raw[1], 4, 2))) {
        return failure(TextureErrorCode::UnsupportedSamplerState,
                       TextureField::MinificationFilter, 1,
                       bits(raw[1], 4, 2), 0);
    }
    if (!decode_mip_filter(result.mip_filter, bits(raw[1], 6, 2))) {
        return failure(TextureErrorCode::UnsupportedSamplerState,
                       TextureField::MipmapFilter, 1, bits(raw[1], 6, 2), 0);
    }
    if (!decode_reduction(result.reduction, bits(raw[1], 10, 2))) {
        return failure(TextureErrorCode::UnsupportedSamplerState,
                       TextureField::ReductionFilter, 1,
                       bits(raw[1], 10, 2), 0);
    }
    // The 13-bit mip-LOD bias is a signed value with 8 fraction bits: shift
    // left then arithmetic-shift right to sign-extend.
    const std::uint32_t bias_code = bits(raw[1], 12, 13);
    const auto signed_bias =
        static_cast<std::int32_t>(bias_code << 19) >> 19;
    result.lod_bias = static_cast<float>(signed_bias) / 256.0f;
    result.float_coordinate_normalization = bits(raw[1], 25, 1) != 0;
    result.trilinear_optimization =
        sampler_five_bit_fraction_table[bits(raw[1], 26, 5)];
    const auto min_lod = bits(raw[2], 0, 12);
    const auto max_lod = bits(raw[2], 12, 12);
    if (min_lod > max_lod) {
        return failure(TextureErrorCode::InvalidLodRange,
                       TextureField::LodRange, 2, min_lod, max_lod);
    }
    result.min_lod = static_cast<float>(min_lod) / 256.0f;
    result.max_lod = static_cast<float>(max_lod) / 256.0f;
    result.srgb_border_color = {
        static_cast<std::uint8_t>(bits(raw[2], 24, 8)),
        static_cast<std::uint8_t>(bits(raw[3], 12, 8)),
        static_cast<std::uint8_t>(bits(raw[3], 20, 8)),
    };
    // Words 4-7 carry the border colour as four IEEE-754 words; every bit
    // pattern is a valid encoding, and the host mapping matches it exactly.
    for (unsigned component = 0; component < 4; ++component) {
        result.border_color[component] =
            std::bit_cast<float>(raw[4 + component]);
    }
    return result;
}

struct TextureDescriptorCache::Impl {
    struct Key {
        std::uint64_t pool;
        std::uint32_t index;
        bool operator==(const Key&) const = default;
    };
    struct Hash {
        std::size_t operator()(Key key) const noexcept {
            return std::hash<std::uint64_t>{}(key.pool) ^
                (std::hash<std::uint32_t>{}(key.index) * 0x9e3779b9u);
        }
    };
    template<class Descriptor> struct Entry {
        Descriptor descriptor;
        RetainedRange range;
        std::uint64_t stamp, gpu_write;
    };
    std::unordered_map<Key, Entry<TextureDescriptor>, Hash> textures;
    std::unordered_map<Key, Entry<SamplerDescriptor>, Hash> samplers;
    const ResourceMemory* owner = nullptr;
    std::uint64_t generation = 0;

    // CPU stamps cannot certify GPU writes. The tiny descriptor interval is
    // inspected without resolving or copying bytes; a GPU-dirty pool falls
    // through to the normal coherent read, including its dependency errors.
    static std::optional<std::uint64_t> gpu_version(
        ResourceMemory& memory, BackingRange range) {
        auto inspected = memory.inspect(range);
        const auto* intervals = std::get_if<std::vector<DirtyRange>>(&inspected);
        if (!intervals || intervals->empty()) return std::nullopt;
        std::uint64_t version = 0;
        for (const auto& interval : *intervals) {
            if (interval.gpu_dirty) return std::nullopt;
            version = std::max(version, interval.version.gpu_write);
        }
        return version;
    }

    template<class Descriptor, class Decode>
    std::variant<Descriptor, TextureError> read(ResourceMemory& memory,
        Key key, std::unordered_map<Key, Entry<Descriptor>, Hash>& entries,
        Decode decode) {
        const auto current_generation = memory.address_space().generation();
        if (owner != &memory || generation != current_generation) {
            textures.clear();
            samplers.clear();
            owner = &memory;
            generation = current_generation;
        }
        std::uint64_t address = 0;
        if (!checked_add(key.pool, std::uint64_t(key.index) * texture_descriptor_size, address))
            return failure(TextureErrorCode::AddressOverflow, TextureField::PoolAddress);
        auto* registry = dynamic_cast<ResourceMemoryRegistry*>(&memory);
        if (const auto found = entries.find(key); found != entries.end()) {
            const auto& entry = found->second;
            if (registry && !registry->cpu_written_since(entry.range.bytes, entry.stamp) &&
                gpu_version(memory, entry.range.bytes) == entry.gpu_write) {
                // Inspection does not expose pending writers or failed GPU
                // feedback. Preserve the coherent read's admission checks
                // without copying the already CPU-authoritative bytes.
                if (std::holds_alternative<std::monostate>(memory.resolve(
                        entry.range.bytes, ResolveReason::CpuEngineRead)))
                    return entry.descriptor;
            }
            entries.erase(found);
        }
        const auto stamp = registry ? registry->cpu_write_stamp() : 0;
        std::array<std::byte, texture_descriptor_size> bytes{};
        if (const auto status = memory.address_space().read(address, bytes); !status)
            return memory_failure(TextureField::PoolAddress, address, status);
        auto decoded = decode(bytes);
        if (stamp && std::holds_alternative<Descriptor>(decoded)) {
            auto retained = memory.retain(address, texture_descriptor_size, ResourceAccess::Read);
            if (auto* range = std::get_if<RetainedRange>(&retained);
                range && range->lease && !registry->cpu_written_since(range->bytes, stamp)) {
                if (auto version = gpu_version(memory, range->bytes)) {
                    // Bound retained pool leases even when guest pools churn.
                    if (entries.size() >= 4096) entries.clear();
                    entries.emplace(key, Entry<Descriptor>{std::get<Descriptor>(decoded),
                        std::move(*range), stamp, *version});
                }
            }
        }
        return decoded;
    }
};

TextureDescriptorCache::TextureDescriptorCache() : impl_(std::make_unique<Impl>()) {}
TextureDescriptorCache::~TextureDescriptorCache() = default;
TextureDescriptorResult TextureDescriptorCache::texture(
    ResourceMemory& memory, std::uint64_t pool, std::uint32_t index) {
    return impl_->read(memory, {pool, index}, impl_->textures, decode_texture_descriptor);
}
SamplerDescriptorResult TextureDescriptorCache::sampler(
    ResourceMemory& memory, std::uint64_t pool, std::uint32_t index) {
    return impl_->read(memory, {pool, index}, impl_->samplers, decode_sampler_descriptor);
}

static TextureBindingResult resolve_texture_binding_impl(
    const RenderStop& stop, std::size_t stage,
    std::uint32_t shader_handle_byte_offset,
    const GpuAddressSpace& memory, bool resolve_sampler,
    std::optional<std::uint8_t> handle_cbuf, bool read_backing,
    ResourceMemory* resource_memory = nullptr, TextureDescriptorCache* descriptors = nullptr) {
    if (stop.kind != RenderKind::Draw || stop.engine != engine_class::threed) {
        return failure(TextureErrorCode::NotDrawStop);
    }
    if (stage >= stop.constant_buffers.size()) {
        return failure(TextureErrorCode::StageOutOfRange,
                       TextureField::ConstantBuffer, 0, stage,
                       stop.constant_buffers.size());
    }
    if (resolve_sampler && stop.registers[sampler_binding_register] != 0) {
        return failure(TextureErrorCode::UnsupportedSamplerState,
                       TextureField::PoolAddress, sampler_binding_register,
                       stop.registers[sampler_binding_register], 0);
    }
    const std::uint32_t slot = handle_cbuf ? *handle_cbuf :
        stop.registers[texture_buffer_selector_register];
    if (slot >= stop.constant_buffers[stage].size()) {
        return failure(TextureErrorCode::ConstantBufferSlotOutOfRange,
                       TextureField::ConstantBuffer,
                       texture_buffer_selector_register, slot,
                       stop.constant_buffers[stage].size());
    }
    const ConstantBufferDescription& binding = stop.constant_buffers[stage][slot];
    if (!binding.enabled) {
        return failure(TextureErrorCode::ConstantBufferDisabled,
                       TextureField::ConstantBuffer, slot);
    }
    if (shader_handle_byte_offset > binding.size ||
        binding.size - shader_handle_byte_offset < 4) {
        return failure(TextureErrorCode::HandleOutsideConstantBuffer,
                       TextureField::HandleOffset, slot,
                       shader_handle_byte_offset, binding.size);
    }
    std::uint64_t handle_address = 0;
    if (!checked_add(binding.address, shader_handle_byte_offset,
                     handle_address)) {
        return failure(TextureErrorCode::AddressOverflow,
                       TextureField::HandleOffset, slot, binding.address,
                       shader_handle_byte_offset);
    }
    std::array<std::byte, 4> handle_bytes{};
    if (const MemoryResult status = memory.read(handle_address, handle_bytes);
        !status) {
        return memory_failure(TextureField::HandleOffset, handle_address, status);
    }
    std::uint32_t handle = 0;
    for (unsigned byte = 0; byte < handle_bytes.size(); ++byte) {
        handle |= std::uint32_t{
                      std::to_integer<std::uint8_t>(handle_bytes[byte])}
                  << (byte * 8);
    }
    const std::uint32_t texture_id = handle & 0xfffff;
    const std::uint32_t sampler_id = handle >> 20;
    const std::uint32_t texture_limit = stop.registers[texture_pool_register + 2];
    const std::uint32_t sampler_limit = stop.registers[sampler_pool_register + 2];
    if (texture_id > texture_limit) {
        return failure(TextureErrorCode::TextureIndexOutOfRange,
                       TextureField::PoolLimit, texture_pool_register + 2,
                       texture_id, texture_limit);
    }
    if (resolve_sampler && sampler_id > sampler_limit) {
        return failure(TextureErrorCode::SamplerIndexOutOfRange,
                       TextureField::PoolLimit, sampler_pool_register + 2,
                       sampler_id, sampler_limit);
    }
    std::uint64_t texture_descriptor_address = 0;
    std::uint64_t sampler_descriptor_address = 0;
    if (!checked_add(pool_base(stop, texture_pool_register),
                     std::uint64_t{texture_id} * texture_descriptor_size,
                     texture_descriptor_address) ||
        (resolve_sampler &&
         !checked_add(pool_base(stop, sampler_pool_register),
                      std::uint64_t{sampler_id} * texture_descriptor_size,
                      sampler_descriptor_address))) {
        return failure(TextureErrorCode::AddressOverflow,
                       TextureField::PoolAddress);
    }
    std::array<std::byte, texture_descriptor_size> texture_bytes{};
    if ([] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) {
        armigo::diag::opt_in(
            "gpu_texture_binding stage=%zu handle_gpu=%#llx handle=%#x tic=%#llx tsc=%#llx sampler=%u\n",
            stage, static_cast<unsigned long long>(handle_address), handle,
            static_cast<unsigned long long>(texture_descriptor_address),
            static_cast<unsigned long long>(sampler_descriptor_address), unsigned(resolve_sampler));
    }
    TextureDescriptorResult texture_result;
    if (descriptors) {
        texture_result = descriptors->texture(*resource_memory,
            pool_base(stop, texture_pool_register), texture_id);
    } else {
        if (const auto status = memory.read(texture_descriptor_address, texture_bytes); !status)
            return memory_failure(TextureField::PoolAddress, texture_descriptor_address, status);
        texture_result = decode_texture_descriptor(texture_bytes);
    }
    if (const auto* error = std::get_if<TextureError>(&texture_result)) {
        return *error;
    }
    const TextureDescriptor texture = std::get<TextureDescriptor>(texture_result);
    if ([] { static const bool on = std::getenv("ARMIGO_TRACE_GPU_MAPPINGS") != nullptr; return on; }()) {
        armigo::diag::opt_in( "gpu_texture_backing tic=%#llx gpu=%#llx size=%#zx "
            "words=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
            static_cast<unsigned long long>(texture_descriptor_address),
            static_cast<unsigned long long>(texture.address), texture.backing_extent,
            texture.raw_words[0], texture.raw_words[1], texture.raw_words[2], texture.raw_words[3],
            texture.raw_words[4], texture.raw_words[5], texture.raw_words[6], texture.raw_words[7]);
    }
    SamplerDescriptor sampler;
    if (resolve_sampler) {
        SamplerDescriptorResult sampler_result;
        if (descriptors) {
            sampler_result = descriptors->sampler(*resource_memory,
                pool_base(stop, sampler_pool_register), sampler_id);
        } else {
            std::array<std::byte, texture_descriptor_size> sampler_bytes{};
            if (const auto status = memory.read(sampler_descriptor_address, sampler_bytes); !status)
                return memory_failure(TextureField::PoolAddress, sampler_descriptor_address, status);
            sampler_result = decode_sampler_descriptor(sampler_bytes);
        }
        if (const auto* error = std::get_if<TextureError>(&sampler_result)) return *error;
        sampler = std::get<SamplerDescriptor>(sampler_result);
    }

    // Null sampled resources are distinct from missing mappings. Descriptor
    // pools and samplers still have to be readable and valid; only the decoded
    // zero backing address bypasses guest storage access.
    if (texture.address == 0) {
        return ResolvedTextureBinding{
            handle, texture_id, sampler_id, texture_descriptor_address,
            sampler_descriptor_address, texture, sampler, 0, true};
    }

    if (!read_backing) {
        return ResolvedTextureBinding{
            handle, texture_id, sampler_id, texture_descriptor_address,
            sampler_descriptor_address, texture, sampler};
    }

    std::array<std::byte, 4096> scratch{};
    std::uint64_t current = texture.address;
    std::size_t remaining = texture.backing_extent;
    while (remaining != 0) {
        std::size_t count = std::min(remaining, scratch.size());
        for (const Mapping& mapping : memory.mappings()) {
            if (current >= mapping.gpu_base &&
                current - mapping.gpu_base < mapping.mapped_size) {
                count = std::min<std::size_t>(
                    count, static_cast<std::size_t>(
                               mapping.mapped_size -
                               (current - mapping.gpu_base)));
                break;
            }
        }
        if (const MemoryResult status =
                memory.read(current, std::span<std::byte>{scratch}.first(count));
            !status) {
            return memory_failure(TextureField::BackingRange, current, status);
        }
        current += count;
        remaining -= count;
    }

    return ResolvedTextureBinding{
        handle, texture_id, sampler_id, texture_descriptor_address,
        sampler_descriptor_address, texture, sampler,
        texture.backing_extent};
}

TextureBindingResult resolve_texture_binding(
    const RenderStop& stop, std::size_t stage,
    std::uint32_t shader_handle_byte_offset,
    const GpuAddressSpace& memory, bool resolve_sampler,
    std::optional<std::uint8_t> handle_cbuf) noexcept {
    return resolve_texture_binding_impl(stop, stage, shader_handle_byte_offset,
                                       memory, resolve_sampler, handle_cbuf, true);
}

RetainedTextureBindingResult resolve_texture_binding_descriptors(
    const RenderStop& stop, std::size_t stage,
    std::uint32_t shader_handle_byte_offset,
    ResourceMemory& memory, bool resolve_sampler,
    std::optional<std::uint8_t> handle_cbuf, TextureDescriptorCache* descriptors) {
    auto decoded = resolve_texture_binding_impl(stop, stage, shader_handle_byte_offset,
        memory.address_space(), resolve_sampler, handle_cbuf, false, &memory, descriptors);
    if (const auto* error = std::get_if<TextureError>(&decoded)) return *error;
    auto binding = std::get<ResolvedTextureBinding>(std::move(decoded));
    RetainedRange backing;
    if (!binding.is_null) {
        auto retained = memory.retain(binding.texture.address,
                                     binding.texture.backing_extent, ResourceAccess::Read);
        if (const auto* error = std::get_if<RendererError>(&retained)) {
            return RendererError{{CommandStatus::MemoryFault, error->command().detail,
                                  binding.texture.address}};
        }
        backing = std::get<RetainedRange>(std::move(retained));
        if (!backing.lease || backing.bytes.size != binding.texture.backing_extent ||
            !backing.mapping.address_space || !backing.mapping.mapping ||
            !backing.mapping.generation || !backing.bytes.backing.owner ||
            !backing.bytes.backing.allocation || !backing.bytes.backing.generation) {
            return RendererError{{CommandStatus::MemoryFault,
                "texture descriptor backing has no complete retained identity"}};
        }
    }
    return RetainedTextureBinding{std::move(binding), std::move(backing)};
}

static SamplerDescriptorResult resolve_sampler_binding_impl(
    const RenderStop& stop, std::size_t stage, std::uint8_t slot,
    std::uint32_t offset, const GpuAddressSpace& memory,
    ResourceMemory* resource_memory = nullptr, TextureDescriptorCache* descriptors = nullptr) {
    if (stop.kind != RenderKind::Draw || stop.engine != engine_class::threed)
        return failure(TextureErrorCode::NotDrawStop);
    if (stage >= stop.constant_buffers.size())
        return failure(TextureErrorCode::StageOutOfRange);
    if (stop.registers[sampler_binding_register] != 0)
        return failure(TextureErrorCode::UnsupportedSamplerState,
                       TextureField::PoolAddress, sampler_binding_register,
                       stop.registers[sampler_binding_register], 0);
    if (slot >= stop.constant_buffers[stage].size())
        return failure(TextureErrorCode::ConstantBufferSlotOutOfRange);
    const auto& binding = stop.constant_buffers[stage][slot];
    if (!binding.enabled) return failure(TextureErrorCode::ConstantBufferDisabled);
    if (offset > binding.size || binding.size - offset < 4)
        return failure(TextureErrorCode::HandleOutsideConstantBuffer);
    std::uint64_t address = 0;
    if (!checked_add(binding.address, offset, address))
        return failure(TextureErrorCode::AddressOverflow);
    std::array<std::byte, 4> bytes{};
    if (const auto status = memory.read(address, bytes); !status)
        return memory_failure(TextureField::HandleOffset, address, status);
    std::uint32_t handle = 0;
    for (unsigned i = 0; i != 4; ++i)
        handle |= std::uint32_t(std::to_integer<std::uint8_t>(bytes[i])) << (i * 8);
    const auto index = handle >> 20;
    if (index > stop.registers[sampler_pool_register + 2])
        return failure(TextureErrorCode::SamplerIndexOutOfRange,
                       TextureField::PoolLimit, sampler_pool_register + 2,
                       index, stop.registers[sampler_pool_register + 2]);
    if (!checked_add(pool_base(stop, sampler_pool_register),
                     std::uint64_t(index) * texture_descriptor_size, address))
        return failure(TextureErrorCode::AddressOverflow);
    if (descriptors)
        return descriptors->sampler(*resource_memory, pool_base(stop, sampler_pool_register), index);
    std::array<std::byte, texture_descriptor_size> descriptor{};
    if (const auto status = memory.read(address, descriptor); !status)
        return memory_failure(TextureField::PoolAddress, address, status);
    return decode_sampler_descriptor(descriptor);
}

SamplerDescriptorResult resolve_sampler_binding(
    const RenderStop& stop, std::size_t stage, std::uint8_t slot,
    std::uint32_t offset, const GpuAddressSpace& memory) noexcept {
    return resolve_sampler_binding_impl(stop, stage, slot, offset, memory);
}

SamplerDescriptorResult resolve_sampler_binding(
    const RenderStop& stop, std::size_t stage, std::uint8_t slot,
    std::uint32_t offset, ResourceMemory& memory, TextureDescriptorCache& descriptors) {
    return resolve_sampler_binding_impl(stop, stage, slot, offset,
        memory.address_space(), &memory, &descriptors);
}

TextureLayoutResult<TextureLayoutDescription> texture_view_layout(
    const TextureDescriptor& texture) {
    const TextureFormatInfo format_info = texture_format_info(texture.format);
    TextureLayoutDescription description{};
    description.width = texture.width;
    description.height = texture.height;
    description.depth = texture.depth;
    description.levels = texture.levels;
    description.layers = texture.layers;
    description.block_width = format_info.block_width;
    description.block_height = format_info.block_height;
    description.bytes_per_block = format_info.bytes_per_block;
    description.gob_block_height_log2 = texture.block_height_log2;
    description.gob_block_depth_log2 = texture.block_depth_log2;
    description.gob_block_width_log2 = 0;
    description.storage = texture.layout == TextureLayout::Pitch
                              ? TextureStorageLayout::Pitch
                              : TextureStorageLayout::BlockLinear;
    description.pitch_bytes = texture.row_bytes;
    auto calculated = calculate_texture_size(std::move(description));
    if (std::holds_alternative<TextureLayoutError>(calculated)) {
        return calculated;
    }
    TextureLayoutDescription& layout =
        std::get<TextureLayoutDescription>(calculated);
    std::size_t stride = layout.layer_stride;
    if (texture.layers > 1) {
        // Layered views keep the allocated mip chain's stride; a shorter
        // recorded stride cannot address the guest's layered storage. Prefix
        // views require this even though their base level remains zero.
        if (texture.layer_stride < stride) {
            return TextureLayoutError{TextureLayoutErrorCode::UnsupportedLayout,
                                      texture.layer_stride, stride};
        }
        stride = texture.layer_stride;
    }
    // One layer's mip chain, plus the strides of the layers before it.
    std::size_t chain = 0;
    for (const TextureLevelLayout& mip : layout.mips) {
        if (!checked_add(chain, mip.size, chain)) {
            return TextureLayoutError{TextureLayoutErrorCode::ExtentOverflow};
        }
    }
    std::size_t total = chain;
    if (texture.layers > 1) {
        std::size_t layered = 0;
        if (!checked_multiply(stride, texture.layers - 1, layered) ||
            !checked_add(layered, chain, total)) {
            return TextureLayoutError{TextureLayoutErrorCode::ExtentOverflow};
        }
    }
    layout.layer_stride = stride;
    layout.total_size = total;
    return layout;
}

LinearTextureResult read_linear_texture(const TextureDescriptor& texture,
                                        const GpuAddressSpace& memory) {
    // Staging keeps guest channel order; the platform applies the TIC swizzle
    // through its texture view.
    const TextureFormatInfo format_info = texture_format_info(texture.format);
    // Staging preserves raw depth/stencil bytes; the platform selects and
    // converts the sampled aspect before uploading to the GPU.
    if (texture.width == 0 || texture.height == 0 || texture.depth == 0 ||
        texture.levels == 0 || texture.layers == 0 || texture.samples != 1) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::BackingRange);
    }
    if (texture.block_height_log2 > 5 || texture.block_depth_log2 > 5) {
        return failure(TextureErrorCode::UnsupportedLayout,
                       TextureField::BlockHeight, 0,
                       texture.block_height_log2, 5);
    }
    // Sampled textures are capped per axis before anything is staged.
    if (texture.width > max_sampled_texture_axis ||
        texture.height > max_sampled_texture_axis) {
        return failure(TextureErrorCode::InvalidDimensions,
                       TextureField::BackingRange, 0,
                       texture.width > max_sampled_texture_axis
                           ? texture.width
                           : texture.height,
                       max_sampled_texture_axis);
    }
    if (texture.backing_extent == 0 ||
        texture.backing_extent > max_texture_backing_size) {
        return failure(TextureErrorCode::ExtentLimit,
                       TextureField::BackingRange, 0, texture.backing_extent,
                       max_texture_backing_size);
    }

    // Re-derive the whole storage layout from the descriptor before staging
    // or reading anything, so a forged descriptor can neither stage a huge
    // output nor read a layout its extent does not describe.
    auto layout_result = texture_view_layout(texture);
    if (const auto* layout_error =
            std::get_if<TextureLayoutError>(&layout_result)) {
        return layout_failure(*layout_error);
    }
    const TextureLayoutDescription& layout =
        std::get<TextureLayoutDescription>(layout_result);
    if (layout.total_size != texture.backing_extent) {
        return failure(TextureErrorCode::UnsupportedTextureState,
                       TextureField::BackingRange, 0, texture.backing_extent,
                       layout.total_size);
    }
    const TextureLevelLayout& base_mip = layout.mips.front();
    if (texture.layout == TextureLayout::Pitch) {
        if (texture.row_bytes < base_mip.linear_row_bytes ||
            texture.storage_height != base_mip.block_rows) {
            return failure(TextureErrorCode::UnsupportedTextureState,
                           TextureField::BackingRange, 0, texture.row_bytes,
                           base_mip.linear_row_bytes);
        }
    } else {
        std::size_t expected_row = 0;
        std::size_t expected_storage = 0;
        if (!checked_multiply(std::size_t{base_mip.block_columns},
                              format_info.bytes_per_block, expected_row) ||
            !align_up(expected_row, gob_width_bytes, expected_row) ||
            !align_up(std::size_t{base_mip.block_rows},
                      gob_height_rows << base_mip.gob_block_height_log2,
                      expected_storage) ||
            texture.row_bytes != expected_row ||
            texture.storage_height != expected_storage) {
            return failure(TextureErrorCode::UnsupportedTextureState,
                           TextureField::BackingRange, 0, texture.row_bytes,
                           expected_row);
        }
    }

    std::vector<std::byte> tiled(texture.backing_extent);
    {
        std::uint64_t current = texture.address;
        std::size_t done = 0;
        std::size_t remaining = texture.backing_extent;
        while (remaining != 0) {
            std::size_t count = remaining;
            for (const Mapping& mapping : memory.mappings()) {
                if (current >= mapping.gpu_base &&
                    current - mapping.gpu_base < mapping.mapped_size) {
                    count = std::min<std::size_t>(
                        count, static_cast<std::size_t>(
                                   mapping.mapped_size -
                                   (current - mapping.gpu_base)));
                    break;
                }
            }
            if (const MemoryResult status =
                    memory.read(current, std::span<std::byte>(tiled)
                                             .subspan(done, count));
                !status) {
                return memory_failure(TextureField::BackingRange, current,
                                      status);
            }
            current += count;
            done += count;
            remaining -= count;
        }
    }

    auto linear = convert_texture_to_linear(layout, tiled);
    if (const auto* layout_error = std::get_if<TextureLayoutError>(&linear)) {
        return layout_failure(*layout_error);
    }

    // The converted bytes are layer-major, then level, then depth plane, then
    // block row; the slices let an upload address every level and layer.
    std::size_t layer_linear = 0;
    for (const TextureLevelLayout& mip : layout.mips) {
        if (!checked_add(layer_linear, mip.linear_bytes, layer_linear)) {
            return failure(TextureErrorCode::ExtentOverflow,
                           TextureField::BackingRange);
        }
    }
    LinearTextureData data;
    data.width = texture.width;
    data.height = texture.height;
    data.depth = texture.depth;
    data.row_bytes = base_mip.linear_row_bytes;
    data.bytes = std::move(std::get<std::vector<std::byte>>(linear));
    data.slices.reserve(std::size_t{texture.layers} * layout.mips.size());
    for (std::uint32_t layer = 0; layer < texture.layers; ++layer) {
        std::size_t cursor = 0;
        if (!checked_multiply(std::size_t{layer}, layer_linear, cursor)) {
            return failure(TextureErrorCode::ExtentOverflow,
                           TextureField::BackingRange);
        }
        for (std::uint32_t level = 0; level < layout.mips.size(); ++level) {
            const TextureLevelLayout& mip = layout.mips[level];
            LinearTextureSlice slice;
            slice.level = level;
            slice.layer = layer;
            slice.width = mip.width;
            slice.height = mip.height;
            slice.depth = mip.depth;
            slice.offset = cursor;
            slice.row_bytes = mip.linear_row_bytes;
            slice.size = mip.linear_bytes;
            data.slices.push_back(slice);
            cursor += mip.linear_bytes;
        }
    }
    return data;
}

}  // namespace armigo::graphics
