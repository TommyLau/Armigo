// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/texture.h>

#include <armigo/graphics/color_clear.h>
#include <armigo/graphics/resource_memory_registry.h>
#include <armigo/graphics/texture_format.h>

#include "../../src/graphics/texture_residency.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

void put32(std::span<std::byte> bytes, std::size_t offset,
           std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index) {
        bytes[offset + index] =
            static_cast<std::byte>(value >> (index * 8));
    }
}

// Verifies the depth and colour storage-alias scope contract.
void test_depth_alias_storage_compatibility() {
    const auto key = [](TextureFormat format) {
        TextureStorageKey key;
        key.bytes = BackingRange{BackingId{7, 1, 1}, 256, 4096};
        key.descriptor.format = format;
        key.descriptor.width = 8;
        key.descriptor.height = 8;
        return key;
    };
    using Scope = TextureAliasScope;
    // Exact storage stays compatible in every scope, and sRGB/linear forms of
    // one format keep sharing storage.
    CHECK(compatible_texture_storage(key(TextureFormat::D16Unorm),
                                     key(TextureFormat::D16Unorm)));
    CHECK(compatible_texture_storage(key(TextureFormat::R8G8B8A8Srgb),
                                     key(TextureFormat::R8G8B8A8Unorm)));
    // Without a scope the alias never fires: colour-coded and depth keys of
    // one range stay distinct storages for copies and render targets.
    CHECK(!compatible_texture_storage(key(TextureFormat::D16Unorm),
                                      key(TextureFormat::R16Unorm)));
    CHECK(!compatible_texture_storage(key(TextureFormat::D32Float),
                                      key(TextureFormat::R32Float)));
    // Sampler lookups resolve the colour-coded request onto depth storage:
    // D16/R16 and D32/R32.
    CHECK(compatible_texture_storage(key(TextureFormat::D16Unorm),
                                     key(TextureFormat::R16Unorm), Scope::Sampler));
    CHECK(compatible_texture_storage(key(TextureFormat::D32Float),
                                     key(TextureFormat::R32Float), Scope::Sampler));
    // The reverse has no native form: a colour-format host texture cannot
    // serve a depth consumer.
    CHECK(!compatible_texture_storage(key(TextureFormat::R16Unorm),
                                      key(TextureFormat::D16Unorm), Scope::Sampler));
    CHECK(!compatible_texture_storage(key(TextureFormat::R32Float),
                                      key(TextureFormat::D32Float), Scope::Sampler));
    // Other colour pairs stay distinct even for samplers.
    CHECK(!compatible_texture_storage(key(TextureFormat::D16Unorm),
                                      key(TextureFormat::R16Snorm), Scope::Sampler));
    // The alias changes the pixel format alone: a request whose block height
    // or backing extent differs names a different guest layout and must not
    // be served from the depth allocation, whose layout conversions read a
    // source span sized by one descriptor only.
    const auto shaped = [](TextureFormat format, std::uint32_t block_height_log2,
                           std::size_t extent) {
        TextureStorageKey shaped_key;
        shaped_key.bytes = BackingRange{BackingId{7, 1, 1}, 256, extent};
        shaped_key.descriptor.format = format;
        shaped_key.descriptor.width = 8;
        shaped_key.descriptor.height = 8;
        shaped_key.descriptor.block_height_log2 =
            static_cast<std::uint8_t>(block_height_log2);
        shaped_key.descriptor.backing_extent = extent;
        return shaped_key;
    };
    CHECK(!compatible_texture_storage(shaped(TextureFormat::D16Unorm, 4, 98304),
                                      shaped(TextureFormat::R16Unorm, 1, 73728),
                                      Scope::Sampler));
    CHECK(!compatible_texture_storage(shaped(TextureFormat::D32Float, 4, 98304),
                                      shaped(TextureFormat::R32Float, 1, 73728),
                                      Scope::Sampler));
    CHECK(compatible_texture_storage(shaped(TextureFormat::D16Unorm, 4, 98304),
                                     shaped(TextureFormat::R16Unorm, 4, 98304),
                                     Scope::Sampler));
    CHECK(!compatible_texture_storage(shaped(TextureFormat::D32FloatS8Uint, 4, 98304),
                                      shaped(TextureFormat::R32G32Float, 1, 73728),
                                      Scope::TwodEngine));
    CHECK(compatible_texture_storage(shaped(TextureFormat::D32FloatS8Uint, 4, 98304),
                                     shaped(TextureFormat::R32G32Float, 4, 98304),
                                     Scope::TwodEngine));
    // The 2D engine's depth aliases additionally
    // admit D32FS8 storage as R32G32Float; Armigo's table has no S8Uint or
    // B8G8R8A8Unorm entries, so no other pair of that branch exists.
    CHECK(!compatible_texture_storage(key(TextureFormat::D32FloatS8Uint),
                                      key(TextureFormat::R32G32Float), Scope::Sampler));
    CHECK(compatible_texture_storage(key(TextureFormat::D32FloatS8Uint),
                                     key(TextureFormat::R32G32Float),
                                     Scope::TwodEngine));
    CHECK(!compatible_texture_storage(key(TextureFormat::R32G32Float),
                                      key(TextureFormat::D32FloatS8Uint),
                                      Scope::TwodEngine));
    // The alias maps name both directions of the admitted pairs.
    CHECK(depth_alias_storage(TextureFormat::R16Unorm) ==
          std::optional(TextureFormat::D16Unorm));
    CHECK(depth_alias_storage(TextureFormat::R32Float) ==
          std::optional(TextureFormat::D32Float));
    CHECK(!depth_alias_storage(TextureFormat::R16Snorm).has_value());
    CHECK(colour_alias_storage(TextureFormat::D16Unorm) ==
          std::optional(TextureFormat::R16Unorm));
    CHECK(colour_alias_storage(TextureFormat::D32Float) ==
          std::optional(TextureFormat::R32Float));
    CHECK(!colour_alias_storage(TextureFormat::D32FloatS8Uint).has_value());
}

std::array<std::byte, 32> texture_descriptor(
    std::uint64_t address = 0x10000, std::uint32_t width = 73,
    std::uint32_t height = 45, std::uint32_t block_height = 2,
    std::uint32_t format = 9, bool srgb = false) {
    std::array<std::byte, 32> bytes{};
    std::uint32_t word0 = format;
    for (unsigned component = 0; component < 4; ++component) {
        word0 |= 2u << (7 + component * 3);
        word0 |= (component + 2u) << (19 + component * 3);
    }
    put32(bytes, 0, word0);
    put32(bytes, 4, static_cast<std::uint32_t>(address));
    put32(bytes, 8, static_cast<std::uint32_t>(address >> 32) | (3u << 21));
    // Bits 16-18 are intentionally opaque and preserved.
    put32(bytes, 12, (block_height << 3) | (5u << 16));
    put32(bytes, 16, (width - 1) | (1u << 23) | (3u << 29) |
                          (srgb ? 1u << 22 : 0u));
    put32(bytes, 20, (height - 1) | (1u << 31));
    // Fields ignored by this lookup remain nonzero to prevent accidental rejection.
    put32(bytes, 24, 1u | (2u << 24));
    return bytes;
}

std::array<std::byte, 32> sampler_descriptor(
    std::uint16_t min_lod = 0x100, std::uint16_t max_lod = 0x380,
    bool srgb = true) {
    std::array<std::byte, 32> bytes{};
    std::uint32_t word0 = 2u | (2u << 3) | (2u << 6);
    word0 |= (srgb ? 1u : 0u) << 13;
    word0 |= 1u << 15;  // Opaque to the inspected reference value path.
    put32(bytes, 0, word0);
    put32(bytes, 4, 2u | (2u << 4) | (3u << 6) | (1u << 8));
    put32(bytes, 8, min_lod | (std::uint32_t{max_lod} << 12));
    put32(bytes, 28, std::bit_cast<std::uint32_t>(1.0f));
    return bytes;
}

std::uint32_t get_word(const std::array<std::byte, 32>& bytes,
                       std::size_t word) {
    std::uint32_t value = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
        value |= std::uint32_t{std::to_integer<std::uint8_t>(
                     bytes[word * 4 + byte])}
                 << (byte * 8);
    }
    return value;
}

void set_word(std::array<std::byte, 32>& bytes, std::size_t word,
              std::uint32_t value) {
    put32(bytes, word * 4, value);
}

std::array<std::byte, 32> with_component_type(std::array<std::byte, 32> bytes,
                                              unsigned component,
                                              std::uint32_t type) {
    const unsigned shift = 7 + component * 3;
    set_word(bytes, 0, (get_word(bytes, 0) & ~(7u << shift)) | (type << shift));
    return bytes;
}

std::array<std::byte, 32> with_swizzle(std::array<std::byte, 32> bytes,
                                       unsigned component,
                                       std::uint32_t source) {
    const unsigned shift = 19 + component * 3;
    set_word(bytes, 0,
             (get_word(bytes, 0) & ~(7u << shift)) | (source << shift));
    return bytes;
}

std::array<std::byte, 32> with_all_component_types(
    std::array<std::byte, 32> bytes, std::uint32_t type) {
    std::uint32_t word0 = get_word(bytes, 0);
    for (unsigned component = 0; component < 4; ++component) {
        const unsigned shift = 7 + component * 3;
        word0 = (word0 & ~(7u << shift)) | (type << shift);
    }
    set_word(bytes, 0, word0);
    return bytes;
}

// TIC texture target in word 4 bits 23-26.
std::array<std::byte, 32> with_target(std::array<std::byte, 32> bytes,
                                      std::uint32_t target) {
    set_word(bytes, 4,
             (get_word(bytes, 4) & ~(0xfu << 23)) | (target << 23));
    return bytes;
}

// Encoded mip level count minus one in word 3 bits 28-31, with the inclusive
// maximum level the consistent encoding names in word 7.
std::array<std::byte, 32> with_mip_levels(std::array<std::byte, 32> bytes,
                                          std::uint32_t levels_minus_one) {
    set_word(bytes, 3,
             (get_word(bytes, 3) & ~(0xfu << 28)) |
                 (levels_minus_one << 28));
    set_word(bytes, 7,
             (get_word(bytes, 7) & ~0xf0u) | (levels_minus_one << 4));
    return bytes;
}

// Base level and inclusive maximum level in word 7 bits 0-7.
std::array<std::byte, 32> with_mip_range(std::array<std::byte, 32> bytes,
                                         std::uint32_t base,
                                         std::uint32_t max_inclusive) {
    set_word(bytes, 7,
             (get_word(bytes, 7) & ~0xffu) | base |
                 (max_inclusive << 4));
    return bytes;
}

// Linear (pitch) descriptor type in word 2, with the pitch stride encoded in
// the low half of word 3 where block-linear keeps its GOB fields.
std::array<std::byte, 32> with_pitch(std::array<std::byte, 32> bytes,
                                     std::size_t stride) {
    set_word(bytes, 2,
             (get_word(bytes, 2) & ~(7u << 21)) | (2u << 21));
    set_word(bytes, 3, static_cast<std::uint32_t>(stride >> 5));
    return bytes;
}

// Encoded depth or layer count minus one in word 5 bits 16-29.
std::array<std::byte, 32> with_depth_field(std::array<std::byte, 32> bytes,
                                           std::uint32_t count_minus_one) {
    set_word(bytes, 5,
             (get_word(bytes, 5) & ~(0x3fffu << 16)) |
                 (count_minus_one << 16));
    return bytes;
}

// log2 GOBs per block-of-GOBs along Z in word 3 bits 6-8.
std::array<std::byte, 32> with_block_depth_log2(
    std::array<std::byte, 32> bytes, std::uint32_t log2) {
    set_word(bytes, 3,
             (get_word(bytes, 3) & ~(7u << 6)) | (log2 << 6));
    return bytes;
}

// TIC swizzle source encodings to the decoded descriptor value.
TextureSwizzle decoded_swizzle(std::uint32_t source) {
    switch (source) {
        case 0:
            return TextureSwizzle::Zero;
        case 2:
            return TextureSwizzle::Red;
        case 3:
            return TextureSwizzle::Green;
        case 4:
            return TextureSwizzle::Blue;
        case 5:
            return TextureSwizzle::Alpha;
        case 6:
            return TextureSwizzle::OneSInt;
        case 7:
            return TextureSwizzle::OneFloat;
    }
    return TextureSwizzle::Zero;
}

// An encoded TIC format word: format index, four component types, and the
// word-4 sRGB flag OR-ed into bit 19.
std::uint32_t tic_word(std::uint32_t base, std::uint32_t r, std::uint32_t g,
                       std::uint32_t b, std::uint32_t a, bool srgb = false) {
    return base | (r << 7) | (g << 10) | (b << 13) | (a << 16) |
           (srgb ? (1u << 19) : 0u);
}

const TextureDescriptor* accepted(const TextureDescriptorResult& result) {
    const auto* value = std::get_if<TextureDescriptor>(&result);
    CHECK(value != nullptr);
    return value;
}

const SamplerDescriptor* accepted(const SamplerDescriptorResult& result) {
    const auto* value = std::get_if<SamplerDescriptor>(&result);
    CHECK(value != nullptr);
    return value;
}

template <typename Result>
void rejected_impl(const Result& result, TextureErrorCode code,
                   unsigned line) {
    const auto* error = std::get_if<TextureError>(&result);
    CHECK(error != nullptr);
    if (error != nullptr) {
        if (error->code != code) {
            std::fprintf(stderr,
                         "FAIL %d: rejection code %u != %u (%s, caller %u)\n",
                         __LINE__, static_cast<unsigned>(error->code),
                         static_cast<unsigned>(code), error->message(), line);
            ++failures;
        }
        CHECK(error->message()[0] != '\0');
    }
}

// Reports the caller's line for a failed rejection so a wrong-code failure
// points at its call site.
#define rejected(result, code) rejected_impl(result, code, __LINE__)

void test_texture_decode() {
    const auto bytes = texture_descriptor();
    const TextureDescriptorResult result = decode_texture_descriptor(bytes);
    const TextureDescriptor* texture = accepted(result);
    if (texture == nullptr) {
        return;
    }
    CHECK(texture->address == 0x10000);
    CHECK(texture->width == 73 && texture->height == 45 && texture->depth == 1);
    CHECK(texture->levels == 1 && texture->layers == 1 && texture->samples == 1);
    CHECK(texture->block_height_log2 == 2);
    CHECK(texture->row_bytes == 320);
    CHECK(texture->storage_height == 64);
    CHECK(texture->backing_extent == 20480);
    CHECK(texture->normalized_coordinates && !texture->srgb);
    CHECK(texture->format == TextureFormat::R10G10B10A2Unorm);
    CHECK((texture->swizzle == std::array{TextureSwizzle::Red,
                                          TextureSwizzle::Green,
                                          TextureSwizzle::Blue,
                                          TextureSwizzle::Alpha}));
    CHECK(texture->raw_words[3] == get_word(bytes, 3));
    CHECK(texture->raw_words[6] == get_word(bytes, 6));

    const auto smallest = texture_descriptor(0x20000, 1, 1, 0);
    const TextureDescriptorResult smallest_result =
        decode_texture_descriptor(smallest);
    const TextureDescriptor* one = accepted(smallest_result);
    if (one != nullptr) {
        CHECK(one->row_bytes == 64 && one->storage_height == 8);
        CHECK(one->backing_extent == 512);
    }
    const auto tallest_block = texture_descriptor(0x30000, 17, 257, 5);
    const TextureDescriptorResult tallest_result =
        decode_texture_descriptor(tallest_block);
    const TextureDescriptor* tall = accepted(tallest_result);
    if (tall != nullptr) {
        CHECK(tall->row_bytes == 128 && tall->storage_height == 512);
        CHECK(tall->backing_extent == 65536);
    }
}

void test_rgba8_texture_formats() {
    for (const bool srgb : {false, true}) {
        const auto bytes = texture_descriptor(0x10000, 73, 45, 2, 8, srgb);
        const TextureDescriptorResult result = decode_texture_descriptor(bytes);
        const TextureDescriptor* texture = accepted(result);
        if (texture == nullptr) {
            continue;
        }
        // The sRGB flag selects the sRGB table entry, not just a flag.
        CHECK(texture->format == (srgb ? TextureFormat::R8G8B8A8Srgb
                                       : TextureFormat::R8G8B8A8Unorm));
        CHECK(texture->srgb == srgb);
        CHECK(texture->address == 0x10000);
        CHECK(texture->width == 73 && texture->height == 45);
        CHECK(texture->row_bytes == 320 && texture->storage_height == 64);
        CHECK(texture->backing_extent == 20480);
        CHECK(texture->raw_words[0] == get_word(bytes, 0));
        CHECK(texture->raw_words[4] == get_word(bytes, 4));
    }
    const auto packed_bytes = texture_descriptor(0x10000, 73, 45, 2, 9, false);
    const TextureDescriptorResult packed_result =
        decode_texture_descriptor(packed_bytes);
    const TextureDescriptor* packed = accepted(packed_result);
    if (packed != nullptr) {
        CHECK(packed->format == TextureFormat::R10G10B10A2Unorm);
        CHECK(!packed->srgb);
    }
    const TextureDescriptorResult unsupported_srgb =
        decode_texture_descriptor(texture_descriptor(0x10000, 73, 45, 2, 9, true));
    rejected(unsupported_srgb, TextureErrorCode::UnsupportedTextureState);
    if (const auto* error = std::get_if<TextureError>(&unsupported_srgb)) {
        CHECK(error->field == TextureField::TextureSrgb);
    }
    for (const std::uint32_t format : {1u, 7u, 0x23u, 0x7fu}) {
        for (const bool srgb : {false, true}) {
            rejected(decode_texture_descriptor(
                         texture_descriptor(0x10000, 73, 45, 2, format, srgb)),
                     TextureErrorCode::UnsupportedTextureFormat);
        }
    }
    for (const bool srgb : {false, true}) {
        for (unsigned component = 0; component < 4; ++component) {
            for (const std::uint32_t type : {0u, 1u, 3u, 4u, 5u, 6u, 7u}) {
                rejected(decode_texture_descriptor(with_component_type(
                             texture_descriptor(0x10000, 73, 45, 2, 8, srgb),
                             component, type)),
                         TextureErrorCode::UnsupportedComponentType);
            }
            // Every defined swizzle source decodes to its descriptor value;
            // only the undefined encoding 1 is rejected.
            for (const std::uint32_t source : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
                const auto result = decode_texture_descriptor(with_swizzle(
                    texture_descriptor(0x10000, 73, 45, 2, 8, srgb), component,
                    source));
                if (source == 1u) {
                    rejected(result, TextureErrorCode::UnsupportedSwizzle);
                    continue;
                }
                if (const TextureDescriptor* texture = accepted(result)) {
                    CHECK(texture->swizzle[component] ==
                          decoded_swizzle(source));
                }
            }
        }
    }
}

void test_texture_rejections() {
    struct Mutation {
        std::size_t word;
        std::uint32_t value;
        TextureErrorCode error;
    };
    const std::vector<Mutation> mutations = {
        {0, 10, TextureErrorCode::UnsupportedTextureFormat},
        {0, get_word(texture_descriptor(), 0) | 0x80000000u,
         TextureErrorCode::UnsupportedTextureFormat},
        {0, get_word(texture_descriptor(), 0) ^ (1u << 7),
         TextureErrorCode::UnsupportedComponentType},
        {2, (get_word(texture_descriptor(), 2) & ~(7u << 21)),
         TextureErrorCode::UnsupportedLayout},
        {2, get_word(texture_descriptor(), 2) | (1u << 24),
         TextureErrorCode::UnsupportedTextureState},
        {3, get_word(texture_descriptor(), 3) | 1u,
         TextureErrorCode::UnsupportedLayout},
        {3, (get_word(texture_descriptor(), 3) & ~(7u << 3)) | (6u << 3),
         TextureErrorCode::UnsupportedLayout},
        {3, get_word(texture_descriptor(), 3) | (1u << 6),
         TextureErrorCode::UnsupportedLayout},
        {3, get_word(texture_descriptor(), 3) | (1u << 10),
         TextureErrorCode::UnsupportedLayout},
        {3, get_word(texture_descriptor(), 3) | (1u << 26),
         TextureErrorCode::UnsupportedTextureState},
        {3, get_word(texture_descriptor(), 3) | (1u << 27),
         TextureErrorCode::UnsupportedTextureState},
        {4, get_word(texture_descriptor(), 4) | (1u << 22),
         TextureErrorCode::UnsupportedTextureState},
        {4, (get_word(texture_descriptor(), 4) & ~(0xfu << 23)) | (9u << 23),
         TextureErrorCode::UnsupportedTextureType},
        {4, get_word(texture_descriptor(), 4) | (1u << 16),
         TextureErrorCode::UnsupportedTextureState},
        {5, get_word(texture_descriptor(), 5) | (1u << 30),
         TextureErrorCode::UnsupportedTextureState},
        {5, get_word(texture_descriptor(), 5) & ~(1u << 31),
         TextureErrorCode::UnsupportedTextureState},
        {7, 1u, TextureErrorCode::UnsupportedTextureState},
        {7, 1u << 8, TextureErrorCode::UnsupportedTextureState},
        {7, 1u << 12, TextureErrorCode::UnsupportedTextureState},
    };
    for (const Mutation& mutation : mutations) {
        auto bytes = texture_descriptor();
        set_word(bytes, mutation.word, mutation.value);
        rejected(decode_texture_descriptor(bytes), mutation.error);
    }
    std::array<std::byte, 31> short_input{};
    rejected(decode_texture_descriptor(short_input),
             TextureErrorCode::DescriptorTooSmall);
    const auto huge = texture_descriptor(0x40000, 65536, 65536, 5);
    rejected(decode_texture_descriptor(huge), TextureErrorCode::ExtentLimit);
    for (unsigned block_height = 1; block_height <= 5; ++block_height) {
        const auto boundary = 4u << block_height;
        rejected(decode_texture_descriptor(texture_descriptor(0x10000, 13, boundary, block_height)),
                 TextureErrorCode::UnsupportedLayout);
        CHECK(std::holds_alternative<TextureDescriptor>(decode_texture_descriptor(
            texture_descriptor(0x10000, 13, boundary + 1, block_height))));
    }
}

void test_sampler_decode_and_rejections() {
    const auto bytes = sampler_descriptor(0x100, 0xfff, true);
    const SamplerDescriptorResult result = decode_sampler_descriptor(bytes);
    const SamplerDescriptor* sampler = accepted(result);
    if (sampler != nullptr) {
        CHECK(sampler->min_lod == 1.0f);
        CHECK(sampler->max_lod == 4095.0f / 256.0f);
        CHECK(sampler->lod_bias == 0.0f && sampler->max_anisotropy == 1.0f);
        CHECK(sampler->srgb_border_conversion);
        CHECK(sampler->wrap[0] == SamplerWrap::ClampToEdge &&
              sampler->wrap[1] == SamplerWrap::ClampToEdge &&
              sampler->wrap[2] == SamplerWrap::ClampToEdge);
        CHECK(sampler->magnification == SamplerFilter::Linear &&
              sampler->minification == SamplerFilter::Linear &&
              sampler->mip_filter == SamplerMipFilter::Linear &&
              sampler->reduction == SamplerReduction::WeightedAverage);
        CHECK(!sampler->depth_compare &&
              sampler->compare_function == SamplerCompareFunction::Never);
        CHECK(sampler->trilinear_optimization == 0.0f &&
              !sampler->float_coordinate_normalization);
        CHECK(sampler->raw_words[0] == get_word(bytes, 0));
    }
    const auto linear_bytes = sampler_descriptor(0, 0x200, false);
    const auto linear_result = decode_sampler_descriptor(linear_bytes);
    const SamplerDescriptor* linear = accepted(linear_result);
    if (linear != nullptr) {
        CHECK(!linear->srgb_border_conversion && linear->max_lod == 2.0f);
    }

    // Every wrap code 0-7 is a defined address mode on each axis.
    for (unsigned mode = 0; mode < 8; ++mode) {
        auto candidate = sampler_descriptor();
        for (unsigned axis = 0; axis < 3; ++axis) {
            set_word(candidate, 0,
                     (get_word(candidate, 0) & ~(7u << (axis * 3))) |
                         (mode << (axis * 3)));
        }
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state != nullptr);
        if (state != nullptr) {
            const auto expected = static_cast<SamplerWrap>(mode);
            CHECK(state->wrap[0] == expected && state->wrap[1] == expected &&
                  state->wrap[2] == expected);
        }
    }
    // Depth-compare enable and every comparison function decode together.
    for (unsigned op = 0; op < 8; ++op) {
        auto comparison = sampler_descriptor();
        set_word(comparison, 0, get_word(comparison, 0) | (1u << 9) | (op << 10));
        const auto decoded_result = decode_sampler_descriptor(comparison);
        const auto* state = accepted(decoded_result);
        CHECK(state && state->depth_compare);
        if (state) {
            CHECK(state->compare_function ==
                  static_cast<SamplerCompareFunction>(op));
        }
    }
    {
        // Disabled comparison still decodes the encoded function.
        auto candidate = sampler_descriptor();
        set_word(candidate, 0, get_word(candidate, 0) | (4u << 10));
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state && !state->depth_compare &&
              state->compare_function == SamplerCompareFunction::Greater);
    }
    // Magnification, minification and mip filters decode every defined code.
    for (const unsigned code : {1u, 2u}) {
        auto candidate = sampler_descriptor();
        set_word(candidate, 1, (get_word(candidate, 1) & ~3u) | code);
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state && state->magnification ==
                           static_cast<SamplerFilter>(code));
    }
    for (const unsigned code : {1u, 2u}) {
        auto candidate = sampler_descriptor();
        set_word(candidate, 1, (get_word(candidate, 1) & ~(3u << 4)) |
                                   (code << 4));
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state && state->minification ==
                           static_cast<SamplerFilter>(code));
    }
    for (const unsigned code : {1u, 2u, 3u}) {
        auto candidate = sampler_descriptor();
        set_word(candidate, 1, (get_word(candidate, 1) & ~(3u << 6)) |
                                   (code << 6));
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state && state->mip_filter ==
                           static_cast<SamplerMipFilter>(code));
    }
    for (const unsigned code : {0u, 1u, 2u}) {
        auto candidate = sampler_descriptor();
        set_word(candidate, 1, (get_word(candidate, 1) & ~(3u << 10)) |
                                   (code << 10));
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state && state->reduction ==
                           static_cast<SamplerReduction>(code));
    }
    // The anisotropy request table matches the guest ratios.
    {
        const std::array<float, 8> ratios{1.0f, 2.0f, 4.0f, 6.0f,
                                          8.0f, 10.0f, 12.0f, 16.0f};
        for (unsigned code = 0; code < ratios.size(); ++code) {
            auto candidate = sampler_descriptor();
            set_word(candidate, 0, (get_word(candidate, 0) & ~(7u << 20)) |
                                       (code << 20));
            const auto decoded_result = decode_sampler_descriptor(candidate);
            const auto* state = accepted(decoded_result);
            CHECK(state && state->max_anisotropy == ratios[code]);
        }
    }
    // The 13-bit mip-LOD bias sign-extends with 8 fraction bits.
    {
        const std::array<std::pair<std::uint32_t, float>, 5> biases{{
            {0u, 0.0f},
            {1u, 1.0f / 256.0f},
            {0x0fffu, 4095.0f / 256.0f},
            {0x1000u, -4096.0f / 256.0f},
            {0x1fffu, -1.0f / 256.0f},
        }};
        for (const auto& [code, expected] : biases) {
            auto candidate = sampler_descriptor();
            set_word(candidate, 1,
                     (get_word(candidate, 1) & ~0x1fff0000u) | (code << 12));
            const auto decoded_result = decode_sampler_descriptor(candidate);
            const auto* state = accepted(decoded_result);
            CHECK(state && state->lod_bias == expected);
        }
    }
    // Trilinear optimization decodes through the guest's 5-bit table.
    {
        const std::array<std::pair<std::uint32_t, float>, 3> codes{{
            {0u, 0.0f},
            {1u, 0.055555556f},
            {31u, 0.46666667f},
        }};
        for (const auto& [code, expected] : codes) {
            auto candidate = sampler_descriptor();
            set_word(candidate, 1, (get_word(candidate, 1) & ~(0x1fu << 26)) |
                                       (code << 26));
            const auto decoded_result = decode_sampler_descriptor(candidate);
            const auto* state = accepted(decoded_result);
            CHECK(state && state->trilinear_optimization == expected);
        }
        // The word-1 float-coordinate-normalization bit is retained as a flag.
        auto candidate = sampler_descriptor();
        set_word(candidate, 1, get_word(candidate, 1) | (1u << 25));
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state && state->float_coordinate_normalization);
    }
    // Float RGBA and packed sRGB RGB are distinct fields. Packed red occupies
    // word 2 bits 24-31; green and blue occupy word 3 bits 12-19 and 20-27.
    {
        auto candidate = sampler_descriptor();
        set_word(candidate, 2,
                 (get_word(candidate, 2) & 0x00ffffffu) | (0x12u << 24));
        set_word(candidate, 3, (get_word(candidate, 3) & 0xf0000fffu) |
                                   (0x80u << 12) | (0xfeu << 20));
        for (unsigned word = 0; word < 4; ++word) {
            set_word(candidate, 4 + word,
                     std::bit_cast<std::uint32_t>(
                         static_cast<float>(word) + 0.5f));
        }
        const auto decoded_result = decode_sampler_descriptor(candidate);
        const auto* state = accepted(decoded_result);
        CHECK(state != nullptr);
        if (state != nullptr) {
            const std::array<std::uint8_t, 3> packed{0x12, 0x80, 0xfe};
            CHECK(state->srgb_border_color == packed);
            for (unsigned component = 0; component < 4; ++component) {
                CHECK(state->border_color[component] ==
                      static_cast<float>(component) + 0.5f);
            }
        }
    }

    // Reserved filter and reduction codes are invalid encodings; each is a
    // two-bit field.
    const auto with_field = [](std::array<std::byte, 32> base, unsigned word,
                               unsigned shift, unsigned code) {
        set_word(base, word,
                 (get_word(base, word) & ~(3u << shift)) | (code << shift));
        return base;
    };
    const std::array<std::tuple<unsigned, unsigned, unsigned, TextureField>,
                     6>
        invalid{{
            {1, 0, 0, TextureField::MagnificationFilter},
            {1, 0, 3, TextureField::MagnificationFilter},
            {1, 4, 0, TextureField::MinificationFilter},
            {1, 4, 3, TextureField::MinificationFilter},
            {1, 6, 0, TextureField::MipmapFilter},
            {1, 10, 3, TextureField::ReductionFilter},
        }};
    for (const auto& [word, shift, code, field] : invalid) {
        const auto decoded = decode_sampler_descriptor(
            with_field(sampler_descriptor(), word, shift, code));
        const auto* error = std::get_if<TextureError>(&decoded);
        CHECK(error != nullptr);
        if (error != nullptr) {
            CHECK(error->code == TextureErrorCode::UnsupportedSamplerState);
            CHECK(error->field == field);
            CHECK(error->message()[0] != '\0');
        }
    }
    rejected(decode_sampler_descriptor(sampler_descriptor(0x300, 0x200)),
             TextureErrorCode::InvalidLodRange);
}

struct Backing {
    static constexpr std::uint64_t gpu_base = 0x1000;
    static constexpr std::uint64_t guest_base = 0x800000;
    std::vector<std::byte> bytes = std::vector<std::byte>(0x30000);
    std::uint64_t fail_at = std::numeric_limits<std::uint64_t>::max();
    std::size_t bytes_read = 0;

    static bool read(void* raw, std::uint64_t address,
                     std::span<std::byte> destination) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < guest_base) {
            return false;
        }
        const std::uint64_t offset = address - guest_base;
        if (offset > self.bytes.size() ||
            destination.size() > self.bytes.size() - offset ||
            (self.fail_at >= address &&
             self.fail_at < address + destination.size())) {
            return false;
        }
        std::memcpy(destination.data(), self.bytes.data() + offset,
                    destination.size());
        self.bytes_read += destination.size();
        return true;
    }

    MemoryCallbacks callbacks() noexcept { return {read, nullptr, this}; }

    void store(std::uint64_t gpu_address, std::span<const std::byte> input) {
        const auto offset = static_cast<std::size_t>(gpu_address - gpu_base);
        std::copy(input.begin(), input.end(), bytes.begin() + offset);
    }
};

GpuAddressSpace make_memory(Backing& backing, std::uint64_t mapped_size = 0x30000) {
    const Mapping mapping{Backing::gpu_base, mapped_size, 0,
                          Backing::guest_base, backing.bytes.size(), 0x1000};
    AddressSpaceResult result = make_address_space({&mapping, 1}, backing.callbacks());
    CHECK(std::holds_alternative<GpuAddressSpace>(result));
    return std::get<GpuAddressSpace>(std::move(result));
}

struct ResolveFixture {
    Backing backing;
    GpuAddressSpace memory;
    RenderStop stop{};
    std::uint32_t handle_offset = 12;

    ResolveFixture() : memory(make_memory(backing)) {
        constexpr std::uint64_t sampler_pool = 0x2000;
        constexpr std::uint64_t texture_pool = 0x3000;
        constexpr std::uint32_t texture_id = 2;
        constexpr std::uint32_t sampler_id = 1;
        constexpr std::uint64_t texture_address = 0x10000;
        stop.kind = RenderKind::Draw;
        stop.engine = engine_class::threed;
        stop.registers[0x48d] = 0;
        stop.registers[0x557] = 0;
        stop.registers[0x558] = sampler_pool;
        stop.registers[0x559] = 7;
        stop.registers[0x55d] = 0;
        stop.registers[0x55e] = texture_pool;
        stop.registers[0x55f] = 7;
        stop.registers[0x982] = 3;
        stop.constant_buffers[4][3] = {0x1800, 64, true};
        std::array<std::byte, 4> handle{};
        put32(handle, 0, texture_id | (sampler_id << 20));
        backing.store(0x1800 + handle_offset, handle);
        const auto tic = texture_descriptor(texture_address);
        const auto tsc = sampler_descriptor();
        backing.store(texture_pool + texture_id * 32, tic);
        backing.store(sampler_pool + sampler_id * 32, tsc);
        std::fill(backing.bytes.begin() +
                      static_cast<std::ptrdiff_t>(texture_address - Backing::gpu_base),
                  backing.bytes.begin() +
                      static_cast<std::ptrdiff_t>(texture_address - Backing::gpu_base + 20480),
                  std::byte{0x5a});
    }
};

// Uses the real registry to distinguish mapped/retained evidence from reads.
struct DescriptorBackend final : ResourceMemoryBackend {
    explicit DescriptorBackend(Backing& backing) : backing(backing) {}
    Backing& backing;
    std::vector<std::size_t> reads;
    RendererStatus read_cpu(BackingRange range, std::span<std::byte> bytes) override {
        reads.push_back(bytes.size());
        if (!Backing::read(&backing, Backing::guest_base + range.offset, bytes))
            return RendererError{{CommandStatus::MemoryFault, "synthetic descriptor read failed"}};
        return renderer_success();
    }
    RendererStatus write_cpu(BackingRange, std::span<const std::byte>) override {
        return RendererError{{CommandStatus::Unsupported, "synthetic descriptor write is unexpected"}};
    }
    RendererStatus wait_gpu(CompletionToken) override {
        return RendererError{{CommandStatus::Unsupported, "synthetic descriptor wait is unexpected"}};
    }
};

void test_descriptor_only_resolution() {
    ResolveFixture fixture;
    auto backend = std::make_shared<DescriptorBackend>(fixture.backing);
    auto made = make_resource_memory({1, backend});
    CHECK(std::holds_alternative<std::shared_ptr<ResourceMemoryRegistry>>(made));
    auto memory = std::get<std::shared_ptr<ResourceMemoryRegistry>>(made);
    const auto ok = [](const auto& status) { return std::holds_alternative<std::monostate>(status); };
    const BackingId backing{1, 1, 1};
    const MappingId original{1, 1, 1};
    const auto size = fixture.backing.bytes.size();
    CHECK(ok(memory->register_backing({backing, size, backend})));
    CHECK(ok(memory->register_mapping({original, backing,
        {Backing::gpu_base, size, 0, Backing::guest_base, size, 0x1000}})));
    const auto resolve = [&](bool sampler = true) {
        backend->reads.clear();
        fixture.backing.bytes_read = 0;
        return resolve_texture_binding_descriptors(fixture.stop, 4,
            fixture.handle_offset, *memory, sampler);
    };
    // A payload read would fail. Descriptor admission must still succeed.
    fixture.backing.fail_at = Backing::guest_base + 0x10000;
    auto result = resolve();
    CHECK(std::holds_alternative<RetainedTextureBinding>(result));
    CHECK((backend->reads == std::vector<std::size_t>{4, 32, 32}));
    CHECK(fixture.backing.bytes_read == 68);
    auto retained = std::get<RetainedTextureBinding>(result);
    CHECK(retained.binding.verified_backing_bytes == 0);
    CHECK(retained.backing.bytes.size == 20480);
    CHECK(retained.backing.mapping == original);
    CHECK(retained.backing.lease != nullptr);
    CHECK(ok(memory->validate_retained(retained.backing, ResourceAccess::Read)));
    result = resolve(false);
    CHECK(std::holds_alternative<RetainedTextureBinding>(result));
    CHECK((backend->reads == std::vector<std::size_t>{4, 32}));
    CHECK(fixture.backing.bytes_read == 36);
    // The legacy API must still try the unreadable payload.
    rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
        memory->address_space()), TextureErrorCode::MemoryRead);
    fixture.backing.fail_at = std::numeric_limits<std::uint64_t>::max();

    // Corrupt live TIC/TSC data must not be hidden by an earlier valid binding.
    auto malformed = texture_descriptor(0x10000);
    put32(malformed, 0, 0);
    fixture.backing.store(0x3040, malformed);
    result = resolve();
    CHECK(std::holds_alternative<TextureError>(result));
    CHECK((backend->reads == std::vector<std::size_t>{4, 32}));
    fixture.backing.store(0x3040, texture_descriptor(0x10000));
    fixture.backing.store(0x2020, sampler_descriptor(0x300, 0x200));
    result = resolve();
    CHECK(std::holds_alternative<TextureError>(result));
    CHECK((backend->reads == std::vector<std::size_t>{4, 32, 32}));
    fixture.backing.store(0x2020, sampler_descriptor());

    // Reusing the mapping number and VA changes the generation and backing
    // offset. An older passive lease must not validate against the replacement.
    CHECK(ok(memory->invalidate_mappings(std::span{&original, 1})));
    const MappingId replacement{1, 1, 2};
    CHECK(ok(memory->register_mapping({replacement, backing,
        {Backing::gpu_base, 0x3000, 0, Backing::guest_base, size, 0x1000}})));
    const MappingId payload_mapping{1, 2, 1};
    CHECK(ok(memory->register_mapping({payload_mapping, backing,
        {0x10000, 20480, 0x20000, Backing::guest_base, size, 0x1000}})));
    CHECK(!ok(memory->validate_retained(retained.backing, ResourceAccess::Read)));
    result = resolve();
    CHECK(std::holds_alternative<RetainedTextureBinding>(result));
    retained = std::get<RetainedTextureBinding>(result);
    CHECK(retained.backing.mapping == payload_mapping);
    CHECK(retained.backing.bytes.offset == 0x20000);
    CHECK((backend->reads == std::vector<std::size_t>{4, 32, 32}));

    CHECK(ok(memory->invalidate_mappings(std::span{&payload_mapping, 1})));
    result = resolve();
    CHECK(std::holds_alternative<RendererError>(result));
    CHECK((backend->reads == std::vector<std::size_t>{4, 32, 32}));
    // A mapped starting address is insufficient if the final byte is absent.
    CHECK(ok(memory->register_mapping({{1, 2, 2}, backing,
        {0x10000, 16384, 0x20000, Backing::guest_base, size, 0x1000}})));
    result = resolve();
    CHECK(std::holds_alternative<RendererError>(result));
    fixture.backing.store(0x3040, texture_descriptor(0));
    result = resolve();
    CHECK(std::holds_alternative<RetainedTextureBinding>(result));
    const auto& null = std::get<RetainedTextureBinding>(result);
    CHECK(null.binding.is_null && null.binding.verified_backing_bytes == 0);
    CHECK(!null.backing.lease && null.backing.bytes.size == 0);
    CHECK((backend->reads == std::vector<std::size_t>{4, 32, 32}));
    fixture.backing.fail_at = Backing::guest_base + 0x2020 - Backing::gpu_base;
    result = resolve();
    CHECK(std::holds_alternative<TextureError>(result));
}

void test_resolver_and_memory_validation() {
    ResolveFixture fixture;
    const TextureBindingResult result = resolve_texture_binding(
        fixture.stop, 4, fixture.handle_offset, fixture.memory);
    const auto* binding = std::get_if<ResolvedTextureBinding>(&result);
    CHECK(binding != nullptr);
    if (binding != nullptr) {
        CHECK(binding->texture_id == 2 && binding->sampler_id == 1);
        CHECK(binding->texture_descriptor_address == 0x3040);
        CHECK(binding->sampler_descriptor_address == 0x2020);
        CHECK(binding->verified_backing_bytes == 20480);
        CHECK(binding->texture.backing_extent == 20480);
    }
    CHECK(fixture.backing.bytes_read == 4 + 32 + 32 + 20480);

    ResolveFixture unreadable;
    unreadable.backing.fail_at = Backing::guest_base + 0x10000;
    rejected(resolve_texture_binding(unreadable.stop, 4,
                                     unreadable.handle_offset,
                                     unreadable.memory),
             TextureErrorCode::MemoryRead);
}

void test_independent_resource_provenance() {
    ResolveFixture fixture;
    // An explicit resource provenance reads its own cbuf, independently of the
    // selector used by older reflected bindings. Selector facts are checked by
    // the binding planner before either resolver runs.
    fixture.stop.registers[0x982] = 0;
    const auto texture = resolve_texture_binding(fixture.stop, 4,
        fixture.handle_offset, fixture.memory, false, 3);
    CHECK(std::holds_alternative<ResolvedTextureBinding>(texture));
    rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                     fixture.memory, false),
             TextureErrorCode::ConstantBufferDisabled);

    // A sampler-only handle must not touch its irrelevant TIC or texture data.
    fixture.backing.bytes_read = 0;
    fixture.stop.registers[0x55f] = 0;
    fixture.stop.registers[0x55e] = 0;
    auto sampler = resolve_sampler_binding(fixture.stop, 4, 3,
                                           fixture.handle_offset, fixture.memory);
    CHECK(std::holds_alternative<SamplerDescriptor>(sampler));
    CHECK(fixture.backing.bytes_read == 4 + texture_descriptor_size);
    if (const auto* decoded = std::get_if<SamplerDescriptor>(&sampler))
        CHECK(decoded->min_lod == 1.0f && decoded->max_lod == 3.5f);

    fixture.stop.registers[0x559] = 0;
    rejected(resolve_sampler_binding(fixture.stop, 4, 3, fixture.handle_offset,
                                     fixture.memory), TextureErrorCode::SamplerIndexOutOfRange);
    fixture.stop.registers[0x559] = 7;
    fixture.stop.constant_buffers[4][3].size = fixture.handle_offset + 3;
    rejected(resolve_sampler_binding(fixture.stop, 4, 3, fixture.handle_offset,
                                     fixture.memory), TextureErrorCode::HandleOutsideConstantBuffer);
    fixture.stop.constant_buffers[4][3].size = 64;
    fixture.backing.fail_at = Backing::guest_base + (0x2020 - Backing::gpu_base);
    rejected(resolve_sampler_binding(fixture.stop, 4, 3, fixture.handle_offset,
                                     fixture.memory), TextureErrorCode::MemoryRead);
}

void test_resolver_null_texture_transition() {
    ResolveFixture fixture;
    constexpr std::uint64_t descriptor_address = 0x3040;
    constexpr std::uint64_t texture_address = 0x10000;
    constexpr std::size_t texture_size = 20480;
    const Mapping descriptors{Backing::gpu_base, 0x3000, 0,
                              Backing::guest_base, fixture.backing.bytes.size(), 0x1000};
    const auto mapped_memory = [&](std::uint64_t backing_offset) {
        const std::array mappings{
            descriptors,
            Mapping{texture_address, texture_size, backing_offset,
                    Backing::guest_base, fixture.backing.bytes.size(), 0x1000}};
        auto result = make_address_space(mappings, fixture.backing.callbacks());
        CHECK(std::holds_alternative<GpuAddressSpace>(result));
        return std::get<GpuAddressSpace>(std::move(result));
    };
    const auto resolve = [&] {
        return resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                       fixture.memory);
    };
    const auto check_shape = [&](const ResolvedTextureBinding& binding) {
        CHECK(binding.texture_id == 2 && binding.sampler_id == 1);
        CHECK(binding.texture_descriptor_address == descriptor_address);
        CHECK(binding.sampler_descriptor_address == 0x2020);
        CHECK(binding.texture.width == 73 && binding.texture.height == 45);
        CHECK(binding.texture.format == TextureFormat::R10G10B10A2Unorm);
        CHECK(binding.texture.type == TextureType::Texture2D);
        CHECK(binding.texture.backing_extent == texture_size);
    };

    fixture.memory = mapped_memory(texture_address - Backing::gpu_base);
    auto result = resolve();
    const auto* binding = std::get_if<ResolvedTextureBinding>(&result);
    CHECK(binding != nullptr);
    if (binding != nullptr) {
        check_shape(*binding);
        CHECK(!binding->is_null);
        CHECK(binding->verified_backing_bytes == texture_size);
    }

    // Pools and handles survive removal of the old texture mapping.
    fixture.memory = std::get<GpuAddressSpace>(make_address_space(
        std::span{&descriptors, 1}, fixture.backing.callbacks()));
    fixture.backing.store(descriptor_address, texture_descriptor(0));
    fixture.backing.bytes_read = 0;
    result = resolve();
    binding = std::get_if<ResolvedTextureBinding>(&result);
    CHECK(binding != nullptr);
    if (binding != nullptr) {
        check_shape(*binding);
        CHECK(binding->is_null);
        CHECK(binding->texture.address == 0);
        CHECK(binding->verified_backing_bytes == 0);
    }
    CHECK(fixture.backing.bytes_read == 4 + 32 + 32);

    // A null backing cannot hide an unreadable TIC or sampler descriptor.
    for (const auto address : {descriptor_address, std::uint64_t{0x2020}}) {
        fixture.backing.fail_at = Backing::guest_base + address - Backing::gpu_base;
        result = resolve();
        const auto* error = std::get_if<TextureError>(&result);
        CHECK(error != nullptr);
        if (error != nullptr) {
            CHECK(error->code == TextureErrorCode::MemoryRead);
            CHECK(error->field == TextureField::PoolAddress);
            CHECK(error->memory_error == MemoryErrorCode::ReadFailed);
        }
    }
    fixture.backing.fail_at = std::numeric_limits<std::uint64_t>::max();

    // Nonzero unmapped storage remains a fault, including the removed address.
    fixture.backing.store(descriptor_address, texture_descriptor(texture_address));
    result = resolve();
    const auto* error = std::get_if<TextureError>(&result);
    CHECK(error != nullptr);
    if (error != nullptr) {
        CHECK(error->code == TextureErrorCode::MemoryRead);
        CHECK(error->field == TextureField::BackingRange);
        CHECK(error->memory_error == MemoryErrorCode::UnmappedAddress);
    }

    // Reusing the same GPU address and pool slot must read the new backing.
    constexpr std::uint64_t replacement_offset = 0x20000;
    std::fill_n(fixture.backing.bytes.begin() + replacement_offset,
                texture_size, std::byte{0xa6});
    fixture.memory = mapped_memory(replacement_offset);
    fixture.backing.bytes_read = 0;
    result = resolve();
    binding = std::get_if<ResolvedTextureBinding>(&result);
    CHECK(binding != nullptr);
    if (binding != nullptr) {
        check_shape(*binding);
        CHECK(!binding->is_null);
        CHECK(binding->texture.address == texture_address);
        CHECK(binding->verified_backing_bytes == texture_size);
        std::array<std::byte, 4> texel{};
        CHECK(fixture.memory.read(binding->texture.address, texel));
        CHECK(std::all_of(texel.begin(), texel.end(),
                          [](std::byte value) { return value == std::byte{0xa6}; }));
    }
    CHECK(fixture.backing.bytes_read >= 4 + 32 + 32 + texture_size);
}

void test_resolver_rejections() {
    {
        ResolveFixture fixture;
        rejected(resolve_texture_binding(fixture.stop, 5, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::StageOutOfRange);
    }
    {
        ResolveFixture fixture;
        fixture.stop.registers[0x982] = 18;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::ConstantBufferSlotOutOfRange);
    }
    {
        ResolveFixture fixture;
        fixture.stop.constant_buffers[4][3].enabled = false;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::ConstantBufferDisabled);
    }
    {
        ResolveFixture fixture;
        fixture.stop.constant_buffers[4][3].size = fixture.handle_offset + 3;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::HandleOutsideConstantBuffer);
    }
    {
        ResolveFixture fixture;
        fixture.stop.constant_buffers[4][3].address =
            std::numeric_limits<std::uint64_t>::max() - 2;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::AddressOverflow);
    }
    {
        ResolveFixture fixture;
        fixture.stop.registers[0x48d] = 1;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::UnsupportedSamplerState);
    }
    {
        ResolveFixture fixture;
        fixture.stop.registers[0x55f] = 1;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::TextureIndexOutOfRange);
    }
    {
        ResolveFixture fixture;
        fixture.stop.registers[0x559] = 0;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::SamplerIndexOutOfRange);
    }
    {
        ResolveFixture fixture;
        fixture.stop.registers[0x55d] = 0xffffffffu;
        fixture.stop.registers[0x55e] = 0xfffffff0u;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::AddressOverflow);
    }
    {
        ResolveFixture fixture;
        fixture.backing.fail_at = Backing::guest_base + (0x3040 - Backing::gpu_base);
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::MemoryRead);
    }
    {
        ResolveFixture fixture;
        fixture.stop.kind = RenderKind::Clear;
        rejected(resolve_texture_binding(fixture.stop, 4, fixture.handle_offset,
                                         fixture.memory),
                 TextureErrorCode::NotDrawStop);
    }
}

void test_bc1_texture_formats() {
    // TIC format 0x24 is BC1 with 4x4-texel blocks of 8 bytes; rows and
    // storage count blocks while GOB rounding stays in bytes.
    const auto bytes = texture_descriptor(0x10000, 33, 9, 0, 0x24, false);
    const TextureDescriptorResult result = decode_texture_descriptor(bytes);
    const TextureDescriptor* texture = accepted(result);
    if (texture != nullptr) {
        CHECK(texture->format == TextureFormat::Bc1RgbaUnorm);
        CHECK(!texture->srgb);
        // 9 block columns * 8 bytes round to a 128-byte GOB row; the naive
        // 33-texel row would round to 192 bytes.
        CHECK(texture->row_bytes == 128);
        CHECK(texture->storage_height == 8);
        CHECK(texture->backing_extent == 1024);
        CHECK(texture->address == 0x10000);
        CHECK(texture->width == 33 && texture->height == 9);
    }
    const auto srgb_bytes = texture_descriptor(0x10000, 33, 9, 0, 0x24, true);
    const TextureDescriptorResult srgb_result =
        decode_texture_descriptor(srgb_bytes);
    const TextureDescriptor* srgb_texture = accepted(srgb_result);
    if (srgb_texture != nullptr) {
        CHECK(srgb_texture->format == TextureFormat::Bc1RgbaSrgb);
        CHECK(srgb_texture->srgb);
    }
    const auto smallest = texture_descriptor(0x20000, 1, 1, 0, 0x24, false);
    const TextureDescriptorResult smallest_result =
        decode_texture_descriptor(smallest);
    if (const TextureDescriptor* one = accepted(smallest_result)) {
        CHECK(one->row_bytes == 64 && one->storage_height == 8);
        CHECK(one->backing_extent == 512);
    }
    // The GOB height policy counts 4x4-texel block rows for BC1: 33x20 has
    // 5 block rows, so an encoded log2=2 stack (half-height 16 rows) reads
    // the wrong layout and must be rejected instead of honored.
    rejected(decode_texture_descriptor(texture_descriptor(0x30000, 33, 20, 2, 0x24, false)),
             TextureErrorCode::UnsupportedLayout);
    // A tall image carries its encoded stack: 65 texels are 17 block rows.
    const auto tall = texture_descriptor(0x30000, 5, 65, 2, 0x24, false);
    const TextureDescriptorResult tall_result = decode_texture_descriptor(tall);
    if (const TextureDescriptor* raised = accepted(tall_result)) {
        // 2 block columns * 8 bytes round to 64; 17 block rows round to 32.
        CHECK(raised->row_bytes == 64 && raised->storage_height == 32);
        CHECK(raised->backing_extent == 2048);
    }
    for (unsigned block_height = 1; block_height <= 5; ++block_height) {
        const auto boundary = 16u << block_height;  // exactly 4<<bh block rows
        rejected(decode_texture_descriptor(texture_descriptor(0x10000, 5, boundary, block_height, 0x24, false)),
                 TextureErrorCode::UnsupportedLayout);
        CHECK(std::holds_alternative<TextureDescriptor>(decode_texture_descriptor(
            texture_descriptor(0x10000, 5, boundary + 1, block_height, 0x24, false))));
    }
    const auto huge = texture_descriptor(0x40000, 65536, 65536, 5, 0x24, false);
    rejected(decode_texture_descriptor(huge), TextureErrorCode::ExtentLimit);
    // B8G8R8G8 has no table entry, while the compressed neighbours are
    // distinct table formats with their own block geometry.
    rejected(decode_texture_descriptor(
                 texture_descriptor(0x10000, 33, 9, 0, 0x23u, false)),
             TextureErrorCode::UnsupportedTextureFormat);
    const std::array<std::tuple<std::uint32_t, TextureFormat, std::size_t,
                                std::size_t>,
                     5>
        neighbours{{{0x25u, TextureFormat::Bc2Unorm, 192, 1536},
                    {0x26u, TextureFormat::Bc3Unorm, 192, 1536},
                    {0x27u, TextureFormat::Bc4Unorm, 128, 1024},
                    {0x40u, TextureFormat::Astc4x4Unorm, 192, 1536},
                    {0x0au, TextureFormat::Etc2RgbPtaUnorm, 128, 1024}}};
    for (const auto& [format, expected, row, extent] : neighbours) {
        const auto neighbour_result = decode_texture_descriptor(
            texture_descriptor(0x10000, 33, 9, 0, format, false));
        if (const TextureDescriptor* neighbour = accepted(neighbour_result)) {
            CHECK(neighbour->format == expected);
            CHECK(neighbour->row_bytes == row);
            CHECK(neighbour->storage_height == 8);
            CHECK(neighbour->backing_extent == extent);
        }
    }
    // BC1 keeps the UNORM component requirement on its legacy word.
    for (unsigned component = 0; component < 4; ++component) {
        for (const std::uint32_t type : {0u, 1u, 3u, 4u, 5u, 6u, 7u}) {
            rejected(decode_texture_descriptor(with_component_type(
                         texture_descriptor(0x10000, 33, 9, 0, 0x24, false),
                         component, type)),
                     TextureErrorCode::UnsupportedComponentType);
        }
        // Every defined swizzle source decodes; encoding 1 is undefined.
        for (const std::uint32_t source : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
            const auto result = decode_texture_descriptor(with_swizzle(
                texture_descriptor(0x10000, 33, 9, 0, 0x24, false), component,
                source));
            if (source == 1u) {
                rejected(result, TextureErrorCode::UnsupportedSwizzle);
                continue;
            }
            if (const TextureDescriptor* swizzled = accepted(result)) {
                CHECK(swizzled->swizzle[component] == decoded_swizzle(source));
            }
        }
    }
}

void test_texture_type_decode() {
    struct Case {
        std::uint32_t word;
        TextureType type;
        std::uint32_t depth_field;
        std::uint32_t depth;
        std::uint32_t layers;
        std::uint32_t height;
        std::uint32_t block_height;
        std::size_t layer_chain;
    };
    // The depth field means 3D depth, array layer count, or cubemap faces
    // depending on the target; plain shapes ignore it, and 1D shapes force a
    // height-one image. A height-one image carries a one-GOB encoded stack,
    // so those cases use block height 0. layer_chain is one layer's storage:
    // 73x45 RGBA8 levels one, or 73x1 for the 1D shapes.
    const std::vector<Case> cases = {
        {0, TextureType::Texture1D, 8, 1, 1, 1, 0, 2560},
        {1, TextureType::Texture2D, 8, 1, 1, 45, 2, 20480},
        {4, TextureType::Texture1DArray, 8, 1, 9, 1, 0, 2560},
        {5, TextureType::Texture2DArray, 8, 1, 9, 45, 2, 20480},
        {7, TextureType::Texture2DRect, 0, 1, 1, 45, 2, 20480},
    };
    for (const Case& item : cases) {
        auto bytes = with_depth_field(
            with_target(texture_descriptor(0x10000, 73, 45, item.block_height,
                                           9),
                        item.word),
            item.depth_field);
        const TextureDescriptorResult result = decode_texture_descriptor(bytes);
        const TextureDescriptor* texture = accepted(result);
        if (texture == nullptr) {
            continue;
        }
        CHECK(texture->type == item.type);
        CHECK(texture->depth == item.depth);
        CHECK(texture->layers == item.layers);
        CHECK(texture->height == item.height);
        CHECK(texture->width == 73);
        CHECK(texture->levels == 1);
        CHECK(texture->normalized_coordinates);
        // Level-0 shape is target-independent; layered extents stack the
        // aligned layer chain once per layer.
        CHECK(texture->row_bytes == 320);
        CHECK(texture->storage_height == (item.height == 1 ? 8u : 64u));
        CHECK(texture->backing_extent == item.layer_chain * item.layers);
        CHECK(texture->layer_stride == item.layer_chain);
    }
    // A 2D descriptor ignores its depth field.
    CHECK(std::holds_alternative<TextureDescriptor>(decode_texture_descriptor(
        with_depth_field(texture_descriptor(), 8))));
    // A 3D descriptor admits an encoded GOB block depth: 16x16x4 with a
    // two-GOB block depth stores two 2048-byte slice stacks per level.
    const TextureDescriptorResult volumetric = decode_texture_descriptor(
        with_block_depth_log2(with_depth_field(
            with_target(texture_descriptor(0x10000, 16, 16, 0, 8), 2), 3),
            1));
    if (const TextureDescriptor* volume = accepted(volumetric)) {
        CHECK(volume->type == TextureType::Texture3D);
        CHECK(volume->depth == 4 && volume->layers == 1);
        CHECK(volume->block_depth_log2 == 1);
        CHECK(volume->row_bytes == 64 && volume->storage_height == 16);
        CHECK(volume->backing_extent == 4096);
    }
    // A cubemap stores six faces regardless of its depth field; a cubemap
    // array multiplies the field by six.
    const TextureDescriptorResult cube =
        decode_texture_descriptor(with_target(
            texture_descriptor(0x10000, 16, 16, 0, 8), 3));
    if (const TextureDescriptor* faces = accepted(cube)) {
        CHECK(faces->type == TextureType::Cubemap);
        CHECK(faces->depth == 1 && faces->layers == 6);
        CHECK(faces->backing_extent == 6 * 1024);
    }
    const TextureDescriptorResult cube_array = decode_texture_descriptor(
        with_depth_field(with_target(texture_descriptor(0x10000, 8, 8, 0, 8),
                                     8),
                         1));
    if (const TextureDescriptor* stacked_faces = accepted(cube_array)) {
        CHECK(stacked_faces->type == TextureType::CubemapArray);
        CHECK(stacked_faces->layers == 12);
        CHECK(stacked_faces->backing_extent == 12 * 512);
    }
    // Block depths beyond the layout cap and off-3D block depths stay
    // rejections.
    rejected(decode_texture_descriptor(with_block_depth_log2(
                 with_target(texture_descriptor(), 2), 6)),
             TextureErrorCode::UnsupportedLayout);
    rejected(decode_texture_descriptor(with_block_depth_log2(
                 with_target(texture_descriptor(), 1), 1)),
             TextureErrorCode::UnsupportedLayout);
    // Buffer textures have no staging path and targets 9-15 are reserved.
    rejected(decode_texture_descriptor(with_target(texture_descriptor(), 6)),
             TextureErrorCode::UnsupportedTextureType);
    for (const std::uint32_t reserved : {9u, 12u, 15u}) {
        rejected(decode_texture_descriptor(with_target(texture_descriptor(),
                                                       reserved)),
                 TextureErrorCode::UnsupportedTextureType);
    }
    // The 2D-rect target carries non-normalized coordinates; other targets
    // keep the normalized requirement.
    auto rect = with_target(texture_descriptor(), 7);
    set_word(rect, 5, get_word(rect, 5) & ~(1u << 31));
    const TextureDescriptorResult rect_result =
        decode_texture_descriptor(rect);
    if (const TextureDescriptor* rectangle = accepted(rect_result)) {
        CHECK(rectangle->type == TextureType::Texture2DRect);
        CHECK(!rectangle->normalized_coordinates);
    }
    auto flat = texture_descriptor();
    set_word(flat, 5, get_word(flat, 5) & ~(1u << 31));
    rejected(decode_texture_descriptor(flat),
             TextureErrorCode::UnsupportedTextureState);
    // The rect target name promises a single level.
    rejected(decode_texture_descriptor(with_mip_levels(with_target(
                 texture_descriptor(), 7), 1)),
             TextureErrorCode::UnsupportedTextureState);
}

void test_mip_level_decode() {
    // The encoded count clamps to the chain the widest axis can carry
    // 73x45 tops out at log2(73) + 1 = 7 levels.
    const TextureDescriptorResult clamped = decode_texture_descriptor(
        with_mip_levels(texture_descriptor(), 15));
    const TextureDescriptor* chain = accepted(clamped);
    if (chain != nullptr) {
        CHECK(chain->levels == 7);
        // 20480 + 6144 + 2048 + 4 * 512 storage bytes.
        CHECK(chain->backing_extent == 30720);
        CHECK(chain->layer_stride == 30720);
    }
    // A base..max range names a sub-chain: a prefix view keeps the address.
    const TextureDescriptorResult prefix = decode_texture_descriptor(
        with_mip_range(with_mip_levels(texture_descriptor(), 6), 0, 2));
    if (const TextureDescriptor* head = accepted(prefix)) {
        CHECK(head->base_level == 0 && head->levels == 3);
        CHECK(head->address == 0x10000);
        CHECK(head->backing_extent == 20480 + 6144 + 2048);
    }
    // A sub-chain view starts at its base level's storage: 0x10000 plus the
    // 26624-byte offset of level 2, with the extents and GOB blocks of that
    // level and the remaining five levels of the clamped chain.
    const TextureDescriptorResult subchain = decode_texture_descriptor(
        with_mip_range(with_mip_levels(texture_descriptor(), 6), 2, 6));
    const TextureDescriptor* view = accepted(subchain);
    if (view != nullptr) {
        CHECK(view->base_level == 2 && view->levels == 5);
        CHECK(view->address == 0x10000 + 26624);
        CHECK(view->width == 18 && view->height == 11);
        CHECK(view->block_height_log2 == 1);
        CHECK(view->row_bytes == 128 && view->storage_height == 16);
        CHECK(view->backing_extent == 2048 + 4 * 512);
    }
    // A layered sub-chain keeps the full chain's layer stride: the guest
    // stores full mip chains per layer.
    const TextureDescriptorResult layered = decode_texture_descriptor(
        with_mip_range(with_mip_levels(with_depth_field(with_target(
                           texture_descriptor(), 5), 1), 6),
                       1, 6));
    if (const TextureDescriptor* stacked = accepted(layered)) {
        CHECK(stacked->layers == 2 && stacked->base_level == 1);
        CHECK(stacked->layer_stride == 30720);
        CHECK(stacked->backing_extent == 30720 + (30720 - 20480));
    }
    // The count beyond the axis capacity still serves the clamped chain.
    const TextureDescriptorResult beyond = decode_texture_descriptor(
        with_mip_range(with_mip_levels(texture_descriptor(), 15), 0, 15));
    if (const TextureDescriptor* capped = accepted(beyond)) {
        CHECK(capped->levels == 7);
    }
    // Ranges outside the chain are structured rejections.
    rejected(decode_texture_descriptor(
                 with_mip_range(with_mip_levels(texture_descriptor(), 6),
                                2, 1)),
             TextureErrorCode::UnsupportedTextureState);
    rejected(decode_texture_descriptor(
                 with_mip_range(texture_descriptor(), 5, 6)),
             TextureErrorCode::UnsupportedTextureState);
}

void test_pitch_layout_decode() {
    // Linear storage keeps its stride in word 3: 32x8 RGBA8 with a 128-byte
    // pitch stores 8 rows of 128 bytes.
    const TextureDescriptorResult result = decode_texture_descriptor(
        with_pitch(texture_descriptor(0x10000, 32, 8, 0, 8), 128));
    const TextureDescriptor* linear = accepted(result);
    if (linear != nullptr) {
        CHECK(linear->layout == TextureLayout::Pitch);
        CHECK(linear->type == TextureType::Texture2D);
        CHECK(linear->row_bytes == 128 && linear->storage_height == 8);
        CHECK(linear->backing_extent == 1024);
        CHECK(linear->layer_stride == 1024);
    }
    // A 1D-array pitch texture takes its layers from the depth field.
    const TextureDescriptorResult layered = decode_texture_descriptor(
        with_depth_field(with_pitch(with_target(texture_descriptor(
                                             0x10000, 8, 45, 0, 8), 4), 32),
                         2));
    if (const TextureDescriptor* strips = accepted(layered)) {
        CHECK(strips->type == TextureType::Texture1DArray);
        CHECK(strips->height == 1 && strips->layers == 3);
        CHECK(strips->backing_extent == 3 * 32);
    }
    // A pitch below the tightly packed row, mip levels, a mip range, and
    // volume or cube targets are structured rejections.
    rejected(decode_texture_descriptor(
                 with_pitch(texture_descriptor(0x10000, 32, 8, 0, 8), 96)),
             TextureErrorCode::UnsupportedLayout);
    rejected(decode_texture_descriptor(with_mip_levels(
                 with_pitch(texture_descriptor(0x10000, 32, 8, 0, 8), 128),
                 1)),
             TextureErrorCode::UnsupportedLayout);
    rejected(decode_texture_descriptor(with_pitch(
                 with_mip_range(texture_descriptor(0x10000, 32, 8, 0, 8),
                                0, 0x8),
                 128)),
             TextureErrorCode::UnsupportedTextureState);
    rejected(decode_texture_descriptor(with_pitch(
                 with_target(texture_descriptor(0x10000, 32, 8, 0, 8), 2),
                 128)),
             TextureErrorCode::UnsupportedLayout);
    rejected(decode_texture_descriptor(with_pitch(
                 with_target(texture_descriptor(0x10000, 32, 8, 0, 8), 3),
                 128)),
             TextureErrorCode::UnsupportedLayout);
}

void test_3d_mip_layout_fidelity() {
    // The layout carries the GOB block-depth shrink across its level walk
    // (its mipGobBlocksInZ persists between levels): 64x64x16, one byte per
    // texel, an eight-GOB block depth, and four levels halve the depth block
    // at level 2 (16x16x4 fills half of eight) and level 3 (8x8x2 fills half
    // of four). Level storage repeats the walk's addressing exactly, so a
    // per-level re-derivation from the encoded value would swizzle wrong.
    TextureLayoutDescription description{};
    description.width = 64;
    description.height = 64;
    description.depth = 16;
    description.levels = 4;
    description.layers = 1;
    description.bytes_per_block = 1;
    description.gob_block_height_log2 = 0;
    description.gob_block_depth_log2 = 3;
    const TextureLayoutResult<TextureLayoutDescription> result =
        calculate_texture_size(description);
    CHECK(std::holds_alternative<TextureLayoutDescription>(result));
    if (const auto* layout = std::get_if<TextureLayoutDescription>(&result)) {
        CHECK(layout->mips.size() == 4);
        CHECK(layout->mips[0].size == 65536 && layout->mips[0].offset == 0);
        CHECK(layout->mips[1].size == 16384 &&
              layout->mips[1].offset == 65536 &&
              layout->mips[1].gob_block_depth_log2 == 3);
        CHECK(layout->mips[2].size == 4096 &&
              layout->mips[2].offset == 81920 &&
              layout->mips[2].gob_block_depth_log2 == 2);
        CHECK(layout->mips[3].size == 1024 &&
              layout->mips[3].offset == 86016 &&
              layout->mips[3].gob_block_depth_log2 == 1);
        CHECK(layout->total_size == 87040);
    }
}


void test_read_texture_views() {
    // Round-trips one descriptor: decode, rebuild its layout, build the guest
    // block-linear image from a synthetic linear pattern with the layout
    // layer's inverse conversion, then read it back and require the exact
    // pattern plus per-level/per-layer slice addressing.
    auto round_trip = [](const std::array<std::byte, 32>& bytes) {
        const TextureDescriptorResult decoded =
            decode_texture_descriptor(bytes);
        const TextureDescriptor* texture = accepted(decoded);
        if (texture == nullptr) {
            return;
        }
        const auto layout_result = texture_view_layout(*texture);
        const auto* layout =
            std::get_if<TextureLayoutDescription>(&layout_result);
        CHECK(layout != nullptr);
        if (layout == nullptr) {
            return;
        }
        CHECK(texture->backing_extent == layout->total_size);
        CHECK(texture->layer_stride == layout->layer_stride);
        std::size_t layer_linear = 0;
        for (const TextureLevelLayout& mip : layout->mips) {
            layer_linear += mip.linear_bytes;
        }
        std::vector<std::byte> linear(layer_linear * layout->layers);
        for (std::size_t index = 0; index < linear.size(); ++index) {
            linear[index] = std::byte((index * 31 + 7) & 0xff);
        }
        const auto tiled = convert_texture_to_block_linear(*layout, linear);
        const auto* guest = std::get_if<std::vector<std::byte>>(&tiled);
        CHECK(guest != nullptr);
        if (guest == nullptr) {
            return;
        }
        Backing backing;
        GpuAddressSpace memory = make_memory(backing);
        backing.store(texture->address, *guest);
        const std::size_t before = backing.bytes_read;
        LinearTextureResult read = read_linear_texture(*texture, memory);
        LinearTextureData* data = std::get_if<LinearTextureData>(&read);
        if (data == nullptr) {
            if (const auto* error = std::get_if<TextureError>(&read)) {
                std::fprintf(stderr, "round trip: %s\n", error->message());
            }
            CHECK(false);
            return;
        }
        CHECK(backing.bytes_read - before >= texture->backing_extent);
        CHECK(data->bytes == linear);
        CHECK(data->width == texture->width &&
              data->height == texture->height &&
              data->depth == texture->depth);
        CHECK(data->row_bytes == layout->mips.front().linear_row_bytes);
        CHECK(data->slices.size() == layout->mips.size() * layout->layers);
        std::size_t cursor = 0;
        bool addressed = true;
        for (std::uint32_t layer = 0; layer < layout->layers; ++layer) {
            for (std::uint32_t level = 0; level < layout->mips.size();
                 ++level) {
                const TextureLevelLayout& mip = layout->mips[level];
                const LinearTextureSlice& slice =
                    data->slices[std::size_t{layer} * layout->mips.size() +
                                 level];
                addressed = addressed && slice.layer == layer &&
                            slice.level == level && slice.offset == cursor &&
                            slice.row_bytes == mip.linear_row_bytes &&
                            slice.size == mip.linear_bytes &&
                            slice.width == mip.width &&
                            slice.height == mip.height &&
                            slice.depth == mip.depth;
                cursor += mip.linear_bytes;
            }
        }
        CHECK(addressed);
        CHECK(cursor == layer_linear * layout->layers);
    };

    // 2D with a mip chain: 73x45 RGBA8, seven levels, 17324 linear bytes.
    round_trip(with_mip_levels(texture_descriptor(0x10000, 73, 45, 2, 8), 6));
    // 2D array: 16x16, two levels, four layers.
    round_trip(with_depth_field(
        with_mip_levels(texture_descriptor(0x10000, 16, 16, 0, 8), 1), 3));
    // 3D: 16x16x4, two levels, a two-GOB block depth.
    round_trip(with_block_depth_log2(
        with_depth_field(
            with_mip_levels(
                with_target(texture_descriptor(0x10000, 16, 16, 0, 8), 2),
                1),
            3),
        1));
    // Cube: two levels, six faces.
    round_trip(with_target(
        with_mip_levels(texture_descriptor(0x10000, 16, 16, 0, 8), 1), 3));
    // Cube array: 8x8, one level, twelve faces.
    round_trip(with_depth_field(
        with_target(texture_descriptor(0x10000, 8, 8, 0, 8), 8), 11));
    // BC1 with levels; ASTC 8x8 with levels.
    round_trip(with_mip_levels(texture_descriptor(0x10000, 33, 33, 0, 0x24), 2));
    round_trip(with_mip_levels(texture_descriptor(0x10000, 64, 48, 0, 0x44), 1));
    // Pitch storage, four bytes per texel and one byte per texel.
    round_trip(with_pitch(texture_descriptor(0x10000, 32, 8, 0, 8), 128));
    round_trip(with_pitch(texture_descriptor(0x10000, 32, 8, 0, 0x1d), 64));

    // A layered prefix view retains the allocated seven-level chain stride.
    // Give every level and layer distinct data so reading layer 1 from the
    // prefix chain's shorter 28672-byte stride cannot pass accidentally.
    {
        const auto full_bytes = with_mip_levels(
            with_depth_field(
                with_target(texture_descriptor(0x10000, 73, 45, 2, 8), 5),
                1),
            6);
        const TextureDescriptorResult full_decode =
            decode_texture_descriptor(full_bytes);
        const TextureDescriptor* full_texture = accepted(full_decode);
        CHECK(full_texture != nullptr);
        if (full_texture == nullptr) {
            return;
        }
        const auto full_layout_result = texture_view_layout(*full_texture);
        const auto* full_layout =
            std::get_if<TextureLayoutDescription>(&full_layout_result);
        CHECK(full_layout != nullptr);
        if (full_layout == nullptr) {
            return;
        }
        CHECK(full_layout->layer_stride == 30720);

        std::size_t full_layer_linear = 0;
        for (const TextureLevelLayout& mip : full_layout->mips) {
            full_layer_linear += mip.linear_bytes;
        }
        std::vector<std::byte> full_linear(full_layer_linear * 2);
        for (std::size_t index = 0; index < full_linear.size(); ++index) {
            full_linear[index] = std::byte((index * 47 + 13) & 0xff);
        }
        const auto full_tiled_result =
            convert_texture_to_block_linear(*full_layout, full_linear);
        const auto* full_tiled =
            std::get_if<std::vector<std::byte>>(&full_tiled_result);
        CHECK(full_tiled != nullptr);
        if (full_tiled == nullptr) {
            return;
        }

        const TextureDescriptorResult prefix_decode =
            decode_texture_descriptor(with_mip_range(full_bytes, 0, 2));
        const TextureDescriptor* prefix = accepted(prefix_decode);
        CHECK(prefix != nullptr);
        if (prefix == nullptr) {
            return;
        }
        CHECK(prefix->layer_stride == 30720);
        CHECK(prefix->backing_extent == 30720 + 28672);
        const auto prefix_layout_result = texture_view_layout(*prefix);
        const auto* prefix_layout =
            std::get_if<TextureLayoutDescription>(&prefix_layout_result);
        CHECK(prefix_layout != nullptr);
        if (prefix_layout == nullptr) {
            return;
        }
        CHECK(prefix_layout->layer_stride == 30720);

        std::size_t prefix_layer_linear = 0;
        for (const TextureLevelLayout& mip : prefix_layout->mips) {
            prefix_layer_linear += mip.linear_bytes;
        }
        std::vector<std::byte> expected_prefix;
        expected_prefix.reserve(prefix_layer_linear * 2);
        for (std::size_t layer = 0; layer < 2; ++layer) {
            const auto begin = full_linear.begin() +
                               static_cast<std::ptrdiff_t>(layer *
                                                           full_layer_linear);
            expected_prefix.insert(
                expected_prefix.end(), begin,
                begin + static_cast<std::ptrdiff_t>(prefix_layer_linear));
        }

        Backing backing;
        GpuAddressSpace memory = make_memory(backing);
        backing.store(full_texture->address, *full_tiled);
        LinearTextureResult uploaded = read_linear_texture(*prefix, memory);
        const auto* upload = std::get_if<LinearTextureData>(&uploaded);
        CHECK(upload != nullptr);
        if (upload != nullptr) {
            CHECK(upload->bytes == expected_prefix);
        }

        // Writing layer 1 level 0 must begin at the allocated-chain stride;
        // the omitted tail of layer 0 remains byte-for-byte unchanged.
        std::vector<std::byte> writeback_guest = *full_tiled;
        const std::vector<std::byte> omitted_tail(
            writeback_guest.begin() + 28672,
            writeback_guest.begin() + 30720);
        std::vector<std::byte> replacement(
            prefix_layout->mips[0].linear_bytes, std::byte{0xa7});
        const auto writeback = write_level_from_linear(
            *prefix_layout, replacement, writeback_guest, 0, 1);
        CHECK(std::holds_alternative<std::monostate>(writeback));
        CHECK(std::equal(omitted_tail.begin(), omitted_tail.end(),
                         writeback_guest.begin() + 28672));
        const TextureLevelResult written =
            convert_level_to_linear(*full_layout, writeback_guest, 0, 1);
        const auto* written_level = std::get_if<TextureLevelData>(&written);
        CHECK(written_level != nullptr);
        if (written_level != nullptr) {
            CHECK(written_level->bytes == replacement);
        }
    }

    // A sub-chain view reads the tail of a full chain.
    {
        const auto base_bytes =
            with_mip_levels(texture_descriptor(0x10000, 73, 45, 2, 8), 6);
        const TextureDescriptorResult base_decode =
            decode_texture_descriptor(base_bytes);
        const TextureDescriptor* base = accepted(base_decode);
        CHECK(base != nullptr);
        if (base == nullptr) {
            return;
        }
        const auto base_layout = texture_view_layout(*base);
        const TextureLayoutDescription& full =
            std::get<TextureLayoutDescription>(base_layout);
        std::size_t layer_linear = 0;
        for (const TextureLevelLayout& mip : full.mips) {
            layer_linear += mip.linear_bytes;
        }
        std::vector<std::byte> linear(layer_linear);
        for (std::size_t index = 0; index < linear.size(); ++index) {
            linear[index] = std::byte((index * 31 + 7) & 0xff);
        }
        const auto tiled = convert_texture_to_block_linear(full, linear);
        Backing backing;
        GpuAddressSpace memory = make_memory(backing);
        backing.store(base->address,
                      std::get<std::vector<std::byte>>(tiled));
        const TextureDescriptorResult view_decode =
            decode_texture_descriptor(with_mip_range(base_bytes, 2, 6));
        const TextureDescriptor* view = accepted(view_decode);
        CHECK(view != nullptr);
        if (view == nullptr) {
            return;
        }
        LinearTextureResult read = read_linear_texture(*view, memory);
        LinearTextureData* data = std::get_if<LinearTextureData>(&read);
        CHECK(data != nullptr);
        if (data != nullptr) {
            // The view's linear output starts at level 2 of the pattern.
            const std::size_t offset =
                full.mips[0].linear_bytes + full.mips[1].linear_bytes;
            CHECK(data->bytes.size() == layer_linear - offset);
            bool tail = true;
            for (std::size_t index = 0; index < data->bytes.size(); ++index) {
                tail = tail && data->bytes[index] == linear[offset + index];
            }
            CHECK(tail);
            CHECK(data->slices.size() == 5);
            CHECK(data->slices[0].width == 18 &&
                  data->slices[0].height == 11);
        }
    }
    // Out-of-bounds and inconsistent backings are explicit errors.
    {
        const auto bytes =
            with_mip_levels(texture_descriptor(0x3f000, 73, 45, 2, 8), 6);
        const TextureDescriptorResult outside_decode =
            decode_texture_descriptor(bytes);
        if (const TextureDescriptor* texture = accepted(outside_decode)) {
            Backing backing;
            GpuAddressSpace memory = make_memory(backing);
            rejected(read_linear_texture(*texture, memory),
                     TextureErrorCode::MemoryRead);
        }
        const TextureDescriptorResult chain_decode = decode_texture_descriptor(
            with_mip_levels(texture_descriptor(0x10000, 73, 45, 2, 8), 6));
        const TextureDescriptor* texture = accepted(chain_decode);
        CHECK(texture != nullptr);
        if (texture == nullptr) {
            return;
        }
        Backing backing;
        GpuAddressSpace memory = make_memory(backing);
        TextureDescriptor shortened = *texture;
        shortened.backing_extent -= 1;
        rejected(read_linear_texture(shortened, memory),
                 TextureErrorCode::UnsupportedTextureState);
        TextureDescriptor widened = *texture;
        widened.backing_extent += 1;
        rejected(read_linear_texture(widened, memory),
                 TextureErrorCode::UnsupportedTextureState);
        // A layered sub-chain view must carry the full chain's stride; a
        // shorter recorded one cannot address the guest's layered storage.
        const TextureDescriptorResult layered_decode =
            decode_texture_descriptor(with_mip_range(
                with_mip_levels(with_depth_field(
                                    with_target(texture_descriptor(0x10000,
                                                                  73, 45, 2,
                                                                  8), 5),
                                    1),
                                6),
                1, 6));
        if (const TextureDescriptor* stacked = accepted(layered_decode)) {
            TextureDescriptor short_stride = *stacked;
            short_stride.layer_stride = 1;
            rejected(read_linear_texture(short_stride, memory),
                     TextureErrorCode::UnsupportedLayout);
            LinearTextureResult layered_read =
                read_linear_texture(*stacked, memory);
            CHECK(std::holds_alternative<LinearTextureData>(layered_read));
        }
    }
}

void test_read_linear_texture() {
    Backing backing;
    GpuAddressSpace memory = make_memory(backing);

    // A 12x8 BC1 image stores 3x2 blocks; the GOB placement of each block is
    // hand-derived in byte coordinates ((y & 1) << 4) + (x_bytes & 0xf):
    // block (0,1) at 16, (1,1) at 24, (2,1) at 48.
    constexpr std::uint64_t bc1_address = 0x10000;
    const auto bc1_descriptor = texture_descriptor(bc1_address, 12, 8, 0, 0x24, false);
    const TextureDescriptorResult bc1_decode =
        decode_texture_descriptor(bc1_descriptor);
    const TextureDescriptor* bc1 = accepted(bc1_decode);
    CHECK(bc1 != nullptr);
    if (bc1 == nullptr) {
        return;
    }
    CHECK(bc1->row_bytes == 64 && bc1->backing_extent == 512);
    const std::size_t bc1_offset =
        static_cast<std::size_t>(bc1_address - Backing::gpu_base);
    // Block (column,row) lives at the byte-coordinate GOB position
    // ((row & 1) << 4) + (column * 8 & 0xf) + ((column * 8 & 0x1f) >> 4 << 5)
    // inside the first GOB: 0, 8, 32 in row 0 and 16, 24, 48 in row 1.
    const std::array<std::size_t, 6> swizzled{0, 8, 32, 16, 24, 48};
    for (unsigned row = 0; row < 2; ++row) {
        for (unsigned column = 0; column < 3; ++column) {
            const std::size_t index = row * 3 + column;
            for (unsigned lane = 0; lane < 8; ++lane) {
                backing.bytes[bc1_offset + swizzled[index] + lane] =
                    static_cast<std::byte>(index * 8 + lane + 1);
            }
        }
    }
    LinearTextureResult bc1_read = read_linear_texture(*bc1, memory);
    LinearTextureData* bc1_data = std::get_if<LinearTextureData>(&bc1_read);
    CHECK(bc1_data != nullptr);
    if (bc1_data != nullptr) {
        CHECK(bc1_data->width == 12 && bc1_data->height == 8);
        CHECK(bc1_data->row_bytes == 24);
        CHECK(bc1_data->bytes.size() == 48);
        bool ordered = true;
        for (unsigned row = 0; row < 2 && ordered; ++row) {
            for (unsigned column = 0; column < 3; ++column) {
                const std::size_t index = row * 3 + column;
                for (unsigned lane = 0; lane < 8; ++lane) {
                    const std::size_t source = swizzled[index] + lane;
                    const std::byte expected =
                        static_cast<std::byte>(index * 8 + lane + 1);
                    ordered = ordered &&
                        bc1_data->bytes[index * 8 + lane] == expected &&
                        backing.bytes[bc1_offset + source] == expected;
                }
            }
        }
        CHECK(ordered);
    }

    // Uncompressed rows deswizzle through the same checked helper.
    constexpr std::uint64_t rgba8_address = 0x18000;
    const auto rgba8_descriptor =
        texture_descriptor(rgba8_address, 5, 12, 1, 8, false);
    const TextureDescriptorResult rgba8_decode =
        decode_texture_descriptor(rgba8_descriptor);
    const TextureDescriptor* rgba8 = accepted(rgba8_decode);
    CHECK(rgba8 != nullptr);
    if (rgba8 == nullptr) {
        return;
    }
    CHECK(rgba8->row_bytes == 64 && rgba8->backing_extent == 1024);
    ColorClearPlan plan;
    plan.target_address = rgba8_address;
    plan.width = 5;
    plan.height = 12;
    plan.layout = ColorTargetLayout::BlockLinear;
    plan.block_height_log2 = rgba8->block_height_log2;
    plan.row_bytes = rgba8->row_bytes;
    plan.storage_size = rgba8->backing_extent;
    const std::size_t rgba8_offset =
        static_cast<std::size_t>(rgba8_address - Backing::gpu_base);
    for (std::uint32_t y = 0; y < 12; ++y) {
        for (std::uint32_t x = 0; x < 5; ++x) {
            const auto offset = std::get<std::size_t>(pixel_byte_offset(plan, x, y));
            backing.bytes[rgba8_offset + offset] = static_cast<std::byte>(x + 1);
            backing.bytes[rgba8_offset + offset + 1] = static_cast<std::byte>(y + 1);
            backing.bytes[rgba8_offset + offset + 2] = std::byte{0xab};
            backing.bytes[rgba8_offset + offset + 3] = std::byte{0xcd};
        }
    }
    LinearTextureResult rgba8_read = read_linear_texture(*rgba8, memory);
    LinearTextureData* rgba8_data = std::get_if<LinearTextureData>(&rgba8_read);
    if (rgba8_data == nullptr) {
        if (const auto* error = std::get_if<TextureError>(&rgba8_read)) {
            std::fprintf(stderr, "rgba8 read: %s\n", error->message());
        }
    }
    CHECK(rgba8_data != nullptr);
    if (rgba8_data != nullptr) {
        CHECK(rgba8_data->width == 5 && rgba8_data->height == 12);
        CHECK(rgba8_data->row_bytes == 20);
        CHECK(rgba8_data->bytes.size() == 240);
        bool packed = true;
        for (std::uint32_t y = 0; y < 12 && packed; ++y) {
            for (std::uint32_t x = 0; x < 5; ++x) {
                const auto* pixel = rgba8_data->bytes.data() +
                                    (std::size_t{y} * 5 + x) * 4;
                packed = packed && pixel[0] == std::byte(x + 1) &&
                         pixel[1] == std::byte(y + 1) &&
                         pixel[2] == std::byte{0xab} &&
                         pixel[3] == std::byte{0xcd};
            }
        }
        CHECK(packed);
    }

    // Cross-GOB placements with an encoded one-GOB-per-stack-log2 height:
    // 8x65 is 2x17 blocks (band size 1024 bytes, 512 per GOB), and block
    // rows 8+ live in the second GOB while rows 16+ live in the second band.
    constexpr std::uint64_t stack_address = 0x20000;
    const auto stack_descriptor =
        texture_descriptor(stack_address, 8, 65, 1, 0x24, false);
    const TextureDescriptorResult stack_decode =
        decode_texture_descriptor(stack_descriptor);
    const TextureDescriptor* stack = accepted(stack_decode);
    CHECK(stack != nullptr);
    if (stack != nullptr) {
        CHECK(stack->row_bytes == 64 && stack->backing_extent == 2048);
        const std::size_t stack_offset =
            static_cast<std::size_t>(stack_address - Backing::gpu_base);
        const std::array<std::tuple<unsigned, unsigned, std::size_t>, 6>
            placed{{{0, 0, 0}, {1, 0, 8}, {0, 8, 512}, {1, 9, 536},
                    {0, 16, 1024}, {1, 16, 1032}}};
        for (const auto& [column, row, swizzled] : placed) {
            const std::size_t index = row * 2 + column;
            for (unsigned lane = 0; lane < 8; ++lane) {
                backing.bytes[stack_offset + swizzled + lane] =
                    static_cast<std::byte>((index + lane * 32 + 1) & 0xff);
            }
        }
        LinearTextureResult stack_read = read_linear_texture(*stack, memory);
        LinearTextureData* stack_data = std::get_if<LinearTextureData>(&stack_read);
        CHECK(stack_data != nullptr);
        if (stack_data != nullptr) {
            CHECK(stack_data->row_bytes == 16);
            CHECK(stack_data->bytes.size() == 272);
            bool placed_ok = true;
            for (const auto& [column, row, swizzled] : placed) {
                const std::size_t index = row * 2 + column;
                for (unsigned lane = 0; lane < 8; ++lane) {
                    placed_ok = placed_ok &&
                        stack_data->bytes[index * 8 + lane] ==
                            std::byte((index + lane * 32 + 1) & 0xff);
                }
            }
            CHECK(placed_ok);
            // Naive row-major placement would read block (0,16) from input
            // offset 256, which holds zeros; its content lives at 1024 and
            // the output slot carries its first-byte pattern.
            CHECK(stack_data->bytes[256] == std::byte{33});
        }
    }

    // A second GOB column: 40x65 is 10x17 blocks on a 128-byte stride.
    constexpr std::uint64_t wide_address = 0x24000;
    const auto wide_descriptor =
        texture_descriptor(wide_address, 40, 65, 1, 0x24, false);
    const TextureDescriptorResult wide_decode =
        decode_texture_descriptor(wide_descriptor);
    const TextureDescriptor* wide = accepted(wide_decode);
    CHECK(wide != nullptr);
    if (wide != nullptr) {
        CHECK(wide->row_bytes == 128 && wide->backing_extent == 4096);
        const std::size_t wide_offset =
            static_cast<std::size_t>(wide_address - Backing::gpu_base);
        const std::array<std::tuple<unsigned, unsigned, std::size_t>, 2>
            wide_placed{{{8, 0, 1024}, {9, 16, 3080}}};
        for (const auto& [column, row, swizzled] : wide_placed) {
            const std::size_t index = row * 10 + column;
            for (unsigned lane = 0; lane < 8; ++lane) {
                backing.bytes[wide_offset + swizzled + lane] =
                    static_cast<std::byte>((index + lane * 32 + 1) & 0xff);
            }
        }
        LinearTextureResult wide_read = read_linear_texture(*wide, memory);
        LinearTextureData* wide_data = std::get_if<LinearTextureData>(&wide_read);
        CHECK(wide_data != nullptr);
        if (wide_data != nullptr) {
            CHECK(wide_data->row_bytes == 80);
            bool placed_ok = true;
            for (const auto& [column, row, swizzled] : wide_placed) {
                const std::size_t index = row * 10 + column;
                for (unsigned lane = 0; lane < 8; ++lane) {
                    placed_ok = placed_ok &&
                        wide_data->bytes[index * 8 + lane] ==
                            std::byte((index + lane * 32 + 1) & 0xff);
                }
            }
            CHECK(placed_ok);
        }
    }

    // Reads outside the mapping and mid-backing faults are explicit errors.
    const auto unmapped = texture_descriptor(0x3f000, 12, 8, 0, 0x24, false);
    const TextureDescriptorResult unmapped_result =
        decode_texture_descriptor(unmapped);
    if (const TextureDescriptor* outside = accepted(unmapped_result)) {
        rejected(read_linear_texture(*outside, memory),
                 TextureErrorCode::MemoryRead);
    }
    {
        Backing faulty;
        GpuAddressSpace faulty_memory = make_memory(faulty);
        faulty.fail_at = Backing::guest_base + bc1_offset + 256;
        rejected(read_linear_texture(*bc1, faulty_memory),
                 TextureErrorCode::MemoryRead);
    }

    // Extent fields that no longer re-derive to the same layout are rejected
    // before any read or staging allocation.
    TextureDescriptor stretched = *bc1;
    stretched.levels = 2;
    rejected(read_linear_texture(stretched, memory),
             TextureErrorCode::UnsupportedTextureState);
    TextureDescriptor layered = *bc1;
    layered.layers = 2;
    rejected(read_linear_texture(layered, memory),
             TextureErrorCode::UnsupportedTextureState);
    TextureDescriptor deep = *bc1;
    deep.depth = 2;
    rejected(read_linear_texture(deep, memory),
             TextureErrorCode::UnsupportedTextureState);
    TextureDescriptor multisampled = *bc1;
    multisampled.samples = 2;
    rejected(read_linear_texture(multisampled, memory),
             TextureErrorCode::UnsupportedTextureState);
    TextureDescriptor empty = *bc1;
    empty.width = 0;
    rejected(read_linear_texture(empty, memory),
             TextureErrorCode::UnsupportedTextureState);
    TextureDescriptor reswizzled = *bc1;
    reswizzled.swizzle = {TextureSwizzle::Blue, TextureSwizzle::Green,
                          TextureSwizzle::Red, TextureSwizzle::Alpha};
    // Staging keeps guest channel order; the swizzle belongs to the view.
    CHECK(std::holds_alternative<LinearTextureData>(read_linear_texture(reswizzled, memory)));
    TextureDescriptor oversized = *bc1;
    oversized.backing_extent = max_texture_backing_size + 1;
    rejected(read_linear_texture(oversized, memory),
             TextureErrorCode::ExtentLimit);
    TextureDescriptor missing = *bc1;
    missing.backing_extent = 0;
    rejected(read_linear_texture(missing, memory),
             TextureErrorCode::ExtentLimit);

    // Every sampled format is capped per axis before anything is staged.
    for (const std::uint32_t code : {9u, 8u, 0x24u}) {
        const auto too_wide = texture_descriptor(0x28000, 65536, 4, 0, code, false);
        const TextureDescriptorResult wide_result =
            decode_texture_descriptor(too_wide);
        if (const TextureDescriptor* wide_texture = accepted(wide_result)) {
            const std::size_t before = backing.bytes_read;
            rejected(read_linear_texture(*wide_texture, memory),
                     TextureErrorCode::InvalidDimensions);
            CHECK(backing.bytes_read == before);
        }
        const auto too_tall = texture_descriptor(0x28000, 13, 65536, 0, code, false);
        const TextureDescriptorResult tall_axis =
            decode_texture_descriptor(too_tall);
        if (const TextureDescriptor* tall_texture = accepted(tall_axis)) {
            const std::size_t before = backing.bytes_read;
            rejected(read_linear_texture(*tall_texture, memory),
                     TextureErrorCode::InvalidDimensions);
            CHECK(backing.bytes_read == before);
        }
    }

    // Forged descriptors are rejected before any read or staging allocation.
    TextureDescriptor forged = *bc1;
    forged.width = 1u << 30;
    forged.height = 4;
    forged.row_bytes = 64;
    forged.backing_extent = 512;
    const std::size_t unread = backing.bytes_read;
    rejected(read_linear_texture(forged, memory),
             TextureErrorCode::InvalidDimensions);
    CHECK(backing.bytes_read == unread);
    // Formats whose level-0 shape matches the stored fields pass through the
    // layout layer: one-byte texels keep the same 64-byte GOB row and 512-byte
    // extent as the BC1 image above, and 12-byte texels are admitted through a
    // freshly decoded descriptor of their own.
    TextureDescriptor one_byte = *bc1;
    one_byte.format = TextureFormat::R8Unorm;
    const LinearTextureResult one_byte_read =
        read_linear_texture(one_byte, memory);
    CHECK(std::holds_alternative<LinearTextureData>(one_byte_read));
    const auto triple_bytes = with_all_component_types(
        texture_descriptor(0x2c000, 8, 4, 0, 0x02), 7);
    const TextureDescriptorResult triple_decode =
        decode_texture_descriptor(triple_bytes);
    const TextureDescriptor* triple_texture = accepted(triple_decode);
    CHECK(triple_texture != nullptr);
    if (triple_texture != nullptr) {
        CHECK(triple_texture->format == TextureFormat::R32G32B32Float);
        CHECK(triple_texture->row_bytes == 128);
        CHECK(triple_texture->backing_extent == 1024);
        const LinearTextureResult triple_read =
            read_linear_texture(*triple_texture, memory);
        CHECK(std::holds_alternative<LinearTextureData>(triple_read));
    }
    // Raw depth staging follows the same validated layout as color data.
    const auto depth_decode = decode_texture_descriptor(with_all_component_types(
        texture_descriptor(0x2c000, 8, 4, 0, 0x2f), 7));
    const auto* depth = accepted(depth_decode);
    CHECK(depth != nullptr);
    if (depth) CHECK(std::holds_alternative<LinearTextureData>(read_linear_texture(*depth, memory)));
    const std::size_t readable = backing.bytes_read;
    TextureDescriptor tall_log2 = *bc1;
    tall_log2.block_height_log2 = 6;
    rejected(read_linear_texture(tall_log2, memory),
             TextureErrorCode::UnsupportedLayout);
    // A 33-wide BC1 row needs a 128-byte stride; a 64-byte lie is rejected.
    TextureDescriptor lying_row = *bc1;
    lying_row.width = 33;
    lying_row.row_bytes = 64;
    lying_row.storage_height = 8;
    lying_row.backing_extent = 512;
    rejected(read_linear_texture(lying_row, memory),
             TextureErrorCode::UnsupportedTextureState);
    TextureDescriptor lying_extent = *bc1;
    lying_extent.backing_extent += 1;
    rejected(read_linear_texture(lying_extent, memory),
             TextureErrorCode::UnsupportedTextureState);
    TextureDescriptor lying_storage = *bc1;
    lying_storage.storage_height *= 2;
    rejected(read_linear_texture(lying_storage, memory),
             TextureErrorCode::UnsupportedTextureState);
    CHECK(backing.bytes_read == readable);
}

void test_format_table() {
    // Each family resolves to its table format with the defined storage
    // geometry: block extent, bytes per block, and component count.
    struct Entry {
        std::uint32_t word;
        bool srgb;
        TextureFormat format;
        std::uint8_t block_width;
        std::uint8_t block_height;
        std::uint8_t bytes_per_block;
        std::uint8_t components;
    };
    constexpr std::uint32_t u = 2, s = 1, ui = 4, si = 3, f = 7;
    const std::vector<Entry> entries = {
        // Single channel
        {tic_word(0x1d, u, u, u, u), false, TextureFormat::R8Unorm, 1, 1, 1, 1},
        {tic_word(0x1d, s, s, s, s), false, TextureFormat::R8Snorm, 1, 1, 1, 1},
        {tic_word(0x1d, ui, ui, ui, ui), false, TextureFormat::R8Uint, 1, 1, 1, 1},
        {tic_word(0x1d, si, si, si, si), false, TextureFormat::R8Sint, 1, 1, 1, 1},
        {tic_word(0x1b, u, u, u, u), false, TextureFormat::R16Unorm, 1, 1, 2, 1},
        {tic_word(0x1b, s, s, s, s), false, TextureFormat::R16Snorm, 1, 1, 2, 1},
        {tic_word(0x1b, ui, ui, ui, ui), false, TextureFormat::R16Uint, 1, 1, 2, 1},
        {tic_word(0x1b, si, si, si, si), false, TextureFormat::R16Sint, 1, 1, 2, 1},
        {tic_word(0x1b, f, f, f, f), false, TextureFormat::R16Float, 1, 1, 2, 1},
        {tic_word(0x0f, f, f, f, f), false, TextureFormat::R32Float, 1, 1, 4, 1},
        {tic_word(0x0f, ui, ui, ui, ui), false, TextureFormat::R32Uint, 1, 1, 4, 1},
        {tic_word(0x0f, si, si, si, si), false, TextureFormat::R32Sint, 1, 1, 4, 1},
        // Two channel
        {tic_word(0x18, u, u, u, u), false, TextureFormat::R8G8Unorm, 1, 1, 2, 2},
        {tic_word(0x18, s, s, s, s), false, TextureFormat::R8G8Snorm, 1, 1, 2, 2},
        {tic_word(0x18, ui, ui, ui, ui), false, TextureFormat::R8G8Uint, 1, 1, 2, 2},
        {tic_word(0x18, si, si, si, si), false, TextureFormat::R8G8Sint, 1, 1, 2, 2},
        {tic_word(0x0c, u, u, u, u), false, TextureFormat::R16G16Unorm, 1, 1, 4, 2},
        {tic_word(0x0c, s, s, s, s), false, TextureFormat::R16G16Snorm, 1, 1, 4, 2},
        {tic_word(0x0c, ui, ui, ui, ui), false, TextureFormat::R16G16Uint, 1, 1, 4, 2},
        {tic_word(0x0c, si, si, si, si), false, TextureFormat::R16G16Sint, 1, 1, 4, 2},
        {tic_word(0x0c, f, f, f, f), false, TextureFormat::R16G16Float, 1, 1, 4, 2},
        {tic_word(0x04, f, f, f, f), false, TextureFormat::R32G32Float, 1, 1, 8, 2},
        {tic_word(0x04, ui, ui, ui, ui), false, TextureFormat::R32G32Uint, 1, 1, 8, 2},
        {tic_word(0x04, si, si, si, si), false, TextureFormat::R32G32Sint, 1, 1, 8, 2},
        // Three channel
        {tic_word(0x02, f, f, f, f), false, TextureFormat::R32G32B32Float, 1, 1, 12, 3},
        {tic_word(0x02, ui, ui, ui, ui), false, TextureFormat::R32G32B32Uint, 1, 1, 12, 3},
        {tic_word(0x02, si, si, si, si), false, TextureFormat::R32G32B32Sint, 1, 1, 12, 3},
        // Four channel
        {tic_word(0x08, u, u, u, u), false, TextureFormat::R8G8B8A8Unorm, 1, 1, 4, 4},
        {tic_word(0x08, u, u, u, u), true, TextureFormat::R8G8B8A8Srgb, 1, 1, 4, 4},
        {tic_word(0x08, s, s, s, s), false, TextureFormat::R8G8B8A8Snorm, 1, 1, 4, 4},
        {tic_word(0x08, ui, ui, ui, ui), false, TextureFormat::R8G8B8A8Uint, 1, 1, 4, 4},
        {tic_word(0x08, si, si, si, si), false, TextureFormat::R8G8B8A8Sint, 1, 1, 4, 4},
        {tic_word(0x03, u, u, u, u), false, TextureFormat::R16G16B16A16Unorm, 1, 1, 8, 4},
        {tic_word(0x03, s, s, s, s), false, TextureFormat::R16G16B16A16Snorm, 1, 1, 8, 4},
        {tic_word(0x03, ui, ui, ui, ui), false, TextureFormat::R16G16B16A16Uint, 1, 1, 8, 4},
        {tic_word(0x03, si, si, si, si), false, TextureFormat::R16G16B16A16Sint, 1, 1, 8, 4},
        {tic_word(0x03, f, f, f, f), false, TextureFormat::R16G16B16A16Float, 1, 1, 8, 4},
        {tic_word(0x01, f, f, f, f), false, TextureFormat::R32G32B32A32Float, 1, 1, 16, 4},
        {tic_word(0x01, ui, ui, ui, ui), false, TextureFormat::R32G32B32A32Uint, 1, 1, 16, 4},
        {tic_word(0x01, si, si, si, si), false, TextureFormat::R32G32B32A32Sint, 1, 1, 16, 4},
        // Depth and stencil layouts
        {tic_word(0x3a, u, u, u, u), false, TextureFormat::D16Unorm, 1, 1, 2, 1},
        {tic_word(0x3a, u, ui, ui, ui), false, TextureFormat::D16Unorm, 1, 1, 2, 1},
        {tic_word(0x2f, f, f, f, f), false, TextureFormat::D32Float, 1, 1, 4, 1},
        {tic_word(0x2f, f, ui, ui, ui), false, TextureFormat::D32Float, 1, 1, 4, 1},
        {tic_word(0x0e, ui, u, u, u), false, TextureFormat::D24UnormS8Uint, 1, 1, 4, 2},
        {tic_word(0x29, ui, u, u, u), false, TextureFormat::D24UnormS8Uint, 1, 1, 4, 2},
        {tic_word(0x29, ui, u, ui, ui), false, TextureFormat::D24UnormS8Uint, 1, 1, 4, 2},
        {tic_word(0x2a, u, ui, ui, ui), false, TextureFormat::X8UintD24Unorm, 1, 1, 4, 2},
        {tic_word(0x2b, u, ui, ui, ui), false, TextureFormat::S8UintD24Unorm, 1, 1, 4, 2},
        {tic_word(0x05, f, ui, u, u), false, TextureFormat::D32FloatS8Uint, 1, 1, 8, 2},
        {tic_word(0x30, f, ui, u, u), false, TextureFormat::D32FloatS8Uint, 1, 1, 8, 2},
        // Packed small formats
        {tic_word(0x1e, u, u, u, u), false, TextureFormat::R4G4Unorm, 1, 1, 1, 2},
        {tic_word(0x12, u, u, u, u), false, TextureFormat::R4G4B4A4Unorm, 1, 1, 2, 4},
        {tic_word(0x14, u, u, u, u), false, TextureFormat::R5G5B5A1Unorm, 1, 1, 2, 4},
        {tic_word(0x13, u, u, u, u), false, TextureFormat::A5B5G5R1Unorm, 1, 1, 2, 4},
        {tic_word(0x15, u, u, u, u), false, TextureFormat::R5G6B5Unorm, 1, 1, 2, 3},
        {tic_word(0x09, u, u, u, u), false, TextureFormat::R10G10B10A2Unorm, 1, 1, 4, 4},
        {tic_word(0x09, ui, ui, ui, ui), false, TextureFormat::R10G10B10A2Uint, 1, 1, 4, 4},
        {tic_word(0x21, f, f, f, f), false, TextureFormat::R11G11B10Float, 1, 1, 4, 3},
        {tic_word(0x20, f, f, f, f), false, TextureFormat::R9G9B9E5Float, 1, 1, 4, 4},
        // BC1-7 in every defined encoding
        {tic_word(0x24, u, u, u, u), false, TextureFormat::Bc1RgbaUnorm, 4, 4, 8, 4},
        {tic_word(0x24, u, u, u, u), true, TextureFormat::Bc1RgbaSrgb, 4, 4, 8, 4},
        {tic_word(0x25, u, u, u, u), false, TextureFormat::Bc2Unorm, 4, 4, 16, 4},
        {tic_word(0x25, u, u, u, u), true, TextureFormat::Bc2Srgb, 4, 4, 16, 4},
        {tic_word(0x26, u, u, u, u), false, TextureFormat::Bc3Unorm, 4, 4, 16, 4},
        {tic_word(0x26, u, u, u, u), true, TextureFormat::Bc3Srgb, 4, 4, 16, 4},
        {tic_word(0x27, u, u, u, u), false, TextureFormat::Bc4Unorm, 4, 4, 8, 1},
        {tic_word(0x27, s, s, s, s), false, TextureFormat::Bc4Snorm, 4, 4, 8, 1},
        {tic_word(0x28, u, u, u, u), false, TextureFormat::Bc5Unorm, 4, 4, 16, 2},
        {tic_word(0x28, s, s, s, s), false, TextureFormat::Bc5Snorm, 4, 4, 16, 2},
        {tic_word(0x17, u, u, u, u), false, TextureFormat::Bc7Unorm, 4, 4, 16, 4},
        {tic_word(0x17, u, u, u, u), true, TextureFormat::Bc7Srgb, 4, 4, 16, 4},
        {tic_word(0x10, f, f, f, f), false, TextureFormat::Bc6HSfloat, 4, 4, 16, 4},
        {tic_word(0x11, f, f, f, f), false, TextureFormat::Bc6HUfloat, 4, 4, 16, 4},
        // ETC2 in both encodings
        {tic_word(0x06, u, u, u, u), false, TextureFormat::Etc2RgbUnorm, 4, 4, 8, 3},
        {tic_word(0x06, u, u, u, u), true, TextureFormat::Etc2RgbSrgb, 4, 4, 8, 3},
        {tic_word(0x0a, u, u, u, u), false, TextureFormat::Etc2RgbPtaUnorm, 4, 4, 8, 4},
        {tic_word(0x0a, u, u, u, u), true, TextureFormat::Etc2RgbPtaSrgb, 4, 4, 8, 4},
        {tic_word(0x0b, u, u, u, u), false, TextureFormat::Etc2RgbaUnorm, 4, 4, 16, 4},
        {tic_word(0x0b, u, u, u, u), true, TextureFormat::Etc2RgbaSrgb, 4, 4, 16, 4},
        // ASTC 2D block sizes in both encodings
        {tic_word(0x40, u, u, u, u), false, TextureFormat::Astc4x4Unorm, 4, 4, 16, 4},
        {tic_word(0x40, u, u, u, u), true, TextureFormat::Astc4x4Srgb, 4, 4, 16, 4},
        {tic_word(0x50, u, u, u, u), false, TextureFormat::Astc5x4Unorm, 5, 4, 16, 4},
        {tic_word(0x50, u, u, u, u), true, TextureFormat::Astc5x4Srgb, 5, 4, 16, 4},
        {tic_word(0x41, u, u, u, u), false, TextureFormat::Astc5x5Unorm, 5, 5, 16, 4},
        {tic_word(0x41, u, u, u, u), true, TextureFormat::Astc5x5Srgb, 5, 5, 16, 4},
        {tic_word(0x51, u, u, u, u), false, TextureFormat::Astc6x5Unorm, 6, 5, 16, 4},
        {tic_word(0x51, u, u, u, u), true, TextureFormat::Astc6x5Srgb, 6, 5, 16, 4},
        {tic_word(0x42, u, u, u, u), false, TextureFormat::Astc6x6Unorm, 6, 6, 16, 4},
        {tic_word(0x42, u, u, u, u), true, TextureFormat::Astc6x6Srgb, 6, 6, 16, 4},
        {tic_word(0x55, u, u, u, u), false, TextureFormat::Astc8x5Unorm, 8, 5, 16, 4},
        {tic_word(0x55, u, u, u, u), true, TextureFormat::Astc8x5Srgb, 8, 5, 16, 4},
        {tic_word(0x52, u, u, u, u), false, TextureFormat::Astc8x6Unorm, 8, 6, 16, 4},
        {tic_word(0x52, u, u, u, u), true, TextureFormat::Astc8x6Srgb, 8, 6, 16, 4},
        {tic_word(0x44, u, u, u, u), false, TextureFormat::Astc8x8Unorm, 8, 8, 16, 4},
        {tic_word(0x44, u, u, u, u), true, TextureFormat::Astc8x8Srgb, 8, 8, 16, 4},
        {tic_word(0x56, u, u, u, u), false, TextureFormat::Astc10x5Unorm, 10, 5, 16, 4},
        {tic_word(0x56, u, u, u, u), true, TextureFormat::Astc10x5Srgb, 10, 5, 16, 4},
        {tic_word(0x57, u, u, u, u), false, TextureFormat::Astc10x6Unorm, 10, 6, 16, 4},
        {tic_word(0x57, u, u, u, u), true, TextureFormat::Astc10x6Srgb, 10, 6, 16, 4},
        {tic_word(0x53, u, u, u, u), false, TextureFormat::Astc10x8Unorm, 10, 8, 16, 4},
        {tic_word(0x53, u, u, u, u), true, TextureFormat::Astc10x8Srgb, 10, 8, 16, 4},
        {tic_word(0x45, u, u, u, u), false, TextureFormat::Astc10x10Unorm, 10, 10, 16, 4},
        {tic_word(0x45, u, u, u, u), true, TextureFormat::Astc10x10Srgb, 10, 10, 16, 4},
        {tic_word(0x54, u, u, u, u), false, TextureFormat::Astc12x10Unorm, 12, 10, 16, 4},
        {tic_word(0x54, u, u, u, u), true, TextureFormat::Astc12x10Srgb, 12, 10, 16, 4},
        {tic_word(0x46, u, u, u, u), false, TextureFormat::Astc12x12Unorm, 12, 12, 16, 4},
        {tic_word(0x46, u, u, u, u), true, TextureFormat::Astc12x12Srgb, 12, 12, 16, 4},
    };
    for (const Entry& entry : entries) {
        TextureFormat resolved{};
        CHECK(lookup_texture_format(entry.word, entry.srgb, resolved));
        CHECK(resolved == entry.format);
        const TextureFormatInfo info = texture_format_info(entry.format);
        CHECK(info.block_width == entry.block_width);
        CHECK(info.block_height == entry.block_height);
        CHECK(info.bytes_per_block == entry.bytes_per_block);
        CHECK(info.components == entry.components);
        CHECK(is_block_compressed(info) ==
              ((entry.block_width | entry.block_height) != 1));
    }

    // Depth and stencil flags distinguish the combined layouts.
    CHECK(texture_format_info(TextureFormat::D16Unorm).depth);
    CHECK(!texture_format_info(TextureFormat::D16Unorm).stencil);
    CHECK(texture_format_info(TextureFormat::D32Float).depth);
    CHECK(!texture_format_info(TextureFormat::D32Float).stencil);
    CHECK(texture_format_info(TextureFormat::D24UnormS8Uint).depth);
    CHECK(texture_format_info(TextureFormat::D24UnormS8Uint).stencil);
    CHECK(texture_format_info(TextureFormat::X8UintD24Unorm).depth);
    CHECK(!texture_format_info(TextureFormat::X8UintD24Unorm).stencil);
    CHECK(texture_format_info(TextureFormat::S8UintD24Unorm).stencil);
    CHECK(texture_format_info(TextureFormat::D32FloatS8Uint).depth &&
          texture_format_info(TextureFormat::D32FloatS8Uint).stencil);

    // Words without a table entry, sRGB on single-encoding formats, mixed
    // component types, and the reserved type encodings stay unresolved.
    TextureFormat resolved{};
    for (const std::uint32_t word : {0x07u, 0x16u, 0x1cu, 0x22u, 0x23u,
                                     0x7fu}) {
        CHECK(!lookup_texture_format(word, false, resolved));
        CHECK(!lookup_texture_format(word, true, resolved));
    }
    CHECK(!lookup_texture_format(tic_word(0x0f, f, f, f, f), true, resolved));
    CHECK(!lookup_texture_format(tic_word(0x09, u, u, u, u), true, resolved));
    CHECK(!lookup_texture_format(tic_word(0x08, u, s, s, s), false, resolved));
    CHECK(!lookup_texture_format(tic_word(0x1d, 0, 0, 0, 0), false, resolved));
    CHECK(!lookup_texture_format(tic_word(0x1d, 5, 5, 5, 5), false, resolved));
    CHECK(!lookup_texture_format(tic_word(0x1d, 6, 6, 6, 6), false, resolved));
}

}  // namespace

int main() {
    test_depth_alias_storage_compatibility();
    test_texture_decode();
    test_rgba8_texture_formats();
    test_bc1_texture_formats();
    test_format_table();
    test_texture_rejections();
    test_texture_type_decode();
    test_mip_level_decode();
    test_pitch_layout_decode();
    test_sampler_decode_and_rejections();
    test_resolver_and_memory_validation();
    test_descriptor_only_resolution();
    test_independent_resource_provenance();
    test_resolver_null_texture_transition();
    test_resolver_rejections();
    test_read_texture_views();
    test_3d_mip_layout_fidelity();
    test_read_linear_texture();
    std::printf("Texture descriptor failures: %u\n", failures);
    return failures == 0 ? 0 : 1;
}
