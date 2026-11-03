// ============================================================================
// Armigo Tests - NVN Replay State Objects
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "commands.h"
#include "entry_table.h"

#include <cstdio>
#include <cstring>

using namespace armigo::nvn;
namespace {

unsigned failures = 0;
#define CHECK(value)                                                                        \
    do {                                                                                    \
        if (!(value)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #value); ++failures; } \
    } while (false)

template <Entry E>
auto api_function() {
    return reinterpret_cast<typename EntrySignature<E>::type>(armigo_nvn_bootstrap_loader(entry_name(E).data()));
}

// Every state object entry point must resolve after registration.
constexpr Entry state_entries[] = {
    Entry::nvnBlendStateSetDefaults, Entry::nvnBlendStateSetBlendTarget, Entry::nvnBlendStateSetBlendFunc,
    Entry::nvnBlendStateSetBlendEquation, Entry::nvnBlendStateSetAdvancedMode, Entry::nvnBlendStateSetAdvancedOverlap,
    Entry::nvnBlendStateSetAdvancedPremultipliedSrc, Entry::nvnBlendStateSetAdvancedNormalizedDst,
    Entry::nvnBlendStateGetBlendTarget, Entry::nvnBlendStateGetBlendFunc, Entry::nvnBlendStateGetBlendEquation,
    Entry::nvnBlendStateGetAdvancedMode, Entry::nvnBlendStateGetAdvancedOverlap,
    Entry::nvnBlendStateGetAdvancedPremultipliedSrc, Entry::nvnBlendStateGetAdvancedNormalizedDst,
    Entry::nvnColorStateSetDefaults, Entry::nvnColorStateSetBlendEnable, Entry::nvnColorStateSetLogicOp,
    Entry::nvnColorStateSetAlphaTest, Entry::nvnColorStateGetBlendEnable, Entry::nvnColorStateGetLogicOp,
    Entry::nvnColorStateGetAlphaTest,
    Entry::nvnChannelMaskStateSetDefaults, Entry::nvnChannelMaskStateSetChannelMask,
    Entry::nvnChannelMaskStateGetChannelMask,
    Entry::nvnMultisampleStateSetDefaults, Entry::nvnMultisampleStateSetMultisampleEnable,
    Entry::nvnMultisampleStateSetSamples, Entry::nvnMultisampleStateSetAlphaToCoverageEnable,
    Entry::nvnMultisampleStateSetAlphaToCoverageDither, Entry::nvnMultisampleStateSetRasterSamples,
    Entry::nvnMultisampleStateSetCoverageModulationMode, Entry::nvnMultisampleStateSetCoverageToColorEnable,
    Entry::nvnMultisampleStateSetCoverageToColorOutput, Entry::nvnMultisampleStateSetSampleLocationsEnable,
    Entry::nvnMultisampleStateSetSampleLocationsGridEnable, Entry::nvnMultisampleStateSetSampleLocations,
    Entry::nvnMultisampleStateGetMultisampleEnable, Entry::nvnMultisampleStateGetSamples,
    Entry::nvnMultisampleStateGetAlphaToCoverageEnable, Entry::nvnMultisampleStateGetAlphaToCoverageDither,
    Entry::nvnMultisampleStateGetRasterSamples, Entry::nvnMultisampleStateGetCoverageModulationMode,
    Entry::nvnMultisampleStateGetCoverageToColorEnable, Entry::nvnMultisampleStateGetCoverageToColorOutput,
    Entry::nvnMultisampleStateGetSampleLocationsEnable, Entry::nvnMultisampleStateGetSampleLocationsGrid,
    Entry::nvnMultisampleStateGetSampleLocationsGridEnable,
    Entry::nvnPolygonStateSetDefaults, Entry::nvnPolygonStateSetCullFace, Entry::nvnPolygonStateSetFrontFace,
    Entry::nvnPolygonStateSetPolygonMode, Entry::nvnPolygonStateSetPolygonOffsetEnables,
    Entry::nvnPolygonStateGetCullFace, Entry::nvnPolygonStateGetFrontFace, Entry::nvnPolygonStateGetPolygonMode,
    Entry::nvnPolygonStateGetPolygonOffsetEnables,
    Entry::nvnDepthStencilStateSetDefaults, Entry::nvnDepthStencilStateSetDepthTestEnable,
    Entry::nvnDepthStencilStateSetDepthWriteEnable, Entry::nvnDepthStencilStateSetDepthFunc,
    Entry::nvnDepthStencilStateSetStencilTestEnable, Entry::nvnDepthStencilStateSetStencilFunc,
    Entry::nvnDepthStencilStateSetStencilOp, Entry::nvnDepthStencilStateGetDepthTestEnable,
    Entry::nvnDepthStencilStateGetDepthWriteEnable, Entry::nvnDepthStencilStateGetDepthFunc,
    Entry::nvnDepthStencilStateGetStencilTestEnable, Entry::nvnDepthStencilStateGetStencilFunc,
    Entry::nvnDepthStencilStateGetStencilOp,
    Entry::nvnVertexAttribStateSetDefaults, Entry::nvnVertexAttribStateSetFormat,
    Entry::nvnVertexAttribStateSetStreamIndex, Entry::nvnVertexAttribStateGetFormat,
    Entry::nvnVertexAttribStateGetStreamIndex,
    Entry::nvnVertexStreamStateSetDefaults, Entry::nvnVertexStreamStateSetStride,
    Entry::nvnVertexStreamStateSetDivisor, Entry::nvnVertexStreamStateGetStride,
    Entry::nvnVertexStreamStateGetDivisor,
};

} // namespace

int main() {
    select_api_version({55, 15});
    register_state_entries();
    for (const Entry entry : state_entries) {
        const auto function = armigo_nvn_bootstrap_loader(entry_name(entry).data());
        CHECK(function != nullptr);
        // A registered entry must not share the address of a stub.
        CHECK(function != armigo_nvn_bootstrap_loader("nvnQueueFinish"));
    }

    // ------------------------------------------------------------- blend
    const auto blend_defaults = api_function<Entry::nvnBlendStateSetDefaults>();
    const auto blend_target_set = api_function<Entry::nvnBlendStateSetBlendTarget>();
    const auto blend_func_set = api_function<Entry::nvnBlendStateSetBlendFunc>();
    const auto blend_equation_set = api_function<Entry::nvnBlendStateSetBlendEquation>();
    const auto blend_mode_set = api_function<Entry::nvnBlendStateSetAdvancedMode>();
    const auto blend_overlap_set = api_function<Entry::nvnBlendStateSetAdvancedOverlap>();
    const auto blend_premultiplied_set = api_function<Entry::nvnBlendStateSetAdvancedPremultipliedSrc>();
    const auto blend_normalized_set = api_function<Entry::nvnBlendStateSetAdvancedNormalizedDst>();
    const auto blend_target_get = api_function<Entry::nvnBlendStateGetBlendTarget>();
    const auto blend_func_get = api_function<Entry::nvnBlendStateGetBlendFunc>();
    const auto blend_equation_get = api_function<Entry::nvnBlendStateGetBlendEquation>();
    const auto blend_mode_get = api_function<Entry::nvnBlendStateGetAdvancedMode>();
    const auto blend_overlap_get = api_function<Entry::nvnBlendStateGetAdvancedOverlap>();
    const auto blend_premultiplied_get = api_function<Entry::nvnBlendStateGetAdvancedPremultipliedSrc>();
    const auto blend_normalized_get = api_function<Entry::nvnBlendStateGetAdvancedNormalizedDst>();

    NVNblendState blend{};
    std::memset(&blend, 0xa5, sizeof(blend));  // defaults must overwrite garbage
    blend_defaults(&blend);
    CHECK(blend_target_get(&blend) == 0);
    NVNblendFunc sr = NVN_BLEND_FUNC_ZERO, dr = NVN_BLEND_FUNC_ZERO, sa = NVN_BLEND_FUNC_ZERO,
                 da = NVN_BLEND_FUNC_ZERO;
    blend_func_get(&blend, &sr, &dr, &sa, &da);
    CHECK(sr == NVN_BLEND_FUNC_ONE && dr == NVN_BLEND_FUNC_ZERO && sa == NVN_BLEND_FUNC_ONE &&
          da == NVN_BLEND_FUNC_ZERO);
    NVNblendEquation er = NVN_BLEND_EQUATION_MIN, ea = NVN_BLEND_EQUATION_MAX;
    blend_equation_get(&blend, &er, &ea);
    CHECK(er == NVN_BLEND_EQUATION_ADD && ea == NVN_BLEND_EQUATION_ADD);
    CHECK(blend_mode_get(&blend) == NVN_BLEND_ADVANCED_MODE_NONE);
    CHECK(blend_overlap_get(&blend) == NVN_BLEND_ADVANCED_OVERLAP_UNCORRELATED);
    CHECK(!blend_premultiplied_get(&blend));
    CHECK(!blend_normalized_get(&blend));

    // Roundtrip of every field, including the widest enumeration values that
    // fit the packed widths.
    blend_target_set(&blend, 5);
    CHECK(blend_target_get(&blend) == 5);
    blend_func_set(&blend, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA,
                   NVN_BLEND_FUNC_SRC1_COLOR, NVN_BLEND_FUNC_SRC_ALPHA_SATURATE);
    blend_func_get(&blend, &sr, &dr, &sa, &da);
    CHECK(sr == NVN_BLEND_FUNC_ZERO && dr == NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA &&
          sa == NVN_BLEND_FUNC_SRC1_COLOR && da == NVN_BLEND_FUNC_SRC_ALPHA_SATURATE);
    blend_equation_set(&blend, NVN_BLEND_EQUATION_MIN, NVN_BLEND_EQUATION_MAX);
    blend_equation_get(&blend, &er, &ea);
    CHECK(er == NVN_BLEND_EQUATION_MIN && ea == NVN_BLEND_EQUATION_MAX);
    blend_mode_set(&blend, NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY);
    blend_overlap_set(&blend, NVN_BLEND_ADVANCED_OVERLAP_CONJOINT);
    blend_premultiplied_set(&blend, 1);
    blend_normalized_set(&blend, 0xff);
    CHECK(blend_mode_get(&blend) == NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY);
    CHECK(blend_overlap_get(&blend) == NVN_BLEND_ADVANCED_OVERLAP_CONJOINT);
    CHECK(blend_premultiplied_get(&blend));
    CHECK(blend_normalized_get(&blend));

    // The packed basic fields must decode identically through the layout the
    // command records keep, so replays read what the setters stored.
    BlendStateBits recorded{};
    std::memcpy(&recorded, &blend, sizeof(recorded));
    CHECK(recorded.target == 5);
    CHECK(recorded.source_color == static_cast<std::uint32_t>(NVN_BLEND_FUNC_ZERO));
    CHECK(recorded.destination_color == static_cast<std::uint32_t>(NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA));
    CHECK(recorded.source_alpha == static_cast<std::uint32_t>(NVN_BLEND_FUNC_SRC1_COLOR));
    CHECK(recorded.destination_alpha == static_cast<std::uint32_t>(NVN_BLEND_FUNC_SRC_ALPHA_SATURATE));
    CHECK(recorded.color_equation == static_cast<std::uint32_t>(NVN_BLEND_EQUATION_MIN));
    CHECK(recorded.alpha_equation == static_cast<std::uint32_t>(NVN_BLEND_EQUATION_MAX));

    // Games copy state objects by value; the copy carries the whole state.
    const NVNblendState blend_copy = blend;
    CHECK(blend_target_get(&blend_copy) == 5);
    CHECK(blend_mode_get(&blend_copy) == NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY);
    CHECK(blend_overlap_get(&blend_copy) == NVN_BLEND_ADVANCED_OVERLAP_CONJOINT);
    CHECK(blend_premultiplied_get(&blend_copy));
    CHECK(blend_normalized_get(&blend_copy));

    // Booleans are 32-bit registers with only the low 8 bits defined.
    blend_premultiplied_set(&blend, 0x100);
    blend_normalized_set(&blend, 0x100);
    CHECK(!blend_premultiplied_get(&blend));
    CHECK(!blend_normalized_get(&blend));
    blend_premultiplied_set(&blend, 0x101);
    blend_normalized_set(&blend, 2);
    CHECK(blend_premultiplied_get(&blend));
    CHECK(blend_normalized_get(&blend));

    // A target index packs into three bits.
    blend_target_set(&blend, 9);
    CHECK(blend_target_get(&blend) == 1);
    blend_target_set(&blend, 7);
    CHECK(blend_target_get(&blend) == 7);

    // Null out pointers are ignored; a null object reads as defaults.
    blend_func_get(&blend, nullptr, nullptr, nullptr, nullptr);
    blend_equation_get(&blend, nullptr, nullptr);
    CHECK(blend_target_get(nullptr) == 0);
    CHECK(blend_mode_get(nullptr) == NVN_BLEND_ADVANCED_MODE_NONE);
    CHECK(!blend_premultiplied_get(nullptr));
    blend_target_set(nullptr, 3);

    // ------------------------------------------------------------- color
    const auto color_defaults = api_function<Entry::nvnColorStateSetDefaults>();
    const auto color_blend_set = api_function<Entry::nvnColorStateSetBlendEnable>();
    const auto color_logic_set = api_function<Entry::nvnColorStateSetLogicOp>();
    const auto color_alpha_set = api_function<Entry::nvnColorStateSetAlphaTest>();
    const auto color_blend_get = api_function<Entry::nvnColorStateGetBlendEnable>();
    const auto color_logic_get = api_function<Entry::nvnColorStateGetLogicOp>();
    const auto color_alpha_get = api_function<Entry::nvnColorStateGetAlphaTest>();

    NVNcolorState color{};
    std::memset(&color, 0xff, sizeof(color));
    color_defaults(&color);
    for (int i = 0; i < 8; ++i) CHECK(!color_blend_get(&color, i));
    CHECK(color_logic_get(&color) == NVN_LOGIC_OP_COPY);
    CHECK(color_alpha_get(&color) == NVN_ALPHA_FUNC_ALWAYS);

    // Per-target enables are independent and survive each other.
    color_blend_set(&color, 1, 1);
    color_blend_set(&color, 4, 0xff);
    color_blend_set(&color, 7, 1);
    for (int i = 0; i < 8; ++i)
        CHECK(color_blend_get(&color, i) == (i == 1 || i == 4 || i == 7));
    color_blend_set(&color, 4, 0);
    CHECK(!color_blend_get(&color, 4));
    CHECK(color_blend_get(&color, 1) && color_blend_get(&color, 7));

    color_logic_set(&color, NVN_LOGIC_OP_XOR);
    color_alpha_set(&color, NVN_ALPHA_FUNC_LESS);
    CHECK(color_logic_get(&color) == NVN_LOGIC_OP_XOR);
    CHECK(color_alpha_get(&color) == NVN_ALPHA_FUNC_LESS);

    // Low-byte booleans: 0x100 is false, 0x101 is true.
    color_blend_set(&color, 2, 0x100);
    CHECK(!color_blend_get(&color, 2));
    color_blend_set(&color, 2, 0x101);
    CHECK(color_blend_get(&color, 2));

    // Out-of-range target indices are ignored by the setter and read false.
    color_blend_set(&color, 8, 1);
    color_blend_set(&color, -1, 1);
    CHECK(!color_blend_get(&color, 8));
    CHECK(!color_blend_get(&color, -1));
    CHECK(!color_blend_get(nullptr, 0));
    color_blend_set(nullptr, 0, 1);
    color_logic_set(nullptr, NVN_LOGIC_OP_AND);
    CHECK(color_logic_get(nullptr) == NVN_LOGIC_OP_CLEAR);
    CHECK(color_alpha_get(nullptr) == NVN_ALPHA_FUNC_ALWAYS);

    // ------------------------------------------------------------- channel mask
    const auto mask_defaults = api_function<Entry::nvnChannelMaskStateSetDefaults>();
    const auto mask_set = api_function<Entry::nvnChannelMaskStateSetChannelMask>();
    const auto mask_get = api_function<Entry::nvnChannelMaskStateGetChannelMask>();

    NVNchannelMaskState mask{};
    std::memset(&mask, 0, sizeof(mask));
    mask_defaults(&mask);
    for (int i = 0; i < 8; ++i) {
        NVNboolean r = 0, g = 0, b = 0, a = 0;
        mask_get(&mask, i, &r, &g, &b, &a);
        CHECK(r && g && b && a);
    }

    // Writes touch only their own target's nibble.
    mask_set(&mask, 0, 1, 0, 1, 0);
    mask_set(&mask, 3, 0, 1, 0, 1);
    mask_set(&mask, 7, 1, 1, 0, 0);
    NVNboolean r = 1, g = 1, b = 1, a = 1;
    mask_get(&mask, 0, &r, &g, &b, &a);
    CHECK(r && !g && b && !a);
    mask_get(&mask, 1, &r, &g, &b, &a);
    CHECK(r && g && b && a);
    mask_get(&mask, 3, &r, &g, &b, &a);
    CHECK(!r && g && !b && a);
    mask_get(&mask, 7, &r, &g, &b, &a);
    CHECK(r && g && !b && !a);

    // Low-byte booleans per channel.
    mask_set(&mask, 2, 0x100, 0x101, 0, 0xff);
    mask_get(&mask, 2, &r, &g, &b, &a);
    CHECK(!r && g && !b && a);

    // Out-of-range indices leave the state unchanged and read all false.
    mask_set(&mask, 8, 1, 1, 1, 1);
    mask_set(&mask, -2, 1, 1, 1, 1);
    mask_get(&mask, 1, &r, &g, &b, &a);
    CHECK(r && g && b && a);
    mask_get(&mask, 8, &r, &g, &b, &a);
    CHECK(!r && !g && !b && !a);
    mask_get(&mask, -1, &r, &g, &b, &a);
    CHECK(!r && !g && !b && !a);
    mask_get(&mask, 0, nullptr, nullptr, nullptr, nullptr);
    mask_get(nullptr, 0, &r, &g, &b, &a);
    CHECK(!r && !g && !b && !a);

    // ------------------------------------------------------------- multisample
    const auto multisample_defaults = api_function<Entry::nvnMultisampleStateSetDefaults>();
    const auto multisample_enable_set = api_function<Entry::nvnMultisampleStateSetMultisampleEnable>();
    const auto multisample_samples_set = api_function<Entry::nvnMultisampleStateSetSamples>();
    const auto multisample_a2c_set = api_function<Entry::nvnMultisampleStateSetAlphaToCoverageEnable>();
    const auto multisample_dither_set = api_function<Entry::nvnMultisampleStateSetAlphaToCoverageDither>();
    const auto multisample_raster_set = api_function<Entry::nvnMultisampleStateSetRasterSamples>();
    const auto multisample_modulation_set = api_function<Entry::nvnMultisampleStateSetCoverageModulationMode>();
    const auto multisample_c2c_set = api_function<Entry::nvnMultisampleStateSetCoverageToColorEnable>();
    const auto multisample_c2c_output_set = api_function<Entry::nvnMultisampleStateSetCoverageToColorOutput>();
    const auto multisample_locations_set = api_function<Entry::nvnMultisampleStateSetSampleLocationsEnable>();
    const auto multisample_grid_set = api_function<Entry::nvnMultisampleStateSetSampleLocationsGridEnable>();
    const auto multisample_locations_table_set = api_function<Entry::nvnMultisampleStateSetSampleLocations>();
    const auto multisample_enable_get = api_function<Entry::nvnMultisampleStateGetMultisampleEnable>();
    const auto multisample_samples_get = api_function<Entry::nvnMultisampleStateGetSamples>();
    const auto multisample_a2c_get = api_function<Entry::nvnMultisampleStateGetAlphaToCoverageEnable>();
    const auto multisample_dither_get = api_function<Entry::nvnMultisampleStateGetAlphaToCoverageDither>();
    const auto multisample_raster_get = api_function<Entry::nvnMultisampleStateGetRasterSamples>();
    const auto multisample_modulation_get = api_function<Entry::nvnMultisampleStateGetCoverageModulationMode>();
    const auto multisample_c2c_get = api_function<Entry::nvnMultisampleStateGetCoverageToColorEnable>();
    const auto multisample_c2c_output_get = api_function<Entry::nvnMultisampleStateGetCoverageToColorOutput>();
    const auto multisample_locations_get = api_function<Entry::nvnMultisampleStateGetSampleLocationsEnable>();
    const auto multisample_grid_get = api_function<Entry::nvnMultisampleStateGetSampleLocationsGrid>();
    const auto multisample_grid_enable_get = api_function<Entry::nvnMultisampleStateGetSampleLocationsGridEnable>();

    NVNmultisampleState multisample{};
    std::memset(&multisample, 0xff, sizeof(multisample));
    multisample_defaults(&multisample);
    CHECK(multisample_enable_get(&multisample));
    CHECK(multisample_samples_get(&multisample) == 0);
    CHECK(!multisample_a2c_get(&multisample));
    CHECK(!multisample_dither_get(&multisample));
    CHECK(multisample_raster_get(&multisample) == 0);
    CHECK(multisample_modulation_get(&multisample) == NVN_COVERAGE_MODULATION_MODE_NONE);
    CHECK(!multisample_c2c_get(&multisample));
    CHECK(multisample_c2c_output_get(&multisample) == 0);
    CHECK(!multisample_locations_get(&multisample));
    CHECK(!multisample_grid_enable_get(&multisample));
    int grid_w = -1, grid_h = -1;
    multisample_grid_get(&multisample, &grid_w, &grid_h);
    CHECK(grid_w == 1 && grid_h == 1);

    multisample_samples_set(&multisample, 4);
    CHECK(multisample_samples_get(&multisample) == 4);
    multisample_grid_get(&multisample, &grid_w, &grid_h);
    CHECK(grid_w == 2 && grid_h == 2);
    multisample_raster_set(&multisample, 8);
    multisample_modulation_set(&multisample, NVN_COVERAGE_MODULATION_MODE_RGBA);
    multisample_c2c_set(&multisample, 1);
    multisample_c2c_output_set(&multisample, 5);
    multisample_locations_set(&multisample, 1);
    multisample_grid_set(&multisample, 1);
    CHECK(multisample_raster_get(&multisample) == 8);
    CHECK(multisample_modulation_get(&multisample) == NVN_COVERAGE_MODULATION_MODE_RGBA);
    CHECK(multisample_c2c_get(&multisample));
    CHECK(multisample_c2c_output_get(&multisample) == 5);
    CHECK(multisample_locations_get(&multisample));
    CHECK(multisample_grid_enable_get(&multisample));

    // The sample-location grid follows the sample count.
    multisample_samples_set(&multisample, 2);
    multisample_grid_get(&multisample, &grid_w, &grid_h);
    CHECK(grid_w == 2 && grid_h == 1);
    multisample_samples_set(&multisample, 8);
    multisample_grid_get(&multisample, &grid_w, &grid_h);
    CHECK(grid_w == 4 && grid_h == 2);
    multisample_samples_set(&multisample, 16);
    multisample_grid_get(&multisample, &grid_w, &grid_h);
    CHECK(grid_w == 4 && grid_h == 4);
    multisample_samples_set(&multisample, 3);
    multisample_grid_get(&multisample, &grid_w, &grid_h);
    CHECK(grid_w == 1 && grid_h == 1);
    multisample_grid_get(&multisample, nullptr, nullptr);
    CHECK(!multisample_enable_get(nullptr));
    CHECK(multisample_samples_get(nullptr) == 0);

    // Low-byte booleans across every boolean setter of the object.
    multisample_enable_set(&multisample, 0x100);
    CHECK(!multisample_enable_get(&multisample));
    multisample_enable_set(&multisample, 0x101);
    CHECK(multisample_enable_get(&multisample));
    multisample_a2c_set(&multisample, 0x100);
    CHECK(!multisample_a2c_get(&multisample));
    multisample_a2c_set(&multisample, 1);
    CHECK(multisample_a2c_get(&multisample));
    multisample_dither_set(&multisample, 0x100);
    CHECK(!multisample_dither_get(&multisample));
    multisample_dither_set(&multisample, 0xff);
    CHECK(multisample_dither_get(&multisample));
    multisample_c2c_set(&multisample, 0x100);
    CHECK(!multisample_c2c_get(&multisample));
    multisample_c2c_set(&multisample, 0x101);
    CHECK(multisample_c2c_get(&multisample));
    multisample_locations_set(&multisample, 0x100);
    CHECK(!multisample_locations_get(&multisample));
    multisample_locations_set(&multisample, 1);
    CHECK(multisample_locations_get(&multisample));
    multisample_grid_set(&multisample, 0x100);
    CHECK(!multisample_grid_enable_get(&multisample));
    multisample_grid_set(&multisample, 0x101);
    CHECK(multisample_grid_enable_get(&multisample));

    // Positions survive copying the state and occupy the same 24-byte API object.
    const float positions[4] = {0.25f, 0.25f, 0.75f, 0.75f};
    multisample_locations_table_set(&multisample, 0, 2, positions);
    multisample_locations_table_set(&multisample, 0, 0, nullptr);
    NVNmultisampleState copied_multisample = multisample;
    const auto& location_state = *reinterpret_cast<const MultisampleStateBits*>(&copied_multisample);
    CHECK(location_state.sample_locations[0] == 0x44);
    CHECK(location_state.sample_locations[1] == 0xcc);
    CHECK(location_state.sample_location_mask == 3);
    multisample_locations_table_set(&multisample, 15, 2, positions);
    CHECK(std::memcmp(&multisample, &copied_multisample, sizeof(multisample)) == 0);
    CHECK(multisample_samples_get(&multisample) == 3);

    // ------------------------------------------------------------- polygon
    const auto polygon_defaults = api_function<Entry::nvnPolygonStateSetDefaults>();
    const auto polygon_cull_set = api_function<Entry::nvnPolygonStateSetCullFace>();
    const auto polygon_front_set = api_function<Entry::nvnPolygonStateSetFrontFace>();
    const auto polygon_mode_set = api_function<Entry::nvnPolygonStateSetPolygonMode>();
    const auto polygon_offset_set = api_function<Entry::nvnPolygonStateSetPolygonOffsetEnables>();
    const auto polygon_cull_get = api_function<Entry::nvnPolygonStateGetCullFace>();
    const auto polygon_front_get = api_function<Entry::nvnPolygonStateGetFrontFace>();
    const auto polygon_mode_get = api_function<Entry::nvnPolygonStateGetPolygonMode>();
    const auto polygon_offset_get = api_function<Entry::nvnPolygonStateGetPolygonOffsetEnables>();

    NVNpolygonState polygon{};
    std::memset(&polygon, 0xff, sizeof(polygon));
    polygon_defaults(&polygon);
    CHECK(polygon_cull_get(&polygon) == NVN_FACE_NONE);
    CHECK(polygon_front_get(&polygon) == NVN_FRONT_FACE_CCW);
    CHECK(polygon_mode_get(&polygon) == NVN_POLYGON_MODE_FILL);
    CHECK(polygon_offset_get(&polygon) == NVN_POLYGON_OFFSET_ENABLE_NONE);

    polygon_cull_set(&polygon, NVN_FACE_FRONT_AND_BACK);
    polygon_front_set(&polygon, NVN_FRONT_FACE_CW);
    polygon_mode_set(&polygon, NVN_POLYGON_MODE_LINE);
    polygon_offset_set(&polygon, NVN_POLYGON_OFFSET_ENABLE_POINT | NVN_POLYGON_OFFSET_ENABLE_FILL);
    CHECK(polygon_cull_get(&polygon) == NVN_FACE_FRONT_AND_BACK);
    CHECK(polygon_front_get(&polygon) == NVN_FRONT_FACE_CW);
    CHECK(polygon_mode_get(&polygon) == NVN_POLYGON_MODE_LINE);
    CHECK(polygon_offset_get(&polygon) ==
          (NVN_POLYGON_OFFSET_ENABLE_POINT | NVN_POLYGON_OFFSET_ENABLE_FILL));
    // Offset enables pack into three bits.
    polygon_offset_set(&polygon, 0xff);
    CHECK(polygon_offset_get(&polygon) == 7);
    CHECK(polygon_cull_get(nullptr) == NVN_FACE_NONE);
    CHECK(polygon_front_get(nullptr) == NVN_FRONT_FACE_CCW);
    CHECK(polygon_mode_get(nullptr) == NVN_POLYGON_MODE_FILL);
    CHECK(polygon_offset_get(nullptr) == NVN_POLYGON_OFFSET_ENABLE_NONE);
    polygon_cull_set(nullptr, NVN_FACE_BACK);

    // ------------------------------------------------------------- depth stencil
    const auto depth_defaults = api_function<Entry::nvnDepthStencilStateSetDefaults>();
    const auto depth_test_set = api_function<Entry::nvnDepthStencilStateSetDepthTestEnable>();
    const auto depth_write_set = api_function<Entry::nvnDepthStencilStateSetDepthWriteEnable>();
    const auto depth_func_set = api_function<Entry::nvnDepthStencilStateSetDepthFunc>();
    const auto stencil_test_set = api_function<Entry::nvnDepthStencilStateSetStencilTestEnable>();
    const auto stencil_func_set = api_function<Entry::nvnDepthStencilStateSetStencilFunc>();
    const auto stencil_op_set = api_function<Entry::nvnDepthStencilStateSetStencilOp>();
    const auto depth_test_get = api_function<Entry::nvnDepthStencilStateGetDepthTestEnable>();
    const auto depth_write_get = api_function<Entry::nvnDepthStencilStateGetDepthWriteEnable>();
    const auto depth_func_get = api_function<Entry::nvnDepthStencilStateGetDepthFunc>();
    const auto stencil_test_get = api_function<Entry::nvnDepthStencilStateGetStencilTestEnable>();
    const auto stencil_func_get = api_function<Entry::nvnDepthStencilStateGetStencilFunc>();
    const auto stencil_op_get = api_function<Entry::nvnDepthStencilStateGetStencilOp>();

    NVNdepthStencilState depth{};
    std::memset(&depth, 0xff, sizeof(depth));
    depth_defaults(&depth);
    CHECK(!depth_test_get(&depth));
    CHECK(depth_write_get(&depth));
    CHECK(depth_func_get(&depth) == NVN_DEPTH_FUNC_LESS);
    CHECK(!stencil_test_get(&depth));
    CHECK(stencil_func_get(&depth, NVN_FACE_FRONT) == NVN_STENCIL_FUNC_ALWAYS);
    CHECK(stencil_func_get(&depth, NVN_FACE_BACK) == NVN_STENCIL_FUNC_ALWAYS);
    NVNstencilOp fail = NVN_STENCIL_OP_ZERO, zfail = NVN_STENCIL_OP_ZERO, zpass = NVN_STENCIL_OP_ZERO;
    stencil_op_get(&depth, NVN_FACE_FRONT, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_KEEP && zfail == NVN_STENCIL_OP_KEEP && zpass == NVN_STENCIL_OP_KEEP);
    stencil_op_get(&depth, NVN_FACE_BACK, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_KEEP && zfail == NVN_STENCIL_OP_KEEP && zpass == NVN_STENCIL_OP_KEEP);

    // Depth roundtrip and low-byte booleans.
    depth_test_set(&depth, 0x100);
    depth_write_set(&depth, 0x100);
    CHECK(!depth_test_get(&depth));
    CHECK(!depth_write_get(&depth));
    depth_test_set(&depth, 0x101);
    depth_write_set(&depth, 0x101);
    CHECK(depth_test_get(&depth));
    CHECK(depth_write_get(&depth));
    depth_func_set(&depth, NVN_DEPTH_FUNC_GEQUAL);
    CHECK(depth_func_get(&depth) == NVN_DEPTH_FUNC_GEQUAL);
    stencil_test_set(&depth, 1);
    CHECK(stencil_test_get(&depth));

    // Face handling: each face bit addresses its own stencil; no bit changes
    // nothing, and both bits write both faces.
    stencil_func_set(&depth, NVN_FACE_FRONT, NVN_STENCIL_FUNC_LESS);
    CHECK(stencil_func_get(&depth, NVN_FACE_FRONT) == NVN_STENCIL_FUNC_LESS);
    CHECK(stencil_func_get(&depth, NVN_FACE_BACK) == NVN_STENCIL_FUNC_ALWAYS);
    stencil_func_set(&depth, NVN_FACE_BACK, NVN_STENCIL_FUNC_GREATER);
    CHECK(stencil_func_get(&depth, NVN_FACE_FRONT) == NVN_STENCIL_FUNC_LESS);
    CHECK(stencil_func_get(&depth, NVN_FACE_BACK) == NVN_STENCIL_FUNC_GREATER);
    // Getters read the back stencil only for the exact back-face value;
    // combined or absent faces read the front stencil.
    CHECK(stencil_func_get(&depth, NVN_FACE_FRONT_AND_BACK) == NVN_STENCIL_FUNC_LESS);
    CHECK(stencil_func_get(&depth, NVN_FACE_NONE) == NVN_STENCIL_FUNC_LESS);
    stencil_func_set(&depth, NVN_FACE_NONE, NVN_STENCIL_FUNC_NEVER);
    CHECK(stencil_func_get(&depth, NVN_FACE_FRONT) == NVN_STENCIL_FUNC_LESS);
    CHECK(stencil_func_get(&depth, NVN_FACE_BACK) == NVN_STENCIL_FUNC_GREATER);
    stencil_func_set(&depth, NVN_FACE_FRONT_AND_BACK, NVN_STENCIL_FUNC_EQUAL);
    CHECK(stencil_func_get(&depth, NVN_FACE_FRONT) == NVN_STENCIL_FUNC_EQUAL);
    CHECK(stencil_func_get(&depth, NVN_FACE_BACK) == NVN_STENCIL_FUNC_EQUAL);

    stencil_op_set(&depth, NVN_FACE_BACK, NVN_STENCIL_OP_ZERO, NVN_STENCIL_OP_INVERT, NVN_STENCIL_OP_INCR_WRAP);
    stencil_op_get(&depth, NVN_FACE_FRONT, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_KEEP && zfail == NVN_STENCIL_OP_KEEP && zpass == NVN_STENCIL_OP_KEEP);
    stencil_op_get(&depth, NVN_FACE_BACK, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_ZERO && zfail == NVN_STENCIL_OP_INVERT && zpass == NVN_STENCIL_OP_INCR_WRAP);
    stencil_op_get(&depth, NVN_FACE_FRONT_AND_BACK, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_KEEP && zfail == NVN_STENCIL_OP_KEEP && zpass == NVN_STENCIL_OP_KEEP);
    stencil_op_set(&depth, NVN_FACE_NONE, NVN_STENCIL_OP_DECR, NVN_STENCIL_OP_DECR, NVN_STENCIL_OP_DECR);
    stencil_op_get(&depth, NVN_FACE_BACK, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_ZERO && zfail == NVN_STENCIL_OP_INVERT && zpass == NVN_STENCIL_OP_INCR_WRAP);
    stencil_op_set(&depth, NVN_FACE_FRONT_AND_BACK, NVN_STENCIL_OP_REPLACE, NVN_STENCIL_OP_DECR_WRAP,
                   NVN_STENCIL_OP_INCR);
    stencil_op_get(&depth, NVN_FACE_FRONT, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_REPLACE && zfail == NVN_STENCIL_OP_DECR_WRAP && zpass == NVN_STENCIL_OP_INCR);
    stencil_op_get(&depth, NVN_FACE_BACK, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_REPLACE && zfail == NVN_STENCIL_OP_DECR_WRAP && zpass == NVN_STENCIL_OP_INCR);
    stencil_op_get(&depth, NVN_FACE_FRONT, nullptr, nullptr, nullptr);

    // A copied object keeps the full depth/stencil state.
    const NVNdepthStencilState depth_copy = depth;
    CHECK(depth_test_get(&depth_copy));
    CHECK(depth_func_get(&depth_copy) == NVN_DEPTH_FUNC_GEQUAL);
    CHECK(stencil_func_get(&depth_copy, NVN_FACE_FRONT) == NVN_STENCIL_FUNC_EQUAL);
    CHECK(stencil_func_get(&depth_copy, NVN_FACE_BACK) == NVN_STENCIL_FUNC_EQUAL);
    stencil_op_get(&depth_copy, NVN_FACE_BACK, &fail, &zfail, &zpass);
    CHECK(fail == NVN_STENCIL_OP_REPLACE && zfail == NVN_STENCIL_OP_DECR_WRAP && zpass == NVN_STENCIL_OP_INCR);

    CHECK(!depth_test_get(nullptr));
    CHECK(!depth_write_get(nullptr));
    CHECK(depth_func_get(nullptr) == NVN_DEPTH_FUNC_ALWAYS);
    CHECK(!stencil_test_get(nullptr));
    CHECK(stencil_func_get(nullptr, NVN_FACE_BACK) == NVN_STENCIL_FUNC_ALWAYS);
    stencil_op_get(nullptr, NVN_FACE_FRONT, &fail, &zfail, &zpass);
    depth_test_set(nullptr, 1);

    // ------------------------------------------------------------- vertex attrib
    const auto attrib_defaults = api_function<Entry::nvnVertexAttribStateSetDefaults>();
    const auto attrib_format_set = api_function<Entry::nvnVertexAttribStateSetFormat>();
    const auto attrib_stream_set = api_function<Entry::nvnVertexAttribStateSetStreamIndex>();
    const auto attrib_format_get = api_function<Entry::nvnVertexAttribStateGetFormat>();
    const auto attrib_stream_get = api_function<Entry::nvnVertexAttribStateGetStreamIndex>();

    NVNvertexAttribState attrib{};
    std::memset(&attrib, 0xff, sizeof(attrib));
    attrib_defaults(&attrib);
    NVNformat format = NVN_FORMAT_RGBA32F;
    ptrdiff_t offset = -1;
    attrib_format_get(&attrib, &format, &offset);
    CHECK(format == NVN_FORMAT_NONE && offset == 0);
    CHECK(attrib_stream_get(&attrib) == 0);

    attrib_format_set(&attrib, NVN_FORMAT_RGBA16F, 0x12345);
    attrib_stream_set(&attrib, 15);
    attrib_format_get(&attrib, &format, &offset);
    CHECK(format == NVN_FORMAT_RGBA16F && offset == 0x12345);
    CHECK(attrib_stream_get(&attrib) == 15);
    attrib_format_set(&attrib, NVN_FORMAT_BGR5A1, (ptrdiff_t(1) << 19) - 1);
    attrib_stream_set(&attrib, 31);
    attrib_format_get(&attrib, &format, &offset);
    CHECK(format == NVN_FORMAT_BGR5A1 && offset == (ptrdiff_t(1) << 19) - 1);
    CHECK(attrib_stream_get(&attrib) == 31);

    // Offsets beyond the 19-bit packed range are diagnosed and truncate.
    attrib_format_set(&attrib, NVN_FORMAT_R8, ptrdiff_t(1) << 19);
    attrib_format_get(&attrib, &format, &offset);
    CHECK(format == NVN_FORMAT_R8 && offset == 0);
    attrib_format_set(&attrib, NVN_FORMAT_R8, -1);
    attrib_format_get(&attrib, &format, &offset);
    CHECK(format == NVN_FORMAT_R8 && offset == (ptrdiff_t(1) << 19) - 1);
    attrib_format_get(&attrib, nullptr, nullptr);
    attrib_format_get(nullptr, &format, &offset);
    CHECK(attrib_stream_get(nullptr) == 0);
    attrib_format_set(nullptr, NVN_FORMAT_R8, 0);

    // ------------------------------------------------------------- vertex stream
    const auto stream_defaults = api_function<Entry::nvnVertexStreamStateSetDefaults>();
    const auto stream_stride_set = api_function<Entry::nvnVertexStreamStateSetStride>();
    const auto stream_divisor_set = api_function<Entry::nvnVertexStreamStateSetDivisor>();
    const auto stream_stride_get = api_function<Entry::nvnVertexStreamStateGetStride>();
    const auto stream_divisor_get = api_function<Entry::nvnVertexStreamStateGetDivisor>();

    NVNvertexStreamState stream{};
    std::memset(&stream, 0xff, sizeof(stream));
    stream_defaults(&stream);
    CHECK(stream_stride_get(&stream) == 0);
    CHECK(stream_divisor_get(&stream) == 0);

    stream_stride_set(&stream, 0x13579);
    stream_divisor_set(&stream, 3);
    CHECK(stream_stride_get(&stream) == 0x13579);
    CHECK(stream_divisor_get(&stream) == 3);
    stream_stride_set(&stream, 0x100000000LL);
    CHECK(stream_stride_get(&stream) == 0);  // strides pack into 32 bits
    stream_divisor_set(&stream, 0);
    CHECK(stream_divisor_get(&stream) == 0);
    CHECK(stream_stride_get(nullptr) == 0);
    CHECK(stream_divisor_get(nullptr) == 0);
    stream_stride_set(nullptr, 16);

    std::printf("NVN replay state failures: %u\n", failures);
    return failures ? 1 : 0;
}
