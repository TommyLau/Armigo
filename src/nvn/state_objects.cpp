// ============================================================================
// Armigo NVN - State Objects
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
// Setters and getters of the immutable state objects (blend, color, channel
// mask, multisample, polygon, depth/stencil, vertex attributes, vertex
// streams). The state is packed in place inside the object storage the game
// allocated, so a game copying an object by value copies the state with it;
// command records keep the packed bytes as they are.
//
// Behavior:
//   - Defaults follow the API defaults of each state object.
//   - Booleans arrive from the guest in a 32-bit register of which only the
//     low 8 bits are defined; every boolean parameter is read that way.
//   - Enumeration values and indices are stored as given into their packed
//     hardware-width fields; out-of-range values truncate like the packed
//     layout does.
//   - Per-target indices (blend enable, channel mask) accept 0..7; anything
//     else is ignored, and getters clamp to a false/zero result.
//   - Stencil setters take a face mask: the front bit applies to the front
//     stencil, the back bit to the back stencil, no bit changes nothing.
//     Stencil getters select the back face only for the exact back-face
//     value; every other face reads the front stencil.

#include "commands.h"
#include "entry_table.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace armigo::nvn {
namespace {

template <typename Bits>
Bits& packed(void* object) noexcept {
    return *reinterpret_cast<Bits*>(object);
}
template <typename Bits>
const Bits& packed(const void* object) noexcept {
    return *reinterpret_cast<const Bits*>(object);
}

void diagnostic(const char* text) noexcept { armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN state: %s\n", text); }

// ---------------------------------------------------------------- blend

// The advanced blending fields share the 64-bit unit the packed basic fields
// occupy; the API leaves the object exactly this much storage. Bits 0..36 are
// the layout of BlendStateBits, in the same order; the tests verify that the
// shared prefix decodes identically through the public view.
struct BlendStorage {
    std::uint64_t target : 3, source_color : 7, destination_color : 7, source_alpha : 7, destination_alpha : 7,
        color_equation : 3, alpha_equation : 3, advanced_mode : 6, advanced_overlap : 2,
        advanced_premultiplied_source : 1, advanced_normalized_destination : 1;
};
static_assert(sizeof(BlendStorage) == sizeof(BlendStateBits));

void blend_set_defaults(NVNblendState* object) {
    if (!object) return;
    std::memset(object, 0, sizeof(*object));
    auto& state = packed<BlendStorage>(object);
    // No blending: source ONE, destination ZERO, equation ADD for both color
    // and alpha. Advanced blending starts disabled: mode NONE, overlap
    // UNCORRELATED, premultiplied source and normalized destination false.
    state.source_color = NVN_BLEND_FUNC_ONE;
    state.source_alpha = NVN_BLEND_FUNC_ONE;
    state.destination_color = NVN_BLEND_FUNC_ZERO;
    state.destination_alpha = NVN_BLEND_FUNC_ZERO;
    state.color_equation = NVN_BLEND_EQUATION_ADD;
    state.alpha_equation = NVN_BLEND_EQUATION_ADD;
}

void blend_set_target(NVNblendState* object, int target) {
    if (!object) return;
    packed<BlendStorage>(object).target = static_cast<std::uint64_t>(target);
}

void blend_set_func(NVNblendState* object, NVNblendFunc source_rgb, NVNblendFunc destination_rgb,
                    NVNblendFunc source_alpha, NVNblendFunc destination_alpha) {
    if (!object) return;
    auto& state = packed<BlendStorage>(object);
    state.source_color = static_cast<std::uint64_t>(source_rgb);
    state.destination_color = static_cast<std::uint64_t>(destination_rgb);
    state.source_alpha = static_cast<std::uint64_t>(source_alpha);
    state.destination_alpha = static_cast<std::uint64_t>(destination_alpha);
}

void blend_set_equation(NVNblendState* object, NVNblendEquation color, NVNblendEquation alpha) {
    if (!object) return;
    auto& state = packed<BlendStorage>(object);
    state.color_equation = static_cast<std::uint64_t>(color);
    state.alpha_equation = static_cast<std::uint64_t>(alpha);
}

void blend_set_advanced_mode(NVNblendState* object, NVNblendAdvancedMode mode) {
    if (!object) return;
    packed<BlendStorage>(object).advanced_mode = static_cast<std::uint64_t>(mode);
}

void blend_set_advanced_overlap(NVNblendState* object, NVNblendAdvancedOverlap overlap) {
    if (!object) return;
    packed<BlendStorage>(object).advanced_overlap = static_cast<std::uint64_t>(overlap);
}

void blend_set_advanced_premultiplied_source(NVNblendState* object, std::uint32_t enable) {
    if (!object) return;
    packed<BlendStorage>(object).advanced_premultiplied_source = guest_bool(enable) ? 1 : 0;
}

void blend_set_advanced_normalized_destination(NVNblendState* object, std::uint32_t enable) {
    if (!object) return;
    packed<BlendStorage>(object).advanced_normalized_destination = guest_bool(enable) ? 1 : 0;
}

int blend_get_target(const NVNblendState* object) {
    return object ? static_cast<int>(packed<BlendStorage>(object).target) : 0;
}

void blend_get_func(const NVNblendState* object, NVNblendFunc* source_rgb, NVNblendFunc* destination_rgb,
                    NVNblendFunc* source_alpha, NVNblendFunc* destination_alpha) {
    if (!object) return;
    const auto& state = packed<BlendStorage>(object);
    if (source_rgb) *source_rgb = static_cast<NVNblendFunc>(state.source_color);
    if (destination_rgb) *destination_rgb = static_cast<NVNblendFunc>(state.destination_color);
    if (source_alpha) *source_alpha = static_cast<NVNblendFunc>(state.source_alpha);
    if (destination_alpha) *destination_alpha = static_cast<NVNblendFunc>(state.destination_alpha);
}

void blend_get_equation(const NVNblendState* object, NVNblendEquation* color, NVNblendEquation* alpha) {
    if (!object) return;
    const auto& state = packed<BlendStorage>(object);
    if (color) *color = static_cast<NVNblendEquation>(state.color_equation);
    if (alpha) *alpha = static_cast<NVNblendEquation>(state.alpha_equation);
}

NVNblendAdvancedMode blend_get_advanced_mode(const NVNblendState* object) {
    return object ? static_cast<NVNblendAdvancedMode>(packed<BlendStorage>(object).advanced_mode)
                  : NVN_BLEND_ADVANCED_MODE_NONE;
}

NVNblendAdvancedOverlap blend_get_advanced_overlap(const NVNblendState* object) {
    return object ? static_cast<NVNblendAdvancedOverlap>(packed<BlendStorage>(object).advanced_overlap)
                  : NVN_BLEND_ADVANCED_OVERLAP_UNCORRELATED;
}

NVNboolean blend_get_advanced_premultiplied_source(const NVNblendState* object) {
    return object && packed<BlendStorage>(object).advanced_premultiplied_source ? 1 : 0;
}

NVNboolean blend_get_advanced_normalized_destination(const NVNblendState* object) {
    return object && packed<BlendStorage>(object).advanced_normalized_destination ? 1 : 0;
}

// ---------------------------------------------------------------- color

void color_set_defaults(NVNcolorState* object) {
    if (!object) return;
    std::memset(object, 0, sizeof(*object));
    auto& state = packed<ColorStateBits>(object);
    state.logic_op = NVN_LOGIC_OP_COPY;
    state.alpha_test = NVN_ALPHA_FUNC_ALWAYS;
}

void color_set_blend_enable(NVNcolorState* object, int index, std::uint32_t enable) {
    if (!object || index < 0 || index > 7) return;
    auto& state = packed<ColorStateBits>(object);
    if (guest_bool(enable))
        state.blend_enable |= 1u << index;
    else
        state.blend_enable &= ~(1u << index);
}

void color_set_logic_op(NVNcolorState* object, NVNlogicOp op) {
    if (!object) return;
    packed<ColorStateBits>(object).logic_op = static_cast<std::uint32_t>(op);
}

void color_set_alpha_test(NVNcolorState* object, NVNalphaFunc func) {
    if (!object) return;
    packed<ColorStateBits>(object).alpha_test = static_cast<std::uint32_t>(func);
}

NVNboolean color_get_blend_enable(const NVNcolorState* object, int index) {
    if (!object || index < 0 || index > 7) return 0;
    return (packed<ColorStateBits>(object).blend_enable >> index) & 1u;
}

NVNlogicOp color_get_logic_op(const NVNcolorState* object) {
    return object ? static_cast<NVNlogicOp>(packed<ColorStateBits>(object).logic_op) : NVN_LOGIC_OP_CLEAR;
}

NVNalphaFunc color_get_alpha_test(const NVNcolorState* object) {
    return object ? static_cast<NVNalphaFunc>(packed<ColorStateBits>(object).alpha_test) : NVN_ALPHA_FUNC_ALWAYS;
}

// ---------------------------------------------------------------- channel mask

void channel_mask_set_defaults(NVNchannelMaskState* object) {
    if (!object) return;
    std::memset(object, 0xff, sizeof(*object));
}

void channel_mask_set(NVNchannelMaskState* object, int index, std::uint32_t r, std::uint32_t g, std::uint32_t b,
                      std::uint32_t a) {
    if (!object || index < 0 || index > 7) return;
    const std::uint32_t channels = (guest_bool(r) ? 1u : 0u) | (guest_bool(g) ? 2u : 0u) | (guest_bool(b) ? 4u : 0u) |
                                   (guest_bool(a) ? 8u : 0u);
    const unsigned shift = 4 * static_cast<unsigned>(index);
    auto& state = packed<ChannelMaskStateBits>(object);
    state.mask = (state.mask & ~(0xfu << shift)) | (channels << shift);
}

void channel_mask_get(const NVNchannelMaskState* object, int index, NVNboolean* r, NVNboolean* g, NVNboolean* b,
                      NVNboolean* a) {
    const std::uint32_t channels =
        object && index >= 0 && index <= 7 ? (packed<ChannelMaskStateBits>(object).mask >> (4 * index)) & 0xfu : 0u;
    if (r) *r = (channels & 1u) != 0;
    if (g) *g = (channels & 2u) != 0;
    if (b) *b = (channels & 4u) != 0;
    if (a) *a = (channels & 8u) != 0;
}

// ---------------------------------------------------------------- multisample

// Grid of sample locations a sample count uses: 1x 1x1, 2x 2x1, 4x 2x2, 8x
// 4x2, 16x 4x4; any other count falls back to a single cell.
constexpr int sample_grid_width(int samples) noexcept {
    return samples == 2 || samples == 4 ? 2 : samples == 8 || samples == 16 ? 4 : 1;
}
constexpr int sample_grid_height(int samples) noexcept {
    return samples == 4 || samples == 8 ? 2 : samples == 16 ? 4 : 1;
}


void multisample_set_defaults(NVNmultisampleState* object) {
    if (!object) return;
    std::memset(object, 0, sizeof(*object));
    packed<MultisampleStateBits>(object).enable = 1;
}

void multisample_set_enable(NVNmultisampleState* object, std::uint32_t enable) {
    if (!object) return;
    packed<MultisampleStateBits>(object).enable = guest_bool(enable) ? 1 : 0;
}

void multisample_set_samples(NVNmultisampleState* object, int samples) {
    if (!object) return;
    packed<MultisampleStateBits>(object).samples = static_cast<std::uint8_t>(samples);
}

void multisample_set_alpha_to_coverage(NVNmultisampleState* object, std::uint32_t enable) {
    if (!object) return;
    packed<MultisampleStateBits>(object).alpha_to_coverage = guest_bool(enable) ? 1 : 0;
}

void multisample_set_alpha_to_coverage_dither(NVNmultisampleState* object, std::uint32_t enable) {
    if (!object) return;
    packed<MultisampleStateBits>(object).alpha_to_coverage_dither = guest_bool(enable) ? 1 : 0;
}

int multisample_get_samples(const NVNmultisampleState* object) {
    return object ? packed<MultisampleStateBits>(object).samples : 0;
}

NVNboolean multisample_get_enable(const NVNmultisampleState* object) {
    return object && packed<MultisampleStateBits>(object).enable ? 1 : 0;
}

NVNboolean multisample_get_alpha_to_coverage(const NVNmultisampleState* object) {
    return object && packed<MultisampleStateBits>(object).alpha_to_coverage ? 1 : 0;
}

NVNboolean multisample_get_alpha_to_coverage_dither(const NVNmultisampleState* object) {
    return object && packed<MultisampleStateBits>(object).alpha_to_coverage_dither ? 1 : 0;
}

void multisample_set_raster_samples(NVNmultisampleState* object, int samples) {
    if (!object) return;
    packed<MultisampleStateBits>(object).raster_samples = static_cast<std::uint8_t>(samples);
}

int multisample_get_raster_samples(NVNmultisampleState* object) {
    return object ? packed<const MultisampleStateBits>(object).raster_samples : 0;
}

void multisample_set_coverage_modulation(NVNmultisampleState* object, NVNcoverageModulationMode mode) {
    if (!object) return;
    packed<MultisampleStateBits>(object).coverage_modulation = static_cast<std::uint8_t>(mode);
}

NVNcoverageModulationMode multisample_get_coverage_modulation(const NVNmultisampleState* object) {
    return object ? static_cast<NVNcoverageModulationMode>(packed<MultisampleStateBits>(object).coverage_modulation)
                  : NVN_COVERAGE_MODULATION_MODE_NONE;
}

void multisample_set_coverage_to_color_enable(NVNmultisampleState* object, std::uint32_t enable) {
    if (!object) return;
    packed<MultisampleStateBits>(object).coverage_to_color_enable = guest_bool(enable) ? 1 : 0;
}

NVNboolean multisample_get_coverage_to_color_enable(const NVNmultisampleState* object) {
    return object && packed<MultisampleStateBits>(object).coverage_to_color_enable ? 1 : 0;
}

void multisample_set_coverage_to_color_output(NVNmultisampleState* object, int index) {
    if (!object) return;
    packed<MultisampleStateBits>(object).coverage_to_color_output = static_cast<std::uint8_t>(index);
}

int multisample_get_coverage_to_color_output(const NVNmultisampleState* object) {
    return object ? packed<MultisampleStateBits>(object).coverage_to_color_output : 0;
}

void multisample_set_sample_locations_enable(NVNmultisampleState* object, std::uint32_t enable) {
    if (!object) return;
    packed<MultisampleStateBits>(object).sample_locations_enable = guest_bool(enable) ? 1 : 0;
}

NVNboolean multisample_get_sample_locations_enable(const NVNmultisampleState* object) {
    return object && packed<MultisampleStateBits>(object).sample_locations_enable ? 1 : 0;
}

void multisample_get_sample_locations_grid(NVNmultisampleState* object, int* width, int* height) {
    const int samples = multisample_get_samples(object);
    if (width) *width = sample_grid_width(samples);
    if (height) *height = sample_grid_height(samples);
}

void multisample_set_sample_locations_grid_enable(NVNmultisampleState* object, std::uint32_t enable) {
    if (!object) return;
    packed<MultisampleStateBits>(object).sample_locations_grid = guest_bool(enable) ? 1 : 0;
}

NVNboolean multisample_get_sample_locations_grid_enable(const NVNmultisampleState* object) {
    return object && packed<MultisampleStateBits>(object).sample_locations_grid ? 1 : 0;
}

void multisample_set_sample_locations(NVNmultisampleState* object, int start, int count, const float* locations) {
    if (!object || start < 0 || count < 0 || start > 16 || count > 16 - start || (count && !locations)) {
        diagnostic("invalid programmable sample location range");
        return;
    }
    for (int i = 0; i < count * 2; ++i) {
        if (!std::isfinite(locations[i]) || locations[i] < 0.0f || locations[i] > 1.0f) {
            diagnostic("sample location coordinate is outside the unit pixel");
            return;
        }
    }
    auto& state = packed<MultisampleStateBits>(object);
    for (int i = 0; i < count; ++i) {
        const auto quantize = [](float value) {
            const auto grid = static_cast<unsigned>(value * 16.0f);
            return grid < 16u ? grid : 15u;
        };
        state.sample_locations[start + i] = static_cast<std::uint8_t>(
            quantize(locations[2 * i]) | (quantize(locations[2 * i + 1]) << 4));
        state.sample_location_mask |= static_cast<std::uint16_t>(1u << (start + i));
    }
}

// ---------------------------------------------------------------- polygon

void polygon_set_defaults(NVNpolygonState* object) {
    if (!object) return;
    std::memset(object, 0, sizeof(*object));
    auto& state = packed<PolygonStateBits>(object);
    state.cull = NVN_FACE_NONE;
    state.front_face = NVN_FRONT_FACE_CCW;
    state.mode = NVN_POLYGON_MODE_FILL;
}

void polygon_set_cull_face(NVNpolygonState* object, NVNface face) {
    if (!object) return;
    packed<PolygonStateBits>(object).cull = static_cast<std::uint32_t>(face);
}

void polygon_set_front_face(NVNpolygonState* object, NVNfrontFace face) {
    if (!object) return;
    packed<PolygonStateBits>(object).front_face = static_cast<std::uint32_t>(face);
}

void polygon_set_mode(NVNpolygonState* object, NVNpolygonMode mode) {
    if (!object) return;
    packed<PolygonStateBits>(object).mode = static_cast<std::uint32_t>(mode);
}

void polygon_set_offset_enables(NVNpolygonState* object, int enables) {
    if (!object) return;
    packed<PolygonStateBits>(object).offset_enables = static_cast<std::uint32_t>(enables);
}

NVNface polygon_get_cull_face(const NVNpolygonState* object) {
    return object ? static_cast<NVNface>(packed<PolygonStateBits>(object).cull) : NVN_FACE_NONE;
}

NVNfrontFace polygon_get_front_face(const NVNpolygonState* object) {
    return object ? static_cast<NVNfrontFace>(packed<PolygonStateBits>(object).front_face) : NVN_FRONT_FACE_CCW;
}

NVNpolygonMode polygon_get_mode(const NVNpolygonState* object) {
    return object ? static_cast<NVNpolygonMode>(packed<PolygonStateBits>(object).mode) : NVN_POLYGON_MODE_FILL;
}

NVNpolygonOffsetEnable polygon_get_offset_enables(const NVNpolygonState* object) {
    return object ? static_cast<NVNpolygonOffsetEnable>(packed<PolygonStateBits>(object).offset_enables)
                  : NVN_POLYGON_OFFSET_ENABLE_NONE;
}

// ---------------------------------------------------------------- depth stencil

void depth_stencil_set_defaults(NVNdepthStencilState* object) {
    if (!object) return;
    std::memset(object, 0, sizeof(*object));
    auto& state = packed<DepthStencilStateBits>(object);
    // Depth writes on with a LESS comparison; the stencil test off with
    // ALWAYS functions and KEEP operations on both faces.
    state.depth_write = 1;
    state.depth_function = NVN_DEPTH_FUNC_LESS;
    state.front_function = NVN_STENCIL_FUNC_ALWAYS;
    state.back_function = NVN_STENCIL_FUNC_ALWAYS;
    state.front_fail = NVN_STENCIL_OP_KEEP;
    state.front_depth_fail = NVN_STENCIL_OP_KEEP;
    state.front_pass = NVN_STENCIL_OP_KEEP;
    state.back_fail = NVN_STENCIL_OP_KEEP;
    state.back_depth_fail = NVN_STENCIL_OP_KEEP;
    state.back_pass = NVN_STENCIL_OP_KEEP;
}

void depth_stencil_set_depth_test(NVNdepthStencilState* object, std::uint32_t enable) {
    if (!object) return;
    packed<DepthStencilStateBits>(object).depth_test = guest_bool(enable) ? 1 : 0;
}

void depth_stencil_set_depth_write(NVNdepthStencilState* object, std::uint32_t enable) {
    if (!object) return;
    packed<DepthStencilStateBits>(object).depth_write = guest_bool(enable) ? 1 : 0;
}

void depth_stencil_set_depth_func(NVNdepthStencilState* object, NVNdepthFunc func) {
    if (!object) return;
    packed<DepthStencilStateBits>(object).depth_function = static_cast<std::uint64_t>(func);
}

void depth_stencil_set_stencil_test(NVNdepthStencilState* object, std::uint32_t enable) {
    if (!object) return;
    packed<DepthStencilStateBits>(object).stencil_test = guest_bool(enable) ? 1 : 0;
}

void depth_stencil_set_stencil_func(NVNdepthStencilState* object, NVNface face, NVNstencilFunc func) {
    if (!object) return;
    auto& state = packed<DepthStencilStateBits>(object);
    const auto value = static_cast<std::uint64_t>(func);
    if (static_cast<int>(face) & NVN_FACE_FRONT) state.front_function = value;
    if (static_cast<int>(face) & NVN_FACE_BACK) state.back_function = value;
}

void depth_stencil_set_stencil_op(NVNdepthStencilState* object, NVNface face, NVNstencilOp fail, NVNstencilOp depth_fail,
                                  NVNstencilOp pass) {
    if (!object) return;
    auto& state = packed<DepthStencilStateBits>(object);
    if (static_cast<int>(face) & NVN_FACE_FRONT) {
        state.front_fail = static_cast<std::uint64_t>(fail);
        state.front_depth_fail = static_cast<std::uint64_t>(depth_fail);
        state.front_pass = static_cast<std::uint64_t>(pass);
    }
    if (static_cast<int>(face) & NVN_FACE_BACK) {
        state.back_fail = static_cast<std::uint64_t>(fail);
        state.back_depth_fail = static_cast<std::uint64_t>(depth_fail);
        state.back_pass = static_cast<std::uint64_t>(pass);
    }
}

NVNboolean depth_stencil_get_depth_test(const NVNdepthStencilState* object) {
    return object && packed<DepthStencilStateBits>(object).depth_test ? 1 : 0;
}

NVNboolean depth_stencil_get_depth_write(const NVNdepthStencilState* object) {
    return object && packed<DepthStencilStateBits>(object).depth_write ? 1 : 0;
}

NVNdepthFunc depth_stencil_get_depth_func(const NVNdepthStencilState* object) {
    return object ? static_cast<NVNdepthFunc>(packed<DepthStencilStateBits>(object).depth_function)
                  : NVN_DEPTH_FUNC_ALWAYS;
}

NVNboolean depth_stencil_get_stencil_test(const NVNdepthStencilState* object) {
    return object && packed<DepthStencilStateBits>(object).stencil_test ? 1 : 0;
}

NVNstencilFunc depth_stencil_get_stencil_func(const NVNdepthStencilState* object, NVNface face) {
    if (!object) return NVN_STENCIL_FUNC_ALWAYS;
    const auto& state = packed<DepthStencilStateBits>(object);
    return static_cast<NVNstencilFunc>(face == NVN_FACE_BACK ? state.back_function : state.front_function);
}

void depth_stencil_get_stencil_op(const NVNdepthStencilState* object, NVNface face, NVNstencilOp* fail,
                                  NVNstencilOp* depth_fail, NVNstencilOp* pass) {
    if (!object) return;
    const auto& state = packed<DepthStencilStateBits>(object);
    const bool back = face == NVN_FACE_BACK;
    if (fail) *fail = static_cast<NVNstencilOp>(back ? state.back_fail : state.front_fail);
    if (depth_fail) *depth_fail = static_cast<NVNstencilOp>(back ? state.back_depth_fail : state.front_depth_fail);
    if (pass) *pass = static_cast<NVNstencilOp>(back ? state.back_pass : state.front_pass);
}

// ---------------------------------------------------------------- vertex attrib

void vertex_attrib_set_defaults(NVNvertexAttribState* object) {
    if (!object) return;
    std::memset(object, 0, sizeof(*object));
}

void vertex_attrib_set_format(NVNvertexAttribState* object, NVNformat format, ptrdiff_t offset) {
    if (!object) return;
    if (offset < 0 || offset >= (ptrdiff_t(1) << 19))
        armigo::diag::write(armigo::diag::Level::error,  "Armigo NVN state: vertex attribute offset %td is outside the packable range\n", offset);
    auto& state = packed<VertexAttributeStateBits>(object);
    state.format = static_cast<std::uint32_t>(format);
    state.offset = static_cast<std::uint32_t>(offset);
}

void vertex_attrib_set_stream_index(NVNvertexAttribState* object, int index) {
    if (!object) return;
    packed<VertexAttributeStateBits>(object).stream = static_cast<std::uint32_t>(index);
}

void vertex_attrib_get_format(const NVNvertexAttribState* object, NVNformat* format, ptrdiff_t* offset) {
    if (!object) return;
    const auto& state = packed<VertexAttributeStateBits>(object);
    if (format) *format = static_cast<NVNformat>(state.format);
    if (offset) *offset = static_cast<ptrdiff_t>(state.offset);
}

int vertex_attrib_get_stream_index(const NVNvertexAttribState* object) {
    return object ? static_cast<int>(packed<VertexAttributeStateBits>(object).stream) : 0;
}

// ---------------------------------------------------------------- vertex stream

void vertex_stream_set_defaults(NVNvertexStreamState* object) {
    if (!object) return;
    std::memset(object, 0, sizeof(*object));
}

void vertex_stream_set_stride(NVNvertexStreamState* object, ptrdiff_t stride) {
    if (!object) return;
    packed<VertexStreamStateBits>(object).stride = static_cast<std::uint32_t>(stride);
}

void vertex_stream_set_divisor(NVNvertexStreamState* object, int divisor) {
    if (!object) return;
    packed<VertexStreamStateBits>(object).divisor = static_cast<std::uint32_t>(divisor);
}

ptrdiff_t vertex_stream_get_stride(const NVNvertexStreamState* object) {
    return object ? static_cast<ptrdiff_t>(packed<VertexStreamStateBits>(object).stride) : 0;
}

int vertex_stream_get_divisor(const NVNvertexStreamState* object) {
    return object ? static_cast<int>(packed<VertexStreamStateBits>(object).divisor) : 0;
}

} // namespace

void register_state_entries() noexcept {
    // Blend.
    install<Entry::nvnBlendStateSetDefaults>(&blend_set_defaults);
    install<Entry::nvnBlendStateSetBlendTarget>(&blend_set_target);
    install<Entry::nvnBlendStateSetBlendFunc>(&blend_set_func);
    install<Entry::nvnBlendStateSetBlendEquation>(&blend_set_equation);
    install<Entry::nvnBlendStateSetAdvancedMode>(&blend_set_advanced_mode);
    install<Entry::nvnBlendStateSetAdvancedOverlap>(&blend_set_advanced_overlap);
    install<Entry::nvnBlendStateSetAdvancedPremultipliedSrc>(&blend_set_advanced_premultiplied_source);
    install<Entry::nvnBlendStateSetAdvancedNormalizedDst>(&blend_set_advanced_normalized_destination);
    install<Entry::nvnBlendStateGetBlendTarget>(&blend_get_target);
    install<Entry::nvnBlendStateGetBlendFunc>(&blend_get_func);
    install<Entry::nvnBlendStateGetBlendEquation>(&blend_get_equation);
    install<Entry::nvnBlendStateGetAdvancedMode>(&blend_get_advanced_mode);
    install<Entry::nvnBlendStateGetAdvancedOverlap>(&blend_get_advanced_overlap);
    install<Entry::nvnBlendStateGetAdvancedPremultipliedSrc>(&blend_get_advanced_premultiplied_source);
    install<Entry::nvnBlendStateGetAdvancedNormalizedDst>(&blend_get_advanced_normalized_destination);
    // Color.
    install<Entry::nvnColorStateSetDefaults>(&color_set_defaults);
    install<Entry::nvnColorStateSetBlendEnable>(&color_set_blend_enable);
    install<Entry::nvnColorStateSetLogicOp>(&color_set_logic_op);
    install<Entry::nvnColorStateSetAlphaTest>(&color_set_alpha_test);
    install<Entry::nvnColorStateGetBlendEnable>(&color_get_blend_enable);
    install<Entry::nvnColorStateGetLogicOp>(&color_get_logic_op);
    install<Entry::nvnColorStateGetAlphaTest>(&color_get_alpha_test);
    // Channel mask.
    install<Entry::nvnChannelMaskStateSetDefaults>(&channel_mask_set_defaults);
    install<Entry::nvnChannelMaskStateSetChannelMask>(&channel_mask_set);
    install<Entry::nvnChannelMaskStateGetChannelMask>(&channel_mask_get);
    // Multisample.
    install<Entry::nvnMultisampleStateSetDefaults>(&multisample_set_defaults);
    install<Entry::nvnMultisampleStateSetMultisampleEnable>(&multisample_set_enable);
    install<Entry::nvnMultisampleStateSetSamples>(&multisample_set_samples);
    install<Entry::nvnMultisampleStateSetAlphaToCoverageEnable>(&multisample_set_alpha_to_coverage);
    install<Entry::nvnMultisampleStateSetAlphaToCoverageDither>(&multisample_set_alpha_to_coverage_dither);
    install<Entry::nvnMultisampleStateSetRasterSamples>(&multisample_set_raster_samples);
    install<Entry::nvnMultisampleStateSetCoverageModulationMode>(&multisample_set_coverage_modulation);
    install<Entry::nvnMultisampleStateSetCoverageToColorEnable>(&multisample_set_coverage_to_color_enable);
    install<Entry::nvnMultisampleStateSetCoverageToColorOutput>(&multisample_set_coverage_to_color_output);
    install<Entry::nvnMultisampleStateSetSampleLocationsEnable>(&multisample_set_sample_locations_enable);
    install<Entry::nvnMultisampleStateSetSampleLocationsGridEnable>(&multisample_set_sample_locations_grid_enable);
    install<Entry::nvnMultisampleStateSetSampleLocations>(&multisample_set_sample_locations);
    install<Entry::nvnMultisampleStateGetMultisampleEnable>(&multisample_get_enable);
    install<Entry::nvnMultisampleStateGetSamples>(&multisample_get_samples);
    install<Entry::nvnMultisampleStateGetAlphaToCoverageEnable>(&multisample_get_alpha_to_coverage);
    install<Entry::nvnMultisampleStateGetAlphaToCoverageDither>(&multisample_get_alpha_to_coverage_dither);
    install<Entry::nvnMultisampleStateGetRasterSamples>(&multisample_get_raster_samples);
    install<Entry::nvnMultisampleStateGetCoverageModulationMode>(&multisample_get_coverage_modulation);
    install<Entry::nvnMultisampleStateGetCoverageToColorEnable>(&multisample_get_coverage_to_color_enable);
    install<Entry::nvnMultisampleStateGetCoverageToColorOutput>(&multisample_get_coverage_to_color_output);
    install<Entry::nvnMultisampleStateGetSampleLocationsEnable>(&multisample_get_sample_locations_enable);
    install<Entry::nvnMultisampleStateGetSampleLocationsGrid>(&multisample_get_sample_locations_grid);
    install<Entry::nvnMultisampleStateGetSampleLocationsGridEnable>(&multisample_get_sample_locations_grid_enable);
    // Polygon.
    install<Entry::nvnPolygonStateSetDefaults>(&polygon_set_defaults);
    install<Entry::nvnPolygonStateSetCullFace>(&polygon_set_cull_face);
    install<Entry::nvnPolygonStateSetFrontFace>(&polygon_set_front_face);
    install<Entry::nvnPolygonStateSetPolygonMode>(&polygon_set_mode);
    install<Entry::nvnPolygonStateSetPolygonOffsetEnables>(&polygon_set_offset_enables);
    install<Entry::nvnPolygonStateGetCullFace>(&polygon_get_cull_face);
    install<Entry::nvnPolygonStateGetFrontFace>(&polygon_get_front_face);
    install<Entry::nvnPolygonStateGetPolygonMode>(&polygon_get_mode);
    install<Entry::nvnPolygonStateGetPolygonOffsetEnables>(&polygon_get_offset_enables);
    // Depth stencil.
    install<Entry::nvnDepthStencilStateSetDefaults>(&depth_stencil_set_defaults);
    install<Entry::nvnDepthStencilStateSetDepthTestEnable>(&depth_stencil_set_depth_test);
    install<Entry::nvnDepthStencilStateSetDepthWriteEnable>(&depth_stencil_set_depth_write);
    install<Entry::nvnDepthStencilStateSetDepthFunc>(&depth_stencil_set_depth_func);
    install<Entry::nvnDepthStencilStateSetStencilTestEnable>(&depth_stencil_set_stencil_test);
    install<Entry::nvnDepthStencilStateSetStencilFunc>(&depth_stencil_set_stencil_func);
    install<Entry::nvnDepthStencilStateSetStencilOp>(&depth_stencil_set_stencil_op);
    install<Entry::nvnDepthStencilStateGetDepthTestEnable>(&depth_stencil_get_depth_test);
    install<Entry::nvnDepthStencilStateGetDepthWriteEnable>(&depth_stencil_get_depth_write);
    install<Entry::nvnDepthStencilStateGetDepthFunc>(&depth_stencil_get_depth_func);
    install<Entry::nvnDepthStencilStateGetStencilTestEnable>(&depth_stencil_get_stencil_test);
    install<Entry::nvnDepthStencilStateGetStencilFunc>(&depth_stencil_get_stencil_func);
    install<Entry::nvnDepthStencilStateGetStencilOp>(&depth_stencil_get_stencil_op);
    // Vertex attributes and streams.
    install<Entry::nvnVertexAttribStateSetDefaults>(&vertex_attrib_set_defaults);
    install<Entry::nvnVertexAttribStateSetFormat>(&vertex_attrib_set_format);
    install<Entry::nvnVertexAttribStateSetStreamIndex>(&vertex_attrib_set_stream_index);
    install<Entry::nvnVertexAttribStateGetFormat>(&vertex_attrib_get_format);
    install<Entry::nvnVertexAttribStateGetStreamIndex>(&vertex_attrib_get_stream_index);
    install<Entry::nvnVertexStreamStateSetDefaults>(&vertex_stream_set_defaults);
    install<Entry::nvnVertexStreamStateSetStride>(&vertex_stream_set_stride);
    install<Entry::nvnVertexStreamStateSetDivisor>(&vertex_stream_set_divisor);
    install<Entry::nvnVertexStreamStateGetStride>(&vertex_stream_get_stride);
    install<Entry::nvnVertexStreamStateGetDivisor>(&vertex_stream_get_divisor);
}

} // namespace armigo::nvn
