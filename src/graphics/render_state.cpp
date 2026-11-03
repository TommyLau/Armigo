// ============================================================================
// Armigo Graphics - GPU Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Render-state decoding covers rasterization, scissors, depth clamp and test,
// alpha test, viewport transforms, polygon mode, depth bias, stencil state,
// culling, front face, color masks, blending, logic operations, and multisampling.
// Maxwell register layouts and GL-derived packed aliases are decoded directly.
// Depth-bias control selects format-dependent bias or force-UNORM. The host
// passes raw units divided by two; the constant quantum is 2^-n for UNORM
// and 2^-24 for force-UNORM float depth, separate from Metal's own scale.

#include <armigo/graphics/render_state.h>

#include <armigo/graphics/texture_format.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

namespace armigo::graphics {
namespace {

// Maxwell 3D method offsets below follow the sequential guest register layout
// (the register file is uint words, so method = byte/4).
// They agree with the existing Armigo frontend, which programs and consumes
// the same offsets (viewport swizzle at 0x286 + 8*i, surface clip at 0x3fd,
// blend state array at 0x780).
constexpr std::size_t raster_enable_method = 0x0df;
constexpr std::size_t viewport_transform_base_method = 0x280;  // 8 words each
constexpr std::size_t viewport_extents_base_method = 0x300;    // 4 words each
constexpr std::size_t depth_mode_method = 0x35f;
constexpr std::size_t polygon_mode_front_method = 0x36b;
constexpr std::size_t polygon_mode_back_method = 0x36c;
constexpr std::size_t depth_bias_enable_method = 0x370;  // point, line, fill
constexpr std::size_t blend_ucode_enable_method = 0x374;
constexpr std::size_t scissor_base_method = 0x380;  // 4 words each
constexpr std::size_t stencil_back_masks_method = 0x3d5;  // func_ref, mask, func_mask
constexpr std::size_t color_mask_shared_method = 0x3e4;
constexpr std::size_t surface_clip_method = 0x3fd;  // {x, width}, {y, height}
constexpr std::size_t depth_bias_control_method = 0x444;
constexpr std::size_t depth_test_enable_method = 0x4b3;
constexpr std::size_t blend_independent_method = 0x4b9;
constexpr std::size_t depth_write_enable_method = 0x4ba;
constexpr std::size_t alpha_test_enable_method = 0x4bb;
constexpr std::size_t depth_test_func_method = 0x4c3;
constexpr std::size_t blend_constant_method = 0x4c7;  // ColorF, 4 words
constexpr std::size_t blend_state_common_method = 0x4cf;  // 8 words, alpha dst at +7
constexpr std::size_t blend_enable_common_method = 0x4d7;
constexpr std::size_t blend_enable_base_method = 0x4d8;  // BlendEnable[i]
constexpr std::size_t stencil_test_method = 0x4e0;  // 8 words
constexpr std::size_t y_control_method = 0x4eb;
constexpr std::size_t multisample_enable_method = 0x54d;
constexpr std::size_t multisample_control_method = 0x54f;
constexpr std::size_t depth_bias_factor_method = 0x55b;
constexpr std::size_t stencil_back_test_method = 0x565;  // 5 words
constexpr std::size_t depth_bias_units_method = 0x56f;
constexpr std::size_t msaa_mode_method = 0x574;
constexpr std::size_t depth_bias_clamp_method = 0x61f;
constexpr std::size_t face_state_method = 0x646;  // cull_enable, front_face, cull_face
constexpr std::size_t viewport_transform_enable_method = 0x64b;
constexpr std::size_t view_volume_clip_control_method = 0x64f;
constexpr std::size_t logic_op_state_method = 0x671;  // enable, op
constexpr std::size_t color_mask_base_method = 0x680;  // RtColorMask[i]
constexpr std::size_t blend_state_base_method = 0x780;  // BlendState[i], 8 words

constexpr std::size_t total_targets = 8;
constexpr std::size_t total_viewports = 16;

// Maxwell packed encodings are the GL constants with a second alias per value,
// collapsed here into one name.
constexpr std::uint32_t front_face_clockwise = 0x900;
constexpr std::uint32_t front_face_counter_clockwise = 0x901;
constexpr std::uint32_t cull_front = 0x404;
constexpr std::uint32_t cull_back = 0x405;
constexpr std::uint32_t cull_front_and_back = 0x408;
constexpr std::uint32_t polygon_point = 0x1b00;
constexpr std::uint32_t polygon_line = 0x1b01;
constexpr std::uint32_t polygon_fill = 0x1b02;

// YControl packed flags.
constexpr std::uint32_t y_control_negate_y = 1u << 0;
constexpr std::uint32_t y_control_triangle_rast_flip = 1u << 4;

// ViewVolumeClipControl packed flags. Only the depth-clamp
// bit names observable renderer behavior; real guest profiles set further
// undocumented bits (the admitted draw profile programs 0x181d), so unknown
// bits stay ignored rather than rejected.
constexpr std::uint32_t clip_control_depth_clamp_disabled = 1u << 11;

// Viewport swizzle word: three bits per component, X in bits 0-2 up to W in
// bits 12-14. Positive encodings are the even values 0/2/4/6.
constexpr std::uint32_t viewport_swizzle_component_mask = 0x7;

// MultisampleControl packed fields.
constexpr std::uint32_t multisample_control_alpha_to_coverage = 1u << 0;
constexpr std::uint32_t multisample_control_alpha_to_one = 1u << 4;

constexpr double coordinate_bound = 1099511627776.0;  // 2^40

constexpr bool flag(std::uint32_t word) noexcept
{
    // Boolean32 semantics: only bit 0 names the value.
    return (word & 1u) != 0;
}

constexpr std::uint32_t low_half(std::uint32_t value) noexcept
{
    return value & 0xffff;
}

constexpr std::uint32_t high_half(std::uint32_t value) noexcept
{
    return value >> 16;
}

RendererResult<RenderState> state_failure(const RenderStop& stop,
                                          std::size_t method,
                                          std::string_view detail) noexcept
{
    return RendererError{CommandResult{CommandStatus::Unsupported,
                                       detail,
                                       stop.original.address,
                                       0,
                                       0,
                                       static_cast<std::uint32_t>(method),
                                       stop.registers[method],
                                       engine_class::threed}};
}

// Checked signed helpers keep scissor arithmetic overflow-free by contract;
// the guest encodings (u16 fields, u16 screen scissor) are individually small.
constexpr bool checked_add(std::int64_t left, std::int64_t right, std::int64_t& sum) noexcept
{
    if (right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) {
        return false;
    }
    if (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right) {
        return false;
    }
    sum = left + right;
    return true;
}

// Rounds a finite, bounded viewport coordinate to the enclosing integer pixel
// grid. Lower edges round down and upper edges round up so the generated
// scissor cannot discard a pixel whose center lies inside a fractional
// viewport. Returns false for values outside the representable range.
bool bounded_floor(double value, std::int64_t& out) noexcept
{
    if (!std::isfinite(value) || std::abs(value) > coordinate_bound) {
        return false;
    }
    out = static_cast<std::int64_t>(std::floor(value));
    return true;
}

bool bounded_ceil(double value, std::int64_t& out) noexcept
{
    if (!std::isfinite(value) || std::abs(value) > coordinate_bound) {
        return false;
    }
    out = static_cast<std::int64_t>(std::ceil(value));
    return true;
}

bool decode_compare(std::uint32_t word, CompareFunction& out) noexcept
{
    // Compare operations: 1-8 plus 0x200-0x207 aliases.
    const std::uint32_t raw = word & 0xffu;
    if (word >= 0x200 && word <= 0x207) {
        out = static_cast<CompareFunction>(word - 0x200);
        return true;
    }
    if (raw >= 1 && raw <= 8) {
        out = static_cast<CompareFunction>(raw - 1);
        return true;
    }
    return false;
}

bool decode_stencil(std::uint32_t word, StencilOperation& out) noexcept
{
    // Stencil operations: 1-8 plus GL aliases.
    switch (word) {
    case 1:
    case 0x1e00:
        out = StencilOperation::Keep;
        return true;
    case 2:
        out = StencilOperation::Zero;
        return true;
    case 0x0:
        out = StencilOperation::Zero;
        return true;
    case 3:
    case 0x1e01:
        out = StencilOperation::Replace;
        return true;
    case 4:
    case 0x1e02:
        out = StencilOperation::IncrementClamp;
        return true;
    case 5:
    case 0x1e03:
        out = StencilOperation::DecrementClamp;
        return true;
    case 6:
    case 0x150a:
        out = StencilOperation::Invert;
        return true;
    case 7:
    case 0x8507:
        out = StencilOperation::IncrementWrap;
        return true;
    case 8:
    case 0x8508:
        out = StencilOperation::DecrementWrap;
        return true;
    default:
        return false;
    }
}

bool decode_blend_operation(std::uint32_t word, BlendOperation& out) noexcept
{
    // Blend operations: 1-5 plus GL aliases.
    switch (word) {
    case 1:
    case 0x8006:
        out = BlendOperation::Add;
        return true;
    case 2:
    case 0x800a:
        out = BlendOperation::Subtract;
        return true;
    case 3:
    case 0x800b:
        out = BlendOperation::ReverseSubtract;
        return true;
    case 4:
    case 0x8007:
        out = BlendOperation::Min;
        return true;
    case 5:
    case 0x8008:
        out = BlendOperation::Max;
        return true;
    default:
        return false;
    }
}

// Writes the Armigo factor and sets dual_source when the word names the
// second shader source. The flag is never cleared; callers reset it first.
bool decode_blend_factor(std::uint32_t word, BlendFactor& out, bool& dual_source) noexcept
{
    // Blend factors. The raw values 0x10-0x13
    // and GL aliases 0xc900-0xc903 name the second shader source; Armigo decodes
    // them so callers can reject dual-source blending explicitly.
    switch (word) {
    case 0x1:
    case 0x4000:
        out = BlendFactor::Zero;
        return true;
    case 0x2:
    case 0x4001:
        out = BlendFactor::One;
        return true;
    case 0x3:
    case 0x4300:
        out = BlendFactor::SourceColor;
        return true;
    case 0x4:
    case 0x4301:
        out = BlendFactor::OneMinusSourceColor;
        return true;
    case 0x5:
    case 0x4302:
        out = BlendFactor::SourceAlpha;
        return true;
    case 0x6:
    case 0x4303:
        out = BlendFactor::OneMinusSourceAlpha;
        return true;
    case 0x7:
    case 0x4304:
        out = BlendFactor::DestinationAlpha;
        return true;
    case 0x8:
    case 0x4305:
        out = BlendFactor::OneMinusDestinationAlpha;
        return true;
    case 0x9:
    case 0x4306:
        out = BlendFactor::DestinationColor;
        return true;
    case 0xa:
    case 0x4307:
        out = BlendFactor::OneMinusDestinationColor;
        return true;
    case 0xb:
    case 0x4308:
        out = BlendFactor::SourceAlphaSaturate;
        return true;
    case 0xc001:
        out = BlendFactor::ConstantColor;
        return true;
    case 0xc002:
        out = BlendFactor::OneMinusConstantColor;
        return true;
    case 0xc003:
        out = BlendFactor::ConstantAlpha;
        return true;
    case 0xc004:
        out = BlendFactor::OneMinusConstantAlpha;
        return true;
    case 0x10:
    case 0xc900:
        out = BlendFactor::Source1Color;
        dual_source = true;
        return true;
    case 0x11:
    case 0xc901:
        out = BlendFactor::OneMinusSource1Color;
        dual_source = true;
        return true;
    case 0x12:
    case 0xc902:
        out = BlendFactor::Source1Alpha;
        dual_source = true;
        return true;
    case 0x13:
    case 0xc903:
        out = BlendFactor::OneMinusSource1Alpha;
        dual_source = true;
        return true;
    default:
        return false;
    }
}

// One decoded target's blend state. The shared layout carries an undocumented word
// between the alpha source factor and the alpha destination factor, so the
// destination index differs between the shared and independent layouts.
struct BlendWords {
    BlendOperation color_operation = BlendOperation::Add;
    BlendFactor color_source = BlendFactor::One;
    BlendFactor color_destination = BlendFactor::Zero;
    BlendOperation alpha_operation = BlendOperation::Add;
    BlendFactor alpha_source = BlendFactor::One;
    BlendFactor alpha_destination = BlendFactor::Zero;
    bool separate_alpha = false;
    bool dual_source = false;
};

[[nodiscard]] std::optional<BlendWords> decode_blend_words(const RenderStop& stop,
                                                           std::size_t base,
                                                           bool common_layout) noexcept
{
    const std::size_t alpha_destination_index = common_layout ? 7 : 6;
    BlendWords words;
    words.separate_alpha = flag(stop.registers[base]);
    if (!decode_blend_operation(stop.registers[base + 1], words.color_operation) ||
        !decode_blend_operation(stop.registers[base + 4], words.alpha_operation) ||
        !decode_blend_factor(stop.registers[base + 2], words.color_source, words.dual_source) ||
        !decode_blend_factor(stop.registers[base + 3], words.color_destination, words.dual_source) ||
        !decode_blend_factor(stop.registers[base + 5], words.alpha_source, words.dual_source) ||
        !decode_blend_factor(stop.registers[base + alpha_destination_index],
                             words.alpha_destination,
                             words.dual_source)) {
        return std::nullopt;
    }
    return words;
}

}  // namespace

RendererResult<RenderState> plan_render_state(const RenderStop& stop,
                                              const RenderTargetPlan& targets)
{
    if (stop.engine != engine_class::threed) {
        return state_failure(stop, 0, "render state requires the 3D engine register file");
    }

    // Alpha tests have no Metal renderer
    // path, so reached alpha-test state stops instead of dropping the test.
    if (flag(stop.registers[alpha_test_enable_method])) {
        return state_failure(stop, alpha_test_enable_method, "alpha test is unsupported");
    }
    // UpdateLogicOpState: Metal has no logical raster operation.
    if (flag(stop.registers[logic_op_state_method])) {
        return state_failure(stop, logic_op_state_method, "logic operations are unsupported");
    }
    // UpdateBlendState: blend microcode has no substitute; ordinary blending
    // must never silently replace it.
    if (stop.registers[blend_ucode_enable_method] != 0) {
        return state_failure(stop, blend_ucode_enable_method, "blend microcode is unsupported");
    }
    // GetDepthMode decodes the register as bit 0. The admitted shader
    // coordinate profile is MinusOneToOne; the zero-to-one convention needs
    // its own shader transform and stops here until it is implemented.
    if ((stop.registers[depth_mode_method] & 1u) != 0) {
        return state_failure(stop, depth_mode_method, "zero-to-one depth mode is unsupported");
    }

    RenderState state{};

    // ---- Blend constant (UpdateBlendState reads ColorF BlendConstant).
    for (std::size_t component = 0; component < 4; ++component) {
        state.blend_constant[component] =
            std::bit_cast<float>(stop.registers[blend_constant_method + component]);
        if (!std::isfinite(state.blend_constant[component])) {
            return state_failure(stop, blend_constant_method + component,
                                 "blend constant is not finite");
        }
    }

    // ---- Color write masks (UpdateRtColorMask). Channels sit at bits 0, 4,
    // 8, 12 of each RtColorMask word. RtColorMaskShared routes every target
    // through word zero. The masks are logical-output indexed, like the blend
    // arrays; RtColorState permutation is attachment planning's concern.
    const bool mask_shared = flag(stop.registers[color_mask_shared_method]);
    for (std::size_t target = 0; target < total_targets; ++target) {
        const std::size_t index = mask_shared ? 0 : target;
        const std::uint32_t packed = stop.registers[color_mask_base_method + index];
        state.color[target].write_mask = static_cast<std::uint8_t>((packed & 0x1u) |
                                                                   ((packed >> 3) & 0x2u) |
                                                                   ((packed >> 6) & 0x4u) |
                                                                   ((packed >> 9) & 0x8u));
    }

    // ---- Blend equations and factors (UpdateBlendState).
    const bool independent_blend = flag(stop.registers[blend_independent_method]);
    // Shared blending has two possible enable sources. Resolve ambiguity at admission: the shared enable and all eight per-target
    // enables must agree, otherwise the register file is ambiguous and stops.
    if (!independent_blend) {
        const bool common_enable = flag(stop.registers[blend_enable_common_method]);
        for (std::size_t target = 0; target < total_targets; ++target) {
            if (flag(stop.registers[blend_enable_base_method + target]) != common_enable) {
                return state_failure(stop, blend_enable_common_method,
                                     "shared blend enables disagree across targets");
            }
        }
    }
    for (std::size_t target = 0; target < total_targets; ++target) {
        ColorBlendState& blend = state.color[target];
        blend.enabled = independent_blend
            ? flag(stop.registers[blend_enable_base_method + target])
            : flag(stop.registers[blend_enable_common_method]);

        // Disabled targets keep the interface defaults: their words are inert
        // guest state (unprogrammed slots hold zero operations) and the host
        // ignores them.
        if (!blend.enabled) {
            continue;
        }

        const std::optional<BlendWords> words = independent_blend
            ? decode_blend_words(stop, blend_state_base_method + target * 8, false)
            : decode_blend_words(stop, blend_state_common_method, true);
        if (!words.has_value()) {
            return state_failure(stop,
                                 independent_blend ? blend_state_base_method + target * 8
                                                   : blend_state_common_method,
                                 "blend state has an unsupported operation or factor");
        }

        // SeparateAlpha is decoded but does not gate the alpha words; the alpha
        // equation always comes from the alpha words.
        // Dual-source blending needs a second fragment output and a tested
        // pipeline form; reached dual-source factors stop for now.
        if (words->dual_source) {
            return state_failure(stop,
                                 independent_blend ? blend_state_base_method + target * 8
                                                   : blend_state_common_method,
                                 "dual-source blend factors are unsupported");
        }

        BlendEquation rgb;
        rgb.operation = words->color_operation;
        rgb.source = words->color_source;
        rgb.destination = words->color_destination;
        BlendEquation alpha;
        alpha.operation = words->alpha_operation;
        alpha.source = words->alpha_source;
        alpha.destination = words->alpha_destination;

        // FilterBlendFactor (UpdateBlendState): targets whose guest format has
        // no alpha channel must not read destination alpha, because hosts
        // express those formats on RGBA storage. Replace destination-alpha
        // factors with the equivalent constant.
        const auto& attachment = targets.colors[target];
        if (attachment.has_value() &&
            (!attachment->has_alpha ||
             texture_format_info(attachment->texture.format).components != 4)) {
            auto filter = [](BlendFactor& factor) noexcept {
                if (factor == BlendFactor::DestinationAlpha) {
                    factor = BlendFactor::One;
                } else if (factor == BlendFactor::OneMinusDestinationAlpha) {
                    factor = BlendFactor::Zero;
                }
            };
            filter(rgb.source);
            filter(rgb.destination);
            filter(alpha.source);
            filter(alpha.destination);
        }

        blend.rgb = rgb;
        blend.alpha = alpha;
    }

    // ---- Depth test (UpdateDepthTestState). The compare function is only
    // observable while the test is enabled, so unprogrammed words stay inert.
    DepthStencilState& depth_stencil = state.depth_stencil;
    depth_stencil.depth_test = flag(stop.registers[depth_test_enable_method]);
    depth_stencil.depth_write = flag(stop.registers[depth_write_enable_method]);
    if (depth_stencil.depth_test &&
        !decode_compare(stop.registers[depth_test_func_method], depth_stencil.depth_compare)) {
        return state_failure(stop, depth_test_func_method, "depth compare function is unsupported");
    }

    // ---- Stencil (UpdateStencilTestState). The front face block carries the
    // operations, masks and reference; with two-sided testing clear, the back
    // face inherits all of the front state, including the reference. State is
    // validated while the test is enabled and stays at defaults otherwise.
    depth_stencil.stencil_test = flag(stop.registers[stencil_test_method]);
    if (depth_stencil.stencil_test) {
        StencilFaceState& front = depth_stencil.front;
        if (!decode_stencil(stop.registers[stencil_test_method + 1], front.fail) ||
            !decode_stencil(stop.registers[stencil_test_method + 2], front.depth_fail) ||
            !decode_stencil(stop.registers[stencil_test_method + 3], front.pass) ||
            !decode_compare(stop.registers[stencil_test_method + 4], front.compare)) {
            return state_failure(stop, stencil_test_method, "front stencil state is unsupported");
        }
        front.compare_mask = stop.registers[stencil_test_method + 6];
        front.write_mask = stop.registers[stencil_test_method + 7];
        state.front_stencil_reference = stop.registers[stencil_test_method + 5];

        StencilFaceState& back = depth_stencil.back;
        if (flag(stop.registers[stencil_back_test_method])) {
            if (!decode_stencil(stop.registers[stencil_back_test_method + 1], back.fail) ||
                !decode_stencil(stop.registers[stencil_back_test_method + 2], back.depth_fail) ||
                !decode_stencil(stop.registers[stencil_back_test_method + 3], back.pass) ||
                !decode_compare(stop.registers[stencil_back_test_method + 4], back.compare)) {
                return state_failure(stop, stencil_back_test_method,
                                     "back stencil state is unsupported");
            }
            // StencilBackMasks: function reference, write mask, compare mask.
            state.back_stencil_reference = stop.registers[stencil_back_masks_method];
            back.write_mask = stop.registers[stencil_back_masks_method + 1];
            back.compare_mask = stop.registers[stencil_back_masks_method + 2];
        } else {
            back = front;
            state.back_stencil_reference = state.front_stencil_reference;
        }
    }

    // ---- Rasterizer (UpdateFaceState, UpdateFrontFace, UpdatePolygonMode,
    // UpdateRasterizerState, UpdateDepthClampState).
    RasterizerState& rasterizer = state.rasterizer;
    rasterizer.enabled = flag(stop.registers[raster_enable_method]);

    const bool cull_enable = flag(stop.registers[face_state_method]);
    switch (stop.registers[face_state_method + 2]) {
    case cull_front:
        rasterizer.cull = cull_enable ? CullMode::Front : CullMode::None;
        break;
    case cull_back:
        rasterizer.cull = cull_enable ? CullMode::Back : CullMode::None;
        break;
    case cull_front_and_back:
        rasterizer.cull = cull_enable ? CullMode::FrontAndBack : CullMode::None;
        break;
    default:
        return state_failure(stop, face_state_method + 2, "cull face selection is unsupported");
    }

    FrontFace front_face = FrontFace::Clockwise;
    switch (stop.registers[face_state_method + 1]) {
    case front_face_clockwise:
        front_face = FrontFace::Clockwise;
        break;
    case front_face_counter_clockwise:
        front_face = FrontFace::CounterClockwise;
        break;
    default:
        return state_failure(stop, face_state_method + 1, "front face selection is unsupported");
    }
    // Convert origin and signed viewport orientation together. The native
    // viewport below uses positive extents; a negative effective scale reverses
    // winding once. In the established positive-X/negative-Y profile this
    // cancels the upper-left origin reversal (verified against native pixels).
    const std::uint32_t y_control = stop.registers[y_control_method];
    const bool upper_left_origin = (y_control & y_control_triangle_rast_flip) == 0;
    bool viewport_reverses = false;
    if (stop.registers[viewport_transform_enable_method] != 0) {
        const float sx = std::bit_cast<float>(stop.registers[viewport_transform_base_method]);
        float sy = std::bit_cast<float>(stop.registers[viewport_transform_base_method + 1]);
        if ((y_control & y_control_negate_y) != 0) sy = -sy;
        viewport_reverses = (sx < 0) != (sy < 0);
    }
    if (upper_left_origin != viewport_reverses) {
        front_face = front_face == FrontFace::CounterClockwise ? FrontFace::Clockwise
                                                               : FrontFace::CounterClockwise;
    }
    rasterizer.front_face = front_face;

    auto decode_fill = [&](std::size_t method, FillMode& out) noexcept -> bool {
        switch (stop.registers[method]) {
        case polygon_point:
            out = FillMode::Point;
            return true;
        case polygon_line:
            out = FillMode::Line;
            return true;
        case polygon_fill:
            out = FillMode::Fill;
            return true;
        default:
            return false;
        }
    };
    if (!decode_fill(polygon_mode_front_method, rasterizer.front_fill) ||
        !decode_fill(polygon_mode_back_method, rasterizer.back_fill)) {
        return state_failure(stop, polygon_mode_front_method, "polygon fill mode is unsupported");
    }
    if (rasterizer.front_fill != rasterizer.back_fill) {
        return state_failure(stop, polygon_mode_back_method, "unequal fill modes are unsupported");
    }
    // Metal supports equal fill/line modes, but no polygon point mode. Polygon
    // modes only affect triangle primitives, so point and line draws ignore it.
    // Same classes as the draw plan: begin-op topologies 0-3 and override
    // topologies 1-3 are points and lines; everything else rasterizes as
    // triangles.
    const bool triangles = [&] {
        if (!stop.draw) return true;
        if (stop.registers[0x652]) return stop.registers[0x65c] < 1 || stop.registers[0x65c] > 3;
        return stop.draw->topology > 3;
    }();
    if (triangles && rasterizer.front_fill == FillMode::Point) {
        return state_failure(stop, polygon_mode_front_method,
                             "point polygon fill is unsupported");
    }

    // UpdateDepthClampState: DepthClampDisabled clear means clamping is on.
    rasterizer.depth_clip = (stop.registers[view_volume_clip_control_method] &
                             clip_control_depth_clamp_disabled) != 0
        ? DepthClipMode::Clip
        : DepthClipMode::Clamp;

    // Maxwell units are twice NVIDIA's conventional polygon-offset API units.
    // NVK's exact representation establishes the raw-register quantum: 2^-n
    // for UNORM and 2^-24 for force-UNORM float depth. Keep the normalized
    // displacement separate from the native backend's format-dependent scale.
    // Measured Metal float-depth quanta vary with depth, so a nonzero constant
    // is added in the fragment while slope-only bias stays on the encoder.
    DepthBiasState& bias = state.depth_bias;
    bias.point = flag(stop.registers[depth_bias_enable_method]);
    bias.line = flag(stop.registers[depth_bias_enable_method + 1]);
    bias.fill = flag(stop.registers[depth_bias_enable_method + 2]);
    bias.factor = std::bit_cast<float>(stop.registers[depth_bias_factor_method]);
    bias.units = std::bit_cast<float>(stop.registers[depth_bias_units_method]);
    bias.clamp = std::bit_cast<float>(stop.registers[depth_bias_clamp_method]);
    if (!std::isfinite(bias.factor) || !std::isfinite(bias.units) || !std::isfinite(bias.clamp)) {
        return state_failure(stop, depth_bias_factor_method, "depth bias value is not finite");
    }
    if (!bias.fill && (bias.point || bias.line)) {
        return state_failure(stop, depth_bias_enable_method,
                             "line/point-only polygon depth bias is unsupported");
    }
    if (bias.fill && rasterizer.front_fill != FillMode::Fill) {
        return state_failure(stop, depth_bias_enable_method,
                             "depth bias on non-filled polygons is unsupported");
    }
    if (bias.fill && targets.depth) {
        const auto control = stop.registers[depth_bias_control_method];
        if (control & ~1u)
            return state_failure(stop, depth_bias_control_method,
                                 "depth bias control bits are unsupported");
        const auto format = targets.depth->texture.format;
        const bool floating = format == TextureFormat::D32Float ||
                              format == TextureFormat::D32FloatS8Uint;
        if (floating && (control & 1u) && bias.units != 0)
            return state_failure(stop, depth_bias_control_method,
                                 "format-dependent floating depth bias is unsupported");
        switch (format) {
        case TextureFormat::D16Unorm:
            bias.constant = std::ldexp(bias.units, -16);
            break;
        case TextureFormat::D24UnormS8Uint:
        case TextureFormat::X8UintD24Unorm:
        case TextureFormat::S8UintD24Unorm:
        case TextureFormat::D32Float:
        case TextureFormat::D32FloatS8Uint:
            bias.constant = std::ldexp(bias.units, -24);
            break;
        default:
            return state_failure(stop, depth_bias_units_method,
                                 "depth bias attachment format is unsupported");
        }
    }

    // ---- Coverage (UpdateMultisampleState, TextureMsaaModeConverter).
    CoverageState& coverage = state.coverage;
    coverage.multisample = flag(stop.registers[multisample_enable_method]);
    const std::uint32_t multisample_control = stop.registers[multisample_control_method];
    coverage.alpha_to_coverage =
        (multisample_control & multisample_control_alpha_to_coverage) != 0;
    coverage.alpha_to_one = (multisample_control & multisample_control_alpha_to_one) != 0;
    std::uint32_t samples = 1;
    switch (stop.registers[msaa_mode_method]) {
    case 0:
        samples = 1;
        break;
    case 2:  // 2x2
        samples = 4;
        break;
    case 4:  // 4x2
        samples = 8;
        break;
    case 5:  // 2x1
        samples = 2;
        break;
    case 6:  // 4x4
        samples = 16;
        break;
    default:
        return state_failure(stop, msaa_mode_method, "multisample mode is unsupported");
    }
    coverage.samples = samples;

    // ---- Viewports (UpdateViewportTransform).
    const std::uint32_t surface_clip_horizontal = stop.registers[surface_clip_method];
    const std::uint32_t surface_clip_vertical = stop.registers[surface_clip_method + 1];
    const std::int64_t surface_left = low_half(surface_clip_horizontal);
    const std::int64_t surface_top = low_half(surface_clip_vertical);
    const std::int64_t surface_width = high_half(surface_clip_horizontal);
    const std::int64_t surface_height = high_half(surface_clip_vertical);
    const std::int64_t surface_right = surface_left + surface_width;
    const std::int64_t surface_bottom = surface_top + surface_height;
    const bool y_negate = (y_control & y_control_negate_y) != 0;
    const bool transform_disable = stop.registers[viewport_transform_enable_method] == 0;

    if (transform_disable) {
        // With the transform pipeline disabled every viewport is the screen
        // scissor rectangle and the full depth range. The primary viewport
        // must still cover pixels.
        if (surface_width <= 0 || surface_height <= 0) {
            return state_failure(stop, surface_clip_method,
                                 "primary viewport extent is not positive");
        }
        ViewportState disabled_viewport;
        disabled_viewport.x = static_cast<double>(surface_left);
        disabled_viewport.y = static_cast<double>(surface_top);
        disabled_viewport.width = static_cast<double>(surface_width);
        disabled_viewport.height = static_cast<double>(surface_height);
        disabled_viewport.near_depth = 0;
        disabled_viewport.far_depth = 1;
        state.viewports.assign(total_viewports, disabled_viewport);
    } else {
        state.viewports.reserve(total_viewports);
        for (std::size_t viewport = 0; viewport < total_viewports; ++viewport) {
            const std::size_t base = viewport_transform_base_method + viewport * 8;
            const double scale_x = std::abs(std::bit_cast<float>(stop.registers[base]));
            double scale_y = std::bit_cast<float>(stop.registers[base + 1]);
            const double scale_z = std::bit_cast<float>(stop.registers[base + 2]);
            const double translate_x = std::bit_cast<float>(stop.registers[base + 3]);
            const double translate_y = std::bit_cast<float>(stop.registers[base + 4]);
            const double translate_z = std::bit_cast<float>(stop.registers[base + 5]);
            const std::uint32_t swizzle = stop.registers[base + 6];

            // Metal exposes no viewport swizzle; any negative component
            // convention stops here instead of being folded into scale.
            for (std::size_t component = 0; component < 4; ++component) {
                const std::uint32_t encoding =
                    (swizzle >> (static_cast<unsigned>(component) * 4)) &
                    viewport_swizzle_component_mask;
                if ((encoding & 1u) != 0) {
                    return state_failure(stop, base + 6, "viewport swizzle is unsupported");
                }
            }

            if (y_negate) {
                scale_y = -scale_y;
            }

            // The window rectangle is translate minus/plus scale; extents are
            // emitted positive in the host's top-left window space. The host
            // encodes all sixteen viewports. The primary viewport must have
            // a positive extent; unused slots may remain empty.
            ViewportState viewport_state;
            viewport_state.x = translate_x - scale_x;
            viewport_state.y = translate_y - std::abs(scale_y);
            viewport_state.width = scale_x * 2;
            viewport_state.height = std::abs(scale_y) * 2;
            if (!std::isfinite(viewport_state.x) || !std::isfinite(viewport_state.y) ||
                !std::isfinite(viewport_state.width) || !std::isfinite(viewport_state.height) ||
                !std::isfinite(scale_z) || !std::isfinite(translate_z)) {
                return state_failure(stop, base, "viewport transform is not finite");
            }
            if (viewport == 0 && (viewport_state.width <= 0 || viewport_state.height <= 0)) {
                return state_failure(stop, base, "primary viewport extent is not positive");
            }

            const double clip_near = std::bit_cast<float>(
                stop.registers[viewport_extents_base_method + viewport * 4 + 2]);
            const double clip_far = std::bit_cast<float>(
                stop.registers[viewport_extents_base_method + viewport * 4 + 3]);
            if (!std::isfinite(clip_near) || !std::isfinite(clip_far) || clip_near < 0 ||
                clip_far > 1 || clip_near > clip_far) {
                return state_failure(stop, viewport_extents_base_method + viewport * 4 + 2,
                                     "viewport depth clip bounds are invalid");
            }
            // Custom clip bounds are not implemented. Viewport 0 is always
            // selected, so it stops; the other slots usually hold reset values
            // (a 1x1 extent with bounds 0/0) and matter only to a shader that
            // writes a viewport index, which is recorded so that case can stop.
            viewport_state.custom_depth_clip = clip_near != 0 || clip_far != 1;
            if (viewport == 0 && viewport_state.width > 0 && viewport_state.height > 0 &&
                viewport_state.custom_depth_clip) {
                return state_failure(stop, viewport_extents_base_method + viewport * 4 + 2,
                                     "independent viewport depth clip bounds are unsupported");
            }

            // ViewportExtents carries clip bounds, not the depth transform.
            // Apply the transform registers to Metal's zero-to-one viewport
            // range; reversed transforms remain ordered as the guest set them.
            const double depth_near = translate_z - scale_z;
            const double depth_far = translate_z + scale_z;
            if (depth_near < 0 || depth_near > 1 || depth_far < 0 || depth_far > 1) {
                return state_failure(stop, base + 2, "viewport depth transform is unsupported");
            }
            viewport_state.near_depth = depth_near;
            viewport_state.far_depth = depth_far;
            state.viewports.push_back(viewport_state);
        }
    }

    // ---- Scissors (UpdateScissorState), intersected with the matching
    // viewport. Disabled scissors widen to the full guest range; an empty
    // intersection is a valid no-coverage area, never an unsigned wrap.
    const auto viewport_pixel_bounds =
        [&](const ViewportState& viewport, std::int64_t& left, std::int64_t& right,
            std::int64_t& bottom, std::int64_t& top) noexcept -> bool {
        return bounded_floor(viewport.x, left) &&
               bounded_ceil(viewport.x + viewport.width, right) &&
               bounded_floor(viewport.y, bottom) && bounded_ceil(viewport.y + viewport.height, top);
    };

    state.scissors.reserve(total_viewports);
    for (std::size_t viewport = 0; viewport < total_viewports; ++viewport) {
        const std::size_t base = scissor_base_method + viewport * 4;
        std::int64_t scissor_left = 0;
        std::int64_t scissor_bottom = 0;
        std::int64_t scissor_right = 0xffff;
        std::int64_t scissor_top = 0xffff;
        if (flag(stop.registers[base]) &&
            !(low_half(stop.registers[base + 1]) == 0 && high_half(stop.registers[base + 1]) == 0xffff &&
              low_half(stop.registers[base + 2]) == 0 && high_half(stop.registers[base + 2]) == 0xffff)) {
            scissor_left = static_cast<std::int64_t>(low_half(stop.registers[base + 1]));
            scissor_right = static_cast<std::int64_t>(high_half(stop.registers[base + 1]));
            scissor_bottom = static_cast<std::int64_t>(low_half(stop.registers[base + 2]));
            scissor_top = static_cast<std::int64_t>(high_half(stop.registers[base + 2]));

            std::int64_t scissor_height = 0;
            if (!checked_add(scissor_top, -scissor_bottom, scissor_height)) {
                return state_failure(stop, base + 1, "scissor rectangle overflows");
            }
            if (y_negate) {
                // UpdateScissorState mirrors the scissor against the surface
                // clip bottom and clamps at zero.
                std::int64_t mirrored = 0;
                if (!checked_add(surface_bottom, -scissor_height, mirrored) ||
                    !checked_add(mirrored, -scissor_bottom, mirrored)) {
                    return state_failure(stop, base + 2, "scissor Y mirror overflows");
                }
                if (mirrored < 0) {
                    if (!checked_add(scissor_height, mirrored, scissor_height)) {
                        return state_failure(stop, base + 2, "scissor Y mirror overflows");
                    }
                    mirrored = 0;
                }
                scissor_bottom = mirrored;
                scissor_top = mirrored + scissor_height;
            }
        }

        const ViewportState& viewport_state = state.viewports[viewport];
        std::int64_t viewport_left = 0;
        std::int64_t viewport_right = 0;
        std::int64_t viewport_bottom = 0;
        std::int64_t viewport_top = 0;
        if (!viewport_pixel_bounds(viewport_state, viewport_left, viewport_right, viewport_bottom,
                                   viewport_top)) {
            return state_failure(stop, viewport_transform_base_method + viewport * 8,
                                 "viewport extent exceeds the coordinate range");
        }

        RenderArea area;
        area.x = std::max({scissor_left, viewport_left, surface_left});
        area.y = std::max({scissor_bottom, viewport_bottom, surface_top});
        const std::int64_t right = std::min({scissor_right, viewport_right, surface_right});
        const std::int64_t top = std::min({scissor_top, viewport_top, surface_bottom});
        area.width = right > area.x ? static_cast<std::uint64_t>(right - area.x) : 0;
        area.height = top > area.y ? static_cast<std::uint64_t>(top - area.y) : 0;
        state.scissors.push_back(area);
    }

    return state;
}

}  // namespace armigo::graphics
