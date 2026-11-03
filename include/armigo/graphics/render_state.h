// ============================================================================
// Armigo Graphics - Renderer Contracts
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/render_target.h>

namespace armigo::graphics {

enum class CompareFunction : std::uint8_t {
    Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always
};
enum class BlendOperation : std::uint8_t { Add, Subtract, ReverseSubtract, Min, Max };
enum class BlendFactor : std::uint8_t {
    Zero, One, SourceColor, OneMinusSourceColor, SourceAlpha, OneMinusSourceAlpha,
    DestinationColor, OneMinusDestinationColor, DestinationAlpha,
    OneMinusDestinationAlpha, ConstantColor, OneMinusConstantColor,
    ConstantAlpha, OneMinusConstantAlpha, SourceAlphaSaturate,
    Source1Color, OneMinusSource1Color, Source1Alpha, OneMinusSource1Alpha
};
struct BlendEquation {
    BlendOperation operation = BlendOperation::Add;
    BlendFactor source = BlendFactor::One, destination = BlendFactor::Zero;
    bool operator==(const BlendEquation&) const = default;
};
struct ColorBlendState {
    bool enabled = false;
    BlendEquation rgb, alpha;
    std::uint8_t write_mask = 15;
    bool operator==(const ColorBlendState&) const = default;
};
enum class StencilOperation : std::uint8_t {
    Keep, Zero, Replace, IncrementClamp, DecrementClamp, Invert,
    IncrementWrap, DecrementWrap
};
struct StencilFaceState {
    CompareFunction compare = CompareFunction::Always;
    StencilOperation fail = StencilOperation::Keep;
    StencilOperation depth_fail = StencilOperation::Keep;
    StencilOperation pass = StencilOperation::Keep;
    std::uint32_t compare_mask = ~0u, write_mask = ~0u;
    bool operator==(const StencilFaceState&) const = default;
};
struct DepthStencilState {
    bool depth_test = false, depth_write = false, stencil_test = false;
    CompareFunction depth_compare = CompareFunction::Always;
    StencilFaceState front, back;
    bool operator==(const DepthStencilState&) const = default;
};
enum class CullMode : std::uint8_t { None, Front, Back, FrontAndBack };
enum class FrontFace : std::uint8_t { Clockwise, CounterClockwise };
enum class FillMode : std::uint8_t { Fill, Line, Point };
enum class DepthClipMode : std::uint8_t { Clip, Clamp };
struct RasterizerState {
    bool enabled = true;
    CullMode cull = CullMode::None;
    FrontFace front_face = FrontFace::CounterClockwise;
    FillMode front_fill = FillMode::Fill, back_fill = FillMode::Fill;
    DepthClipMode depth_clip = DepthClipMode::Clip;
    bool operator==(const RasterizerState&) const = default;
};
struct ViewportState {
    // Near/far encode the depth scale and translation. Guest depth clip
    // bounds are independent and require separate backend support.
    double x = 0, y = 0, width = 0, height = 0, near_depth = 0, far_depth = 1;
    // Set when the guest clip bounds differ from [0, 1]; only viewport 0
    // rejects them in the planner, so a draw that can select another viewport
    // must check this before relying on it.
    bool custom_depth_clip = false;
    bool operator==(const ViewportState&) const = default;
};
struct DepthBiasState {
    bool point = false, line = false, fill = false;
    // Raw Maxwell units; constant is the equivalent window-depth displacement.
    float factor = 0, units = 0, clamp = 0;
    float constant = 0;
    bool operator==(const DepthBiasState&) const = default;
};
struct CoverageState {
    std::uint32_t samples = 1, sample_mask = ~0u;
    bool multisample = false, alpha_to_coverage = false, alpha_to_one = false;
    bool operator==(const CoverageState&) const = default;
};
struct RenderState {
    // Logical output order, same as RenderTargetPlan, including holes.
    std::array<ColorBlendState, 8> color;
    DepthStencilState depth_stencil;
    RasterizerState rasterizer;
    std::vector<ViewportState> viewports;
    std::vector<RenderArea> scissors;
    std::array<float, 4> blend_constant{};
    std::uint32_t front_stencil_reference = 0, back_stencil_reference = 0;
    DepthBiasState depth_bias;
    CoverageState coverage;
    bool operator==(const RenderState&) const = default;
};

// L7: decode Maxwell words, not host enum numeric values. Return only fully
// validated state. Expand shared masks (bits 0/4/8/12) and two-sided stencil
// inheritance explicitly. Resolve shared-blend/SeparateAlpha disagreements
// before admission; unknown modes stop instead of guessing.
// Apply Y/depth/winding conversion exactly once in concert with admitted shader
// coordinate conventions. Validate finite viewport/bias/constants and checked
// signed scissor intersections. Empty scissors and viewport/scissor arrays are
// valid. Unknown swizzles, dual-source blend, logic ops, alpha test, blend
// microcode and point/unequal fill modes remain Unsupported
// until a tested backend/emulation implements them. Enum presence is not support.
[[nodiscard]] RendererResult<RenderState> plan_render_state(
    const RenderStop& stop, const RenderTargetPlan& targets);

} // namespace armigo::graphics
