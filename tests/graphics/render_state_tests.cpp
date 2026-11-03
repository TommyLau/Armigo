// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/render_state.h>
#include <armigo/graphics/texture_format.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <variant>

namespace {

using namespace armigo::graphics;

unsigned failures = 0;
#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

constexpr std::uint32_t pack_halves(std::uint32_t low, std::uint32_t high) noexcept
{
    return low | (high << 16);
}

constexpr std::uint32_t word_of(float value) noexcept
{
    return std::bit_cast<std::uint32_t>(value);
}

// A supported 1280x720 draw profile modeled on the established single-target
// admission: filled triangles, independent blending on target zero, shared
// full write masks, no depth/stencil/alpha/logic/bias/multisample features,
// counterclockwise guest winding with an upper-left raster origin.
RenderStop make_stop()
{
    RenderStop stop{};
    stop.engine = engine_class::threed;
    stop.kind = RenderKind::Draw;
    stop.original = {0x586, 3, 0, 0x2000, true};

    stop.registers[0x0df] = 1;                       // RasterizeEnable
    // ViewportTransform[0]: scale (640, -360, 0.5), translate (640, 360, 0.5),
    // positive swizzles (0x6420).
    stop.registers[0x280] = word_of(640.0F);
    stop.registers[0x281] = word_of(-360.0F);
    stop.registers[0x282] = word_of(0.5F);
    stop.registers[0x283] = word_of(640.0F);
    stop.registers[0x284] = word_of(360.0F);
    stop.registers[0x285] = word_of(0.5F);
    stop.registers[0x286] = 0x6420;
    // ViewportExtents[0]: 0..1280 x 0..720, depth 0..1.
    stop.registers[0x300] = pack_halves(0, 1280);
    stop.registers[0x301] = pack_halves(0, 720);
    stop.registers[0x302] = word_of(0.0F);
    stop.registers[0x303] = word_of(1.0F);
    stop.registers[0x35f] = 0;                       // DepthMode MinusOneToOne
    stop.registers[0x36b] = 0x1b02;                  // PolygonModeFront Fill
    stop.registers[0x36c] = 0x1b02;                  // PolygonModeBack Fill
    stop.registers[0x374] = 0;                       // BlendUcodeEnable off
    stop.registers[0x380] = 1;                       // ScissorState[0] enable word
    stop.registers[0x381] = pack_halves(0, 0xffff);  // disabled full-range pattern
    stop.registers[0x382] = pack_halves(0, 0xffff);
    stop.registers[0x3e4] = 1;                       // RtColorMaskShared
    stop.registers[0x3fd] = pack_halves(0, 1280);    // ScreenScissorState
    stop.registers[0x3fe] = pack_halves(0, 720);
    stop.registers[0x4b3] = 0;                       // DepthTestEnable
    stop.registers[0x4b9] = 1;                       // BlendIndependent
    stop.registers[0x4ba] = 0;                       // DepthWriteEnable
    stop.registers[0x4bb] = 0;                       // AlphaTestEnable
    stop.registers[0x4c3] = 8;                       // DepthTestFunc Always
    stop.registers[0x4c7] = word_of(0.125F);         // BlendConstant
    stop.registers[0x4c8] = word_of(0.25F);
    stop.registers[0x4c9] = word_of(0.5F);
    stop.registers[0x4ca] = word_of(0.75F);
    stop.registers[0x4d7] = 0;                       // BlendEnableCommon
    stop.registers[0x4d8] = 1;                       // BlendEnable[0]
    stop.registers[0x4e0] = 0;                       // StencilTestState.Enable
    stop.registers[0x4eb] = 0;                       // YControl
    stop.registers[0x54d] = 0;                       // MultisampleEnable
    stop.registers[0x54f] = 0;                       // MultisampleControl
    stop.registers[0x55b] = word_of(0.0F);           // DepthBiasFactor
    stop.registers[0x565] = 0;                       // StencilBackTestState.TwoSided
    stop.registers[0x56f] = word_of(0.0F);           // DepthBiasUnits
    stop.registers[0x574] = 0;                       // RtMsaaMode 1x1
    stop.registers[0x61f] = word_of(0.0F);           // DepthBiasClamp
    stop.registers[0x646] = 0;                       // CullEnable
    stop.registers[0x647] = 0x901;                   // FrontFace CCW
    stop.registers[0x648] = 0x405;                   // CullFace Back
    stop.registers[0x64b] = 1;                       // ViewportTransformEnable
    stop.registers[0x64f] = 0x181d;                  // ViewVolumeClipControl profile
    stop.registers[0x671] = 0;                       // LogicOpState.Enable
    stop.registers[0x680] = 0x1111;                  // RtColorMask[0]
    // BlendState[0]: SeparateAlpha 0, Add, SrcAlpha, OneMinusSrcAlpha twice.
    stop.registers[0x780] = 0;
    stop.registers[0x781] = 1;
    stop.registers[0x782] = 5;
    stop.registers[0x783] = 6;
    stop.registers[0x784] = 1;
    stop.registers[0x785] = 5;
    stop.registers[0x786] = 6;
    return stop;
}

RenderTargetPlan make_plan(TextureFormat format = TextureFormat::R8G8B8A8Unorm)
{
    RenderTargetPlan plan;
    AttachmentPlan color;
    color.physical_target = 0;
    color.texture.format = format;
    color.texture.width = 1280;
    color.texture.height = 720;
    plan.colors[0] = color;
    plan.area = {0, 0, 1280, 720};
    plan.samples = 1;
    return plan;
}

RenderTargetPlan make_decoded_plan(std::uint32_t encoded_format)
{
    RenderStop stop{};
    stop.engine = engine_class::threed;
    stop.kind = RenderKind::Draw;
    stop.registers[0x487] = 1;
    stop.registers[0x574] = 0;
    stop.registers[0x200] = 0;
    stop.registers[0x201] = 0x100000;
    stop.registers[0x202] = 8;
    stop.registers[0x203] = 8;
    stop.registers[0x204] = encoded_format;
    stop.registers[0x205] = 0;
    stop.registers[0x206] = 1;
    TargetDescription& target = stop.color_targets[0];
    target.address = 0x100000;
    target.width = 8;
    target.height = 8;
    target.format = encoded_format;
    target.depth = 1;

    const auto result = plan_render_targets(stop);
    CHECK(std::holds_alternative<RenderTargetPlan>(result));
    return std::holds_alternative<RenderTargetPlan>(result)
        ? std::get<RenderTargetPlan>(result)
        : RenderTargetPlan{};
}

std::optional<RenderState> require_state(const RendererResult<RenderState>& result)
{
    CHECK(std::holds_alternative<RenderState>(result));
    if (std::holds_alternative<RenderState>(result)) {
        return std::get<RenderState>(result);
    }
    return std::nullopt;
}

void check_rejected(const RendererResult<RenderState>& result, std::uint32_t method = 0)
{
    CHECK(std::holds_alternative<RendererError>(result));
    if (std::holds_alternative<RendererError>(result)) {
        const CommandResult command = std::get<RendererError>(result).command();
        CHECK(!command.ok());
        CHECK(command.status == CommandStatus::Unsupported);
        CHECK(command.engine == engine_class::threed);
        if (method != 0) {
            CHECK(command.method == method);
        }
    }
}

// ---- Default profile ------------------------------------------------------

void test_default_profile()
{
    const auto state = require_state(plan_render_state(make_stop(), make_plan()));
    if (!state.has_value()) {
        return;
    }

    CHECK(state->rasterizer.enabled);
    CHECK(state->rasterizer.cull == CullMode::None);
    // Negative viewport Y cancels the upper-left origin winding reversal.
    CHECK(state->rasterizer.front_face == FrontFace::CounterClockwise);
    CHECK(state->rasterizer.front_fill == FillMode::Fill);
    CHECK(state->rasterizer.back_fill == FillMode::Fill);
    CHECK(state->rasterizer.depth_clip == DepthClipMode::Clip);

    CHECK(!state->depth_stencil.depth_test);
    CHECK(!state->depth_stencil.depth_write);
    CHECK(state->depth_stencil.depth_compare == CompareFunction::Always);
    CHECK(!state->depth_stencil.stencil_test);
    CHECK(state->depth_stencil.front.compare == CompareFunction::Always);
    CHECK(state->depth_stencil.front.fail == StencilOperation::Keep);
    CHECK(state->depth_stencil.back == state->depth_stencil.front);
    CHECK(state->front_stencil_reference == 0);
    CHECK(state->back_stencil_reference == 0);

    CHECK(state->color[0].enabled);
    CHECK(state->color[0].rgb == (BlendEquation{BlendOperation::Add,
                                                BlendFactor::SourceAlpha,
                                                BlendFactor::OneMinusSourceAlpha}));
    CHECK(state->color[0].alpha == state->color[0].rgb);
    CHECK(state->color[0].write_mask == 15);
    for (std::size_t target = 1; target < 8; ++target) {
        CHECK(!state->color[target].enabled);
        CHECK(state->color[target].write_mask == 15);
    }

    const std::array blend_constant{0.125F, 0.25F, 0.5F, 0.75F};
    CHECK(state->blend_constant == blend_constant);

    CHECK(state->viewports.size() == 16);
    CHECK(state->viewports[0] ==
          (ViewportState{0.0, 0.0, 1280.0, 720.0, 0.0, 1.0}));
    CHECK(state->scissors.size() == 16);
    CHECK(state->scissors[0] == (RenderArea{0, 0, 1280, 720}));

    CHECK(!state->depth_bias.point && !state->depth_bias.line && !state->depth_bias.fill);
    CHECK(state->coverage.samples == 1);
    CHECK(!state->coverage.multisample);
    CHECK(!state->coverage.alpha_to_coverage);
    CHECK(!state->coverage.alpha_to_one);
}

// ---- Blend decode tables --------------------------------------------------

void test_blend_factor_table()
{
    // Maxwell words, including the GL alias encodings, name one Armigo factor.
    constexpr std::array table{
        std::pair{0x1u, BlendFactor::Zero},
        std::pair{0x4000u, BlendFactor::Zero},
        std::pair{0x2u, BlendFactor::One},
        std::pair{0x4001u, BlendFactor::One},
        std::pair{0x3u, BlendFactor::SourceColor},
        std::pair{0x4300u, BlendFactor::SourceColor},
        std::pair{0x4u, BlendFactor::OneMinusSourceColor},
        std::pair{0x4301u, BlendFactor::OneMinusSourceColor},
        std::pair{0x5u, BlendFactor::SourceAlpha},
        std::pair{0x4302u, BlendFactor::SourceAlpha},
        std::pair{0x6u, BlendFactor::OneMinusSourceAlpha},
        std::pair{0x4303u, BlendFactor::OneMinusSourceAlpha},
        std::pair{0x7u, BlendFactor::DestinationAlpha},
        std::pair{0x4304u, BlendFactor::DestinationAlpha},
        std::pair{0x8u, BlendFactor::OneMinusDestinationAlpha},
        std::pair{0x4305u, BlendFactor::OneMinusDestinationAlpha},
        std::pair{0x9u, BlendFactor::DestinationColor},
        std::pair{0x4306u, BlendFactor::DestinationColor},
        std::pair{0xau, BlendFactor::OneMinusDestinationColor},
        std::pair{0x4307u, BlendFactor::OneMinusDestinationColor},
        std::pair{0xbu, BlendFactor::SourceAlphaSaturate},
        std::pair{0x4308u, BlendFactor::SourceAlphaSaturate},
        std::pair{0xc001u, BlendFactor::ConstantColor},
        std::pair{0xc002u, BlendFactor::OneMinusConstantColor},
        std::pair{0xc003u, BlendFactor::ConstantAlpha},
        std::pair{0xc004u, BlendFactor::OneMinusConstantAlpha},
    };
    for (const auto& [word, expected] : table) {
        RenderStop stop = make_stop();
        // Keep SeparateAlpha clear by writing the factor into all four slots.
        stop.registers[0x782] = word;  // color source
        stop.registers[0x783] = word;  // color destination
        stop.registers[0x785] = word;  // alpha source
        stop.registers[0x786] = word;  // alpha destination
        const auto state = require_state(plan_render_state(stop, make_plan()));
        if (!state.has_value()) {
            continue;
        }
        CHECK(state->color[0].rgb.source == expected);
        CHECK(state->color[0].rgb.destination == expected);
        CHECK(state->color[0].alpha.source == expected);
        CHECK(state->color[0].alpha.destination == expected);
    }
}

void test_blend_operation_table()
{
    constexpr std::array table{
        std::pair{0x1u, BlendOperation::Add},
        std::pair{0x8006u, BlendOperation::Add},
        std::pair{0x2u, BlendOperation::Subtract},
        std::pair{0x800au, BlendOperation::Subtract},
        std::pair{0x3u, BlendOperation::ReverseSubtract},
        std::pair{0x800bu, BlendOperation::ReverseSubtract},
        std::pair{0x4u, BlendOperation::Min},
        std::pair{0x8007u, BlendOperation::Min},
        std::pair{0x5u, BlendOperation::Max},
        std::pair{0x8008u, BlendOperation::Max},
    };
    for (const auto& [word, expected] : table) {
        RenderStop stop = make_stop();
        stop.registers[0x781] = word;  // color operation
        stop.registers[0x784] = word;  // alpha operation
        const auto state = require_state(plan_render_state(stop, make_plan()));
        if (!state.has_value()) {
            continue;
        }
        CHECK(state->color[0].rgb.operation == expected);
        CHECK(state->color[0].alpha.operation == expected);
    }
}

void test_blend_unknown_encodings_rejected()
{
    for (const std::uint32_t word : {0x0u, 0xcu, 0x14u, 0x4309u, 0xc005u, 0x8005u}) {
        RenderStop stop = make_stop();
        stop.registers[0x782] = word;
        check_rejected(plan_render_state(stop, make_plan()));
    }
    for (const std::uint32_t word : {0x0u, 0x6u, 0x8009u}) {
        RenderStop stop = make_stop();
        stop.registers[0x781] = word;
        check_rejected(plan_render_state(stop, make_plan()));
    }
}

void test_blend_dual_source_rejected_when_enabled()
{
    // Both the raw and GL alias encodings of every second-source factor stop
    // while blending is enabled.
    for (const std::uint32_t word : {0x10u, 0x11u, 0x12u, 0x13u, 0xc900u, 0xc901u, 0xc902u, 0xc903u}) {
        RenderStop stop = make_stop();
        stop.registers[0x782] = word;
        stop.registers[0x785] = word;
        check_rejected(plan_render_state(stop, make_plan()));
    }
    // With blending disabled the factors are inert: the plan accepts the
    // words without reaching the dual-source rejection.
    RenderStop stop = make_stop();
    stop.registers[0x4d8] = 0;
    stop.registers[0x782] = 0x10;
    stop.registers[0x785] = 0x10;
    const auto state = require_state(plan_render_state(stop, make_plan()));
    if (state.has_value()) {
        CHECK(!state->color[0].enabled);
    }
}

void test_independent_blend_across_targets()
{
    RenderStop stop = make_stop();
    // Even targets blend with distinct equations, odd targets stay disabled.
    for (std::size_t target = 0; target < 8; ++target) {
        const bool enabled = (target % 2) == 0;
        stop.registers[0x4d8 + target] = enabled ? 1 : 0;
        const std::size_t base = 0x780 + target * 8;
        stop.registers[base] = 1;  // SeparateAlpha set: independent alpha words
        stop.registers[base + 1] = static_cast<std::uint32_t>(1 + (target % 5));
        // Words 1-6 map linearly onto the Armigo factor order.
        stop.registers[base + 2] = static_cast<std::uint32_t>(1 + (target % 6));
        stop.registers[base + 3] = 0x4001;  // One
        stop.registers[base + 4] = static_cast<std::uint32_t>(1 + ((target + 2) % 5));
        stop.registers[base + 5] = 0x1;  // Zero
        stop.registers[base + 6] = 0xb;  // SourceAlphaSaturate
    }
    const auto state = require_state(plan_render_state(stop, make_plan()));
    if (!state.has_value()) {
        return;
    }
    for (std::size_t target = 0; target < 8; ++target) {
        const ColorBlendState& blend = state->color[target];
        CHECK(blend.enabled == ((target % 2) == 0));
        if (!blend.enabled) {
            // Disabled targets keep interface defaults; words are inert.
            CHECK(blend.rgb == BlendEquation{});
            continue;
        }
        const std::size_t base = 0x780 + target * 8;
        CHECK(blend.rgb.operation ==
              static_cast<BlendOperation>(stop.registers[base + 1] - 1));
        CHECK(blend.rgb.source == static_cast<BlendFactor>(stop.registers[base + 2] - 1));
        CHECK(blend.rgb.destination == BlendFactor::One);
        CHECK(blend.alpha.operation ==
              static_cast<BlendOperation>(stop.registers[base + 4] - 1));
        CHECK(blend.alpha.source == BlendFactor::Zero);
        CHECK(blend.alpha.destination == BlendFactor::SourceAlphaSaturate);
    }
}

void test_shared_blend_replicates_and_requires_agreement()
{
    RenderStop stop = make_stop();
    stop.registers[0x4b9] = 0;  // shared blending
    stop.registers[0x4cf] = 1;      // BlendStateCommon.SeparateAlpha
    stop.registers[0x4d0] = 3;      // ReverseSubtract
    stop.registers[0x4d1] = 0x4303; // OneMinusSourceAlpha
    stop.registers[0x4d2] = 0x4302; // SourceAlpha
    stop.registers[0x4d3] = 4;      // alpha Min
    stop.registers[0x4d4] = 0x4001; // alpha source One
    stop.registers[0x4d6] = 0x4000; // shared layout alpha destination at +7
    stop.registers[0x4d7] = 1;
    for (std::size_t target = 0; target < 8; ++target) {
        stop.registers[0x4d8 + target] = 1;
    }
    const auto state = require_state(plan_render_state(stop, make_plan()));
    if (state.has_value()) {
        for (std::size_t target = 0; target < 8; ++target) {
            CHECK(state->color[target].enabled);
            CHECK(state->color[target].rgb ==
                  (BlendEquation{BlendOperation::ReverseSubtract,
                                 BlendFactor::OneMinusSourceAlpha,
                                 BlendFactor::SourceAlpha}));
            CHECK(state->color[target].alpha ==
                  (BlendEquation{BlendOperation::Min,
                                 BlendFactor::One,
                                 BlendFactor::Zero}));
        }
    }

    // Shared mode with one per-target enable disagreeing is ambiguous.
    RenderStop disagreement = stop;
    disagreement.registers[0x4da] = 0;
    check_rejected(plan_render_state(disagreement, make_plan()), 0x4d7);
}

void test_separate_alpha_decisions()
{
    // SeparateAlpha set: alpha words may differ from color words.
    RenderStop separate = make_stop();
    separate.registers[0x780] = 1;
    separate.registers[0x784] = 5;  // alpha Max
    separate.registers[0x785] = 0x4000;
    separate.registers[0x786] = 0x4001;
    const auto state = require_state(plan_render_state(separate, make_plan()));
    if (state.has_value()) {
        CHECK(state->color[0].rgb.operation == BlendOperation::Add);
        CHECK(state->color[0].alpha.operation == BlendOperation::Max);
        CHECK(state->color[0].alpha.source == BlendFactor::Zero);
        CHECK(state->color[0].alpha.destination == BlendFactor::One);
    }

    // SeparateAlpha clear: the alpha words still define the alpha equation,
    // in both decoded paths.
    RenderStop clear = make_stop();
    clear.registers[0x784] = 5;  // alpha Max
    const auto words = require_state(plan_render_state(clear, make_plan()));
    if (words.has_value()) {
        CHECK(words->color[0].rgb.operation == BlendOperation::Add);
        CHECK(words->color[0].alpha.operation == BlendOperation::Max);
    }
}

void test_color_write_masks()
{
    // Shared masks route target zero's word to every slot; channels sit at
    // bits 0, 4, 8 and 12, and bits above the alpha channel are ignored.
    RenderStop stop = make_stop();
    stop.registers[0x3e4] = 1;
    stop.registers[0x680] = 0x1001 | 0x10000;  // R and A, plus a stray high bit
    const auto shared = require_state(plan_render_state(stop, make_plan()));
    if (shared.has_value()) {
        for (std::size_t target = 0; target < 8; ++target) {
            CHECK(shared->color[target].write_mask == 0x9);
        }
    }

    // Independent masks decode each word.
    RenderStop independent = make_stop();
    independent.registers[0x3e4] = 0;
    independent.registers[0x680] = 0x1;
    independent.registers[0x681] = 0x10;
    independent.registers[0x682] = 0x100;
    independent.registers[0x683] = 0x1000;
    independent.registers[0x684] = 0x0;
    const auto per_target = require_state(plan_render_state(independent, make_plan()));
    if (per_target.has_value()) {
        CHECK(per_target->color[0].write_mask == 0x1);
        CHECK(per_target->color[1].write_mask == 0x2);
        CHECK(per_target->color[2].write_mask == 0x4);
        CHECK(per_target->color[3].write_mask == 0x8);
        CHECK(per_target->color[4].write_mask == 0x0);
    }
}

void test_blend_constant_and_finiteness()
{
    RenderStop stop = make_stop();
    stop.registers[0x4c9] = word_of(std::nanf(""));
    check_rejected(plan_render_state(stop, make_plan()), 0x4c9);
}

void test_no_alpha_blend_factor_filter()
{
    // A three-component target must not read destination alpha.
    RenderStop stop = make_stop();
    stop.registers[0x780] = 1;      // SeparateAlpha, keep alpha words distinct
    stop.registers[0x782] = 0x7;    // DestinationAlpha source
    stop.registers[0x783] = 0x8;    // OneMinusDestinationAlpha destination
    stop.registers[0x784] = 1;      // alpha Add
    stop.registers[0x785] = 0x5;    // SourceAlpha
    stop.registers[0x786] = 0x6;

    const auto filtered = require_state(plan_render_state(stop, make_plan(TextureFormat::R11G11B10Float)));
    if (filtered.has_value()) {
        CHECK(filtered->color[0].rgb.source == BlendFactor::One);
        CHECK(filtered->color[0].rgb.destination == BlendFactor::Zero);
    }

    // Four-component targets read destination alpha unchanged.
    const auto kept = require_state(plan_render_state(stop, make_plan(TextureFormat::R8G8B8A8Unorm)));
    if (kept.has_value()) {
        CHECK(kept->color[0].rgb.source == BlendFactor::DestinationAlpha);
        CHECK(kept->color[0].rgb.destination == BlendFactor::OneMinusDestinationAlpha);
    }

    // Holes have no format facts, so the factors pass through unfiltered.
    RenderTargetPlan hole_plan = make_plan(TextureFormat::R8G8B8A8Unorm);
    hole_plan.colors[0] = std::nullopt;
    const auto holed = require_state(plan_render_state(stop, hole_plan));
    if (holed.has_value()) {
        CHECK(holed->color[0].rgb.source == BlendFactor::DestinationAlpha);
    }

    // Padding channels in RGBX encodings are not destination alpha even when
    // the host storage format has four components.
    for (const std::uint32_t encoded : {0xc3u, 0xc4u, 0xc5u, 0xceu, 0xf9u, 0xfau}) {
        const auto rgbx = require_state(plan_render_state(stop, make_decoded_plan(encoded)));
        if (rgbx.has_value()) {
            CHECK(rgbx->color[0].rgb.source == BlendFactor::One);
            CHECK(rgbx->color[0].rgb.destination == BlendFactor::Zero);
        }
    }

    // The Maxwell 0xc6 encoding carries alpha and must retain the factors.
    const auto rgba16 = require_state(plan_render_state(stop, make_decoded_plan(0xc6)));
    if (rgba16.has_value()) {
        CHECK(rgba16->color[0].rgb.source == BlendFactor::DestinationAlpha);
        CHECK(rgba16->color[0].rgb.destination == BlendFactor::OneMinusDestinationAlpha);
    }
}

// ---- Depth and stencil ----------------------------------------------------

void test_depth_state_table()
{
    constexpr std::array table{
        std::pair{0x1u, CompareFunction::Never},
        std::pair{0x2u, CompareFunction::Less},
        std::pair{0x3u, CompareFunction::Equal},
        std::pair{0x4u, CompareFunction::LessEqual},
        std::pair{0x5u, CompareFunction::Greater},
        std::pair{0x6u, CompareFunction::NotEqual},
        std::pair{0x7u, CompareFunction::GreaterEqual},
        std::pair{0x8u, CompareFunction::Always},
        std::pair{0x200u, CompareFunction::Never},
        std::pair{0x207u, CompareFunction::Always},
    };
    for (const auto& [word, expected] : table) {
        RenderStop stop = make_stop();
        stop.registers[0x4b3] = 1;  // the compare is only observable when enabled
        stop.registers[0x4c3] = word;
        const auto state = require_state(plan_render_state(stop, make_plan()));
        if (!state.has_value()) {
            continue;
        }
        CHECK(state->depth_stencil.depth_compare == expected);
    }

    // With the test disabled the function stays at defaults whatever the word.
    RenderStop disabled = make_stop();
    disabled.registers[0x4b3] = 0;
    disabled.registers[0x4c3] = 0;
    const auto inert = require_state(plan_render_state(disabled, make_plan()));
    if (inert.has_value()) {
        CHECK(inert->depth_stencil.depth_compare == CompareFunction::Always);
    }

    RenderStop enabled = make_stop();
    enabled.registers[0x4b3] = 1;
    enabled.registers[0x4ba] = 1;
    const auto state = require_state(plan_render_state(enabled, make_plan()));
    if (state.has_value()) {
        CHECK(state->depth_stencil.depth_test);
        CHECK(state->depth_stencil.depth_write);
    }
    RenderStop unknown = make_stop();
    unknown.registers[0x4b3] = 1;
    unknown.registers[0x4c3] = 0x9;
    check_rejected(plan_render_state(unknown, make_plan()), 0x4c3);
}

void test_stencil_two_sided_inheritance()
{
    RenderStop stop = make_stop();
    stop.registers[0x4e0] = 1;                 // stencil test enabled
    stop.registers[0x4e1] = 0x1e00;            // front fail Keep (GL alias)
    stop.registers[0x4e2] = 2;                 // front depth fail Zero
    stop.registers[0x4e3] = 3;                 // front pass Replace
    stop.registers[0x4e4] = 4;                 // front compare LessEqual
    stop.registers[0x4e5] = 7;                 // front reference
    stop.registers[0x4e6] = 0xff;              // front compare mask
    stop.registers[0x4e7] = 0xf0;              // front write mask
    const auto single = require_state(plan_render_state(stop, make_plan()));
    if (single.has_value()) {
        CHECK(single->depth_stencil.stencil_test);
        CHECK(single->depth_stencil.front.fail == StencilOperation::Keep);
        CHECK(single->depth_stencil.front.depth_fail == StencilOperation::Zero);
        CHECK(single->depth_stencil.front.pass == StencilOperation::Replace);
        CHECK(single->depth_stencil.front.compare == CompareFunction::LessEqual);
        CHECK(single->depth_stencil.front.compare_mask == 0xff);
        CHECK(single->depth_stencil.front.write_mask == 0xf0);
        CHECK(single->front_stencil_reference == 7);
        // Two-sided clear: the back face inherits all front state.
        CHECK(single->depth_stencil.back == single->depth_stencil.front);
        CHECK(single->back_stencil_reference == 7);
    }

    RenderStop two_sided = stop;
    two_sided.registers[0x565] = 1;            // TwoSided
    two_sided.registers[0x566] = 0x8507;       // back fail IncrementWrap
    two_sided.registers[0x567] = 0x150a;       // back depth fail Invert
    two_sided.registers[0x568] = 8;            // back pass DecrementWrap
    two_sided.registers[0x569] = 0x205;        // back compare NotEqual (GL alias)
    two_sided.registers[0x3d5] = 3;            // back reference
    two_sided.registers[0x3d6] = 0x33;         // back write mask
    two_sided.registers[0x3d7] = 0x77;         // back compare mask
    const auto back = require_state(plan_render_state(two_sided, make_plan()));
    if (back.has_value()) {
        CHECK(back->depth_stencil.back.fail == StencilOperation::IncrementWrap);
        CHECK(back->depth_stencil.back.depth_fail == StencilOperation::Invert);
        CHECK(back->depth_stencil.back.pass == StencilOperation::DecrementWrap);
        CHECK(back->depth_stencil.back.compare == CompareFunction::NotEqual);
        CHECK(back->depth_stencil.back.compare_mask == 0x77);
        CHECK(back->depth_stencil.back.write_mask == 0x33);
        CHECK(back->back_stencil_reference == 3);
        CHECK(back->depth_stencil.front.compare == CompareFunction::LessEqual);
    }

    RenderStop unknown = make_stop();
    unknown.registers[0x4e0] = 1;
    unknown.registers[0x4e1] = 0x9;
    check_rejected(plan_render_state(unknown, make_plan()), 0x4e0);
}

// ---- Rasterizer -----------------------------------------------------------

void test_cull_and_winding_table()
{
    struct CullCase {
        std::uint32_t enable;
        std::uint32_t face;
        CullMode expected;
    };
    constexpr std::array cull_table{
        CullCase{0, 0x404, CullMode::None},
        CullCase{0, 0x405, CullMode::None},
        CullCase{0, 0x408, CullMode::None},
        CullCase{1, 0x404, CullMode::Front},
        CullCase{1, 0x405, CullMode::Back},
        CullCase{1, 0x408, CullMode::FrontAndBack},
    };
    for (const auto& test : cull_table) {
        RenderStop stop = make_stop();
        stop.registers[0x646] = test.enable;
        stop.registers[0x648] = test.face;
        const auto state = require_state(plan_render_state(stop, make_plan()));
        if (state.has_value()) {
            CHECK(state->rasterizer.cull == test.expected);
        }
    }
    RenderStop unknown_cull = make_stop();
    unknown_cull.registers[0x646] = 1;
    unknown_cull.registers[0x648] = 0x406;
    check_rejected(plan_render_state(unknown_cull, make_plan()), 0x648);

    // Negative viewport Y cancels the origin flip when TriangleRastFlip is
    // clear; setting it leaves the viewport reversal alone.
    struct WindingCase {
        std::uint32_t guest;
        std::uint32_t y_control;
        FrontFace expected;
    };
    constexpr std::array winding_table{
        WindingCase{0x900, 0x0, FrontFace::Clockwise},
        WindingCase{0x901, 0x0, FrontFace::CounterClockwise},
        WindingCase{0x900, 0x10, FrontFace::CounterClockwise},
        WindingCase{0x901, 0x10, FrontFace::Clockwise},
    };
    for (const auto& test : winding_table) {
        RenderStop stop = make_stop();
        stop.registers[0x647] = test.guest;
        stop.registers[0x4eb] = test.y_control;
        const auto state = require_state(plan_render_state(stop, make_plan()));
        if (state.has_value()) {
            CHECK(state->rasterizer.front_face == test.expected);
        }
    }
    // Origin, viewport signs and YNegate each reverse orientation once.
    for (bool clockwise : {false, true})
        for (bool origin_flip : {false, true})
            for (bool negative_x : {false, true})
                for (bool negative_y : {false, true})
                    for (bool negate_y : {false, true}) {
                        auto stop = make_stop();
                        stop.registers[0x647] = clockwise ? 0x900 : 0x901;
                        stop.registers[0x4eb] = (origin_flip ? 0x10u : 0u) | (negate_y ? 1u : 0u);
                        stop.registers[0x280] = word_of(negative_x ? -640.f : 640.f);
                        stop.registers[0x281] = word_of(negative_y ? -360.f : 360.f);
                        const auto state = require_state(plan_render_state(stop, make_plan()));
                        if (state) {
                            const bool native_clockwise = clockwise ^ !origin_flip ^ negative_x ^ negative_y ^ negate_y;
                            CHECK(state->rasterizer.front_face == (native_clockwise
                                ? FrontFace::Clockwise : FrontFace::CounterClockwise));
                        }
                    }
    RenderStop unknown_face = make_stop();
    unknown_face.registers[0x647] = 0x902;
    check_rejected(plan_render_state(unknown_face, make_plan()), 0x647);
}

void test_fill_modes_unsupported()
{
    RenderStop unequal = make_stop();
    unequal.registers[0x36b] = 0x1b02;
    unequal.registers[0x36c] = 0x1b01;
    check_rejected(plan_render_state(unequal, make_plan()), 0x36c);

    // Equal line modes map to native Metal triangle line rasterization.
    RenderStop lines = make_stop();
    lines.registers[0x36b] = 0x1b01;
    lines.registers[0x36c] = 0x1b01;
    auto line_result = plan_render_state(lines, make_plan());
    const auto* line_state = std::get_if<RenderState>(&line_result);
    CHECK(line_state && line_state->rasterizer.front_fill == FillMode::Line);

    RenderStop points = make_stop();
    points.registers[0x36b] = 0x1b00;
    points.registers[0x36c] = 0x1b00;
    check_rejected(plan_render_state(points, make_plan()));

    RenderStop unknown = make_stop();
    unknown.registers[0x36b] = 0x1b03;
    check_rejected(plan_render_state(unknown, make_plan()), 0x36b);
}

void test_depth_clip_and_raster_discard()
{
    RenderStop clip = make_stop();
    clip.registers[0x64f] = 0x181d;  // DepthClampDisabled set
    auto state = require_state(plan_render_state(clip, make_plan()));
    if (state.has_value()) {
        CHECK(state->rasterizer.depth_clip == DepthClipMode::Clip);
    }

    RenderStop clamp = make_stop();
    clamp.registers[0x64f] = 0x1001;  // undocumented profile bits, clamp enabled
    state = require_state(plan_render_state(clamp, make_plan()));
    if (state.has_value()) {
        CHECK(state->rasterizer.depth_clip == DepthClipMode::Clamp);
    }

    RenderStop discard = make_stop();
    discard.registers[0xdf] = 0;
    state = require_state(plan_render_state(discard, make_plan()));
    if (state.has_value()) {
        CHECK(!state->rasterizer.enabled);
    }
}

// ---- Coverage -------------------------------------------------------------

void test_coverage_table()
{
    struct MsaaCase {
        std::uint32_t mode;
        std::uint32_t samples;
        bool supported;
    };
    constexpr std::array table{
        MsaaCase{0, 1, true},
        MsaaCase{1, 0, false},
        MsaaCase{2, 4, true},
        MsaaCase{3, 0, false},
        MsaaCase{4, 8, true},
        MsaaCase{5, 2, true},
        MsaaCase{6, 16, true},
        MsaaCase{7, 0, false},
    };
    for (const auto& test : table) {
        RenderStop stop = make_stop();
        stop.registers[0x574] = test.mode;
        if (!test.supported) {
            check_rejected(plan_render_state(stop, make_plan()), 0x574);
            continue;
        }
        const auto state = require_state(plan_render_state(stop, make_plan()));
        if (state.has_value()) {
            CHECK(state->coverage.samples == test.samples);
        }
    }

    RenderStop controls = make_stop();
    controls.registers[0x54d] = 1;
    controls.registers[0x54f] = 0x11;  // alpha to coverage | alpha to one
    const auto state = require_state(plan_render_state(controls, make_plan()));
    if (state.has_value()) {
        CHECK(state->coverage.multisample);
        CHECK(state->coverage.alpha_to_coverage);
        CHECK(state->coverage.alpha_to_one);
        CHECK(state->coverage.sample_mask == ~0u);
    }
}

// ---- Explicit unsupported features ---------------------------------------

void test_explicit_unsupported_features()
{
    RenderStop alpha_test = make_stop();
    alpha_test.registers[0x4bb] = 1;
    check_rejected(plan_render_state(alpha_test, make_plan()), 0x4bb);

    RenderStop logic_op = make_stop();
    logic_op.registers[0x671] = 1;
    check_rejected(plan_render_state(logic_op, make_plan()), 0x671);

    RenderStop blend_ucode = make_stop();
    blend_ucode.registers[0x374] = 1;
    check_rejected(plan_render_state(blend_ucode, make_plan()), 0x374);

    RenderStop bias_nan = make_stop();
    bias_nan.registers[0x55b] = word_of(std::nanf(""));
    check_rejected(plan_render_state(bias_nan, make_plan()), 0x55b);

    RenderStop depth_mode = make_stop();
    depth_mode.registers[0x35f] = 1;  // zero-to-one convention
    check_rejected(plan_render_state(depth_mode, make_plan()), 0x35f);

    RenderStop wrong_engine = make_stop();
    wrong_engine.engine = engine_class::compute;
    check_rejected(plan_render_state(wrong_engine, make_plan()));
}

void test_depth_bias()
{
    for (const auto format : {TextureFormat::D16Unorm, TextureFormat::D24UnormS8Uint,
                             TextureFormat::X8UintD24Unorm, TextureFormat::S8UintD24Unorm,
                             TextureFormat::D32Float, TextureFormat::D32FloatS8Uint}) {
        auto plan = make_plan();
        plan.depth.emplace();
        plan.depth->texture.format = format;
        auto stop = make_stop();
        stop.registers[0x370] = 3;
        stop.registers[0x371] = 2; // Boolean32 ignores all but bit zero.
        stop.registers[0x372] = 1;
        stop.registers[0x55b] = word_of(-1.25f);
        stop.registers[0x56f] = word_of(32.0f);
        stop.registers[0x61f] = word_of(-0.005f);
        const auto state = require_state(plan_render_state(stop, plan));
        if (state) {
            CHECK(state->depth_bias.point && !state->depth_bias.line && state->depth_bias.fill);
            CHECK(state->depth_bias.units == 32.0f);
            CHECK(state->depth_bias.factor == -1.25f);
            CHECK(state->depth_bias.clamp == -0.005f);
            CHECK(state->depth_bias.constant == std::ldexp(32.0f,
                format == TextureFormat::D16Unorm ? -16 : -24));
        }
        stop.registers[0x444] = 1;
        if (format == TextureFormat::D32Float || format == TextureFormat::D32FloatS8Uint)
            check_rejected(plan_render_state(stop, plan), 0x444);
        else
            CHECK(std::holds_alternative<RenderState>(plan_render_state(stop, plan)));
        stop.registers[0x56f] = 0; // Slope-only does not use the constant representation.
        CHECK(std::holds_alternative<RenderState>(plan_render_state(stop, plan)));
        stop.registers[0x444] = 2;
        check_rejected(plan_render_state(stop, plan), 0x444);
    }
    for (const auto method : {0x370, 0x371}) {
        auto stop = make_stop();
        stop.registers[method] = 1;
        check_rejected(plan_render_state(stop, make_plan()), 0x370);
    }
    auto stop = make_stop();
    stop.registers[0x372] = 1;
    stop.registers[0x56f] = word_of(16.0f);
    // A null attachment cannot turn stale format/control registers into bias.
    stop.registers[0x444] = ~0u;
    const auto null_depth = require_state(plan_render_state(stop, make_plan()));
    if (null_depth) CHECK(null_depth->depth_bias.constant == 0.0f);
    stop.registers[0x36b] = stop.registers[0x36c] = 0x1b01;
    check_rejected(plan_render_state(stop, make_plan()), 0x370);
    for (const auto method : {0x55b, 0x56f, 0x61f}) {
        auto invalid = make_stop();
        invalid.registers[method] = word_of(INFINITY);
        check_rejected(plan_render_state(invalid, make_plan()), 0x55b);
    }
}

// ---- Viewports ------------------------------------------------------------

void test_viewport_transforms()
{
    // Secondary viewport decodes independently of the primary, including its
    // own depth extents.
    RenderStop stop = make_stop();
    stop.registers[0x280 + 3 * 8] = word_of(100.0F);   // ScaleX
    stop.registers[0x281 + 3 * 8] = word_of(50.0F);    // ScaleY
    stop.registers[0x282 + 3 * 8] = word_of(0.5F);     // ScaleZ
    stop.registers[0x283 + 3 * 8] = word_of(200.0F);   // TranslateX
    stop.registers[0x284 + 3 * 8] = word_of(300.0F);   // TranslateY
    stop.registers[0x285 + 3 * 8] = word_of(0.5F);     // TranslateZ
    stop.registers[0x30e] = word_of(0.0F);             // Extents[3] clip near
    stop.registers[0x30f] = word_of(1.0F);             // Extents[3] clip far
    const auto state = require_state(plan_render_state(stop, make_plan()));
    if (!state.has_value()) {
        return;
    }
    CHECK(state->viewports[3] == (ViewportState{100.0, 250.0, 200.0, 100.0, 0.0, 1.0}));

    // Y negate flips the effective Y scale, mirroring the window rectangle.
    RenderStop negate = make_stop();
    negate.registers[0x4eb] = 1;
    const auto flipped = require_state(plan_render_state(negate, make_plan()));
    if (flipped.has_value()) {
        // ScaleY -360 negated to 360: rectangle y becomes 360 - 360 = 0 with
        // height 720 again, matching translate minus scale.
        CHECK(flipped->viewports[0] == (ViewportState{0.0, 0.0, 1280.0, 720.0, 0.0, 1.0}));
    }

    // Positive Y scale emits the same window interval with positive extents.
    RenderStop positive = make_stop();
    positive.registers[0x281] = word_of(360.0F);
    const auto upright = require_state(plan_render_state(positive, make_plan()));
    if (upright.has_value()) {
        CHECK(upright->viewports[0] == (ViewportState{0.0, 0.0, 1280.0, 720.0, 0.0, 1.0}));
    }

    // Subregion: translate (900, 540), scale (280, -240).
    RenderStop subregion = make_stop();
    subregion.registers[0x280] = word_of(280.0F);
    subregion.registers[0x281] = word_of(-240.0F);
    subregion.registers[0x283] = word_of(900.0F);
    subregion.registers[0x284] = word_of(540.0F);
    const auto region = require_state(plan_render_state(subregion, make_plan()));
    if (region.has_value()) {
        CHECK(region->viewports[0] == (ViewportState{620.0, 300.0, 560.0, 480.0, 0.0, 1.0}));
    }
}

void test_viewport_swizzles_rejected()
{
    // Positive encodings are the even values; each negative component stops
    // (X, Y, Z and W negatives are the odd encodings 1, 3, 5 and 7).
    for (const std::uint32_t word : {0x6421u, 0x6423u, 0x6430u, 0x6450u, 0x6470u}) {
        RenderStop stop = make_stop();
        stop.registers[0x286] = word;
        check_rejected(plan_render_state(stop, make_plan()), 0x286);
    }
}

void test_viewport_depth_ranges()
{
    // Clip bounds are independent of the fixed admitted depth transform.
    // Until the backend clips them independently, non-default bounds stop
    // instead of remapping guest depth through a Metal viewport.
    RenderStop swap = make_stop();
    swap.registers[0x282] = word_of(-0.5F);
    const auto swapped = require_state(plan_render_state(swap, make_plan()));
    if (swapped.has_value()) {
        CHECK(swapped->viewports[0].near_depth == 1.0);
        CHECK(swapped->viewports[0].far_depth == 0.0);
    }

    RenderStop transform = make_stop();
    transform.registers[0x282] = word_of(0.25F);
    const auto transformed = require_state(plan_render_state(transform, make_plan()));
    if (transformed.has_value()) {
        CHECK(transformed->viewports[0].near_depth == 0.25);
        CHECK(transformed->viewports[0].far_depth == 0.75);
    }

    RenderStop bounded = make_stop();
    bounded.registers[0x302] = word_of(0.25F);
    bounded.registers[0x303] = word_of(0.75F);
    check_rejected(plan_render_state(bounded, make_plan()), 0x302);

    // Unused viewport slots keep reset values (a 1x1 transform with clip
    // bounds 0/0); they plan, are flagged, and do not stop viewport 0.
    RenderStop reset_slots = make_stop();
    for (std::uint32_t viewport = 1; viewport < 16; ++viewport) {
        const std::uint32_t transform_base = 0x280 + viewport * 8;
        reset_slots.registers[transform_base + 0] = word_of(0.5F);
        reset_slots.registers[transform_base + 1] = word_of(-0.5F);
        reset_slots.registers[transform_base + 2] = word_of(0.5F);
        reset_slots.registers[transform_base + 3] = word_of(0.5F);
        reset_slots.registers[transform_base + 4] = word_of(0.5F);
        reset_slots.registers[transform_base + 5] = word_of(0.5F);
        reset_slots.registers[transform_base + 6] = 0x6420;
        reset_slots.registers[0x300 + viewport * 4 + 2] = word_of(0.0F);
        reset_slots.registers[0x300 + viewport * 4 + 3] = word_of(0.0F);
    }
    const auto reset_state = require_state(plan_render_state(reset_slots, make_plan()));
    if (reset_state.has_value()) {
        CHECK(!reset_state->viewports[0].custom_depth_clip);
        CHECK(reset_state->viewports.size() > 1 && reset_state->viewports[1].custom_depth_clip);
    }

    RenderStop outside = make_stop();
    outside.registers[0x303] = word_of(1.5F);
    check_rejected(plan_render_state(outside, make_plan()), 0x302);

    RenderStop inverted = make_stop();
    inverted.registers[0x302] = word_of(0.75F);
    inverted.registers[0x303] = word_of(0.25F);
    check_rejected(plan_render_state(inverted, make_plan()));

    RenderStop nan = make_stop();
    nan.registers[0x280] = word_of(std::nanf(""));
    check_rejected(plan_render_state(nan, make_plan()));

    RenderStop zero = make_stop();
    zero.registers[0x280] = 0;
    check_rejected(plan_render_state(zero, make_plan()));
}

void test_viewport_transform_disable()
{
    RenderStop stop = make_stop();
    stop.registers[0x64b] = 0;
    const auto state = require_state(plan_render_state(stop, make_plan()));
    if (state.has_value()) {
        for (std::size_t viewport = 0; viewport < 16; ++viewport) {
            CHECK(state->viewports[viewport] ==
                  (ViewportState{0.0, 0.0, 1280.0, 720.0, 0.0, 1.0}));
        }
    }

    RenderStop empty = make_stop();
    empty.registers[0x64b] = 0;
    empty.registers[0x3fd] = 0;
    empty.registers[0x3fe] = 0;
    check_rejected(plan_render_state(empty, make_plan()), 0x3fd);
}

// ---- Scissors -------------------------------------------------------------

void test_scissor_intersection()
{
    // Enabled scissor clips the viewport.
    RenderStop stop = make_stop();
    stop.registers[0x381] = pack_halves(10, 2000);  // X1=10, X2=2000
    stop.registers[0x382] = pack_halves(20, 300);   // Y1=20, Y2=300
    auto state = require_state(plan_render_state(stop, make_plan()));
    if (state.has_value()) {
        CHECK(state->scissors[0] == (RenderArea{10, 20, 1270, 280}));
    }

    // Scissor entirely outside the viewport is an empty area, not a wrap.
    RenderStop outside = make_stop();
    outside.registers[0x381] = pack_halves(2000, 3000);
    outside.registers[0x382] = pack_halves(0, 100);
    state = require_state(plan_render_state(outside, make_plan()));
    if (state.has_value()) {
        CHECK(state->scissors[0] == (RenderArea{2000, 0, 0, 100}));
    }

    // Inverted extents decode as empty coverage on that axis.
    RenderStop inverted = make_stop();
    inverted.registers[0x381] = pack_halves(100, 50);
    inverted.registers[0x382] = pack_halves(0, 10);
    state = require_state(plan_render_state(inverted, make_plan()));
    if (state.has_value()) {
        CHECK(state->scissors[0].width == 0);
        CHECK(state->scissors[0].height == 10);
    }

    // Per-viewport scissors stay independent; unselected viewports hold zero
    // extents, so their intersections are empty.
    RenderStop second = make_stop();
    second.registers[0x380 + 4] = 1;
    second.registers[0x381 + 4] = pack_halves(0, 500);
    second.registers[0x382 + 4] = pack_halves(0, 500);
    state = require_state(plan_render_state(second, make_plan()));
    if (state.has_value()) {
        CHECK(state->scissors[0] == (RenderArea{0, 0, 1280, 720}));
        CHECK(state->scissors[1] == (RenderArea{0, 0, 0, 0}));
        CHECK(state->scissors[2] == (RenderArea{0, 0, 0, 0}));
    }
}

void test_surface_clip_intersection()
{
    RenderStop stop = make_stop();
    stop.registers[0x3fd] = pack_halves(2, 4);
    stop.registers[0x3fe] = pack_halves(1, 5);
    const auto state = require_state(plan_render_state(stop, make_plan()));
    if (state.has_value()) {
        CHECK(state->scissors[0] == (RenderArea{2, 1, 4, 5}));
    }
}

void test_fractional_viewport_scissor()
{
    RenderStop stop = make_stop();
    stop.registers[0x280] = word_of(3.75F);
    stop.registers[0x283] = word_of(4.0F);
    stop.registers[0x281] = word_of(-3.75F);
    stop.registers[0x284] = word_of(4.0F);
    stop.registers[0x3fd] = pack_halves(0, 8);
    stop.registers[0x3fe] = pack_halves(0, 8);
    const auto state = require_state(plan_render_state(stop, make_plan()));
    if (state.has_value()) {
        CHECK(state->viewports[0].x == 0.25);
        CHECK(state->viewports[0].y == 0.25);
        CHECK(state->scissors[0] == (RenderArea{0, 0, 8, 8}));
    }
}

void test_scissor_y_negate_mirror()
{
    RenderStop stop = make_stop();
    stop.registers[0x4eb] = 1;  // NegateY
    stop.registers[0x381] = pack_halves(10, 100);
    stop.registers[0x382] = pack_halves(20, 200);  // height 180
    // y mirrors to 720 - 180 - 20 = 520.
    auto state = require_state(plan_render_state(stop, make_plan()));
    if (state.has_value()) {
        CHECK(state->scissors[0] == (RenderArea{10, 520, 90, 180}));
    }

    // Mirroring below zero clamps at the top edge and shrinks the height.
    RenderStop clamped = make_stop();
    clamped.registers[0x4eb] = 1;
    clamped.registers[0x381] = pack_halves(0, 100);
    clamped.registers[0x382] = pack_halves(700, 880);  // height 180 -> clamp
    state = require_state(plan_render_state(clamped, make_plan()));
    if (state.has_value()) {
        CHECK(state->scissors[0] == (RenderArea{0, 0, 100, 20}));
    }
}

// ---- Independent plan slot handling ---------------------------------------

void test_holes_keep_register_decoding()
{
    // Attachment holes do not disturb the logical-output blend and mask
    // decoding: slot one's state comes from its own registers.
    RenderStop stop = make_stop();
    stop.registers[0x3e4] = 0;  // independent masks
    stop.registers[0x4d8] = 0;  // hole slot zero does not blend
    stop.registers[0x4d9] = 1;  // BlendEnable[1]
    stop.registers[0x788] = 1;  // BlendState[1] SeparateAlpha
    stop.registers[0x789] = 5;  // Max
    stop.registers[0x78a] = 0x4302;
    stop.registers[0x78b] = 0x4303;
    stop.registers[0x78c] = 5;
    stop.registers[0x78d] = 0x4302;
    stop.registers[0x78e] = 0x4303;
    stop.registers[0x681] = 0x100;  // write mask blue only

    RenderTargetPlan plan = make_plan();
    plan.colors[1] = plan.colors[0];
    plan.colors[1]->physical_target = 3;
    plan.colors[0] = std::nullopt;  // hole at logical slot zero

    const auto state = require_state(plan_render_state(stop, plan));
    if (state.has_value()) {
        CHECK(!state->color[0].enabled);
        CHECK(state->color[0].write_mask == 15);  // its own independent word
        CHECK(state->color[1].enabled);
        CHECK(state->color[1].rgb.operation == BlendOperation::Max);
        CHECK(state->color[1].write_mask == 0x4);
    }
}

}  // namespace

int main()
{
    test_default_profile();
    test_blend_factor_table();
    test_blend_operation_table();
    test_blend_unknown_encodings_rejected();
    test_blend_dual_source_rejected_when_enabled();
    test_independent_blend_across_targets();
    test_shared_blend_replicates_and_requires_agreement();
    test_separate_alpha_decisions();
    test_color_write_masks();
    test_blend_constant_and_finiteness();
    test_no_alpha_blend_factor_filter();
    test_depth_state_table();
    test_stencil_two_sided_inheritance();
    test_cull_and_winding_table();
    test_fill_modes_unsupported();
    test_depth_clip_and_raster_discard();
    test_coverage_table();
    test_explicit_unsupported_features();
    test_depth_bias();
    test_viewport_transforms();
    test_viewport_swizzles_rejected();
    test_viewport_depth_ranges();
    test_viewport_transform_disable();
    test_scissor_intersection();
    test_surface_clip_intersection();
    test_fractional_viewport_scissor();
    test_scissor_y_negate_mirror();
    test_holes_keep_register_decoding();

    if (failures != 0) {
        std::fprintf(stderr, "%u checks failed\n", failures);
        return 1;
    }
    std::printf("render_state tests passed\n");
    return 0;
}
