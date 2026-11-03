// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Full clears use render-pass load actions. Partial, masked, and scissored clears
// emit real operations that preserve untouched data. Layer ranges map to Metal
// slice selection and layered render passes.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#import <Metal/Metal.h>

#include <armigo/graphics/render_target.h>

namespace armigo::platform::metal {

class PassCounters;

// Native aspect targets of one planned clear. The color view serves
// color-aspect clears; the depth and stencil views serve the depth/stencil
// aspects (one packed texture may serve both). level and slice select the
// attachment's first cleared subresource; the cleared layer range comes from
// the plan's target.layer/layer_count, so array clears bind the same base
// slice and cover plan.target.layer_count consecutive slices.
struct ClearBindings {
    id<MTLTexture> color = nil;
    id<MTLTexture> depth = nil;
    id<MTLTexture> stencil = nil;
    std::uint32_t level = 0;
    std::uint32_t slice = 0;
    // When set, the clear encodes its pass into this open command buffer and
    // adds its resources to this residency set (committed by its owner)
    // instead of creating, beginning and ending its own command buffer.
    id<MTL4CommandBuffer> into = nil;
    id<MTLResidencySet> into_residency = nil;
    PassCounters* pass_counters = nullptr;
};

// Objects the encoded commands need alive through commit and feedback. The
// caller submits commands with its own commit options and feedback handler
// and must keep these retained until GPU quiescence.
struct EncodedClear {
    id<MTL4CommandBuffer> commands = nil;
    id<MTL4CommandAllocator> allocator = nil;
    NSMutableArray<id>* retained = nil;
};

// Performs the same plan/binding checks as encode_clear without allocating or
// encoding. Submission admission must call this before committing GPU intents.
[[nodiscard]] graphics::RendererStatus
validate_clear(const graphics::ClearPlan& plan,
               const ClearBindings& bindings) noexcept;

// Encodes one planned clear into a fresh Metal 4 command buffer. Full
// admitted clears fold into the load action of an empty pass; a full-range
// clear of an arrayed attachment folds into the load action of an empty
// layered pass (renderTargetArrayLength). Partial, masked, scissored or
// aspect-preserving clears encode a load-action-Load pass plus a full-screen
// draw bounded by the scissor rect and per-channel write masks (depth/stencil
// writes bounded likewise); layered draws instance one vertex per cleared
// layer, addressing slices through render_target_array_index. Colour and
// depth/stencil aspects of a combined clear encode in one pass. Commits
// nothing and waits for nothing. Unsupported or contradictory plans return a
// structured error without encoding any work.
[[nodiscard]] graphics::RendererResult<EncodedClear>
encode_clear(id<MTLDevice> device,
             const graphics::ClearPlan& plan,
             const ClearBindings& bindings) noexcept;

// One clear pipeline configuration: the clear library's vertex and fragment
// functions and the attachment state the pipeline is built for (invalid
// formats and a zero mask for aspects the clear leaves alone, and the
// attachments' sample count).
struct ClearPipelineConfig {
    std::string vertex, fragment;
    MTLPixelFormat color = MTLPixelFormatInvalid;
    MTLColorWriteMask mask = MTLColorWriteMaskNone;
    MTLPixelFormat depth = MTLPixelFormatInvalid;
    MTLPixelFormat stencil = MTLPixelFormatInvalid;
    std::uint32_t samples = 1;
    bool operator==(const ClearPipelineConfig&) const = default;
};

// Compiles the clear shader library for device ahead of the first partial
// clear. The library compiles once per device and each pipeline once per
// configuration, also when threads ask together (the others wait); a compile
// that fails is tried again by the next clear that needs it. Safe to call from
// any thread.
void prepare_clear_shaders(id<MTLDevice> device) noexcept;
// Builds one clear pipeline ahead of its first clear, from a configuration an
// earlier session recorded. False when it names no clear function or Metal
// refuses it.
bool prepare_clear_pipeline(id<MTLDevice> device, const ClearPipelineConfig& config) noexcept;

// One compile of the clear library (no configuration) or of a clear pipeline,
// or (wait) more than a millisecond spent waiting for another thread's.
struct ClearCompile {
    std::optional<ClearPipelineConfig> pipeline;
    std::uint64_t nanoseconds = 0;
    bool wait = false;
};
// The clear compiles and long waits of the calling thread since its last
// call, oldest first, for the driver's compile accounting and pipeline
// record. A compile another thread ran is reported to that thread; a thread
// that waited for it more than a millisecond reports the wait.
[[nodiscard]] std::vector<ClearCompile> take_clear_compiles();

}  // namespace armigo::platform::metal
