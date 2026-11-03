// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/platform/presenter.h>
#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/presentation_clock.h>
#include <armigo/services/graphics_buffer_backing.h>
#include <armigo/services/graphics_presentation.h>
#include <functional>
#include <memory>
#include <stop_token>

namespace armigo::runtime {

// Lane G binds one application's driver/export namespace to VI. Reuse existing
// GraphicsBufferBacking/GraphicsPresentation; no second nvmap or surface model.
// Pin resolves and revalidates export identity/owner, then pins the whole surface
// through consumer completion. Numeric export IDs alone are not global keys.
// Ioctl replies keep the client's header placement. Queue completion follows
// real consumer work. Unsupported metadata/fences are errors, never NoFence success.
struct GraphicsBackingAccess {
    services::GraphicsBufferBackingLookup lookup;
    // Offset/length are relative to backing; reject overflow and out-of-range.
    // Pin must belong to this retained export, not a later reused guest address.
    std::function<LaunchResult<GuestMemoryPin>(
        const services::GraphicsBufferBacking&, std::uint64_t, std::uint64_t)> pin;
    std::function<LaunchStatus(const services::GraphicsFence&,
        std::optional<ClockInstant>, std::stop_token)> wait_fence;
};
struct PresentationDependencies {
    GraphicsBackingAccess backing;
    std::shared_ptr<PresentationClock> clock;
    std::shared_ptr<platform::Presenter> presenter;
    // Called after completed consumer work, before VI completion is returned.
    // Borrowed only during the callback; copy owned images to retain them.
    std::function<LaunchStatus(const platform::PresentedFrame&)> observe_frame;
    // Whether the consumer must return this frame's completed pixels, asked
    // once per frame on the presentation worker before it is presented. A
    // headless native frame that is not wanted is consumed as its texture on
    // the GPU without a CPU readback. Empty reads back every frame.
    std::function<bool(std::uint64_t frame_number)> wants_readback;
};
class PresentationBinding {
public:
    virtual ~PresentationBinding() = default;
    // Returned callback retains this binding's implementation and dependencies.
    // It must not reenter VI. Sequence: validate/pin, wait real producer fences,
    // decode, pace once, present/read back, observe, then completed=true. Any
    // failure remains completed=false and the first structured failure is kept.
    // If present fails with inflight work, retain the surface pin until presenter
    // drain succeeds. A drain failure retains dependencies; destruction requires
    // successful drain. Never release ACQUIRED just because a deadline expired.
    virtual services::GraphicsPresenter callback() = 0;
    virtual std::optional<LaunchFailure> failure() const = 0;
    virtual void request_stop() noexcept = 0;
    virtual LaunchStatus drain(std::optional<ClockInstant> deadline) = 0;
};
[[nodiscard]] LaunchResult<std::shared_ptr<PresentationBinding>> bind_presentation(
    PresentationDependencies dependencies);

} // namespace armigo::runtime
