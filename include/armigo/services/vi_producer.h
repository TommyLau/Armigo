// ============================================================================
// Armigo Services - In-Process Video Interface Producer
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/image_readback.h>
#include <armigo/horizon/event.h>
#include <armigo/services/graphics_presentation.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace armigo::services::vi {

struct NativeTexture {
    std::uint64_t serial = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::function<graphics::Rgba8ReadbackResult(std::array<std::int32_t, 4>)> read_pixels;
    // The platform texture itself (an id<MTLTexture> on Metal), kept alive by
    // this handle, so a presenter on the same device can draw it directly.
    std::shared_ptr<const void> native_texture;
    // Resolve the frame's current allocation after its rendering fence. The
    // returned snapshot owns that allocation until presentation completes.
    std::function<std::optional<NativeTexture>()> resolve;
    // Native GPU producers may capture a per-frame allocation in their CPU
    // submission stream. The returned handle resolves after the rendering
    // fence; cancel releases unused ordered work on every drop/error path.
    std::function<std::optional<NativeTexture>(std::uint64_t)> capture;
    std::function<void()> cancel;
    // Platform-specific ordered GPU publication context, owned by this frame.
    std::shared_ptr<void> native_submission;
    // The captured consumer has a reserved position after all producer GPU
    // work. Resolve and publish it before waiting for producer CPU retirement.
    bool ordered_gpu_dependencies = false;
    // Owns the exact producer result; called after ordered consumer completion.
    std::function<bool()> ordered_producer_ready;
};

struct NativeDequeue {
    std::int32_t slot = -1;
    std::uint64_t fence = 0;
};

class NativeWindowProducer {
public:
    virtual ~NativeWindowProducer() = default;
    virtual bool register_textures(std::span<const NativeTexture> textures) = 0;
    virtual bool unregister_textures() = 0;
    virtual NativeDequeue dequeue() = 0;
    virtual GraphicsPresentationResult present(
        std::int32_t slot, std::uint64_t fence,
        std::function<bool(std::uint64_t)> wait_fence,
        std::array<std::int32_t, 4> crop, int interval) = 0;
    virtual std::shared_ptr<horizon::Event> free_event() const = 0;
    // Dequeues only the first count registered textures from now on (NVN
    // window active textures); false when unsupported or out of range.
    virtual bool set_active_textures(std::size_t count) { (void)count; return false; }
    // Host pacing period for scheduling graphics workers. This query does not
    // wait, advance presentation, or alter guest-visible refresh events.
    virtual std::chrono::nanoseconds scheduling_period(int) const { return {}; }
    // The guest's requested refresh multiple, even when host waits are disabled.
    // Zero means no cadence is known. This query never advances the refresh clock.
    virtual std::chrono::nanoseconds guest_present_period(int) const { return {}; }
};

std::shared_ptr<NativeWindowProducer> find_native_window(void* native_window);
void unbind_native_window(void* native_window);

// Called on the guest's thread at the start of every present through a native
// window producer, before the frame is queued: a test driver that steps input
// with the guest's own frames. An empty function clears it.
void set_guest_present_observer(std::function<void()> observer);
// Runs the observer, if any (the producer's present calls it).
void notify_guest_present();

}  // namespace armigo::services::vi
