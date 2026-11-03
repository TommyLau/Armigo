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
    // Host pacing period for scheduling graphics workers. This query does not
    // wait, advance presentation, or alter guest-visible refresh events.
    virtual std::chrono::nanoseconds scheduling_period(int) const { return {}; }
};

std::shared_ptr<NativeWindowProducer> find_native_window(void* native_window);
void unbind_native_window(void* native_window);

}  // namespace armigo::services::vi
