// ============================================================================
// Armigo Services - Synchronous Graphics Presentation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/image_readback.h>
#include <armigo/services/graphics_buffer_backing.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace armigo::services {

struct GraphicsFence {
    std::int32_t syncpoint = -1;
    std::uint32_t threshold = 0;
};

struct GraphicsPresentation {
    GraphicsBufferBacking backing;
    // A native producer supplies an already synchronized, owned RGBA8 image.
    // Native frames are paced by VI before reaching the shared presenter.
    std::shared_ptr<const graphics::Rgba8Image> native_pixels;
    // Reads the presented native texture back to RGBA8 on demand. The presenter
    // calls it only for frames it keeps, so an ordinary present hands the GPU
    // frame to the display without a CPU copy.
    std::function<graphics::Rgba8ReadbackResult(std::array<std::int32_t, 4>)>
        native_read_pixels;
    std::uint64_t native_texture_serial = 0;
    // The presented platform texture (an id<MTLTexture> on Metal), valid once
    // the frame's fence has completed. A windowed presenter on the same device
    // draws it without the CPU copy.
    std::shared_ptr<const void> native_texture;
    // Opaque submission context retains the captured allocation and its GPU order.
    std::shared_ptr<void> native_submission;
    // Ordered native consumers validate this after GPU/ordered CPU completion,
    // before visible presentation or successful frame observation. It owns its
    // captures and caches the producer result across consumer/VI checks.
    std::function<bool()> native_producer_ready;
    bool native_vsync_paced = false;
    std::uint32_t slot = 0;
    std::uint64_t frame_number = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t pitch = 0;
    std::uint32_t block_height_log2 = 0;
    std::uint64_t surface_offset = 0;
    std::uint64_t surface_size = 0;
    std::int64_t timestamp = 0;
    bool auto_timestamp = false;
    std::array<std::int32_t, 4> crop{};
    std::uint32_t scaling_mode = 0;
    std::uint32_t transform = 0;
    std::uint32_t sticky_transform = 0;
    std::int32_t swap_interval = 0;
    std::uint32_t fence_count = 0;
    std::array<GraphicsFence, 4> producer_fences{};
};

struct GraphicsPresentationResult {
    bool completed = false;
    std::string diagnostic;
    // Host-stop provenance for this callback only; never a guest error code.
    bool cancelled = false;
};

// The callback runs synchronously without the queue mutex. It must not re-enter
// VI. Success means all producer fences and consumer GPU/readback work finished;
// only then may the service release the buffer with NoFence. The callback owns
// host pacing policy and must honor the supplied swap interval explicitly.
using GraphicsPresenter =
    std::function<GraphicsPresentationResult(const GraphicsPresentation&)>;

}  // namespace armigo::services
