// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/graphics/image_readback.h>
#include <armigo/services/graphics_presentation.h>
#include <chrono>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <variant>

namespace armigo::platform {

// Lane H owns Apple objects privately. This is a synchronous host interface,
// not a Horizon ABI. Lane G validates VI metadata, waits producer fences and
// decodes through pinned backing before calling present. Readback copies a
// completed RGBA8 target only, after the caller has waited. Success requires
// actual consumer GPU/readback completion, not submission or a drawable request.
// Unsupported formats/transforms and unavailable drawables are explicit failures;
// neither a skipped frame nor cancellation permits VI to free its acquired slot.
enum class PresentationMode { headless, windowed };
struct PresenterOptions {
    PresentationMode mode = PresentationMode::headless;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool fullscreen = false;
    // Windowed only. The width of the game image the drawable serves: the
    // drawable keeps the window's shape and is no larger than an image this
    // wide fitted into it needs (0: the window's full pixel size).
    std::uint32_t content_width = 0;
    // Present in step with the display (V-Sync). Off presents as soon as a
    // frame is ready; the guest's display cadence still paces the game.
    // Always on for iOS.
    bool display_sync = true;
};
struct PresentationFrame {
    services::GraphicsPresentation queued;
    // Immutable, tightly packed RGBA8 in the admitted crop/orientation. The
    // initial bridge uploads these decoded pixels; no raw guest-pointer casts.
    // Absent when the presenter consumes queued.native_texture itself.
    std::shared_ptr<const graphics::Rgba8Image> pixels;
    // Whether the presenter must return this frame's completed pixels. A
    // headless presenter not asked for them still consumes the frame on the
    // GPU, copying a native texture into presenter-owned storage, and returns
    // no readback. Windowed readback is requested explicitly by capture policy.
    bool readback = true;
};
enum class PresentationError {
    invalid_frame, unsupported, drawable_unavailable, cancelled,
    deadline_expired, host_failure,
};
struct PresentationFailure {
    PresentationError code = PresentationError::invalid_frame;
    std::string detail;
};
struct PresentedFrame {
    std::uint64_t frame_number = 0;
    std::chrono::steady_clock::time_point completed_at{};
    // Required for a frame that asked for readback: pixels read after real
    // consumer completion. Unread frames may omit it. The image owns its bytes.
    std::optional<graphics::Rgba8Image> readback;
};
using PresentResult = std::variant<PresentationFailure, PresentedFrame>;
using PresenterStatus = std::variant<PresentationFailure, std::monostate>;
class Presenter {
public:
    virtual ~Presenter() = default;
    virtual PresentationMode mode() const noexcept = 0;
    // Actual drawable-present notifications, for performance observation.
    // Presenters without that notification report zero.
    virtual std::uint64_t presented_count() const noexcept { return 0; }
    // Serialized on the presentation worker, never under VI's queue mutex.
    // deadline bounds drawable/completion waits, not an extra pacing sleep.
    // Borrowed frame lives through the call. Failure may leave inflight work:
    // retain pixels, backing owner, staging and drawable resources internally
    // until drain succeeds. Timeout is not proof of completion. Lane G retains
    // the actual guest pin too. Destruction requires successful drain.
    virtual PresentResult present(const PresentationFrame& frame,
        std::optional<std::chrono::steady_clock::time_point> deadline,
        std::stop_token stop) = 0;
    virtual PresenterStatus drain(
        std::optional<std::chrono::steady_clock::time_point> deadline) = 0;
};
using PresenterCreation = std::variant<PresentationFailure, std::shared_ptr<Presenter>>;
// Creates a headless presenter; windowed creation uses create_application in
// application.h and is unsupported here. Success always carries a nonnull object.
// Window events/destruction stay on AppKit's main thread; presentation work must
// not block that pump. No AppKit/Metal type appears in the portable contract.
[[nodiscard]] PresenterCreation create_presenter(const PresenterOptions& options);

} // namespace armigo::platform
