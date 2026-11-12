// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic coverage for the windowed presenter path:
// an authored GPU image is presented to a real CAMetalLayer drawable through
// the application-owned presenter. The test creates a real window when a
// display session is available and skips cleanly otherwise (reported on
// stdout), mirroring the lane H acceptance rule.

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <armigo/platform/application.h>

#include <atomic>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <variant>

namespace armigo::platform {
std::uint64_t presenter_test_presented_count(const Presenter& presenter);
} // namespace armigo::platform

namespace {

using namespace armigo::platform;
namespace graphics = armigo::graphics;

unsigned failures = 0;

#define CHECK(...)                                                          \
    do {                                                                    \
        if (!(__VA_ARGS__)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); \
            ++failures;                                                     \
        }                                                                   \
    } while (false)

// Pumps AppKit events so window ordering, drawable acquisition and presented
// callbacks make progress while the test waits.
void pump_until(const std::chrono::steady_clock::time_point& deadline, const std::function<bool()>& done) {
    while (!done() && std::chrono::steady_clock::now() < deadline) {
        @autoreleasepool {
            NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                untilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]
                                                   inMode:NSDefaultRunLoopMode
                                                  dequeue:YES];
            if (event) [NSApp sendEvent:event];
        }
    }
}

graphics::Rgba8Image gradient_image(std::uint32_t width, std::uint32_t height) {
    graphics::Rgba8Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(std::size_t{width} * height * 4);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            auto* pixel = image.pixels.data() + (std::size_t{y} * width + x) * 4;
            pixel[0] = static_cast<std::byte>((x * 13) & 0xFF);
            pixel[1] = static_cast<std::byte>((y * 5) & 0xFF);
            pixel[2] = static_cast<std::byte>(((x + y) * 9) & 0xFF);
            pixel[3] = static_cast<std::byte>(0xFF);
        }
    }
    return image;
}

} // namespace

int main() {
    @autoreleasepool {
        PresenterOptions options;
        options.mode = PresentationMode::windowed;
        options.width = 64;
        options.height = 48;
        std::atomic<bool> close_requested_seen = false;
        ApplicationCallbacks callbacks;
        callbacks.close_requested = [&close_requested_seen] {
            // Models the shell reacting to a close: nonblocking, never
            // destroying the application from inside the callback.
            close_requested_seen.store(true);
        };
        ApplicationCreation creation = create_application(options, callbacks);
        if (const auto* failure = std::get_if<PresentationFailure>(&creation)) {
            if (failure->code == PresentationError::unsupported) {
                // No display session, no windowserver access, or no Metal 4
                // device: the windowed acceptance cannot run here. Reported,
                // not hidden behind a fake success.
                std::printf("SKIP: windowed presenter unavailable: %s\n", failure->detail.c_str());
                return 0;
            }
            std::fprintf(stderr, "FAIL: application creation failed: %s\n", failure->detail.c_str());
            ++failures;
            return 1;
        }
        Application& application = *std::get<std::unique_ptr<Application>>(creation);
        // An active application lets the window server actually composite the
        // presented drawable.
        [NSApp activateIgnoringOtherApps:YES];

        std::shared_ptr<Presenter> presenter = application.presenter();
        CHECK(presenter);
        if (!presenter) return 1;
        CHECK(presenter->mode() == PresentationMode::windowed);

        graphics::Rgba8Image image = gradient_image(64, 48);
        PresentationFrame frame;
        frame.queued.frame_number = 5;
        frame.readback = false;
        frame.pixels = std::make_shared<const graphics::Rgba8Image>(std::move(image));
        const PresentResult result = presenter->present(frame, std::chrono::steady_clock::now() +
                                                                     std::chrono::seconds(10), {});
        const auto* presented = std::get_if<PresentedFrame>(&result);
        if (const auto* failed = std::get_if<PresentationFailure>(&result))
            std::fprintf(stderr, "present failure: code=%d detail=%s\n", int(failed->code),
                         failed->detail.c_str());
        CHECK(presented);
        if (presented) {
            CHECK(presented->frame_number == 5);
            // An ordinary windowed frame consumes the drawable without a readback.
            CHECK(!presented->readback.has_value());
        }

        frame.queued.frame_number = 6;
        frame.readback = true;
        const auto captured = presenter->present(frame, std::chrono::steady_clock::now() + std::chrono::seconds(10), {});
        const auto* captured_frame = std::get_if<PresentedFrame>(&captured);
        CHECK(captured_frame && captured_frame->readback);
        if (captured_frame && captured_frame->readback)
            CHECK(captured_frame->readback->pixels == frame.pixels->pixels);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
            width:64 height:48 mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
        CHECK(texture != nil);
        if (texture) {
            auto bgra = frame.pixels->pixels;
            for (std::size_t offset = 0; offset < bgra.size(); offset += 4) std::swap(bgra[offset], bgra[offset + 2]);
            [texture replaceRegion:MTLRegionMake2D(0, 0, 64, 48) mipmapLevel:0 withBytes:bgra.data() bytesPerRow:64 * 4];
            PresentationFrame native;
            native.queued.frame_number = 7;
            native.queued.width = 64; native.queued.height = 48;
            native.queued.native_texture = std::shared_ptr<const void>((__bridge_retained const void*)texture,
                [](const void* value) { CFRelease(value); });
            const auto read = presenter->present(native, std::chrono::steady_clock::now() + std::chrono::seconds(10), {});
            const auto* completed = std::get_if<PresentedFrame>(&read);
            CHECK(completed && completed->readback);
            if (completed && completed->readback) CHECK(completed->readback->pixels == frame.pixels->pixels);
        }

        // The presented handler is the compositor's own acknowledgement. It
        // can only fire when the window server actually composites this
        // window; a session that keeps the window uncomposited can never
        // deliver it, which is an environment limit, not a presenter one.
        NSWindow* window = nil;
        for (NSWindow* candidate : [NSApp windows]) {
            if ([candidate.contentView.layer isKindOfClass:[CAMetalLayer class]]) window = candidate;
            for (NSView* subview in candidate.contentView.subviews)
                if ([subview.layer isKindOfClass:[CAMetalLayer class]]) window = candidate;
        }
        const BOOL composited = (window.occlusionState & NSWindowOcclusionStateVisible) != 0;
        pump_until(std::chrono::steady_clock::now() + std::chrono::seconds(3), [&] {
            return presenter_test_presented_count(*presenter) >= 1;
        });
        if (composited) CHECK(presenter_test_presented_count(*presenter) >= 1);
        else std::printf("SKIP: window is not composited in this session; presented observation unavailable\n");

        const PresenterStatus drained = presenter->drain(std::chrono::steady_clock::now() +
                                                         std::chrono::seconds(10));
        CHECK(std::holds_alternative<std::monostate>(drained));

        application.request_quit();
    }
    if (failures) {
        std::fprintf(stderr, "%u windowed presenter checks failed\n", failures);
        return 1;
    }
    std::printf("windowed presenter checks passed\n");
    return 0;
}
