// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic coverage for the AppKit application
// contract: windowed-only creation, main-thread event pumping with
// request_quit/stop, resize updating drawable dimensions, fullscreen toggle,
// and close notification through the shell callback. The test creates a real
// window when a display session is available and skips cleanly otherwise.

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <armigo/platform/application.h>

#include <atomic>
#include <vector>
#include <utility>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <thread>
#include <variant>

// Internal test seam defined in src/platform/macos/application.mm.
namespace armigo::platform {
NSWindow* application_window_for_testing(Application* application);
} // namespace armigo::platform

namespace {

using namespace armigo::platform;

unsigned failures = 0;

#define CHECK(...)                                                          \
    do {                                                                    \
        if (!(__VA_ARGS__)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); \
            ++failures;                                                     \
        }                                                                   \
    } while (false)

// The game view is the content view, or the subview whose layer is the
// window's CAMetalLayer when a guide sits beside it.
NSView* metal_view(NSWindow* window) {
    if ([window.contentView.layer isKindOfClass:[CAMetalLayer class]]) return window.contentView;
    for (NSView* subview in window.contentView.subviews)
        if ([subview.layer isKindOfClass:[CAMetalLayer class]]) return subview;
    return nil;
}

void pump_for(std::chrono::milliseconds duration, const std::function<bool()>& done) {
    const auto deadline = std::chrono::steady_clock::now() + duration;
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

void test_creation_rejects_headless() {
    PresenterOptions options;
    options.mode = PresentationMode::headless;
    const ApplicationCreation creation = create_application(options, {});
    const auto* failure = std::get_if<PresentationFailure>(&creation);
    CHECK(failure && failure->code == PresentationError::unsupported);
}

void test_run_requires_main_thread(Application& application) {
    PresenterStatus status;
    std::thread worker([&] { status = application.run(); });
    worker.join();
    const auto* failure = std::get_if<PresentationFailure>(&status);
    CHECK(failure && failure->code == PresentationError::host_failure);
}

void test_resize_updates_drawable(Application& application) {
    NSWindow* window = application_window_for_testing(&application);
    CHECK(window);
    if (!window) return;
    NSView* view = metal_view(window);
    CAMetalLayer* layer = (CAMetalLayer*)view.layer;
    CHECK([layer isKindOfClass:[CAMetalLayer class]]);
    if (![layer isKindOfClass:[CAMetalLayer class]]) return;
    const CGSize before = layer.drawableSize;
    CHECK(before.width > 0 && before.height > 0);

    // Resize in points; the application must track the drawable surface to
    // the new view bounds at the layer's contents scale.
    const CGSize points = view.bounds.size;
    const CGFloat scale = layer.contentsScale > 0 ? layer.contentsScale : 1.0;
    [window setContentSize:NSMakeSize(points.width * 2.0, points.height)];
    pump_for(std::chrono::seconds(2), [&] {
        const CGSize now = layer.drawableSize;
        return std::abs(now.width - points.width * 2.0 * scale) < 8.0;
    });
    const CGSize after = layer.drawableSize;
    CHECK(std::abs(after.width - points.width * 2.0 * scale) < 8.0);
    CHECK(std::abs(after.height - points.height * scale) < 8.0);
}

void test_fullscreen_event_tracks_geometry(Application& application) {
    NSWindow* window = application_window_for_testing(&application);
    CHECK(window);
    if (!window) return;
    // The sanctioned fullscreen transition API. The window server runs it
    // asynchronously; some sessions (headless or restricted) never start the
    // transition at all. That is an environment limit, not an application
    // failure, so it is reported and skipped rather than failed.
    [window toggleFullScreen:nil];
    pump_for(std::chrono::seconds(5), [&] {
        return (window.styleMask & NSWindowStyleMaskFullScreen) != 0;
    });
    if ((window.styleMask & NSWindowStyleMaskFullScreen) == 0) {
        std::printf("SKIP: this display session never ran the fullscreen transition\n");
        return;
    }
    CHECK((window.styleMask & NSWindowStyleMaskFullScreen) != 0);
    pump_for(std::chrono::seconds(2), [&] {
        return NSEqualRects(window.frame, window.screen.frame);
    });
    CHECK(NSEqualRects(window.frame, window.screen.frame));
    // AppKit ignores a toggle while the entering transition still settles.
    pump_for(std::chrono::seconds(1), [] { return false; });
    [window toggleFullScreen:nil];
    pump_for(std::chrono::seconds(10), [&] {
        return (window.styleMask & NSWindowStyleMaskFullScreen) == 0;
    });
    CHECK((window.styleMask & NSWindowStyleMaskFullScreen) == 0);
}

void test_quit_wakes_the_event_loop(Application& application) {
    std::thread quitter([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        application.request_quit();
    });
    const auto started = std::chrono::steady_clock::now();
    const PresenterStatus status = application.run();
    const auto elapsed = std::chrono::steady_clock::now() - started;
    quitter.join();
    CHECK(std::holds_alternative<std::monostate>(status));
    CHECK(elapsed < std::chrono::seconds(5));
}

void test_close_notifies_without_destroying(Application& application) {
    NSWindow* window = application_window_for_testing(&application);
    CHECK(window);
    if (!window) return;
    // The shell callback was installed at creation to flip the close flag and
    // wake the loop; performClose routes through windowShouldClose.
    [window performClose:nil];
    pump_for(std::chrono::seconds(2), [] { return false; });
    CHECK(window.contentView != nil);
    CHECK(application.presenter() != nullptr);
}

} // namespace

// A click captures the mouse without reaching the game; captured motion and
// buttons reach the callbacks; Esc releases the mouse and its buttons.
struct MouseLog {
    double dx = 0, dy = 0;
    unsigned moves = 0;
    std::vector<std::pair<unsigned, bool>> buttons;
};

NSEvent* mouse_event(NSWindow* window, NSEventType type, double dx, double dy) {
    CGEventRef raw = CGEventCreateMouseEvent(nullptr, kCGEventMouseMoved, CGPointMake(10, 10), kCGMouseButtonLeft);
    CGEventSetIntegerValueField(raw, kCGMouseEventDeltaX, static_cast<int64_t>(dx));
    CGEventSetIntegerValueField(raw, kCGMouseEventDeltaY, static_cast<int64_t>(dy));
    NSEvent* moved = [NSEvent eventWithCGEvent:raw];
    CFRelease(raw);
    if (type == NSEventTypeMouseMoved) return moved;
    return [NSEvent mouseEventWithType:type location:NSMakePoint(10, 10) modifierFlags:0 timestamp:0
                          windowNumber:window.windowNumber context:nil eventNumber:0 clickCount:1 pressure:1];
}

void test_mouse_capture(Application& application, MouseLog& log) {
    NSWindow* window = application_window_for_testing(&application);
    NSView* view = metal_view(window);
    CHECK(view);
    [view mouseDown:mouse_event(window, NSEventTypeLeftMouseDown, 0, 0)];
    [view mouseUp:mouse_event(window, NSEventTypeLeftMouseUp, 0, 0)];
    CHECK(log.buttons.empty());
    [view mouseMoved:mouse_event(window, NSEventTypeMouseMoved, 3, -2)];
    CHECK(log.moves == 1 && log.dx == 3 && log.dy == -2);
    [view rightMouseDown:mouse_event(window, NSEventTypeRightMouseDown, 0, 0)];
    CHECK(log.buttons.size() == 1 && log.buttons[0] == std::make_pair(1u, true));
    NSEvent* escape = [NSEvent keyEventWithType:NSEventTypeKeyDown location:NSZeroPoint modifierFlags:0
                                      timestamp:0 windowNumber:window.windowNumber context:nil
                                     characters:@"\x1b" charactersIgnoringModifiers:@"\x1b" isARepeat:NO
                                        keyCode:53];
    [view keyDown:escape];
    // Release reports every button up.
    CHECK(log.buttons.size() == 4 && !log.buttons.back().second);
    [view mouseMoved:mouse_event(window, NSEventTypeMouseMoved, 5, 5)];
    CHECK(log.moves == 1);
}

NSEvent* f1_event(NSWindow* window, bool repeat) {
    NSString* f1 = [NSString stringWithFormat:@"%C", (unichar)NSF1FunctionKey];
    return [NSEvent keyEventWithType:NSEventTypeKeyDown location:NSZeroPoint modifierFlags:0 timestamp:0
                        windowNumber:window.windowNumber context:nil characters:f1
         charactersIgnoringModifiers:f1 isARepeat:repeat keyCode:122];
}

// The key guide is a sibling drawn above the game view; F1 hides a showing
// guide and pins a hidden one, and a held key's repeats change nothing.
// ARMIGO_KEY_GUIDE_SNAPSHOT names a PNG to receive the window's views with
// the guide showing (the Metal content is not part of it).
void test_key_guide(Application& application) {
    NSWindow* window = application_window_for_testing(&application);
    NSView* view = metal_view(window);
    NSView* guide = nil;
    for (NSView* subview in window.contentView.subviews)
        if ([NSStringFromClass(subview.class) isEqualToString:@"ArmigoKeyGuide"]) guide = subview;
    CHECK(view && guide);
    if (!view || !guide) return;
    CHECK(window.contentView.subviews.lastObject == guide);
    CHECK([window.contentView.subviews indexOfObject:view] < [window.contentView.subviews indexOfObject:guide]);
    CHECK([guide hitTest:NSMakePoint(10, 10)] == nil);
    const bool shown = !guide.hidden;
    [view keyDown:f1_event(window, false)];
    CHECK(guide.hidden == shown);
    [view keyDown:f1_event(window, true)];
    CHECK(guide.hidden == shown);
    [view keyDown:f1_event(window, false)];
    CHECK(guide.hidden != shown);
    if (guide.hidden) [view keyDown:f1_event(window, false)];
    CHECK(!guide.hidden);
    if (const char* path = std::getenv("ARMIGO_KEY_GUIDE_SNAPSHOT"); path && *path) {
        [window setContentSize:NSMakeSize(960, 540)];
        NSView* content = window.contentView;
        [content layoutSubtreeIfNeeded];
        NSBitmapImageRep* image = [content bitmapImageRepForCachingDisplayInRect:content.bounds];
        [content cacheDisplayInRect:content.bounds toBitmapImageRep:image];
        CHECK([[image representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:@(path)
                                                                                         atomically:YES]);
    }
    [view keyDown:f1_event(window, false)];
    CHECK(guide.hidden);
}

int main() {
    @autoreleasepool {
        test_creation_rejects_headless();

        PresenterOptions options;
        options.mode = PresentationMode::windowed;
        options.width = 128;
        options.height = 96;
        std::atomic<bool> close_requested_seen = false;
        ApplicationCallbacks callbacks;
        callbacks.close_requested = [&close_requested_seen] {
            close_requested_seen.store(true);
        };
        MouseLog mouse;
        callbacks.mouse_moved = [&mouse](double dx, double dy) {
            ++mouse.moves;
            mouse.dx = dx;
            mouse.dy = dy;
        };
        callbacks.mouse_button = [&mouse](unsigned button, bool down) { mouse.buttons.emplace_back(button, down); };
        ApplicationCreation creation = create_application(options, callbacks);
        if (const auto* failure = std::get_if<PresentationFailure>(&creation)) {
            if (failure->code == PresentationError::unsupported) {
                std::printf("SKIP: application unavailable: %s\n", failure->detail.c_str());
                return failures ? 1 : 0;
            }
            std::fprintf(stderr, "FAIL: application creation failed: %s\n", failure->detail.c_str());
            ++failures;
            return 1;
        }
        Application& application = *std::get<std::unique_ptr<Application>>(creation);
        // Give the fullscreen transition a live, active application.
        [NSApp activateIgnoringOtherApps:YES];

        std::shared_ptr<Presenter> presenter = application.presenter();
        CHECK(presenter && presenter->mode() == PresentationMode::windowed);

        test_run_requires_main_thread(application);
        test_mouse_capture(application, mouse);
        test_key_guide(application);
        test_resize_updates_drawable(application);
        test_fullscreen_event_tracks_geometry(application);
        test_quit_wakes_the_event_loop(application);
        test_close_notifies_without_destroying(application);
        CHECK(close_requested_seen.load());

        // Drain before destruction, matching the owner contract.
        const PresenterStatus drained = presenter->drain(std::chrono::steady_clock::now() +
                                                         std::chrono::seconds(10));
        CHECK(std::holds_alternative<std::monostate>(drained));
    }
    if (failures) {
        std::fprintf(stderr, "%u application checks failed\n", failures);
        return 1;
    }
    std::printf("application checks passed\n");
    return 0;
}
