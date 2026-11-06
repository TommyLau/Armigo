// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#include <armigo/common/diagnostics.h>
// The application installs a layer-backed CAMetalLayer with display sync and
// framebuffer-only drawables, then presents without adding a pacing policy.

#include <armigo/platform/application.h>

#include "display_refresh.h"
#include "key_guide.h"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <variant>

// Objective-C declarations must live at global scope, outside the C++
// namespaces below.

// Keyboard input reaches the guest through the GameController framework, so
// the view only swallows key events (no system beep) and handles Esc, which
// releases a captured mouse. While captured, pointer motion and buttons go to
// the application's callbacks.
@interface ArmigoMetalView : NSView
@property (nonatomic, copy) void (^onMouseMoved)(double dx, double dy);
@property (nonatomic, copy) void (^onMouseButton)(unsigned button, BOOL down);
@property (nonatomic) BOOL captured;
// The button whose click captured the mouse until it is released, else -1.
@property (nonatomic) int capturingButton;
- (void)releaseMouse;
@end

@implementation ArmigoMetalView

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (self) {
        self.wantsLayer = YES;
        self.capturingButton = -1;
    }
    return self;
}

- (void)captureMouse {
    if (self.captured || (!self.onMouseMoved && !self.onMouseButton)) return;
    self.captured = YES;
    CGAssociateMouseAndMouseCursorPosition(false);
    [NSCursor hide];
}

- (void)releaseMouse {
    if (!self.captured) return;
    self.captured = NO;
    self.capturingButton = -1;
    for (unsigned button = 0; button < 3; ++button)
        if (self.onMouseButton) self.onMouseButton(button, NO);
    CGAssociateMouseAndMouseCursorPosition(true);
    [NSCursor unhide];
}

- (void)keyDown:(NSEvent*)event {
    if (event.keyCode == 53) [self releaseMouse];  // Esc
    // F1 toggles the on-screen key guide. It is not a game key. 122 is the
    // virtual key code of F1; NSF1FunctionKey is what a keyboard reports when
    // the key arrives as a function character.
    const unichar character = event.charactersIgnoringModifiers.length > 0
                                  ? [event.charactersIgnoringModifiers characterAtIndex:0]
                                  : 0;
    // Ignore repeats so holding F1 does not flicker the guide.
    if (!event.isARepeat && (event.keyCode == 122 || character == NSF1FunctionKey))
        armigo::platform::toggle_key_guide(self);
}

- (void)keyUp:(NSEvent*)event {
    (void)event;
}

- (void)flagsChanged:(NSEvent*)event {
    (void)event;
}

- (void)pointerButton:(unsigned)button down:(BOOL)down {
    // The click that captures the mouse, press and release, is not a game input.
    if (!self.captured) {
        if (down) {
            [self captureMouse];
            if (self.captured) self.capturingButton = int(button);
        }
        return;
    }
    if (!down && self.capturingButton == int(button)) {
        self.capturingButton = -1;
        return;
    }
    if (self.onMouseButton) self.onMouseButton(button, down);
}

- (void)mouseDown:(NSEvent*)event { (void)event; [self pointerButton:0 down:YES]; }
- (void)mouseUp:(NSEvent*)event { (void)event; [self pointerButton:0 down:NO]; }
- (void)rightMouseDown:(NSEvent*)event { (void)event; [self pointerButton:1 down:YES]; }
- (void)rightMouseUp:(NSEvent*)event { (void)event; [self pointerButton:1 down:NO]; }
- (void)otherMouseDown:(NSEvent*)event { (void)event; [self pointerButton:2 down:YES]; }
- (void)otherMouseUp:(NSEvent*)event { (void)event; [self pointerButton:2 down:NO]; }

- (void)pointerMoved:(NSEvent*)event {
    if (self.captured && self.onMouseMoved) self.onMouseMoved(event.deltaX, event.deltaY);
}

- (void)mouseMoved:(NSEvent*)event { [self pointerMoved:event]; }
- (void)mouseDragged:(NSEvent*)event { [self pointerMoved:event]; }
- (void)rightMouseDragged:(NSEvent*)event { [self pointerMoved:event]; }
- (void)otherMouseDragged:(NSEvent*)event { [self pointerMoved:event]; }

+ (Class)layerClass {
    return [CAMetalLayer class];
}


@end

@interface ArmigoWindowDelegate : NSObject <NSWindowDelegate>
@property (nonatomic, copy) void (^onClose)(void);
@property (nonatomic, copy) void (^onClosed)(void);
@property (nonatomic, copy) void (^onGeometry)(void);
@property (nonatomic, copy) void (^onScreenChanged)(NSWindow*);
@property (nonatomic, copy) void (^onResignKey)(void);
@property (nonatomic, copy) void (^onBecomeKey)(void);
@end

@implementation ArmigoWindowDelegate

- (BOOL)windowShouldClose:(NSWindow*)window {
    (void)window;
    if (self.onClose) self.onClose();
    return YES;
}

- (void)windowWillClose:(NSNotification*)notification {
    (void)notification;
    if (self.onClosed) self.onClosed();
}

// Resize, fullscreen transitions and backing-scale changes all move the
// drawable surface. They never touch guest buffer metadata; the presenter
// scales the frame onto whatever drawable the layer currently provides.
- (void)windowDidResignKey:(NSNotification*)notification {
    (void)notification;
    if (self.onResignKey) self.onResignKey();
}

- (void)windowDidBecomeKey:(NSNotification*)notification {
    (void)notification;
    if (self.onBecomeKey) self.onBecomeKey();
}

- (void)windowDidChangeScreen:(NSNotification*)notification {
    if (self.onScreenChanged) self.onScreenChanged((NSWindow*)notification.object);
}

- (void)windowDidResize:(NSNotification*)notification {
    (void)notification;
    if (self.onGeometry) self.onGeometry();
}

- (void)windowDidChangeBackingProperties:(NSNotification*)notification {
    (void)notification;
    if (self.onGeometry) self.onGeometry();
}

- (void)windowDidEnterFullScreen:(NSNotification*)notification {
    (void)notification;
    if (self.onGeometry) self.onGeometry();
}

- (void)windowDidExitFullScreen:(NSNotification*)notification {
    (void)notification;
    if (self.onGeometry) self.onGeometry();
}

@end

// Quit from the menu, the Dock or a logout closes the window instead of
// terminating the process: the close stops the guest through the same path as
// the close button, and the shell exits once the runtime has shut down.
@interface ArmigoApplicationDelegate : NSObject <NSApplicationDelegate>
@property (nonatomic, weak) NSWindow* window;
@end

@implementation ArmigoApplicationDelegate

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication*)sender {
    (void)sender;
    NSWindow* window = self.window;
    if (window && (window.visible || window.miniaturized)) [window performClose:nil];
    return NSTerminateCancel;
}

@end

namespace armigo::platform {

// Defined in src/platform/metal/presenter.mm. Internal seam so the AppKit
// window can hand its CAMetalLayer to the Metal presenter without an
// Apple-typed public contract.
std::shared_ptr<Presenter> create_layer_presenter(const PresenterOptions& options, CAMetalLayer* layer,
                                                  PresentationFailure* failure_out);
CGSize content_drawable_size(CGSize pixels, std::uint32_t content_width);

namespace {

// Bounded pump latency for request_quit and stop tokens. The event loop always
// wakes through real event delivery; this interval is the upper bound on how
// long a quit request can sit unnoticed while no events arrive.
constexpr NSTimeInterval kPumpInterval = 0.05;

PresentationFailure failure(PresentationError code, std::string detail) {
    return PresentationFailure{code, std::move(detail)};
}

// Shutdown state shared between the application and the window delegate. The
// delegate's block owns a reference, so a close notification that races the
// application destructor cannot touch freed state.
struct AppState {
    std::atomic<bool> quit{false};
    std::atomic<bool> window_closed{false};
};

// The menu bar of a plain executable is empty; give it the commands a player
// expects: Quit (which reaches applicationShouldTerminate:), Minimize and
// Full Screen.
void install_main_menu() {
    NSMenu* bar = [NSMenu new];
    NSMenuItem* app_item = [NSMenuItem new];
    NSMenu* app_menu = [NSMenu new];
    [app_menu addItemWithTitle:@"Quit Armigo" action:@selector(terminate:) keyEquivalent:@"q"];
    app_item.submenu = app_menu;
    [bar addItem:app_item];

    NSMenuItem* window_item = [NSMenuItem new];
    NSMenu* window_menu = [[NSMenu alloc] initWithTitle:@"Window"];
    [window_menu addItemWithTitle:@"Minimize" action:@selector(performMiniaturize:) keyEquivalent:@"m"];
    NSMenuItem* fullscreen = [window_menu addItemWithTitle:@"Enter Full Screen"
                                                    action:@selector(toggleFullScreen:)
                                             keyEquivalent:@"f"];
    fullscreen.keyEquivalentModifierMask = NSEventModifierFlagControl | NSEventModifierFlagCommand;
    window_item.submenu = window_menu;
    [bar addItem:window_item];
    NSApp.mainMenu = bar;
    NSApp.windowsMenu = window_menu;
}

class AppKitApplication final : public Application {
public:
    AppKitApplication(NSWindow* window, ArmigoWindowDelegate* delegate, ArmigoApplicationDelegate* app_delegate,
                      id<NSObject> activity, std::unique_ptr<DisplayRefreshLink> refresh,
                      std::shared_ptr<Presenter> presenter, ApplicationCallbacks callbacks,
                      std::shared_ptr<AppState> state)
        : window_(window), delegate_(delegate), app_delegate_(app_delegate), activity_(activity),
          refresh_(std::move(refresh)), presenter_(std::move(presenter)), callbacks_(std::move(callbacks)),
          state_(std::move(state)) {}

    ~AppKitApplication() override {
        if (![NSThread isMainThread]) {
            // The contract makes destruction a main-thread step; refuse to
            // touch AppKit from anywhere else instead of misbehaving quietly.
            armigo::diag::write(armigo::diag::Level::error,  "Armigo application destroyed off the main thread\n");
            return;
        }
        // No refresh is delivered once this returns.
        refresh_.reset();
        if (app_delegate_) {
            if (NSApp.delegate == app_delegate_) NSApp.delegate = nil;
            app_delegate_ = nil;
        }
        if (activity_) {
            [[NSProcessInfo processInfo] endActivity:activity_];
            activity_ = nil;
        }
        if (delegate_) {
            delegate_.onClose = nil;
            window_.delegate = nil;
            delegate_ = nil;
        }
        if (window_) {
            [window_ close];
            window_ = nil;
        }
        // The presenter member is destroyed after the window: its destructor
        // waits for real GPU completion of any inflight work, so no drawable,
        // staging or pixel resource is released before completion.
    }

    std::shared_ptr<Presenter> presenter() const noexcept override { return presenter_; }

    DisplayCapabilities display_capabilities() const noexcept override {
        NSScreen* screen = window_.screen;
        if (!screen) return {};
        return query_display_capabilities([screen.deviceDescription[@"NSScreenNumber"] unsignedIntValue]);
    }

    PresenterStatus run(std::stop_token stop) override {
        if (![NSThread isMainThread])
            return failure(PresentationError::host_failure, "the application event loop runs on the main thread");
        while (!stop.stop_requested() && !state_->quit.load(std::memory_order_relaxed)) {
            @autoreleasepool {
                NSEvent* event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                    untilDate:[NSDate dateWithTimeIntervalSinceNow:kPumpInterval]
                                                       inMode:NSDefaultRunLoopMode
                                                      dequeue:YES];
                if (event) [NSApp sendEvent:event];
            }
        }
        return std::monostate{};
    }

    void request_quit() noexcept override {
        state_->quit.store(true, std::memory_order_relaxed);
    }

    NSWindow* window_for_testing() const { return window_; }

private:
    NSWindow* window_ = nil;  // Main thread only.
    ArmigoWindowDelegate* delegate_ = nil;  // Main thread only.
    ArmigoApplicationDelegate* app_delegate_ = nil;  // Main thread only; NSApp holds it weakly.
    id<NSObject> activity_ = nil;  // Keeps the display awake while the window exists.
    std::unique_ptr<DisplayRefreshLink> refresh_;  // Refreshes of the window's display.
    std::shared_ptr<Presenter> presenter_;
    ApplicationCallbacks callbacks_;
    std::shared_ptr<AppState> state_;
};

} // namespace

ApplicationCreation create_application(const PresenterOptions& options, ApplicationCallbacks callbacks) {
    @autoreleasepool {
        if (options.mode != PresentationMode::windowed)
            return failure(PresentationError::unsupported, "application creation requires mode=windowed");
        if (![NSThread isMainThread])
            return failure(PresentationError::host_failure, "application creation requires the main thread");
        if ([NSScreen screens].count == 0)
            return failure(PresentationError::unsupported, "no display session is available for a window");
        if (![NSApplication sharedApplication])
            return failure(PresentationError::host_failure, "NSApplication initialization failed");
        // A plain executable starts as a background process: make it a regular
        // application so its window can appear, take focus and enter fullscreen.
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp finishLaunching];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4])
            return failure(PresentationError::unsupported, "no Metal 4 device is available");

        const NSSize content{
            options.width > 0 ? static_cast<CGFloat>(options.width) : 1280.0,
            options.height > 0 ? static_cast<CGFloat>(options.height) : 720.0};
        NSWindow* window = [[NSWindow alloc]
            initWithContentRect:NSMakeRect(0, 0, content.width, content.height)
                      styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                                NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
                        backing:NSBackingStoreBuffered
                          defer:NO];
        ArmigoMetalView* view = [[ArmigoMetalView alloc] initWithFrame:NSMakeRect(0, 0, content.width,
                                                                                      content.height)];
        view.wantsLayer = YES;
        CAMetalLayer* metal_layer = [CAMetalLayer layer];
        view.layer = metal_layer;
        if (!window || !view || view.layer != metal_layer) {
            std::string detail = "window or layer-backed view creation failed (";
            detail += window ? "window ok, " : "window nil, ";
            detail += view ? "view ok" : "view nil";
            detail += ")";
            return failure(PresentationError::host_failure, detail);
        }
        window.title = @"Armigo";
        window.releasedWhenClosed = NO;
        // The Metal view fills a plain container so the key guide can sit
        // above it as a sibling. A subview of the CAMetalLayer is not a
        // reliable overlay: the drawable is presented into that layer.
        NSView* container = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, content.width, content.height)];
        view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        [container addSubview:view];
        window.contentView = container;
        view.frame = container.bounds;
        install_key_guide(container, view);
        // Keeps the drawable from collapsing to nothing while resizing.
        window.contentMinSize = NSMakeSize(320, 180);
        [window center];

        auto state = std::make_shared<AppState>();
        PresentationFailure presenter_failure;
        std::shared_ptr<Presenter> presenter = create_layer_presenter(options, metal_layer,
                                                                      &presenter_failure);
        if (!presenter) {
            [window close];
            return ApplicationCreation(std::move(presenter_failure));
        }

        ApplicationCallbacks handler = std::move(callbacks);
        ArmigoWindowDelegate* delegate = [ArmigoWindowDelegate new];
        delegate.onClose = [state, handler]() {
            // Close notifies the shell to stop the guest; it never destroys
            // inflight graphics resources from inside the delegate.
            if (handler.close_requested) handler.close_requested();
        };
        delegate.onClosed = [state]() {
            state->window_closed.store(true, std::memory_order_relaxed);
        };
        delegate.onScreenChanged = [handler](NSWindow* changed_window) {
            NSScreen* screen = changed_window.screen;
            if (screen && handler.display_changed)
                handler.display_changed(query_display_capabilities(
                    [screen.deviceDescription[@"NSScreenNumber"] unsignedIntValue]));
        };
        const std::uint32_t content_width = options.content_width;
        delegate.onGeometry = [view, metal_layer, content_width]() {
            // View bounds update synchronously with window geometry; layer
            // bounds can lag until the next layout pass.
            const CGFloat scale = metal_layer.contentsScale > 0 ? metal_layer.contentsScale : 1.0;
            const CGSize bounds = view.bounds.size;
            metal_layer.drawableSize =
                content_drawable_size(CGSizeMake(bounds.width * scale, bounds.height * scale), content_width);
        };
        if (handler.mouse_moved) {
            auto moved = handler.mouse_moved;
            view.onMouseMoved = ^(double dx, double dy) { moved(dx, dy); };
        }
        if (handler.mouse_button) {
            auto button = handler.mouse_button;
            view.onMouseButton = ^(unsigned index, BOOL down) { button(index, down); };
        }
        __weak ArmigoMetalView* weak_view = view;
        auto focus_changed = handler.focus_changed;
        delegate.onResignKey = ^{
            [weak_view releaseMouse];
            if (focus_changed) focus_changed(false);
        };
        delegate.onBecomeKey = ^{
            if (focus_changed) focus_changed(true);
        };
        window.acceptsMouseMovedEvents = YES;
        window.delegate = delegate;
        window.collectionBehavior = NSWindowCollectionBehaviorFullScreenPrimary;
        [window makeKeyAndOrderFront:nil];
        [window makeFirstResponder:view];
        [NSApp activate];
        metal_layer.contentsScale = window.screen ? window.screen.backingScaleFactor : 1.0;
        delegate.onGeometry();
        if (options.fullscreen) [window toggleFullScreen:nil];

        install_main_menu();
        ArmigoApplicationDelegate* app_delegate = [ArmigoApplicationDelegate new];
        app_delegate.window = window;
        NSApp.delegate = app_delegate;
        // A title played with a controller sends no input the system counts
        // as user activity; keep the display awake while the window exists.
        id<NSObject> activity = [[NSProcessInfo processInfo]
            beginActivityWithOptions:NSActivityUserInitiated | NSActivityIdleDisplaySleepDisabled |
                                     NSActivityLatencyCritical
                              reason:@"Running a title"];

        // The view's display link follows the window to whichever screen it is on.
        std::unique_ptr<DisplayRefreshLink> refresh;
        if (handler.display_refresh) refresh = start_display_refresh(view, handler.display_refresh);

        // The window does not retain its delegate, so the application owns it.
        auto application = std::make_unique<AppKitApplication>(window, delegate, app_delegate, activity,
                                                               std::move(refresh), std::move(presenter),
                                                               std::move(handler), std::move(state));
        return ApplicationCreation(std::move(application));
    }
}

int host_thermal_state() noexcept {
    return -1;
}

// Internal seam for lifecycle tests: returns the application's window without
// exposing AppKit types in any public header.
NSWindow* application_window_for_testing(Application* application) {
    auto* impl = dynamic_cast<AppKitApplication*>(application);
    return impl ? impl->window_for_testing() : nil;
}

} // namespace armigo::platform
