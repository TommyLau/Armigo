// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The application installs a layer-backed CAMetalLayer with display sync and
// framebuffer-only drawables, then presents without adding a pacing policy.

#include <armigo/platform/application.h>
#include <armigo/platform/pcm_sink.h>
#include "../macos/display_refresh.h"
#include "touch_controls.h"

#import <UIKit/UIKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <atomic>
#include <memory>
#include <utility>

@interface ArmigoIosGameController : UIViewController
@end
@implementation ArmigoIosGameController
- (UIInterfaceOrientationMask)supportedInterfaceOrientations { return UIInterfaceOrientationMaskLandscape; }
- (BOOL)prefersStatusBarHidden { return YES; }
- (BOOL)prefersHomeIndicatorAutoHidden { return YES; }
@end

namespace armigo::platform {
std::shared_ptr<Presenter> create_layer_presenter(const PresenterOptions&, CAMetalLayer*, PresentationFailure*);
CGSize content_drawable_size(CGSize pixels, std::uint32_t content_width);
namespace {

struct State { std::atomic<bool> quit{false}; };
struct Subscriptions {
    NSMutableArray<id>* tokens = [NSMutableArray array];
    Subscriptions() = default;
    Subscriptions(const Subscriptions&) = delete;
    Subscriptions(Subscriptions&& other) noexcept : tokens(other.tokens) { other.tokens = nil; }
    void clear() {
        for (id token in tokens) [NSNotificationCenter.defaultCenter removeObserver:token];
        [tokens removeAllObjects];
    }
    ~Subscriptions() { clear(); }
};

class IosApplication final : public Application {
public:
    IosApplication(UIWindow* window, std::shared_ptr<Presenter> presenter,
                   std::unique_ptr<DisplayRefreshLink> refresh,
                   std::shared_ptr<State> state, Subscriptions observers)
        : window_(window), presenter_(std::move(presenter)), refresh_(std::move(refresh)),
          state_(std::move(state)), observers_(std::move(observers)) {}
    ~IosApplication() override {
        refresh_.reset();
        observers_.clear();
        window_.hidden = YES;
        UIApplication.sharedApplication.idleTimerDisabled = NO;
    }
    std::shared_ptr<Presenter> presenter() const noexcept override { return presenter_; }
    PresenterStatus run(std::stop_token stop) override {
        if (!NSThread.isMainThread)
            return PresentationFailure{PresentationError::host_failure, "UIKit event pumping requires the main thread."};
        std::stop_callback cancelled(stop, [state = state_] {
            state->quit.store(true);
            CFRunLoopWakeUp(CFRunLoopGetMain());
        });
        // UIApplicationMain owns the outer lifecycle. The windowed shell is
        // entered from a scene callback and pumps the same main run loop until
        // its runtime worker has stopped and drained the presenter.
        while (!state_->quit.load()) {
            @autoreleasepool {
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.05, true);
            }
        }
        return std::monostate{};
    }
    bool set_display_refresh_rate(double rate) noexcept override {
        return refresh_ && refresh_->set_rate(rate);
    }
    void request_quit() noexcept override {
        state_->quit.store(true);
        CFRunLoopWakeUp(CFRunLoopGetMain());
    }
private:
    UIWindow* window_;
    std::shared_ptr<Presenter> presenter_;
    std::unique_ptr<DisplayRefreshLink> refresh_;
    std::shared_ptr<State> state_;
    Subscriptions observers_;
};
} // namespace

int host_thermal_state() noexcept {
    return static_cast<int>(NSProcessInfo.processInfo.thermalState);
}

ApplicationCreation create_application(const PresenterOptions& options, ApplicationCallbacks callbacks) {
    if (!NSThread.isMainThread || options.mode != PresentationMode::windowed)
        return PresentationFailure{PresentationError::unsupported, "UIKit requires a windowed application on the main thread."};
    @autoreleasepool {
        UIWindowScene* selected = nil;
        for (UIScene* scene in UIApplication.sharedApplication.connectedScenes) {
            if (![scene isKindOfClass:UIWindowScene.class]) continue;
            if (!selected || scene.activationState == UISceneActivationStateForegroundActive)
                selected = (UIWindowScene*)scene;
            if (scene.activationState == UISceneActivationStateForegroundActive) break;
        }
        if (!selected)
            return PresentationFailure{PresentationError::host_failure, "A connected UIKit window scene is required."};
        UIWindow* window = [[UIWindow alloc] initWithWindowScene:selected];
        ArmigoIosGameController* controller = [ArmigoIosGameController new];
        ArmigoIosMetalView* view = [[ArmigoIosMetalView alloc] initWithFrame:window.bounds];
        controller.view = view;
        window.rootViewController = controller;
        CAMetalLayer* layer = (CAMetalLayer*)view.layer;
        layer.opaque = YES;
        layer.contentsScale = selected.screen.scale;
        layer.drawableSize = content_drawable_size(CGSizeMake(window.bounds.size.width * layer.contentsScale,
                                                              window.bounds.size.height * layer.contentsScale),
                                                   options.content_width);
        PresentationFailure error;
        auto presenter = create_layer_presenter(options, layer, &error);
        if (!presenter) return error;
        auto state = std::make_shared<State>();
        Subscriptions observers;
        const auto focus = callbacks.focus_changed;
        const auto close = callbacks.close_requested;
        NSNotificationCenter* notifications = NSNotificationCenter.defaultCenter;
        [observers.tokens addObject:[notifications addObserverForName:UIApplicationWillResignActiveNotification
            object:nil queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification*) {
                armigo_ios_release_touch_controls(view);
                if (focus) focus(false);
            }]];
        [observers.tokens addObject:[notifications addObserverForName:UIApplicationDidBecomeActiveNotification
            object:nil queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification*) {
                if (focus) focus(true);
                pcm_sink_resume_devices();
            }]];
        [observers.tokens addObject:[notifications addObserverForName:UIApplicationWillTerminateNotification
            object:nil queue:NSOperationQueue.mainQueue usingBlock:^(NSNotification*) {
                if (close) close();
            }]];
        std::unique_ptr<DisplayRefreshLink> refresh;
        if (callbacks.display_refresh) {
            refresh = start_display_refresh(selected.screen, std::move(callbacks.display_refresh));
            if (!refresh) {
                return PresentationFailure{PresentationError::host_failure, "Cannot start the iOS display refresh link."};
            }
        }
        UIApplication.sharedApplication.idleTimerDisabled = YES;
        [window makeKeyAndVisible];
        armigo_ios_start_touch_controls(view, observers.tokens);
        [view setNeedsLayout];
        [view layoutIfNeeded];
        if (focus) focus(true);
        return std::unique_ptr<Application>(new IosApplication(window, std::move(presenter),
            std::move(refresh), std::move(state), std::move(observers)));
    }
}
} // namespace armigo::platform
