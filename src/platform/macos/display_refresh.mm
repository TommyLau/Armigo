// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "display_refresh.h"

#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif
#import <QuartzCore/QuartzCore.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <pthread.h>
#include <thread>
#include <utility>

@interface ArmigoDisplayRefreshTarget : NSObject
- (instancetype)initWithHandler:(armigo::platform::DisplayRefreshCallback)handler;
- (void)refresh:(CADisplayLink*)link;
@end

@implementation ArmigoDisplayRefreshTarget {
    armigo::platform::DisplayRefreshCallback _handler;
}

- (instancetype)initWithHandler:(armigo::platform::DisplayRefreshCallback)handler {
    if ((self = [super init])) _handler = std::move(handler);
    return self;
}

- (void)refresh:(CADisplayLink*)link {
    const auto vblank = armigo::platform::steady_from_host_time(link.timestamp);
    const auto period = std::chrono::nanoseconds{std::llround((link.targetTimestamp - link.timestamp) * 1e9)};
    if (_handler) _handler(vblank, period);
}

@end

namespace armigo::platform {

std::chrono::steady_clock::time_point steady_from_host_time(double seconds) {
    // The two clocks are read with host time on both sides of the steady
    // read; a pair split by preemption is retried, and the tightest of a few
    // tries is used, so the offset is not off by a scheduling delay.
    auto best_steady = std::chrono::steady_clock::now();
    double best_host = CACurrentMediaTime(), best_window = 1.0;
    for (int attempt = 0; attempt < 4 && best_window > 2e-6; ++attempt) {
        const double before = CACurrentMediaTime();
        const auto steady = std::chrono::steady_clock::now();
        const double after = CACurrentMediaTime();
        if (after - before < best_window) {
            best_window = after - before;
            best_steady = steady;
            best_host = (before + after) / 2;
        }
    }
    return best_steady + std::chrono::nanoseconds{std::llround((seconds - best_host) * 1e9)};
}

namespace {

class RunLoopDisplayRefresh final : public DisplayRefreshLink {
public:
    explicit RunLoopDisplayRefresh(CADisplayLink* link) : link_(link) {
        std::promise<CFRunLoopRef> started;
        auto loop = started.get_future();
        thread_ = std::thread([link, started = std::move(started)]() mutable {
            pthread_setname_np("armigo.display-refresh");
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            @autoreleasepool {
                [link addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSRunLoopCommonModes];
                CFRunLoopRef current = CFRunLoopGetCurrent();
                CFRetain(current);
                started.set_value(current);
                CFRunLoopRun();
            }
        });
        loop_ = loop.get();
    }

    bool set_rate(double rate) noexcept override {
#if TARGET_OS_IPHONE
        if (!std::isfinite(rate) || rate <= 0) return false;
        CADisplayLink* link = link_;
        CFRunLoopPerformBlock(loop_, kCFRunLoopCommonModes, ^{
            link.preferredFrameRateRange = CAFrameRateRangeMake(rate, rate, rate);
        });
        CFRunLoopWakeUp(loop_);
        return true;
#else
        static_cast<void>(rate);
        return false;
#endif
    }

    ~RunLoopDisplayRefresh() override {
        CADisplayLink* link = link_;
        // The link is invalidated on the thread whose run loop it is on; the
        // run returns once its only source is gone and the loop is stopped.
        CFRunLoopPerformBlock(loop_, kCFRunLoopCommonModes, ^{
            [link invalidate];
            CFRunLoopStop(CFRunLoopGetCurrent());
        });
        CFRunLoopWakeUp(loop_);
        thread_.join();
        CFRelease(loop_);
    }

private:
    CADisplayLink* link_ = nil;
    CFRunLoopRef loop_ = nullptr;
    std::thread thread_;
};

} // namespace

std::unique_ptr<DisplayRefreshLink> start_display_refresh(id source, DisplayRefreshCallback handler) {
    if (!handler) return nullptr;
    auto* target = [[ArmigoDisplayRefreshTarget alloc] initWithHandler:std::move(handler)];
#if TARGET_OS_IPHONE
    UIScreen* screen = [source isKindOfClass:UIScreen.class] ? source :
        [source isKindOfClass:UIView.class] ? ((UIView*)source).window.windowScene.screen : nil;
    if (!screen) return nullptr;
    CADisplayLink* link = [CADisplayLink displayLinkWithTarget:target selector:@selector(refresh:)];
    double rate = screen.maximumFramesPerSecond;
    if (const char* override = std::getenv("ARMIGO_VI_REFRESH_HZ")) {
        const double selected = std::strtod(override, nullptr);
        if (selected > 0 && (rate <= 0 || selected < rate)) rate = selected;
    }
    if (rate > 0) link.preferredFrameRateRange = CAFrameRateRangeMake(rate, rate, rate);
#else
    if (![source respondsToSelector:@selector(displayLinkWithTarget:selector:)]) return nullptr;
    CADisplayLink* link = [source displayLinkWithTarget:target selector:@selector(refresh:)];
#endif
    if (!link) return nullptr;
    return std::make_unique<RunLoopDisplayRefresh>(link);
}

} // namespace armigo::platform
