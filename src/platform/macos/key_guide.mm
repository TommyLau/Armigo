// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import "key_guide.h"

#import <GameController/GameController.h>

#include <objc/runtime.h>

namespace {
// The player's Settings window hard-codes English and has no string table, so
// the guide does too.
NSString* const kGuideText =
    @"Keyboard and mouse\n"
     "\n"
     "Move      W A S D     Camera    I J K L\n"
     "A         E           B         Shift\n"
     "X         Space       Y         U\n"
     "L         F           R         R\n"
     "ZL        Q           ZR        O\n"
     "+         Enter       −         Tab\n"
     "Clicks    C left, V right\n"
     "\n"
     "Mouse     Click to look; left Y, right ZR\n"
     "Esc       Release the mouse\n"
     "F1        Show or hide this guide";

const void* const kGuideAssociation = &kGuideAssociation;
constexpr NSTimeInterval kVisibleSeconds = 10.0;
}  // namespace

@interface ArmigoKeyGuide : NSView
@property (nonatomic) NSInteger epoch;
@property (nonatomic) BOOL pinned;
@property (nonatomic) CGFloat fontSize;
@property (nonatomic, strong) NSTextField* label;
@property (nonatomic, strong) NSLayoutConstraint* preferredHeight;
@property (nonatomic, strong) id connectObserver;
@property (nonatomic, strong) id disconnectObserver;
- (void)showTransient;
- (void)hideImmediately;
- (void)toggle;
@end

@implementation ArmigoKeyGuide

- (instancetype)initWithFrame:(NSRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;
    self.wantsLayer = YES;
    self.layer.cornerRadius = 12;
    self.layer.masksToBounds = YES;
    self.layer.backgroundColor = [NSColor colorWithSRGBRed:0 green:0 blue:0 alpha:0.78].CGColor;
    self.layer.borderWidth = 1;
    self.layer.borderColor = [NSColor colorWithSRGBRed:1 green:1 blue:1 alpha:0.28].CGColor;
    self.hidden = YES;
    self.fontSize = 13;
    NSTextField* label = [NSTextField labelWithString:kGuideText];
    label.font = [NSFont monospacedSystemFontOfSize:self.fontSize weight:NSFontWeightMedium];
    label.textColor = NSColor.whiteColor;
    label.maximumNumberOfLines = 0;
    label.lineBreakMode = NSLineBreakByWordWrapping;
    label.cell.wraps = YES;
    label.cell.usesSingleLineMode = NO;
    label.preferredMaxLayoutWidth = 392;
    label.translatesAutoresizingMaskIntoConstraints = NO;
    label.refusesFirstResponder = YES;
    [self addSubview:label];
    self.label = label;
    // The bottom edge yields when the window is shorter than the text, so a
    // tiny window stays free of unsatisfiable constraints. The view clips.
    NSLayoutConstraint* bottom = [label.bottomAnchor constraintEqualToAnchor:self.bottomAnchor constant:-12];
    bottom.priority = NSLayoutPriorityDefaultHigh;
    [NSLayoutConstraint activateConstraints:@[
        [label.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:14],
        [label.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-14],
        [label.topAnchor constraintEqualToAnchor:self.topAnchor constant:12],
        bottom,
    ]];
    [self setContentHuggingPriority:NSLayoutPriorityDefaultHigh
                      forOrientation:NSLayoutConstraintOrientationVertical];
    [self setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                    forOrientation:NSLayoutConstraintOrientationVertical];
    return self;
}

- (BOOL)acceptsFirstResponder {
    return NO;
}

// Clicks fall through to the game view. The guide never becomes key.
- (NSView*)hitTest:(NSPoint)point {
    (void)point;
    return nil;
}

- (void)layout {
    [super layout];
    NSView* host = self.superview;
    if (host == nil) return;
    CGFloat size = host.bounds.size.height * 0.016;
    if (size < 10) size = 10;
    if (size > 14) size = 14;
    if (size != self.fontSize) {
        self.fontSize = size;
        self.label.font = [NSFont monospacedSystemFontOfSize:size weight:NSFontWeightMedium];
    }
    const CGFloat width = self.bounds.size.width - 28;
    if (width > 40 && self.label.preferredMaxLayoutWidth != width)
        self.label.preferredMaxLayoutWidth = width;
    NSView* limit_host = self.superview;
    CGFloat text_height = self.label.fittingSize.height + 24;
    if (limit_host != nil) {
        const CGFloat cap = limit_host.bounds.size.height - 32;
        if (cap > 40 && text_height > cap) text_height = cap;
    }
    if (self.preferredHeight != nil && self.preferredHeight.constant != text_height)
        self.preferredHeight.constant = text_height;
}

- (void)showTransient {
    self.pinned = NO;
    self.alphaValue = 1;
    self.hidden = NO;
    const NSInteger epoch = ++self.epoch;
    __weak ArmigoKeyGuide* weak = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(kVisibleSeconds * NSEC_PER_SEC)),
                   dispatch_get_main_queue(), ^{
                       [weak fadeFromEpoch:epoch];
                   });
}

- (void)fadeFromEpoch:(NSInteger)epoch {
    if (epoch != self.epoch || self.hidden || self.pinned) return;
    [self fadeStep:epoch];
}

- (void)fadeStep:(NSInteger)epoch {
    if (epoch != self.epoch || self.hidden || self.pinned) return;
    if (self.alphaValue <= 0.08) {
        self.hidden = YES;
        self.alphaValue = 1;
        return;
    }
    self.alphaValue -= 0.08;
    __weak ArmigoKeyGuide* weak = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.04 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        [weak fadeStep:epoch];
    });
}

- (void)hideImmediately {
    ++self.epoch;
    self.pinned = NO;
    self.alphaValue = 1;
    self.hidden = YES;
}

- (void)toggle {
    if (!self.hidden && self.alphaValue > 0.05) {
        [self hideImmediately];
        return;
    }
    ++self.epoch;
    self.pinned = YES;
    self.alphaValue = 1;
    self.hidden = NO;
}

- (void)controllersChanged {
    if (GCController.controllers.count == 0) [self showTransient];
    else [self hideImmediately];
}

- (void)dealloc {
    NSNotificationCenter* center = NSNotificationCenter.defaultCenter;
    if (self.connectObserver) [center removeObserver:self.connectObserver];
    if (self.disconnectObserver) [center removeObserver:self.disconnectObserver];
}

@end

namespace armigo::platform {

void install_key_guide(NSView* container, NSView* metal_view) {
    if (container == nil || metal_view == nil) return;
    ArmigoKeyGuide* guide = [[ArmigoKeyGuide alloc] initWithFrame:NSZeroRect];
    guide.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:guide];

    NSLayoutConstraint* preferred_width = [guide.widthAnchor constraintEqualToConstant:420];
    preferred_width.priority = NSLayoutPriorityDefaultHigh;
    NSLayoutConstraint* preferred_height = [guide.heightAnchor constraintEqualToConstant:240];
    preferred_height.priority = NSLayoutPriorityDefaultHigh;
    guide.preferredHeight = preferred_height;
    [NSLayoutConstraint activateConstraints:@[
        [guide.leadingAnchor constraintEqualToAnchor:container.leadingAnchor constant:16],
        [guide.topAnchor constraintEqualToAnchor:container.topAnchor constant:16],
        [guide.trailingAnchor constraintLessThanOrEqualToAnchor:container.trailingAnchor constant:-16],
        [guide.bottomAnchor constraintLessThanOrEqualToAnchor:container.bottomAnchor constant:-16],
        [guide.widthAnchor constraintLessThanOrEqualToAnchor:container.widthAnchor constant:-32],
        [guide.heightAnchor constraintLessThanOrEqualToAnchor:container.heightAnchor constant:-32],
        preferred_width,
        preferred_height,
    ]];

    __weak ArmigoKeyGuide* weak = guide;
    NSNotificationCenter* center = NSNotificationCenter.defaultCenter;
    void (^changed)(NSNotification*) = ^(NSNotification*) { [weak controllersChanged]; };
    guide.connectObserver = [center addObserverForName:GCControllerDidConnectNotification
                                                object:nil
                                                 queue:NSOperationQueue.mainQueue
                                            usingBlock:changed];
    guide.disconnectObserver = [center addObserverForName:GCControllerDidDisconnectNotification
                                                   object:nil
                                                    queue:NSOperationQueue.mainQueue
                                               usingBlock:changed];
    objc_setAssociatedObject(metal_view, kGuideAssociation, guide, OBJC_ASSOCIATION_RETAIN);
    [guide controllersChanged];
}

void toggle_key_guide(NSView* metal_view) {
    ArmigoKeyGuide* guide = objc_getAssociatedObject(metal_view, kGuideAssociation);
    [guide toggle];
}

}  // namespace armigo::platform
