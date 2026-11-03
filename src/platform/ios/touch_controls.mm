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
#include "touch_controls.h"

#include <armigo/platform/apple_input.h>
#include <armigo/platform/touch_controls.h>

#import <GameController/GameController.h>
#import <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace armigo::platform;

@interface ArmigoScriptTouch : UITouch
@property(nonatomic) CGPoint scriptPoint;
@property(nonatomic) UITouchPhase scriptPhase;
@end
@implementation ArmigoScriptTouch
- (CGPoint)locationInView:(UIView*)view { (void)view; return self.scriptPoint; }
- (CGPoint)previousLocationInView:(UIView*)view { (void)view; return self.scriptPoint; }
- (NSTimeInterval)timestamp { return CACurrentMediaTime(); }
- (UITouchPhase)phase { return self.scriptPhase; }
@end

@interface ArmigoTouchPad : UIView
- (void)releaseAll;
@end

@implementation ArmigoTouchPad {
    std::array<CAShapeLayer*, touch_control_count> shapes_;
    std::array<CATextLayer*, touch_control_count> text_;
    std::array<CAShapeLayer*, 2> knobs_;
    std::array<unsigned, touch_control_count> held_;
    std::array<TouchStick, 2> sticks_;
    TouchLayout layout_;
    CFMutableDictionaryRef owners_;
}

static UIColor* line_color() {
    return [UIColor colorWithWhite:1 alpha:0.55];
}

static NSString* label_for_control(std::size_t index) {
    static NSString* const labels[] = {
        nil, nil, @"A", @"B", @"X", @"Y", @"▲", @"▼", @"◀", @"▶",
        @"L", @"ZL", @"R", @"ZR", @"−", @"+", @"LS", @"RS",
    };
    return labels[index];
}

- (instancetype)initWithFrame:(CGRect)frame {
    self = [super initWithFrame:frame];
    if (!self) return nil;
    self.multipleTouchEnabled = YES;
    self.backgroundColor = UIColor.clearColor;
    shapes_.fill(nil);
    text_.fill(nil);
    knobs_.fill(nil);
    held_.fill(0);
    sticks_ = {};
    layout_ = {};
    owners_ = CFDictionaryCreateMutable(nullptr, 0, nullptr, nullptr);
    const CGFloat scale = self.traitCollection.displayScale;
    for (std::size_t i = 0; i < touch_control_count; ++i) {
        CAShapeLayer* shape = [CAShapeLayer layer];
        shape.fillColor = [UIColor colorWithWhite:1 alpha:0.06].CGColor;
        shape.strokeColor = line_color().CGColor;
        shape.lineWidth = 1.5;
        shape.shadowColor = UIColor.blackColor.CGColor;
        shape.shadowOpacity = 0.35;
        shape.shadowRadius = 1.5;
        shape.shadowOffset = CGSizeZero;
        [self.layer addSublayer:shape];
        shapes_[i] = shape;
        NSString* label = label_for_control(i);
        if (!label) continue;
        CATextLayer* text = [CATextLayer layer];
        text.string = label;
        text.alignmentMode = kCAAlignmentCenter;
        text.foregroundColor = [UIColor colorWithWhite:1 alpha:0.8].CGColor;
        text.contentsScale = scale;
        text.font = (__bridge CFTypeRef)[UIFont systemFontOfSize:16 weight:UIFontWeightSemibold];
        text.fontSize = label.length > 1 ? 15 : 17;
        text.shadowColor = UIColor.blackColor.CGColor;
        text.shadowOpacity = 0.5;
        text.shadowRadius = 1;
        text.shadowOffset = CGSizeZero;
        [self.layer addSublayer:text];
        text_[i] = text;
    }
    for (auto& knob : knobs_) {
        knob = [CAShapeLayer layer];
        knob.fillColor = [UIColor colorWithWhite:1 alpha:0.18].CGColor;
        knob.strokeColor = line_color().CGColor;
        knob.lineWidth = 1.5;
        [self.layer addSublayer:knob];
    }
    return self;
}

- (void)dealloc {
    if (owners_) CFRelease(owners_);
}

- (void)layoutSubviews {
    [super layoutSubviews];
    const UIEdgeInsets safe = self.safeAreaInsets;
    layout_ = make_touch_layout(self.bounds.size.width, self.bounds.size.height,
        {safe.left, safe.top, safe.right, safe.bottom});
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    for (std::size_t i = 0; i < touch_control_count; ++i) {
        const auto& control = layout_[i];
        const CGRect box = control.width > 0
            ? CGRectMake(control.x - control.width / 2, control.y - control.height / 2,
                         control.width, control.height)
            : CGRectMake(control.x - control.radius, control.y - control.radius,
                         control.radius * 2, control.radius * 2);
        shapes_[i].path = (control.width > 0
            ? [UIBezierPath bezierPathWithRoundedRect:box cornerRadius:control.height / 2]
            : [UIBezierPath bezierPathWithOvalInRect:box]).CGPath;
        if (text_[i]) text_[i].frame = CGRectMake(CGRectGetMinX(box), control.y - 10, CGRectGetWidth(box), 20);
    }
    [CATransaction commit];
    [self redraw];
}

- (void)redraw {
    [CATransaction begin];
    [CATransaction setDisableActions:YES];
    for (std::size_t i = 2; i < touch_control_count; ++i) {
        shapes_[i].fillColor = [UIColor colorWithWhite:1 alpha:held_[i] != 0 ? 0.35 : 0.06].CGColor;
    }
    for (std::size_t stick = 0; stick < 2; ++stick) {
        const auto& control = layout_[stick];
        const double x = control.x + sticks_[stick][0] * control.radius;
        const double y = control.y - sticks_[stick][1] * control.radius;
        const double knob_radius = 24 * (control.radius > 0 ? control.radius / 58.0 : 1.0);
        knobs_[stick].path = [UIBezierPath bezierPathWithOvalInRect:
            CGRectMake(x - knob_radius, y - knob_radius, knob_radius * 2, knob_radius * 2)].CGPath;
        knobs_[stick].fillColor = [UIColor colorWithWhite:1 alpha:held_[stick] != 0 ? 0.35 : 0.18].CGColor;
    }
    [CATransaction commit];
}

- (void)publish {
    set_apple_touch_input(touch_input_state(held_, sticks_));
    [self redraw];
}

- (void)touchesBegan:(NSSet<UITouch*>*)touches withEvent:(UIEvent*)event {
    (void)event;
    for (UITouch* touch in touches) {
        const CGPoint point = [touch locationInView:self];
        const int control = touch_control_at(layout_, point.x, point.y);
        if (control < 0) continue;
        CFDictionarySetValue(owners_, (__bridge const void*)touch, (const void*)(intptr_t)(control + 1));
        ++held_[control];
        if (control < 2) sticks_[control] = clamped_touch_stick(layout_[control], point.x, point.y);
    }
    [self publish];
}

- (void)touchesMoved:(NSSet<UITouch*>*)touches withEvent:(UIEvent*)event {
    (void)event;
    bool moved = false;
    for (UITouch* touch in touches) {
        const int control = (int)(intptr_t)CFDictionaryGetValue(owners_, (__bridge const void*)touch) - 1;
        if (control >= 0 && control < 2) {
            const CGPoint point = [touch locationInView:self];
            sticks_[control] = clamped_touch_stick(layout_[control], point.x, point.y);
            moved = true;
        }
    }
    if (moved) [self publish];
}

- (void)lift:(NSSet<UITouch*>*)touches {
    for (UITouch* touch in touches) {
        const int control = (int)(intptr_t)CFDictionaryGetValue(owners_, (__bridge const void*)touch) - 1;
        if (control < 0) continue;
        CFDictionaryRemoveValue(owners_, (__bridge const void*)touch);
        if (held_[control] > 0 && --held_[control] == 0 && control < 2) sticks_[control] = {};
    }
    [self publish];
}

- (void)touchesEnded:(NSSet<UITouch*>*)touches withEvent:(UIEvent*)event {
    (void)event;
    [self lift:touches];
}

- (void)touchesCancelled:(NSSet<UITouch*>*)touches withEvent:(UIEvent*)event {
    (void)event;
    [self lift:touches];
}

- (void)releaseAll {
    CFDictionaryRemoveAllValues(owners_);
    held_.fill(0);
    sticks_ = {};
    set_apple_touch_input({});
    [self redraw];
}

- (void)scriptPhase:(NSString*)phase slot:(NSInteger)slot point:(CGPoint)point {
    static NSMutableDictionary<NSNumber*, ArmigoScriptTouch*>* fingers;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ fingers = [NSMutableDictionary dictionary]; });
    NSNumber* key = @(slot);
    ArmigoScriptTouch* touch = fingers[key];
    if ([phase isEqualToString:@"down"]) {
        if (touch) {
            touch.scriptPhase = UITouchPhaseEnded;
            [self touchesEnded:[NSSet setWithObject:touch] withEvent:nil];
        }
        touch = [ArmigoScriptTouch new];
        touch.scriptPoint = point;
        touch.scriptPhase = UITouchPhaseBegan;
        fingers[key] = touch;
        [self touchesBegan:[NSSet setWithObject:touch] withEvent:nil];
    } else if (touch && [phase isEqualToString:@"move"]) {
        touch.scriptPoint = point;
        touch.scriptPhase = UITouchPhaseMoved;
        [self touchesMoved:[NSSet setWithObject:touch] withEvent:nil];
    } else if (touch && [phase isEqualToString:@"up"]) {
        touch.scriptPhase = UITouchPhaseEnded;
        [self touchesEnded:[NSSet setWithObject:touch] withEvent:nil];
        [fingers removeObjectForKey:key];
    } else if (touch && [phase isEqualToString:@"cancel"]) {
        touch.scriptPhase = UITouchPhaseCancelled;
        [self touchesCancelled:[NSSet setWithObject:touch] withEvent:nil];
        [fingers removeObjectForKey:key];
    }
}

@end

@implementation ArmigoIosMetalView
+ (Class)layerClass { return CAMetalLayer.class; }
- (void)layoutSubviews {
    [super layoutSubviews];
    CAMetalLayer* layer = (CAMetalLayer*)self.layer;
    const CGFloat scale = self.window.windowScene.screen.scale;
    layer.contentsScale = scale > 0 ? scale : 1;
    layer.drawableSize = CGSizeMake(self.bounds.size.width * layer.contentsScale,
                                   self.bounds.size.height * layer.contentsScale);
}
@end

static bool touch_pad_wanted() {
    for (GCController* controller in GCController.controllers) {
        if (controller.extendedGamepad) return false;
    }
    return true;
}

static void update_touch_pad(ArmigoIosMetalView* view) {
    const bool wanted = touch_pad_wanted();
    const bool shown = view.touchPad && !view.touchPad.hidden;
    if (wanted == shown) return;
    if (wanted && !view.touchPad) {
        view.touchPad = [[ArmigoTouchPad alloc] initWithFrame:view.bounds];
        view.touchPad.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
        [view addSubview:view.touchPad];
    }
    view.touchPad.hidden = !wanted;
    if (!wanted) [view.touchPad releaseAll];
    armigo::diag::write(armigo::diag::Level::info,  "Armigo iOS input: %s\n",
        wanted ? "no controller, showing the on-screen controller" : "controller present, on-screen controller hidden");
}

static void start_touch_script(ArmigoIosMetalView* view) {
    const char* name = std::getenv("ARMIGO_TOUCH_SCRIPT");
    if (!name || !*name || !view.touchPad) return;
    NSString* documents = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    NSString* path = [documents stringByAppendingPathComponent:@(name)];
    NSString* text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (!text) {
        armigo::diag::write(armigo::diag::Level::error,  "Armigo iOS touch replay: cannot read %s\n", path.fileSystemRepresentation);
        return;
    }
    NSMutableArray<NSArray<NSString*>*>* steps = [NSMutableArray array];
    for (NSString* raw in [text componentsSeparatedByString:@"\n"]) {
        NSArray<NSString*>* fields = [raw componentsSeparatedByCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
        if (fields.count >= 3 && ([fields[1] isEqualToString:@"down"] || [fields[1] isEqualToString:@"move"])) {
            [steps addObject:fields];
        } else if (fields.count >= 3 && [fields[1] isEqualToString:@"up"]) {
            [steps addObject:fields];
        }
    }
    [steps sortWithOptions:NSSortStable usingComparator:^NSComparisonResult(NSArray<NSString*>* a, NSArray<NSString*>* b) {
        return [@([a[0] doubleValue]) compare:@([b[0] doubleValue])];
    }];
    __block NSUInteger next = 0;
    const double started = CACurrentMediaTime();
    [NSTimer scheduledTimerWithTimeInterval:1.0 / 120 repeats:YES block:^(NSTimer* timer) {
        const double elapsed = CACurrentMediaTime() - started;
        while (next < steps.count && [steps[next][0] doubleValue] <= elapsed) {
            NSArray<NSString*>* fields = steps[next++];
            NSString* phase = fields[1];
            const NSInteger slot = fields[2].integerValue;
            CGPoint point = CGPointZero;
            if (fields.count > 3) sscanf(fields[3].UTF8String, "%lf,%lf", &point.x, &point.y);
            [view.touchPad scriptPhase:phase slot:slot point:point];
        }
        if (next == steps.count) [timer invalidate];
    }];
}

void armigo_ios_start_touch_controls(ArmigoIosMetalView* view, NSMutableArray<id>* tokens) {
    update_touch_pad(view);
    start_touch_script(view);
    NSNotificationCenter* notifications = NSNotificationCenter.defaultCenter;
    for (NSNotificationName name in @[GCControllerDidConnectNotification, GCControllerDidDisconnectNotification]) {
        [tokens addObject:[notifications addObserverForName:name object:nil queue:NSOperationQueue.mainQueue
            usingBlock:^(NSNotification*) { update_touch_pad(view); }]];
    }
}

void armigo_ios_release_touch_controls(ArmigoIosMetalView* view) {
    [view.touchPad releaseAll];
}
