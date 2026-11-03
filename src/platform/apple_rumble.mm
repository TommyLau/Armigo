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
// Rumble: the first controller with haptics gets one Core Haptics engine per
// motor, each playing an endless continuous event whose intensity follows the
// motor's level. A controller with left and right handle actuators plays each
// motor on its own side; one with a single actuator gets one engine on its
// default locality, driven by the stronger side.

#import <armigo/platform/controller_rumble.h>

#import <CoreHaptics/CoreHaptics.h>
#import <Foundation/Foundation.h>
#import <GameController/GameController.h>

#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>

@interface ArmigoRumble : NSObject {
@public
    int motors;                                   // 2: [0] left, [1] right handle; 1: [0] default locality
    CHHapticEngine* engine[2];
    id<CHHapticAdvancedPatternPlayer> player[2];  // nil until the first non-zero level, and after a stop or reset
    float level[2];                               // intensity last sent to the player
}
@end
@implementation ArmigoRumble
@end

namespace armigo::platform {
namespace {

// Both levels in one word (left in the low half), so a pair is never mixed.
std::atomic<std::uint64_t> requested_levels{0};
std::atomic<bool> apply_pending{false};

dispatch_queue_t rumble_queue() {
    static dispatch_queue_t queue = dispatch_queue_create("armigo.rumble", DISPATCH_QUEUE_SERIAL);
    return queue;
}

// Accessed only on the rumble queue.
GCController* attached_controller = nil;
ArmigoRumble* attached_rumble = nil;

void rumble_apply();

void report_once(const char* message, NSError* error) {
    static std::atomic<bool> reported{false};
    if (reported.exchange(true)) return;
    armigo::diag::write(armigo::diag::Level::error,  "Armigo input: %s: %s\n", message,
                 error ? error.localizedDescription.UTF8String : "no detail");
}

void motor_set(ArmigoRumble* rumble, int motor, float value) {
    if (value == rumble->level[motor]) return;
    NSError* error = nil;
    if (!rumble->player[motor]) {
        if (value <= 0) {
            rumble->level[motor] = 0;
            return;
        }
        // Start at the requested intensity: the pattern's intensity parameter
        // applies from the first sample.
        CHHapticEvent* event = [[CHHapticEvent alloc]
            initWithEventType:CHHapticEventTypeHapticContinuous
                   parameters:@[[[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticIntensity
                                                                              value:1]]
                 relativeTime:0
                     duration:GCHapticDurationInfinite];
        CHHapticDynamicParameter* start = [[CHHapticDynamicParameter alloc]
            initWithParameterID:CHHapticDynamicParameterIDHapticIntensityControl value:value relativeTime:0];
        CHHapticPattern* pattern = [[CHHapticPattern alloc] initWithEvents:@[event] parameters:@[start] error:&error];
        id<CHHapticAdvancedPatternPlayer> player = nil;
        if (pattern && [rumble->engine[motor] startAndReturnError:&error])
            player = [rumble->engine[motor] createAdvancedPlayerWithPattern:pattern error:&error];
        if (!player || ![player startAtTime:CHHapticTimeImmediate error:&error]) {
            report_once("rumble could not start", error);
            return;
        }
        rumble->player[motor] = player;
        rumble->level[motor] = value;
        return;
    }
    CHHapticDynamicParameter* update = [[CHHapticDynamicParameter alloc]
        initWithParameterID:CHHapticDynamicParameterIDHapticIntensityControl value:value relativeTime:0];
    if (![rumble->player[motor] sendParameters:@[update] atTime:CHHapticTimeImmediate error:&error]) {
        report_once("rumble level update failed", error);
        rumble->player[motor] = nil;  // started again on the next non-zero level
        rumble->level[motor] = 0;
        return;
    }
    rumble->level[motor] = value;
}

ArmigoRumble* rumble_create(GCController* controller) {
    GCDeviceHaptics* haptics = controller.haptics;
    if (!haptics) return nil;
    NSSet<GCHapticsLocality>* have = haptics.supportedLocalities;
    const bool split = [have containsObject:GCHapticsLocalityLeftHandle] &&
                       [have containsObject:GCHapticsLocalityRightHandle];
    GCHapticsLocality where[2] = {split ? GCHapticsLocalityLeftHandle : GCHapticsLocalityDefault,
                                  GCHapticsLocalityRightHandle};
    ArmigoRumble* rumble = [ArmigoRumble new];
    for (int motor = 0; motor < (split ? 2 : 1); ++motor) {
        CHHapticEngine* engine = [haptics createEngineWithLocality:where[motor]];
        if (!engine) {
            report_once("the controller has no haptic engine; rumble off", nil);
            return nil;
        }
        // A stopped or reset engine loses its players; the next level starts it again.
        __weak ArmigoRumble* weak = rumble;
        void (^lost)(void) = ^{
            dispatch_async(rumble_queue(), ^{
                ArmigoRumble* strong = weak;
                if (!strong) return;
                strong->player[motor] = nil;
                strong->level[motor] = 0;
                // Restart the current level: the guest may hold a constant
                // value and send nothing new.
                rumble_apply();
            });
        };
        engine.stoppedHandler = ^(CHHapticEngineStoppedReason) { lost(); };
        engine.resetHandler = lost;
        rumble->engine[motor] = engine;
    }
    rumble->motors = split ? 2 : 1;
    return rumble;
}

void rumble_drop() {
    if (!attached_rumble) return;
    for (int motor = 0; motor < attached_rumble->motors; ++motor) {
        [attached_rumble->player[motor] stopAtTime:CHHapticTimeImmediate error:nil];
        [attached_rumble->engine[motor] stopWithCompletionHandler:nil];
    }
    attached_rumble->motors = 0;
    attached_rumble = nil;
    attached_controller = nil;
}

// Follows the controller the input poll reads (the first one), then applies
// the latest levels to it.
void rumble_apply() {
    GCController* controller = [GCController controllers].firstObject;
    if (controller != attached_controller) {
        rumble_drop();
        attached_controller = controller;
        if (controller) attached_rumble = rumble_create(controller);
    }
    if (!attached_rumble) return;
    const auto levels = requested_levels.load();
    const float left = std::bit_cast<float>(static_cast<std::uint32_t>(levels));
    const float right = std::bit_cast<float>(static_cast<std::uint32_t>(levels >> 32));
    if (attached_rumble->motors == 2) {
        motor_set(attached_rumble, 0, left);
        motor_set(attached_rumble, 1, right);
    } else if (attached_rumble->motors == 1) {
        motor_set(attached_rumble, 0, std::fmax(left, right));
    }
}

float clamped(float value) {
    return std::isfinite(value) ? std::fmin(std::fmax(value, 0.0f), 1.0f) : 0.0f;
}

} // namespace

void set_controller_rumble(float left, float right) noexcept {
    requested_levels.store(std::uint64_t{std::bit_cast<std::uint32_t>(clamped(left))} |
                           std::uint64_t{std::bit_cast<std::uint32_t>(clamped(right))} << 32);
    // Coalesce: the guest sends one value per motor many times a second; the
    // queue applies only the latest pair.
    if (apply_pending.exchange(true)) return;
    dispatch_async(rumble_queue(), ^{
        apply_pending.store(false);
        rumble_apply();
    });
}

float rumble_level(float amplitude_low, float amplitude_high) noexcept {
    const float low = amplitude_low > 0 ? amplitude_low : 0;  // NaN and negatives: 0
    const float high = amplitude_high > 0 ? amplitude_high : 0;
    const float level = std::sqrt(low * low + high * high);
    return level < 1 ? level : 1;
}

} // namespace armigo::platform
