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

#import <UIKit/UIKit.h>

@class ArmigoTouchPad;

@interface ArmigoIosMetalView : UIView
@property(nonatomic, strong) ArmigoTouchPad* touchPad;
@end

void armigo_ios_start_touch_controls(ArmigoIosMetalView* view, NSMutableArray<id>* tokens);
void armigo_ios_release_touch_controls(ArmigoIosMetalView* view);
