// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
#pragma once
#import <Metal/Metal.h>

// Preserve real execution timestamps while injecting one deterministic error
// at the feedback boundary, without submitting invalid GPU commands.
@interface ArmigoFailedCommitFeedback : NSObject <MTL4CommitFeedback>
@property(nonatomic, strong) id<MTL4CommitFeedback> original;
@end
@implementation ArmigoFailedCommitFeedback
- (NSError*)error { return [NSError errorWithDomain:@"ArmigoSyntheticCommit" code:27 userInfo:nil]; }
- (CFTimeInterval)GPUStartTime { return self.original.GPUStartTime; }
- (CFTimeInterval)GPUEndTime { return self.original.GPUEndTime; }
@end

inline id<MTL4CommitFeedback> failed_commit_feedback(id<MTL4CommitFeedback> original) {
    auto result = [ArmigoFailedCommitFeedback new];
    result.original = original;
    return result;
}
