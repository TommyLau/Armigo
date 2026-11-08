// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "bundle.h"

#import <CoreServices/CoreServices.h>
#import <Foundation/Foundation.h>

namespace armigo::pack {

std::string register_application(const std::filesystem::path& app_path) {
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithFileSystemRepresentation:app_path.c_str() isDirectory:YES relativeToURL:nil];
        if (!url) return "Cannot name " + app_path.string() + " for LaunchServices.";
        // Updating replaces whatever LaunchServices recorded for this path.
        const OSStatus status = LSRegisterURL((__bridge CFURLRef)url, true);
        if (status != noErr) {
            return "LaunchServices did not register " + app_path.string() + " (status " + std::to_string(status) + ").";
        }
    }
    return {};
}

} // namespace armigo::pack
