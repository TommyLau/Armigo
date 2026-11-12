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

std::variant<std::string, std::filesystem::path> packed_application_executable(
    const std::filesystem::path& app_path) {
    const auto plist = app_path / "Contents" / "Info.plist";
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithFileSystemRepresentation:plist.c_str() isDirectory:NO relativeToURL:nil];
        NSData* data = [NSData dataWithContentsOfURL:url];
        id info = data ? [NSPropertyListSerialization propertyListWithData:data options:NSPropertyListImmutable
                                                                  format:nil error:nil] : nil;
        if (![info isKindOfClass:NSDictionary.class])
            return "Cannot read the packed application's Info.plist: " + plist.string() + ".";
        id value = ((NSDictionary*)info)[@"CFBundleExecutable"];
        if (![value isKindOfClass:NSString.class] || ![(NSString*)value UTF8String])
            return "The packed application's Info.plist has no valid CFBundleExecutable: " + plist.string() + ".";
        const std::string name([(NSString*)value UTF8String],
                               [(NSString*)value lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
        if (name.empty() || name.find_first_not_of(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos)
            return "The packed application's CFBundleExecutable is not a valid packed executable name: " + plist.string() + ".";
        return app_path / "Contents" / "MacOS" / name;
    }
}

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
