// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "application_host.h"
#include "armigo/compat/title_settings.h"
#include "player_settings.h"
#include "report_json.h"
#include "runner_shell.h"
#include <armigo/services/settings.h>

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

// The Settings window of a packed app (the app menu's Settings..., Command-,):
// the same choices the iOS player offers in Settings, kept under the same
// keys in the app's defaults. They take effect the next time the game starts.
@interface ArmigoSettingsWindow : NSObject <NSWindowDelegate>
- (void)install;
- (void)show:(id)sender;
- (BOOL)writeSnapshot:(NSString*)path;
@end

@implementation ArmigoSettingsWindow {
    NSWindow* _window;
}

- (void)install {
    NSMenu* app = NSApp.mainMenu.itemArray.firstObject.submenu;
    if (!app) {
        // The player builds its menu bar when its window opens.
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 2), dispatch_get_main_queue(), ^{
            [self install];
        });
        return;
    }
    NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:@"Settings…" action:@selector(show:) keyEquivalent:@","];
    item.target = self;
    [app insertItem:item atIndex:0];
    [app insertItem:NSMenuItem.separatorItem atIndex:1];
}

- (NSPopUpButton*)popUpForKey:(NSString*)key titles:(NSArray<NSString*>*)titles values:(NSArray*)values {
    NSPopUpButton* button = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    id current = [NSUserDefaults.standardUserDefaults objectForKey:key];
    for (NSUInteger index = 0; index < titles.count; ++index) {
        [button addItemWithTitle:titles[index]];
        button.lastItem.representedObject = @[key, values[index]];
        if ([[current description] isEqualToString:[values[index] description]]) [button selectItem:button.lastItem];
    }
    button.target = self;
    button.action = @selector(choose:);
    [button.widthAnchor constraintEqualToConstant:240].active = YES;
    return button;
}

- (NSTextField*)heading:(NSString*)title {
    NSTextField* label = [NSTextField labelWithString:title];
    label.font = [NSFont boldSystemFontOfSize:NSFont.systemFontSize];
    return label;
}

- (void)build {
    NSPopUpButton* mode = [self popUpForKey:@"console_mode" titles:@[@"Docked", @"Handheld"]
                                     values:@[@"docked", @"handheld"]];
    NSPopUpButton* resolution = [self popUpForKey:@"resolution"
        titles:@[@"Auto (Native)", @"720p", @"900p", @"1080p", @"1440p", @"4K", @"Native"]
        values:@[@"auto", @"720p", @"900p", @"1080p", @"1440p", @"2160p", @"native"]];
    // No 120 fps: the title's menu and loading flow still depends on the frame
    // rate there (a scripted run takes another path at 120 Hz than at 60 Hz).
    NSPopUpButton* rate = [self popUpForKey:@"frame_rate" titles:@[@"Auto", @"30 fps", @"60 fps"]
                                     values:@[@0, @30, @60]];
    NSPopUpButton* vsync = [self popUpForKey:@"vsync" titles:@[@"On", @"Off"] values:@[@YES, @NO]];
    NSPopUpButton* overlay = [self popUpForKey:@"performance_overlay" titles:@[@"Off", @"On"] values:@[@NO, @YES]];
    NSGridView* grid = [NSGridView gridViewWithViews:@[
        @[[self heading:@"Display"], NSGridCell.emptyContentView],
        @[[NSTextField labelWithString:@"Console Mode:"], mode],
        @[[NSTextField labelWithString:@"Resolution:"], resolution],
        @[[NSTextField labelWithString:@"Frame Rate:"], rate],
        @[[NSTextField labelWithString:@"V-Sync:"], vsync],
        @[[self heading:@"Overlay"], NSGridCell.emptyContentView],
        @[[NSTextField labelWithString:@"Performance Overlay:"], overlay],
    ]];
    [grid columnAtIndex:0].xPlacement = NSGridCellPlacementTrailing;
    [grid cellAtColumnIndex:0 rowIndex:5].xPlacement = NSGridCellPlacementLeading;
    [grid cellAtColumnIndex:0 rowIndex:0].xPlacement = NSGridCellPlacementLeading;
    grid.rowSpacing = 10;
    NSTextField* note = [NSTextField wrappingLabelWithString:
        @"Game language follows this app's language in system Settings.\n"
        @"Changes take effect the next time the game starts."];
    note.textColor = NSColor.secondaryLabelColor;
    note.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    NSStackView* stack = [NSStackView stackViewWithViews:@[grid, note]];
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeLeading;
    stack.spacing = 16;
    stack.edgeInsets = NSEdgeInsetsMake(20, 20, 20, 20);
    [note.widthAnchor constraintEqualToConstant:380].active = YES;
    _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 420, 320)
                                          styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"Settings";
    _window.releasedWhenClosed = NO;
    _window.contentView = stack;
    [_window center];
}

- (void)show:(id)sender {
    (void)sender;
    if (!_window) [self build];
    [_window makeKeyAndOrderFront:nil];
}

- (BOOL)writeSnapshot:(NSString*)path {
    [NSApplication sharedApplication];
    if (!_window) [self build];
    NSView* view = _window.contentView;
    [view layoutSubtreeIfNeeded];
    NSBitmapImageRep* image = [view bitmapImageRepForCachingDisplayInRect:view.bounds];
    [view cacheDisplayInRect:view.bounds toBitmapImageRep:image];
    return [[image representationUsingType:NSBitmapImageFileTypePNG properties:@{}] writeToFile:path atomically:YES];
}

- (void)choose:(NSPopUpButton*)sender {
    NSArray* choice = sender.selectedItem.representedObject;
    [NSUserDefaults.standardUserDefaults setObject:choice[1] forKey:choice[0]];
}
@end

int main(int argc, char* argv[]) {
    armigo::diag::use_player_defaults();
    armigo::tools::raise_open_file_limit();
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::printf("Usage: armigo run --exefs DIR --romfs DIR [--shaders DIR] [options]\n"
                    "Usage: armigo pso-archive --shaders DIR --manifest FILE... [--cache DIR] [--threads N] "
                    "[--warm-only]\n%s\n",
                    armigo::tools::runner_run_usage().c_str());
        return 0;
    }
    if (argc >= 2 && std::string_view(argv[1]) == "pso-archive") {
        return armigo::tools::run_pso_archive_command(argc, argv);
    }
    // A packed app opened from the Finder gets no arguments and runs the game
    // it carries in its own bundle.
    const bool packed_launch = argc == 1 && !armigo::compat::packed_resources().empty();
    if (!packed_launch && (argc < 2 || std::string_view(argv[1]) != "run")) {
        armigo::diag::error("%s", "Expected --help, run, or pso-archive.\n");
        return 2;
    }
    @autoreleasepool {
        const std::vector<char*> arguments(argv + (packed_launch ? 1 : 2), argv + argc);
        auto parsed = armigo::tools::parse_runner_arguments(
            armigo::platform::PresentationMode::windowed, arguments);
        // The packed game, whether opened from the Finder or run without
        // explicit inputs, starts with the player's settings; runs with
        // explicit inputs stay on the deterministic defaults (American English,
        // the Americas) unless flags choose otherwise.
        const auto packed = armigo::compat::packed_resources();
        const bool packed_game = parsed.ok && !parsed.help && !packed.empty() && !parsed.arguments.container_file &&
                                 (parsed.arguments.exefs_directory.empty() ||
                                  parsed.arguments.exefs_directory == packed / "game" / "exefs");
        if (packed_game) {
            auto manifest = armigo::tools::read_language_manifest(packed / "languages.txt");
            if (const auto* error = std::get_if<std::string>(&manifest)) {
                armigo::diag::write(armigo::diag::Level::error,  "%s\n", error->c_str());
                return 1;
            }
            const auto locales = armigo::tools::preferred_host_locales();
            std::vector<std::string_view> preferred(locales.begin(), locales.end());
            const std::string language = armigo::tools::apply_language_manifest(
                std::get<armigo::tools::LanguageManifest>(manifest), preferred);
            if (language.empty()) {
                armigo::diag::error("%s", "This game supports none of your preferred languages.\n");
                return 1;
            }
            const auto choices = armigo::tools::read_player_choices();
            // Before anything here creates a Metal device.
            armigo::tools::apply_performance_overlay(choices);
            // A fixed frame rate pins the guest's display cadence; Auto runs
            // at the display's rate.
            if (choices.frame_rate && !std::getenv("ARMIGO_VI_REFRESH_HZ"))
                setenv("ARMIGO_VI_REFRESH_HZ", std::to_string(choices.frame_rate).c_str(), 1);
            // Native is the main display's pixels; flags given on the command
            // line keep their own operation mode and window.
            NSScreen* screen = NSScreen.mainScreen;
            const CGFloat scale = screen ? screen.backingScaleFactor : 1;
            const auto width = armigo::tools::resolution_width(choices.resolution, false,
                static_cast<std::uint32_t>(screen.frame.size.width * scale),
                static_cast<std::uint32_t>(screen.frame.size.height * scale));
            if (!width) {
                armigo::diag::error("%s", "The Resolution setting is not recognised.\n");
                return 1;
            }
            const auto mode = parsed.arguments.operation_mode;
            armigo::tools::apply_player_choices(choices, width, parsed.arguments);
            if (std::any_of(arguments.begin(), arguments.end(),
                            [](const char* argument) { return std::string_view(argument) == "--operation-mode"; }))
                parsed.arguments.operation_mode = mode;
            static ArmigoSettingsWindow* settings;
            settings = [ArmigoSettingsWindow new];
            if (const char* snapshot = std::getenv("ARMIGO_SETTINGS_SNAPSHOT"); snapshot && *snapshot) {
                // Verification aid: write the Settings window's contents as a PNG.
                if (![settings writeSnapshot:@(snapshot)]) armigo::diag::write(armigo::diag::Level::error,  "Cannot write %s\n", snapshot);
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                [settings install];
            });
            armigo::diag::write(armigo::diag::Level::info,  "Armigo settings: language=%s region=%u mode=%s resolution=%s/%u frame_rate=%s "
                         "vsync=%s overlay=%s\n", language.c_str(), armigo::services::set::console_region_code(),
                         choices.handheld ? "handheld" : "docked", choices.resolution.c_str(), width,
                         choices.frame_rate ? std::to_string(choices.frame_rate).c_str() : "auto",
                         choices.vsync ? "on" : "off", choices.performance_overlay ? "on" : "off");
        }
        if (parsed.help) {
            std::printf("Usage: armigo run --exefs DIR --romfs DIR [--shaders DIR] [options]\n"
                        "%s\n",
                        armigo::tools::runner_run_usage().c_str());
            return 0;
        }
        if (!parsed.ok) {
            armigo::diag::write(armigo::diag::Level::error,  "armigo run: %s\n", parsed.error.c_str());
            return 2;
        }
        const auto environment = armigo::tools::default_shell_environment();
        const auto report = armigo::tools::run_windowed_shell(
            parsed.arguments, environment, armigo::tools::create_application_host);
        const auto fields = armigo::tools::report_fields(report);
        if (!armigo::tools::emit_shell_report(@"run-windowed", fields,
                                              parsed.arguments.report_file)) {
            armigo::tools::exit_process_if_runtime_retained(1);
            return 1;
        }
        const int status = static_cast<int>(report.exit_code);
        armigo::tools::exit_process_if_runtime_retained(status);
        return status;
    }
}
