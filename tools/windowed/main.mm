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
#include "player_frame_rate_menu.h"
#include "player_language_menu.h"
#include "player_overlay_setting.h"
#include "report_json.h"
#include "runner_shell.h"
#include <armigo/services/settings.h>

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#include <mach-o/dyld.h>
#include <spawn.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern char** environ;

// The Settings window (the app menu's Settings..., Command-,). A packed app
// keeps its display choices under the same keys as iOS and its game language
// separately; declared 120 fps modes require capable Mac displays. Choices take
// effect on the next launch. A command-line run offers the title's frame rate
// alone. New frame-rate, language and overlay choices offer a cooperative restart.
@interface ArmigoSettingsWindow : NSObject <NSWindowDelegate>
- (instancetype)initWithFrameRates:(ArmigoFrameRateMenu*)frameRates
                         languages:(ArmigoLanguageMenu*)languages
                           overlay:(ArmigoOverlaySetting*)overlay
                     frameRateOnly:(BOOL)frameRateOnly;
- (void)install;
- (void)show:(id)sender;
- (BOOL)writeSnapshot:(NSString*)path;
@end

@implementation ArmigoSettingsWindow {
    NSWindow* _window;
    ArmigoFrameRateMenu* _frameRates;
    ArmigoLanguageMenu* _languages;
    ArmigoOverlaySetting* _overlay;
    NSPopUpButton* _overlayChoice;
    NSTextField* _overlayStatus;
    BOOL _frameRateOnly;
    NSPopUpButton* _rate;
    NSTextField* _rateStatus;
    NSPopUpButton* _language;
    NSTextField* _languageStatus;
}

- (instancetype)initWithFrameRates:(ArmigoFrameRateMenu*)frameRates
                         languages:(ArmigoLanguageMenu*)languages
                           overlay:(ArmigoOverlaySetting*)overlay
                     frameRateOnly:(BOOL)frameRateOnly {
    self = [super init];
    if (!self) return nil;
    _frameRates = frameRates;
    _languages = languages;
    _overlay = overlay;
    _frameRateOnly = frameRateOnly;
    __weak ArmigoSettingsWindow* weak = self;
    frameRates.changed = ^{ [weak refreshFrameRate]; };
    languages.changed = ^{ [weak refreshLanguage]; };
    overlay.changed = ^{ [weak refreshOverlay]; };
    return self;
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

- (void)refreshFrameRate {
    if (!_rate) return;
    armigo::tools::populate_player_frame_rate_menu(_rate, _frameRates.selection, _frameRates.storedRate);
    _rateStatus.stringValue = _frameRates.status;
}

- (void)refreshLanguage {
    if (!_language) return;
    [_languages populate:_language];
    _languageStatus.stringValue = _languages.status;
}

- (void)refreshOverlay {
    if (!_overlayChoice) return;
    [_overlay populate:_overlayChoice];
    _overlayStatus.stringValue = _overlay.status;
}

- (void)build {
    _rate = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    _rate.target = self;
    _rate.action = @selector(chooseFrameRate:);
    [_rate.widthAnchor constraintEqualToConstant:240].active = YES;
    _rateStatus = [NSTextField labelWithString:@""];
    _rateStatus.textColor = NSColor.secondaryLabelColor;
    _rateStatus.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    [self refreshFrameRate];
    if (_languages) {
        _language = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
        _language.target = self;
        _language.action = @selector(chooseLanguage:);
        [_language.widthAnchor constraintEqualToConstant:240].active = YES;
        _languageStatus = [NSTextField labelWithString:@""];
        _languageStatus.textColor = NSColor.secondaryLabelColor;
        _languageStatus.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
        [self refreshLanguage];
    }
    NSArray<NSArray<NSView*>*>* rows;
    NSString* note;
    if (_frameRateOnly) {
        rows = @[
            @[[self heading:@"Display"], NSGridCell.emptyContentView],
            @[[NSTextField labelWithString:@"Frame Rate:"], _rate],
            @[NSGridCell.emptyContentView, _rateStatus],
        ];
        note = @"The frame rate is kept for this title. A new frame rate restarts the game;\n"
               @"the command line sets the other options.";
    } else {
        NSPopUpButton* mode = [self popUpForKey:@"console_mode" titles:@[@"Docked", @"Handheld"]
                                         values:@[@"docked", @"handheld"]];
        NSPopUpButton* resolution = [self popUpForKey:@"resolution"
            titles:@[@"Auto (Native)", @"720p", @"900p", @"1080p", @"1440p", @"4K", @"Native"]
            values:@[@"auto", @"720p", @"900p", @"1080p", @"1440p", @"2160p", @"native"]];
        NSPopUpButton* vsync = [self popUpForKey:@"vsync" titles:@[@"On", @"Off"] values:@[@YES, @NO]];
        _overlayChoice = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
        [_overlayChoice.widthAnchor constraintEqualToConstant:240].active = YES;
        _overlayStatus = [NSTextField wrappingLabelWithString:@""];
        [_overlayStatus.widthAnchor constraintEqualToConstant:240].active = YES;
        _overlayStatus.textColor = NSColor.secondaryLabelColor;
        _overlayStatus.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
        [self refreshOverlay];
        rows = @[
            @[[self heading:@"Game"], NSGridCell.emptyContentView],
            @[[NSTextField labelWithString:@"Language:"], _language],
            @[NSGridCell.emptyContentView, _languageStatus],
            @[[self heading:@"Display"], NSGridCell.emptyContentView],
            @[[NSTextField labelWithString:@"Console Mode:"], mode],
            @[[NSTextField labelWithString:@"Resolution:"], resolution],
            @[[NSTextField labelWithString:@"Frame Rate:"], _rate],
            @[NSGridCell.emptyContentView, _rateStatus],
            @[[NSTextField labelWithString:@"V-Sync:"], vsync],
            @[[self heading:@"Overlay"], NSGridCell.emptyContentView],
            @[[NSTextField labelWithString:@"Performance Overlay:"], _overlayChoice],
            @[NSGridCell.emptyContentView, _overlayStatus],
        ];
        note = @"Auto follows this app's language, then the system's preferred languages.\n"
               @"A new language, frame rate or performance overlay offers to restart; other changes take effect next launch.";
    }
    NSGridView* grid = [NSGridView gridViewWithViews:rows];
    [grid columnAtIndex:0].xPlacement = NSGridCellPlacementTrailing;
    for (NSInteger row = 0; row < grid.numberOfRows; ++row) {
        NSView* first = [grid cellAtColumnIndex:0 rowIndex:row].contentView;
        if ([first isKindOfClass:[NSTextField class]] && ((NSTextField*)first).font.fontDescriptor.symbolicTraits &
                                                             NSFontDescriptorTraitBold)
            [grid cellAtColumnIndex:0 rowIndex:row].xPlacement = NSGridCellPlacementLeading;
    }
    grid.rowSpacing = 10;
    NSTextField* label = [NSTextField wrappingLabelWithString:note];
    label.textColor = NSColor.secondaryLabelColor;
    label.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
    NSStackView* stack = [NSStackView stackViewWithViews:@[grid, label]];
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeLeading;
    stack.spacing = 16;
    stack.edgeInsets = NSEdgeInsetsMake(20, 20, 20, 20);
    [label.widthAnchor constraintEqualToConstant:380].active = YES;
    _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 420, _frameRateOnly ? 180 : 480)
                                          styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"Settings";
    _window.releasedWhenClosed = NO;
    // Opens over the game in full screen, on whichever space is active.
    _window.collectionBehavior = NSWindowCollectionBehaviorFullScreenAuxiliary |
                                 NSWindowCollectionBehaviorMoveToActiveSpace;
    _window.contentView = stack;
    [_window center];
}

- (void)show:(id)sender {
    (void)sender;
    if (!_window) [self build];
    [self refreshFrameRate];
    [self refreshLanguage];
    [self refreshOverlay];
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

- (void)chooseFrameRate:(NSPopUpButton*)sender {
    NSArray* choice = sender.selectedItem.representedObject;
    [_frameRates chooseRate:[choice[1] intValue]];
}

- (void)chooseLanguage:(NSPopUpButton*)sender {
    [_languages chooseLanguage:((NSString*)sender.selectedItem.representedObject).UTF8String];
}
@end

namespace {

// The command line and environment the game started with, kept for a restart
// from the settings menus, and whether one was asked for.
struct Restart {
    std::vector<std::string> arguments;
    std::vector<std::string> environment;
    bool requested = false;
    bool fullscreen = false;
};
Restart restart;

// The game's own window: the application delegate's, or the main window.
NSWindow* game_window() {
    id delegate = NSApp.delegate;
    if ([delegate respondsToSelector:@selector(window)]) return [delegate window];
    return NSApp.mainWindow;
}

// Clicks the button titled title in view or below it; false when there is none.
bool press_button(NSView* view, NSString* title) {
    if ([view isKindOfClass:NSButton.class] && [((NSButton*)view).title isEqualToString:title]) {
        [(NSButton*)view performClick:nil];
        return true;
    }
    for (NSView* subview in view.subviews)
        if (press_button(subview, title)) return true;
    return false;
}

// Installs the settings menus and window for the running game. Restarting
// closes the game the way Quit does and starts it again once the runtime has
// shut down.
void install_player_controls(const armigo::tools::MacFrameRateSelection& selection,
                             const std::string& key, bool frame_rate_only,
                             const armigo::tools::LanguageManifest* language_manifest = nullptr,
                             const std::string& language_key = {}, const std::string& running_language = {},
                             bool running_overlay = false, bool overlay_override = false) {
    static ArmigoFrameRateMenu* frame_rates;
    static ArmigoLanguageMenu* languages;
    static ArmigoSettingsWindow* settings;
    static ArmigoOverlaySetting* overlay;
    frame_rates = [[ArmigoFrameRateMenu alloc] initWithSelection:selection key:key];
    frame_rates.restart = ^{
        NSWindow* window = game_window();
        restart.fullscreen = window && (window.styleMask & NSWindowStyleMaskFullScreen);
        restart.requested = true;
        armigo::diag::write(armigo::diag::Level::info, "Armigo: restarting the game for its new settings.\n");
        [NSApp terminate:nil];
    };
    languages = language_manifest
        ? [[ArmigoLanguageMenu alloc] initWithManifest:*language_manifest key:language_key runningLanguage:running_language]
        : nil;
    languages.restart = frame_rates.restart;
    overlay = frame_rate_only ? nil : [[ArmigoOverlaySetting alloc]
        initWithDefaults:NSUserDefaults.standardUserDefaults key:@"performance_overlay"
        runningEnabled:running_overlay environmentOverride:overlay_override];
    overlay.restart = frame_rates.restart;
    settings = [[ArmigoSettingsWindow alloc] initWithFrameRates:frame_rates languages:languages overlay:overlay
                                                 frameRateOnly:frame_rate_only];
    if (const char* snapshot = std::getenv("ARMIGO_SETTINGS_SNAPSHOT"); snapshot && *snapshot) {
        // Verification aid: write the Settings window's contents as a PNG.
        if (![settings writeSnapshot:@(snapshot)])
            armigo::diag::write(armigo::diag::Level::error, "Cannot write %s\n", snapshot);
    }
    if (const char* choice = std::getenv("ARMIGO_FRAME_RATE_CHOICE"); choice && *choice) {
        // Verification aid: RATE@SECONDS brings the game to the front that
        // long after it starts, as a player's click would, chooses RATE in the
        // Frame Rate menu and clicks Restart Now in the prompt it shows. Timers
        // fire in every run loop mode, so the click also reaches a modal alert.
        int rate = 0;
        double seconds = 0;
        if (std::sscanf(choice, "%d@%lf", &rate, &seconds) == 2 && seconds >= 0) {
            auto after = [](double delay, void (^step)(void)) {
                NSTimer* timer = [NSTimer timerWithTimeInterval:delay repeats:NO block:^(NSTimer*) { step(); }];
                [NSRunLoop.mainRunLoop addTimer:timer forMode:NSRunLoopCommonModes];
            };
            after(seconds, ^{
                [NSApp activate];
                [game_window() makeKeyAndOrderFront:nil];
                after(0.5, ^{
                    after(0.5, ^{
                        for (NSWindow* window in NSApp.windows) {
                            const bool sheet = window.sheet;
                            if (press_button(window.contentView, @"Restart Now")) {
                                armigo::diag::write(armigo::diag::Level::info, "Armigo: clicked Restart Now in %s.\n",
                                                    sheet ? "a sheet" : "an alert");
                                return;
                            }
                        }
                        armigo::diag::error("%s", "Armigo: no Restart Now prompt to answer.\n");
                    });
                    [frame_rates chooseRate:rate];
                });
            });
        }
    }
    dispatch_async(dispatch_get_main_queue(), ^{
        [frame_rates install];
        [languages install];
        [settings install];
    });
}

// Replaces this process with the game started again: same process id, so the
// terminal and anything waiting on it keep following the game. Returns only
// when that fails.
void restart_game() {
    char path[PATH_MAX];
    std::uint32_t size = sizeof path;
    if (_NSGetExecutablePath(path, &size) != 0) {
        armigo::diag::error("%s", "Armigo: the game cannot restart: its executable path is unavailable.\n");
        return;
    }
    const auto command = armigo::tools::restart_command(restart.arguments, restart.environment, restart.fullscreen);
    std::vector<char*> arguments;
    for (const auto& argument : command.arguments) arguments.push_back(const_cast<char*>(argument.c_str()));
    arguments.push_back(nullptr);
    std::vector<char*> environment;
    for (const auto& entry : command.environment) environment.push_back(const_cast<char*>(entry.c_str()));
    environment.push_back(nullptr);
    std::fflush(nullptr);
    posix_spawnattr_t attributes;
    posix_spawn_file_actions_t files;
    int error = posix_spawnattr_init(&attributes);
    if (error == 0) {
        error = posix_spawn_file_actions_init(&files);
        // Only the standard streams carry over; every other descriptor closes.
        if (error == 0) {
            error = posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETEXEC | POSIX_SPAWN_CLOEXEC_DEFAULT);
            for (int descriptor : {0, 1, 2})
                if (error == 0) error = posix_spawn_file_actions_addinherit_np(&files, descriptor);
            pid_t pid = 0;
            if (error == 0)
                error = posix_spawn(&pid, path, &files, &attributes, arguments.data(), environment.data());
            posix_spawn_file_actions_destroy(&files);
        }
        posix_spawnattr_destroy(&attributes);
    }
    armigo::diag::error("Armigo: the game cannot restart: %s\n", std::strerror(error));
}

} // namespace

int main(int argc, char* argv[]) {
    restart.arguments.assign(argv, argv + argc);
    for (char** entry = environ; *entry; ++entry) restart.environment.emplace_back(*entry);
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
    armigo::diag::use_player_defaults(!packed_launch);
    if (!packed_launch && (argc < 2 || std::string_view(argv[1]) != "run")) {
        armigo::diag::error("%s", "Expected --help, run, or pso-archive.\n");
        return 2;
    }
    @autoreleasepool {
        const std::vector<char*> arguments(argv + (packed_launch ? 1 : 2), argv + argc);
        auto parsed = armigo::tools::parse_runner_arguments(
            armigo::platform::PresentationMode::windowed, arguments);
        armigo::tools::WindowedLaunchConfiguration configure_launch;
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
            const auto language_manifest = std::get<armigo::tools::LanguageManifest>(std::move(manifest));
            const auto language_key = armigo::tools::language_preference_key(std::nullopt);
            const auto language_choice = armigo::tools::game_language_choice(language_manifest,
                armigo::tools::read_language_preference(language_key));
            if (language_choice.fell_back) {
                armigo::diag::write(armigo::diag::Level::warning,
                    "Armigo settings: the stored game language is not supported by this title; using Auto.\n");
                armigo::tools::write_language_preference(language_key, "auto");
            }
            const auto locales = armigo::tools::preferred_host_locales();
            std::vector<std::string_view> preferred(locales.begin(), locales.end());
            const std::string language = armigo::tools::apply_language_manifest(
                language_manifest, preferred, language_choice.preference);
            if (language.empty()) {
                armigo::diag::error("%s", "This game supports none of your preferred languages.\n");
                return 1;
            }
            const auto choices = armigo::tools::read_player_choices();
            // Before anything here creates a Metal device.
            const bool overlay_override = std::getenv("MTL_HUD_ENABLED") != nullptr;
            armigo::tools::apply_performance_overlay(choices);
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
            // Window geometry and synchronization are needed before creating
            // the host. Frame-rate policy waits for that window's display.
            parsed.arguments.content_width = width;
            parsed.arguments.display_sync = choices.vsync;
            if (std::none_of(arguments.begin(), arguments.end(),
                            [](const char* argument) { return std::string_view(argument) == "--operation-mode"; }))
                parsed.arguments.operation_mode = choices.handheld ? armigo::services::OperationMode::Handheld
                                                                  : armigo::services::OperationMode::Console;
            configure_launch = armigo::tools::mac_player_launch_configuration(choices, width,
                [language_manifest, language_key, language, width, overlay_override](const armigo::tools::PlayerChoices& effective,
                                  const armigo::tools::MacFrameRateSelection& selection) {
                install_player_controls(selection, armigo::tools::frame_rate_preference_key(std::nullopt), false,
                    &language_manifest, language_key, language, effective.performance_overlay, overlay_override);
                armigo::diag::write(armigo::diag::Level::info,
                    "Armigo settings: language=%s region=%u mode=%s resolution=%s/%u frame_rate=%s "
                    "vsync=%s overlay=%s\n", language.c_str(), armigo::services::set::console_region_code(),
                    effective.handheld ? "handheld" : "docked", effective.resolution.c_str(), width,
                    effective.frame_rate ? std::to_string(effective.frame_rate).c_str() : "auto",
                    effective.vsync ? "on" : "off", effective.performance_overlay ? "on" : "off");
            });
        } else if (parsed.ok && !parsed.help) {
            // A title run from the command line keeps the frame rate chosen
            // for it in this app; the command line sets everything else.
            if (const auto policy = armigo::tools::find_title_frame_rate_policy(parsed.arguments)) {
                const auto key = armigo::tools::frame_rate_preference_key(policy->title_id);
                configure_launch = armigo::tools::mac_title_frame_rate_configuration(
                    armigo::tools::title_frame_rate_choices(*policy, armigo::tools::read_frame_rate_preference(key)),
                    [key](const armigo::tools::PlayerChoices& effective,
                          const armigo::tools::MacFrameRateSelection& selection) {
                    install_player_controls(selection, key, true);
                    armigo::diag::write(armigo::diag::Level::info, "Armigo settings: frame_rate=%s\n",
                        effective.frame_rate ? std::to_string(effective.frame_rate).c_str() : "auto");
                });
            }
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
        const auto configured = armigo::tools::apply_development_run_options(parsed.arguments,
            [NSBundle.mainBundle.infoDictionary[@"ArmigoDevelopmentPack"] boolValue]);
        if (const auto* error = std::get_if<armigo::runtime::LaunchFailure>(&configured)) {
            armigo::diag::error("Armigo development run: %s\n", error->detail.c_str());
            return 2;
        }
        const auto environment = armigo::tools::default_shell_environment();
        const auto report = armigo::tools::run_windowed_shell(
            parsed.arguments, environment, armigo::tools::create_application_host, configure_launch);
        const auto fields = armigo::tools::report_fields(report);
        const bool reported = packed_launch ? true : armigo::tools::emit_shell_report(@"run-windowed", fields,
                                                               parsed.arguments.report_file);
        if (packed_launch) {
            if (const auto* error = std::get_if<armigo::runtime::LaunchFailure>(&report.result.outcome)) {
                NSAlert* alert = [NSAlert new];
                alert.messageText = @"The game could not continue";
                alert.informativeText = [NSString stringWithUTF8String:error->detail.c_str()];
                [alert runModal];
            }
        }
        // A runtime that did not stop within its bound still has live threads
        // (possibly writing saves); it ends with the process, never restarts.
        if (restart.requested && !report.runtime_retained && report.result.quiescent)
            restart_game();
        else if (restart.requested)
            armigo::diag::error("%s", "Armigo: the game did not shut down cleanly, so it was not restarted; "
                                      "start it again to use the new settings.\n");
        if (!reported) {
            armigo::tools::exit_process_if_runtime_retained(1);
            return 1;
        }
        const int status = static_cast<int>(report.exit_code);
        armigo::tools::exit_process_if_runtime_retained(status);
        return status;
    }
}
