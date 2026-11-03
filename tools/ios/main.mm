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
#include "player_settings.h"
#include "report_json.h"
#include "runner_shell.h"
#include <armigo/runtime/launch_profile.h>
#include <armigo/runtime/embedded_modules.h>
#include <armigo/services/settings.h>

#import <UIKit/UIKit.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

// Applies the player's settings (player_settings.h) to this launch: the game's
// language from the app's and the device's languages, and the Settings page's
// choices. The screen is the device's in pixels.
void apply_player_settings(CGSize screen, armigo::tools::RunnerArguments& arguments) {
    auto choices = armigo::tools::read_player_choices();
    const bool automatic = choices.frame_rate == 0;
    choices.frame_rate = choices.frame_rate == 30 ? 30 : 60;
    choices.vsync = true;
    setenv("ARMIGO_VI_REFRESH_HZ", choices.frame_rate == 30 ? "30" : "60", 1);
    setenv("ARMIGO_IOS_FRAME_RATE_AUTO", automatic ? "1" : "0", 1);
    arguments.refresh_hz = choices.frame_rate;

    const auto root = std::filesystem::path(NSBundle.mainBundle.bundlePath.fileSystemRepresentation) / "Armigo";
    auto manifest = armigo::tools::read_language_manifest(root / "languages.txt");
    if (const auto* error = std::get_if<std::string>(&manifest)) throw std::runtime_error(*error);
    const auto locales = armigo::tools::preferred_host_locales();
    std::vector<std::string_view> preferred(locales.begin(), locales.end());
    const std::string language =
        armigo::tools::apply_language_manifest(std::get<armigo::tools::LanguageManifest>(manifest), preferred);
    if (language.empty()) throw std::runtime_error("The game supports none of the device's languages.");
    const auto width = armigo::tools::resolution_width(choices.resolution, true,
        static_cast<std::uint32_t>(std::max(screen.width, screen.height)),
        static_cast<std::uint32_t>(std::min(screen.width, screen.height)));
    if (!width) throw std::runtime_error("The Resolution setting is not recognised.");
    armigo::tools::apply_player_choices(choices, width, arguments);
    armigo::diag::write(armigo::diag::Level::info,  "Armigo iOS settings: language=%s region=%u mode=%s resolution=%s/%u frame_rate=%s%d "
        "overlay=%s\n", language.c_str(), armigo::services::set::console_region_code(),
        choices.handheld ? "handheld" : "docked", choices.resolution.c_str(), width, automatic ? "auto/" : "",
        choices.frame_rate, choices.performance_overlay ? "on" : "off");
}

NSString* run_bundled_title(CGSize screen) {
    using namespace armigo;
    @autoreleasepool {
        try {
            const std::filesystem::path root = std::filesystem::path(NSBundle.mainBundle.bundlePath.fileSystemRepresentation) / "Armigo";
            const auto exefs = root / "game/exefs";
            const auto romfs = root / "game/romfs";
            const auto settings = root / "title/profile.json";
            const auto shaders = root / "shaders";
            if (!std::filesystem::is_directory(exefs) || !std::filesystem::is_directory(romfs) ||
                !std::filesystem::is_regular_file(settings) ||
                !std::filesystem::is_directory(shaders))
                return @"Required game files are missing from this app.";
            const auto* linked = armigo_linked_embedded_bundle ? armigo_linked_embedded_bundle() : nullptr;
            if (!linked || linked->manifest.empty() || linked->backings.empty())
                return @"This player executable has no linked game code.";
            const auto npdm_path = exefs / "main.npdm";
            if (std::filesystem::file_size(npdm_path) > 1024 * 1024)
                return @"The bundled process metadata is invalid.";
            NSData* metadata = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:npdm_path.c_str()]];
            if (!metadata) return @"The bundled process metadata could not be read.";
            auto derived = runtime::derive_launch_profile(
                {static_cast<const std::byte*>(metadata.bytes), metadata.length},
                runtime::NpdmTrustPolicy::structurally_validated_extracted_input);
            if (const auto* error = std::get_if<runtime::LaunchFailure>(&derived)) {
                armigo::diag::write(armigo::diag::Level::error,  "Armigo iOS profile: %s\n", error->detail.c_str());
                return @"The bundled process metadata is not supported.";
            }
            char title[17];
            std::snprintf(title, sizeof(title), "%016llX",
                static_cast<unsigned long long>(std::get<runtime::LaunchProfile>(derived).title_id));
            NSFileManager* manager = NSFileManager.defaultManager;
            NSURL* documents = [manager URLForDirectory:NSDocumentDirectory inDomain:NSUserDomainMask
                                     appropriateForURL:nil create:YES error:nil];
            NSURL* caches = [manager URLForDirectory:NSCachesDirectory inDomain:NSUserDomainMask
                                  appropriateForURL:nil create:YES error:nil];
            if (!documents || !caches) return @"The app's save and cache folders are unavailable.";
            const auto saves = std::filesystem::path(documents.path.fileSystemRepresentation) / "Armigo" / title;
            const auto cache = std::filesystem::path(caches.path.fileSystemRepresentation) / "Armigo" / title;
            std::filesystem::create_directories(saves);
            std::filesystem::create_directories(cache);
            armigo::diag::open_last_run_log((cache / "last-run.log").c_str());
            if (!std::freopen((saves / "boot.log").c_str(), "w", stderr))
                return @"The app's startup log could not be opened.";
            setenv("ARMIGO_COMPAT_PATCHES", settings.c_str(), 1);
            setenv("ARMIGO_SHADER_CACHE", (cache / "shaders").c_str(), 1);
            setenv("ARMIGO_PSO_CACHE", (cache / "pipelines").c_str(), 1);
            if (::chdir((root / "title").c_str()) != 0) return @"The title settings folder is unavailable.";
            tools::RunnerArguments arguments;
            arguments.mode = platform::PresentationMode::windowed;
            arguments.exefs_directory = exefs;
            arguments.romfs_directory = romfs;
            // The iOS adapter obtains code from the linked provider. The shell
            // still requires an existing directory to select embedded loading.
            arguments.embedded_directory = root;
            arguments.shader_package_directory = shaders;
            arguments.save_directory = saves;
            arguments.report_file = saves / "run.json";
            arguments.fullscreen = true;
            apply_player_settings(screen, arguments);
            const auto addons = root / "game/addons";
            if (std::filesystem::exists(addons)) for (const auto& entry : std::filesystem::directory_iterator(addons)) {
                if (entry.is_directory()) arguments.addon_directories.push_back(entry.path());
            }
            const auto report = tools::run_windowed_shell(arguments, tools::default_shell_environment(),
                                                          tools::create_application_host);
            const auto fields = tools::report_fields(report);
            tools::emit_shell_report(@"run-ios", fields, arguments.report_file);
            tools::exit_process_if_runtime_retained(static_cast<int>(report.exit_code));
            return report.exit_code == tools::RunnerExitCode::success ? @"The game has stopped." :
                @"The game could not continue. Its startup log is available in the app's Documents folder.";
        } catch (const std::exception& error) {
            armigo::diag::write(armigo::diag::Level::error,  "Armigo iOS startup: %s\n", error.what());
            return @"The bundled game could not be opened.";
        }
    }
}
}

@interface ArmigoPlayerSceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(nonatomic, strong) UIWindow* window;
@end
@implementation ArmigoPlayerSceneDelegate
- (void)scene:(UIScene*)scene willConnectToSession:(UISceneSession*)session options:(UISceneConnectionOptions*)options {
    (void)session; (void)options;
    if (![scene isKindOfClass:UIWindowScene.class]) return;
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene*)scene];
    UIViewController* controller = [UIViewController new];
    controller.view.backgroundColor = UIColor.systemBackgroundColor;
    UILabel* label = [[UILabel alloc] initWithFrame:controller.view.bounds];
    label.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    label.numberOfLines = 0;
    label.textAlignment = NSTextAlignmentCenter;
    label.text = @"Starting game…";
    [controller.view addSubview:label];
    self.window.rootViewController = controller;
    [self.window makeKeyAndVisible];
    static bool started = false;
    if (started) return;
    started = true;
    // Enter outside a main-dispatch-queue block so the nested shell run loop
    // can continue servicing that queue while UIKit owns the outer lifecycle.
    const CGSize screen = ((UIWindowScene*)scene).screen.nativeBounds.size;
    CFRunLoopPerformBlock(CFRunLoopGetMain(), kCFRunLoopDefaultMode, ^{ label.text = run_bundled_title(screen); });
    CFRunLoopWakeUp(CFRunLoopGetMain());
}
@end

@interface ArmigoPlayerDelegate : UIResponder <UIApplicationDelegate>
@end
@implementation ArmigoPlayerDelegate
- (UISceneConfiguration*)application:(UIApplication*)application
    configurationForConnectingSceneSession:(UISceneSession*)session options:(UISceneConnectionOptions*)options {
    (void)application; (void)options;
    UISceneConfiguration* configuration = [[UISceneConfiguration alloc] initWithName:@"Armigo" sessionRole:session.role];
    configuration.delegateClass = ArmigoPlayerSceneDelegate.class;
    return configuration;
}
@end

int main(int argc, char** argv) {
    armigo::diag::use_player_defaults();
    armigo::tools::raise_open_file_limit();
    // Before UIKit or the game creates a Metal device.
    armigo::tools::apply_performance_overlay(armigo::tools::read_player_choices());
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(ArmigoPlayerDelegate.class)); }
}
