// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "bundle.h"
#include "ios.h"
#include <armigo/pack/naming.h>

#import <Foundation/Foundation.h>

#include <array>
#include <cstring>
#include <fstream>
#include <unistd.h>

namespace pack = armigo::pack;
namespace fs = std::filesystem;

namespace {
int failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); ++failures; } } while (false)

constexpr std::array<const char*, 16> languages{
    "en-US", "en-GB", "ja", "fr", "de", "es-419", "es", "it", "nl",
    "fr-CA", "pt", "ru", "ko", "zh-Hant", "zh-Hans", "pt-BR"};
constexpr std::array<const char*, 16> localizations{
    "en", "en-GB", "ja", "fr", "de", "es-419", "es", "it", "nl",
    "fr-CA", "pt-PT", "ru", "ko", "zh-Hant", "zh-Hans", "pt-BR"};

pack::PackedNames names(std::uint32_t mask) {
    std::vector<std::byte> nacp(pack::nacp::control_file_size);
    for (std::size_t i = 0; i < languages.size(); ++i) {
        const std::string title = "Synthetic " + std::string(languages[i]);
        std::memcpy(nacp.data() + i * 0x300, title.data(), title.size());
    }
    for (unsigned i = 0; i < 4; ++i) nacp[0x302c + i] = std::byte((mask >> (i * 8)) & 0xff);
    const auto result = pack::names_from_control(0x0100000000000001, nacp);
    CHECK(std::holds_alternative<pack::PackedNames>(result));
    return std::get<pack::PackedNames>(result);
}

pack::BundleSpec spec_for(std::uint32_t mask, const char* platform) {
    const auto title = names(mask);
    pack::BundleSpec spec;
    spec.platform = platform;
    spec.display_name = title.display_name;
    spec.file_name = title.file_name;
    spec.executable_name = title.executable_name;
    spec.bundle_identifier = title.bundle_identifier;
    spec.languages = title.languages;
    spec.localized_titles = title.localized_titles;
    return spec;
}

void write_bundle(const pack::BundleSpec& spec, const fs::path& path) {
    const auto contents = spec.platform == "ios" ? path : path / "Contents";
    const auto resources = spec.platform == "ios" ? path : contents / "Resources";
    fs::create_directories(resources);
    std::ofstream(contents / "Info.plist") << pack::info_plist(spec);
    for (const auto& [directory, strings] : pack::localized_names(spec)) {
        fs::create_directories(resources / directory);
        std::ofstream(resources / directory / "InfoPlist.strings") << strings;
    }
}

void check_bundle(const fs::path& executable, const fs::path& app, const char* preference,
                  const char* expected) {
    // Run inside the fixture so the main bundle advertises the game's languages.
    NSBundle* bundle = [NSBundle bundleWithPath:[NSString stringWithUTF8String:app.c_str()]];
    const auto executable_name = std::string([bundle.infoDictionary[@"CFBundleExecutable"] UTF8String]);
    const auto player = (fs::exists(app / "Contents") ? app / "Contents/MacOS" : app) / executable_name;
    if (!fs::exists(player)) {
        fs::create_directories(player.parent_path());
        fs::create_hard_link(executable, player);
    }
    // Each process gets its own language preferences before Foundation reads them.
    const auto error = pack::run_pack_tool({player.string(), "--bundle", app.string(), expected,
                                           "-AppleLanguages", "(" + std::string(preference) + ")"});
    if (!error.empty()) std::printf("%s: %s\n", preference, error.c_str());
    CHECK(error.empty());
}
} // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc > 3 && std::string_view(argv[1]) == "--bundle") {
            NSBundle* bundle = [NSBundle bundleWithPath:[NSString stringWithUTF8String:argv[2]]];
            if (![[NSProcessInfo processInfo].processName isEqualToString:bundle.infoDictionary[@"CFBundleExecutable"]]) {
                std::printf("The process name does not match CFBundleExecutable.\n");
                return 1;
            }
            NSString* expected = [NSString stringWithFormat:@"Synthetic %s", argv[3]];
            for (NSString* key in @[@"CFBundleDisplayName", @"CFBundleName"]) {
                NSString* actual = [bundle objectForInfoDictionaryKey:key];
                if (![actual isEqualToString:expected]) {
                    std::printf("%s: expected %s, got %s\n", key.UTF8String, expected.UTF8String,
                                actual.UTF8String);
                    return 1;
                }
            }
            if (std::filesystem::exists(fs::path(argv[2]) / "Contents")) {
                NSString* finder = [[NSFileManager defaultManager] displayNameAtPath:
                    [NSString stringWithUTF8String:argv[2]]];
                if (![finder isEqualToString:expected] && ![finder isEqualToString:[expected stringByAppendingString:@".app"]]) {
                    std::printf("Expected Finder name %s, got %s\n", expected.UTF8String, finder.UTF8String);
                    return 1;
                }
            }
            return 0;
        }
        const fs::path root = fs::temp_directory_path() / ("armigo-localizations-" + std::to_string(getpid()));
        const auto executable = fs::absolute(argv[0]);
        for (const char* platform : {"macos", "ios"}) {
            const auto spec = spec_for(0xffff, platform);
            const auto localized = pack::localized_names(spec);
            CHECK(localized.size() == languages.size());
            for (std::size_t i = 0; i < languages.size(); ++i) {
                CHECK(localized[i].first == std::string(localizations[i]) + ".lproj");
                CHECK(localized[i].second == "\"CFBundleDisplayName\" = \"Synthetic " +
                      std::string(languages[i]) + "\";\n\"CFBundleName\" = \"Synthetic " + languages[i] + "\";\n");
                const auto single = spec_for(1u << i, platform);
                NSString* plist = [NSString stringWithUTF8String:pack::info_plist(single).c_str()];
                NSDictionary* dict = [NSPropertyListSerialization propertyListWithData:
                    [plist dataUsingEncoding:NSUTF8StringEncoding] options:NSPropertyListImmutable format:nil error:nil];
                CHECK([dict[@"CFBundleDevelopmentRegion"] isEqualToString:
                    [NSString stringWithUTF8String:localizations[i]]]);
                const fs::path app = root / platform / std::to_string(i) / (single.file_name + ".app");
                write_bundle(single, app);
                check_bundle(executable, app, "ar", languages[i]);
            }
            const fs::path all = root / platform / "all" / (spec.file_name + ".app");
            write_bundle(spec, all);
            const std::pair<const char*, const char*> preferences[]{
                {"en-US", "en-US"}, {"en-GB", "en-GB"}, {"en-IN", "en-GB"},
                {"ja-JP", "ja"}, {"fr-FR", "fr"}, {"de-DE", "de"}, {"es-MX", "es-419"},
                {"es-AR", "es-419"}, {"es-419", "es-419"}, {"es-ES", "es"}, {"it-IT", "it"},
                {"nl-NL", "nl"}, {"fr-CA", "fr-CA"}, {"pt-PT", "pt"}, {"pt-BR", "pt-BR"},
                {"ru-RU", "ru"}, {"ko-KR", "ko"}, {"zh-HK", "zh-Hant"}, {"zh-TW", "zh-Hant"},
                {"zh-CN", "zh-Hans"}, {"zh-SG", "zh-Hans"}, {"ar", "en-US"}};
            for (const auto& [preference, expected] : preferences) check_bundle(executable, all, preference, expected);
            const auto asian = spec_for((1u << 12) | (1u << 13) | (1u << 14), platform);
            CHECK(asian.display_name == "Synthetic ko");
            const fs::path asian_app = root / platform / "asian" / (asian.file_name + ".app");
            write_bundle(asian, asian_app);
            check_bundle(executable, asian_app, "zh-HK", "zh-Hant");
            check_bundle(executable, asian_app, "zh-CN", "zh-Hans");
            check_bundle(executable, asian_app, "en-US", "ko");
            const auto chinese_spec = spec_for((1u << 13) | (1u << 14), platform);
            const fs::path chinese = root / platform / "chinese" / (chinese_spec.file_name + ".app");
            write_bundle(chinese_spec, chinese);
            check_bundle(executable, chinese, "en-US", "zh-Hant");
        }
        fs::remove_all(root);
        return failures ? 1 : 0;
    }
}
