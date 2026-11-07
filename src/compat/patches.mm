// ============================================================================
// Armigo Compatibility - Runtime Game Patches
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include <algorithm>
#include <atomic>

#include <armigo/compat/patches.h>
#include <armigo/titles/01007EF00011E000/resolution.h>
#include <armigo/execution/gate_patch.h>
#include <bit>
#include <mutex>
#include <armigo/graphics/presentation_hints.h>
#include <armigo/aarch64/instruction.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <charconv>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace armigo::compat {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
NSDictionary* object(id value, std::initializer_list<const char*> keys,
                     std::initializer_list<const char*> optional = {}) {
    require([value isKindOfClass:[NSDictionary class]], "Compatibility record must be an object.");
    NSDictionary* result = value;
    std::size_t optional_count = 0;
    for (const auto key : optional) if (result[@(key)] != nil) ++optional_count;
    require(result.count == keys.size() + optional_count, "Compatibility record has missing or unknown fields.");
    for (const auto key : keys)
        require(result[@(key)] != nil, "Compatibility record is missing a required field.");
    return result;
}
std::string string(id value) {
    require([value isKindOfClass:[NSString class]], "Compatibility value must be a string.");
    return [value UTF8String];
}
std::string cache_member_name(id value) {
    auto name = string(value);
    require(!name.empty() && name.size() <= 256,
            "Compatibility shader cache name is empty or too long.");
    require(name.front() != '/' && name.find('\\') == std::string::npos &&
                name.find("//") == std::string::npos && name.find("..") == std::string::npos,
            "Compatibility shader cache name is not an archive member name.");
    return name;
}
// A profile note is plain English of 1 to 200 characters. NSString's length
// counts UTF-16 code units, which is the character count for this text.
std::string note_text(id value) {
    require([value isKindOfClass:[NSString class]], "Compatibility note must be a string.");
    NSString* text = value;
    require([text length] >= 1 && [text length] <= 200,
            "Compatibility note must be 1 to 200 characters.");
    const char* utf8 = [text UTF8String];
    require(utf8 != nullptr, "Compatibility note must be a string.");
    return utf8;
}
std::uint64_t hex(id value, std::size_t digits = 0) {
    auto text = string(value);
    require(text.starts_with("0x"), "Compatibility integers require hexadecimal strings with 0x prefix.");
    text.erase(0, 2);
    require(!text.empty() && text.size() <= 16 && (!digits || text.size() == digits),
            "Compatibility hexadecimal integer has invalid width.");
    std::uint64_t number = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), number, 16);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
            "Compatibility hexadecimal integer is invalid.");
    return number;
}
std::uint32_t layout_count(id value, std::uint32_t limit) {
    require([value isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
            [value doubleValue] >= 0 && [value doubleValue] <= limit &&
            [value doubleValue] == [value unsignedIntValue], "Compatibility layout number is out of range.");
    return [value unsignedIntValue];
}
bool exclusive_word_modes(const WordPatch& left, const WordPatch& right) {
    return left.frame_rate && right.frame_rate && left.frame_rate != right.frame_rate &&
           left.expected_instruction == right.expected_instruction;
}
bool valid_frame_rates(const FrameRateSettings& settings) {
    if (settings.default_rate != 30 && settings.default_rate != 60) return false;
    if (settings.supported_rates.empty() || settings.supported_rates.size() > 4) return false;
    std::set<std::uint32_t> rates;
    for (auto rate : settings.supported_rates)
        if ((rate != 30 && rate != 60 && rate != 120 && rate != variable_frame_rate) || !rates.insert(rate).second) return false;
    std::set<std::uint32_t> experimental;
    for (auto rate : settings.experimental_rates)
        if (!rates.contains(rate) || !experimental.insert(rate).second) return false;
    return rates.contains(settings.default_rate);
}
std::uint32_t frame_rate_value(id value, bool private_rate = false) {
    if (private_rate && [value isKindOfClass:[NSString class]] && [value isEqual:@"variable"])
        return variable_frame_rate;
    require([value isKindOfClass:[NSNumber class]] &&
            CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
            ([value isEqual:@30] || [value isEqual:@60] || (private_rate && [value isEqual:@120])),
            private_rate ? "Compatibility mode must be 30, 60, 120 or variable."
                         : "Compatibility default and display frame rates must be 30 or 60.");
    return [value unsignedIntValue];
}
}
PatchReadResult read_patch_descriptions(const std::filesystem::path& path) {
    @autoreleasepool {
        try {
            std::ifstream stream(path, std::ios::binary | std::ios::ate);
            require(bool(stream), "Cannot open compatibility patch file.");
            const auto size = stream.tellg();
            require(size > 0 && size <= 1024 * 1024, "Compatibility patch file exceeds size bounds.");
            std::string bytes(static_cast<std::size_t>(size), '\0');
            stream.seekg(0);
            require(bool(stream.read(bytes.data(), size)), "Cannot read compatibility patch file.");
            NSData* data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
            NSError* error = nil;
            id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            require(root != nil && error == nil, "Compatibility patch file is invalid JSON.");
            auto top = object(root, {"version", "patches"});
            require([top[@"version"] isKindOfClass:[NSNumber class]] &&
                    [top[@"version"] isEqual:@1], "Unsupported compatibility patch schema version.");
            id entries = top[@"patches"];
            require([entries isKindOfClass:[NSArray class]] && [entries count] <= 128,
                    "Compatibility patches must be a bounded array.");
            PatchDescriptions result;
            for (id entry in entries) {
                require([entry isKindOfClass:[NSDictionary class]], "Compatibility record must be an object.");
                const bool has_patch = entry[@"module"] != nil;
                auto record = has_patch
                    ? object(entry, {"title_id", "main_build_id", "module", "original_callee", "calls", "layout"}, {"nvn_driver", "startup_user", "pipelines", "words", "present_link", "resolution", "shader_inputs", "note", "frame_rate", "frame_rates"})
                    : object(entry, {"title_id", "main_build_id"}, {"nvn_driver", "startup_user", "pipelines", "words", "present_link", "resolution", "shader_inputs", "note", "frame_rate", "frame_rates"});
                PatchDescription patch;
                patch.identity.title_id = hex(record[@"title_id"], 16);
                auto build = string(record[@"main_build_id"]);
                require(build.size() == 64, "Compatibility build ID must contain 64 hexadecimal digits.");
                for (std::size_t i = 0; i < 32; ++i) {
                    unsigned value = 0;
                    auto parsed = std::from_chars(build.data() + i * 2, build.data() + i * 2 + 2, value, 16);
                    require(parsed.ec == std::errc{} && parsed.ptr == build.data() + i * 2 + 2,
                            "Compatibility build ID is invalid.");
                    patch.identity.main_build_id[i] = std::byte(value);
                }
                require(patch.identity.title_id != 0, "Compatibility title ID must be nonzero.");
                if (id enabled = record[@"nvn_driver"]) {
                    require(CFGetTypeID((__bridge CFTypeRef)enabled) == CFBooleanGetTypeID(),
                            "Compatibility nvn_driver must be a JSON boolean.");
                    patch.nvn_driver = [enabled boolValue];
                }
                if (id required = record[@"startup_user"]) {
                    require(CFGetTypeID((__bridge CFTypeRef)required) == CFBooleanGetTypeID(),
                            "Compatibility startup_user must be a JSON boolean.");
                    patch.startup_user = [required boolValue];
                }
                // The title's pipeline list: a file next to the profile.
                if (id list = record[@"pipelines"]) {
                    require([list isKindOfClass:[NSString class]] && [list length] != 0,
                            "Compatibility pipelines must be a non-empty JSON string.");
                    const std::string name([list UTF8String]);
                    require(name.find('/') == std::string::npos && name != "." && name != "..",
                            "Compatibility pipelines must name a file in the profile's folder.");
                    const auto file = path.parent_path() / name;
                    std::error_code missing;
                    require(std::filesystem::is_regular_file(file, missing),
                            "Compatibility pipeline list is missing from the profile's folder.");
                    patch.pso_manifest = file.string();
                }
                if (id inputs = record[@"shader_inputs"]) {
                    auto selected = object(inputs, {"containers", "vertex_depth"},
                                           {"caches", "fallbacks", "ignored_needle_offsets"});
                    const auto containers = string(selected[@"containers"]);
                    if (containers == "bnsh") patch.shader_inputs.containers = ShaderContainerSet::Bnsh;
                    else if (containers == "gfd-cache") patch.shader_inputs.containers = ShaderContainerSet::GfdCache;
                    else require(false, "Compatibility shader containers are bnsh or gfd-cache.");
                    const auto depth = string(selected[@"vertex_depth"]);
                    if (depth == "minus-one-to-one") patch.shader_inputs.vertex_depth = ShaderVertexDepth::NegativeOneToOne;
                    else if (depth == "zero-to-one") patch.shader_inputs.vertex_depth = ShaderVertexDepth::ZeroToOne;
                    else require(false, "Compatibility vertex depth is minus-one-to-one or zero-to-one.");
                    if (id caches = selected[@"caches"]) {
                        require([caches isKindOfClass:[NSArray class]] && [caches count] <= 64,
                                "Compatibility shader caches must be a bounded array.");
                        std::set<std::string> cache_names;
                        for (id item in caches) {
                            auto name = cache_member_name(item);
                            require(cache_names.insert(name).second, "Duplicate compatibility shader cache name.");
                            patch.shader_inputs.caches.push_back(std::move(name));
                        }
                    }
                    if (id fallbacks = selected[@"fallbacks"]) {
                        require([fallbacks isKindOfClass:[NSArray class]] && [fallbacks count] <= 16,
                                "Compatibility shader cache fallbacks must be a bounded array.");
                        std::set<std::string> fallback_names;
                        for (id item in fallbacks) {
                            auto fallback = object(item, {"cache", "alternatives"});
                            ShaderCacheFallback entry_fallback;
                            entry_fallback.cache = cache_member_name(fallback[@"cache"]);
                            require(std::find(patch.shader_inputs.caches.begin(), patch.shader_inputs.caches.end(),
                                              entry_fallback.cache) != patch.shader_inputs.caches.end(),
                                    "Compatibility shader cache fallback does not name a required cache.");
                            require(fallback_names.insert(entry_fallback.cache).second,
                                    "Duplicate compatibility shader cache fallback.");
                            id alternatives = fallback[@"alternatives"];
                            require([alternatives isKindOfClass:[NSArray class]] && [alternatives count] > 0 &&
                                        [alternatives count] <= 8,
                                    "Compatibility shader cache alternatives must be a nonempty bounded array.");
                            std::set<std::string> alternative_names;
                            for (id alternative in alternatives) {
                                auto name = cache_member_name(alternative);
                                require(name != entry_fallback.cache,
                                        "Compatibility shader cache alternative repeats the required cache.");
                                require(alternative_names.insert(name).second,
                                        "Duplicate compatibility shader cache alternative.");
                                entry_fallback.alternatives.push_back(std::move(name));
                            }
                            patch.shader_inputs.fallbacks.push_back(std::move(entry_fallback));
                        }
                    }
                    if (id offsets = selected[@"ignored_needle_offsets"]) {
                        require([offsets isKindOfClass:[NSArray class]] && [offsets count] <= 64,
                                "Compatibility ignored shader offsets must be a bounded array.");
                        std::set<std::uint64_t> seen_offsets;
                        for (id item in offsets) {
                            const auto offset = hex(item);
                            require(seen_offsets.insert(offset).second, "Duplicate compatibility ignored shader offset.");
                            patch.shader_inputs.ignored_needle_offsets.push_back(offset);
                        }
                    }
                    if (patch.shader_inputs.containers == ShaderContainerSet::Bnsh) {
                        require(patch.shader_inputs.caches.empty() && patch.shader_inputs.fallbacks.empty() &&
                                    patch.shader_inputs.ignored_needle_offsets.empty(),
                                "bnsh shader input does not name caches or ignored offsets.");
                    }
                }
                if (id link = record[@"present_link"]) {
                    require(CFGetTypeID((__bridge CFTypeRef)link) == CFBooleanGetTypeID(),
                            "Compatibility present_link must be a JSON boolean.");
                    patch.present_link = [link boolValue];
                }
                if (id note = record[@"note"]) patch.note = note_text(note);
                if (id rate = record[@"frame_rate"]) {
                    auto settings = object(rate, {"default", "display_refresh_rate"}, {"experimental_rates"});
                    patch.frame_rate = FrameRateSettings{frame_rate_value(settings[@"default"]),
                        frame_rate_value(settings[@"display_refresh_rate"]), {}};
                    id supported = record[@"frame_rates"];
                    require([supported isKindOfClass:[NSArray class]] && [supported count] > 0 && [supported count] <= 4,
                            "A title frame-rate policy requires an explicit frame_rates array.");
                    for (id value in supported) patch.frame_rate->supported_rates.push_back(frame_rate_value(value, true));
                    if (id experimental = settings[@"experimental_rates"]) {
                        require([experimental isKindOfClass:[NSArray class]] && [experimental count] <= 4,
                                "Experimental frame rates must be a bounded array of declared modes.");
                        for (id value in experimental)
                            patch.frame_rate->experimental_rates.push_back(frame_rate_value(value, true));
                    }
                    require(valid_frame_rates(*patch.frame_rate),
                            "Supported frame rates must be unique and include the title default.");
                }
                require(!record[@"frame_rates"] || patch.frame_rate.has_value(),
                        "Supported frame rates require a title frame-rate policy.");
                if (id words = record[@"words"]) {
                    require([words isKindOfClass:[NSArray class]] && [words count] > 0 && [words count] <= 128,
                            "Compatibility words must be a nonempty bounded array.");
                    for (id item in words) {
                        auto word = object(item, {"offset", "expected_instruction", "replacement_instruction", "note"},
                                           {"nvn_driver_only", "frame_rate"});
                        WordPatch patch_word{hex(word[@"offset"]),
                            static_cast<std::uint32_t>(hex(word[@"expected_instruction"], 8)),
                            static_cast<std::uint32_t>(hex(word[@"replacement_instruction"], 8))};
                        patch_word.note = note_text(word[@"note"]);
                        if (id rate = word[@"frame_rate"]) {
                            require(patch.frame_rate.has_value(),
                                    "A rate-specific word requires a title frame-rate policy.");
                            patch_word.frame_rate = frame_rate_value(rate, true);
                            require(std::find(patch.frame_rate->supported_rates.begin(), patch.frame_rate->supported_rates.end(),
                                              patch_word.frame_rate) != patch.frame_rate->supported_rates.end(),
                                    "A word's frame rate is not declared in frame_rates.");
                        }
                        if (id only = word[@"nvn_driver_only"]) {
                            require(CFGetTypeID((__bridge CFTypeRef)only) == CFBooleanGetTypeID(),
                                    "Compatibility nvn_driver_only must be a JSON boolean.");
                            patch_word.nvn_driver_only = [only boolValue];
                        }
                        for (const auto& previous : patch.words) {
                            if (previous.offset != patch_word.offset) continue;
                            require(exclusive_word_modes(previous, patch_word),
                                    "Overlapping compatibility word modes or inconsistent preimages.");
                        }
                        patch.words.push_back(patch_word);
                    }
                }
                if (id resolution = record[@"resolution"]) {
                    auto sites = object(resolution, {"tv_size", "screen_widths", "display_size", "window_crop", "floor",
                        "render_target_pool", "graphics_heap", "process_heap", "root_heap", "load_scale",
                        "swapchain_textures", "pool_size", "minimum_render_width"}, {"maximum_width", "target_rate", "gpu_budget_note", "gpu_budget_rate"});
                    const auto pair = [&](id value) {
                        require([value isKindOfClass:[NSArray class]] && [value count] == 2,
                                "Compatibility resolution site pairs need two offsets.");
                        return std::array<std::uint64_t, 2>{hex(value[0]), hex(value[1])};
                    };
                    const auto count = [&](id value, std::uint32_t limit) {
                        require([value isKindOfClass:[NSNumber class]] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() &&
                                [value doubleValue] >= 0 && [value doubleValue] <= limit &&
                                [value doubleValue] == [value unsignedIntValue],
                                "Compatibility resolution number is out of range.");
                        return std::uint32_t([value unsignedIntValue]);
                    };
                    ResolutionSites site;
                    // One pair, or a list of pairs for every copy of the initializer.
                    id tv = sites[@"tv_size"];
                    require([tv isKindOfClass:[NSArray class]] && [tv count] > 0, "Compatibility resolution site pairs need two offsets.");
                    if ([tv[0] isKindOfClass:[NSArray class]]) {
                        require([tv count] <= 8, "Compatibility resolution tv_size lists at most eight pairs.");
                        for (id item in tv) site.tv_size.push_back(pair(item));
                    } else {
                        site.tv_size.push_back(pair(tv));
                    }
                    site.screen_widths = pair(sites[@"screen_widths"]);
                    site.display_size = pair(sites[@"display_size"]);
                    site.window_crop = pair(sites[@"window_crop"]);
                    site.floor = pair(sites[@"floor"]);
                    site.render_target_pool = hex(sites[@"render_target_pool"]);
                    site.graphics_heap = hex(sites[@"graphics_heap"]);
                    site.process_heap = hex(sites[@"process_heap"]);
                    site.root_heap = hex(sites[@"root_heap"]);
                    site.load_scale = hex(sites[@"load_scale"]);
                    site.swapchain_textures = count(sites[@"swapchain_textures"], 8);
                    id pool = sites[@"pool_size"];
                    require([pool isKindOfClass:[NSArray class]] && [pool count] == 2,
                            "Compatibility resolution pool_size needs width and height.");
                    site.pool_width = count(pool[0], 8192);
                    site.pool_height = count(pool[1], 8192);
                    site.minimum_render_width = count(sites[@"minimum_render_width"], 8192);
                    if (id value = sites[@"maximum_width"]) site.maximum_width = count(value, 8192);
                    if (id value = sites[@"target_rate"]) site.target_rate = count(value, 1000);
                    if (id value = sites[@"gpu_budget_note"]) site.gpu_budget_note = note_text(value);
                    if (id value = sites[@"gpu_budget_rate"]) {
                        site.gpu_budget_rate = count(value, 120);
                        require(site.gpu_budget_rate == 120 && !site.gpu_budget_note.empty() &&
                                patch.frame_rate && std::find(patch.frame_rate->supported_rates.begin(),
                                    patch.frame_rate->supported_rates.end(), 120) != patch.frame_rate->supported_rates.end(),
                                "GPU-work resolution defaults require a note and a declared 120 mode.");
                    }
                    require(site.swapchain_textures && site.pool_width && site.pool_height &&
                            site.minimum_render_width && site.target_rate,
                            "Compatibility resolution parameters must be nonzero.");
                    patch.resolution = site;
                }
                if (has_patch) {
                    patch.module = string(record[@"module"]);
                    require(patch.module == "frame-rate", "Unknown compatibility module.");
                    patch.original_callee = hex(record[@"original_callee"]);
                    id calls = record[@"calls"];
                    require([calls isKindOfClass:[NSArray class]] && [calls count] > 0 && [calls count] <= 64,
                            "Compatibility calls must be a nonempty bounded array.");
                    std::set<std::uint64_t> offsets;
                    for (id item in calls) {
                        auto call = object(item, {"offset", "expected_instruction", "note"}, {"frame_rate", "nvn_driver_only", "phases"});
                        CallPatch site{hex(call[@"offset"]), static_cast<std::uint32_t>(hex(call[@"expected_instruction"], 8))};
                        site.note = note_text(call[@"note"]);
                        if (id phases = call[@"phases"]) {
                            auto fields = object(phases, {"original_callee", "counter", "phase_count"});
                            site.phases = PhaseCallPatch{hex(fields[@"original_callee"]),
                                hex(fields[@"counter"]), layout_count(fields[@"phase_count"], 65535)};
                            require(site.phases->phase_count != 0 && site.phases->counter % 4 == 0,
                                    "Compatibility phase counter must be aligned and its period positive.");
                        }
                        if (id rate = call[@"frame_rate"]) {
                            site.frame_rate = frame_rate_value(rate, true);
                            require(patch.frame_rate && std::find(patch.frame_rate->supported_rates.begin(),
                                patch.frame_rate->supported_rates.end(), site.frame_rate) != patch.frame_rate->supported_rates.end(),
                                "A call's frame rate is not declared in frame_rates.");
                        }
                        if (id driver = call[@"nvn_driver_only"]) {
                            require(CFGetTypeID((__bridge CFTypeRef)driver) == CFBooleanGetTypeID(),
                                    "Compatibility nvn_driver_only must be a JSON boolean.");
                            site.nvn_driver_only = [driver boolValue];
                        }
                        require(offsets.insert(site.offset).second, "Duplicate compatibility call offset.");
                        patch.calls.push_back(site);
                    }
                    const id layout_record = record[@"layout"];
                    require([layout_record isKindOfClass:[NSDictionary class]], "Compatibility layout must be an object.");
                    const id variant = layout_record[@"variant"];
                    const bool countdown = variant && [variant isEqual:@"fractional-countdown-30"];
                    const bool repeating = variant && [variant isEqual:@"fractional-repeat-30"];
                    if (repeating || countdown || (variant && [variant isEqual:@"fractional-counter-30"])) {
                        auto layout = object(layout_record, {"variant", "framework_pointer", "framework_period",
                            "object_register", "state", "active_state", "counter"}, {"repeat_flag", "repeat_frames"});
                        patch.layout.variant = repeating ? titles::FrameRateVariant::FractionalRepeat30
                            : countdown ? titles::FrameRateVariant::FractionalCountdown30
                            : titles::FrameRateVariant::FractionalCounter30;
                        patch.layout.framework_pointer = hex(layout[@"framework_pointer"]);
                        patch.layout.framework_period = hex(layout[@"framework_period"]);
                        patch.layout.object_register = layout_count(layout[@"object_register"], 30);
                        patch.layout.state = hex(layout[@"state"]);
                        patch.layout.active_state = layout_count(layout[@"active_state"], UINT32_MAX);
                        patch.layout.counter = hex(layout[@"counter"]);
                        require(patch.layout.counter % 4 == 0 && patch.layout.state % 4 == 0 &&
                                patch.layout.counter != patch.layout.state,
                                "Compatibility counter fields must be aligned and distinct.");
                        if (repeating) {
                            patch.layout.repeat_flag = hex(layout[@"repeat_flag"]);
                            patch.layout.repeat_frames = layout_count(layout[@"repeat_frames"], UINT32_MAX);
                            require(patch.layout.repeat_frames != 0,
                                    "Compatibility repeat delay must be positive.");
                            const auto flag = patch.layout.repeat_flag;
                            require(!(flag >= patch.layout.state && flag - patch.layout.state < 4) &&
                                    !(flag >= patch.layout.counter && flag - patch.layout.counter < 4),
                                    "Compatibility repeat flag must not overlap the state or counter.");
                        } else {
                            require(!layout[@"repeat_flag"] && !layout[@"repeat_frames"],
                                    "Repeat fields require the fractional-repeat-30 layout.");
                        }
                    } else {
                        require(!variant || [variant isEqual:@"interval-ratio-60"], "Unknown compatibility layout variant.");
                        auto layout = object(layout_record, {"framework_pointer", "framework_period", "override_flag", "ratio", "interval", "base_interval"},
                                             {"frame_step", "variant"});
                        patch.layout = {hex(layout[@"framework_pointer"]), hex(layout[@"framework_period"]),
                            hex(layout[@"override_flag"]), hex(layout[@"ratio"]), hex(layout[@"interval"]), hex(layout[@"base_interval"])};
                        if (id step = layout[@"frame_step"]) {
                            patch.layout.frame_step = hex(step);
                            require(patch.layout.frame_step != 0, "Compatibility frame step must be nonzero when present.");
                        }
                    }
                }
                for (const auto& call : patch.calls) if (call.phases)
                    require(patch.layout.variant == titles::FrameRateVariant::IntervalRatio60 &&
                            patch.layout.frame_step != 0,
                            "Compatibility phase calls require a published frame step.");
                require(match_patch(result, patch.identity) == nullptr, "Duplicate compatibility title/build identity.");
                result.push_back(std::move(patch));
            }
            return result;
        } catch (const std::exception& error) { return std::string(error.what()); }
    }
}
std::variant<std::vector<WordPatch>, std::string> select_patch_words(
    const PatchDescription& patch, std::string_view requested_rate, bool nvn_driver_enabled,
    std::uint32_t* admitted_rate) {
    if (admitted_rate) *admitted_rate = 0;
    if (!patch.frame_rate) return patch.words;
    auto rate = patch.frame_rate->default_rate;
    if (!requested_rate.empty() && requested_rate != "0") {
        if (requested_rate == "30") rate = 30;
        else if (requested_rate == "60") rate = 60;
        else if (requested_rate == "120") rate = 120;
        else if (requested_rate == "variable") rate = variable_frame_rate;
        else return "The title frame rate must be Auto (0), 30, 60, 120 or variable.";
    }
    if (!valid_frame_rates(*patch.frame_rate)) return "The title's supported frame rates are invalid.";
    if (std::find(patch.frame_rate->supported_rates.begin(), patch.frame_rate->supported_rates.end(), rate) ==
        patch.frame_rate->supported_rates.end())
        return "The requested frame rate is not supported by this title profile.";
    std::vector<WordPatch> result;
    std::set<std::uint64_t> selected_offsets;
    for (const auto& word : patch.words)
        if ((nvn_driver_enabled || !word.nvn_driver_only) && (word.frame_rate == 0 || word.frame_rate == rate)) {
            if (!selected_offsets.insert(word.offset).second)
                return "The selected title mode has overlapping instruction words.";
            result.push_back(word);
        }
    // The measured-period module already scales its common interval hooks
    // across admitted rates; it needs no duplicate words for each frequency.
    const bool measured_interval = patch.module == "frame-rate" &&
        patch.layout.variant == titles::FrameRateVariant::IntervalRatio60 && patch.layout.frame_step &&
        std::any_of(patch.calls.begin(), patch.calls.end(), [=](const auto& call) {
            return !call.frame_rate && !call.phases && !call.gpu_budget &&
                (nvn_driver_enabled || !call.nvn_driver_only);
        });
    if (!measured_interval && rate != patch.frame_rate->default_rate && std::none_of(result.begin(), result.end(),
            [rate](const auto& word) { return word.frame_rate == rate; }) &&
        std::none_of(patch.calls.begin(), patch.calls.end(), [=](const auto& call) {
            return call.frame_rate == rate && (nvn_driver_enabled || !call.nvn_driver_only);
        }))
        return "The requested frame rate has no effective patches for the selected graphics driver.";
    if (admitted_rate) *admitted_rate = rate == variable_frame_rate ? 0 : rate;
    return result;
}
std::variant<std::vector<CallPatch>, std::string> select_patch_calls(
    const PatchDescription& patch, std::string_view requested_rate, bool nvn_driver_enabled) {
    const auto words = select_patch_words(patch, requested_rate, nvn_driver_enabled);
    if (const auto* error = std::get_if<std::string>(&words)) return *error;
    auto rate = patch.frame_rate ? patch.frame_rate->default_rate : 0;
    if (patch.frame_rate && requested_rate == "variable") rate = variable_frame_rate;
    else if (patch.frame_rate && !requested_rate.empty() && requested_rate != "0")
        std::from_chars(requested_rate.data(), requested_rate.data() + requested_rate.size(), rate);
    std::vector<CallPatch> result;
    for (const auto& call : patch.calls)
        if ((nvn_driver_enabled || !call.nvn_driver_only) && (call.frame_rate == 0 || call.frame_rate == rate))
            result.push_back(call);
    return result;
}
struct GpuBudgetContext {
    std::mutex mutex;
    titles::title_01007ef00011e000::GpuBudgetState state;
};

void initialize_gpu_budget(InstalledPatch& patch) {
    if (std::any_of(patch.description.calls.begin(), patch.description.calls.end(),
                    [](const auto& call) { return call.gpu_budget.has_value(); }))
        patch.gpu_budget = std::make_shared<GpuBudgetContext>();
}

execution::CallHookPatchResult make_patch_hook(const CallPatch& site, std::uint32_t actual,
    std::uint64_t address, std::uint64_t trampoline, execution::CallHookCodeRegion gate) {
    if (!site.gpu_budget) return execution::make_call_hook_patch(actual, address, trampoline, gate);
    // Only the declared 32-bit MOVZ, LSL #16 load-scale form is intercepted.
    // A branch (not BL) preserves the instruction site's live link register.
    const auto reg = actual & 31;
    if ((actual & 0xffe00000u) != 0x52a00000u || reg == 18 || reg >= 30)
        return {.error = execution::CallHookPatchError::unsupported_site};
    const auto emitted = execution::make_gate_site_patch(actual, address, trampoline,
        {gate.address, gate.size}, {.move_wide = true});
    if (!emitted.ok()) return {.error = execution::CallHookPatchError::invalid_layout};
    const auto& patch = *emitted.patch;
    return {.patch = execution::CallHookPatch{address, trampoline, actual, patch.replacement_instruction,
        patch.trampoline, {patch.dispatch_site.return_address, address + 4, address + 4}}};
}

execution::GateStatus handle_patch(const InstalledPatch& patch, execution::GateCall& call,
                                    const horizon::GuestBuffers& memory,
                                    std::uint64_t call_return_address, PatchFeedback feedback) noexcept {
    if (patch.description.module == "frame-rate") {
        {
            const CallPatch* site = nullptr;
            for (const auto& candidate : patch.description.calls)
                if (call_return_address >= patch.module_base &&
                    call_return_address - patch.module_base >= 4 &&
                    call_return_address - patch.module_base - 4 == candidate.offset)
                    site = &candidate;
            if (!site) {
                call.set_failure_message("Compatibility call site has no retained description.");
                return execution::GateStatus::failed;
            }
            if (site->gpu_budget) {
                if (!patch.gpu_budget) {
                    call.set_failure_message("Compatibility GPU budget has no initialized state.");
                    return execution::GateStatus::failed;
                }
                const auto gpu_ns = feedback.gpu_work_ns ? feedback.gpu_work_ns(feedback.context) : 0;
                std::lock_guard lock(patch.gpu_budget->mutex);
                const auto scale = titles::title_01007ef00011e000::gpu_budget_scale(
                    patch.gpu_budget->state, gpu_ns, static_cast<std::int64_t>(call.x(0)),
                    site->gpu_budget->target_rate);
                if (scale && call.set_x(site->expected_instruction & 31, std::bit_cast<std::uint32_t>(*scale)))
                    return execution::GateStatus::resume;
                call.set_failure_message("Compatibility GPU budget state is invalid.");
                return execution::GateStatus::failed;
            }
            if (site->phases) {
                const auto& phases = *site->phases;
                titles::FramePhaseDispatch due;
                if (titles::advance_frame_phases({patch.description.layout.frame_step,
                        phases.counter, phases.phase_count}, patch.module_base, call.x(0), memory, due) &&
                    call.set_x(1, due.first) && call.set_x(2, due.count))
                    return execution::GateStatus::resume;
                call.set_failure_message("Compatibility phase handler could not access valid scheduler state.");
                return execution::GateStatus::failed;
            }
        }
        std::uint32_t maximum_rate = 0;
        const bool updated = titles::update_frame_rate(
            patch.description.layout, patch.module_base,
            call.x(patch.description.layout.variant != titles::FrameRateVariant::IntervalRatio60
                ? patch.description.layout.object_register : 0),
            static_cast<std::uint32_t>(call.x(1)), memory, &maximum_rate);
        graphics::set_maximum_presentation_rate(maximum_rate);
        if (updated) return execution::GateStatus::resume;
    }
    call.set_failure_message("Compatibility call handler could not access valid frame-rate state.");
    return execution::GateStatus::failed;
}
const PatchDescription* match_patch(std::span<const PatchDescription> patches,
                                    const TitleIdentity& identity) noexcept {
    for (const auto& patch : patches) if (patch.identity == identity) return &patch;
    return nullptr;
}
std::string validate_patch(const PatchDescription& patch, const loader::NsoModuleImage& image) {
    if (patch.frame_rate && !valid_frame_rates(*patch.frame_rate))
        return "The title's supported frame rates are invalid.";
    const auto& text = image.segment(loader::NsoSegmentKind::text);
    const auto executable = [&](std::uint64_t offset) {
        return offset % 4 == 0 && offset >= text.image_offset &&
            offset - text.image_offset < text.content_size && text.content_size - (offset - text.image_offset) >= 4 &&
            offset <= image.bytes.size() && image.bytes.size() - offset >= 4;
    };
    for (const auto& word : patch.words) {
        for (const auto& previous : patch.words) {
            if (&previous == &word) break;
            if (previous.offset == word.offset && !exclusive_word_modes(previous, word))
                return "Overlapping compatibility word modes or inconsistent preimages.";
        }
        if (word.frame_rate != 0 &&
            (!patch.frame_rate || std::find(patch.frame_rate->supported_rates.begin(),
                patch.frame_rate->supported_rates.end(), word.frame_rate) == patch.frame_rate->supported_rates.end()))
            return "Compatibility instruction word has an invalid frame-rate condition.";
        if (!executable(word.offset)) return "Compatibility instruction word is outside executable text.";
        std::uint32_t actual = 0;
        std::memcpy(&actual, image.bytes.data() + word.offset, 4);
        if (actual != word.expected_instruction) {
            armigo::diag::write(armigo::diag::Level::warning, 
                         "Armigo compatibility: instruction word at %#llx is %08x, expected %08x; "
                         "not patched.\n",
                         static_cast<unsigned long long>(word.offset), actual,
                         word.expected_instruction);
            return "Compatibility instruction word preimage mismatch.";
        }
        for (const auto& call : patch.calls)
            if (call.offset == word.offset) return "Compatibility instruction word overlaps a call hook.";
    }
    if (patch.module.empty() && patch.calls.empty()) return {};
    if (patch.module != "frame-rate") return "Unknown compatibility module.";
    if (patch.layout.variant == titles::FrameRateVariant::FractionalCounter30 ||
        patch.layout.variant == titles::FrameRateVariant::FractionalCountdown30 ||
        patch.layout.variant == titles::FrameRateVariant::FractionalRepeat30) {
        if (patch.layout.object_register > 30 || patch.layout.counter % 4 || patch.layout.state % 4 ||
            patch.layout.counter == patch.layout.state || patch.layout.frame_step != 0)
            return "Compatibility fractional counter layout is invalid.";
        if (patch.layout.variant == titles::FrameRateVariant::FractionalRepeat30 &&
            (patch.layout.repeat_frames == 0 ||
             (patch.layout.repeat_flag >= patch.layout.state && patch.layout.repeat_flag - patch.layout.state < 4) ||
             (patch.layout.repeat_flag >= patch.layout.counter && patch.layout.repeat_flag - patch.layout.counter < 4)))
            return "Compatibility fractional repeat layout is invalid.";
    } else if (patch.layout.variant != titles::FrameRateVariant::IntervalRatio60) {
        return "Unknown compatibility layout variant.";
    }
    if (patch.calls.empty() || patch.calls.size() > 64) return "Compatibility call count is invalid.";
    if (!executable(patch.original_callee)) return "Compatibility original callee is outside executable text.";
    const auto& data = image.segment(loader::NsoSegmentKind::data);
    if (patch.layout.framework_pointer < data.image_offset ||
        patch.layout.framework_pointer - data.image_offset > data.mapped_size ||
        data.mapped_size - (patch.layout.framework_pointer - data.image_offset) < 8)
        return "Compatibility framework pointer is outside module data.";
    if (patch.layout.frame_step != 0 &&
        (patch.layout.frame_step % 4 != 0 || patch.layout.frame_step < data.image_offset ||
         patch.layout.frame_step - data.image_offset > data.mapped_size ||
         data.mapped_size - (patch.layout.frame_step - data.image_offset) < 4))
        return "Compatibility frame step is outside module data.";
    if (patch.layout.frame_step != 0 && patch.layout.frame_step + 4 > patch.layout.framework_pointer &&
        patch.layout.frame_step < patch.layout.framework_pointer + 8)
        return "Compatibility frame step overlaps the framework pointer.";
    std::set<std::uint64_t> seen;
    for (const auto& call : patch.calls) {
        if (call.gpu_budget && (call.phases || call.gpu_budget->target_rate != 120 ||
            (call.expected_instruction & 0xffe00000u) != 0x52a00000u ||
            (call.expected_instruction & 31) == 18 || (call.expected_instruction & 31) >= 30))
            return "Compatibility GPU budget instruction is invalid.";
        if (call.phases &&
            (!call.phases->phase_count || call.phases->phase_count > 65535 ||
             call.phases->counter % 4 || !patch.layout.frame_step ||
             patch.layout.variant != titles::FrameRateVariant::IntervalRatio60))
            return "Compatibility phase call layout is invalid.";
        if (!executable(call_callee(patch, call)))
            return "Compatibility call callee is outside executable text.";
        if (call.frame_rate != 0 && (!patch.frame_rate ||
            std::find(patch.frame_rate->supported_rates.begin(), patch.frame_rate->supported_rates.end(),
                      call.frame_rate) == patch.frame_rate->supported_rates.end()))
            return "Compatibility call has an invalid frame-rate condition.";
        if (!executable(call.offset) || !seen.insert(call.offset).second)
            return "Compatibility call site is invalid or duplicated.";
        std::uint32_t word = 0;
        std::memcpy(&word, image.bytes.data() + call.offset, 4);
        if (word != call.expected_instruction) return "Compatibility call instruction preimage mismatch.";
        const auto instruction = aarch64::decode(word);
        const auto target = aarch64::referenced_address(instruction, call.offset);
        if (!call.gpu_budget &&
            (instruction.opcode != aarch64::Opcode::bl || !target || *target != call_callee(patch, call)))
            return "Compatibility instruction is not a BL to the declared original callee.";
    }
    return {};
}

namespace {
std::atomic<bool> g_present_link{false};
}

void set_present_link(bool enabled) noexcept {
    g_present_link.store(enabled, std::memory_order_relaxed);
    graphics::set_maximum_presentation_rate(0);
}

bool present_link() noexcept {
    return g_present_link.load(std::memory_order_relaxed);
}

} // namespace armigo::compat
