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
#include <atomic>

#include <armigo/compat/patches.h>
#include <armigo/graphics/presentation_hints.h>
#include <armigo/aarch64/instruction.h>
#import <Foundation/Foundation.h>
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
                    ? object(entry, {"title_id", "main_build_id", "module", "original_callee", "calls", "layout"}, {"nvn_driver", "startup_user", "pipelines", "words", "present_link", "resolution"})
                    : object(entry, {"title_id", "main_build_id"}, {"nvn_driver", "startup_user", "pipelines", "words", "present_link", "resolution"});
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
                if (id link = record[@"present_link"]) {
                    require(CFGetTypeID((__bridge CFTypeRef)link) == CFBooleanGetTypeID(),
                            "Compatibility present_link must be a JSON boolean.");
                    patch.present_link = [link boolValue];
                }
                if (id words = record[@"words"]) {
                    require([words isKindOfClass:[NSArray class]] && [words count] > 0 && [words count] <= 64,
                            "Compatibility words must be a nonempty bounded array.");
                    std::set<std::uint64_t> word_offsets;
                    for (id item in words) {
                        auto word = object(item, {"offset", "expected_instruction", "replacement_instruction"},
                                           {"nvn_driver_only"});
                        WordPatch patch_word{hex(word[@"offset"]),
                            static_cast<std::uint32_t>(hex(word[@"expected_instruction"], 8)),
                            static_cast<std::uint32_t>(hex(word[@"replacement_instruction"], 8))};
                        if (id only = word[@"nvn_driver_only"]) {
                            require(CFGetTypeID((__bridge CFTypeRef)only) == CFBooleanGetTypeID(),
                                    "Compatibility nvn_driver_only must be a JSON boolean.");
                            patch_word.nvn_driver_only = [only boolValue];
                        }
                        require(word_offsets.insert(patch_word.offset).second, "Duplicate compatibility word offset.");
                        patch.words.push_back(patch_word);
                    }
                }
                if (id resolution = record[@"resolution"]) {
                    auto sites = object(resolution, {"tv_size", "screen_widths", "display_size", "window_crop", "floor",
                        "render_target_pool", "graphics_heap", "process_heap", "root_heap", "load_scale",
                        "swapchain_textures", "pool_size", "minimum_render_width"}, {"maximum_width", "target_rate"});
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
                        auto call = object(item, {"offset", "expected_instruction"});
                        CallPatch site{hex(call[@"offset"]), static_cast<std::uint32_t>(hex(call[@"expected_instruction"], 8))};
                        require(offsets.insert(site.offset).second, "Duplicate compatibility call offset.");
                        patch.calls.push_back(site);
                    }
                    auto layout = object(record[@"layout"], {"framework_pointer", "framework_period", "override_flag", "ratio", "interval", "base_interval"},
                                         {"frame_step"});
                    patch.layout = {hex(layout[@"framework_pointer"]), hex(layout[@"framework_period"]),
                        hex(layout[@"override_flag"]), hex(layout[@"ratio"]), hex(layout[@"interval"]), hex(layout[@"base_interval"])};
                    if (id step = layout[@"frame_step"]) {
                        patch.layout.frame_step = hex(step);
                        require(patch.layout.frame_step != 0, "Compatibility frame step must be nonzero when present.");
                    }
                }
                require(match_patch(result, patch.identity) == nullptr, "Duplicate compatibility title/build identity.");
                result.push_back(std::move(patch));
            }
            return result;
        } catch (const std::exception& error) { return std::string(error.what()); }
    }
}
execution::GateStatus handle_patch(const InstalledPatch& patch, execution::GateCall& call,
                                    const horizon::GuestBuffers& memory) noexcept {
    if (patch.description.module == "frame-rate") {
        std::uint32_t maximum_rate = 0;
        const bool updated = titles::title_01007ef00011e000::update_frame_rate(
            patch.description.layout, patch.module_base,
            call.x(0), static_cast<std::uint32_t>(call.x(1)), memory, &maximum_rate);
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
    const auto& text = image.segment(loader::NsoSegmentKind::text);
    const auto executable = [&](std::uint64_t offset) {
        return offset % 4 == 0 && offset >= text.image_offset &&
            offset - text.image_offset < text.content_size && text.content_size - (offset - text.image_offset) >= 4 &&
            offset <= image.bytes.size() && image.bytes.size() - offset >= 4;
    };
    for (const auto& word : patch.words) {
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
        if (!executable(call.offset) || !seen.insert(call.offset).second)
            return "Compatibility call site is invalid or duplicated.";
        std::uint32_t word = 0;
        std::memcpy(&word, image.bytes.data() + call.offset, 4);
        if (word != call.expected_instruction) return "Compatibility call instruction preimage mismatch.";
        const auto instruction = aarch64::decode(word);
        const auto target = aarch64::referenced_address(instruction, call.offset);
        if (instruction.opcode != aarch64::Opcode::bl || !target || *target != patch.original_callee)
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
