// ============================================================================
// Armigo Tests - 01007EF00011E000 Internal Resolution Words
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/titles/01007EF00011E000/resolution.h>
#include <armigo/loader/nso_module.h>

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)

using namespace armigo;
namespace title = titles::title_01007ef00011e000;

std::uint32_t move(bool wide, bool keep, unsigned shift, std::uint16_t immediate, unsigned rd) {
    return (std::uint32_t(wide) << 31) | ((keep ? 3u : 2u) << 29) | (0b100101u << 23) |
        ((shift / 16) << 21) | (std::uint32_t(immediate) << 5) | rd;
}
std::uint16_t immediate(std::uint32_t word) { return std::uint16_t((word >> 5) & 0xffff); }

// A synthetic module: a console output of 1280x720, a 16 MiB render-target pool
// sized for 1280x720, a 32 MiB graphics heap, small process and root heaps, a
// 0.75 floor and a load scale of 30.
struct Fixture {
    loader::NsoModuleImage image;
    compat::ResolutionSites sites;
    std::map<std::uint64_t, std::uint32_t> words;
    Fixture() {
        image.bytes.resize(0x1000);
        image.segments[0].image_offset = 0;
        image.segments[0].content_size = 0x400;
        const auto put = [&](std::uint64_t offset, std::uint32_t word) {
            std::memcpy(image.bytes.data() + offset, &word, 4);
            words[offset] = word;
        };
        const auto floor = std::bit_cast<std::uint32_t>(0.75f), scale = std::bit_cast<std::uint32_t>(30.0f);
        put(0x10, move(true, false, 0, 1280, 9));
        put(0x14, move(true, true, 32, 720, 9));
        put(0x18, move(true, false, 0, 1280, 10));   // an inlined copy of the swapchain size
        put(0x1c, move(true, true, 32, 720, 10));
        put(0x20, move(true, false, 16, std::bit_cast<std::uint32_t>(1280.0f) >> 16, 8));
        put(0x24, move(true, true, 48, std::bit_cast<std::uint32_t>(1024.0f) >> 16, 8));
        put(0x30, move(true, false, 0, 1280, 8));
        put(0x34, move(true, true, 32, 720, 8));
        put(0x40, move(false, false, 0, 1280, 3));
        put(0x44, move(false, false, 0, 720, 4));
        put(0x50, move(false, false, 0, std::uint16_t(floor), 8));
        put(0x54, move(false, true, 16, std::uint16_t(floor >> 16), 8));
        put(0x60, move(false, false, 16, 0x100, 10));
        put(0x64, move(false, false, 16, 0x200, 0));
        put(0x68, move(false, false, 16, 0x8000, 0));
        put(0x6c, move(false, false, 16, 0x7f00, 8));
        put(0x70, move(false, false, 16, scale >> 16, 8));
        sites.tv_size = {{0x10, 0x14}, {0x18, 0x1c}};
        sites.screen_widths = {0x20, 0x24};
        sites.display_size = {0x30, 0x34};
        sites.window_crop = {0x40, 0x44};
        sites.floor = {0x50, 0x54};
        sites.render_target_pool = 0x60;
        sites.graphics_heap = 0x64;
        sites.process_heap = 0x68;
        sites.root_heap = 0x6c;
        sites.load_scale = 0x70;
        sites.swapchain_textures = 3;
        sites.pool_width = 1280;
        sites.pool_height = 720;
        sites.minimum_render_width = 1024;
    }
};

std::map<std::uint64_t, std::uint32_t> by_offset(const std::vector<compat::WordPatch>& words) {
    std::map<std::uint64_t, std::uint32_t> result;
    for (const auto& word : words) result[word.offset] = word.replacement_instruction;
    return result;
}

void check_disabled() {
    Fixture fixture;
    const auto planned = title::resolution_words(fixture.sites, {0, 60}, fixture.image);
    CHECK(std::holds_alternative<title::ResolutionPlan>(planned) &&
          std::get<title::ResolutionPlan>(planned).words.empty());
}

void check_scaled() {
    Fixture fixture;
    const auto planned = title::resolution_words(fixture.sites, {2567, 60}, fixture.image);
    CHECK(std::holds_alternative<title::ResolutionPlan>(planned));
    if (!std::holds_alternative<title::ResolutionPlan>(planned)) return;
    const auto& plan = std::get<title::ResolutionPlan>(planned);
    CHECK(plan.width == 2560 && plan.height == 1440 && plan.words.size() == 17);
    for (const auto& word : plan.words) {
        CHECK(fixture.words.count(word.offset) && fixture.words[word.offset] == word.expected_instruction);
        // Only the 16-bit immediate changes.
        CHECK((word.replacement_instruction & ~(0xffffu << 5)) == (word.expected_instruction & ~(0xffffu << 5)));
    }
    auto words = by_offset(plan.words);
    const auto wide = std::uint16_t(std::bit_cast<std::uint32_t>(2560.0f) >> 16);
    CHECK(immediate(words[0x10]) == 2560 && immediate(words[0x14]) == 1440);
    CHECK(immediate(words[0x18]) == 2560 && immediate(words[0x1c]) == 1440);
    CHECK(immediate(words[0x20]) == wide && immediate(words[0x24]) == wide);
    CHECK(immediate(words[0x30]) == 2560 && immediate(words[0x34]) == 1440);
    CHECK(immediate(words[0x40]) == 2560 && immediate(words[0x44]) == 1440);
    // Four times the pixels: a 64 MiB pool. The swapchain grows from
    // 5120 x 768 to 10240 x 1536 bytes per texture, three textures, rounded up
    // to 1 MiB; the heaps grow by both, rounded up to 2 MiB.
    CHECK(plan.render_target_pool == 0x4000000 && immediate(words[0x60]) == 0x400);
    const std::uint64_t graphics = 0x2000000 + (10240ull * 1536 - 5120ull * 768) * 3;
    CHECK(plan.graphics_heap == ((graphics + 0xfffff) & ~0xfffffull));
    const auto extra = ((plan.graphics_heap - 0x2000000 + plan.render_target_pool - 0x1000000) + 0x1fffff) & ~0x1fffffull;
    CHECK(plan.process_heap == 0x80000000ull + extra && plan.root_heap == 0x7f000000ull + extra);
    CHECK(immediate(words[0x68]) == std::uint16_t(plan.process_heap >> 16));
    // Floor 1024 / 2560, below the game's 0.75.
    const auto floor = (std::uint32_t(immediate(words[0x54])) << 16) | immediate(words[0x50]);
    CHECK(std::bit_cast<float>(floor) == 1024.0f / 2560.0f && plan.floor == 0.4f);
    // The load scale loads with one halfword and budgets about 0.71 of a 60 fps frame.
    const auto scale = std::bit_cast<std::uint32_t>(plan.load_scale);
    CHECK((scale & 0xffff) == 0 && immediate(words[0x70]) == scale >> 16);
    CHECK(plan.load_scale > 140.0f && plan.load_scale < 155.0f);
}

void check_floor_keeps_game_floor() {
    Fixture fixture;
    const auto planned = title::resolution_words(fixture.sites, {1280, 60}, fixture.image);
    CHECK(std::holds_alternative<title::ResolutionPlan>(planned) &&
          std::get<title::ResolutionPlan>(planned).floor == 0.75f);
}

void check_rejections() {
    {
        Fixture fixture;
        const auto word = move(false, true, 16, 0x100, 10);   // a MOVK where a MOVZ is expected
        std::memcpy(fixture.image.bytes.data() + 0x60, &word, 4);
        CHECK(std::holds_alternative<std::string>(title::resolution_words(fixture.sites, {1920, 60}, fixture.image)));
    }
    {
        Fixture fixture;
        const auto word = move(false, false, 0, 1281, 3);   // crop width disagrees with the swapchain width
        std::memcpy(fixture.image.bytes.data() + 0x40, &word, 4);
        CHECK(std::holds_alternative<std::string>(title::resolution_words(fixture.sites, {1920, 60}, fixture.image)));
    }
    {
        Fixture fixture;
        const auto word = move(true, true, 32, 1080, 10);   // a copy of the swapchain size disagrees
        std::memcpy(fixture.image.bytes.data() + 0x1c, &word, 4);
        CHECK(std::holds_alternative<std::string>(title::resolution_words(fixture.sites, {1920, 60}, fixture.image)));
    }
    {
        Fixture fixture;
        fixture.sites.tv_size.clear();
        CHECK(std::holds_alternative<std::string>(title::resolution_words(fixture.sites, {1920, 60}, fixture.image)));
    }
    {
        Fixture fixture;
        fixture.sites.load_scale = 0x800;   // outside text
        CHECK(std::holds_alternative<std::string>(title::resolution_words(fixture.sites, {1920, 60}, fixture.image)));
    }
}

void check_settings() {
    compat::ResolutionSites sites;
    sites.maximum_width = 0;
    sites.target_rate = 60;
    unsetenv("ARMIGO_COMPAT_RESOLUTION_WIDTH");
    unsetenv("ARMIGO_COMPAT_RESOLUTION_RATE");
    auto settings = title::resolution_settings(sites);
    CHECK(std::holds_alternative<title::ResolutionSettings>(settings) &&
          std::get<title::ResolutionSettings>(settings).maximum_width == 0);
    setenv("ARMIGO_COMPAT_RESOLUTION_WIDTH", "3456", 1);
    setenv("ARMIGO_COMPAT_RESOLUTION_RATE", "120", 1);
    settings = title::resolution_settings(sites);
    CHECK(std::holds_alternative<title::ResolutionSettings>(settings) &&
          std::get<title::ResolutionSettings>(settings).maximum_width == 3456 &&
          std::get<title::ResolutionSettings>(settings).target_rate == 120);
    setenv("ARMIGO_COMPAT_RESOLUTION_WIDTH", "34x", 1);
    CHECK(std::holds_alternative<std::string>(title::resolution_settings(sites)));
    setenv("ARMIGO_COMPAT_RESOLUTION_WIDTH", "5000", 1);
    CHECK(std::holds_alternative<std::string>(title::resolution_settings(sites)));
    unsetenv("ARMIGO_COMPAT_RESOLUTION_WIDTH");
    unsetenv("ARMIGO_COMPAT_RESOLUTION_RATE");
}
} // namespace

int main() {
    check_disabled();
    check_scaled();
    check_floor_keeps_game_floor();
    check_rejections();
    check_settings();
    if (failures) {
        std::fprintf(stderr, "%u resolution checks failed\n", failures);
        return 1;
    }
    std::puts("01007EF00011E000 resolution word checks passed");
    return 0;
}
