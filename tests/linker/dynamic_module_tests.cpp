// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/linker/dynamic_module.h>

#include <cstdio>
#include <initializer_list>
#include <utility>

namespace {
using namespace armigo::linker;
using Image = std::vector<std::byte>;

void put(Image& image, std::size_t offset, std::uint64_t value, unsigned width = 8) {
    for (unsigned index = 0; index < width; ++index) {
        image.at(offset + index) = std::byte((value >> (index * 8)) & 0xff);
    }
}

void tags(Image& image, std::initializer_list<std::pair<std::uint64_t, std::uint64_t>> entries) {
    std::size_t at = 0x80;
    for (const auto& [tag, value] : entries) {
        put(image, at, tag);
        put(image, at + 8, value);
        at += 16;
    }
    put(image, at, 0);
    put(image, at + 8, 0);
}

Image fixture() {
    Image image(0x1000);
    put(image, 4, 0x40, 4);
    put(image, 0x40, 0x30444f4d, 4);
    put(image, 0x44, 0x40, 4);
    tags(image, {{4, 0x300}, {5, 0x400}, {10, 5}, {6, 0x500}, {11, 24},
                 {7, 0x600}, {8, 24}, {9, 24},
                 {36, 0x700}, {35, 16}, {37, 8}, {12, 0x20},
                 {25, 0x900}, {27, 8}});
    put(image, 0x300, 1, 4);
    put(image, 0x304, 2, 4);
    put(image, 0x308, 1, 4);
    image[0x401] = std::byte{'f'};
    image[0x402] = std::byte{'o'};
    image[0x403] = std::byte{'o'};
    put(image, 0x518, 1, 4);
    put(image, 0x51c, 0x12, 1);
    put(image, 0x51e, 1, 2);
    put(image, 0x520, 0x20);
    put(image, 0x528, 4);
    put(image, 0x600, 0x800);
    put(image, 0x608, (std::uint64_t{1} << 32) | 257);
    put(image, 0x610, static_cast<std::uint64_t>(-8));
    put(image, 0x700, 0x808);
    put(image, 0x708, 0b1011); // Direct target followed by bits selecting two words.
    return image;
}

bool check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
    }
    return condition;
}
}

int main() {
    auto image = fixture();
    const auto unchanged = image;
    auto result = read_dynamic_module(image);
    if (!check(result.ok(), result.message.c_str()) ||
        !check(image == unchanged, "Parsing modified the input.") ||
        !check(result.module->symbols.size() == 2 &&
               result.module->symbols[1].name == "foo", "Symbol decoding failed.") ||
        !check(result.module->relocations.size() == 1 &&
               result.module->relocations[0].addend == -8, "RELA decoding failed.") ||
        !check(result.module->packed_relative_targets ==
               std::vector<std::uint64_t>{0x808, 0x810, 0x820}, "RELR bitmap decoding failed.") ||
        !check(result.module->init == 0x20 && result.module->init_array_size == 8,
               "Initialization metadata decoding failed.")) {
        return 1;
    }

    for (const auto size : {0u, 7u, 0x44u, 0x100u, 0x507u, 0x903u}) {
        auto truncated = fixture();
        truncated.resize(size);
        if (!check(!read_dynamic_module(truncated).ok(), "A truncated image was accepted.")) {
            return 1;
        }
    }
    for (const auto& [offset, value] :
         std::initializer_list<std::pair<std::size_t, std::uint64_t>>{
             {0x608, 1031}, {0x608, (std::uint64_t{9} << 32) | 257},
             {0x600, 0xffc}, {0x600, 0xfffffffffffffff8ull},
             {0x700, 3}, {0x708, 0x1000},
             {0x304, 0xffffffffu}, {0x518, 100u}}) {
        auto malformed = fixture();
        put(malformed, offset, value, offset == 0x304 || offset == 0x518 ? 4 : 8);
        if (!check(!read_dynamic_module(malformed).ok(), "Malformed metadata was accepted.")) {
            return 1;
        }
    }
    auto invalid_relative = fixture();
    put(invalid_relative, 0x608, (std::uint64_t{1} << 32) | 1027);
    if (!check(!read_dynamic_module(invalid_relative).ok(), "Relative relocation had a symbol.")) {
        return 1;
    }
    DynamicLimits limited;
    limited.max_relocations = 2;
    if (!check(read_dynamic_module(fixture(), limited).error == DynamicError::resource_limit,
               "The relocation budget was not enforced.")) {
        return 1;
    }
    for (const auto none_kind : {0u, 256u}) {
        auto no_operation = fixture();
        put(no_operation, 0x608, none_kind);
        result = read_dynamic_module(no_operation);
        if (!check(result.ok() && result.module->relocations.empty(),
                   "A valid NONE relocation was rejected.")) {
            return 1;
        }
        limited = {};
        limited.max_relocation_records = 0;
        if (!check(read_dynamic_module(no_operation, limited).error == DynamicError::resource_limit,
                   "NONE records bypassed the encoded-record budget.")) {
            return 1;
        }
    }
    auto bad_null_symbol = fixture();
    put(bad_null_symbol, 0x500, 1, 4);
    if (!check(read_dynamic_module(bad_null_symbol).error == DynamicError::invalid_symbol,
               "A named null symbol was accepted.")) {
        return 1;
    }
    limited = {};
    limited.max_total_name_bytes = 2;
    if (!check(read_dynamic_module(fixture(), limited).error == DynamicError::resource_limit,
               "The aggregate name budget was not enforced.")) {
        return 1;
    }
    auto gnu = fixture();
    put(gnu, 0x80, 0x6ffffef5);
    put(gnu, 0x300, 1, 4);
    put(gnu, 0x304, 1, 4);
    put(gnu, 0x308, 1, 4);
    put(gnu, 0x30c, 0, 4);
    put(gnu, 0x310, 0);
    put(gnu, 0x318, 1, 4);
    put(gnu, 0x31c, 1, 4);
    result = read_dynamic_module(gnu);
    if (!check(result.ok() && result.module->symbols.size() == 2,
               "GNU-only symbol bounds failed.")) {
        return 1;
    }
    put(gnu, 0x318, 0);
    put(gnu, 0x608, 1027);
    result = read_dynamic_module(gnu);
    if (!check(result.ok() && result.module->symbols.size() == 1,
               "An empty GNU hash table lost the unhashed symbol prefix.")) {
        return 1;
    }
    put(gnu, 0x304, 0, 4);
    if (!check(read_dynamic_module(gnu).error == DynamicError::invalid_symbol,
               "A module with no null symbol was accepted.")) {
        return 1;
    }
    auto plt = fixture();
    // Keep a complete symbol table while selecting dependency and PLT records.
    tags(plt, {{4, 0x300}, {5, 0x400}, {10, 5}, {6, 0x500}, {11, 24},
               {1, 1}, {23, 0x600}, {2, 24}, {20, 7}});
    put(plt, 0x608, (std::uint64_t{1} << 32) | 1026);
    result = read_dynamic_module(plt);
    if (!check(result.ok() && result.module->dependencies == std::vector<std::string>{"foo"} &&
               result.module->relocations[0].kind == RelocationKind::jump_slot,
               "PLT or dependency parsing failed.")) {
        return 1;
    }
    auto duplicate = fixture();
    put(duplicate, 0x160, 5);
    if (!check(!read_dynamic_module(duplicate).ok(), "A duplicate dynamic tag was accepted.")) {
        return 1;
    }
    std::puts("Dynamic symbol, RELA, RELR, and malformed-input checks passed.");
    return 0;
}
