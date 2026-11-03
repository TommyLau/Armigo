// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "../loader/nso_fixture.h"

#include <armigo/aarch64/instruction.h>
#include <armigo/loader/exefs_directory.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace armigo::runtime::test {

inline constexpr std::uint64_t text_size = 0x2000;
inline constexpr std::uint64_t rodata_offset = 0x4000;
inline constexpr std::uint64_t rodata_size = 0x4000;
inline constexpr std::uint64_t data_offset = 0x8000;
inline constexpr std::uint64_t image_size = 0x9000;
inline constexpr std::uint64_t pointer_slot_offset = 0x6000;
inline constexpr std::uint64_t original_pointer_slot_offset = data_offset;
inline constexpr std::uint64_t packed_relative_offset = 0x8020;
inline constexpr std::uint64_t packed_relative_addend = 0x8028;
inline constexpr std::uint64_t literal_offset = 0x300;
inline constexpr std::uint64_t unreachable_word_offset = 0x308;
inline constexpr std::uint32_t unreachable_instruction_word = 0xd4000841u; // SVC #0x42.
inline constexpr std::uint32_t literal_x18_instruction_word = 0x91001652u; // ADD X18, X18, #5.
inline constexpr std::uint64_t literal_value =
    (std::uint64_t{literal_x18_instruction_word} << 32) | unreachable_instruction_word;

struct SymbolSpec {
    std::string name;
    std::uint64_t value = 0;
    std::uint64_t size = 0;
    bool defined = true;
};

struct RelocationSpec {
    std::uint64_t target = 0;
    std::uint32_t symbol = 0;
    std::uint32_t kind = 257;
    std::int64_t addend = 0;
};

struct ModuleSpec {
    std::string name;
    std::vector<std::string> dependencies;
    std::vector<SymbolSpec> symbols;
    std::vector<RelocationSpec> relocations;
    std::vector<std::uint64_t> packed_relative_targets;
    std::vector<std::byte> text = std::vector<std::byte>(text_size);
    std::vector<std::byte> rodata = std::vector<std::byte>(rodata_size);
    std::vector<std::byte> data = std::vector<std::byte>(image_size - data_offset);
    std::uint64_t initializer = 0;
};

inline void put(std::vector<std::byte>& bytes, std::size_t offset,
                std::uint64_t value, unsigned width = 8) {
    for (unsigned index = 0; index < width; ++index) {
        bytes.at(offset + index) = std::byte((value >> (8 * index)) & 0xffu);
    }
}

inline void put_word(ModuleSpec& module, std::uint64_t image_offset, std::uint32_t word) {
    if (image_offset < module.text.size()) {
        put(module.text, image_offset, word, 4);
    } else if (image_offset >= rodata_offset && image_offset < data_offset) {
        put(module.rodata, image_offset - rodata_offset, word, 4);
    } else {
        put(module.data, image_offset - data_offset, word, 4);
    }
}

inline void put_wide(ModuleSpec& module, std::uint64_t image_offset, std::uint64_t value) {
    if (image_offset < text_size) {
        put(module.text, image_offset, value);
    } else if (image_offset >= rodata_offset && image_offset < data_offset) {
        put(module.rodata, image_offset - rodata_offset, value);
    } else {
        put(module.data, image_offset - data_offset, value);
    }
}

inline std::vector<std::byte> build_module(ModuleSpec module) {
    using loader::test::FixtureSegment;
    using loader::test::FixtureSpec;

    const auto module_rodata_offset = std::max<std::uint64_t>(
        rodata_offset, (module.text.size() + 0xfff) & ~std::uint64_t{0xfff});
    const auto module_data_offset = module_rodata_offset + module.rodata.size();
    put(module.text, 4, module_rodata_offset, 4);
    auto& ro = module.rodata;
    put(ro, 0, 0x30444f4d, 4);
    put(ro, 4, 0x40, 4);
    put(ro, 8, 0x5000, 4);
    put(ro, 12, 0x5000, 4);
    put(ro, 16, 0x0f00, 4);
    put(ro, 20, 0x0f00, 4);
    put(ro, 24, 0x4100, 4);

    const std::uint64_t dynamic_offset = module_rodata_offset + 0x40;
    const std::uint64_t hash_offset = module_rodata_offset + 0x400;
    const std::uint64_t strings_offset = module_rodata_offset + 0x800;
    const std::uint64_t symbols_offset = module_rodata_offset + 0xc00;
    const std::uint64_t relocations_offset = module_rodata_offset + 0x1200;
    const std::uint64_t relr_offset = module_rodata_offset + 0x1600;

    std::vector<std::byte> strings(1);
    auto append_string = [&](std::string_view value) {
        const auto offset = static_cast<std::uint32_t>(strings.size());
        for (const char character : value) strings.push_back(std::byte(character));
        strings.push_back(std::byte{0});
        return offset;
    };
    std::vector<std::uint32_t> dependency_names;
    for (const auto& dependency : module.dependencies) {
        dependency_names.push_back(append_string(dependency));
    }
    std::vector<std::uint32_t> symbol_names;
    for (const auto& symbol : module.symbols) {
        symbol_names.push_back(append_string(symbol.name));
    }
    if (strings.size() > 0x300 || module.symbols.size() > 32 ||
        module.relocations.size() > 32 || module.packed_relative_targets.size() > 32) {
        return {};
    }
    std::memcpy(ro.data() + (strings_offset - module_rodata_offset), strings.data(), strings.size());

    const auto symbol_count = static_cast<std::uint32_t>(module.symbols.size() + 1);
    put(ro, hash_offset - module_rodata_offset, 1, 4);
    put(ro, hash_offset - module_rodata_offset + 4, symbol_count, 4);
    put(ro, hash_offset - module_rodata_offset + 8, symbol_count > 1 ? 1 : 0, 4);
    for (std::uint32_t index = 0; index < symbol_count; ++index) {
        put(ro, hash_offset - module_rodata_offset + 12 + index * 4, 0, 4);
    }
    for (std::size_t index = 0; index < module.symbols.size(); ++index) {
        const auto& symbol = module.symbols[index];
        const auto at = symbols_offset - module_rodata_offset + (index + 1) * 24;
        put(ro, at, symbol_names[index], 4);
        put(ro, at + 4, 0x12, 1);
        put(ro, at + 6, symbol.defined ? 1 : 0, 2);
        put(ro, at + 8, symbol.value);
        put(ro, at + 16, symbol.size);
    }
    for (std::size_t index = 0; index < module.relocations.size(); ++index) {
        const auto& relocation = module.relocations[index];
        const auto at = relocations_offset - module_rodata_offset + index * 24;
        put(ro, at, relocation.target);
        put(ro, at + 8, (std::uint64_t{relocation.symbol} << 32) | relocation.kind);
        put(ro, at + 16, static_cast<std::uint64_t>(relocation.addend));
    }
    for (std::size_t index = 0; index < module.packed_relative_targets.size(); ++index) {
        put(ro, relr_offset - module_rodata_offset + index * 8,
            module.packed_relative_targets[index]);
    }

    std::size_t dynamic = dynamic_offset - module_rodata_offset;
    auto tag = [&](std::uint64_t key, std::uint64_t value) {
        put(ro, dynamic, key);
        put(ro, dynamic + 8, value);
        dynamic += 16;
    };
    tag(4, hash_offset);
    tag(5, strings_offset);
    tag(10, strings.size());
    tag(6, symbols_offset);
    tag(11, 24);
    for (const auto dependency : dependency_names) tag(1, dependency);
    if (!module.relocations.empty()) {
        tag(7, relocations_offset);
        tag(8, module.relocations.size() * 24);
        tag(9, 24);
    }
    if (!module.packed_relative_targets.empty()) {
        tag(36, relr_offset);
        tag(35, module.packed_relative_targets.size() * 8);
        tag(37, 8);
    }
    if (module.initializer != 0) tag(12, module.initializer);
    tag(0, 0);

    FixtureSpec fixture;
    fixture.segments[0] = FixtureSegment{std::move(module.text), 0, false, true};
    fixture.segments[1] = FixtureSegment{std::move(module.rodata),
                                         static_cast<std::uint32_t>(module_rodata_offset), false, true};
    fixture.segments[2] = FixtureSegment{std::move(module.data),
                                         static_cast<std::uint32_t>(module_data_offset), false, true};
    fixture.rodata_extents = {{{0, 0}, {0x800, 0x300},
                               {0xc00, static_cast<std::uint32_t>(symbol_count * 24)}}};
    return loader::test::build_nso(fixture);
}

inline std::uint32_t load64(unsigned rt, unsigned rn, std::uint32_t byte_offset = 0) {
    return aarch64::encode_load_unsigned_offset(
        aarch64::Opcode::ldr_literal_x, rt, rn, byte_offset).word;
}

inline std::uint32_t store64(unsigned rt, unsigned rn, std::uint32_t byte_offset = 0) {
    return 0xf9000000u | ((byte_offset / 8) << 10) | (rn << 5) | rt;
}

inline loader::ExeFsDirectoryInput two_module_input(std::string entry_name = "_start",
                                                    bool provider_entry = false) {
    ModuleSpec consumer;
    consumer.name = "main";
    consumer.dependencies = {"sdk"};
    consumer.symbols = {
        {std::move(entry_name), 0x100, 0x80, true},
        {"initializer", 0x200, 0x10, true},
        {"local_helper", 0x240, 0x14, true},
        {"local_tail", 0x260, 0x4, true},
        {"provider_helper", 0, 0, false},
    };
    consumer.relocations = {{pointer_slot_offset, 5, 257, 0}};
    consumer.packed_relative_targets = {packed_relative_offset};
    consumer.initializer = 0x200;
    put_wide(consumer, packed_relative_offset, packed_relative_addend);
    put_wide(consumer, literal_offset, literal_value);
    put_word(consumer, unreachable_word_offset, unreachable_instruction_word);

    std::uint64_t pc = 0x100;
    auto emit = [&](std::uint32_t word) {
        put_word(consumer, pc, word);
        pc += 4;
    };
    emit(0xd10083ffu); // SUB SP, SP, #32.
    emit(0xa9017bf3u); // STP X19, X30, [SP, #16].
    emit(0xaa0003f3u); // MOV X19, X0.
    emit(aarch64::encode_add_immediate(true, 18, 18, 5).word);
    emit(aarch64::encode_cbz(false, true, 18, pc, pc + 8).word);
    emit(store64(18, 19, 32));
    emit(aarch64::encode_bl(pc, 0x200).word);
    emit(0xaa1303e0u); // MOV X0, X19.
    emit(aarch64::encode_adr(18, pc, 0x240).word);
    emit(aarch64::encode_blr(18).word);
    emit(0xaa1303e0u); // MOV X0, X19.
    emit(aarch64::encode_adr(18, pc, pointer_slot_offset).word);
    emit(load64(18, 18));
    emit(aarch64::encode_blr(18).word);
    emit(aarch64::encode_mrs(
        9, aarch64::system_register_encoding(3, 3, 13, 0, 2)).word);
    emit(store64(9, 19, 24));
    emit(aarch64::encode_svc(0x42).word);
    emit(aarch64::encode_load_literal(
        aarch64::Opcode::ldr_literal_x, 10, pc, literal_offset).word);
    emit(store64(10, 19, 48));
    emit(0xaa1303e0u); // MOV X0, X19.
    emit(0xa9417bf3u); // LDP X19, X30, [SP, #16].
    emit(0x910083ffu); // ADD SP, SP, #32.
    emit(aarch64::encode_ret().word);

    pc = 0x200;
    put_word(consumer, pc, load64(9, 0)); pc += 4;
    put_word(consumer, pc, aarch64::encode_add_immediate(true, 9, 9, 1).word); pc += 4;
    put_word(consumer, pc, store64(9, 0)); pc += 4;
    put_word(consumer, pc, aarch64::encode_ret().word);
    pc = 0x240;
    put_word(consumer, pc, load64(9, 0, 8)); pc += 4;
    put_word(consumer, pc, aarch64::encode_add_immediate(true, 9, 9, 1).word); pc += 4;
    put_word(consumer, pc, store64(9, 0, 8)); pc += 4;
    put_word(consumer, pc, aarch64::encode_adr(18, pc, 0x260).word); pc += 4;
    put_word(consumer, pc, aarch64::encode_br(18).word);
    put_word(consumer, 0x260, aarch64::encode_ret().word);

    ModuleSpec provider;
    provider.name = "sdk";
    provider.symbols = {{"provider_helper", 0x100, 0x10, true}};
    if (provider_entry) provider.symbols.push_back({"_start", 0x100, 0x10, true});
    put_word(provider, 0x100, load64(9, 0, 16));
    put_word(provider, 0x104, aarch64::encode_add_immediate(true, 9, 9, 1).word);
    put_word(provider, 0x108, store64(9, 0, 16));
    put_word(provider, 0x10c, aarch64::encode_ret().word);

    loader::ExeFsDirectoryInput input;
    input.modules.push_back({consumer.name, build_module(std::move(consumer))});
    input.modules.push_back({provider.name, build_module(std::move(provider))});
    return input;
}

inline loader::ExeFsDirectoryInput rootless_companion_input() {
    ModuleSpec main;
    main.name = "main";
    main.symbols = {{"_start", 0x100, 4, true}};
    put_word(main, 0x100, aarch64::encode_ret().word);

    ModuleSpec companion;
    companion.name = "sdk";

    loader::ExeFsDirectoryInput input;
    input.modules.push_back({main.name, build_module(std::move(main))});
    input.modules.push_back({companion.name, build_module(std::move(companion))});
    return input;
}

inline loader::ExeFsDirectoryInput original_process_input(bool include_rtld = true) {
    ModuleSpec main;
    main.name = "main";
    main.dependencies = {"sdk"};
    main.symbols = {
        {"main_body", 0x100, 4, true},
        {"service", 0, 0, false},
    };
    main.relocations = {{original_pointer_slot_offset, 2, 257, 0}};
    if (!include_rtld) {
        main.dependencies.clear();
        main.symbols.resize(1);
        main.relocations.clear();
    }
    put_word(main, 0, aarch64::encode_b(0, 0x100).word);
    put_word(main, 0x100, aarch64::encode_ret().word);

    ModuleSpec sdk;
    sdk.name = "sdk";
    sdk.symbols = {{"service", 0x100, 4, true}};
    put_word(sdk, 0x100, aarch64::encode_ret().word);

    loader::ExeFsDirectoryInput input;
    input.modules.push_back({main.name, build_module(std::move(main))});
    if (include_rtld) {
        ModuleSpec rtld;
        rtld.name = "rtld";
        rtld.symbols = {{"rtld_body", 0x100, 4, true}};
        put_word(rtld, 0, aarch64::encode_b(0, 0x100).word);
        put_word(rtld, 0x100, aarch64::encode_ret().word);
        input.modules.push_back({rtld.name, build_module(std::move(rtld))});
    }
    if (include_rtld) input.modules.push_back({sdk.name, build_module(std::move(sdk))});
    input.process_metadata = std::vector<std::byte>{std::byte{0x4d}, std::byte{0x50}};
    return input;
}

inline loader::ExeFsDirectoryInput fixup_count_input(std::size_t adr_count) {
    ModuleSpec module;
    module.name = "main";
    constexpr std::uint64_t entry = 0x100;
    const auto function_size = (adr_count + 1) * sizeof(std::uint32_t);
    module.text.resize(entry + function_size);
    module.symbols = {{"_start", entry, function_size, true}};
    for (std::size_t index = 0; index < adr_count; ++index) {
        const auto pc = entry + index * sizeof(std::uint32_t);
        put_word(module, pc, aarch64::encode_adr(18, pc, pc).word);
    }
    put_word(module, entry + adr_count * sizeof(std::uint32_t),
             aarch64::encode_ret().word);

    loader::ExeFsDirectoryInput input;
    input.modules.push_back({module.name, build_module(std::move(module))});
    return input;
}

} // namespace armigo::runtime::test
