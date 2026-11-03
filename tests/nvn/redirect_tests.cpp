// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "redirect.h"
#include <armigo/aarch64/instruction.h>
#include <array>
#include <cstdio>
#include <cstring>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)
}
int main() {
    using namespace armigo;
    std::array<std::byte, 128> bytes{};
    linker::DynamicModule metadata;
    metadata.symbols.push_back({"nvnBootstrapLoader", 32, 32, 1, 0x12, 0});
    execution::PreparedImage prepared;
    prepared.original_image_size = bytes.size();
    for (std::uint64_t at = 32; at < 48; at += 4) {
        constexpr std::uint32_t nop = 0xd503201f;
        std::memcpy(bytes.data() + at, &nop, 4);
        prepared.coverage.push_back({at, nop, nop});
    }
    const auto original = bytes;
    runtime::ModulePreparationView view{0, "sdk", 0x100000, bytes, {}, 0, metadata};
    const auto redirect = [&] { return nvn::redirect_bootstrap_loader(view, prepared, 0, 64); };
    CHECK(redirect().ok());
    std::uint32_t words[2]{};
    std::uintptr_t address = 0;
    std::memcpy(words, bytes.data() + 32, 8);
    std::memcpy(&address, bytes.data() + 40, 8);
    CHECK(words[0] == 0x58000050u && words[1] == 0xd61f0200u);
    CHECK(address == reinterpret_cast<std::uintptr_t>(&nvn::armigo_nvn_bootstrap_loader));
    CHECK(std::memcmp(bytes.data(), original.data(), 32) == 0);
    CHECK(std::memcmp(bytes.data() + 48, original.data() + 48, 80) == 0);
    bytes = original;
    std::array<std::byte, 16384> keys{};
    std::array<std::byte, 64> appendix{};
    view.appended_data = keys;
    view.appended_data_offset = 0x4000;
    view.appended_code = appendix;
    view.appended_code_offset = 0x8000;
    CHECK(nvn::redirect_bootstrap_loader(view, prepared, 0, 64, 0).ok());
    std::uint32_t branch = 0;
    std::memcpy(&branch, bytes.data() + 32, 4);
    CHECK(aarch64::referenced_address(aarch64::decode(branch), view.base_address + 32) ==
          view.base_address + 0x8000);
    std::memcpy(&address, keys.data() + 16, 8);
    CHECK(address == reinterpret_cast<std::uintptr_t>(&nvn::armigo_nvn_bootstrap_loader));
    const auto first_code = appendix;
    bytes = original;
    keys = {};
    appendix = {};
    view.base_address += 0x100000000ull;
    CHECK(nvn::redirect_bootstrap_loader(view, prepared, 0, 64, 0).ok());
    CHECK(appendix == first_code);
    bytes = original;
    keys = {};
    appendix = {};
    CHECK(!nvn::redirect_bootstrap_loader(view, prepared, 0, 64, 60).ok());
    CHECK(bytes == original && keys == (std::array<std::byte, 16384>{}) &&
          appendix == (std::array<std::byte, 64>{}));
    const auto rejects = [&] {
        CHECK(!redirect().ok()); CHECK(bytes == original);
        CHECK(!nvn::redirect_bootstrap_loader(view, prepared, 0, 64, 0).ok());
        CHECK(bytes == original && keys == (std::array<std::byte, 16384>{}) &&
              appendix == (std::array<std::byte, 64>{}));
    };
    prepared.replacements.push_back({44, 0, 0}); rejects(); prepared.replacements.clear();
    prepared.fixups.push_back({execution::PreparedFixupKind::module_address_mov_wide, 20, {}});
    rejects(); prepared.fixups.clear();
    prepared.fixups.push_back({execution::PreparedFixupKind::gate_dispatcher_address, 28, {}});
    rejects(); prepared.fixups.clear();
    metadata.relocations.push_back({28}); rejects(); metadata.relocations.clear();
    metadata.packed_relative_targets.push_back(44); rejects(); metadata.packed_relative_targets.clear();
    metadata.symbols.push_back(metadata.symbols.front()); rejects(); metadata.symbols.pop_back();
    metadata.symbols.front().section = 0; rejects(); metadata.symbols.front().section = 1;
    metadata.symbols.front().size = 0; rejects();
    metadata.symbols.front().size = 32;
    metadata.symbols.push_back({"adjacentFunction", 40, 8, 1, 0x12, 0});
    rejects(); metadata.symbols.pop_back();
    metadata.symbols.front().size = 12; rejects(); metadata.symbols.front().size = 32;
    metadata.symbols.front().info = 0x11; rejects(); metadata.symbols.front().info = 0x12;
    metadata.symbols.front().value = 60; rejects(); metadata.symbols.front().value = 32;
    prepared.coverage.pop_back(); rejects();
    const std::string_view known = "NintendoSDK_NVN-20_5_22-Release";
    const std::string_view unknown = "NintendoSDK_NVN-99_99_99-Release";
    const auto image = std::as_bytes(std::span(known.data(), known.size()));
    CHECK(!nvn::select_driver(false, image).version);
    CHECK(nvn::select_driver(true, image).version == (nvn::ApiVersion{55, 15}));
    CHECK(nvn::select_driver(false, image, "driver").version.has_value());
    CHECK(!nvn::select_driver(true, image, "stream").version);
    CHECK(!nvn::select_driver(true, std::as_bytes(std::span(unknown.data(), unknown.size())), "driver").version);
    CHECK(!nvn::select_driver(true, {}).version);
    CHECK(nvn::select_driver(true, image, "invalid").error != nullptr);
    std::printf("NVN redirect/selection failures: %u\n", failures);
    return failures ? 1 : 0;
}
