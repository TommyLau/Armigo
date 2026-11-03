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
#include <armigo/execution/embedded_patch.h>
#include <limits>

#include <array>
#include <cstring>

namespace armigo::nvn {
DriverSelection select_driver(bool title_enabled, std::span<const std::byte> image,
                              std::string_view override_value) noexcept {
    if (!override_value.empty() && override_value != "driver" && override_value != "stream")
        return {{}, "ARMIGO_NVN must be driver or stream."};
    if (override_value == "stream" || (!title_enabled && override_value != "driver")) return {};
    const auto package = driver_package(image);
    const auto version = package ? known_api_version(*package) : std::nullopt;
    return {version, nullptr};
}

RedirectResult redirect_bootstrap_loader(const runtime::ModulePreparationView& view,
    const execution::PreparedImage& prepared, std::uint64_t text_offset,
    std::uint64_t text_size, std::optional<std::size_t> embedded_stub_offset) noexcept {
    const linker::Symbol* exported = nullptr;
    for (const auto& symbol : view.metadata.symbols) {
        if (symbol.name != "nvnBootstrapLoader" || !symbol.defined()) continue;
        if (exported) return {"NVN bootstrap export is ambiguous."};
        exported = &symbol;
    }
    if (!exported) return {"NVN bootstrap export is missing."};
    const auto at = exported->value;
    if ((exported->info & 15u) != 2 || (exported->info >> 4) == 0 ||
        (exported->visibility & 3u) == 1 || (exported->visibility & 3u) == 2 ||
        exported->section >= 0xff00 || at % 4 ||
        exported->size < 16 ||
        at < text_offset || at - text_offset > text_size ||
        text_size - (at - text_offset) < 16 ||
        at > view.image.size() || view.image.size() - at < 16 ||
        prepared.original_image_size != view.image.size())
        return {"NVN bootstrap export is not a complete executable entry."};
    for (const auto& symbol : view.metadata.symbols) {
        if (&symbol != exported && symbol.defined() && symbol.value > at && symbol.value - at < 16)
            return {"Another symbol overlaps the NVN bootstrap entry."};
    }
    const auto overlaps = [at](std::uint64_t offset, std::uint64_t size) {
        return offset <= at ? at - offset < size : offset - at < 16;
    };
    for (const auto& patch : prepared.replacements)
        if (overlaps(patch.image_offset, 4)) return {"Preparation rewrites the NVN bootstrap entry."};
    for (const auto& fixup : prepared.fixups) {
        std::uint64_t width = 0;
        switch (fixup.kind) {
        case execution::PreparedFixupKind::thread_slot_load: width = 4; break;
        case execution::PreparedFixupKind::gate_dispatcher_address: width = 8; break;
        case execution::PreparedFixupKind::module_address_mov_wide: width = 16; break;
        case execution::PreparedFixupKind::branch26_module_target: width = 4; break;
        default: return {"Unknown preparation fixup cannot admit an NVN redirect."};
        }
        if (overlaps(fixup.image_offset, width)) return {"Preparation fixes up the NVN bootstrap entry."};
    }
    for (const auto& relocation : view.metadata.relocations)
        if (overlaps(relocation.target, 8)) return {"A relocation overlaps the NVN bootstrap entry."};
    for (const auto target : view.metadata.packed_relative_targets)
        if (overlaps(target, 8)) return {"A packed relocation overlaps the NVN bootstrap entry."};
    for (std::uint64_t offset = at; offset < at + 16; offset += 4) {
        const execution::PreparedWordPatch* covered = nullptr;
        for (const auto& word : prepared.coverage) {
            if (word.image_offset != offset) continue;
            if (covered) return {"NVN bootstrap preparation coverage is ambiguous."};
            covered = &word;
        }
        std::uint32_t actual = 0;
        std::memcpy(&actual, view.image.data() + offset, 4);
        if (!covered || covered->expected_word != actual || covered->replacement_word != actual)
            return {"NVN bootstrap entry lacks unchanged prepared coverage."};
    }
    if (embedded_stub_offset) {
        constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
        const auto stub = *embedded_stub_offset;
        const auto key_offset = view.appended_data_offset + execution::embedded_nvn_bootstrap_offset;
        if (view.appended_data.size() < execution::embedded_key_page_size ||
            view.appended_data_offset % execution::embedded_key_page_size ||
            view.appended_data_offset < view.image.size() ||
            view.appended_data_offset > view.appended_code_offset ||
            view.appended_data.size() > view.appended_code_offset - view.appended_data_offset ||
            stub % 4 || stub > view.appended_code.size() || view.appended_code.size() - stub < 12 ||
            view.appended_code_offset > maximum - view.base_address ||
            view.appended_code.size() > maximum - view.base_address - view.appended_code_offset)
            return {"Embedded NVN redirect has invalid appendix or key-page bounds."};
        const auto trampoline = view.base_address + view.appended_code_offset + stub;
        const auto key = view.base_address + key_offset;
        const auto enter = aarch64::encode_b(view.base_address + at, trampoline);
        const auto page = aarch64::encode_adrp(16, trampoline, key);
        if (!enter.ok() || !page.ok()) return {"Embedded NVN redirect cannot reach its target."};
        const std::array<std::uint32_t, 3> code{page.word,
            0xf9400210u | (static_cast<std::uint32_t>((key & 4095) / 8) << 10), 0xd61f0200u};
        const std::array<std::uint32_t, 4> branch{enter.word, 0xd503201f, 0xd503201f, 0xd503201f};
        const auto destination = reinterpret_cast<std::uintptr_t>(&armigo_nvn_bootstrap_loader);
        std::memcpy(view.appended_code.data() + stub, code.data(), sizeof(code));
        std::memcpy(view.appended_data.data() + execution::embedded_nvn_bootstrap_offset, &destination, 8);
        std::memcpy(view.image.data() + at, branch.data(), sizeof(branch));
        return {nullptr, at};
    }
    const std::array<std::uint32_t, 2> branch{0x58000050u, 0xd61f0200u};
    const auto destination = reinterpret_cast<std::uintptr_t>(&armigo_nvn_bootstrap_loader);
    static_assert(sizeof(destination) == 8);
    std::memcpy(view.image.data() + at, branch.data(), 8);
    std::memcpy(view.image.data() + at + 8, &destination, 8);
    return {nullptr, at};
}
} // namespace armigo::nvn
