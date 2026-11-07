// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/nup.h>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2 || std::string_view(argv[1]) == "--help") {
        std::puts("Usage: armigo-nup-inspect ARCHIVE.nup\nVerifies the table and selected ExeFS; RomFS payloads remain unread.");
        return argc == 2 ? 0 : 2;
    }
    auto opened = armigo::loader::NupArchive::open(argv[1]);
    if (auto failure = std::get_if<armigo::loader::NupFailure>(&opened)) {
        std::fprintf(stderr, "%s: %s\n", failure->operation.c_str(), failure->detail.c_str());
        return 1;
    }
    auto archive = std::get<std::shared_ptr<armigo::loader::NupArchive>>(opened);
    auto selected = archive->select();
    if (auto failure = std::get_if<armigo::loader::NupFailure>(&selected)) {
        std::fprintf(stderr, "%s: %s\n", failure->operation.c_str(), failure->detail.c_str());
        return 1;
    }
    const auto& selection = std::get<armigo::loader::NupSelection>(selected);
    auto input = archive->read_exefs(selection);
    if (auto failure = std::get_if<armigo::loader::NupFailure>(&input)) {
        std::fprintf(stderr, "%s: %s\n", failure->operation.c_str(), failure->detail.c_str());
        return 1;
    }
    std::printf("title=%016llx version=%u update=%u packages=%zu files=%zu romfs=%zu\n",
                static_cast<unsigned long long>(selection.title_id), selection.version,
                selection.update_applied, archive->packages().size(), archive->files().size(),
                selection.romfs.size());
    const auto& exefs = std::get<armigo::loader::ExeFsDirectoryInput>(input);
    for (const auto& module : exefs.modules)
        std::printf("verified module=%s bytes=%zu\n", module.name.c_str(), module.bytes.size());
    std::printf("npdm_bytes=%zu\n", exefs.process_metadata ? exefs.process_metadata->size() : 0);
    return 0;
}
