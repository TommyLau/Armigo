// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/loader/nup.h>
#include "sha256.h"
#include "nup_fixture.h"
#include "test_support.h"
#include <fstream>
#include <unistd.h>

using namespace armigo::loader;
namespace {
using namespace nup_fixture;
void rejects(Bytes bytes, const char* operation, bool hash = true) {
    if (hash) rehash(bytes);
    Temporary temp;
    const auto result = temp.open(bytes);
    const auto* failure = std::get_if<NupFailure>(&result);
    ARMIGO_CHECK(failure != nullptr);
    if (failure) ARMIGO_CHECK(failure->operation == operation);
}
}
int main() {
    for (bool after : {false, true}) {
        Temporary temp;
        auto opened = temp.open(fixture(after));
        ARMIGO_CHECK(std::holds_alternative<std::shared_ptr<NupArchive>>(opened));
        if (!std::holds_alternative<std::shared_ptr<NupArchive>>(opened)) continue;
        auto archive = std::get<std::shared_ptr<NupArchive>>(opened);
        auto selected = archive->select();
        ARMIGO_CHECK(std::holds_alternative<NupSelection>(selected));
        auto selection = std::get<NupSelection>(selected);
        ARMIGO_CHECK(selection.update_applied && selection.version == 0x20000);
        ARMIGO_CHECK(selection.exefs.at("main") == 3);
        ARMIGO_CHECK(selection.romfs.at("old") == 4 && selection.romfs.at("keep") == 2);
        std::array<std::byte, 8> output{};
        ARMIGO_CHECK(std::get<std::size_t>(archive->read(4, 2, output)) == 2);
        ARMIGO_CHECK(output[0] == std::byte{0x55});
        ARMIGO_CHECK(std::get<std::size_t>(archive->read(4, UINT64_MAX, output)) == 0);
        ARMIGO_CHECK(!archive->verify(4));
        auto exefs = archive->read_exefs(selection);
        ARMIGO_CHECK(std::get<ExeFsDirectoryInput>(exefs).modules[0].bytes[0] == std::byte{0x44});
        ARMIGO_CHECK(std::holds_alternative<NupFailure>(archive->read_exefs(selection, {3, 10})));
        ARMIGO_CHECK(std::holds_alternative<NupFailure>(archive->select(0x1234)));
    }
    auto bytes = fixture(); bytes[pool] ^= std::byte{1}; rejects(bytes, "nup.toc_hash", false);
    bytes = fixture(); put(bytes, toc + 8, UINT64_MAX, 8); rejects(bytes, "nup.counts");
    bytes = fixture(); put(bytes, 8, UINT64_MAX, 8); rejects(bytes, "nup.bounds", false);
    bytes = fixture(); put(bytes, file + 16, UINT64_MAX - 15, 8); rejects(bytes, "nup.extent");
    bytes = fixture(); put(bytes, file + 24, 8, 8); rejects(bytes, "nup.extent");
    bytes = fixture(); put(bytes, child, 0); rejects(bytes, "nup.tree");
    bytes = fixture(); put(bytes, child + 4, 1); rejects(bytes, "nup.tree");
    bytes = fixture(); put(bytes, child, 99); rejects(bytes, "nup.tree");
    bytes = fixture(); put(bytes, file, 39); rejects(bytes, "nup.extent");
    bytes = fixture(); bytes[pool + 14] = std::byte{'/'}; rejects(bytes, "nup.extent");
    bytes = fixture(); put(bytes, package + 0x28, UINT32_MAX); rejects(bytes, "nup.metadata");
    bytes = fixture(); put(bytes, toc_end + 4, 10); rejects(bytes, "nup.metadata");
    bytes = fixture(); put(bytes, package + 0x40 + 12, 1); rejects(bytes, "nup.tree");
    bytes = fixture(); bytes.resize(70); rejects(bytes, "nup.header", false);
    bytes = fixture(); bytes.pop_back(); rejects(bytes, "nup.bounds", false);
    {
        Temporary temp;
        bytes = fixture(); bytes[blob] ^= std::byte{1};
        auto archive = std::get<std::shared_ptr<NupArchive>>(temp.open(bytes));
        ARMIGO_CHECK(archive->verify(0)->operation == "nup.file_hash");
        bytes = fixture(); put(bytes, file + 64, 1, 1); rehash(bytes);
        archive = std::get<std::shared_ptr<NupArchive>>(temp.open(bytes));
        ARMIGO_CHECK(archive->verify(0)->operation == "nup.codec");
        bytes = fixture(); put(bytes, package + 0x40, 0x0400000000003800ull, 8); rehash(bytes);
        archive = std::get<std::shared_ptr<NupArchive>>(temp.open(bytes));
        const auto selection = std::get<NupSelection>(archive->select());
        ARMIGO_CHECK(!selection.update_applied && selection.exefs.at("main") == 0);
    }
    {
        Temporary temp;
        bytes = fixture(); put(bytes, package + 0x40 + 25, 1, 1); rehash(bytes);
        auto archive = std::get<std::shared_ptr<NupArchive>>(temp.open(bytes));
        const auto selected = std::get<NupSelection>(archive->select());
        ARMIGO_CHECK(selected.romfs.size() == 1 && !selected.romfs.contains("keep"));
        bytes = fixture(); put(bytes, file + 3 * 0x50 + 64, 1, 1); rehash(bytes);
        archive = std::get<std::shared_ptr<NupArchive>>(temp.open(bytes));
        ARMIGO_CHECK(std::get<NupFailure>(archive->read_exefs(
            std::get<NupSelection>(archive->select()))).operation == "nup.codec");
        bytes = fixture(); bytes[blob + 3 * 16] ^= std::byte{1};
        archive = std::get<std::shared_ptr<NupArchive>>(temp.open(bytes));
        ARMIGO_CHECK(std::get<NupFailure>(archive->read_exefs(
            std::get<NupSelection>(archive->select()))).operation == "nup.file_hash");
    }
    return armigo::loader::test::finish("NUP reader");
}
