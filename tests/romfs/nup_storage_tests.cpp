// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/romfs/nup_storage.h>
#include "nup_fixture.h"

using namespace armigo;
using namespace nup_fixture;

int main() {
    Temporary package;
    auto opened = package.open(fixture());
    auto archive = std::get<std::shared_ptr<loader::NupArchive>>(opened);
    auto selection = std::get<loader::NupSelection>(archive->select());
    std::string error;
    auto storage = romfs::NupStorage::open(archive, selection, error);
    ARMIGO_CHECK(storage != nullptr);
    if (!storage) return armigo::loader::test::finish("romfs.nup_storage");
    const auto root = package.path.string() + "-tree";
    std::filesystem::create_directory(root);
    for (const auto& [name, value] : {std::pair{"old", 0x55}, std::pair{"keep", 0x33}}) {
        std::ofstream out(std::filesystem::path(root) / name, std::ios::binary);
        for (unsigned i = 0; i < 4; ++i) out.put(static_cast<char>(value));
    }
    romfs::ImageLayout layout;
    const auto expected = romfs::build(root, layout, error);
    ARMIGO_CHECK(expected && storage->size() == expected->size());
    std::vector<std::byte> bytes(storage->size());
    ARMIGO_CHECK(storage->read(0, bytes) == bytes.size());
    ARMIGO_CHECK(expected && *expected == bytes);
    for (std::size_t offset = 0; offset < bytes.size(); offset += 7) {
        std::array<std::byte, 19> chunk{};
        const auto count = storage->read(offset, chunk);
        const auto wanted = std::min(chunk.size(), bytes.size() - offset);
        ARMIGO_CHECK(count == wanted);
        ARMIGO_CHECK(std::equal(chunk.begin(), chunk.begin() + wanted, bytes.begin() + offset));
    }
    std::array<std::byte, 3> out{};
    ARMIGO_CHECK(storage->read(UINT64_MAX, out) == 0);
    ARMIGO_CHECK(storage->read(storage->size(), out) == 0);
    selection.romfs["bad"] = UINT32_MAX;
    ARMIGO_CHECK(!romfs::NupStorage::open(archive, selection, error));
    std::filesystem::remove_all(root);
    return armigo::loader::test::finish("romfs.nup_storage");
}
