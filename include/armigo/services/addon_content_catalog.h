// ============================================================================
// Armigo Services - Horizon Service Implementations
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/romfs/storage.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace armigo::services {

// One installed add-on content item: its data id (the application's add-on
// content base id plus the item's index) and the RomFS store fsp-srv opens for
// that data id.
struct AddOnContentItem {
    std::uint64_t data_id = 0;
    std::shared_ptr<romfs::Storage> storage;
};

// The add-on content installed for the running application, shared by aoc:u
// (which counts and lists it) and fsp-srv (which opens its data storage).
// Items are unique and in ascending data-id order; each data id lies in
// [base_id + 1, base_id + 0xfff].
struct AddOnContentCatalog {
    std::uint64_t base_id = 0;
    std::vector<AddOnContentItem> items;

    const AddOnContentItem* find(std::uint64_t data_id) const noexcept {
        for (const auto& item : items)
            if (item.data_id == data_id) return &item;
        return nullptr;
    }
};

}  // namespace armigo::services
