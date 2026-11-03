// ============================================================================
// Armigo RomFS - On-demand storage interface
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace armigo::romfs {

// A read-only byte store the file-system service can read at arbitrary offsets.
// Implementations serve the bytes on demand; callers never receive a pointer to
// the whole store.
class Storage {
public:
    virtual ~Storage() = default;
    virtual std::uint64_t size() const noexcept = 0;
    // Reads up to destination.size() bytes at `offset`. Returns the number of
    // bytes read (short only at the end of the store) or nullopt on an I/O
    // error. A read at or beyond size() returns 0.
    virtual std::optional<std::size_t> read(std::uint64_t offset,
                                            std::span<std::byte> destination) = 0;
};

}  // namespace armigo::romfs
