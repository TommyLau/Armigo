// ============================================================================
// Armigo RomFS - Directory-backed virtual storage
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/loader/nup.h>
#include <armigo/romfs/romfs_image.h>
#include <armigo/romfs/storage.h>

namespace armigo::romfs {

// A virtual RomFS image whose selected files remain raw extents in one NUP.
class NupStorage final : public Storage {
public:
    static std::shared_ptr<NupStorage> open(std::shared_ptr<loader::NupArchive> archive,
                                            const loader::NupSelection& selection,
                                            std::string& error);
    std::uint64_t size() const noexcept override { return image_.size; }
    std::optional<std::size_t> read(std::uint64_t offset, std::span<std::byte> output) override;
private:
    std::shared_ptr<loader::NupArchive> archive_;
    VirtualImage image_;
    std::vector<std::uint32_t> members_;
};

} // namespace armigo::romfs
