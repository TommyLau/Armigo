// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "container_nca.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::loader {

struct ContainerEntry {
    std::string name;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

struct ContainerScan {
    std::vector<ContainerEntry> entries;
    std::vector<container_detail::ContainerTicketRef> tickets;
};

// Records every entry in an NSP, standalone NCA, or the secure, normal and
// update XCI partitions, retaining names so callers can distinguish NCAs from tickets.
[[nodiscard]] bool scan_container_entries(std::FILE* file, ContainerScan& scan,
                                          ContainerFailure& failure);

[[nodiscard]] bool sanitize_romfs_name(std::string_view name) noexcept;

} // namespace armigo::loader
