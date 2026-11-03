// ============================================================================
// Armigo Horizon - Minimal Kernel Compatibility
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/handle_table.h>

#include <span>
#include <stdexcept>

namespace armigo::horizon {

// Immutable, independently owned contents for a read-only shared-memory object.
// Handle copies pin this same object. This type does not create a host mapping
// or implement MapSharedMemory, guest permissions, addresses or unmapping.
class ReadOnlySharedMemory final : public KernelObject {
public:
    explicit ReadOnlySharedMemory(std::span<const std::byte> payload)
        : bytes_(copy_payload(payload)) {}

    std::span<const std::byte> bytes() const noexcept { return bytes_; }

private:
    static std::vector<std::byte> copy_payload(std::span<const std::byte> payload) {
        if (payload.empty() || payload.size() % 0x1000 != 0)
            throw std::invalid_argument("Shared-memory payload must contain complete 4096-byte pages");
        return {payload.begin(), payload.end()};
    }

    const std::vector<std::byte> bytes_;
};

}  // namespace armigo::horizon
