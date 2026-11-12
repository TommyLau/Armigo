// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/3ds/android_memory.h"

#include <cerrno>
#include <cstdio>
#include <sys/mman.h>

int main() {
    using namespace armigo::n3ds;
    GuestMemory memory;
    constexpr std::uint32_t base = 0x0b000000;
    constexpr std::uint32_t size = 8192;
    if (!memory.Initialize(4 * 4096, MapAndroidGuestMemory, nullptr)) {
        return 1;
    }
    std::uint32_t output{};
    if (memory.Commit(3, base, 0, size, 3, output) || output != base ||
        memory.CommittedBytes() != size) {
        std::printf("FAIL: Android exact mapping, host error %d\n", memory.LastHostError());
        return 1;
    }
    auto* bytes = reinterpret_cast<volatile unsigned char*>(static_cast<std::uintptr_t>(output));
    for (std::uint32_t i = 0; i < size; ++i) {
        if (bytes[i] != 0) {
            std::printf("FAIL: anonymous mapping not zero-filled\n");
            return 1;
        }
    }
    bytes[0] = 0x12;
    bytes[size - 1] = 0x34;
    if (bytes[0] != 0x12 || bytes[size - 1] != 0x34) {
        return 1;
    }
    GuestMemory other;
    if (!other.Initialize(size, MapAndroidGuestMemory, nullptr)) {
        return 1;
    }
    output = 0x5678;
    if (other.Commit(3, base, 0, size, 3, output) != 0xc860180a ||
        other.LastHostError() != EEXIST || output != 0x5678 ||
        other.MappingCount() != 0 || other.CommittedBytes() != 0 ||
        bytes[0] != 0x12 || bytes[size - 1] != 0x34) {
        std::printf("FAIL: collision must preserve existing memory and ledger\n");
        return 1;
    }
    if (munmap(reinterpret_cast<void*>(static_cast<std::uintptr_t>(base)), size)) {
        return 1;
    }
    std::printf("PASS: native Android exact RW mapping, zero fill and collision rollback\n");
    return 0;
}
