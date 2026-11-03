// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// SHA-256 through Apple's CommonCrypto, available on macOS, iOS, and iPadOS.

#include "sha256.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cstdint>

namespace armigo::loader::detail {

std::optional<Sha256Digest> sha256(std::span<const std::byte> data) noexcept {
    static_assert(CC_SHA256_DIGEST_LENGTH == std::tuple_size_v<Sha256Digest>);
    CC_SHA256_CTX context;
    if (CC_SHA256_Init(&context) != 1) {
        return std::nullopt;
    }
    // CC_LONG is 32 bits wide; feed larger inputs in bounded chunks.
    constexpr std::size_t kChunk = std::size_t{1} << 30;
    std::size_t position = 0;
    while (position < data.size()) {
        const std::size_t length = std::min(kChunk, data.size() - position);
        if (CC_SHA256_Update(&context, data.data() + position, static_cast<CC_LONG>(length)) != 1) {
            return std::nullopt;
        }
        position += length;
    }
    Sha256Digest digest{};
    if (CC_SHA256_Final(reinterpret_cast<unsigned char*>(digest.data()), &context) != 1) {
        return std::nullopt;
    }
    return digest;
}

} // namespace armigo::loader::detail
