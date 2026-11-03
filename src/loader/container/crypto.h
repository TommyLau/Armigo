// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Container crypto primitives over Apple system frameworks: AES-128
// ECB/CTR/XTS through CommonCrypto and RSA-2048-PSS-SHA256 verification
// through Security. Internal to the container reader; not a public header.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace armigo::loader::container_detail {

// Encrypts one 16-byte block with AES-128 ECB; every span must be 16 bytes.
[[nodiscard]] bool aes_ecb_encrypt_block(std::span<const std::byte> key,
                                         std::span<const std::byte> input,
                                         std::span<std::byte> output) noexcept;

// Decrypts one 16-byte block with AES-128 ECB; every span must be 16 bytes.
[[nodiscard]] bool aes_ecb_decrypt_block(std::span<const std::byte> key,
                                         std::span<const std::byte> input,
                                         std::span<std::byte> output) noexcept;

// AES-128 CTR keystream application over one contiguous span. `counter` is
// the full 16-byte big-endian counter of the first block and advances
// across the span; CTR encryption and decryption are the same operation.
// The key must be 16 bytes.
[[nodiscard]] bool aes_ctr_crypt(std::span<const std::byte> key,
                                 std::array<std::uint8_t, 16> counter,
                                 std::span<std::byte> data) noexcept;

// AES-128 XTS decryption of whole sectors: `data.size()` must be a
// multiple of `sector_size`, and the key must be 32 bytes. `first_sector`
// is the data-unit number of data[0..sector_size); each sector's tweak is
// the little-endian encoding of its data-unit number.
[[nodiscard]] bool aes_xts_decrypt(std::span<const std::byte> key,
                                   std::span<std::byte> data,
                                   std::uint64_t first_sector,
                                   std::size_t sector_size) noexcept;

// Computes the SHA-256 digest of `data` into `out`, which must be 32 bytes.
[[nodiscard]] bool sha256_digest(std::span<const std::byte> data,
                                 std::span<std::byte> out) noexcept;

// RSA-2048 PSS signature verification (SHA-256, 32-byte salt, public
// exponent 65537). Returns false for every malformed input; never throws.
[[nodiscard]] bool rsa2048_pss_sha256_verify(
    std::span<const std::uint8_t, 256> modulus, std::span<const std::byte> message,
    std::span<const std::uint8_t, 256> signature) noexcept;

} // namespace armigo::loader::container_detail
