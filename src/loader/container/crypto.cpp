// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Container crypto primitives. The AES block, CTR and XTS arithmetic is
// Armigo's own, built on CommonCrypto's raw AES-ECB primitive; XTS follows
// IEEE 1619 with little-endian data-unit tweaks as the NCA format uses.
// PSS verification uses Security.framework's message-PSS-SHA256 algorithm,
// whose fixed 32-byte salt matches the container signature parameters.

#include "crypto.h"

#include <CommonCrypto/CommonCrypto.h>
#include <CommonCrypto/CommonDigest.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace armigo::loader::container_detail {
namespace {

// Multiplies the tweak, a little-endian 128-bit value, by alpha in
// GF(2^128) (IEEE 1619): shift left by one bit and fold a carry out of the
// top bit back in with the polynomial 0x87.
void multiply_tweak(std::array<std::uint8_t, 16>& tweak) noexcept {
    const std::uint8_t carry = tweak[15] >> 7;
    for (std::size_t i = 15; i > 0; --i) {
        tweak[i] = static_cast<std::uint8_t>((tweak[i] << 1) | (tweak[i - 1] >> 7));
    }
    tweak[0] = static_cast<std::uint8_t>(tweak[0] << 1);
    if (carry != 0) {
        tweak[0] ^= 0x87;
    }
}

// Appends a DER length using the short or long form.
void append_der_length(std::vector<std::uint8_t>& out, std::size_t length) {
    if (length < 0x80) {
        out.push_back(static_cast<std::uint8_t>(length));
        return;
    }
    std::uint8_t bytes[4];
    std::size_t count = 0;
    std::size_t value = length;
    while (value != 0) {
        bytes[count++] = static_cast<std::uint8_t>(value & 0xFF);
        value >>= 8;
    }
    out.push_back(static_cast<std::uint8_t>(0x80 | count));
    for (std::size_t i = count; i > 0; --i) {
        out.push_back(bytes[i - 1]);
    }
}

} // namespace

bool aes_ecb_encrypt_block(std::span<const std::byte> key, std::span<const std::byte> input,
                           std::span<std::byte> output) noexcept {
    if (key.size() != 16 || input.size() != 16 || output.size() != 16) {
        return false;
    }
    std::size_t moved = 0;
    const CCCryptorStatus status =
        CCCrypt(kCCEncrypt, kCCAlgorithmAES, kCCOptionECBMode, key.data(), kCCKeySizeAES128,
                nullptr, input.data(), input.size(), output.data(), output.size(), &moved);
    return status == kCCSuccess && moved == output.size();
}

bool aes_ecb_decrypt_block(std::span<const std::byte> key, std::span<const std::byte> input,
                           std::span<std::byte> output) noexcept {
    if (key.size() != 16 || input.size() != 16 || output.size() != 16) {
        return false;
    }
    std::size_t moved = 0;
    const CCCryptorStatus status =
        CCCrypt(kCCDecrypt, kCCAlgorithmAES, kCCOptionECBMode, key.data(), kCCKeySizeAES128,
                nullptr, input.data(), input.size(), output.data(), output.size(), &moved);
    return status == kCCSuccess && moved == output.size();
}

bool aes_ctr_crypt(std::span<const std::byte> key, std::array<std::uint8_t, 16> counter,
                   std::span<std::byte> data) noexcept {
    if (key.size() != 16) {
        return false;
    }
    if (data.empty()) {
        return true;
    }
    CCCryptorRef cryptor = nullptr;
    const CCCryptorStatus created = CCCryptorCreateWithMode(
        kCCDecrypt, kCCModeCTR, kCCAlgorithmAES, ccNoPadding, counter.data(), key.data(),
        kCCKeySizeAES128, nullptr, 0, 0, 0, &cryptor);
    if (created != kCCSuccess) {
        return false;
    }
    std::size_t moved = 0;
    const CCCryptorStatus updated = CCCryptorUpdate(cryptor, data.data(), data.size(),
                                                    data.data(), data.size(), &moved);
    CCCryptorRelease(cryptor);
    return updated == kCCSuccess && moved == data.size();
}

bool aes_xts_decrypt(std::span<const std::byte> key, std::span<std::byte> data,
                     std::uint64_t first_sector, std::size_t sector_size) noexcept {
    if (key.size() != 32 || sector_size == 0 || sector_size % 16 != 0 ||
        data.size() % sector_size != 0) {
        return false; // Whole data units only; ciphertext stealing is not
                      // part of the NCA formats used.
    }
    const std::span<const std::byte, 16> data_key = key.first<16>();
    const std::span<const std::byte, 16> tweak_key = key.last<16>();
    std::array<std::byte, 16> encrypted_tweak{};
    std::array<std::byte, 16> block_input{};
    std::array<std::byte, 16> block_output{};
    for (std::size_t offset = 0; offset < data.size(); offset += sector_size) {
        // NCA XTS places the data-unit number big-endian at the end of the
        // tweak (unlike IEEE 1619's little-endian layout); the tweak is then
        // encrypted with the second half of the key.
        std::array<std::byte, 16> tweak{};
        std::uint64_t sector = first_sector + offset / sector_size;
        for (std::size_t i = 0; i < 8; ++i) {
            tweak[15 - i] = std::byte(static_cast<std::uint8_t>(sector & 0xFF));
            sector >>= 8;
        }
        if (!aes_ecb_encrypt_block(tweak_key, tweak, encrypted_tweak)) {
            return false;
        }
        const std::size_t unit_length = std::min(sector_size, data.size() - offset);
        for (std::size_t within = 0; within < unit_length; within += 16) {
            for (std::size_t i = 0; i < 16; ++i) {
                block_input[i] = data[offset + within + i] ^ encrypted_tweak[i];
            }
            std::size_t moved = 0;
            const CCCryptorStatus status = CCCrypt(
                kCCDecrypt, kCCAlgorithmAES, kCCOptionECBMode, data_key.data(), kCCKeySizeAES128,
                nullptr, block_input.data(), block_input.size(), block_output.data(),
                block_output.size(), &moved);
            if (status != kCCSuccess || moved != block_output.size()) {
                return false;
            }
            for (std::size_t i = 0; i < 16; ++i) {
                data[offset + within + i] = block_output[i] ^ encrypted_tweak[i];
            }
            std::array<std::uint8_t, 16> next{};
            for (std::size_t i = 0; i < 16; ++i) {
                next[i] = static_cast<std::uint8_t>(encrypted_tweak[i]);
            }
            multiply_tweak(next);
            for (std::size_t i = 0; i < 16; ++i) {
                encrypted_tweak[i] = std::byte(next[i]);
            }
        }
    }
    return true;
}

bool rsa2048_pss_sha256_verify(std::span<const std::uint8_t, 256> modulus,
                               std::span<const std::byte> message,
                               std::span<const std::uint8_t, 256> signature) noexcept {
    // Encode the public key as a PKCS#1 RSAPublicKey DER structure, the
    // input format of SecKeyCreateWithData for RSA keys.
    std::vector<std::uint8_t> der;
    der.push_back(0x30); // SEQUENCE
    const std::size_t content_position = der.size();
    der.push_back(0x00); // length placeholder, long-form patched below
    der.push_back(0x00);
    der.push_back(0x00);
    der.push_back(0x02); // INTEGER modulus
    append_der_length(der, modulus.size() + 1); // leading zero keeps the sign bit clear
    der.push_back(0x00);
    der.insert(der.end(), modulus.begin(), modulus.end());
    der.push_back(0x02); // INTEGER publicExponent
    der.push_back(0x03);
    der.push_back(0x01);
    der.push_back(0x00);
    der.push_back(0x01);
    {
        // Header: the tag byte plus the three long-form length bytes.
        const std::size_t content_length = der.size() - content_position - 3;
        der[content_position] = 0x82;
        der[content_position + 1] = static_cast<std::uint8_t>(content_length >> 8);
        der[content_position + 2] = static_cast<std::uint8_t>(content_length);
    }

    const CFDataRef key_data =
        CFDataCreate(kCFAllocatorDefault, der.data(), static_cast<CFIndex>(der.size()));
    if (!key_data) {
        return false;
    }
    const void* attribute_keys[] = {kSecAttrKeyType, kSecAttrKeyClass};
    const void* attribute_values[] = {kSecAttrKeyTypeRSA, kSecAttrKeyClassPublic};
    const CFDictionaryRef attributes = CFDictionaryCreate(
        kCFAllocatorDefault, attribute_keys, attribute_values, 2, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);
    if (!attributes) {
        CFRelease(key_data);
        return false;
    }
    CFErrorRef error = nullptr;
    SecKeyRef key = SecKeyCreateWithData(key_data, attributes, &error);
    CFRelease(key_data);
    CFRelease(attributes);
    if (!key) {
        return false;
    }
    bool verified = false;
    {
        const CFDataRef message_data = CFDataCreateWithBytesNoCopy(
            kCFAllocatorDefault, reinterpret_cast<const UInt8*>(message.data()),
            static_cast<CFIndex>(message.size()), kCFAllocatorNull);
        const CFDataRef signature_data = CFDataCreateWithBytesNoCopy(
            kCFAllocatorDefault, reinterpret_cast<const UInt8*>(signature.data()),
            static_cast<CFIndex>(signature.size()), kCFAllocatorNull);
        if (message_data && signature_data) {
            verified = SecKeyVerifySignature(key, kSecKeyAlgorithmRSASignatureMessagePSSSHA256,
                                             message_data, signature_data, &error);
        }
        if (message_data) CFRelease(message_data);
        if (signature_data) CFRelease(signature_data);
    }
    CFRelease(key);
    return verified;
}

bool sha256_digest(std::span<const std::byte> data, std::span<std::byte> out) noexcept {
    if (out.size() != CC_SHA256_DIGEST_LENGTH ||
        data.size() > static_cast<std::size_t>(std::numeric_limits<CC_LONG>::max())) {
        return false;
    }
    return CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()),
                     reinterpret_cast<std::uint8_t*>(out.data())) != nullptr;
}

} // namespace armigo::loader::container_detail
