// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Synthetic PFS0/HFS0/NCA/RomFS builders for the container reader tests.
// Every key, container and content byte here is generated at test runtime
// from fixed synthetic patterns; no game data, captured bytes or real key
// material appear in this repository. Signature fixtures use a test RSA key
// pair generated with Security.framework, whose public modulus is injected
// as the trust anchor under test; production anchors are never consulted.

#pragma once

#include <armigo/loader/container.h>

#include <CommonCrypto/CommonCrypto.h>
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace container_test {

namespace loader = armigo::loader;

// ---------------------------------------------------------------------------
// Little-endian scalar writers over byte images.
// ---------------------------------------------------------------------------

inline void put_u16(std::vector<std::byte>& image, std::size_t offset, std::uint16_t value) {
    for (std::size_t i = 0; i < 2; ++i) {
        image[offset + i] = std::byte(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

inline void put_u32(std::vector<std::byte>& image, std::size_t offset, std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
        image[offset + i] = std::byte(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

inline void put_u64(std::vector<std::byte>& image, std::size_t offset, std::uint64_t value) {
    for (std::size_t i = 0; i < 8; ++i) {
        image[offset + i] = std::byte(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

// ---------------------------------------------------------------------------
// AES helpers for the fixture side (encryption); the library under test owns
// the matching decryption, so a roundtrip cross-checks both directions.
// ---------------------------------------------------------------------------

inline bool fixture_aes_ecb_encrypt(const std::byte* key, const std::byte* input,
                                    std::byte* output) {
    std::size_t moved = 0;
    return CCCrypt(kCCEncrypt, kCCAlgorithmAES, kCCOptionECBMode, key, kCCKeySizeAES128, nullptr,
                   input, 16, output, 16, &moved) == kCCSuccess &&
           moved == 16;
}

inline bool fixture_aes_ctr_apply(const std::byte* key, const std::uint8_t counter[16],
                                  std::vector<std::byte>& data) {
    if (data.empty()) {
        return true;
    }
    CCCryptorRef cryptor = nullptr;
    if (CCCryptorCreateWithMode(kCCEncrypt, kCCModeCTR, kCCAlgorithmAES, ccNoPadding, counter,
                                key, kCCKeySizeAES128, nullptr, 0, 0, 0,
                                &cryptor) != kCCSuccess) {
        return false;
    }
    std::size_t moved = 0;
    const bool ok = CCCryptorUpdate(cryptor, data.data(), data.size(), data.data(), data.size(),
                                    &moved) == kCCSuccess &&
                    moved == data.size();
    CCCryptorRelease(cryptor);
    return ok;
}

// XTS-AES-128 encryption of whole 0x200-byte sectors, written independently
// of the library's decryptor (same IEEE 1619 tweak arithmetic).
inline bool fixture_xts_encrypt(const std::byte* key32, std::vector<std::byte>& data,
                                std::uint64_t first_sector, std::size_t sector_size) {
    for (std::size_t offset = 0; offset < data.size(); offset += sector_size) {
        std::byte tweak[16] = {};
        std::uint64_t sector = first_sector + offset / sector_size;
        // NCA layout: the data-unit number is big-endian in bytes 8..15.
        for (std::size_t i = 0; i < 8; ++i) {
            tweak[15 - i] = std::byte(static_cast<std::uint8_t>(sector & 0xFF));
            sector >>= 8;
        }
        std::byte encrypted_tweak[16];
        if (!fixture_aes_ecb_encrypt(key32 + 16, tweak, encrypted_tweak)) {
            return false;
        }
        const std::size_t length = std::min(sector_size, data.size() - offset);
        for (std::size_t within = 0; within < length; within += 16) {
            std::byte block[16];
            std::byte cipher[16];
            for (std::size_t i = 0; i < 16; ++i) {
                block[i] = data[offset + within + i] ^ encrypted_tweak[i];
            }
            if (!fixture_aes_ecb_encrypt(key32, block, cipher)) {
                return false;
            }
            for (std::size_t i = 0; i < 16; ++i) {
                data[offset + within + i] = cipher[i] ^ encrypted_tweak[i];
            }
            // Multiply by alpha: the tweak is a little-endian 128-bit value.
            std::byte next[16];
            const std::uint8_t carry = static_cast<std::uint8_t>(encrypted_tweak[15]) >> 7;
            for (int i = 15; i > 0; --i) {
                next[i] = std::byte(static_cast<std::uint8_t>(
                    (static_cast<std::uint8_t>(encrypted_tweak[i]) << 1) |
                    (static_cast<std::uint8_t>(encrypted_tweak[i - 1]) >> 7)));
            }
            next[0] = std::byte(static_cast<std::uint8_t>(static_cast<std::uint8_t>(encrypted_tweak[0]) << 1));
            if (carry != 0) {
                next[0] = std::byte(static_cast<std::uint8_t>(next[0]) ^ 0x87);
            }
            std::memcpy(encrypted_tweak, next, 16);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// SHA-256 for building the fixtures' hash chains; the library under test
// computes the same digests through its own CommonCrypto wrapper.
// ---------------------------------------------------------------------------

inline std::array<std::byte, 32> fixture_sha256(std::span<const std::byte> data) {
    std::array<std::byte, 32> digest{};
    CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()),
              reinterpret_cast<std::uint8_t*>(digest.data()));
    return digest;
}

// ---------------------------------------------------------------------------
// A test-only RSA-2048 key pair. The public modulus is injected as a trust
// anchor; the private key signs fixture headers and ACIDs.
// ---------------------------------------------------------------------------

class TestRsaKey {
public:
    TestRsaKey() = default;
    TestRsaKey(const TestRsaKey&) = delete;
    TestRsaKey& operator=(const TestRsaKey&) = delete;
    TestRsaKey(TestRsaKey&& other) noexcept { *this = std::move(other); }
    TestRsaKey& operator=(TestRsaKey&& other) noexcept {
        std::swap(private_key_, other.private_key_);
        std::swap(public_key_, other.public_key_);
        modulus_ = other.modulus_;
        return *this;
    }
    ~TestRsaKey() {
        if (private_key_) CFRelease(private_key_);
        if (public_key_) CFRelease(public_key_);
    }

    [[nodiscard]] static std::optional<TestRsaKey> generate() {
        CFDictionaryRef attributes = nullptr;
        const void* keys[] = {kSecAttrKeyType, kSecAttrKeySizeInBits};
        const void* values[] = {kSecAttrKeyTypeRSA, CFSTR("2048")};
        attributes = CFDictionaryCreate(kCFAllocatorDefault, keys, values, 2,
                                        &kCFTypeDictionaryKeyCallBacks,
                                        &kCFTypeDictionaryValueCallBacks);
        if (!attributes) {
            return std::nullopt;
        }
        CFErrorRef error = nullptr;
        SecKeyRef private_key = SecKeyCreateRandomKey(attributes, &error);
        CFRelease(attributes);
        if (!private_key) {
            return std::nullopt;
        }
        SecKeyRef public_key = SecKeyCopyPublicKey(private_key);
        if (!public_key) {
            CFRelease(private_key);
            return std::nullopt;
        }
        TestRsaKey key;
        key.private_key_ = private_key;
        key.public_key_ = public_key;
        key.modulus_ = public_modulus(public_key);
        return key;
    }

    [[nodiscard]] const std::array<std::uint8_t, 256>& modulus() const { return modulus_; }

    // RSA-PSS (SHA-256, 32-byte salt) over `message`, matching the
    // parameters the container verifier expects.
    [[nodiscard]] bool sign(std::span<const std::byte> message,
                            std::array<std::uint8_t, 256>& signature) const {
        const CFDataRef message_data = CFDataCreateWithBytesNoCopy(
            kCFAllocatorDefault, reinterpret_cast<const UInt8*>(message.data()),
            static_cast<CFIndex>(message.size()), kCFAllocatorNull);
        if (!message_data) {
            return false;
        }
        CFErrorRef error = nullptr;
        const CFDataRef signature_data = SecKeyCreateSignature(
            private_key_, kSecKeyAlgorithmRSASignatureMessagePSSSHA256, message_data, &error);
        CFRelease(message_data);
        if (!signature_data || CFDataGetLength(signature_data) != 256) {
            if (signature_data) CFRelease(signature_data);
            return false;
        }
        std::memcpy(signature.data(), CFDataGetBytePtr(signature_data), 256);
        CFRelease(signature_data);
        return true;
    }

private:
    // The modulus inside the PKCS#1 DER external representation.
    static std::array<std::uint8_t, 256> public_modulus(SecKeyRef public_key) {
        std::array<std::uint8_t, 256> modulus{};
        CFErrorRef error = nullptr;
        const CFDataRef data = SecKeyCopyExternalRepresentation(public_key, &error);
        if (!data) {
            return modulus;
        }
        const std::uint8_t* der = CFDataGetBytePtr(data);
        const CFIndex length = CFDataGetLength(data);
        // SEQUENCE { INTEGER modulus, INTEGER exponent }; locate the modulus.
        CFIndex position = 0;
        if (position < length && der[position] == 0x30) {
            ++position;
            if (position < length && (der[position] & 0x80) != 0) {
                position += der[position] & 0x7F;
                ++position;
            } else {
                ++position;
            }
            if (position < length && der[position] == 0x02) {
                ++position;
                if (position < length && (der[position] & 0x80) != 0) {
                    const std::uint8_t length_bytes = der[position] & 0x7F;
                    CFIndex integer_length = 0;
                    for (std::uint8_t i = 0; i < length_bytes; ++i) {
                        integer_length = (integer_length << 8) | der[position + 1 + i];
                    }
                    position += 1 + length_bytes;
                    if (integer_length == 257) {
                        ++position; // leading zero keeps the sign positive
                        --integer_length;
                    }
                    if (integer_length == 256 && position + 256 <= length) {
                        std::memcpy(modulus.data(), der + position, 256);
                    }
                }
            }
        }
        CFRelease(data);
        return modulus;
    }

    SecKeyRef private_key_ = nullptr;
    SecKeyRef public_key_ = nullptr;
    std::array<std::uint8_t, 256> modulus_{};
};

// ---------------------------------------------------------------------------
// Container builders.
// ---------------------------------------------------------------------------

// Builds a partition image from named entries: PFS0 with 0x18-byte
// entries, or HFS0 (gamecard partitions) with 0x40-byte entries.
inline std::vector<std::byte> build_partition(
    std::uint32_t magic, std::size_t entry_size,
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& entries) {
    std::string string_table;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> sizes;
    std::vector<std::uint32_t> name_offsets;
    std::uint64_t cursor = 0;
    for (const auto& [name, content] : entries) {
        offsets.push_back(cursor);
        sizes.push_back(content.size());
        name_offsets.push_back(static_cast<std::uint32_t>(string_table.size()));
        string_table += name;
        string_table.push_back('\0');
        cursor += content.size();
    }
    std::vector<std::byte> image(0x10 + entry_size * entries.size() + string_table.size(),
                                 std::byte{0});
    put_u32(image, 0, magic);
    put_u32(image, 4, static_cast<std::uint32_t>(entries.size()));
    put_u32(image, 8, static_cast<std::uint32_t>(string_table.size()));
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const std::size_t entry = 0x10 + entry_size * i;
        put_u64(image, entry, offsets[i]);
        put_u64(image, entry + 8, sizes[i]);
        put_u32(image, entry + 0x10, name_offsets[i]);
        std::memcpy(image.data() + 0x10 + entry_size * entries.size() + name_offsets[i],
                    string_table.data() + name_offsets[i], string_table.size() - name_offsets[i]);
    }
    for (const auto& [name, content] : entries) {
        image.insert(image.end(), content.begin(), content.end());
    }
    return image;
}

inline std::vector<std::byte> build_pfs0(
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& entries) {
    return build_partition(0x30534650, 0x18, entries); // "PFS0"
}

inline std::vector<std::byte> build_hfs0(
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& entries) {
    return build_partition(0x30534648, 0x40, entries); // "HFS0"
}

// Builds a minimal NPDM whose ACID carries `sign_key`'s modulus and a
// PSS signature made with the same test key. The structural sections
// satisfy the repository's NPDM reader.
inline std::vector<std::byte> build_npdm(const TestRsaKey& sign_key) {
    const std::array<std::uint8_t, 256>& acid_modulus = sign_key.modulus();
    // META 0x80 | ACI0 0x48 | ACID 0x248
    constexpr std::size_t kAci0Offset = 0x80;
    constexpr std::size_t kAci0Size = 0x48;
    constexpr std::size_t kAcidOffset = kAci0Offset + kAci0Size;
    constexpr std::size_t kAcidSize = 0x248;
    constexpr std::size_t kAcidDataSize = 0x240;
    std::vector<std::byte> image(kAcidOffset + kAcidSize, std::byte{0});
    put_u32(image, 0, 0x4154454D); // "META"
    image[0x0C] = std::byte{0};    // flags: ACID-signed, sign key index 0
    put_u32(image, 0x18, 0x00010000); // version
    put_u32(image, 0x1C, 0x100000);   // main stack size
    put_u32(image, 0x70, kAci0Offset);
    put_u32(image, 0x74, kAci0Size);
    put_u32(image, 0x78, kAcidOffset);
    put_u32(image, 0x7C, kAcidSize);

    // ACI0: header plus one empty-range subsection triple and a small KAC.
    put_u32(image, kAci0Offset, 0x30494341); // "ACI0"
    put_u32(image, kAci0Offset + 0x20, 0x40); // fs access header: rel 0x40 size 0
    put_u32(image, kAci0Offset + 0x24, 0);
    put_u32(image, kAci0Offset + 0x28, 0x40); // service access control
    put_u32(image, kAci0Offset + 0x2C, 0);
    put_u32(image, kAci0Offset + 0x30, 0x40); // kernel access control
    put_u32(image, kAci0Offset + 0x34, 4);

    // ACID: signature and modulus are filled by the caller through sign().
    put_u32(image, kAcidOffset + 0x200, 0x44494341); // "ACID"
    put_u32(image, kAcidOffset + 0x204, static_cast<std::uint32_t>(kAcidSize - 0x100));
    put_u32(image, kAcidOffset + 0x220, kAcidDataSize); // fs access control rel, size 0
    put_u32(image, kAcidOffset + 0x224, 0);
    put_u32(image, kAcidOffset + 0x228, kAcidDataSize); // service access control
    put_u32(image, kAcidOffset + 0x22C, 0);
    put_u32(image, kAcidOffset + 0x230, kAcidDataSize); // kernel access control
    put_u32(image, kAcidOffset + 0x234, 8);
    for (std::size_t i = 0; i < 256; ++i) {
        image[kAcidOffset + 0x100 + i] = std::byte(acid_modulus[i]);
    }
    // The ACID signature covers the section from its modulus field to the
    // end of its declared size.
    std::array<std::uint8_t, 256> signature{};
    const std::span<const std::byte> message = std::span<const std::byte>(image).subspan(
        kAcidOffset + 0x100, kAcidSize - 0x100);
    if (!sign_key.sign(message, signature)) {
        return {};
    }
    for (std::size_t i = 0; i < 256; ++i) {
        image[kAcidOffset + i] = std::byte(signature[i]);
    }
    return image;
}

// The synthetic trust anchor: the test key's modulus for both the NCA
// header and the ACID.
inline loader::ContainerTrustAnchors test_anchors(const TestRsaKey& key) {
    loader::ContainerTrustAnchors anchors;
    anchors.nca_header_moduli.push_back(key.modulus());
    anchors.acid_moduli.push_back(key.modulus());
    return anchors;
}

inline std::string to_hex(std::span<const std::byte> value) {
    static const char* digits = "0123456789abcdef";
    std::string text;
    for (std::byte b : value) {
        text.push_back(digits[static_cast<std::uint8_t>(b) >> 4]);
        text.push_back(digits[static_cast<std::uint8_t>(b) & 0xF]);
    }
    return text;
}

// A synthetic key set: the keys file text plus the raw key bytes.
struct TestKeys {
    std::array<std::byte, 32> header_key{};
    std::array<std::byte, 16> area_key{};
    std::array<std::byte, 16> titlekek{}; // for ticket title keys; update tests
                                          // append 'titlekek_00' to keys_text
    std::string keys_text; // prod.keys content with hex values

    [[nodiscard]] static TestKeys generate() {
        TestKeys keys;
        for (std::size_t i = 0; i < keys.header_key.size(); ++i) {
            keys.header_key[i] = std::byte(static_cast<std::uint8_t>(0x40 + i));
        }
        for (std::size_t i = 0; i < keys.area_key.size(); ++i) {
            keys.area_key[i] = std::byte(static_cast<std::uint8_t>(0xA0 + i));
        }
        for (std::size_t i = 0; i < keys.titlekek.size(); ++i) {
            keys.titlekek[i] = std::byte(static_cast<std::uint8_t>(0xD0 + i));
        }
        keys.keys_text = "header_key = " + to_hex(keys.header_key) +
                         "\n# a comment line\n\nkey_area_key_application_00 = " +
                         to_hex(keys.area_key) + "\n";
        return keys;
    }
};

// A synthetic RomFS image: root with one directory ("data") holding
// file_b, and file_a in the root. `file_a_offset`/`file_b_offset` locate
// the data extents inside the image.
struct BuiltRomFsImage {
    std::vector<std::byte> image;
    std::size_t file_a_offset = 0;
    std::size_t file_a_size = 0;
    std::size_t file_b_offset = 0;
    std::size_t file_b_size = 0;
};

inline BuiltRomFsImage build_romfs_image(std::span<const std::byte> file_a,
                                         std::span<const std::byte> file_b) {
    constexpr std::uint32_t kEmpty = 0xFFFFFFFF;
    // Layout: header 0x50 | dir hash 0x10 @0x50 | dir metadata @0x60 (root
    // 0x18, subdir 0x18 + 4-byte name) | file hash 0x10 | file metadata
    // (file_b 0x20+11, file_a 0x20+10) | data.
    const std::size_t dir_table_offset = 0x60;
    const std::size_t dir_table_size = 0x18 * 2 + 4;
    const std::size_t file_table_offset = dir_table_offset + dir_table_size;
    const std::size_t file_table_size = (0x20 + 10) + (0x20 + 9);
    const std::size_t data_offset = file_table_offset + 0x10 + file_table_size;
    const std::size_t romfs_size = data_offset + file_a.size() + file_b.size();
    std::vector<std::byte> romfs(romfs_size, std::byte{0});
    put_u64(romfs, 0x00, 0x50);
    put_u64(romfs, 0x08, 0x50);                       // dir hash table
    put_u64(romfs, 0x10, 0x10);
    put_u64(romfs, 0x18, dir_table_offset);           // dir metadata
    put_u64(romfs, 0x20, dir_table_size);
    put_u64(romfs, 0x28, file_table_offset);          // file hash table
    put_u64(romfs, 0x30, 0x10);
    put_u64(romfs, 0x38, file_table_offset + 0x10);   // file metadata
    put_u64(romfs, 0x40, file_table_size);
    put_u64(romfs, 0x48, data_offset);
    // Root directory: child -> subdir, file -> file_a.
    put_u32(romfs, dir_table_offset + 0x00, kEmpty);             // parent
    put_u32(romfs, dir_table_offset + 0x04, kEmpty);             // sibling
    put_u32(romfs, dir_table_offset + 0x08, 0x18);               // child dir
    put_u32(romfs, dir_table_offset + 0x0C, 0x20 + 10);          // file entry
    put_u32(romfs, dir_table_offset + 0x10, kEmpty);             // hash
    put_u32(romfs, dir_table_offset + 0x14, 0);                  // name size
    // Subdirectory "data": parent root, no children, file -> file_b.
    put_u32(romfs, dir_table_offset + 0x18 + 0x00, 0);
    put_u32(romfs, dir_table_offset + 0x18 + 0x04, kEmpty);
    put_u32(romfs, dir_table_offset + 0x18 + 0x08, kEmpty);
    put_u32(romfs, dir_table_offset + 0x18 + 0x0C, 0);           // file entry
    put_u32(romfs, dir_table_offset + 0x18 + 0x10, kEmpty);
    put_u32(romfs, dir_table_offset + 0x18 + 0x14, 4);
    std::memcpy(romfs.data() + dir_table_offset + 0x18 + 0x18, "data", 4);
    // file_b in "data" (first entry), file_a in root (second entry).
    const std::size_t file_b_entry = file_table_offset + 0x10;
    const std::size_t file_a_entry = file_b_entry + 0x20 + 10;
    put_u32(romfs, file_b_entry + 0x00, 0x18);        // parent dir
    put_u32(romfs, file_b_entry + 0x04, kEmpty);      // sibling
    put_u64(romfs, file_b_entry + 0x08, 0);
    put_u64(romfs, file_b_entry + 0x10, file_b.size());
    put_u32(romfs, file_b_entry + 0x18, kEmpty);
    put_u32(romfs, file_b_entry + 0x1C, 10);
    std::memcpy(romfs.data() + file_b_entry + 0x20, "nested.bin", 10);
    put_u32(romfs, file_a_entry + 0x00, 0);           // parent root
    put_u32(romfs, file_a_entry + 0x04, kEmpty);
    put_u64(romfs, file_a_entry + 0x08, file_b.size());
    put_u64(romfs, file_a_entry + 0x10, file_a.size());
    put_u32(romfs, file_a_entry + 0x18, kEmpty);
    put_u32(romfs, file_a_entry + 0x1C, 9);
    std::memcpy(romfs.data() + file_a_entry + 0x20, "hello.txt", 9);
    std::memcpy(romfs.data() + data_offset + file_b.size(), file_a.data(), file_a.size());
    std::memcpy(romfs.data() + data_offset, file_b.data(), file_b.size());
    BuiltRomFsImage built;
    built.image = std::move(romfs);
    built.file_a_offset = data_offset + file_b.size();
    built.file_a_size = file_a.size();
    built.file_b_offset = data_offset;
    built.file_b_size = file_b.size();
    return built;
}

// A synthetic RomFS tree spec: named directories and files in link order,
// so tests can demand wide sibling chains, deep nesting and dense
// directories. File bytes are a fixed pattern of the fill byte.
struct RomFsFileSpec {
    std::string name;
    std::uint64_t size = 0;
    std::uint8_t fill = 0; // byte i of the file is (i * 3 + fill) & 0xFF
};

struct RomFsDirSpec {
    std::string name; // unused for the tree root
    std::vector<RomFsDirSpec> dirs;
    std::vector<RomFsFileSpec> files;
};

struct BuiltRomFsTree {
    std::vector<std::byte> image;
    // Every file as a '/'-separated path below the root, in spec order.
    std::vector<std::pair<std::string, RomFsFileSpec>> files;
};

// Emits a complete RomFS image (header, metadata tables, data extents) for
// the spec, with the parent/sibling/child links the on-disk directory and
// file tables define. Entry offsets are four-byte aligned as the format's
// table writer aligns them.
inline BuiltRomFsTree build_romfs_tree(const RomFsDirSpec& root) {
    constexpr std::uint32_t kEmpty = 0xFFFFFFFF;
    auto align4 = [](std::uint32_t value) {
        return (value + 3u) & ~3u;
    };
    struct DirPlace {
        const RomFsDirSpec* spec = nullptr;
        std::uint32_t parent = kEmpty;
        std::uint32_t offset = 0;
        std::string path;
    };
    struct FilePlace {
        const RomFsFileSpec* spec = nullptr;
        std::uint32_t parent = 0;
        std::uint32_t offset = 0;
        std::uint64_t data_offset = 0;
        std::string path;
    };
    std::vector<DirPlace> dirs;
    std::vector<FilePlace> files;
    dirs.push_back({&root, kEmpty, 0, {}});
    std::uint32_t dir_cursor = align4(0x18); // root: fixed part, empty name
    std::uint64_t data_cursor = 0;
    for (std::size_t i = 0; i < dirs.size(); ++i) {
        const DirPlace place = dirs[i];
        for (const RomFsDirSpec& child : place.spec->dirs) {
            DirPlace next;
            next.spec = &child;
            next.parent = place.offset;
            next.offset = dir_cursor;
            dir_cursor = align4(dir_cursor + 0x18 +
                                static_cast<std::uint32_t>(child.name.size()));
            next.path = place.path.empty() ? child.name : place.path + "/" + child.name;
            dirs.push_back(std::move(next));
        }
        for (const RomFsFileSpec& file : place.spec->files) {
            FilePlace next;
            next.spec = &file;
            next.parent = place.offset;
            next.data_offset = data_cursor;
            data_cursor += file.size;
            next.path = place.path.empty() ? file.name : place.path + "/" + file.name;
            files.push_back(std::move(next));
        }
    }
    std::uint32_t file_cursor = 0;
    for (FilePlace& file : files) {
        file.offset = file_cursor;
        file_cursor = align4(file_cursor + 0x20 +
                             static_cast<std::uint32_t>(file.spec->name.size()));
    }

    // Sibling order follows the spec order collected per parent above.
    std::map<std::uint32_t, std::vector<std::uint32_t>> child_dirs;
    std::map<std::uint32_t, std::vector<std::uint32_t>> child_files;
    for (std::size_t i = 1; i < dirs.size(); ++i) {
        child_dirs[dirs[i].parent].push_back(dirs[i].offset);
    }
    for (const FilePlace& file : files) {
        child_files[file.parent].push_back(file.offset);
    }

    // Layout mirrors build_romfs_image: header 0x50, one-block hash tables,
    // directory metadata, file metadata, data extents.
    const std::size_t dir_table_offset = 0x60;
    const std::size_t dir_table_size = dir_cursor;
    const std::size_t file_table_offset = dir_table_offset + dir_table_size;
    const std::size_t file_table_size = file_cursor;
    const std::size_t data_offset = file_table_offset + 0x10 + file_table_size;
    std::vector<std::byte> image(data_offset + static_cast<std::size_t>(data_cursor),
                                 std::byte{0});
    put_u64(image, 0x00, 0x50);
    put_u64(image, 0x08, 0x50);                       // dir hash table
    put_u64(image, 0x10, 0x10);
    put_u64(image, 0x18, dir_table_offset);           // dir metadata
    put_u64(image, 0x20, dir_table_size);
    put_u64(image, 0x28, file_table_offset);          // file hash table
    put_u64(image, 0x30, 0x10);
    put_u64(image, 0x38, file_table_offset + 0x10);   // file metadata
    put_u64(image, 0x40, file_table_size);
    put_u64(image, 0x48, data_offset);

    auto sibling_of = [&](const std::map<std::uint32_t, std::vector<std::uint32_t>>& lists,
                          std::uint32_t parent, std::uint32_t offset) -> std::uint32_t {
        const auto it = lists.find(parent);
        if (it == lists.end()) {
            return kEmpty;
        }
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            if (it->second[i] == offset) {
                return i + 1 < it->second.size() ? it->second[i + 1] : kEmpty;
            }
        }
        return kEmpty;
    };

    for (const DirPlace& dir : dirs) {
        const std::size_t at = dir_table_offset + dir.offset;
        put_u32(image, at + 0x00, dir.parent);
        put_u32(image, at + 0x04, sibling_of(child_dirs, dir.parent, dir.offset));
        const auto children = child_dirs.find(dir.offset);
        put_u32(image, at + 0x08,
                children != child_dirs.end() && !children->second.empty()
                    ? children->second.front()
                    : kEmpty);
        const auto dir_files = child_files.find(dir.offset);
        put_u32(image, at + 0x0C,
                dir_files != child_files.end() && !dir_files->second.empty()
                    ? dir_files->second.front()
                    : kEmpty);
        put_u32(image, at + 0x10, kEmpty);
        put_u32(image, at + 0x14,
                static_cast<std::uint32_t>(dir.spec->name.size()));
        if (dir.offset != 0) {
            std::memcpy(image.data() + at + 0x18, dir.spec->name.data(),
                        dir.spec->name.size());
        }
    }
    for (const FilePlace& file : files) {
        const std::size_t at = file_table_offset + 0x10 + file.offset;
        put_u32(image, at + 0x00, file.parent);
        put_u32(image, at + 0x04, sibling_of(child_files, file.parent, file.offset));
        put_u64(image, at + 0x08, file.data_offset);
        put_u64(image, at + 0x10, file.spec->size);
        put_u32(image, at + 0x18, kEmpty);
        put_u32(image, at + 0x1C, static_cast<std::uint32_t>(file.spec->name.size()));
        std::memcpy(image.data() + at + 0x20, file.spec->name.data(),
                    file.spec->name.size());
        const std::size_t data_at = data_offset + static_cast<std::size_t>(file.data_offset);
        for (std::uint64_t i = 0; i < file.spec->size; ++i) {
            image[data_at + static_cast<std::size_t>(i)] =
                std::byte(static_cast<std::uint8_t>(i * 3 + file.spec->fill));
        }
    }

    BuiltRomFsTree built;
    built.image = std::move(image);
    for (const FilePlace& file : files) {
        built.files.emplace_back(file.path, *file.spec);
    }
    return built;
}

// The IVFC hash tree over a RomFS image. Levels are sized bottom-up (one
// digest per hashed block of the level below) and laid out top-down on
// 0x200-byte boundaries with the image last, so images larger than one
// level-four block grow the upper levels instead of overrunning them.
struct BuiltIvfcSection {
    std::vector<std::byte> section; // levels 1..5 then the image
    std::array<std::uint64_t, 6> level_offsets{};
    std::array<std::uint64_t, 6> level_sizes{};
    std::array<std::byte, 32> master_hash{}; // over the padded level-1 block
};

inline BuiltIvfcSection build_ivfc_section(std::span<const std::byte> romfs_image) {
    constexpr std::uint64_t kHashBlockSize = 0x200;
    constexpr std::uint64_t kDigestSize = 0x20;
    BuiltIvfcSection built;
    std::array<std::uint64_t, 6> sizes{};
    sizes[5] = romfs_image.size();
    for (std::size_t level = 5; level >= 1; --level) {
        const std::uint64_t entries =
            (sizes[level] + kHashBlockSize - 1) / kHashBlockSize;
        sizes[level - 1] = entries * kDigestSize;
    }
    std::uint64_t cursor = 0;
    for (std::size_t level = 0; level < 6; ++level) {
        built.level_offsets[level] = cursor;
        built.level_sizes[level] = sizes[level];
        cursor += ((sizes[level] + kHashBlockSize - 1) / kHashBlockSize) * kHashBlockSize;
    }
    const std::uint64_t image_blocks =
        (romfs_image.size() + kHashBlockSize - 1) / kHashBlockSize;
    built.section.assign(static_cast<std::size_t>(built.level_offsets[5]) +
                             image_blocks * static_cast<std::size_t>(kHashBlockSize),
                         std::byte{0});
    std::memcpy(built.section.data() + built.level_offsets[5], romfs_image.data(),
                romfs_image.size());
    for (std::size_t level = 5; level >= 1; --level) {
        const std::uint64_t entries =
            (sizes[level] + kHashBlockSize - 1) / kHashBlockSize;
        for (std::uint64_t entry = 0; entry < entries; ++entry) {
            std::vector<std::byte> block(static_cast<std::size_t>(kHashBlockSize), std::byte{0});
            const std::size_t begin = static_cast<std::size_t>(
                built.level_offsets[level] + entry * kHashBlockSize);
            const std::size_t take = static_cast<std::size_t>(std::min<std::uint64_t>(
                kHashBlockSize, sizes[level] - entry * kHashBlockSize));
            std::memcpy(block.data(), built.section.data() + begin, take);
            const std::array<std::byte, 32> digest = fixture_sha256(block);
            std::memcpy(built.section.data() +
                            static_cast<std::size_t>(built.level_offsets[level - 1]) +
                            static_cast<std::size_t>(entry * kDigestSize),
                        digest.data(), 0x20);
        }
    }
    built.master_hash = fixture_sha256(std::span<const std::byte>(
        built.section.data() + static_cast<std::size_t>(built.level_offsets[0]),
        static_cast<std::size_t>(kHashBlockSize)));
    return built;
}

// Builds an NCA3 program image around the supplied ExeFS partition and
// RomFS images. `sign_key` signs both header signatures; its modulus must
// be the injected trust anchor and the ACID modulus inside `npdm`. The
// image carries complete hash chains (FS header hash table, PFS0
// hierarchical SHA-256, RomFS IVFC tree) so extraction verifies them.
struct BuiltNca {
    std::vector<std::byte> image;
    std::array<std::byte, 16> content_key{};
    std::array<std::byte, 16> nca_id{};
    std::array<std::byte, 32> header_hash{};
    std::array<std::byte, 32> content_hash{};
    // Absolute file offsets of the hash-verified regions, for corruption
    // tests that flip a single byte and expect the matching structured
    // failure.
    std::uint64_t exefs_table_offset = 0; // first hash table byte
    std::uint64_t exefs_data_offset = 0;  // first PFS0 partition byte
    std::uint64_t romfs_section_offset = 0;
    std::array<std::uint64_t, 6> ivfc_level_offsets{}; // start of each level's data
};

inline BuiltNca build_program_nca(
    const TestKeys& keys, const TestRsaKey& sign_key, std::span<const std::byte> npdm,
    bool encrypt_header = true, bool with_rights_id = false, std::uint64_t title_id = 0,
    std::span<const std::byte> romfs_image = {}, std::uint8_t content_type = 0,
    const std::vector<std::pair<std::string, std::vector<std::byte>>>* pfs0_entries = nullptr) {
    // The ExeFS PFS0 partition: main.npdm plus a synthetic module file, or a
    // caller-supplied meta/control partition for pack-input fixtures.
    std::vector<std::byte> exefs_pattern(0x300);
    for (std::size_t i = 0; i < exefs_pattern.size(); ++i) {
        exefs_pattern[i] = std::byte(static_cast<std::uint8_t>(i * 7 + 1));
    }
    const std::vector<std::pair<std::string, std::vector<std::byte>>> default_pfs0_entries = {
        {"main", exefs_pattern}, {"main.npdm", {npdm.begin(), npdm.end()}}};
    const std::vector<std::byte> pfs0 =
        build_pfs0(pfs0_entries ? *pfs0_entries : default_pfs0_entries);

    // The RomFS image: a caller-supplied synthetic tree, or by default the
    // root with one file and one directory with one file.
    std::vector<std::byte> default_file_a(0x123);
    for (std::size_t i = 0; i < default_file_a.size(); ++i) {
        default_file_a[i] = std::byte(static_cast<std::uint8_t>(i * 3 + 5));
    }
    std::vector<std::byte> default_file_b(0x47);
    for (std::size_t i = 0; i < default_file_b.size(); ++i) {
        default_file_b[i] = std::byte(static_cast<std::uint8_t>(i * 11 + 9));
    }
    std::vector<std::byte> default_romfs;
    if (romfs_image.empty()) {
        default_romfs = build_romfs_image(default_file_a, default_file_b).image;
    }
    const std::span<const std::byte> romfs =
        romfs_image.empty() ? std::span<const std::byte>(default_romfs) : romfs_image;

    // Section layout: header 0xC00, ExeFS at 0x1000 (hash table then PFS0),
    // RomFS next boundary (IVFC tree then the image).
    constexpr std::uint64_t kHashBlockSize = 0x200;
    constexpr std::uint32_t kHashBlockLog2 = 9;
    const std::uint64_t exefs_offset = 0x1000;

    // The ExeFS section: a hierarchical SHA-256 table over the PFS0
    // partition, then the partition itself. The final partial data block
    // hashes unpadded, as the format's partition hash table requires.
    const std::uint64_t pfs0_offset = 0x200;
    const std::size_t pfs0_size = pfs0.size();
    const std::size_t table_entries =
        static_cast<std::size_t>((pfs0_size + kHashBlockSize - 1) / kHashBlockSize);
    std::vector<std::byte> exefs_section(
        static_cast<std::size_t>(pfs0_offset) + ((pfs0_size + 0x1FF) & ~std::size_t{0x1FF}),
        std::byte{0});
    std::memcpy(exefs_section.data() + pfs0_offset, pfs0.data(), pfs0_size);
    for (std::size_t entry = 0; entry < table_entries; ++entry) {
        const std::size_t begin = static_cast<std::size_t>(entry * kHashBlockSize);
        const std::size_t end =
            std::min<std::size_t>(begin + static_cast<std::size_t>(kHashBlockSize), pfs0_size);
        const std::array<std::byte, 32> digest =
            fixture_sha256(std::span<const std::byte>(pfs0).subspan(begin, end - begin));
        std::memcpy(exefs_section.data() + entry * 0x20, digest.data(), 0x20);
    }

    // The RomFS section: an IVFC hash tree over the RomFS image (shared
    // with the BKTR fixture builder).
    const BuiltIvfcSection romfs_tree = build_ivfc_section(romfs);
    const std::vector<std::byte>& romfs_section = romfs_tree.section;
    const std::array<std::uint64_t, 6> level_offsets = romfs_tree.level_offsets;
    const std::array<std::uint64_t, 6>& level_sizes = romfs_tree.level_sizes;

    const std::uint64_t romfs_offset = exefs_offset + exefs_section.size();
    BuiltNca built;
    built.content_key = {std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44},
                         std::byte{0x55}, std::byte{0x66}, std::byte{0x77}, std::byte{0x88},
                         std::byte{0x99}, std::byte{0xAA}, std::byte{0xBB}, std::byte{0xCC},
                         std::byte{0xDD}, std::byte{0xEE}, std::byte{0x0F}, std::byte{0xF0}};
    built.exefs_table_offset = exefs_offset;
    built.exefs_data_offset = exefs_offset + pfs0_offset;
    built.romfs_section_offset = romfs_offset;
    for (std::size_t level = 0; level < 6; ++level) {
        built.ivfc_level_offsets[level] = romfs_offset + level_offsets[level];
    }
    std::vector<std::byte> image(romfs_offset + romfs_section.size(), std::byte{0});
    put_u32(image, 0x200, 0x3341434E); // "NCA3"
    image[0x204] = std::byte{0};       // distribution type: download
    image[0x205] = std::byte{content_type};
    image[0x206] = std::byte{1};       // crypto type: master key generation 0
    image[0x207] = std::byte{0};       // key area index: application
    put_u64(image, 0x208, image.size());
    put_u64(image, 0x210, title_id);
    image[0x220] = std::byte{0};       // crypto type 2
    image[0x221] = std::byte{0};       // fixed key generation 0
    if (with_rights_id) {
        for (std::size_t i = 0; i < 16; ++i) {
            image[0x230 + i] = std::byte(static_cast<std::uint8_t>(i + 1));
        }
    }
    // Section entries (media units of 0x200).
    put_u32(image, 0x240, static_cast<std::uint32_t>(exefs_offset / 0x200));
    put_u32(image, 0x244,
            static_cast<std::uint32_t>((exefs_offset + exefs_section.size()) / 0x200));
    put_u32(image, 0x250, static_cast<std::uint32_t>(romfs_offset / 0x200));
    put_u32(image, 0x254,
            static_cast<std::uint32_t>((romfs_offset + romfs_section.size()) / 0x200));

    // ExeFS FS header: PFS0 partition with CTR and its hierarchical
    // SHA-256 superblock.
    const std::size_t exefs_header = 0x400;
    image[exefs_header + 0x02] = std::byte{1}; // partition: PFS0
    image[exefs_header + 0x03] = std::byte{2}; // fs type: PFS0
    image[exefs_header + 0x04] = std::byte{3}; // crypt: CTR
    {
        const std::size_t superblock = exefs_header + 0x08;
        const std::array<std::byte, 32> master =
            fixture_sha256(std::span<const std::byte>(exefs_section.data(), table_entries * 0x20));
        std::memcpy(image.data() + superblock + 0x00, master.data(), 0x20);
        put_u32(image, superblock + 0x20, static_cast<std::uint32_t>(kHashBlockSize)); // block size
        put_u32(image, superblock + 0x24, 2);                    // levels
        put_u64(image, superblock + 0x28, 0);                    // hash table offset
        put_u64(image, superblock + 0x30, table_entries * 0x20); // hash table size
        put_u64(image, superblock + 0x38, pfs0_offset);          // PFS0 partition offset
        put_u64(image, superblock + 0x40, pfs0_size);            // PFS0 partition size
    }
    const std::array<std::byte, 8> exefs_ctr = {std::byte{0x01}, std::byte{0x23}, std::byte{0x45},
                                                std::byte{0x67}, std::byte{0x89}, std::byte{0xAB},
                                                std::byte{0xCD}, std::byte{0xEF}};
    for (std::size_t i = 0; i < 8; ++i) {
        image[exefs_header + 0x140 + i] = exefs_ctr[i];
    }
    // RomFS FS header: IVFC superblock with the six level descriptors and
    // the master hash over the zero-padded level-1 block.
    const std::size_t romfs_header = 0x600;
    image[romfs_header + 0x02] = std::byte{0}; // partition: RomFS
    image[romfs_header + 0x03] = std::byte{3}; // fs type: RomFS
    image[romfs_header + 0x04] = std::byte{3}; // crypt: CTR
    {
        const std::size_t superblock = romfs_header + 0x08;
        put_u32(image, superblock + 0x00, 0x43465649); // "IVFC"
        put_u32(image, superblock + 0x04, 0x00010000); // id
        put_u32(image, superblock + 0x08, 0x20);       // master hash size
        put_u32(image, superblock + 0x0C, 7);          // six levels plus the master
        for (std::size_t level = 0; level < 6; ++level) {
            const std::size_t descriptor = superblock + 0x10 + level * 0x18;
            put_u64(image, descriptor, level_offsets[level]);
            put_u64(image, descriptor + 0x08, level_sizes[level]);
            put_u32(image, descriptor + 0x10, kHashBlockLog2);
        }
        const std::array<std::byte, 32> master = fixture_sha256(std::span<const std::byte>(
            romfs_section.data() + static_cast<std::size_t>(level_offsets[0]),
            static_cast<std::size_t>(kHashBlockSize)));
        std::memcpy(image.data() + superblock + 0xC0, master.data(), 0x20);
    }
    const std::array<std::byte, 8> romfs_ctr = {std::byte{0xFE}, std::byte{0xDC}, std::byte{0xBA},
                                                std::byte{0x98}, std::byte{0x76}, std::byte{0x54},
                                                std::byte{0x32}, std::byte{0x10}};
    for (std::size_t i = 0; i < 8; ++i) {
        image[romfs_header + 0x140 + i] = romfs_ctr[i];
    }

    // Key area: ECB-encrypt the slot keys with the key-area key; slot 2 is
    // the content key.
    for (std::size_t slot = 0; slot < 4; ++slot) {
        std::array<std::byte, 16> plain{};
        if (slot == 2) {
            plain = built.content_key;
        } else {
            for (std::size_t i = 0; i < 16; ++i) {
                plain[i] = std::byte(static_cast<std::uint8_t>(0x30 + slot * 16 + i));
            }
        }
        std::array<std::byte, 16> cipher{};
        (void)fixture_aes_ecb_encrypt(keys.area_key.data(), plain.data(), cipher.data());
        std::memcpy(image.data() + 0x300 + slot * 16, cipher.data(), 16);
    }

    // Section content: CTR-encrypt with the content key. The counter is the
    // FS header's 8 bytes reversed in the high half and the 16-byte block
    // number of the absolute offset in the low half.
    auto encrypt_section = [&](std::uint64_t section_offset, std::size_t section_size,
                               const std::array<std::byte, 8>& counter_bytes,
                               const std::vector<std::byte>& plain) {
        std::vector<std::byte> data(section_size, std::byte{0});
        std::memcpy(data.data(), plain.data(), plain.size());
        for (std::uint64_t block = 0; block * 16 < data.size(); ++block) {
            std::uint8_t counter[16] = {};
            for (std::size_t i = 0; i < 8; ++i) {
                counter[i] = static_cast<std::uint8_t>(counter_bytes[7 - i]);
            }
            const std::uint64_t block_number =
                (section_offset + block * 16) >> 4;
            for (std::size_t i = 0; i < 8; ++i) {
                counter[15 - i] = static_cast<std::uint8_t>(block_number >> (8 * i));
            }
            // Apply one block at a time so each block's counter is exact.
            const std::size_t end =
                std::min<std::size_t>((block + 1) * 16, data.size());
            std::vector<std::byte> chunk(data.begin() + static_cast<std::ptrdiff_t>(block * 16),
                                         data.begin() + static_cast<std::ptrdiff_t>(end));
            (void)fixture_aes_ctr_apply(built.content_key.data(), counter, chunk);
            std::memcpy(data.data() + block * 16, chunk.data(), chunk.size());
        }
        std::memcpy(image.data() + section_offset, data.data(), section_size);
    };
    encrypt_section(exefs_offset, exefs_section.size(), exefs_ctr, exefs_section);
    encrypt_section(romfs_offset, romfs_section.size(), romfs_ctr, romfs_section);

    // The FS header hash table: one SHA-256 per 0x200-byte FS header at
    // 0x280 + 0x20 * index, inside the signed header region. Written after
    // the FS headers are final and before the header is signed.
    for (std::size_t index = 0; index < 4; ++index) {
        const std::array<std::byte, 32> digest = fixture_sha256(
            std::span<const std::byte>(image).subspan(0x400 + index * 0x200, 0x200));
        std::memcpy(image.data() + 0x280 + index * 0x20, digest.data(), 0x20);
    }

    // Signatures over the signed header region, then header encryption.
    std::array<std::uint8_t, 256> signature{};
    const std::span<const std::byte> message = std::span<const std::byte>(image).subspan(
        0x200, 0x200);
    (void)sign_key.sign(message, signature);
    for (std::size_t i = 0; i < 256; ++i) {
        image[i] = std::byte(signature[i]);                  // signature 1
        image[0x100 + i] = std::byte(signature[i]);          // signature 2 (same test key)
    }
    built.header_hash = fixture_sha256(std::span<const std::byte>(image).first(0xC00));
    std::copy_n(built.header_hash.begin(), built.nca_id.size(), built.nca_id.begin());
    if (encrypt_header) {
        std::vector<std::byte> header(image.begin(), image.begin() + 0xC00);
        (void)fixture_xts_encrypt(keys.header_key.data(), header, 0, 0x200);
        std::memcpy(image.data(), header.data(), 0xC00);
    }
    built.content_hash = fixture_sha256(image);
    built.image = std::move(image);
    return built;
}

// ---------------------------------------------------------------------------
// BKTR update fixture: an update program NCA whose RomFS section is a
// BKTR patch over a base program NCA's RomFS, plus its common ticket.
// ---------------------------------------------------------------------------

// A common ticket (v2, RSA-2048 SHA-256 signature type 0x10004) carrying
// the ticket-encrypted title key and the rights id.
inline std::vector<std::byte> build_common_ticket(const std::array<std::byte, 16>& rights_id,
                                                  const std::array<std::byte, 16>& title_key,
                                                  const std::array<std::byte, 16>& titlekek) {
    std::vector<std::byte> ticket(0x350, std::byte{0});
    put_u32(ticket, 0x000, 0x10004); // signature type: RSA-2048 SHA-256
    std::array<std::byte, 16> encrypted{};
    (void)fixture_aes_ecb_encrypt(titlekek.data(), title_key.data(), encrypted.data());
    std::memcpy(ticket.data() + 0x180, encrypted.data(), 16); // title key block
    ticket[0x280] = std::byte{2};                             // format version
    ticket[0x285] = std::byte{1};                             // common key generation
    std::memcpy(ticket.data() + 0x2A0, rights_id.data(), 16);
    return ticket;
}

struct BuiltUpdateNca {
    std::vector<std::byte> image;
    std::vector<std::byte> ticket;
    std::array<std::byte, 16> title_key{};
    std::array<std::byte, 16> rights_id{};
    std::string rights_id_hex;          // 32 lowercase hex digits
    std::uint64_t bktr_section_offset = 0; // image offset of the BKTR section
    // Absolute image offsets of patch-only bytes (relocation to the update
    // section), for corruption tests that expect an IVFC failure.
    std::uint64_t patch_table_offset = 0; // inside the patched level tables
    std::uint64_t patch_data_offset = 0;  // inside the patched image data
};

// Builds an update program NCA with titlekey encryption (rights id) and a
// BKTR RomFS section layered over the base fixture's RomFS. The patched
// image equals the base section image except hello.txt's bytes (same
// size), so the relocation tree alternates patch/base/patch:
//   [0x000,0x840) patch  (IVFC levels 1-5 tables; all differ)
//   [0x840,0xA48) base   (level-5 padding and image block 0 head; equal)
//   [0xA48,0xE00) patch  (image bytes holding hello.txt; differ)
// The patch data holds two subsections with different counters.
inline BuiltUpdateNca build_update_program_nca(const TestKeys& keys, const TestRsaKey& sign_key,
                                               std::span<const std::byte> npdm,
                                               std::uint64_t title_id,
                                               std::uint64_t subsection_total_adjustment = 0) {
    constexpr std::uint64_t kHashBlockSize = 0x200;
    constexpr std::uint32_t kHashBlockLog2 = 9;
    BuiltUpdateNca built;

    // The title key doubles as the update NCA's content key.
    for (std::size_t i = 0; i < built.title_key.size(); ++i) {
        built.title_key[i] = std::byte(static_cast<std::uint8_t>(0xE0 + i));
    }
    for (std::size_t i = 0; i < 8; ++i) {
        built.rights_id[i] = std::byte(static_cast<std::uint8_t>(title_id >> (8 * (7 - i))));
    }
    built.rights_id[15] = std::byte{1}; // key generation (last byte) 1 -> titlekek_00
    built.rights_id_hex = to_hex(built.rights_id);
    built.ticket = build_common_ticket(built.rights_id, built.title_key, keys.titlekek);

    // The patched RomFS image: the base layout with different hello.txt
    // bytes of the same length.
    std::vector<std::byte> hello_base(0x123);
    for (std::size_t i = 0; i < hello_base.size(); ++i) {
        hello_base[i] = std::byte(static_cast<std::uint8_t>(i * 3 + 5));
    }
    std::vector<std::byte> hello_patched(0x123);
    for (std::size_t i = 0; i < hello_patched.size(); ++i) {
        hello_patched[i] = std::byte(static_cast<std::uint8_t>(i * 5 + 0x71));
    }
    std::vector<std::byte> nested(0x47);
    for (std::size_t i = 0; i < nested.size(); ++i) {
        nested[i] = std::byte(static_cast<std::uint8_t>(i * 11 + 9));
    }
    const BuiltRomFsImage patched = build_romfs_image(hello_patched, nested);
    const BuiltIvfcSection base_tree = build_ivfc_section(
        build_romfs_image(hello_base, nested).image);
    const BuiltIvfcSection patched_tree = build_ivfc_section(patched.image);
    // The virtual image the BKTR section describes.
    const std::vector<std::byte>& virtual_image = patched_tree.section;
    const std::uint64_t virtual_size = virtual_image.size();
    // The base ranges must be byte-identical in both images; the builder
    // fails loudly here rather than emitting a fixture that cannot verify.
    const std::uint64_t kBaseRangeBegin = 0x840;
    const std::uint64_t kBaseRangeEnd = 0xA48;
    for (std::uint64_t i = kBaseRangeBegin; i < kBaseRangeEnd; ++i) {
        if (virtual_image[std::size_t(i)] != base_tree.section[std::size_t(i)]) {
            return built; // empty image marks a broken fixture contract
        }
    }

    // Patch physical layout: the two patch ranges back to back, the second
    // padded to a 16-byte boundary.
    const std::uint64_t kPatch1Begin = 0x000;
    const std::uint64_t kPatch1End = 0x840;
    const std::uint64_t kPatch2Begin = 0xA48;
    const std::uint64_t kPatch2End = virtual_size;
    std::vector<std::byte> patch;
    patch.insert(patch.end(), virtual_image.begin() + std::int64_t(kPatch1Begin),
                 virtual_image.begin() + std::int64_t(kPatch1End));
    const std::uint64_t patch2_phys = patch.size();
    patch.insert(patch.end(), virtual_image.begin() + std::int64_t(kPatch2Begin),
                 virtual_image.begin() + std::int64_t(kPatch2End));
    // The patch data extent is 16-byte aligned so the tree regions that
    // follow start on CTR block boundaries.
    while (patch.size() % 16 != 0) {
        patch.push_back(std::byte{0});
    }
    const std::uint64_t patch_end = patch.size();

    // Bucket trees. Relocation: three entries, patch/base/patch, one
    // bucket. Subsection: two entries with different counters, one bucket.
    constexpr std::size_t kBucketSize = 0x4000;
    const std::size_t kTreeSize = kBucketSize * 2; // header page plus one bucket
    auto make_tree_block = [](std::uint32_t num_buckets, std::uint64_t total,
                              const std::vector<std::pair<std::size_t, std::vector<std::byte>>>&
                                  bucket_payloads) {
        std::vector<std::byte> block(kTreeSize, std::byte{0});
        put_u32(block, 0x04, num_buckets);
        put_u64(block, 0x08, total);
        std::size_t bucket_base = kBucketSize; // buckets follow the header page
        for (std::uint32_t i = 0; i < num_buckets; ++i) {
            put_u64(block, 0x10 + std::size_t{i} * 8, bucket_payloads[i].first);
            std::memcpy(block.data() + bucket_base, bucket_payloads[i].second.data(),
                        bucket_payloads[i].second.size());
            bucket_base += kBucketSize;
        }
        return block;
    };
    std::vector<std::byte> reloc_bucket(0x10 + 3 * 0x14, std::byte{0});
    put_u32(reloc_bucket, 0x04, 3);
    put_u64(reloc_bucket, 0x08, virtual_size);
    auto put_reloc = [&](std::size_t index, std::uint64_t virt, std::uint64_t phys,
                         std::uint32_t is_patch) {
        const std::size_t at = 0x10 + index * 0x14;
        put_u64(reloc_bucket, at, virt);
        put_u64(reloc_bucket, at + 0x08, phys);
        put_u32(reloc_bucket, at + 0x10, is_patch);
    };
    put_reloc(0, kPatch1Begin, 0, 1);
    put_reloc(1, kPatch1End, kPatch1End, 0); // base range at its own offset
    put_reloc(2, kPatch2Begin, patch2_phys, 1);
    const std::vector<std::byte> relocation_tree =
        make_tree_block(1, virtual_size, {{0, reloc_bucket}});

    constexpr std::uint32_t kSubsectionCtr1 = 0x11111111;
    constexpr std::uint32_t kSubsectionCtr2 = 0x22222222;
    constexpr std::uint64_t kSubsectionSplit = 0x400; // inside the first patch range
    std::vector<std::byte> subsection_bucket(0x10 + 2 * 0x10, std::byte{0});
    put_u32(subsection_bucket, 0x04, 2);
    put_u64(subsection_bucket, 0x08, patch_end);
    // Entry layout: u64 physical offset, u32 reserved, u32 counter value.
    put_u64(subsection_bucket, 0x10, 0);
    put_u32(subsection_bucket, 0x1C, kSubsectionCtr1);
    put_u64(subsection_bucket, 0x20, kSubsectionSplit);
    put_u32(subsection_bucket, 0x2C, kSubsectionCtr2);
    // Section layout: patch data at 0, relocation tree, subsection tree,
    // padded to the media-unit section size. The subsection block's total is
    // either the patch-data extent or the subsection tree's own offset.
    const std::uint64_t kSectionAlignment = 0x200;
    auto round_up_media = [&](std::uint64_t value) {
        return (value + kSectionAlignment - 1) & ~(kSectionAlignment - 1);
    };
    const std::uint64_t relocation_offset = patch_end;
    const std::uint64_t subsection_offset = relocation_offset + kTreeSize;
    const std::uint64_t section_size = round_up_media(subsection_offset + kTreeSize);
    const std::vector<std::byte> subsection_tree =
        make_tree_block(1, relocation_offset + subsection_total_adjustment,
                        {{0, subsection_bucket}});
    built.bktr_section_offset = 0; // set below, after the ExeFS layout

    // The ExeFS: a PFS0 with a patched module and the same NPDM.
    std::vector<std::byte> module(0x300);
    for (std::size_t i = 0; i < module.size(); ++i) {
        module[i] = std::byte(static_cast<std::uint8_t>(i * 7 + 2));
    }
    const std::vector<std::byte> pfs0 = build_pfs0(
        {{"main", module}, {"main.npdm", {npdm.begin(), npdm.end()}}});
    const std::uint64_t pfs0_offset = 0x200;
    const std::size_t table_entries =
        static_cast<std::size_t>((pfs0.size() + kHashBlockSize - 1) / kHashBlockSize);
    std::vector<std::byte> exefs_section(
        static_cast<std::size_t>(pfs0_offset) +
            ((pfs0.size() + std::size_t{kHashBlockSize} - 1) & ~std::size_t{kHashBlockSize - 1}),
        std::byte{0});
    std::memcpy(exefs_section.data() + pfs0_offset, pfs0.data(), pfs0.size());
    for (std::size_t entry = 0; entry < table_entries; ++entry) {
        const std::size_t begin = static_cast<std::size_t>(entry * kHashBlockSize);
        const std::size_t end =
            std::min<std::size_t>(begin + static_cast<std::size_t>(kHashBlockSize), pfs0.size());
        const std::array<std::byte, 32> digest =
            fixture_sha256(std::span<const std::byte>(pfs0).subspan(begin, end - begin));
        std::memcpy(exefs_section.data() + entry * 0x20, digest.data(), 0x20);
    }

    const std::uint64_t exefs_offset = 0x1000;
    const std::uint64_t bktr_offset = exefs_offset + exefs_section.size();
    built.bktr_section_offset = bktr_offset;
    built.patch_table_offset = bktr_offset + 0x10;  // patched level-1 table byte
    built.patch_data_offset = bktr_offset + patch2_phys + 0x30; // patched image byte

    std::vector<std::byte> image(std::size_t(bktr_offset + section_size), std::byte{0});
    put_u32(image, 0x200, 0x3341434E); // "NCA3"
    image[0x204] = std::byte{0};       // content type: program
    image[0x206] = std::byte{0};       // crypto type
    image[0x207] = std::byte{0};       // key area index (unused for titlekey)
    put_u64(image, 0x208, image.size());
    put_u64(image, 0x210, title_id);
    image[0x220] = std::byte{0};
    image[0x221] = std::byte{0};
    std::memcpy(image.data() + 0x230, built.rights_id.data(), 16);
    put_u32(image, 0x240, static_cast<std::uint32_t>(exefs_offset / 0x200));
    put_u32(image, 0x244, static_cast<std::uint32_t>(bktr_offset / 0x200));
    put_u32(image, 0x250, static_cast<std::uint32_t>(bktr_offset / 0x200));
    put_u32(image, 0x254,
            static_cast<std::uint32_t>((bktr_offset + section_size) / 0x200));

    // ExeFS FS header (plain CTR).
    const std::size_t exefs_header = 0x400;
    image[exefs_header + 0x02] = std::byte{1};
    image[exefs_header + 0x03] = std::byte{2};
    image[exefs_header + 0x04] = std::byte{3};
    {
        const std::size_t superblock = exefs_header + 0x08;
        const std::array<std::byte, 32> master =
            fixture_sha256(std::span<const std::byte>(exefs_section.data(),
                                                      table_entries * 0x20));
        std::memcpy(image.data() + superblock, master.data(), 0x20);
        put_u32(image, superblock + 0x20, static_cast<std::uint32_t>(kHashBlockSize));
        put_u32(image, superblock + 0x24, 2);
        put_u64(image, superblock + 0x28, 0);
        put_u64(image, superblock + 0x30, table_entries * 0x20);
        put_u64(image, superblock + 0x38, pfs0_offset);
        put_u64(image, superblock + 0x40, pfs0.size());
    }
    const std::array<std::byte, 8> exefs_ctr = {std::byte{0x11}, std::byte{0x0A},
                                                std::byte{0x2B}, std::byte{0x3C},
                                                std::byte{0x4D}, std::byte{0x5E},
                                                std::byte{0x6F}, std::byte{0x70}};
    for (std::size_t i = 0; i < 8; ++i) {
        image[exefs_header + 0x140 + i] = exefs_ctr[i];
    }

    // RomFS FS header: IVFC superblock (virtual level descriptors of the
    // patched image) plus the BKTR bucket-tree headers, crypt type 4.
    const std::size_t romfs_header = 0x600;
    image[romfs_header + 0x02] = std::byte{0};
    image[romfs_header + 0x03] = std::byte{3};
    image[romfs_header + 0x04] = std::byte{4}; // crypt: BKTR
    {
        const std::size_t superblock = romfs_header + 0x08;
        put_u32(image, superblock + 0x00, 0x43465649); // "IVFC"
        put_u32(image, superblock + 0x04, 0x00010000);
        put_u32(image, superblock + 0x08, 0x20);
        put_u32(image, superblock + 0x0C, 7);
        for (std::size_t level = 0; level < 6; ++level) {
            const std::size_t descriptor = superblock + 0x10 + level * 0x18;
            put_u64(image, descriptor, patched_tree.level_offsets[level]);
            put_u64(image, descriptor + 0x08, patched_tree.level_sizes[level]);
            put_u32(image, descriptor + 0x10, kHashBlockLog2);
        }
        std::memcpy(image.data() + superblock + 0xC0, patched_tree.master_hash.data(), 0x20);
        // Relocation header: patch data then the relocation tree.
        put_u64(image, superblock + 0xF8, relocation_offset);
        put_u64(image, superblock + 0x100, kTreeSize);
        put_u32(image, superblock + 0x108, 0x52544B42); // "BKTR"
        put_u32(image, superblock + 0x110, 0);          // version
        put_u32(image, superblock + 0x114, 1);          // entry count (buckets)
        // Subsection header: the subsection tree fills to the section end.
        put_u64(image, superblock + 0x118, subsection_offset);
        put_u64(image, superblock + 0x120, section_size - subsection_offset);
        put_u32(image, superblock + 0x128, 0x52544B42);
        put_u32(image, superblock + 0x130, 0);
        put_u32(image, superblock + 0x134, 1);
    }
    const std::array<std::byte, 8> romfs_ctr = {std::byte{0x0F}, std::byte{0x1E},
                                                std::byte{0x2D}, std::byte{0x3C},
                                                std::byte{0x4B}, std::byte{0x5A},
                                                std::byte{0x69}, std::byte{0x78}};
    for (std::size_t i = 0; i < 8; ++i) {
        image[romfs_header + 0x140 + i] = romfs_ctr[i];
    }

    // Key area stays zero: titlekey content carries its key in the ticket.

    // Section encryption.
    {
        std::vector<std::byte> section(std::size_t(section_size), std::byte{0});
        std::memcpy(section.data(), patch.data(), patch.size());
        std::memcpy(section.data() + relocation_offset, relocation_tree.data(), kTreeSize);
        std::memcpy(section.data() + subsection_offset, subsection_tree.data(), kTreeSize);
        auto counter_at = [&](std::uint64_t offset, std::uint32_t ctr_val,
                              std::array<std::uint8_t, 16>& counter) {
            for (std::size_t i = 0; i < 4; ++i) {
                counter[i] = static_cast<std::uint8_t>(romfs_ctr[7 - i]);
            }
            counter[4] = static_cast<std::uint8_t>(ctr_val >> 24);
            counter[5] = static_cast<std::uint8_t>(ctr_val >> 16);
            counter[6] = static_cast<std::uint8_t>(ctr_val >> 8);
            counter[7] = static_cast<std::uint8_t>(ctr_val);
            const std::uint64_t block = (bktr_offset + offset) >> 4;
            for (std::size_t i = 0; i < 8; ++i) {
                counter[15 - i] = static_cast<std::uint8_t>(block >> (8 * i));
            }
        };
        auto apply = [&](std::uint64_t offset, std::size_t length,
                         const std::array<std::uint8_t, 16>& counter) {
            std::vector<std::byte> chunk(section.begin() + std::int64_t(offset),
                                         section.begin() + std::int64_t(offset + length));
            (void)fixture_aes_ctr_apply(built.title_key.data(), counter.data(), chunk);
            std::memcpy(section.data() + offset, chunk.data(), length);
        };
        // Patch data: per-block counters from the covering subsection.
        for (std::uint64_t offset = 0; offset < patch_end; offset += 16) {
            const std::uint32_t ctr_val =
                offset < kSubsectionSplit ? kSubsectionCtr1 : kSubsectionCtr2;
            std::array<std::uint8_t, 16> counter{};
            counter_at(offset, ctr_val, counter);
            apply(offset, 16, counter);
        }
        // Bucket trees: plain section CTR, as the reader decrypts them.
        for (std::uint64_t offset = relocation_offset; offset < section_size; offset += 16) {
            std::array<std::uint8_t, 16> counter{};
            for (std::size_t i = 0; i < 8; ++i) {
                counter[i] = static_cast<std::uint8_t>(romfs_ctr[7 - i]);
            }
            const std::uint64_t block = (bktr_offset + offset) >> 4;
            for (std::size_t i = 0; i < 8; ++i) {
                counter[15 - i] = static_cast<std::uint8_t>(block >> (8 * i));
            }
            const std::uint64_t take = std::min<std::uint64_t>(16, section_size - offset);
            apply(offset, std::size_t(take), counter);
        }
        std::memcpy(image.data() + bktr_offset, section.data(), std::size_t(section_size));
    }
    // ExeFS encryption: plain section CTR with the title key.
    {
        std::vector<std::byte> data(exefs_section.size(), std::byte{0});
        std::memcpy(data.data(), exefs_section.data(), exefs_section.size());
        for (std::uint64_t block = 0; block * 16 < data.size(); ++block) {
            std::array<std::uint8_t, 16> counter{};
            for (std::size_t i = 0; i < 8; ++i) {
                counter[i] = static_cast<std::uint8_t>(exefs_ctr[7 - i]);
            }
            const std::uint64_t block_number = (exefs_offset + block * 16) >> 4;
            for (std::size_t i = 0; i < 8; ++i) {
                counter[15 - i] = static_cast<std::uint8_t>(block_number >> (8 * i));
            }
            const std::size_t end =
                std::min<std::size_t>((block + 1) * 16, data.size());
            std::vector<std::byte> chunk(data.begin() + std::int64_t(block * 16),
                                         data.begin() + std::int64_t(end));
            (void)fixture_aes_ctr_apply(built.title_key.data(), counter.data(), chunk);
            std::memcpy(data.data() + block * 16, chunk.data(), chunk.size());
        }
        std::memcpy(image.data() + exefs_offset, data.data(), data.size());
    }

    // FS header hash table, signatures, header encryption.
    for (std::size_t index = 0; index < 4; ++index) {
        const std::array<std::byte, 32> digest = fixture_sha256(
            std::span<const std::byte>(image).subspan(0x400 + index * 0x200, 0x200));
        std::memcpy(image.data() + 0x280 + index * 0x20, digest.data(), 0x20);
    }
    std::array<std::uint8_t, 256> signature{};
    const std::span<const std::byte> message =
        std::span<const std::byte>(image).subspan(0x200, 0x200);
    (void)sign_key.sign(message, signature);
    for (std::size_t i = 0; i < 256; ++i) {
        image[i] = std::byte(signature[i]);
        image[0x100 + i] = std::byte(signature[i]);
    }
    std::vector<std::byte> header(image.begin(), image.begin() + 0xC00);
    (void)fixture_xts_encrypt(keys.header_key.data(), header, 0, 0x200);
    std::memcpy(image.data(), header.data(), 0xC00);
    built.image = std::move(image);
    return built;
}

// Wraps entries in a plaintext NSP (PFS0 at offset zero).
inline std::vector<std::byte> build_nsp(
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& entries) {
    return build_pfs0(entries);
}

// Wraps content in secure and optional update gamecard partitions.
inline std::vector<std::byte> build_xci_partitions(
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& secure_entries,
    const std::vector<std::pair<std::string, std::vector<std::byte>>>& update_entries = {}) {
    const std::vector<std::byte> secure = build_hfs0(secure_entries);
    std::vector<std::pair<std::string, std::vector<std::byte>>> root_entries = {
        {"secure", secure}};
    if (!update_entries.empty()) {
        root_entries.emplace_back("update", build_hfs0(update_entries));
    }
    std::vector<std::byte> gamecard(0x1000, std::byte{0});
    put_u32(gamecard, 0x100, 0x44414548); // "HEAD"
    const std::vector<std::byte> root = build_hfs0(root_entries);
    gamecard.resize(0x400 + root.size());
    put_u64(gamecard, 0x130, 0x400);
    std::memcpy(gamecard.data() + 0x400, root.data(), root.size());
    return gamecard;
}

inline std::vector<std::byte> build_xci(const std::string& nca_name,
                                        const std::vector<std::byte>& content) {
    return build_xci_partitions({{nca_name, content}});
}

// Writes bytes to a new file and returns true on success.
inline bool write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        return false;
    }
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    return written == bytes.size();
}

} // namespace container_test
