// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// NCA3 program-content reading. Written for Armigo from the NCA on-disk
// format. The signature checks follow the format's trust structure: the
// fixed-key header signature and the NPDM ACID signature must both verify
// before any content byte is extracted, and the NCA's second signature,
// made with the ACID modulus, binds the NPDM to this archive. Content
// verification continues the same chain: the header's FS header hash table
// authenticates each FS header, whose superblock master hash authenticates
// the section's hash tables, which authenticate the data blocks. Hash chain
// semantics (block sizes, padded versus unpadded final blocks, level
// descriptors) follow the on-disk format as documented by the hactool
// project; the C++ here is Armigo's own.

#include "container_nca.h"

#include "crypto.h"
#include "npdm.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <variant>
#include <vector>

namespace armigo::loader::container_detail {
namespace {

constexpr std::uint32_t kNca3Magic = 0x3341434E; // "NCA3" little-endian

// Fixed NCA3 header offsets, in bytes.
constexpr std::size_t kFixedSignatureOffset = 0x000;
constexpr std::size_t kNpdmSignatureOffset = 0x100;
constexpr std::size_t kSignedMessageOffset = 0x200;
constexpr std::size_t kSignedMessageSize = 0x200;
constexpr std::size_t kContentTypeOffset = 0x205;
constexpr std::size_t kCryptoTypeOffset = 0x206;
constexpr std::size_t kKeyAreaIndexOffset = 0x207;
constexpr std::size_t kCryptoType2Offset = 0x220;
constexpr std::size_t kFixedKeyGenerationOffset = 0x221;
constexpr std::size_t kTitleIdOffset = 0x210;
constexpr std::size_t kRightsIdOffset = 0x230;
constexpr std::size_t kSectionEntryBase = 0x240;
constexpr std::size_t kSectionHashesOffset = 0x280; // 4 x 0x20 SHA-256 over FS headers
constexpr std::size_t kEncryptedKeysOffset = 0x300;
constexpr std::size_t kFsHeaderBase = 0x400;
constexpr std::size_t kFsHeaderSize = 0x200;
constexpr std::size_t kFsHeaderCount = 4;

// Fixed FS-header offsets.
constexpr std::size_t kFsPartitionType = 0x02;
constexpr std::size_t kFsType = 0x03;
constexpr std::size_t kFsCryptType = 0x04;
constexpr std::size_t kFsSuperblock = 0x08;
constexpr std::size_t kFsSectionCtr = 0x140;

// PFS0 superblock fields, relative to the superblock: the master hash over
// the hash table, the hash block size in bytes, the level count, the hash
// table position and extent, and the hashed PFS0 partition.
constexpr std::size_t kSuperMasterHash = 0x00;
constexpr std::size_t kSuperBlockSize = 0x20;
constexpr std::size_t kSuperHashTableOffset = 0x28;
constexpr std::size_t kSuperHashTableSize = 0x30;
constexpr std::size_t kSuperPfs0Offset = 0x38;
constexpr std::size_t kSuperPfs0Size = 0x40;
// IVFC fields, relative to the superblock.
constexpr std::size_t kIvfcMagic = 0x00;
constexpr std::size_t kIvfcMasterHashSize = 0x08;
constexpr std::size_t kIvfcNumLevels = 0x0C;
constexpr std::size_t kIvfcLevelBase = 0x10;
constexpr std::size_t kIvfcLevelStride = 0x18;
constexpr std::size_t kIvfcMasterHash = 0xC0;
constexpr std::size_t kIvfcDataLevel = 5;

constexpr std::uint8_t kCryptNone = 1;
constexpr std::uint8_t kCryptCtr = 3;
constexpr std::uint8_t kCryptBktr = 4;

// BKTR superblock layout, relative to the superblock (the FS header's
// 0x08 byte): the RomFS IVFC header, 0x18 reserved bytes, then the
// relocation and subsection bucket-tree headers.
constexpr std::size_t kBktrRelocationHeaderOffset = 0x0F8;
constexpr std::size_t kBktrSubsectionHeaderOffset = 0x118;
constexpr std::uint32_t kBktrMagic = 0x52544B42; // "BKTR"
// Bucket-tree block: a 0x10-byte header followed by one u64 bucket offset
// per bucket, then fixed 0x4000-byte buckets. A relocation entry is
// 0x14 bytes (virtual, physical, patch flag); a subsection entry is 0x10
// bytes (physical offset, reserved, counter value), giving 0x3FF slots.
constexpr std::size_t kBktrBlockHeaderSize = 0x10;
constexpr std::size_t kBktrBucketSize = 0x4000;
constexpr std::size_t kBktrBucketHeaderSize = 0x10;
constexpr std::size_t kBktrRelocationEntrySize = 0x14;
constexpr std::size_t kBktrSubsectionEntrySize = 0x10;
constexpr std::size_t kBktrMaxBucketEntries = 0x3FF;
// Bound on one bucket-tree block before its allocation. Real update trees
// stay far below this; the bound only rejects implausible headers.
constexpr std::uint64_t kMaxBktrTreeBytes = std::uint64_t{64} << 20;

// Common ticket (v2): fixed fields used for titlekey resolution.
constexpr std::size_t kTicketMinSize = 0x2B0; // through the rights id
constexpr std::size_t kTicketSignatureTypeOffset = 0x000;
constexpr std::size_t kTicketTitleKeyBlockOffset = 0x180;
constexpr std::size_t kTicketRightsIdOffset = 0x2A0;
constexpr std::uint32_t kTicketRsa2048Sha256 = 0x10004;
constexpr std::size_t kTicketReadCap = 0x4000;

constexpr std::uint32_t kPfs0Magic = 0x30534650; // "PFS0"
constexpr std::uint32_t kAcidMagic = 0x44494341; // "ACID"
constexpr std::size_t kAcidHeaderSize = 0x240;
constexpr std::size_t kAcidSignatureSize = 0x100;
constexpr std::size_t kAcidModulusOffset = 0x100;

std::uint8_t read_u8_at(std::span<const std::byte> image, std::size_t offset) noexcept {
    return static_cast<std::uint8_t>(image[offset]);
}

bool read_exact(std::FILE* file, std::uint64_t offset, std::span<std::byte> out) noexcept {
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
        return false;
    }
    return std::fread(out.data(), 1, out.size(), file) == out.size();
}

// Builds a full 16-byte counter: the FS header's 8 CTR bytes (stored once
// reversed in the section) in the high half, and the 16-byte block number
// of `absolute_offset` big-endian in the low half.
std::array<std::uint8_t, 16> counter_for(const NcaSection& section,
                                         std::uint64_t absolute_offset) noexcept {
    std::array<std::uint8_t, 16> counter{};
    for (std::size_t i = 0; i < 8; ++i) {
        counter[i] = section.counter[i];
    }
    std::uint64_t block = absolute_offset >> 4;
    for (std::size_t i = 0; i < 8; ++i) {
        counter[15 - i] = static_cast<std::uint8_t>(block & 0xFF);
        block >>= 8;
    }
    return counter;
}

const char* key_area_key_prefix(std::uint8_t index) noexcept {
    switch (index) {
        case 0: return "key_area_key_application";
        case 1: return "key_area_key_ocean";
        case 2: return "key_area_key_system";
        default: return nullptr;
    }
}

// Key generation mapping shared by key-area keys and titlekeks: the stored
// generation counts from one, except that zero stays zero.
std::uint32_t key_index_for_generation(std::uint32_t generation) noexcept {
    return generation == 0 ? 0 : generation - 1;
}

// A generation-indexed key the keys file lacks: its keys predate the
// firmware or game that encrypted this content.
std::string newer_keys_needed(std::uint32_t key_generation) {
    char text[160];
    std::snprintf(text, sizeof(text),
                  " The keys file is older than this content (keys older than firmware): it needs key "
                  "generation %02x (master_key_%02x); supply keys dumped from firmware at least that new.",
                  key_generation, key_generation);
    return text;
}

// Encodes 16 bytes as 32 lowercase hex characters (used to name rights ids
// in diagnostics; never key values).
std::string hex_name(std::span<const std::byte> value) {
    static const char* digits = "0123456789abcdef";
    std::string text;
    for (std::byte b : value) {
        text.push_back(digits[static_cast<std::uint8_t>(b) >> 4]);
        text.push_back(digits[static_cast<std::uint8_t>(b) & 0xF]);
    }
    return text;
}

// AES-CTR counter for a BKTR patch range: the section's 4 leading counter
// bytes, the subsection's 32-bit counter value big-endian, then the 8-byte
// block number of the physical offset (NCA-relative, as for plain CTR).
std::array<std::uint8_t, 16> bktr_counter(const NcaSection& section, std::uint32_t ctr_val,
                                          std::uint64_t physical_offset) noexcept {
    std::array<std::uint8_t, 16> counter{};
    for (std::size_t i = 0; i < 4; ++i) {
        counter[i] = section.counter[i];
    }
    counter[4] = static_cast<std::uint8_t>(ctr_val >> 24);
    counter[5] = static_cast<std::uint8_t>(ctr_val >> 16);
    counter[6] = static_cast<std::uint8_t>(ctr_val >> 8);
    counter[7] = static_cast<std::uint8_t>(ctr_val);
    std::uint64_t block = (section.offset_in_nca + physical_offset) >> 4;
    for (std::size_t i = 0; i < 8; ++i) {
        counter[15 - i] = static_cast<std::uint8_t>(block & 0xFF);
        block >>= 8;
    }
    return counter;
}

// One relocation tree entry: where a virtual range lives.
struct BktrRelocation {
    std::uint64_t virt_offset = 0;
    std::uint64_t phys_offset = 0;
    std::uint32_t is_patch = 0;
};

// One subsection tree entry: which counter decrypts a physical range.
struct BktrSubsection {
    std::uint64_t offset = 0;
    std::uint32_t ctr_val = 0;
};

// Bucket addresses inside one decrypted tree block. The 0x10-byte header and
// the u64 bucket offset table fill the first 0x4000 bytes; bucket i starts at
// 0x4000 * (i + 1).
constexpr std::size_t kBktrBucketTableBegin = kBktrBlockHeaderSize;

std::size_t bktr_bucket_base(std::uint32_t num_buckets, std::uint32_t index) noexcept {
    static_cast<void>(num_buckets);
    return (std::size_t{index} + 1) * kBktrBucketSize;
}

// First u64 of a bucket's offset table window: the virtual or physical start
// of the range that bucket covers.
std::uint64_t bktr_bucket_start(std::span<const std::byte> block, std::uint32_t index) noexcept {
    return read_u64_le(block, kBktrBucketTableBegin + std::size_t{index} * 8);
}

// Finds the relocation entry covering `virtual_offset` and the virtual
// bound where its range ends: the next entry, the next bucket's start, or
// the image total for the tree's last entry.
[[nodiscard]] bool find_bktr_relocation(const NcaSection& section, std::uint64_t virtual_offset,
                                        BktrRelocation& out, std::uint64_t& range_end) noexcept {
    const std::span<const std::byte> block = section.relocation_block;
    if (virtual_offset >= section.patch_virtual_size || block.size() < kBktrBlockHeaderSize) {
        return false;
    }
    const std::uint32_t num_buckets = read_u32_le(block, 0x04);
    // Select the bucket whose virtual window contains the offset: the last
    // bucket offset at or below it (bucket 0 starts the image).
    std::uint32_t bucket = 0;
    for (std::uint32_t i = 1; i < num_buckets; ++i) {
        if (bktr_bucket_start(block, i) <= virtual_offset) {
            bucket = i;
        }
    }
    const std::size_t base = bktr_bucket_base(num_buckets, bucket);
    if (base + kBktrBucketSize > block.size()) {
        return false;
    }
    const std::uint32_t count = read_u32_le(block, base + 0x04);
    if (count == 0 || count > kBktrMaxBucketEntries) {
        return false;
    }
    // The last entry whose virtual start is at or below the offset.
    std::size_t found = count;
    for (std::size_t i = count; i-- > 0;) {
        if (read_u64_le(block, base + kBktrBucketHeaderSize + i * kBktrRelocationEntrySize) <=
            virtual_offset) {
            found = i;
            break;
        }
    }
    if (found == count) {
        return false;
    }
    const std::size_t entry = base + kBktrBucketHeaderSize + found * kBktrRelocationEntrySize;
    out.virt_offset = read_u64_le(block, entry);
    out.phys_offset = read_u64_le(block, entry + 0x08);
    out.is_patch = read_u32_le(block, entry + 0x10);
    if (found + 1 < count) {
        range_end = read_u64_le(block, base + kBktrBucketHeaderSize +
                                            (found + 1) * kBktrRelocationEntrySize);
    } else {
        range_end = bucket + 1 < num_buckets ? bktr_bucket_start(block, bucket + 1)
                                             : section.patch_virtual_size;
    }
    return range_end > virtual_offset;
}

// Finds the subsection entry covering `physical_offset` and the physical
// bound where its range ends: the next entry, the next bucket's first entry,
// or the patch data end for the tree's last entry.
[[nodiscard]] bool find_bktr_subsection(const NcaSection& section, std::uint64_t physical_offset,
                                        BktrSubsection& out, std::uint64_t& range_end) noexcept {
    const std::span<const std::byte> block = section.subsection_block;
    if (block.size() < kBktrBlockHeaderSize) {
        return false;
    }
    const std::uint32_t num_buckets = read_u32_le(block, 0x04);
    std::uint32_t bucket = 0;
    for (std::uint32_t i = 1; i < num_buckets; ++i) {
        if (bktr_bucket_start(block, i) <= physical_offset) {
            bucket = i;
        }
    }
    const std::size_t base = bktr_bucket_base(num_buckets, bucket);
    if (base + kBktrBucketSize > block.size()) {
        return false;
    }
    const std::uint32_t count = read_u32_le(block, base + 0x04);
    if (count == 0 || count > kBktrMaxBucketEntries) {
        return false;
    }
    std::size_t found = count;
    for (std::size_t i = count; i-- > 0;) {
        if (read_u64_le(block, base + kBktrBucketHeaderSize + i * kBktrSubsectionEntrySize) <=
            physical_offset) {
            found = i;
            break;
        }
    }
    if (found == count) {
        return false;
    }
    const std::size_t entry = base + kBktrBucketHeaderSize + found * kBktrSubsectionEntrySize;
    out.offset = read_u64_le(block, entry);
    out.ctr_val = read_u32_le(block, entry + 0x0C);
    if (found + 1 < count) {
        range_end = read_u64_le(block, base + kBktrBucketHeaderSize +
                                            (found + 1) * kBktrSubsectionEntrySize);
    } else if (bucket + 1 < num_buckets) {
        range_end = read_u64_le(block, bktr_bucket_base(num_buckets, bucket + 1) +
                                            kBktrBucketHeaderSize);
    } else {
        range_end = section.patch_data_end;
    }
    return range_end > physical_offset;
}

// Decrypts one patch range: an aligned CTR window over the update section
// keyed with the subsection's counter.
[[nodiscard]] bool read_patch_range(const SectionView& view, std::uint64_t physical_offset,
                                    std::uint32_t ctr_val, std::span<std::byte> out) noexcept {
    if (out.empty()) {
        return true;
    }
    const NcaSection& section = *view.section;
    if (physical_offset > section.size || out.size() > section.size - physical_offset) {
        return false;
    }
    const std::uint64_t position = section.offset_in_nca + physical_offset;
    const std::uint64_t aligned_position = position & ~std::uint64_t{0xF};
    const std::size_t prefix = static_cast<std::size_t>(position - aligned_position);
    const std::size_t window_length = (prefix + out.size() + 0xF) & ~std::size_t{0xF};
    std::vector<std::byte> window(window_length);
    if (!read_exact(view.file, section.offset + (aligned_position - section.offset_in_nca),
                    window)) {
        return false;
    }
    if (!aes_ctr_crypt(view.program->content_key,
                       bktr_counter(section, ctr_val, aligned_position - section.offset_in_nca),
                       window)) {
        return false;
    }
    std::memcpy(out.data(), window.data() + prefix, out.size());
    return true;
}

// Largest hash block the reader accepts. Legitimate images use 16 KiB
// (IVFC) or 64 KiB (PFS0) blocks; the bound only rejects implausible
// superblock values before their allocations are made.
constexpr std::uint64_t kMaxHashBlockSize = std::uint64_t{1} << 22;

enum class TableOutcome {
    // Every block digest matched its expected entry.
    verified,
    // A block digest differed; `bad_entry` holds the failing entry index.
    mismatch,
    // The hashed data was malformed or unreadable; `failure` is set.
    failure,
};

// Bytes occupied by the SHA-256 digest of every block in `data_len`.
// The entry count is a ceil-divide, then scaled by 0x20. Either step
// failing means the table cannot be addressed, so the caller rejects
// the section instead of wrapping the product to an empty read.
[[nodiscard]] bool hash_digest_bytes(std::uint64_t data_len, std::uint64_t block_size,
                                     std::uint64_t& digest_bytes) noexcept {
    if (block_size == 0 || data_len > ~std::uint64_t{0} - (block_size - 1)) {
        return false;
    }
    const std::uint64_t entries = (data_len + block_size - 1) / block_size;
    if (entries > ~std::uint64_t{0} / 0x20) {
        return false;
    }
    digest_bytes = entries * 0x20;
    return true;
}

// IVFC stores a block size as a log2. Values above 22 are not a block
// this reader will hash, and they are rejected before the shift.
[[nodiscard]] bool ivfc_block_size(std::uint32_t log2, std::uint64_t& block_size) noexcept {
    if (log2 > 22) {
        return false;
    }
    block_size = std::uint64_t{1} << log2;
    return true;
}

// Verifies `data_len` bytes at `data_offset` (section-relative) against
// `expected`, the consecutive 0x20-byte digests of its blocks, hashing
// blocks of `block_size` bytes. The final partial block is zero-padded to
// the block size when `full_block` is set, and hashes at its true length
// otherwise; the NCA format uses padded blocks in the IVFC tree and
// unpadded blocks for the PFS0 hash table.
[[nodiscard]] TableOutcome verify_block_digests(const SectionView& view,
                                                std::uint64_t data_offset,
                                                std::uint64_t data_len,
                                                std::uint64_t block_size, bool full_block,
                                                std::span<const std::byte> expected,
                                                std::uint64_t& bad_entry,
                                                ContainerFailure& failure) {
    const NcaSection& section = *view.section;
    if (block_size == 0 || block_size > kMaxHashBlockSize || data_len == 0) {
        failure = {ContainerError::section_invalid,
                   "The section's superblock names an unusable hash block size."};
        return TableOutcome::failure;
    }
    // A layered view reads the patched image, whose extent is the relocation
    // tree's virtual size, not the physical section.
    const std::uint64_t image_size =
        view.base_program ? section.patch_virtual_size : section.size;
    std::uint64_t digest_bytes = 0;
    if (!hash_digest_bytes(data_len, block_size, digest_bytes) ||
        expected.size() < digest_bytes || data_offset > image_size ||
        data_len > image_size - data_offset) {
        failure = {ContainerError::section_invalid,
                   "The section's hash table or hashed data leaves its section."};
        return TableOutcome::failure;
    }
    std::vector<std::byte> block(static_cast<std::size_t>(block_size));
    std::array<std::byte, 0x20> digest{};
    for (std::uint64_t entry = 0, done = 0; done < data_len; ++entry, done += block_size) {
        const std::uint64_t take = std::min(block_size, data_len - done);
        const std::size_t hashed = full_block ? static_cast<std::size_t>(block_size)
                                              : static_cast<std::size_t>(take);
        if (full_block) {
            std::fill(block.begin(), block.end(), std::byte{0});
        }
        if (!read_section_view(view, data_offset + done,
                               std::span<std::byte>(block.data(), static_cast<std::size_t>(take)))) {
            failure = {ContainerError::section_invalid, "Reading a hashed data block failed."};
            return TableOutcome::failure;
        }
        if (!sha256_digest(std::span<const std::byte>(block.data(), hashed), digest)) {
            failure = {ContainerError::section_invalid, "Hashing a data block failed."};
            return TableOutcome::failure;
        }
        if (!std::equal(digest.begin(), digest.end(),
                        expected.begin() + static_cast<std::ptrdiff_t>(entry * 0x20))) {
            bad_entry = entry;
            return TableOutcome::mismatch;
        }
    }
    return TableOutcome::verified;
}

// Reads `size` bytes at `offset` (section-relative) as an expected digest
// table for verify_block_digests.
[[nodiscard]] bool read_hash_table(const SectionView& view, std::uint64_t offset,
                                   std::uint64_t size, std::vector<std::byte>& table,
                                   ContainerFailure& failure) {
    const NcaSection& section = *view.section;
    // Layered views address the patched image; see verify_block_digests.
    const std::uint64_t image_size =
        view.base_program ? section.patch_virtual_size : section.size;
    if (offset > image_size || size > image_size - offset) {
        failure = {ContainerError::section_invalid,
                   "The section's hash table leaves its section."};
        return false;
    }
    table.resize(static_cast<std::size_t>(size));
    if (!read_section_view(view, offset, table)) {
        failure = {ContainerError::section_invalid, "Reading the section's hash table failed."};
        return false;
    }
    return true;
}

// Verifies one section's content chain from the FS header's master hash down
// through the hash tables to the data blocks. Extraction runs only after
// every present section verified, so each byte it reads is bound to the
// signed header region.
[[nodiscard]] bool verify_section_hash_chain(const SectionView& view,
                                             ContainerFailure& failure) {
    const NcaSection& section = *view.section;
    const NcaSuperblock& superblock = section.superblock;
    std::uint64_t bad_entry = 0;
    if (section.partition_type == kPartitionPfs0 && section.fs_type == kFsTypePfs0) {
        // The master hash covers the whole hash table as one unpadded block.
        const TableOutcome master = verify_block_digests(
            view, superblock.hash_table_offset, superblock.hash_table_size,
            superblock.hash_table_size, false, superblock.master_hash, bad_entry, failure);
        if (master == TableOutcome::failure) {
            return false;
        }
        if (master == TableOutcome::mismatch) {
            failure = {ContainerError::content_hash_invalid,
                       "The ExeFS section's hash table does not match the superblock master "
                       "hash."};
            return false;
        }
        std::uint64_t digest_bytes = 0;
        if (!hash_digest_bytes(superblock.partition_size, superblock.partition_block_size,
                               digest_bytes)) {
            failure = {ContainerError::section_invalid,
                       "The section's hash table or hashed data leaves its section."};
            return false;
        }
        std::vector<std::byte> table;
        if (!read_hash_table(view, superblock.hash_table_offset, digest_bytes, table, failure)) {
            return false;
        }
        const TableOutcome data = verify_block_digests(
            view, section.superblock_offset, superblock.partition_size,
            superblock.partition_block_size, false, table, bad_entry, failure);
        if (data == TableOutcome::failure) {
            return false;
        }
        if (data == TableOutcome::mismatch) {
            failure = {ContainerError::content_hash_invalid,
                       "The ExeFS section's hash table entry " + std::to_string(bad_entry) +
                           " does not match its PFS0 data block."};
            return false;
        }
        return true;
    }
    if (section.partition_type == kPartitionRomFs && section.fs_type == kFsTypeRomFs) {
        const NcaSuperblock::Level& first = superblock.ivfc_levels[0];
        std::uint64_t master_block = 0;
        if (!ivfc_block_size(first.block_size_log2, master_block)) {
            failure = {ContainerError::section_invalid,
                       "The section's superblock names an unusable hash block size."};
            return false;
        }
        const TableOutcome master =
            verify_block_digests(view, first.offset, first.size, master_block, true,
                                 superblock.master_hash, bad_entry, failure);
        if (master == TableOutcome::failure) {
            return false;
        }
        if (master == TableOutcome::mismatch) {
            failure = {ContainerError::content_hash_invalid,
                       "The RomFS section's IVFC level 1 table does not match the superblock "
                       "master hash."};
            return false;
        }
        for (std::size_t level = 1; level < superblock.ivfc_levels.size(); ++level) {
            const NcaSuperblock::Level& data = superblock.ivfc_levels[level];
            const NcaSuperblock::Level& table_level = superblock.ivfc_levels[level - 1];
            std::uint64_t block_size = 0;
            std::uint64_t digest_bytes = 0;
            // The parent level is the digest table. Its declared size bounds
            // the read, so a wrapped entry count cannot become an empty table
            // and a real count cannot read past the parent.
            if (!ivfc_block_size(data.block_size_log2, block_size) ||
                !hash_digest_bytes(data.size, block_size, digest_bytes) ||
                digest_bytes > table_level.size) {
                failure = {ContainerError::section_invalid,
                           "The section's hash table or hashed data leaves its section."};
                return false;
            }
            std::vector<std::byte> table;
            if (!read_hash_table(view, table_level.offset, digest_bytes, table, failure)) {
                return false;
            }
            const TableOutcome outcome =
                verify_block_digests(view, data.offset, data.size, block_size, true, table,
                                     bad_entry, failure);
            if (outcome == TableOutcome::failure) {
                return false;
            }
            if (outcome == TableOutcome::mismatch) {
                failure = {ContainerError::content_hash_invalid,
                           "The RomFS section's IVFC level " + std::to_string(level + 1) +
                               " does not match its hash table (entry " +
                               std::to_string(bad_entry) + ")."};
                return false;
            }
        }
        return true;
    }
    return true;
}

// Verifies the ACID signature inside main.npdm and the NCA signature that
// binds the NPDM to this archive. `acid_modulus` receives the ACID's RSA
// modulus for the binding check and is purely an intermediate value.
[[nodiscard]] bool verify_npdm_and_binding(const NcaProgram& nca, const NcaSection& exefs,
                                           std::FILE* file,
                                           const ContainerTrustAnchors& anchors,
                                           ContainerFailure& failure) {
    Pfs0Partition partition;
    if (!read_pfs0_partition(nca, exefs, file, exefs.superblock_offset, partition, failure)) {
        return false;
    }
    std::size_t npdm_index = partition.file_count;
    for (std::size_t i = 0; i < partition.file_count; ++i) {
        if (partition.entry_name(i) == "main.npdm") {
            npdm_index = i;
            break;
        }
    }
    if (npdm_index == partition.file_count) {
        failure = {ContainerError::npdm_invalid, "The ExeFS does not contain 'main.npdm'."};
        return false;
    }
    const std::uint64_t npdm_size = partition.entry_size(npdm_index);
    if (npdm_size == 0 || npdm_size > exefs.size || npdm_size > (1u << 24)) {
        failure = {ContainerError::npdm_invalid, "The 'main.npdm' size is out of range."};
        return false;
    }
    std::vector<std::byte> npdm(npdm_size);
    if (!read_nca_section(nca, exefs, file, partition.entry_file_offset(npdm_index), npdm)) {
        failure = {ContainerError::section_invalid, "Reading 'main.npdm' failed."};
        return false;
    }
    const auto parsed = parse_npdm(npdm);
    if (auto* failed = std::get_if<NpdmParseFailure>(&parsed)) {
        failure = {ContainerError::npdm_invalid,
                   "The ExeFS 'main.npdm' is not structurally valid: " + failed->detail};
        return false;
    }
    const ParsedNpdm& metadata = std::get<ParsedNpdm>(parsed);
    if (metadata.acid.size < kAcidHeaderSize || metadata.acid.offset > npdm.size() ||
        metadata.acid.size > npdm.size() - metadata.acid.offset) {
        failure = {ContainerError::npdm_invalid,
                   "The NPDM does not contain a complete ACID section."};
        return false;
    }
    const std::span<const std::byte> acid =
        std::span<const std::byte>(npdm).subspan(metadata.acid.offset, metadata.acid.size);
    if (read_u32_le(acid, 0x200) != kAcidMagic) {
        failure = {ContainerError::npdm_invalid, "The NPDM's ACID section has a bad magic."};
        return false;
    }
    // The signed region is the ACID from its modulus field to the length
    // its own header declares.
    const std::uint32_t signed_size = read_u32_le(acid, 0x204);
    if (signed_size != acid.size() - kAcidSignatureSize) {
        failure = {ContainerError::npdm_invalid,
                   "The NPDM's ACID section has an unexpected signed length."};
        return false;
    }
    const std::uint8_t sign_key_index = read_u8_at(npdm, 4);
    if (sign_key_index >= anchors.acid_moduli.size()) {
        failure = {ContainerError::unsupported_nca,
                   "The NPDM names an ACID signature key with no trust anchor."};
        return false;
    }
    std::array<std::uint8_t, 256> acid_modulus{};
    std::memcpy(acid_modulus.data(), acid.data() + kAcidModulusOffset, 256);
    const std::span<const std::uint8_t, 256> acid_signature(
        reinterpret_cast<const std::uint8_t*>(acid.data()), kAcidSignatureSize);
    const std::span<const std::byte> acid_message = acid.subspan(kAcidSignatureSize);
    if (!rsa2048_pss_sha256_verify(anchors.acid_moduli[sign_key_index], acid_message,
                                           acid_signature)) {
        failure = {ContainerError::signature_invalid,
                   "The NPDM ACID signature does not verify against the fixed public key."};
        return false;
    }

    // The NCA's second signature is made with the ACID modulus.
    const std::span<const std::uint8_t, 256> binding_signature(
        reinterpret_cast<const std::uint8_t*>(nca.header.data() + kNpdmSignatureOffset),
        kAcidSignatureSize);
    const std::span<const std::byte> header_message =
        std::span<const std::byte>(nca.header).subspan(kSignedMessageOffset, kSignedMessageSize);
    if (!rsa2048_pss_sha256_verify(acid_modulus, header_message, binding_signature)) {
        failure = {ContainerError::signature_invalid,
                   "The NCA signature over the NPDM (signature 2) does not verify."};
        return false;
    }
    return true;
}

// Resolves the content key of a rights-id (titlekey) NCA. The matching
// common ticket inside the same container wins; otherwise the user's
// rights-id table supplies the title key. The ticket's title key block
// decrypts with the titlekek named by the rights id's generation, using
// the same generation mapping as key-area keys. Key values never reach
// diagnostics: failures name the rights id or the key by name.
[[nodiscard]] bool resolve_titlekey(const ContainerKeys& keys, const TitleKeySource& source,
                                    std::span<const std::byte> rights_id,
                                    std::array<std::byte, 16>& content_key,
                                    ContainerFailure& failure) {
    const std::string rights_hex = hex_name(rights_id);
    if (source.tickets != nullptr && source.container_file != nullptr) {
        for (const ContainerTicketRef& ref : *source.tickets) {
            if (ref.size < kTicketMinSize || ref.size > kTicketReadCap) {
                continue; // cannot hold the fixed ticket fields
            }
            std::vector<std::byte> ticket(static_cast<std::size_t>(ref.size));
            if (!read_exact(source.container_file, ref.offset, ticket)) {
                failure = {ContainerError::ticket_invalid,
                           "Reading a ticket from the container failed."};
                return false;
            }
            const std::span<const std::byte> ticket_rights =
                std::span<const std::byte>(ticket).subspan(kTicketRightsIdOffset, 0x10);
            if (!std::equal(rights_id.begin(), rights_id.end(), ticket_rights.begin())) {
                continue;
            }
            if (read_u32_le(ticket, kTicketSignatureTypeOffset) != kTicketRsa2048Sha256) {
                failure = {ContainerError::ticket_invalid,
                           "The ticket matching rights id " + rights_hex +
                               " has an unsupported signature type."};
                return false;
            }
            const std::uint32_t generation =
                key_index_for_generation(static_cast<std::uint8_t>(rights_id[15]));
            char key_name[32];
            std::snprintf(key_name, sizeof(key_name), "titlekek_%02x", generation);
            const std::vector<std::byte>* titlekek = keys.find(key_name);
            if (!titlekek || titlekek->size() != 16) {
                failure = {ContainerError::key_missing,
                           std::string("The keys do not provide '") + key_name +
                               "' (16 bytes), required to decrypt the ticket title key." +
                               newer_keys_needed(generation)};
                return false;
            }
            const std::span<const std::byte> encrypted =
                std::span<const std::byte>(ticket).subspan(kTicketTitleKeyBlockOffset, 16);
            if (!aes_ecb_decrypt_block(*titlekek, encrypted, content_key)) {
                failure = {ContainerError::ticket_invalid,
                           "The ticket's title key block could not be decrypted."};
                return false;
            }
            return true;
        }
    }
    if (source.title_keys != nullptr) {
        const std::vector<std::byte>* title_key = source.title_keys->find(rights_hex);
        if (title_key) {
            if (title_key->size() != 16) {
                failure = {ContainerError::key_missing,
                           "The title key for rights id " + rights_hex + " must be 16 bytes."};
                return false;
            }
            std::memcpy(content_key.data(), title_key->data(), 16);
            return true;
        }
    }
    failure = {ContainerError::key_missing,
               "No ticket or title key was found for rights id " + rights_hex + "."};
    return false;
}

} // namespace

bool read_section_view(const SectionView& view, std::uint64_t offset_within_section,
                       std::span<std::byte> out) {
    if (!view.program || !view.section || !view.file) {
        return false;
    }
    if (!view.base_program || !view.base_section) {
        return read_nca_section(*view.program, *view.section, view.file, offset_within_section,
                                out);
    }
    // Walk the patched image: each output stretch is bounded by the
    // covering relocation entry and, for patch ranges, by the covering
    // subsection, so every stretch decrypts with one counter.
    const NcaSection& patch = *view.section;
    std::size_t done = 0;
    while (done < out.size()) {
        const std::uint64_t virtual_offset = offset_within_section + done;
        BktrRelocation relocation;
        std::uint64_t relocation_end = 0;
        if (!find_bktr_relocation(patch, virtual_offset, relocation, relocation_end)) {
            return false;
        }
        const std::uint64_t available = relocation_end - virtual_offset;
        if (relocation.virt_offset > virtual_offset) {
            return false;
        }
        const std::uint64_t delta = virtual_offset - relocation.virt_offset;
        // Physical start and the range length are both checked by
        // subtraction. Adding either one first can wrap a high physical
        // address to zero, which then passes the later bounds check.
        if (relocation.phys_offset > ~std::uint64_t{0} - delta ||
            available > ~std::uint64_t{0} - delta - relocation.phys_offset) {
            return false;
        }
        const std::uint64_t physical = relocation.phys_offset + delta;
        if (relocation.is_patch == 0) {
            const std::uint64_t take = std::min<std::uint64_t>(out.size() - done, available);
            if (!read_nca_section(*view.base_program, *view.base_section,
                                  view.base_file ? view.base_file : view.file, physical,
                                  out.subspan(done, std::size_t(take)))) {
                return false;
            }
            done += std::size_t(take);
            continue;
        }
        BktrSubsection subsection;
        std::uint64_t subsection_end = 0;
        if (!find_bktr_subsection(patch, physical, subsection, subsection_end)) {
            return false;
        }
        const std::uint64_t take = std::min<std::uint64_t>(
            {out.size() - done, available, subsection_end - physical});
        if (!read_patch_range(view, physical, subsection.ctr_val,
                              out.subspan(done, std::size_t(take)))) {
            return false;
        }
        done += std::size_t(take);
    }
    return true;
}

bool verify_section_view_chain(const SectionView& view, ContainerFailure& failure) {
    return verify_section_hash_chain(view, failure);
}

// A PFS0 partition located inside a section. Entries hold byte ranges
// relative to the partition's data area; reads go through read_nca_section.

std::uint64_t Pfs0Partition::entry_file_offset(std::size_t index) const noexcept {
    return partition_base + data_offset + read_u64_le(entries, index * 0x18);
}

std::uint64_t Pfs0Partition::entry_size(std::size_t index) const noexcept {
    return read_u64_le(entries, index * 0x18 + 8);
}

std::string_view Pfs0Partition::entry_name(std::size_t index) const noexcept {
    const std::uint32_t name_offset = read_u32_le(entries, index * 0x18 + 0x10);
    if (name_offset >= string_table.size()) {
        return {};
    }
    const auto* begin = reinterpret_cast<const char*>(string_table.data()) + name_offset;
    const auto* end = reinterpret_cast<const char*>(string_table.data()) + string_table.size();
    const auto* terminator = static_cast<const char*>(std::memchr(begin, '\0', end - begin));
    return std::string_view(begin, terminator ? static_cast<std::size_t>(terminator - begin)
                                              : static_cast<std::size_t>(end - begin));
}

bool read_pfs0_partition(const NcaProgram& nca, const NcaSection& section, std::FILE* file,
                         std::uint64_t partition_offset, Pfs0Partition& partition,
                         ContainerFailure& failure) {
    std::array<std::byte, 0x10> header{};
    if (!read_nca_section(nca, section, file, partition_offset, header)) {
        failure = {ContainerError::section_invalid, "Reading the PFS0 partition header failed."};
        return false;
    }
    if (read_u32_le(header, 0) != kPfs0Magic) {
        failure = {ContainerError::section_invalid, "The partition is not PFS0."};
        return false;
    }
    partition.partition_base = partition_offset;
    partition.file_count = read_u32_le(header, 4);
    const std::uint32_t string_table_size = read_u32_le(header, 8);
    if (partition.file_count == 0 || partition.file_count > 0x1000 ||
        string_table_size > 0x100000) {
        failure = {ContainerError::section_invalid, "The PFS0 partition header is malformed."};
        return false;
    }
    partition.entries.resize(std::size_t{0x18} * partition.file_count);
    if (!read_nca_section(nca, section, file, partition_offset + 0x10, partition.entries)) {
        failure = {ContainerError::section_invalid, "Reading the PFS0 entries failed."};
        return false;
    }
    partition.string_table.resize(string_table_size);
    if (!read_nca_section(nca, section, file, partition_offset + 0x10 + partition.entries.size(),
                          partition.string_table)) {
        failure = {ContainerError::section_invalid, "Reading the PFS0 names failed."};
        return false;
    }
    partition.data_offset =
        0x10 + partition.entries.size() + partition.string_table.size();
    // Entries must stay inside the hashed partition extent, not merely the
    // section, so every byte extraction reads is covered by the verified
    // hash table.
    const std::uint64_t hashed_end = section.superblock.partition_size;
    for (std::size_t i = 0; i < partition.file_count; ++i) {
        const std::uint64_t relative = partition.entry_file_offset(i) - partition.partition_base;
        const std::uint64_t size = partition.entry_size(i);
        if (relative > hashed_end || size > hashed_end - relative) {
            failure = {ContainerError::section_invalid,
                       "A PFS0 entry extends past the hashed partition."};
            return false;
        }
    }
    return true;
}

std::uint32_t read_u32_le(std::span<const std::byte> image, std::size_t offset) noexcept {
    if (offset + 4 > image.size()) {
        return 0;
    }
    return static_cast<std::uint32_t>(static_cast<std::uint8_t>(image[offset])) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(image[offset + 1])) << 8) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(image[offset + 2])) << 16) |
           (static_cast<std::uint32_t>(static_cast<std::uint8_t>(image[offset + 3])) << 24);
}

std::uint64_t read_u64_le(std::span<const std::byte> image, std::size_t offset) noexcept {
    if (offset + 8 > image.size()) {
        return 0;
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(image[offset + i]))
                 << (8 * i);
    }
    return value;
}

bool read_nca_section(const NcaProgram& nca, const NcaSection& section, std::FILE* file,
                      std::uint64_t offset_within_section, std::span<std::byte> out) noexcept {
    if (out.empty()) {
        return true;
    }
    if (offset_within_section > section.size || out.size() > section.size - offset_within_section) {
        return false;
    }
    if (section.crypt_type == kCryptNone) {
        return read_exact(file, section.offset + offset_within_section, out);
    }
    // CTR counters count 16-byte blocks from the NCA image start, so the
    // alignment arithmetic runs in NCA-relative coordinates and the disk
    // read maps back through the section's absolute offset.
    const std::uint64_t position = section.offset_in_nca + offset_within_section;
    const std::uint64_t aligned_position = position & ~std::uint64_t{0xF};
    const std::size_t prefix = static_cast<std::size_t>(position - aligned_position);
    const std::size_t aligned_length = (prefix + out.size() + 0xF) & ~std::size_t{0xF};
    const std::uint64_t aligned_absolute =
        section.offset + (aligned_position - section.offset_in_nca);
    std::vector<std::byte> window(aligned_length);
    if (!read_exact(file, aligned_absolute, window)) {
        return false;
    }
    if (!aes_ctr_crypt(nca.content_key, counter_for(section, aligned_position), window)) {
        return false;
    }
    std::memcpy(out.data(), window.data() + prefix, out.size());
    return true;
}

CandidateOutcome open_nca_candidate(std::FILE* file, std::uint64_t base,
                                    const ContainerKeys& keys,
                                    const TitleKeySource& title_keys,
                                    const ContainerTrustAnchors& anchors,
                                    int wanted_content_type, std::uint64_t wanted_title_id,
                                    bool patch_sections_allowed, bool require_nca,
                                    NcaProgram& program, ContainerFailure& failure) {
    if (!read_exact(file, base, program.header)) {
        return CandidateOutcome::skip;
    }
    const std::span<const std::byte> image = program.header;

    if (read_u32_le(image, kSignedMessageOffset) != kNca3Magic) {
        const std::vector<std::byte>* header_key = keys.find("header_key");
        if (!header_key || header_key->size() != 32) {
            failure = {ContainerError::key_missing,
                       "The keys do not provide 'header_key' (32 bytes), required for NCA "
                       "header decryption."};
            return CandidateOutcome::failure;
        }
        program.required_key_names.push_back("header_key");
        std::array<std::byte, 32> xts_key{};
        std::memcpy(xts_key.data(), header_key->data(), 32);
        if (!aes_xts_decrypt(xts_key, program.header, 0, 0x200)) {
            for (volatile std::byte& value : xts_key) {
                value = std::byte{0};
            }
            failure = {ContainerError::section_invalid,
                       "NCA header decryption failed; 'header_key' may be wrong."};
            return CandidateOutcome::failure;
        }
        for (volatile std::byte& value : xts_key) {
            value = std::byte{0};
        }
        if (read_u32_le(image, kSignedMessageOffset) != kNca3Magic) {
            if (require_nca) {
                failure = {ContainerError::unsupported_nca,
                           "The NCA header magic check failed after decryption; 'header_key' is "
                           "missing or incorrect."};
                return CandidateOutcome::failure;
            }
            return CandidateOutcome::skip;
        }
    }

    program.content_type = read_u8_at(image, kContentTypeOffset);
    if (wanted_content_type >= 0 && program.content_type != wanted_content_type) {
        return CandidateOutcome::skip;
    }
    program.title_id = read_u64_le(image, kTitleIdOffset);
    // With a wanted title id (the base program's), a different title in the
    // same container is not a match and not an admission failure.
    if (wanted_title_id != 0 && program.title_id != wanted_title_id) {
        return CandidateOutcome::skip;
    }

    // From here the candidate has the selected content type and title, so
    // every failure is structured rather than a silent skip.

    // Program extraction retains fixed-key and NPDM signature admission.
    // Metadata classification instead relies on the FS and section hash
    // chains below, which also cover repacked meta/control NCAs.
    if (program.content_type == 0) {
        const std::size_t generation = read_u8_at(image, kFixedKeyGenerationOffset);
        if (generation >= anchors.nca_header_moduli.size()) {
            failure = {ContainerError::unsupported_nca,
                       "The NCA fixed-key generation has no trust anchor."};
            return CandidateOutcome::failure;
        }
        const std::span<const std::uint8_t, 256> signature(
            reinterpret_cast<const std::uint8_t*>(image.data() + kFixedSignatureOffset),
            kAcidSignatureSize);
        const std::span<const std::byte> message =
            image.subspan(kSignedMessageOffset, kSignedMessageSize);
        if (!rsa2048_pss_sha256_verify(anchors.nca_header_moduli[generation], message,
                                       signature)) {
            failure = {ContainerError::signature_invalid,
                       "The NCA header signature does not verify against the fixed public key."};
            return CandidateOutcome::failure;
        }
    }

    // The FS header hash table binds each 0x200-byte FS header into the
    // header region; the superblocks and their master hashes live inside
    // those headers and anchor every section's content chain.
    for (std::size_t index = 0; index < kFsHeaderCount; ++index) {
        // Unused section slots have an empty section entry and no hash.
        const std::size_t entry = kSectionEntryBase + index * 0x10;
        if (read_u32_le(image, entry) == 0 && read_u32_le(image, entry + 4) == 0) {
            continue;
        }
        std::array<std::byte, 0x20> digest{};
        if (!sha256_digest(image.subspan(kFsHeaderBase + index * kFsHeaderSize, kFsHeaderSize),
                           digest) ||
            !std::equal(digest.begin(), digest.end(),
                        image.begin() + kSectionHashesOffset + index * 0x20)) {
            failure = {ContainerError::content_hash_invalid,
                       "The NCA's FS header hash for FS header " + std::to_string(index) +
                           " does not match its header."};
            return CandidateOutcome::failure;
        }
    }

    // Titlekey (rights-id) content resolves its content key from a matching
    // ticket or the user's rights-id table; key-area content decrypts its
    // key area. The two paths are exclusive.
    const std::span<const std::byte> rights_id = image.subspan(kRightsIdOffset, 0x10);
    const bool has_rights_id = std::any_of(rights_id.begin(), rights_id.end(),
                                           [](std::byte b) { return b != std::byte{0}; });
    if (has_rights_id) {
        if (!resolve_titlekey(keys, title_keys, rights_id, program.content_key, failure)) {
            return CandidateOutcome::failure;
        }
        const std::uint32_t generation =
            key_index_for_generation(static_cast<std::uint8_t>(rights_id[15]));
        char key_name[32];
        std::snprintf(key_name, sizeof(key_name), "titlekek_%02x", generation);
        program.required_key_names.push_back(key_name);
    } else {
        // The key-area key is selected by key index and key generation; the
        // first two master key generations share the generation-0 key.
        const std::uint8_t area_index = read_u8_at(image, kKeyAreaIndexOffset);
        const char* area_prefix = key_area_key_prefix(area_index);
        if (!area_prefix) {
            failure = {ContainerError::unsupported_nca,
                       "The NCA names an unsupported key-area index."};
            return CandidateOutcome::failure;
        }
        const std::uint32_t master_generation =
            std::max<std::uint32_t>(read_u8_at(image, kCryptoTypeOffset),
                                    read_u8_at(image, kCryptoType2Offset));
        const std::uint32_t key_generation = key_index_for_generation(master_generation);
        char key_name[80];
        std::snprintf(key_name, sizeof(key_name), "%s_%02x", area_prefix, key_generation);
        const std::vector<std::byte>* area_key = keys.find(key_name);
        if (!area_key || area_key->size() != 16) {
            failure = {ContainerError::key_missing,
                       std::string("The keys do not provide '") + key_name +
                           "' (16 bytes), required for NCA key-area decryption." +
                           newer_keys_needed(key_generation)};
            return CandidateOutcome::failure;
        }
        program.required_key_names.push_back(key_name);

        // The key area decrypts beside the header, never inside it: the signed
        // header region includes the still-encrypted key area, so mutating the
        // header would break the later binding-signature check. The content
        // key is the third slot.
        std::array<std::byte, 16> area_key_bytes{};
        std::memcpy(area_key_bytes.data(), area_key->data(), 16);
        for (std::size_t slot = 0; slot < 4; ++slot) {
            const std::span<const std::byte> encrypted =
                image.subspan(kEncryptedKeysOffset + slot * 16, 16);
            std::array<std::byte, 16> decrypted{};
            if (slot == 2) {
                if (!aes_ecb_decrypt_block(area_key_bytes, encrypted, program.content_key)) {
                    failure = {ContainerError::section_invalid, "NCA key-area decryption failed."};
                    return CandidateOutcome::failure;
                }
                continue;
            }
            if (!aes_ecb_decrypt_block(area_key_bytes, encrypted, decrypted)) {
                failure = {ContainerError::section_invalid, "NCA key-area decryption failed."};
                return CandidateOutcome::failure;
            }
            for (volatile std::byte& value : decrypted) {
                value = std::byte{0};
            }
        }
        for (volatile std::byte& value : area_key_bytes) {
            value = std::byte{0};
        }
    }

    // Sections: a present entry has a nonzero start sector.
    for (std::size_t index = 0; index < 4; ++index) {
        const std::size_t entry = kSectionEntryBase + index * 0x10;
        const std::uint64_t start_sectors = read_u32_le(image, entry);
        const std::uint64_t end_sectors = read_u32_le(image, entry + 4);
        if (start_sectors == 0) {
            continue;
        }
        NcaSection section;
        section.offset = base + (start_sectors << 9);
        section.offset_in_nca = start_sectors << 9;
        section.size = (end_sectors << 9) - (start_sectors << 9);
        const std::size_t fs_header_offset = kFsHeaderBase + index * kFsHeaderSize;
        section.partition_type = read_u8_at(image, fs_header_offset + kFsPartitionType);
        section.fs_type = read_u8_at(image, fs_header_offset + kFsType);
        section.crypt_type = read_u8_at(image, fs_header_offset + kFsCryptType);
        for (std::size_t i = 0; i < 8; ++i) {
            section.counter[i] = static_cast<std::uint8_t>(image[fs_header_offset + kFsSectionCtr + 7 - i]);
        }
        if (section.crypt_type != kCryptNone && section.crypt_type != kCryptCtr &&
            section.crypt_type != kCryptBktr) {
            failure = {ContainerError::unsupported_nca,
                       "The NCA section uses an unsupported encryption type; only plaintext, "
                       "AES-CTR and BKTR sections are supported."};
            return CandidateOutcome::failure;
        }
        if (section.crypt_type == kCryptBktr &&
            (!patch_sections_allowed || section.partition_type != kPartitionRomFs ||
             section.fs_type != kFsTypeRomFs)) {
            failure = {ContainerError::unsupported_nca,
                       "The NCA has a BKTR section outside an update's RomFS; a base "
                       "container is required to layer it."};
            return CandidateOutcome::failure;
        }
        if (end_sectors <= start_sectors) {
            failure = {ContainerError::section_invalid,
                       "The NCA section entry has an empty or reversed range."};
            return CandidateOutcome::failure;
        }
        const std::span<const std::byte> superblock =
            image.subspan(fs_header_offset + kFsSuperblock, 0x138);
        NcaSuperblock& parsed_superblock = section.superblock;
        if (section.partition_type == kPartitionPfs0 && section.fs_type == kFsTypePfs0) {
            std::memcpy(parsed_superblock.master_hash.data(),
                        superblock.data() + kSuperMasterHash, 0x20);
            // The block size is a 32-bit byte count followed by the
            // two-level count, so it reads as u32; a u64 read would pull
            // the level count into the value.
            parsed_superblock.partition_block_size = read_u32_le(superblock, kSuperBlockSize);
            parsed_superblock.hash_table_offset = read_u64_le(superblock, kSuperHashTableOffset);
            parsed_superblock.hash_table_size = read_u64_le(superblock, kSuperHashTableSize);
            section.superblock_offset = read_u64_le(superblock, kSuperPfs0Offset);
            parsed_superblock.partition_size = read_u64_le(superblock, kSuperPfs0Size);
            if (section.superblock_offset >= section.size) {
                failure = {ContainerError::section_invalid,
                           "The ExeFS partition offset leaves its section."};
                return CandidateOutcome::failure;
            }
            if (parsed_superblock.partition_block_size == 0 ||
                parsed_superblock.partition_block_size > kMaxHashBlockSize ||
                (parsed_superblock.partition_block_size &
                 (parsed_superblock.partition_block_size - 1)) != 0) {
                failure = {ContainerError::section_invalid,
                           "The ExeFS superblock names an unusable hash block size."};
                return CandidateOutcome::failure;
            }
            if (parsed_superblock.partition_size == 0 ||
                parsed_superblock.partition_size >
                    section.size - section.superblock_offset) {
                failure = {ContainerError::section_invalid,
                           "The ExeFS partition extent leaves its section."};
                return CandidateOutcome::failure;
            }
            std::uint64_t digest_bytes = 0;
            if (!hash_digest_bytes(parsed_superblock.partition_size,
                                   parsed_superblock.partition_block_size, digest_bytes) ||
                parsed_superblock.hash_table_size < digest_bytes ||
                parsed_superblock.hash_table_offset > section.size ||
                parsed_superblock.hash_table_size >
                    section.size - parsed_superblock.hash_table_offset) {
                failure = {ContainerError::section_invalid,
                           "The ExeFS hash table extent is malformed."};
                return CandidateOutcome::failure;
            }
        } else if (section.partition_type == kPartitionRomFs && section.fs_type == kFsTypeRomFs) {
            if (read_u32_le(superblock, kIvfcMagic) != 0x43465649) { // "IVFC"
                failure = {ContainerError::section_invalid,
                           "The RomFS section's IVFC superblock has a bad magic."};
                return CandidateOutcome::failure;
            }
            // The count includes the master hash level: six stored levels.
            if (read_u32_le(superblock, kIvfcNumLevels) != 7) {
                failure = {ContainerError::section_invalid,
                           "The RomFS section's IVFC superblock does not have six levels."};
                return CandidateOutcome::failure;
            }
            if (read_u32_le(superblock, kIvfcMasterHashSize) != 0x20) {
                failure = {ContainerError::section_invalid,
                           "The RomFS section's IVFC superblock has an unexpected master hash "
                           "size."};
                return CandidateOutcome::failure;
            }
            std::memcpy(parsed_superblock.master_hash.data(),
                        superblock.data() + kIvfcMasterHash, 0x20);
            for (std::size_t level = 0; level < parsed_superblock.ivfc_levels.size(); ++level) {
                const std::size_t base = kIvfcLevelBase + level * kIvfcLevelStride;
                NcaSuperblock::Level& descriptor = parsed_superblock.ivfc_levels[level];
                descriptor.offset = read_u64_le(superblock, base);
                descriptor.size = read_u64_le(superblock, base + 0x08);
                descriptor.block_size_log2 = read_u32_le(superblock, base + 0x10);
                // A BKTR section's levels address the patched image, which can be
                // larger than the stored section; they are bounded against the
                // relocation tree's image size once it is read.
                const bool patched = section.crypt_type == kCryptBktr;
                if (descriptor.block_size_log2 > 22 || descriptor.size == 0 ||
                    (!patched && (descriptor.offset > section.size ||
                                  descriptor.size > section.size - descriptor.offset))) {
                    failure = {ContainerError::section_invalid,
                               "The RomFS section's IVFC level " + std::to_string(level + 1) +
                                   " descriptor leaves its section."};
                    return CandidateOutcome::failure;
                }
            }
            section.romfs_offset = parsed_superblock.ivfc_levels[kIvfcDataLevel].offset;
            parsed_superblock.romfs_size = parsed_superblock.ivfc_levels[kIvfcDataLevel].size;
            // A BKTR data offset addresses the patched image, which can be
            // larger than the stored section. The relocation tree bounds it.
            if (section.crypt_type != kCryptBktr && section.romfs_offset >= section.size) {
                failure = {ContainerError::section_invalid,
                           "The RomFS data offset leaves its section."};
                return CandidateOutcome::failure;
            }
            if (section.crypt_type == kCryptBktr) {
                // The superblock carries the relocation and subsection
                // bucket-tree headers after the IVFC header; the trees
                // describe the patched image the IVFC chain covers.
                const std::size_t relocation_header = kBktrRelocationHeaderOffset;
                const std::size_t subsection_header = kBktrSubsectionHeaderOffset;
                struct BktrTreeHeader {
                    std::uint64_t offset = 0;
                    std::uint64_t size = 0;
                };
                auto read_tree_header = [&](std::size_t at, BktrTreeHeader& out) -> bool {
                    if (read_u32_le(superblock, at + 0x10) != kBktrMagic) {
                        return false;
                    }
                    out.offset = read_u64_le(superblock, at);
                    out.size = read_u64_le(superblock, at + 0x08);
                    return true;
                };
                BktrTreeHeader relocation;
                BktrTreeHeader subsection;
                if (!read_tree_header(relocation_header, relocation) ||
                    !read_tree_header(subsection_header, subsection)) {
                    failure = {ContainerError::section_invalid,
                               "The BKTR section's bucket-tree headers have a bad magic."};
                    return CandidateOutcome::failure;
                }
                // The trees follow the patch data back to back inside the
                // section; the section may end with alignment padding.
                if (relocation.offset > section.size ||
                    relocation.size > section.size - relocation.offset ||
                    subsection.offset > section.size ||
                    subsection.size > section.size - subsection.offset ||
                    relocation.offset + relocation.size != subsection.offset) {
                    failure = {ContainerError::section_invalid,
                               "The BKTR section's bucket-tree layout is malformed."};
                    return CandidateOutcome::failure;
                }
                if (relocation.size == 0 || relocation.size > kMaxBktrTreeBytes ||
                    subsection.size == 0 || subsection.size > kMaxBktrTreeBytes) {
                    failure = {ContainerError::section_invalid,
                               "The BKTR section's bucket trees have an unusable size."};
                    return CandidateOutcome::failure;
                }
            }
        } else {
            failure = {ContainerError::section_invalid,
                       "The NCA section has an unsupported file-system type."};
            return CandidateOutcome::failure;
        }
        program.sections.push_back(section);
    }

    // Content chains verify before any metadata or file is read, so a
    // corrupted section reports its hash failure rather than a downstream
    // signature or structure error. A BKTR section's chain runs over the
    // patched image and needs its base layer, so its caller verifies it
    // through verify_section_view_chain before extracting.
    for (const NcaSection& section : program.sections) {
        if (section.is_bktr()) {
            continue;
        }
        const SectionView view{&program, &section, nullptr, nullptr, file};
        if (!verify_section_hash_chain(view, failure)) {
            return CandidateOutcome::failure;
        }
    }

    // BKTR bucket trees load after the plain chains verify: a wrong content
    // key then surfaces as its section's hash failure instead of a tree
    // structure error, matching the chain-first admission order.
    {
        std::size_t section_index = 0;
        for (std::size_t index = 0; index < 4; ++index) {
            const std::size_t entry = kSectionEntryBase + index * 0x10;
            if (read_u32_le(image, entry) == 0) {
                continue;
            }
            NcaSection& section = program.sections.at(section_index++);
            if (!section.is_bktr()) {
                continue;
            }
            const std::size_t fs_header_offset = kFsHeaderBase + index * kFsHeaderSize;
            const std::span<const std::byte> superblock =
                image.subspan(fs_header_offset + kFsSuperblock, 0x138);
            struct BktrTreeExtent {
                std::uint64_t offset = 0;
                std::uint64_t size = 0;
            };
            auto tree_extent = [&](std::size_t at) -> BktrTreeExtent {
                return {read_u64_le(superblock, at), read_u64_le(superblock, at + 0x08)};
            };
            const BktrTreeExtent relocation = tree_extent(kBktrRelocationHeaderOffset);
            const BktrTreeExtent subsection = tree_extent(kBktrSubsectionHeaderOffset);
            // The trees sit in the encrypted section; plain section CTR
            // decrypts them before the subsection mapping exists.
            // AES-CTR-EX subsections cover the patch data, which ends where the
            // relocation tree begins.
            section.patch_data_end = relocation.offset;
            section.relocation_block.resize(static_cast<std::size_t>(relocation.size));
            if (!read_nca_section(program, section, file, relocation.offset,
                                  section.relocation_block)) {
                failure = {ContainerError::section_invalid,
                           "Reading the BKTR relocation tree failed."};
                return CandidateOutcome::failure;
            }
            section.subsection_block.resize(static_cast<std::size_t>(subsection.size));
            if (!read_nca_section(program, section, file, subsection.offset,
                                  section.subsection_block)) {
                failure = {ContainerError::section_invalid,
                           "Reading the BKTR subsection tree failed."};
                return CandidateOutcome::failure;
            }
            // Structure checks before any read consults the trees.
            auto validate_tree = [&](const std::vector<std::byte>& block) -> bool {
                if (block.size() < kBktrBlockHeaderSize) {
                    return false;
                }
                const std::uint32_t num_buckets = read_u32_le(block, 0x04);
                const std::uint64_t total = read_u64_le(block, 0x08);
                if (num_buckets == 0 || total == 0) {
                    return false;
                }
                const std::size_t table_bytes =
                    kBktrBucketTableBegin + std::size_t{num_buckets} * 8;
                if (table_bytes > kBktrBucketSize ||
                    block.size() / kBktrBucketSize < std::size_t{num_buckets} + 1) {
                    return false;
                }
                // Bucket windows partition the address space from zero.
                if (read_u64_le(block, kBktrBucketTableBegin) != 0) {
                    return false;
                }
                for (std::uint32_t i = 1; i < num_buckets; ++i) {
                    if (read_u64_le(block, kBktrBucketTableBegin + std::size_t{i} * 8) <
                        read_u64_le(block, kBktrBucketTableBegin + std::size_t{i - 1} * 8)) {
                        return false;
                    }
                }
                return true;
            };
            if (!validate_tree(section.relocation_block) ||
                !validate_tree(section.subsection_block)) {
                failure = {ContainerError::section_invalid,
                           "The BKTR section's bucket trees are malformed."};
                return CandidateOutcome::failure;
            }
            const std::uint64_t subsection_total =
                read_u64_le(section.subsection_block, 0x08);
            if (subsection_total != section.patch_data_end &&
                subsection_total != subsection.offset) {
                failure = {ContainerError::section_invalid,
                           "The BKTR section's bucket trees are malformed."};
                return CandidateOutcome::failure;
            }
            section.patch_virtual_size = read_u64_le(section.relocation_block, 0x08);
            // Every IVFC level lives inside the patched image the
            // relocation tree describes.
            const NcaSuperblock& parsed_superblock = section.superblock;
            for (std::size_t level = 0; level < parsed_superblock.ivfc_levels.size(); ++level) {
                const NcaSuperblock::Level& descriptor = parsed_superblock.ivfc_levels[level];
                if (descriptor.offset > section.patch_virtual_size ||
                    descriptor.size > section.patch_virtual_size - descriptor.offset) {
                    failure = {ContainerError::section_invalid,
                               "The BKTR section's IVFC level " + std::to_string(level + 1) +
                                   " leaves the patched image."};
                    return CandidateOutcome::failure;
                }
            }
        }
    }

    if (program.content_type != 0) {
        return CandidateOutcome::verified;
    }

    // The ACID check reads main.npdm through the ExeFS section, so it runs
    // after the section table is ready.
    const NcaSection* exefs = nullptr;
    for (const NcaSection& section : program.sections) {
        if (section.partition_type == kPartitionPfs0 && section.fs_type == kFsTypePfs0) {
            exefs = &section;
            break;
        }
    }
    if (!exefs) {
        failure = {ContainerError::npdm_invalid, "The program NCA has no ExeFS (PFS0) section."};
        return CandidateOutcome::failure;
    }
    if (!verify_npdm_and_binding(program, *exefs, file, anchors, failure)) {
        return CandidateOutcome::failure;
    }
    return CandidateOutcome::verified;
}

CandidateOutcome open_program_nca_candidate(std::FILE* file, std::uint64_t base,
                                            const ContainerKeys& keys,
                                            const TitleKeySource& title_keys,
                                            const ContainerTrustAnchors& anchors,
                                            std::uint64_t wanted_title_id,
                                            bool patch_sections_allowed,
                                            NcaProgram& program, ContainerFailure& failure) {
    return open_nca_candidate(file, base, keys, title_keys, anchors, 0, wanted_title_id,
                              patch_sections_allowed, false, program, failure);
}

} // namespace armigo::loader::container_detail
