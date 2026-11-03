// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// NCA3 content reading: header decryption (AES-XTS with the header key),
// program signature verification, key-area decryption and AES-CTR section
// access. Written for Armigo from the NCA on-disk format; offsets follow the
// NCA3 header layout. Internal to the container reader.

#pragma once

#include <armigo/loader/container.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::loader::container_detail {

// Section-type discriminators shared by the NCA reader and the container
// orchestration.
inline constexpr std::uint8_t kPartitionRomFs = 0;
inline constexpr std::uint8_t kPartitionPfs0 = 1;
inline constexpr std::uint8_t kFsTypePfs0 = 2;
inline constexpr std::uint8_t kFsTypeRomFs = 3;

// Little-endian scalar readers over a byte image. Offsets are fixed by the
// format; callers keep them inside the image.
[[nodiscard]] std::uint32_t read_u32_le(std::span<const std::byte> image,
                                        std::size_t offset) noexcept;
[[nodiscard]] std::uint64_t read_u64_le(std::span<const std::byte> image,
                                        std::size_t offset) noexcept;

// Superblock hash parameters carried by a verified FS header. They are read
// from the decrypted header only after the header's own entry in the NCA
// header's FS header hash table verified, so they anchor the section's
// content chain to the signed header region.
struct NcaSuperblock {
    // SHA-256 over the outermost hash table: the PFS0 hash table, or the
    // RomFS IVFC level-1 table.
    std::array<std::byte, 0x20> master_hash{};
    // PFS0 (hierarchical SHA-256): the hash table's section-relative
    // position and extent, the hashed PFS0 extent that follows it, and the
    // hash block size in bytes. The final data block hashes at its true
    // length, without padding.
    std::uint64_t hash_table_offset = 0;
    std::uint64_t hash_table_size = 0;
    std::uint64_t partition_size = 0;
    std::uint64_t partition_block_size = 0;
    // RomFS (IVFC): one descriptor per level, 1..6; level 6 describes the
    // RomFS image itself. Blocks hash zero-padded at `1 << block_size_log2`.
    struct Level {
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        std::uint32_t block_size_log2 = 0;
    };
    std::array<Level, 6> ivfc_levels{};
    // The hashed extent of the level-6 image; extraction reads stay inside
    // it so every byte read is covered by the verified chain.
    std::uint64_t romfs_size = 0;
};

// One decrypted NCA section. `offset` is the absolute file offset of the
// section's first byte; `offset_in_nca` is the section's position within
// the NCA image, which drives the CTR counter because NCA section counters
// count blocks from the NCA start, not from the enclosing container.
struct NcaSection {
    std::uint64_t offset = 0;
    std::uint64_t offset_in_nca = 0;
    std::uint64_t size = 0;
    std::uint8_t partition_type = 0; // 0 RomFS superblock, 1 PFS0 superblock
    std::uint8_t fs_type = 0;        // 2 PFS0, 3 RomFS
    std::uint8_t crypt_type = 0;     // 1 none, 2 XTS, 3 CTR, 4 BKTR
    std::uint64_t superblock_offset = 0; // PFS0: partition offset within the section.
    std::uint64_t romfs_offset = 0;      // RomFS: IVFC level-5 logical offset.
    std::array<std::uint8_t, 16> counter{}; // CTR counter at the section start.
    NcaSuperblock superblock; // hash parameters the FS header authenticates
    // BKTR (patch RomFS) bucket trees, present only for crypt type 4. The
    // trees are read and held decrypted; the section's IVFC extents are
    // virtual offsets into the patched image they describe.
    std::vector<std::byte> relocation_block;
    std::vector<std::byte> subsection_block;
    std::uint64_t patch_virtual_size = 0; // relocation tree's total image size
    std::uint64_t patch_data_end = 0;     // physical end of the BKTR patch data

    [[nodiscard]] bool is_bktr() const noexcept { return crypt_type == 4; }
};

// A decrypted NCA3 image whose section hash chains verified. Program images
// also pass signature admission. Sections may still be absent; the caller
// selects the ExeFS and RomFS sections it needs.
struct NcaProgram {
    std::array<std::byte, 0xC00> header{};
    std::vector<NcaSection> sections;
    std::array<std::byte, 16> content_key{};
    std::uint64_t title_id = 0;
    std::uint8_t content_type = 0;
    std::vector<std::string> required_key_names;

    void clear_sensitive() noexcept {
        for (volatile std::byte& value : content_key) {
            value = std::byte{0};
        }
    }
};

// A base/patch read layer for one section's bytes. When `base_program` is
// set, `program`/`section` are the update's BKTR RomFS section and reads go
// through its relocation and subsection trees: patch ranges decrypt from
// the update section with the per-subsection counter, the rest comes from
// the base section. A null `base_program` reads the section directly.
struct SectionView {
    const NcaProgram* program = nullptr;
    const NcaSection* section = nullptr;
    const NcaProgram* base_program = nullptr;
    const NcaSection* base_section = nullptr;
    std::FILE* file = nullptr;      // container file of `section`
    std::FILE* base_file = nullptr; // container file of `base_section`
};

// Reads `out` from `view` at `offset_within_section`; BKTR views map
// virtual offsets through the patch layer, plain views read directly.
[[nodiscard]] bool read_section_view(const SectionView& view, std::uint64_t offset_within_section,
                                     std::span<std::byte> out);

// Verifies one section's content hash chain through `view`. A candidate
// open verifies plain sections itself; a caller layers a BKTR section over
// its base image first and then verifies the patched chain with this.
[[nodiscard]] bool verify_section_view_chain(const SectionView& view, ContainerFailure& failure);

// A PFS0 partition located inside an NCA section; entries hold byte ranges
// relative to the partition's data area, read through read_nca_section.
struct Pfs0Partition {
    std::uint64_t partition_base = 0; // section-relative PFS0 header position
    std::uint64_t data_offset = 0;    // section-relative first data byte
    std::uint32_t file_count = 0;
    std::vector<std::byte> entries; // 0x18 bytes per entry
    std::vector<std::byte> string_table;

    [[nodiscard]] std::uint64_t entry_file_offset(std::size_t index) const noexcept;
    [[nodiscard]] std::uint64_t entry_size(std::size_t index) const noexcept;
    [[nodiscard]] std::string_view entry_name(std::size_t index) const noexcept;
};

// Reads the PFS0 partition table at `partition_offset` within `section`.
[[nodiscard]] bool read_pfs0_partition(const NcaProgram& nca, const NcaSection& section,
                                       std::FILE* file, std::uint64_t partition_offset,
                                       Pfs0Partition& partition, ContainerFailure& failure);

// A ticket file found in a container partition, read only when a rights id
// asks for it.
struct ContainerTicketRef {
    std::uint64_t offset = 0; // absolute container offset
    std::uint64_t size = 0;
};

// Titlekey evidence for rights-id content: the ticket files of the same
// container plus the user's optional rights-id table. Tickets are read
// from `container_file` on demand.
struct TitleKeySource {
    std::FILE* container_file = nullptr;
    const std::vector<ContainerTicketRef>* tickets = nullptr;
    const ContainerKeys* title_keys = nullptr;
};

// Outcome of one candidate probe while searching a container.
enum class CandidateOutcome {
    skip,
    verified,
    failure,
};

// Opens any NCA content type. `wanted_content_type` of -1 accepts every type;
// `require_nca` turns a bad post-decryption magic into an error that names the
// header key instead of skipping a non-NCA container entry.
[[nodiscard]] CandidateOutcome open_nca_candidate(std::FILE* file, std::uint64_t base,
                                                  const ContainerKeys& keys,
                                                  const TitleKeySource& title_keys,
                                                  const ContainerTrustAnchors& anchors,
                                                  int wanted_content_type,
                                                  std::uint64_t wanted_title_id,
                                                  bool patch_sections_allowed, bool require_nca,
                                                  NcaProgram& program, ContainerFailure& failure);

// Program-only extraction entry point, including NPDM and binding signatures.
[[nodiscard]] CandidateOutcome open_program_nca_candidate(std::FILE* file,
                                                          std::uint64_t base,
                                                          const ContainerKeys& keys,
                                                          const TitleKeySource& title_keys,
                                                          const ContainerTrustAnchors& anchors,
                                                          std::uint64_t wanted_title_id,
                                                          bool patch_sections_allowed,
                                                          NcaProgram& program,
                                                          ContainerFailure& failure);

// Decrypts `out` from a CTR or plaintext section at `offset_within_section`.
[[nodiscard]] bool read_nca_section(const NcaProgram& nca, const NcaSection& section,
                                    std::FILE* file, std::uint64_t offset_within_section,
                                    std::span<std::byte> out) noexcept;

} // namespace armigo::loader::container_detail
