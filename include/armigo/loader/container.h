// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Program-container input for user-supplied NSP, XCI and standalone NCA files.
//
// Extracts a program NCA's ExeFS and RomFS into a caller-owned output
// directory and verifies container integrity with the user's own keys
// before any byte is written: the NCA header RSA-2048-PSS signature against
// the fixed public modulus, the NPDM ACID signature against the fixed ACID
// modulus, the NCA signature that binds the NPDM to its NCA, the NCA
// header's FS header hash table, and each section's content hash chain from
// the FS header's superblock master hash through the hash tables to the
// data blocks. Extraction succeeds only when every verification passes;
// there is no "accept unverified" path here.
//
// Implements the required PFS0, HFS0, NCA, and IVFC on-disk format checks.
//
// Key material is used in place and never included in diagnostics: failure
// details name a missing key by name only, never by value.

#pragma once

#include <armigo/loader/exefs_directory.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::loader {

// ---------------------------------------------------------------------------
// Keys file (prod.keys text format)
// ---------------------------------------------------------------------------

// Decoded keys from a user-supplied keys file. Values are held but never
// logged; `detail` strings in this interface name keys, never key bytes.
class ContainerKeys {
public:
    ContainerKeys() = default;
    ContainerKeys(const ContainerKeys&) = default;
    ContainerKeys& operator=(const ContainerKeys& other);
    ContainerKeys(ContainerKeys&&) noexcept = default;
    ContainerKeys& operator=(ContainerKeys&& other) noexcept;
    ~ContainerKeys();

    // Moves in one decoded entry. Duplicate names keep the first value and
    // are reported by the parser.
    void add(std::string name, std::vector<std::byte> value);
    void clear() noexcept;

    // The decoded bytes for `name`, or null when absent.
    [[nodiscard]] const std::vector<std::byte>* find(std::string_view name) const;

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

private:
    std::vector<std::pair<std::string, std::vector<std::byte>>> entries_;
};

struct ContainerKeysFailure {
    enum class Code : std::uint8_t {
        // The keys path could not be read as a file.
        file_unavailable,
        // The file is not valid prod.keys text. `detail` has the line number.
        malformed_entry,
        // A hex value has an odd length or non-hex digits. `detail` names the key.
        bad_hex_value,
        // A key name appears more than once. `detail` names the key.
        duplicate_key,
    };
    Code code = Code::file_unavailable;
    std::string detail;
};

// Parses prod.keys-format text: lines of `name = hex`, with blank lines and
// `#`/`;` comments. Decoding is the parser's job so raw key text never
// outlives this call.
[[nodiscard]] std::variant<ContainerKeysFailure, ContainerKeys> parse_container_keys(
    std::span<const std::byte> text);
[[nodiscard]] std::variant<ContainerKeysFailure, ContainerKeys> read_container_keys_file(
    const std::filesystem::path& keys_file);

// ---------------------------------------------------------------------------
// Trust anchors
// ---------------------------------------------------------------------------

// RSA-2048 public moduli accepted for container signatures. Production uses
// the fixed well-known public constants; tests inject their own moduli so a
// test key pair can sign synthetic containers. Private keys never appear in
// this interface.
struct ContainerTrustAnchors {
    // Index is the NCA header "fixed key generation" field.
    std::vector<std::array<std::uint8_t, 256>> nca_header_moduli;
    // Index is the NPDM "ACID signature key index" field.
    std::vector<std::array<std::uint8_t, 256>> acid_moduli;

    // The fixed public verification constants used by production.
    [[nodiscard]] static ContainerTrustAnchors production();
};

// ---------------------------------------------------------------------------
// Container extraction
// ---------------------------------------------------------------------------

struct ContainerExtractionOptions {
    // Created when missing. The extraction writes only below this directory.
    std::filesystem::path output_directory;
    ContainerKeys keys;
    ContainerTrustAnchors anchors = ContainerTrustAnchors::production();
    // Title keys for rights-id (titlekey) content, keyed by the 32-digit
    // lowercase hexadecimal rights id. Consulted when the container has no
    // matching ticket. Same text format as the keys file.
    ContainerKeys title_keys;
    // Optional update container. When set, the extracted program content is
    // layered: the update's program NCA (matched to the base by title id)
    // supplies the ExeFS, and its BKTR-patched RomFS is applied over the
    // base program NCA's RomFS. Both containers verify fully; the base
    // NCA's own admission is unchanged.
    std::filesystem::path update_container_file; // empty: no update layer
    // Bounds for the extracted ExeFS files, per file and in total. RomFS
    // files stream straight to disk without accumulation and are not
    // capped: real RomFS images exceed the ExeFS collection limits.
    ExeFsReadLimits limits;
};

struct ContainerExtraction {
    std::filesystem::path exefs_directory;
    std::filesystem::path romfs_directory;
    // True when an update container supplied the active program content.
    bool update_applied = false;
    // The program title id the extraction selected; both containers carried
    // it when an update was applied.
    std::uint64_t title_id = 0;
    // Title versions are deliberately absent: NCA headers carry no title
    // version, and CNMT (meta NCA) parsing is out of scope for this reader.
};

enum class ContainerError : std::uint8_t {
    // The container path could not be opened or read.
    container_unavailable,
    // A required key is absent from the supplied keys. `detail` names it.
    key_missing,
    // The file is neither an NSP (PFS0) nor an XCI (HFS0 gamecard).
    unknown_container_format,
    // A partition table or its bounds are malformed. `detail` says which.
    partition_invalid,
    // No program NCA was found in the container. With an update container,
    // `detail` also covers "none matching the base title id".
    program_nca_missing,
    // The NCA uses a format outside this reader: NCA0/NCA2, or an
    // unsupported key generation. `detail` names the reason.
    unsupported_nca,
    // A ticket for titlekey content is malformed or has an unsupported
    // signature type. `detail` says which.
    ticket_invalid,
    // An NCA header or ACID signature does not verify.
    signature_invalid,
    // The ExeFS has no main.npdm, or it is not a structurally valid NPDM.
    npdm_invalid,
    // A section's decrypted content is malformed. `detail` says which.
    section_invalid,
    // The host refused a file operation during extraction. `detail` has it.
    io_failure,
    // Host memory for extraction could not be allocated.
    allocation_failed,
    // A section's content hash chain does not verify. `detail` names the
    // section, the failing level and the failing hash table entry.
    content_hash_invalid,
};

struct ContainerFailure {
    ContainerError code = ContainerError::container_unavailable;
    std::string detail;
};

const char* to_string(ContainerError error) noexcept;

// Extracts the program NCA of an NSP or XCI into `options.output_directory`
// as an `exefs` and a `romfs` subdirectory, verifying the container's
// signatures and content hash chains first. Returns the extraction only
// when the NCA header signature, the NPDM ACID signature, the NCA-to-NPDM
// binding signature, the FS header hash table, and every section's hash
// chain (superblock master hash, hash tables, data blocks) verify; a failed
// verification is a structured failure, never an unverified extraction.
// The container file is only read; the output directory is only written
// below the supplied path.
//
// With `options.update_container_file` set, the same admission runs on both
// containers. The update's program NCA must carry the base program NCA's
// title id; rights-id content resolves its title key from a matching ticket
// inside the same container or from `options.title_keys`. The ExeFS comes
// from the update, and the RomFS is the update section's BKTR-patched image
// over the base section, whose IVFC chain the reader verifies over the
// patched bytes exactly as it verifies an unpatched RomFS.
[[nodiscard]] std::variant<ContainerFailure, ContainerExtraction> extract_program_container(
    const std::filesystem::path& container_file, const ContainerExtractionOptions& options);

// A data or public-data NCA's RomFS, written as the tree itself (no extra
// romfs directory) below options.output_directory.
struct DataRomFsExtraction {
    std::filesystem::path romfs_directory;
    std::uint64_t title_id = 0;
    std::uint8_t content_type = 0;
};

// Extracts the RomFS of the add-on data NCA whose title id is `title_id`.
// Public data (content type 5) is tried before data (content type 4); the
// content type is checked before the title id, so a program NCA in the same
// container is not opened. The NCA's signatures and section hash chains are
// verified. `options.update_container_file` is ignored.
[[nodiscard]] std::variant<ContainerFailure, DataRomFsExtraction> extract_data_romfs(
    const std::filesystem::path& container_file, const ContainerExtractionOptions& options,
    std::uint64_t title_id);

} // namespace armigo::loader
