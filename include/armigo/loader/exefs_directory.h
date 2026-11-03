// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// ExeFS directory collector for a user-supplied module directory.
//
// Collects the admitted files of a host directory laid out like an ExeFS
// into owned memory for the future public runner. The collector only
// transports bytes: it does not parse NPDM fields, does not derive
// permissions or a process configuration, does not decide module placement
// or SDK startup order, and does not validate NSO containers. The subsdk
// filename sequence follows the original executable roster admission rule.
//
// The caller must keep the input tree unchanged during the call. The
// collection is not an atomic snapshot of the whole directory and does not
// protect against adversarial modification; each candidate is checked to
// have kept the length observed when it was sized, and nothing else.

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace armigo::loader {

/// One admitted file and its complete contents.
struct ExeFsFile {
    /// Exact admitted directory-entry name, for example "main".
    std::string name;
    /// The file's bytes at collection time. Not interpreted here.
    std::vector<std::byte> bytes;
};

/// The collected contents of an ExeFS directory.
struct ExeFsDirectoryInput {
    /// Admitted module files, sorted byte-wise by name. This is a
    /// deterministic directory order, not a load or initialization order;
    /// module loading decides its own order.
    std::vector<ExeFsFile> modules;
    /// Raw bytes of "main.npdm" when the directory supplies it; absent
    /// otherwise. Returned unparsed; consumers own all interpretation.
    std::optional<std::vector<std::byte>> process_metadata;
    // Later subsdk names ignored after the first missing numbered slot.
    std::vector<std::string> ignored_modules;
};

/// Resource bounds applied before any candidate is read or allocated.
struct ExeFsReadLimits {
    /// Largest single admitted file.
    std::size_t max_file_bytes = std::size_t{256} * 1024 * 1024;
    /// Largest combined size of all admitted files, including the
    /// process metadata.
    std::size_t max_total_bytes = std::size_t{512} * 1024 * 1024;
    // Explicit synthetic inputs may use numbered names outside the original roster.
    bool collect_all_subsdk = false;
};

/// Structured failure reported by `read_exefs_directory`.
enum class ExeFsDirectoryError : std::uint8_t {
    /// The path could not be listed as a directory: it is missing, not a
    /// directory, or cannot be read. `detail` holds the reason.
    directory_unavailable,
    /// The required "main" module is not present.
    main_module_missing,
    /// An admitted name exists but is a symbolic link, a directory, or
    /// another non-regular file. `detail` names the entry.
    candidate_not_regular,
    /// An admitted candidate could not be examined. `detail` names the
    /// entry and the reason.
    candidate_stat_failed,
    /// A candidate exceeds `ExeFsReadLimits::max_file_bytes`. `detail`
    /// names the entry and both sizes.
    candidate_too_large,
    /// The candidates together exceed `ExeFsReadLimits::max_total_bytes`.
    /// `detail` names the entry that crossed the limit.
    total_too_large,
    /// A candidate's length changed between sizing and reading. `detail`
    /// names the entry and both lengths.
    size_changed,
    /// A candidate could not be opened or read. `detail` names the entry
    /// and the reason.
    read_failed,
    /// Host memory for the collected bytes could not be allocated.
    allocation_failed,
};

/// English diagnostic text for an `ExeFsDirectoryError` value. Never
/// returns null.
const char* to_string(ExeFsDirectoryError error) noexcept;

/// A failure code with an owned English detail string.
struct ExeFsDirectoryFailure {
    ExeFsDirectoryError code = ExeFsDirectoryError::directory_unavailable;
    std::string detail;
};

/// Outcome of `read_exefs_directory`: the collected input on success, a
/// failure otherwise. Nothing is returned partially collected.
using ExeFsDirectoryResult = std::variant<ExeFsDirectoryInput, ExeFsDirectoryFailure>;

/// Collects the admitted files of an ExeFS-shaped directory.
///
/// Admits exactly these directory-entry names, matched case-sensitively:
/// "main" (required), "rtld", "sdk", "subsdk0" through "subsdk9" (each
/// optional), and "main.npdm" (optional process metadata). Subsdk collection
/// stops at the first missing numbered slot; later subsdk entries are reported
/// in ignored_modules without being opened or charged against read limits. Every other
/// entry is ignored without being read. Only regular files are admitted;
/// symbolic links, directories, and other file kinds are rejected.
///
/// The directory is only read; nothing in it is written. The result owns
/// its bytes. The caller must keep the tree unchanged for the duration of
/// the call.
ExeFsDirectoryResult read_exefs_directory(const std::filesystem::path& directory,
                                          const ExeFsReadLimits& limits = {});

} // namespace armigo::loader
