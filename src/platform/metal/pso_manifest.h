// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace armigo::platform::metal::nvn_driver {

// The pipeline warm-up manifest: one line per shader stage or pipeline the
// driver created, in the order it was first used. Fields are tab separated:
//   S <stage identity>                          a shader stage's Metal library
//   R <vertex identity> <fragment identity|-> <v1:state hex>   a render pipeline
//   M <task identity|-> <mesh identity> <fragment identity|-> <v1:state hex>
//     a native mesh render pipeline, with the same attachment/raster state
//     encoding as R and no vertex input layout
//     Mesh stage references append @<source digest> and #<native input id>
//     when known, identifying exact emitted and control/payload variants.
//   C <stage identity>                          a compute pipeline.
//     A packaged compute list is these rows, one per packaged compute stage.
//   K <vertex> <fragment> <color> <mask> <depth> <stencil> <samples>   a clear
//     pipeline: the clear functions, then the Metal pixel formats, color write mask
//     and sample count as decimal numbers
//   O <vertex identity> <fragment identity> <v1:state hex>   a render
//     pipeline's open form only, for pipeline archive builds (the driver's
//     warm-up skips it): stages a title may pair, not one it was seen to use
// Stage identities name a stage by its program, not by the package it was
// built into, so a manifest stays valid when the shader package is rebuilt.
// Lines starting with '#' are comments. Row order is the build order: never
// sort a manifest.
struct PsoRow {
    char kind = 0;
    std::vector<std::string> fields;

    [[nodiscard]] std::string text() const;
    bool operator==(const PsoRow&) const = default;
};

// A row for a line, or nothing for a comment, blank line or malformed row
// (unknown kind or wrong field count).
[[nodiscard]] std::optional<PsoRow> parse_pso_row(std::string_view line);
// Every valid row in file order; a repeated row is dropped at its later place.
[[nodiscard]] std::vector<PsoRow> parse_pso_manifest(std::string_view text);
// The seed's rows in order, then the user's rows the seed lacks, in order.
[[nodiscard]] std::vector<PsoRow> merge_pso_manifests(const std::vector<PsoRow>& seed,
                                                      const std::vector<PsoRow>& user);

// File name of the packaged compute list, under the shader package's native section.
inline constexpr std::string_view compute_pipeline_manifest_file = "compute-pipelines.tsv";

// One C row per identity, in the order given. An empty identity, one that
// cannot be a single field, and a later repeat are left out. The rows are
// not sorted.
[[nodiscard]] std::string compute_stage_manifest(const std::vector<std::string>& identities);

// The title list first, when it is set, then the packaged compute list when
// that file is present. An empty result is a pack with nothing to archive.
[[nodiscard]] std::vector<std::string> select_pipeline_manifests(std::string_view title_list,
                                                                std::string_view compute_list,
                                                                bool compute_list_exists);

[[nodiscard]] std::string pso_hex_encode(const void* bytes, std::size_t size);
// False unless `text` is exactly 2 * size lowercase hex digits.
[[nodiscard]] bool pso_hex_decode(std::string_view text, void* bytes, std::size_t size);

// Appends rows as they are first used. The record file receives every row once
// (rows already in it from an earlier run are not written again); the user
// file receives the rows missing from the manifests loaded at launch and from
// itself, so each session widens the next launch's warm-up. Writes are
// buffered and flushed about once a second (and when the file closes or the
// process exits), so a caller on a hot path pays for no system call per row.
class PsoRecorder {
public:
    // Either path may be empty. `known` is what the launch's manifests hold.
    PsoRecorder(const std::string& record_path, const std::string& user_path,
                const std::vector<PsoRow>& known);
    ~PsoRecorder();
    PsoRecorder(const PsoRecorder&) = delete;
    PsoRecorder& operator=(const PsoRecorder&) = delete;

    // Thread safe.
    void note(const PsoRow& row);
    [[nodiscard]] bool enabled() const noexcept { return record_.file || user_.file; }

private:
    struct Sink {
        std::FILE* file = nullptr;
        std::unordered_set<std::string> seen;
        std::chrono::steady_clock::time_point flushed = std::chrono::steady_clock::now();
    };
    static void open(Sink& sink, const std::string& path);
    static void append(Sink& sink, const std::string& line);

    std::mutex mutex_;
    Sink record_, user_;
};

// Render pipelines ("<vertex identity> <fragment identity>") an archive build
// compiled but could not archive, because Metal refused their open form. The
// build keeps them in this file of its archive directory, so a later warm-up
// that reads the archive can tell those misses from pipelines the archive
// should have held.
inline constexpr char unarchived_pipelines_file[] = "unarchived.tsv";
// Writes the keys (sorted, once each) into `directory`; false when the file
// cannot be written.
[[nodiscard]] bool write_unarchived_pipelines(const std::filesystem::path& directory,
                                              std::vector<std::string> keys);
// The keys of every archive directory under `cache_root`.
[[nodiscard]] std::unordered_set<std::string> read_unarchived_pipelines(const std::filesystem::path& cache_root);
// How many of a warm-up's missed keys the archive build recorded as unarchived.
[[nodiscard]] std::uint64_t explained_archive_misses(const std::vector<std::string>& missed,
                                                     const std::unordered_set<std::string>& unarchived);

} // namespace armigo::platform::metal::nvn_driver
