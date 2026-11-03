// ============================================================================
// Armigo Shader - Stage Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/shader_translation/stage_translation.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::shader::translation {

// A cached stage as read back: its kind (from the reflection), reflection and
// program; the Metal source stays in its file until a pipeline needs it.
struct CachedStage {
    shader_input::Stage stage = shader_input::Stage::Vertex;
    std::string reflection;
    std::vector<std::byte> program;
    std::filesystem::path source_file;
};

// Translated stages kept on disk between runs. Each entry is a directory
// holding the files a package stage holds besides its Metal library
// (source.metal, reflection.json, program.bin), under
// ROOT/<translator source version>/<key>. A key digests the stage's control
// and code bytes and its translation options, so an entry is valid exactly for
// the translator version that wrote it, whatever title or run asks again.
class StageCache {
public:
    explicit StageCache(std::filesystem::path root);

    // The entry key of a stage: a SHA-256 over the options text, the control
    // bytes and the code bytes, as 64 lowercase hexadecimal digits.
    [[nodiscard]] static std::string key(std::span<const std::byte> control,
                                         std::span<const std::byte> code,
                                         std::string_view options);

    // The entry's reflection and program and its source file, or nothing when
    // it is absent, incomplete, larger than a stage can be or names no stage.
    [[nodiscard]] std::optional<CachedStage> load(std::string_view key) const;
    // The keys of the entries present.
    [[nodiscard]] std::vector<std::string> keys() const;
    // Writes an entry beside its final place and renames it there, so a reader
    // never sees a partial entry; an entry already present is kept. False when
    // the entry could not be written.
    bool store(std::string_view key, const TranslatedStage& stage) const;
    // Removes what no run can use any more: the directories of other translator
    // versions beside this one, and staging directories older than
    // `partial_age` (left by a writer that stopped mid-store).
    void prune(std::chrono::seconds partial_age) const;

    // ROOT/<translator source version>.
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }

private:
    std::filesystem::path directory_;
};

} // namespace armigo::shader::translation
