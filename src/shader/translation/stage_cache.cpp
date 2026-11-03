// ============================================================================
// Armigo Shader - Stage Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_translation/stage_cache.h>

#include <armigo/shader/source_version.h>
#include "sha256.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <system_error>
#include <vector>

namespace armigo::shader::translation {
namespace {
namespace fs = std::filesystem;

// Bounds of one entry's files: an entry larger than any stage is not one.
constexpr std::uintmax_t source_limit = 16u << 20;
constexpr std::uintmax_t reflection_limit = 4u << 20;
constexpr std::uintmax_t program_limit = 4u << 20;

std::optional<std::vector<std::byte>> read_file(const fs::path& path, std::uintmax_t limit) {
    std::error_code error;
    if (!fs::is_regular_file(path, error)) return std::nullopt;
    const auto size = fs::file_size(path, error);
    if (error || size > limit) return std::nullopt;
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    std::ifstream file(path, std::ios::binary);
    if (!file || (!bytes.empty() && !file.read(reinterpret_cast<char*>(bytes.data()),
                                               static_cast<std::streamsize>(bytes.size()))))
        return std::nullopt;
    return bytes;
}

bool write_file(const fs::path& path, std::span<const std::byte> bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(reinterpret_cast<const char*>(bytes.data()),
                             static_cast<std::streamsize>(bytes.size())))
        return false;
    file.close();
    return static_cast<bool>(file);
}

std::span<const std::byte> text(std::string_view value) {
    return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

bool valid_key(std::string_view key) {
    if (key.size() != 64) return false;
    for (const char c : key)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}
} // namespace

StageCache::StageCache(fs::path root)
    : directory_(std::move(root) / std::string(armigo::shader::source_version)) {}

std::string StageCache::key(std::span<const std::byte> control, std::span<const std::byte> code,
                            std::string_view options) {
    // Each part is preceded by its length, so no two inputs share a digest input.
    std::vector<std::byte> input;
    input.reserve(24 + options.size() + control.size() + code.size());
    const auto append = [&](std::span<const std::byte> part) {
        const auto size = static_cast<std::uint64_t>(part.size());
        for (unsigned i = 0; i != 8; ++i) input.push_back(static_cast<std::byte>((size >> (8 * i)) & 255));
        input.insert(input.end(), part.begin(), part.end());
    };
    append(text(options));
    append(control);
    append(code);
    const auto hash = armigo::loader::detail::sha256(input);
    if (!hash) return {};
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : *hash) {
        const auto value = std::to_integer<unsigned>(byte);
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}

std::optional<CachedStage> StageCache::load(std::string_view key) const {
    if (!valid_key(key)) return std::nullopt;
    const auto entry = directory_ / std::string(key);
    std::error_code error;
    const auto source_size = fs::file_size(entry / "source.metal", error);
    if (error || !fs::is_regular_file(entry / "source.metal", error) || source_size > source_limit)
        return std::nullopt;
    const auto reflection = read_file(entry / "reflection.json", reflection_limit);
    auto program = read_file(entry / "program.bin", program_limit);
    if (!reflection || !program) return std::nullopt;
    CachedStage result;
    result.reflection.assign(reinterpret_cast<const char*>(reflection->data()), reflection->size());
    // The translator writes the stage kind as the reflection's "stage" field.
    if (result.reflection.find("\"stage\":\"vertex\"") != std::string::npos)
        result.stage = shader_input::Stage::Vertex;
    else if (result.reflection.find("\"stage\":\"fragment\"") != std::string::npos)
        result.stage = shader_input::Stage::Fragment;
    else if (result.reflection.find("\"stage\":\"compute\"") != std::string::npos)
        result.stage = shader_input::Stage::Compute;
    else
        return std::nullopt;
    result.program = std::move(*program);
    result.source_file = entry / "source.metal";
    return result;
}

std::vector<std::string> StageCache::keys() const {
    std::vector<std::string> result;
    std::error_code error;
    for (fs::directory_iterator it(directory_, error), end; !error && it != end; it.increment(error)) {
        auto name = it->path().filename().string();
        if (valid_key(name)) result.push_back(std::move(name));
    }
    return result;
}

void StageCache::prune(std::chrono::seconds partial_age) const {
    std::error_code error;
    const auto current = directory_.filename().string();
    for (fs::directory_iterator it(directory_.parent_path(), error), end; !error && it != end; it.increment(error)) {
        const auto name = it->path().filename().string();
        std::error_code ignored;
        // Version directories are named by the translator's 64-digit source version.
        if (name != current && valid_key(name) && it->is_directory(ignored)) fs::remove_all(it->path(), ignored);
    }
    const auto now = fs::file_time_type::clock::now();
    for (fs::directory_iterator it(directory_, error), end; !error && it != end; it.increment(error)) {
        if (!it->path().filename().string().starts_with(".partial-")) continue;
        std::error_code ignored;
        const auto written = fs::last_write_time(it->path(), ignored);
        if (!ignored && now - written > partial_age) fs::remove_all(it->path(), ignored);
    }
}

bool StageCache::store(std::string_view key, const TranslatedStage& stage) const {
    if (!valid_key(key)) return false;
    std::error_code error;
    const auto entry = directory_ / std::string(key);
    if (fs::is_directory(entry, error)) return true;
    fs::create_directories(directory_, error);
    if (error) return false;
    static std::atomic<std::uint64_t> sequence{0};
    const auto nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + '-' +
                       std::to_string(sequence.fetch_add(1));
    const auto temporary = directory_ / (".partial-" + std::string(key) + '-' + nonce);
    if (!fs::create_directory(temporary, error) || error) return false;
    const bool written = write_file(temporary / "source.metal", text(stage.source)) &&
                         write_file(temporary / "reflection.json", text(stage.reflection)) &&
                         write_file(temporary / "program.bin", stage.program);
    if (written) fs::rename(temporary, entry, error);
    if (!written || error) {
        std::error_code ignored;
        fs::remove_all(temporary, ignored);
        // Another writer may have placed the same entry first.
        return fs::is_directory(entry, ignored);
    }
    return true;
}

} // namespace armigo::shader::translation
