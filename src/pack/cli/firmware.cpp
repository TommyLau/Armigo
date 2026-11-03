// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "firmware.h"

#include "ios.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <system_error>
#include <variant>
#include <vector>
#include <optional>

namespace armigo::pack {
namespace {

constexpr std::array<std::uint64_t, 6> required_system_data{
    0x0100000000000809, 0x0100000000000810, 0x0100000000000811,
    0x0100000000000812, 0x0100000000000813, 0x0100000000000814};

std::string system_data_id(std::uint64_t title_id) {
    char text[17];
    std::snprintf(text, sizeof(text), "%016llX", static_cast<unsigned long long>(title_id));
    return text;
}

std::string extraction_error(const loader::ContainerFailure& failure) {
    return std::string(loader::to_string(failure.code)) + ": " + failure.detail;
}

std::variant<ContentSet, std::string> required_content(
    const Inputs& inputs, const std::filesystem::path& firmware_input, std::uint64_t title_id) {
    const auto found = std::find_if(inputs.content_sets.begin(), inputs.content_sets.end(),
        [&](const ContentSet& candidate) {
            return candidate.title_id == title_id &&
                (candidate.location.file == firmware_input || candidate.location.file.parent_path() == firmware_input);
        });
    if (found == inputs.content_sets.end()) {
        return "Firmware is missing system data " + system_data_id(title_id) + ".";
    }
    if (!found->find_content(ContentRecordType::data)) {
        return "Firmware system data " + system_data_id(title_id) + " has no data NCA.";
    }
    return *found;
}

std::variant<std::filesystem::path, std::string> extract_system_data(
    const ContentSet& content, const loader::ContainerKeys& keys,
    const std::filesystem::path& output) {
    const auto* record = content.find_content(ContentRecordType::data);
    auto container = content.location.file;
    const auto sibling = container.parent_path() / (record->nca_id + ".nca");
    std::error_code error;
    if (std::filesystem::is_regular_file(sibling, error)) container = sibling;
    loader::ContainerExtractionOptions options;
    options.output_directory = output;
    options.keys = keys;
    auto extracted = loader::extract_data_romfs(container, options, content.title_id);
    if (const auto* failure = std::get_if<loader::ContainerFailure>(&extracted)) {
        return extraction_error(*failure);
    }
    return output;
}

std::variant<FirmwareResources, std::string> read_version(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::array<char, 0x80> bytes{};
    if (!input.read(bytes.data(), static_cast<std::streamsize>(bytes.size())) ||
        bytes[8] != 'N' || bytes[9] != 'X') {
        return "Firmware system version data is malformed.";
    }
    std::string version(bytes.data() + 0x68, 0x18);
    version = version.substr(0, version.find('\0'));
    if (version.empty() || !std::all_of(version.begin(), version.end(), [](unsigned char c) {
            return c >= 0x20 && c < 0x7f;
        })) {
        return "Firmware system version data has no readable version.";
    }
    return FirmwareResources{version};
}

// The entry names of a zip archive, from its central directory (with the
// Zip64 end record when the plain one overflows).
std::variant<std::vector<std::string>, std::string> zip_entry_names(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return std::string("Cannot read the firmware zip.");
    const auto size = static_cast<std::uint64_t>(input.tellg());
    const auto read_at = [&](std::uint64_t offset, std::size_t length) -> std::optional<std::vector<unsigned char>> {
        if (offset > size || length > size - offset) return std::nullopt;
        std::vector<unsigned char> bytes(length);
        input.seekg(static_cast<std::streamoff>(offset));
        if (!input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(length))) return std::nullopt;
        return bytes;
    };
    const auto le = [](const std::vector<unsigned char>& bytes, std::size_t at, unsigned width) {
        std::uint64_t value = 0;
        for (unsigned i = 0; i != width; ++i) value |= std::uint64_t{bytes[at + i]} << (8 * i);
        return value;
    };
    // The end-of-central-directory record lies in the last 64 KiB + 22 bytes.
    const std::uint64_t window = std::min<std::uint64_t>(size, 0x10000 + 22);
    const auto tail = read_at(size - window, static_cast<std::size_t>(window));
    if (!tail || window < 22) return std::string("The firmware zip has no central directory.");
    std::optional<std::size_t> end;
    for (std::size_t at = tail->size() - 22 + 1; at-- > 0;) {
        if (le(*tail, at, 4) == 0x06054b50u) { end = at; break; }
    }
    if (!end) return std::string("The firmware zip has no central directory.");
    std::uint64_t entries = le(*tail, *end + 10, 2);
    std::uint64_t directory = le(*tail, *end + 16, 4);
    if (entries == 0xffff || directory == 0xffffffffu) {
        const std::uint64_t locator = size - window + *end;
        if (locator < 20) return std::string("The firmware zip's Zip64 records are missing.");
        const auto record = read_at(locator - 20, 20);
        if (!record || le(*record, 0, 4) != 0x07064b50u) return std::string("The firmware zip's Zip64 records are missing.");
        const auto end64 = read_at(le(*record, 8, 8), 56);
        if (!end64 || le(*end64, 0, 4) != 0x06064b50u) return std::string("The firmware zip's Zip64 records are malformed.");
        entries = le(*end64, 32, 8);
        directory = le(*end64, 48, 8);
    }
    if (entries > 100000) return std::string("The firmware zip has too many entries.");
    std::vector<std::string> names;
    std::uint64_t at = directory;
    for (std::uint64_t index = 0; index != entries; ++index) {
        const auto header = read_at(at, 46);
        if (!header || le(*header, 0, 4) != 0x02014b50u) return std::string("The firmware zip's central directory is malformed.");
        const auto name_length = le(*header, 28, 2), extra = le(*header, 30, 2), comment = le(*header, 32, 2);
        const auto name = read_at(at + 46, static_cast<std::size_t>(name_length));
        if (!name) return std::string("The firmware zip's central directory is malformed.");
        names.emplace_back(name->begin(), name->end());
        at += 46 + name_length + extra + comment;
    }
    return names;
}

} // namespace

std::variant<FirmwareResources, std::string> read_firmware_version(
    const std::filesystem::path& path) {
    return read_version(path);
}

std::variant<std::filesystem::path, std::string> prepare_firmware_input(
    const std::filesystem::path& input, const std::filesystem::path& work_directory) {
    std::error_code error;
    if (std::filesystem::is_directory(input, error)) return input;
    if (!std::filesystem::is_regular_file(input, error)) {
        return "Firmware input is not a folder or zip file: " + input.string() + ".";
    }
    std::filesystem::remove_all(work_directory, error);
    std::filesystem::create_directories(work_directory, error);
    if (error) return std::string("Cannot prepare the firmware staging directory.");
    // Firmware archives are flat: one NCA file per entry. The names are
    // checked before anything is unpacked, so no entry can reach outside the
    // staging directory; ditto, which ships with macOS, then unpacks.
    auto names = zip_entry_names(input);
    if (const auto* failure = std::get_if<std::string>(&names)) return *failure;
    for (const auto& name : std::get<std::vector<std::string>>(names)) {
        if (name.empty() || name.front() == '.' || name.find_first_of("/\\") != std::string::npos ||
            !name.ends_with(".nca"))
            return "Firmware zip must contain only NCA files at its top level: " + name + ".";
    }
    const auto failure = run_pack_tool({"/usr/bin/ditto", "-x", "-k", input.string(), work_directory.string()});
    if (!failure.empty()) return "Cannot unpack the firmware zip: " + failure;
    for (const auto& entry : std::filesystem::directory_iterator(work_directory, error)) {
        if (!entry.is_regular_file(error) || entry.path().extension() != ".nca") {
            std::filesystem::remove_all(work_directory, error);
            return "Firmware zip must contain only NCA files at its top level: " +
                   entry.path().filename().string() + ".";
        }
    }
    if (error) return "Cannot read the unpacked firmware: " + error.message() + ".";
    return work_directory;
}

std::variant<FirmwareResources, std::string> extract_firmware_resources(
    const Inputs& inputs, const std::filesystem::path& firmware_input,
    const loader::ContainerKeys& keys, const std::filesystem::path& output_directory) {
    std::string version;
    for (const std::uint64_t title_id : required_system_data) {
        auto content = required_content(inputs, firmware_input, title_id);
        if (const auto* error = std::get_if<std::string>(&content)) return *error;
        const auto output = output_directory / system_data_id(title_id);
        auto extracted = extract_system_data(std::get<ContentSet>(content), keys, output);
        if (const auto* error = std::get_if<std::string>(&extracted)) return *error;
        if (title_id == required_system_data.front()) {
            auto parsed = read_version(output / "file");
            std::error_code error;
            std::filesystem::remove_all(output, error);
            if (const auto* failure = std::get_if<std::string>(&parsed)) return *failure;
            version = std::get<FirmwareResources>(parsed).version;
        }
    }
    return FirmwareResources{version};
}

} // namespace armigo::pack
