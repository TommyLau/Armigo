// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/loader/container.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace armigo::pack {

enum class ContentType : std::uint8_t {
    system_program = 0x00,
    system_data = 0x01,
    system_update = 0x02,
    boot_image_package = 0x03,
    boot_image_package_safe = 0x04,
    boot_image_package_recovery = 0x05,
    boot_image_package_exfat = 0x06,
    boot_image_package_exfat_safe = 0x07,
    application = 0x80,
    patch = 0x81,
    add_on_content = 0x82,
};

enum class ContentRecordType : std::uint8_t {
    meta = 0,
    program = 1,
    data = 2,
    control = 3,
    html_document = 4,
    legal_information = 5,
    delta_fragment = 6,
};

[[nodiscard]] const char* to_string(ContentType type) noexcept;
[[nodiscard]] const char* to_string(ContentRecordType type) noexcept;
[[nodiscard]] std::string format_title_id(std::uint64_t title_id);

struct ContentLocation {
    std::filesystem::path file;
    std::size_t content_set_index = 0;

    [[nodiscard]] bool operator==(const ContentLocation&) const = default;
};

struct ContentRecord {
    std::string nca_id;
    std::array<std::byte, 32> content_hash{};
    ContentRecordType type = ContentRecordType::meta;
    std::uint64_t size = 0;
    std::uint8_t id_offset = 0;
};

struct ContentSet {
    ContentLocation location;
    std::string meta_nca_id;
    std::uint64_t title_id = 0;
    std::uint32_t version = 0;
    ContentType type = ContentType::application;
    std::uint64_t application_id = 0;
    std::uint32_t required_download_system_version = 0;
    std::uint32_t required_system_version = 0;
    std::uint32_t required_application_version = 0;
    std::vector<ContentRecord> content_records;

    [[nodiscard]] const ContentRecord* find_content(ContentRecordType type) const noexcept;
};

struct RefusedInput {
    std::filesystem::path file;
    std::optional<std::size_t> content_set_index;
    std::string reason;
};

struct SupersededUpdate {
    ContentSet update;
    std::string reason;
};

struct NamedData {
    std::string name;
    std::vector<std::byte> bytes;
};

struct ControlData {
    ContentLocation source;
    std::vector<std::byte> nacp;
    std::vector<NamedData> icons;
};

struct KeysReport {
    std::filesystem::path keys_file;
    bool validated = false;
    std::vector<std::string> required_key_names;
};

struct Inputs {
    ContentSet application;
    std::optional<ContentSet> patch;
    std::vector<ContentSet> add_on_content;
    std::vector<ContentSet> content_sets;
    std::vector<RefusedInput> refused;
    std::vector<SupersededUpdate> superseded_updates;
    std::vector<SupersededUpdate> duplicate_updates;
    std::vector<SupersededUpdate> superseded_add_on_content;
    ControlData control;
    KeysReport keys;
};

struct InputsFailure {
    std::string detail;
};

struct InputsOptions {
    std::filesystem::path keys_file;
    loader::ContainerTrustAnchors anchors = loader::ContainerTrustAnchors::production();
};

// Expands files and non-recursive folders, validates every NCA against the
// supplied keys, parses all CNMT content sets, selects one title and reads its
// active control data. Key failures are fatal; unrelated titles and unusable
// inputs are reported separately when a valid application remains.
[[nodiscard]] std::variant<InputsFailure, Inputs> classify_inputs(
    const std::vector<std::filesystem::path>& input_paths, const InputsOptions& options);

} // namespace armigo::pack
