// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "container_fixture.h"
#include "inputs.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace container_test;
namespace pack = armigo::pack;
namespace loader = armigo::loader;

int failures = 0;

#define CHECK(condition)                                                         \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
            ++failures;                                                          \
        }                                                                        \
    } while (false)

struct Fixture {
    TestKeys keys = TestKeys::generate();
    std::optional<TestRsaKey> sign_key;
    std::vector<std::byte> npdm;
    std::filesystem::path directory;

    Fixture() {
        sign_key = TestRsaKey::generate();
        if (sign_key) {
            npdm = build_npdm(*sign_key);
        }
        directory = std::filesystem::temp_directory_path() / "armigo_pack_inputs_tests";
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);
    }

    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

struct ContentSpec {
    std::uint64_t title_id = 0;
    std::uint64_t application_id = 0;
    std::uint32_t version = 0;
    pack::ContentType type = pack::ContentType::application;
    bool with_control = true;
    std::uint8_t control_fill = 0;
    bool malicious_icon = false;
};

struct BuiltContent {
    std::vector<std::pair<std::string, std::vector<std::byte>>> entries;
    std::vector<std::byte> nacp;
    std::vector<std::pair<std::string, std::vector<std::byte>>> icons;
    std::string content_nca_id;
    std::string control_nca_id;
    std::uint64_t title_id = 0;
};

std::vector<std::byte> filled_bytes(std::size_t size, std::uint8_t fill) {
    std::vector<std::byte> bytes(size);
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = std::byte(static_cast<std::uint8_t>(i * 3 + fill));
    }
    return bytes;
}

void put_u48(std::vector<std::byte>& image, std::size_t offset, std::uint64_t value) {
    for (std::size_t i = 0; i < 6; ++i) {
        image[offset + i] = std::byte(static_cast<std::uint8_t>(value >> (8 * i)));
    }
}

BuiltContent build_content(Fixture& fixture, const ContentSpec& spec) {
    struct Record {
        BuiltNca nca;
        pack::ContentRecordType type = pack::ContentRecordType::program;
    };
    std::vector<Record> records;
    BuiltContent built;
    built.title_id = spec.title_id;

    if (spec.type != pack::ContentType::application && spec.type != pack::ContentType::patch) {
        records.push_back({build_program_nca(fixture.keys, *fixture.sign_key, fixture.npdm, true,
                                             false, spec.title_id, {}, 4),
                           pack::ContentRecordType::data});
    } else {
        records.push_back({build_program_nca(fixture.keys, *fixture.sign_key, fixture.npdm, true,
                                             false, spec.title_id),
                           pack::ContentRecordType::program});
    }

    if (spec.with_control) {
        built.nacp = filled_bytes(0x4000, spec.control_fill);
        built.icons = {{"icon_AmericanEnglish.dat", filled_bytes(0x123, spec.control_fill + 1)},
                       {"icon_Japanese.dat", filled_bytes(0xE7, spec.control_fill + 2)}};
        RomFsDirSpec control_root;
        control_root.files = {{"control.nacp", built.nacp.size(), spec.control_fill},
                              {"icon_AmericanEnglish.dat", built.icons[0].second.size(),
                               static_cast<std::uint8_t>(spec.control_fill + 1)},
                              {"icon_Japanese.dat", built.icons[1].second.size(),
                               static_cast<std::uint8_t>(spec.control_fill + 2)}};
        if (spec.malicious_icon) {
            control_root.files.push_back({"icon_../../payload.dat", 0x20, 0x77});
        }
        const BuiltRomFsTree romfs = build_romfs_tree(control_root);
        records.push_back({build_program_nca(fixture.keys, *fixture.sign_key, fixture.npdm, true,
                                             false, spec.title_id, romfs.image, 2),
                           pack::ContentRecordType::control});
    }

    std::size_t extended_size = 0;
    switch (spec.type) {
        case pack::ContentType::application:
        case pack::ContentType::patch:
        case pack::ContentType::add_on_content:
            extended_size = 0x10;
            break;
        default:
            break;
    }
    std::vector<std::byte> cnmt(0x20 + extended_size + records.size() * 0x38, std::byte{0});
    put_u64(cnmt, 0x00, spec.title_id);
    put_u32(cnmt, 0x08, spec.version);
    cnmt[0x0C] = std::byte{static_cast<std::uint8_t>(spec.type)};
    put_u16(cnmt, 0x0E, static_cast<std::uint16_t>(extended_size));
    put_u16(cnmt, 0x10, static_cast<std::uint16_t>(records.size()));
    put_u32(cnmt, 0x18, 0x00050000);
    if (spec.type == pack::ContentType::application) {
        put_u64(cnmt, 0x20, spec.title_id | 0x800);
        put_u32(cnmt, 0x28, 0x00060000);
        put_u32(cnmt, 0x2C, 0x00070000);
    } else if (spec.type == pack::ContentType::patch) {
        put_u64(cnmt, 0x20,
                spec.application_id ? spec.application_id : spec.title_id & ~std::uint64_t{0x800});
        put_u32(cnmt, 0x28, 0x00090000);
    } else if (spec.type == pack::ContentType::add_on_content) {
        put_u64(cnmt, 0x20, spec.application_id);
        put_u32(cnmt, 0x28, 0x00080000);
    }
    for (std::size_t i = 0; i < records.size(); ++i) {
        const Record& record = records[i];
        const std::size_t at = 0x20 + extended_size + i * 0x38;
        std::memcpy(cnmt.data() + at, record.nca.content_hash.data(),
                    record.nca.content_hash.size());
        std::memcpy(cnmt.data() + at + 0x20, record.nca.nca_id.data(), record.nca.nca_id.size());
        put_u48(cnmt, at + 0x30, record.nca.image.size());
        cnmt[at + 0x36] = std::byte{static_cast<std::uint8_t>(record.type)};
    }
    const std::vector<std::pair<std::string, std::vector<std::byte>>> meta_entries = {
        {"meta.cnmt", cnmt}};
    BuiltNca meta = build_program_nca(fixture.keys, *fixture.sign_key, fixture.npdm, true, false,
                                      spec.title_id, {}, 1, &meta_entries);

    for (Record& record : records) {
        const std::string id = to_hex(record.nca.nca_id);
        if (record.type == pack::ContentRecordType::control) {
            built.control_nca_id = id;
        } else {
            built.content_nca_id = id;
        }
        built.entries.emplace_back(id + ".nca", std::move(record.nca.image));
    }
    built.entries.emplace_back(to_hex(meta.nca_id) + ".nca", std::move(meta.image));
    return built;
}

void write_container(const std::filesystem::path& path, const BuiltContent& content) {
    const std::vector<std::byte> nsp = build_nsp(content.entries);
    CHECK(write_file(path, nsp));
}

void write_keys(const std::filesystem::path& path, const std::string& text) {
    CHECK(write_file(path, std::span(reinterpret_cast<const std::byte*>(text.data()), text.size())));
}

pack::InputsOptions options_for(const Fixture& fixture, const std::filesystem::path& keys_file) {
    pack::InputsOptions options;
    options.keys_file = keys_file;
    options.anchors = test_anchors(*fixture.sign_key);
    return options;
}

const pack::Inputs* successful_inputs(
    const std::variant<pack::InputsFailure, pack::Inputs>& result) {
    const auto* inputs = std::get_if<pack::Inputs>(&result);
    CHECK(inputs != nullptr);
    if (!inputs) {
        const auto* failure = std::get_if<pack::InputsFailure>(&result);
        if (failure) {
            std::printf("  classification failed: %s\n", failure->detail.c_str());
        }
    }
    return inputs;
}

bool has_refusal(const pack::Inputs& inputs, std::string_view text) {
    return std::any_of(inputs.refused.begin(), inputs.refused.end(), [&](const pack::RefusedInput& item) {
        return item.reason.find(text) != std::string::npos;
    });
}

void test_selection(Fixture& fixture, const BuiltContent& application,
                    const BuiltContent& newest_patch) {
    constexpr std::uint64_t kApplication = 0x01007EF00011E000;
    constexpr std::uint64_t kOtherApplication = 0x0100ABCD00112000;
    const BuiltContent old_patch = build_content(
        fixture, {kApplication | 0x800, kOtherApplication, 100, pack::ContentType::patch, false, 0});
    const BuiltContent dlc = build_content(
        fixture, {kApplication + 0x1001, kApplication, 1, pack::ContentType::add_on_content, false, 0});
    const BuiltContent other_dlc = build_content(
        fixture, {kOtherApplication + 0x1001, kOtherApplication, 1,
                  pack::ContentType::add_on_content, false, 0});
    const BuiltContent other_patch = build_content(
        fixture, {kOtherApplication | 0x800, 0, 500, pack::ContentType::patch, false, 0});

    const std::filesystem::path folder = fixture.directory / "mixed";
    std::filesystem::create_directories(folder);
    const std::vector<std::byte> xci = build_xci_partitions(application.entries, old_patch.entries);
    CHECK(write_file(folder / "game.xci", xci));
    write_container(folder / "newest.nsp", newest_patch);
    write_container(folder / "dlc.nsp", dlc);
    write_container(folder / "other-dlc.nsp", other_dlc);
    write_container(folder / "other-patch.nsp", other_patch);
    const std::vector<std::byte> compressed = {std::byte{0}, std::byte{1}};
    CHECK(write_file(folder / "compressed.nsz", compressed));
    CHECK(write_file(folder / "ignored.txt", compressed));
    const std::filesystem::path keys_file = fixture.directory / "prod.keys";
    write_keys(keys_file, fixture.keys.keys_text);

    auto result = pack::classify_inputs({folder}, options_for(fixture, keys_file));
    const pack::Inputs* inputs = successful_inputs(result);
    if (!inputs) {
        return;
    }
    CHECK(inputs->application.title_id == kApplication);
    CHECK(inputs->application.required_system_version == 0x00060000);
    CHECK(inputs->application.required_application_version == 0x00070000);
    CHECK(inputs->application.location.file.filename() == "game.xci");
    CHECK(inputs->application.location.content_set_index == 0);
    CHECK(inputs->patch.has_value());
    CHECK(inputs->patch && inputs->patch->version == 200);
    CHECK(inputs->patch && inputs->patch->application_id == kApplication);
    CHECK(inputs->patch && inputs->patch->required_system_version == 0x00090000);
    CHECK(inputs->superseded_updates.size() == 1);
    CHECK(inputs->superseded_updates.size() == 1 &&
          inputs->superseded_updates[0].update.version == 100);
    CHECK(inputs->superseded_updates.size() == 1 &&
          inputs->superseded_updates[0].update.application_id == kOtherApplication);
    CHECK(inputs->superseded_updates.size() == 1 &&
          inputs->superseded_updates[0].update.location.file.filename() == "game.xci");
    CHECK(inputs->add_on_content.size() == 1);
    CHECK(inputs->add_on_content.size() == 1 &&
          inputs->add_on_content[0].title_id == kApplication + 0x1001);
    CHECK(inputs->content_sets.size() == 6);
    CHECK(has_refusal(*inputs, pack::format_title_id(kOtherApplication + 0x1001)));
    CHECK(has_refusal(*inputs, pack::format_title_id(kOtherApplication | 0x800)));
    CHECK(has_refusal(*inputs, "Compressed dumps are not supported"));
    CHECK(!has_refusal(*inputs, "ignored.txt"));
    CHECK(inputs->control.nacp == newest_patch.nacp);
    CHECK(inputs->control.source.file.filename() == "newest.nsp");
    CHECK(inputs->control.icons.size() == 2);
    CHECK(inputs->control.icons.size() == 2 &&
          inputs->control.icons[0].name == "icon_AmericanEnglish.dat" &&
          inputs->control.icons[0].bytes == newest_patch.icons[0].second);
    CHECK(inputs->control.icons.size() == 2 &&
          inputs->control.icons[1].bytes == newest_patch.icons[1].second);
    CHECK(inputs->keys.validated);
    CHECK(std::find(inputs->keys.required_key_names.begin(), inputs->keys.required_key_names.end(),
                    "header_key") != inputs->keys.required_key_names.end());
    CHECK(std::find(inputs->keys.required_key_names.begin(), inputs->keys.required_key_names.end(),
                    "key_area_key_application_00") != inputs->keys.required_key_names.end());
}

std::pair<std::string, std::vector<std::byte>>* find_entry(
    BuiltContent& content, const std::string& id) {
    const auto match = std::find_if(content.entries.begin(), content.entries.end(),
                                    [&](const auto& entry) { return entry.first == id + ".nca"; });
    return match == content.entries.end() ? nullptr : &*match;
}

void test_content_identity(Fixture& fixture, const BuiltContent& application,
                           const BuiltContent& other) {
    const std::filesystem::path keys_file = fixture.directory / "prod.keys";
    BuiltContent renamed = application;
    auto* control = find_entry(renamed, application.control_nca_id);
    CHECK(control != nullptr);
    if (control) {
        control->first = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.nca";
    }
    write_container(fixture.directory / "renamed.nsp", renamed);
    auto renamed_result = pack::classify_inputs({fixture.directory / "renamed.nsp"},
                                                options_for(fixture, keys_file));
    const pack::Inputs* renamed_inputs = successful_inputs(renamed_result);
    CHECK(renamed_inputs && renamed_inputs->control.nacp == application.nacp);

    BuiltContent stand_in = application;
    auto* stand_in_control = find_entry(stand_in, application.control_nca_id);
    const auto other_control = std::find_if(
        other.entries.begin(), other.entries.end(), [&](const auto& entry) {
            return entry.first == other.control_nca_id + ".nca";
        });
    CHECK(stand_in_control != nullptr && other_control != other.entries.end());
    if (stand_in_control && other_control != other.entries.end()) {
        stand_in_control->second = other_control->second;
    }
    write_container(fixture.directory / "stand-in.nsp", stand_in);
    auto stand_in_result = pack::classify_inputs({fixture.directory / "stand-in.nsp"},
                                                 options_for(fixture, keys_file));
    const auto* failure = std::get_if<pack::InputsFailure>(&stand_in_result);
    CHECK(failure != nullptr);
    CHECK(failure && failure->detail.find(application.control_nca_id) != std::string::npos);
}

void test_verification_failures(Fixture& fixture, const BuiltContent& application) {
    const std::filesystem::path keys_file = fixture.directory / "prod.keys";
    BuiltContent corrupt = application;
    auto* program = find_entry(corrupt, application.content_nca_id);
    CHECK(program != nullptr && program->second.size() > 0x1800);
    if (program && program->second.size() > 0x1800) {
        program->second[0x1800] ^= std::byte{0xFF};
    }
    write_container(fixture.directory / "corrupt-program.nsp", corrupt);
    auto result = pack::classify_inputs({fixture.directory / "corrupt-program.nsp"},
                                        options_for(fixture, keys_file));
    const auto* failure = std::get_if<pack::InputsFailure>(&result);
    CHECK(failure != nullptr);
    CHECK(failure && failure->detail.find("hash table entry") != std::string::npos);

    BuiltContent with_junk = application;
    with_junk.entries.emplace_back("cccccccccccccccccccccccccccccccc.nca",
                                   std::vector<std::byte>(0x1000, std::byte{0}));
    write_container(fixture.directory / "junk.nsp", with_junk);
    auto junk_result = pack::classify_inputs({fixture.directory / "junk.nsp"},
                                             options_for(fixture, keys_file));
    const pack::Inputs* junk_inputs = successful_inputs(junk_result);
    CHECK(junk_inputs && has_refusal(*junk_inputs, "cccccccccccccccccccccccccccccccc.nca"));
    CHECK(junk_inputs && has_refusal(*junk_inputs, "not NCA3"));
}

void test_isolated_bad_cnmt(Fixture& fixture, const BuiltContent& application) {
    const std::vector<std::pair<std::string, std::vector<std::byte>>> broken_entries = {
        {"broken.cnmt", {std::byte{0}, std::byte{1}, std::byte{2}}}};
    BuiltNca broken = build_program_nca(fixture.keys, *fixture.sign_key, fixture.npdm, true, false,
                                        0x0100000000002000, {}, 1, &broken_entries);
    const std::vector<std::pair<std::string, std::vector<std::byte>>> update_entries = {
        {to_hex(broken.nca_id) + ".cnmt.nca", std::move(broken.image)}};
    const std::vector<std::byte> xci = build_xci_partitions(application.entries, update_entries);
    const std::filesystem::path path = fixture.directory / "bad-cnmt.xci";
    CHECK(write_file(path, xci));
    auto result = pack::classify_inputs({path}, options_for(fixture, fixture.directory / "prod.keys"));
    const pack::Inputs* inputs = successful_inputs(result);
    CHECK(inputs && inputs->application.title_id == application.title_id);
    CHECK(inputs && has_refusal(*inputs, "CNMT"));
    CHECK(inputs && std::any_of(inputs->refused.begin(), inputs->refused.end(),
                                [](const pack::RefusedInput& refusal) {
                                    return refusal.content_set_index &&
                                           *refusal.content_set_index == 1;
                                }));
}

void test_icon_sanitization(Fixture& fixture) {
    const BuiltContent malicious = build_content(
        fixture, {0x0100000000003000, 0, 0, pack::ContentType::application, true, 4, true});
    write_container(fixture.directory / "malicious-icon.nsp", malicious);
    auto result = pack::classify_inputs({fixture.directory / "malicious-icon.nsp"},
                                        options_for(fixture, fixture.directory / "prod.keys"));
    const auto* failure = std::get_if<pack::InputsFailure>(&result);
    CHECK(failure != nullptr);
    CHECK(failure && failure->detail.find("unusable name") != std::string::npos);
}

void test_system_content_is_ignored(Fixture& fixture, const BuiltContent& application) {
    constexpr std::uint64_t kSystemTitle = 0x0100000000004000;
    BuiltContent system = build_content(
        fixture, {kSystemTitle, 0, 1, pack::ContentType::system_update, false, 0});
    auto* data = find_entry(system, system.content_nca_id);
    CHECK(data != nullptr && data->second.size() > 0x1800);
    if (data && data->second.size() > 0x1800) {
        data->second[0x1800] ^= std::byte{0xFF};
    }
    const std::vector<std::byte> xci = build_xci_partitions(application.entries, system.entries);
    const std::filesystem::path path = fixture.directory / "system.xci";
    CHECK(write_file(path, xci));
    auto result = pack::classify_inputs({path}, options_for(fixture, fixture.directory / "prod.keys"));
    const pack::Inputs* inputs = successful_inputs(result);
    CHECK(inputs && !has_refusal(*inputs, pack::format_title_id(kSystemTitle)));
    CHECK(inputs && !has_refusal(*inputs, "hash table"));
    CHECK(inputs && inputs->content_sets.size() == 2);
}

void test_duplicate_and_addon_versions(Fixture& fixture, const BuiltContent& application,
                                       const BuiltContent& patch) {
    constexpr std::uint64_t kApplication = 0x01007EF00011E000;
    const BuiltContent duplicate = build_content(
        fixture, {kApplication | 0x800, 0, 200, pack::ContentType::patch, true, 0x50});
    write_container(fixture.directory / "base.nsp", application);
    write_container(fixture.directory / "patch-a.nsp", patch);
    write_container(fixture.directory / "patch-b.nsp", duplicate);
    auto patches = pack::classify_inputs(
        {fixture.directory / "base.nsp", fixture.directory / "patch-a.nsp",
         fixture.directory / "patch-b.nsp"},
        options_for(fixture, fixture.directory / "prod.keys"));
    const pack::Inputs* patch_inputs = successful_inputs(patches);
    CHECK(patch_inputs && patch_inputs->duplicate_updates.size() == 1);
    CHECK(patch_inputs && patch_inputs->superseded_updates.empty());

    const BuiltContent addon_v1 = build_content(
        fixture, {kApplication + 0x1001, kApplication, 1, pack::ContentType::add_on_content, false, 0});
    const BuiltContent addon_v2 = build_content(
        fixture, {kApplication + 0x1001, kApplication, 2, pack::ContentType::add_on_content, false, 0});
    write_container(fixture.directory / "addon-v1.nsp", addon_v1);
    write_container(fixture.directory / "addon-v2.nsp", addon_v2);
    auto addons = pack::classify_inputs(
        {fixture.directory / "base.nsp", fixture.directory / "addon-v1.nsp",
         fixture.directory / "addon-v2.nsp"},
        options_for(fixture, fixture.directory / "prod.keys"));
    const pack::Inputs* addon_inputs = successful_inputs(addons);
    CHECK(addon_inputs && addon_inputs->add_on_content.size() == 1);
    CHECK(addon_inputs && addon_inputs->add_on_content.size() == 1 &&
          addon_inputs->add_on_content[0].version == 2);
    CHECK(addon_inputs && addon_inputs->superseded_add_on_content.size() == 1);
}

void test_partition_offset_overflow(Fixture& fixture, const BuiltContent& application) {
    std::vector<std::byte> malformed = build_nsp(
        {{"overflow.nca", std::vector<std::byte>(0x1000, std::byte{0})}});
    put_u64(malformed, 0x10, ~std::uint64_t{0});
    const std::filesystem::path path = fixture.directory / "overflow.nsp";
    CHECK(write_file(path, malformed));
    write_container(fixture.directory / "base.nsp", application);
    auto result = pack::classify_inputs(
        {fixture.directory / "base.nsp", path},
        options_for(fixture, fixture.directory / "prod.keys"));
    const pack::Inputs* inputs = successful_inputs(result);
    CHECK(inputs && has_refusal(*inputs, "overflow"));

    auto parsed_keys = loader::read_container_keys_file(fixture.directory / "prod.keys");
    loader::ContainerExtractionOptions options;
    options.output_directory = fixture.directory / "overflow-output";
    options.anchors = test_anchors(*fixture.sign_key);
    options.keys = std::move(std::get<loader::ContainerKeys>(parsed_keys));
    const auto extraction = loader::extract_program_container(path, options);
    const auto* failure = std::get_if<loader::ContainerFailure>(&extraction);
    CHECK(failure && failure->code == loader::ContainerError::partition_invalid);
    CHECK(failure && failure->detail.find("overflow") != std::string::npos);
}

void test_key_failures(Fixture& fixture, const BuiltContent& application) {
    const std::filesystem::path app_path = fixture.directory / "base.nsp";
    write_container(app_path, application);
    const std::string missing_area_text =
        "header_key = " + to_hex(fixture.keys.header_key) + "\n";
    const std::filesystem::path missing_file = fixture.directory / "missing.keys";
    write_keys(missing_file, missing_area_text);
    auto missing = pack::classify_inputs({app_path}, options_for(fixture, missing_file));
    const auto* missing_failure = std::get_if<pack::InputsFailure>(&missing);
    CHECK(missing_failure != nullptr);
    CHECK(missing_failure &&
          missing_failure->detail.find("key_area_key_application_00") != std::string::npos);
    CHECK(missing_failure && missing_failure->detail.find(to_hex(fixture.keys.area_key)) == std::string::npos);

    TestKeys wrong = fixture.keys;
    wrong.header_key[0] ^= std::byte{0xFF};
    wrong.keys_text = "header_key = " + to_hex(wrong.header_key) +
                      "\nkey_area_key_application_00 = " + to_hex(wrong.area_key) + "\n";
    const std::filesystem::path wrong_file = fixture.directory / "wrong.keys";
    write_keys(wrong_file, wrong.keys_text);
    auto wrong_result = pack::classify_inputs({app_path}, options_for(fixture, wrong_file));
    const auto* wrong_failure = std::get_if<pack::InputsFailure>(&wrong_result);
    CHECK(wrong_failure != nullptr);
    CHECK(wrong_failure && wrong_failure->detail.find("header_key") != std::string::npos);
    CHECK(wrong_failure && wrong_failure->detail.find(to_hex(wrong.header_key)) == std::string::npos);
}

} // namespace

int main() {
    Fixture fixture;
    CHECK(fixture.sign_key.has_value());
    CHECK(!fixture.npdm.empty());
    if (!fixture.sign_key || fixture.npdm.empty()) {
        return 1;
    }
    constexpr std::uint64_t kApplication = 0x01007EF00011E000;
    const BuiltContent application = build_content(
        fixture, {kApplication, 0, 0, pack::ContentType::application, true, 3});
    const BuiltContent newest_patch = build_content(
        fixture, {kApplication | 0x800, 0, 200, pack::ContentType::patch, true, 0x40});

    test_selection(fixture, application, newest_patch);
    const std::filesystem::path keys_file = fixture.directory / "prod.keys";
    write_keys(keys_file, fixture.keys.keys_text);
    write_container(fixture.directory / "base.nsp", application);
    auto two = pack::classify_inputs({fixture.directory / "base.nsp"},
                                     options_for(fixture, keys_file));
    CHECK(successful_inputs(two) != nullptr);
    const BuiltContent second = build_content(
        fixture, {0x0100ABCD00112000, 0, 0, pack::ContentType::application, true, 9});
    write_container(fixture.directory / "second.nsp", second);
    auto two_applications = pack::classify_inputs(
        {fixture.directory / "base.nsp", fixture.directory / "second.nsp"},
        options_for(fixture, keys_file));
    const auto* duplicate = std::get_if<pack::InputsFailure>(&two_applications);
    CHECK(duplicate != nullptr);
    CHECK(duplicate && duplicate->detail.find(pack::format_title_id(kApplication)) != std::string::npos);
    CHECK(duplicate &&
          duplicate->detail.find(pack::format_title_id(0x0100ABCD00112000)) != std::string::npos);
    test_content_identity(fixture, application, second);
    test_verification_failures(fixture, application);
    test_isolated_bad_cnmt(fixture, application);
    test_icon_sanitization(fixture);
    test_system_content_is_ignored(fixture, application);
    test_duplicate_and_addon_versions(fixture, application, newest_patch);
    test_partition_offset_overflow(fixture, application);
    test_key_failures(fixture, application);

    if (failures == 0) {
        std::printf("pack input tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
