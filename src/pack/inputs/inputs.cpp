// ============================================================================
// Armigo Loader - Guest Module Loading
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "inputs.h"

#include "container/container_nca.h"
#include "container/container_scan.h"
#include "container/crypto.h"

#include <CommonCrypto/CommonCrypto.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string_view>
#include <utility>

namespace armigo::pack {
namespace {

namespace detail = loader::container_detail;

constexpr std::uint64_t kMaxCnmtBytes = 16u << 20;
constexpr std::uint64_t kMaxControlMetadataBytes = 16u << 20;
constexpr std::uint64_t kMaxControlFileBytes = 16u << 20;

std::uint16_t read_u16(std::span<const std::byte> bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(static_cast<std::uint8_t>(bytes[offset])) |
           static_cast<std::uint16_t>(static_cast<std::uint8_t>(bytes[offset + 1]) << 8);
}

std::uint64_t read_u48(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 6; ++i) {
        value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(bytes[offset + i])) << (8 * i);
    }
    return value;
}

std::string hex_bytes(std::span<const std::byte> bytes) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string text;
    text.reserve(bytes.size() * 2);
    for (const std::byte byte : bytes) {
        text.push_back(kDigits[static_cast<std::uint8_t>(byte) >> 4]);
        text.push_back(kDigits[static_cast<std::uint8_t>(byte) & 0xF]);
    }
    return text;
}

bool hash_stored_entry(std::FILE* file, const loader::ContainerEntry& entry,
                       std::array<std::byte, 32>& hash) {
    if (::fseeko(file, static_cast<off_t>(entry.offset), SEEK_SET) != 0) {
        return false;
    }
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::vector<std::byte> chunk(1u << 20);
    std::uint64_t done = 0;
    while (done < entry.size) {
        const std::size_t take =
            static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(), entry.size - done));
        const std::size_t read = std::fread(chunk.data(), 1, take, file);
        if (read != take) {
            return false;
        }
        CC_SHA256_Update(&context, chunk.data(), static_cast<CC_LONG>(read));
        done += read;
    }
    std::array<std::uint8_t, 32> digest{};
    CC_SHA256_Final(digest.data(), &context);
    for (std::size_t i = 0; i < hash.size(); ++i) {
        hash[i] = std::byte{digest[i]};
    }
    return true;
}

std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
    });
    return text;
}

bool ends_with(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.substr(text.size() - suffix.size()) == suffix;
}

struct OpenedNca {
    OpenedNca() = default;
    OpenedNca(const OpenedNca&) = delete;
    OpenedNca& operator=(const OpenedNca&) = delete;
    OpenedNca(OpenedNca&&) noexcept = default;
    OpenedNca& operator=(OpenedNca&&) noexcept = default;
    ~OpenedNca() {
        nca.clear_sensitive();
    }

    loader::ContainerEntry entry;
    detail::NcaProgram nca;
    std::string header_id;
    std::array<std::byte, 32> content_hash{};
    bool has_nca_header = false;
    bool has_content_hash = false;
    std::optional<loader::ContainerFailure> failure;
    bool failure_referenced = false;
};

struct OpenedContainer {
    ~OpenedContainer() {
        if (file != nullptr) {
            std::fclose(file);
        }
    }

    std::filesystem::path path;
    std::FILE* file = nullptr;
    loader::ContainerScan scan;
    std::vector<OpenedNca> ncas;
};

struct ParsedSet {
    ContentSet data;
    OpenedContainer* container = nullptr;
    OpenedNca* meta = nullptr;
};

struct ScanResult {
    enum class Status {
        parsed,
        refused,
        fatal,
    } status = Status::parsed;
    std::string detail;
};

bool read_cnmt(const OpenedContainer& container, const OpenedNca& opened,
               ContentSet& content_set, std::string& error) {
    const detail::NcaSection* pfs0_section = nullptr;
    for (const detail::NcaSection& section : opened.nca.sections) {
        if (section.partition_type == detail::kPartitionPfs0 &&
            section.fs_type == detail::kFsTypePfs0) {
            pfs0_section = &section;
            break;
        }
    }
    if (pfs0_section == nullptr) {
        error = "The meta NCA has no PFS0 section.";
        return false;
    }

    detail::Pfs0Partition partition;
    loader::ContainerFailure failure;
    if (!detail::read_pfs0_partition(opened.nca, *pfs0_section, container.file,
                                     pfs0_section->superblock_offset, partition, failure)) {
        error = failure.detail;
        return false;
    }
    std::size_t cnmt_index = partition.file_count;
    for (std::size_t i = 0; i < partition.file_count; ++i) {
        if (ends_with(partition.entry_name(i), ".cnmt")) {
            if (cnmt_index != partition.file_count) {
                error = "The meta NCA contains more than one CNMT file.";
                return false;
            }
            cnmt_index = i;
        }
    }
    if (cnmt_index == partition.file_count) {
        error = "The meta NCA contains no CNMT file.";
        return false;
    }
    const std::uint64_t size = partition.entry_size(cnmt_index);
    if (size < 0x20 || size > kMaxCnmtBytes) {
        error = "The CNMT file size is outside the supported range.";
        return false;
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    const detail::SectionView view{&opened.nca, pfs0_section, nullptr, nullptr, container.file};
    if (!detail::read_section_view(view, partition.entry_file_offset(cnmt_index), bytes)) {
        error = "Reading the CNMT file failed.";
        return false;
    }

    content_set.meta_nca_id = opened.header_id;
    content_set.title_id = detail::read_u64_le(bytes, 0x00);
    content_set.version = detail::read_u32_le(bytes, 0x08);
    content_set.type = static_cast<ContentType>(static_cast<std::uint8_t>(bytes[0x0C]));
    const std::uint16_t extended_size = read_u16(bytes, 0x0E);
    const std::uint16_t content_count = read_u16(bytes, 0x10);
    const std::uint16_t content_meta_count = read_u16(bytes, 0x12);
    content_set.required_download_system_version = detail::read_u32_le(bytes, 0x18);
    const std::size_t extended_offset = 0x20;
    if (extended_size > bytes.size() - extended_offset) {
        error = "The CNMT extended header leaves the file.";
        return false;
    }
    const std::size_t records_offset = extended_offset + extended_size;
    const std::size_t records_size = static_cast<std::size_t>(content_count) * 0x38;
    const std::size_t meta_size = static_cast<std::size_t>(content_meta_count) * 0x10;
    if (records_offset > bytes.size() || records_size > bytes.size() - records_offset ||
        meta_size > bytes.size() - records_offset - records_size) {
        error = "The CNMT content records leave the file.";
        return false;
    }

    switch (content_set.type) {
        case ContentType::application:
            if (extended_size < 0x10) {
                error = "The Application CNMT extended header is truncated.";
                return false;
            }
            content_set.application_id = content_set.title_id;
            content_set.required_system_version =
                detail::read_u32_le(bytes, extended_offset + 8);
            content_set.required_application_version =
                detail::read_u32_le(bytes, extended_offset + 0x0C);
            break;
        case ContentType::patch:
            if (extended_size < 0x10) {
                error = "The Patch CNMT extended header is truncated.";
                return false;
            }
            content_set.application_id = detail::read_u64_le(bytes, extended_offset);
            content_set.required_system_version =
                detail::read_u32_le(bytes, extended_offset + 8);
            break;
        case ContentType::add_on_content:
            if (extended_size < 0x10) {
                error = "The AddOnContent CNMT extended header is truncated.";
                return false;
            }
            content_set.application_id = detail::read_u64_le(bytes, extended_offset);
            content_set.required_application_version =
                detail::read_u32_le(bytes, extended_offset + 8);
            break;
        default:
            content_set.application_id = content_set.title_id;
            break;
    }

    content_set.content_records.reserve(content_count);
    for (std::size_t i = 0; i < content_count; ++i) {
        const std::size_t at = records_offset + i * 0x38;
        ContentRecord record;
        record.nca_id = hex_bytes(std::span<const std::byte>(bytes).subspan(at + 0x20, 0x10));
        std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(at), record.content_hash.size(),
                    record.content_hash.begin());
        record.size = read_u48(bytes, at + 0x30);
        record.type = static_cast<ContentRecordType>(static_cast<std::uint8_t>(bytes[at + 0x36]));
        record.id_offset = static_cast<std::uint8_t>(bytes[at + 0x37]);
        content_set.content_records.push_back(std::move(record));
    }
    return true;
}

bool is_system_content(ContentType type) {
    return static_cast<std::uint8_t>(type) < 0x80;
}

ScanResult scan_container_file(const std::filesystem::path& path, const InputsOptions& options,
                               const loader::ContainerKeys& keys,
                               std::vector<std::unique_ptr<OpenedContainer>>& containers,
                               std::vector<ParsedSet>& sets, std::set<std::string>& key_names,
                               std::vector<RefusedInput>& refused, std::size_t& opened_nca_count,
                               std::size_t& failed_nca_count, std::string& first_open_failure) {
    const std::size_t refusals_before = refused.size();
    auto container = std::make_unique<OpenedContainer>();
    container->path = path;
    container->file = std::fopen(path.c_str(), "rb");
    if (container->file == nullptr) {
        return {ScanResult::Status::refused, "Could not open the input file."};
    }
    loader::ContainerFailure failure;
    if (!loader::scan_container_entries(container->file, container->scan, failure)) {
        return {ScanResult::Status::refused, failure.detail};
    }

    container->ncas.reserve(container->scan.entries.size());
    for (const loader::ContainerEntry& entry : container->scan.entries) {
        if (lowercase(std::filesystem::path(entry.name).extension().string()) != ".nca") {
            continue;
        }
        OpenedNca& opened = container->ncas.emplace_back();
        opened.entry = entry;
        const detail::TitleKeySource title_keys{container->file, &container->scan.tickets, nullptr};
        const detail::CandidateOutcome outcome = detail::open_nca_candidate(
            container->file, entry.offset, keys, title_keys, options.anchors, -1, 0, true, true,
            opened.nca, failure);
        if (outcome == detail::CandidateOutcome::failure &&
            failure.code == loader::ContainerError::key_missing) {
            return {ScanResult::Status::fatal,
                    path.filename().string() + ": " + entry.name + ": " + failure.detail};
        }
        if (outcome != detail::CandidateOutcome::verified) {
            ++failed_nca_count;
            if (first_open_failure.empty()) {
                first_open_failure = path.filename().string() + ": " + entry.name + ": " +
                                     (outcome == detail::CandidateOutcome::failure
                                          ? failure.detail
                                          : "The entry is truncated or is not NCA3 content.");
            }
            opened.failure = outcome == detail::CandidateOutcome::failure
                ? failure
                : loader::ContainerFailure{
                      loader::ContainerError::unsupported_nca,
                      "NCA entry '" + entry.name + "' is truncated or is not NCA3 content."};
        } else {
            ++opened_nca_count;
        }

        if (detail::read_u32_le(opened.nca.header, 0x200) == 0x3341434E) {
            opened.has_nca_header = true;
            std::array<std::byte, 32> header_hash{};
            if (!detail::sha256_digest(opened.nca.header, header_hash)) {
                return {ScanResult::Status::refused, "Hashing an NCA header failed."};
            }
            opened.header_id = hex_bytes(std::span<const std::byte>(header_hash).first<0x10>());
        }
        if (!hash_stored_entry(container->file, entry, opened.content_hash)) {
            return {ScanResult::Status::refused,
                    "Hashing stored NCA entry '" + entry.name + "' failed."};
        }
        opened.has_content_hash = true;
        key_names.insert(opened.nca.required_key_names.begin(), opened.nca.required_key_names.end());
    }

    std::size_t content_set_index = 0;
    std::vector<ParsedSet> file_sets;
    for (OpenedNca& opened : container->ncas) {
        if (opened.failure || opened.nca.content_type != 1) {
            continue;
        }
        ParsedSet parsed;
        parsed.container = container.get();
        parsed.meta = &opened;
        parsed.data.location = {path, content_set_index};
        std::string error;
        if (!read_cnmt(*container, opened, parsed.data, error)) {
            refused.push_back({path, content_set_index,
                               "Content set " + std::to_string(content_set_index) + ": " + error});
        } else {
            file_sets.push_back(std::move(parsed));
        }
        ++content_set_index;
    }

    for (ParsedSet& parsed : file_sets) {
        bool verified = true;
        const bool system_content = is_system_content(parsed.data.type);
        for (const ContentRecord& record : parsed.data.content_records) {
            if (record.type == ContentRecordType::delta_fragment) {
                continue;
            }
            OpenedNca* match = nullptr;
            for (OpenedNca& candidate : container->ncas) {
                const bool content_matches = candidate.has_content_hash &&
                                             candidate.content_hash == record.content_hash;
                const bool failure_matches = candidate.failure && candidate.has_nca_header &&
                                             candidate.header_id == record.nca_id;
                if (content_matches || failure_matches) {
                    match = &candidate;
                    break;
                }
            }
            if (system_content) {
                if (match && match->failure) {
                    match->failure_referenced = true;
                }
                continue;
            }
            if (match == nullptr) {
                refused.push_back({path, parsed.data.location.content_set_index,
                                   "Content NCA " + record.nca_id + " (" +
                                       std::string(to_string(record.type)) +
                                       ") was not found by its CNMT content hash " +
                                       hex_bytes(record.content_hash) + "."});
                verified = false;
                break;
            }
            if (match->failure) {
                match->failure_referenced = true;
                refused.push_back({path, parsed.data.location.content_set_index,
                                   match->entry.name + ": " + match->failure->detail});
                verified = false;
                break;
            }
        }
        if (verified) {
            sets.push_back(std::move(parsed));
        }
    }

    for (OpenedNca& opened : container->ncas) {
        if (opened.failure && !opened.failure_referenced) {
            refused.push_back({path, std::nullopt,
                               opened.entry.name + ": " +
                                   (opened.has_nca_header
                                        ? opened.failure->detail
                                        : "The entry is not NCA3 content or could not be decrypted.")});
        }
    }
    if (file_sets.empty() && refused.size() == refusals_before) {
        return {ScanResult::Status::refused, "No CNMT meta NCA was found in the container."};
    }
    containers.push_back(std::move(container));
    return {};
}

bool read_control(const ParsedSet& parsed, ControlData& control, std::string& error) {
    const ContentRecord* record = parsed.data.find_content(ContentRecordType::control);
    if (record == nullptr) {
        error = "Title " + format_title_id(parsed.data.title_id) + " has no Control content.";
        return false;
    }
    const OpenedNca* opened = nullptr;
    for (const OpenedNca& candidate : parsed.container->ncas) {
        if (!candidate.failure && candidate.has_content_hash &&
            candidate.content_hash == record->content_hash) {
            opened = &candidate;
            break;
        }
    }
    if (opened == nullptr) {
        error = "Control NCA " + record->nca_id + " is absent from '" +
                parsed.container->path.filename().string() + "'.";
        return false;
    }
    if (opened->nca.content_type != 2) {
        error = "CNMT Control record " + record->nca_id + " does not name a control NCA.";
        return false;
    }
    const detail::NcaSection* romfs = nullptr;
    for (const detail::NcaSection& section : opened->nca.sections) {
        if (section.partition_type == detail::kPartitionRomFs &&
            section.fs_type == detail::kFsTypeRomFs) {
            romfs = &section;
            break;
        }
    }
    if (romfs == nullptr || romfs->is_bktr()) {
        error = "The control NCA has no directly readable RomFS section.";
        return false;
    }

    const detail::SectionView view{&opened->nca, romfs, nullptr, nullptr, parsed.container->file};
    std::array<std::byte, 0x50> header{};
    if (!detail::read_section_view(view, romfs->romfs_offset, header) ||
        detail::read_u64_le(header, 0) != 0x50) {
        error = "The control RomFS header is missing or malformed.";
        return false;
    }
    const std::uint64_t dir_meta_offset = detail::read_u64_le(header, 0x18);
    const std::uint64_t dir_meta_size = detail::read_u64_le(header, 0x20);
    const std::uint64_t file_meta_offset = detail::read_u64_le(header, 0x38);
    const std::uint64_t file_meta_size = detail::read_u64_le(header, 0x40);
    const std::uint64_t data_offset = detail::read_u64_le(header, 0x48);
    const std::uint64_t romfs_size = romfs->superblock.romfs_size;
    if (dir_meta_size < 0x18 || dir_meta_size > kMaxControlMetadataBytes ||
        file_meta_size > kMaxControlMetadataBytes || dir_meta_offset > romfs_size ||
        dir_meta_size > romfs_size - dir_meta_offset || file_meta_offset > romfs_size ||
        file_meta_size > romfs_size - file_meta_offset || data_offset > romfs_size) {
        error = "The control RomFS metadata leaves the verified image.";
        return false;
    }
    std::vector<std::byte> directories(static_cast<std::size_t>(dir_meta_size));
    std::vector<std::byte> files(static_cast<std::size_t>(file_meta_size));
    if (!detail::read_section_view(view, romfs->romfs_offset + dir_meta_offset, directories) ||
        !detail::read_section_view(view, romfs->romfs_offset + file_meta_offset, files)) {
        error = "Reading the control RomFS metadata failed.";
        return false;
    }
    const std::uint32_t root_file = detail::read_u32_le(directories, 0x0C);
    constexpr std::uint32_t kEmpty = 0xFFFFFFFF;
    std::size_t visits = 0;
    bool found_nacp = false;
    for (std::uint32_t entry = root_file; entry != kEmpty;
         entry = detail::read_u32_le(files, entry + 0x04)) {
        if (static_cast<std::size_t>(entry) + 0x20 > files.size() ||
            ++visits > files.size() / 0x20) {
            error = "The control RomFS file graph is malformed.";
            return false;
        }
        const std::uint32_t name_size = detail::read_u32_le(files, entry + 0x1C);
        if (static_cast<std::size_t>(entry) + 0x20 + name_size > files.size()) {
            error = "A control RomFS filename leaves the metadata table.";
            return false;
        }
        const std::string name(
            reinterpret_cast<const char*>(files.data()) + entry + 0x20, name_size);
        const bool is_nacp = name == "control.nacp";
        const bool is_icon = name.starts_with("icon_") && name.ends_with(".dat");
        if (!is_nacp && !is_icon) {
            continue;
        }
        if (!loader::sanitize_romfs_name(name)) {
            error = "Control file '" + name + "' has an unusable name.";
            return false;
        }
        if (is_nacp && found_nacp) {
            error = "The control RomFS contains more than one control.nacp file.";
            return false;
        }
        const std::uint64_t relative = detail::read_u64_le(files, entry + 0x08);
        const std::uint64_t size = detail::read_u64_le(files, entry + 0x10);
        if (size > kMaxControlFileBytes || relative > romfs_size - data_offset ||
            size > romfs_size - data_offset - relative) {
            error = "Control file '" + name + "' has an unsupported size or extent.";
            return false;
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(size));
        if (!detail::read_section_view(view, romfs->romfs_offset + data_offset + relative, bytes)) {
            error = "Reading control file '" + name + "' failed.";
            return false;
        }
        if (is_nacp) {
            control.nacp = std::move(bytes);
            found_nacp = true;
        } else {
            control.icons.push_back({name, std::move(bytes)});
        }
    }
    if (!found_nacp) {
        error = "The control RomFS contains no control.nacp file.";
        return false;
    }
    std::sort(control.icons.begin(), control.icons.end(), [](const NamedData& a, const NamedData& b) {
        return a.name < b.name;
    });
    control.source = parsed.data.location;
    return true;
}

void expand_input_paths(const std::vector<std::filesystem::path>& input_paths,
                        std::vector<std::filesystem::path>& files,
                        std::vector<RefusedInput>& refused) {
    auto add_file = [&](const std::filesystem::path& path, bool from_folder) {
        const std::string extension = lowercase(path.extension().string());
        if (extension == ".nsz" || extension == ".xcz") {
            refused.push_back({path, std::nullopt, "Compressed dumps are not supported."});
            return;
        }
        if (extension != ".nsp" && extension != ".xci" && !(from_folder && extension == ".nca")) {
            if (!from_folder) {
                refused.push_back(
                    {path, std::nullopt, "Unrecognized input; expected an .xci or .nsp file."});
            }
            return;
        }
        files.push_back(path);
    };

    for (const std::filesystem::path& path : input_paths) {
        std::error_code error;
        const bool directory = std::filesystem::is_directory(path, error);
        if (!error && directory) {
            std::vector<std::filesystem::path> children;
            std::filesystem::directory_iterator iterator(path, error);
            const std::filesystem::directory_iterator end;
            while (!error && iterator != end) {
                const std::filesystem::path child = iterator->path();
                std::error_code type_error;
                if (std::filesystem::is_regular_file(child, type_error) && !type_error) {
                    children.push_back(child);
                }
                iterator.increment(error);
            }
            if (error) {
                refused.push_back({path, std::nullopt, "Could not read the input folder."});
                continue;
            }
            std::sort(children.begin(), children.end());
            for (const std::filesystem::path& child : children) {
                add_file(child, true);
            }
            continue;
        }
        std::error_code file_error;
        if (!std::filesystem::is_regular_file(path, file_error) || file_error) {
            refused.push_back({path, std::nullopt, "The input path is not a readable file or folder."});
            continue;
        }
        add_file(path, false);
    }
}

std::variant<InputsFailure, Inputs> classify_inputs_impl(
    const std::vector<std::filesystem::path>& input_paths, const InputsOptions& options) {
    auto parsed_keys = loader::read_container_keys_file(options.keys_file);
    if (const auto* failure = std::get_if<loader::ContainerKeysFailure>(&parsed_keys)) {
        return InputsFailure{"Could not load the keys file: " + failure->detail};
    }
    loader::ContainerKeys keys = std::move(std::get<loader::ContainerKeys>(parsed_keys));

    std::vector<std::filesystem::path> files;
    std::vector<RefusedInput> refused;
    expand_input_paths(input_paths, files, refused);

    std::vector<std::unique_ptr<OpenedContainer>> containers;
    std::vector<ParsedSet> parsed_sets;
    std::set<std::string> key_names;
    std::size_t opened_nca_count = 0;
    std::size_t failed_nca_count = 0;
    std::string first_open_failure;
    for (const std::filesystem::path& file : files) {
        ScanResult scan = scan_container_file(file, options, keys, containers, parsed_sets, key_names,
                                              refused, opened_nca_count, failed_nca_count,
                                              first_open_failure);
        if (scan.status == ScanResult::Status::fatal) {
            return InputsFailure{scan.detail};
        }
        if (scan.status == ScanResult::Status::refused) {
            refused.push_back({file, std::nullopt, scan.detail});
        }
    }
    if (opened_nca_count == 0 && failed_nca_count != 0) {
        return InputsFailure{first_open_failure};
    }

    const ParsedSet* application = nullptr;
    for (const ParsedSet& candidate : parsed_sets) {
        if (candidate.data.type != ContentType::application) {
            continue;
        }
        if (application != nullptr) {
            return InputsFailure{
                "Two Application content sets were supplied: " +
                format_title_id(application->data.title_id) + " and " +
                format_title_id(candidate.data.title_id) + "."};
        }
        application = &candidate;
    }
    if (application == nullptr) {
        std::string detail = "No Application content set was found in the supplied inputs.";
        for (const RefusedInput& refusal : refused) {
            detail += " " + refusal.file.filename().string() + ": " + refusal.reason;
        }
        return InputsFailure{std::move(detail)};
    }

    Inputs result;
    result.application = application->data;
    const ParsedSet* newest_patch = nullptr;
    for (const ParsedSet& candidate : parsed_sets) {
        if (&candidate == application || is_system_content(candidate.data.type)) {
            continue;
        }
        if (candidate.data.type == ContentType::patch &&
            candidate.data.title_id == (application->data.title_id | 0x800)) {
            if (newest_patch == nullptr || candidate.data.version > newest_patch->data.version) {
                if (newest_patch != nullptr) {
                    result.superseded_updates.push_back(
                        {newest_patch->data, "A newer patch was selected."});
                }
                newest_patch = &candidate;
            } else if (candidate.data.version == newest_patch->data.version) {
                result.duplicate_updates.push_back(
                    {candidate.data, "Duplicate patch version " +
                                         std::to_string(candidate.data.version) + "."});
            } else {
                result.superseded_updates.push_back(
                    {candidate.data, "A newer patch was selected."});
            }
            continue;
        }
        if (candidate.data.type == ContentType::add_on_content &&
            candidate.data.application_id == application->data.title_id) {
            auto existing = std::find_if(result.add_on_content.begin(), result.add_on_content.end(),
                                         [&](const ContentSet& addon) {
                                             return addon.title_id == candidate.data.title_id;
                                         });
            if (existing == result.add_on_content.end()) {
                result.add_on_content.push_back(candidate.data);
            } else if (candidate.data.version > existing->version) {
                result.superseded_add_on_content.push_back(
                    {*existing, "A newer add-on version was selected."});
                *existing = candidate.data;
            } else if (candidate.data.version == existing->version) {
                result.superseded_add_on_content.push_back(
                    {candidate.data, "Duplicate add-on version; the first copy was retained."});
            } else {
                result.superseded_add_on_content.push_back(
                    {candidate.data, "A newer add-on version was selected."});
            }
            continue;
        }
        refused.push_back({candidate.data.location.file, candidate.data.location.content_set_index,
                           "Title " + format_title_id(candidate.data.title_id) +
                               " belongs to another title."});
    }
    if (newest_patch != nullptr) {
        result.patch = newest_patch->data;
        for (SupersededUpdate& update : result.superseded_updates) {
            update.reason = "Superseded by patch version " +
                            std::to_string(newest_patch->data.version) + ".";
        }
    }
    result.refused = std::move(refused);
    result.content_sets.reserve(parsed_sets.size());
    for (const ParsedSet& parsed : parsed_sets) {
        result.content_sets.push_back(parsed.data);
    }

    const ParsedSet* control_set = application;
    if (newest_patch != nullptr) {
        const ContentRecord* control_record =
            newest_patch->data.find_content(ContentRecordType::control);
        const bool available = control_record != nullptr &&
            std::any_of(newest_patch->container->ncas.begin(), newest_patch->container->ncas.end(),
                        [&](const OpenedNca& candidate) {
                            return !candidate.failure && candidate.has_content_hash &&
                                   candidate.content_hash == control_record->content_hash;
                        });
        if (available) {
            control_set = newest_patch;
        }
    }
    std::string control_error;
    if (!read_control(*control_set, result.control, control_error)) {
        return InputsFailure{control_error};
    }
    result.keys.keys_file = options.keys_file;
    result.keys.validated = true;
    result.keys.required_key_names.assign(key_names.begin(), key_names.end());
    return result;
}

} // namespace

const char* to_string(ContentType type) noexcept {
    switch (type) {
        case ContentType::system_program: return "SystemProgram";
        case ContentType::system_data: return "SystemData";
        case ContentType::system_update: return "SystemUpdate";
        case ContentType::boot_image_package: return "BootImagePackage";
        case ContentType::boot_image_package_safe: return "BootImagePackageSafe";
        case ContentType::boot_image_package_recovery: return "BootImagePackageRecovery";
        case ContentType::boot_image_package_exfat: return "BootImagePackageExFat";
        case ContentType::boot_image_package_exfat_safe: return "BootImagePackageExFatSafe";
        case ContentType::application: return "Application";
        case ContentType::patch: return "Patch";
        case ContentType::add_on_content: return "AddOnContent";
    }
    return "Unknown";
}

const char* to_string(ContentRecordType type) noexcept {
    switch (type) {
        case ContentRecordType::meta: return "Meta";
        case ContentRecordType::program: return "Program";
        case ContentRecordType::data: return "Data";
        case ContentRecordType::control: return "Control";
        case ContentRecordType::html_document: return "HtmlDocument";
        case ContentRecordType::legal_information: return "LegalInformation";
        case ContentRecordType::delta_fragment: return "DeltaFragment";
    }
    return "Unknown";
}

std::string format_title_id(std::uint64_t title_id) {
    static constexpr char kDigits[] = "0123456789ABCDEF";
    std::string text(16, '0');
    for (std::size_t i = 0; i < text.size(); ++i) {
        text[text.size() - 1 - i] = kDigits[title_id & 0xF];
        title_id >>= 4;
    }
    return text;
}

const ContentRecord* ContentSet::find_content(ContentRecordType type) const noexcept {
    const auto match = std::find_if(content_records.begin(), content_records.end(),
                                    [&](const ContentRecord& record) {
                                        return record.type == type;
                                    });
    return match == content_records.end() ? nullptr : &*match;
}

std::variant<InputsFailure, Inputs> classify_inputs(
    const std::vector<std::filesystem::path>& input_paths, const InputsOptions& options) {
    try {
        return classify_inputs_impl(input_paths, options);
    } catch (const std::bad_alloc&) {
        return InputsFailure{"Host memory for pack input classification could not be allocated."};
    }
}

} // namespace armigo::pack
