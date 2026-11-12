// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/package.h>

#include "package_internal.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace armigo::shader_native {

static_assert(maximum_native_source_bytes == shader::package::detail::maximum_source_size);
static_assert(maximum_unstored_source_bytes == shader::package::detail::maximum_library_size);
namespace {

namespace fs = std::filesystem;
namespace detail = armigo::shader::package::detail;
using detail::Json;
using detail::JsonNumber;

constexpr std::uint64_t maximum_count = 1'000'000'000ull;

bool hex64(std::string_view text) {
    if (text.size() != 64) return false;
    for (const unsigned char character : text) {
        if (std::isupper(character)) return false;
        if (!std::isxdigit(character)) return false;
    }
    return true;
}

bool refusal_key(std::string_view text) {
    if (text.empty() || text.size() > 64) return false;
    for (const unsigned char character : text) {
        if (std::isalnum(character) || character == '-' || character == '_') continue;
        return false;
    }
    return true;
}

const char* kind_name(NativeStageKind stage) {
    switch (stage) {
        case NativeStageKind::vertex: return "vertex";
        case NativeStageKind::fragment: return "fragment";
        case NativeStageKind::compute: return "compute";
        case NativeStageKind::task: return "task";
        case NativeStageKind::mesh: return "mesh";
    }
    return "";
}

std::optional<NativeStageKind> parse_kind(std::string_view text) {
    if (text == "vertex") return NativeStageKind::vertex;
    if (text == "fragment") return NativeStageKind::fragment;
    if (text == "compute") return NativeStageKind::compute;
    if (text == "task") return NativeStageKind::task;
    if (text == "mesh") return NativeStageKind::mesh;
    return std::nullopt;
}

// A mesh stage is stored once per depth convention, as a vertex stage is.
// A task stage has the single depth of a fragment or compute stage.
bool depth_variants(NativeStageKind stage) {
    return stage == NativeStageKind::vertex || stage == NativeStageKind::mesh;
}

const char* depth_name(NativeDepth depth) {
    switch (depth) {
        case NativeDepth::stage: return "stage";
        case NativeDepth::minus_one_to_one: return "minus-one-to-one";
        case NativeDepth::zero_to_one: return "zero-to-one";
    }
    return "";
}

std::optional<NativeDepth> parse_depth(std::string_view text) {
    if (text == "stage") return NativeDepth::stage;
    if (text == "minus-one-to-one") return NativeDepth::minus_one_to_one;
    if (text == "zero-to-one") return NativeDepth::zero_to_one;
    return std::nullopt;
}

std::string slot_directory(std::string_view identity, NativeDepth depth) {
    return std::string("stages/") + std::string(identity) + "/" + depth_name(depth);
}

Json count_object(const std::map<std::string, std::uint64_t>& counts, bool exact_stages) {
    Json::Object object;
    if (exact_stages) {
        for (const char* stage : {"compute", "fragment", "vertex"}) {
            const auto found = counts.find(stage);
            object.emplace(stage, Json{JsonNumber{std::to_string(found == counts.end() ? 0 : found->second)}});
        }
    } else {
        for (const auto& [key, value] : counts) object.emplace(key, Json{JsonNumber{std::to_string(value)}});
    }
    return Json{std::move(object)};
}

Json number(std::uint64_t value) { return Json{JsonNumber{std::to_string(value)}}; }

std::uint64_t take_count(const Json& object, std::string_view key) {
    const Json& value = detail::member(object, key);
    detail::require(detail::is_uint(value, maximum_count), "Native shader count is not an integer");
    return detail::uint_value(value);
}

std::map<std::string, std::uint64_t> take_stage_counts(const Json& object, std::string_view key) {
    const Json& value = detail::member(object, key);
    static constexpr std::string_view keys[] = {"compute", "fragment", "vertex"};
    detail::require_keys(value, keys, "native shader stage counts");
    std::map<std::string, std::uint64_t> counts;
    for (const std::string_view stage : keys) counts.emplace(stage, take_count(value, stage));
    return counts;
}

std::map<std::string, std::uint64_t> take_refusals(const Json& object) {
    const Json& value = detail::member(object, "refusals");
    detail::require(value.is_object(), "Native shader refusals are not an object");
    std::map<std::string, std::uint64_t> counts;
    for (const auto& [key, item] : value.object()) {
        detail::require(refusal_key(key), "Native shader refusal name is not admitted");
        detail::require(detail::is_uint(item, maximum_count), "Native shader refusal count is not an integer");
        counts.emplace(key, detail::uint_value(item));
    }
    return counts;
}

void require_stage_counts(const std::map<std::string, std::uint64_t>& counts) {
    detail::require(counts.size() == 3 && counts.contains("vertex") && counts.contains("fragment") &&
                        counts.contains("compute"),
                    "Native shader stage counts must name vertex, fragment and compute");
    for (const auto& [key, value] : counts) {
        detail::require(value <= maximum_count, "Native shader stage count exceeds its policy");
    }
}

std::string validate_stage(const NativeStageWrite& stage, bool write_source) {
    try {
        detail::require(hex64(stage.stage_identity), "Native stage identity must be 64 lowercase hexadecimal digits");
        detail::require(!stage.program.empty() && stage.program.size() <= detail::maximum_program_size,
                        "Native program size is outside the identity bounds");
        detail::require(!stage.variants.empty() && stage.variants.size() <= 2, "Native stage variant count is invalid");
        std::vector<NativeDepth> seen;
        for (const NativeVariantWrite& variant : stage.variants) {
            detail::require(std::find(seen.begin(), seen.end(), variant.depth) == seen.end(),
                            "Native stage repeats a depth");
            seen.push_back(variant.depth);
            const bool depths = depth_variants(stage.stage);
            detail::require(depths ? variant.depth != NativeDepth::stage : variant.depth == NativeDepth::stage,
                            "Native stage depth does not match its stage");
            detail::require(!variant.reflection.empty() && variant.reflection.size() <= detail::maximum_reflection_size,
                            "Native reflection exceeds its size policy");
            detail::require(!variant.source.empty() &&
                                variant.source.size() <= (write_source ? maximum_native_source_bytes
                                                                       : maximum_unstored_source_bytes),
                            "Native Metal source exceeds its size policy");
            detail::require(!variant.library.empty() && variant.library.size() <= detail::maximum_library_size,
                            "Native Metal library exceeds its size policy");
        }
        if (!depth_variants(stage.stage)) detail::require(stage.variants.size() == 1, "Native stage variant count is invalid");
        return {};
    } catch (const std::exception& error) {
        return error.what();
    }
}

Json variant_json(const NativeEntry& entry, const NativeVariant& variant) {
    Json::Object object;
    object.emplace("depth", Json{std::string(depth_name(variant.depth))});
    object.emplace("directory", Json{slot_directory(entry.stage_identity, variant.depth)});
    object.emplace("library_digest", Json{variant.library_digest});
    object.emplace("library_size", number(variant.library_size));
    object.emplace("reflection_digest", Json{variant.reflection_digest});
    object.emplace("reflection_size", number(variant.reflection_size));
    object.emplace("source_digest", Json{variant.source_digest});
    object.emplace("source_size", number(variant.source_size));
    return Json{std::move(object)};
}

Json entry_json(const NativeEntry& entry) {
    Json::Array variants;
    for (const NativeVariant& variant : entry.variants) variants.push_back(variant_json(entry, variant));
    Json::Object object;
    object.emplace("program_digest", Json{entry.program_digest});
    object.emplace("program_size", number(entry.program_size));
    object.emplace("stage", Json{std::string(kind_name(entry.stage))});
    object.emplace("stage_identity", Json{entry.stage_identity});
    object.emplace("variants", Json{std::move(variants)});
    return Json{std::move(object)};
}

Json section_json(const NativeSection& section) {
    Json::Array depths;
    if (section.specialization.vertex_depth_minus_one_to_one) depths.push_back(Json{std::string("minus-one-to-one")});
    if (section.specialization.vertex_depth_zero_to_one) depths.push_back(Json{std::string("zero-to-one")});
    Json::Object specialization;
    specialization.emplace("fragment_position",
                           Json{std::string(section.specialization.fragment_position_matches ? "metal-position-w"
                                                                                             : "unspecified")});
    specialization.emplace("pixel_rate", Json{std::string(section.specialization.pixel_rate_disabled ? "disabled" : "enabled")});
    specialization.emplace("vertex_depths", Json{std::move(depths)});
    specialization.emplace("vertex_fetch", Json{section.specialization.vertex_fetch});
    Json::Object counts;
    counts.emplace("by_stage", count_object(section.counts.by_stage, true));
    counts.emplace("compiled", number(section.counts.compiled));
    counts.emplace("compiled_by_stage", count_object(section.counts.compiled_by_stage, true));
    counts.emplace("duplicates", number(section.counts.duplicates));
    counts.emplace("enumerated", number(section.counts.enumerated));
    counts.emplace("refusals", count_object(section.counts.refusals, false));
    counts.emplace("translated", number(section.counts.translated));
    counts.emplace("translated_by_stage", count_object(section.counts.translated_by_stage, true));
    counts.emplace("unique", number(section.counts.unique));
    Json::Array entries;
    for (const NativeEntry& entry : section.entries) entries.push_back(entry_json(entry));
    Json::Object root;
    root.emplace("counts", Json{std::move(counts)});
    root.emplace("entries", Json{std::move(entries)});
    root.emplace("schema", Json{std::string(native_section_schema)});
    root.emplace("specialization", Json{std::move(specialization)});
    return Json{std::move(root)};
}

NativeVariant parse_variant(const Json& value, std::string_view identity) {
    static constexpr std::string_view keys[] = {"depth", "directory", "library_digest", "library_size",
                                                 "reflection_digest", "reflection_size", "source_digest", "source_size"};
    detail::require_keys(value, keys, "native shader variant");
    detail::require(detail::member(value, "depth").is_string(), "Native shader variant depth is not a string");
    const auto depth = parse_depth(detail::member(value, "depth").string());
    detail::require(depth.has_value(), "Native shader variant depth is not admitted");
    detail::require(detail::member(value, "directory").is_string(), "Native shader variant directory is not a string");
    detail::require(detail::member(value, "directory").string() == slot_directory(identity, *depth),
                    "Native shader variant directory does not match its identity");
    NativeVariant variant;
    variant.depth = *depth;
    for (const std::string_view digest : {"library_digest", "reflection_digest", "source_digest"}) {
        detail::require(detail::member(value, digest).is_string() && hex64(detail::member(value, digest).string()),
                        "Native shader digest must be 64 lowercase hexadecimal digits");
    }
    variant.library_digest = detail::member(value, "library_digest").string();
    variant.reflection_digest = detail::member(value, "reflection_digest").string();
    variant.source_digest = detail::member(value, "source_digest").string();
    variant.library_size = take_count(value, "library_size");
    variant.reflection_size = take_count(value, "reflection_size");
    variant.source_size = take_count(value, "source_size");
    detail::require(variant.library_size > 0 && variant.library_size <= detail::maximum_library_size &&
                        variant.reflection_size > 0 && variant.reflection_size <= detail::maximum_reflection_size &&
                        variant.source_size > 0 && variant.source_size <= maximum_unstored_source_bytes,
                    "Native shader variant size is outside its policy");
    return variant;
}

NativeEntry parse_entry(const Json& value) {
    static constexpr std::string_view keys[] = {"program_digest", "program_size", "stage", "stage_identity", "variants"};
    detail::require_keys(value, keys, "native shader entry");
    detail::require(detail::member(value, "stage_identity").is_string() && hex64(detail::member(value, "stage_identity").string()),
                    "Native stage identity must be 64 lowercase hexadecimal digits");
    detail::require(detail::member(value, "program_digest").is_string() && hex64(detail::member(value, "program_digest").string()),
                    "Native program digest must be 64 lowercase hexadecimal digits");
    detail::require(detail::member(value, "stage").is_string(), "Native shader stage is not a string");
    const auto stage = parse_kind(detail::member(value, "stage").string());
    detail::require(stage.has_value(), "Native shader stage is not admitted");
    NativeEntry entry;
    entry.stage_identity = detail::member(value, "stage_identity").string();
    entry.program_digest = detail::member(value, "program_digest").string();
    entry.program_size = take_count(value, "program_size");
    entry.stage = *stage;
    detail::require(entry.program_size > 0 && entry.program_size <= detail::maximum_program_size,
                    "Native program size is outside the identity bounds");
    const Json& variants = detail::member(value, "variants");
    detail::require(variants.is_array() && !variants.array().empty() && variants.array().size() <= 2,
                    "Native stage variant count is invalid");
    for (const Json& variant : variants.array()) entry.variants.push_back(parse_variant(variant, entry.stage_identity));
    std::vector<NativeDepth> seen;
    for (const NativeVariant& variant : entry.variants) {
        detail::require(std::find(seen.begin(), seen.end(), variant.depth) == seen.end(), "Native stage repeats a depth");
        seen.push_back(variant.depth);
        const bool depths = depth_variants(entry.stage);
        detail::require(depths ? variant.depth != NativeDepth::stage : variant.depth == NativeDepth::stage && entry.variants.size() == 1,
                        "Native stage depth does not match its stage");
    }
    return entry;
}

NativeSection parse_section(const Json& root) {
    static constexpr std::string_view root_keys[] = {"counts", "entries", "schema", "specialization"};
    detail::require_keys(root, root_keys, "native shader manifest");
    detail::require(detail::member(root, "schema").is_string() &&
                        detail::member(root, "schema").string() == native_section_schema,
                    "Native shader manifest schema is not admitted");
    const Json& specialization = detail::member(root, "specialization");
    static constexpr std::string_view specialization_keys[] = {"fragment_position", "pixel_rate", "vertex_depths", "vertex_fetch"};
    detail::require_keys(specialization, specialization_keys, "native shader specialization");
    NativeSection section;
    detail::require(detail::member(specialization, "pixel_rate").is_string(), "Native pixel rate is not a string");
    const std::string& pixel_rate = detail::member(specialization, "pixel_rate").string();
    detail::require(pixel_rate == "disabled" || pixel_rate == "enabled", "Native pixel rate is not admitted");
    section.specialization.pixel_rate_disabled = pixel_rate == "disabled";
    detail::require(detail::member(specialization, "vertex_fetch").is_bool(), "Native vertex fetch is not a boolean");
    section.specialization.vertex_fetch = detail::member(specialization, "vertex_fetch").boolean();
    detail::require(detail::member(specialization, "fragment_position").is_string(), "Native fragment position is not a string");
    const std::string& position = detail::member(specialization, "fragment_position").string();
    detail::require(position == "metal-position-w" || position == "unspecified", "Native fragment position is not admitted");
    section.specialization.fragment_position_matches = position == "metal-position-w";
    section.specialization.vertex_depth_minus_one_to_one = false;
    section.specialization.vertex_depth_zero_to_one = false;
    const Json& depths = detail::member(specialization, "vertex_depths");
    detail::require(depths.is_array() && depths.array().size() <= 2, "Native vertex depths are not admitted");
    for (const Json& depth : depths.array()) {
        detail::require(depth.is_string(), "Native vertex depth is not a string");
        if (depth.string() == "minus-one-to-one") {
            detail::require(!section.specialization.vertex_depth_minus_one_to_one, "Native vertex depths repeat a value");
            section.specialization.vertex_depth_minus_one_to_one = true;
        } else if (depth.string() == "zero-to-one") {
            detail::require(!section.specialization.vertex_depth_zero_to_one, "Native vertex depths repeat a value");
            section.specialization.vertex_depth_zero_to_one = true;
        } else {
            detail::require(false, "Native vertex depth is not admitted");
        }
    }
    const Json& counts = detail::member(root, "counts");
    static constexpr std::string_view count_keys[] = {"by_stage", "compiled", "compiled_by_stage", "duplicates", "enumerated",
                                                       "refusals", "translated", "translated_by_stage", "unique"};
    detail::require_keys(counts, count_keys, "native shader counts");
    section.counts.enumerated = take_count(counts, "enumerated");
    section.counts.unique = take_count(counts, "unique");
    section.counts.translated = take_count(counts, "translated");
    section.counts.compiled = take_count(counts, "compiled");
    section.counts.duplicates = take_count(counts, "duplicates");
    section.counts.by_stage = take_stage_counts(counts, "by_stage");
    section.counts.translated_by_stage = take_stage_counts(counts, "translated_by_stage");
    section.counts.compiled_by_stage = take_stage_counts(counts, "compiled_by_stage");
    section.counts.refusals = take_refusals(counts);
    const Json& entries = detail::member(root, "entries");
    detail::require(entries.is_array() && entries.array().size() <= detail::maximum_identities,
                    "Native shader entry count exceeds the identity policy");
    std::vector<std::string> identities;
    for (const Json& entry : entries.array()) {
        section.entries.push_back(parse_entry(entry));
        identities.push_back(section.entries.back().stage_identity);
    }
    detail::require(std::is_sorted(identities.begin(), identities.end()) &&
                        std::adjacent_find(identities.begin(), identities.end()) == identities.end(),
                    "Native shader entries must be sorted and unique");
    return section;
}

std::span<const std::byte> bytes_of(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

void write_bytes(const fs::path& path, std::span<const std::byte> bytes) { detail::write_file(path, bytes); }

bool contained(const fs::path& path, const fs::path& root) {
    const auto compared = std::mismatch(root.begin(), root.end(), path.begin(), path.end());
    return compared.first == root.end();
}

} // namespace

struct NativeSectionWriter::State {
    fs::path root;
    fs::path native;
    std::vector<NativeEntry> entries;
    bool write_source = true;
    bool finished = false;
};

struct NativePackage::State {
    // Program digest plus stage kind. A repeated pair keeps the first identity.
    struct ProgramKey {
        std::string digest;
        NativeStageKind stage = NativeStageKind::fragment;
        bool operator<(const ProgramKey& other) const {
            if (digest != other.digest) return digest < other.digest;
            return static_cast<unsigned>(stage) < static_cast<unsigned>(other.stage);
        }
    };

    fs::path native;
    NativeSection section;
    std::map<std::string, std::size_t, std::less<>> index;
    mutable std::once_flag program_index_once;
    mutable std::map<ProgramKey, std::string> program_index;
};

NativeSectionWriter::NativeSectionWriter(std::unique_ptr<State> state) : state_(std::move(state)) {}
NativeSectionWriter::NativeSectionWriter(NativeSectionWriter&&) noexcept = default;
NativeSectionWriter& NativeSectionWriter::operator=(NativeSectionWriter&&) noexcept = default;
NativeSectionWriter::~NativeSectionWriter() = default;
NativePackage::NativePackage(std::unique_ptr<State> state) : state_(std::move(state)) {}
NativePackage::NativePackage(NativePackage&&) noexcept = default;
NativePackage& NativePackage::operator=(NativePackage&&) noexcept = default;
NativePackage::~NativePackage() = default;

std::variant<std::string, NativeSectionWriter> NativeSectionWriter::create(const fs::path& package_root, bool write_source) {
    try {
        std::error_code error;
        detail::require(fs::is_directory(package_root, error) && !fs::is_symlink(package_root, error),
                        "Native shader package root must be a directory");
        const fs::path native = package_root / "native";
        detail::require(!fs::exists(native, error), "Native shader section already exists");
        fs::create_directories(native / "stages", error);
        detail::require(!error, "Cannot create the native shader section");
        auto state = std::make_unique<State>();
        state->root = package_root;
        state->native = native;
        state->write_source = write_source;
        return NativeSectionWriter(std::move(state));
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
}

std::variant<std::string, std::monostate> NativeSectionWriter::add(const NativeStageWrite& stage) {
    try {
        detail::require(state_ && !state_->finished, "Native shader section is already finished");
        if (const std::string failure = validate_stage(stage, state_->write_source); !failure.empty()) return failure;
        for (const NativeEntry& existing : state_->entries) {
            detail::require(existing.stage_identity != stage.stage_identity, "Native shader section repeats a stage identity");
        }
        const fs::path stage_root = state_->native / "stages" / stage.stage_identity;
        std::error_code error;
        fs::create_directories(stage_root, error);
        detail::require(!error, "Cannot create a native shader stage");
        write_bytes(stage_root / "program.bin", stage.program);
        NativeEntry entry;
        entry.stage_identity = stage.stage_identity;
        entry.stage = stage.stage;
        entry.program_digest = detail::sha256(stage.program);
        entry.program_size = stage.program.size();
        for (const NativeVariantWrite& variant : stage.variants) {
            const fs::path directory = stage_root / depth_name(variant.depth);
            fs::create_directory(directory, error);
            detail::require(!error && fs::is_directory(directory, error), "Cannot create a native shader variant");
            write_bytes(directory / "reflection.json", variant.reflection);
            // The digest is recorded either way. A packed stage omits the file.
            if (state_->write_source) write_bytes(directory / "source.metal", bytes_of(variant.source));
            write_bytes(directory / "shader.metallib", variant.library);
            NativeVariant stored;
            stored.depth = variant.depth;
            stored.reflection_digest = detail::sha256(variant.reflection);
            stored.source_digest = detail::sha256(bytes_of(variant.source));
            stored.library_digest = detail::sha256(variant.library);
            stored.reflection_size = variant.reflection.size();
            stored.source_size = variant.source.size();
            stored.library_size = variant.library.size();
            entry.variants.push_back(std::move(stored));
        }
        std::sort(entry.variants.begin(), entry.variants.end(), [](const NativeVariant& left, const NativeVariant& right) {
            return std::string_view(depth_name(left.depth)) < depth_name(right.depth);
        });
        state_->entries.push_back(std::move(entry));
        return std::monostate{};
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
}

// The Maxwell reflection cap is one mebibyte. A native manifest names every
// admitted program, so it has its own bound and is rejected before it is published.
constexpr std::size_t maximum_manifest_bytes = 128u * 1024u * 1024u;

std::variant<std::string, NativeSection> NativeSectionWriter::finish(NativeCounts counts, NativeSpecialization specialization) {
    try {
        detail::require(state_ && !state_->finished, "Native shader section is already finished");
        require_stage_counts(counts.by_stage);
        require_stage_counts(counts.translated_by_stage);
        require_stage_counts(counts.compiled_by_stage);
        for (const auto& [key, value] : counts.refusals) {
            detail::require(refusal_key(key) && value <= maximum_count, "Native shader refusal is not admitted");
        }
        for (const std::uint64_t value : {counts.enumerated, counts.unique, counts.translated, counts.compiled, counts.duplicates}) {
            detail::require(value <= maximum_count, "Native shader count exceeds its policy");
        }
        NativeSection section;
        section.specialization = specialization;
        section.counts = std::move(counts);
        section.entries = std::move(state_->entries);
        std::sort(section.entries.begin(), section.entries.end(), [](const NativeEntry& left, const NativeEntry& right) {
            return left.stage_identity < right.stage_identity;
        });
        const std::string text = detail::canonical_json(section_json(section)) + "\n";
        detail::require(text.size() <= maximum_manifest_bytes, "Native shader manifest exceeds its size policy");
        const fs::path temporary = state_->native / "manifest.json.tmp";
        write_bytes(temporary, bytes_of(text));
        std::error_code error;
        fs::rename(temporary, state_->native / "manifest.json", error);
        detail::require(!error, "Cannot publish the native shader manifest");
        state_->finished = true;
        return section;
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
}

std::variant<std::string, NativePackage> NativePackage::open(const fs::path& package_root) {
    try {
        std::error_code error;
        const fs::path root = fs::weakly_canonical(package_root, error);
        detail::require(!error && fs::is_directory(root, error), "Native shader package root must be a directory");
        const fs::path native = root / "native";
        const fs::path manifest = native / "manifest.json";
        detail::require(fs::is_regular_file(manifest, error) && !fs::is_symlink(manifest, error),
                        "The package has no native shader section");
        const auto bytes = detail::read_regular(manifest, maximum_manifest_bytes, "native shader manifest");
        const Json parsed = detail::parse_json(bytes);
        auto state = std::make_unique<State>();
        state->native = fs::canonical(native, error);
        detail::require(!error, "Cannot resolve the native shader section");
        state->section = parse_section(parsed);
        for (std::size_t index = 0; index < state->section.entries.size(); ++index) {
            state->index.emplace(state->section.entries[index].stage_identity, index);
        }
        return NativePackage(std::move(state));
    } catch (const std::exception& error) {
        return std::string(error.what());
    }
}

const NativeSection& NativePackage::section() const { return state_->section; }

std::optional<RuntimeStageIdentity> parse_runtime_stage_identity(std::string_view text) {
    const auto first = text.find('.');
    if (first != 1) return std::nullopt;
    const auto second = text.find('.', first + 1);
    if (second == std::string_view::npos || text.find('.', second + 1) != std::string_view::npos) return std::nullopt;
    const auto digest = text.substr(first + 1, second - first - 1);
    if (!hex64(digest)) return std::nullopt;
    const auto suffix = text.substr(second + 1);
    RuntimeStageIdentity parsed;
    parsed.program_digest.assign(digest);
    switch (text[0]) {
    case '0': parsed.stage = NativeStageKind::vertex; break;
    case '1': parsed.stage = NativeStageKind::fragment; break;
    case '3': parsed.stage = NativeStageKind::task; break;
    case '4': parsed.stage = NativeStageKind::mesh; break;
    case '5': parsed.stage = NativeStageKind::compute; break;
    default: return std::nullopt;
    }
    if (parsed.stage == NativeStageKind::vertex || parsed.stage == NativeStageKind::mesh) {
        if (suffix == "1") parsed.vertex_depth_zero_to_one = true;
        else if (suffix == "2") parsed.vertex_depth_zero_to_one = false;
        else return std::nullopt;
    }
    return parsed;
}

std::optional<std::string> NativePackage::identity_for_program(std::string_view program_digest_hex,
                                                               NativeStageKind stage) const {
    std::call_once(state_->program_index_once, [this] {
        for (const NativeEntry& entry : state_->section.entries)
            state_->program_index.emplace(State::ProgramKey{entry.program_digest, entry.stage}, entry.stage_identity);
    });
    const auto found = state_->program_index.find(State::ProgramKey{std::string(program_digest_hex), stage});
    if (found == state_->program_index.end()) return std::nullopt;
    return found->second;
}

std::variant<NativeStageFiles, NativeMiss> NativePackage::find(std::string_view stage_identity, const NativeLookup& lookup) const {
    try {
        const NativeSpecialization& specialization = state_->section.specialization;
        if (lookup.pixel_rate_disabled != specialization.pixel_rate_disabled) {
            return NativeMiss{NativeMiss::Reason::specialization, "pixel rate"};
        }
        if (lookup.stage == NativeStageKind::vertex && lookup.vertex_fetch != specialization.vertex_fetch) {
            return NativeMiss{NativeMiss::Reason::specialization, "vertex fetch"};
        }
        if (lookup.stage == NativeStageKind::fragment &&
            lookup.fragment_position_matches != specialization.fragment_position_matches) {
            return NativeMiss{NativeMiss::Reason::specialization, "fragment position"};
        }
        // A mesh stage selects its library by the vertex depth convention.
        const bool depths = depth_variants(lookup.stage);
        if (depths) {
            const bool admitted = lookup.vertex_depth_zero_to_one ? specialization.vertex_depth_zero_to_one
                                                                   : specialization.vertex_depth_minus_one_to_one;
            if (!admitted) return NativeMiss{NativeMiss::Reason::specialization, "vertex depth"};
        }
        const auto found = state_->index.find(stage_identity);
        if (found == state_->index.end()) return NativeMiss{NativeMiss::Reason::absent, {}};
        const NativeEntry& entry = state_->section.entries[found->second];
        if (entry.stage != lookup.stage) return NativeMiss{NativeMiss::Reason::absent, "stage"};
        const NativeDepth depth = depths ? (lookup.vertex_depth_zero_to_one ? NativeDepth::zero_to_one : NativeDepth::minus_one_to_one)
                                          : NativeDepth::stage;
        const auto variant = std::find_if(entry.variants.begin(), entry.variants.end(),
                                           [&](const NativeVariant& candidate) { return candidate.depth == depth; });
        if (variant == entry.variants.end()) return NativeMiss{NativeMiss::Reason::absent, "depth"};
        const fs::path stage_root = state_->native / "stages" / entry.stage_identity;
        const fs::path directory = stage_root / depth_name(depth);
        std::error_code stage_error;
        const fs::path canonical_stage = fs::canonical(stage_root, stage_error);
        std::error_code variant_error;
        const fs::path canonical_variant = fs::canonical(directory, variant_error);
        if (stage_error || variant_error || !contained(canonical_stage, state_->native) ||
            !contained(canonical_variant, canonical_stage)) {
            return NativeMiss{NativeMiss::Reason::corrupt, "Native shader stage path leaves the package"};
        }
        NativeStageFiles files;
        files.program = detail::read_regular(canonical_stage / "program.bin", detail::maximum_program_size, "program.bin");
        files.reflection = detail::read_regular(canonical_variant / "reflection.json", detail::maximum_reflection_size, "reflection.json");
        files.library = detail::read_regular(canonical_variant / "shader.metallib", detail::maximum_library_size, "shader.metallib");
        // Metal source is a compiler input. A published stage may omit
        // source.metal; its digest stays in the reflection and is not hashed
        // again unless the file is present. A symlink is not that omission.
        bool source_present = false;
        {
            const fs::path source_path = canonical_variant / "source.metal";
            std::error_code symlink_error;
            std::error_code exists_error;
            const bool symlink = fs::is_symlink(source_path, symlink_error);
            const bool exists = fs::exists(source_path, exists_error);
            if (!symlink && !exists && exists_error) {
                return NativeMiss{NativeMiss::Reason::corrupt, exists_error.message()};
            }
            if (symlink || exists) {
                const auto source = detail::read_regular(source_path, detail::maximum_source_size, "source.metal");
                files.source.assign(reinterpret_cast<const char*>(source.data()), source.size());
                source_present = true;
            }
        }
        const bool source_mismatch = source_present && (files.source.size() != variant->source_size ||
                                                        detail::sha256(bytes_of(files.source)) != variant->source_digest);
        if (files.program.size() != entry.program_size || detail::sha256(files.program) != entry.program_digest ||
            files.reflection.size() != variant->reflection_size || detail::sha256(files.reflection) != variant->reflection_digest ||
            source_mismatch ||
            files.library.size() != variant->library_size || detail::sha256(files.library) != variant->library_digest) {
            return NativeMiss{NativeMiss::Reason::corrupt, "Native shader stage digest does not match the manifest"};
        }
        return files;
    } catch (const std::exception& error) {
        return NativeMiss{NativeMiss::Reason::corrupt, error.what()};
    }
}

} // namespace armigo::shader_native
