// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "prepared_artifact.h"
#include "preparation_analysis.h"
#include "sha256.h"

#include <armigo/execution/gate_code.h>
#include <armigo/linker/link_plan.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace armigo::runtime {
namespace {
using execution::PreparedLocation;
using preparation::ModulePayload;
constexpr std::string_view file_magic = "ARMGPREP";
constexpr std::string_view image_magic = "ARMGIMG1";
constexpr std::string_view artifact_file = "modules.armigo-prepared";
constexpr std::uint64_t synthetic_base = 0x100000000ull;
constexpr std::uint64_t synthetic_stride = 0x100000000ull;
constexpr std::size_t max_modules = 64;
constexpr std::size_t max_records = 1'000'000;

struct FormatError : std::runtime_error { using std::runtime_error::runtime_error; };
struct Bytes {
    std::vector<std::byte> data;
    void number(std::uint64_t value, unsigned size) {
        if (size > default_prepared_artifact_limit - data.size())
            throw FormatError("Prepared artifact exceeds the format size limit.");
        for (unsigned i = 0; i < size; ++i) data.push_back(std::byte(value >> (i * 8)));
    }
    void raw(std::span<const std::byte> bytes) {
        if (bytes.size() > default_prepared_artifact_limit - data.size())
            throw FormatError("Prepared artifact exceeds the format size limit.");
        data.insert(data.end(), bytes.begin(), bytes.end());
    }
    void string(std::string_view value) {
        number(value.size(), 4);
        raw(std::as_bytes(std::span(value.data(), value.size())));
    }
    void magic(std::string_view value) {
        raw(std::as_bytes(std::span(value.data(), value.size())));
    }
    void checksum() {
        const auto hash = loader::detail::sha256(data);
        if (!hash) throw FormatError("Prepared artifact hashing failed.");
        raw(*hash);
    }
};
struct Reader {
    std::span<const std::byte> data;
    std::size_t cursor = 0;
    std::span<const std::byte> take(std::size_t size) {
        if (size > data.size() - cursor) throw FormatError("Truncated prepared artifact.");
        const auto result = data.subspan(cursor, size); cursor += size; return result;
    }
    std::uint64_t number(unsigned size) {
        std::uint64_t value = 0; const auto bytes = take(size);
        for (unsigned i = 0; i < size; ++i) value |= std::uint64_t(bytes[i]) << (i * 8);
        return value;
    }
    std::size_t count(std::size_t minimum_record_bytes, std::size_t maximum = max_records) {
        const auto value = number(4);
        if (value > maximum || value > (data.size() - cursor) / minimum_record_bytes)
            throw FormatError("Prepared artifact record count exceeds its bound.");
        return static_cast<std::size_t>(value);
    }
    std::string string(std::size_t maximum) {
        const auto size = number(4);
        if (size > maximum) throw FormatError("Prepared artifact string exceeds its bound.");
        const auto bytes = take(static_cast<std::size_t>(size));
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    void magic(std::string_view expected) {
        const auto bytes = take(expected.size());
        if (std::memcmp(bytes.data(), expected.data(), expected.size()) != 0)
            throw FormatError("Unrecognized prepared artifact encoding.");
    }
    void end() { if (cursor != data.size()) throw FormatError("Trailing prepared artifact data."); }
};
Reader checked_reader(std::span<const std::byte> bytes) {
    if (bytes.size() < 32) throw FormatError("Missing prepared artifact checksum.");
    const auto body = bytes.first(bytes.size() - 32);
    const auto hash = loader::detail::sha256(body);
    if (!hash || !std::equal(hash->begin(), hash->end(), bytes.end() - 32))
        throw FormatError("Prepared artifact checksum mismatch.");
    return {body};
}
void location(Bytes& bytes, PreparedLocation value) {
    bytes.number(value.module_index, 4); bytes.number(value.image_offset, 8);
}
PreparedLocation location(Reader& reader) {
    const auto index = static_cast<std::uint32_t>(reader.number(4));
    return {index, reader.number(8)};
}
void patches(Bytes& bytes, const std::vector<execution::PreparedWordPatch>& values) {
    bytes.number(values.size(), 4);
    for (const auto& value : values) {
        bytes.number(value.image_offset, 8); bytes.number(value.expected_word, 4);
        bytes.number(value.replacement_word, 4);
    }
}
std::vector<execution::PreparedWordPatch> patches(Reader& reader) {
    std::vector<execution::PreparedWordPatch> values(reader.count(16));
    for (auto& value : values) {
        value.image_offset = reader.number(8);
        value.expected_word = static_cast<std::uint32_t>(reader.number(4));
        value.replacement_word = static_cast<std::uint32_t>(reader.number(4));
    }
    return values;
}
std::vector<std::byte> encode_payload(const ModulePayload& payload) {
    Bytes bytes; bytes.magic(image_magic); bytes.number(2, 4);
    bytes.number(payload.original_process ? 1 : 0, 4);
    bytes.number(payload.load_rank, 4); bytes.number(payload.text_offset, 8);
    bytes.number(payload.text_size, 8); bytes.number(payload.roots.size(), 4);
    for (const auto root : payload.roots) bytes.number(root, 8);
    const auto& image = payload.image;
    bytes.number(image.module_index, 4); bytes.number(image.original_image_size, 8);
    bytes.number(image.appendix_offset, 8);
    patches(bytes, image.coverage); patches(bytes, image.replacements);
    bytes.number(image.appendix.size(), 8); bytes.raw(image.appendix);
    bytes.number(image.gate_sites.size(), 4);
    for (const auto& site : image.gate_sites) {
        bytes.number(site.return_address_offset, 8); bytes.number(site.instruction, 4);
        bytes.number(site.allowed_indirect_targets.size(), 4);
        for (const auto& target : site.allowed_indirect_targets) location(bytes, target);
    }
    bytes.number(image.fixups.size(), 4);
    for (const auto& fixup : image.fixups) {
        bytes.number(static_cast<unsigned>(fixup.kind), 4);
        bytes.number(fixup.image_offset, 8); location(bytes, fixup.target);
    }
    bytes.checksum(); return std::move(bytes.data);
}
std::size_t limit(const PreparationOptions& options) {
    return options.max_artifact_bytes == 0 ? default_prepared_artifact_limit :
        std::min(options.max_artifact_bytes, default_prepared_artifact_limit);
}
bool identity_matches(const PreparationOptions& options, std::string_view installed) {
    return options.required_preparation_identity.empty() ||
        options.required_preparation_identity == installed;
}
bool valid_name(std::string_view name) {
    return !name.empty() && name.size() <= 256 && name.find_first_of("/\\") == name.npos &&
        name.find('\0') == name.npos && name != "." && name != "..";
}
std::vector<std::byte> encode_artifacts(const PreparedModuleArtifacts& artifacts) {
    Bytes bytes; bytes.magic(file_magic); bytes.number(artifacts.format_version, 4);
    bytes.string(artifacts.preparation_identity); bytes.number(artifacts.modules.size(), 4);
    for (const auto& module : artifacts.modules) {
        bytes.string(module.module_name); bytes.raw(module.source_sha256);
        bytes.number(module.payload.size(), 8); bytes.raw(module.payload);
    }
    bytes.checksum(); return std::move(bytes.data);
}
LaunchFailure allocation_failure() {
    return preparation::failure(LaunchError::resource_limit, "prepared-artifact", "Preparation allocation failed.");
}
} // namespace

std::string installed_preparation_identity() {
    // Bump this algorithm revision whenever analysis, emission, or ABI changes.
    // The template digest independently prevents mixing different gate builds.
    const auto gate = execution::make_local_gate_template();
    const auto digest = loader::detail::sha256(gate.code);
    if (!digest) throw std::runtime_error("Native gate template hashing failed.");
    std::string identity = "armigo-a64-preparation-v1-r6-";
    constexpr char hex[] = "0123456789abcdef";
    for (const auto byte : *digest) {
        const auto value = std::to_integer<unsigned>(byte);
        identity.push_back(hex[value >> 4]); identity.push_back(hex[value & 15]);
    }
    return identity;
}

namespace preparation {
std::optional<std::vector<std::size_t>> original_module_order(
    const loader::ExeFsDirectoryInput& input) {
    static constexpr std::array<std::string_view, 13> names{
        "rtld", "main", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "subsdk4",
        "subsdk5", "subsdk6", "subsdk7", "subsdk8", "subsdk9", "sdk"};
    std::map<std::string_view, std::size_t, std::less<>> indices;
    for (std::size_t i = 0; i < input.modules.size(); ++i)
        if (!indices.emplace(input.modules[i].name, i).second) return {};
    if (!indices.contains("main")) return {};
    for (const auto& [name, index] : indices) {
        (void)index;
        if (std::find(names.begin(), names.end(), name) == names.end()) return {};
    }
    std::vector<std::size_t> order;
    bool subsdk_gap = false;
    for (std::size_t slot = 0; slot < names.size(); ++slot) {
        const auto found = indices.find(names[slot]);
        if (slot >= 2 && slot < 12) {
            if (found == indices.end()) subsdk_gap = true;
            if (subsdk_gap) continue;
        }
        if (found != indices.end()) order.push_back(found->second);
    }
    if (!indices.contains("rtld") && order.size() != 1) return {};
    return order;
}

std::optional<loader::ExeFsDirectoryInput> admitted_input(
    const loader::ExeFsDirectoryInput& input, PreparationMode mode) {
    if (mode == PreparationMode::synthetic_test) return {};
    const auto order = original_module_order(input);
    if (!order || order->size() == input.modules.size()) return {};
    loader::ExeFsDirectoryInput admitted;
    admitted.process_metadata = input.process_metadata;
    admitted.ignored_modules = input.ignored_modules;
    for (std::size_t i = 0; i < input.modules.size(); ++i) {
        if (std::find(order->begin(), order->end(), i) != order->end())
            admitted.modules.push_back(input.modules[i]);
        else admitted.ignored_modules.push_back(input.modules[i].name);
    }
    return admitted;
}
LaunchFailure failure(LaunchError code, std::string operation, std::string detail) {
    return {.stage = LaunchStage::preparation, .code = code,
        .operation = std::move(operation), .detail = std::move(detail)};
}
bool covers(const execution::PreparedImage& image, std::uint64_t offset) noexcept {
    const auto found = std::lower_bound(image.coverage.begin(), image.coverage.end(), offset,
        [](const auto& word, auto value) { return word.image_offset < value; });
    return found != image.coverage.end() && found->image_offset == offset;
}
LaunchResult<DecodedInput> decode_input(const loader::ExeFsDirectoryInput& input, PreparationMode mode) {
    if (auto admitted = admitted_input(input, mode)) return decode_input(*admitted, mode);
    if (input.modules.empty() || input.modules.size() > max_modules)
        return failure(LaunchError::invalid_input, "decode-modules", "Invalid input module count.");
    const auto canonical = mode == PreparationMode::synthetic_test
        ? std::optional<std::vector<std::size_t>>{} : original_module_order(input);
    if ((input.process_metadata || mode == PreparationMode::original_process) && !canonical)
        return failure(LaunchError::unsupported, "module-order",
            "Original process modules do not match the fixed executable slot roster.");
    DecodedInput result;
    result.original_process = canonical.has_value();
    std::map<std::string, std::size_t, std::less<>> names;
    std::size_t total_bytes = 0;
    for (std::size_t i = 0; i < input.modules.size(); ++i) {
        const auto& source = input.modules[i];
        if (!valid_name(source.name) || !names.emplace(source.name, i).second)
            return failure(LaunchError::invalid_input, "decode-modules", "Module names must be valid and unique.");
        loader::NsoReadOptions options;
        options.limits.max_image_bytes = 2ull * 1024 * 1024 * 1024 - total_bytes;
        auto read = loader::read_nso_module(source.bytes, options);
        if (!read.ok()) return failure(LaunchError::invalid_input, "decode-modules", loader::to_string(read.error.code));
        total_bytes += read.image->bytes.size();
        auto dynamic = linker::read_dynamic_module(read.image->bytes);
        if (!dynamic.ok()) return failure(LaunchError::invalid_input, "decode-modules", dynamic.message);
        result.images.push_back(std::move(*read.image));
        result.metadata.push_back(std::move(*dynamic.module));
    }
    if (canonical) { result.order = *canonical; return result; }
    std::vector<unsigned char> state(input.modules.size());
    std::function<bool(std::size_t)> visit = [&](std::size_t i) {
        if (state[i] == 1) return false;
        if (state[i] == 2) return true;
        state[i] = 1;
        for (const auto& dependency : result.metadata[i].dependencies) {
            const auto found = names.find(dependency);
            if (found == names.end() || !visit(found->second)) return false;
        }
        state[i] = 2; result.order.push_back(i); return true;
    };
    for (std::size_t i = 0; i < input.modules.size(); ++i) {
        if (!visit(i)) return failure(LaunchError::unsupported, "module-order",
            "A dependency is absent or cyclic; load ownership is unresolved.");
    }
    return result;
}
LaunchResult<ModulePayload> decode_payload(std::span<const std::byte> payload) {
    try {
        auto reader = checked_reader(payload); reader.magic(image_magic);
        if (reader.number(4) != 2) throw FormatError("Unsupported prepared image version.");
        ModulePayload result;
        const auto mode = reader.number(4);
        if (mode > 1) throw FormatError("Invalid prepared execution mode.");
        result.original_process = mode == 1;
        result.load_rank = static_cast<std::uint32_t>(reader.number(4));
        result.text_offset = reader.number(8); result.text_size = reader.number(8);
        result.roots.resize(reader.count(8));
        for (auto& root : result.roots) root = reader.number(8);
        auto& image = result.image;
        image.module_index = static_cast<std::uint32_t>(reader.number(4));
        image.original_image_size = reader.number(8); image.appendix_offset = reader.number(8);
        image.coverage = patches(reader); image.replacements = patches(reader);
        const auto appendix_size = reader.number(8);
        if (appendix_size > default_prepared_artifact_limit) throw FormatError("Prepared appendix exceeds its bound.");
        const auto appendix = reader.take(static_cast<std::size_t>(appendix_size));
        image.appendix.assign(appendix.begin(), appendix.end());
        image.gate_sites.resize(reader.count(16));
        for (auto& site : image.gate_sites) {
            site.return_address_offset = reader.number(8);
            site.instruction = static_cast<std::uint32_t>(reader.number(4));
            site.allowed_indirect_targets.resize(reader.count(12));
            for (auto& target : site.allowed_indirect_targets) target = location(reader);
        }
        image.fixups.resize(reader.count(24, prepared_fixup_limit));
        for (auto& fixup : image.fixups) {
            const auto kind = reader.number(4);
            if (kind > static_cast<unsigned>(execution::PreparedFixupKind::branch26_module_target))
                throw FormatError("Unsupported prepared fixup kind.");
            fixup.kind = static_cast<execution::PreparedFixupKind>(kind);
            fixup.image_offset = reader.number(8); fixup.target = location(reader);
        }
        reader.end(); return result;
    } catch (const FormatError& error) {
        return failure(LaunchError::artifact_mismatch, "decode-prepared-image", error.what());
    } catch (const std::bad_alloc&) { return allocation_failure(); }
}

LaunchResult<std::vector<ModulePayload>> validate_artifacts(
    const loader::ExeFsDirectoryInput& input, const PreparedModuleArtifacts& artifacts,
    const PreparationOptions& options) {
    try {
        const auto identity = installed_preparation_identity();
        if (!identity_matches(options, identity) || artifacts.preparation_identity != identity ||
            artifacts.format_version != prepared_modules_format_version)
            return failure(LaunchError::artifact_mismatch, "validate-artifacts", "Preparer identity or format version mismatch.");
        if (options.mode == PreparationMode::automatic && !artifacts.modules.empty()) {
            auto effective = options;
            {
                auto first = decode_payload(artifacts.modules.front().payload);
                if (const auto* failed = std::get_if<LaunchFailure>(&first)) return *failed;
                effective.mode = std::get<ModulePayload>(first).original_process
                    ? PreparationMode::original_process : PreparationMode::synthetic_test;
            }
            return validate_artifacts(input, artifacts, effective);
        }
        if (auto admitted = admitted_input(input, options.mode))
            return validate_artifacts(*admitted, artifacts, options);
        if (input.modules.empty() || input.modules.size() > max_modules ||
            artifacts.modules.size() != input.modules.size())
            return failure(LaunchError::artifact_mismatch, "validate-artifacts", "Prepared manifest does not match the input set.");
        std::size_t remaining = limit(options);
        std::set<std::string> names;
        std::vector<ModulePayload> result;
        std::vector<bool> ranks(input.modules.size());
        for (std::size_t i = 0; i < input.modules.size(); ++i) {
            const auto& source = input.modules[i]; const auto& artifact = artifacts.modules[i];
            if (!valid_name(source.name) || !names.insert(source.name).second || artifact.module_name != source.name)
                return failure(LaunchError::artifact_mismatch, "validate-artifacts", "Prepared manifest names or order differ.");
            if (artifact.payload.size() > remaining)
                return failure(LaunchError::resource_limit, "validate-artifacts", "Prepared payload exceeds the artifact budget.");
            remaining -= artifact.payload.size();
            const auto digest = loader::detail::sha256(source.bytes);
            if (!digest || *digest != artifact.source_sha256)
                return failure(LaunchError::artifact_mismatch, "validate-artifacts", "Complete NSO source identity mismatch.");
            auto decoded = decode_payload(artifact.payload);
            if (auto* failed = std::get_if<LaunchFailure>(&decoded)) return *failed;
            auto payload = std::get<ModulePayload>(std::move(decoded));
            const auto& image = payload.image;
            if ((!result.empty() && payload.original_process != result.front().original_process) ||
                (options.mode == PreparationMode::original_process && !payload.original_process) ||
                (options.mode == PreparationMode::synthetic_test && payload.original_process) ||
                (input.process_metadata && !payload.original_process))
                return failure(LaunchError::artifact_mismatch, "validate-artifacts",
                    "Prepared execution mode differs from the requested process ABI.");
            if (image.module_index != i || payload.load_rank >= ranks.size() || ranks[payload.load_rank] ||
                image.original_image_size == 0 || image.original_image_size > 2ull * 1024 * 1024 * 1024 ||
                image.appendix_offset != ((image.original_image_size + 0x3fff) & ~std::uint64_t{0x3fff}) ||
                payload.text_size == 0 || payload.text_offset % 4 ||
                payload.text_size > image.original_image_size ||
                payload.text_offset > image.original_image_size - payload.text_size)
                return failure(LaunchError::artifact_mismatch, "validate-artifacts", "Invalid prepared image layout or load rank.");
            ranks[payload.load_rank] = true;
            std::uint64_t previous = 0; bool first = true;
            for (const auto& word : image.coverage) {
                if (word.image_offset % 4 || word.image_offset < payload.text_offset ||
                    word.image_offset - payload.text_offset >= payload.text_size ||
                    payload.text_size - (word.image_offset - payload.text_offset) < 4 ||
                    (!first && word.image_offset <= previous))
                    return failure(LaunchError::artifact_mismatch, "validate-artifacts", "Invalid prepared execution coverage.");
                previous = word.image_offset; first = false;
            }
            if (!std::is_sorted(payload.roots.begin(), payload.roots.end()) ||
                std::adjacent_find(payload.roots.begin(), payload.roots.end()) != payload.roots.end())
                return failure(LaunchError::artifact_mismatch, "validate-artifacts", "Prepared roots are not canonical.");
            for (const auto root : payload.roots) if (!covers(image, root))
                return failure(LaunchError::artifact_mismatch, "validate-artifacts", "A prepared root has no instruction coverage.");
            result.push_back(std::move(payload));
        }
        const auto valid_target = [&](PreparedLocation target) {
            return target.module_index < result.size() && covers(result[target.module_index].image, target.image_offset);
        };
        for (const auto& payload : result) {
            for (const auto& site : payload.image.gate_sites)
                for (const auto target : site.allowed_indirect_targets) if (!valid_target(target))
                    return failure(LaunchError::artifact_mismatch, "validate-artifacts", "A gate target lacks prepared coverage.");
            for (const auto& fixup : payload.image.fixups) {
                if (fixup.kind == execution::PreparedFixupKind::branch26_module_target && !valid_target(fixup.target))
                    return failure(LaunchError::artifact_mismatch, "validate-artifacts", "A branch target lacks prepared coverage.");
                if (fixup.kind == execution::PreparedFixupKind::module_address_mov_wide &&
                    (fixup.target.module_index >= result.size() || fixup.target.image_offset >=
                        result[fixup.target.module_index].image.original_image_size))
                    return failure(LaunchError::artifact_mismatch, "validate-artifacts", "An address fixup is outside its module image.");
            }
        }
        return result;
    } catch (const std::bad_alloc&) { return allocation_failure(); }
      catch (const std::exception& error) { return failure(LaunchError::host_failure, "validate-artifacts", error.what()); }
}
} // namespace preparation

LaunchResult<PreparedModuleArtifacts> prepare_modules(
    const loader::ExeFsDirectoryInput& input, const PreparationOptions& options) {
    try {
        if (auto admitted = preparation::admitted_input(input, options.mode))
            return prepare_modules(*admitted, options);
        const auto identity = installed_preparation_identity();
        if (!identity_matches(options, identity))
            return preparation::failure(LaunchError::artifact_mismatch, "prepare-modules", "Requested preparer identity is not installed.");
        auto decoded = preparation::decode_input(input, options.mode);
        if (auto* failed = std::get_if<LaunchFailure>(&decoded)) return *failed;
        auto parsed = std::get<preparation::DecodedInput>(std::move(decoded));
        const auto count = input.modules.size();
        std::vector<std::vector<std::byte>> linked(count);
        std::vector<std::vector<linker::ModuleRange>> writable(count);
        std::vector<linker::LinkModule> link_modules;
        std::vector<std::span<std::byte>> destinations;
        for (std::size_t i = 0; i < count; ++i) {
            linked[i] = parsed.images[i].bytes;
            for (const auto& segment : parsed.images[i].segments) {
                if (segment.kind != loader::NsoSegmentKind::text && segment.mapped_size)
                    writable[i].push_back({segment.image_offset, segment.mapped_size});
            }
        }
        for (const auto i : parsed.order) {
            link_modules.push_back({input.modules[i].name, linked[i], &parsed.metadata[i],
                synthetic_base + i * synthetic_stride, linked[i].size(), writable[i]});
            destinations.push_back(linked[i]);
        }
        linker::LinkOptions link_options;
        link_options.preemption = parsed.original_process ? linker::PreemptionModel::load_order
            : linker::PreemptionModel::own_definition_first;
        auto link = linker::plan_link(link_modules, link_options);
        if (!link.ok()) return preparation::failure(LaunchError::unsupported, "offline-link", link.message);
        const auto applied = link.plan->apply(destinations);
        if (!applied.ok()) return preparation::failure(LaunchError::invalid_input, "offline-link", applied.message);
        std::vector<std::vector<preparation::RelocatedPointer>> pointer_proofs(count);
        for (std::size_t rank = 0; rank < count; ++rank) {
            const auto owner = parsed.order[rank];
            for (const auto& write : link.plan->writes(rank)) {
                if (write.source != linker::WriteSource::relative &&
                    write.source != linker::WriteSource::packed_relative &&
                    write.source != linker::WriteSource::module_symbol) continue;
                if (write.provider >= parsed.order.size())
                    return preparation::failure(LaunchError::host_failure, "offline-link", "Relocation provider index is inconsistent.");
                const auto provider = parsed.order[write.provider];
                const auto base = synthetic_base + provider * synthetic_stride;
                // The runtime linker rebases with this provider, not whichever
                // synthetic image a large addend happens to point into.
                if (write.value < base || write.value - base >= linked[provider].size()) continue;
                pointer_proofs[owner].push_back({write.offset,
                    {static_cast<std::uint32_t>(provider), write.value - base}});
            }
            std::sort(pointer_proofs[owner].begin(), pointer_proofs[owner].end(),
                [](const auto& left, const auto& right) { return left.image_offset < right.image_offset; });
        }
        std::vector<std::vector<std::uint64_t>> entry_roots(count);
        if (parsed.original_process) entry_roots[parsed.order.front()].push_back(0);
        std::vector<preparation::AnalysisModule> views;
        for (std::size_t i = 0; i < count; ++i)
            views.push_back({&parsed.images[i], &parsed.metadata[i], linked[i], synthetic_base + i * synthetic_stride,
                pointer_proofs[i], entry_roots[i], parsed.original_process});
        const auto analysis = preparation::analyze_modules(views);
        if (!analysis.ok()) return preparation::failure(LaunchError::unsupported, "classify-code", analysis.error);
        PreparedModuleArtifacts artifacts{prepared_modules_format_version, identity, {}, input.ignored_modules};
        for (std::size_t i = 0; i < count; ++i) {
            std::vector<execution::CoveredInstruction> words;
            const auto& analyzed = analysis.modules[i];
            for (const auto& word : analyzed.code) {
                const auto instruction = aarch64::decode(word.word);
                if (aarch64::is_pc_relative(instruction.opcode)) {
                    const auto base = synthetic_base + i * synthetic_stride;
                    const auto target = aarch64::referenced_address(instruction, base + word.offset);
                    // Original PC-relative instructions stay in place. Their
                    // displacement is invariant only within one module; cross-
                    // module address generation needs a distinct fixup contract.
                    if (!target || *target < base || *target - base >= linked[i].size())
                        return preparation::failure(LaunchError::unsupported, "prepare-image",
                            "Cross-module or external PC-relative references require unsupported address fixups.");
                }
                words.push_back({word.offset, word.word, {}, {}});
            }
            for (const auto& site : analyzed.indirect_sites) {
                const auto found = std::lower_bound(words.begin(), words.end(), site.offset,
                    [](const auto& word, auto offset) { return word.image_offset < offset; });
                if (found == words.end() || found->image_offset != site.offset)
                    return preparation::failure(LaunchError::host_failure, "prepare-image", "Indirect analysis is inconsistent.");
                for (const auto& target : site.targets) {
                    const PreparedLocation location{target.module_index, target.image_offset};
                    found->indirect_targets.push_back({location, location});
                }
            }
            execution::PreparedImageOptions image_options;
            image_options.module_index = static_cast<std::uint32_t>(i);
            image_options.module_count = static_cast<std::uint32_t>(count);
            // This artifact requires the guarded-stack, disjoint-storage native
            // execution contract for both synthetic and original exact-SP entry.
            // Reservation clearance and whole-window x18 transfers are proved
            // per site, retaining all whole-program ingress boundaries.
            image_options.scalar_x18_policy = {false, true, true};
            image_options.function_entries = analyzed.function_entries;
            image_options.external_entries = analyzed.external_entries;
            auto prepared = execution::prepare_image(linked[i].size(), words, image_options);
            if (!prepared.ok()) {
                std::string message = "A covered instruction cannot be prepared (status " +
                    std::to_string(static_cast<unsigned>(prepared.error)) + ").";
                if (!prepared.exclusive_proof.diagnostics.empty()) {
                    const auto& diagnostic = prepared.exclusive_proof.diagnostics.front();
                    message += " Exclusive window refusal: module=" + std::to_string(i);
                    if (diagnostic.window_offset) message += " window=" + std::to_string(*diagnostic.window_offset);
                    message += " site=" + std::to_string(diagnostic.instruction_offset) + " reason=" +
                        execution::exclusive_window_error_name(diagnostic.reason) + ".";
                }
                return preparation::failure(LaunchError::unsupported, "prepare-image", std::move(message));
            }
            if (prepared.image->fixups.size() > prepared_fixup_limit)
                return preparation::failure(LaunchError::resource_limit, "prepare-image",
                    "Prepared fixup count exceeds the format limit.");
            ModulePayload payload;
            payload.original_process = parsed.original_process;
            payload.load_rank = static_cast<std::uint32_t>(std::find(parsed.order.begin(), parsed.order.end(), i) - parsed.order.begin());
            const auto& text = parsed.images[i].segment(loader::NsoSegmentKind::text);
            payload.text_offset = text.image_offset; payload.text_size = text.content_size;
            for (const auto& root : analysis.roots) if (root.module_index == i) payload.roots.push_back(root.image_offset);
            std::sort(payload.roots.begin(), payload.roots.end());
            payload.roots.erase(std::unique(payload.roots.begin(), payload.roots.end()), payload.roots.end());
            payload.image = std::move(*prepared.image);
            const auto digest = loader::detail::sha256(input.modules[i].bytes);
            if (!digest) return preparation::failure(LaunchError::host_failure, "prepare-image", "NSO hashing failed.");
            artifacts.modules.push_back({input.modules[i].name, *digest, encode_payload(payload)});
        }
        if (encode_artifacts(artifacts).size() > limit(options))
            return preparation::failure(LaunchError::resource_limit, "prepare-modules", "Prepared output exceeds the artifact budget.");
        return artifacts;
    } catch (const std::bad_alloc&) { return allocation_failure(); }
      catch (const FormatError& error) { return preparation::failure(LaunchError::resource_limit, "prepare-modules", error.what()); }
      catch (const std::exception& error) { return preparation::failure(LaunchError::host_failure, "prepare-modules", error.what()); }
}

LaunchResult<PreparedModuleArtifacts> read_prepared_modules(
    const std::filesystem::path& directory, const loader::ExeFsDirectoryInput& input,
    const PreparationOptions& options) {
    try {
        const auto path = directory / artifact_file;
        const int descriptor = ::open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | O_NOFOLLOW);
        if (descriptor < 0) return preparation::failure(LaunchError::invalid_input, "read-prepared", "Cannot open the prepared artifact file.");
        struct Close { int fd; ~Close() { ::close(fd); } } close{descriptor};
        struct stat status{};
        if (::fstat(descriptor, &status) || !S_ISREG(status.st_mode) || status.st_size < 0)
            return preparation::failure(LaunchError::invalid_input, "read-prepared", "Prepared artifact must be a regular file.");
        if (static_cast<std::uint64_t>(status.st_size) > limit(options))
            return preparation::failure(LaunchError::resource_limit, "read-prepared", "Prepared artifact exceeds the read budget.");
        std::vector<std::byte> bytes(static_cast<std::size_t>(status.st_size));
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto size = ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
            if (size < 0 && errno == EINTR) continue;
            if (size <= 0) throw FormatError("Prepared artifact read failed or file changed.");
            offset += static_cast<std::size_t>(size);
        }
        std::byte extra; ssize_t size;
        do { size = ::read(descriptor, &extra, 1); } while (size < 0 && errno == EINTR);
        if (size != 0) throw FormatError("Prepared artifact length changed while reading.");
        auto reader = checked_reader(bytes); reader.magic(file_magic);
        PreparedModuleArtifacts artifacts;
        artifacts.format_version = static_cast<std::uint32_t>(reader.number(4));
        artifacts.preparation_identity = reader.string(256);
        artifacts.modules.resize(reader.count(44, max_modules));
        for (auto& module : artifacts.modules) {
            module.module_name = reader.string(256);
            const auto digest = reader.take(32);
            std::copy(digest.begin(), digest.end(), module.source_sha256.begin());
            const auto length = reader.number(8);
            if (length > limit(options)) throw FormatError("Prepared payload exceeds the read budget.");
            const auto payload = reader.take(static_cast<std::size_t>(length));
            module.payload.assign(payload.begin(), payload.end());
        }
        reader.end();
        auto validated = preparation::validate_artifacts(input, artifacts, options);
        if (const auto* failed = std::get_if<LaunchFailure>(&validated)) return *failed;
        artifacts.ignored_modules = input.ignored_modules;
        const auto& payloads = std::get<std::vector<ModulePayload>>(validated);
        const auto mode = payloads.front().original_process
            ? PreparationMode::original_process : PreparationMode::synthetic_test;
        if (auto admitted = preparation::admitted_input(input, mode))
            artifacts.ignored_modules = admitted->ignored_modules;
        return artifacts;
    } catch (const std::bad_alloc&) { return allocation_failure(); }
      catch (const FormatError& error) { return preparation::failure(LaunchError::artifact_mismatch, "read-prepared", error.what()); }
      catch (const std::exception& error) { return preparation::failure(LaunchError::host_failure, "read-prepared", error.what()); }
}

LaunchStatus write_prepared_modules(const std::filesystem::path& directory,
                                   const PreparedModuleArtifacts& artifacts) {
    try {
        if (artifacts.format_version != prepared_modules_format_version ||
            artifacts.preparation_identity != installed_preparation_identity() ||
            artifacts.modules.empty() || artifacts.modules.size() > max_modules)
            return preparation::failure(LaunchError::artifact_mismatch, "write-prepared", "Unsupported prepared artifact envelope.");
        std::set<std::string> names;
        for (const auto& module : artifacts.modules) {
            if (!valid_name(module.module_name) || !names.insert(module.module_name).second)
                return preparation::failure(LaunchError::invalid_input, "write-prepared", "Invalid prepared manifest names.");
            const auto payload = preparation::decode_payload(module.payload);
            if (const auto* failed = std::get_if<LaunchFailure>(&payload)) return *failed;
        }
        const auto bytes = encode_artifacts(artifacts);
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error || !std::filesystem::is_directory(directory))
            return preparation::failure(LaunchError::host_failure, "write-prepared", "Cannot create the artifact directory.");
        auto temporary = (directory / ".armigo-prepared-XXXXXX").string();
        const int descriptor = ::mkstemp(temporary.data());
        if (descriptor < 0) return preparation::failure(LaunchError::host_failure, "write-prepared", "Cannot create an artifact temporary file.");
        struct Cleanup { int fd; std::string path; ~Cleanup() { ::close(fd); ::unlink(path.c_str()); } } cleanup{descriptor, temporary};
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto size = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
            if (size < 0 && errno == EINTR) continue;
            if (size <= 0) return preparation::failure(LaunchError::host_failure, "write-prepared", "Prepared artifact write failed.");
            offset += static_cast<std::size_t>(size);
        }
        const auto destination = directory / artifact_file;
        if (::rename(temporary.c_str(), destination.c_str()) != 0)
            return preparation::failure(LaunchError::host_failure, "write-prepared", "Cannot publish the prepared artifact file.");
        return std::monostate{};
    } catch (const std::bad_alloc&) { return allocation_failure(); }
      catch (const FormatError& error) { return preparation::failure(LaunchError::resource_limit, "write-prepared", error.what()); }
      catch (const std::exception& error) { return preparation::failure(LaunchError::host_failure, "write-prepared", error.what()); }
}

} // namespace armigo::runtime
