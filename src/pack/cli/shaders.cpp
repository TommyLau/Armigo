// ============================================================================
// Armigo Tools - Development and Verification Tools
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "shaders.h"

#include <armigo/loader/nso_module.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/package.h>
#include <armigo/shader_input/archive.h>
#include <armigo/shader_input/container.h>
#include <armigo/shader_input/gfd_cache.h>
#include <armigo/shader_input/glslc_output.h>
#include <armigo/shader_input/shader_input.h>
#include <armigo/shader_translation/stage_translation.h>
#include "sha256.h"

#include <sys/qos.h>
#include <sys/sysctl.h>

#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::pack {
namespace {

namespace fs = std::filesystem;
namespace si = armigo::shader_input;
namespace tr = armigo::shader::translation;
namespace gm = armigo::shader::graphics_msl;

constexpr std::size_t kMemoryLimit = 512ull * 1024ull * 1024ull;
constexpr std::size_t kTraversalLimit = 64ull << 30;
constexpr std::size_t kMaxErrors = 8;
constexpr std::array<std::byte, 8> kBnshSignature{std::byte{'B'}, std::byte{'N'}, std::byte{'S'},
                                                  std::byte{'H'}, std::byte{0},    std::byte{0},
                                                  std::byte{0},    std::byte{0}};

using Bytes = std::span<const std::byte>;

struct RemoveTree {
    fs::path path;
    ~RemoveTree() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

bool starts_with(Bytes input, std::string_view magic) {
    return input.size() >= magic.size() &&
           std::memcmp(input.data(), magic.data(), magic.size()) == 0;
}

bool ends_with_ignore_case(std::string_view text, std::string_view suffix) {
    if (text.size() < suffix.size()) return false;
    for (std::size_t index = 0; index < suffix.size(); ++index) {
        const auto left = static_cast<unsigned char>(text[text.size() - suffix.size() + index]);
        const auto right = static_cast<unsigned char>(suffix[index]);
        if (std::tolower(left) != std::tolower(right)) return false;
    }
    return true;
}

si::ByteSpan view_of(Bytes input) {
    return {reinterpret_cast<const std::uint8_t*>(input.data()), input.size()};
}

Bytes bytes_of(si::ByteSpan span) {
    return {reinterpret_cast<const std::byte*>(span.data), span.size};
}

std::string hex_digest(Bytes input) {
    const auto value = armigo::loader::detail::sha256(input);
    if (!value) {
        throw std::runtime_error("System SHA-256 failed");
    }
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string text;
    text.reserve(value->size() * 2);
    for (const std::byte byte : *value) {
        const auto digit = std::to_integer<unsigned>(byte);
        text += kDigits[digit >> 4];
        text += kDigits[digit & 0x0f];
    }
    return text;
}

bool ignorable(si::ContainerError::Code code) {
    switch (code) {
        case si::ContainerError::Code::MissingResourceBlock:
        case si::ContainerError::Code::Magic:
        case si::ContainerError::Code::Truncated:
        case si::ContainerError::Code::InvalidSpan:
            return true;
        default:
            return false;
    }
}

void write_bytes(const fs::path& path, Bytes data) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream || !stream.write(reinterpret_cast<const char*>(data.data()),
                                 static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

struct Enumerator {
    fs::path stages;
    std::set<std::string> seen;
    std::size_t unique = 0;
    std::size_t unsupported = 0;
    std::size_t structural_errors = 0;
    bool strict_stages = false;
    std::atomic<std::size_t> scanned_bytes{0};
    std::vector<std::string> errors;
    std::string hard;
    std::mutex mutex;

    void note(std::string message) {
        const std::lock_guard lock(mutex);
        ++structural_errors;
        if (errors.size() < kMaxErrors) {
            errors.push_back(std::move(message));
        }
    }

    void fail(std::string message) {
        const std::lock_guard lock(mutex);
        if (hard.empty()) hard = std::move(message);
    }

    [[nodiscard]] bool stopped() {
        const std::lock_guard lock(mutex);
        return !hard.empty();
    }

    [[nodiscard]] std::string hard_error() {
        const std::lock_guard lock(mutex);
        return hard;
    }

    [[nodiscard]] std::size_t unique_count() {
        const std::lock_guard lock(mutex);
        return unique;
    }

    void record(si::Stage kind, Bytes control, Bytes code) {
        const auto key = kind == si::Stage::Vertex || kind == si::Stage::Fragment || kind == si::Stage::Compute
            ? hex_digest(control) + "-" + hex_digest(code)
            : std::string{};
        const std::lock_guard lock(mutex);
        if (kind != si::Stage::Vertex && kind != si::Stage::Fragment && kind != si::Stage::Compute) {
            if (strict_stages) {
                if (hard.empty()) hard = "A shader stage is not vertex, fragment, or compute.";
            } else {
                ++unsupported;
            }
            return;
        }
        if (!seen.insert(key).second) {
            return;
        }
        const auto directory = stages / key;
        fs::create_directory(directory);
        write_bytes(directory / "control.bin", control);
        write_bytes(directory / "code.bin", code);
        ++unique;
    }

    void record(const si::ContainerStage& stage) {
        record(stage.shader.stage, bytes_of(stage.shader.control), bytes_of(stage.shader.code));
    }
};

void scan(Bytes input, const std::string& source, unsigned depth, Enumerator& enumerator);

void accept_container(const si::BnshResult& parsed, const std::string& source, std::size_t offset,
                      Enumerator& enumerator) {
    if (const auto* error = std::get_if<si::ContainerError>(&parsed)) {
        if (!ignorable(error->code)) {
            enumerator.note(source + " offset " + std::to_string(offset) + ": " + error->message());
        }
        return;
    }
    for (const auto& stage : std::get<si::BnshContainer>(parsed).stages) {
        enumerator.record(stage);
    }
}

void scan(Bytes input, const std::string& source, unsigned depth, Enumerator& enumerator) {
    if (enumerator.stopped()) {
        return;
    }
    if (depth > 8) {
        enumerator.fail("Archive nesting exceeds the extraction policy: " + source);
        return;
    }
    const std::size_t scanned = enumerator.scanned_bytes.fetch_add(input.size(), std::memory_order_relaxed);
    if (scanned > kTraversalLimit || input.size() > kTraversalLimit - scanned) {
        enumerator.fail("Archive traversal exceeds the extraction policy: " + source);
        return;
    }
    if (starts_with(input, "Yaz0")) {
        auto decoded = si::decompress_yaz0(input, kMemoryLimit);
        if (const auto* error = std::get_if<si::ArchiveError>(&decoded)) {
            if (error->code == si::ArchiveError::Code::Yaz0OutputLimitExceeded) {
                enumerator.fail(source + " exceeds the in-memory limit.");
            } else {
                enumerator.note(source + ": " + error->message());
            }
            return;
        }
        scan(std::get<std::vector<std::byte>>(decoded), source + "::decoded", depth + 1, enumerator);
        return;
    }
    if (starts_with(input, "SARC")) {
        auto parsed = si::parse_sarc(input);
        if (const auto* error = std::get_if<si::ArchiveError>(&parsed)) {
            enumerator.note(source + ": " + error->message());
            return;
        }
        std::size_t index = 0;
        for (const auto& member : std::get<si::SarcArchive>(parsed).members) {
            const auto name = member.has_name ? std::string(member.name) : std::to_string(index);
            scan(member.data, source + "::" + name, depth + 1, enumerator);
            ++index;
            if (enumerator.stopped()) {
                return;
            }
        }
        return;
    }
    if (starts_with(input, "BAHS")) {
        auto parsed = si::parse_bahs(view_of(input));
        if (const auto* error = std::get_if<si::ContainerError>(&parsed)) {
            if (!ignorable(error->code)) {
                enumerator.note(source + ": " + error->message());
            }
        } else {
            for (const auto& stage : std::get<si::BahsContainer>(parsed).stages) {
                enumerator.record(stage);
            }
        }
    }
    auto cursor = input.begin();
    while (cursor != input.end() && !enumerator.stopped()) {
        const auto found = std::search(cursor, input.end(), kBnshSignature.begin(), kBnshSignature.end());
        if (found == input.end()) {
            break;
        }
        const auto offset = static_cast<std::size_t>(found - input.begin());
        accept_container(si::parse_bnsh(view_of(input.subspan(offset))), source, offset, enumerator);
        cursor = found + static_cast<std::ptrdiff_t>(kBnshSignature.size());
    }
}

std::vector<std::byte> read_file(const fs::path& path, std::uintmax_t size) {
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    if (!stream || !stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return data;
}

void accept_bnsh_file(const fs::path& path, std::ifstream& random, std::uint64_t at,
                      std::uint64_t file_size, Enumerator& enumerator) {
    if (at > file_size || file_size - at < 0x88) {
        return;
    }
    std::array<std::byte, 0x88> header{};
    random.clear();
    random.seekg(static_cast<std::streamoff>(at));
    if (!random.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()))) {
        enumerator.fail("Cannot read " + path.string());
        return;
    }
    if (std::memcmp(header.data(), kBnshSignature.data(), kBnshSignature.size()) != 0 ||
        std::memcmp(header.data() + 0x60, "grsc", 4) != 0) {
        return;
    }
    const auto available = static_cast<std::size_t>(file_size - at);
    const auto take = std::min(available, kMemoryLimit);
    std::vector<std::byte> bytes(take);
    random.clear();
    random.seekg(static_cast<std::streamoff>(at));
    if (!random.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        enumerator.fail("Cannot read " + path.string());
        return;
    }
    const auto parsed = si::parse_bnsh(view_of(bytes));
    accept_container(parsed, path.string(), static_cast<std::size_t>(at), enumerator);
    if (take < available) {
        const auto* error = std::get_if<si::ContainerError>(&parsed);
        // A signature that happens to sit in a large non-shader file is not a
        // truncated container. Note the limit only when the parser itself ran
        // out of the bytes this read kept.
        if (error != nullptr && (error->code == si::ContainerError::Code::Truncated ||
                                 error->code == si::ContainerError::Code::Range)) {
            enumerator.note(path.string() + " offset " + std::to_string(at) +
                            " exceeds the in-memory limit; the shader container was not fully read.");
        }
    }
}

void scan_large(const fs::path& path, std::uintmax_t size, Enumerator& enumerator) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        enumerator.fail("Cannot read " + path.string());
        return;
    }
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    const auto got_magic = static_cast<std::size_t>(input.gcount());
    const auto starts = [&](std::string_view token) {
        return got_magic >= token.size() && std::string_view(magic.data(), token.size()) == token;
    };
    if (starts("Yaz0") || starts("SARC") || starts("NSO0") || starts("BAHS")) {
        enumerator.fail(path.string() + " exceeds the in-memory limit (" + std::to_string(size) + " bytes).");
        return;
    }
    std::ifstream random(path, std::ios::binary);
    if (!random) {
        enumerator.fail("Cannot read " + path.string());
        return;
    }
    input.clear();
    input.seekg(0);
    constexpr std::size_t kChunk = 1u << 20;
    std::vector<std::byte> window(kChunk + 7);
    std::uint64_t window_base = 0;
    std::size_t carry = 0;
    while (!enumerator.stopped()) {
        input.read(reinterpret_cast<char*>(window.data() + carry), static_cast<std::streamsize>(kChunk));
        const auto got = static_cast<std::size_t>(input.gcount());
        const auto count = carry + got;
        std::size_t search_from = 0;
        while (!enumerator.stopped()) {
            const auto begin = window.begin() + static_cast<std::ptrdiff_t>(search_from);
            const auto end = window.begin() + static_cast<std::ptrdiff_t>(count);
            const auto found = std::search(begin, end, kBnshSignature.begin(), kBnshSignature.end());
            if (found == end) {
                break;
            }
            const auto index = static_cast<std::size_t>(found - window.begin());
            accept_bnsh_file(path, random, window_base + index, size, enumerator);
            search_from = index + kBnshSignature.size();
            if (search_from >= count) {
                break;
            }
        }
        if (got < kChunk) {
            break;
        }
        std::memmove(window.data(), window.data() + count - 7, 7);
        window_base += count - 7;
        carry = 7;
    }
}

void scan_path(const fs::path& path, Enumerator& enumerator) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if (error) {
        enumerator.fail("Cannot read the size of " + path.string());
        return;
    }
    if (size > kMemoryLimit) {
        scan_large(path, size, enumerator);
        return;
    }
    const auto data = read_file(path, size);
    if (starts_with(data, "NSO0")) {
        const auto module = armigo::loader::read_nso_module(data);
        if (!module.ok()) {
            enumerator.fail(path.string() + ": " + armigo::loader::to_string(module.error.code));
            return;
        }
        scan(module.image->bytes, path.string(), 0, enumerator);
        return;
    }
    scan(data, path.string(), 0, enumerator);
}

unsigned worker_count(unsigned maximum);

void enumerate_tree(const fs::path& root, Enumerator& enumerator) {
    std::error_code error;
    if (!fs::is_directory(root, error)) {
        enumerator.fail("Missing extracted directory " + root.string());
        return;
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(root, error)) {
        if (error) {
            enumerator.fail("Cannot list " + root.string());
            return;
        }
        std::error_code link;
        if (entry.is_symlink(link) || !entry.is_regular_file(link)) {
            continue;
        }
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) return;
    struct Queue {
        const std::vector<fs::path>& files;
        Enumerator& enumerator;
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> processed{0};
    } queue{files, enumerator};
    const auto worker = [](void* raw) -> void* {
        auto& queue = *static_cast<Queue*>(raw);
        for (;;) {
            const auto index = queue.next.fetch_add(1);
            if (index >= queue.files.size() || queue.enumerator.stopped()) return nullptr;
            scan_path(queue.files[index], queue.enumerator);
            const std::size_t processed = queue.processed.fetch_add(1) + 1;
            if (processed % 200 == 0) {
                armigo::diag::write(armigo::diag::Level::info,  "Scanned %zu files, %zu shader stages\n", processed,
                             queue.enumerator.unique_count());
                std::fflush(stderr);
            }
        }
    };
    const unsigned workers = std::min<std::size_t>(worker_count(14), files.size());
    std::vector<pthread_t> threads(workers);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, 64u << 20);
    pthread_attr_set_qos_class_np(&attributes, QOS_CLASS_USER_INITIATED, 0);
    unsigned created = 0;
    for (; created < workers; ++created) {
        if (pthread_create(&threads[created], &attributes, worker, &queue) != 0) {
            enumerator.fail("A shader enumeration thread could not start.");
            break;
        }
    }
    pthread_attr_destroy(&attributes);
    for (unsigned index = 0; index < created; ++index) pthread_join(threads[index], nullptr);
}

void collect_cpk_archives(const fs::path& root, std::vector<fs::path>& archives, Enumerator& enumerator) {
    std::error_code error;
    if (!fs::is_directory(root, error)) {
        enumerator.fail("Missing extracted directory " + root.string());
        return;
    }
    for (const auto& entry : fs::recursive_directory_iterator(root, error)) {
        if (error) {
            enumerator.fail("Cannot list " + root.string());
            return;
        }
        std::error_code link;
        if (entry.is_symlink(link) || !entry.is_regular_file(link)) continue;
        if (ends_with_ignore_case(entry.path().filename().string(), ".cpk")) {
            archives.push_back(entry.path());
        }
    }
    if (error) enumerator.fail("Cannot list " + root.string());
}

// The main module's flat image, or empty after recording why it cannot be read.
std::vector<std::byte> read_main_image(const fs::path& exefs, Enumerator& enumerator) {
    std::error_code error;
    const fs::path main_file = fs::is_regular_file(exefs / "main", error) ? exefs / "main" : exefs / "main.nso";
    if (!fs::is_regular_file(main_file, error)) {
        enumerator.fail("The main module is missing from the extracted ExeFS.");
        return {};
    }
    const auto size = fs::file_size(main_file, error);
    if (error || size > kMemoryLimit) {
        enumerator.fail(main_file.string() + " exceeds the in-memory limit.");
        return {};
    }
    const auto data = read_file(main_file, size);
    if (!starts_with(data, "NSO0")) {
        enumerator.fail(main_file.string() + " is not an NSO module.");
        return {};
    }
    auto module = armigo::loader::read_nso_module(data);
    if (!module.ok()) {
        enumerator.fail(main_file.string() + ": " + armigo::loader::to_string(module.error.code));
        return {};
    }
    return std::move(module.image->bytes);
}

void enumerate_gfd(const fs::path& exefs, const fs::path& romfs, const std::vector<fs::path>& addons,
                   const ShaderPackInput& input, Enumerator& enumerator) {
    std::vector<fs::path> archives;
    collect_cpk_archives(romfs, archives, enumerator);
    for (const auto& addon : addons) {
        if (enumerator.stopped()) return;
        collect_cpk_archives(addon, archives, enumerator);
    }
    if (enumerator.stopped()) return;
    std::sort(archives.begin(), archives.end());
    struct CacheMember {
        fs::path archive;
        si::CpkMember member;
    };
    std::vector<CacheMember> members;
    std::set<std::string> names;
    for (const auto& archive : archives) {
        auto listing = si::list_cpk(archive);
        if (const auto* failure = std::get_if<si::GfdCacheError>(&listing)) {
            enumerator.fail(failure->message);
            return;
        }
        for (auto& member : std::get<si::CpkListing>(listing).members) {
            if (!ends_with_ignore_case(member.name, ".gsc")) continue;
            if (!names.insert(member.name).second) {
                enumerator.fail("Shader cache " + member.name + " is listed more than once.");
                return;
            }
            members.push_back(CacheMember{archive, std::move(member)});
        }
    }
    std::vector<std::string> present;
    present.reserve(members.size());
    for (const auto& item : members) present.push_back(item.member.name);
    const auto missing = si::missing_shader_caches(present, input.caches, input.fallbacks);
    if (!missing.empty()) {
        enumerator.fail("Required shader cache is missing: " + missing.front());
        return;
    }
    for (const auto& item : members) {
        if (enumerator.stopped()) return;
        armigo::diag::write(armigo::diag::Level::info, "Shader cache %s\n", item.member.name.c_str());
        std::fflush(stderr);
        auto bytes = si::read_cpk_member(item.archive, item.member);
        if (const auto* failure = std::get_if<si::GfdCacheError>(&bytes)) {
            enumerator.fail(failure->message);
            return;
        }
        const auto& decoded = std::get<std::vector<std::byte>>(bytes);
        auto stages = si::parse_gsc(decoded, item.member.name);
        if (const auto* failure = std::get_if<si::GfdCacheError>(&stages)) {
            enumerator.fail(failure->message);
            return;
        }
        for (const auto& stage : std::get<std::vector<si::OwnedShaderStage>>(stages)) {
            enumerator.record(stage.stage, stage.control, stage.code);
        }
    }
    if (enumerator.stopped()) return;
    const auto image = read_main_image(exefs, enumerator);
    if (image.empty()) return;
    const auto packages = si::scan_hhneedle(image, "main",
                                            input.ignored_needle_offsets);
    if (!packages.rejected.empty()) {
        enumerator.fail(packages.rejected.front());
        return;
    }
    if (!packages.uncovered_controls.empty()) {
        enumerator.fail("Shader control at main image offset " +
                        std::to_string(packages.uncovered_controls.front()) +
                        " is not part of a packaged stage.");
        return;
    }
    armigo::diag::write(armigo::diag::Level::info, "Shader packages in main: %zu stages\n", packages.stages.size());
    std::fflush(stderr);
    for (const auto& stage : packages.stages) {
        enumerator.record(stage.stage, stage.control, stage.code);
    }
}

// The GLSLC compiler outputs compiled into the main module. Every shader
// control in the image must belong to an output that validates.
void enumerate_glslc(const fs::path& exefs, Enumerator& enumerator) {
    const auto image = read_main_image(exefs, enumerator);
    if (image.empty()) return;
    const auto scan = si::scan_glslc_outputs(image, "main");
    if (!scan.rejected.empty()) {
        enumerator.fail("GLSLC output " + scan.rejected.front());
        return;
    }
    if (!scan.uncovered_controls.empty()) {
        enumerator.fail("Shader control at main image offset " + std::to_string(scan.uncovered_controls.front()) +
                        " is not part of a GLSLC output.");
        return;
    }
    if (scan.stages.empty()) {
        enumerator.fail("The main module holds no GLSLC output.");
        return;
    }
    armigo::diag::write(armigo::diag::Level::info, "GLSLC outputs in main: %zu outputs, %zu stages\n", scan.outputs,
                        scan.stages.size());
    std::fflush(stderr);
    for (const auto& stage : scan.stages) {
        enumerator.record(stage.stage, stage.control, stage.code);
    }
}

// Every stage of the title's container set, recorded into the enumerator.
void enumerate_containers(const fs::path& exefs, const fs::path& romfs, const std::vector<fs::path>& addons,
                          const ShaderPackInput& input, Enumerator& enumerator) {
    if (input.containers == ShaderPackInput::Containers::Glslc) {
        enumerator.strict_stages = true;
        enumerate_glslc(exefs, enumerator);
    } else if (input.containers == ShaderPackInput::Containers::GfdCache) {
        enumerator.strict_stages = true;
        enumerate_gfd(exefs, romfs, addons, input, enumerator);
    } else {
        enumerate_tree(exefs, enumerator);
        if (!enumerator.stopped()) {
            enumerate_tree(romfs, enumerator);
        }
        for (const auto& addon : addons) {
            if (enumerator.stopped()) {
                break;
            }
            enumerate_tree(addon, enumerator);
        }
    }
}

std::uint32_t word_at(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof value);
    return value;
}

tr::StageOptions stage_options(si::Stage stage, std::span<const std::byte> control,
                               gm::VertexDepthConvention vertex_depth, bool hoist_implicit_samples) {
    tr::StageOptions options;
    options.texture_handle_cbuf = 2;
    options.texture_descriptors_derived = true;
    options.hoist_implicit_samples = hoist_implicit_samples;
    if (stage == si::Stage::Vertex) {
        options.emission.vertex_depth = vertex_depth;
    } else if (stage == si::Stage::Compute) {
        constexpr std::size_t kLocal = 0x748;
        constexpr std::size_t kShared = 0x754;
        if (control.size() < kShared + 4) {
            throw std::runtime_error("Compute shader control is too short for its launch facts");
        }
        std::array<std::uint32_t, 3> local{};
        for (unsigned axis = 0; axis < 3; ++axis) {
            local[axis] = word_at(control, kLocal + 4 * axis);
        }
        options.compute_local_size = local;
        options.compute_shared_memory_bytes = word_at(control, kShared);
    }
    return options;
}

tr::TranslatedStage translate_with_retry(std::span<const std::byte> control, std::span<const std::byte> code,
                                        si::Stage stage, gm::VertexDepthConvention vertex_depth,
                                        bool hoist_implicit_samples) {
    auto options = stage_options(stage, control, vertex_depth, hoist_implicit_samples);
    try {
        return tr::translate_stage(control, code, options);
    } catch (const std::exception& first) {
        std::string message = first.what();
        std::transform(message.begin(), message.end(), message.begin(),
                       [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
        if (stage != si::Stage::Fragment || message.find("position convention") == std::string::npos) {
            throw;
        }
        options.emission.fragment_position =
            gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        return tr::translate_stage(control, code, options);
    }
}

bool translate_directory(const fs::path& directory, gm::VertexDepthConvention vertex_depth,
                         bool hoist_implicit_samples, std::string& error) {
    try {
        const auto control_size = fs::file_size(directory / "control.bin");
        const auto code_size = fs::file_size(directory / "code.bin");
        const auto control = read_file(directory / "control.bin", control_size);
        const auto code = read_file(directory / "code.bin", code_size);
        const auto parsed = si::parse_shader(
            {view_of(control), view_of(code), {}});
        if (const auto* failure = std::get_if<si::ParseError>(&parsed)) {
            error = directory.string() + ": " + failure->message();
            return false;
        }
        const auto stage = std::get<si::ParsedShader>(parsed).stage;
        const auto translated = translate_with_retry(control, code, stage, vertex_depth, hoist_implicit_samples);
        const auto source = std::as_bytes(std::span<const char>(translated.source.data(), translated.source.size()));
        const auto reflection = std::as_bytes(
            std::span<const char>(translated.reflection.data(), translated.reflection.size()));
        write_bytes(directory / "source.metal", source);
        write_bytes(directory / "reflection.json", reflection);
        write_bytes(directory / "program.bin", translated.program);
        return true;
    } catch (const std::exception& failure) {
        error = directory.string() + ": " + failure.what();
        std::error_code ignored;
        fs::remove(directory / "source.metal", ignored);
        fs::remove(directory / "reflection.json", ignored);
        fs::remove(directory / "program.bin", ignored);
        return false;
    }
}

struct TranslateQueue {
    std::vector<fs::path> stages;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> translated{0};
    std::atomic<std::size_t> failed{0};
    std::mutex mutex;
    std::vector<std::string> errors;
    gm::VertexDepthConvention vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    bool hoist_implicit_samples = false;
};

void* translate_worker(void* raw) {
    auto& queue = *static_cast<TranslateQueue*>(raw);
    for (;;) {
        const auto index = queue.next.fetch_add(1);
        if (index >= queue.stages.size()) {
            return nullptr;
        }
        std::string error;
        if (!translate_directory(queue.stages[index], queue.vertex_depth, queue.hoist_implicit_samples, error)) {
            queue.failed.fetch_add(1);
            const std::lock_guard lock(queue.mutex);
            if (queue.errors.size() < kMaxErrors) {
                queue.errors.push_back(std::move(error));
            }
            continue;
        }
        const auto count = queue.translated.fetch_add(1) + 1;
        if (count == 1 || count % 500 == 0) {
            armigo::diag::write(armigo::diag::Level::info,  "Translated %zu shader stages\n", count);
            std::fflush(stderr);
        }
    }
}

unsigned worker_count(unsigned maximum) {
    int cores = 0;
    std::size_t size = sizeof cores;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nullptr, 0) != 0 || cores <= 0) {
        cores = 4;
    }
    return static_cast<unsigned>(std::clamp(cores, 1, static_cast<int>(maximum)));
}

std::variant<std::string, ShaderReport> build_shader_package_impl(
    const fs::path& exefs, const fs::path& romfs, const std::vector<fs::path>& addons, const fs::path& staging,
    const fs::path& package, const fs::path& compile_cache, const fs::path& translate_tool,
    const ShaderPackInput& input) {
    std::error_code error;
    if (!fs::is_regular_file(translate_tool, error)) {
        return "The shader translator is missing next to the packer.";
    }
    fs::remove_all(staging, error);
    fs::create_directories(staging / "stages", error);
    if (error) {
        return "Cannot create shader staging " + staging.string();
    }
    RemoveTree cleanup{staging};
    fs::remove_all(package, error);
    Enumerator enumerator;
    enumerator.stages = staging / "stages";
    enumerate_containers(exefs, romfs, addons, input, enumerator);
    if (enumerator.stopped()) {
        return enumerator.hard_error();
    }
    armigo::diag::write(armigo::diag::Level::info,  "Enumerated %zu shader stages (%zu unsupported, %zu structural errors)\n",
                 enumerator.unique, enumerator.unsupported, enumerator.structural_errors);
    for (const auto& message : enumerator.errors) {
        armigo::diag::write(armigo::diag::Level::error,  "Shader enumeration: %s\n", message.c_str());
    }
    std::fflush(stderr);
    if (enumerator.unique == 0) {
        return std::string("No shader stages were found in the extracted game.");
    }
    TranslateQueue queue;
    queue.vertex_depth = input.vertex_depth == ShaderPackInput::VertexDepth::ZeroToOne
        ? gm::VertexDepthConvention::ZeroToOne
        : gm::VertexDepthConvention::NegativeOneToOne;
    queue.hoist_implicit_samples = input.hoist_implicit_samples;
    for (const auto& entry : fs::directory_iterator(enumerator.stages)) {
        if (entry.is_directory()) {
            queue.stages.push_back(entry.path());
        }
    }
    const unsigned workers = worker_count(14);
    std::vector<pthread_t> threads(workers);
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, 64u << 20);
    pthread_attr_set_qos_class_np(&attributes, QOS_CLASS_USER_INITIATED, 0);
    for (unsigned index = 0; index < workers; ++index) {
        const int created = pthread_create(&threads[index], &attributes, translate_worker, &queue);
        if (created != 0) {
            pthread_attr_destroy(&attributes);
            for (unsigned joined = 0; joined < index; ++joined) {
                pthread_join(threads[joined], nullptr);
            }
            return std::string("A shader translation thread could not start.");
        }
    }
    pthread_attr_destroy(&attributes);
    for (pthread_t thread : threads) {
        pthread_join(thread, nullptr);
    }
    for (const auto& message : queue.errors) {
        armigo::diag::write(armigo::diag::Level::error,  "Shader translation: %s\n", message.c_str());
    }
    const auto translated = queue.translated.load();
    armigo::diag::write(armigo::diag::Level::info,  "Translated %zu stages.\n", translated);
    std::fflush(stderr);
    if (translated == 0) {
        return std::string("No shader stages were translated.");
    }
    if (input.containers == ShaderPackInput::Containers::GfdCache && queue.failed.load() != 0) {
        return queue.errors.empty() ? std::string("A shader stage did not translate.") : queue.errors.front();
    }
    armigo::shader::package::BuildOptions options;
    for (const auto& entry : fs::directory_iterator(enumerator.stages)) {
        if (entry.is_directory() && fs::is_regular_file(entry.path() / "reflection.json")) {
            options.stages.push_back(entry.path());
        }
    }
    std::sort(options.stages.begin(), options.stages.end());
    options.translate_tool = translate_tool;
    options.output = package;
    options.hoist_implicit_samples = input.hoist_implicit_samples;
    if (!compile_cache.empty()) options.compile_cache = compile_cache;
    options.jobs = worker_count(8);
    auto built = armigo::shader::package::build(options);
    if (const auto* failure = std::get_if<std::string>(&built)) {
        fs::remove_all(package, error);
        return *failure;
    }
    ShaderReport report;
    report.enumerated = enumerator.unique;
    report.translated = translated;
    report.unsupported = enumerator.unsupported;
    report.structural_errors = enumerator.structural_errors + queue.failed.load();
    return report;
}

} // namespace

std::variant<std::string, ShaderEnumeration> enumerate_shader_stages(
    const fs::path& exefs, const fs::path& romfs, const std::vector<fs::path>& addons, const fs::path& stages,
    const ShaderPackInput& input) {
    try {
        Enumerator enumerator;
        enumerator.stages = stages;
        enumerate_containers(exefs, romfs, addons, input, enumerator);
        if (enumerator.stopped()) return enumerator.hard_error();
        return ShaderEnumeration{enumerator.unique, enumerator.unsupported, enumerator.structural_errors,
                                 enumerator.errors};
    } catch (const std::bad_alloc&) {
        return std::string("Host memory for shader packing could not be allocated.");
    } catch (const std::exception& failure) {
        return std::string(failure.what());
    }
}

std::variant<std::string, ShaderReport> build_shader_package(
    const fs::path& exefs, const fs::path& romfs, const std::vector<fs::path>& addons, const fs::path& staging,
    const fs::path& package, const fs::path& compile_cache, const fs::path& translate_tool,
    const ShaderPackInput& input) {
    try {
        return build_shader_package_impl(exefs, romfs, addons, staging, package, compile_cache, translate_tool,
                                         input);
    } catch (const std::bad_alloc&) {
        return std::string("Host memory for shader packing could not be allocated.");
    } catch (const std::exception& failure) {
        return std::string(failure.what());
    }
}

} // namespace armigo::pack
