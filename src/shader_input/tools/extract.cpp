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
#include <armigo/loader/nso_module.h>
#include <armigo/shader_input/archive.h>
#include <armigo/shader_input/container.h>
#include "sha256.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
using namespace armigo::shader_input;
using Bytes = std::span<const std::byte>;
constexpr std::size_t input_limit = 512ull * 1024 * 1024;
constexpr std::size_t expansion_limit = 512ull * 1024 * 1024;
constexpr std::size_t total_expansion_limit = 64ull << 30;

std::string quote(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') {
            result += '\\';
            result += static_cast<char>(byte);
        } else if (byte < 0x20) {
            result += "\\u00";
            result += hex[byte >> 4];
            result += hex[byte & 15];
        } else {
            result += static_cast<char>(byte);
        }
    }
    return result + '"';
}

std::string digest(Bytes bytes) {
    const auto value = armigo::loader::detail::sha256(bytes);
    if (!value) throw std::runtime_error("System SHA-256 failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string text;
    for (const auto byte : *value) {
        const auto value = std::to_integer<unsigned>(byte);
        text += hex[value >> 4];
        text += hex[value & 15];
    }
    return text;
}

Bytes bytes(ByteSpan span) {
    return {reinterpret_cast<const std::byte*>(span.data), span.size};
}

ByteSpan view(Bytes span) {
    return {reinterpret_cast<const std::uint8_t*>(span.data()), span.size()};
}

bool magic(Bytes input, std::string_view signature) {
    return input.size() >= signature.size() &&
        std::memcmp(input.data(), signature.data(), signature.size()) == 0;
}

std::vector<std::byte> read_file(const fs::path& path) {
    const auto size = fs::file_size(path);
    if (size > input_limit) throw std::runtime_error("Input exceeds the file-size policy: " + path.string());
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    std::ifstream file(path, std::ios::binary);
    if (!file || !file.read(reinterpret_cast<char*>(data.data()), data.size()))
        throw std::runtime_error("Cannot read input: " + path.string());
    return data;
}

void write_file(const fs::path& path, Bytes data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(reinterpret_cast<const char*>(data.data()), data.size()))
        throw std::runtime_error("Cannot write extraction output: " + path.string());
}

fs::path prepare_output(const fs::path& requested, const std::vector<fs::path>& roots) {
    const auto output = fs::weakly_canonical(fs::absolute(requested));
    for (auto parent = output;; parent = parent.parent_path()) {
        if (fs::exists(parent / ".git"))
            throw std::runtime_error("Shader outputs must be outside every Git work tree");
        if (parent == parent.parent_path()) break;
    }
    for (const auto& root : roots) {
        if (!fs::is_directory(root)) continue;
        const auto input = fs::weakly_canonical(fs::absolute(root));
        const auto [left, right] = std::mismatch(input.begin(), input.end(), output.begin(), output.end());
        static_cast<void>(right);
        if (left == input.end())
            throw std::runtime_error("Extraction output must not be inside an input directory");
    }
    if (fs::exists(output) && (!fs::is_directory(output) || !fs::is_empty(output)))
        throw std::runtime_error("Extraction output must be a new or empty directory");
    fs::create_directories(output / "stages");
    return output;
}

struct Extractor {
    fs::path output;
    std::ofstream inventory;
    std::set<std::string> inputs;
    std::set<std::string> programs;
    std::vector<std::pair<std::string, std::vector<std::byte>>> observations;
    std::map<std::string, std::size_t> matches;
    std::size_t occurrences = 0;
    std::size_t rejected = 0;
    std::size_t containers = 0;
    std::size_t expanded_bytes = 0;
    std::size_t visited_spans = 0;
    std::size_t scanned_bytes = 0;
    std::size_t hashed_bytes = 0;
    bool first = true;

    explicit Extractor(fs::path destination) : output(std::move(destination)),
        inventory(output / "inventory.json", std::ios::binary) {
        if (!inventory) throw std::runtime_error("Cannot create extraction inventory");
        inventory << "{\"schema\":\"armigo-shader-input-v1\",\"runnable\":false,\"occurrences\":[\n";
    }

    void stage(const ContainerStage& record, const std::string& source, std::size_t base) {
        const auto& shader = record.shader;
        if (occurrences >= 1000000)
            throw std::runtime_error("Stage occurrences exceed the extraction policy");
        const auto hash_work = shader.control.size + shader.code.size + shader.c1_offset - shader.code_offset;
        if (hash_work > total_expansion_limit - hashed_bytes)
            throw std::runtime_error("Shader hashing exceeds the aggregate extraction policy");
        hashed_bytes += hash_work;
        const auto control_hash = digest(bytes(shader.control));
        const auto code_hash = digest(bytes(shader.code));
        // Graphics starts at SPH; compute has no header. C1 remains binding
        // data, while the graphics wrapper and control are container metadata.
        const auto program = bytes(shader.code).subspan(shader.code_offset,
            shader.c1_offset - shader.code_offset);
        const auto key = digest(program);
        const auto input = control_hash + "-" + code_hash;
        if (inputs.insert(input).second) {
            fs::create_directory(output / "stages" / input);
            write_file(output / "stages" / input / "control.bin", bytes(shader.control));
            write_file(output / "stages" / input / "code.bin", bytes(shader.code));
        }
        programs.insert(key);
        ++occurrences;
        if (!first) inventory << ",\n";
        first = false;
        inventory << "{\"source\":" << quote(source) << ",\"container_offset\":" << base
            << ",\"descriptor_offset\":" << record.descriptor_offset
            << ",\"stage\":" << static_cast<unsigned>(shader.stage)
            << ",\"input\":" << quote(input) << ",\"program_key\":" << quote(key)
            << ",\"program_bytes\":" << program.size();
        if (record.variation) inventory << ",\"variation\":" << *record.variation;
        if (record.slot) inventory << ",\"slot\":" << static_cast<unsigned>(*record.slot);
        inventory << '}';
        for (const auto& [name, observed] : observations) {
            if (observed.size() >= program.size() &&
                std::equal(program.begin(), program.end(), observed.begin())) {
                ++matches[name];
                armigo::diag::info("Bound program match: %s stage %u key %s bytes %zu\n",
                    name.c_str(), static_cast<unsigned>(shader.stage), key.c_str(), program.size());
            }
        }
    }

    void scan(Bytes input, const std::string& source, unsigned depth = 0) {
        if (depth > 8) throw std::runtime_error("Archive nesting exceeds the extraction policy: " + source);
        if (++visited_spans > 1000000 || input.size() > total_expansion_limit - scanned_bytes)
            throw std::runtime_error("Archive traversal exceeds the aggregate extraction policy");
        scanned_bytes += input.size();
        if (magic(input, "Yaz0")) {
            auto decoded = decompress_yaz0(input, expansion_limit);
            if (auto* error = std::get_if<ArchiveError>(&decoded))
                throw std::runtime_error(source + ": " + error->message());
            const auto& data = std::get<std::vector<std::byte>>(decoded);
            if (data.size() > total_expansion_limit - expanded_bytes)
                throw std::runtime_error("Expanded data exceeds the aggregate extraction policy");
            expanded_bytes += data.size();
            scan(data, source + "::decoded", depth + 1);
            return;
        }
        if (magic(input, "SARC")) {
            auto parsed = parse_sarc(input);
            if (auto* error = std::get_if<ArchiveError>(&parsed))
                throw std::runtime_error(source + ": " + error->message());
            std::size_t index = 0;
            for (const auto& member : std::get<SarcArchive>(parsed).members) {
                const auto name = member.has_name ? std::string(member.name) : std::to_string(index);
                scan(member.data, source + "::" + name, depth + 1);
                ++index;
            }
            return;
        }
        if (magic(input, "BAHS")) {
            auto parsed = parse_bahs(view(input));
            if (auto* error = std::get_if<ContainerError>(&parsed))
                throw std::runtime_error(source + ": " + error->message());
            const auto& archive = std::get<BahsContainer>(parsed);
            ++containers;
            rejected += archive.rejected_candidates;
            for (const auto& record : archive.stages) stage(record, source, 0);
        }
        constexpr std::array signature{std::byte{'B'}, std::byte{'N'}, std::byte{'S'},
            std::byte{'H'}, std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0}};
        auto cursor = input.begin();
        while (cursor != input.end()) {
            const auto found = std::search(cursor, input.end(), signature.begin(), signature.end());
            if (found == input.end()) break;
            const auto offset = static_cast<std::size_t>(found - input.begin());
            auto parsed = parse_bnsh(view(input.subspan(offset)));
            if (auto* error = std::get_if<ContainerError>(&parsed))
                throw std::runtime_error(source + ": " + error->message());
            ++containers;
            for (const auto& record : std::get<BnshContainer>(parsed).stages) stage(record, source, offset);
            cursor = found + signature.size();
        }
    }

    void finish() {
        inventory << "\n],\"unique_inputs\":" << inputs.size() << ",\"unique_programs\":"
            << programs.size() << ",\"containers\":" << containers
            << ",\"rejected_candidates\":" << rejected << ",\"observations\":{";
        bool first_observation = true;
        for (const auto& [name, data] : observations) {
            static_cast<void>(data);
            if (!first_observation) inventory << ',';
            first_observation = false;
            inventory << quote(name) << ':' << matches[name];
        }
        inventory << "}}\n";
        inventory.close();
        if (!inventory) throw std::runtime_error("Cannot finish extraction inventory");
        std::cout << "Extracted " << inputs.size() << " unique inputs / " << programs.size()
            << " program keys from " << occurrences << " stage occurrences; " << rejected
            << " rejected candidates\n";
        if (rejected != 0) throw std::runtime_error("Recognized shader candidates failed structural validation");
        for (const auto& [name, data] : observations) {
            static_cast<void>(data);
            if (matches[name] == 0) throw std::runtime_error("Bound program has no extracted match: " + name);
        }
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<fs::path> roots;
        std::vector<fs::path> observations;
        fs::path output;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help") {
                std::cout << "Usage: armigo-shader-extract --output DIRECTORY [--compare-program FILE] INPUT...\n"
                    << "Reads files/directories, decodes NSO modules, Yaz0/SARC and BNSH/BAHS containers.\n"
                    << "Output must be outside Git work trees. No shader translation is performed.\n";
                return 0;
            }
            if (argument == "--output" || argument == "--compare-program") {
                if (++index == argc) throw std::runtime_error("Option requires a path: " + argument);
                if (argument == "--output") output = argv[index];
                else observations.emplace_back(argv[index]);
            } else if (argument.starts_with("--")) {
                throw std::runtime_error("Unknown option: " + argument);
            } else roots.emplace_back(argument);
        }
        if (output.empty() || roots.empty()) throw std::runtime_error("An output directory and inputs are required");
        std::vector<fs::path> files;
        for (const auto& root : roots) {
            if (fs::is_directory(root)) {
                for (const auto& item : fs::recursive_directory_iterator(root))
                    if (item.is_regular_file()) files.push_back(item.path());
            } else files.push_back(root);
        }
        std::sort(files.begin(), files.end());
        files.erase(std::unique(files.begin(), files.end()), files.end());
        Extractor extractor(prepare_output(output, roots));
        for (const auto& path : observations) extractor.observations.emplace_back(path.string(), read_file(path));
        std::size_t processed = 0;
        for (const auto& path : files) {
            const auto data = read_file(path);
            if (magic(data, "NSO0")) {
                auto module = armigo::loader::read_nso_module(data);
                if (!module.ok()) throw std::runtime_error(path.string() + ": " + armigo::loader::to_string(module.error.code));
                extractor.scan(module.image->bytes, path.string());
            } else extractor.scan(data, path.string());
            if (++processed % 100 == 0) armigo::diag::info("Scanned %zu files\n", processed);
        }
        extractor.finish();
        return 0;
    } catch (const std::exception& error) {
        armigo::diag::error("Shader extraction failed: %s\n", error.what());
        return 1;
    }
}
