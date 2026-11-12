// ============================================================================
// Armigo Tests - Shader Reader Characterization
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Pins the packer's two shader readers. Each case is a fresh temporary tree.
// The expected file stores one FNV-1a-64 digest per case.
// Usage: shader_golden_tests <expected> [--write]
// ARMIGO_GOLDEN_DUMP=<file> also writes the canonical text the digests hash.

// Before the project headers. A C header included first can lock <inttypes.h>
// without the printf macros, and a later <cinttypes> then cannot define PRIx64.
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <cinttypes>

#include "shader_golden_fixtures.h"
#include "native_shaders.h"
#include "shaders.h"
#include "fixture.h"

#include <armigo/shader_native/frontend.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace fx = armigo::pack::shader_golden;
namespace sn = armigo::shader_native;
namespace pl = armigo::shader::pipeline;
namespace snt = armigo::shader_native::test;
using armigo::pack::NativeShaderReport;
using armigo::pack::NativeShaderRequest;
using armigo::pack::ShaderEnumeration;
using armigo::pack::ShaderPackInput;

std::uint64_t fnv1a(std::string_view text) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const unsigned char byte : text) hash = (hash ^ byte) * 0x100000001b3ull;
    return hash;
}

std::uint64_t fnv1a(const std::vector<std::byte>& bytes) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const std::byte byte : bytes) {
        hash = (hash ^ static_cast<unsigned char>(byte)) * 0x100000001b3ull;
    }
    return hash;
}

std::string hex16(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof text, "%016" PRIx64, value);
    return text;
}

void replace_all(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    for (std::size_t at = 0; (at = text.find(from, at)) != std::string::npos; at += to.size()) {
        text.replace(at, from.size(), to);
    }
}

class Recorder {
public:
    void add(const std::string& key, const std::string& canonical) {
        lines_[key] = hex16(fnv1a(canonical));
        dump_ << "== " << key << "\n" << canonical;
    }
    const std::map<std::string, std::string>& lines() const { return lines_; }
    std::string dump() const { return dump_.str(); }

private:
    std::map<std::string, std::string> lines_;
    std::ostringstream dump_;
};

void write_bytes(const fs::path& path, const void* data, std::size_t size) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output || (size != 0 && !output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size)))) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

void write_bytes(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
    write_bytes(path, bytes.data(), bytes.size());
}

void write_bytes(const fs::path& path, const std::vector<std::byte>& bytes) {
    write_bytes(path, bytes.data(), bytes.size());
}

std::vector<std::byte> read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    input.seekg(0, std::ios::end);
    const auto size = static_cast<std::streamoff>(input.tellg());
    input.seekg(0);
    if (size < 0) throw std::runtime_error("Cannot read " + path.string());
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (size != 0 && !input.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return bytes;
}

// One extracted tree. Error text replaces this directory with "<tmp>".
class Tree {
public:
    Tree() {
        const auto pattern = (fs::temp_directory_path() / "armigo-shader-golden-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        if (mkdtemp(name.data()) == nullptr) throw std::runtime_error("Cannot create a temporary directory");
        root_ = name.data();
        fs::create_directory(exefs());
        fs::create_directory(romfs());
        fs::create_directory(stages());
    }
    Tree(const Tree&) = delete;
    Tree& operator=(const Tree&) = delete;
    ~Tree() {
        std::error_code error;
        fs::remove_all(root_, error);
    }

    fs::path exefs() const { return root_ / "exefs"; }
    fs::path romfs() const { return root_ / "romfs"; }
    fs::path stages() const { return root_ / "stages"; }
    fs::path root() const { return root_; }

    void omit_romfs() {
        std::error_code error;
        fs::remove_all(romfs(), error);
    }

    // enumerate_only still publishes an empty package shell, and that shell
    // asks the translator for a 64-digit source version.
    const fs::path& translator() {
        if (!translator_.empty()) return translator_;
        translator_ = root_ / "translate-stand-in";
        const char script[] =
            "#!/bin/sh\n"
            "printf '%s\\n' 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n";
        write_bytes(translator_, script, sizeof script - 1);
        if (chmod(translator_.c_str(), 0755) != 0) {
            throw std::runtime_error("Cannot mark the translator stand-in executable");
        }
        return translator_;
    }

    std::string scrub(std::string text) const {
        std::error_code error;
        const std::string canonical = fs::weakly_canonical(root_, error).string();
        const std::string root = root_.string();
        if (!error && canonical.size() > root.size()) replace_all(text, canonical, "<tmp>");
        replace_all(text, root, "<tmp>");
        if (!error && !canonical.empty() && canonical.size() <= root.size()) replace_all(text, canonical, "<tmp>");
        return text;
    }

private:
    fs::path root_;
    fs::path translator_;
};

std::string characterize_stages(const std::string& name, Tree& tree, const ShaderPackInput& input) {
    std::ostringstream out;
    out << name << '\n';
    const auto result = armigo::pack::enumerate_shader_stages(tree.exefs(), tree.romfs(), {}, tree.stages(), input);
    if (const auto* error = std::get_if<std::string>(&result)) {
        out << "error=" << tree.scrub(*error) << '\n';
        return out.str();
    }
    const auto& found = std::get<ShaderEnumeration>(result);
    out << "unique=" << found.unique << "\nunsupported=" << found.unsupported
        << "\nstructural_errors=" << found.structural_errors << '\n';
    std::vector<fs::path> directories;
    for (const auto& entry : fs::directory_iterator(tree.stages())) {
        if (entry.is_directory()) directories.push_back(entry.path());
    }
    std::sort(directories.begin(), directories.end(), [](const fs::path& left, const fs::path& right) {
        return left.filename() < right.filename();
    });
    for (const fs::path& directory : directories) {
        out << directory.filename().string() << ' ' << hex16(fnv1a(read_bytes(directory / "control.bin"))) << ' '
            << hex16(fnv1a(read_bytes(directory / "code.bin"))) << '\n';
    }
    std::vector<std::string> errors = found.errors;
    for (std::string& error : errors) error = tree.scrub(error);
    std::sort(errors.begin(), errors.end());
    out << "errors:\n";
    for (const std::string& error : errors) out << error << '\n';
    return out.str();
}

std::string characterize_native(const std::string& name, Tree& tree,
                                const std::variant<std::string, NativeShaderReport>& result) {
    std::ostringstream out;
    out << name << '\n';
    if (const auto* error = std::get_if<std::string>(&result)) {
        out << "error=" << tree.scrub(*error) << '\n';
        return out.str();
    }
    const auto& report = std::get<NativeShaderReport>(result);
    out << "enumerated=" << report.enumerated << "\nunique=" << report.unique << "\ntranslated=" << report.translated
        << "\ncompiled=" << report.compiled << "\nduplicates=" << report.duplicates << "\nby_stage:\n";
    for (const auto& [stage, count] : report.by_stage) out << stage << '=' << count << '\n';
    out << "refusals:\n";
    for (const auto& [reason, count] : report.refusals) out << reason << '=' << count << '\n';
    out << "admitted:\n";
    for (const std::string& line : report.admitted) out << line << '\n';
    return out.str();
}

ShaderPackInput containers(armigo::compat::ShaderContainerSet kind) {
    ShaderPackInput input;
    input.settings.containers = kind;
    return input;
}

void add_stages(Recorder& recorder, const std::string& name, const ShaderPackInput& input,
                const std::function<void(Tree&)>& prepare) {
    Tree tree;
    prepare(tree);
    recorder.add(name, characterize_stages(name, tree, input));
}

std::variant<std::string, NativeShaderReport> read_native(Tree& tree, const fs::path& nup, const fs::path& output,
                                                          const std::vector<std::string>& members,
                                                          const std::vector<std::string>& identities) {
    std::error_code error;
    fs::remove_all(output, error);
    NativeShaderRequest request;
    request.nup = nup;
    request.output = output;
    request.translate_tool = tree.translator();
    request.members = members;
    request.identities = identities;
    request.enumerate_only = true;
    request.jobs = 1;
    return armigo::pack::build_native_shader_package(request);
}

void require_census(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string file_text(const fs::path& path) {
    const std::vector<std::byte> bytes = read_bytes(path);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::vector<std::string> directory_names(const fs::path& directory) {
    std::vector<std::string> names;
    std::error_code error;
    if (!fs::exists(directory, error)) return names;
    for (const auto& entry : fs::directory_iterator(directory)) names.push_back(entry.path().filename().string());
    std::sort(names.begin(), names.end());
    return names;
}

struct TranslatedVariant {
    std::string filename;
    std::string source;
    std::string entry_point;
    std::string stage;
};

// Same front-end options the packer passes for this depth.
TranslatedVariant translate_variant(const sn::Program& program, sn::NativeDepth depth, bool clip_depth) {
    sn::FrontendOptions options;
    options.pixel_rate = sn::PixelRateState::disabled;
    options.vertex_fetch = false;
    const char* label = "stage";
    if (clip_depth) {
        const bool zero = depth == sn::NativeDepth::zero_to_one;
        options.vertex_depth = zero ? pl::DepthConvention::zero_to_one : pl::DepthConvention::negative_one_to_one;
        label = zero ? "zero-to-one" : "minus-one-to-one";
    } else if (program.stage == sn::Stage::fragment) {
        options.fragment_position = pl::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
    }
    auto translated = sn::translate_native(program, options);
    if (const auto* diagnostic = std::get_if<pl::Diagnostic>(&translated)) {
        throw std::runtime_error("translator refused " + diagnostic->message);
    }
    const auto& artifact = std::get<pl::MslArtifact>(translated);
    TranslatedVariant variant;
    variant.filename = program.stage_identity + '-' + label + ".metal";
    variant.source = artifact.source;
    variant.entry_point = artifact.entry_point;
    variant.stage = program.stage == sn::Stage::vertex ? "vertex" : "fragment";
    return variant;
}

std::variant<std::string, NativeShaderReport> build_graphics(Tree& tree, const fs::path& nup, const fs::path& output,
                                                            bool translate_only, const std::optional<fs::path>& sources,
                                                            std::size_t maximum_source_bytes) {
    std::error_code error;
    fs::remove_all(output, error);
    NativeShaderRequest request;
    request.nup = nup;
    request.output = output;
    request.translate_tool = tree.translator();
    request.jobs = 1;
    request.translate_only = translate_only;
    request.source_directory = sources;
    request.maximum_source_bytes = maximum_source_bytes;
    return armigo::pack::build_native_shader_package(request);
}

NativeShaderReport expect_report(std::variant<std::string, NativeShaderReport> result) {
    if (const auto* error = std::get_if<std::string>(&result)) throw std::runtime_error("native package: " + *error);
    return std::get<NativeShaderReport>(std::move(result));
}

std::uint64_t counted(const std::map<std::string, std::uint64_t>& counts, const std::string& key) {
    const auto found = counts.find(key);
    return found == counts.end() ? 0 : found->second;
}

std::string describe_report(const NativeShaderReport& report) {
    std::string text = "translated=" + std::to_string(report.translated) + " compiled=" + std::to_string(report.compiled) +
                       " refusals";
    for (const auto& [key, value] : report.refusals) text += ' ' + key + '=' + std::to_string(value);
    if (report.refusals.empty()) text += " none";
    return text;
}

// A translate-only census of one synthetic vertex and fragment. The fragment
// carries extra moves so one global source limit can refuse that stage and
// admit the vertex.
void check_census_sources() {
    snt::Fixture vertex({snt::exit_instruction()}, sn::Stage::vertex);
    std::vector<sn::Instruction> words;
    for (unsigned index = 0; index < 96; ++index) words.push_back(snt::move(index % 8, 0x3f800000u + index));
    words.push_back(snt::exit_instruction());
    snt::Fixture fragment(std::move(words), sn::Stage::fragment);
    const sn::Program vertex_program = vertex.program();
    const sn::Program fragment_program = fragment.program();
    const std::vector<TranslatedVariant> variants = {
        translate_variant(vertex_program, sn::NativeDepth::minus_one_to_one, true),
        translate_variant(vertex_program, sn::NativeDepth::zero_to_one, true),
        translate_variant(fragment_program, sn::NativeDepth::stage, false),
    };
    for (const TranslatedVariant& variant : variants) {
        require_census(!variant.source.empty(), "translator produced an empty source for " + variant.filename);
        require_census(!variant.entry_point.empty() && variant.source.find(variant.entry_point) != std::string::npos,
                       "translator source omits its entry point in " + variant.filename);
    }

    Tree tree;
    const fs::path nup = tree.root() / "title.nup";
    write_bytes(nup, fx::nup_archive({fx::RomMember{
                         "Shader/graphics.bin", fx::native_graphics_bnsh(vertex.control, vertex.code, fragment.control,
                                                                        fragment.code)}}));

    const fs::path sources = tree.root() / "sources";
    fs::create_directory(sources);
    const NativeShaderReport written =
        expect_report(build_graphics(tree, nup, tree.root() / "out-sources", true, sources, sn::maximum_native_source_bytes));
    require_census(written.unique == 2 && written.translated == 2, "source census " + describe_report(written));
    require_census(counted(written.refusals, "source-size") == 0, "default limit counted source-size");
    std::vector<std::string> expected_names;
    for (const TranslatedVariant& variant : variants) expected_names.push_back(variant.filename);
    std::sort(expected_names.begin(), expected_names.end());
    const std::vector<std::string> written_names = directory_names(sources);
    if (written_names != expected_names) {
        std::string message = "source directory entries:";
        for (const std::string& name : written_names) message += ' ' + name;
        message += "; expected";
        for (const std::string& name : expected_names) message += ' ' + name;
        throw std::runtime_error(message);
    }
    for (const TranslatedVariant& variant : variants) {
        require_census(file_text(sources / variant.filename) == variant.source,
                       "written source differs from the translator: " + variant.filename);
    }

    const fs::path quiet = tree.root() / "quiet-sources";
    fs::create_directory(quiet);
    const NativeShaderReport silent = expect_report(
        build_graphics(tree, nup, tree.root() / "out-quiet", true, std::nullopt, sn::maximum_native_source_bytes));
    require_census(silent.translated == 2, "census without a source directory " + describe_report(silent));
    require_census(directory_names(quiet).empty(), "a census without source_directory wrote files");

    std::vector<std::size_t> vertex_sizes;
    std::vector<std::size_t> fragment_sizes;
    for (const TranslatedVariant& variant : variants) {
        (variant.stage == "vertex" ? vertex_sizes : fragment_sizes).push_back(variant.source.size());
    }
    const std::size_t vertex_min = *std::min_element(vertex_sizes.begin(), vertex_sizes.end());
    const std::size_t vertex_max = *std::max_element(vertex_sizes.begin(), vertex_sizes.end());
    const std::size_t fragment_min = *std::min_element(fragment_sizes.begin(), fragment_sizes.end());
    const std::size_t fragment_max = *std::max_element(fragment_sizes.begin(), fragment_sizes.end());
    std::size_t limit = 0;
    std::size_t oversized_count = 0;
    std::size_t kept_variants = 0;
    std::string oversized_stage;
    std::string kept_stage;
    std::string oversized_identity;
    std::string kept_identity;
    if (fragment_min > vertex_max) {
        limit = vertex_max;
        oversized_count = fragment_sizes.size();
        kept_variants = vertex_sizes.size();
        oversized_stage = "fragment";
        kept_stage = "vertex";
        oversized_identity = fragment_program.stage_identity;
        kept_identity = vertex_program.stage_identity;
    } else if (vertex_min > fragment_max) {
        limit = fragment_max;
        oversized_count = vertex_sizes.size();
        kept_variants = fragment_sizes.size();
        oversized_stage = "vertex";
        kept_stage = "fragment";
        oversized_identity = vertex_program.stage_identity;
        kept_identity = fragment_program.stage_identity;
    } else {
        throw std::runtime_error("synthetic stages do not separate by source size: vertex " + std::to_string(vertex_min) +
                                 "-" + std::to_string(vertex_max) + " fragment " + std::to_string(fragment_min) + "-" +
                                 std::to_string(fragment_max));
    }
    for (const bool translate_only : {true, false}) {
        const char* const mode = translate_only ? "translate-only" : "package";
        const fs::path output = tree.root() / (translate_only ? "out-census-limit" : "out-package-limit");
        const NativeShaderReport report =
            expect_report(build_graphics(tree, nup, output, translate_only, std::nullopt, limit));
        require_census(counted(report.refusals, "source-size") == oversized_count,
                       std::string(mode) + " source-size count " + describe_report(report));
        require_census(counted(report.translated_by_stage, oversized_stage) == 0,
                       std::string(mode) + " published the oversized stage in the report: " + describe_report(report));
        require_census(counted(report.translated_by_stage, kept_stage) == 1,
                       std::string(mode) + " dropped the stage under the limit: " + describe_report(report));
        if (translate_only) continue;
        require_census(counted(report.compiled_by_stage, kept_stage) == kept_variants,
                       std::string(mode) + " did not compile the stage under the limit: " + describe_report(report));
        const fs::path stages = output / "native" / "stages";
        require_census(!fs::exists(stages / oversized_identity), "oversized stage was published");
        require_census(fs::is_directory(stages / kept_identity), "kept stage was not published: " + describe_report(report));
        const std::string manifest = file_text(output / "native" / "manifest.json");
        require_census(manifest.find(oversized_identity) == std::string::npos, "manifest names the oversized stage");
        require_census(manifest.find(kept_identity) != std::string::npos, "manifest omits the stage under the limit");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <expected> [--write]\n", argv[0]);
        return 2;
    }
    const fs::path expected = argv[1];
    const bool write = argc > 2 && std::string_view(argv[2]) == "--write";
    try {
        check_census_sources();
        Recorder recorder;
        const auto bnsh = containers(armigo::compat::ShaderContainerSet::Bnsh);
        const std::vector<std::uint8_t> graphics = fx::vertex_fragment_bnsh();
        add_stages(recorder, "bnsh_loose", bnsh, [&](Tree& tree) {
            const std::uint8_t note[] = {1, 2, 3, 4};
            write_bytes(tree.exefs() / "note.bin", note, sizeof note);
            write_bytes(tree.romfs() / "a.bnsh", graphics);
        });
        add_stages(recorder, "bnsh_duplicate", bnsh, [&](Tree& tree) {
            write_bytes(tree.romfs() / "a.bnsh", graphics);
            write_bytes(tree.romfs() / "b.bnsh", graphics);
            std::vector<std::uint8_t> embedded(1 + graphics.size(), 0xa5);
            std::memcpy(embedded.data() + 1, graphics.data(), graphics.size());
            write_bytes(tree.romfs() / "embedded.bin", embedded);
        });
        add_stages(recorder, "bnsh_in_sarc_yaz0", bnsh, [&](Tree& tree) {
            write_bytes(tree.romfs() / "pack.yaz0",
                        fx::yaz0_literals(fx::sarc_member(graphics, "stage.bnsh")));
        });
        add_stages(recorder, "bahs", bnsh, [&](Tree& tree) {
            write_bytes(tree.romfs() / "shader.bahs", fx::bahs_container());
        });
        add_stages(recorder, "structural", bnsh,
                  [&](Tree& tree) { write_bytes(tree.romfs() / "broken.bnsh", fx::bnsh_structural()); });
        add_stages(recorder, "unsupported_stage", bnsh,
                  [&](Tree& tree) { write_bytes(tree.romfs() / "geometry.bnsh", fx::geometry_bnsh()); });
        add_stages(recorder, "missing_romfs", bnsh, [](Tree& tree) { tree.omit_romfs(); });

        add_stages(recorder, "gfd_cache", containers(armigo::compat::ShaderContainerSet::GfdCache), [](Tree& tree) {
            write_bytes(tree.romfs() / "cache.cpk", fx::gsc_cpk());
            write_bytes(tree.exefs() / "main", fx::nso_without_shaders());
        });
        ShaderPackInput missing_cache = containers(armigo::compat::ShaderContainerSet::GfdCache);
        missing_cache.settings.caches = {"missing.gsc"};
        add_stages(recorder, "gfd_missing_cache", missing_cache, [](Tree&) {});

        add_stages(recorder, "glslc_output", containers(armigo::compat::ShaderContainerSet::Glslc), [](Tree& tree) {
            write_bytes(tree.exefs() / "main", fx::nso_with_glslc_vertex());
        });
        add_stages(recorder, "glslc_none", containers(armigo::compat::ShaderContainerSet::Glslc), [](Tree& tree) {
            write_bytes(tree.exefs() / "main", fx::nso_without_shaders());
        });

        {
            Tree tree;
            const fs::path nup = tree.root() / "title.nup";
            write_bytes(nup, fx::native_shader_nup());
            const auto all = read_native(tree, nup, tree.root() / "out-all", {}, {});
            recorder.add("native_all", characterize_native("native_all", tree, all));
            recorder.add("native_member", characterize_native("native_member", tree,
                                                              read_native(tree, nup, tree.root() / "out-member",
                                                                          {"b.bin"}, {})));
            recorder.add("native_missing_member",
                         characterize_native("native_missing_member", tree,
                                             read_native(tree, nup, tree.root() / "out-missing", {"nope.bin"}, {})));
            std::string identity;
            if (const auto* report = std::get_if<NativeShaderReport>(&all)) {
                if (!report->admitted.empty()) {
                    const auto gap = report->admitted.front().find(' ');
                    identity = report->admitted.front().substr(0, gap);
                }
            }
            if (identity.empty()) {
                recorder.add("native_identity", "native_identity\nerror=no admitted stage identity\n");
            } else {
                recorder.add("native_identity",
                             characterize_native("native_identity", tree,
                                                 read_native(tree, nup, tree.root() / "out-identity", {}, {identity})));
            }
        }

        if (const char* dump = std::getenv("ARMIGO_GOLDEN_DUMP"); dump && *dump) {
            std::ofstream out(dump);
            out << recorder.dump();
        }
        if (write) {
            std::ofstream out(expected);
            if (!out) {
                std::fprintf(stderr, "cannot write %s\n", expected.string().c_str());
                return 1;
            }
            for (const auto& [key, value] : recorder.lines()) out << key << " = " << value << "\n";
            std::printf("wrote %zu cases\n", recorder.lines().size());
            return 0;
        }
        std::map<std::string, std::string> wanted;
        std::ifstream in(expected);
        for (std::string line; std::getline(in, line);) {
            const auto separator = line.rfind(" = ");
            if (separator != std::string::npos) wanted[line.substr(0, separator)] = line.substr(separator + 3);
        }
        unsigned failures = 0;
        for (const auto& [key, value] : recorder.lines()) {
            const auto found = wanted.find(key);
            if (found == wanted.end()) {
                std::fprintf(stderr, "new case: %s\n", key.c_str());
                ++failures;
            } else if (found->second != value) {
                std::fprintf(stderr, "changed: %s\n", key.c_str());
                ++failures;
            }
        }
        for (const auto& [key, value] : wanted) {
            static_cast<void>(value);
            if (!recorder.lines().count(key)) {
                std::fprintf(stderr, "missing case: %s\n", key.c_str());
                ++failures;
            }
        }
        if (failures) {
            std::fprintf(stderr, "%u shader cases differ from %s (ARMIGO_GOLDEN_DUMP=<file> prints them)\n", failures,
                         expected.string().c_str());
            return 1;
        }
        std::printf("%zu shader cases match\n", recorder.lines().size());
        return 0;
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "shader golden fixture failed: %s\n", failure.what());
        return 1;
    }
}
