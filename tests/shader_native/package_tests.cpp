// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/package.h>

#include <CommonCrypto/CommonDigest.h>

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <random>
#include <stdexcept>

namespace sn = armigo::shader_native;
namespace fs = std::filesystem;

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct TemporaryDirectory {
    fs::path path;
    explicit TemporaryDirectory() {
        std::random_device random;
        path = fs::temp_directory_path() / ("armigo-native-package-" + std::to_string(random()));
        fs::create_directory(path);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

std::vector<std::byte> bytes(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), reinterpret_cast<const std::byte*>(text.data()) + text.size()};
}

std::string identity(char digit) { return std::string(64, digit); }

sn::NativeCounts counts() {
    sn::NativeCounts value;
    value.enumerated = 4;
    value.unique = 2;
    value.translated = 2;
    value.compiled = 3;
    value.duplicates = 1;
    value.by_stage = {{"vertex", 1}, {"fragment", 1}, {"compute", 0}};
    value.translated_by_stage = value.by_stage;
    value.compiled_by_stage = {{"vertex", 2}, {"fragment", 1}, {"compute", 0}};
    value.refusals = {{"unsupported-stage", 1}, {"metal-compile", 2}};
    return value;
}

sn::NativeStageWrite fragment_stage() {
    sn::NativeStageWrite stage;
    stage.stage_identity = identity('a');
    stage.stage = sn::NativeStageKind::fragment;
    stage.program = bytes("fragment-program-bytes");
    sn::NativeVariantWrite variant;
    variant.depth = sn::NativeDepth::stage;
    variant.reflection = bytes("{\"schema\":\"reflection\"}\n");
    variant.source = "fragment metal source\n";
    variant.library = bytes("fragment-library");
    stage.variants.push_back(std::move(variant));
    return stage;
}

sn::NativeStageWrite vertex_stage() {
    sn::NativeStageWrite stage;
    stage.stage_identity = identity('b');
    stage.stage = sn::NativeStageKind::vertex;
    stage.program = bytes("vertex-program-bytes");
    for (const sn::NativeDepth depth : {sn::NativeDepth::minus_one_to_one, sn::NativeDepth::zero_to_one}) {
        sn::NativeVariantWrite variant;
        variant.depth = depth;
        variant.reflection = bytes(depth == sn::NativeDepth::zero_to_one ? "zero-reflection" : "minus-reflection");
        variant.source = depth == sn::NativeDepth::zero_to_one ? "zero source" : "minus source";
        variant.library = bytes(depth == sn::NativeDepth::zero_to_one ? "zero-library" : "minus-library");
        stage.variants.push_back(std::move(variant));
    }
    return stage;
}

sn::NativePackage publish(const fs::path& root) {
    auto writer = sn::NativeSectionWriter::create(root);
    check(std::holds_alternative<sn::NativeSectionWriter>(writer), "writer");
    auto& section = std::get<sn::NativeSectionWriter>(writer);
    check(std::holds_alternative<std::monostate>(section.add(fragment_stage())), "fragment add");
    check(std::holds_alternative<std::monostate>(section.add(vertex_stage())), "vertex add");
    auto finished = section.finish(counts(), {});
    check(std::holds_alternative<sn::NativeSection>(finished), "finish");
    auto opened = sn::NativePackage::open(root);
    check(std::holds_alternative<sn::NativePackage>(opened), "open");
    return std::get<sn::NativePackage>(std::move(opened));
}

std::string sha256_hex(std::string_view text) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH] = {};
    CC_SHA256(text.data(), static_cast<CC_LONG>(text.size()), digest);
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(CC_SHA256_DIGEST_LENGTH * 2, '0');
    for (std::size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; ++index) {
        result[index * 2] = digits[digest[index] >> 4];
        result[index * 2 + 1] = digits[digest[index] & 0x0f];
    }
    return result;
}

std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    check(static_cast<bool>(stream), "manifest readable");
    std::string text;
    stream.seekg(0, std::ios::end);
    text.resize(static_cast<std::size_t>(stream.tellg()));
    stream.seekg(0);
    stream.read(text.data(), static_cast<std::streamsize>(text.size()));
    check(static_cast<bool>(stream), "manifest readable");
    return text;
}

void omitted_source() {
    TemporaryDirectory directory;
    const auto stage = fragment_stage();
    const std::string source = stage.variants.front().source;
    auto created = sn::NativeSectionWriter::create(directory.path, false);
    check(std::holds_alternative<sn::NativeSectionWriter>(created), "omitted writer");
    auto& writer = std::get<sn::NativeSectionWriter>(created);
    check(std::holds_alternative<std::monostate>(writer.add(stage)), "omitted add");
    auto finished = writer.finish(counts(), {});
    check(std::holds_alternative<sn::NativeSection>(finished), "omitted finish");

    const fs::path variant = directory.path / "native" / "stages" / identity('a') / "stage";
    check(!fs::exists(variant / "source.metal"), "omitted source absent");
    const std::string digest = sha256_hex(source);
    const std::string manifest = read_text(directory.path / "native" / "manifest.json");
    check(manifest.find("\"source_digest\":\"" + digest + "\"") != std::string::npos, "omitted source digest");
    check(manifest.find("\"source_size\":" + std::to_string(source.size())) != std::string::npos, "omitted source size");

    auto opened = sn::NativePackage::open(directory.path);
    check(std::holds_alternative<sn::NativePackage>(opened), "omitted open");
    const auto& package = std::get<sn::NativePackage>(opened);
    check(package.section().entries.front().variants.front().source_digest == digest, "omitted section digest");
    check(package.section().entries.front().variants.front().source_size == source.size(), "omitted section size");
    sn::NativeLookup fragment;
    fragment.stage = sn::NativeStageKind::fragment;
    auto hit = package.find(identity('a'), fragment);
    check(std::holds_alternative<sn::NativeStageFiles>(hit), "omitted hit");
    const auto& files = std::get<sn::NativeStageFiles>(hit);
    check(files.source.empty(), "omitted source");
    check(files.program == bytes("fragment-program-bytes"), "omitted program");
    check(files.reflection == bytes("{\"schema\":\"reflection\"}\n"), "omitted reflection");
    check(files.library == bytes("fragment-library"), "omitted library");

    {
        std::fstream stream(variant / "shader.metallib", std::ios::binary | std::ios::in | std::ios::out);
        check(stream.good(), "omitted library file");
        stream.seekp(0);
        stream.put('X');
    }
    auto corrupt = package.find(identity('a'), fragment);
    check(std::holds_alternative<sn::NativeMiss>(corrupt) &&
              std::get<sn::NativeMiss>(corrupt).reason == sn::NativeMiss::Reason::corrupt,
          "omitted corrupt library");
}

sn::NativeStageWrite one_variant(char digit, sn::NativeStageKind kind, sn::NativeDepth depth, std::string_view name) {
    sn::NativeStageWrite stage;
    stage.stage_identity = identity(digit);
    stage.stage = kind;
    stage.program = bytes(std::string(name) + "-program");
    sn::NativeVariantWrite variant;
    variant.depth = depth;
    variant.reflection = bytes(std::string(name) + "-reflection");
    variant.source = std::string(name) + " source";
    variant.library = bytes(std::string(name) + "-library");
    stage.variants.push_back(std::move(variant));
    return stage;
}

sn::NativeStageWrite mesh_stage(char digit) {
    sn::NativeStageWrite stage = one_variant(digit, sn::NativeStageKind::mesh, sn::NativeDepth::minus_one_to_one, "mesh-minus");
    stage.program = bytes("mesh-program");
    sn::NativeVariantWrite zero;
    zero.depth = sn::NativeDepth::zero_to_one;
    zero.reflection = bytes("mesh-zero-reflection");
    zero.source = "mesh zero source";
    zero.library = bytes("mesh-zero-library");
    stage.variants.push_back(std::move(zero));
    stage.variants.front().reflection = bytes("mesh-minus-reflection");
    stage.variants.front().source = "mesh minus source";
    stage.variants.front().library = bytes("mesh-minus-library");
    return stage;
}

sn::NativePackage finish_stages(const fs::path& root, std::initializer_list<sn::NativeStageWrite> stages,
                                sn::NativeSpecialization specialization = {}) {
    auto created = sn::NativeSectionWriter::create(root);
    check(std::holds_alternative<sn::NativeSectionWriter>(created), "task mesh writer");
    auto& writer = std::get<sn::NativeSectionWriter>(created);
    for (const sn::NativeStageWrite& stage : stages)
        check(std::holds_alternative<std::monostate>(writer.add(stage)), "task mesh add");
    auto finished = writer.finish(counts(), specialization);
    check(std::holds_alternative<sn::NativeSection>(finished), "task mesh finish");
    auto opened = sn::NativePackage::open(root);
    check(std::holds_alternative<sn::NativePackage>(opened), "task mesh open");
    return std::get<sn::NativePackage>(std::move(opened));
}

// A section whose stages are only vertex, fragment, and compute still opens.
void legacy_manifest_loads() {
    TemporaryDirectory directory;
    const sn::NativePackage package = publish(directory.path);
    check(package.section().entries.size() == 2, "legacy entries");
    for (const auto& entry : package.section().entries) {
        check(entry.stage == sn::NativeStageKind::vertex || entry.stage == sn::NativeStageKind::fragment ||
                  entry.stage == sn::NativeStageKind::compute,
              "legacy stage kind");
    }
    const std::string manifest = read_text(directory.path / "native" / "manifest.json");
    check(manifest.find("\"stage\":\"task\"") == std::string::npos, "legacy manifest has no task kind");
    check(manifest.find("\"stage\":\"mesh\"") == std::string::npos, "legacy manifest has no mesh kind");
    sn::NativeLookup fragment;
    fragment.stage = sn::NativeStageKind::fragment;
    check(std::holds_alternative<sn::NativeStageFiles>(package.find(identity('a'), fragment)), "legacy fragment");
}

void task_and_mesh_stages() {
    TemporaryDirectory directory;
    const sn::NativePackage package = finish_stages(
        directory.path, {one_variant('c', sn::NativeStageKind::task, sn::NativeDepth::stage, "task"), mesh_stage('d')});
    check(package.section().entries.size() == 2, "task mesh entries");
    bool saw_task = false;
    bool saw_mesh = false;
    for (const auto& entry : package.section().entries) {
        if (entry.stage_identity == identity('c')) {
            saw_task = entry.stage == sn::NativeStageKind::task && entry.variants.size() == 1 &&
                       entry.variants.front().depth == sn::NativeDepth::stage;
        }
        if (entry.stage_identity == identity('d')) {
            saw_mesh = entry.stage == sn::NativeStageKind::mesh && entry.variants.size() == 2;
        }
    }
    check(saw_task, "task entry");
    check(saw_mesh, "mesh entry");

    sn::NativeLookup task;
    task.stage = sn::NativeStageKind::task;
    auto task_hit = package.find(identity('c'), task);
    check(std::holds_alternative<sn::NativeStageFiles>(task_hit), "task hit");
    check(std::get<sn::NativeStageFiles>(task_hit).library == bytes("task-library"), "task library");
    check(std::get<sn::NativeStageFiles>(task_hit).program == bytes("task-program"), "task program");

    sn::NativeLookup mesh;
    mesh.stage = sn::NativeStageKind::mesh;
    mesh.vertex_depth_zero_to_one = false;
    auto minus = package.find(identity('d'), mesh);
    check(std::holds_alternative<sn::NativeStageFiles>(minus), "mesh minus hit");
    check(std::get<sn::NativeStageFiles>(minus).library == bytes("mesh-minus-library"), "mesh minus library");
    check(std::get<sn::NativeStageFiles>(minus).reflection == bytes("mesh-minus-reflection"), "mesh minus reflection");
    mesh.vertex_depth_zero_to_one = true;
    auto zero = package.find(identity('d'), mesh);
    check(std::holds_alternative<sn::NativeStageFiles>(zero), "mesh zero hit");
    check(std::get<sn::NativeStageFiles>(zero).library == bytes("mesh-zero-library"), "mesh zero library");
    check(std::get<sn::NativeStageFiles>(zero).source == "mesh zero source", "mesh zero source");

    TemporaryDirectory narrow_directory;
    sn::NativeSpecialization one_depth;
    one_depth.vertex_depth_minus_one_to_one = false;
    const sn::NativePackage narrow = finish_stages(narrow_directory.path, {mesh_stage('e')}, one_depth);
    check(!narrow.section().specialization.vertex_depth_minus_one_to_one, "narrow omits minus");
    check(narrow.section().specialization.vertex_depth_zero_to_one, "narrow keeps zero");
    sn::NativeLookup wrong;
    wrong.stage = sn::NativeStageKind::mesh;
    wrong.vertex_depth_zero_to_one = false;
    auto missed = narrow.find(identity('e'), wrong);
    check(std::holds_alternative<sn::NativeMiss>(missed) &&
              std::get<sn::NativeMiss>(missed).reason == sn::NativeMiss::Reason::specialization,
          "wrong mesh depth");
    wrong.vertex_depth_zero_to_one = true;
    check(std::holds_alternative<sn::NativeStageFiles>(narrow.find(identity('e'), wrong)), "admitted mesh depth");
}
} // namespace

int main() {
    try {
        TemporaryDirectory directory;
        const sn::NativePackage package = publish(directory.path);
        check(package.section().counts.enumerated == 4, "enumerated");
        check(package.section().counts.refusals.at("unsupported-stage") == 1, "refusal");
        check(package.section().counts.compiled_by_stage.at("vertex") == 2, "compiled vertex");
        check(package.section().entries.size() == 2, "entries");
        check(package.section().entries.front().stage_identity == identity('a'), "sorted");
        check(package.section().specialization.pixel_rate_disabled, "pixel rate");
        check(!package.section().specialization.vertex_fetch, "vertex fetch");
        check(package.section().specialization.vertex_depth_zero_to_one, "zero depth");
        check(package.section().specialization.fragment_position_matches, "fragment position");

        sn::NativeLookup fragment;
        fragment.stage = sn::NativeStageKind::fragment;
        auto hit = package.find(identity('a'), fragment);
        check(std::holds_alternative<sn::NativeStageFiles>(hit), "fragment hit");
        const auto& files = std::get<sn::NativeStageFiles>(hit);
        check(files.library == bytes("fragment-library"), "library bytes");
        check(files.source == "fragment metal source\n", "source");
        check(files.program == bytes("fragment-program-bytes"), "program");

        sn::NativeLookup vertex;
        vertex.stage = sn::NativeStageKind::vertex;
        vertex.vertex_depth_zero_to_one = true;
        auto zero = package.find(identity('b'), vertex);
        check(std::holds_alternative<sn::NativeStageFiles>(zero), "zero hit");
        check(std::get<sn::NativeStageFiles>(zero).library == bytes("zero-library"), "zero library");
        const fs::path zero_dir = directory.path / "native" / "stages" / identity('b') / "zero-to-one";
        fs::remove(zero_dir / "source.metal");
        auto source_free = package.find(identity('b'), vertex);
        check(std::holds_alternative<sn::NativeStageFiles>(source_free), "source-free hit");
        check(std::get<sn::NativeStageFiles>(source_free).source.empty(), "source-free text");
        check(std::get<sn::NativeStageFiles>(source_free).library == bytes("zero-library"), "source-free library");
        check(std::get<sn::NativeStageFiles>(source_free).reflection == bytes("zero-reflection"), "source-free reflection");
        check(std::get<sn::NativeStageFiles>(source_free).program == bytes("vertex-program-bytes"), "source-free program");
        {
            std::fstream stream(zero_dir / "shader.metallib", std::ios::binary | std::ios::in | std::ios::out);
            check(stream.good(), "source-free library file");
            stream.seekp(0);
            stream.put('X');
        }
        auto source_free_corrupt = package.find(identity('b'), vertex);
        check(std::holds_alternative<sn::NativeMiss>(source_free_corrupt) &&
                  std::get<sn::NativeMiss>(source_free_corrupt).reason == sn::NativeMiss::Reason::corrupt,
              "source-free corrupt library");
        vertex.vertex_depth_zero_to_one = false;
        auto minus = package.find(identity('b'), vertex);
        check(std::holds_alternative<sn::NativeStageFiles>(minus), "minus hit");
        check(std::get<sn::NativeStageFiles>(minus).source == "minus source", "minus source");

        vertex.vertex_fetch = true;
        auto fetch = package.find(identity('b'), vertex);
        check(std::holds_alternative<sn::NativeMiss>(fetch) &&
                  std::get<sn::NativeMiss>(fetch).reason == sn::NativeMiss::Reason::specialization,
              "vertex fetch miss");
        fragment.fragment_position_matches = false;
        auto position = package.find(identity('a'), fragment);
        check(std::holds_alternative<sn::NativeMiss>(position) &&
                  std::get<sn::NativeMiss>(position).reason == sn::NativeMiss::Reason::specialization,
              "fragment position miss");
        fragment.fragment_position_matches = true;
        fragment.pixel_rate_disabled = false;
        auto rate = package.find(identity('a'), fragment);
        check(std::holds_alternative<sn::NativeMiss>(rate) &&
                  std::get<sn::NativeMiss>(rate).reason == sn::NativeMiss::Reason::specialization,
              "pixel rate miss");
        fragment.pixel_rate_disabled = true;
        auto missing = package.find(identity('c'), fragment);
        check(std::holds_alternative<sn::NativeMiss>(missing) &&
                  std::get<sn::NativeMiss>(missing).reason == sn::NativeMiss::Reason::absent,
              "absent");

        const fs::path library = directory.path / "native" / "stages" / identity('a') / "stage" / "shader.metallib";
        {
            std::fstream stream(library, std::ios::binary | std::ios::in | std::ios::out);
            check(stream.good(), "library file");
            stream.seekp(0);
            stream.put('X');
        }
        auto corrupt = package.find(identity('a'), fragment);
        check(std::holds_alternative<sn::NativeMiss>(corrupt) &&
                  std::get<sn::NativeMiss>(corrupt).reason == sn::NativeMiss::Reason::corrupt,
              "corrupt library");

        {
            std::fstream manifest(directory.path / "native" / "manifest.json", std::ios::binary | std::ios::in | std::ios::out);
            check(manifest.good(), "manifest file");
            manifest.seekp(0);
            manifest.put('x');
        }
        auto reopened = sn::NativePackage::open(directory.path);
        check(std::holds_alternative<std::string>(reopened), "damaged manifest");

        auto missing_section = sn::NativePackage::open(directory.path / "absent");
        check(std::holds_alternative<std::string>(missing_section), "missing section");
        omitted_source();
        legacy_manifest_loads();
        task_and_mesh_stages();
        std::cout << "Native shader package tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Native shader package tests failed: " << error.what() << "\n";
        return 1;
    }
}
