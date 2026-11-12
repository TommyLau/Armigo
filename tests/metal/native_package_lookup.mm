// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// A packaged native stage is used at first use. A stage the package does not
// hold is translated, and the two outcomes are counted separately.

#include "nvn_native_stages.h"
#include "entry_table.h"
#include "fixture.h"
#include "programs.h"

#include <armigo/platform/metal_shader_package.h>
#include <armigo/shader_native/package.h>

#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>
#include <tuple>

namespace {
namespace sn = armigo::shader_native;
namespace pl = armigo::shader::pipeline;
namespace gm = armigo::shader::graphics_msl;
namespace nvn = armigo::nvn;

unsigned failures = 0;
#define CHECK(condition)                                                  \
    do {                                                                  \
        if (!(condition)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); \
            ++failures;                                                   \
        }                                                                 \
    } while (false)

std::vector<std::byte> bytes_of(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), reinterpret_cast<const std::byte*>(text.data()) + text.size()};
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        std::random_device random;
        path = std::filesystem::temp_directory_path() / ("armigo-native-lookup-" + std::to_string(random()));
        std::filesystem::create_directory(path);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

template <nvn::Entry E>
auto entry_function() {
    return reinterpret_cast<typename nvn::EntrySignature<E>::type>(nvn::installed_entry(E));
}

// A mesh quad the front end translates: four clip positions, six index bytes.
sn::test::Fixture mesh_quad() {
    const auto store = [](unsigned data, std::uint32_t offset, unsigned mode) {
        auto instruction = sn::test::raw(0x927);
        sn::test::field(instruction, 32, 8, data);
        sn::test::field(instruction, 24, 8, 255);
        sn::test::field(instruction, 40, 24, offset);
        sn::test::field(instruction, 72, 8, mode);
        return instruction;
    };
    std::vector<sn::Instruction> code{sn::test::move(1, 0xbf800000u), sn::test::move(2, 0x3f800000u), sn::test::move(6, 1),
                                      sn::test::move(7, 2), sn::test::move(8, 3)};
    for (unsigned vertex = 0; vertex < 4; ++vertex) {
        code.push_back(store(vertex & 1 ? 2 : 1, 0 * 128 + vertex * 4, 0xf8));
        code.push_back(store(vertex & 2 ? 2 : 1, 1 * 128 + vertex * 4, 0xf8));
        code.push_back(store(255, 2 * 128 + vertex * 4, 0xf8));
        code.push_back(store(2, 3 * 128 + vertex * 4, 0xf8));
    }
    for (unsigned at = 0; const unsigned index : {255u, 6u, 7u, 7u, 6u, 8u}) code.push_back(store(index, 4 + at++, 0x80));
    code.push_back(sn::test::exit_instruction());
    sn::test::Fixture mesh(std::move(code), sn::Stage::mesh);
    sn::test::put(mesh.control, 0x58, 32);
    mesh.code[0x36] = std::byte{1};
    sn::test::put(mesh.control, 0x5c, 0x03000000u);
    sn::test::put(mesh.control, 0x60, 4 | (2u << 16));
    return mesh;
}

std::string launch_record(std::string_view reflection) {
    constexpr std::string_view key = "\"mesh_launch\":";
    const auto at = reflection.find(key);
    if (at == std::string_view::npos) return {};
    const auto end = reflection.find('}', at);
    if (end == std::string_view::npos) return {};
    return std::string(reflection.substr(at, end + 1 - at));
}
} // namespace

int main() {
    using sn::test::Fixture;
    using sn::test::put;
    try {
        const Fixture packaged_fixture({sn::test::move(0, 0x3f800000), sn::test::move(1, 0), sn::test::move(2, 0),
                                        sn::test::move(3, 0x3f800000), sn::test::exit_instruction()});
        const auto program = packaged_fixture.program();
        sn::FrontendOptions options;
        options.pixel_rate = sn::PixelRateState::disabled;
        options.vertex_fetch = false;
        options.fragment_position = pl::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
        gm::EmitOptions emission;
        emission.fragment_position = gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        auto translated = sn::translate_native(program, options);
        CHECK(std::holds_alternative<pl::MslArtifact>(translated));
        if (!std::holds_alternative<pl::MslArtifact>(translated)) return 1;
        const auto& artifact = std::get<pl::MslArtifact>(translated);
        const auto reflection = armigo::platform::metal::native_stage_reflection(artifact, emission, pl::Stage::fragment, program.image);
        const std::vector<std::byte> library = bytes_of("packaged-metal-library");

        // A direct driver storage binding needs no GPU constant-buffer
        // window. Its validated host-binding fact remains in the package.
        {
            auto pair = sn::test::raw(0xb82);
            for (auto [at, width, value] : {std::tuple{16u, 8u, 8u}, {24u, 8u, 255u}, {54u, 5u, 0u},
                                          {38u, 16u, 0x510u}, {73u, 3u, 5u}}) sn::test::field(pair, at, width, value);
            auto load = sn::test::raw(0x381);
            for (auto [at, width, value] : {std::tuple{16u, 8u, 0u}, {24u, 8u, 8u}, {72u, 1u, 1u},
                                          {73u, 3u, 6u}, {77u, 4u, 5u}, {81u, 3u, 7u}}) sn::test::field(load, at, width, value);
            Fixture direct_fixture({pair, load, sn::test::exit_instruction()}); put(direct_fixture.code, 0x48, 0xf);
            const auto direct_program = direct_fixture.program();
            auto direct_translation = sn::translate_native(direct_program, options);
            CHECK(std::holds_alternative<pl::MslArtifact>(direct_translation));
            if (!std::holds_alternative<pl::MslArtifact>(direct_translation)) return 1;
            const auto& direct = std::get<pl::MslArtifact>(direct_translation);
            CHECK(direct.source.find("armigo_guest_address") == std::string::npos);
            CHECK(direct.reflection.resources.size() == 1 &&
                  std::holds_alternative<pl::StorageBufferSource>(std::get<pl::BufferUse>(direct.reflection.resources[0].use).source));
            const auto metadata = armigo::platform::metal::native_stage_reflection(
                direct, emission, pl::Stage::fragment, direct_program.image);
            sn::NativeStageWrite direct_stage;
            direct_stage.stage_identity = direct_program.stage_identity; direct_stage.stage = sn::NativeStageKind::fragment;
            direct_stage.program.assign(direct_program.image.begin(), direct_program.image.end());
            sn::NativeVariantWrite direct_variant;
            direct_variant.depth = sn::NativeDepth::stage; direct_variant.source = direct.source;
            direct_variant.reflection = bytes_of(metadata); direct_variant.library = library;
            direct_stage.variants.push_back(std::move(direct_variant));
            TemporaryDirectory direct_directory;
            auto direct_writer = sn::NativeSectionWriter::create(direct_directory.path);
            CHECK(std::holds_alternative<sn::NativeSectionWriter>(direct_writer));
            if (!std::holds_alternative<sn::NativeSectionWriter>(direct_writer)) return 1;
            const auto accepted = std::get<sn::NativeSectionWriter>(direct_writer).add(direct_stage);
            if (const auto* failure = std::get_if<std::string>(&accepted)) std::fprintf(stderr, "direct storage add: %s\n", failure->c_str());
            CHECK(std::holds_alternative<std::monostate>(accepted));
            const auto parsed = armigo::platform::make_source_stage(armigo::shader_input::Stage::Fragment,
                direct_program.image, bytes_of(metadata), direct.source);
            if (const auto* error = std::get_if<armigo::platform::MetalShaderPackageError>(&parsed))
                std::fprintf(stderr, "direct storage reflection: %s\n", error->detail.c_str());
            CHECK(std::holds_alternative<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(parsed));
            if (const auto* stage = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&parsed)) {
                CHECK((*stage)->storage_buffers.size() == 1);
                if (!(*stage)->storage_buffers.empty()) CHECK((*stage)->storage_buffers[0].pool_pointer);
            }
            auto invalid = metadata;
            const auto fact = invalid.find("bound_storage_descriptor");
            CHECK(fact != std::string::npos);
            if (fact != std::string::npos) invalid.replace(fact, std::string_view("bound_storage_descriptor").size(), "negate_y");
            CHECK(std::holds_alternative<armigo::platform::MetalShaderPackageError>(armigo::platform::make_source_stage(
                armigo::shader_input::Stage::Fragment, direct_program.image, bytes_of(invalid), direct.source)));
        }

        // The same mesh program, translated as the run-time path translates it
        // and stored for both depth conventions.
        const Fixture mesh_fixture = mesh_quad();
        const auto mesh_program = mesh_fixture.program();
        const auto mesh_launch = armigo::platform::metal::native_mesh_launch(mesh_program);
        CHECK(mesh_launch.has_value() && mesh_program.mesh_launch.has_value());
        if (!mesh_launch || !mesh_program.mesh_launch) return 1;
        const auto& facts = *mesh_program.mesh_launch;
        CHECK(!mesh_launch->task && mesh_launch->threads == facts.threads && mesh_launch->payload_bytes == facts.payload_bytes &&
              mesh_launch->launched_by_task == facts.task && mesh_launch->primitive_vertices == facts.primitive_vertices &&
              mesh_launch->max_vertices == facts.max_vertices && mesh_launch->max_primitives == facts.max_primitives);
        std::string mesh_reflections[2];
        std::string mesh_sources[2];
        const gm::VertexDepthConvention mesh_depths[2] = {gm::VertexDepthConvention::NegativeOneToOne,
                                                           gm::VertexDepthConvention::ZeroToOne};
        for (int index = 0; index < 2; ++index) {
            sn::FrontendOptions mesh_options;
            mesh_options.pixel_rate = sn::PixelRateState::disabled;
            mesh_options.vertex_fetch = false;
            mesh_options.vertex_depth = mesh_depths[index] == gm::VertexDepthConvention::ZeroToOne
                                            ? pl::DepthConvention::zero_to_one
                                            : pl::DepthConvention::negative_one_to_one;
            gm::EmitOptions mesh_emission;
            mesh_emission.vertex_fetch = false;
            mesh_emission.vertex_depth = mesh_depths[index];
            auto mesh_translated = sn::translate_native(mesh_program, mesh_options);
            if (const auto* refused = std::get_if<pl::Diagnostic>(&mesh_translated)) {
                std::fprintf(stderr, "mesh translation: %s\n", refused->message.c_str());
                return 1;
            }
            const auto& mesh_artifact = std::get<pl::MslArtifact>(mesh_translated);
            mesh_reflections[index] = armigo::platform::metal::native_stage_reflection(
                mesh_artifact, mesh_emission, pl::Stage::vertex, mesh_program.image, &*mesh_launch);
            mesh_sources[index] = mesh_artifact.source;
        }
        const std::string runtime_launch = launch_record(mesh_reflections[0]);
        CHECK(runtime_launch == launch_record(mesh_reflections[1]));
        CHECK(runtime_launch.find("\"threads\":" + std::to_string(mesh_launch->threads) + ",") != std::string::npos);
        CHECK(runtime_launch.find("\"payload_bytes\":" + std::to_string(mesh_launch->payload_bytes) + ",") != std::string::npos);
        CHECK(runtime_launch.find(std::string("\"task\":") + (mesh_launch->launched_by_task ? "true" : "false")) != std::string::npos);
        CHECK(runtime_launch.find("\"primitive_vertices\":" + std::to_string(mesh_launch->primitive_vertices) + ",") != std::string::npos);
        CHECK(runtime_launch.find("\"max_vertices\":" + std::to_string(mesh_launch->max_vertices) + ",") != std::string::npos);
        CHECK(runtime_launch.find("\"max_primitives\":" + std::to_string(mesh_launch->max_primitives) + "}") != std::string::npos);
        const std::vector<std::byte> mesh_minus_library = bytes_of("mesh-minus-library");
        const std::vector<std::byte> mesh_zero_library = bytes_of("mesh-zero-library");

        TemporaryDirectory directory;
        auto writer = sn::NativeSectionWriter::create(directory.path);
        CHECK(std::holds_alternative<sn::NativeSectionWriter>(writer));
        if (!std::holds_alternative<sn::NativeSectionWriter>(writer)) return 1;
        sn::NativeStageWrite stage;
        stage.stage_identity = program.stage_identity;
        stage.stage = sn::NativeStageKind::fragment;
        stage.program.assign(program.image.begin(), program.image.end());
        sn::NativeVariantWrite variant;
        variant.depth = sn::NativeDepth::stage;
        variant.reflection = bytes_of(reflection);
        variant.source = artifact.source;
        variant.library = library;
        stage.variants.push_back(std::move(variant));
        sn::NativeStageWrite mesh_stage;
        mesh_stage.stage_identity = mesh_program.stage_identity;
        mesh_stage.stage = sn::NativeStageKind::mesh;
        mesh_stage.program.assign(mesh_program.image.begin(), mesh_program.image.end());
        for (int index = 0; index < 2; ++index) {
            sn::NativeVariantWrite mesh_variant;
            mesh_variant.depth = index == 0 ? sn::NativeDepth::minus_one_to_one : sn::NativeDepth::zero_to_one;
            mesh_variant.reflection = bytes_of(mesh_reflections[index]);
            mesh_variant.source = mesh_sources[index];
            mesh_variant.library = index == 0 ? mesh_minus_library : mesh_zero_library;
            mesh_stage.variants.push_back(std::move(mesh_variant));
        }
        auto added = std::get<sn::NativeSectionWriter>(writer).add(stage);
        CHECK(std::holds_alternative<std::monostate>(added));
        auto mesh_added = std::get<sn::NativeSectionWriter>(writer).add(mesh_stage);
        CHECK(std::holds_alternative<std::monostate>(mesh_added));
        if (const auto* failure = std::get_if<std::string>(&mesh_added)) std::fprintf(stderr, "mesh add: %s\n", failure->c_str());
        sn::NativeCounts counts;
        counts.by_stage = {{"vertex", 0}, {"fragment", 1}, {"compute", 0}};
        counts.translated_by_stage = counts.by_stage;
        counts.compiled_by_stage = counts.by_stage;
        counts.unique = 1;
        counts.translated = 1;
        counts.compiled = 1;
        counts.enumerated = 1;
        auto finished = std::get<sn::NativeSectionWriter>(writer).finish(counts, {});
        CHECK(std::holds_alternative<sn::NativeSection>(finished));
        if (const auto* failure = std::get_if<std::string>(&added)) std::fprintf(stderr, "add: %s\n", failure->c_str());
        if (const auto* failure = std::get_if<std::string>(&finished)) std::fprintf(stderr, "finish: %s\n", failure->c_str());

        nvn::select_api_version({231, 40});
        nvn::register_program_entries();
        nvn::set_native_shader_package(directory.path);
        nvn::set_native_program_resolver();
        const auto initialize = entry_function<nvn::Entry::nvnProgramInitialize>();
        const auto set = entry_function<nvn::Entry::nvnProgramSetShaders>();
        auto runtime = packaged_fixture;
        put(runtime.control, 0x30, 0);
        const nvn::NVNshaderData hit_shader{reinterpret_cast<std::uint64_t>(runtime.code.data()), runtime.control.data()};
        nvn::NVNprogram guest{};
        CHECK(initialize(&guest, nullptr));
        CHECK(set(&guest, 1, &hit_shader) == 1);
        const auto& live = *reinterpret_cast<const nvn::ProgramState*>(&guest);
        auto resolved = nvn::retain_program(live);
        CHECK(resolved && resolved->stages[1]);
        if (resolved && resolved->stages[1]) {
            CHECK(resolved->stages[1]->metallib == library);
            CHECK(resolved->stages[1]->msl_source.empty());
        }

        const Fixture missed_fixture({sn::test::move(0, 0x3f000000), sn::test::move(1, 0), sn::test::move(2, 0),
                                      sn::test::move(3, 0x3f800000), sn::test::exit_instruction()});
        auto missed = missed_fixture;
        put(missed.control, 0x30, 0);
        const nvn::NVNshaderData miss_shader{reinterpret_cast<std::uint64_t>(missed.code.data()), missed.control.data()};
        CHECK(set(&guest, 1, &miss_shader) == 1);
        resolved = nvn::retain_program(live);
        CHECK(resolved && resolved->stages[1]);
        if (resolved && resolved->stages[1]) {
            CHECK(resolved->stages[1]->metallib.empty());
            CHECK(!resolved->stages[1]->msl_source.empty());
        }
        auto mesh_runtime = mesh_fixture;
        put(mesh_runtime.control, 0x30, 0);
        const nvn::NVNshaderData mesh_shader{reinterpret_cast<std::uint64_t>(mesh_runtime.code.data()), mesh_runtime.control.data()};
        CHECK(set(&guest, 1, &mesh_shader) == 1);
        resolved = nvn::retain_program(live);
        CHECK(resolved && resolved->stages[4] && resolved->failed_stages == 0);
        if (resolved && resolved->stages[4]) {
            CHECK(resolved->stages[4]->metallib == mesh_minus_library);
            CHECK(resolved->stages[4]->stage == armigo::shader_input::Stage::TessellationEvaluation);
            const auto runtime_stage = armigo::platform::make_source_stage(
                armigo::shader_input::Stage::TessellationEvaluation, mesh_program.image,
                {reinterpret_cast<const std::byte*>(mesh_reflections[0].data()), mesh_reflections[0].size()}, mesh_sources[0]);
            const auto* runtime_made = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&runtime_stage);
            CHECK(runtime_made != nullptr);
            if (runtime_made) CHECK(resolved->stages[4]->mesh == (*runtime_made)->mesh);
        }
        auto packaged = sn::NativePackage::open(directory.path);
        CHECK(std::holds_alternative<sn::NativePackage>(packaged));
        if (std::holds_alternative<sn::NativePackage>(packaged)) {
            const auto& section = std::get<sn::NativePackage>(packaged);
            for (const bool zero_to_one : {false, true}) {
                sn::NativeLookup mesh_lookup;
                mesh_lookup.stage = sn::NativeStageKind::mesh;
                mesh_lookup.vertex_depth_zero_to_one = zero_to_one;
                auto found = section.find(mesh_program.stage_identity, mesh_lookup);
                CHECK(std::holds_alternative<sn::NativeStageFiles>(found));
                if (!std::holds_alternative<sn::NativeStageFiles>(found)) continue;
                const auto& files = std::get<sn::NativeStageFiles>(found);
                const std::string text(reinterpret_cast<const char*>(files.reflection.data()), files.reflection.size());
                CHECK(launch_record(text) == runtime_launch);
            }
        }
        const auto use = nvn::native_shader_package_counts();
        CHECK(use.hits == 2);
        CHECK(use.misses == 1);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "native package lookup failed: %s\n", error.what());
        return 1;
    }
    std::printf("Native package lookup failures: %u\n", failures);
    return failures ? 1 : 0;
}
