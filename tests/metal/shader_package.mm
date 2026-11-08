// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/metal_shader_package.h>
#include "../../src/nvn/programs.h"
#include "../../src/nvn/entry_table.h"
#include <armigo/shader/program_identity.h>
#include <armigo/shader/source_version.h>
#include <armigo/shader_translation/stage_translation.h>
#include "../shader/translation/stage_fixture.h"
#include "sha256.h"
#import <Metal/Metal.h>

#import <Foundation/Foundation.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using namespace armigo;
namespace gm = shader::graphics_msl;
unsigned failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); ++failures; } } while (false)

std::string hash(std::span<const std::byte> bytes) {
    const auto digest = loader::detail::sha256(bytes);
    CHECK(digest.has_value());
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (auto byte : *digest) {
        const auto value = std::to_integer<unsigned>(byte);
        result += digits[value >> 4];
        result += digits[value & 15];
    }
    return result;
}

std::string hash(std::string_view text) {
    return hash(std::as_bytes(std::span(text.data(), text.size())));
}

void write(const fs::path& path, std::span<const std::byte> bytes) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    CHECK(bool(stream));
}

void write(const fs::path& path, std::string_view text) {
    write(path, std::as_bytes(std::span(text.data(), text.size())));
}

std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::vector<std::byte> program(shader_input::Stage stage, std::byte fill) {
    std::vector<std::byte> result(shader::program_identity::prefix_size, fill);
    const std::uint32_t word = shader_input::sph_stage_type_for(stage) <<
                               shader_input::format::sph_stage_shift;
    for (unsigned byte = 0; byte < 4; ++byte) result[byte] = std::byte(word >> (byte * 8));
    return result;
}

std::string reflection(shader_input::Stage stage, std::span<const std::byte> bytes) {
    const bool vertex = stage == shader_input::Stage::Vertex;
    return std::string("{\"schema\":\"armigo-shader-stage-v1\",\"source_version\":\"") +
        shader::source_version + "\",\"program_key\":\"" + hash(bytes) +
        "\",\"program_size\":" + std::to_string(bytes.size()) +
        ",\"program_file\":\"program.bin\",\"source_file\":\"source.metal\","
        "\"source_digest\":\"" + std::string(64, '0') + "\",\"stage\":\"" +
        (vertex ? "vertex" : "fragment") + "\",\"entry_point\":\"" +
        (vertex ? "armigo_vertex_main" : "armigo_fragment_main") +
        "\",\"vertex_depth\":\"" + (vertex ? "minus-one-to-one" : "unspecified") +
        "\",\"fragment_position\":\"" + (vertex ? "unspecified" : "reciprocal-clip-w") +
        "\",\"vertex_attributes\":" +
        (vertex ? "[{\"generic_slot\":0,\"attribute_index\":0,\"format\":\"float32x3\"}]" : "[]") +
        ",\"constant_buffers\":[],\"textures\":[],\"interface\":{"
        "\"position_input_mask\":" + std::string(vertex ? "0" : "8") +
        ",\"position_output_mask\":" + std::string(vertex ? "15" : "0") +
        ",\"color_output_mask\":" + std::string(vertex ? "0" : "15") +
        ",\"generic_inputs\":[{\"slot\":0,\"component_mask\":" +
        std::string(vertex ? "7" : "3") + ",\"interpolation\":\"" +
        (vertex ? "screen-linear" : "perspective") + "\"}],\"generic_outputs\":" +
        (vertex ? "[{\"slot\":0,\"component_mask\":3,\"interpolation\":\"screen-linear\"}]" : "[]") +
        "}}\n";
}

void replace(std::string& value, std::string_view before, std::string_view after);

// armigo-shader-stage-v2 in the admitted runtime subset: float32x3 streams at
// the reflected ordinal, one constant buffer, and dependencies the loader can
// verify against guest state. Either stage may carry bound texture/sampler
// entries sharing one handle word plus the consumed handle-buffer fact.
std::string reflection_v2(shader_input::Stage stage, std::span<const std::byte> bytes,
                          const std::string& fragment_textures = "[]",
                          const std::string& fragment_extra_dependencies = "",
                          const std::string& vertex_textures = "[]",
                          const std::string& vertex_extra_dependencies = "") {
    const bool vertex = stage == shader_input::Stage::Vertex;
    return std::string("{\"schema\":\"armigo-shader-stage-v2\",\"source_version\":\"") +
        shader::source_version + "\",\"program_key\":\"" + hash(bytes) +
        "\",\"program_size\":" + std::to_string(bytes.size()) +
        ",\"program_file\":\"program.bin\",\"source_file\":\"source.metal\","
        "\"source_digest\":\"" + std::string(64, '0') + "\",\"stage\":\"" +
        (vertex ? "vertex" : "fragment") + "\",\"entry_point\":\"" +
        (vertex ? "armigo_vertex_main" : "armigo_fragment_main") +
        "\",\"vertex_depth\":\"" + (vertex ? "minus-one-to-one" : "unspecified") +
        "\",\"fragment_position\":\"" + (vertex ? "unspecified" : "reciprocal-clip-w") +
        "\",\"c1_digest\":null,\"profile\":{\"identity\":\"armigo-macos-metal4\",\"version\":1},"
        "\"constant_buffers\":" +
        (vertex ? "[{\"guest_binding\":0,\"size_kind\":\"constant_buffer_window\","
                  "\"window_bytes\":64,\"alignment\":4,\"policy\":\"runtime_zero_words_v1\","
                  "\"metal_buffer_index\":0}]" : "[]") +
        ",\"support_buffers\":" +
        (vertex ? "[{\"schema\":\"armigo-cb-sizes\",\"version\":1,\"byte_extent\":72,"
                  "\"alignment\":4,\"metal_buffer_index\":1}]" : "[]") +
        ",\"textures\":" + (vertex ? vertex_textures : fragment_textures) + ",\"vertex_streams\":" +
        (vertex ? "[{\"slot\":0,\"attribute_index\":0,\"format\":\"f32\",\"components\":3,"
                  "\"byte_stride\":12}]" : "[]") +
        ",\"vertex_inputs\":[]" +
        std::string(",\"interface\":{\"position_input_mask\":") + (vertex ? "0" : "8") +
        ",\"position_output_mask\":" + std::string(vertex ? "15" : "0") +
        ",\"color_output_mask\":" + std::string(vertex ? "0" : "15") +
        ",\"color_outputs\":" + std::string(vertex ? "[]" :
            "[{\"location\":0,\"component_mask\":15,\"scalar\":\"f32\",\"components\":4}]") +
        ",\"writes_fragment_depth\":false,\"writes_sample_mask\":false" +
        ",\"generic_inputs\":[" + (vertex
            ? "{\"slot\":0,\"component_mask\":7,\"scalar\":\"f32\",\"components\":3,"
              "\"interpolation\":\"linear\"}"
            : "{\"slot\":0,\"component_mask\":3,\"scalar\":\"f32\",\"components\":4,"
              "\"interpolation\":\"perspective\"}") +
        "],\"generic_outputs\":" +
        (vertex ? "[{\"slot\":0,\"component_mask\":3,\"scalar\":\"f32\",\"components\":4,"
                  "\"interpolation\":\"linear\"}]" : "[]") +
        "},\"dependencies\":[" + (vertex
            ? "{\"kind\":\"depth_convention\",\"index\":0,\"byte_offset\":0,\"component\":0,"
              "\"origin\":\"explicit_option\",\"value\":\"minus-one-to-one\"},"
              "{\"kind\":\"vertex_attribute_format\",\"index\":0,\"byte_offset\":0,"
              "\"component\":0,\"origin\":\"explicit_option\","
              "\"value\":{\"format\":\"f32\",\"components\":3}}" + vertex_extra_dependencies
            : "{\"kind\":\"fragment_output_type\",\"index\":0,\"byte_offset\":0,"
              "\"component\":0,\"origin\":\"derived\","
              "\"value\":{\"scalar\":\"f32\",\"components\":4}},"
              "{\"kind\":\"input_interpolation\",\"index\":0,\"byte_offset\":0,"
              "\"component\":0,\"origin\":\"validated_header\","
              "\"value\":{\"mode\":\"perspective\",\"location\":\"center\"}}" +
              fragment_extra_dependencies) + "]}\n";
}

std::string reflection_v3(shader_input::Stage stage, std::span<const std::byte> bytes,
                          const std::string& fragment_resources,
                          const std::string& fragment_extra_dependencies,
                          const std::string& vertex_resources,
                          const std::string& vertex_extra_dependencies,
                          const std::string& fragment_argument_buffer,
                          const std::string& vertex_argument_buffer) {
    auto result = reflection_v2(stage, bytes, fragment_resources,
                                fragment_extra_dependencies, vertex_resources,
                                vertex_extra_dependencies);
    replace(result, "\"schema\":\"armigo-shader-stage-v2\"",
            "\"schema\":\"armigo-shader-stage-v3\"");
    replace(result, "\"profile\":{\"identity\":\"armigo-macos-metal4\",\"version\":1},",
            "\"profile\":{\"identity\":\"armigo-macos-metal4\",\"version\":3},"
            "\"argument_buffer\":" +
            (stage == shader_input::Stage::Vertex ? vertex_argument_buffer
                                                   : fragment_argument_buffer) + ",");
    return result;
}

std::string v3_texture_resources(unsigned count, bool argument_buffer) {
    std::string result = "[";
    for (unsigned index = 0; index < count; ++index) {
        if (index) result += ',';
        result += "{\"cbuf_slot\":2,\"handle_byte_offset\":" + std::to_string(index * 4) +
            ",\"kind\":\"texture\",\"dimension\":\"d2\",\"array\":false,"
            "\"multisample\":false,\"scalar\":\"f32\",\"component_format\":\"floating\","
            "\"normalized_coordinates\":true,\"comparison\":false,\"resource_id\":" +
            std::to_string(index) + ",\"array_length\":1,\"location\":";
        if (argument_buffer) {
            result += "{\"kind\":\"argument_buffer_member\",\"byte_offset\":" +
                std::to_string(index * 8) + ",\"count\":1,\"stride\":8}";
        } else {
            result += "{\"kind\":\"direct\",\"namespace\":\"texture\",\"first\":" +
                std::to_string(index) + ",\"count\":1}";
        }
        result += ",\"resource_kind\":\"sampled\",\"access\":\"read\"}";
    }
    return result + ']';
}

struct Fixture {
    fs::path root;
    std::string chain;
    std::vector<std::byte> vertex;
    std::vector<std::byte> fragment;
    std::string vertex_variant;
    std::string fragment_variant;
};

Fixture package(const fs::path& root, bool ported = false,
                const std::string& fragment_textures = "[]",
                const std::string& fragment_extra_dependencies = "",
                const std::string& vertex_textures = "[]",
                const std::string& vertex_extra_dependencies = "",
                bool v3 = false,
                const std::string& fragment_argument_buffer = "null",
                const std::string& vertex_argument_buffer = "null",
                std::string_view sample_hoisting = {}) {
    Fixture result{root, {}, {}, {}, {}, {}};
    result.vertex = program(shader_input::Stage::Vertex, std::byte{0x31});
    result.fragment = program(shader_input::Stage::Fragment, std::byte{0x72});
    result.vertex_variant = hash(ported ? "ported vertex variant" : "vertex variant");
    result.fragment_variant = hash(ported ? "ported fragment variant" : "fragment variant");
    const std::string profile = std::string("{\"platform\":\"macos-metal4\",\"source_version\":\"") +
        shader::source_version + "\",\"tool_version\":\"sdk/metal\"}";
    result.chain = hash("armigo-metal-chain-v1\n" + profile);
    fs::create_directories(root / "stages" / result.vertex_variant);
    fs::create_directories(root / "stages" / result.fragment_variant);
    const std::array libraries{std::string("synthetic vertex metallib"),
                               std::string("synthetic fragment metallib")};
    const std::array programs{std::span<const std::byte>(result.vertex),
                              std::span<const std::byte>(result.fragment)};
    const std::array variants{result.vertex_variant, result.fragment_variant};
    const std::array stages{std::string("vertex"), std::string("fragment")};
    std::string entries;
    for (unsigned index = 0; index < 2; ++index) {
        const auto directory = root / "stages" / variants[index];
        const auto stage_type = index == 0 ? shader_input::Stage::Vertex
                                           : shader_input::Stage::Fragment;
        const auto reflected = v3
            ? reflection_v3(stage_type, programs[index], fragment_textures,
                            fragment_extra_dependencies, vertex_textures,
                            vertex_extra_dependencies, fragment_argument_buffer,
                            vertex_argument_buffer)
            : ported ? reflection_v2(stage_type, programs[index], fragment_textures,
                                     fragment_extra_dependencies, vertex_textures,
                                     vertex_extra_dependencies)
                     : reflection(stage_type, programs[index]);
        write(directory / "program.bin", programs[index]);
        write(directory / "reflection.json", reflected);
        write(directory / "shader.metallib", libraries[index]);
        if (!entries.empty()) entries += ',';
        entries += "{\"stage\":\"" + stages[index] + "\",\"program_key\":\"" +
            hash(programs[index]) + "\",\"prefix_key\":\"" + hash(programs[index]) +
            "\",\"program_size\":" + std::to_string(programs[index].size()) +
            ",\"directory\":\"stages/" + variants[index] +
            "\",\"reflection_digest\":\"" + hash(reflected) +
            "\",\"library_digest\":\"" + hash(libraries[index]) +
            "\",\"library_size\":" + std::to_string(libraries[index].size()) +
            ",\"variant_id\":\"" + variants[index] + "\"}";
    }
    const std::string manifest = std::string("{\"schema\":\"armigo-metal-package-v1\","
        "\"source_version\":\"") + shader::source_version + "\",\"chain_id\":\"" +
        result.chain + "\",\"platform\":\"macos-metal4\",\"profile_text\":";
    NSString* profile_string = [NSString stringWithUTF8String:profile.c_str()];
    NSData* escaped_data = [NSJSONSerialization dataWithJSONObject:@[profile_string] options:0 error:nil];
    std::string escaped(static_cast<const char*>(escaped_data.bytes), escaped_data.length);
    escaped = escaped.substr(1, escaped.size() - 2);
    std::string hoisting;
    if (!sample_hoisting.empty())
        hoisting = ",\"sample_hoisting\":\"" + std::string(sample_hoisting) + "\"";
    write(root / "manifest.json", manifest + escaped +
        ",\"complete_inventory\":false,\"runnable\":false" + hoisting +
        ",\"entries\":[" + entries + "]}\n");
    return result;
}

struct Backing {
    std::vector<std::byte> bytes = std::vector<std::byte>(0x4000);
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> output) noexcept {
        auto& self = *static_cast<Backing*>(raw);
        if (address < 0x1000 || output.size() > self.bytes.size() - std::min<std::size_t>(
                self.bytes.size(), static_cast<std::size_t>(address - 0x1000))) return false;
        std::memcpy(output.data(), self.bytes.data() + address - 0x1000, output.size());
        return true;
    }
};

graphics::GpuAddressSpace memory(Backing& backing) {
    const graphics::Mapping mapping{0x1000, 0x4000, 0, 0x1000, 0x4000, 0x1000};
    return std::get<graphics::GpuAddressSpace>(graphics::make_address_space(
        {&mapping, 1}, {Backing::read, nullptr, &backing}));
}

graphics::RenderStop snapshot(const Fixture& fixture, Backing& backing) {
    std::copy(fixture.vertex.begin(), fixture.vertex.end(), backing.bytes.begin());
    std::copy(fixture.fragment.begin(), fixture.fragment.end(), backing.bytes.begin() + 0x1000);
    graphics::RenderStop stop;
    stop.kind = graphics::RenderKind::Draw;
    stop.engine = graphics::engine_class::threed;
    stop.programs[1] = {0x1000, 0, 0, true};
    stop.programs[5] = {0x2000, 0, 0, true};
    stop.registers[0x35f] = 0;
    stop.registers[0x458] = (2u << 21) | (7u << 27);
    stop.registers[0x700] = 0x1000 | 12;
    stop.registers[0x701] = 0;
    stop.registers[0x702] = 0x3000;
    stop.registers[0x7c0] = 0;
    stop.registers[0x7c1] = 0x3fff;
    return stop;
}

void replace(std::string& value, std::string_view before, std::string_view after) {
    const auto position = value.find(before);
    CHECK(position != std::string::npos);
    if (position != std::string::npos) value.replace(position, before.size(), after);
}

void mutate_reflection(const Fixture& fixture, const std::string& variant,
                       std::string_view before, std::string_view after) {
    const auto path = fixture.root / "stages" / variant / "reflection.json";
    auto text = read_text(path);
    const auto old_digest = hash(text);
    replace(text, before, after);
    write(path, text);
    auto manifest = read_text(fixture.root / "manifest.json");
    replace(manifest, old_digest, hash(text));
    write(fixture.root / "manifest.json", manifest);
}

// Adds a stage entry to a fixture's manifest: a variant directory holding
// `bytes` as its program and `reflected` as its reflection.
std::string add_entry(const Fixture& fixture, const std::string& stage, const std::string& name,
                      std::span<const std::byte> bytes, const std::string& reflected) {
    const auto variant = hash(name);
    const auto directory = fixture.root / "stages" / variant;
    fs::create_directories(directory);
    write(directory / "program.bin", bytes);
    write(directory / "reflection.json", reflected);
    const std::string library = "synthetic " + name + " metallib";
    write(directory / "shader.metallib", library);
    const auto prefix = bytes.first(shader::program_identity::prefix_size);
    const std::string entry = ",{\"stage\":\"" + stage + "\",\"program_key\":\"" + hash(bytes) +
        "\",\"prefix_key\":\"" + hash(prefix) + "\",\"program_size\":" + std::to_string(bytes.size()) +
        ",\"directory\":\"stages/" + variant + "\",\"reflection_digest\":\"" + hash(reflected) +
        "\",\"library_digest\":\"" + hash(library) + "\",\"library_size\":" +
        std::to_string(library.size()) + ",\"variant_id\":\"" + variant + "\"}";
    auto manifest = read_text(fixture.root / "manifest.json");
    const auto end = manifest.rfind("]}");
    CHECK(end != std::string::npos);
    if (end != std::string::npos) manifest.insert(end, entry);
    write(fixture.root / "manifest.json", manifest);
    return variant;
}

// A stage translated at run time becomes a compiled stage that carries its
// Metal source, with the identity a package stage of the same program has;
// the source compiles with Metal, and a stage whose source or kind differs
// from its reflection is refused.
void test_source_stage() {
    namespace tr = armigo::shader::translation;
    const auto container = armigo::test::stage_fixture::fragment_stage();
    const auto translated = tr::translate_stage(container.control, container.code, {});
    const std::span<const std::byte> reflection{
        reinterpret_cast<const std::byte*>(translated.reflection.data()), translated.reflection.size()};
    auto made = armigo::platform::make_source_stage(armigo::shader_input::Stage::Fragment, translated.program,
                                                    reflection, translated.source);
    const auto* stage = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&made);
    if (const auto* error = std::get_if<armigo::platform::MetalShaderPackageError>(&made))
        std::fprintf(stderr, "source stage: %s\n", error->detail.c_str());
    CHECK(stage && *stage);
    if (!stage || !*stage) return;
    CHECK((*stage)->metallib.empty() && (*stage)->msl_source == translated.source);
    CHECK((*stage)->stage == armigo::shader_input::Stage::Fragment && !(*stage)->entry_point.empty());
    CHECK((*stage)->identity.rfind("1.", 0) == 0 && (*stage)->identity.size() == 2 + 64 + 2);
    CHECK((*stage)->guest_code_bytes == translated.program.size());
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        auto* options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@((*stage)->msl_source.c_str())
                                                      options:options error:&error];
        CHECK(library != nil);
        CHECK([[library functionNames] containsObject:@((*stage)->entry_point.c_str())]);
    }
    // A stage whose source stays in a file reads and checks it when asked.
    const auto file = fs::temp_directory_path() /
        ("armigo-source-stage-" + std::to_string(NSProcessInfo.processInfo.processIdentifier) + ".metal");
    std::ofstream(file, std::ios::binary) << translated.source;
    auto filed = armigo::platform::make_source_stage(armigo::shader_input::Stage::Fragment, translated.program,
                                                     reflection, {}, file.string());
    const auto* on_disk = std::get_if<std::shared_ptr<const armigo::platform::CompiledGraphicsStage>>(&filed);
    CHECK(on_disk && *on_disk && (*on_disk)->msl_source.empty() && (*on_disk)->msl_source_file == file.string());
    if (on_disk && *on_disk) {
        const auto text = armigo::platform::stage_source(**on_disk);
        CHECK(std::holds_alternative<std::string>(text) && std::get<std::string>(text) == translated.source);
        std::ofstream(file, std::ios::binary | std::ios::app) << "// changed\n";
        CHECK(std::holds_alternative<armigo::platform::MetalShaderPackageError>(
            armigo::platform::stage_source(**on_disk)));
    }
    std::error_code ignored;
    fs::remove(file, ignored);
    // Neither or both of the source and its file is refused.
    CHECK(std::holds_alternative<armigo::platform::MetalShaderPackageError>(
        armigo::platform::make_source_stage(armigo::shader_input::Stage::Fragment, translated.program,
                                            reflection, {}, {})));
    auto tampered = translated.source + "\n";
    CHECK(std::holds_alternative<armigo::platform::MetalShaderPackageError>(
        armigo::platform::make_source_stage(armigo::shader_input::Stage::Fragment, translated.program,
                                            reflection, tampered)));
    CHECK(std::holds_alternative<armigo::platform::MetalShaderPackageError>(
        armigo::platform::make_source_stage(armigo::shader_input::Stage::Vertex, translated.program,
                                            reflection, translated.source)));
}

void test_nvn_program_package(const fs::path& base) {
    const auto fixture = package(base / "nvn-program");
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    auto* package_pointer = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened);
    CHECK(package_pointer);
    if (!package_pointer) return;
    nvn::set_program_package(*package_pointer);
    nvn::select_api_version({55, 15});
    nvn::register_program_entries();
    using namespace nvn::api;
    const auto initialize = reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnProgramInitialize>::type>(
        nvn::armigo_nvn_bootstrap_loader("nvnProgramInitialize"));
    const auto set = reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnProgramSetShaders>::type>(
        nvn::armigo_nvn_bootstrap_loader("nvnProgramSetShaders"));
    const auto extended = reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnProgramSetShadersExt>::type>(
        nvn::armigo_nvn_bootstrap_loader("nvnProgramSetShadersExt"));
    const auto finalize = reinterpret_cast<nvn::EntrySignature<nvn::Entry::nvnProgramFinalize>::type>(
        nvn::armigo_nvn_bootstrap_loader("nvnProgramFinalize"));
    std::array<std::byte, 0x780> control{};
    std::vector<std::byte> code(0x30 + fixture.vertex.size());
    const auto put = [](std::span<std::byte> bytes, std::size_t at, std::uint32_t value) {
        std::memcpy(bytes.data() + at, &value, sizeof(value));
    };
    put(control, 0, shader_input::format::control_magic);
    put(control, 4, 1);
    put(control, 8, 14);
    put(control, 0x20, control.size());
    put(control, 0x700, code.size());
    put(control, 0x704, code.size());
    put(control, 0x708, 0x30);
    put(control, 0x714, 0);
    put(code, 0, shader_input::format::code_wrapper_magic);
    std::copy(fixture.vertex.begin(), fixture.vertex.end(), code.begin() + 0x30);
    NVNprogram object{};
    CHECK(initialize(&object, nullptr));
    NVNshaderData descriptor{reinterpret_cast<std::uint64_t>(code.data()), control.data()};
    CHECK(set(&object, 1, &descriptor));
    auto& state = *reinterpret_cast<nvn::ProgramState*>(&object);
    auto executable = nvn::retain_program(state);
    CHECK(executable && executable->stages[0]);
    if (executable) CHECK(executable->stages[0]->stage == shader_input::Stage::Vertex);
    std::array<std::byte, 24> metadata{};
    NVNshaderDataExt ext{descriptor.data, descriptor.control, metadata.data()};
    CHECK(extended(&object, 1, &ext));
    const auto serial = state.serial;
    metadata[0] = std::byte{1};
    CHECK(!extended(&object, 1, &ext));
    CHECK(state.serial == serial);
    code.back() ^= std::byte{1};
    // The package lacks the changed stage, so it is translated at run time,
    // where it fails. Translated in the background (the default), SetShaders
    // publishes the program and the stage stays empty when the program is
    // retained; once the failure is known, the same stage is refused at once.
    // With ARMIGO_SHADER_THREADS=0 SetShaders translates and refuses it itself.
    // Either way the log names the stage the package lacks, once.
    const auto log_path = base / "nvn-program-miss.log";
    std::fflush(stderr);
    const int saved_stderr = dup(STDERR_FILENO);
    const int log_file = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log_file >= 0) dup2(log_file, STDERR_FILENO);
    const char* threads = std::getenv("ARMIGO_SHADER_THREADS");
    if (!(threads && *threads && std::strtoul(threads, nullptr, 10) == 0)) {
        CHECK(set(&object, 1, &descriptor));
        CHECK(state.serial != serial);
        const auto changed = nvn::retain_program(state);
        CHECK(changed && !changed->stages[0]);
    }
    const auto known = state.serial;
    CHECK(!set(&object, 1, &descriptor));
    CHECK(state.serial == known);
    std::fflush(stderr);
    if (saved_stderr >= 0) {
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
    }
    if (log_file >= 0) close(log_file);
    {
        std::ifstream log(log_path);
        const std::string text((std::istreambuf_iterator<char>(log)), std::istreambuf_iterator<char>());
        const std::string miss = "Armigo shader runtime: stage not in the package, prepared at run time: vertex program ";
        const auto first = text.find(miss);
        CHECK(first != std::string::npos && text.find(miss, first + 1) == std::string::npos);
    }
    CHECK(executable && !executable->stages[0]->metallib.empty());
    finalize(&object);
    CHECK(executable && nvn::retain_program(executable->state) == executable);
    nvn::set_program_package(nullptr);
}

void test_open_lookup_cache_and_failures(const fs::path& base) {
    auto fixture = package(base / "good");
    auto refused = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, false});
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(refused));
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
    auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
    Backing backing;
    auto address_space = memory(backing);
    auto stop = snapshot(fixture, backing);
    auto selected = package_ptr->lookup(stop, address_space);
    CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(selected));
    if (const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected)) {
        CHECK(pair->vertex && pair->fragment);
        CHECK(!pair->vertex->metallib.empty() && !pair->fragment->metallib.empty());
        // The v1 schema synthesizes its known target-0 float32x4 record.
        CHECK(pair->fragment->color_outputs.size() == 1);
        if (pair->fragment->color_outputs.size() == 1) {
            const auto& output = pair->fragment->color_outputs.front();
            CHECK(output.location == 0 && output.component_mask == 15 &&
                  output.scalar == platform::ColorOutputScalar::f32 && output.components == 4);
        }
        CHECK(pair->vertex->color_outputs.empty());
        CHECK(!pair->fragment->writes_fragment_depth && !pair->fragment->writes_sample_mask);
    }
    const auto stage = package_ptr->lookup_stage(shader_input::Stage::Vertex, fixture.vertex,
        shader::graphics_msl::VertexDepthConvention::NegativeOneToOne);
    CHECK(std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(stage));
    if (const auto* executable = std::get_if<std::shared_ptr<const platform::CompiledGraphicsStage>>(&stage))
        CHECK(*executable == std::get<platform::CompiledGraphicsPair>(selected).vertex);
    auto changed_program = fixture.vertex;
    changed_program.back() ^= std::byte{1};
    CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup_stage(
        shader_input::Stage::Vertex, changed_program,
        shader::graphics_msl::VertexDepthConvention::NegativeOneToOne)));
    CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup_stage(
        shader_input::Stage::Vertex, fixture.vertex,
        shader::graphics_msl::VertexDepthConvention::ZeroToOne)));
    fs::remove(fixture.root / "stages" / fixture.vertex_variant / "shader.metallib");
    fs::remove(fixture.root / "stages" / fixture.fragment_variant / "shader.metallib");
    CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(package_ptr->lookup(stop, address_space)));

    backing.bytes[0] ^= std::byte{1};
    const auto miss = package_ptr->lookup(stop, address_space);
    CHECK(std::holds_alternative<graphics::CommandResult>(miss));
    const auto* miss_error = std::get_if<graphics::CommandResult>(&miss);
    const std::string saved_detail = miss_error ? std::string(miss_error->detail) : std::string();
    const std::string_view saved_view = miss_error ? miss_error->detail : std::string_view();
    backing.bytes[0] ^= std::byte{1};
    stop.registers[0x35f] = 2;
    CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup(stop, address_space)));
    for (unsigned attempt = 0; attempt < 80; ++attempt) {
        stop.programs[0].enabled = true;
        CHECK(std::holds_alternative<graphics::CommandResult>(
            package_ptr->lookup(stop, address_space)));
        stop.programs[0].enabled = false;
    }
    CHECK(saved_view == saved_detail);
}

void test_manifest_and_artifact_rejection(const fs::path& base) {
    {
        auto fixture = package(base / "chain");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {std::string(64, '0'), true})));
    }
    {
        auto fixture = package(base / "platform");
        auto path = fixture.root / "manifest.json";
        auto text = read_text(path);
        replace(text, "\"platform\":\"macos-metal4\"", "\"platform\":\"ios-metal4\"");
        write(path, text);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "program");
        std::ofstream stream(fixture.root / "stages" / fixture.vertex_variant / "program.bin",
                             std::ios::binary | std::ios::app);
        stream.put('x'); stream.close();
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "reflection");
        mutate_reflection(fixture, fixture.fragment_variant,
                          "armigo_fragment_main", "unsupported_fragment");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "constant-buffer-metadata");
        mutate_reflection(fixture, fixture.vertex_variant, "\"constant_buffers\":[]",
            "\"constant_buffers\":[{\"guest_binding\":0,\"minimum_size\":2,\"metal_buffer_index\":18}]");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "vertex-interpolation");
        mutate_reflection(fixture, fixture.vertex_variant,
                          "\"interpolation\":\"screen-linear\"",
                          "\"interpolation\":\"perspective\"");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "reflection-digest");
        std::ofstream stream(fixture.root / "stages" / fixture.fragment_variant / "reflection.json",
                             std::ios::binary | std::ios::app);
        stream.put(' '); stream.close();
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "library");
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        std::ofstream stream(fixture.root / "stages" / fixture.vertex_variant / "shader.metallib",
                             std::ios::binary | std::ios::app);
        stream.put('x'); stream.close();
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        auto result = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened)->lookup(stop, address_space);
        CHECK(std::holds_alternative<graphics::CommandResult>(result));
    }
    {
        auto fixture = package(base / "types");
        auto path = fixture.root / "manifest.json";
        std::string text = read_text(path);
        replace(text, "\"complete_inventory\":false", "\"complete_inventory\":0");
        write(path, text);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "integer-bool");
        auto path = fixture.root / "manifest.json";
        auto text = read_text(path);
        const std::string value = "\"library_size\":" +
            std::to_string(std::string("synthetic vertex metallib").size());
        replace(text, value, "\"library_size\":true");
        write(path, text);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "path");
        auto path = fixture.root / "manifest.json";
        auto text = read_text(path);
        replace(text, "stages/" + fixture.vertex_variant, "stages/../" + fixture.vertex_variant);
        write(path, text);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "duplicate-key");
        auto path = fixture.root / "manifest.json";
        auto text = read_text(path);
        text.insert(1, "\"\\u0073chema\":\"armigo-metal-package-v1\",");
        write(path, text);
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        auto* error = std::get_if<platform::MetalShaderPackageError>(&opened);
        CHECK(error && error->detail.find("duplicate JSON object key") != std::string::npos);
    }
    {
        // The same key twice, both without escapes.
        auto fixture = package(base / "duplicate-plain-key");
        auto path = fixture.root / "manifest.json";
        auto text = read_text(path);
        text.insert(1, "\"schema\":\"armigo-metal-package-v1\",");
        write(path, text);
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        auto* error = std::get_if<platform::MetalShaderPackageError>(&opened);
        CHECK(error && error->detail.find("duplicate JSON object key") != std::string::npos);
    }
    {
        auto fixture = package(base / "declared-program-budget");
        NSData* bytes = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:(fixture.root / "manifest.json").c_str()]];
        NSMutableDictionary* manifest = [[NSJSONSerialization JSONObjectWithData:bytes
            options:NSJSONReadingMutableContainers error:nil] mutableCopy];
        NSMutableArray* entries = manifest[@"entries"];
        for (NSMutableDictionary* entry in entries) entry[@"program_size"] = @(4u << 20);
        for (unsigned index = 0; index < 127; ++index) {
            NSMutableDictionary* duplicate = [entries[0] mutableCopy];
            const auto variant = hash("program budget " + std::to_string(index));
            duplicate[@"variant_id"] = [NSString stringWithUTF8String:variant.c_str()];
            duplicate[@"directory"] = [NSString stringWithUTF8String:
                ("stages/" + variant).c_str()];
            [entries addObject:duplicate];
            fs::create_directories(fixture.root / "stages" / variant);
        }
        NSData* changed = [NSJSONSerialization dataWithJSONObject:manifest options:0 error:nil];
        write(fixture.root / "manifest.json",
              {static_cast<const std::byte*>(changed.bytes), changed.length});
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "reflection-budget");
        NSData* bytes = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:(fixture.root / "manifest.json").c_str()]];
        NSMutableDictionary* manifest = [[NSJSONSerialization JSONObjectWithData:bytes
            options:NSJSONReadingMutableContainers error:nil] mutableCopy];
        NSMutableArray* entries = manifest[@"entries"];
        const auto original = fixture.root / "stages" / fixture.vertex_variant;
        for (unsigned index = 0; index < 32; ++index) {
            NSMutableDictionary* duplicate = [entries[0] mutableCopy];
            const auto variant = hash("reflection budget " + std::to_string(index));
            duplicate[@"variant_id"] = [NSString stringWithUTF8String:variant.c_str()];
            duplicate[@"directory"] = [NSString stringWithUTF8String:
                ("stages/" + variant).c_str()];
            [entries addObject:duplicate];
            const auto directory = fixture.root / "stages" / variant;
            fs::create_directories(directory);
            fs::copy_file(original / "program.bin", directory / "program.bin");
            fs::copy_file(original / "shader.metallib", directory / "shader.metallib");
            write(directory / "reflection.json", std::string_view());
            fs::resize_file(directory / "reflection.json", 4u << 20);
        }
        NSData* changed = [NSJSONSerialization dataWithJSONObject:manifest options:0 error:nil];
        write(fixture.root / "manifest.json",
              {static_cast<const std::byte*>(changed.bytes), changed.length});
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
    }
    {
        auto fixture = package(base / "ambiguous");
        NSData* bytes = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:(fixture.root / "manifest.json").c_str()]];
        NSMutableDictionary* manifest = [[NSJSONSerialization JSONObjectWithData:bytes
            options:NSJSONReadingMutableContainers error:nil] mutableCopy];
        NSMutableArray* entries = manifest[@"entries"];
        NSMutableDictionary* duplicate = [entries[0] mutableCopy];
        const std::string variant = hash("ambiguous vertex variant");
        duplicate[@"variant_id"] = [NSString stringWithUTF8String:variant.c_str()];
        duplicate[@"directory"] = [NSString stringWithUTF8String:
            ("stages/" + variant).c_str()];
        [entries addObject:duplicate];
        const auto original = fixture.root / "stages" / fixture.vertex_variant;
        const auto copied = fixture.root / "stages" / variant;
        fs::create_directories(copied);
        for (const auto& name : {"program.bin", "reflection.json", "shader.metallib"})
            fs::copy_file(original / name, copied / name);
        NSData* changed = [NSJSONSerialization dataWithJSONObject:manifest options:0 error:nil];
        write(fixture.root / "manifest.json",
              {static_cast<const std::byte*>(changed.bytes), changed.length});
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        auto result = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened)->lookup(
            stop, address_space);
        CHECK(std::holds_alternative<graphics::CommandResult>(result));
    }
}

void test_ported_v2_reflections(const fs::path& base) {
    // A well-formed v2 pair in the admitted subset loads and selects like v1.
    auto fixture = package(base / "ported", true);
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
    auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
    Backing backing;
    auto address_space = memory(backing);
    auto stop = snapshot(fixture, backing);
    auto selected = package_ptr->lookup(stop, address_space);
    if (const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected)) {
        CHECK(pair->vertex && pair->fragment);
        CHECK(pair->vertex->vertex_depth == gm::VertexDepthConvention::NegativeOneToOne);
        CHECK(pair->vertex->vertex_attributes.size() == 1);
        CHECK(pair->vertex->constant_buffers.empty());
        CHECK(pair->vertex->constant_buffer_windows.size() == 1);
        CHECK(pair->vertex->constant_buffer_windows[0].window_bytes == 64);
        CHECK(pair->vertex->constant_buffer_windows[0].guest_binding == 0);
        CHECK(pair->vertex->constant_buffer_windows[0].metal_buffer_index == 0);
        CHECK(pair->vertex->constant_buffer_support_index == 1);
        CHECK(pair->fragment->constant_buffer_windows.empty());
        CHECK(!pair->fragment->constant_buffer_support_index);
        CHECK(pair->fragment->fragment_position ==
              gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW);
    } else {
        CHECK(false);
    }
    // Depth variants stay distinct: the zero-to-one depth dependency must not
    // match the snapshot's minus-one-to-one convention register.
    auto other = package(base / "ported-depth", true);
    mutate_reflection(other, other.vertex_variant, "minus-one-to-one\","
        "\"fragment_position", "zero-to-one\",\"fragment_position");
    mutate_reflection(other, other.vertex_variant,
        "\"value\":\"minus-one-to-one\"", "\"value\":\"zero-to-one\"");
    auto other_opened = platform::MetalShaderPackage::open(other.root, {other.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(other_opened));
    if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(other_opened)) {
        Backing other_backing;
        auto other_memory = memory(other_backing);
        auto other_stop = snapshot(other, other_backing);
        CHECK(std::holds_alternative<graphics::CommandResult>(
            std::get<std::shared_ptr<platform::MetalShaderPackage>>(other_opened)->lookup(
                other_stop, other_memory)));
    }

    // Structured rejections for anything outside the admitted runtime subset.
    const std::pair<const char*, void (*)(Fixture&)> rejections[] = {
        {"profile", [](Fixture& f) {
            mutate_reflection(f, f.vertex_variant,
                "\"identity\":\"armigo-macos-metal4", "\"identity\":\"other-metal");
        }},
        // Multisample textures remain outside the single-sample L4 profile.
        {"texture", [](Fixture& f) {
            mutate_reflection(f, f.vertex_variant, "\"textures\":[]",
                "\"textures\":[{\"cbuf_slot\":2,\"handle_byte_offset\":16,\"kind\":"
                "\"texture\",\"dimension\":\"d2\",\"array\":false,\"multisample\":true,"
                "\"scalar\":\"f32\",\"component_format\":\"floating\","
                "\"normalized_coordinates\":true,\"comparison\":false,"
                "\"texture_index\":0}]");
        }},
        {"stream", [](Fixture& f) {
            mutate_reflection(f, f.vertex_variant, "\"format\":\"f32",
                              "\"format\":\"unorm8");
        }},
        {"dependency", [](Fixture& f) {
            mutate_reflection(f, f.vertex_variant,
                "\"kind\":\"vertex_attribute_format", "\"kind\":\"topology");
        }},
        {"extent", [](Fixture& f) {
            mutate_reflection(f, f.vertex_variant, "\"window_bytes\":64",
                              "\"window_bytes\":62");
        }},
    };
    for (const auto& [label, mutate] : rejections) {
        auto broken = package(base / ("ported-" + std::string(label)), true);
        mutate(broken);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }

    // Deferred entries: a profile rejection no longer rejects the package, and
    // looking the entry up reports the recorded reason as Unsupported.
    {
        auto deferred = package(base / "ported-deferred", true);
        rejections[1].second(deferred);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(deferred.root, {deferred.chain, false, true})));
        auto opened_deferred =
            platform::MetalShaderPackage::open(deferred.root, {deferred.chain, true, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened_deferred));
        if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened_deferred)) {
            Backing deferred_backing;
            auto deferred_memory = memory(deferred_backing);
            auto deferred_stop = snapshot(deferred, deferred_backing);
            const auto result = std::get<std::shared_ptr<platform::MetalShaderPackage>>(
                opened_deferred)->lookup(deferred_stop, deferred_memory);
            const auto* stop_result = std::get_if<graphics::CommandResult>(&result);
            CHECK(stop_result && stop_result->status == graphics::CommandStatus::Unsupported &&
                  std::string(stop_result->detail).find("outside the runtime profile: Texture descriptor") !=
                      std::string::npos);
        }
        // Integrity failures still reject the whole package with deferral on.
        auto corrupt = package(base / "ported-deferred-corrupt", true);
        {
            std::ofstream program(corrupt.root / "stages" / corrupt.vertex_variant / "program.bin",
                                  std::ios::binary | std::ios::app);
            program.put('\0');
        }
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(corrupt.root, {corrupt.chain, true, true})));
    }
    // The program header's local memory size is a consumed fact the runtime
    // admits; it sizes an array inside the compiled stage.
    {
        auto local = package(base / "ported-local-memory", true);
        mutate_reflection(local, local.vertex_variant, "\"dependencies\":[",
            "\"dependencies\":[{\"kind\":\"local_memory_bytes\",\"index\":0,\"byte_offset\":0,"
            "\"component\":0,\"origin\":\"validated_header\",\"value\":16},");
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
            platform::MetalShaderPackage::open(local.root, {local.chain, true})));
    }
}

void test_v3_resource_reflections(const fs::path& base) {
    auto direct = package(base / "v3-direct", true, v3_texture_resources(17, false),
                          "", "[]", "", true);
    auto opened = platform::MetalShaderPackage::open(direct.root, {direct.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (const auto* package_ptr =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened)) {
        Backing backing;
        auto address_space = memory(backing);
        auto selected = (*package_ptr)->lookup(snapshot(direct, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && pair->fragment->resources.size() == 17);
        CHECK(pair && pair->fragment->textures.size() == 17);
        if (pair) {
            CHECK(!pair->fragment->argument_buffer);
            CHECK(!pair->fragment->texture_handle_cbuf);
            CHECK(pair->fragment->resources.back().resource_id == 16);
            CHECK(pair->fragment->resources.back().location.first == 16);
            CHECK(pair->fragment->resources.back().location.count == 1);
        }
    }

    const std::string layout =
        "{\"abi_version\":1,\"buffer_index\":2,\"byte_extent\":16,\"alignment\":8}";
    auto argument = package(base / "v3-argument", true, v3_texture_resources(2, true),
                            "", "[]", "", true, layout);
    auto argument_opened =
        platform::MetalShaderPackage::open(argument.root, {argument.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(argument_opened));
    if (const auto* package_ptr =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&argument_opened)) {
        Backing backing;
        auto address_space = memory(backing);
        const auto selected = (*package_ptr)->lookup(snapshot(argument, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && pair->fragment->argument_buffer.has_value());
        CHECK(pair && !pair->fragment->legacy_texture_bindings_compatible);
    }

    const std::tuple<const char*, const char*, const char*> malformed[] = {
        {"misaligned", "\"byte_offset\":8", "\"byte_offset\":9"},
        {"overlap", "\"byte_offset\":8", "\"byte_offset\":0"},
        {"overflow", "\"byte_offset\":8", "\"byte_offset\":4294967288"},
        {"extent", "\"byte_extent\":16", "\"byte_extent\":24"},
        {"boolean-abi", "\"abi_version\":1", "\"abi_version\":true"},
        {"float-alignment", "\"alignment\":8", "\"alignment\":8.0"},
        {"sampled-write", "\"access\":\"read\"", "\"access\":\"write\""},
        {"provenance", "\"handle_byte_offset\":4", "\"handle_byte_offset\":2"},
        {"count", "\"byte_offset\":8,\"count\":1",
                  "\"byte_offset\":8,\"count\":2"},
    };
    for (const auto& [name, before, after] : malformed) {
        auto broken = package(base / ("v3-argument-" + std::string(name)), true,
                              v3_texture_resources(2, true), "", "[]", "", true, layout);
        mutate_reflection(broken, broken.fragment_variant, before, after);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }

    const std::string independent =
        "[{\"cbuf_slot\":2,\"handle_byte_offset\":0,\"kind\":\"texture\","
        "\"dimension\":\"d2\",\"array\":false,\"multisample\":false,"
        "\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":true,\"comparison\":false,\"resource_id\":0,"
        "\"array_length\":1,\"location\":{\"kind\":\"direct\","
        "\"namespace\":\"texture\",\"first\":0,\"count\":1},"
        "\"resource_kind\":\"sampled\",\"access\":\"read\"},"
        "{\"cbuf_slot\":2,\"handle_byte_offset\":4,\"kind\":\"sampler\","
        "\"comparison\":false,\"normalized_coordinates\":true,\"resource_id\":1,"
        "\"array_length\":1,\"location\":{\"kind\":\"direct\","
        "\"namespace\":\"sampler\",\"first\":0,\"count\":1}}]";
    const std::string sampler_fact =
        ",{\"kind\":\"texture_descriptor\",\"index\":2,\"byte_offset\":4,"
        "\"component\":0,\"origin\":\"derived\",\"value\":{"
        "\"dimension\":\"d2\",\"array\":false,\"multisample\":false,"
        "\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":true,\"comparison\":false}}";
    {
        auto resources = independent;
        replace(resources, "\"handle_byte_offset\":4", "\"handle_byte_offset\":0");
        auto border = package(base / "v3-custom-border", true, resources,
                              "", "[]", "", true);
        mutate_reflection(border, border.fragment_variant, "\"argument_buffer\":null,",
            "\"argument_buffer\":null,\"custom_border\":{\"abi\":\"coverage-v1\","
            "\"buffer_index\":2,\"pairs\":[{\"texture\":0,\"sampler\":1}]},");
        auto border_opened = platform::MetalShaderPackage::open(border.root, {border.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(border_opened));
        if (const auto* loaded = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&border_opened)) {
            Backing backing;
            auto address_space = memory(backing);
            const auto selected = (*loaded)->lookup(snapshot(border, backing), address_space);
            const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
            CHECK(pair && pair->fragment->custom_border);
            if (pair && pair->fragment->custom_border) {
                CHECK(pair->fragment->custom_border->buffer_index == 2);
                CHECK(pair->fragment->custom_border->pairs.size() == 1);
                CHECK(pair->fragment->custom_border->pairs[0].sampler_resource == 1);
            }
        }
        mutate_reflection(border, border.fragment_variant, "\"sampler\":1", "\"sampler\":9");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(border.root, {border.chain, true})));
        mutate_reflection(border, border.fragment_variant, "\"sampler\":9", "\"sampler\":1");
        mutate_reflection(border, border.fragment_variant, "\"buffer_index\":2", "\"buffer_index\":30");
        mutate_reflection(border, border.fragment_variant, "\"custom_border\":",
                          "\"depth_bias_mode\":\"shader-v1\",\"custom_border\":");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(border.root, {border.chain, true})));
    }
    {
        // The same coverage-v1 capability admits a vertex stage; the support
        // index must stay clear of the CB, support and vertex-stream indices.
        auto resources = independent;
        replace(resources, "\"handle_byte_offset\":4", "\"handle_byte_offset\":0");
        auto border = package(base / "v3-custom-border-vertex", true, "[]", "",
                              resources, "", true);
        mutate_reflection(border, border.vertex_variant, "\"argument_buffer\":null,",
            "\"argument_buffer\":null,\"custom_border\":{\"abi\":\"coverage-v1\","
            "\"buffer_index\":2,\"pairs\":[{\"texture\":0,\"sampler\":1}]},");
        auto border_opened = platform::MetalShaderPackage::open(border.root, {border.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(border_opened));
        if (const auto* loaded = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&border_opened)) {
            Backing backing;
            auto address_space = memory(backing);
            const auto selected = (*loaded)->lookup(snapshot(border, backing), address_space);
            const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
            CHECK(pair && pair->vertex->custom_border);
            if (pair && pair->vertex->custom_border) {
                CHECK(pair->vertex->custom_border->buffer_index == 2);
                CHECK(pair->vertex->custom_border->pairs.size() == 1);
                CHECK(pair->vertex->custom_border->pairs[0].sampler_resource == 1);
            }
        }
        mutate_reflection(border, border.vertex_variant, "\"buffer_index\":2", "\"buffer_index\":18");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(border.root, {border.chain, true})));
    }
    auto separate = package(base / "v3-independent", true, independent,
                            sampler_fact, "[]", "", true);
    auto separate_opened =
        platform::MetalShaderPackage::open(separate.root, {separate.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(separate_opened));
    if (const auto* package_ptr =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&separate_opened)) {
        Backing backing;
        auto address_space = memory(backing);
        const auto selected = (*package_ptr)->lookup(snapshot(separate, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && !pair->fragment->resources.empty());
        CHECK(pair && !pair->fragment->legacy_texture_bindings_compatible);
    }

    auto array_resources = independent;
    replace(array_resources, "\"array_length\":1,", "\"array_length\":17,");
    replace(array_resources, "\"array_length\":1,", "\"array_length\":17,");
    replace(array_resources, "{\"kind\":\"direct\",\"namespace\":\"texture\",\"first\":0,\"count\":1}",
            "{\"kind\":\"argument_buffer_member\",\"byte_offset\":0,\"count\":17,\"stride\":8}");
    replace(array_resources, "{\"kind\":\"direct\",\"namespace\":\"sampler\",\"first\":0,\"count\":1}",
            "{\"kind\":\"argument_buffer_member\",\"byte_offset\":136,\"count\":17,\"stride\":8}");
    const std::string array_facts = sampler_fact +
        ",{\"kind\":\"texture_array_length\",\"index\":2,\"byte_offset\":0,"
        "\"component\":0,\"origin\":\"derived\",\"value\":17},"
        "{\"kind\":\"texture_array_length\",\"index\":2,\"byte_offset\":4,"
        "\"component\":0,\"origin\":\"derived\",\"value\":17}";
    auto arrays = package(base / "v3-independent-arrays", true, array_resources,
                          array_facts, "[]", "", true,
                          "{\"abi_version\":1,\"buffer_index\":2,\"byte_extent\":272,\"alignment\":8}");
    auto arrays_opened = platform::MetalShaderPackage::open(arrays.root, {arrays.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(arrays_opened));
    if (const auto* package_ptr =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&arrays_opened)) {
        Backing backing;
        auto address_space = memory(backing);
        const auto selected = (*package_ptr)->lookup(snapshot(arrays, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && pair->fragment->argument_buffer.has_value());
        CHECK(pair && !pair->fragment->legacy_texture_bindings_compatible);
    }
    mutate_reflection(arrays, arrays.fragment_variant, "\"value\":17", "\"value\":16");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(arrays.root, {arrays.chain, true})));

    auto fp16 = package(base / "v3-fp16", true, v3_texture_resources(1, false),
                        "", "[]", "", true);
    mutate_reflection(fp16, fp16.fragment_variant, "\"scalar\":\"f32\"",
                      "\"scalar\":\"f16\"");
    auto fp16_opened = platform::MetalShaderPackage::open(fp16.root, {fp16.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(fp16_opened));
    if (const auto* package_ptr =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&fp16_opened)) {
        Backing backing;
        auto address_space = memory(backing);
        const auto selected = (*package_ptr)->lookup(snapshot(fp16, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && !pair->fragment->resources.empty());
        CHECK(pair && !pair->fragment->legacy_texture_bindings_compatible);
    }
}

void test_v3_draw_support_reflections(const fs::path& base) {
    const std::string support =
        "{\"schema\":\"armigo-draw-parameters\",\"version\":1,"
        "\"byte_extent\":4,\"alignment\":4,\"metal_buffer_index\":2}";
    const auto add_support = [&](Fixture& fixture, const std::string& value, bool vertex = true) {
        mutate_reflection(fixture, vertex ? fixture.vertex_variant : fixture.fragment_variant,
            "\"argument_buffer\":null,", "\"argument_buffer\":null,\"draw_support\":" + value + ",");
    };
    auto valid = package(base / "v3-draw-support", true, "[]", "", "[]", "", true);
    add_support(valid, support);
    auto opened = platform::MetalShaderPackage::open(valid.root, {valid.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (const auto* package_ptr =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened)) {
        Backing backing;
        auto address_space = memory(backing);
        const auto selected = (*package_ptr)->lookup(snapshot(valid, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && pair->vertex->draw_support_buffer_index == 2);
        CHECK(pair && !pair->fragment->draw_support_buffer_index);
    }
    const std::tuple<const char*, const char*, const char*> malformed[] = {
        {"schema", "armigo-draw-parameters", "future-draw-parameters"},
        {"version", "\"version\":1", "\"version\":2"},
        {"extent", "\"byte_extent\":4", "\"byte_extent\":8"},
        {"alignment", "\"alignment\":4", "\"alignment\":8"},
        {"metal-limit", "\"metal_buffer_index\":2", "\"metal_buffer_index\":31"},
        {"constant-collision", "\"metal_buffer_index\":2", "\"metal_buffer_index\":0"},
        {"support-collision", "\"metal_buffer_index\":2", "\"metal_buffer_index\":1"},
        {"stream-collision", "\"metal_buffer_index\":2", "\"metal_buffer_index\":18"},
    };
    for (const auto& [name, before, after] : malformed) {
        auto broken = package(base / ("v3-draw-support-" + std::string(name)),
                              true, "[]", "", "[]", "", true);
        auto invalid = support;
        replace(invalid, before, after);
        add_support(broken, invalid);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }
    auto fragment = package(base / "v3-draw-support-fragment", true, "[]", "", "[]", "", true);
    add_support(fragment, support, false);
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(fragment.root, {fragment.chain, true})));
    // Version 2 adds the render-target height and serves fragment stages only.
    const std::string height =
        "{\"schema\":\"armigo-draw-parameters\",\"version\":2,"
        "\"byte_extent\":8,\"alignment\":4,\"metal_buffer_index\":2}";
    auto fragment_height = package(base / "v3-draw-support-fragment-height", true, "[]", "", "[]", "", true);
    add_support(fragment_height, height, false);
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
        platform::MetalShaderPackage::open(fragment_height.root, {fragment_height.chain, true})));
    auto vertex_height = package(base / "v3-draw-support-vertex-height", true, "[]", "", "[]", "", true);
    add_support(vertex_height, height);
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(vertex_height.root, {vertex_height.chain, true})));
}

void test_v3_storage_buffer_reflections(const fs::path& base) {
    const std::string binding =
        "{\"cbuf_slot\":0,\"descriptor_byte_offset\":8,\"metal_buffer_index\":2,"
        "\"access\":\"read_write\",\"alignment\":4}";
    const auto add_storage = [&](Fixture& fixture, std::string_view value = {}) {
        const auto storage = value.empty() ? "[" + binding + "]" : std::string(value);
        mutate_reflection(fixture, fixture.vertex_variant, "\"argument_buffer\":null,",
            "\"argument_buffer\":null,\"storage_buffers\":" +
            storage + ",");
    };

    auto valid = package(base / "v3-storage", true, "[]", "", "[]", "", true);
    add_storage(valid);
    auto opened = platform::MetalShaderPackage::open(valid.root, {valid.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (const auto* package_ptr =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened)) {
        Backing backing;
        auto address_space = memory(backing);
        const auto selected = (*package_ptr)->lookup(snapshot(valid, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && pair->vertex->storage_buffers.size() == 1);
        if (pair && pair->vertex->storage_buffers.size() == 1) {
            const auto& storage = pair->vertex->storage_buffers.front();
            CHECK(storage.cbuf_slot == 0 && storage.descriptor_byte_offset == 8);
            CHECK(storage.metal_buffer_index == 2 && storage.alignment == 4);
            CHECK(storage.access == platform::CompiledShaderResourceAccess::ReadWrite);
        }
        CHECK(pair && pair->fragment->storage_buffers.empty());
    }

    auto explicit_empty = package(base / "v3-storage-empty", true, "[]", "", "[]", "", true);
    add_storage(explicit_empty, "[]");
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
        platform::MetalShaderPackage::open(
            explicit_empty.root, {explicit_empty.chain, true})));

    const std::tuple<const char*, const char*, const char*> malformed[] = {
        {"slot", "\"cbuf_slot\":0", "\"cbuf_slot\":18"},
        {"missing-window", "\"cbuf_slot\":0", "\"cbuf_slot\":1"},
        {"offset-alignment", "\"descriptor_byte_offset\":8",
                             "\"descriptor_byte_offset\":4"},
        {"window-overrun", "\"descriptor_byte_offset\":8",
                           "\"descriptor_byte_offset\":56"},
        {"offset-limit", "\"descriptor_byte_offset\":8",
                         "\"descriptor_byte_offset\":65528"},
        {"metal-limit", "\"metal_buffer_index\":2", "\"metal_buffer_index\":31"},
        {"constant-collision", "\"metal_buffer_index\":2",
                               "\"metal_buffer_index\":0"},
        {"support-collision", "\"metal_buffer_index\":2",
                              "\"metal_buffer_index\":1"},
        {"stream-collision", "\"metal_buffer_index\":2",
                             "\"metal_buffer_index\":18"},
        {"access", "\"access\":\"read_write\"", "\"access\":\"atomic\""},
        {"alignment", "\"alignment\":4", "\"alignment\":8"},
    };
    for (const auto& [name, before, after] : malformed) {
        auto broken = package(base / ("v3-storage-" + std::string(name)),
                              true, "[]", "", "[]", "", true);
        add_storage(broken);
        mutate_reflection(broken, broken.vertex_variant, before, after);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }

    const std::string layout =
        "{\"abi_version\":1,\"buffer_index\":2,\"byte_extent\":8,\"alignment\":8}";
    auto argument_collision = package(
        base / "v3-storage-argument-collision", true, "[]", "",
        v3_texture_resources(1, true), "", true, "null", layout);
    mutate_reflection(argument_collision, argument_collision.vertex_variant,
        "\"argument_buffer\":" + layout + ",",
        "\"argument_buffer\":" + layout + ",\"storage_buffers\":[" + binding + "],");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(
            argument_collision.root, {argument_collision.chain, true})));

    auto depth_collision = package(
        base / "v3-storage-depth-collision", true, "[]", "", "[]", "", true);
    mutate_reflection(depth_collision, depth_collision.fragment_variant,
        "\"constant_buffers\":[]",
        "\"constant_buffers\":[{\"guest_binding\":0,"
        "\"size_kind\":\"constant_buffer_window\",\"window_bytes\":64,"
        "\"alignment\":4,\"policy\":\"runtime_zero_words_v1\","
        "\"metal_buffer_index\":0}]");
    mutate_reflection(depth_collision, depth_collision.fragment_variant,
        "\"support_buffers\":[]",
        "\"support_buffers\":[{\"schema\":\"armigo-cb-sizes\",\"version\":1,"
        "\"byte_extent\":72,\"alignment\":4,\"metal_buffer_index\":1}]");
    auto depth_binding = binding;
    replace(depth_binding, "\"metal_buffer_index\":2", "\"metal_buffer_index\":30");
    mutate_reflection(depth_collision, depth_collision.fragment_variant,
        "\"argument_buffer\":null,",
        "\"argument_buffer\":null,\"storage_buffers\":[" + depth_binding + "],");
    mutate_reflection(depth_collision, depth_collision.fragment_variant,
        "\"c1_digest\":null", "\"c1_digest\":null,\"depth_bias_mode\":\"shader-v1\"");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(
            depth_collision.root, {depth_collision.chain, true})));

    auto v2 = package(base / "v2-storage-field", true);
    mutate_reflection(v2, v2.vertex_variant, "\"profile\":",
                      "\"storage_buffers\":[],\"profile\":");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(v2.root, {v2.chain, true})));
}

// Class-typed vertex inputs replace the stream entries: the guest register
// supplies format and stream at draw time, and the variant only matches when
// the decoded format's class agrees with the translated input.
void test_ported_v2_class_inputs(const fs::path& base) {
    const auto to_class = [](Fixture& fixture) {
        mutate_reflection(fixture, fixture.vertex_variant,
            "\"vertex_streams\":[{\"slot\":0,\"attribute_index\":0,\"format\":\"f32\","
            "\"components\":3,\"byte_stride\":12}]",
            "\"vertex_streams\":[]");
        mutate_reflection(fixture, fixture.vertex_variant, "\"vertex_inputs\":[]",
            "\"vertex_inputs\":[{\"location\":0,\"class\":\"uint\"}]");
        mutate_reflection(fixture, fixture.vertex_variant,
            "{\"slot\":0,\"component_mask\":7,\"scalar\":\"f32\",\"components\":3,"
            "\"interpolation\":\"linear\"}",
            "{\"slot\":0,\"component_mask\":15,\"scalar\":\"u32\",\"components\":4,"
            "\"interpolation\":\"linear\"}");
        // The stream-format fact has no declared streams anymore; the derived
        // interpolation fact keeps the dependency list honest.
        mutate_reflection(fixture, fixture.vertex_variant,
            "{\"kind\":\"vertex_attribute_format\",\"index\":0,\"byte_offset\":0,"
            "\"component\":0,\"origin\":\"explicit_option\","
            "\"value\":{\"format\":\"f32\",\"components\":3}}",
            "{\"kind\":\"input_interpolation\",\"index\":0,\"byte_offset\":0,"
            "\"component\":0,\"origin\":\"derived\","
            "\"value\":{\"mode\":\"linear\",\"location\":\"center\"}}");
    };
    auto fixture = package(base / "ported-class", true);
    to_class(fixture);
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
    auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
    Backing backing;
    auto address_space = memory(backing);
    auto stop = snapshot(fixture, backing);
    // The snapshot's stream zero fetches float32x3; an unsigned byte format on
    // the same stream matches the uint input class.
    stop.registers[0x458] = (0x0au << 21) | (4u << 27);
    auto selected = package_ptr->lookup(stop, address_space);
    CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(selected));
    if (const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected)) {
        CHECK(pair->vertex && pair->fragment);
        CHECK(pair->vertex->vertex_attributes.empty());
        CHECK(pair->vertex->vertex_class_inputs.size() == 1);
        CHECK(pair->vertex->vertex_class_inputs[0].location == 0);
        CHECK(pair->vertex->vertex_class_inputs[0].input_class ==
              graphics::VertexFormatClass::uint_class);
    }
    // A float32x3 guest format feeding a uint input refetches the same words
    // as uint32x3; a normalized byte format cannot and is a stop.
    Backing refetch_backing;
    auto refetch_memory = memory(refetch_backing);
    auto refetch_stop = snapshot(fixture, refetch_backing);
    CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(package_ptr->lookup(
        refetch_stop, refetch_memory)));
    Backing mismatch_backing;
    auto mismatch_memory = memory(mismatch_backing);
    auto mismatch_stop = snapshot(fixture, mismatch_backing);
    mismatch_stop.registers[0x458] = (0x0au << 21) | (2u << 27);
    CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup(
        mismatch_stop, mismatch_memory)));
    // Scaled formats have no Metal fetch and never match a uint input.
    Backing scaled_backing;
    auto scaled_memory = memory(scaled_backing);
    auto scaled_stop = snapshot(fixture, scaled_backing);
    scaled_stop.registers[0x458] = (0x0au << 21) | (5u << 27);
    CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup(
        scaled_stop, scaled_memory)));
    // Declaring class inputs while stream entries remain present is rejected.
    auto mixed = package(base / "ported-class-mixed", true);
    mutate_reflection(mixed, mixed.vertex_variant, "\"vertex_inputs\":[]",
        "\"vertex_inputs\":[{\"location\":0,\"class\":\"uint\"}]");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(mixed.root, {mixed.chain, true})));
    // A fragment stage must not declare class inputs.
    auto fragment_class = package(base / "ported-class-fragment", true);
    mutate_reflection(fragment_class, fragment_class.fragment_variant,
        "\"vertex_inputs\":[]",
        "\"vertex_inputs\":[{\"location\":0,\"class\":\"float\"}]");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(fragment_class.root, {fragment_class.chain, true})));
}

void test_ported_v2_constant_buffers(const fs::path& base) {
    const std::string window =
        "{\"guest_binding\":0,\"size_kind\":\"constant_buffer_window\",\"window_bytes\":64,"
        "\"alignment\":4,\"policy\":\"runtime_zero_words_v1\",\"metal_buffer_index\":0}";
    const std::string support =
        "{\"schema\":\"armigo-cb-sizes\",\"version\":1,\"byte_extent\":72,"
        "\"alignment\":4,\"metal_buffer_index\":1}";
    const std::pair<std::string, std::string> malformed[] = {
        {"\"guest_binding\":0", "\"guest_binding\":18"},
        {"\"guest_binding\":0", "\"guest_binding\":true"},
        {"\"window_bytes\":64", "\"window_bytes\":0"},
        {"\"window_bytes\":64", "\"window_bytes\":3"},
        {"\"window_bytes\":64", "\"window_bytes\":6"},
        {"\"window_bytes\":64", "\"window_bytes\":65540"},
        {"\"window_bytes\":64", "\"window_bytes\":true"},
        {"\"window_bytes\":64", "\"window_bytes\":4.5"},
        {"\"window_bytes\":64", "\"byte_extent\":64"},
        {"\"size_kind\":\"constant_buffer_window\"", "\"size_kind\":\"accessed_extent\""},
        {"\"policy\":\"runtime_zero_words_v1\"", "\"policy\":\"none\""},
        {"\"alignment\":4", "\"alignment\":1"},
        {"\"metal_buffer_index\":0", "\"metal_buffer_index\":31"},
        {"\"metal_buffer_index\":0", "\"metal_buffer_index\":18"},
        {"\"metal_buffer_index\":0", "\"metal_buffer_index\":true"},
        {"\"constant_buffers\":[" + window + "]", "\"constant_buffers\":[" + window + "," + window + "]"},
        {"\"constant_buffers\":[" + window + "]", "\"constant_buffers\":[]"},
        {"\"support_buffers\":[" + support + "]", "\"support_buffers\":[]"},
        {"\"support_buffers\":[" + support + "]", "\"support_buffers\":[" + support + "," + support + "]"},
        {"\"support_buffers\":[" + support + "],", ""},
        {"\"schema\":\"armigo-cb-sizes\"", "\"schema\":\"unknown\""},
        {"\"version\":1,\"byte_extent\":72", "\"version\":0,\"byte_extent\":72"},
        {"\"version\":1,\"byte_extent\":72", "\"version\":2,\"byte_extent\":72"},
        {"\"byte_extent\":72", "\"byte_extent\":68"},
        {"\"byte_extent\":72", "\"byte_extent\":true"},
        {"\"byte_extent\":72,\"alignment\":4", "\"byte_extent\":72,\"alignment\":8"},
        {"\"metal_buffer_index\":1", "\"metal_buffer_index\":0"},
        {"\"metal_buffer_index\":1", "\"metal_buffer_index\":18"},
        {"\"metal_buffer_index\":1", "\"metal_buffer_index\":31"},
        {"\"metal_buffer_index\":1", "\"metal_buffer_index\":true"},
        {"\"metal_buffer_index\":1", "\"metal_buffer_index\":1,\"extra\":0"},
    };
    unsigned ordinal = 0;
    for (const auto& [before, after] : malformed) {
        auto broken = package(base / ("cb-invalid-" + std::to_string(ordinal++)), true);
        mutate_reflection(broken, broken.vertex_variant, before, after);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }
    // The entire Metal namespace is available when actual stream indices do
    // not collide. Window size is independent of the live binding's size.
    for (const unsigned extent : {4u, 65536u}) {
        auto valid = package(base / ("cb-boundary-" + std::to_string(extent)), true);
        mutate_reflection(valid, valid.vertex_variant, "\"window_bytes\":64",
                          "\"window_bytes\":" + std::to_string(extent));
        mutate_reflection(valid, valid.vertex_variant, "\"metal_buffer_index\":0",
                          "\"metal_buffer_index\":30");
        mutate_reflection(valid, valid.vertex_variant, "\"metal_buffer_index\":1",
                          "\"metal_buffer_index\":29");
        auto opened = platform::MetalShaderPackage::open(valid.root, {valid.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (const auto* ptr = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened)) {
            Backing backing;
            auto address_space = memory(backing);
            auto stop = snapshot(valid, backing);
            auto selected = (*ptr)->lookup(stop, address_space);
            const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
            CHECK(pair);
            if (pair) {
                CHECK(pair->vertex->constant_buffer_windows[0].window_bytes == extent);
                CHECK(pair->vertex->constant_buffer_windows[0].metal_buffer_index == 30);
                CHECK(pair->vertex->constant_buffer_support_index == 29);
            }
        }
    }
}

// Bound textures and samplers in the v2 admitted subset: one handle word
// binds both halves, proven from a single constant-buffer slot that the
// runtime re-checks against guest state at draw time.
void test_ported_v2_textures(const fs::path& base) {
    const auto texture_entry = [](unsigned slot, unsigned offset, unsigned index,
                                  const char* dimension = "d2") {
        return "{\"cbuf_slot\":" + std::to_string(slot) + ",\"handle_byte_offset\":" +
            std::to_string(offset) + ",\"kind\":\"texture\",\"dimension\":\"" + dimension +
            "\",\"array\":false,\"multisample\":false,\"scalar\":\"f32\","
            "\"component_format\":\"unorm\",\"normalized_coordinates\":true,"
            "\"comparison\":false,\"texture_index\":" + std::to_string(index) + "}";
    };
    const auto sampler_entry = [](unsigned slot, unsigned offset, unsigned index,
                                  bool comparison = false) {
        return "{\"cbuf_slot\":" + std::to_string(slot) + ",\"handle_byte_offset\":" +
            std::to_string(offset) + ",\"kind\":\"sampler\",\"comparison\":" +
            (comparison ? "true" : "false") + ",\"normalized_coordinates\":true,"
            "\"sampler_index\":" + std::to_string(index) + "}";
    };
    const auto pair_at = [&](unsigned slot, unsigned offset) {
        return "[" + texture_entry(slot, offset, 3) + "," +
               sampler_entry(slot, offset, 1) + "]";
    };
    const auto handle_fact = [](unsigned slot) {
        return ",{\"kind\":\"texture_handle_cbuf\",\"index\":0,\"byte_offset\":0,"
               "\"component\":0,\"origin\":\"explicit_option\",\"value\":" +
               std::to_string(slot) + "}";
    };
    const std::string pair = pair_at(2, 32);
    auto fixture = package(base / "ported-textured", true, pair, handle_fact(2));
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
    auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
    Backing backing;
    auto address_space = memory(backing);
    auto stop = snapshot(fixture, backing);
    auto selected = package_ptr->lookup(stop, address_space);
    if (const auto* pair_result = std::get_if<platform::CompiledGraphicsPair>(&selected)) {
        CHECK(pair_result->vertex && pair_result->fragment);
        CHECK(pair_result->vertex->textures.empty());
        CHECK(pair_result->fragment->texture_handle_cbuf == std::optional<std::uint8_t>(2));
        CHECK(pair_result->fragment->textures.size() == 1);
        CHECK(pair_result->fragment->texture_facts.size() == 1);
        if (pair_result->fragment->textures.size() == 1) {
            const auto& binding = pair_result->fragment->textures.front();
            CHECK(binding.handle_byte_offset == 32);
            CHECK(binding.texture_index == 3);
            CHECK(binding.sampler_index == 1);
        }
        if (pair_result->fragment->texture_facts.size() == 1) {
            const auto& fact = pair_result->fragment->texture_facts.front();
            CHECK(fact.cbuf_slot == 2 && fact.handle_byte_offset == 32);
            CHECK(fact.dimension == platform::TextureFactDimension::d2 && !fact.array &&
                  !fact.multisample);
            CHECK(fact.scalar == platform::TextureSampledScalar::f32);
            CHECK(fact.component_format == platform::TextureComponentFormat::unorm);
            CHECK(fact.normalized_coordinates && !fact.comparison);
        }
    } else {
        CHECK(false);
    }

    // Structured rejections for texture entries outside the admitted subset.
    const std::pair<const char*, std::pair<std::string, std::string>> rejections[] = {
        {"unpaired-sampler", {"[" + sampler_entry(2, 32, 1) + "]", handle_fact(2)}},
        {"duplicate-texture-index", {pair + "," + texture_entry(2, 36, 3) + "," +
                                     sampler_entry(2, 36, 2), handle_fact(2)}},
        {"duplicate-sampler-index", {pair + "," + texture_entry(2, 36, 4) + "," +
                                     sampler_entry(2, 36, 1), handle_fact(2)}},
        {"unaligned-handle", {"[" + texture_entry(2, 34, 3) + "," +
                              sampler_entry(2, 34, 1) + "]", handle_fact(2)}},
        {"unsupported-dimension", {"[" + texture_entry(2, 32, 3, "buffer") + "," +
                                   sampler_entry(2, 32, 1) + "]", handle_fact(2)}},
        {"comparison-sampler", {"[" + texture_entry(2, 32, 3) + "," +
                                sampler_entry(2, 32, 1, true) + "]", handle_fact(2)}},
        {"provenance-split", {"[" + texture_entry(2, 32, 3) + "," +
                              sampler_entry(3, 32, 1) + "]", handle_fact(2)}},
        {"fact-slot-mismatch", {pair, handle_fact(4)}},
        {"fact-out-of-range", {pair, handle_fact(18)}},
    };
    for (const auto& [label, parts] : rejections) {
        auto broken = package(base / ("ported-texture-" + std::string(label)), true,
                              parts.first, parts.second);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }
    // Fetch/query-only MSL has a texture argument but no sampler argument.
    const std::string texture_only = "[" + texture_entry(2, 32, 3) + "]";
    auto samplerless = package(base / "ported-texture-samplerless", true,
                               texture_only, handle_fact(2));
    auto samplerless_opened = platform::MetalShaderPackage::open(
        samplerless.root, {samplerless.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
        samplerless_opened));
    if (const auto* admitted =
            std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&samplerless_opened)) {
        Backing samplerless_backing;
        auto samplerless_memory = memory(samplerless_backing);
        auto samplerless_stop = snapshot(samplerless, samplerless_backing);
        const auto samplerless_selected = (*admitted)->lookup(samplerless_stop,
                                                              samplerless_memory);
        const auto* samplerless_pair =
            std::get_if<platform::CompiledGraphicsPair>(&samplerless_selected);
        CHECK(samplerless_pair && samplerless_pair->fragment->textures.size() == 1);
        CHECK(samplerless_pair && samplerless_pair->fragment->texture_facts.size() == 1);
        if (samplerless_pair && samplerless_pair->fragment->textures.size() == 1)
            CHECK(samplerless_pair->fragment->textures.front().sampler_index ==
                  shader::graphics_msl::TextureBinding::no_sampler);
    }
    // Entries with an explicit per-instruction slot and no consumed fact are
    // still admitted: their shared slot becomes the draw-time contract.
    auto explicit_fixture = package(base / "ported-texture-explicit", true, pair, "");
    auto explicit_opened = platform::MetalShaderPackage::open(explicit_fixture.root,
                                                              {explicit_fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(explicit_opened));
    if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(explicit_opened)) {
        Backing explicit_backing;
        auto explicit_memory = memory(explicit_backing);
        auto explicit_stop = snapshot(explicit_fixture, explicit_backing);
        auto explicit_selected = std::get<std::shared_ptr<platform::MetalShaderPackage>>(
            explicit_opened)->lookup(explicit_stop, explicit_memory);
        if (const auto* explicit_pair =
                std::get_if<platform::CompiledGraphicsPair>(&explicit_selected)) {
            CHECK(explicit_pair->fragment->texture_handle_cbuf == std::optional<std::uint8_t>(2));
        } else {
            CHECK(false);
        }
    }
}

// The widened v2 profile: flat, perspective and linear vertex outputs;
// a fragment perspective input without a declared position input (the draw
// backend requires the compiled position convention instead); and
// vertex-stage texture pairs bound through the shared descriptor path.
void test_ported_v2_widened_interfaces(const fs::path& base) {
    const auto output_form = [](const char* interpolation) {
        return "\"component_mask\":3,\"scalar\":\"f32\",\"components\":4,"
               "\"interpolation\":" + std::string(interpolation);
    };
    const std::pair<const char*, const char*> forms[] = {
        {"perspective", "\"perspective\""}, {"flat", "\"flat\""}, {"linear", "\"linear\""}};
    for (const auto& [name, form] : forms) {
        auto fixture = package(base / ("widened-output-" + std::string(name)), true);
        mutate_reflection(fixture, fixture.vertex_variant, output_form("\"linear\""),
                          output_form(form));
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) {
            Backing backing;
            auto address_space = memory(backing);
            auto stop = snapshot(fixture, backing);
            CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(
                std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened)->lookup(
                    stop, address_space)));
        }
    }
    // The draw94 shape: perspective fragment inputs read the compiled
    // convention rather than a declared position input.
    auto no_position = package(base / "widened-fragment-no-position", true);
    mutate_reflection(no_position, no_position.fragment_variant,
                      "\"position_input_mask\":8", "\"position_input_mask\":0");
    auto no_position_opened = platform::MetalShaderPackage::open(no_position.root,
                                                                 {no_position.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
        no_position_opened));
    if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
            no_position_opened)) {
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(no_position, backing);
        auto selected = std::get<std::shared_ptr<platform::MetalShaderPackage>>(
            no_position_opened)->lookup(stop, address_space);
        if (const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected)) {
            CHECK(pair->fragment->interface.position_input_mask == 0);
            CHECK(pair->fragment->fragment_position ==
                  gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW);
        } else {
            CHECK(false);
        }
    }
    // Perspective inputs without the compiled position convention remain a stop.
    auto missing = package(base / "widened-perspective-no-convention", true);
    mutate_reflection(missing, missing.fragment_variant,
                      "\"fragment_position\":\"reciprocal-clip-w\"",
                      "\"fragment_position\":\"unspecified\"");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(missing.root, {missing.chain, true})));

    // Vertex-stage texture pairs bind like fragment ones through the guest
    // descriptor path, provenance included.
    const std::string vertex_textures =
        "[{\"cbuf_slot\":2,\"handle_byte_offset\":32,\"kind\":\"texture\",\"dimension\":"
        "\"d2\",\"array\":false,\"multisample\":false,\"scalar\":\"f32\","
        "\"component_format\":\"unorm\",\"normalized_coordinates\":true,"
        "\"comparison\":false,\"texture_index\":3},"
        "{\"cbuf_slot\":2,\"handle_byte_offset\":32,\"kind\":\"sampler\","
        "\"comparison\":false,\"normalized_coordinates\":true,\"sampler_index\":1}]";
    const std::string vertex_handle_fact =
        ",{\"kind\":\"texture_handle_cbuf\",\"index\":0,\"byte_offset\":0,"
        "\"component\":0,\"origin\":\"explicit_option\",\"value\":2}";
    auto textured = package(base / "widened-vertex-textures", true, "[]", "",
                            vertex_textures, vertex_handle_fact);
    auto textured_opened = platform::MetalShaderPackage::open(textured.root,
                                                              {textured.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
        textured_opened));
    if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
            textured_opened)) {
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(textured, backing);
        auto selected = std::get<std::shared_ptr<platform::MetalShaderPackage>>(
            textured_opened)->lookup(stop, address_space);
        if (const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected)) {
            CHECK(pair->vertex->textures.size() == 1);
            CHECK(pair->vertex->texture_handle_cbuf == std::optional<std::uint8_t>(2));
            CHECK(pair->vertex->texture_facts.size() == 1);
            CHECK(pair->fragment->textures.empty());
        } else {
            CHECK(false);
        }
    }
    // A fragment input without a covering vertex output still fails to link.
    auto unlinked = package(base / "widened-unlinked", true);
    mutate_reflection(unlinked, unlinked.vertex_variant,
                      ",\"generic_outputs\":[{\"slot\":0,\"component_mask\":3,"
                      "\"scalar\":\"f32\",\"components\":4,\"interpolation\":\"linear\"}]",
                      ",\"generic_outputs\":[]");
    auto unlinked_opened = platform::MetalShaderPackage::open(unlinked.root,
                                                              {unlinked.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
        unlinked_opened));
    if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
            unlinked_opened)) {
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(unlinked, backing);
        CHECK(std::holds_alternative<graphics::CommandResult>(
            std::get<std::shared_ptr<platform::MetalShaderPackage>>(unlinked_opened)->lookup(
                stop, address_space)));
    }
}

// Every reflected v2 texture becomes a complete draw-time descriptor contract.
// A consumed texture_descriptor dependency, when present, must restate that
// contract exactly regardless of its evidence origin.
void test_ported_v2_texture_descriptor_facts(const fs::path& base) {
    const std::string textures = "[{\"cbuf_slot\":2,\"handle_byte_offset\":32,\"kind\":"
        "\"texture\",\"dimension\":\"d2\",\"array\":false,\"multisample\":false,"
        "\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":true,\"comparison\":false,\"texture_index\":3},"
        "{\"cbuf_slot\":2,\"handle_byte_offset\":32,\"kind\":\"sampler\","
        "\"comparison\":false,\"normalized_coordinates\":true,\"sampler_index\":1}]";
    const std::string value = "{\"dimension\":\"d2\",\"array\":false,\"multisample\":false,"
        "\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":true,\"comparison\":false}";
    const auto fact = [](const std::string& fact_value, unsigned index = 2,
                         unsigned offset = 32, const char* origin = "derived") {
        return ",{\"kind\":\"texture_descriptor\",\"index\":" + std::to_string(index) +
               ",\"byte_offset\":" + std::to_string(offset) + ",\"component\":0,"
               "\"origin\":\"" + origin + "\",\"value\":" + fact_value + "}";
    };
    auto fixture = package(base / "ported-fact", true, textures,
                           fact(value) + ",{\"kind\":\"texture_handle_cbuf\",\"index\":0,"
                               "\"byte_offset\":0,\"component\":0,\"origin\":"
                               "\"explicit_option\",\"value\":2}");
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
    Backing backing;
    auto address_space = memory(backing);
    auto stop = snapshot(fixture, backing);
    auto selected = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened)->lookup(
        stop, address_space);
    if (const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected)) {
        CHECK(pair->fragment->texture_facts.size() == 1);
        if (pair->fragment->texture_facts.size() == 1) {
            const auto& admitted = pair->fragment->texture_facts.front();
            CHECK(admitted.cbuf_slot == 2);
            CHECK(admitted.handle_byte_offset == 32);
            CHECK(admitted.dimension == platform::TextureFactDimension::d2);
            CHECK(!admitted.array && !admitted.multisample);
            CHECK(admitted.scalar == platform::TextureSampledScalar::f32);
            CHECK(admitted.component_format == platform::TextureComponentFormat::floating);
            CHECK(admitted.normalized_coordinates);
            CHECK(!admitted.comparison);
        }
        CHECK(pair->vertex->texture_facts.empty());
    } else {
        CHECK(false);
    }

    // The serialized origin does not weaken agreement. In particular,
    // descriptor assumptions and captured facts obey the same exact contract.
    auto captured_fixture = package(base / "ported-captured-fact", true, textures,
                                    fact(value, 2, 32, "captured_state"));
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
        platform::MetalShaderPackage::open(captured_fixture.root,
                                            {captured_fixture.chain, true})));

    const auto descriptor = [](const char* dimension, bool array, bool multisample,
                               const char* scalar, const char* format, bool normalized,
                               bool comparison) {
        return "{\"dimension\":\"" + std::string(dimension) + "\",\"array\":" +
            (array ? "true" : "false") + ",\"multisample\":" +
            (multisample ? "true" : "false") + ",\"scalar\":\"" + scalar +
            "\",\"component_format\":\"" + format +
            "\",\"normalized_coordinates\":" + (normalized ? "true" : "false") +
            ",\"comparison\":" + (comparison ? "true" : "false") + "}";
    };
    const auto reflected = [](const std::string& contract, bool sampler,
                              bool normalized, bool comparison) {
        std::string result = "[{\"cbuf_slot\":2,\"handle_byte_offset\":32,\"kind\":"
            "\"texture\"," + contract.substr(1, contract.size() - 2) +
            ",\"texture_index\":3}";
        if (sampler)
            result += ",{\"cbuf_slot\":2,\"handle_byte_offset\":32,\"kind\":"
                "\"sampler\",\"comparison\":" + std::string(comparison ? "true" : "false") +
                ",\"normalized_coordinates\":" + (normalized ? "true" : "false") +
                ",\"sampler_index\":1}";
        return result + "]";
    };
    struct Admission {
        const char* label;
        const char* dimension;
        bool array;
        const char* scalar;
        const char* format;
        bool normalized;
        bool comparison;
        bool sampler;
        platform::TextureFactDimension expected_dimension;
        platform::TextureSampledScalar expected_scalar;
        platform::TextureComponentFormat expected_format;
    };
    const Admission admissions[] = {
        {"d1", "d1", false, "f32", "floating", true, false, true,
         platform::TextureFactDimension::d1, platform::TextureSampledScalar::f32,
         platform::TextureComponentFormat::floating},
        {"d1-array", "d1", true, "f32", "unorm", true, false, true,
         platform::TextureFactDimension::d1, platform::TextureSampledScalar::f32,
         platform::TextureComponentFormat::unorm},
        {"d2-array-depth", "d2", true, "f32", "depth", true, true, true,
         platform::TextureFactDimension::d2, platform::TextureSampledScalar::f32,
         platform::TextureComponentFormat::depth},
        {"d3", "d3", false, "f32", "snorm", true, false, true,
         platform::TextureFactDimension::d3, platform::TextureSampledScalar::f32,
         platform::TextureComponentFormat::snorm},
        {"cube", "cube", false, "f32", "floating", true, false, true,
         platform::TextureFactDimension::cube, platform::TextureSampledScalar::f32,
         platform::TextureComponentFormat::floating},
        {"cube-array", "cube", true, "f32", "depth", true, true, true,
         platform::TextureFactDimension::cube, platform::TextureSampledScalar::f32,
         platform::TextureComponentFormat::depth},
        {"uint-fetch", "d2", false, "u32", "unsigned_integer", true, false, false,
         platform::TextureFactDimension::d2, platform::TextureSampledScalar::u32,
         platform::TextureComponentFormat::unsigned_integer},
        {"sint-fetch", "d3", false, "s32", "signed_integer", true, false, false,
         platform::TextureFactDimension::d3, platform::TextureSampledScalar::s32,
         platform::TextureComponentFormat::signed_integer},
        {"rectangle-fetch", "d2", false, "f32", "unorm", false, false, false,
         platform::TextureFactDimension::d2, platform::TextureSampledScalar::f32,
         platform::TextureComponentFormat::unorm},
    };
    for (const auto& item : admissions) {
        const auto contract = descriptor(item.dimension, item.array, false, item.scalar,
                                         item.format, item.normalized, item.comparison);
        auto admitted_fixture = package(base / ("ported-target-" + std::string(item.label)),
            true, reflected(contract, item.sampler, item.normalized, item.comparison));
        auto admitted_opened = platform::MetalShaderPackage::open(
            admitted_fixture.root, {admitted_fixture.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(
            admitted_opened));
        if (const auto* admitted_package =
                std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&admitted_opened)) {
            Backing admitted_backing;
            auto admitted_memory = memory(admitted_backing);
            auto admitted_stop = snapshot(admitted_fixture, admitted_backing);
            const auto admitted_selected = (*admitted_package)->lookup(
                admitted_stop, admitted_memory);
            const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&admitted_selected);
            CHECK(pair && pair->fragment->texture_facts.size() == 1);
            CHECK(pair && pair->fragment->textures.size() == 1);
            if (pair && pair->fragment->texture_facts.size() == 1) {
                const auto& admitted_fact = pair->fragment->texture_facts.front();
                CHECK(admitted_fact.dimension == item.expected_dimension);
                CHECK(admitted_fact.array == item.array && !admitted_fact.multisample);
                CHECK(admitted_fact.scalar == item.expected_scalar);
                CHECK(admitted_fact.component_format == item.expected_format);
                CHECK(admitted_fact.normalized_coordinates == item.normalized);
                CHECK(admitted_fact.comparison == item.comparison);
            }
            if (pair && pair->fragment->textures.size() == 1)
                CHECK(pair->fragment->textures.front().sampler_index ==
                      (item.sampler ? 1 : shader::graphics_msl::TextureBinding::no_sampler));
        }
    }

    const auto rejected_reflection = [&](const char* label, const std::string& contract,
                                         bool sampler, bool normalized,
                                         bool comparison) {
        auto rejected_fixture = package(base / ("ported-target-rejected-" +
            std::string(label)), true,
            reflected(contract, sampler, normalized, comparison));
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(rejected_fixture.root,
                                                {rejected_fixture.chain, true})));
    };
    // These profiles still have no valid single-sample MSL/runtime path.
    rejected_reflection("multisample",
        descriptor("d2", false, true, "f32", "floating", true, false),
        false, true, false);
    rejected_reflection("d3-array",
        descriptor("d3", true, false, "f32", "floating", true, false),
        false, true, false);
    rejected_reflection("unnormalized-d1",
        descriptor("d1", false, false, "f32", "floating", false, false),
        false, false, false);
    rejected_reflection("unnormalized-sampling",
        descriptor("d2", false, false, "f32", "unorm", false, false),
        true, false, false);
    rejected_reflection("integer-sampling",
        descriptor("d2", false, false, "u32", "unsigned_integer", true, false),
        true, true, false);
    rejected_reflection("scalar-format",
        descriptor("d2", false, false, "u32", "floating", true, false),
        false, true, false);
    rejected_reflection("depth-shape",
        descriptor("d1", false, false, "f32", "depth", true, false),
        false, true, false);

    const auto depth_contract = [](std::string text) {
        for (const auto& [before, after] : {
                std::pair{std::string("\"comparison\":false"), std::string("\"comparison\":true")},
                std::pair{std::string("\"component_format\":\"floating\""),
                          std::string("\"component_format\":\"depth\"")}}) {
            std::size_t at = 0;
            while ((at = text.find(before, at)) != std::string::npos) {
                text.replace(at, before.size(), after);
                at += after.size();
            }
        }
        return text;
    };
    auto depth_fixture = package(base / "ported-depth-fact", true,
                                 depth_contract(textures), fact(depth_contract(value)));
    auto depth_opened = platform::MetalShaderPackage::open(
        depth_fixture.root, {depth_fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(depth_opened));
    if (const auto* admitted = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&depth_opened)) {
        Backing depth_backing;
        auto depth_memory = memory(depth_backing);
        auto depth_stop = snapshot(depth_fixture, depth_backing);
        const auto selected_depth = (*admitted)->lookup(depth_stop, depth_memory);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected_depth);
        CHECK(pair && pair->fragment->texture_facts.size() == 1);
        if (pair && pair->fragment->texture_facts.size() == 1)
            CHECK(pair->fragment->texture_facts.front().comparison);
    }
    // Depth comparison and floating-color contracts cannot be interchanged.
    for (const auto& [name, entries, dependency] : {
            std::tuple{"depth-float-fact", depth_contract(textures), fact(value)},
            std::tuple{"float-depth-fact", textures, fact(depth_contract(value))},
            std::tuple{"depth-float-component", depth_contract(textures),
                fact("{\"dimension\":\"d2\",\"array\":false,\"multisample\":false,"
                     "\"scalar\":\"f32\",\"component_format\":\"floating\","
                     "\"normalized_coordinates\":true,\"comparison\":true}")}}) {
        auto invalid = package(base / name, true, entries, dependency);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(invalid.root, {invalid.chain, true})));
    }

    const auto mutated = [&](const char* name, const std::string& extra) {
        auto broken = package(base / ("ported-fact-" + std::string(name)), true,
                              textures, extra);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    };
    // Names no reflected bound handle at the recorded key.
    mutated("unbound-handle", fact(value, 2, 36));
    // Slot provenance disagrees with the bound handle's slot.
    mutated("slot-mismatch", fact(value, 4, 32));
    // Every descriptor field must agree with the reflected runtime contract.
    mutated("array-value", fact("{\"dimension\":\"d2\",\"array\":true,\"multisample\":false,"
        "\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":true,\"comparison\":false}"));
    mutated("multisample-value", fact("{\"dimension\":\"d2\",\"array\":false,"
        "\"multisample\":true,\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":true,\"comparison\":false}"));
    mutated("cube-value", fact("{\"dimension\":\"cube\",\"array\":false,\"multisample\":false,"
        "\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":true,\"comparison\":false}"));
    mutated("integer-scalar", fact("{\"dimension\":\"d2\",\"array\":false,\"multisample\":false,"
        "\"scalar\":\"u32\",\"component_format\":\"unsigned_integer\","
        "\"normalized_coordinates\":true,\"comparison\":false}"));
    mutated("unorm-format", fact("{\"dimension\":\"d2\",\"array\":false,\"multisample\":false,"
        "\"scalar\":\"f32\",\"component_format\":\"unorm\","
        "\"normalized_coordinates\":true,\"comparison\":false}"));
    mutated("unnormalized-value", fact("{\"dimension\":\"d2\",\"array\":false,"
        "\"multisample\":false,\"scalar\":\"f32\",\"component_format\":\"floating\","
        "\"normalized_coordinates\":false,\"comparison\":false}"));
    // The fact must agree with the bound handle's reflected comparison state.
    mutated("comparison-mismatch", fact("{\"dimension\":\"d2\",\"array\":false,"
        "\"multisample\":false,\"scalar\":\"f32\",\"component_format\":\"depth\","
        "\"normalized_coordinates\":true,\"comparison\":true}"));
    // Duplicate dependency for one handle key.
    mutated("duplicate", fact(value) + fact(value));
    // The dependency value must be the descriptor object.
    mutated("scalar-value", ",{\"kind\":\"texture_descriptor\",\"index\":2,"
        "\"byte_offset\":32,\"component\":0,\"origin\":\"derived\",\"value\":0}");
}

// Typed colour-output records in the v2 interface: per-target exports
// validated against their fragment_output_type dependencies and the summary
// mask. Admission of which target sets the renderer binds happens at draw time.
void test_ported_v2_colour_outputs(const fs::path& base) {
    const auto record = [](unsigned location, unsigned mask, const char* scalar,
                           unsigned components) {
        return "{\"location\":" + std::to_string(location) + ",\"component_mask\":" +
               std::to_string(mask) + ",\"scalar\":" + std::string(scalar) +
               ",\"components\":" + std::to_string(components) + "}";
    };
    const auto fact = [](unsigned target, const char* scalar, unsigned components) {
        return "{\"kind\":\"fragment_output_type\",\"index\":" + std::to_string(target) +
               ",\"byte_offset\":0,\"component\":0,\"origin\":\"derived\","
               "\"value\":{\"scalar\":" + std::string(scalar) +
               ",\"components\":" + std::to_string(components) + "}}";
    };
    const std::string default_record = record(0, 15, "\"f32\"", 4);
    const std::string default_fact = fact(0, "\"f32\"", 4);
    // A target-0-plus-target-2 stage (mask 0x0f0f) with agreeing facts loads,
    // as does an unsigned-integer export class and an explicit depth write.
    {
        auto fixture = package(base / "colour-mrt", true);
        mutate_reflection(fixture, fixture.fragment_variant, default_record,
                          default_record + "," + record(2, 15, "\"f32\"", 4));
        mutate_reflection(fixture, fixture.fragment_variant, default_fact,
                          default_fact + "," + fact(2, "\"f32\"", 4));
        mutate_reflection(fixture, fixture.fragment_variant,
                          "\"color_output_mask\":15", "\"color_output_mask\":3855");
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (const auto* loaded =
                std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened)) {
            Backing backing;
            auto address_space = memory(backing);
            auto stop = snapshot(fixture, backing);
            auto selected = (*loaded)->lookup(stop, address_space);
            const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
            CHECK(pair && pair->fragment->color_outputs.size() == 2);
            if (pair && pair->fragment->color_outputs.size() == 2)
                CHECK(pair->fragment->color_outputs[1].location == 2);
        }
    }
    {
        auto fixture = package(base / "colour-uint", true);
        mutate_reflection(fixture, fixture.fragment_variant, default_record,
                          record(0, 15, "\"u32\"", 4));
        mutate_reflection(fixture, fixture.fragment_variant, default_fact,
                          fact(0, "\"u32\"", 4));
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    }
    {
        auto fixture = package(base / "colour-depth-flag", true);
        mutate_reflection(fixture, fixture.fragment_variant,
                          "\"writes_fragment_depth\":false",
                          "\"writes_fragment_depth\":true");
        auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (const auto* loaded =
                std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened)) {
            Backing backing;
            auto address_space = memory(backing);
            auto stop = snapshot(fixture, backing);
            auto selected = (*loaded)->lookup(stop, address_space);
            const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
            CHECK(pair && pair->fragment->writes_fragment_depth);
        }
    }

    // Structured rejections: malformed records, duplicate targets, summary
    // disagreements and record/fact type mismatches all refuse to load. Each
    // entry names the stage variant it mutates.
    const std::tuple<const char*, bool, std::string, std::string> rejections[] = {
        {"summary-disagreement", false, default_record, record(0, 5, "\"f32\"", 4)},
        {"summary-records-missing", false, default_record, ""},
        {"summary-mask-stale", false, "\"color_output_mask\":15",
         "\"color_output_mask\":3855"},
        {"duplicate-location", false, default_record,
         default_record + "," + record(0, 15, "\"f32\"", 4)},
        {"mask-outside-components", false, default_record, record(0, 15, "\"f32\"", 2)},
        {"unsupported-scalar", false, default_record, record(0, 15, "\"f64\"", 4)},
        {"empty-mask", false, default_record, record(0, 0, "\"f32\"", 4)},
        {"extra-field", false, default_record,
         "{\"location\":0,\"component_mask\":15,\"scalar\":\"f32\",\"components\":4,"
         "\"interpolation\":\"flat\"}"},
        {"record-type-mismatch", false, default_fact, fact(0, "\"u32\"", 4)},
        {"record-components-mismatch", false, default_fact, fact(0, "\"f32\"", 2)},
        {"missing-fact", false, default_fact + ",", ""},
        {"extra-fact", false, default_fact, default_fact + "," + fact(1, "\"f32\"", 4)},
        {"fact-offset", false,
         ",\"byte_offset\":0,\"component\":0,\"origin\":\"derived\","
         "\"value\":{\"scalar\":\"f32\",\"components\":4}},"
         "{\"kind\":\"input_interpolation\"",
         ",\"byte_offset\":32,\"component\":0,\"origin\":\"derived\","
         "\"value\":{\"scalar\":\"f32\",\"components\":4}},"
         "{\"kind\":\"input_interpolation\"}"},
        {"vertex-fact", true,
         "\"origin\":\"explicit_option\","
         "\"value\":{\"format\":\"f32\",\"components\":3}}",
         "\"origin\":\"explicit_option\","
         "\"value\":{\"format\":\"f32\",\"components\":3}},"
         "{\"kind\":\"fragment_output_type\",\"index\":0,\"byte_offset\":0,"
         "\"component\":0,\"origin\":\"derived\","
         "\"value\":{\"scalar\":\"f32\",\"components\":4}}"},
    };
    unsigned ordinal = 0;
    for (const auto& [label, on_vertex, before, after] : rejections) {
        auto broken = package(base / ("colour-invalid-" + std::to_string(ordinal++)), true);
        mutate_reflection(broken, on_vertex ? broken.vertex_variant : broken.fragment_variant,
                          before, after);
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }
}


void test_adaptive_fragment_outputs(const fs::path& base) {
    const auto make_adaptive = [](const Fixture& fixture) {
        const auto& variant = fixture.fragment_variant;
        mutate_reflection(fixture, variant, "\"schema\":\"armigo-shader-stage-v2\"",
            "\"schema\":\"armigo-shader-stage-v2\",\"fragment_output_mode\":\"render-target-v1\"");
        mutate_reflection(fixture, variant,
            "\"location\":0,\"component_mask\":15,\"scalar\":\"f32\"",
            "\"location\":0,\"component_mask\":15,\"scalar\":\"u32\"");
        mutate_reflection(fixture, variant,
            "\"value\":{\"scalar\":\"f32\",\"components\":4}",
            "\"value\":{\"scalar\":\"u32\",\"components\":4}");
    };
    auto fixture = package(base / "adaptive-outputs", true);
    make_adaptive(fixture);
    mutate_reflection(fixture, fixture.fragment_variant,
        "\"color_outputs\":[{\"location\":0,\"component_mask\":15,\"scalar\":\"u32\",\"components\":4}]",
        "\"color_outputs\":[{\"location\":0,\"component_mask\":15,\"scalar\":\"u32\",\"components\":4},"
        "{\"location\":2,\"component_mask\":15,\"scalar\":\"u32\",\"components\":4}]");
    mutate_reflection(fixture, fixture.fragment_variant,
        "\"color_output_mask\":15", "\"color_output_mask\":3855");
    mutate_reflection(fixture, fixture.fragment_variant, "\"dependencies\":[",
        "\"dependencies\":[{\"kind\":\"fragment_output_type\",\"index\":2,\"byte_offset\":0,"
        "\"component\":0,\"origin\":\"derived\",\"value\":{\"scalar\":\"u32\",\"components\":4}},");
    auto opened = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
    auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
    const auto unspecialized = package_ptr->lookup_stage(shader_input::Stage::Fragment, fixture.fragment,
        shader::graphics_msl::VertexDepthConvention::NegativeOneToOne);
    CHECK(std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(unspecialized));
    if (const auto* executable = std::get_if<std::shared_ptr<const platform::CompiledGraphicsStage>>(&unspecialized)) {
        CHECK((*executable)->fragment_output_types_from_render_targets);
        CHECK((*executable)->color_outputs[0].scalar == platform::ColorOutputScalar::u32);
    }
    Backing backing;
    auto address_space = memory(backing);
    auto stop = snapshot(fixture, backing);
    // Logical outputs 0 and 2 use physical targets 5 and 1; logical 1 is a hole.
    stop.registers[0x487] = 3 | (5 << 4) | (7 << 7) | (1 << 10);
    const auto target = [&](unsigned physical, unsigned format) {
        auto& description = stop.color_targets[physical];
        description.address = 0x100000 + physical * 0x10000;
        description.width = 8;
        description.height = 8;
        description.format = format;
        description.depth = 1;
        stop.registers[0x200 + physical * 16 + 6] = 1;
    };
    target(5, 0xc2); // RGBA32 unsigned integer.
    target(1, 0xc1); // RGBA32 signed integer.
    const auto selected = package_ptr->lookup(stop, address_space);
    const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
    CHECK(pair != nullptr);
    if (!pair) {
        std::fprintf(stderr, "adaptive lookup: %.*s\n",
            int(std::get<graphics::CommandResult>(selected).detail.size()),
            std::get<graphics::CommandResult>(selected).detail.data());
        return;
    }
    CHECK(pair->fragment->fragment_output_types_from_render_targets);
    CHECK(pair->fragment->color_outputs[0].scalar == platform::ColorOutputScalar::u32);
    CHECK(pair->fragment->color_outputs[1].scalar == platform::ColorOutputScalar::s32);
    const auto first = pair->fragment;
    target(5, 0xc0); // The same guest program now exports float at logical 0.
    const auto changed = package_ptr->lookup(stop, address_space);
    const auto* changed_pair = std::get_if<platform::CompiledGraphicsPair>(&changed);
    CHECK(changed_pair && changed_pair->fragment != first);
    if (changed_pair) {
        CHECK(changed_pair->fragment->color_outputs[0].scalar == platform::ColorOutputScalar::f32);
        CHECK(changed_pair->fragment->color_outputs[1].scalar == platform::ColorOutputScalar::s32);
        CHECK(changed_pair->fragment->metallib == first->metallib);
    }
    CHECK(first->color_outputs[0].scalar == platform::ColorOutputScalar::u32);
    target(5, 0xc2);
    const auto repeated = package_ptr->lookup(stop, address_space);
    const auto* repeated_pair = std::get_if<platform::CompiledGraphicsPair>(&repeated);
    CHECK(repeated_pair && repeated_pair->fragment == first);
    target(1, 0);
    const auto unbound = package_ptr->lookup(stop, address_space);
    const auto* unbound_pair = std::get_if<platform::CompiledGraphicsPair>(&unbound);
    CHECK(unbound_pair && unbound_pair->fragment != first);
    if (unbound_pair) {
        CHECK(unbound_pair->fragment->color_outputs[0].scalar ==
              platform::ColorOutputScalar::u32);
        CHECK(unbound_pair->fragment->color_outputs[1].scalar ==
              platform::ColorOutputScalar::u32);
        const auto unbound_repeated = package_ptr->lookup(stop, address_space);
        const auto* unbound_repeated_pair =
            std::get_if<platform::CompiledGraphicsPair>(&unbound_repeated);
        CHECK(unbound_repeated_pair &&
              unbound_repeated_pair->fragment == unbound_pair->fragment);
    }
    target(1, 0xffff);
    CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup(stop, address_space)));

    for (const bool vertex : {false, true}) {
        auto broken = package(base / (vertex ? "adaptive-vertex" : "adaptive-float-storage"), true);
        mutate_reflection(broken, vertex ? broken.vertex_variant : broken.fragment_variant,
            "\"schema\":\"armigo-shader-stage-v2\"",
            "\"schema\":\"armigo-shader-stage-v2\",\"fragment_output_mode\":\"render-target-v1\"");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }
    auto unknown = package(base / "adaptive-unknown-mode", true);
    make_adaptive(unknown);
    mutate_reflection(unknown, unknown.fragment_variant, "render-target-v1", "unknown-mode");
    CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
        platform::MetalShaderPackage::open(unknown.root, {unknown.chain, true})));
}

void test_depth_bias_capability(const fs::path& base) {
    const auto enable = [](const auto& fixture, const auto& variant) {
        mutate_reflection(fixture, variant,
            "\"schema\":\"armigo-shader-stage-v2\"",
            "\"schema\":\"armigo-shader-stage-v2\",\"depth_bias_mode\":\"shader-v1\"");
    };
    auto valid = package(base / "depth-bias-capability", true);
    enable(valid, valid.fragment_variant);
    auto opened = platform::MetalShaderPackage::open(valid.root, {valid.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
    if (const auto* ptr = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&opened)) {
        Backing backing;
        auto address_space = memory(backing);
        auto selected = (*ptr)->lookup(snapshot(valid, backing), address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && pair->fragment->emulates_depth_bias && !pair->vertex->emulates_depth_bias);
    }
    for (unsigned mode = 0; mode != 3; ++mode) {
        auto broken = package(base / ("depth-bias-invalid-" + std::to_string(mode)), true);
        const auto variant = mode == 0 ? broken.vertex_variant : broken.fragment_variant;
        enable(broken, variant);
        if (mode == 1) mutate_reflection(broken, variant, "shader-v1", "unknown");
        if (mode == 2) mutate_reflection(broken, variant,
            "\"writes_fragment_depth\":false", "\"writes_fragment_depth\":true");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(broken.root, {broken.chain, true})));
    }
}

void test_lazy_entries(const fs::path& base) {
    const auto lazy = [](const Fixture& fixture) {
        platform::MetalShaderPackage::Options options{fixture.chain, true};
        options.lazy_entries = true;
        return options;
    };
    {
        // A lazy open reads no reflection: it matches what the eager open
        // returns, and stages are found by lookup and by identity text.
        auto fixture = package(base / "lazy-good");
        auto eager = platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true});
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(eager));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(eager) ||
            !std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto eager_package = std::get<std::shared_ptr<platform::MetalShaderPackage>>(eager);
        auto lazy_package = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        auto expected = eager_package->lookup(stop, address_space);
        auto actual = lazy_package->lookup(stop, address_space);
        CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(expected));
        CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(actual));
        if (std::holds_alternative<platform::CompiledGraphicsPair>(expected) &&
            std::holds_alternative<platform::CompiledGraphicsPair>(actual)) {
            const auto& want = std::get<platform::CompiledGraphicsPair>(expected);
            const auto& got = std::get<platform::CompiledGraphicsPair>(actual);
            CHECK(got.vertex->metallib == want.vertex->metallib);
            CHECK(got.fragment->metallib == want.fragment->metallib);
            CHECK(got.vertex->identity == want.vertex->identity);
            CHECK(got.fragment->identity == want.fragment->identity);
            CHECK(got.fragment->color_outputs.size() == want.fragment->color_outputs.size());
            const auto fresh = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
            auto by_identity = std::get<std::shared_ptr<platform::MetalShaderPackage>>(fresh)
                ->stage_by_identity(want.vertex->identity);
            CHECK(by_identity && by_identity->metallib == want.vertex->metallib);
            CHECK(!std::get<std::shared_ptr<platform::MetalShaderPackage>>(fresh)
                       ->stage_by_identity(want.vertex->identity + "9"));
            CHECK(!std::get<std::shared_ptr<platform::MetalShaderPackage>>(fresh)
                       ->stage_by_identity("not an identity"));
        }
    }
    {
        // Background preparation yields the same stages, and a package that is
        // destroyed while the threads run joins them.
        auto fixture = package(base / "lazy-prefetch");
        auto options = lazy(fixture);
        options.prefetch_threads = 3;
        auto opened = platform::MetalShaderPackage::open(fixture.root, options);
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(package_ptr->lookup(stop, address_space)));
        package_ptr.reset();
        auto brief = platform::MetalShaderPackage::open(fixture.root, options);
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(brief));
    }
    {
        // Stages the title selects first are prepared first, once each: the
        // outcome counts every stage once, whatever the hint lists, and a
        // hint after the first or on a package without threads changes nothing.
        auto fixture = package(base / "lazy-prefetch-first");
        auto options = lazy(fixture);
        options.prefetch_threads = 2;
        auto opened = platform::MetalShaderPackage::open(fixture.root, options);
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        const auto selected = package_ptr->lookup(stop, address_space);
        const auto* pair = std::get_if<platform::CompiledGraphicsPair>(&selected);
        CHECK(pair && pair->vertex && pair->fragment);
        if (!pair || !pair->vertex || !pair->fragment) return;
        const std::vector<std::string> hint{pair->fragment->identity, "9.not-a-digest.1", "no dot",
                                            pair->fragment->identity, pair->vertex->identity};
        package_ptr->prefetch_first(hint);
        package_ptr->prefetch_first(hint);
        const auto outcome = package_ptr->finish_prefetch();
        CHECK(outcome.failed == 0);
        CHECK(outcome.prepared == 2);
        auto unthreaded = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(unthreaded));
        if (std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(unthreaded)) {
            std::get<std::shared_ptr<platform::MetalShaderPackage>>(unthreaded)->prefetch_first(hint);
            CHECK(std::get<std::shared_ptr<platform::MetalShaderPackage>>(unthreaded)
                      ->finish_prefetch().prepared == 0);
        }
    }
    {
        // A reflection file that does not start with its schema takes the full
        // parse and loads as before.
        auto fixture = package(base / "lazy-schema-not-first");
        mutate_reflection(fixture, fixture.vertex_variant, "{\"schema\":", "{ \"schema\":");
        mutate_reflection(fixture, fixture.fragment_variant, "{\"schema\":", "{ \"schema\":");
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(package_ptr->lookup(stop, address_space)));
    }
    {
        // A stage that fails in the background is reported by the prefetch
        // outcome and again, naming the stage, by the lookup that selects it;
        // it never counts as prepared, and the stages around it are unaffected.
        auto fixture = package(base / "lazy-prefetch-digest");
        std::ofstream stream(fixture.root / "stages" / fixture.fragment_variant / "reflection.json",
                             std::ios::binary | std::ios::app);
        stream.put(' '); stream.close();
        auto options = lazy(fixture);
        options.prefetch_threads = 3;
        auto opened = platform::MetalShaderPackage::open(fixture.root, options);
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        const auto outcome = package_ptr->finish_prefetch();
        CHECK(outcome.failed == 1);
        CHECK(outcome.prepared == 1);
        CHECK(outcome.first_failure.find(fixture.fragment_variant) != std::string::npos);
        CHECK(outcome.first_failure.find("Reflection digest differs") != std::string::npos);
        // Waiting again returns the same outcome.
        CHECK(package_ptr->finish_prefetch().failed == 1);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        for (int attempt = 0; attempt != 3; ++attempt) {
            const auto result = package_ptr->lookup(stop, address_space);
            CHECK(std::holds_alternative<graphics::CommandResult>(result));
            if (const auto* failure = std::get_if<graphics::CommandResult>(&result)) {
                CHECK(failure->detail.find(fixture.fragment_variant) != std::string_view::npos);
                CHECK(failure->detail.find("Reflection digest differs") != std::string_view::npos);
            }
        }
        CHECK(std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(
            package_ptr->lookup_stage(shader_input::Stage::Vertex, fixture.vertex,
                shader::graphics_msl::VertexDepthConvention::NegativeOneToOne)));
    }
    {
        // Lookups that select stages while the prefetch threads prepare them,
        // and a package destroyed at any point of that, all agree.
        auto fixture = package(base / "lazy-prefetch-race");
        auto options = lazy(fixture);
        options.prefetch_threads = 4;
        for (int round = 0; round != 200; ++round) {
            auto opened = platform::MetalShaderPackage::open(fixture.root, options);
            CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
            if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
            auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
            if (round % 4 == 3) continue;  // destroyed with the threads possibly mid-entry
            bool vertex_found = false;
            std::thread other([&] {
                vertex_found = std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(
                    package_ptr->lookup_stage(shader_input::Stage::Vertex, fixture.vertex,
                        shader::graphics_msl::VertexDepthConvention::NegativeOneToOne));
            });
            const auto fragment_found = std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(
                package_ptr->lookup_stage(shader_input::Stage::Fragment, fixture.fragment,
                    shader::graphics_msl::VertexDepthConvention::NegativeOneToOne));
            other.join();
            CHECK(vertex_found);
            CHECK(fragment_found);
            if (round % 4 == 0) package_ptr.reset();
            else CHECK(package_ptr->finish_prefetch().failed == 0);
        }
    }
    {
        // Two variants of one program (here, by clip-depth convention) share an
        // identity: the index collapses them, and each lookup picks its variant.
        auto fixture = package(base / "lazy-variants");
        auto zero_to_one = reflection(shader_input::Stage::Vertex, fixture.vertex);
        replace(zero_to_one, "minus-one-to-one", "zero-to-one");
        add_entry(fixture, "vertex", "zero-to-one vertex variant", fixture.vertex, zero_to_one);
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        for (const auto depth : {shader::graphics_msl::VertexDepthConvention::NegativeOneToOne,
                                 shader::graphics_msl::VertexDepthConvention::ZeroToOne}) {
            const auto found = package_ptr->lookup_stage(shader_input::Stage::Vertex, fixture.vertex, depth);
            const auto* stage = std::get_if<std::shared_ptr<const platform::CompiledGraphicsStage>>(&found);
            CHECK(stage && *stage && (*stage)->vertex_depth == depth);
        }
    }
    {
        // A program that is the first bytes of a longer one is told apart by
        // its full digest; when the guest's code is the longer program both
        // match, which a lazy package reports as ambiguous (the eager open
        // rejects such a package).
        auto fixture = package(base / "lazy-prefix-relation");
        auto longer = fixture.vertex;
        longer.resize(longer.size() + 32, std::byte{0x5a});
        add_entry(fixture, "vertex", "longer vertex program", longer,
                  reflection(shader_input::Stage::Vertex, longer));
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        CHECK(std::holds_alternative<platform::CompiledGraphicsPair>(package_ptr->lookup(stop, address_space)));
        // A fresh package: the first one remembers the code it matched.
        auto again = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(again));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(again)) return;
        std::copy(longer.begin(), longer.end(), backing.bytes.begin());
        const auto ambiguous = std::get<std::shared_ptr<platform::MetalShaderPackage>>(again)
                                   ->lookup(stop, address_space);
        const auto* failure = std::get_if<graphics::CommandResult>(&ambiguous);
        CHECK(failure);
        if (failure) CHECK(failure->detail.find("matches multiple identities") != std::string_view::npos);
    }
    {
        // Program artifacts are checked when their stage is prepared, and a
        // defect is reported by the background preparation and by the lookup
        // that selects the stage, naming it: a program for another stage kind,
        // a prefix digest the program does not have, and a missing program.
        struct Case {
            const char* name;
            void (*damage)(const Fixture&);
            const char* reported;
        };
        const Case cases[] = {
            {"lazy-header-mismatch", [](const Fixture& f) {
                 // The fragment program under the vertex stage, with every digest
                 // consistent, so only the stage header disagrees.
                 const auto wrong = program(shader_input::Stage::Fragment, std::byte{0x31});
                 const auto reflected = reflection(shader_input::Stage::Vertex, wrong);
                 const auto directory = f.root / "stages" / f.vertex_variant;
                 write(directory / "program.bin", wrong);
                 write(directory / "reflection.json", reflected);
                 auto manifest = read_text(f.root / "manifest.json");
                 replace(manifest, "\"reflection_digest\":\"" + hash(reflection(shader_input::Stage::Vertex, f.vertex)),
                         "\"reflection_digest\":\"" + hash(reflected));
                 replace(manifest, "\"program_key\":\"" + hash(f.vertex), "\"program_key\":\"" + hash(wrong));
                 replace(manifest, "\"prefix_key\":\"" + hash(f.vertex), "\"prefix_key\":\"" + hash(wrong));
                 write(f.root / "manifest.json", manifest);
             }, "Program identity rejected"},
            {"lazy-prefix-digest", [](const Fixture& f) {
                 auto manifest = read_text(f.root / "manifest.json");
                 replace(manifest, "\"prefix_key\":\"" + hash(f.vertex),
                         "\"prefix_key\":\"" + hash("another prefix"));
                 write(f.root / "manifest.json", manifest);
             }, "prefix digest differs"},
            {"lazy-program-missing", [](const Fixture& f) {
                 fs::remove(f.root / "stages" / f.vertex_variant / "program.bin");
             }, nullptr},
        };
        for (const auto& item : cases) {
            auto fixture = package(base / item.name);
            item.damage(fixture);
            auto options = lazy(fixture);
            options.prefetch_threads = 2;
            auto opened = platform::MetalShaderPackage::open(fixture.root, options);
            CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
            if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
            auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
            const auto outcome = package_ptr->finish_prefetch();
            CHECK(outcome.failed == 1);
            CHECK(outcome.prepared == 1);
            CHECK(outcome.first_failure.find(fixture.vertex_variant) != std::string::npos);
            if (item.reported) CHECK(outcome.first_failure.find(item.reported) != std::string::npos);
            // The stage is never served, and its neighbour is unaffected.
            CHECK(!std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(
                package_ptr->lookup_stage(shader_input::Stage::Vertex, fixture.vertex,
                    shader::graphics_msl::VertexDepthConvention::NegativeOneToOne)));
            CHECK(std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(
                package_ptr->lookup_stage(shader_input::Stage::Fragment, fixture.fragment,
                    shader::graphics_msl::VertexDepthConvention::NegativeOneToOne)));
        }
    }
    {
        // A stage outside the runtime profile is rejected by the eager open and
        // reported by the lookup that selects it; other stages are unaffected.
        auto fixture = package(base / "lazy-reflection");
        mutate_reflection(fixture, fixture.fragment_variant,
                          "armigo_fragment_main", "unsupported_fragment");
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup(stop, address_space)));
        CHECK(std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(
            package_ptr->lookup_stage(shader_input::Stage::Vertex, fixture.vertex,
                shader::graphics_msl::VertexDepthConvention::NegativeOneToOne)));
        // The failure is repeatable: the entry is not left half prepared.
        CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup(stop, address_space)));
    }
    {
        auto fixture = package(base / "lazy-reflection-digest");
        std::ofstream stream(fixture.root / "stages" / fixture.fragment_variant / "reflection.json",
                             std::ios::binary | std::ios::app);
        stream.put(' '); stream.close();
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        Backing backing;
        auto address_space = memory(backing);
        auto stop = snapshot(fixture, backing);
        CHECK(std::holds_alternative<graphics::CommandResult>(package_ptr->lookup(stop, address_space)));
    }
    {
        // A lazy open reads no program artifact: a program that differs from
        // the manifest is reported, naming its stage, by the lookup that
        // selects it, and the other stage is unaffected.
        auto fixture = package(base / "lazy-program");
        std::ofstream stream(fixture.root / "stages" / fixture.vertex_variant / "program.bin",
                             std::ios::binary | std::ios::app);
        stream.put('x'); stream.close();
        CHECK(std::holds_alternative<platform::MetalShaderPackageError>(
            platform::MetalShaderPackage::open(fixture.root, {fixture.chain, true})));
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        const auto vertex = package_ptr->lookup_stage(shader_input::Stage::Vertex, fixture.vertex,
            shader::graphics_msl::VertexDepthConvention::NegativeOneToOne);
        CHECK(std::holds_alternative<graphics::CommandResult>(vertex));
        if (const auto* failure = std::get_if<graphics::CommandResult>(&vertex)) {
            CHECK(failure->detail.find(fixture.vertex_variant) != std::string_view::npos);
            CHECK(failure->detail.find("Program file size differs") != std::string_view::npos);
        }
        CHECK(std::holds_alternative<std::shared_ptr<const platform::CompiledGraphicsStage>>(
            package_ptr->lookup_stage(shader_input::Stage::Fragment, fixture.fragment,
                shader::graphics_msl::VertexDepthConvention::NegativeOneToOne)));
    }
    {
        // Same size, different bytes: the digests catch it.
        auto fixture = package(base / "lazy-program-bytes");
        const auto path = fixture.root / "stages" / fixture.fragment_variant / "program.bin";
        std::fstream stream(path, std::ios::binary | std::ios::in | std::ios::out);
        stream.seekp(static_cast<std::streamoff>(fixture.fragment.size() / 2)); stream.put('\x5a'); stream.close();
        auto opened = platform::MetalShaderPackage::open(fixture.root, lazy(fixture));
        CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened));
        if (!std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(opened)) return;
        auto package_ptr = std::get<std::shared_ptr<platform::MetalShaderPackage>>(opened);
        const auto fragment = package_ptr->lookup_stage(shader_input::Stage::Fragment, fixture.fragment,
            shader::graphics_msl::VertexDepthConvention::NegativeOneToOne);
        CHECK(std::holds_alternative<graphics::CommandResult>(fragment));
        if (const auto* failure = std::get_if<graphics::CommandResult>(&fragment))
            CHECK(failure->detail.find("full digest or prefix digest differs") != std::string_view::npos);
    }
}

Fixture package_with_hoisting(const fs::path& root, std::string_view sample_hoisting) {
    return package(root, false, "[]", "", "[]", "", false, "null", "null", sample_hoisting);
}

// A manifest without sample_hoisting leaves hoisting off. "implicit" turns it
// on. Any other value is a compatibility failure. The chain is the plain
// profile's hash either way.
void test_sample_hoisting(const fs::path& base) {
    auto absent = package(base / "hoist-absent");
    auto absent_opened = platform::MetalShaderPackage::open(absent.root, {absent.chain, true});
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(absent_opened));
    if (const auto* opened = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&absent_opened))
        CHECK(*opened && !(*opened)->hoists_implicit_samples());

    auto implicit = package_with_hoisting(base / "hoist-implicit", "implicit");
    CHECK(implicit.chain == absent.chain);
    auto implicit_opened = platform::MetalShaderPackage::open(implicit.root, {implicit.chain, true});
    if (const auto* error = std::get_if<platform::MetalShaderPackageError>(&implicit_opened))
        std::fprintf(stderr, "implicit hoisting: %s\n", error->detail.c_str());
    CHECK(std::holds_alternative<std::shared_ptr<platform::MetalShaderPackage>>(implicit_opened));
    if (const auto* opened = std::get_if<std::shared_ptr<platform::MetalShaderPackage>>(&implicit_opened))
        CHECK(*opened && (*opened)->hoists_implicit_samples());

    auto explicit_fixture = package_with_hoisting(base / "hoist-explicit", "explicit");
    CHECK(explicit_fixture.chain == absent.chain);
    auto explicit_opened =
        platform::MetalShaderPackage::open(explicit_fixture.root, {explicit_fixture.chain, true});
    const auto* error = std::get_if<platform::MetalShaderPackageError>(&explicit_opened);
    if (!error) {
        std::fprintf(stderr, "FAIL explicit sample hoisting was accepted\n");
        ++failures;
    } else {
        if (error->detail != "Package sample hoisting is unsupported")
            std::fprintf(stderr, "explicit hoisting detail: %s\n", error->detail.c_str());
        CHECK(error->code == platform::MetalShaderPackageError::Code::Compatibility);
        CHECK(error->detail == "Package sample hoisting is unsupported");
    }
}

}  // namespace

int main() {
    @autoreleasepool {
        const auto root = fs::temp_directory_path() /
            ("armigo-metal-package-test-" + std::to_string(NSProcessInfo.processInfo.processIdentifier));
        std::error_code ignored;
        fs::remove_all(root, ignored);
        fs::create_directories(root);
        test_source_stage();
        test_nvn_program_package(root);
        test_open_lookup_cache_and_failures(root);
        test_manifest_and_artifact_rejection(root);
        test_ported_v2_reflections(root);
        test_v3_resource_reflections(root);
        test_v3_storage_buffer_reflections(root);
        test_v3_draw_support_reflections(root);
        test_ported_v2_class_inputs(root);
        test_ported_v2_widened_interfaces(root);
        test_ported_v2_constant_buffers(root);
        test_ported_v2_textures(root);
        test_ported_v2_texture_descriptor_facts(root);
        test_ported_v2_colour_outputs(root);
        test_adaptive_fragment_outputs(root);
        test_depth_bias_capability(root);
        test_lazy_entries(root);
        test_sample_hoisting(root);
        fs::remove_all(root, ignored);
        return failures == 0 ? 0 : 1;
    }
}
