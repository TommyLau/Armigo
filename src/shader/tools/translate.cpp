// ============================================================================
// Armigo Tools - Offline Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// The translation driver exposes host facts through Armigo's frozen pipeline
// contracts. Storage-buffer reflection records descriptor origins and access
// metadata. The derived-descriptor pre-scan applies the per-encoding texture
// shape and depth-compare checks implemented by the texture emitter.

#include <armigo/common/diagnostics.h>
#include <armigo/shader/graphics_msl.h>
#include <armigo/shader/graphics_forward.h>
#include <armigo/shader/source_version.h>
#include <armigo/shader_translation/stage_translation.h>
#include "pipeline/facts.h"
#include "pipeline/phases.h"
#include "sha256.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace si = armigo::shader_input;
namespace mx = armigo::shader::maxwell;
namespace gx = armigo::shader::graphics;
namespace gm = armigo::shader::graphics_msl;
namespace pl = armigo::shader::pipeline;
using Bytes = std::span<const std::byte>;

struct Arguments {
    fs::path control, code, output;
    gm::EmitOptions emission;
    std::string translator{"current"};
    bool translator_seen{};
    // Evidence for the bound-texture handle buffer (3D register 0x982). The
    // ported phases consume it as the texture_handle_cbuf fact; without the
    // option that fact stays Unknown.
    std::optional<std::uint8_t> texture_handle_cbuf;
    // --texture-descriptors derived: pre-scan the program's texture
    // instructions and answer the texture_descriptor facts their encodings
    // prove, for draw-time verification of everything the code cannot know.
    bool texture_descriptors_derived{};
    bool vertex_input_seen{};
    std::optional<pl::ComputeLocalSize> compute_local_size;
    std::optional<std::uint32_t> compute_shared_memory_bytes;
};

constexpr std::string_view usage =
    "Usage: armigo-shader-translate --control FILE --code FILE --output DIR\n"
    "  [--translator current|ported]\n"
    "  [--vertex-float3 SLOT:ATTR ...]\n"
    "  [--vertex-depth minus-one-to-one|zero-to-one]\n"
    "  [--vertex-input descriptor|fetch]\n"
    "  [--fragment-position reciprocal-clip-w]\n"
    "  [--texture-handle-cbuf SLOT]\n"
    "  [--texture-descriptors derived]\n"
    "  [--compute-local-size X,Y,Z --compute-shared-memory BYTES]\n"
    "  armigo-shader-translate --version\n";

std::string quote(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20 || c >= 0x7f) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

unsigned number(std::string_view text, unsigned maximum) {
    unsigned value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        value > maximum) throw std::runtime_error("Invalid numeric option value");
    return value;
}

Arguments arguments(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (i + 1 == argc) throw std::runtime_error("Missing value for " + std::string(option));
        const std::string_view value = argv[++i];
        if (value.empty()) throw std::runtime_error("Empty option value");
        if (option == "--control" || option == "--code" || option == "--output") {
            auto& path = option == "--control" ? result.control :
                         option == "--code" ? result.code : result.output;
            if (!path.empty()) throw std::runtime_error("Duplicate " + std::string(option));
            path = value;
        } else if (option == "--translator") {
            if (result.translator_seen) throw std::runtime_error("Duplicate --translator");
            if (value != "current" && value != "ported")
                throw std::runtime_error("Unsupported translator selection");
            result.translator = value;
            result.translator_seen = true;
        } else if (option == "--vertex-float3") {
            const auto colon = value.find(':');
            if (colon == std::string_view::npos) throw std::runtime_error("Expected SLOT:ATTR");
            result.emission.vertex_attributes.push_back({
                static_cast<std::uint8_t>(number(value.substr(0, colon), 31)),
                static_cast<std::uint8_t>(number(value.substr(colon + 1), 30)),
                gm::VertexFormat::Float32x3});
            if (result.emission.vertex_attributes.size() > 32)
                throw std::runtime_error("Too many vertex format declarations");
        } else if (option == "--vertex-input") {
            if (result.vertex_input_seen) throw std::runtime_error("Duplicate --vertex-input");
            if (value != "descriptor" && value != "fetch")
                throw std::runtime_error("Unsupported vertex input mode");
            result.emission.vertex_fetch = value == "fetch";
            result.vertex_input_seen = true;
        } else if (option == "--vertex-depth") {
            if (result.emission.vertex_depth != gm::VertexDepthConvention::Unspecified)
                throw std::runtime_error("Duplicate --vertex-depth");
            if (value == "minus-one-to-one")
                result.emission.vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
            else if (value == "zero-to-one")
                result.emission.vertex_depth = gm::VertexDepthConvention::ZeroToOne;
            else throw std::runtime_error("Unsupported vertex depth convention");
        } else if (option == "--fragment-position") {
            if (result.emission.fragment_position != gm::FragmentPositionConvention::Unspecified)
                throw std::runtime_error("Duplicate --fragment-position");
            if (value != "reciprocal-clip-w")
                throw std::runtime_error("Unsupported fragment position convention");
            result.emission.fragment_position =
                gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        } else if (option == "--compute-local-size") {
            if (result.compute_local_size) throw std::runtime_error("Duplicate --compute-local-size");
            const auto first = value.find(','), second = value.find(',', first == std::string_view::npos ? 0 : first + 1);
            if (first == std::string_view::npos || second == std::string_view::npos)
                throw std::runtime_error("Expected compute local size X,Y,Z");
            result.compute_local_size = pl::ComputeLocalSize{{number(value.substr(0, first), 1024),
                number(value.substr(first + 1, second - first - 1), 1024), number(value.substr(second + 1), 64)}};
        } else if (option == "--compute-shared-memory") {
            if (result.compute_shared_memory_bytes) throw std::runtime_error("Duplicate --compute-shared-memory");
            result.compute_shared_memory_bytes = number(value, UINT32_MAX);
        } else if (option == "--texture-handle-cbuf") {
            if (result.texture_handle_cbuf)
                throw std::runtime_error("Duplicate --texture-handle-cbuf");
            result.texture_handle_cbuf = static_cast<std::uint8_t>(number(value, 17));
        } else if (option == "--texture-descriptors") {
            if (result.texture_descriptors_derived)
                throw std::runtime_error("Duplicate --texture-descriptors");
            if (value != "derived")
                throw std::runtime_error("Unsupported texture descriptor source");
            result.texture_descriptors_derived = true;
        } else {
            throw std::runtime_error("Unknown option: " + std::string(option));
        }
    }
    if (result.control.empty() || result.code.empty() || result.output.empty())
        throw std::runtime_error("--control, --code and --output are required");
    std::sort(result.emission.vertex_attributes.begin(), result.emission.vertex_attributes.end(),
              [](const auto& a, const auto& b) {
                  if (a.generic_slot != b.generic_slot) return a.generic_slot < b.generic_slot;
                  return a.attribute_index < b.attribute_index;
              });
    return result;
}

bool inside(const fs::path& child, const fs::path& parent) {
    const auto matched = std::mismatch(parent.begin(), parent.end(), child.begin(), child.end());
    return matched.first == parent.end();
}

fs::path validate_output(const Arguments& args) {
    const auto output = fs::weakly_canonical(fs::absolute(args.output));
    for (auto parent = output;; parent = parent.parent_path()) {
        std::error_code ec;
        const auto marker = fs::symlink_status(parent / ".git", ec);
        if (ec && ec != std::errc::no_such_file_or_directory)
            throw fs::filesystem_error("Cannot inspect Git boundary", parent, ec);
        if (marker.type() != fs::file_type::not_found && marker.type() != fs::file_type::none)
            throw std::runtime_error("Shader outputs must be outside every Git work tree");
        if (parent == parent.parent_path()) break;
    }
    for (const auto& path : {args.control, args.code}) {
        const auto source = fs::canonical(path);
        const auto spelled_parent = fs::weakly_canonical(fs::absolute(path).parent_path());
        if (inside(output, source.parent_path()) || inside(output, spelled_parent))
            throw std::runtime_error("Shader output must not be inside an input directory");
    }
    if (fs::exists(output) && (!fs::is_directory(output) || !fs::is_empty(output)))
        throw std::runtime_error("Shader output must be a new or empty directory");
    return output;
}

std::vector<std::byte> read_file(const fs::path& path, std::size_t limit,
                                  std::string_view label) {
    if (!fs::is_regular_file(path)) throw std::runtime_error(std::string(label) + " is not a regular file");
    const auto size = fs::file_size(path);
    if (size > limit) throw std::runtime_error(std::string(label) + " exceeds the input size limit");
    std::vector<std::byte> result(static_cast<std::size_t>(size));
    std::ifstream file(path, std::ios::binary);
    if (!file || (!result.empty() &&
        !file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(size))) ||
        file.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Cannot read stable " + std::string(label) + " input");
    return result;
}

si::ByteSpan view(Bytes bytes) {
    return {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
}

Bytes text_bytes(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

std::string digest(Bytes bytes) {
    const auto hash = armigo::loader::detail::sha256(bytes);
    if (!hash) throw std::runtime_error("System SHA-256 failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : *hash) {
        const auto value = std::to_integer<unsigned>(byte);
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}

const char* depth_name(gm::VertexDepthConvention depth) {
    switch (depth) {
    case gm::VertexDepthConvention::NegativeOneToOne: return "minus-one-to-one";
    case gm::VertexDepthConvention::ZeroToOne: return "zero-to-one";
    case gm::VertexDepthConvention::Unspecified: return "unspecified";
    }
    throw std::runtime_error("Invalid emitted depth convention");
}

const char* interpolation_name(gx::Interpolation mode) {
    switch (mode) {
    case gx::Interpolation::Constant: return "constant";
    case gx::Interpolation::Perspective: return "perspective";
    case gx::Interpolation::ScreenLinear: return "screen-linear";
    }
    throw std::runtime_error("Invalid emitted interpolation mode");
}

std::string reflection(const gm::EmittedStage& stage, const gm::EmitOptions& options,
                       Bytes program) {
    std::ostringstream out;
    out << "{\"schema\":\"armigo-shader-stage-v1\",\"source_version\":"
        << quote(armigo::shader::source_version)
        << ",\"program_key\":" << quote(digest(program))
        << ",\"program_size\":" << program.size()
        << ",\"program_file\":\"program.bin\",\"source_file\":\"source.metal\""
        << ",\"source_digest\":" << quote(digest(text_bytes(stage.source)))
        << ",\"stage\":" << quote(stage.stage == si::Stage::Vertex ? "vertex" : "fragment")
        << ",\"entry_point\":" << quote(stage.entry_point)
        << ",\"vertex_depth\":" << quote(depth_name(stage.vertex_depth))
        << ",\"fragment_position\":"
        << quote(options.fragment_position == gm::FragmentPositionConvention::Unspecified
                     ? "unspecified" : "reciprocal-clip-w")
        << ",\"vertex_attributes\":[";
    bool first = true;
    for (const auto& item : options.vertex_attributes) {
        if (!first) out << ',';
        first = false;
        out << "{\"generic_slot\":" << unsigned(item.generic_slot)
            << ",\"attribute_index\":" << unsigned(item.attribute_index)
            << ",\"format\":\"float32x3\"}";
    }
    out << "],\"constant_buffers\":[";
    first = true;
    for (const auto& item : stage.constant_buffers) {
        if (!first) out << ',';
        first = false;
        out << "{\"guest_binding\":" << unsigned(item.guest_binding)
            << ",\"minimum_size\":" << item.minimum_size
            << ",\"metal_buffer_index\":" << unsigned(item.metal_buffer_index) << '}';
    }
    out << "],\"textures\":[";
    first = true;
    for (const auto& item : stage.textures) {
        if (!first) out << ',';
        first = false;
        out << "{\"handle_byte_offset\":" << item.handle_byte_offset
            << ",\"texture_index\":" << unsigned(item.texture_index)
            << ",\"sampler_index\":" << unsigned(item.sampler_index) << '}';
    }
    out << "],\"interface\":{\"position_input_mask\":" << unsigned(stage.interface.position_input_mask)
        << ",\"position_output_mask\":" << unsigned(stage.interface.position_output_mask)
        << ",\"color_output_mask\":" << stage.interface.color_output_mask;
    const auto generic = [&](std::string_view name, const auto& items) {
        out << ',' << quote(name) << ":[";
        bool separator = false;
        for (const auto& item : items) {
            if (separator) out << ',';
            separator = true;
            out << "{\"slot\":" << unsigned(item.slot)
                << ",\"component_mask\":" << unsigned(item.component_mask)
                << ",\"interpolation\":" << quote(interpolation_name(item.interpolation)) << '}';
        }
        out << ']';
    };
    generic("generic_inputs", stage.interface.generic_inputs);
    generic("generic_outputs", stage.interface.generic_outputs);
    out << "}}\n";
    return out.str();
}

void write_file(const fs::path& path, Bytes data) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(reinterpret_cast<const char*>(data.data()),
                             static_cast<std::streamsize>(data.size())))
        throw std::runtime_error("Cannot write translated stage file");
    file.close();
    if (!file) throw std::runtime_error("Cannot finish translated stage file");
}

void publish(const Arguments& args, const fs::path& output, Bytes source,
             std::string_view metadata, Bytes program) {
    fs::create_directories(output.parent_path());
    fs::path temporary;
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (unsigned attempt = 0; attempt != 128; ++attempt) {
        const auto candidate = output.parent_path() /
            (".armigo-stage-" + std::to_string(nonce) + '-' + std::to_string(attempt));
        if (fs::create_directory(candidate)) { temporary = candidate; break; }
    }
    if (temporary.empty()) throw std::runtime_error("Cannot reserve translation staging directory");
    try {
        write_file(temporary / "source.metal", source);
        write_file(temporary / "reflection.json", text_bytes(metadata));
        write_file(temporary / "program.bin", program);
        if (validate_output(args) != output)
            throw std::runtime_error("Output boundary changed during translation");
        if (fs::exists(output) && !fs::remove(output))
            throw std::runtime_error("Cannot replace empty output directory");
        fs::rename(temporary, output);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(temporary, ignored);
        throw;
    }
}

[[noreturn]] void decode_failure(const mx::DecodeError& error) {
    throw std::runtime_error("decode code=" + std::to_string(unsigned(error.code)) +
        " pc=" + std::to_string(error.pc) + " field=" +
        std::to_string(unsigned(error.field)) + " value=" + std::to_string(error.value) +
        " expected=" + std::to_string(error.expected) + ": " + error.message());
}


void translate_ported(const Arguments& args) {
    const auto output = validate_output(args);
    const auto control = read_file(args.control, 64u * 1024u, "control");
    const auto code = read_file(args.code, 512u * 1024u * 1024u, "code");
    armigo::shader::translation::StageOptions options;
    options.emission = args.emission;
    options.texture_handle_cbuf = args.texture_handle_cbuf;
    options.texture_descriptors_derived = args.texture_descriptors_derived;
    if (args.compute_local_size) options.compute_local_size = args.compute_local_size->dimensions;
    options.compute_shared_memory_bytes = args.compute_shared_memory_bytes;
    const auto stage = armigo::shader::translation::translate_stage(control, code, options);
    publish(args, output, text_bytes(stage.source), stage.reflection, stage.program);
}


gx::LowerResult lower_shader(si::ByteSpan control, si::ByteSpan code) {
    auto decoded = mx::decode(control, code);
    if (const auto* program = std::get_if<mx::DecodedProgram>(&decoded)) {
        return gx::lower(*program);
    }
    const auto& error = std::get<mx::DecodeError>(decoded);
    if (error.code != mx::DecodeError::Code::UnsupportedPredicate &&
        error.code != mx::DecodeError::Code::UnsupportedControlFlow) {
        decode_failure(error);
    }
    auto forward = mx::decode_forward_program(control, code);
    if (const auto* program = std::get_if<mx::ForwardProgram>(&forward)) {
        return gx::lower_forward(*program);
    }
    decode_failure(std::get<mx::DecodeError>(forward));
}

void translate(const Arguments& args) {
    if (args.translator != "ported" && (args.compute_local_size || args.compute_shared_memory_bytes))
        throw std::runtime_error("Compute launch options require --translator ported");
    if (args.translator != "ported" && args.texture_handle_cbuf)
        throw std::runtime_error("--texture-handle-cbuf requires --translator ported");
    if (args.translator != "ported" && args.texture_descriptors_derived)
        throw std::runtime_error("--texture-descriptors requires --translator ported");
    if (args.translator == "ported") return translate_ported(args);
    const auto output = validate_output(args);
    const auto control = read_file(args.control, 64u * 1024u, "control");
    const auto code = read_file(args.code, 512u * 1024u * 1024u, "code");
    auto parsed = si::parse_shader({view(control), view(code), {}});
    if (const auto* error = std::get_if<si::ParseError>(&parsed))
        throw std::runtime_error("parse code=" + std::to_string(unsigned(error->code)) +
            " pc=0 field=container value=" + std::to_string(error->value) +
            " bound=" + std::to_string(error->bound) + ": " + error->message());
    const auto& shader = std::get<si::ParsedShader>(parsed);
    if (shader.stage != si::Stage::Vertex && shader.stage != si::Stage::Fragment)
        throw std::runtime_error("stage code=unsupported pc=0 field=stage");
    if (shader.stage == si::Stage::Vertex &&
        args.emission.fragment_position != gm::FragmentPositionConvention::Unspecified)
        throw std::runtime_error("--fragment-position is not a vertex-stage option");
    if (shader.stage == si::Stage::Fragment &&
        (!args.emission.vertex_attributes.empty() ||
         args.emission.vertex_depth != gm::VertexDepthConvention::Unspecified))
        throw std::runtime_error("Vertex options are not fragment-stage options");
    auto lowered = lower_shader(view(control), view(code));
    if (const auto* error = std::get_if<gx::Error>(&lowered))
        throw std::runtime_error("lower code=" + std::to_string(unsigned(error->code)) +
            " pc=" + std::to_string(error->pc) + " field=detail value=" +
            std::to_string(error->detail) + ": " + error->message());
    const auto& program = std::get<gx::Program>(lowered);
    auto emitted = gm::emit(program, args.emission);
    if (const auto* error = std::get_if<gm::EmitError>(&emitted)) {
        const auto pc = error->node < program.nodes().size() ? program.nodes()[error->node].pc : 0;
        throw std::runtime_error("emit code=" + std::to_string(unsigned(error->code)) +
            " pc=" + std::to_string(pc) + " field=detail value=" +
            std::to_string(error->detail) + " node=" + std::to_string(error->node) +
            ": " + error->message());
    }
    const auto& stage = std::get<gm::EmittedStage>(emitted);
    const auto identity = Bytes{code}.subspan(shader.code_offset,
                                             shader.c1_offset - shader.code_offset);
    const auto metadata = reflection(stage, args.emission, identity);
    publish(args, output, text_bytes(stage.source), metadata, identity);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--version") {
            std::cout << armigo::shader::source_version << '\n';
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << usage;
            return 0;
        }
        translate(arguments(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        armigo::diag::error("armigo-shader-translate: %s\n", error.what());
        return 1;
    }
}
