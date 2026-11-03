// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
//
// Independently authored synthetic pixel oracles for expanded persistent
// Metal engine state. The fixture compiles its shaders offline and supplies
// only synthetic guest memory through the public engine-backend contract.

#include <armigo/graphics/engines.h>
#include <armigo/graphics/texture_layout.h>
#include <armigo/platform/metal_engine_renderer.h>
#include <armigo/shader/graphics_msl.h>

#include "pipeline/phases.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <variant>
#include <vector>

namespace {

using namespace armigo::graphics;
using namespace armigo::platform;
namespace gm = armigo::shader::graphics_msl;
namespace sp = armigo::shader::pipeline;

unsigned failures = 0;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);             \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

constexpr std::uint32_t kWidth = 8;
constexpr std::uint32_t kHeight = 8;
constexpr std::uint64_t kGpuBase = 0x100000;
constexpr std::uint64_t kGuestBase = 0x800000;
constexpr std::size_t kBackingSize = 0x20000;
constexpr std::uint64_t kIndexAddress = kGpuBase + 0x1000;
constexpr std::uint64_t kInstanceCbufAddress = kGpuBase + 0x1800;
constexpr std::uint64_t kInstanceStorageAddress = kGpuBase + 0x2000;
constexpr std::uint64_t kInstanceAttributeAddress = kGpuBase + 0x3000;
constexpr std::uint64_t kInstancePositionAddress = kGpuBase + 0x4000;
constexpr std::uint64_t kColor0Address = kGpuBase + 0x10000;
constexpr std::uint64_t kColor1Address = kGpuBase + 0x11000;
constexpr std::uint64_t kColor2Address = kGpuBase + 0x12000;
constexpr std::uint64_t kDepthAddress = kGpuBase + 0x14000;

constexpr std::uint32_t pack_halves(std::uint32_t low,
                                    std::uint32_t high) noexcept {
  return low | (high << 16);
}

constexpr std::uint32_t word_of(float value) noexcept {
  return std::bit_cast<std::uint32_t>(value);
}

constexpr std::uint32_t
rt_control(std::uint32_t count,
           const std::array<std::uint32_t, 8> &permutation) noexcept {
  std::uint32_t packed = count & 0xf;
  for (std::uint32_t index = 0; index < permutation.size(); ++index)
    packed |= (permutation[index] & 7) << (4 + index * 3);
  return packed;
}

struct Backing {
  std::vector<std::byte> bytes =
      std::vector<std::byte>(kBackingSize, std::byte{0x29});
  unsigned writes = 0;

  static bool read(void *raw, std::uint64_t address,
                   std::span<std::byte> output) noexcept {
    auto &backing = *static_cast<Backing *>(raw);
    if (address < kGuestBase || address - kGuestBase > backing.bytes.size() ||
        output.size() > backing.bytes.size() - (address - kGuestBase))
      return false;
    std::memcpy(output.data(), backing.bytes.data() + address - kGuestBase,
                output.size());
    return true;
  }

  static bool write(void *raw, std::uint64_t address,
                    std::span<const std::byte> input) noexcept {
    auto &backing = *static_cast<Backing *>(raw);
    if (address < kGuestBase || address - kGuestBase > backing.bytes.size() ||
        input.size() > backing.bytes.size() - (address - kGuestBase))
      return false;
    ++backing.writes;
    std::memcpy(backing.bytes.data() + address - kGuestBase, input.data(),
                input.size());
    return true;
  }

  GpuAddressSpace memory(const std::shared_ptr<Backing> &owner) {
    const Mapping mapping{kGpuBase,     bytes.size(), 0,    kGuestBase,
                          bytes.size(), 0x1000,       owner};
    auto made = make_address_space(std::span{&mapping, 1}, {read, write, this});
    CHECK(std::holds_alternative<GpuAddressSpace>(made));
    return std::get<GpuAddressSpace>(std::move(made));
  }
};

std::vector<std::byte> compile_msl(const std::string &source,
                                   bool translated = false);

std::vector<std::byte> compile_metallib() {
  static constexpr const char *source = R"metal(
#include <metal_stdlib>
using namespace metal;

struct VOut { float4 position [[position]]; };
constant float2 k_full[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
constant float2 k_faces[6] = {
    float2(-1.0, -1.0), float2(0.0, -1.0), float2(-1.0, 1.0),
    float2(0.0, -1.0), float2(1.0, 1.0), float2(1.0, -1.0)};

vertex VOut vs_z025(uint id [[vertex_id]]) { return {float4(k_full[id % 3], 0.25, 1.0)}; }
vertex VOut vs_z075(uint id [[vertex_id]]) { return {float4(k_full[id % 3], 0.75, 1.0)}; }
vertex VOut vs_z0125(uint id [[vertex_id]]) { return {float4(k_full[id % 3], 0.125, 1.0)}; }
vertex VOut vs_z050(uint id [[vertex_id]]) { return {float4(k_full[id % 3], 0.5, 1.0)}; }
constant float k_sloped_depth[3] = {0.125, 0.625, 0.125};
vertex VOut vs_sloped_depth(uint id [[vertex_id]]) {
    return {float4(k_full[id % 3], k_sloped_depth[id % 3], 1.0)};
}
constant float k_shallow_depth[3] = {0.5, 0.5 + 0x1p-22f, 0.5};
vertex VOut vs_shallow_depth(uint id [[vertex_id]]) {
    return {float4(k_full[id % 3], k_shallow_depth[id % 3], 1.0)};
}
vertex VOut vs_shallow_depth_reversed(uint id [[vertex_id]]) {
    return {float4(k_full[id % 3], 1.0 - k_shallow_depth[id % 3], 1.0)};
}
vertex VOut vs_faces(uint id [[vertex_id]]) { return {float4(k_faces[id], 0.5, 1.0)}; }

struct ViewOut {
    float4 position [[position]];
    ushort viewport [[viewport_array_index]];
};
vertex ViewOut vs_viewports(uint id [[vertex_id]]) {
    return {float4(k_full[id % 3], 0.5, 1.0), ushort(id / 3)};
}

struct MrtOut {
    float4 f [[color(0)]];
    uint u [[color(1)]];
    int s [[color(2)]];
};
fragment MrtOut fs_mrt() { return {float4(0.25, 0.5, 0.75, 1.0), 0x11223344u, -37}; }
struct FloatMrtOut {
    float4 first [[color(0)]];
    float4 second [[color(1)]];
};
fragment FloatMrtOut fs_float_mrt_base() {
    return {float4(0.2, 0.4, 0.6, 0.8), float4(0.1, 0.3, 0.5, 0.7)};
}
fragment FloatMrtOut fs_float_mrt_blend() {
    return {float4(0.8, 0.2, 0.1, 0.5), float4(0.6, 0.4, 0.2, 0.25)};
}
fragment float4 fs_a() { return float4(0.2, 0.4, 0.6, 0.8); }
fragment float4 fs_b() { return float4(0.8, 0.2, 0.1, 0.5); }
fragment float4 fs_red() { return float4(1.0, 0.0, 0.0, 1.0); }
fragment float4 fs_green() { return float4(0.0, 1.0, 0.0, 1.0); }
fragment uint fs_uint() { return 0x55667788u; }

struct ColorOut {
    float4 position [[position]];
    float4 color [[user(locn0)]];
};
constant float4 k_index_color[6] = {
    float4(1.0, 0.0, 0.0, 1.0), float4(1.0, 0.0, 0.0, 1.0),
    float4(1.0, 0.0, 0.0, 1.0), float4(0.0, 1.0, 0.0, 1.0),
    float4(0.0, 1.0, 0.0, 1.0), float4(0.0, 1.0, 0.0, 1.0)};
vertex ColorOut vs_indexed(uint id [[vertex_id]]) {
    return {float4(k_faces[id], 0.5, 1.0), k_index_color[id]};
}
vertex ColorOut vs_instance(uint id [[vertex_id]], uint instance [[instance_id]]) {
    const float value = float(instance & 15u) / 15.0;
    return {float4(k_full[id % 3], 0.5, 1.0), float4(value, 1.0 - value, 0.0, 1.0)};
}
constant float2 k_fan[4] = {
    float2(-1.0, -1.0), float2(1.0, -1.0),
    float2(1.0, 1.0), float2(-1.0, 1.0)};
vertex VOut vs_fan(uint id [[vertex_id]]) { return {float4(k_fan[id], 0.5, 1.0)}; }
fragment float4 fs_color(ColorOut in [[stage_in]]) { return in.color; }
struct TranslatedInstanceInput {
    float record [[user(locn0_0), center_no_perspective]];
    float fetched [[user(locn0_1), center_no_perspective]];
    float instance [[user(locn0_2), center_no_perspective]];
    float alpha [[user(locn0_3), center_no_perspective]];
};
fragment float4 fs_translated_instance(TranslatedInstanceInput in [[stage_in]]) {
    return float4(in.record, in.fetched, in.instance, in.alpha);
}
)metal";
  return compile_msl(source);
}

std::vector<std::byte> compile_msl(const std::string &source, bool translated) {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("armigo-l11-state-" + std::to_string(::getpid()));
  std::filesystem::create_directories(directory);
  const auto source_path = directory / "state.metal";
  const auto air_path = directory / "state.air";
  const auto library_path = directory / "state.metallib";
  std::ofstream output(source_path, std::ios::binary);
  output << source;
  output.close();
  const auto run = [&](const std::string &command) {
    const int status = std::system(command.c_str());
    if (status != 0) {
      std::fprintf(stderr, "FAIL: fixture compilation failed: %s\n",
                   command.c_str());
      ++failures;
    }
    return status == 0;
  };
  const auto quote = [](const std::filesystem::path &path) {
    std::string result{"'"};
    for (const char character : path.string()) {
      if (character == '\'')
        result += "'\\''";
      else
        result += character;
    }
    return result + "'";
  };
  const std::string tools = "xcrun --sdk macosx ";
  // Constant folding can leave declared SSA temporaries unused in generated
  // stages; all other warnings remain errors for these synthetic libraries.
  const std::string warnings = translated ? "-Werror -Wno-unused-variable"
                                          : "-Werror";
  const bool compiled =
      run(tools +
          "metal -std=metal4.0 -mmacosx-version-min=26.0 " + warnings + " -c " +
          quote(source_path) + " -o " + quote(air_path)) &&
      run(tools + "metallib " + quote(air_path) + " -o " + quote(library_path));
  std::vector<std::byte> library;
  if (compiled) {
    std::ifstream input(library_path, std::ios::binary);
    const std::vector<char> data{std::istreambuf_iterator<char>(input), {}};
    const auto *begin = reinterpret_cast<const std::byte *>(data.data());
    library.assign(begin, begin + data.size());
  }
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
  return library;
}

// Independently authored instructions cross the production decoder, IR, SSA,
// legalization, resource planning and MSL emitter. The pixel tests therefore
// exercise the actual depth export and discard code used by translated stages.
sp::MslArtifact translated_depth_bias_fragment(bool discard_left) {
  struct Facts final : sp::FactSource {
    sp::Knowledge<sp::FactValue> lookup(sp::FactKey key) const override {
      if (key.kind == sp::FactKind::fragment_output_type)
        return sp::Known<sp::FactValue>{sp::IoType{sp::ScalarType::u32, 4},
            {sp::OriginKind::explicit_option, "synthetic-depth-bias"}};
      return sp::Unknown{"Synthetic fragment has no additional draw-state facts"};
    }
  } source;
  auto take = []<class T>(sp::Result<T> result) -> T {
    if (const auto *error = std::get_if<sp::Diagnostic>(&result))
      throw std::runtime_error(error->message);
    return std::get<T>(std::move(result));
  };
  sp::StageImage image;
  image.stage = sp::Stage::fragment;
  const auto append = [&](std::uint64_t bits) {
    for (unsigned byte = 0; byte < 8; ++byte)
      image.bytes.push_back(std::byte((bits >> (byte * 8)) & 255));
  };
  const auto instruction = [&](std::uint64_t bits) {
    if (image.bytes.size() % 32 == 0)
      append(0);
    append(bits);
  };
  const auto literal = [&](unsigned reg, float value) {
    instruction(0x0100000000070000ull | reg |
                (std::uint64_t{word_of(value)} << 20));
  };
  if (discard_left) {
    // ALD R4, PositionX; FSETP.LT.AND P0, PT, R4, R5, PT; @P0 KIL.
    instruction(0xefd800000007ff04ull | (std::uint64_t{0x70} << 20));
    literal(5, 4.0f);
    instruction(0x5bb0000000070007ull | (std::uint64_t{4} << 8) |
                (std::uint64_t{5} << 20) | (std::uint64_t{7} << 39) |
                (std::uint64_t{1} << 48));
    instruction(0xe33000000000000full);
  }
  for (unsigned lane = 0; lane < 4; ++lane) {
    literal(lane, lane == 0 || lane == 3 ? 1.0f : 0.0f);
    instruction(0xeff000000007ff00ull | lane |
                (std::uint64_t{0x80 + lane * 4} << 20));
  }
  instruction(0xe30000000007000full);
  sp::FactReader facts(source);
  const sp::MetalProfile profile{
      "synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
  auto decoded = take(sp::decode(image, facts, {}));
  auto registers = take(sp::emit_ir(decoded, facts, {}));
  auto ssa = take(sp::construct_ssa(std::move(registers), {}));
  auto legal = take(sp::legalize(std::move(ssa), facts, profile));
  auto structured = take(sp::structure(std::move(legal), {}));
  auto plan = take(sp::plan_resources(structured, profile));
  sp::EmitOptions options;
  options.fragment_output_types_from_render_targets = true;
  options.emulate_depth_bias = true;
  options.fragment_position =
      sp::FragmentPositionConvention::metal_position_w_matches_guest_position_w;
  return take(sp::emit_msl(structured, plan, options));
}

// This independently authored point shader crosses every production
// translation phase. It uses the guest-local InstanceId to select a storage
// record while a class-reflected vertex input observes the native absolute
// instance fetch selected by the draw planner.
sp::MslArtifact translated_instance_vertex() {
  struct Facts final : sp::FactSource {
    sp::Knowledge<sp::FactValue> lookup(sp::FactKey key) const override {
      if (key == sp::FactKey{sp::FactKind::depth_convention})
        return sp::Known<sp::FactValue>{sp::DepthConvention::negative_one_to_one,
            {sp::OriginKind::explicit_option, "synthetic-instance-regression"}};
      return sp::Unknown{"Synthetic instance shader keeps draw inputs dynamic"};
    }
  } source;
  const auto take = []<class T>(sp::Result<T> result) -> T {
    if (const auto *error = std::get_if<sp::Diagnostic>(&result))
      throw std::runtime_error(error->message);
    return std::get<T>(std::move(result));
  };
  const auto opcode = [](std::string_view bits) {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < bits.size(); ++index)
      if (bits[index] == '1')
        result |= std::uint64_t{1} << (63 - index);
    return result;
  };
  sp::StageImage image;
  image.stage = sp::Stage::vertex;
  const auto append = [&](std::uint64_t bits) {
    for (unsigned byte = 0; byte < 8; ++byte)
      image.bytes.push_back(std::byte((bits >> (byte * 8)) & 255));
  };
  const auto instruction = [&](std::uint64_t bits) {
    if (image.bytes.size() % 32 == 0)
      append(0);
    append(bits);
  };
  const auto literal = [&](unsigned reg, float value) {
    instruction(0x0100000000070000ull | reg |
                (std::uint64_t{word_of(value)} << 20));
  };
  const auto attribute_load = [&](unsigned reg, unsigned byte_offset) {
    instruction(0xefd800000007ff00ull | reg |
                (std::uint64_t{byte_offset} << 20));
  };
  const auto attribute_store = [&](unsigned reg, unsigned byte_offset,
                                   unsigned words = 1) {
    instruction(0xeff000000007ff00ull | reg |
                (std::uint64_t{byte_offset} << 20) |
                (std::uint64_t{words - 1} << 47));
  };
  const auto constant = [&](unsigned reg, unsigned byte_offset) {
    instruction(opcode("1110111110010") | reg | (255ull << 8) | (7ull << 16) |
                (std::uint64_t{byte_offset} << 20) | (2ull << 36) |
                (4ull << 48));
  };

  attribute_load(4, 0x2f8); // InstanceId.
  attribute_load(5, 0x80);  // Generic attribute zero, component X.
  constant(0, 16);
  constant(1, 20);
  // ISCADD R0, R4, R0, 2 forms descriptor_base + InstanceId * 4 while
  // retaining the constant-buffer provenance needed for storage reflection.
  instruction(opcode("0101110000011") | (std::uint64_t{4} << 8) |
              (7ull << 16) | (std::uint64_t{0} << 20) | (std::uint64_t{2} << 39));
  // LDG.E R6, [R0] reads one float record selected by guest-local InstanceId.
  instruction(opcode("1110111011010") | 6ull | (0ull << 8) | (7ull << 16) |
              (1ull << 45) | (4ull << 48));
  // I2F.F32.U32 preserves InstanceId numerically for the blue pixel channel.
  instruction(0x5cb8000000070a00ull | 7ull | (std::uint64_t{4} << 20));
  literal(14, 1.0f);
  instruction(opcode("0101110001101") | 5ull | (std::uint64_t{5} << 8) |
              (7ull << 16) | (std::uint64_t{14} << 20));
  instruction(opcode("0101110001101") | 6ull | (std::uint64_t{6} << 8) |
              (7ull << 16) | (std::uint64_t{14} << 20));
  literal(8, 1.0f);
  attribute_store(6, 0x80);
  attribute_store(5, 0x84);
  attribute_store(7, 0x88);
  attribute_store(8, 0x8c);
  for (unsigned component = 0; component < 4; ++component) {
    const unsigned reg = 9 + component;
    attribute_load(reg, 0x90 + component * 4);
    instruction(opcode("0101110001101") | reg | (std::uint64_t{reg} << 8) |
                (7ull << 16) | (std::uint64_t{14} << 20));
  }
  attribute_store(9, 0x70, 4);
  instruction(0xe30000000007000full);

  sp::FactReader facts(source);
  const sp::MetalProfile profile{
      "synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
  auto decoded = take(sp::decode(image, facts, {}));
  auto registers = take(sp::emit_ir(decoded, facts, {}));
  auto ssa = take(sp::construct_ssa(std::move(registers), {}));
  auto legal = take(sp::legalize(std::move(ssa), facts, profile));
  auto structured = take(sp::structure(std::move(legal), {}));
  auto plan = take(sp::plan_resources(structured, profile));
  return take(sp::emit_msl(structured, plan, {}));
}

std::shared_ptr<CompiledGraphicsStage>
compile_translated_instance_stage(const sp::MslArtifact &artifact) {
  auto stage = std::make_shared<CompiledGraphicsStage>();
  stage->metallib = compile_msl(artifact.source, true);
  if (stage->metallib.empty())
    throw std::runtime_error("translated instance Metal compilation failed");
  stage->stage = armigo::shader_input::Stage::Vertex;
  stage->entry_point = artifact.entry_point;
  stage->vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
  stage->fixed_point_size = artifact.reflection.fixed_point_size;
  stage->writes_point_size = artifact.reflection.fixed_point_size;
  const auto byte_index = [](std::uint32_t value, const char *label) {
    if (value > std::numeric_limits<std::uint8_t>::max())
      throw std::runtime_error(std::string(label) + " exceeds uint8_t");
    return static_cast<std::uint8_t>(value);
  };
  const auto direct_buffer_index = [&](const sp::PlannedResource &resource) {
    const auto *range = std::get_if<sp::MetalRange>(&resource.location);
    if (!range || range->space != sp::MetalNamespace::buffer ||
        range->count != 1)
      throw std::runtime_error("translated instance resource is not one direct buffer");
    return byte_index(range->first, "translated instance buffer index");
  };
  for (const auto &io : artifact.reflection.io) {
    if (io.semantic == sp::IoSemantic::position &&
        io.access == sp::Access::write) {
      stage->interface.position_output_mask |= io.mask;
    } else if (io.semantic == sp::IoSemantic::generic) {
      const gm::GenericInterface interface{
          byte_index(io.location, "translated instance IO location"), io.mask,
          armigo::shader::graphics::Interpolation::ScreenLinear};
      if (io.access == sp::Access::read) {
        stage->interface.generic_inputs.push_back(interface);
        if (!io.vertex_class)
          throw std::runtime_error(
              "translated instance input lacks a reflected vertex class");
        VertexFormatClass input_class;
        switch (*io.vertex_class) {
        case sp::VertexAttributeClass::float_class:
          input_class = VertexFormatClass::float_class;
          break;
        case sp::VertexAttributeClass::uint_class:
          input_class = VertexFormatClass::uint_class;
          break;
        case sp::VertexAttributeClass::sint_class:
          input_class = VertexFormatClass::sint_class;
          break;
        }
        stage->vertex_class_inputs.push_back({interface.slot, input_class});
      } else if (io.access == sp::Access::write) {
        stage->interface.generic_outputs.push_back(interface);
      }
    }
  }
  for (const auto &resource : artifact.reflection.resources) {
    const auto *buffer = std::get_if<sp::BufferUse>(&resource.use);
    if (!buffer)
      throw std::runtime_error("translated instance unexpectedly reflects a non-buffer");
    const auto metal_index = direct_buffer_index(resource);
    if (const auto *constant =
            std::get_if<sp::ConstantBufferSource>(&buffer->source)) {
      if (buffer->size.bytes > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("translated instance constant window is too large");
      stage->constant_buffer_windows.push_back(
          {constant->slot, static_cast<std::uint32_t>(buffer->size.bytes),
           metal_index});
    } else if (const auto *support =
                   std::get_if<sp::SupportBufferSource>(&buffer->source)) {
      if (support->schema != sp::constant_buffer_support_schema ||
          support->version != sp::constant_buffer_support_version)
        throw std::runtime_error("translated instance support ABI is unexpected");
      stage->constant_buffer_support_index = metal_index;
    } else if (const auto *storage =
                   std::get_if<sp::StorageBufferSource>(&buffer->source)) {
      CompiledShaderResourceAccess access;
      switch (buffer->access) {
      case sp::Access::read:
        access = CompiledShaderResourceAccess::Read;
        break;
      case sp::Access::write:
        access = CompiledShaderResourceAccess::Write;
        break;
      case sp::Access::read_write:
        access = CompiledShaderResourceAccess::ReadWrite;
        break;
      }
      stage->storage_buffers.push_back(
          {storage->descriptor.cbuf_slot,
           storage->descriptor.handle_byte_offset, metal_index, access,
           buffer->size.alignment});
    } else {
      throw std::runtime_error("translated instance buffer source is unsupported");
    }
  }
  if (artifact.reflection.draw_support_buffer_index) {
    stage->draw_support_buffer_index = byte_index(
        *artifact.reflection.draw_support_buffer_index,
        "translated instance draw-support buffer index");
  }
  if (!stage->draw_support_buffer_index)
    throw std::runtime_error("translated instance did not reflect draw support");
  return stage;
}

enum class Program : std::uint32_t {
  Mrt,
  FloatMrtBase,
  FloatMrtBlend,
  A,
  B,
  RedNear,
  GreenFar,
  Faces,
  Viewports,
  Uint,
  Indexed,
  Instance,
  TranslatedInstance,
  Fan,
  Depth0125,
  Depth025,
  Depth050,
  Depth075,
  SlopedDepth,
  SlopedDepthNative,
  Depth025Discard,
  DepthOnly025,
  Depth075Discard,
  ShallowDepth,
  ShallowDepthDiscard,
  ShallowDepthReversed,
  ShallowDepthReversedDiscard,
  ShallowDepthNative,
  ShallowDepthReversedNative,
};

struct Fixtures {
  std::vector<std::byte> library;
  std::array<std::shared_ptr<CompiledGraphicsStage>, 2> generated_fragments;
  std::shared_ptr<CompiledGraphicsStage> translated_instance;

  CompiledGraphicsPair pair(const RenderStop &stop) const {
    const Program program = static_cast<Program>(stop.registers[0x900]);
    auto vertex = std::make_shared<CompiledGraphicsStage>();
    auto fragment = std::make_shared<CompiledGraphicsStage>();
    vertex->metallib = fragment->metallib = library;
    vertex->stage = armigo::shader_input::Stage::Vertex;
    fragment->stage = armigo::shader_input::Stage::Fragment;
    vertex->vertex_depth = gm::VertexDepthConvention::NegativeOneToOne;
    vertex->interface.position_output_mask = 15;
    vertex->entry_point = "vs_z025";
    if (program == Program::GreenFar || program == Program::Depth075 ||
        program == Program::Depth075Discard)
      vertex->entry_point = "vs_z075";
    else if (program == Program::Faces)
      vertex->entry_point = "vs_faces";
    else if (program == Program::Viewports)
      vertex->entry_point = "vs_viewports";
    else if (program == Program::Depth0125)
      vertex->entry_point = "vs_z0125";
    else if (program == Program::Depth050)
      vertex->entry_point = "vs_z050";
    else if (program == Program::SlopedDepth ||
             program == Program::SlopedDepthNative)
      vertex->entry_point = "vs_sloped_depth";
    else if (program == Program::ShallowDepth ||
             program == Program::ShallowDepthDiscard ||
             program == Program::ShallowDepthNative)
      vertex->entry_point = "vs_shallow_depth";
    else if (program == Program::ShallowDepthReversed ||
             program == Program::ShallowDepthReversedDiscard ||
             program == Program::ShallowDepthReversedNative)
      vertex->entry_point = "vs_shallow_depth_reversed";
    else if (program == Program::Indexed)
      vertex->entry_point = "vs_indexed";
    else if (program == Program::Instance)
      vertex->entry_point = "vs_instance";
    else if (program == Program::TranslatedInstance)
      vertex = translated_instance;
    else if (program == Program::Fan)
      vertex->entry_point = "vs_fan";
    const bool depth_bias_program =
        program == Program::Depth0125 || program == Program::Depth025 ||
        program == Program::DepthOnly025 ||
        program == Program::Depth050 || program == Program::Depth075 ||
        program == Program::SlopedDepth || program == Program::Depth025Discard ||
        program == Program::Depth075Discard || program == Program::ShallowDepth ||
        program == Program::ShallowDepthDiscard ||
        program == Program::ShallowDepthReversed ||
        program == Program::ShallowDepthReversedDiscard;
    fragment->entry_point =
        (program == Program::Mrt)             ? "fs_mrt"
        : (program == Program::FloatMrtBase)  ? "fs_float_mrt_base"
        : (program == Program::FloatMrtBlend) ? "fs_float_mrt_blend"
        : (program == Program::A)             ? "fs_a"
        : (program == Program::B)             ? "fs_b"
        : (program == Program::GreenFar)      ? "fs_green"
        : (program == Program::Uint)          ? "fs_uint"
        : (program == Program::TranslatedInstance) ? "fs_translated_instance"
        : (program == Program::Indexed || program == Program::Instance)
            ? "fs_color"
            : "fs_red";
    if (program == Program::Mrt) {
      fragment->color_outputs = {
          {0, 15, ColorOutputScalar::f32, 4},
          {1, 1, ColorOutputScalar::u32, 1},
          {2, 1, ColorOutputScalar::s32, 1},
      };
      fragment->interface.color_output_mask = 0x11f;
    } else if (program == Program::FloatMrtBase ||
               program == Program::FloatMrtBlend) {
      fragment->color_outputs = {
          {0, 15, ColorOutputScalar::f32, 4},
          {1, 15, ColorOutputScalar::f32, 4},
      };
      fragment->interface.color_output_mask = 0xff;
    } else if (program == Program::Uint) {
      fragment->color_outputs = {{0, 1, ColorOutputScalar::u32, 1}};
      fragment->interface.color_output_mask = 1;
    } else {
      fragment->color_outputs = {{0, 15, ColorOutputScalar::f32, 4}};
      fragment->interface.color_output_mask = 15;
    }
    if (program == Program::Indexed || program == Program::Instance ||
        program == Program::TranslatedInstance) {
      vertex->interface.generic_outputs = {
          {0, 15, armigo::shader::graphics::Interpolation::ScreenLinear}};
      fragment->interface.generic_inputs = {
          {0, 15, armigo::shader::graphics::Interpolation::ScreenLinear}};
    }
    if (depth_bias_program) {
      const bool discard_left = program == Program::Depth025Discard ||
                                program == Program::Depth075Discard ||
                                program == Program::ShallowDepthDiscard ||
                                program == Program::ShallowDepthReversedDiscard;
      fragment = generated_fragments[discard_left ? 1 : 0];
      if (program == Program::DepthOnly025) {
        fragment = std::make_shared<CompiledGraphicsStage>(*fragment);
        // Package lookup retains raw words when no attachment supplies a type.
        for (auto &output : fragment->color_outputs)
          output.scalar = ColorOutputScalar::u32;
      }
    }
    return {std::move(vertex), std::move(fragment)};
  }

  static CompiledGraphicsLookupResult lookup(void *raw, const RenderStop &stop,
                                             const GpuAddressSpace &) {
    return static_cast<Fixtures *>(raw)->pair(stop);
  }
};

RenderCallbacks make_backend(const std::shared_ptr<Fixtures> &fixtures) {
  MetalEngineOptions options;
  options.mode = MetalEngineMode::Persistent;
  options.lookup = Fixtures::lookup;
  options.lookup_context = fixtures.get();
  options.lookup_owner = fixtures;
  auto made = create_metal_engine_backend(std::move(options));
  if (const auto *error = std::get_if<RendererError>(&made)) {
    const auto failure = error->command();
    std::fprintf(stderr, "FAIL: engine backend creation: %.*s\n",
                 int(failure.detail.size()), failure.detail.data());
    ++failures;
    return {};
  }
  return std::get<RenderCallbacks>(std::move(made));
}

void set_color_target(RenderStop &stop, std::uint32_t physical,
                      std::uint64_t address, std::uint32_t format) {
  const std::size_t base = 0x200 + physical * 0x10;
  stop.registers[base] = std::uint32_t(address >> 32);
  stop.registers[base + 1] = std::uint32_t(address);
  stop.registers[base + 2] = kWidth * 4;
  stop.registers[base + 3] = kHeight;
  stop.registers[base + 4] = format;
  stop.registers[base + 5] = 0x1000;
  stop.registers[base + 6] = 1;
  stop.registers[base + 7] = 0x1000;
  stop.color_targets[physical] = {address, kWidth * 4, kHeight, format,
                                  0x1000,  1,          0x1000,  0};
}

void set_depth_target(RenderStop &stop, std::uint32_t format) {
  stop.registers[0x3f8] = std::uint32_t(kDepthAddress >> 32);
  stop.registers[0x3f9] = std::uint32_t(kDepthAddress);
  stop.registers[0x3fa] = format;
  stop.registers[0x3fb] = 0;
  stop.registers[0x3fc] = 0x1000;
  stop.registers[0x48a] = kWidth;
  stop.registers[0x48b] = kHeight;
  stop.registers[0x48c] = (1u << 16) | 1u;
  stop.registers[0x54e] = 1;
  stop.depth = {kDepthAddress, format, 0, 0x1000, kWidth, kHeight, true};
}

RenderStop base_draw(Backing &backing, Program program,
                     std::uint32_t count = 3) {
  RenderStop stop;
  stop.engine = engine_class::threed;
  stop.kind = RenderKind::Draw;
  stop.draw = DrawDescription{true, 4, 0, count, 0, 0, 1};
  stop.registers[0x900] = static_cast<std::uint32_t>(program);
  stop.registers[0x0df] = 1;
  stop.registers[0x36b] = stop.registers[0x36c] = 0x1b02;
  stop.registers[0x3e4] = 0;
  stop.registers[0x4b9] = 1;
  stop.registers[0x4c3] = 8;
  stop.registers[0x647] = 0x901;
  stop.registers[0x648] = 0x405;
  stop.registers[0x64b] = 1;
  stop.registers[0x64f] = 0x181d;
  stop.registers[0x680] = stop.registers[0x681] = stop.registers[0x682] =
      0x1111;
  stop.registers[0x3ef] = 0xffff;
  stop.registers[0x280] = word_of(4.0f);
  stop.registers[0x281] = word_of(-4.0f);
  stop.registers[0x282] = word_of(0.5f);
  stop.registers[0x283] = word_of(4.0f);
  stop.registers[0x284] = word_of(4.0f);
  stop.registers[0x285] = word_of(0.5f);
  stop.registers[0x286] = 0x6420;
  stop.registers[0x300] = stop.registers[0x301] = 8u << 16;
  stop.registers[0x302] = word_of(0.0f);
  stop.registers[0x303] = word_of(1.0f);
  stop.registers[0x380] = 1;
  stop.registers[0x381] = stop.registers[0x382] = 8u << 16;
  stop.registers[0x3fd] = stop.registers[0x3fe] = 8u << 16;
  stop.registers[0x5f3] = std::uint32_t(kIndexAddress);
  stop.registers[0x5f5] = std::uint32_t(kIndexAddress + count * 2 - 1);
  stop.registers[0x5f6] = 1;
  stop.programs[1].enabled = stop.programs[5].enabled = true;
  const std::array<std::uint16_t, 6> indices{0, 1, 2, 3, 4, 5};
  std::memcpy(backing.bytes.data() + (kIndexAddress - kGpuBase), indices.data(),
              count * sizeof(indices[0]));
  return stop;
}

struct FrontendCompletion {
  RenderCallbacks rendering;
  unsigned increments = 0;
  std::uint32_t last_syncpoint = 0;

  static bool increment(void *raw, std::uint32_t syncpoint) noexcept {
    auto &self = *static_cast<FrontendCompletion *>(raw);
    ++self.increments;
    self.last_syncpoint = syncpoint;
    return true;
  }

  static bool expired(void *, std::uint32_t, std::uint32_t) noexcept {
    return true;
  }

  static std::uint64_t ticks(void *) noexcept { return 0x1122334455667788ull; }

  CompletionCallbacks callbacks() noexcept {
    CompletionCallbacks result;
    result.increment = increment;
    result.expired = expired;
    result.ticks = ticks;
    result.context = this;
    result.rendering = rendering;
    return result;
  }
};

CommandResult frontend_call(EngineFrontEnd &frontend,
                            const GpuAddressSpace &memory,
                            FrontendCompletion &completion,
                            std::uint32_t method, std::uint32_t argument,
                            std::uint32_t engine = engine_class::threed) {
  return frontend.call({method, 0, argument, 0x4000 + method * 4, true}, engine,
                       memory, completion.callbacks());
}

void configure_frontend(EngineFrontEnd &frontend, const RenderStop &stop,
                        const GpuAddressSpace &memory,
                        FrontendCompletion &completion) {
  const auto write = [&](std::uint32_t method) {
    const auto result = frontend_call(frontend, memory, completion, method,
                                      stop.registers[method]);
    if (!result.ok()) {
      std::fprintf(stderr, "FAIL: frontend setup method %x: %.*s\n", method,
                   int(result.detail.size()), result.detail.data());
      ++failures;
    }
  };
  for (const std::uint32_t method :
       {0x0dfu, 0x280u, 0x281u, 0x282u, 0x283u, 0x284u, 0x285u, 0x286u,
        0x300u, 0x301u, 0x302u, 0x303u, 0x35fu, 0x36bu, 0x36cu, 0x380u,
        0x381u, 0x382u, 0x3e4u, 0x3efu, 0x3fdu, 0x3feu, 0x487u, 0x4b3u,
        0x4b9u, 0x4bau, 0x4c3u, 0x4d5u, 0x54du, 0x54fu, 0x574u, 0x646u,
        0x647u, 0x648u, 0x649u, 0x64bu, 0x64fu, 0x671u, 0x680u, 0x681u,
        0x682u, 0x683u, 0x684u, 0x685u, 0x686u, 0x687u, 0x900u})
    write(method);
  for (std::uint32_t method = 0x200; method <= 0x208; ++method)
    write(method);
  CHECK(frontend_call(frontend, memory, completion, 0x850, 1).ok());
}

bool submit(const RenderCallbacks &backend, const RenderStop &stop,
            const GpuAddressSpace &memory, const char *label,
            bool clear = false) {
  const auto result = clear ? backend.clear(backend.context, stop, memory)
                            : backend.draw(backend.context, stop, memory);
  if (!result.ok()) {
    std::fprintf(stderr, "FAIL: %s: %.*s\n", label, int(result.detail.size()),
                 result.detail.data());
    ++failures;
    return false;
  }
  return true;
}

std::span<std::byte> surface(Backing &backing, std::uint64_t address,
                             std::size_t size = kWidth * kHeight * 4) {
  return std::span{backing.bytes}.subspan(address - kGpuBase, size);
}

std::array<unsigned, 4> rgba8(const Backing &backing, std::uint64_t address,
                              unsigned x, unsigned y) {
  const std::size_t offset = address - kGpuBase + (y * kWidth + x) * 4;
  return {std::to_integer<unsigned>(backing.bytes[offset]),
          std::to_integer<unsigned>(backing.bytes[offset + 1]),
          std::to_integer<unsigned>(backing.bytes[offset + 2]),
          std::to_integer<unsigned>(backing.bytes[offset + 3])};
}

std::uint32_t u32_at(const Backing &backing, std::uint64_t address,
                     unsigned x = 4, unsigned y = 4) {
  std::uint32_t value = 0;
  const std::size_t offset = address - kGpuBase + (y * kWidth + x) * 4;
  std::memcpy(&value, backing.bytes.data() + offset, sizeof(value));
  return value;
}

TextureLayoutDescription depth_layout(std::uint32_t encoded_format) {
  TextureLayoutDescription description;
  description.width = kWidth;
  description.height = kHeight;
  description.depth = description.levels = description.layers = 1;
  description.bytes_per_block =
      encoded_format == 0x19 ? 8 : encoded_format == 0x13 ? 2 : 4;
  description.storage = TextureStorageLayout::BlockLinear;
  auto result = calculate_texture_size(description);
  CHECK(std::holds_alternative<TextureLayoutDescription>(result));
  return std::get<TextureLayoutDescription>(std::move(result));
}

void initialize_depth(Backing &backing, std::uint32_t format, float depth,
                      std::byte stencil_or_padding) {
  const auto layout = depth_layout(format);
  std::vector<std::byte> linear(layout.mips[0].linear_bytes, std::byte{0});
  const std::uint32_t packed_d24 =
      static_cast<std::uint32_t>(std::llround(double(depth) * 16777215.0));
  const std::uint16_t packed_d16 =
      static_cast<std::uint16_t>(std::llround(double(depth) * 65535.0));
  for (std::size_t texel = 0; texel < kWidth * kHeight; ++texel) {
    if (format == 0x19) {
      std::memcpy(linear.data() + texel * 8, &depth, sizeof(depth));
      linear[texel * 8 + 4] = stencil_or_padding;
      linear[texel * 8 + 5] = std::byte{0xa1};
      linear[texel * 8 + 6] = std::byte{0xb2};
      linear[texel * 8 + 7] = std::byte{0xc3};
    } else if (format == 0x0a) {
      std::memcpy(linear.data() + texel * 4, &depth, sizeof(depth));
    } else if (format == 0x13) {
      std::memcpy(linear.data() + texel * 2, &packed_d16,
                  sizeof(packed_d16));
    } else {
      linear[texel * 4] = stencil_or_padding;
      linear[texel * 4 + 1] = std::byte(packed_d24 & 0xff);
      linear[texel * 4 + 2] = std::byte((packed_d24 >> 8) & 0xff);
      linear[texel * 4 + 3] = std::byte((packed_d24 >> 16) & 0xff);
    }
  }
  auto guest = surface(backing, kDepthAddress, layout.total_size);
  CHECK(std::holds_alternative<std::monostate>(
      write_level_from_linear(layout, linear, guest, 0, 0)));
}

TextureLevelData read_depth(const Backing &backing, std::uint32_t format) {
  const auto layout = depth_layout(format);
  const auto guest = std::span{backing.bytes}.subspan(kDepthAddress - kGpuBase,
                                                      layout.total_size);
  auto result = convert_level_to_linear(layout, guest, 0, 0);
  CHECK(std::holds_alternative<TextureLevelData>(result));
  return std::get<TextureLevelData>(std::move(result));
}

float depth_value(const TextureLevelData &data, std::uint32_t format,
                  unsigned x = 4, unsigned y = 4) {
  const std::size_t texel = y * kWidth + x;
  if (format == 0x19) {
    float value = 0;
    std::memcpy(&value, data.bytes.data() + texel * 8, sizeof(value));
    return value;
  }
  if (format == 0x0a) {
    float value = 0;
    std::memcpy(&value, data.bytes.data() + texel * 4, sizeof(value));
    return value;
  }
  if (format == 0x13) {
    std::uint16_t packed = 0;
    std::memcpy(&packed, data.bytes.data() + texel * 2, sizeof(packed));
    return float(packed) / 65535.0f;
  }
  const auto *pixel = data.bytes.data() + texel * 4;
  const std::uint32_t packed = std::to_integer<unsigned>(pixel[1]) |
                               (std::to_integer<unsigned>(pixel[2]) << 8) |
                               (std::to_integer<unsigned>(pixel[3]) << 16);
  return float(packed) / 16777215.0f;
}

bool expected_depth(const TextureLevelData &data, std::uint32_t format,
                    unsigned x, unsigned y, float expected) {
  const float actual = depth_value(data, format, x, y);
  if (format == 0x19 || format == 0x0a)
    return actual == expected;
  if (format == 0x13) {
    const auto actual_d16 = std::llround(double(actual) * 65535.0);
    const auto expected_d16 = std::llround(double(expected) * 65535.0);
    return std::llabs(actual_d16 - expected_d16) <= 1;
  }
  const auto actual_d24 = std::llround(double(actual) * 16777215.0);
  const auto expected_d24 = std::llround(double(expected) * 16777215.0);
  return std::llabs(actual_d24 - expected_d24) <= 1;
}

float depth_bias_resolution(std::uint32_t format, float depth) {
  if (format == 0x13)
    return std::ldexp(1.0f, -16);
  if (format == 0x14 || format == 0x15 || format == 0x16)
    return std::ldexp(1.0f, -24);
  // DEPTH_BIAS_CONTROL=0 selects the force-UNORM D32 increment.
  static_cast<void>(depth);
  return std::ldexp(1.0f, -24);
}

// This oracle states the guest contract directly. It deliberately does not
// share the renderer's conversion or native Metal depth-bias parameters.
float biased_depth(std::uint32_t format, float depth, float slope,
                   float factor, float units, float clamp) {
  float bias = units * depth_bias_resolution(format, depth) + factor * slope;
  if (clamp > 0.0f)
    bias = std::min(bias, clamp);
  else if (clamp < 0.0f)
    bias = std::max(bias, clamp);
  return std::clamp(depth + bias, 0.0f, 1.0f);
}

bool expected_biased_depth(const TextureLevelData &data, std::uint32_t format,
                           unsigned x, unsigned y, float expected) {
  if (format != 0x0a && format != 0x19)
    return expected_depth(data, format, x, y, expected);
  const float actual = depth_value(data, format, x, y);
  const float ulp = std::nextafter(expected, 1.0f) - expected;
  return std::abs(actual - expected) <= ulp * 2.0f;
}

bool expected_constant_biased_depth(const TextureLevelData &data,
                                    std::uint32_t format, unsigned x,
                                    unsigned y, float expected) {
  if (format == 0x13) {
    const auto actual =
        std::llround(double(depth_value(data, format, x, y)) * 65535.0);
    const auto wanted = std::llround(double(expected) * 65535.0);
    return actual == wanted;
  }
  if (format == 0x14 || format == 0x15 || format == 0x16) {
    const auto actual =
        std::llround(double(depth_value(data, format, x, y)) * 16777215.0);
    const auto wanted = std::llround(double(expected) * 16777215.0);
    return actual == wanted;
  }
  return expected_biased_depth(data, format, x, y, expected);
}

void configure_depth_bias_draw(RenderStop &stop, std::uint32_t format,
                               float factor, float units, float clamp,
                               bool enabled = true) {
  stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
  set_color_target(stop, 0, kColor0Address, 0xd5);
  set_depth_target(stop, format);
  stop.registers[0x4b3] = stop.registers[0x4ba] = 1;
  stop.registers[0x4c3] = 8;
  stop.registers[0x444] = 0;
  stop.registers[0x372] = enabled ? 1 : 0;
  stop.registers[0x55b] = word_of(factor);
  stop.registers[0x56f] = word_of(units);
  stop.registers[0x61f] = word_of(clamp);
}

void test_constant_depth_bias(const std::shared_ptr<Fixtures> &fixtures) {
  struct BiasCase {
    std::uint32_t format;
    Program program;
    float depth;
    float units;
  };
  const std::array cases{
      BiasCase{0x13, Program::Depth0125, 0.125f, 8.0f},
      BiasCase{0x13, Program::Depth025, 0.25f, 8.0f},
      BiasCase{0x13, Program::Depth050, 0.5f, -8.0f},
      BiasCase{0x13, Program::Depth075, 0.75f, -8.0f},
      BiasCase{0x14, Program::Depth0125, 0.125f, 8.0f},
      BiasCase{0x14, Program::Depth025, 0.25f, 8.0f},
      BiasCase{0x14, Program::Depth050, 0.5f, -8.0f},
      BiasCase{0x14, Program::Depth075, 0.75f, -8.0f},
      BiasCase{0x19, Program::Depth0125, 0.125f, 8.0f},
      BiasCase{0x19, Program::Depth025, 0.25f, 8.0f},
      BiasCase{0x19, Program::Depth050, 0.5f, -8.0f},
      BiasCase{0x19, Program::Depth075, 0.75f, -8.0f},
      BiasCase{0x0a, Program::Depth0125, 0.125f, 8.0f},
      BiasCase{0x0a, Program::Depth025, 0.25f, 8.0f},
      BiasCase{0x0a, Program::Depth050, 0.5f, -8.0f},
      BiasCase{0x0a, Program::Depth075, 0.75f, -8.0f},
  };
  for (const auto &test : cases) {
    auto backing = std::make_shared<Backing>();
    initialize_depth(*backing, test.format, 0.9f, std::byte{0x5a});
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, test.program);
    configure_depth_bias_draw(stop, test.format, 0.0f, test.units, 0.0f);
    CHECK(submit(backend, stop, memory, "constant depth bias"));
    const auto depth = read_depth(*backing, test.format);
    const float expected =
        biased_depth(test.format, test.depth, 0.0f, 0.0f, test.units, 0.0f);
    for (const unsigned x : {0u, 3u, 4u, 7u}) {
      if (!expected_constant_biased_depth(depth, test.format, x, 4,
                                          expected)) {
        std::fprintf(stderr,
                     "constant bias mismatch format=%x depth=%.9g units=%.9g "
                     "x=%u expected=%.9g actual=%.9g\n",
                     test.format, test.depth, test.units, x, expected,
                     depth_value(depth, test.format, x, 4));
        CHECK(false);
      }
    }
  }
}

float sloped_depth_at(unsigned x) {
  return 0.125f + (float(x) + 0.5f) * (0.5f / 16.0f);
}

void test_sloped_and_clamped_depth_bias(
    const std::shared_ptr<Fixtures> &fixtures) {
  constexpr float slope = 0.5f / 16.0f;
  for (const std::uint32_t format : {0x13u, 0x14u, 0x19u, 0x0au}) {
    for (const Program program :
         {Program::SlopedDepthNative, Program::SlopedDepth}) {
      for (const float factor : {2.0f, -2.0f}) {
        auto backing = std::make_shared<Backing>();
        initialize_depth(*backing, format, 0.9f, std::byte{0x5a});
        auto memory = backing->memory(backing);
        auto backend = make_backend(fixtures);
        auto stop = base_draw(*backing, program);
        configure_depth_bias_draw(stop, format, factor, 0.0f, 0.0f);
        CHECK(submit(backend, stop, memory, "slope-only depth bias"));
        const auto depth = read_depth(*backing, format);
        for (const unsigned x : {0u, 3u, 4u, 7u}) {
          const float source = sloped_depth_at(x);
          const float expected =
              biased_depth(format, source, slope, factor, 0.0f, 0.0f);
          CHECK(expected_biased_depth(depth, format, x, 4, expected));
        }
      }
    }
    {
      auto backing = std::make_shared<Backing>();
      initialize_depth(*backing, format, 0.9f, std::byte{0x5a});
      auto memory = backing->memory(backing);
      auto backend = make_backend(fixtures);
      auto stop = base_draw(*backing, Program::SlopedDepth);
      configure_depth_bias_draw(stop, format, 0.0f, 8.0f, 0.0f);
      CHECK(submit(backend, stop, memory, "sloped constant-only depth bias"));
      const auto depth = read_depth(*backing, format);
      for (const unsigned x : {0u, 3u, 4u, 7u}) {
        const float source = sloped_depth_at(x);
        const float expected =
            biased_depth(format, source, slope, 0.0f, 8.0f, 0.0f);
        CHECK(expected_constant_biased_depth(depth, format, x, 4, expected));
      }
    }
    struct ClampCase {
      float factor;
      float clamp;
    };
    for (const auto test :
         {ClampCase{4.0f, 0.02f}, ClampCase{-4.0f, -0.02f},
          ClampCase{-1.0f, 0.02f}, ClampCase{1.0f, -0.02f}}) {
      auto backing = std::make_shared<Backing>();
      initialize_depth(*backing, format, 0.9f, std::byte{0x5a});
      auto memory = backing->memory(backing);
      auto backend = make_backend(fixtures);
      auto stop = base_draw(*backing, Program::SlopedDepthNative);
      configure_depth_bias_draw(stop, format, test.factor, 0.0f, test.clamp);
      CHECK(submit(backend, stop, memory, "clamped depth bias"));
      const auto depth = read_depth(*backing, format);
      for (const unsigned x : {2u, 5u}) {
        const float source = sloped_depth_at(x);
        const float expected = biased_depth(format, source, slope, test.factor,
                                            0.0f, test.clamp);
        CHECK(expected_biased_depth(depth, format, x, 4, expected));
      }
    }
    for (const float sign : {1.0f, -1.0f}) {
      auto backing = std::make_shared<Backing>();
      initialize_depth(*backing, format, 0.9f, std::byte{0x5a});
      auto memory = backing->memory(backing);
      auto backend = make_backend(fixtures);
      auto stop = base_draw(*backing, Program::SlopedDepth);
      const float units = sign * 0.01f / depth_bias_resolution(format, 0.25f);
      const float factor = sign * 0.5f;
      const float clamp = sign * 0.02f;
      configure_depth_bias_draw(stop, format, factor, units, clamp);
      CHECK(submit(backend, stop, memory, "combined clamped depth bias"));
      const auto depth = read_depth(*backing, format);
      for (const unsigned x : {2u, 5u}) {
        const float source = sloped_depth_at(x);
        const float expected =
            biased_depth(format, source, slope, factor, units, clamp);
        CHECK(expected_biased_depth(depth, format, x, 4, expected));
      }
    }
  }
}

// Positive bias always moves toward larger numerical window depth, including
// reversed-Z projections and GREATER comparisons. A reference midway between the
// unbiased and biased values makes an incorrect sign change visible in color.
// Reversal is supplied by vertex depth (z -> 1-z); the engine still rejects
// nondefault viewport depth transforms, so these cases do not claim that support.
void test_translated_depth_bias_visibility(
    const std::shared_ptr<Fixtures> &fixtures) {
  constexpr std::uint32_t format = 0x0a;
  constexpr float quantum = 0x1p-24f;
  for (const bool reversed : {false, true}) {
    for (const bool greater : {false, true}) {
      for (const bool discard_left : {false, true}) {
        for (const float sign : {-1.0f, 1.0f}) {
          for (const bool enabled : {false, true}) {
            const float source = reversed ? 0.75f : 0.25f;
            const float reference = source + sign * 4.0f * quantum;
            const float expected = source + (enabled ? sign * 8.0f * quantum : 0.0f);
            const bool passes = greater ? expected > reference : expected < reference;
            auto backing = std::make_shared<Backing>();
            initialize_depth(*backing, format, reference, std::byte{0x5a});
            auto memory = backing->memory(backing);
            auto backend = make_backend(fixtures);
            const Program program = reversed
                ? (discard_left ? Program::Depth075Discard : Program::Depth075)
                : (discard_left ? Program::Depth025Discard : Program::Depth025);
            auto stop = base_draw(*backing, program);
            configure_depth_bias_draw(stop, format, 0.0f, sign * 8.0f, 0.0f, enabled);
            stop.registers[0x4c3] = greater ? 5 : 2;
            CHECK(submit(backend, stop, memory, "translated depth visibility"));
            const auto depth = read_depth(*backing, format);
            for (unsigned y = 0; y < kHeight; ++y) {
              for (unsigned x = 0; x < kWidth; ++x) {
                const bool visible = passes && !(discard_left && x < kWidth / 2);
                const auto color = rgba8(*backing, kColor0Address, x, y);
                const std::array<unsigned, 4> wanted = visible
                    ? std::array<unsigned, 4>{255, 0, 0, 255}
                    : std::array<unsigned, 4>{0x29, 0x29, 0x29, 0x29};
                if (color != wanted ||
                    !expected_depth(depth, format, x, y, visible ? expected : reference)) {
                  std::fprintf(stderr,
                      "translated visibility mismatch reversed=%u greater=%u discard=%u "
                      "sign=%.0f enabled=%u pixel=(%u,%u) visible=%u depth=%.9g\n",
                      reversed, greater, discard_left, sign, enabled, x, y,
                      visible, depth_value(depth, format, x, y));
                  CHECK(false);
                }
              }
            }
          }
        }
      }
    }
  }
}

// The primitive has z(x) = 0.5 + x * 2^-26 in window coordinates. Adjacent
// fragment-position values can round to the same float, so deriving slope
// from their difference can lose a factor * slope contribution of 2^-21 here.
// This is an independently specified Metal regression oracle, not a claim
// of measured equivalence with Maxwell rasterization hardware.
void test_translated_shallow_depth_bias(
    const std::shared_ptr<Fixtures> &fixtures) {
  constexpr std::uint32_t format = 0x0a;
  constexpr float slope = 0x1p-26f;
  struct BiasCase {
    float factor;
    float units;
    bool enabled;
    float reference_offset;
  };
  constexpr std::array cases{
      BiasCase{32.0f, 8.0f, true, 12.0f * 0x1p-24f},
      BiasCase{32.0f, 8.0f, false, 12.0f * 0x1p-24f},
      BiasCase{0.0f, 8.0f, true, 12.0f * 0x1p-24f},
      BiasCase{32.0f, 0.0f, true, 4.0f * 0x1p-24f},
      BiasCase{32.0f, -8.0f, true, -6.0f * 0x1p-24f},
      BiasCase{-32.0f, 8.0f, true, 6.0f * 0x1p-24f},
  };
  for (const auto &test : cases) {
    for (const bool reversed : {false, true}) {
      // Native rasterization supplies a depth readback reference with exactly
      // the same primitive and slope state. Adding the independently specified
      // UNORM quantum tests the fallback without assuming sub-ULP interpolation
      // precision. Visibility below also has a separate analytic threshold.
      auto native_backing = std::make_shared<Backing>();
      initialize_depth(*native_backing, format, 0.9f, std::byte{0x5a});
      auto native_memory = native_backing->memory(native_backing);
      auto native_backend = make_backend(fixtures);
      auto native_stop = base_draw(*native_backing, reversed
          ? Program::ShallowDepthReversedNative : Program::ShallowDepthNative);
      configure_depth_bias_draw(native_stop, format, test.factor, 0.0f, 0.0f,
                                test.enabled);
      CHECK(submit(native_backend, native_stop, native_memory,
                   "native shallow depth reference"));
      const auto native_depth = read_depth(*native_backing, format);
      for (const bool greater : {false, true}) {
        for (const bool discard_left : {false, true}) {
          auto backing = std::make_shared<Backing>();
          const float reference = 0.5f + test.reference_offset;
          initialize_depth(*backing, format, reference, std::byte{0x5a});
          auto memory = backing->memory(backing);
          auto backend = make_backend(fixtures);
          const Program program = reversed
              ? (discard_left ? Program::ShallowDepthReversedDiscard
                              : Program::ShallowDepthReversed)
              : (discard_left ? Program::ShallowDepthDiscard : Program::ShallowDepth);
          auto stop = base_draw(*backing, program);
          configure_depth_bias_draw(stop, format, test.factor, test.units, 0.0f,
                                    test.enabled);
          stop.registers[0x4c3] = greater ? 5 : 2;
          CHECK(submit(backend, stop, memory, "translated shallow depth bias"));
          const auto depth = read_depth(*backing, format);
          for (unsigned y = 0; y < kHeight; ++y) {
            for (unsigned x = 0; x < kWidth; ++x) {
              const double source = 0.5 + (reversed ? -1.0 : 1.0) *
                  (double(x) + 0.5) * double(slope);
              const float analytic = float(source + (test.enabled
                  ? double(test.factor) * slope + double(test.units) * 0x1p-24 : 0.0));
              const bool passes = greater ? analytic > reference : analytic < reference;
              const float expected = depth_value(native_depth, format, x, y) +
                  (test.enabled ? test.units * 0x1p-24f : 0.0f);
              const bool visible = passes && !(discard_left && x < kWidth / 2);
              const auto color = rgba8(*backing, kColor0Address, x, y);
              const std::array<unsigned, 4> wanted = visible
                  ? std::array<unsigned, 4>{255, 0, 0, 255}
                  : std::array<unsigned, 4>{0x29, 0x29, 0x29, 0x29};
              const bool depth_matches = visible
                  ? expected_biased_depth(depth, format, x, y, expected)
                  : expected_depth(depth, format, x, y, reference);
              if (color != wanted || !depth_matches) {
                std::fprintf(stderr,
                    "shallow bias mismatch factor=%.0f units=%.0f enabled=%u "
                    "reversed=%u greater=%u discard=%u pixel=(%u,%u) "
                    "visible=%u expected=%.9g actual=%.9g\n",
                    test.factor, test.units, test.enabled, reversed, greater,
                    discard_left, x, y, visible, visible ? expected : reference,
                    depth_value(depth, format, x, y));
                CHECK(false);
              }
            }
          }
        }
      }
    }
  }
}

void test_depth_bias_disable_restores_zero(
    const std::shared_ptr<Fixtures> &fixtures) {
  for (const std::uint32_t format : {0x13u, 0x14u, 0x19u, 0x0au}) {
    auto backing = std::make_shared<Backing>();
    initialize_depth(*backing, format, 0.9f, std::byte{0x5a});
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto biased = base_draw(*backing, Program::Depth025);
    configure_depth_bias_draw(biased, format, 0.0f, 16.0f, 0.0f);
    CHECK(submit(backend, biased, memory, "enable depth bias"));
    auto depth = read_depth(*backing, format);
    const float shifted = biased_depth(format, 0.25f, 0.0f, 0.0f, 16.0f,
                                       0.0f);
    CHECK(expected_biased_depth(depth, format, 4, 4, shifted));

    auto same_shader_unbiased = base_draw(*backing, Program::Depth025);
    configure_depth_bias_draw(same_shader_unbiased, format, 7.0f, 11.0f,
                              -0.01f, false);
    CHECK(submit(backend, same_shader_unbiased, memory,
                 "disable depth bias uniform"));
    depth = read_depth(*backing, format);
    CHECK(expected_constant_biased_depth(depth, format, 4, 4, 0.25f));
    auto pixel = rgba8(*backing, kColor0Address, 4, 4);
    CHECK(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0 &&
          pixel[3] == 255);

    auto switched_shader = base_draw(*backing, Program::B);
    configure_depth_bias_draw(switched_shader, format, 7.0f, 11.0f, -0.01f,
                              false);
    CHECK(submit(backend, switched_shader, memory,
                 "disabled depth bias shader switch"));
    depth = read_depth(*backing, format);
    CHECK(expected_constant_biased_depth(depth, format, 4, 4, 0.25f));
    pixel = rgba8(*backing, kColor0Address, 4, 4);
    CHECK(pixel[0] == 204 && pixel[1] == 51 && pixel[2] == 26);
  }
}

void test_depth_bias_color_only_variant(
    const std::shared_ptr<Fixtures> &fixtures) {
  auto backing = std::make_shared<Backing>();
  auto memory = backing->memory(backing);
  auto backend = make_backend(fixtures);
  auto color_only = base_draw(*backing, Program::Depth025);
  color_only.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
  set_color_target(color_only, 0, kColor0Address, 0xd5);
  color_only.registers[0x372] = 1;
  color_only.registers[0x55b] = word_of(2.0f);
  color_only.registers[0x56f] = word_of(8.0f);
  color_only.registers[0x61f] = word_of(0.02f);
  const auto depth_before = surface(*backing, kDepthAddress, 0x1000);
  const std::vector<std::byte> preserved_depth(depth_before.begin(),
                                                depth_before.end());
  CHECK(submit(backend, color_only, memory, "color-only depth bias variant"));
  for (unsigned y = 0; y < kHeight; ++y) {
    for (unsigned x = 0; x < kWidth; ++x) {
      const auto pixel = rgba8(*backing, kColor0Address, x, y);
      CHECK(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0 &&
            pixel[3] == 255);
    }
  }
  CHECK(std::equal(preserved_depth.begin(), preserved_depth.end(),
                   depth_before.begin()));

  initialize_depth(*backing, 0x14, 0.9f, std::byte{0x5a});
  auto with_depth = base_draw(*backing, Program::Depth025);
  configure_depth_bias_draw(with_depth, 0x14, 0.0f, 8.0f, 0.0f);
  CHECK(submit(backend, with_depth, memory,
               "depth bias attachment format transition"));
  const auto depth = read_depth(*backing, 0x14);
  const float expected = biased_depth(0x14, 0.25f, 0.0f, 0.0f, 8.0f, 0.0f);
  CHECK(expected_constant_biased_depth(depth, 0x14, 4, 4, expected));
}

void test_translated_depth_only_draw(
    const std::shared_ptr<Fixtures> &fixtures) {
  constexpr std::uint32_t format = 0x0a;
  for (const bool bias_enabled : {false, true}) {
    auto backing = std::make_shared<Backing>();
    initialize_depth(*backing, format, 0.9f, std::byte{0x5a});
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, Program::DepthOnly025);
    configure_depth_bias_draw(stop, format, 0.0f, 8.0f, 0.0f, bias_enabled);
    stop.registers[0x487] = rt_control(0, {});
    stop.color_targets = {};
    std::fill_n(stop.registers.begin() + 0x200, 8 * 0x10, 0);
    // Retain the translated color export: attachment specialization must remove
    // it without preventing this depth-only draw from executing its fragment.
    CHECK(submit(backend, stop, memory, "translated depth-only draw"));
    const auto depth = read_depth(*backing, format);
    const float expected = 0.25f + (bias_enabled ? 8.0f * 0x1p-24f : 0.0f);
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x)
        CHECK(expected_constant_biased_depth(depth, format, x, y, expected));
    }
    for (const auto address : {kColor0Address, kColor1Address, kColor2Address}) {
      const auto color = surface(*backing, address, 0x1000);
      CHECK(std::all_of(color.begin(), color.end(),
                        [](std::byte value) { return value == std::byte{0x29}; }));
    }
  }
}

void test_mrt_and_blend(const std::shared_ptr<Fixtures> &fixtures) {
  auto backing = std::make_shared<Backing>();
  auto memory = backing->memory(backing);
  auto backend = make_backend(fixtures);
  auto stop = base_draw(*backing, Program::Mrt);
  stop.registers[0x487] = rt_control(3, {5, 0, 7, 0, 0, 0, 0, 0});
  set_color_target(stop, 5, kColor0Address, 0xd5);
  set_color_target(stop, 0, kColor1Address, 0xe4);
  set_color_target(stop, 7, kColor2Address, 0xe3);
  CHECK(submit(backend, stop, memory, "sparse permuted MRT"));
  for (unsigned y = 0; y < kHeight; ++y) {
    for (unsigned x = 0; x < kWidth; ++x) {
      const auto color = rgba8(*backing, kColor0Address, x, y);
      CHECK(std::abs(int(color[0]) - 64) <= 1);
      CHECK(std::abs(int(color[1]) - 128) <= 1);
      CHECK(std::abs(int(color[2]) - 191) <= 1);
      CHECK(color[3] == 255);
      CHECK(u32_at(*backing, kColor1Address, x, y) == 0x11223344u);
      CHECK(static_cast<std::int32_t>(u32_at(*backing, kColor2Address, x, y)) ==
            -37);
    }
  }

  std::fill(surface(*backing, kColor0Address).begin(),
            surface(*backing, kColor0Address).end(), std::byte{0});
  std::fill(surface(*backing, kColor1Address).begin(),
            surface(*backing, kColor1Address).end(), std::byte{0});
  auto mrt_base = base_draw(*backing, Program::FloatMrtBase);
  mrt_base.registers[0x487] = rt_control(2, {3, 6, 0, 0, 0, 0, 0, 0});
  set_color_target(mrt_base, 3, kColor0Address, 0xd5);
  set_color_target(mrt_base, 6, kColor1Address, 0xd5);
  CHECK(submit(backend, mrt_base, memory, "permuted float MRT base"));
  auto mrt_blend = mrt_base;
  mrt_blend.registers[0x900] =
      static_cast<std::uint32_t>(Program::FloatMrtBlend);
  mrt_blend.registers[0x4d8] = mrt_blend.registers[0x4d9] = 1;
  mrt_blend.registers[0x680] = 0x11;
  mrt_blend.registers[0x681] = 0x1100;
  mrt_blend.registers[0x780] = 0;
  mrt_blend.registers[0x781] = mrt_blend.registers[0x784] = 1;
  mrt_blend.registers[0x782] = mrt_blend.registers[0x785] = 5;
  mrt_blend.registers[0x783] = mrt_blend.registers[0x786] = 6;
  mrt_blend.registers[0x788] = 0;
  mrt_blend.registers[0x789] = mrt_blend.registers[0x78c] = 3;
  mrt_blend.registers[0x78a] = mrt_blend.registers[0x78b] = 2;
  mrt_blend.registers[0x78d] = mrt_blend.registers[0x78e] = 2;
  CHECK(submit(backend, mrt_blend, memory,
               "per-target MRT blend equations and write masks"));
  for (unsigned y = 0; y < kHeight; ++y) {
    for (unsigned x = 0; x < kWidth; ++x) {
      const auto first = rgba8(*backing, kColor0Address, x, y);
      const auto second = rgba8(*backing, kColor1Address, x, y);
      const std::array<unsigned, 4> expected_first{128, 77, 153, 204};
      const std::array<unsigned, 4> expected_second{26, 76, 77, 114};
      for (unsigned component = 0; component < 4; ++component) {
        CHECK(std::abs(int(first[component]) -
                       int(expected_first[component])) <= 2);
        CHECK(std::abs(int(second[component]) -
                       int(expected_second[component])) <= 2);
      }
    }
  }

  std::fill(surface(*backing, kColor0Address).begin(),
            surface(*backing, kColor0Address).end(), std::byte{0});
  auto first = base_draw(*backing, Program::A);
  first.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
  set_color_target(first, 0, kColor0Address, 0xd5);
  CHECK(submit(backend, first, memory, "blend base"));
  auto second = first;
  second.registers[0x900] = static_cast<std::uint32_t>(Program::B);
  second.registers[0x4d8] = 1;
  second.registers[0x680] = 0x11;
  second.registers[0x780] = 0;
  second.registers[0x781] = second.registers[0x784] = 1;
  second.registers[0x782] = second.registers[0x785] = 5;
  second.registers[0x783] = second.registers[0x786] = 6;
  CHECK(submit(backend, second, memory, "independent blend and write mask"));
  const auto blended = rgba8(*backing, kColor0Address, 4, 4);
  CHECK(std::abs(int(blended[0]) - 128) <= 2);
  CHECK(std::abs(int(blended[1]) - 77) <= 2);
  CHECK(std::abs(int(blended[2]) - 153) <= 1);
  CHECK(std::abs(int(blended[3]) - 204) <= 1);
}

void test_rgbx_destination_alpha(const std::shared_ptr<Fixtures> &fixtures) {
  for (const std::uint32_t format : {0xf9u, 0xd5u}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, format);
    for (unsigned texel = 0; texel < kWidth * kHeight; ++texel)
      backing->bytes[kColor0Address - kGpuBase + texel * 4 + 3] = std::byte{0};
    stop.registers[0x4d8] = 1;
    stop.registers[0x780] = 1;
    stop.registers[0x781] = stop.registers[0x784] = 1;
    stop.registers[0x782] = stop.registers[0x785] = 7;
    stop.registers[0x783] = stop.registers[0x786] = 1;
    CHECK(submit(backend, stop, memory, "RGBX destination alpha is one"));
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        const auto pixel = rgba8(*backing, kColor0Address, x, y);
        CHECK(pixel[0] == (format == 0xf9 ? 255 : 0));
        CHECK(pixel[1] == 0 && pixel[2] == 0);
      }
    }
  }
}

void test_surface_clip_and_fractional_viewport(const std::shared_ptr<Fixtures> &fixtures) {
  for (const bool fractional : {false, true}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    if (fractional) {
      stop.registers[0x283] = word_of(4.25f);
      stop.registers[0x284] = word_of(4.25f);
      stop.registers[0x380] = 0;
    } else {
      stop.registers[0x3fd] = pack_halves(2, 4);
      stop.registers[0x3fe] = pack_halves(1, 5);
    }
    CHECK(submit(backend, stop, memory, "surface clip and fractional viewport"));
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        const bool covered = fractional || (x >= 2 && x < 6 && y >= 1 && y < 6);
        const auto pixel = rgba8(*backing, kColor0Address, x, y);
        CHECK(pixel[0] == (covered ? 255 : 0x29));
        CHECK(pixel[1] == (covered ? 0 : 0x29));
      }
    }
  }
}

void test_depth_formats(const std::shared_ptr<Fixtures> &fixtures) {
  for (const std::uint32_t format : {0x19u, 0x14u, 0x16u, 0x15u}) {
    auto backing = std::make_shared<Backing>();
    initialize_depth(*backing, format, 0.6f, std::byte{0x5a});
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto near = base_draw(*backing, Program::RedNear);
    near.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(near, 0, kColor0Address, 0xd5);
    set_depth_target(near, format);
    near.registers[0x4b3] = near.registers[0x4ba] = 1;
    near.registers[0x4c3] = 2;
    CHECK(submit(backend, near, memory, "depth near overwrite"));
    auto far = near;
    far.registers[0x900] = static_cast<std::uint32_t>(Program::GreenFar);
    CHECK(submit(backend, far, memory, "depth far rejection"));
    const auto depth = read_depth(*backing, format);
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        const auto pixel = rgba8(*backing, kColor0Address, x, y);
        CHECK(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0 &&
              pixel[3] == 255);
        CHECK(expected_depth(depth, format, x, y, 0.25f));
      }
    }
    const std::size_t stride = format == 0x19 ? 8 : 4;
    const std::size_t marker = format == 0x19 ? 4 : 0;
    for (std::size_t texel = 0; texel < kWidth * kHeight; ++texel)
      CHECK(depth.bytes[texel * stride + marker] == std::byte{0x5a});
    if (format == 0x19) {
      for (std::size_t texel = 0; texel < kWidth * kHeight; ++texel) {
        CHECK(depth.bytes[texel * 8 + 5] == std::byte{0xa1});
        CHECK(depth.bytes[texel * 8 + 6] == std::byte{0xb2});
        CHECK(depth.bytes[texel * 8 + 7] == std::byte{0xc3});
      }
    }
    far.registers[0x4c3] = 8;
    far.registers[0x4ba] = 0;
    CHECK(submit(backend, far, memory, "depth write disabled"));
    const auto write_disabled_depth = read_depth(*backing, format);
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        const auto color = rgba8(*backing, kColor0Address, x, y);
        CHECK(color[0] == 0 && color[1] == 255 && color[2] == 0 &&
              color[3] == 255);
        CHECK(expected_depth(write_disabled_depth, format, x, y, 0.25f));
      }
    }
  }
}

void test_identical_depth_prepass(const std::shared_ptr<Fixtures> &fixtures) {
  for (const std::uint32_t format : {0x14u, 0x16u, 0x15u}) {
    for (const std::uint32_t compare : {2u, 3u}) {
      auto backing = std::make_shared<Backing>();
      initialize_depth(*backing, format, 0.6f, std::byte{0x5a});
      auto memory = backing->memory(backing);
      auto backend = make_backend(fixtures);
      auto prepass = base_draw(*backing, Program::RedNear);
      prepass.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
      set_color_target(prepass, 0, kColor0Address, 0xd5);
      set_depth_target(prepass, format);
      prepass.registers[0x4b3] = prepass.registers[0x4ba] = 1;
      prepass.registers[0x4c3] = 2;
      CHECK(submit(backend, prepass, memory, "identical-depth prepass"));
      auto color = prepass;
      color.registers[0x900] = static_cast<std::uint32_t>(Program::B);
      color.registers[0x4ba] = 0;
      color.registers[0x4c3] = compare;
      CHECK(submit(backend, color, memory, "identical-depth color pass"));
      for (unsigned y = 0; y < kHeight; ++y) {
        for (unsigned x = 0; x < kWidth; ++x) {
          const auto pixel = rgba8(*backing, kColor0Address, x, y);
          if (compare == 3)
            CHECK(pixel[0] == 204 && pixel[1] == 51 && pixel[2] == 26);
          else
            CHECK(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0);
        }
      }
      // Native CPU writes between callbacks must supersede the retained GPU
      // value even though submission-boundary mode has no store notifications.
      initialize_depth(*backing, format, 0.75f, std::byte{0x6b});
      color.registers[0x4c3] = 2;
      CHECK(submit(backend, color, memory, "CPU depth update after prepass"));
      const auto pixel = rgba8(*backing, kColor0Address, 4, 4);
      CHECK(pixel[0] == 204 && pixel[1] == 51 && pixel[2] == 26);
      const auto depth = read_depth(*backing, format);
      CHECK(expected_depth(depth, format, 4, 4, 0.75f));
      CHECK(depth.bytes[(4 * kWidth + 4) * 4] == std::byte{0x6b});
    }
  }
}

RenderStop depth_clear(std::uint32_t format, std::uint32_t aspects) {
  RenderStop stop;
  stop.engine = engine_class::threed;
  stop.kind = RenderKind::Clear;
  stop.registers[0xdf] = 1;
  stop.registers[0x674] = aspects;
  stop.registers[0x3fd] = stop.registers[0x3fe] = 8u << 16;
  stop.registers[0x364] = word_of(0.75f);
  stop.registers[0x368] = 0xa6;
  stop.registers[0x4e7] = 0xff;
  set_depth_target(stop, format);
  return stop;
}

void test_plane_clears(const std::shared_ptr<Fixtures> &fixtures) {
  for (const std::uint32_t format : {0x19u, 0x14u, 0x16u, 0x15u}) {
    auto backing = std::make_shared<Backing>();
    initialize_depth(*backing, format, 0.4f, std::byte{0x3c});
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    CHECK(submit(backend, depth_clear(format, 1), memory, "depth-only clear",
                 true));
    auto depth = read_depth(*backing, format);
    const std::size_t marker = format == 0x19 ? 4 : 0;
    const std::size_t stride = format == 0x19 ? 8 : 4;
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        const std::size_t texel = y * kWidth + x;
        CHECK(expected_depth(depth, format, x, y, 0.75f));
        CHECK(depth.bytes[texel * stride + marker] == std::byte{0x3c});
      }
    }
    if (format != 0x15) {
      CHECK(submit(backend, depth_clear(format, 2), memory,
                   "stencil-only clear", true));
      depth = read_depth(*backing, format);
      for (unsigned y = 0; y < kHeight; ++y) {
        for (unsigned x = 0; x < kWidth; ++x) {
          const std::size_t texel = y * kWidth + x;
          CHECK(expected_depth(depth, format, x, y, 0.75f));
          CHECK(depth.bytes[texel * stride + marker] == std::byte{0xa6});
        }
      }
    }
    if (format == 0x19) {
      for (std::size_t texel = 0; texel < kWidth * kHeight; ++texel) {
        CHECK(depth.bytes[texel * 8 + 5] == std::byte{0xa1});
        CHECK(depth.bytes[texel * 8 + 6] == std::byte{0xb2});
        CHECK(depth.bytes[texel * 8 + 7] == std::byte{0xc3});
      }
    }
  }
}

RenderStop color_clear(std::uint32_t format, std::uint32_t bits,
                       bool scissored) {
  RenderStop stop;
  stop.engine = engine_class::threed;
  stop.kind = RenderKind::Clear;
  stop.registers[0xdf] = 1;
  stop.registers[0x674] = 1u << 2;
  stop.registers[0x360] = bits;
  stop.registers[0x3fd] = stop.registers[0x3fe] = 8u << 16;
  if (scissored) {
    stop.registers[0x43e] = 1u << 8;
    stop.registers[0x380] = 1;
    stop.registers[0x381] = pack_halves(2, 6);
    stop.registers[0x382] = pack_halves(1, 7);
  }
  set_color_target(stop, 0, kColor0Address, format);
  return stop;
}

void test_integer_clears(const std::shared_ptr<Fixtures> &fixtures) {
  for (const auto [format, bits, scissored] :
       {std::tuple{0xe3u, 0xffffffffu, false},
        std::tuple{0xe3u, 0xffffffdbu, true},
        std::tuple{0xe4u, 0xf1234567u, true}}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    const auto before = backing->bytes;
    const char *label = !scissored       ? "full sint clear"
                        : format == 0xe3 ? "scissored sint clear"
                                         : "scissored uint clear";
    CHECK(submit(backend, color_clear(format, bits, scissored), memory, label,
                 true));
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        const bool touched = !scissored || (x >= 2 && x < 6 && y >= 1 && y < 7);
        const std::size_t offset =
            kColor0Address - kGpuBase + (y * kWidth + x) * 4;
        std::uint32_t actual = 0;
        std::memcpy(&actual, backing->bytes.data() + offset, sizeof(actual));
        if (touched)
          CHECK(actual == bits);
        else
          CHECK(std::equal(backing->bytes.begin() + offset,
                           backing->bytes.begin() + offset + 4,
                           before.begin() + offset));
      }
    }
    const std::size_t end = kColor0Address - kGpuBase + kWidth * kHeight * 4;
    CHECK(std::equal(backing->bytes.begin(),
                     backing->bytes.begin() + (kColor0Address - kGpuBase),
                     before.begin()));
    CHECK(std::equal(backing->bytes.begin() + end, backing->bytes.end(),
                     before.begin() + end));
  }
}

void test_two_sided_stencil(const std::shared_ptr<Fixtures> &fixtures) {
  auto backing = std::make_shared<Backing>();
  initialize_depth(*backing, 0x19, 0.5f, std::byte{0x55});
  auto memory = backing->memory(backing);
  auto backend = make_backend(fixtures);
  auto stop = base_draw(*backing, Program::Faces, 6);
  stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
  set_color_target(stop, 0, kColor0Address, 0xd5);
  set_depth_target(stop, 0x19);
  stop.registers[0x4e0] = 1;
  stop.registers[0x4e1] = 1;
  stop.registers[0x4e2] = 1;
  stop.registers[0x4e3] = 3;
  stop.registers[0x4e4] = 8;
  stop.registers[0x4e5] = 0xa2;
  stop.registers[0x4e6] = 0xff;
  stop.registers[0x4e7] = 0xf0;
  stop.registers[0x565] = 1;
  stop.registers[0x566] = 1;
  stop.registers[0x567] = 1;
  stop.registers[0x568] = 3;
  stop.registers[0x569] = 8;
  stop.registers[0x3d5] = 0x3c;
  stop.registers[0x3d6] = 0x0f;
  stop.registers[0x3d7] = 0xff;
  CHECK(
      submit(backend, stop, memory, "two-sided stencil references and masks"));
  const auto depth = read_depth(*backing, 0x19);
  const auto stencil_at = [&](unsigned x, unsigned y) {
    return depth.bytes[(y * kWidth + x) * 8 + 4];
  };
  CHECK(stencil_at(1, 5) == std::byte{0xa5});
  CHECK(stencil_at(6, 5) == std::byte{0x5c});
  CHECK(expected_depth(depth, 0x19, 1, 5, 0.5f));
  CHECK(expected_depth(depth, 0x19, 6, 5, 0.5f));
}

void test_raster_and_viewports(const std::shared_ptr<Fixtures> &fixtures) {
  {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, Program::Faces, 6);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[0x646] = 1;
    stop.registers[0x648] = 0x405;
    CHECK(submit(backend, stop, memory, "cull and winding"));
    CHECK(rgba8(*backing, kColor0Address, 1, 5)[0] == 255);
    CHECK(rgba8(*backing, kColor0Address, 6, 5)[0] == 0x29);
  }
  {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[0x36b] = stop.registers[0x36c] = 0x1b01;
    stop.registers[0x4ec] = stop.registers[0x4ed] = word_of(1.0f);
    stop.registers[0x55c] = 0;
    stop.registers[0x59b] = 0;
    CHECK(submit(backend, stop, memory, "equal line fill"));
    unsigned touched = 0;
    for (unsigned y = 0; y < kHeight; ++y)
      for (unsigned x = 0; x < kWidth; ++x)
        touched += rgba8(*backing, kColor0Address, x, y)[0] == 255;
    CHECK(touched > 0 && touched < kWidth * kHeight);
  }
  {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, Program::Viewports, 6);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[0x280] = word_of(2.0f);
    stop.registers[0x281] = word_of(-4.0f);
    stop.registers[0x283] = word_of(2.0f);
    stop.registers[0x300] = 4u << 16;
    stop.registers[0x381] = pack_halves(1, 3);
    stop.registers[0x382] = pack_halves(1, 7);
    stop.registers[0x288] = word_of(2.0f);
    stop.registers[0x289] = word_of(-4.0f);
    stop.registers[0x28a] = word_of(0.5f);
    stop.registers[0x28b] = word_of(6.0f);
    stop.registers[0x28c] = word_of(4.0f);
    stop.registers[0x28d] = word_of(0.5f);
    stop.registers[0x28e] = 0x6420;
    stop.registers[0x304] = pack_halves(4, 8);
    stop.registers[0x305] = 8u << 16;
    stop.registers[0x306] = word_of(0.0f);
    stop.registers[0x307] = word_of(1.0f);
    stop.registers[0x384] = 1;
    stop.registers[0x385] = pack_halves(5, 7);
    stop.registers[0x386] = pack_halves(2, 6);
    CHECK(submit(backend, stop, memory, "viewport and scissor arrays"));
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        const bool covered = (x >= 1 && x < 3 && y >= 1 && y < 7) ||
                             (x >= 5 && x < 7 && y >= 2 && y < 6);
        CHECK((rgba8(*backing, kColor0Address, x, y)[0] == 255) == covered);
      }
    }
  }
}

void expect_unsupported(const std::shared_ptr<Fixtures> &fixtures,
                        RenderStop stop,
                        const std::shared_ptr<Backing> &backing,
                        const char *label) {
  auto memory = backing->memory(backing);
  const auto before = backing->bytes;
  auto backend = make_backend(fixtures);
  const auto result = backend.draw(backend.context, stop, memory);
  if (result.status != CommandStatus::Unsupported) {
    std::fprintf(stderr, "FAIL: %s did not stop as unsupported: %.*s\n", label,
                 int(result.detail.size()), result.detail.data());
    ++failures;
  }
  CHECK(backing->writes == 0);
  CHECK(backing->bytes == before);
}

void test_structured_stops(const std::shared_ptr<Fixtures> &fixtures) {
  {
    auto backing = std::make_shared<Backing>();
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[0x302] = word_of(0.25f);
    stop.registers[0x303] = word_of(0.75f);
    expect_unsupported(fixtures, stop, backing, "independent depth clip bounds");
  }
  {
    auto backing = std::make_shared<Backing>();
    auto stop = base_draw(*backing, Program::Uint);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xe4);
    stop.registers[0x4d8] = 1;
    stop.registers[0x780] = 0;
    stop.registers[0x781] = stop.registers[0x784] = 1;
    stop.registers[0x782] = stop.registers[0x785] = 2;
    stop.registers[0x783] = stop.registers[0x786] = 1;
    expect_unsupported(fixtures, stop, backing, "integer blending");
  }
  for (const auto [name, front, back] :
       {std::tuple{"point fill", 0x1b00u, 0x1b00u},
        std::tuple{"unequal fill", 0x1b02u, 0x1b01u}}) {
    auto backing = std::make_shared<Backing>();
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[0x36b] = front;
    stop.registers[0x36c] = back;
    expect_unsupported(fixtures, stop, backing, name);
  }
  // Metal has no line-width, smoothing or stipple state: such lines draw one
  // pixel wide, like unit-width lines.
  for (const auto [label, method, value] :
       {std::tuple{"non-unit line width", 0x4ecu, word_of(2.0f)},
        std::tuple{"non-unit aliased line width", 0x4edu, word_of(2.0f)},
        std::tuple{"line smoothing", 0x55cu, 1u},
        std::tuple{"line stipple", 0x59bu, 1u}}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[0x36b] = stop.registers[0x36c] = 0x1b01;
    stop.registers[0x4ec] = stop.registers[0x4ed] = word_of(1.0f);
    stop.registers[0x55c] = 0;
    stop.registers[0x59b] = 0;
    stop.registers[method] = value;
    CHECK(submit(backend, stop, memory, label));
    unsigned touched = 0;
    for (unsigned y = 0; y < kHeight; ++y)
      for (unsigned x = 0; x < kWidth; ++x)
        touched += rgba8(*backing, kColor0Address, x, y)[0] == 255;
    CHECK(touched > 0 && touched < kWidth * kHeight);
  }
  for (const auto [label, method] :
       {std::tuple{"point-only depth bias", 0x370u},
        std::tuple{"line-only depth bias", 0x371u}}) {
    auto backing = std::make_shared<Backing>();
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[method] = 1;
    stop.registers[0x55b] = word_of(1.0f);
    expect_unsupported(fixtures, stop, backing, label);
  }
  {
    auto backing = std::make_shared<Backing>();
    initialize_depth(*backing, 0x13, 0.5f, std::byte{0});
    auto stop = base_draw(*backing, Program::RedNear);
    configure_depth_bias_draw(stop, 0x13, 0.0f, 1.0f, 0.0f);
    expect_unsupported(fixtures, stop, backing,
                       "constant depth bias without shader ABI");
  }
  {
    auto backing = std::make_shared<Backing>();
    initialize_depth(*backing, 0x0a, 0.5f, std::byte{0});
    auto stop = base_draw(*backing, Program::Depth050);
    configure_depth_bias_draw(stop, 0x0a, 0.0f, 1.0f, 0.0f);
    stop.registers[0x444] = 1;
    expect_unsupported(fixtures, stop, backing,
                       "format-dependent floating depth bias");
  }
  for (const auto [label, method, value] :
       {std::tuple{"disabled sample mask", 0x3efu, 0xfffeu},
        std::tuple{"non-default color key", 0x4d5u, 1u}}) {
    auto backing = std::make_shared<Backing>();
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    stop.registers[method] = value;
    expect_unsupported(fixtures, stop, backing, label);
  }
}

void test_frontend_inline_indices(const std::shared_ptr<Fixtures> &fixtures) {
  auto backing = std::make_shared<Backing>();
  auto memory = backing->memory(backing);
  auto statistics = std::make_shared<MetalEngineStatistics>();
  MetalEngineOptions options;
  options.mode = MetalEngineMode::Persistent;
  options.lookup = Fixtures::lookup;
  options.lookup_context = fixtures.get();
  options.lookup_owner = fixtures;
  options.statistics = statistics;
  auto made = create_metal_engine_backend(std::move(options));
  CHECK(std::holds_alternative<RenderCallbacks>(made));
  if (!std::holds_alternative<RenderCallbacks>(made))
    return;
  FrontendCompletion completion{std::get<RenderCallbacks>(std::move(made))};
  EngineFrontEnd frontend;
  auto stop = base_draw(*backing, Program::Indexed, 6);
  stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
  set_color_target(stop, 0, kColor0Address, 0xd5);
  configure_frontend(frontend, stop, memory, completion);
  CHECK(frontend_call(frontend, memory, completion, 0x50d,
                      std::bit_cast<std::uint32_t>(-5))
            .ok());
  CHECK(frontend_call(frontend, memory, completion, 0x591, 1).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x592, 255).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x5f3, 0xdead0000).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x5f5, 0xdead0fff).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x5f6, 1).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x4c1,
                      5u | (6u << 8) | (7u << 16) | (255u << 24))
            .ok());
  CHECK(
      frontend_call(frontend, memory, completion, 0x57c, 8u | (9u << 16)).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x57a, 10).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x586, 5).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x585, 0).ok());
  CHECK(statistics->completed_draws == 1);
  const auto left = rgba8(*backing, kColor0Address, 1, 5);
  const auto right = rgba8(*backing, kColor0Address, 6, 5);
  CHECK(left[0] == 255 && left[1] == 0);
  CHECK(right[0] == 0 && right[1] == 255);
}

void test_frontend_instances(const std::shared_ptr<Fixtures> &fixtures) {
  auto backing = std::make_shared<Backing>();
  auto memory = backing->memory(backing);
  auto statistics = std::make_shared<MetalEngineStatistics>();
  MetalEngineOptions options;
  options.mode = MetalEngineMode::Persistent;
  options.lookup = Fixtures::lookup;
  options.lookup_context = fixtures.get();
  options.lookup_owner = fixtures;
  options.statistics = statistics;
  auto made = create_metal_engine_backend(std::move(options));
  CHECK(std::holds_alternative<RenderCallbacks>(made));
  if (!std::holds_alternative<RenderCallbacks>(made))
    return;
  FrontendCompletion completion{std::get<RenderCallbacks>(std::move(made))};
  EngineFrontEnd frontend;
  auto stop = base_draw(*backing, Program::Instance);
  stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
  set_color_target(stop, 0, kColor0Address, 0xd5);
  configure_frontend(frontend, stop, memory, completion);
  CHECK(frontend_call(frontend, memory, completion, 0x35d, 0).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x35e, 3).ok());
  CHECK(frontend_call(frontend, memory, completion, 0x50e, 3).ok());
  const auto draw_and_check = [&](std::uint32_t begin, unsigned instance) {
    CHECK(frontend_call(frontend, memory, completion, 0x586, begin).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x585, 0).ok());
    const auto pixel = rgba8(*backing, kColor0Address, 4, 4);
    CHECK(std::abs(int(pixel[0]) - int(instance * 17)) <= 1);
    CHECK(std::abs(int(pixel[1]) - int(255 - instance * 17)) <= 1);
  };
  draw_and_check(4, 3);
  draw_and_check((1u << 26) | 4u, 4);
  draw_and_check((1u << 27) | 4u, 4);
  draw_and_check(4, 3);
  CHECK(statistics->completed_draws == 4);
}

void test_translated_frontend_instances(
    const std::shared_ptr<Fixtures> &fixtures) {
  const auto run_sequence = [&](std::uint32_t base_instance,
                                std::uint32_t divisor) {
    auto backing = std::make_shared<Backing>();
    std::array<float, 32> records{};
    std::array<float, 32> attributes{};
    for (std::size_t index = 0; index < records.size(); ++index) {
      records[index] = float(index + 1) / 32.0f;
      attributes[index] = float(index + 1) / 32.0f;
    }
    std::memcpy(backing->bytes.data() +
                    (kInstanceStorageAddress - kGpuBase),
                records.data(), sizeof(records));
    std::memcpy(backing->bytes.data() +
                    (kInstanceAttributeAddress - kGpuBase),
                attributes.data(), sizeof(attributes));
    const std::array<std::array<float, 4>, 3> positions{{
        {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f},
        {-1.0f, 3.0f, 0.5f, 1.0f}}};
    std::memcpy(backing->bytes.data() + (kInstancePositionAddress - kGpuBase),
                positions.data(), sizeof(positions));
    const std::array<std::uint32_t, 3> descriptor{
        std::uint32_t(kInstanceStorageAddress),
        std::uint32_t(kInstanceStorageAddress >> 32),
        std::uint32_t(sizeof(records))};
    std::memcpy(backing->bytes.data() +
                    (kInstanceCbufAddress - kGpuBase) + 16,
                descriptor.data(), sizeof(descriptor));

    auto memory = backing->memory(backing);
    auto statistics = std::make_shared<MetalEngineStatistics>();
    MetalEngineOptions options;
    options.mode = MetalEngineMode::Persistent;
    options.lookup = Fixtures::lookup;
    options.lookup_context = fixtures.get();
    options.lookup_owner = fixtures;
    options.statistics = statistics;
    auto made = create_metal_engine_backend(std::move(options));
    CHECK(std::holds_alternative<RenderCallbacks>(made));
    if (!std::holds_alternative<RenderCallbacks>(made))
      return;
    FrontendCompletion completion{std::get<RenderCallbacks>(std::move(made))};
    EngineFrontEnd frontend;
    auto stop = base_draw(*backing, Program::TranslatedInstance, 3);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    configure_frontend(frontend, stop, memory, completion);

    CHECK(frontend_call(frontend, memory, completion, 0x35d, 0).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x35e, 3).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x50e, base_instance)
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x458,
                        (0x12u << 21) | (7u << 27))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x620, 1).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x700, 0x1000 | 4u)
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x701,
                        std::uint32_t(kInstanceAttributeAddress >> 32))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x702,
                        std::uint32_t(kInstanceAttributeAddress))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x703, divisor).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x459,
                        1u | (1u << 21) | (7u << 27)).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x704, 0x1000 | 16u).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x705,
                        std::uint32_t(kInstancePositionAddress >> 32)).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x706,
                        std::uint32_t(kInstancePositionAddress)).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x7c2,
                        std::uint32_t((kInstancePositionAddress + sizeof(positions) - 1) >> 32)).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x7c3,
                        std::uint32_t(kInstancePositionAddress + sizeof(positions) - 1)).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x7c0,
                        std::uint32_t((kInstanceAttributeAddress +
                                      sizeof(attributes) - 1) >>
                                     32))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x7c1,
                        std::uint32_t(kInstanceAttributeAddress +
                                      sizeof(attributes) - 1))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x8e0, 28).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x8e1,
                        std::uint32_t(kInstanceCbufAddress >> 32))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x8e2,
                        std::uint32_t(kInstanceCbufAddress))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x904, (2u << 4) | 1u)
              .ok());

    const auto draw_and_check = [&](std::uint32_t begin,
                                    std::uint32_t instance_index) {
      auto result = frontend_call(frontend, memory, completion, 0x586, begin);
      if (result.ok()) result = frontend_call(frontend, memory, completion, 0x585, 0);
      if (!result.ok()) {
        std::fprintf(stderr, "translated instance draw failed: %.*s\n",
                     int(result.detail.size()), result.detail.data());
        CHECK(false);
        return;
      }
      const auto pixel = rgba8(*backing, kColor0Address, 4, 4);
      const auto fetch = (base_instance + instance_index) / divisor;
      const int expected_record = int(std::lround(records[instance_index] * 255));
      const int expected_attribute = int(std::lround(attributes[fetch] * 255));
      if (std::abs(int(pixel[0]) - expected_record) > 1 ||
          std::abs(int(pixel[1]) - expected_attribute) > 1) {
        std::fprintf(stderr,
                     "translated instance mismatch base=%u divisor=%u "
                     "instance=%u fetch=%u pixel=(%u,%u,%u) expected=(%d,%d)\n",
                     base_instance, divisor, instance_index, fetch, pixel[0],
                     pixel[1], pixel[2], expected_record, expected_attribute);
        CHECK(false);
      }
    };

    draw_and_check(4, 0);
    for (std::uint32_t instance = 1; instance <= 17; ++instance)
      draw_and_check((1u << 26) | 4u, instance);
    draw_and_check((1u << 27) | 4u, 17);
    draw_and_check((1u << 26) | (1u << 27) | 4u, 18);
    draw_and_check(4, 0);
    CHECK(statistics->completed_draws == 21);
  };

  run_sequence(0, 1);
  run_sequence(0, 3);
  run_sequence(7, 1);
  run_sequence(7, 3);
}

void configure_index_buffer(EngineFrontEnd &frontend,
                            const GpuAddressSpace &memory,
                            FrontendCompletion &completion,
                            const RenderStop &stop) {
  for (const std::uint32_t method : {0x5f3u, 0x5f5u, 0x5f6u})
    CHECK(frontend_call(frontend, memory, completion, method,
                        stop.registers[method])
              .ok());
}

void test_frontend_compact_and_fan(const std::shared_ptr<Fixtures> &fixtures) {
  {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto backend = make_backend(fixtures);
    FrontendCompletion completion{backend};
    EngineFrontEnd frontend;
    auto stop = base_draw(*backing, Program::Instance);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    configure_frontend(frontend, stop, memory, completion);
    CHECK(frontend_call(frontend, memory, completion, 0x50e, 2).ok());
    const std::uint32_t compact = (4u << 28) | (3u << 16);
    CHECK(frontend_call(frontend, memory, completion, 0x485, compact).ok());
    CHECK(std::abs(int(rgba8(*backing, kColor0Address, 4, 4)[0]) - 34) <= 1);
    CHECK(frontend_call(frontend, memory, completion, 0x486, compact).ok());
    CHECK(std::abs(int(rgba8(*backing, kColor0Address, 4, 4)[0]) - 51) <= 1);
  }
  for (const std::uint32_t first_method : {0x5f9u, 0x5fau, 0x5fbu}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    FrontendCompletion completion{make_backend(fixtures)};
    EngineFrontEnd frontend;
    auto stop = base_draw(*backing, Program::Indexed, 6);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    configure_frontend(frontend, stop, memory, completion);
    configure_index_buffer(frontend, memory, completion, stop);
    CHECK(frontend_call(frontend, memory, completion, first_method,
                        (4u << 28) | (3u << 16))
              .ok());
    CHECK(rgba8(*backing, kColor0Address, 1, 5)[0] == 255);
  }
  for (const std::uint32_t subsequent_method : {0x5fcu, 0x5fdu, 0x5feu}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    FrontendCompletion completion{make_backend(fixtures)};
    EngineFrontEnd frontend;
    auto stop = base_draw(*backing, Program::Indexed, 6);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    configure_frontend(frontend, stop, memory, completion);
    configure_index_buffer(frontend, memory, completion, stop);
    CHECK(frontend_call(frontend, memory, completion, 0x5f9,
                        (4u << 28) | (3u << 16))
              .ok());
    CHECK(frontend_call(frontend, memory, completion, subsequent_method,
                        (4u << 28) | (3u << 16) | 3u)
              .ok());
    CHECK(rgba8(*backing, kColor0Address, 1, 5)[0] == 255);
    CHECK(rgba8(*backing, kColor0Address, 6, 5)[1] == 255);
  }
  for (const bool override_topology : {false, true}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    FrontendCompletion completion{make_backend(fixtures)};
    EngineFrontEnd frontend;
    auto stop = base_draw(*backing, Program::Fan, 4);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    configure_frontend(frontend, stop, memory, completion);
    CHECK(frontend_call(frontend, memory, completion, 0x35d, 0).ok());
    CHECK(frontend_call(frontend, memory, completion, 0x35e, 4).ok());
    if (override_topology) {
      CHECK(frontend_call(frontend, memory, completion, 0x652, 1).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x65c, 0x1015).ok());
    }
    CHECK(frontend_call(frontend, memory, completion, 0x586,
                        override_topology ? 4u : 6u)
              .ok());
    CHECK(frontend_call(frontend, memory, completion, 0x585, 0).ok());
    for (unsigned y = 0; y < kHeight; ++y)
      for (unsigned x = 0; x < kWidth; ++x)
        CHECK(rgba8(*backing, kColor0Address, x, y)[0] == 255);
  }
}

void test_frontend_no_coverage(const std::shared_ptr<Fixtures> &fixtures) {
  enum class EmptyCase { ZeroCount, RestartOnly, DegenerateFan };
  for (const EmptyCase empty : {EmptyCase::ZeroCount, EmptyCase::RestartOnly,
                                EmptyCase::DegenerateFan}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    auto statistics = std::make_shared<MetalEngineStatistics>();
    MetalEngineOptions options;
    options.mode = MetalEngineMode::Persistent;
    options.lookup = Fixtures::lookup;
    options.lookup_context = fixtures.get();
    options.lookup_owner = fixtures;
    options.statistics = statistics;
    auto made = create_metal_engine_backend(std::move(options));
    CHECK(std::holds_alternative<RenderCallbacks>(made));
    if (!std::holds_alternative<RenderCallbacks>(made))
      continue;
    FrontendCompletion completion{std::get<RenderCallbacks>(std::move(made))};
    EngineFrontEnd frontend;
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    configure_frontend(frontend, stop, memory, completion);
    const auto before = backing->bytes;
    if (empty == EmptyCase::ZeroCount) {
      CHECK(frontend_call(frontend, memory, completion, 0x35d, 0).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x35e, 0).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x586, 4).ok());
    } else if (empty == EmptyCase::RestartOnly) {
      CHECK(frontend_call(frontend, memory, completion, 0x591, 1).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x592, 255).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x57a, 255).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x586, 5).ok());
    } else {
      CHECK(frontend_call(frontend, memory, completion, 0x35d, 0).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x35e, 2).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x586, 6).ok());
    }
    CHECK(frontend_call(frontend, memory, completion, 0x585, 0).ok());
    CHECK(statistics->completed_draws == 1);
    CHECK(backing->bytes == before);
    CHECK(frontend_call(frontend, memory, completion, 0x1d, (9u << 8) | 1u, 0)
              .ok());
    CHECK(completion.increments == 1 && completion.last_syncpoint == 9);
  }
}

void test_frontend_inline_stops(const std::shared_ptr<Fixtures> &fixtures) {
  for (const bool continuation : {false, true}) {
    auto backing = std::make_shared<Backing>();
    auto memory = backing->memory(backing);
    FrontendCompletion completion{make_backend(fixtures)};
    EngineFrontEnd frontend;
    auto stop = base_draw(*backing, Program::RedNear);
    stop.registers[0x487] = rt_control(1, {0, 0, 0, 0, 0, 0, 0, 0});
    set_color_target(stop, 0, kColor0Address, 0xd5);
    configure_frontend(frontend, stop, memory, completion);
    const auto before = backing->bytes;
    CHECK(frontend_call(frontend, memory, completion, 0x4c1,
                        0u | (1u << 8) | (2u << 16) | (3u << 24))
              .ok());
    CommandResult stopped;
    if (continuation) {
      stopped =
          frontend_call(frontend, memory, completion, 0x586, (1u << 26) | 4u);
    } else {
      CHECK(frontend_call(frontend, memory, completion, 0x5f7, 1).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x5f8, 3).ok());
      CHECK(frontend_call(frontend, memory, completion, 0x586, 4).ok());
      stopped = frontend_call(frontend, memory, completion, 0x585, 0);
    }
    CHECK(stopped.status == CommandStatus::Unsupported);
    CHECK(backing->bytes == before);
  }
}

} // namespace

int main() {
  auto fixtures = std::make_shared<Fixtures>();
  fixtures->library = compile_metallib();
  CHECK(!fixtures->library.empty());
  if (fixtures->library.empty())
    return 1;
  try {
    fixtures->translated_instance =
        compile_translated_instance_stage(translated_instance_vertex());
    for (unsigned discard_left = 0; discard_left < 2; ++discard_left) {
      const auto artifact = translated_depth_bias_fragment(discard_left != 0);
      auto stage = std::make_shared<CompiledGraphicsStage>();
      stage->metallib = compile_msl(artifact.source, true);
      CHECK(!stage->metallib.empty());
      if (stage->metallib.empty())
        return 1;
      stage->stage = armigo::shader_input::Stage::Fragment;
      stage->entry_point = artifact.entry_point;
      stage->emulates_depth_bias = artifact.reflection.emulate_depth_bias;
      stage->fragment_output_types_from_render_targets = true;
      stage->fragment_position =
          gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
      for (const auto &io : artifact.reflection.io) {
        if (io.semantic == sp::IoSemantic::color) {
          stage->interface.color_output_mask |= io.mask << (io.location * 4);
          stage->color_outputs.push_back(
              {static_cast<std::uint8_t>(io.location), io.mask,
               ColorOutputScalar::f32, io.type.components});
        } else if (io.semantic == sp::IoSemantic::fragment_coord) {
          stage->interface.position_input_mask |= io.mask;
        }
      }
      fixtures->generated_fragments[discard_left] = std::move(stage);
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: synthetic shader translation: %s\n", error.what());
    return 1;
  }
  test_mrt_and_blend(fixtures);
  test_rgbx_destination_alpha(fixtures);
  test_surface_clip_and_fractional_viewport(fixtures);
  test_depth_formats(fixtures);
  test_constant_depth_bias(fixtures);
  test_sloped_and_clamped_depth_bias(fixtures);
  test_translated_depth_bias_visibility(fixtures);
  test_translated_shallow_depth_bias(fixtures);
  test_depth_bias_disable_restores_zero(fixtures);
  test_depth_bias_color_only_variant(fixtures);
  test_translated_depth_only_draw(fixtures);
  test_identical_depth_prepass(fixtures);
  test_plane_clears(fixtures);
  test_integer_clears(fixtures);
  test_two_sided_stencil(fixtures);
  test_raster_and_viewports(fixtures);
  test_structured_stops(fixtures);
  test_frontend_inline_indices(fixtures);
  test_frontend_instances(fixtures);
  test_translated_frontend_instances(fixtures);
  test_frontend_compact_and_fan(fixtures);
  test_frontend_no_coverage(fixtures);
  test_frontend_inline_stops(fixtures);
  if (failures == 0)
    std::puts("Verified expanded persistent Metal engine state.");
  return failures == 0 ? 0 : 1;
}
