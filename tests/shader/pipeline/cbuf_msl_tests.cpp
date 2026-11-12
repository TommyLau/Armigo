// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "pipeline/phases.h"
#include "proven_cbuf_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace armigo::shader::test::cbuf_msl {
std::string gpu_argument_buffer_source();

std::string gpu_fragment_source(unsigned case_index);
std::string gpu_small_window_source(unsigned words);
}

namespace {
using namespace armigo::shader::pipeline;

int failures{};
std::filesystem::path fixture_directory;
bool compile_metal{};

void check(bool value, const std::string &message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

Operand word(std::uint32_t bits) {
  return LiteralBits{bits, ValueType::word32};
}

std::size_t occurrences(const std::string &text, const std::string &needle) {
  std::size_t count = 0;
  for (std::size_t position = 0;
       (position = text.find(needle, position)) != std::string::npos;
       position += needle.size())
    ++count;
  return count;
}

std::string quote(const std::string &value) {
  std::string result = "'";
  for (const char character : value)
    result += character == '\'' ? "'\\''" : std::string(1, character);
  return result + "'";
}

struct Fixture {
  StructuredProgram program;
  ResourcePlan plan;
  ResourceId buffer{};
  ResourceId support{};

  explicit Fixture(std::uint8_t slot = 5, std::uint32_t window_bytes = 64,
                   bool color_output = false) {
    program.ir.stage = Stage::fragment;
    program.ir.entry = FunctionId{0};
    program.ir.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
    program.ir.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
    program.regions = {BlockRegion{BlockId{0}}};
    program.functions = {{FunctionId{0}, RegionId{0}}};
    program.profile = {"synthetic-metal4",
                       1,
                       31,
                       128,
                       16,
                       true,
                       false,
                       true,
                       false,
                       false,
                       false};
    plan.stage = Stage::fragment;
    plan.profile = program.profile;

    BufferUse constant;
    constant.source = ConstantBufferSource{slot};
    constant.access = Access::read;
    constant.size = {SizeKind::constant_buffer_window, window_bytes, 4,
                     ConstantBufferPolicy::padded_zero_words_v2};
    buffer = add_resource(constant, 3);

    BufferUse sizes;
    sizes.source =
        SupportBufferSource{std::string(constant_buffer_support_schema),
                            constant_buffer_support_version};
    sizes.access = Access::read;
    sizes.size = {SizeKind::accessed_extent, constant_buffer_support_bytes, 4};
    support = add_resource(sizes, 11);

    if (color_output) {
      IoUse color;
      color.semantic = IoSemantic::color;
      color.access = Access::write;
      color.mask = 15;
      color.type = {ScalarType::u32, 4};
      program.requirements.io.push_back(color);
      plan.io.push_back(color);
    }
  }

  ResourceId add_resource(ResourceUse use, std::uint16_t index) {
    const ResourceId id{
        static_cast<std::uint32_t>(program.requirements.uses.size())};
    program.requirements.uses.push_back(use);
    plan.resources.push_back(
        {id, std::move(use), MetalRange{MetalNamespace::buffer, index, 1}});
    return id;
  }

  ResourceId add_constant_buffer(std::uint8_t slot, std::uint32_t window_bytes,
                                 std::uint16_t index) {
    BufferUse constant;
    constant.source = ConstantBufferSource{slot};
    constant.access = Access::read;
    constant.size = {SizeKind::constant_buffer_window, window_bytes, 4,
                     ConstantBufferPolicy::padded_zero_words_v2};
    return add_resource(constant, index);
  }

  ValueId value(ValueType type = ValueType::word32) {
    const ValueId id{static_cast<std::uint32_t>(program.ir.values.size())};
    program.ir.values.push_back({type, ValueKind::temporary});
    return id;
  }

  OperationId add(Operation operation) {
    operation.effects = required_effects(operation);
    operation.source = {40 + static_cast<Pc>(program.ir.operations.size() * 8),
                        InstructionName::ldc};
    const OperationId id{
        static_cast<std::uint32_t>(program.ir.operations.size())};
    program.ir.operations.push_back(std::move(operation));
    program.ir.blocks[0].operations.push_back(id);
    return id;
  }

  ValueId copied_word(std::uint32_t bits) {
    const auto result = value();
    Operation operation;
    operation.opcode = Opcode::copy;
    operation.inputs.push_back(word(bits));
    operation.outputs.push_back(result);
    add(std::move(operation));
    return result;
  }

  std::vector<ValueId> load(std::uint32_t byte_offset,
                            std::uint8_t element_bytes = 4,
                            std::uint8_t elements = 1,
                            std::optional<ValueId> dynamic = {},
                            bool signed_narrow = false) {
    return load_from(buffer, byte_offset, element_bytes, elements, dynamic,
                     signed_narrow);
  }

  std::vector<ValueId>
  load_from(ResourceId source_buffer, std::uint32_t byte_offset,
            std::uint8_t element_bytes = 4, std::uint8_t elements = 1,
            std::optional<ValueId> dynamic = {}, bool signed_narrow = false) {
    MemoryInfo memory;
    memory.space = AddressSpace::constant_buffer;
    memory.access = Access::read;
    memory.constant_buffer_slot =
        std::get<ConstantBufferSource>(
            std::get<BufferUse>(program.requirements.uses[source_buffer.value])
                .source)
            .slot;
    memory.byte_offset = byte_offset;
    memory.element_bytes = element_bytes;
    memory.elements = elements;
    memory.resource = source_buffer;
    Operation operation;
    operation.opcode = Opcode::load;
    if (dynamic) {
      memory.dynamic_byte_offset = InputIndex{0};
      operation.inputs.push_back(*dynamic);
    }
    if (signed_narrow) {
      operation.numeric = {Domain::signed_integer, element_bytes == 1
                                                       ? ScalarType::s8
                                                       : ScalarType::s16};
    }
    std::vector<ValueId> results;
    for (unsigned i = 0; i < elements; ++i) {
      results.push_back(
          value(element_bytes == 8 ? ValueType::word64 : ValueType::word32));
      operation.outputs.push_back(results.back());
    }
    operation.info = memory;
    add(std::move(operation));
    return results;
  }

  std::vector<ValueId> unpack(ValueId wide) {
    std::vector<ValueId> result{value(), value()};
    Operation operation;
    operation.opcode = Opcode::unpack_word64;
    operation.inputs.push_back(wide);
    operation.outputs = {result[0], result[1]};
    add(std::move(operation));
    return result;
  }

  void store_color(std::vector<Operand> values) {
    const auto repeating = values;
    while (values.size() < 4)
      values.push_back(repeating.empty()
                           ? Operand{word(0)}
                           : repeating[values.size() % repeating.size()]);
    values.resize(4);
    IoInfo info;
    info.semantic = IoSemantic::color;
    info.type = {ScalarType::u32, 4};
    info.component_mask = 15;
    Operation operation;
    operation.opcode = Opcode::store_io;
    operation.inputs = std::move(values);
    for (unsigned i = 0; i < 4; ++i)
      info.data.push_back(InputIndex{static_cast<std::uint16_t>(i)});
    operation.info = info;
    add(std::move(operation));
  }

  Result<MslArtifact> emit() { return emit_msl(program, plan, {}); }
};

std::string accepted(Fixture &fixture, const std::string &label) {
  auto result = fixture.emit();
  if (const auto *diagnostic = std::get_if<Diagnostic>(&result)) {
    check(false, label + ": " + diagnostic->message);
    return {};
  }
  auto artifact = std::get<MslArtifact>(std::move(result));
  check(artifact.reflection.reflection_version == 3,
        label + " keeps reflection version 3");
  if (!fixture_directory.empty()) {
    const auto source_path = fixture_directory / (label + ".metal");
    std::ofstream(source_path) << artifact.source;
    if (compile_metal) {
      const auto output_path = fixture_directory / (label + ".air");
      const auto command =
          "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
          quote((fixture_directory / "module-cache").string()) + " -c " +
          quote(source_path.string()) + " -o " + quote(output_path.string());
      check(std::system(command.c_str()) == 0, label + " Metal 4 compilation");
    }
  }
  return artifact.source;
}

void rejected(Fixture fixture, const std::string &label,
              std::optional<DiagnosticCode> code = std::nullopt) {
  auto result = fixture.emit();
  const auto *diagnostic = std::get_if<Diagnostic>(&result);
  check(diagnostic && diagnostic->phase == Phase::emit_msl &&
            (!code || diagnostic->code == *code),
        label);
  if (diagnostic && code && diagnostic->code != *code)
    std::cerr << "  " << diagnostic->message << '\n';
}

void compile_generated_source(const std::string &source,
                              const std::string &label) {
  if (fixture_directory.empty())
    return;
  const auto source_path = fixture_directory / (label + ".metal");
  std::ofstream(source_path) << source;
  if (!compile_metal)
    return;
  const auto output_path = fixture_directory / (label + ".air");
  const auto command =
      "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
      quote((fixture_directory / "module-cache").string()) + " -c " +
      quote(source_path.string()) + " -o " + quote(output_path.string());
  check(std::system(command.c_str()) == 0, label + " Metal 4 compilation");
}

void proven_window(Fixture& fixture, ResourceId resource = ResourceId{0}) {
  auto& use = std::get<BufferUse>(fixture.program.requirements.uses[resource.value]);
  use.size.constant_buffer_policy = ConstantBufferPolicy::proven_zero_words_v2;
  fixture.plan.resources[resource.value].use = use;
}

void source_tests() {
  for (const auto& item : armigo::shader::test::proven_cbuf::cases) {
    try {
      const auto artifact = armigo::shader::test::proven_cbuf::translate(item);
      const BufferUse* data = nullptr;
      for (const auto& resource : artifact.reflection.resources)
        if (const auto* use = std::get_if<BufferUse>(&resource.use))
          if (const auto* slot = std::get_if<ConstantBufferSource>(&use->source); slot && slot->slot == 3) data = use;
      check(data && data->size.constant_buffer_policy == item.policy && data->size.bytes == item.window,
            std::string(item.name) + " legalization selects the independently checked complete-word contract");
      const bool clamp = artifact.source.find(", " + std::to_string(item.window) + "ul)") != std::string::npos;
      check(clamp == (item.policy == ConstantBufferPolicy::padded_zero_words_v2),
            std::string(item.name) + " emission matches its proven/padded policy");
      check(artifact.source.find("armigo_cbuf_word") == std::string::npos,
            std::string(item.name) + " has no per-word range helper");
      compile_generated_source(artifact.source, std::string("proven_shape_") + item.name);
    } catch (const std::exception& error) { check(false, std::string(item.name) + ": " + error.what()); }
  }
  Fixture proven(5, 64);
  proven_window(proven);
  proven.load(4);
  const auto proved_source = accepted(proven, "cbuf_proven_word");
  check(proved_source.find("min(") == std::string::npos &&
            proved_source.find("armigo_cbuf_word") == std::string::npos &&
            proved_source.find("resource0[((memory_address0 + 0ul) & ~3ul) / 4ul]") != std::string::npos,
        "Proved windows use direct loads with no clamp or per-word guard");
  Fixture proven_end(0, constant_buffer_window_bytes);
  proven_window(proven_end);
  proven_end.load(constant_buffer_window_bytes - 4);
  check(accepted(proven_end, "cbuf_proven_last_word").find("min(") == std::string::npos,
        "The last complete word can use an unpadded proven window");
  Fixture direct(5, 64);
  direct.load(4);
  auto source = accepted(direct, "cbuf_direct_word");
  check(source.find("constant uint* resource0 [[buffer(3)]]") !=
                std::string::npos &&
            source.find("constant uint* resource1 [[buffer(11)]]") !=
                std::string::npos,
        "Constant buffer and support table use planned Metal arguments");
  check(source.find("armigo_cbuf_word") == std::string::npos &&
            source.find("min(4ul, 64ul)") != std::string::npos &&
            source.find("resource0[((memory_address0 + 0ul) & ~3ul) / 4ul]") != std::string::npos,
        "Padded loads use one address clamp and direct word indexing");
  check(source.find("resource1[5u]") == std::string::npos &&
            source.find(" ? ") == std::string::npos,
        "Constant loads have no size-table read or per-word zero branch");

  Fixture dynamic(2, constant_buffer_window_bytes);
  const auto offset = dynamic.copied_word(UINT32_MAX);
  dynamic.load(12, 4, 1, offset);
  source = accepted(dynamic, "cbuf_dynamic_word");
  check(source.find("4294967295u") != std::string::npos &&
            source.find("resource1[2u]") == std::string::npos &&
            source.find(", 65536ul)") != std::string::npos,
        "Dynamic offsets clamp to the zero pad after the 64 KiB window");

  Fixture unaligned(0, 16);
  unaligned.load(6);
  source = accepted(unaligned, "cbuf_unaligned_word");
  check(source.find("resource0[((memory_address0 + 0ul) & ~3ul) / 4ul]") != std::string::npos &&
            source.find("memory_address0 + 0ul") != std::string::npos &&
            source.find("255u") == std::string::npos,
        "Unaligned U32 uses one normalized word load rather than byte "
        "reconstruction");

  Fixture narrow(7, 16);
  narrow.load(3, 1);
  narrow.load(3, 2, 1, {}, true);
  source = accepted(narrow, "cbuf_narrow");
  check(source.find("& 3ul) * 8ul") != std::string::npos &&
            source.find("255u") != std::string::npos &&
            source.find("& 2ul) * 8ul") != std::string::npos &&
            source.find("65535u") != std::string::npos &&
            source.find("as_type<short>") != std::string::npos,
        "Narrow extraction follows byte and aligned-halfword rules after the "
        "padded word load");

  Fixture wide(1, 16);
  wide.load(4, 8);
  source = accepted(wide, "cbuf_word64");
  check(occurrences(source, "resource0[") == 2 &&
            occurrences(source, "min(") == 1 &&
            source.find("+ 4ul") != std::string::npos,
        "B64 uses one clamp followed by two direct word reads");

  const auto first_dynamic =
      armigo::shader::test::cbuf_msl::gpu_fragment_source(9);
  const auto second_dynamic =
      armigo::shader::test::cbuf_msl::gpu_fragment_source(9);
  check(first_dynamic == second_dynamic &&
            first_dynamic.find("resource1[1u]") == std::string::npos &&
            first_dynamic.find("resource1[0u]") == std::string::npos &&
            first_dynamic.find("ulong(value0)") != std::string::npos,
        "One generated variant reads a draw-time offset without runtime size reads");
  compile_generated_source(first_dynamic, "cbuf_runtime_dynamic_gpu_fragment");
  const auto dynamic_wide = armigo::shader::test::cbuf_msl::gpu_fragment_source(13);
  check(occurrences(dynamic_wide, "min(") == 2 &&
            dynamic_wide.find("armigo_cbuf_word") == std::string::npos,
        "Dynamic B64 clamps once; its offset-buffer load has its own static clamp");
  compile_generated_source(dynamic_wide, "cbuf_runtime_wide_gpu_fragment");
  compile_generated_source(
      armigo::shader::test::cbuf_msl::gpu_fragment_source(10),
      "cbuf_signed_byte_gpu_fragment");
  compile_generated_source(
      armigo::shader::test::cbuf_msl::gpu_fragment_source(11),
      "cbuf_signed_half_gpu_fragment");
  compile_generated_source(
      armigo::shader::test::cbuf_msl::gpu_fragment_source(12),
      "cbuf_unaligned_half_gpu_fragment");
}

void validation_tests() {
  Fixture unknown_proven(0, constant_buffer_window_bytes);
  proven_window(unknown_proven);
  const auto offset_buffer = unknown_proven.add_constant_buffer(1, 4, 4);
  const auto unknown = unknown_proven.load_from(offset_buffer, 0).front();
  unknown_proven.load(0, 4, 1, unknown);
  rejected(unknown_proven, "An unbounded offset cannot claim an unpadded proven window",
           DiagnosticCode::missing_fact);
  auto forged_bound = unknown_proven;
  std::get<MemoryInfo>(forged_bound.program.ir.operations.back().info).dynamic_byte_offset_upper_bound = 60;
  rejected(std::move(forged_bound), "A forged offset bound must be independently rejected");
  Fixture crosses_end(0, constant_buffer_window_bytes);
  proven_window(crosses_end);
  crosses_end.load(constant_buffer_window_bytes - 4, 8);
  rejected(std::move(crosses_end), "B64 cannot claim an unpadded window when its high word needs the pad");
  Fixture static_extent(0, 4);
  static_extent.load(12);
  rejected(std::move(static_extent),
           "Static constant-buffer load must fit the reflected window");

  Fixture missing_support;
  missing_support.load(0);
  missing_support.program.requirements.uses.pop_back();
  missing_support.plan.resources.pop_back();
  rejected(std::move(missing_support),
           "Constant-buffer window requires its size table");

  Fixture bad_policy;
  bad_policy.load(0);
  auto &policy_use =
      std::get<BufferUse>(bad_policy.program.requirements.uses[0]);
  policy_use.size.constant_buffer_policy = ConstantBufferPolicy::none;
  bad_policy.plan.resources[0].use = policy_use;
  rejected(std::move(bad_policy),
           "Constant-buffer window requires the word-zero policy");

  Fixture bad_window;
  bad_window.load(0);
  auto &window_use =
      std::get<BufferUse>(bad_window.program.requirements.uses[0]);
  window_use.size.bytes = constant_buffer_window_bytes + 4;
  bad_window.plan.resources[0].use = window_use;
  rejected(std::move(bad_window), "Constant-buffer window is capped at 64 KiB");

  Fixture bad_support;
  bad_support.load(0);
  auto &support_use =
      std::get<BufferUse>(bad_support.program.requirements.uses[1]);
  std::get<SupportBufferSource>(support_use.source).version =
      constant_buffer_support_version + 1;
  bad_support.plan.resources[1].use = support_use;
  rejected(std::move(bad_support), "Unknown support-table version is rejected",
           DiagnosticCode::unsupported_capability);
}

} // namespace

namespace armigo::shader::test::cbuf_msl {

std::string gpu_argument_buffer_source() {
  Fixture fixture(0, 64, true);
  TextureDescriptor descriptor;
  descriptor.format_namespace = "synthetic-format-v1";
  descriptor.shape.dimension = TextureDimension::d2;
  descriptor.normalized_coordinates = true;
  descriptor.format = ComponentFormat::floating;
  TextureUse texture;
  texture.key = {2, 16}; texture.descriptor = descriptor;
  texture.array_length = 17; texture.dynamically_indexed = true; texture.modes = {TextureMode::sample_level};
  const auto texture_id = fixture.add_resource(texture, 0);
  const auto sampler_id = fixture.add_resource(SamplerUse{texture.key, false, true, 17}, 0);
  TextureInfo info;
  info.texture = texture.key; info.sampler = texture.key; info.descriptor = descriptor;
  info.resource = texture_id; info.sampler_resource = sampler_id; info.mode = TextureMode::sample_level;
  info.coordinates = {InputIndex{0}, InputIndex{1}}; info.level_or_bias = InputIndex{2};
  info.array_index = InputIndex{3}; info.component_mask = 1;
  const auto sampled = fixture.value();
  fixture.add({{}, Opcode::texture_sample, {}, {Domain::floating, ScalarType::f32, Rounding::nearest_even},
      {word(0), word(0), word(0), word(16)}, {sampled}, info});
  fixture.store_color({sampled});
  // Compile mixed direct-layout resource types and a texture array beyond 128.
  texture.key = {2, 32}; texture.array_length = 129;
  fixture.add_resource(texture, 0);
  texture.key = {2, 48}; texture.array_length = 1;
  texture.descriptor.format = ComponentFormat::depth; texture.descriptor.comparison = true;
  fixture.add_resource(texture, 0);
  texture.key = {2, 64}; texture.kind = ResourceTextureKind::storage_image;
  texture.descriptor.format = ComponentFormat::unsigned_integer; texture.descriptor.scalar = ScalarType::u32;
  texture.descriptor.comparison = false; texture.access = Access::read_write;
  texture.modes = {TextureMode::fetch};
  fixture.add_resource(texture, 0);
  auto planned = plan_resources(fixture.program, fixture.program.profile);
  if (const auto* diagnostic = std::get_if<Diagnostic>(&planned))
    throw std::runtime_error("Argument-buffer GPU planning failed: " + diagnostic->message);
  fixture.plan = std::get<ResourcePlan>(std::move(planned));
  auto emitted = fixture.emit();
  if (const auto* diagnostic = std::get_if<Diagnostic>(&emitted))
    throw std::runtime_error("Argument-buffer GPU emission failed: " + diagnostic->message);
  auto source = std::get<MslArtifact>(std::move(emitted)).source;
  const auto& layout = *fixture.plan.argument_buffer;
  source += "\nstatic_assert(sizeof(ArmigoStageResources) == " + std::to_string(layout.byte_extent) + ", \"Argument-buffer extent\");\n";
  source += "static_assert(alignof(ArmigoStageResources) == " + std::to_string(layout.alignment) + ", \"Argument-buffer alignment\");\n";
  for (const auto& resource : fixture.plan.resources)
    if (const auto* member = std::get_if<ArgumentBufferMember>(&resource.location))
      source += "static_assert(__builtin_offsetof(ArmigoStageResources, resource" +
          std::to_string(resource.resource.value) + ") == " + std::to_string(member->byte_offset) + ", \"Argument-buffer member offset\");\n";
  return source;
}

std::string gpu_small_window_source(unsigned words) {
  if (words < 1 || words > 4 || words == 3)
    throw std::runtime_error("Small-window fixture width is invalid");
  Fixture fixture(0, 64, true);
  fixture.add_constant_buffer(1, 4, 4);
  const auto proven_offset = 64u - words * 4u;
  const auto dynamic = fixture.copied_word(proven_offset);
  std::vector<Operand> color;
  if (words == 2) {
    const auto halves = fixture.unpack(fixture.load(0, 8, 1, dynamic).front());
    color = {halves[0], halves[1], halves[0], halves[1]};
  } else {
    const auto loaded = fixture.load(0, 4, static_cast<std::uint8_t>(words), dynamic);
    color.assign(loaded.begin(), loaded.end());
  }
  fixture.store_color(std::move(color));
  for (auto& operation : fixture.program.ir.operations)
    if (auto* memory = std::get_if<MemoryInfo>(&operation.info);
        memory && memory->dynamic_byte_offset)
      memory->dynamic_byte_offset_upper_bound = proven_offset;
  auto emitted = fixture.emit();
  if (const auto* error = std::get_if<Diagnostic>(&emitted))
    throw std::runtime_error(error->message);
  auto source = std::get<MslArtifact>(std::move(emitted)).source;
  // Legalization requires a small window's register offset to be bounded.
  // Replace only this synthetic offset producer with test data after emission
  // to exercise the real clamp beyond that proven range, without rewriting
  // its extent or load expressions. Production admission remains unchanged.
  const auto producer = "value0 = " + std::to_string(proven_offset) + "u;";
  const auto at = source.find(producer);
  if (at == std::string::npos || source.find(producer, at + 1) != std::string::npos)
    throw std::runtime_error("Small-window offset producer is not unique");
  source.replace(at, producer.size(), "value0 = resource2[0];");
  return source;
}

std::string gpu_fragment_source(unsigned case_index) {
  if (case_index >= 14 && case_index <= 16)
    return gpu_small_window_source(1u << (case_index - 14));
  Fixture fixture(0,
                  case_index == 2 || case_index == 7 || case_index == 9 || case_index == 13
                      ? constant_buffer_window_bytes
                      : 64,
                  true);
  std::vector<Operand> color;
  if (case_index == 9 || case_index == 13) {
    const auto offset_buffer = fixture.add_constant_buffer(1, 4, 4);
    const auto dynamic = fixture.load_from(offset_buffer, 0).front();
    if (case_index == 13) {
      const auto halves = fixture.unpack(fixture.load(0, 8, 1, dynamic).front());
      color = {halves[0], halves[1], halves[0], halves[1]};
    } else color.push_back(fixture.load(0, 4, 1, dynamic).front());
  } else if (case_index == 2) {
    const auto dynamic = fixture.copied_word(UINT32_MAX);
    color.push_back(fixture.load(0, 4, 1, dynamic).front());
  } else if (case_index == 4) {
    color.push_back(fixture.load(4, 2).front());
  } else if (case_index == 5) {
    color.push_back(fixture.load(2, 1).front());
  } else if (case_index == 6) {
    const auto halves = fixture.unpack(fixture.load(4, 8).front());
    color = {halves[0], halves[1], halves[0], halves[1]};
  } else if (case_index == 10) {
    color.push_back(fixture.load(3, 1, 1, {}, true).front());
  } else if (case_index == 11) {
    color.push_back(fixture.load(2, 2, 1, {}, true).front());
  } else if (case_index == 12) {
    color.push_back(fixture.load(3, 2).front());
  } else {
    const std::uint32_t offset = case_index == 0   ? 4
                                 : case_index == 1 ? 8
                                 : case_index == 3 ? 6
                                 : case_index == 8
                                     ? 0
                                     : constant_buffer_window_bytes - 4;
    color.push_back(fixture.load(offset).front());
  }
  fixture.store_color(std::move(color));
  auto result = fixture.emit();
  if (const auto *diagnostic = std::get_if<Diagnostic>(&result))
    throw std::runtime_error("GPU fragment emission failed: " +
                             diagnostic->message);
  return std::get<MslArtifact>(std::move(result)).source;
}

#if defined(ARMIGO_CBUF_MSL_GPU)
int run_gpu_cases();
#endif

} // namespace armigo::shader::test::cbuf_msl

#if !defined(ARMIGO_CBUF_MSL_FIXTURE_ONLY)
int main(int argc, char **argv) {
  bool run_gpu = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--metal")
      compile_metal = true;
    else if (argument == "--gpu")
      run_gpu = true;
    else if (!argument.empty() && argument[0] != '-') {
      fixture_directory = argument;
      std::filesystem::create_directories(fixture_directory);
    }
  }
  source_tests();
  validation_tests();
  try {
    const auto argument_source = armigo::shader::test::cbuf_msl::gpu_argument_buffer_source();
    check(argument_source.find("resources.resource2[16u].sample(resources.resource3[16u]") != std::string::npos,
          "GPU argument-buffer fixture samples through emitted member expressions");
    if (!fixture_directory.empty()) {
      const auto source_path = fixture_directory / "emitted_argument_buffer.metal";
      std::ofstream(source_path) << argument_source;
      if (compile_metal) {
        const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
            quote((fixture_directory / "module-cache").string()) + " -c " + quote(source_path.string()) +
            " -o " + quote((fixture_directory / "emitted_argument_buffer.air").string());
        check(std::system(command.c_str()) == 0, "Emitted GPU argument-buffer fixture compiles as Metal");
      }
    }
  } catch (const std::exception& error) { check(false, error.what()); }
#if defined(ARMIGO_CBUF_MSL_GPU)
  if (run_gpu && failures == 0)
    return armigo::shader::test::cbuf_msl::run_gpu_cases();
#endif
  if (failures)
    std::cerr << failures << " constant-buffer MSL tests failed\n";
  else
    std::cout << "Constant-buffer MSL tests passed\n";
  return failures ? 1 : 0;
}

#endif
