// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored synthetic tests for per-thread local-memory lowering
// and declaration.

#include "pipeline/phases.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

int failures{};
std::filesystem::path fixture_directory;
bool compile_metal{};

void check(bool value, std::string_view label) {
  if (!value) {
    std::cerr << "FAIL: " << label << '\n';
    ++failures;
  }
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

void compile_source(const std::string &source, const std::string &label) {
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

Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
NumericMode narrow(bool signed_domain, ScalarType scalar) {
  return {signed_domain ? Domain::signed_integer : Domain::unsigned_integer,
          scalar};
}

// --- legalization: the guarded/bounded split -------------------------------

class Facts final : public FactSource {
public:
  std::map<FactKey, FactValue> values;
  Knowledge<FactValue> lookup(FactKey key) const override {
    auto it = values.find(key);
    if (it == values.end())
      return Unknown{"Synthetic fact is unavailable"};
    return Known<FactValue>{it->second,
                            {OriginKind::explicit_option, "local-synthetic"}};
  }
  template <FactKind K>
  void set(FactQuery<K> key, typename FactTraits<K>::value_type value) {
    values[key.key()] = std::move(value);
  }
};

MetalProfile profile() {
  return {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
}

Program empty(Stage stage = Stage::fragment) {
  Program p;
  p.stage = stage;
  p.entry = FunctionId{0};
  p.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
  p.blocks.push_back({FunctionId{0}, {8, {}}, {}, {}, {}, Return{}});
  return p;
}

ValueId add(Program &p, Operation operation) {
  ValueId result{static_cast<std::uint32_t>(p.values.size())};
  p.values.push_back({storage_type(operation.numeric.scalar)});
  operation.outputs = {result};
  p.blocks[0].operations.push_back(
      OperationId{static_cast<std::uint32_t>(p.operations.size())});
  p.operations.push_back(std::move(operation));
  return result;
}

ValueId copy(Program &p, Operand input) {
  return add(p, {{8, InstructionName::mov}, Opcode::copy, Effects::none, {},
                 {std::move(input)}, {}, {}});
}

Operation local_access(bool store, std::uint32_t byte_offset,
                       std::uint8_t element_bytes, std::uint8_t elements,
                       NumericMode mode) {
  MemoryInfo info;
  info.space = AddressSpace::local;
  info.access = store ? Access::write : Access::read;
  info.byte_offset = byte_offset;
  info.element_bytes = element_bytes;
  info.elements = elements;
  info.alignment = element_bytes;
  Operation operation;
  operation.opcode = store ? Opcode::store : Opcode::load;
  operation.effects = required_effects(operation);
  operation.numeric = std::move(mode);
  operation.info = std::move(info);
  return operation;
}

Result<LegalProgram> run(Program p, Facts &facts) {
  FactReader reader(facts);
  return legalize({std::move(p)}, reader, profile());
}

void legalize_split_tests() {
  Facts facts;
  facts.set(FactQuery<FactKind::local_memory_bytes>{}, std::uint32_t{64});

  // A register offset with no provable bound stays runtime-checked instead of
  // failing; no bound may be invented.
  auto p = empty();
  const ValueId unknown{static_cast<std::uint32_t>(p.values.size())};
  p.values.push_back({ValueType::word32});
  p.functions[0].parameters.push_back(unknown);
  auto load = local_access(false, 0, 4, 1, narrow(false, ScalarType::u32));
  std::get<MemoryInfo>(load.info).dynamic_byte_offset = InputIndex{0};
  load.inputs.push_back(unknown);
  const auto loaded = add(p, std::move(load));
  p.functions[0].results = {ValueType::word32};
  p.blocks[0].terminator = Return{{loaded}};
  auto result = run(std::move(p), facts);
  const auto *legal = std::get_if<LegalProgram>(&result);
  check(legal, "Unbounded local offset legalizes as runtime-checked");
  if (!legal) {
    if (const auto *error = std::get_if<Diagnostic>(&result))
      std::cerr << "  " << error->message << '\n';
    return;
  }
  check(std::any_of(legal->ir.dependencies.begin(), legal->ir.dependencies.end(),
                    [](const auto &dependency) {
                      return dependency.key.kind == FactKind::local_memory_bytes &&
                             std::get<std::uint32_t>(dependency.value) == 64;
                    }),
        "Local allocation fact is a recorded dependency");
  const auto *info = std::get_if<MemoryInfo>(&legal->ir.operations.back().info);
  check(info && !info->dynamic_byte_offset_upper_bound,
        "Runtime-checked local access keeps no invented bound");

  // A masked register offset proves its bound and keeps the static check.
  auto bounded_program = empty();
  const auto masked =
      add(bounded_program, {{8, {}}, Opcode::bit_and, Effects::none,
                            narrow(false, ScalarType::u32),
                            {word(48), word(0xfffffffcu)}, {}, {}});
  auto bounded = local_access(false, 0, 4, 1, narrow(false, ScalarType::u32));
  std::get<MemoryInfo>(bounded.info).dynamic_byte_offset = InputIndex{0};
  bounded.inputs.push_back(masked);
  const auto bounded_loaded = add(bounded_program, std::move(bounded));
  bounded_program.functions[0].results = {ValueType::word32};
  bounded_program.blocks[0].terminator = Return{{bounded_loaded}};
  auto bounded_result = run(std::move(bounded_program), facts);
  if (const auto *bounded_legal = std::get_if<LegalProgram>(&bounded_result)) {
    const auto *bounded_info =
        std::get_if<MemoryInfo>(&bounded_legal->ir.operations.back().info);
    check(bounded_info && bounded_info->dynamic_byte_offset_upper_bound ==
                              std::optional<std::uint32_t>{48},
          "Proven bound is recorded on the access");
  } else {
    check(false, "Bounded local offset legalizes");
    if (const auto *error = std::get_if<Diagnostic>(&bounded_result))
      std::cerr << "  " << error->message << '\n';
  }

  // Static extents still fail against the declared allocation.
  auto overrun = local_access(false, 64, 4, 1, narrow(false, ScalarType::u32));
  overrun.inputs = {};
  auto overrun_program = empty();
  const auto overrun_loaded = add(overrun_program, std::move(overrun));
  overrun_program.functions[0].results = {ValueType::word32};
  overrun_program.blocks[0].terminator = Return{{overrun_loaded}};
  const auto overrun_result = run(std::move(overrun_program), facts);
  const auto *overrun_error = std::get_if<Diagnostic>(&overrun_result);
  check(overrun_error && overrun_error->code == DiagnosticCode::unresolved_resource,
        "Static local access beyond the allocation is rejected");
  if (overrun_error)
    std::cerr << "  note: " << overrun_error->message << '\n';

  // A proven bound must still fit: bound 48 + offset 16 + 4 bytes exceeds 64.
  auto bounded_program2 = empty();
  const auto masked2 =
      add(bounded_program2, {{8, {}}, Opcode::bit_and, Effects::none,
                             narrow(false, ScalarType::u32),
                             {word(48), word(0xfffffffcu)}, {}, {}});
  auto bounded2 = local_access(false, 16, 4, 1, narrow(false, ScalarType::u32));
  std::get<MemoryInfo>(bounded2.info).dynamic_byte_offset = InputIndex{0};
  bounded2.inputs.push_back(masked2);
  const auto bounded2_loaded = add(bounded_program2, std::move(bounded2));
  bounded_program2.functions[0].results = {ValueType::word32};
  bounded_program2.blocks[0].terminator = Return{{bounded2_loaded}};
  const auto bounded2_result = run(std::move(bounded_program2), facts);
  const auto *bounded2_error = std::get_if<Diagnostic>(&bounded2_result);
  check(bounded2_error && bounded2_error->code == DiagnosticCode::unresolved_resource,
        "Proven bound beyond the allocation is rejected");

  // armigo-shared-guard-v1: shared memory admits an unproven register offset
  // in a compute stage as a runtime-checked access; emission guards it and an
  // out-of-range load reads zero under the unbounded shared-memory access
  // lowering. Atomic shared accesses keep strict proofs.
  Facts compute_facts;
  compute_facts.set(FactQuery<FactKind::local_memory_bytes>{}, std::uint32_t{64});
  compute_facts.set(FactQuery<FactKind::compute_local_size>{},
                    ComputeLocalSize{{1, 1, 1}});
  compute_facts.set(FactQuery<FactKind::compute_shared_memory_bytes>{},
                    std::uint32_t{64});
  auto shared_program = empty(Stage::compute);
  auto shared_load = local_access(false, 0, 4, 1, narrow(false, ScalarType::u32));
  {
    auto &info = std::get<MemoryInfo>(shared_load.info);
    info.space = AddressSpace::shared;
    info.dynamic_byte_offset = InputIndex{0};
  }
  shared_load.inputs.push_back(word(0)); // Replaced below by a parameter.
  const ValueId parameter{static_cast<std::uint32_t>(shared_program.values.size())};
  shared_program.values.push_back({ValueType::word32});
  shared_program.functions[0].parameters.push_back(parameter);
  shared_load.inputs[0] = parameter;
  const auto shared_loaded = add(shared_program, std::move(shared_load));
  shared_program.functions[0].results = {ValueType::word32};
  shared_program.blocks[0].terminator = Return{{shared_loaded}};
  const auto shared_result = run(std::move(shared_program), compute_facts);
  check(std::holds_alternative<LegalProgram>(shared_result),
        "Shared memory admits an unproven offset as a guarded access");
  const auto &shared_legal = std::get<LegalProgram>(shared_result);
  bool shared_guarded = false;
  for (const auto &operation : shared_legal.ir.operations) {
    const auto *info = std::get_if<MemoryInfo>(&operation.info);
    if (info && info->space == AddressSpace::shared)
        shared_guarded = !info->dynamic_byte_offset_upper_bound.has_value();
  }
  check(shared_guarded, "The guarded shared load records no invented offset bound");

  // A zero-byte allocation cannot express a guarded access.
  Facts empty_facts;
  empty_facts.set(FactQuery<FactKind::local_memory_bytes>{}, std::uint32_t{0});
  auto zero_program = empty();
  const ValueId zero_unknown{static_cast<std::uint32_t>(zero_program.values.size())};
  zero_program.values.push_back({ValueType::word32});
  zero_program.functions[0].parameters.push_back(zero_unknown);
  auto zero_load = local_access(false, 0, 4, 1, narrow(false, ScalarType::u32));
  std::get<MemoryInfo>(zero_load.info).dynamic_byte_offset = InputIndex{0};
  zero_load.inputs.push_back(zero_unknown);
  const auto zero_loaded = add(zero_program, std::move(zero_load));
  zero_program.functions[0].results = {ValueType::word32};
  zero_program.blocks[0].terminator = Return{{zero_loaded}};
  const auto zero_result = run(std::move(zero_program), empty_facts);
  const auto *zero_error = std::get_if<Diagnostic>(&zero_result);
  check(zero_error && zero_error->code == DiagnosticCode::invalid_fact,
        "Runtime-checked access with a zero allocation is rejected");
}

// The stage-buffer contract is a distinct header-sized policy, not an
// invented bound for ordinary shared-memory instructions.
void stage_buffer_policy_tests() {
  Facts facts;
  facts.set(FactQuery<FactKind::compute_local_size>{}, ComputeLocalSize{{1, 1, 1}});
  facts.set(FactQuery<FactKind::compute_shared_memory_bytes>{}, std::uint32_t{64});
  const auto make = [](StageBufferAccess region, Operand address = ValueId{0}) {
    auto p = empty(Stage::compute);
    p.values.push_back({ValueType::word32});
    p.functions[0].parameters.push_back(ValueId{0});
    auto load = local_access(false, 0, 4, 1, narrow(false, ScalarType::u32));
    auto &memory = std::get<MemoryInfo>(load.info);
    memory.space = AddressSpace::shared;
    memory.dynamic_byte_offset = InputIndex{0};
    memory.stage_buffer = region;
    load.inputs = {address};
    const auto loaded = add(p, std::move(load));
    p.functions[0].results = {ValueType::word32};
    p.blocks[0].terminator = Return{{loaded}};
    return p;
  };
  const StageBufferAccess region{StageBufferPolicy::header_direct_v1, 16, 32, 64};
  auto legal = run(make(region), facts);
  check(std::holds_alternative<LegalProgram>(legal),
        "Dynamic stage-buffer indexing accepts the declared region contract");
  if (const auto *accepted = std::get_if<LegalProgram>(&legal)) {
    const auto &memory = std::get<MemoryInfo>(accepted->ir.operations.back().info);
    check(memory.stage_buffer && !memory.dynamic_byte_offset_upper_bound,
          "Stage-buffer contract records no invented dynamic bound");
  }
  for (const auto address : {12u, 48u, 0xfffffffcu}) {
    const auto result = run(make(region, word(address)), facts);
    const auto *error = std::get_if<Diagnostic>(&result);
    check(error && error->code == DiagnosticCode::unsupported_capability,
          "Constant stage-buffer access cannot cross either region boundary");
  }
  check(std::holds_alternative<LegalProgram>(run(make(region, word(44)), facts)),
        "Last complete region word is admitted");
  for (const auto invalid : {
           StageBufferAccess{StageBufferPolicy::header_direct_v1, 16, 32, 68},
           StageBufferAccess{StageBufferPolicy::header_direct_v1, 16, 52, 64},
           StageBufferAccess{StageBufferPolicy::header_direct_v1, 17, 32, 64},
           StageBufferAccess{static_cast<StageBufferPolicy>(2), 16, 32, 64}}) {
    check(std::holds_alternative<Diagnostic>(run(make(invalid), facts)),
          "Invalid allocation, extent, alignment or version is rejected");
  }
  auto local = make(region);
  std::get<MemoryInfo>(local.operations.back().info).space = AddressSpace::local;
  check(std::holds_alternative<Diagnostic>(run(std::move(local), facts)),
        "Stage-buffer policy cannot bypass local-memory checks");
  auto vector = make(region);
  std::get<MemoryInfo>(vector.operations.back().info).elements = 2;
  check(std::holds_alternative<Diagnostic>(run(std::move(vector), facts)),
        "Stage-buffer policy cannot admit a vector memory operation");
}

void native_shared_atomic_policy_tests() {
  Facts facts;
  facts.set(FactQuery<FactKind::compute_local_size>{},
            ComputeLocalSize{{32, 1, 1}});
  facts.set(FactQuery<FactKind::compute_shared_memory_bytes>{},
            std::uint32_t{64});
  const auto make = [](bool native, std::optional<std::uint32_t> literal = {}) {
    auto p = empty(Stage::compute);
    p.values.push_back({ValueType::word32});
    p.functions[0].parameters.push_back(ValueId{0});
    const auto address = add(p, {{8, {}},
                                 Opcode::bit_and,
                                 Effects::none,
                                 {Domain::bits, ScalarType::u32},
                                 {ValueId{0}, word(0xfffffffcu)},
                                 {},
                                 {}});
    MemoryInfo memory;
    memory.space = AddressSpace::shared;
    memory.access = Access::read_write;
    memory.scope = MemoryScope::workgroup;
    memory.atomic = AtomicKind::add;
    memory.alignment = 4;
    memory.dynamic_byte_offset = InputIndex{0};
    memory.data = {InputIndex{1}};
    if (native)
      memory.native_shared_atomic_policy =
          NativeSharedAtomicPolicy::word_address_v1;
    Operation operation{
        {16, {}},
        Opcode::atomic,
        Effects::none,
        {Domain::unsigned_integer, ScalarType::u32},
        {literal ? Operand{word(*literal)} : Operand{address}, word(1)},
        {},
        memory};
    operation.effects = required_effects(operation);
    const auto old = add(p, std::move(operation));
    p.functions[0].results = {ValueType::word32};
    p.blocks[0].terminator = Return{{old}};
    return p;
  };
  check(std::holds_alternative<Diagnostic>(run(make(false), facts)),
        "Legacy shared atomics retain static address bounds");
  check(std::holds_alternative<LegalProgram>(run(make(false, 60), facts)),
        "Legacy bounded atomics are unchanged");
  const auto legal = run(make(true), facts);
  check(std::holds_alternative<LegalProgram>(legal),
        "Native word atomic accepts a runtime aligned index");
  if (const auto *accepted = std::get_if<LegalProgram>(&legal)) {
    const auto &info =
        std::get<MemoryInfo>(accepted->ir.operations.back().info);
    check(info.native_shared_atomic_policy &&
              !info.dynamic_byte_offset_upper_bound,
          "Native word addressing carries no invented upper bound");
    auto structured = structure(*accepted, accepted->ir.limits);
    check(std::holds_alternative<StructuredProgram>(structured),
          "Native atomic policy structures successfully");
    if (const auto *source = std::get_if<StructuredProgram>(&structured)) {
      check(std::holds_alternative<ResourcePlan>(
                plan_resources(*source, profile())),
            "Planning independently accepts native policy");
      for (unsigned mutation = 0; mutation < 10; ++mutation) {
        auto changed = *source;
        auto &operation = changed.ir.operations.back();
        auto &m = std::get<MemoryInfo>(operation.info);
        switch (mutation) {
        case 0:
          m.native_shared_atomic_policy =
              static_cast<NativeSharedAtomicPolicy>(2);
          break;
        case 1:
          m.space = AddressSpace::local;
          break;
        case 2:
          m.atomic = AtomicKind::minimum;
          break;
        case 3:
          m.alignment = 1;
          break;
        case 4:
          m.order = MemoryOrder::acquire_release;
          break;
        case 5:
          m.scope = MemoryScope::device;
          break;
        case 6:
          m.dynamic_byte_offset_upper_bound = 60;
          break;
        case 7:
          m.byte_offset = 4;
          break;
        case 8:
          operation.numeric = {Domain::signed_integer, ScalarType::s32};
          break;
        case 9:
          changed.ir.operations.front().inputs[1] = word(0xffffffffu);
          break;
        }
        check(std::holds_alternative<Diagnostic>(
                  plan_resources(changed, profile())),
              "Independent planning rejects malformed native policy or lost "
              "alignment");
      }
      auto constant = *source;
      constant.ir.operations.back().inputs[0] = word(64);
      check(std::holds_alternative<Diagnostic>(
                plan_resources(constant, profile())),
            "Planning rejects a constant beyond the workgroup allocation");
    }
  }
  for (const auto address : {64u, 65u, 0xfffffffcu})
    check(std::holds_alternative<Diagnostic>(run(make(true, address), facts)),
          "Native policy refuses invalid constant addresses");
  check(std::holds_alternative<LegalProgram>(run(make(true, 60), facts)),
        "Native policy accepts the final complete word");
  auto unaligned = make(true);
  unaligned.operations.front().inputs[1] = word(0xffffffffu);
  check(std::holds_alternative<Diagnostic>(run(std::move(unaligned), facts)),
        "Native policy requires a real word-alignment proof");
}

// --- MSL text emission ------------------------------------------------------

struct Fixture {
  StructuredProgram program;
  ResourcePlan plan;
  std::uint32_t allocation_bytes{64};

  explicit Fixture(bool color_output = false) {
    program.ir.stage = Stage::fragment;
    program.ir.entry = FunctionId{0};
    program.ir.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
    program.ir.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
    program.regions = {BlockRegion{BlockId{0}}};
    program.functions = {{FunctionId{0}, RegionId{0}}};
    program.profile = {"synthetic-metal4", 1, 31, 128, 16, true,
                       false, true, false, false, false};
    plan.stage = Stage::fragment;
    plan.profile = program.profile;
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

  // Records the allocation dependency the way legalization leaves it.
  void declare_allocation() {
    program.ir.dependencies.push_back({FactKey{FactKind::local_memory_bytes},
                                       allocation_bytes,
                                       {OriginKind::validated_header, "sph"}});
    plan.dependencies = program.ir.dependencies;
  }

  ValueId value(ValueType type = ValueType::word32) {
    const ValueId id{static_cast<std::uint32_t>(program.ir.values.size())};
    program.ir.values.push_back({type, ValueKind::temporary});
    return id;
  }

  OperationId add(Operation operation) {
    operation.effects = required_effects(operation);
    operation.source = {40 + static_cast<Pc>(program.ir.operations.size() * 8),
                        InstructionName::ldl};
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
                            std::optional<std::uint32_t> bound = {},
                            bool signed_narrow = false) {
    MemoryInfo memory;
    memory.space = AddressSpace::local;
    memory.access = Access::read;
    memory.byte_offset = byte_offset;
    memory.element_bytes = element_bytes;
    memory.elements = elements;
    memory.alignment = element_bytes;
    Operation operation;
    operation.opcode = Opcode::load;
    const auto narrow_scalar = element_bytes == 1
        ? (signed_narrow ? ScalarType::s8 : ScalarType::u8)
        : (signed_narrow ? ScalarType::s16 : ScalarType::u16);
    operation.numeric = element_bytes == 4 ? narrow(false, ScalarType::u32)
                                           : narrow(signed_narrow, narrow_scalar);
    if (dynamic) {
      memory.dynamic_byte_offset = InputIndex{0};
      operation.inputs.push_back(*dynamic);
    }
    memory.dynamic_byte_offset_upper_bound = bound;
    std::vector<ValueId> results;
    for (unsigned i = 0; i < elements; ++i) {
      results.push_back(value());
      operation.outputs.push_back(results.back());
    }
    operation.info = memory;
    add(std::move(operation));
    return results;
  }

  void store(std::vector<Operand> data, std::uint32_t byte_offset,
             std::uint8_t element_bytes = 4,
             std::optional<ValueId> dynamic = {},
             std::optional<std::uint32_t> bound = {}) {
    MemoryInfo memory;
    memory.space = AddressSpace::local;
    memory.access = Access::write;
    memory.byte_offset = byte_offset;
    memory.element_bytes = element_bytes;
    memory.elements = static_cast<std::uint8_t>(data.size());
    memory.alignment = element_bytes;
    Operation operation;
    operation.opcode = Opcode::store;
    if (dynamic) {
      memory.dynamic_byte_offset = InputIndex{0};
      operation.inputs.push_back(*dynamic);
    }
    for (auto &item : data) {
      memory.data.push_back(
          InputIndex{static_cast<std::uint16_t>(operation.inputs.size())});
      operation.inputs.push_back(std::move(item));
    }
    memory.dynamic_byte_offset_upper_bound = bound;
    operation.info = memory;
    add(std::move(operation));
  }

  void store_color(std::vector<Operand> values) {
    const auto repeating = values;
    while (values.size() < 4)
      values.push_back(repeating.empty() ? Operand{word(0)}
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
    std::cerr << "FAIL: " << label << ": pc=" << diagnostic->pc.value_or(0)
              << ": " << diagnostic->message << '\n';
    ++failures;
    return {};
  }
  auto artifact = std::get<MslArtifact>(std::move(result));
  const auto &dependencies = artifact.reflection.dependencies;
  check(std::any_of(dependencies.begin(), dependencies.end(),
                    [](const auto &dependency) {
                      return dependency.key.kind == FactKind::local_memory_bytes;
                    }),
        label + " keeps the allocation dependency in reflection");
  compile_source(artifact.source, label);
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

void emission_tests() {
  // Declaration: the allocation rounds up to words and zero-initializes.
  Fixture declaration;
  declaration.declare_allocation();
  const auto stored = declaration.copied_word(0x12345678u);
  declaration.store({stored}, 8);
  auto source = accepted(declaration, "local_declaration");
  check(source.find("thread uint armigo_local_words[16] = {};") !=
                std::string::npos,
        "Allocation declares a zero-initialized word array");
  check(source.find("armigo_local_words[(local_address1 + 0ul) / 4ul] = "
                    "value0;") != std::string::npos,
        "Static aligned store writes its containing word directly");
  check(source.find("local_inside") == std::string::npos,
        "Static access emits no runtime guard");

  // A 12-byte allocation rounds up to three words; zero bytes declare nothing.
  Fixture rounded;
  rounded.allocation_bytes = 12;
  rounded.declare_allocation();
  rounded.store({rounded.copied_word(1)}, 8);
  source = accepted(rounded, "local_rounded_allocation");
  check(source.find("thread uint armigo_local_words[3] = {};") !=
                std::string::npos,
        "Non-word allocation rounds up to whole words");
  Fixture unallocated;
  unallocated.allocation_bytes = 0;
  unallocated.declare_allocation();
  unallocated.store({unallocated.copied_word(1)}, 0);
  rejected(std::move(unallocated),
           "Emission requires a nonzero recorded allocation",
           DiagnosticCode::missing_fact);
  Fixture missing;
  missing.store({missing.copied_word(1)}, 0);
  rejected(std::move(missing),
           "Emission requires the allocation dependency",
           DiagnosticCode::missing_fact);

  // Bounded dynamic offset: legalized bound, no runtime guard.
  Fixture bounded;
  bounded.declare_allocation();
  const auto bounded_offset = bounded.copied_word(0);
  bounded.load(0, 4, 1, bounded_offset, 60);
  source = accepted(bounded, "local_bounded_dynamic");
  check(source.find("local_inside") == std::string::npos &&
            source.find("% 4ul) * 8ul") != std::string::npos,
        "Bounded dynamic access loads without a guard");

  // Unproven dynamic offset: the guard covers the whole access.
  Fixture guarded{true};
  guarded.declare_allocation();
  const auto offset = guarded.copied_word(0);
  const auto loaded = guarded.load(0, 4, 1, offset).front();
  guarded.store({guarded.copied_word(7)}, 0, 4, offset);
  guarded.store_color({loaded});
  source = accepted(guarded, "local_guarded_dynamic");
  check(source.find("const bool local_inside1 = local_address1 + 4ul <= "
                    "sizeof(armigo_local_words);") != std::string::npos,
        "Guard covers the whole vector access");
  check(source.find("(local_inside1 ? ") != std::string::npos &&
            source.find(" : 0u)") != std::string::npos,
        "Out-of-range load yields zero");
  check(occurrences(source, "if (local_inside") == 1,
        "Out-of-range store is dropped by one guard");

  // Narrow static accesses extract from and insert into their word.
  Fixture narrow;
  narrow.declare_allocation();
  narrow.load(3, 1);
  narrow.load(2, 2, 1, {}, {}, true);
  narrow.store({narrow.copied_word(0xabu)}, 1, 1);
  source = accepted(narrow, "local_narrow_static");
  check(source.find("& 3ul) * 8ul") != std::string::npos &&
            source.find("255u") != std::string::npos &&
            source.find("as_type<short>(ushort(") != std::string::npos,
        "Static narrow load extracts and sign-extends");
  check(source.find("& ~(255u << 8u)") != std::string::npos &&
            source.find("<< 8u) & (255u << 8u)") != std::string::npos,
        "Static byte store merges into its containing word");

  // Dynamic word access lowers as byte-wise little-endian assembly.
  Fixture dynamic_word{true};
  dynamic_word.declare_allocation();
  const auto word_offset = dynamic_word.copied_word(0);
  const auto word_loaded = dynamic_word.load(0, 4, 1, word_offset).front();
  dynamic_word.store({dynamic_word.copied_word(0xdeadbeefu)}, 0, 4, word_offset);
  dynamic_word.store_color({word_loaded});
  source = accepted(dynamic_word, "local_dynamic_word");
  check(occurrences(source, "% 4ul) * 8ul") >= 8,
        "Dynamic word access addresses individual bytes");
  check(source.find("& 255u) << 8u") != std::string::npos,
        "Dynamic word load assembles little-endian bytes");
  check(source.find(">> 8u) & 255u)") != std::string::npos,
        "Dynamic word store distributes little-endian bytes");

  // 64-bit vectors use two consecutive words when statically aligned.
  Fixture wide{true};
  wide.declare_allocation();
  const auto wide_loaded = wide.load(8, 4, 2);
  wide.store_color({wide_loaded.front()});
  source = accepted(wide, "local_word64");
  check(occurrences(source, "/ 4ul]") == 2 &&
            source.find("+ 4ul)") != std::string::npos,
        "B64 accesses two consecutive words");

  // Unsupported shapes are explicit diagnostics.
  Fixture wide_element;
  wide_element.declare_allocation();
  MemoryInfo invalid;
  invalid.space = AddressSpace::local;
  invalid.access = Access::read;
  invalid.byte_offset = 0;
  invalid.element_bytes = 8;
  invalid.elements = 1;
  Operation operation;
  operation.opcode = Opcode::load;
  operation.outputs.push_back(wide_element.value(ValueType::word64));
  operation.info = invalid;
  wide_element.add(std::move(operation));
  rejected(std::move(wide_element), "64-bit elements have no guest encoding",
           DiagnosticCode::unsupported_capability);
  Fixture modified;
  modified.declare_allocation();
  const auto modified_offset = modified.copied_word(0);
  modified.load(0, 4, 1, modified_offset, 60);
  auto &info =
      std::get<MemoryInfo>(modified.program.ir.operations.back().info);
  info.is_volatile = true;
  rejected(std::move(modified), "Local memory modifiers are unsupported",
           DiagnosticCode::unsupported_modifier);
}

} // namespace

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--metal")
      compile_metal = true;
    else if (!argument.empty() && argument[0] != '-') {
      fixture_directory = argument;
      std::filesystem::create_directories(fixture_directory);
    }
  }
  legalize_split_tests();
  stage_buffer_policy_tests();
  native_shared_atomic_policy_tests();
  emission_tests();
  if (failures)
    std::cerr << failures << " local memory tests failed\n";
  else
    std::cout << "Local memory legalization and MSL tests passed\n";
  return failures ? 1 : 0;
}
