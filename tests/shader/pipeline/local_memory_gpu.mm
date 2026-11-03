// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored encodings and expected results. A compute program
// stores and reloads per-thread local-memory words, including an out-of-range
// dynamic access that the emitted guard must drop, against a CPU model.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pipeline/phases.h"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

constexpr unsigned threads = 32;
constexpr unsigned row_words = 6;
constexpr std::uint32_t allocation = 1024;

void check(bool value, const char *label) { if (!value) throw std::runtime_error(label); }
template <class T> T take(Result<T> result) {
  if (const auto *e = std::get_if<Diagnostic>(&result))
    throw std::runtime_error("phase=" + std::to_string(unsigned(e->phase)) + " pc=" +
                             std::to_string(e->pc.value_or(0)) + ": " + e->message);
  return std::get<T>(std::move(result));
}

struct Facts final : FactSource {
  std::map<FactKey, FactValue> values{
      {{FactKind::compute_local_size}, ComputeLocalSize{{threads, 1, 1}}},
      {{FactKind::compute_shared_memory_bytes}, std::uint32_t{0}},
      {{FactKind::local_memory_bytes}, allocation}};
  Knowledge<FactValue> lookup(FactKey key) const override {
    if (auto it = values.find(key); it != values.end())
      return Known<FactValue>{it->second, {OriginKind::explicit_option, "synthetic-local"}};
    return Unknown{"Synthetic launch fact is unavailable"};
  }
};

MetalProfile profile() { return {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false}; }

std::uint64_t pattern(std::string_view bits) {
  std::uint64_t word{};
  for (unsigned i = 0; i < bits.size(); ++i)
    if (bits[i] == '1') word |= 1ull << (63 - i);
  return word;
}
std::uint64_t s2r(unsigned dst, unsigned selector) {
  return pattern("1111000011001") | dst | (7ull << 16) | (std::uint64_t{selector} << 20);
}
std::uint64_t shl(unsigned dst, unsigned src, unsigned count) {
  return pattern("0011100001001") | dst | (std::uint64_t{src} << 8) | (7ull << 16) |
         (std::uint64_t{count} << 20) | (1ull << 39);
}
// LDL (store=false) / STL: value register, address register, signed byte
// displacement, LS size selector (4 = b32, 0 = u8, 3 = s16).
std::uint64_t local(bool store, unsigned value_reg, unsigned address_reg,
                    unsigned offset, unsigned size) {
  return pattern(store ? "1110111101010" : "1110111101000") | value_reg |
         (std::uint64_t{address_reg} << 8) | (7ull << 16) | (std::uint64_t{offset} << 20) |
         (std::uint64_t{size} << 48);
}
StageImage image(const std::vector<std::uint64_t> &instructions) {
  StageImage input;
  input.stage = Stage::compute;
  auto append = [&](std::uint64_t word) {
    for (unsigned i = 0; i < 8; ++i)
      input.bytes.push_back(std::byte((word >> (8 * i)) & 255));
  };
  for (auto word : instructions) {
    if (input.bytes.size() % 32 == 0) append(0);
    append(word);
  }
  if (input.bytes.size() % 32 == 0) append(0);
  append(pattern("111000110000") | (7ull << 16) | 15);
  return input;
}

Result<LegalProgram> legalize_program(const StageImage &input, const Facts &source) {
  FactReader facts(source);
  auto decoded = decode(input, facts, {});
  if (auto *e = std::get_if<Diagnostic>(&decoded)) return *e;
  auto registers = emit_ir(std::get<DecodedProgram>(decoded), facts, {});
  if (auto *e = std::get_if<Diagnostic>(&registers)) return *e;
  auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(registers)), {});
  if (auto *e = std::get_if<Diagnostic>(&ssa)) return *e;
  return legalize(std::get<SsaProgram>(std::move(ssa)), facts, profile());
}

// Thread i stores its id at word i, reloads it, then exercises the guarded
// out-of-range slot, a byte slot and a signed halfword slot. Derived from the
// loaded word, the out-of-range address has no provable bound, so emission
// guards it: the store is dropped and the load yields zero.
StageImage guest_program() {
  return image({s2r(1, 33),                    // R1 = thread id
                shl(2, 1, 2),                  // R2 = 4i (word slot bytes)
                local(true, 1, 2, 0, 4),       // slot A[i] = i
                local(false, 4, 2, 0, 4),      // R4 = i
                local(false, 6, 2, 768, 4),    // R6 = 0: never stored, zero init
                shl(11, 4, 2),                 // R11 = 4i, unprovable bound
                local(true, 1, 11, 1024, 4),   // dropped: beyond allocation
                local(false, 5, 11, 1024, 4),  // R5 = 0 by the guard
                shl(3, 1, 2),                  // R3 = 4i (byte value)
                local(true, 3, 2, 1, 0),       // byte at 4i+1
                local(false, 7, 2, 1, 0),      // R7 = 4i
                shl(8, 1, 11),                 // R8 = i << 11 (bit 15 for i >= 16)
                shl(9, 1, 1),                  // R9 = 2i (halfword slot bytes)
                local(true, 8, 9, 512, 3),     // signed halfword store
                local(false, 10, 9, 512, 3)}); // R10 = sign-extended i << 11
}

std::uint32_t expected_half(unsigned i) {
  const auto narrowed = static_cast<std::int16_t>(static_cast<std::uint16_t>(i << 11));
  return static_cast<std::uint32_t>(narrowed);
}

// Observes compute results through a synthetic storage binding without
// inventing guest global addresses (see the compute reduction harness).
MslArtifact finish(LegalProgram program) {
  const ResourceId output{static_cast<std::uint32_t>(program.requirements.uses.size())};
  program.requirements.uses.push_back(BufferUse{StorageBufferSource{TextureKey{0, 0}}, Access::write,
                                                {SizeKind::accessed_extent, threads * row_words * 4, 4}});
  std::vector<Operand> exported;
  const Operation *word_load = nullptr, *oor_load = nullptr, *byte_load = nullptr, *half_load = nullptr,
                 *fresh_load = nullptr;
  Operand linear;
  bool have_linear = false;
  // Address emitters fold the signed displacement into an add feeding the
  // dynamic offset, so loads are classified by that displacement.
  auto displacement = [&](const Operation &op) {
    const auto *m = std::get_if<MemoryInfo>(&op.info);
    if (!m || !m->dynamic_byte_offset) return std::uint32_t{0};
    const auto &offset = op.inputs[m->dynamic_byte_offset->value];
    if (const auto *id = std::get_if<ValueId>(&offset))
      for (const auto &candidate : program.ir.operations) {
        if (candidate.opcode != Opcode::add || candidate.outputs.size() != 1) continue;
        const auto *out = std::get_if<ValueId>(&candidate.outputs[0]);
        if (!out || out->value != id->value) continue;
        const auto *literal = std::get_if<LiteralBits>(&candidate.inputs[1]);
        return literal ? static_cast<std::uint32_t>(literal->bits) : 0;
      }
    return std::uint32_t{0};
  };
  for (const auto &op : program.ir.operations) {
    const auto *m = std::get_if<MemoryInfo>(&op.info);
    if (!m || m->space != AddressSpace::local) continue;
    if (op.opcode == Opcode::store && !have_linear) {
      linear = op.inputs[m->data[0].value];
      have_linear = true;
      continue;
    }
    if (op.opcode != Opcode::load) continue;
    if (m->element_bytes == 1) byte_load = &op;
    else if (m->element_bytes == 2) half_load = &op;
    else if (displacement(op) == 1024) oor_load = &op;
    else if (displacement(op) == 768) fresh_load = &op;
    else word_load = &op;
  }
  check(have_linear && word_load && oor_load && byte_load && half_load && fresh_load,
        "Every local access survives legalization");
  exported.push_back(linear);
  exported.push_back(word_load->outputs[0]);
  exported.push_back(fresh_load->outputs[0]);
  exported.push_back(oor_load->outputs[0]);
  exported.push_back(byte_load->outputs[0]);
  exported.push_back(half_load->outputs[0]);
  auto append = [&](Operation op) {
    const OperationId id{static_cast<std::uint32_t>(program.ir.operations.size())};
    program.ir.operations.push_back(std::move(op));
    program.ir.blocks.back().operations.push_back(id);
  };
  const ValueId address{static_cast<std::uint32_t>(program.ir.values.size())};
  program.ir.values.push_back({ValueType::word32});
  append({{}, Opcode::multiply, Effects::none, {Domain::unsigned_integer, ScalarType::u32},
          {linear, LiteralBits{row_words * 4, ValueType::word32}}, {address}, {}});
  unsigned field = 0;
  for (const auto &value : exported) {
    MemoryInfo m;
    m.space = AddressSpace::storage;
    m.access = Access::write;
    m.resource = output;
    m.alignment = 4;
    m.byte_offset = field++ * 4;
    m.dynamic_byte_offset = InputIndex{0};
    m.dynamic_byte_offset_upper_bound = (threads - 1) * row_words * 4;
    m.data = {InputIndex{1}};
    append({{}, Opcode::store, Effects::write_memory, {}, {address, value}, {}, m});
  }
  auto structured = take(structure(std::move(program), {}));
  return take(emit_msl(structured, take(plan_resources(structured, profile())), {}));
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const MslArtifact &artifact) {
  NSError *error = nil;
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion4_0;
  options.mathMode = MTLMathModeSafe;
  id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:artifact.source.c_str()]
                                                options:options error:&error];
  if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
  return library;
}

void dispatch(const MslArtifact &artifact) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  check(device != nil, "Metal device is required");
  id<MTLLibrary> library = compile_library(device, artifact);
  MTL4CompilerDescriptor *compiler_descriptor = [MTL4CompilerDescriptor new];
  NSError *error = nil;
  id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compiler_descriptor error:&error];
  MTL4LibraryFunctionDescriptor *function = [MTL4LibraryFunctionDescriptor new];
  function.library = library;
  function.name = [NSString stringWithUTF8String:artifact.entry_point.c_str()];
  MTL4ComputePipelineDescriptor *descriptor = [MTL4ComputePipelineDescriptor new];
  descriptor.computeFunctionDescriptor = function;
  id<MTLComputePipelineState> pipeline =
      [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
  if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
  std::vector<std::uint32_t> initial(threads * row_words + 8, 0xdeadbeef);
  id<MTLBuffer> output = [device newBufferWithBytes:initial.data()
                                             length:initial.size() * 4
                                            options:MTLResourceStorageModeShared];
  id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
  id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
  id<MTL4CommandBuffer> commands = [device newCommandBuffer];
  MTL4ArgumentTableDescriptor *table_descriptor = [MTL4ArgumentTableDescriptor new];
  table_descriptor.maxBufferBindCount = 31;
  id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
  MTLResidencySetDescriptor *residency_descriptor = [MTLResidencySetDescriptor new];
  residency_descriptor.initialCapacity = 1;
  id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
  check(output && queue && allocator && commands && table && residency, "Metal dispatch allocation failed");
  for (const auto &resource : artifact.reflection.resources)
    if (const auto *use = std::get_if<BufferUse>(&resource.use);
        use && std::holds_alternative<StorageBufferSource>(use->source))
      [table setAddress:output.gpuAddress atIndex:std::get<MetalRange>(resource.location).first];
  [residency addAllocation:output];
  [residency commit];
  [commands beginCommandBufferWithAllocator:allocator];
  [commands useResidencySet:residency];
  id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  [encoder setArgumentTable:table];
  [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [encoder endEncoding];
  [commands endCommandBuffer];
  dispatch_semaphore_t completed = dispatch_semaphore_create(0);
  __block bool failed = false;
  MTL4CommitOptions *commit = [MTL4CommitOptions new];
  NSArray *retained = @[library, compiler, pipeline, output, queue, allocator, commands, table, residency];
  [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
    static_cast<void>(retained.count);
    failed = feedback.error != nil;
    dispatch_semaphore_signal(completed);
  }];
  id<MTL4CommandBuffer> submitted[] = {commands};
  [queue commit:submitted count:1 options:commit];
  check(dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0 && !failed,
        "Metal compute execution failed or timed out");
  const auto *actual = static_cast<const std::uint32_t *>(output.contents);
  for (unsigned i = 0; i < threads; ++i) {
    const auto *row = actual + i * row_words;
    const std::uint32_t expected[row_words] = {i, i, 0, 0, 4 * i, expected_half(i)};
    for (unsigned field = 0; field < row_words; ++field)
      if (row[field] != expected[field]) {
        std::fprintf(stderr, "lane %u field %u: GPU 0x%08x CPU 0x%08x\n", i, field, row[field], expected[field]);
        throw std::runtime_error("Local memory GPU mismatch");
      }
  }
  for (unsigned i = threads * row_words; i < initial.size(); ++i)
    check(actual[i] == 0xdeadbeef, "Compute output guard was modified");
}

} // namespace

int main() {
  @autoreleasepool {
    try {
      const auto artifact = finish(take(legalize_program(guest_program(), Facts{})));
      check(artifact.source.find("thread uint armigo_local_words[256] = {};") != std::string::npos,
            "Allocation declaration reaches the kernel");
      check(artifact.source.find("<= sizeof(armigo_local_words);") != std::string::npos,
            "Unproven out-of-range access is runtime guarded");
      dispatch(artifact);
      std::puts("Local memory decode/legalize/MSL Metal 4 execution passed.");
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "Local memory GPU test: %s\n", e.what());
      return 1;
    }
  }
}
