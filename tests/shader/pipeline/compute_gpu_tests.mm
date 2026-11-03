// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic encodings and expected results are independently authored.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pipeline/phases.h"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned output_words = 17;
using namespace armigo::shader::pipeline;
void check(bool value, const char* label) { if (!value) throw std::runtime_error(label); }
template<class T> T take(Result<T> result) {
    if (const auto* e = std::get_if<Diagnostic>(&result))
        throw std::runtime_error("phase=" + std::to_string(unsigned(e->phase)) + " pc=" +
            std::to_string(e->pc.value_or(0)) + ": " + e->message);
    return std::get<T>(std::move(result));
}
struct Facts final : FactSource {
    std::map<FactKey, FactValue> values{
        {{FactKind::compute_local_size}, ComputeLocalSize{{4, 4, 2}}},
        {{FactKind::compute_shared_memory_bytes}, std::uint32_t{192}}};
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (auto it = values.find(key); it != values.end())
            return Known<FactValue>{it->second, {OriginKind::explicit_option, "synthetic-compute"}};
        return Unknown{"Synthetic launch fact is unavailable"};
    }
};
MetalProfile profile() { return {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, true}; }
std::uint64_t pattern(std::string_view bits) {
    std::uint64_t word{};
    for (unsigned i = 0; i < bits.size(); ++i) if (bits[i] == '1') word |= 1ull << (63 - i);
    return word;
}
std::uint64_t s2r(unsigned dst, unsigned selector) {
    return pattern("1111000011001") | dst | (7ull << 16) | (std::uint64_t{selector} << 20);
}
std::uint64_t shl(unsigned dst, unsigned src, unsigned count) {
    return pattern("0011100001001") | dst | (std::uint64_t{src} << 8) | (7ull << 16) |
        (std::uint64_t{count} << 20) | (1ull << 39);
}
std::uint64_t add(unsigned dst, unsigned lhs, unsigned rhs) {
    return pattern("0101110000010") | dst | (std::uint64_t{lhs} << 8) | (7ull << 16) | (std::uint64_t{rhs} << 20);
}
std::uint64_t shared(bool store, unsigned reg, unsigned addr, unsigned offset = 0, unsigned size = 4) {
    return pattern(store ? "1110111101011" : "1110111101001") | reg |
        (std::uint64_t{addr} << 8) | (7ull << 16) | (std::uint64_t{offset} << 20) | (std::uint64_t{size} << 48);
}
std::uint64_t barrier() {
    return pattern("1111000010101") | (7ull << 16) | (7ull << 39) | (1ull << 43) | (1ull << 44);
}
// ISETP.NE.AND P0, PT, R<lhs>, RZ, PT.
std::uint64_t set_nonzero(unsigned lhs) {
    return pattern("010110110110") | 7ull | (std::uint64_t{lhs} << 8) | (7ull << 16) |
        (255ull << 20) | (7ull << 39) | (5ull << 49);
}
// Logical instruction indices omit the scheduling word in every bundle.
constexpr std::int64_t pc(std::size_t index) { return std::int64_t((index / 3) * 32 + (index % 3) * 8 + 8); }
// @P0 BRA from instruction `from` to instruction `to`.
std::uint64_t branch_if(std::size_t from, std::size_t to) {
    const auto displacement = (pc(to) - pc(from) - 8) & 0xffffff;
    return pattern("111000100100") | 15ull | (std::uint64_t(displacement) << 20);
}
std::uint64_t atomic(unsigned dst, unsigned data, unsigned size = 0, unsigned operation = 0) {
    return pattern("11101100") | dst | (255ull << 8) | (7ull << 16) | (std::uint64_t{data} << 20) |
        (std::uint64_t{size} << 28) | (32ull << 30) | (std::uint64_t{operation} << 52);
}
StageImage image(const std::vector<std::uint64_t>& instructions) {
    StageImage input; input.stage = Stage::compute;
    auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) input.bytes.push_back(std::byte((word >> (8 * i)) & 255));
    };
    for (auto word : instructions) {
        if (input.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    if (input.bytes.size() % 32 == 0) append(0);
    append(pattern("111000110000") | (7ull << 16) | 15);
    return input;
}
Result<LegalProgram> legal(const StageImage& input, const Facts& source) {
    FactReader facts(source);
    auto decoded = decode(input, facts, {});
    if (auto* e = std::get_if<Diagnostic>(&decoded)) return *e;
    auto registers = emit_ir(std::get<DecodedProgram>(decoded), facts, {});
    if (auto* e = std::get_if<Diagnostic>(&registers)) return *e;
    auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(registers)), {});
    if (auto* e = std::get_if<Diagnostic>(&ssa)) return *e;
    return legalize(std::get<SsaProgram>(std::move(ssa)), facts, profile());
}
MslArtifact finish(LegalProgram program) {
    auto structured = take(structure(std::move(program), {}));
    return take(emit_msl(structured, take(plan_resources(structured, profile())), {}));
}
void rejects(const StageImage& input, Facts facts, DiagnosticCode code, const char* label) {
    const auto result = legal(input, facts);
    const auto* error = std::get_if<Diagnostic>(&result);
    check(error && error->code == code, label);
}

MslArtifact reduction() {
    auto input = image({s2r(0, 33), s2r(1, 34), s2r(2, 35), s2r(3, 37),
        shl(5, 1, 2), shl(6, 2, 4), add(7, 0, 5), add(7, 7, 6), shl(8, 7, 2),
        shared(true, 7, 8), shared(true, 255, 255, 128), barrier(), atomic(9, 7), barrier(),
        shared(false, 10, 8) | (1ull << 44), shared(false, 11, 255, 128), s2r(12, 32), s2r(13, 0),
        shared(true, 255, 255, 160), barrier(), shared(true, 0, 0, 160, 0), barrier(),
        shared(false, 14, 255, 160), pattern("000000010000") | 15 | (7ull << 16) | (0x8001ull << 20),
        shared(true, 15, 255, 164, 2), barrier(), shared(false, 16, 255, 164, 3),
        pattern("1110111110011") | (7ull << 16),
        pattern("1110111110011") | (7ull << 16) | (1ull << 8), shared(true, 12, 8),
        s2r(17, 56), shared(true, 17, 255, 168), s2r(18, 57), shared(true, 18, 255, 168),
        s2r(19, 58), shared(true, 19, 255, 168), s2r(20, 59), shared(true, 20, 255, 168),
        s2r(21, 60), shared(true, 21, 255, 168)});
    auto program = take(legal(input, Facts{}));
    check(program.requirements.compute == ComputeConfiguration{ComputeLocalSize{{4, 4, 2}}, 192}, "Launch facts reach requirements");
    Operand x, y, z, group, packed, lane, linear, roundtrip, sum, old, bytes, half;
    bool have_linear = false;
    std::vector<Operand> masks;
    for (const auto& op : program.ir.operations) {
        if (const auto* io = std::get_if<IoInfo>(&op.info); io && op.opcode == Opcode::load_io) {
            if (io->semantic == IoSemantic::thread_id) {
                if (io->component_mask == 1) x = op.outputs[0];
                if (io->component_mask == 2) y = op.outputs[0];
                if (io->component_mask == 4) z = op.outputs[0];
            }
            if (io->semantic == IoSemantic::workgroup_id) group = op.outputs[0];
            if (io->semantic == IoSemantic::subgroup_lane_id) lane = op.outputs[0];
        }
        const auto* m = std::get_if<MemoryInfo>(&op.info);
        if (!m || m->space != AddressSpace::shared) continue;
        if (op.opcode == Opcode::store && m->dynamic_byte_offset && m->element_bytes == 4 && !have_linear) {
            linear = op.inputs[m->data[0].value]; have_linear = true;
        }
        if (op.opcode == Opcode::store && m->dynamic_byte_offset && m->element_bytes == 4)
            packed = op.inputs[m->data[0].value];
        if (op.opcode == Opcode::atomic) old = op.outputs[0];
        if (op.opcode == Opcode::store && m->byte_offset == 168) masks.push_back(op.inputs[m->data[0].value]);
        if (op.opcode == Opcode::load) {
            if (m->dynamic_byte_offset) roundtrip = op.outputs[0];
            else if (m->byte_offset == 128) sum = op.outputs[0];
            else if (m->byte_offset == 160) bytes = op.outputs[0];
            else if (m->byte_offset == 164) half = op.outputs[0];
        }
    }
    check(have_linear, "Decoded shared store retains its linear thread index");
    // The production global-address resolver deliberately rejects missing provenance.
    // A synthetic legal-IR output binding observes compute results without inventing guest addresses.
    const ResourceId output{static_cast<std::uint32_t>(program.requirements.uses.size())};
    program.requirements.uses.push_back(BufferUse{StorageBufferSource{TextureKey{0, 0}}, Access::write,
        {SizeKind::accessed_extent, 32 * output_words * 4, 4}});
    auto append = [&](Operation op) {
        const OperationId id{static_cast<std::uint32_t>(program.ir.operations.size())};
        program.ir.operations.push_back(std::move(op)); program.ir.blocks.back().operations.push_back(id);
    };
    const ValueId address{static_cast<std::uint32_t>(program.ir.values.size())};
    program.ir.values.push_back({ValueType::word32});
    append({{}, Opcode::multiply, Effects::none, {Domain::unsigned_integer, ScalarType::u32},
        {linear, LiteralBits{output_words * 4, ValueType::word32}}, {address}, {}});
    unsigned field = 0;
    std::vector<Operand> exported{x, y, z, group, packed, lane, roundtrip, sum, old, bytes, half, linear};
    check(masks.size() == 5, "Every subgroup mask remains live");
    exported.insert(exported.end(), masks.begin(), masks.end());
    for (const auto& value : exported) {
        MemoryInfo m; m.space = AddressSpace::storage; m.access = Access::write; m.resource = output;
        m.alignment = 4; m.byte_offset = field++ * 4; m.dynamic_byte_offset = InputIndex{0};
        m.dynamic_byte_offset_upper_bound = 31 * output_words * 4; m.data = {InputIndex{1}};
        append({{}, Opcode::store, Effects::write_memory, {}, {address, value}, {}, m});
    }
    return finish(std::move(program));
}

// One SHFL word: dst/src/predicate/mode plus the lane (b) and mask (c) routes.
// The b route is a 5-bit immediate when b_imm is set and a register index
// otherwise; the c route carries the 13-bit mask immediate when c_imm is set.
std::uint64_t shfl(unsigned dst, unsigned src_a, unsigned mode, unsigned pdest,
                   bool b_imm, unsigned b, bool c_imm, unsigned mask) {
    std::uint64_t word = pattern("1110111100010") | dst | (std::uint64_t{src_a} << 8) | (7ull << 16) |
        (std::uint64_t{mode} << 30) | (std::uint64_t{pdest} << 48) | (std::uint64_t{b} << 20);
    if (b_imm) word |= std::uint64_t{1} << 28;
    if (c_imm) word |= std::uint64_t{1} << 29 | (std::uint64_t{mask} << 34);
    else word |= std::uint64_t{mask} << 39;
    return word;
}

// The Maxwell shuffle contract over a 32-lane group uses the segment base from
// the mask's second byte, clamp ceiling from its low byte, per-mode source lane
// and the in-range comparison whose failure keeps the caller's own value.
std::pair<std::uint32_t, std::uint32_t> shuffle_reference(ShuffleKind kind, unsigned index, unsigned mask,
                                                          unsigned lane) {
    const unsigned clamp = mask & 0x1f, segment = (mask >> 8) & 0x1f;
    const unsigned base = lane & segment;
    const unsigned ceiling = (clamp & ~segment) | base;
    unsigned source = 0;
    bool valid = false;
    switch (kind) {
    case ShuffleKind::index:
        source = (index & ~segment) | base;
        valid = source <= ceiling;
        break;
    case ShuffleKind::up:
        source = lane - index;
        valid = static_cast<std::int32_t>(source) >= static_cast<std::int32_t>(base);
        break;
    case ShuffleKind::down:
        source = lane + index;
        valid = source <= ceiling;
        break;
    case ShuffleKind::butterfly:
        source = lane ^ index;
        valid = source <= ceiling;
        break;
    }
    return {valid ? source : lane, valid ? 0xffffffffu : 0u};
}

// A 32x1x1 compute stage executes six SHFL forms across exactly one full
// SIMD-group: index/up/down/butterfly, both immediate routes, a register lane
// route, segment boundaries and clamp ceilings, with the in-range predicate.
MslArtifact shuffle() {
    struct Case { ShuffleKind kind; bool b_imm; unsigned b; unsigned mask; };
    const std::vector<Case> cases{
        {ShuffleKind::index, true, 6, 0x001f},      {ShuffleKind::up, true, 1, 0x101f},
        {ShuffleKind::down, true, 1, 0x101f},       {ShuffleKind::butterfly, true, 16, 0x101f},
        {ShuffleKind::down, true, 8, 0x001f},       {ShuffleKind::index, false, 9, 0x001f},
    };
    std::vector<std::uint64_t> instructions{s2r(0, 0), pattern("000000010000") | 9 | (7ull << 16) | (3ull << 20)};
    for (unsigned k = 0; k < cases.size(); ++k)
        instructions.push_back(shfl(4 + static_cast<unsigned>(k), 0, static_cast<unsigned>(cases[k].kind),
                                    1 + k, cases[k].b_imm, cases[k].b, true, cases[k].mask));
    auto input = image(instructions);
    Facts facts;
    facts.values[{FactKind::compute_local_size}] = ComputeLocalSize{{32, 1, 1}};
    facts.values[{FactKind::compute_shared_memory_bytes}] = std::uint32_t{0};
    auto program = take(legal(input, facts));
    check(program.requirements.compute == ComputeConfiguration{ComputeLocalSize{{32, 1, 1}}, 0},
        "Shuffle launch facts reach requirements");
    const unsigned output_words = 1 + static_cast<unsigned>(cases.size()) * 2;
    Operand lane;
    std::vector<std::pair<Operand, Operand>> results;
    for (const auto& op : program.ir.operations) {
        if (op.opcode != Opcode::shuffle) continue;
        const auto& subgroup = std::get<SubgroupInfo>(op.info);
        check(subgroup.width == 32, "Shuffle keeps the 32-lane contract");
        results.emplace_back(op.outputs[0], op.outputs[1]);
    }
    for (const auto& op : program.ir.operations) {
        if (op.opcode != Opcode::load_io) continue;
        const auto& io = std::get<IoInfo>(op.info);
        if (io.semantic == IoSemantic::subgroup_lane_id && std::holds_alternative<ValueId>(lane) &&
            !std::get<ValueId>(lane).valid()) {
            lane = op.outputs[0];
        }
    }
    check(results.size() == cases.size(), "Every SHFL reaches the legal IR");
    check(std::holds_alternative<ValueId>(lane) && std::get<ValueId>(lane).valid(),
          "The lane builtin load reaches the legal IR");
    const ResourceId output{static_cast<std::uint32_t>(program.requirements.uses.size())};
    program.requirements.uses.push_back(BufferUse{StorageBufferSource{TextureKey{0, 0}}, Access::write,
        {SizeKind::accessed_extent, 32 * output_words * 4, 4}});
    auto append = [&](Operation op) {
        const OperationId id{static_cast<std::uint32_t>(program.ir.operations.size())};
        program.ir.operations.push_back(std::move(op)); program.ir.blocks.back().operations.push_back(id);
    };
    const ValueId address{static_cast<std::uint32_t>(program.ir.values.size())};
    program.ir.values.push_back({ValueType::word32});
    append({{}, Opcode::multiply, Effects::none, {Domain::unsigned_integer, ScalarType::u32},
        {lane, LiteralBits{output_words * 4, ValueType::word32}}, {address}, {}});
    std::vector<Operand> exported{lane};
    for (const auto& [value, valid] : results) {
        const ValueId mask{static_cast<std::uint32_t>(program.ir.values.size())};
        program.ir.values.push_back({ValueType::word32});
        append({{}, Opcode::predicate_to_mask, Effects::subgroup, {}, {valid}, {mask}, {}});
        exported.push_back(value);
        exported.push_back(mask);
    }
    unsigned field = 0;
    for (const auto& value : exported) {
        MemoryInfo m; m.space = AddressSpace::storage; m.access = Access::write; m.resource = output;
        m.alignment = 4; m.byte_offset = field++ * 4; m.dynamic_byte_offset = InputIndex{0};
        m.dynamic_byte_offset_upper_bound = 31 * output_words * 4; m.data = {InputIndex{1}};
        append({{}, Opcode::store, Effects::write_memory, {}, {address, value}, {}, m});
    }
    return finish(std::move(program));
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const MslArtifact& artifact);

void dispatch_shuffle(const MslArtifact& artifact) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    check(device != nil, "Metal device is required");
    NSError* error = nil;
    id<MTLLibrary> library = compile_library(device, artifact);
    MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
    id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library; function.name = [NSString stringWithUTF8String:artifact.entry_point.c_str()];
    MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
    check(pipeline.threadExecutionWidth == 32, "Shuffle SIMD adapter requires 32 lanes");
    constexpr unsigned output_words = 13;
    std::vector<std::uint32_t> initial(32 * output_words + 8, 0xdeadbeef);
    id<MTLBuffer> output = [device newBufferWithBytes:initial.data() length:initial.size() * 4 options:MTLResourceStorageModeShared];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new]; table_descriptor.maxBufferBindCount = 31;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new]; residency_descriptor.initialCapacity = 1;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    check(output && queue && allocator && commands && table && residency, "Metal dispatch allocation failed");
    for (const auto& resource : artifact.reflection.resources)
        if (const auto* use = std::get_if<BufferUse>(&resource.use); use && std::holds_alternative<StorageBufferSource>(use->source))
            [table setAddress:output.gpuAddress atIndex:std::get<MetalRange>(resource.location).first];
    [residency addAllocation:output]; [residency commit];
    [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [encoder endEncoding]; [commands endCommandBuffer];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0); __block bool failed = false;
    MTL4CommitOptions* commit = [MTL4CommitOptions new];
    NSArray* retained = @[library, compiler, pipeline, output, queue, allocator, commands, table, residency];
    [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count); failed = feedback.error != nil; dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands}; [queue commit:submitted count:1 options:commit];
    check(dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0 && !failed,
        "Metal shuffle execution failed or timed out");
    const auto* actual = static_cast<const std::uint32_t*>(output.contents);
    struct Case { ShuffleKind kind; unsigned b; unsigned mask; };
    const std::vector<Case> cases{
        {ShuffleKind::index, 6, 0x001f},   {ShuffleKind::up, 1, 0x101f},   {ShuffleKind::down, 1, 0x101f},
        {ShuffleKind::butterfly, 16, 0x101f}, {ShuffleKind::down, 8, 0x001f}, {ShuffleKind::index, 3, 0x001f},
    };
    std::vector<bool> seen(32);
    for (unsigned i = 0; i < 32; ++i) {
        const auto* row = actual + i * output_words;
        check(row[0] == i, "Lane export mismatch");
        for (unsigned k = 0; k < cases.size(); ++k) {
            const auto [value, mask] = shuffle_reference(cases[k].kind, cases[k].b, cases[k].mask, i);
            if (row[1 + 2 * k] != value || row[2 + 2 * k] != mask) {
                std::fprintf(stderr, "case %u lane %u: value %u expected %u, valid %u expected %u\n",
                    k, i, row[1 + 2 * k], value, row[2 + 2 * k], mask);
            }
            check(row[1 + 2 * k] == value && row[2 + 2 * k] == mask, "Shuffle result or validity mismatch");
        }
        check(!seen[row[0]], "Duplicate SIMD lane"); seen[row[0]] = true;
    }
    for (unsigned i = 32 * output_words; i < initial.size(); ++i) check(actual[i] == 0xdeadbeef, "Shuffle output guard was modified");
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const MslArtifact& artifact) {
    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0; options.mathMode = MTLMathModeSafe;
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:artifact.source.c_str()] options:options error:&error];
    if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
    return library;
}
void dispatch(const MslArtifact& artifact) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    check(device != nil, "Metal device is required");
    NSError* error = nil;
    id<MTLLibrary> library = compile_library(device, artifact);
    MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
    id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compiler_descriptor error:&error];
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = library; function.name = [NSString stringWithUTF8String:artifact.entry_point.c_str()];
    MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (!pipeline) throw std::runtime_error(error.localizedDescription.UTF8String);
    check(pipeline.threadExecutionWidth == 32, "Compute SIMD adapter requires 32 lanes");
    std::vector<std::uint32_t> initial(32 * output_words + 8, 0xdeadbeef);
    id<MTLBuffer> output = [device newBufferWithBytes:initial.data() length:initial.size() * 4 options:MTLResourceStorageModeShared];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new]; table_descriptor.maxBufferBindCount = 31;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new]; residency_descriptor.initialCapacity = 1;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    check(output && queue && allocator && commands && table && residency, "Metal dispatch allocation failed");
    for (const auto& resource : artifact.reflection.resources)
        if (const auto* use = std::get_if<BufferUse>(&resource.use); use && std::holds_alternative<StorageBufferSource>(use->source))
            [table setAddress:output.gpuAddress atIndex:std::get<MetalRange>(resource.location).first];
    [residency addAllocation:output]; [residency commit];
    [commands beginCommandBufferWithAllocator:allocator]; [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline]; [encoder setArgumentTable:table];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 4, 2)];
    [encoder endEncoding]; [commands endCommandBuffer];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0); __block bool failed = false;
    MTL4CommitOptions* commit = [MTL4CommitOptions new];
    NSArray* retained = @[library, compiler, pipeline, output, queue, allocator, commands, table, residency];
    [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count); failed = feedback.error != nil; dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands}; [queue commit:submitted count:1 options:commit];
    check(dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0 && !failed,
        "Metal compute execution failed or timed out");
    const auto* actual = static_cast<const std::uint32_t*>(output.contents);
    std::vector<bool> seen(32);
    for (unsigned i = 0; i < 32; ++i) {
        const auto* row = actual + i * output_words;
        const auto x = i % 4, y = (i / 4) % 4, z = i / 16;
        check(row[0] == x && row[1] == y && row[2] == z && row[3] == 0, "Thread/workgroup coordinates mismatch");
        check(row[4] == (x | (y << 16) | (z << 26)) && row[5] < 32, "Packed thread ID or lane mismatch");
        check(row[6] == i && row[7] == 496 && row[11] == i, "Shared roundtrip or reduction mismatch");
        check(row[8] <= 496, "Atomic return exceeds reduction");
        check(row[9] == 0x03020100 && row[10] == 0xffff8001, "Concurrent byte CAS or signed halfword mismatch");
        check(!seen[row[5]], "Duplicate SIMD lane"); seen[row[5]] = true;
        const auto eq = 1u << row[5], lt = eq - 1, le = eq | lt;
        check(row[12] == eq && row[13] == lt && row[14] == le && row[15] == ~le && row[16] == ~lt,
            "Guest subgroup masks mismatch");
    }
    for (unsigned i = 32 * output_words; i < initial.size(); ++i) check(actual[i] == 0xdeadbeef, "Compute output guard was modified");
}
} // namespace

int main() {
    @autoreleasepool {
        try {
            Facts missing; missing.values.erase({FactKind::compute_local_size});
            rejects(image({}), missing, DiagnosticCode::missing_fact, "Unknown local size must fail");
            missing = Facts{}; missing.values.erase({FactKind::compute_shared_memory_bytes});
            rejects(image({}), missing, DiagnosticCode::missing_fact, "Unknown shared allocation must fail");
            Facts invalid; invalid.values[{FactKind::compute_local_size}] = ComputeLocalSize{{0, 4, 2}};
            rejects(image({}), invalid, DiagnosticCode::unsupported_capability, "Zero launch dimension must fail");
            // The LDS U bit is decoded but unused by the emitter, so
            // a U-bit load must behave exactly like the plain load: the same
            // out-of-range diagnostic, and an in-range load that compiles.
            rejects(image({shared(false, 0, 255, 192) | (1ull << 44)}), Facts{},
                DiagnosticCode::unresolved_resource, "U-bit shared load keeps the plain overrun diagnostic");
            rejects(image({shared(false, 0, 255, 192)}), Facts{}, DiagnosticCode::unresolved_resource,
                "Plain shared load agrees with the U-bit form");
            {
                const auto u_bit_artifact = finish(take(legal(
                    image({shared(false, 0, 8) | (1ull << 44)}), Facts{})));
                check(compile_library(MTLCreateSystemDefaultDevice(), u_bit_artifact) != nil,
                    "U-bit shared load compiles as Metal 4");
            }
            rejects(image({shared(true, 254, 255, 0, 6)}), Facts{}, DiagnosticCode::unsupported_modifier,
                "Shared vector stores may not substitute registers past RZ");
            rejects(image({s2r(0, 1)}), Facts{}, DiagnosticCode::unsupported_instruction, "Unknown special register must fail");
            rejects(image({atomic(0, 255, 2)}), Facts{}, DiagnosticCode::unsupported_modifier, "64-bit atomic must fail");
            rejects(image({pattern("000000010000") | (7ull << 16) | (0xfffffffcull << 20),
                pattern("11101100") | 1 | (7ull << 16) | (255ull << 20) | (1ull << 30)}),
                Facts{}, DiagnosticCode::unresolved_resource, "ATOMS byte overflow must not alias shared word zero");
            rejects(image({pattern("11101100") | (255ull << 8) | (7ull << 16) | (255ull << 20) | (0x3fffffull << 30)}),
                Facts{}, DiagnosticCode::unsupported_modifier, "Negative ATOMS word displacement must be diagnosed");
            rejects(image({shared(false, 0, 255, 192)}), Facts{}, DiagnosticCode::unresolved_resource, "Shared allocation overrun must fail");
            {
                // armigo-shared-guard-v1: the unproven address stays admitted
                // as a guarded access and compiles; out-of-range reads return
                // zero instead of failing translation.
                const auto guarded_artifact = finish(take(legal(
                    image({s2r(0, 37), shared(false, 1, 0)}), Facts{})));
                check(compile_library(MTLCreateSystemDefaultDevice(), guarded_artifact) != nil,
                    "Unbounded shared address compiles as a guarded access");
                // A guarded word store lowers to one CAS per byte.
                const auto guarded_store = finish(take(legal(
                    image({s2r(0, 37), shared(true, 0, 0)}), Facts{})));
                check(compile_library(MTLCreateSystemDefaultDevice(), guarded_store) != nil,
                    "Unbounded shared store compiles as guarded byte stores");
            }
            for (unsigned operation : {0u, 1u, 2u, 5u, 6u, 7u, 8u}) {
                const auto size = operation == 1 || operation == 2 ? 1u : 0u;
                auto atomic_program = image({pattern("000000010000") | (7ull << 16) | (0xffffffffull << 20),
                    shared(true, 255, 255, 128), barrier(), atomic(1, 0, size, operation), barrier(),
                    shared(false, 2, 255, 128)});
                auto atomic_artifact = finish(take(legal(atomic_program, Facts{})));
                check(compile_library(MTLCreateSystemDefaultDevice(), atomic_artifact) != nil,
                    "32-bit atomic family must compile as Metal 4");
            }
            {
                // Barriers keep the guest's placement around control flow:
                // after a conditional store, and inside a branch arm.
                for (bool inside : {false, true}) {
                    auto program = inside
                        ? image({s2r(0, 33), set_nonzero(0), branch_if(2, 5),
                                 shared(true, 0, 255, 0), barrier(), shared(false, 1, 255, 0)})
                        : image({s2r(0, 33), set_nonzero(0), branch_if(2, 4),
                                 shared(true, 0, 255, 0), barrier(), shared(false, 1, 255, 0)});
                    auto barrier_artifact = finish(take(legal(program, Facts{})));
                    check(barrier_artifact.source.find("threadgroup_barrier") != std::string::npos,
                          "Guest barrier is emitted");
                    check(compile_library(MTLCreateSystemDefaultDevice(), barrier_artifact) != nil,
                          "Barrier around guest control flow compiles as Metal 4");
                }
            }
            auto artifact = reduction();
            check(artifact.reflection.stage == Stage::compute && artifact.reflection.compute.has_value(), "Compute v2 reflection is present");
            dispatch(artifact);
            dispatch_shuffle(shuffle());
            std::puts("Compute decode/IR/MSL facts, diagnostics and Metal 4 reduction passed.");
            return 0;
        } catch (const std::exception& e) { std::fprintf(stderr, "Compute test: %s\n", e.what()); return 1; }
    }
}
