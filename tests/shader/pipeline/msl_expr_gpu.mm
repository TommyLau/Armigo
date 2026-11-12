// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// GPU checks for the pure pipeline generators, following the established
// operation harness pattern: a single-block fragment program reads
// constant-buffer words, evaluates the generator under test, and exports the
// raw result word through a 32-bit float target; the CPU reference is the
// tested Armigo helper with the identical contract. NaN results never reach
// the render target: every case guards the export with the NaN test.
#include "pipeline/phases.h"
#include "vertex_builtins_fixture.h"

#include <armigo/shader/fma.h>
#include <armigo/shader/rne_multiply.h>
#include "graphics_operations.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

constexpr NSUInteger kWidth = 4;
constexpr NSUInteger kHeight = 4;
constexpr NSUInteger kPixelBytes = 4 * sizeof(std::uint32_t);

Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
NumericMode floating() { return {Domain::floating, ScalarType::f32, Rounding::nearest_even}; }

struct BuiltinDraw {
    unsigned component{};
    bool indexed{};
    int vertex_base{};
    unsigned instance_base{};
    unsigned instance_count{1};
};

// A point-rasterization case: the translated point_vertex fixture supplies
// the vertex stage (its guest 0x6c export only when shader_size is set, the
// armigo_point_size constant otherwise) and a hand-written fragment exports
// a constant or its [[point_coord]]. The pixel check compares the sprite
// geometry instead of one uniform expected word.
struct PointDraw {
    bool shader_size{};
    float size{3.0f};
    bool point_coord{};
};

struct Case {
    std::string name;
    // Fills one computation over the loaded words value0..value3; the builder
    // appends its fresh result, wraps it in an explicit NaN guard, and stores
    // the guarded word to all four color components. Integer results may be
    // NaN patterns as floats, so those cases guard on the input instead.
    std::function<void(Operation&, std::vector<ValueId>&)> build;
    std::vector<std::uint32_t> constants;
    std::uint32_t expected;
    bool guard_on_input{};
    // Optional real guest instruction; constants then initialize R0..R3.
    std::uint64_t instruction{};
    bool query_lod{};
    float result_tolerance{};
    std::optional<BuiltinDraw> builtin{};
    bool compare_words{};
    bool gather_offsets{};
    // A comparison result: convert the predicate through the all-bits mask and
    // export it without the NaN guard, which would distort a full mask.
    bool predicate_word{};
    // A carry operation: the result word is followed by its carry predicate;
    // export_carry exports that predicate instead of the word.
    bool carry_output{};
    bool export_carry{};
    // Optional follow-up instruction consuming a predicate the guest
    // instruction wrote, so paired-predicate results reach a register.
    std::uint64_t post_instruction{};
    std::optional<PointDraw> point{};
};

// Compare before float export so NaN payloads, signed zeros and integer words
// cannot be normalized by the render target. Only the finite pass/fail value
// crosses that boundary; the decoded arithmetic remains unchanged.
void compare_color_words(StructuredProgram& program, std::uint32_t expected) {
    auto& ir = program.ir;
    for (auto& block : ir.blocks) {
        std::vector<OperationId> operations;
        for (const auto id : block.operations) {
            auto store = ir.operations[id.value];
            if (store.opcode == Opcode::store_io &&
                std::get<IoInfo>(store.info).semantic == IoSemantic::color) {
                for (const auto index : std::get<IoInfo>(store.info).data) {
                    const ValueId matches{static_cast<std::uint32_t>(ir.values.size())};
                    ir.values.push_back({ValueType::predicate, ValueKind::temporary});
                    const ValueId exported{static_cast<std::uint32_t>(ir.values.size())};
                    ir.values.push_back({ValueType::word32, ValueKind::temporary});
                    Operation compare;
                    compare.source = store.source;
                    compare.opcode = Opcode::compare;
                    compare.numeric = {Domain::unsigned_integer, ScalarType::u32};
                    compare.inputs = {store.inputs[index.value], word(expected)};
                    compare.outputs = {matches};
                    compare.info = ComparisonInfo{Compare::equal, false};
                    operations.push_back(OperationId{static_cast<std::uint32_t>(ir.operations.size())});
                    ir.operations.push_back(std::move(compare));
                    Operation select;
                    select.source = store.source;
                    select.opcode = Opcode::select;
                    select.inputs = {matches, word(0x3f800000u), word(0x40000000u)};
                    select.outputs = {exported};
                    operations.push_back(OperationId{static_cast<std::uint32_t>(ir.operations.size())});
                    ir.operations.push_back(std::move(select));
                    store.inputs[index.value] = exported;
                }
                ir.operations[id.value] = std::move(store);
            }
            operations.push_back(id);
        }
        block.operations = std::move(operations);
    }
}

// Encoded regressions keep inputs dynamic all the way through the production
// decoder/emitter and the remaining translation phases. Missing word facts
// prevent offline folding from concealing an MSL backend defect.
Result<MslArtifact> guest_fragment(std::uint64_t instruction, bool query_lod = false,
                                  std::optional<std::uint32_t> expected_word = {},
                                  bool gather_offsets = false,
                                  std::uint64_t post_instruction = 0) {
    struct Facts final : FactSource {
        Knowledge<FactValue> lookup(FactKey key) const override {
            const EvidenceOrigin origin{OriginKind::explicit_option, "synthetic-alu-gpu"};
            if (key.kind == FactKind::constant_buffer_bound_mask) return Known<FactValue>{std::uint32_t{1}, origin};
            if (key.kind == FactKind::constant_buffer_size) return Known<FactValue>{std::uint32_t{64}, origin};
            if (key.kind == FactKind::fragment_output_type) return Known<FactValue>{IoType{ScalarType::f32, 4}, origin};
            if (key.kind == FactKind::texture_handle_cbuf) return Known<FactValue>{std::uint32_t{2}, origin};
            if (key == fact_query::texture({2, 32}).key())
                return Known<FactValue>{TextureDescriptor{{TextureDimension::d2, false, false},
                    ScalarType::f32, ComponentFormat::floating, true, false, 1, "synthetic-lod-v1"}, origin};
            return Unknown{"Runtime register input"};
        }
    } source;
    auto take = []<class T>(Result<T> result) -> T {
        if (const auto* error = std::get_if<Diagnostic>(&result)) throw *error;
        return std::get<T>(std::move(result));
    };
    try {
        StageImage image;
        image.stage = Stage::fragment;
        auto append = [&](std::uint64_t word) {
            for (unsigned i = 0; i < 8; ++i) image.bytes.push_back(std::byte((word >> (8 * i)) & 255));
        };
        auto emit = [&](std::uint64_t word) {
            if (image.bytes.size() % 32 == 0) append(0);
            append(word);
        };
        for (unsigned reg = 0; reg != 4; ++reg)
            emit(0xef9400000007ff00ull | reg | (std::uint64_t{reg * 4} << 20));
        if (query_lod) {
            // IPA reads pixel X/Y, then FMUL scales both coordinates by R2.
            // A 32-wide texture gives rho=32*scale and lambda=log2(rho).
            for (unsigned axis = 0; axis != 2; ++axis) {
                const auto reg = 4 + axis;
                emit(0xe000000000070000ull | reg | (255ull << 8) | (255ull << 20) |
                     (std::uint64_t{0x70 + axis * 4} << 28) | (255ull << 39) | (7ull << 47));
                emit(0x5c68000000070000ull | reg | (std::uint64_t{reg} << 8) | (2ull << 20));
            }
        }
        if (gather_offsets) {
            // IPA reads pixel X/Y for the gather coordinate, then MOV32I fills
            // both packed per-texel offset registers with zero offsets.
            for (unsigned axis = 0; axis != 2; ++axis) {
                const auto reg = 4 + axis;
                emit(0xe000000000070000ull | reg | (255ull << 8) | (255ull << 20) |
                     (std::uint64_t{0x70 + axis * 4} << 28) | (255ull << 39) | (7ull << 47));
            }
            for (unsigned reg = 6; reg != 8; ++reg)
                emit((1ull << 56) | (7ull << 16) | reg);
        }
        emit(instruction);
        if (post_instruction) emit(post_instruction);
        if (query_lod) {
            // Export signed fixed-point values numerically so negative words do
            // not become NaNs when written through a floating color target.
            emit(0x5cb8000000072a00ull); // I2F.F32.S32 R0, R0, nearest-even.
        }
        for (unsigned component = 0; component != 4; ++component)
            emit(0xeff000000007ff00ull | (std::uint64_t{0x80 + component * 4} << 20));
        emit(0xe30000000007000full);
        FactReader facts(source);
        const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        auto decoded = take(decode(image, facts, {}));
        auto registers = take(emit_ir(decoded, facts, {}));
        auto ssa = take(construct_ssa(std::move(registers), {}));
        auto legal = take(legalize(std::move(ssa), facts, profile));
        auto structured = take(structure(std::move(legal), {}));
        if (expected_word) compare_color_words(structured, *expected_word);
        auto plan = take(plan_resources(structured, profile));
        return emit_msl(structured, plan, {});
    } catch (const Diagnostic& error) {
        return error;
    }
}

// A single-block fragment program with one color output and a constant
// buffer; the loaded words feed the case computation, and the guarded result
// fills all four color components.
struct FragmentBuilder {
    StructuredProgram p;
    ResourcePlan plan;
    ResourceId buffer{};
    FragmentBuilder() {
        p.ir.stage = Stage::fragment;
        p.ir.entry = FunctionId{0};
        p.ir.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
        p.ir.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
        p.regions = {BlockRegion{BlockId{0}}};
        p.functions = {{FunctionId{0}, RegionId{0}}};
        p.profile = {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        plan.stage = Stage::fragment;
        plan.profile = p.profile;
        IoUse color;
        color.semantic = IoSemantic::color;
        color.access = Access::write;
        color.mask = 15;
        color.type = {ScalarType::f32, 4};
        p.requirements.io.push_back(color);
        plan.io.push_back(color);
        BufferUse use;
        use.source = ConstantBufferSource{0};
        use.access = Access::read;
        use.size = {SizeKind::constant_buffer_window, 64, 4, ConstantBufferPolicy::padded_zero_words_v2};
        buffer = ResourceId{static_cast<std::uint32_t>(p.requirements.uses.size())};
        p.requirements.uses.push_back(use);
        plan.resources.push_back({buffer, use, MetalRange{MetalNamespace::buffer, 0, 1}});
        // Constant-buffer windows require the draw-time size table.
        BufferUse support;
        support.source = SupportBufferSource{std::string(constant_buffer_support_schema), 1};
        support.access = Access::read;
        support.size = {SizeKind::accessed_extent, 72, 4};
        const ResourceId support_id{static_cast<std::uint32_t>(p.requirements.uses.size())};
        p.requirements.uses.push_back(support);
        plan.resources.push_back({support_id, support, MetalRange{MetalNamespace::buffer, 1, 1}});
    }
    ValueId value(ValueType type = ValueType::word32) {
        ValueId id{static_cast<std::uint32_t>(p.ir.values.size())};
        p.ir.values.push_back({type, ValueKind::temporary});
        return id;
    }
    OperationId add(Operation op) {
        op.effects = required_effects(op);
        op.source = {40 + static_cast<Pc>(p.ir.operations.size() * 8), InstructionName::mov};
        OperationId id{static_cast<std::uint32_t>(p.ir.operations.size())};
        p.ir.operations.push_back(std::move(op));
        p.ir.blocks[0].operations.push_back(id);
        return id;
    }
    std::vector<ValueId> load_words(unsigned count) {
        std::vector<ValueId> loaded;
        for (unsigned i = 0; i < count; ++i) {
            MemoryInfo memory;
            memory.space = AddressSpace::constant_buffer;
            memory.access = Access::read;
            memory.constant_buffer_slot = 0;
            memory.byte_offset = 4 * i;
            memory.element_bytes = 4;
            memory.resource = buffer;
            Operation op;
            op.opcode = Opcode::load;
            op.outputs.push_back(value());
            op.info = memory;
            loaded.push_back(std::get<ValueId>(op.outputs.front()));
            add(std::move(op));
        }
        return loaded;
    }
    ValueId compute(const Case& item) {
        auto loaded = load_words(4);
        Operation compute;
        item.build(compute, loaded);
        auto result = value(item.predicate_word ? ValueType::predicate : ValueType::word32);
        compute.outputs.push_back(result);
        if (item.carry_output) {
            const auto carry = value(ValueType::predicate);
            compute.outputs.push_back(carry);
            if (item.export_carry) result = carry;
        }
        add(std::move(compute));
        ValueId exported;
        if (item.predicate_word || item.export_carry) {
            // A comparison exports through its all-bits mask, then as one of
            // two finite words: a raw full mask would be a NaN float at the
            // render target, whose payload the target does not promise.
            const auto mask = value();
            Operation to_mask;
            to_mask.opcode = Opcode::predicate_to_mask;
            to_mask.inputs.push_back(result);
            to_mask.outputs.push_back(mask);
            add(std::move(to_mask));
            const auto condition = value(ValueType::predicate);
            Operation to_predicate;
            to_predicate.opcode = Opcode::mask_to_predicate;
            to_predicate.inputs.push_back(mask);
            to_predicate.outputs.push_back(condition);
            add(std::move(to_predicate));
            exported = value();
            Operation select;
            select.opcode = Opcode::select;
            select.inputs.push_back(condition);
            select.inputs.push_back(word(0x3f800000u));
            select.inputs.push_back(word(0x40000000u));
            select.outputs.push_back(exported);
            add(std::move(select));
        } else {
            // The render target does not promise NaN payloads, so the exported
            // word is guarded: the NaN test selects a zero word instead.
            const auto guard = value(ValueType::predicate);
            Operation test;
            test.opcode = Opcode::is_nan;
            test.numeric = floating();
            test.inputs.push_back(item.guard_on_input ? loaded.front() : result);
            test.outputs.push_back(guard);
            add(std::move(test));
            exported = value();
            Operation select;
            select.opcode = Opcode::select;
            select.inputs.push_back(guard);
            select.inputs.push_back(word(0));
            select.inputs.push_back(result);
            select.outputs.push_back(exported);
            add(std::move(select));
        }
        IoInfo io;
        io.semantic = IoSemantic::color;
        io.type = {ScalarType::f32, 4};
        io.component_mask = 15;
        Operation store;
        store.opcode = Opcode::store_io;
        for (unsigned i = 0; i < 4; ++i) {
            store.inputs.push_back(exported);
            io.data.push_back(InputIndex{static_cast<std::uint16_t>(i)});
        }
        store.info = io;
        add(std::move(store));
        return exported;
    }
    Result<MslArtifact> emit(std::optional<std::uint32_t> expected_word = {}) {
        // Raw-word cases compare inside the program against the expected
        // word, exactly like the decoded-instruction path: a NaN result must
        // not be normalized by the render target or masked by the export
        // guard before the comparison.
        if (expected_word) compare_color_words(p, *expected_word);
        return emit_msl(p, plan, {});
    }
};

StructuredProgram vertex_program() {
    StructuredProgram p;
    p.ir.stage = Stage::vertex;
    p.ir.entry = FunctionId{0};
    p.ir.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
    p.ir.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
    p.regions = {BlockRegion{BlockId{0}}};
    p.functions = {{FunctionId{0}, RegionId{0}}};
    p.profile = {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
    const Dependency depth{{FactKind::depth_convention}, DepthConvention::negative_one_to_one,
                           {OriginKind::explicit_option, "synthetic-depth"}};
    p.ir.dependencies.push_back(depth);
    IoUse position;
    position.semantic = IoSemantic::position;
    position.access = Access::write;
    position.mask = 15;
    position.type = {ScalarType::f32, 4};
    p.requirements.io.push_back(position);
    IoUse vertex_id;
    vertex_id.semantic = IoSemantic::vertex_id;
    vertex_id.access = Access::read;
    vertex_id.mask = 1;
    vertex_id.type = {ScalarType::u32, 1};
    p.requirements.io.push_back(vertex_id);

    auto value = [&](ValueType type = ValueType::word32) {
        ValueId id{static_cast<std::uint32_t>(p.ir.values.size())};
        p.ir.values.push_back({type, ValueKind::temporary});
        return id;
    };
    auto add = [&](Operation op) {
        op.effects = required_effects(op);
        op.source = {40 + static_cast<Pc>(p.ir.operations.size() * 8), InstructionName::mov};
        OperationId id{static_cast<std::uint32_t>(p.ir.operations.size())};
        p.ir.operations.push_back(std::move(op));
        p.ir.blocks[0].operations.push_back(id);
    };
    // Position vertices by vertex id: the three lanes spread into a triangle
    // covering the whole render target, exactly like the operation harness.
    IoInfo id_info;
    id_info.semantic = IoSemantic::vertex_id;
    id_info.type = {ScalarType::u32, 1};
    const auto lane = value();
    Operation load_id;
    load_id.opcode = Opcode::load_io;
    load_id.outputs.push_back(lane);
    load_id.info = id_info;
    add(std::move(load_id));
    const NumericMode unsigned_mode{Domain::unsigned_integer, ScalarType::u32};
    auto compare_to = [&](ValueId lhs, std::uint32_t bits) {
        const auto result = value(ValueType::predicate);
        Operation compare;
        compare.opcode = Opcode::compare;
        compare.numeric = unsigned_mode;
        compare.inputs.push_back(lhs);
        compare.inputs.push_back(word(bits));
        compare.outputs.push_back(result);
        compare.info = ComparisonInfo{Compare::equal, false};
        add(std::move(compare));
        return result;
    };
    auto choose = [&](ValueId condition, std::uint32_t when_true, Operand when_false) {
        const auto result = value();
        Operation select;
        select.opcode = Opcode::select;
        select.inputs.push_back(condition);
        select.inputs.push_back(word(when_true));
        select.inputs.push_back(std::move(when_false));
        select.outputs.push_back(result);
        add(std::move(select));
        return result;
    };
    const auto is_zero = compare_to(lane, 0);
    const auto is_one = compare_to(lane, 1);
    const auto wide_open = std::bit_cast<std::uint32_t>(-2.0f);
    const auto far_edge = std::bit_cast<std::uint32_t>(6.0f);
    const auto x = choose(is_zero, wide_open, choose(is_one, far_edge, word(wide_open)));
    const auto y = choose(is_zero, wide_open, choose(is_one, wide_open, word(far_edge)));
    const auto zero = value();
    Operation z;
    z.opcode = Opcode::copy;
    z.inputs.push_back(word(std::bit_cast<std::uint32_t>(0.0f)));
    z.outputs.push_back(zero);
    add(std::move(z));
    const auto one = value();
    Operation w;
    w.opcode = Opcode::copy;
    w.inputs.push_back(word(std::bit_cast<std::uint32_t>(1.0f)));
    w.outputs.push_back(one);
    add(std::move(w));
    IoInfo io;
    io.semantic = IoSemantic::position;
    io.type = {ScalarType::f32, 4};
    io.component_mask = 15;
    Operation store;
    store.opcode = Opcode::store_io;
    store.inputs.push_back(x);
    store.inputs.push_back(y);
    store.inputs.push_back(zero);
    store.inputs.push_back(one);
    for (unsigned i = 0; i < 4; ++i) io.data.push_back(InputIndex{static_cast<std::uint16_t>(i)});
    store.info = io;
    add(std::move(store));
    return p;
}

id<MTLLibrary> library(id<MTLDevice> device, const std::string& source) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = MTLLanguageVersion4_0;
    // The translated stages pin safe math with contraction off in their
    // prelude; the probe kernels must observe the same arithmetic policy.
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    id<MTLLibrary> result = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:options error:&error];
    if (!result) std::fprintf(stderr, "Metal source compilation failed: %s\n",
                              error.localizedDescription.UTF8String);
    return result;
}

// Metal's native words for one unary float expression over several input
// words, produced by a one-line probe kernel on this device. The probe uses
// the same fast-math compile options as the translated stages, so these words
// define the results that native operations' guest draws must reproduce.
std::vector<std::uint32_t> native_words(id<MTLDevice> device, id<MTL4Compiler> compiler,
                                        const std::string& expression,
                                        const std::vector<std::uint32_t>& inputs) {
    const std::string source =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void armigo_native_probe(device uint* results [[buffer(0)]],\n"
        "                                const device uint* words [[buffer(1)]],\n"
        "                                uint lane [[thread_position_in_threadgroup]]) {\n"
        "    const float x = as_type<float>(words[lane]);\n"
        "    results[lane] = " + expression + ";\n}\n";
    const auto probe_library = library(device, source);
    NSError* error = nil;
    MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
    function.library = probe_library;
    function.name = @"armigo_native_probe";
    MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
    descriptor.computeFunctionDescriptor = function;
    id<MTLComputePipelineState> pipeline =
        [compiler newComputePipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    id<MTLBuffer> word_buffer = [device newBufferWithBytes:inputs.data()
        length:inputs.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> result_buffer = [device newBufferWithLength:inputs.size() * sizeof(std::uint32_t)
        options:MTLResourceStorageModeShared];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxBufferBindCount = 2;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 2;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    if (!probe_library || !pipeline || !word_buffer || !result_buffer || !queue || !allocator ||
        !commands || !table || !residency) {
        throw std::runtime_error("native MUFU probe could not be created");
    }
    [table setAddress:result_buffer.gpuAddress atIndex:0];
    [table setAddress:word_buffer.gpuAddress atIndex:1];
    [residency addAllocation:word_buffer];
    [residency addAllocation:result_buffer];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setArgumentTable:table];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(inputs.size()), 1, 1)];
    [encoder endEncoding];
    [commands endCommandBuffer];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block bool failed = false;
    MTL4CommitOptions* commit = [MTL4CommitOptions new];
    NSArray* retained = @[probe_library, pipeline, word_buffer, result_buffer, queue, allocator,
                          commands, table, residency];
    [commit addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        failed = feedback.error != nil;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:commit];
    if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0 || failed) {
        throw std::runtime_error("native MUFU probe execution failed or timed out");
    }
    std::vector<std::uint32_t> results(inputs.size(), 0);
    std::memcpy(results.data(), result_buffer.contents, inputs.size() * sizeof(std::uint32_t));
    return results;
}

std::vector<Case> make_cases(id<MTLDevice> device, id<MTL4Compiler> compiler) {
    std::vector<Case> cases;
    // Case stores a C label, so dynamically named cases borrow these strings;
    // the vector lives as long as the case table.
    std::vector<std::string> case_names;
    { // Flushed addition: subnormal inputs become signed zeros.
        Case item{"ftz_add",
                  [](Operation& self, std::vector<ValueId>& loaded) {
                      auto mode = floating();
                      mode.flush_inputs = mode.flush_output = true;
                      self.opcode = Opcode::add;
                      self.numeric = mode;
                      self.inputs.push_back(loaded[0]);
                      self.inputs.push_back(loaded[1]);
                  },
                  {0x00400000u, 0x00400000u}, 0};
        cases.push_back(std::move(item));
    }
    { // The emitter leaves the subnormal flush of arithmetic to the GPU, so
      // pin the device behavior through the emitted MSL: each operand or
      // result here is a subnormal that the guest flush turns into zero.
        constexpr std::uint32_t subnormal = 0x00400000u;
        constexpr std::uint32_t tiny = 0x1e3ce508u;        // 1e-20: its square is subnormal
        constexpr std::uint32_t power_100 = 0x71800000u;   // 2^100
        const auto flushed = [&](const char* name, Opcode opcode, std::vector<std::uint32_t> words,
                                 std::uint32_t expected, bool saturate = false) {
            Case item{name,
                      [opcode, count = words.size(), saturate](Operation& self, std::vector<ValueId>& loaded) {
                          auto mode = floating();
                          mode.flush_inputs = mode.flush_output = true;
                          mode.saturate = saturate;
                          self.opcode = opcode;
                          self.numeric = mode;
                          for (std::size_t i = 0; i != count; ++i) self.inputs.push_back(loaded[i]);
                      },
                      std::move(words), expected};
            item.compare_words = true;
            cases.push_back(std::move(item));
        };
        flushed("ftz_multiply_subnormal_result", Opcode::multiply, {tiny, tiny}, 0u);
        flushed("ftz_multiply_subnormal_input", Opcode::multiply, {subnormal, power_100}, 0u);
        flushed("ftz_fma_subnormal_input", Opcode::fused_multiply_add, {subnormal, power_100, 0u}, 0u);
        flushed("ftz_ceiling_subnormal", Opcode::ceiling, {subnormal}, 0u);
        flushed("ftz_rsqrt_subnormal", Opcode::reciprocal_sqrt, {subnormal}, 0x7f800000u);
        flushed("ftz_saturate_subnormal_input", Opcode::multiply, {subnormal, power_100}, 0u, true);
        Case greater{"ftz_compare_subnormal_above_zero",
                     [](Operation& self, std::vector<ValueId>& loaded) {
                         auto mode = floating();
                         mode.flush_inputs = true;
                         self.opcode = Opcode::compare;
                         self.numeric = mode;
                         self.inputs.push_back(loaded[0]);
                         self.inputs.push_back(loaded[1]);
                         self.info = ComparisonInfo{Compare::greater, false};
                     },
                     {subnormal, 0u}, 0x40000000u};
        greater.predicate_word = true;
        cases.push_back(std::move(greater));
    }
    { // Nearest-even product by the tested integer-assisted helper.
        Case item{"multiply_rne",
                  [](Operation& self, std::vector<ValueId>& loaded) {
                      self.opcode = Opcode::multiply;
                      self.numeric = floating();
                      self.inputs.push_back(loaded[0]);
                      self.inputs.push_back(loaded[1]);
                  },
                  {0x3f800000u, 0x40490fdbu}, armigo::shader::multiply_rne(0x3f800000u, 0x40490fdbu)};
        cases.push_back(std::move(item));
    }
    { // Fused product with one rounding and signed flush.
        Case item{"fma_ftz",
                  [](Operation& self, std::vector<ValueId>& loaded) {
                      self.opcode = Opcode::fused_multiply_add;
                      auto mode = floating();
                      mode.flush_inputs = mode.flush_output = true;
                      self.numeric = mode;
                      self.inputs.push_back(loaded[0]);
                      self.inputs.push_back(loaded[1]);
                      self.inputs.push_back(loaded[2]);
                  },
                  {0x3f800000u, 0x40000000u, 0xc0490fdbu},
                  armigo::shader::fma_rne_ftz(0x3f800000u, 0x40000000u, 0xc0490fdbu)};
        cases.push_back(std::move(item));
    }
    { // Float-to-integer truncation by the tested decomposition.
        const std::uint32_t source = std::bit_cast<std::uint32_t>(-3.7f);
        const std::uint32_t profile = 1u | 2u | (3u << 3);
        Case item{"f32_to_s32_trunc",
                  [](Operation& self, std::vector<ValueId>& loaded) {
                      self.opcode = Opcode::convert;
                      self.numeric = {Domain::floating, ScalarType::f32, Rounding::toward_zero};
                      self.numeric.flush_inputs = true;
                      self.inputs.push_back(loaded[0]);
                      self.info = ConversionInfo{ScalarType::f32, ScalarType::s32};
                  },
                  {source},
                  armigo::shader::graphics::detail::evaluate_float_to_integer({source, 0, 0}, profile), true};
        cases.push_back(std::move(item));
    }
    { // Integer-to-float by the tested bit assembly.
        const std::uint32_t source = 0xdeadbeefu;
        Case item{"u32_to_f32_rne",
                  [](Operation& self, std::vector<ValueId>& loaded) {
                      self.opcode = Opcode::convert;
                      self.numeric = floating();
                      self.inputs.push_back(loaded[0]);
                      self.info = ConversionInfo{ScalarType::u32, ScalarType::f32};
                  },
                  {source}, armigo::shader::graphics::detail::evaluate_integer_to_float({source, 0, 0}, 4u)};
        cases.push_back(std::move(item));
    }
    { // Carry arithmetic adds or subtracts the carry predicate as one unit
      // and reports the 33rd bit: the carry term must not absorb the sum.
        const auto carry = [&](const char* name, Opcode opcode, std::uint32_t left, std::uint32_t right, bool incoming,
                               std::uint32_t expected, bool export_carry) {
            Case item{name,
                      [opcode, incoming](Operation& self, std::vector<ValueId>& loaded) {
                          self.opcode = opcode;
                          self.numeric = {Domain::unsigned_integer, ScalarType::u32};
                          self.inputs.push_back(loaded[0]);
                          self.inputs.push_back(loaded[1]);
                          self.inputs.push_back(LiteralBits{incoming ? 1u : 0u, ValueType::predicate});
                      },
                      {left, right}, expected};
            item.carry_output = true;
            item.export_carry = export_carry;
            item.compare_words = !export_carry;
            cases.push_back(std::move(item));
        };
        carry("add_carry_sum", Opcode::add_carry, 2u, 3u, false, 5u, false);
        carry("add_carry_sum_with_carry_in", Opcode::add_carry, 2u, 3u, true, 6u, false);
        carry("add_carry_wrapped_sum", Opcode::add_carry, 0xffffffffu, 1u, true, 1u, false);
        carry("add_carry_out", Opcode::add_carry, 0xffffffffu, 1u, true, 0x3f800000u, true);
        carry("add_carry_no_out", Opcode::add_carry, 2u, 3u, true, 0x40000000u, true);
        carry("subtract_borrow_difference", Opcode::subtract_borrow, 5u, 3u, true, 1u, false);
        carry("subtract_borrow_out", Opcode::subtract_borrow, 0u, 1u, false, 0x3f800000u, true);
    }
    { // The signed most-significant bit of a negative word is its highest
      // zero bit; zero and minus one report none.
        const auto msb = [&](const char* name, std::uint32_t source, std::uint32_t expected) {
            Case item{name,
                      [](Operation& self, std::vector<ValueId>& loaded) {
                          self.opcode = Opcode::find_msb;
                          self.numeric = {Domain::signed_integer, ScalarType::s32};
                          self.inputs.push_back(loaded[0]);
                      },
                      {source}, expected};
            // The no-bit result is a NaN pattern, so the export guard tests
            // the (finite) input instead of the result.
            item.compare_words = true;
            item.guard_on_input = true;
            cases.push_back(std::move(item));
        };
        msb("find_msb_signed_negative", 0x81010101u, 30u);
        msb("find_msb_signed_minimum", 0x80000000u, 30u);
        msb("find_msb_signed_minus_two", 0xfffffffeu, 0u);
        msb("find_msb_signed_zero", 0u, 0xffffffffu);
        msb("find_msb_signed_positive", 0x00012345u, 16u);
    }
    { // Draw and export sanity: a literal word without any load.
        Case item{"literal_export",
                  [](Operation& self, std::vector<ValueId>&) {
                      self.opcode = Opcode::copy;
                      self.inputs.push_back(word(0x3f800000u));
                  },
                  {0u, 0u, 0u, 0u}, 0x3f800000u};
        cases.push_back(std::move(item));
    }
    { // One NaN extremum operand yields the other under the tested contract.
        Case item{"minimum_nan",
                  [](Operation& self, std::vector<ValueId>& loaded) {
                      self.opcode = Opcode::minimum;
                      self.numeric = floating();
                      self.numeric.flush_inputs = true;
                      self.inputs.push_back(loaded[0]);
                      self.inputs.push_back(loaded[1]);
                  },
                  {0x7fc00000u, 0x3f800000u}, 0x3f800000u};
        cases.push_back(std::move(item));
    }
    { // Guest saturation is the native unit clamp: over-range clamps to one,
      // negatives clamp to plus zero, and NaN clamps to zero.
        const auto saturating = [&](const char* name, std::uint32_t multiplicand,
                                    std::uint32_t expected) {
            Case item{name,
                      [](Operation& self, std::vector<ValueId>& loaded) {
                          auto mode = floating();
                          mode.saturate = true;
                          self.opcode = Opcode::multiply;
                          self.numeric = mode;
                          self.inputs.push_back(loaded[0]);
                          self.inputs.push_back(loaded[1]);
                      },
                      {multiplicand, 0x3f800000u}, expected};
            item.compare_words = true;
            cases.push_back(std::move(item));
        };
        saturating("saturate_above_one", 0x40000000u, 0x3f800000u);
        saturating("saturate_negative", 0xc0000000u, 0u);
        saturating("saturate_nan", 0x7fc00000u, 0u);
    }
    { // The guest clamp over the precise composition min(max(value, low),
      // high): a NaN value yields the low bound, and the result stays a raw
      // word comparison, because the export guard would turn an erroneous
      // propagated NaN into the same zero the low bound demands. A -0 value
      // against a +0 low bound clamps to plus zero; reversed signs clamp to
      // the -0 high bound.
        const auto clamp_case = [&](const char* name, std::array<std::uint32_t, 3> words,
                                    std::uint32_t expected) {
            Case item{name,
                      [](Operation& self, std::vector<ValueId>& loaded) {
                          self.opcode = Opcode::clamp;
                          self.numeric = floating();
                          for (const auto index : {0u, 1u, 2u}) self.inputs.push_back(loaded[index]);
                      },
                      {words[0], words[1], words[2]}, expected};
            item.compare_words = true;
            cases.push_back(std::move(item));
        };
        // Ties between zeros of opposite sign are the device's choice (see
        // shader-float-min-max.md), so no case pins them.
        clamp_case("clamp_nan_yields_low", {0x7fc00000u, 0u, 0x3f800000u}, 0u);
        clamp_case("clamp_into_range", {0xc0490fdbu, 0xbf800000u, 0x40400000u}, 0xbf800000u);
        clamp_case("clamp_above_range", {0x40490fdbu, 0xbf800000u, 0x40400000u}, 0x40400000u);
    }
    { // Ordered comparisons are false whenever a NaN participates; the
      // unordered flag adds the explicit NaN terms. Ordered NE excludes NaN
      // operands explicitly, so NE with a NaN operand is false.
      // Each case exports one of two finite words: true selects 1.0f, false
      // selects 2.0f.
        const auto fsetp = [&](std::string name, Compare relation, bool unordered,
                               std::uint32_t left, std::uint32_t right, bool expected) {
            case_names.push_back(std::move(name));
            Case item{case_names.back().c_str(),
                      [relation, unordered](Operation& self, std::vector<ValueId>& loaded) {
                          self.opcode = Opcode::compare;
                          self.numeric = floating();
                          self.inputs.push_back(loaded[0]);
                          self.inputs.push_back(loaded[1]);
                          self.info = ComparisonInfo{relation, unordered};
                      },
                      {left, right}, expected ? 0x3f800000u : 0x40000000u};
            item.predicate_word = true;
            cases.push_back(std::move(item));
        };
        constexpr std::uint32_t nan_word = 0x7fc00000u;
        constexpr std::uint32_t one = 0x3f800000u;
        constexpr std::uint32_t two = 0x40000000u;
        const std::array<std::pair<Compare, const char*>, 6> relations{{
            {Compare::less, "lt"}, {Compare::equal, "eq"}, {Compare::less_equal, "le"},
            {Compare::greater, "gt"}, {Compare::not_equal, "ne"}, {Compare::greater_equal, "ge"},
        }};
        for (const auto& [relation, suffix] : relations) {
            // NaN on the left, finite on the right: ordered false, unordered
            // true, for every relation.
            fsetp(std::string("fsetp_ordered_") + suffix + "_nan_left", relation, false,
                  nan_word, one, false);
            fsetp(std::string("fsetp_unordered_") + suffix + "_nan_left", relation, true,
                  nan_word, one, true);
            // NaN on the right covers the other operand position.
            fsetp(std::string("fsetp_ordered_") + suffix + "_nan_right", relation, false,
                  one, nan_word, false);
            fsetp(std::string("fsetp_unordered_") + suffix + "_nan_right", relation, true,
                  one, nan_word, true);
            // Distinct finite operands (one below two): the unordered flag
            // changes nothing while both operands are finite.
            const bool holds_below = relation == Compare::less ||
                                     relation == Compare::less_equal ||
                                     relation == Compare::not_equal;
            fsetp(std::string("fsetp_ordered_") + suffix + "_finite", relation, false,
                  one, two, holds_below);
            fsetp(std::string("fsetp_unordered_") + suffix + "_finite", relation, true,
                  one, two, holds_below);
            // Equal operands: true only for the inclusive relations, ordered
            // and unordered alike.
            fsetp(std::string("fsetp_ordered_") + suffix + "_equal", relation, false,
                  one, one, relation == Compare::equal || relation == Compare::less_equal ||
                                relation == Compare::greater_equal);
            fsetp(std::string("fsetp_unordered_") + suffix + "_equal", relation, true,
                  one, one, relation == Compare::equal || relation == Compare::less_equal ||
                                relation == Compare::greater_equal);
        }
    }
    const auto guest = [&](const char* name, std::uint64_t instruction,
                           std::vector<std::uint32_t> constants, std::uint32_t expected) {
        cases.push_back(Case{name, {}, std::move(constants), expected, false, instruction});
    };
    guest("decoded_i2i_signed_clamp", 0x5ce4000000173500ull, {0, 1, 0, 0}, 1);
    guest("decoded_f2i_signed_clamp", 0x5cb0100000171900ull, {0, 0x3f800000, 0, 0}, 1);
    guest("decoded_bfe_width32", 0x3800000200070100ull, {0, 0x89abcdef, 0, 0}, 0x89abcdef);
    guest("decoded_bfi_width32", 0x36f0010200070100ull, {0, 0x89abcdef, 0x12345678, 0}, 0x89abcdef);
    guest("decoded_bfe_dynamic_width32", 0x5c00000000270100ull, {0, 0x89abcdef, 0x2000, 0}, 0x89abcdef);
    guest("decoded_bfi_dynamic_width32", 0x5bf0018000270100ull, {0, 0x89abcdef, 0x2000, 0x12345678}, 0x89abcdef);
    guest("decoded_bfe_signed_width32", 0x3801000200070100ull, {0, 0x89abcdef, 0, 0}, 0x89abcdef);
    guest("decoded_shf_left_zero", 0x5bf8018000270100ull, {0, 1, 0, 2}, 2);
    guest("decoded_shf_right_zero", 0x5cf8018000270100ull, {0, 1, 0, 2}, 1);
    { // I2F of a halfway magnitude under the toward-zero rounding field: the
      // exact integer-assisted helper assembles the bits on the guest grid,
      // 16777218 = 0x4b800001, where the native nearest-even cast would give
      // 0x4b800002. The expectation is the tested CPU conversion, independent
      // of any device behavior.
        const std::uint32_t halfway = 16777219;
        const auto expected = armigo::shader::graphics::detail::evaluate_integer_to_float(
            {halfway, 0, 0}, 4u | (3u << 3));
        guest("decoded_i2f_rz", 0x5cb8018000170a00ull, {0, halfway, 0, 0}, expected);
    }
    guest("decoded_fmul_fmz", 0x5c68200000270100ull, {0, 0, 0x7f800000, 0}, 0);
    guest("decoded_fmul_fmz_zero_nan", 0x5c68200000270100ull, {0, 0, 0x7fc00000, 0}, 0);
    guest("decoded_ffma_fmz", 0x59c0018000270100ull, {0, 0, 0x7f800000, 0x40400000}, 0x40400000);
    guest("decoded_ffma_fmz_subnormal_addend", 0x59c0018000270100ull, {0, 0, 0x7f800000, 1}, 0);
    guest("decoded_ffma_fmz_negative_subnormal_addend", 0x59c0018000270100ull,
          {0, 0, 0x7f800000, 0x80000001}, 0x80000000);
    guest("decoded_f2f_pass_ftz", 0x5ca8100000170a00ull, {0, 1, 0, 0}, 0);
    guest("decoded_f2f_pass_negative_ftz", 0x5ca8100000170a00ull, {0, 0x80000001, 0, 0}, 0x80000000);
    const auto encoded = [](std::string_view bits) {
        std::uint64_t word{};
        for (unsigned i = 0; i < bits.size(); ++i)
            if (bits[i] == '1') word |= std::uint64_t{1} << (63 - i);
        return word;
    };
    const auto add = encoded("0101110001011") | (1ull << 8) | (2ull << 20) | (7ull << 16) |
                     (1ull << 44) | (1ull << 46) | (1ull << 48);
    guest("decoded_fadd_abs_neg_ftz_zero", add, {0, 0x80000001, 0x80000000, 0}, 0x80000000);
    guest("decoded_fadd_abs_neg_ftz_value", add, {0, 0xc0000000, 0x40a00000, 0}, 0x40400000);
    const auto fma = encoded("010110011") | (1ull << 8) | (2ull << 20) | (3ull << 39) |
                     (7ull << 16) | (1ull << 48) | (1ull << 49) | (1ull << 53);
    guest("decoded_ffma_neg_ftz", fma, {0, 0x3f800000, 0x40000000, 0x40400000}, 0xc0a00000);
    const auto hadd = encoded("0101110100010") | (1ull << 8) | (2ull << 20) | (7ull << 16);
    guest("decoded_hadd_pack", hadd, {0, 0x40003c00, 0x44004200, 0}, 0x46004400);
    guest("decoded_hadd_f32", hadd | (1ull << 49), {0, 0x40003c00, 0x44004200, 0}, 0x40800000);
    guest("decoded_hadd_merge_low", hadd | (2ull << 49), {0x4c004800, 0x40003c00, 0x44004200, 0}, 0x4c004400);
    guest("decoded_hadd_merge_high", hadd | (3ull << 49), {0x4c004800, 0x40003c00, 0x44004200, 0}, 0x46004800);
    guest("decoded_hadd_ftz_keeps_high_subnormal", hadd | (2ull << 49) | (1ull << 39),
          {0x00014800, 0x40003c00, 0x44004200, 0}, 0x00014400);
    guest("decoded_hadd_ftz_keeps_low_subnormal", hadd | (3ull << 49) | (1ull << 39),
          {0x4c000001, 0x40003c00, 0x44004200, 0}, 0x46000001);
    guest("decoded_hadd_keeps_low_nan_payload", hadd | (3ull << 49),
          {0x4c007c01, 0x40003c00, 0x44004200, 0}, 0x46007c01);
    const auto compact_hadd = encoded("0111101x0") | (1ull << 8) | (7ull << 16) |
                              (0xf0ull << 20) | (0x100ull << 30);
    guest("decoded_hadd_compact", compact_hadd, {0, 0x40003c00, 0, 0}, 0x44004000);
    const auto hadd32 = encoded("0010110") | (1ull << 8) | (7ull << 16) | (0x40003c00ull << 20);
    guest("decoded_hadd_immediate32", hadd32, {0, 0x40003c00, 0, 0}, 0x44004000);
    // HSETP2 draws compare the unpacked binary32 lanes through the production
    // decoder: NaN, signed zero, infinity and subnormal halves, with the
    // primary predicate exported through SEL between the loaded operand
    // registers. The expected word is whichever raw register the predicate
    // selects; the in-program word comparison keeps NaN patterns away from
    // the float render target. Both guest instructions predicate on P7 so
    // neither runs behind a transient guard.
    const auto hsetp2 = [&](std::uint64_t condition, std::uint64_t modifiers) {
        return encoded("0101110100100") | (2ull << 0) | (4ull << 3) | (1ull << 8) |
               (2ull << 20) | (7ull << 16) | (condition << 35) | (7ull << 39) |
               (3ull << 45) | modifiers;
    };
    const auto predicate_select = encoded("0101110010100") | (1ull << 8) | (2ull << 20) |
                                  (7ull << 16) | (4ull << 39);
    const auto guest_predicate = [&](const char* name, std::uint64_t condition,
                                     std::uint64_t modifiers, std::uint32_t word_a,
                                     std::uint32_t word_b, bool expected_true) {
        Case item{name, {}, {0, word_a, word_b, 0}, expected_true ? word_a : word_b, false,
                  hsetp2(condition, modifiers)};
        item.post_instruction = predicate_select;
        item.compare_words = true;
        cases.push_back(std::move(item));
    };
    // Binary16 NaN against zero: ordered comparisons are false, unordered true.
    guest_predicate("decoded_hsetp2_nan_ordered_lt", 1, 0, 0xfc007c01u, 0x00003c00u, false);
    guest_predicate("decoded_hsetp2_nan_unordered_ltu", 9, 0, 0x3c003c00u, 0xfc007c01u, true);
    // Infinities compare by magnitude in binary32.
    guest_predicate("decoded_hsetp2_infinity_lt", 1, 0, 0x3c00fc00u, 0x40003c00u, true);
    // Signed zeros are equal and never less in binary32.
    guest_predicate("decoded_hsetp2_signed_zero_eq", 2, 0, 0x00000000u, 0x00008000u, true);
    guest_predicate("decoded_hsetp2_signed_zero_lt", 1, 0, 0x00008000u, 0x00000000u, false);
    // A binary16 subnormal exceeds zero until the decoded flush zeroes it at
    // the unpack boundary.
    guest_predicate("decoded_hsetp2_subnormal_lt", 1, 0, 0x00000000u, 0x00000001u, true);
    guest_predicate("decoded_hsetp2_subnormal_ftz_lt", 1, 1ull << 6, 0x00000000u, 0x00000001u,
                    false);
    // The lane-AND mode joins both comparisons before the primary predicate.
    guest_predicate("decoded_hsetp2_lane_and_true", 1, 1ull << 49, 0x40003c00u, 0x44004000u,
                    true);
    guest_predicate("decoded_hsetp2_lane_and_false", 1, 1ull << 49, 0x40003c00u, 0x40004000u,
                    false);
    const auto f2i = encoded("0101110010110") | (1ull << 20) | (7ull << 16) |
                     (2ull << 8) | (2ull << 10) | (1ull << 12) | (1ull << 44);
    const auto exact_guest = [&](const char* name, std::uint64_t instruction,
                                  std::vector<std::uint32_t> constants, std::uint32_t expected) {
        Case item{name, {}, std::move(constants), expected, false, instruction};
        item.compare_words = true;
        cases.push_back(std::move(item));
    };
    { // Same-size F2F integral rounding: every selector against the halfway,
      // signed-zero, infinity and NaN boundaries. The expectations are the
      // independently derived guest roundToIntegral contract: halfway values
      // go to the nearest even integer, infinities and quiet NaNs pass
      // through, signed zero keeps its sign, and a saturating boundary clamps
      // into the unit interval with NaN and negatives mapping to plus zero.
      // The word comparison guards the export so NaN and -0 results survive.
        const auto f2f_round = [&](std::uint64_t selector, bool flush, bool saturate,
                                   bool absolute = false, bool negate = false) {
            std::uint64_t word = encoded("0101110010101") | (1ull << 20) | (7ull << 16) |
                                 (1ull << 47) | (2ull << 8) | (2ull << 10); // Rb = R1, P7,
            // write CC, F32 source and destination.
            // Guest IntegerRound field: bit 42 is the high selector bit and
            // bits 40..39 carry the low two, bit 39 first.
            word |= (1ull << 42) | ((selector & 1) << 39) | (((selector >> 1) & 1) << 40);
            if (flush) word |= 1ull << 44;
            if (saturate) word |= 1ull << 50;
            if (absolute) word |= 1ull << 49;
            if (negate) word |= 1ull << 45;
            return word;
        };
        constexpr std::uint32_t positive_half = 0x40200000u;  // 2.5
        constexpr std::uint32_t negative_half = 0xc0200000u;  // -2.5
        constexpr std::uint32_t three_and_half = 0x40600000u; // 3.5
        constexpr std::uint32_t three_quarters = 0x3f400000u; // 0.75
        constexpr std::uint32_t negative_zero = 0x80000000u;
        constexpr std::uint32_t positive_infinity = 0x7f800000u;
        constexpr std::uint32_t negative_infinity = 0xff800000u;
        constexpr std::uint32_t quiet_nan = 0x7fc00000u;
        const auto halfway = [](std::uint32_t nearest, std::uint32_t toward_negative,
                                std::uint32_t toward_positive, std::uint32_t toward_zero) {
            // Selector order: rint, floor, ceiling, truncate.
            return std::array{nearest, toward_negative, toward_positive, toward_zero};
        };
        const std::array<std::pair<std::uint32_t, std::array<std::uint32_t, 4>>, 7> plain{
            std::pair{positive_half, halfway(0x40000000u, 0x40000000u, 0x40400000u, 0x40000000u)},
            std::pair{negative_half, halfway(0xc0000000u, 0xc0400000u, 0xc0000000u, 0xc0000000u)},
            std::pair{three_and_half, halfway(0x40800000u, 0x40400000u, 0x40800000u, 0x40400000u)},
            std::pair{negative_zero, halfway(negative_zero, negative_zero, negative_zero, negative_zero)},
            std::pair{positive_infinity,
                      halfway(positive_infinity, positive_infinity, positive_infinity, positive_infinity)},
            std::pair{negative_infinity,
                      halfway(negative_infinity, negative_infinity, negative_infinity, negative_infinity)},
            std::pair{quiet_nan, halfway(quiet_nan, quiet_nan, quiet_nan, quiet_nan)}};
        for (std::uint64_t selector = 0; selector != 4; ++selector) {
            const char* direction = selector == 0 ? "rint" : selector == 1 ? "floor"
                                                        : selector == 2   ? "ceil"
                                                                          : "trunc";
            for (const auto& [input, expected] : plain) {
                exact_guest((std::string("decoded_f2f_") + direction + "_word").c_str(),
                            f2f_round(selector + 4, false, false), {0, input, 0, 0},
                            expected[selector]);
            }
            // Flush modifiers: subnormal inputs become signed zero before the
            // rounding, so both signs of subnormal round to signed zero.
            exact_guest((std::string("decoded_f2f_") + direction + "_ftz_subnormal").c_str(),
                        f2f_round(selector + 4, true, false), {0, 1, 0, 0}, 0);
            exact_guest((std::string("decoded_f2f_") + direction + "_ftz_negative_subnormal").c_str(),
                        f2f_round(selector + 4, true, false), {0, 0x80000001u, 0, 0},
                        negative_zero);
            // Saturation follows the rounding: in-range results clamp at the
            // unit bounds and NaN saturates to plus zero.
            exact_guest((std::string("decoded_f2f_") + direction + "_sat_halfway").c_str(),
                        f2f_round(selector + 4, false, true), {0, positive_half, 0, 0},
                        0x3f800000u);
            exact_guest((std::string("decoded_f2f_") + direction + "_sat_negative").c_str(),
                        f2f_round(selector + 4, false, true), {0, negative_half, 0, 0}, 0);
            exact_guest((std::string("decoded_f2f_") + direction + "_sat_quarter").c_str(),
                        f2f_round(selector + 4, false, true), {0, three_quarters, 0, 0},
                        selector == 1 || selector == 3 ? 0u : 0x3f800000u);
            exact_guest((std::string("decoded_f2f_") + direction + "_sat_nan").c_str(),
                        f2f_round(selector + 4, false, true), {0, quiet_nan, 0, 0}, 0);
        }
        // Source modifiers act before the rounding, in the guest order
        // absolute first and negation second.
        exact_guest("decoded_f2f_floor_negate", f2f_round(5, false, false, false, true),
                    {0, positive_half, 0, 0}, 0xc0400000u);
        exact_guest("decoded_f2f_ceil_absolute", f2f_round(6, false, false, true, false),
                    {0, negative_half, 0, 0}, 0x40400000u);
    }
    // Independently authored bit patterns cover both input orders and the
    // boundaries the emitted native extrema keep. The expectation is the tested
    // CPU guest contract: exactly one NaN operand yielding the other, infinities
    // ordered, adjacent normal values distinguished. The emitted precise extrema
    // must reproduce it on this device; no expectation is taken from the device
    // itself. The deliberate departures (zero ties of opposite sign, NaN
    // payloads of a two-NaN result, subnormal operands) are outside these
    // inputs by design: the native operation decides them.
    // A NaN against a zero is the common guest form (FMNMX x, RZ).
    const std::array<std::array<std::uint32_t, 2>, 12> extremum_inputs{{
        {0, 0x3f800000u}, {0x80000000u, 0xbf800000u}, {0x00800000u, 0x00800001u},
        {0x80800000u, 0x80800001u}, {0x7fc12345u, 0x3f800000u}, {0xff812345u, 0xbf800001u},
        {0x7fc12345u, 0x00800000u}, {0x7f800000u, 0xff800000u},
        {0x3f800001u, 0x3f800000u}, {0x80000000u, 0x00800000u},
        {0x7fc12345u, 0x80000000u}, {0xff812345u, 0u},
    }};
    const auto fmnmx = encoded("0101110001100") | (1ull << 8) | (2ull << 20) |
                       (7ull << 16) | (7ull << 39) | (1ull << 44);
    for (auto inputs : extremum_inputs) {
        for (bool reverse : {false, true}) {
            if (reverse) std::swap(inputs[0], inputs[1]);
            for (bool minimum : {false, true}) {
                const armigo::shader::graphics::detail::OperationWords operands{inputs[0], inputs[1], 0};
                const auto expected = minimum
                    ? armigo::shader::graphics::detail::evaluate_float_minimum(operands, 0)
                    : armigo::shader::graphics::detail::evaluate_float_maximum(operands, 0);
                exact_guest(minimum ? "decoded_fmnmx_ftz_min" : "decoded_fmnmx_ftz_max",
                            fmnmx | (minimum ? 0 : 1ull << 42),
                            {0, inputs[0], inputs[1], 0}, expected);
            }
        }
    }
    // The CPU integer decomposition applies input FTZ before directed
    // rounding. Adjacent normal values distinguish that boundary from an
    // incorrect flush after rounding, which would yield +/-1 for subnormals.
    for (const auto input : {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
                             0x00800000u, 0x80800000u, 0x7fc12345u, 0xffc12345u,
                             0x7f812345u, 0xff812345u, 0x40600000u, 0xc0600000u,
                             0x7f800000u, 0xff800000u,
                             // Finite overflow boundaries: 2^31, just above,
                             // 2^32, -2^31 exactly, just below, and large
                             // magnitudes beyond every destination bound.
                             0x4f000000u, 0x4f000001u, 0x4f800000u, 0x4f800001u,
                             0xcf000000u, 0xcf000001u, 0x5f000000u, 0xdf000000u}) {
        for (const bool signed_destination : {false, true}) {
            for (unsigned rounding = 0; rounding != 4; ++rounding) {
                const auto profile = 2u | (signed_destination ? 1u : 0u) | (rounding << 3);
                const auto expected = armigo::shader::graphics::detail::evaluate_float_to_integer(
                    {input, 0, 0}, profile);
                const auto instruction = (f2i & ~(1ull << 12)) |
                    (signed_destination ? 1ull << 12 : 0) | (std::uint64_t{rounding} << 39);
                exact_guest(signed_destination ? "decoded_f2i_ftz_signed" : "decoded_f2i_ftz_unsigned",
                            instruction, {0, input, 0, 0}, expected);
            }
        }
    }
    guest("decoded_f2i_nearest", f2i, {0, 0x40600000, 0, 0}, 4);
    guest("decoded_f2i_floor", f2i | (1ull << 39), {0, 0x40600000, 0, 0}, 3);
    guest("decoded_f2i_ceil", f2i | (2ull << 39), {0, 0x40600000, 0, 0}, 4);
    guest("decoded_f2i_trunc", f2i | (3ull << 39), {0, 0x40600000, 0, 0}, 3);
    guest("decoded_f2i_abs_trunc", f2i | (3ull << 39) | (1ull << 49), {0, 0xc0600000, 0, 0}, 3);
    // FMUL scale selectors 4 and 6 scale input A by 8 and 2 in M8/M2 order; an
    // inverted selector table changes these decoded
    // products bitwise. The expected words use the exact CPU product the
    // emitted helpers reproduce.
    const auto fmul = 0x5c68000000270100ull;
    guest("decoded_fmul_scale_m8", fmul | (4ull << 41), {0, 0x3fa00000, 0x40400000, 0},
          armigo::shader::multiply_rne(
              armigo::shader::multiply_rne(0x3fa00000u, 0x41000000u), 0x40400000u));
    guest("decoded_fmul_scale_m2", fmul | (6ull << 41), {0, 0x3fa00000, 0x40400000, 0},
          armigo::shader::multiply_rne(
              armigo::shader::multiply_rne(0x3fa00000u, 0x40000000u), 0x40400000u));
    // MUFU.RCP and MUFU.RSQ draw against Metal's own native divide of one and
    // native rsqrt, probed on this device with the same compile options; the
    // exact composed helpers and any stale lowering would change these words.
    const auto mufu_rcp = 0x5080000000070100ull | (4ull << 20);
    const auto mufu_rsq = 0x5080000000070100ull | (5ull << 20);
    const std::vector<std::uint32_t> rcp_inputs{
        0x3f000000u, 0x3f800000u, 0x3f800001u, 0x40000000u, 0x40400000u,
        0x00800000u, 0x00000001u, 0x7f800000u, 0x00000000u, 0x80000000u};
    const auto rcp_words = native_words(device, compiler, "as_type<uint>(1.0f / x)", rcp_inputs);
    for (std::size_t index = 0; index < rcp_inputs.size(); ++index) {
        exact_guest("decoded_mufu_rcp_native", mufu_rcp, {0, rcp_inputs[index], 0, 0},
                    rcp_words[index]);
    }
    const std::vector<std::uint32_t> rsq_inputs{
        0x3f800000u, 0x3f800001u, 0x40000000u, 0x40800000u, 0x00800000u,
        0x00000001u, 0x7f800000u, 0x00000000u, 0xbf800000u};
    const auto rsq_words = native_words(device, compiler, "as_type<uint>(rsqrt(x))", rsq_inputs);
    for (std::size_t index = 0; index < rsq_inputs.size(); ++index) {
        exact_guest("decoded_mufu_rsq_native", mufu_rsq, {0, rsq_inputs[index], 0, 0},
                    rsq_words[index]);
    }
    const auto i2f = encoded("0101110010111") | (1ull << 20) | (7ull << 16) |
                     (2ull << 8) | (2ull << 10) | (1ull << 13) | (1ull << 45) | (1ull << 49);
    guest("decoded_i2f_abs_neg", i2f, {0, 0xfffffffd, 0, 0}, 0xc0400000);
    const auto i2i = encoded("0101110011100") | (1ull << 20) | (7ull << 16) |
                     (2ull << 8) | (2ull << 10) | (1ull << 12) | (1ull << 13) | (1ull << 49);
    guest("decoded_i2i_abs", i2i, {0, 0xfffffffd, 0, 0}, 3);
    const auto lop3 = encoded("001111") | (1ull << 8) | (7ull << 16) | (0xffull << 20) |
                      (3ull << 39) | (0x96ull << 48) | (1ull << 56) | (1ull << 47);
    guest("decoded_lop3_negative_immediate_cc", lop3, {0, 0x12345678, 0, 0x89abcdef},
          0x12345678 ^ 0xfff800ff ^ 0x89abcdef);
    for (unsigned mask : {1u, 2u}) {
        const auto tmml = 0xdf58000000070000ull | (4ull << 8) | (2ull << 28) |
                          (std::uint64_t{mask} << 31) | (8ull << 36);
        for (bool magnify : {false, true}) {
            const float scale = magnify ? 1.0f / 64.0f : 1.0f / 8.0f;
            const float expected = mask == 1 ? (magnify ? -256.0f : 512.0f) : (magnify ? 0.0f : 256.0f);
            const char* name = mask == 1 ? (magnify ? "tmml_unclamped_negative" : "tmml_unclamped_positive")
                                        : (magnify ? "tmml_clamped_min" : "tmml_clamped_max");
            cases.push_back(Case{name, {}, {0, 0, std::bit_cast<std::uint32_t>(scale), 0},
                                 std::bit_cast<std::uint32_t>(expected), false, tmml, true});
        }
        // rho=1.5 gives trunc(256*log2(rho))=149; rho=0.75 gives -106.
        // Allow one fixed-point unit for native fractional LOD approximation;
        // the power-of-two and sampler clamp cases above stay bit-exact.
        for (bool magnify : {false, true}) {
            const float scale = magnify ? 3.0f / 128.0f : 3.0f / 64.0f;
            const float expected = magnify ? (mask == 1 ? -106.0f : 0.0f) : 149.0f;
            const char* name = mask == 1 ? (magnify ? "tmml_fractional_unclamped_negative" : "tmml_fractional_unclamped_positive")
                                        : (magnify ? "tmml_fractional_clamped_min" : "tmml_fractional_clamped_positive");
            cases.push_back(Case{name, {}, {0, 0, std::bit_cast<std::uint32_t>(scale), 0},
                                 std::bit_cast<std::uint32_t>(expected), false, tmml, true,
                                 expected == 0.0f ? 0.0f : 1.0f});
        }
    }
    {
        // A per-texel offset gather (TLD4 Ptp) over a uniform 0.25 texture:
        // every gathered texel is 0.25, so each result component is 0.25
        // whatever the live quadrant order. The packed offsets are zero, so
        // the coordinate stays inside the texture at every pixel.
        const auto tld4_ptp = 0xc800000000000000ull | (0ull << 0) | (4ull << 8) |
            (7ull << 16) | (6ull << 20) | (2ull << 28) | (15ull << 31) |
            (8ull << 36) | (7ull << 51) | (2ull << 54);
        cases.push_back(Case{"tld4_ptp_gather_offsets", {}, {0, 0, 0, 0},
                             std::bit_cast<std::uint32_t>(64.0f / 255.0f), false,
                             tld4_ptp, false, 0.0f, {}, false, true});
    }
    for (const auto& draw : {
             BuiltinDraw{0, false, 0, 0, 1}, BuiltinDraw{1, false, 0, 0, 1},
             BuiltinDraw{0, false, 11, 9, 3}, BuiltinDraw{1, false, 11, 9, 3},
             BuiltinDraw{0, true, 0, 0, 1}, BuiltinDraw{1, true, 0, 0, 1},
             BuiltinDraw{0, true, 5, 9, 3}, BuiltinDraw{1, true, 5, 9, 3},
             BuiltinDraw{0, true, -3, 9, 3}, BuiltinDraw{1, true, -3, 9, 3}}) {
        // Every indexed case uses the independently authored index 7. Repeated
        // instances overlap; raster order leaves the last instance's export.
        const auto expected = draw.component == 0 ? (draw.indexed ? 7 : 0) + draw.vertex_base
                                                  : static_cast<int>(draw.instance_count - 1);
        Case item{"decoded_vertex_builtin", {}, {}, std::bit_cast<std::uint32_t>(float(expected))};
        item.builtin = draw;
        cases.push_back(std::move(item));
    }
    // Point rasterization through the translated stages. The sprite centres on
    // pixel (1, 1): a shader-written size 3 covers the 3x3 pixel block 0..2 and
    // the armigo_point_size default covers pixel (1, 1) alone, each with the
    // [[point_coord]] fragment placing (0, 0) at the sprite's upper left.
    {
        Case shader_size{"point_size_shader_written", {}, {}, 0};
        shader_size.point = PointDraw{true, 3.0f, false};
        cases.push_back(std::move(shader_size));
        Case shader_coord{"point_coord_shader_written", {}, {}, 0};
        shader_coord.point = PointDraw{true, 3.0f, true};
        cases.push_back(std::move(shader_coord));
        Case default_size{"point_size_default_constant", {}, {}, 0};
        default_size.point = PointDraw{false, 1.0f, false};
        cases.push_back(std::move(default_size));
        Case default_coord{"point_coord_default_constant", {}, {}, 0};
        default_coord.point = PointDraw{false, 1.0f, true};
        cases.push_back(std::move(default_coord));
    }
    return cases;
}

int run_one(id<MTLDevice> device, id<MTL4Compiler> compiler, id<MTL4CommandQueue> queue, const Case& item) {
    FragmentBuilder builder;
    const bool raw_builder_case = !item.instruction && !item.builtin && !item.point;
    const bool point_case = item.point.has_value();
    if (raw_builder_case) builder.compute(item);
    auto vertex = vertex_program();
    ResourcePlan vertex_plan;
    vertex_plan.stage = Stage::vertex;
    vertex_plan.profile = vertex.profile;
    vertex_plan.dependencies = vertex.ir.dependencies;
    for (const auto& use : vertex.requirements.io) vertex_plan.io.push_back(use);
    const auto fragment_result = item.builtin ? Result<MslArtifact>{MslArtifact{
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct BuiltinInput { float vertex_value [[user(locn0_0), flat]]; float instance_value [[user(locn0_1), flat]]; };\n"
        "fragment float4 armigo_fragment_main(BuiltinInput in [[stage_in]]) { return float4(in." +
            std::string(item.builtin->component == 0 ? "vertex_value" : "instance_value") + "); }\n",
        "armigo_fragment_main", {}}} :
        point_case ? Result<MslArtifact>{MslArtifact{
            item.point->point_coord ?
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "fragment float4 armigo_fragment_main(float2 coord [[point_coord]])"
                " { return float4(coord, 0.0f, 1.0f); }\n" :
                "#include <metal_stdlib>\nusing namespace metal;\n"
                "fragment float4 armigo_fragment_main() { return float4(1.0f, 1.0f, 1.0f, 1.0f); }\n",
            "armigo_fragment_main", {}}} :
        item.instruction
            ? guest_fragment(item.instruction, item.query_lod,
                             item.compare_words ? std::optional{item.expected} : std::nullopt,
                             item.gather_offsets, item.post_instruction)
            : builder.emit(item.compare_words ? std::optional{item.expected} : std::nullopt);
    if (const auto* d = std::get_if<Diagnostic>(&fragment_result)) {
        std::fprintf(stderr, "%s: emission failed: %s\n", item.name.c_str(), d->message.c_str());
        return 1;
    }
    const auto fragment = std::get<MslArtifact>(std::move(fragment_result));
    EmitOptions vertex_options;
    const auto vertex_result = item.builtin ? armigo::shader::test::vertex_builtins::guest_vertex() :
        point_case ? armigo::shader::test::vertex_builtins::point_vertex(
                         item.point->shader_size, item.point->size) :
        emit_msl(vertex, vertex_plan, vertex_options);
    if (const auto* d = std::get_if<Diagnostic>(&vertex_result)) {
        std::fprintf(stderr, "%s: vertex emission failed: %s\n", item.name.c_str(), d->message.c_str());
        return 1;
    }
    const auto vertex_stage = std::get<MslArtifact>(std::move(vertex_result));
    const auto vertex_library = library(device, vertex_stage.source);
    const auto fragment_library = library(device, fragment.source);
    if (!vertex_library || !fragment_library) return 1;
    MTL4LibraryFunctionDescriptor* vertex_function = [MTL4LibraryFunctionDescriptor new];
    vertex_function.library = vertex_library;
    vertex_function.name = @"armigo_vertex_main";
    MTL4LibraryFunctionDescriptor* fragment_function = [MTL4LibraryFunctionDescriptor new];
    fragment_function.library = fragment_library;
    fragment_function.name = @"armigo_fragment_main";
    MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
    MTL4FunctionDescriptor* vertex_descriptor = vertex_function;
    descriptor.vertexFunctionDescriptor = vertex_descriptor;
    descriptor.fragmentFunctionDescriptor = fragment_function;
    // Translated vertex stages write [[point_size]] (the program's or the fixed
    // 1.0), which Metal accepts only with a point or unspecified topology class.
    descriptor.inputPrimitiveTopology = item.builtin || point_case ?
        MTLPrimitiveTopologyClassPoint : MTLPrimitiveTopologyClassUnspecified;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    NSError* error = nil;
    id<MTLRenderPipelineState> state =
        [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
    if (!state) {
        std::fprintf(stderr, "%s: pipeline creation failed: %s\n", item.name.c_str(),
                     error.localizedDescription.UTF8String);
        return 1;
    }
    std::vector<std::uint32_t> constants = item.constants;
    while (constants.size() < 20) constants.push_back(0);
    id<MTLBuffer> constant_buffer = [device newBufferWithBytes:constants.data()
        length:constants.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    std::array<std::uint32_t, 18> sizes{};
    sizes[0] = static_cast<std::uint32_t>(constants.size() * sizeof(std::uint32_t));
    id<MTLBuffer> size_table = [device newBufferWithBytes:sizes.data()
        length:sizes.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    auto* target_descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:kWidth height:kHeight mipmapped:NO];
    target_descriptor.storageMode = MTLStorageModePrivate;
    target_descriptor.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [device newTextureWithDescriptor:target_descriptor];
    id<MTLBuffer> readback = [device newBufferWithLength:kWidth * kHeight * kPixelBytes
                                                 options:MTLResourceStorageModeShared];
    const std::uint32_t index = 7;
    id<MTLBuffer> index_buffer = item.builtin && item.builtin->indexed ?
        [device newBufferWithBytes:&index length:sizeof(index) options:MTLResourceStorageModeShared] : nil;
    MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
    const bool textured = item.query_lod || item.gather_offsets;
    table_descriptor.maxBufferBindCount = 4;
    table_descriptor.maxTextureBindCount = textured ? 1 : 0;
    table_descriptor.maxSamplerStateBindCount = textured ? 1 : 0;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
    MTLResidencySetDescriptor* residency_descriptor = [MTLResidencySetDescriptor new];
    residency_descriptor.initialCapacity = 4;
    id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
    id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
    id<MTL4CommandBuffer> commands = [device newCommandBuffer];
    if (!constant_buffer || !size_table || !target || !readback || !table || !residency || !allocator || !commands) {
        std::fprintf(stderr, "%s: Metal resource creation failed.\n", item.name.c_str());
        return 1;
    }
    if (item.builtin && item.builtin->indexed && !index_buffer) return 1;
    [table setAddress:constant_buffer.gpuAddress atIndex:0];
    [table setAddress:size_table.gpuAddress atIndex:1];
    id<MTLBuffer> draw_support = nil;
    if (vertex_stage.reflection.draw_support_buffer_index) {
        const std::uint32_t first_instance = item.builtin->instance_base;
        draw_support = [device newBufferWithBytes:&first_instance length:sizeof(first_instance)
            options:MTLResourceStorageModeShared];
        if (!draw_support) return 1;
        [table setAddress:draw_support.gpuAddress atIndex:*vertex_stage.reflection.draw_support_buffer_index];
        [residency addAllocation:draw_support];
    }
    id<MTLTexture> lod_texture = nil;
    id<MTLSamplerState> lod_sampler = nil;
    if (textured) {
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                             width:32 height:32 mipmapped:YES];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageShaderRead;
        lod_texture = [device newTextureWithDescriptor:descriptor];
        auto* sampler = [MTLSamplerDescriptor new];
        sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterLinear;
        sampler.mipFilter = MTLSamplerMipFilterLinear;
        sampler.maxAnisotropy = 1;
        sampler.lodMinClamp = 0;
        sampler.lodMaxClamp = 1;
        lod_sampler = [device newSamplerStateWithDescriptor:sampler];
        if (!lod_texture || !lod_sampler) return 1;
        if (item.gather_offsets) {
            // A uniform 0.25 texture: every gathered texel is 0.25, so the
            // expected result stays independent of the live quadrant order.
            const std::vector<std::uint8_t> fill(32 * 32 * 4, 0x40);
            [lod_texture replaceRegion:MTLRegionMake2D(0, 0, 32, 32) mipmapLevel:0
                             withBytes:fill.data() bytesPerRow:32 * 4];
        }
        [table setTexture:lod_texture.gpuResourceID atIndex:0];
        [table setSamplerState:lod_sampler.gpuResourceID atIndex:0];
        [residency addAllocation:lod_texture];
    }
    for (id<MTLAllocation> allocation in @[constant_buffer, size_table, target, readback]) {
        [residency addAllocation:allocation];
    }
    if (index_buffer) [residency addAllocation:index_buffer];
    [residency commit];
    [commands beginCommandBufferWithAllocator:allocator];
    [commands useResidencySet:residency];
    MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = item.builtin || point_case ?
        MTLClearColorMake(-1, -1, -1, -1) : MTLClearColorMake(0, 0, 0, 0);
    id<MTL4RenderCommandEncoder> render = [commands renderCommandEncoderWithDescriptor:pass];
    if (!render) return 1;
    [render setRenderPipelineState:state];
    [render setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
    if (item.builtin) {
        const auto& draw = *item.builtin;
        if (draw.indexed) {
            [render drawIndexedPrimitives:MTLPrimitiveTypePoint indexCount:1 indexType:MTLIndexTypeUInt32
                              indexBuffer:index_buffer.gpuAddress indexBufferLength:sizeof(index)
                            instanceCount:draw.instance_count baseVertex:draw.vertex_base baseInstance:draw.instance_base];
        } else {
            [render drawPrimitives:MTLPrimitiveTypePoint vertexStart:draw.vertex_base vertexCount:1
                    instanceCount:draw.instance_count baseInstance:draw.instance_base];
        }
    } else if (point_case) {
        [render drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:1
                instanceCount:1 baseInstance:0];
    } else {
        [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [render endEncoding];
    id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
    if (!copy) return 1;
    [copy barrierAfterQueueStages:MTLStageFragment beforeStages:MTLStageBlit
                 visibilityOptions:MTL4VisibilityOptionDevice];
    [copy copyFromTexture:target sourceSlice:0 sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(kWidth, kHeight, 1)
                 toBuffer:readback destinationOffset:0
    destinationBytesPerRow:kWidth * kPixelBytes destinationBytesPerImage:0];
    [copy endEncoding];
    [commands endCommandBuffer];
    NSMutableArray* retained = [@[vertex_library, fragment_library, state, constant_buffer, size_table, target, readback,
                                 table, residency, allocator] mutableCopy];
    if (textured) [retained addObjectsFromArray:@[lod_texture, lod_sampler]];
    if (index_buffer) [retained addObject:index_buffer];
    if (draw_support) [retained addObject:draw_support];
    dispatch_semaphore_t completed = dispatch_semaphore_create(0);
    __block NSError* execution_error = nil;
    MTL4CommitOptions* options = [MTL4CommitOptions new];
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(retained.count);
        execution_error = feedback.error;
        dispatch_semaphore_signal(completed);
    }];
    id<MTL4CommandBuffer> submitted[] = {commands};
    [queue commit:submitted count:1 options:options];
    if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0) {
        std::fprintf(stderr, "%s: Metal execution timed out.\n", item.name.c_str());
        return 1;
    }
    if (execution_error) {
        std::fprintf(stderr, "%s: Metal execution failed: %s\n", item.name.c_str(),
                     execution_error.localizedDescription.UTF8String);
        return 1;
    }
    const auto* pixels = static_cast<const std::uint32_t*>(readback.contents);
    const auto expected = item.compare_words ? 0x3f800000u : item.expected;
    if (point_case) {
        // Sprite geometry: clip (-0.25, 0.25) rasterizes to window (1.5, 1.5),
        // the sprite spans one size around that centre, and [[point_coord]]
        // locates each covered pixel centre inside the sprite from its
        // upper-left corner. Uncovered pixels keep the -1 clear word; covered
        // coordinates stay at least half a pixel from every sprite edge, so
        // the fraction tolerance only absorbs coordinate rounding.
        const auto& point = *item.point;
        const float left = 1.5f - point.size / 2.0f;
        int failures = 0;
        for (NSUInteger y = 0; y < kHeight && !failures; ++y) {
            for (NSUInteger x = 0; x < kWidth && !failures; ++x) {
                const bool covered = x + 0.5f >= left && x + 0.5f < left + point.size &&
                                     y + 0.5f >= left && y + 0.5f < left + point.size;
                std::uint32_t expected_lanes[4];
                if (!covered) {
                    expected_lanes[0] = expected_lanes[1] = expected_lanes[2] = expected_lanes[3] =
                        std::bit_cast<std::uint32_t>(-1.0f);
                } else if (point.point_coord) {
                    expected_lanes[0] = std::bit_cast<std::uint32_t>(
                        (static_cast<float>(x) + 0.5f - left) / point.size);
                    expected_lanes[1] = std::bit_cast<std::uint32_t>(
                        (static_cast<float>(y) + 0.5f - left) / point.size);
                    expected_lanes[2] = 0;
                    expected_lanes[3] = 0x3f800000u;
                } else {
                    expected_lanes[0] = expected_lanes[1] = expected_lanes[2] = expected_lanes[3] =
                        0x3f800000u;
                }
                for (unsigned lane = 0; lane < 4; ++lane) {
                    const auto actual = pixels[(y * kWidth + x) * 4 + lane];
                    const float actual_float = std::bit_cast<float>(actual);
                    const bool close = std::isfinite(actual_float) &&
                        std::abs(actual_float - std::bit_cast<float>(expected_lanes[lane])) <= 1.0f / 256.0f;
                    if (actual != expected_lanes[lane] && !close) {
                        std::fprintf(stderr, "%s: pixel %zu,%zu lane %u GPU 0x%08x CPU 0x%08x\n",
                                     item.name.c_str(), static_cast<std::size_t>(x),
                                     static_cast<std::size_t>(y), lane, actual, expected_lanes[lane]);
                        failures = 1;
                    }
                }
            }
        }
        return failures;
    }
    int failures = 0;
    for (NSUInteger pixel = 0; pixel < kWidth * kHeight; ++pixel) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            const auto actual = pixels[pixel * 4 + lane];
            const float actual_float = std::bit_cast<float>(actual);
            const bool approximate_match = item.result_tolerance > 0 && std::isfinite(actual_float) &&
                std::abs(actual_float - std::bit_cast<float>(item.expected)) <= item.result_tolerance;
            if (actual != expected && !approximate_match) {
                if (item.builtin) {
                    const auto& draw = *item.builtin;
                    std::fprintf(stderr, "Builtin component %u, indexed %u, vertex base %d, instance base %u, count %u\n",
                                 draw.component, unsigned(draw.indexed), draw.vertex_base, draw.instance_base, draw.instance_count);
                }
                if (item.compare_words) {
                    std::fprintf(stderr, "Decoded instruction 0x%016llx, input words 0x%08x 0x%08x, expected word 0x%08x\n",
                                 static_cast<unsigned long long>(item.instruction), item.constants[1],
                                 item.constants[2], item.expected);
                }
                std::fprintf(stderr, "%s: pixel %lu lane %u GPU 0x%08x CPU 0x%08x\n", item.name.c_str(),
                             static_cast<unsigned long>(pixel), lane, pixels[pixel * 4 + lane], expected);
                failures = 1;
            }
        }
        if (failures) break;
    }
    return failures;
}
} // namespace

namespace armigo::shader::test::msl_expr {
int run_gpu_cases() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fputs("No Metal device.\n", stderr);
            return 1;
        }
        NSError* error = nil;
        id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        if (!compiler || !queue) {
            std::fputs("Metal 4 compiler or queue unavailable.\n", stderr);
            return 1;
        }
        int failures = 0;
        const auto cases = make_cases(device, compiler);
        for (const auto& item : cases) failures += run_one(device, compiler, queue, item);
        std::printf("%zu pipeline expression GPU cases, %d failed\n", cases.size(), failures);
        return failures == 0 ? 0 : 1;
    }
}
} // namespace armigo::shader::test::msl_expr
