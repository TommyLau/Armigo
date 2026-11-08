// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Verification for the armigo-output-init-v1 emission policy: vertex stages
// with early exits and path-limited output writes are initialized at entry.
// The structured programs and instruction streams below are independently
// authored synthetic fixtures. The prologue stores position (0, 0, 0, 1) and
// zeroed outputs
// before the first guest instruction.
#include "pipeline/phases.h"

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::shader::test::output_init {
std::string gpu_vertex_source(unsigned program);
#if defined(ARMIGO_OUTPUT_INIT_GPU)
int run_gpu_cases();
#endif
}

namespace {
using namespace armigo::shader::pipeline;
int failures{};
std::filesystem::path fixture_directory;
bool compile_metal{};

void check(bool value, const std::string& message) { if (!value) { std::cerr << "FAIL: " << message << '\n'; ++failures; } }
Operand word(std::uint32_t bits) { return LiteralBits{bits, ValueType::word32}; }
Operand fp(float value) { return word(std::bit_cast<std::uint32_t>(value)); }
Operand predicate(bool value) { return LiteralBits{value ? 1u : 0u, ValueType::predicate}; }

std::string quote(const std::string& value) {
    std::string result = "'"; for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c); return result + "'";
}
void compile_with_metal(const std::string& label, const std::string& source) {
    if (fixture_directory.empty()) return;
    const auto path = fixture_directory / (label + ".metal");
    std::ofstream(path) << source;
    if (!compile_metal) return;
    const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
        quote((fixture_directory / "module-cache").string()) + " -c " + quote(path.string()) +
        " -o " + quote((fixture_directory / (label + ".air")).string());
    check(std::system(command.c_str()) == 0, label + " Metal compilation");
}

// Structured-program fixture, following the established stage-test pattern.
struct Fixture {
    StructuredProgram p;
    ResourcePlan plan;
    EmitOptions options;
    Fixture(Stage stage = Stage::fragment) {
        p.ir.stage = stage; p.ir.entry = FunctionId{0};
        p.ir.functions.push_back({8, BlockId{0}, {BlockId{0}}, {}, {}});
        p.ir.blocks.push_back({FunctionId{0}, {}, {}, {}, {}, Return{}});
        p.regions = {BlockRegion{BlockId{0}}}; p.functions = {{FunctionId{0}, RegionId{0}}};
        p.profile = {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        plan.stage = stage; plan.profile = p.profile;
        if (stage == Stage::vertex) {
            p.ir.dependencies.push_back({{FactKind::depth_convention}, DepthConvention::zero_to_one, {OriginKind::explicit_option, "synthetic-output-init"}});
            plan.dependencies = p.ir.dependencies;
            io(IoSemantic::position, Access::write, 15, {ScalarType::f32, 4});
        }
    }
    void io(IoSemantic semantic, Access access, unsigned mask, IoType type, unsigned location = 0) {
        IoUse use; use.semantic = semantic; use.access = access; use.mask = mask; use.type = type; use.location = static_cast<std::uint16_t>(location);
        p.requirements.io.push_back(use); plan.io.push_back(use);
    }
    OperationId operation(Operation op) {
        op.effects = required_effects(op); op.source = {40 + static_cast<Pc>(p.ir.operations.size() * 8), InstructionName::mov};
        OperationId id{static_cast<std::uint32_t>(p.ir.operations.size())};
        p.ir.operations.push_back(std::move(op)); p.ir.blocks[0].operations.push_back(id); return id;
    }
    ValueId value(ValueType type = ValueType::word32) {
        ValueId id{static_cast<std::uint32_t>(p.ir.values.size())}; p.ir.values.push_back({type, ValueKind::temporary}); return id;
    }
    void store(IoSemantic semantic, std::vector<Operand> data, unsigned location = 0) {
        IoInfo info; info.semantic = semantic; info.type = {ScalarType::f32, 4}; info.location = static_cast<std::uint16_t>(location);
        info.component_mask = static_cast<std::uint8_t>((1u << data.size()) - 1);
        for (unsigned i = 0; i < data.size(); ++i) info.data.push_back(InputIndex{static_cast<std::uint16_t>(i)});
        operation({{}, Opcode::store_io, {}, {}, std::move(data), {}, info});
    }
    void isolate_operations() {
        p.ir.blocks.clear(); p.ir.functions[0].blocks.clear();
        for (std::uint32_t i = 0; i < p.ir.operations.size(); ++i) {
            p.ir.functions[0].blocks.push_back(BlockId{i});
            BasicBlock block; block.function = FunctionId{0}; block.operations = {OperationId{i}};
            if (i) block.predecessors = {BlockId{i - 1}};
            block.terminator = i + 1 < p.ir.operations.size() ? Terminator{Jump{BlockId{i + 1}}} : Terminator{Return{}};
            p.ir.blocks.push_back(std::move(block));
        }
    }
    Result<MslArtifact> emit() { return emit_msl(p, plan, options); }
};

// A condition value the definedness analysis cannot prove constant, defined
// before the regions that consume it, so both arms stay live.
ValueId guarded_condition(Fixture& f) {
    const auto condition = f.value(ValueType::predicate);
    f.operation({{}, Opcode::compare, {}, {Domain::unsigned_integer, ScalarType::u32},
        {word(1), word(2)}, {condition}, ComparisonInfo{Compare::equal, false}});
    return condition;
}

void structured_tests() {
    // The early exit writes position X/Y/Z and returns; the remaining path
    // writes all four components. W comes from the entry initialization.
    Fixture early_exit(Stage::vertex);
    const auto condition = guarded_condition(early_exit);
    early_exit.store(IoSemantic::position, {fp(0), fp(0), fp(0.5f)});
    early_exit.store(IoSemantic::position, {fp(0.25f), fp(0.5f), fp(0.75f), fp(1)});
    early_exit.isolate_operations();
    early_exit.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}}}, BlockRegion{BlockId{0}},
        IfRegion{condition, RegionId{3}, RegionId{4}}, SequenceRegion{{RegionId{5}, RegionId{6}}},
        BlockRegion{BlockId{2}}, BlockRegion{BlockId{1}}, ReturnRegion{}};
    const auto early_result = early_exit.emit();
    if (const auto* d = std::get_if<Diagnostic>(&early_result)) {
        check(false, std::string("early_exit_return: ") + d->message);
    } else {
        const auto& artifact = std::get<MslArtifact>(early_result);
        check(artifact.source.find("out.position.w = 1.0f;") != std::string::npos &&
              artifact.source.find("out.position.x = float(0);") != std::string::npos,
            "Early exit initializes position at entry");
        check(artifact.source.find("return out;") != artifact.source.rfind("return out;"),
            "Early exit keeps both return paths");
        compile_with_metal("output_init_early_exit", artifact.source);
    }

    // Position is written on one path and a generic output on the other; both
    // are covered by initialization and neither rejects.
    Fixture split(Stage::vertex);
    const auto split_condition = guarded_condition(split);
    split.io(IoSemantic::generic, Access::write, 15, {ScalarType::f32, 4}, 3);
    split.store(IoSemantic::position, {fp(0), fp(0), fp(0), fp(1)});
    split.store(IoSemantic::generic, {fp(0.75f)}, 3);
    split.isolate_operations();
    split.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}}}, BlockRegion{BlockId{0}},
        IfRegion{split_condition, RegionId{3}, RegionId{4}},
        BlockRegion{BlockId{1}}, BlockRegion{BlockId{2}}};
    const auto split_result = split.emit();
    if (const auto* d = std::get_if<Diagnostic>(&split_result)) {
        check(false, std::string("split_path_outputs: ") + d->message);
    } else {
        const auto& artifact = std::get<MslArtifact>(split_result);
        check(artifact.source.find("out.position.w = 1.0f;") != std::string::npos &&
              artifact.source.find("out.generic3_0 = float(0);") != std::string::npos &&
              artifact.source.find("out.generic3_3 = 1.0f;") != std::string::npos,
            "Path-limited position and generic outputs initialize at entry");
        compile_with_metal("output_init_split_path", artifact.source);
    }

    // A position output that declares only X/Y/Z (GLSLC drops a constant W of
    // 1) gets the hardware default for W at entry; X/Y/Z stay as written.
    Fixture partial(Stage::vertex);
    partial.plan.io[0].mask = 7;
    partial.p.requirements.io[0].mask = 7;
    partial.store(IoSemantic::position, {fp(0.25f), fp(0.5f), fp(0.75f)});
    const auto partial_result = partial.emit();
    if (const auto* d = std::get_if<Diagnostic>(&partial_result)) {
        check(false, std::string("partial_position: ") + d->message);
    } else {
        const auto& artifact = std::get<MslArtifact>(partial_result);
        check(artifact.source.find("out.position.w = 1.0f;") != std::string::npos,
            "Undeclared position W takes the hardware default");
        compile_with_metal("output_init_partial_position", artifact.source);
    }
    // A vertex stage that writes outputs but no position still rejects; one
    // that writes no output at all is a void vertex for rasterizer-discard
    // draws.
    Fixture no_position(Stage::vertex);
    no_position.plan.io.clear();
    no_position.p.requirements.io.clear();
    no_position.io(IoSemantic::generic, Access::write, 15, {ScalarType::f32, 4}, 3);
    no_position.store(IoSemantic::generic, {fp(0.75f)}, 3);
    check(std::holds_alternative<Diagnostic>(no_position.emit()), "Vertex stage without position rejects");
    // A void vertex records no clip-depth convention.
    Fixture no_output(Stage::vertex);
    no_output.plan.io.clear();
    no_output.p.requirements.io.clear();
    no_output.plan.dependencies.clear();
    no_output.p.ir.dependencies.clear();
    const auto void_result = no_output.emit();
    if (const auto* d = std::get_if<Diagnostic>(&void_result)) {
        check(false, std::string("void_vertex: ") + d->message);
    } else {
        const auto& artifact = std::get<MslArtifact>(void_result);
        check(artifact.reflection.requires_rasterizer_discard && artifact.source.find("vertex void ") != std::string::npos,
            "Vertex stage without outputs is a void vertex that requires rasterizer discard");
    }

    // The policy covers writes on live paths only: a store under a literal
    // false guard writes nothing, so the completeness check still rejects.
    Fixture infeasible(Stage::vertex);
    infeasible.store(IoSemantic::position, {fp(0), fp(0), fp(0), fp(1)});
    infeasible.p.regions = {IfRegion{predicate(false), RegionId{1}, {}}, BlockRegion{BlockId{0}}};
    const auto rejected = infeasible.emit();
    const auto* error = std::get_if<Diagnostic>(&rejected);
    check(error && error->code == DiagnosticCode::undefined_read &&
        error->message.find("Stage output is not written") != std::string::npos,
        "A write on no live path stays rejected");

    // Fragment outputs are outside the policy: a path-limited colour write
    // keeps failing the return-path check.
    Fixture fragment;
    const auto fragment_condition = guarded_condition(fragment);
    fragment.io(IoSemantic::color, Access::write, 15, {ScalarType::f32, 4}, 0);
    fragment.store(IoSemantic::color, {fp(1), fp(0), fp(0), fp(1)});
    fragment.isolate_operations();
    fragment.p.regions = {SequenceRegion{{RegionId{1}, RegionId{2}}}, BlockRegion{BlockId{0}},
        IfRegion{fragment_condition, RegionId{3}, {}}, BlockRegion{BlockId{1}}};
    const auto fragment_result = fragment.emit();
    const auto* fragment_error = std::get_if<Diagnostic>(&fragment_result);
    check(fragment_error && fragment_error->code == DiagnosticCode::undefined_read &&
        fragment_error->message.find("Stage output is not written") != std::string::npos,
        "Fragment outputs stay outside the initialization policy");
}

// ---- Synthetic guest streams through the full decode pipeline ----

constexpr std::uint64_t pattern(std::string_view bits) {
    std::uint64_t result{};
    for (std::size_t i = 0; i < bits.size(); ++i)
        if (bits[i] == '1') result |= std::uint64_t{1} << (63 - i);
    return result;
}
constexpr std::uint64_t guarded(std::uint64_t word, unsigned predicate, bool inverted = false) {
    return (word & ~(15ull << 16)) | (std::uint64_t{predicate} << 16) | (std::uint64_t{inverted} << 19);
}
constexpr std::uint64_t mov32i(unsigned destination, std::uint32_t bits) {
    return guarded(0x0100000000000000ull | destination | (std::uint64_t{bits} << 20), 7);
}
// ISETP.LT Pn, PT, left, right, PT; compares two raw registers.
constexpr std::uint64_t set_predicate_less(unsigned destination, unsigned left, unsigned right) {
    return 0x5b60000000000000ull | (std::uint64_t{destination} << 3) | 7ull |
        (std::uint64_t{left} << 8) | (7ull << 16) | (std::uint64_t{right} << 20) |
        (7ull << 39) | (1ull << 49);
}
constexpr std::uint64_t store(unsigned source, unsigned offset, unsigned count = 1,
                              unsigned predicate = 7, bool inverted = false) {
    return guarded(0xeff000000000ff00ull | source | (std::uint64_t{offset} << 20) |
        (std::uint64_t{count - 1} << 47), predicate, inverted);
}
constexpr Pc stream_pc(std::size_t index) {
    return static_cast<Pc>((index / 3) * 32 + (index % 3) * 8 + 8);
}
std::uint64_t branch(std::size_t from, std::size_t to, unsigned predicate) {
    const auto displacement = (static_cast<std::int64_t>(stream_pc(to)) -
        static_cast<std::int64_t>(stream_pc(from)) - 8) & 0xffffff;
    return guarded(pattern("111000100100") | 15 | (std::uint64_t(displacement) << 20), predicate);
}
constexpr std::uint64_t exit_word() { return 0xe30000000007000full; }

struct GuestFacts final : FactSource {
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::depth_convention)
            return Known<FactValue>{DepthConvention::zero_to_one,
                {OriginKind::explicit_option, "synthetic-output-init"}};
        return Unknown{"Synthetic output-init program has no other facts"};
    }
};

StageImage guest_image(const std::vector<std::uint64_t>& words) {
    StageImage image; image.stage = Stage::vertex;
    const auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) image.bytes.push_back(std::byte((word >> (8 * i)) & 255));
    };
    for (const auto word : words) {
        if (image.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    return image;
}

// The vertex id feeds a runtime predicate so neither branch is provably
// constant: static analysis must treat both output writes as live, while the
// observed draw takes only the early-exit arm.
std::vector<std::uint64_t> early_exit_program() {
    return {
        mov32i(0, std::bit_cast<std::uint32_t>(0.0f)),    // position X
        mov32i(1, std::bit_cast<std::uint32_t>(0.0f)),    // position Y
        mov32i(2, std::bit_cast<std::uint32_t>(0.5f)),    // position Z
        mov32i(3, std::bit_cast<std::uint32_t>(1.0f)),    // position W: early exit skips this store
        mov32i(4, std::bit_cast<std::uint32_t>(8.0f)),    // point size keeps the probe visible
        mov32i(5, 99999),
        0xefd800000007ff06ull | (0x2fcull << 20),         // R6 = vertex id
        set_predicate_less(0, 6, 5),                      // P0 = id < 99999
        branch(8, 10, 0),                                 // if P0 skip the W store
        store(3, 0x7c),                                   // position W, normal path only
        store(4, 0x6c),                                   // point size
        store(0, 0x70, 3),                                // position X/Y/Z, both paths
        exit_word(),
    };
}

std::vector<std::uint64_t> split_path_program() {
    return {
        mov32i(0, std::bit_cast<std::uint32_t>(0.0f)),    // position X
        mov32i(1, std::bit_cast<std::uint32_t>(0.0f)),    // position Y
        mov32i(2, std::bit_cast<std::uint32_t>(0.5f)),    // position Z
        mov32i(3, std::bit_cast<std::uint32_t>(1.0f)),    // position W
        mov32i(4, std::bit_cast<std::uint32_t>(8.0f)),    // point size
        mov32i(5, std::bit_cast<std::uint32_t>(0.75f)),   // generic value, guarded path only
        mov32i(6, 99999),
        0xefd800000007ff07ull | (0x2fcull << 20),         // R7 = vertex id
        set_predicate_less(0, 7, 6),                      // P0 = id < 99999
        store(0, 0x70, 4),                                // position, always
        store(4, 0x6c),                                   // point size, always
        store(5, 0x80, 1, 0, true),                       // generic0.x only when P0 is false
        exit_word(),
    };
}

template<class T> T take(Result<T> result) {
    if (const auto* d = std::get_if<Diagnostic>(&result)) throw std::runtime_error(d->message);
    return std::get<T>(std::move(result));
}

Result<MslArtifact> run_guest(const std::vector<std::uint64_t>& words) {
    try {
        GuestFacts facts_source;
        FactReader facts(facts_source);
        const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        auto decoded = take(decode(guest_image(words), facts, {}));
        auto registers = take(emit_ir(decoded, facts, {}));
        auto ssa = take(construct_ssa(std::move(registers), {}));
        auto legal = take(legalize(std::move(ssa), facts, profile));
        auto structured = take(structure(std::move(legal), {}));
        auto plan = take(plan_resources(structured, profile));
        return emit_msl(structured, plan, {});
    } catch (const Diagnostic& error) { return error; }
}

void decoded_tests() {
    auto early = run_guest(early_exit_program());
    if (const auto* d = std::get_if<Diagnostic>(&early)) {
        check(false, std::string("decoded early exit rejected: ") + d->message);
    } else {
        const auto& artifact = std::get<MslArtifact>(early);
        check(artifact.source.find("out.position.w = 1.0f;") != std::string::npos,
            "Decoded early exit initializes position W at entry");
        check(artifact.source.find("out.builtin2 = float(0);") != std::string::npos,
            "Decoded early exit initializes the scalar point-size output");
        compile_with_metal("output_init_decoded_early_exit", artifact.source);
    }
    auto split = run_guest(split_path_program());
    if (const auto* d = std::get_if<Diagnostic>(&split)) {
        check(false, std::string("decoded split path rejected: ") + d->message);
    } else {
        const auto& artifact = std::get<MslArtifact>(split);
        check(artifact.source.find("out.generic0_0 = float(0);") != std::string::npos &&
              artifact.source.find("out.position.w = 1.0f;") != std::string::npos,
            "Decoded split path initializes both outputs at entry");
        compile_with_metal("output_init_decoded_split_path", artifact.source);
    }
}
} // namespace

namespace armigo::shader::test::output_init {

std::string gpu_vertex_source(unsigned program) {
    auto result = run_guest(program == 0 ? early_exit_program() : split_path_program());
    if (const auto* d = std::get_if<Diagnostic>(&result))
        throw std::runtime_error("GPU vertex emission failed: " + d->message);
    return std::get<MslArtifact>(std::move(result)).source;
}

int main(int argc, char** argv) {
    bool run_gpu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--metal") compile_metal = true;
        else if (argument == "--gpu") run_gpu = true;
        else if (!argument.empty() && argument[0] != '-') {
            fixture_directory = argument;
            std::filesystem::create_directories(fixture_directory);
        }
    }
    structured_tests();
    decoded_tests();
#if defined(ARMIGO_OUTPUT_INIT_GPU)
    if (run_gpu && failures == 0) return run_gpu_cases();
    if (run_gpu) return 1;
#endif
    if (failures) std::cerr << failures << " output initialization tests failed\n";
    else std::cout << "Output initialization tests passed\n";
    return failures ? 1 : 0;
}
} // namespace armigo::shader::test::output_init

int main(int argc, char** argv) {
    return armigo::shader::test::output_init::main(argc, argv);
}
