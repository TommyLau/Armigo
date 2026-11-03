// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// FSWZADD (quad swizzle add) verification: synthetic fragment programs run
// through the full pipeline, the MSL quad-lane entry contract, and the CPU
// model of the swizzle-add semantics that the GPU draw test compares
// against. Instruction words below are independently authored synthetic
// fixtures.

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

namespace armigo::shader::test::fswzadd {
std::string gpu_fragment_source(unsigned program);
#if defined(ARMIGO_FSWZADD_GPU)
int run_gpu_cases();
#endif
}

namespace {
using namespace armigo::shader::pipeline;

int failures = 0;
void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::filesystem::path fixture_directory;
bool compile_metal{};

std::string quote(const std::string& value) {
    std::string result = "'";
    for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
    return result + "'";
}

// ---------------------------------------------------------------------------
// The swizzle-add model uses the exact per-code expression over
// the per-lane sign tables. Code 3 multiplies the first source by zero, so a
// NaN or infinity first source propagates and a negative-zero second source
// takes the first source's sign; a plain move would differ in both cases.
// ---------------------------------------------------------------------------

float reference_result(float x, float y, std::uint32_t pn_word, unsigned lane) {
    constexpr float x_table[4] = {1.0F, -1.0F, 1.0F, 0.0F};
    constexpr float y_table[4] = {1.0F, 1.0F, -1.0F, 1.0F};
    const unsigned code = (pn_word >> (lane * 2)) & 3;
    return x * x_table[code] + y * y_table[code];
}

void model_tests() {
    const float x = 1.5F;
    const float y = -2.25F;
    for (std::uint32_t pn = 0; pn <= 255; ++pn) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            const unsigned code = (pn >> (lane * 2)) & 3;
            const float expected[4] = {x + y, y - x, x - y, x * 0.0F + y};
            check(reference_result(x, y, pn, lane) == expected[code],
                  "CPU model matches the per-code expression");
        }
    }
    // The multiply-by-zero move term keeps IEEE edge behavior, unlike a plain
    // move: NaN propagates and signed zero keeps the first source's sign. The
    // checked words put the move code on lane 3 (bits 7..6).
    check(std::bit_cast<std::uint32_t>(reference_result(
              std::bit_cast<float>(0x7fc00000U), 1.0F, 0b11000000U, 3)) == 0x7fc00000U,
        "The move code propagates a NaN first source");
    check(std::bit_cast<std::uint32_t>(reference_result(1.0F, -0.0F, 0b11000000U, 3)) == 0U,
        "The move code gives a negative-zero second source a positive zero");
}

// ---------------------------------------------------------------------------
// Synthetic guest streams: constants in R0/R1, one FSWZADD per colour
// component, one colour store, EXIT.
// ---------------------------------------------------------------------------

constexpr std::uint64_t pattern(std::string_view bits) {
    std::uint64_t result{};
    for (std::size_t i = 0; i < bits.size(); ++i)
        if (bits[i] == '1') result |= std::uint64_t{1} << (63 - i);
    return result;
}
constexpr std::uint64_t guarded(std::uint64_t word, unsigned predicate) {
    return (word & ~(15ull << 16)) | (std::uint64_t{predicate} << 16);
}
constexpr std::uint64_t mov32i(unsigned destination, std::uint32_t bits) {
    return guarded(0x0100000000000000ull | destination | (std::uint64_t{bits} << 20), 7);
}
constexpr std::uint64_t fswzadd(unsigned destination, unsigned a, unsigned b, std::uint32_t pn) {
    return guarded(pattern("0101000011111") | destination | (std::uint64_t{a} << 8) |
                       (std::uint64_t{b} << 20) | (std::uint64_t{pn} << 28), 7);
}
constexpr std::uint64_t store_color(unsigned source, unsigned count) {
    return guarded(0xeff000000000ff00ull | source | (std::uint64_t{0x80} << 20) |
        (std::uint64_t{count - 1} << 47), 7);
}
constexpr std::uint64_t exit_word() { return 0xe30000000007000full; }

StageImage guest_image(Stage stage, const std::vector<std::uint64_t>& words) {
    StageImage image;
    image.stage = stage;
    const auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) image.bytes.push_back(std::byte((word >> (8 * i)) & 255));
    };
    for (const auto word : words) {
        if (image.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    return image;
}

struct FragmentFacts final : FactSource {
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key.kind == FactKind::fragment_output_type)
            return Known<FactValue>{IoType{ScalarType::f32, 4},
                EvidenceOrigin{OriginKind::explicit_option, "synthetic-fswzadd"}};
        return Unknown{"Synthetic swizzle-add program has no other facts"};
    }
};

// Four masks forming a Latin square: across the four colour components every
// two-bit code appears at every quad lane position.
constexpr std::uint32_t component_masks[4] = {0xe4U, 0x39U, 0x4eU, 0x93U};

std::vector<std::uint64_t> fragment_program() {
    return {
        mov32i(0, std::bit_cast<std::uint32_t>(1.5F)),
        mov32i(1, std::bit_cast<std::uint32_t>(-2.25F)),
        fswzadd(4, 0, 1, component_masks[0]),
        fswzadd(5, 0, 1, component_masks[1]),
        fswzadd(6, 0, 1, component_masks[2]),
        fswzadd(7, 0, 1, component_masks[3]),
        store_color(4, 4),
        exit_word(),
    };
}

// The edge-value program multiplies through the move code, pinning the
// multiply form on the GPU: signed zeros keep the first source's
// sign and NaN first or second sources propagate.
std::vector<std::uint64_t> edge_program() {
    return {
        mov32i(0, std::bit_cast<std::uint32_t>(1.0F)),
        mov32i(1, std::bit_cast<std::uint32_t>(-0.0F)),
        mov32i(2, std::bit_cast<std::uint32_t>(-0.0F)),
        mov32i(3, 0x7fc00000U),                           // NaN
        fswzadd(4, 0, 1, 0b11000000U),                    // +1.0, -0.0: move is +0.0
        fswzadd(5, 2, 1, 0b11000000U),                    // -0.0, -0.0: move is -0.0
        fswzadd(6, 3, 0, 0b11000000U),                    // NaN, +1.0: move is NaN
        fswzadd(7, 0, 3, 0b11000000U),                    // +1.0, NaN: move is NaN
        store_color(4, 4),
        exit_word(),
    };
}

template<class T> T take(Result<T> result) {
    if (const auto* d = std::get_if<Diagnostic>(&result)) throw std::runtime_error(d->message);
    return std::get<T>(std::move(result));
}

Result<MslArtifact> run_guest(const std::vector<std::uint64_t>& words, Stage stage) {
    try {
        FragmentFacts facts_source;
        FactReader facts(facts_source);
        const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        auto decoded = decode(guest_image(stage, words), facts, {});
        if (const auto* d = std::get_if<Diagnostic>(&decoded)) return *d;
        auto registers = emit_ir(std::get<DecodedProgram>(std::move(decoded)), facts, {});
        if (const auto* d = std::get_if<Diagnostic>(&registers)) return *d;
        auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(registers)), {});
        if (const auto* d = std::get_if<Diagnostic>(&ssa)) return *d;
        auto legal = legalize(std::get<SsaProgram>(std::move(ssa)), facts, profile);
        if (const auto* d = std::get_if<Diagnostic>(&legal)) return *d;
        auto structured = structure(std::get<LegalProgram>(std::move(legal)), {});
        if (const auto* d = std::get_if<Diagnostic>(&structured)) return *d;
        auto plan = plan_resources(std::get<StructuredProgram>(structured), profile);
        if (const auto* d = std::get_if<Diagnostic>(&plan)) return *d;
        return emit_msl(std::get<StructuredProgram>(structured), std::get<ResourcePlan>(plan), {});
    } catch (const Diagnostic& error) {
        return error;
    }
}

void msl_text_tests() {
    auto artifact = run_guest(fragment_program(), Stage::fragment);
    if (const auto* d = std::get_if<Diagnostic>(&artifact)) {
        check(false, std::string("fragment swizzle-add stage rejected: ") + d->message);
        return;
    }
    const auto& source = std::get<MslArtifact>(std::move(artifact)).source;
    const auto builtin = "builtin" + std::to_string(unsigned(IoSemantic::quad_lane_id));
    check(source.find("[[thread_index_in_quadgroup]]") != std::string::npos,
        "Fragment entry declares the quad position attribute");
    check(source.find(builtin) != std::string::npos,
        "Fragment body reads the quad position builtin");
    check(source.find("quad_shuffle") == std::string::npos &&
          source.find("Helper_SwizzleAdd") == std::string::npos,
        "No quad-transfer helper: the expression needs no data movement");
    check(source.find("* ") != std::string::npos && source.find(" + ") != std::string::npos,
        "The sign-table product expression survives to MSL");

    // Outside fragment stages the quad builtin has no position to read: the
    // same instruction stream in a vertex stage is a structured rejection.
    auto vertex = run_guest(fragment_program(), Stage::vertex);
    const auto* error = std::get_if<Diagnostic>(&vertex);
    check(error != nullptr && error->code == DiagnosticCode::invalid_ir &&
              error->message.find("Invalid compute builtin IO") != std::string::npos,
        "Vertex stages reject the quad position builtin");
}

void fragment_stage_tests() {
    // The GPU fragment sources go through the same pipeline; build them once
    // here so text failures surface without a Metal device too, and compile
    // them with the offline Metal compiler when a fixture directory is given.
    for (unsigned program = 0; program < 2; ++program) {
        try {
            const auto source = armigo::shader::test::fswzadd::gpu_fragment_source(program);
            check(source.find("[[thread_index_in_quadgroup]]") != std::string::npos,
                  "GPU fragment source declares the quad position attribute");
            if (!fixture_directory.empty()) {
                const auto path = fixture_directory / ("fswzadd_fragment_" + std::to_string(program) + ".metal");
                std::ofstream(path) << source;
                if (compile_metal) {
                    const auto command = "xcrun -sdk macosx metal -std=metal4.0 -fmodules-cache-path=" +
                        quote((fixture_directory / "module-cache").string()) + " -c " + quote(path.string()) +
                        " -o " + quote((fixture_directory / ("fswzadd_fragment_" + std::to_string(program) + ".air")).string());
                    check(std::system(command.c_str()) == 0, "Offline Metal compilation");
                }
            }
        } catch (const std::exception& error) {
            check(false, std::string("GPU fragment source failed: ") + error.what());
        }
    }
}
} // namespace

namespace armigo::shader::test::fswzadd {

std::string gpu_fragment_source(unsigned program) {
    const auto words = program == 0 ? fragment_program() : edge_program();
    auto result = run_guest(words, Stage::fragment);
    if (const auto* d = std::get_if<Diagnostic>(&result))
        throw std::runtime_error("GPU fragment emission failed: " + d->message);
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
    model_tests();
    msl_text_tests();
    fragment_stage_tests();
#if defined(ARMIGO_FSWZADD_GPU)
    if (run_gpu && failures == 0) return run_gpu_cases();
    if (run_gpu) return 1;
#endif
    if (failures) std::cerr << failures << " FSWZADD tests failed\n";
    else std::cout << "FSWZADD quad swizzle add tests passed\n";
    return failures ? 1 : 0;
}
} // namespace armigo::shader::test::fswzadd

int main(int argc, char** argv) {
    return armigo::shader::test::fswzadd::main(argc, argv);
}
