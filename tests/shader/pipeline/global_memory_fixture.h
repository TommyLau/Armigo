// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include "pipeline/phases.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::shader::test::global_memory {
using namespace armigo::shader::pipeline;

inline constexpr std::uint8_t descriptor_slot = 2;
inline constexpr std::uint32_t input_descriptor_offset = 16;
inline constexpr std::uint32_t output_descriptor_offset = 32;

enum class AddressSource { descriptor, literal_zero, literal_nonzero, other_descriptor };

struct Configuration {
    std::int32_t load_displacement{};
    std::int32_t store_displacement{};
    std::uint32_t low_addend{};
    std::uint32_t high_addend{};
    AddressSource low_source{AddressSource::descriptor};
    AddressSource high_source{AddressSource::descriptor};
    bool copy_pair{};
    bool forge_store_high{};
    bool extended{true};
    bool mixed_high{};
    bool ambiguous_pair{};
    // A predicate on the runtime input base selects the output descriptor's
    // complete address pair instead: the load has two descriptor origins
    // through multi-target lowering.
    bool branch_pair{};
    // Route the address pair through shared memory: the guest stores both
    // descriptor words into the shared array and loads them back before the
    // global access. Descriptor-origin resolution must recover the same descriptor
    // the direct configuration uses.
    bool shared_table{};
};

struct Facts final : FactSource {
    explicit Facts(const Configuration& config = {}) : shared_table(config.shared_table) {}
    bool shared_table;
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (key == FactKey{FactKind::compute_local_size})
            return Known<FactValue>{ComputeLocalSize{{1, 1, 1}},
                {OriginKind::explicit_option, "synthetic-global-memory"}};
        if (key == FactKey{FactKind::compute_shared_memory_bytes})
            return Known<FactValue>{shared_table ? std::uint32_t{16} : std::uint32_t{0},
                {OriginKind::explicit_option, "synthetic-global-memory"}};
        return Unknown{"Synthetic descriptor contents remain runtime values"};
    }
};

inline MetalProfile profile() {
    return {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, true};
}

template<class T> T take(Result<T> result) {
    if (const auto* error = std::get_if<Diagnostic>(&result))
        throw std::runtime_error("phase=" + std::to_string(unsigned(error->phase)) +
            " pc=" + std::to_string(error->pc.value_or(0)) + ": " + error->message);
    return std::get<T>(std::move(result));
}

inline std::uint64_t pattern(std::string_view bits) {
    std::uint64_t result{};
    for (unsigned i = 0; i < bits.size(); ++i)
        if (bits[i] == '1') result |= std::uint64_t{1} << (63 - i);
    return result;
}

inline std::uint64_t literal(unsigned dest, std::uint32_t value) {
    return pattern("000000010000") | dest | (7ull << 16) | (std::uint64_t{value} << 20);
}

inline std::uint64_t constant(unsigned dest, std::uint32_t byte_offset) {
    return pattern("1110111110010") | dest | (255ull << 8) | (7ull << 16) |
        (std::uint64_t{byte_offset} << 20) | (std::uint64_t{descriptor_slot} << 36) | (4ull << 48);
}

inline std::uint64_t copy(unsigned dest, unsigned source) {
    return pattern("0101110010011") | dest | (7ull << 16) | (std::uint64_t{source} << 20);
}

inline std::uint64_t add(unsigned dest, unsigned left, unsigned right) {
    return pattern("0101110000010") | dest | (std::uint64_t{left} << 8) |
        (7ull << 16) | (std::uint64_t{right} << 20);
}

// LDS/STS shared forms carry a base register, a signed byte offset and one
// 32-bit word.
inline std::uint64_t shared_access(bool store, unsigned reg, unsigned base, std::int32_t offset) {
    return pattern(store ? "1110111101011" : "1110111101001") | reg |
        (std::uint64_t{base} << 8) | (7ull << 16) |
        (std::uint64_t{static_cast<std::uint32_t>(offset) & 0x00ffffffu} << 20) | (4ull << 48);
}

inline std::uint64_t global(bool store, unsigned value, unsigned address,
                            std::int32_t displacement, bool extended = true) {
    return pattern(store ? "1110111011011" : "1110111011010") | value |
        (std::uint64_t{address} << 8) | (7ull << 16) |
        (std::uint64_t{static_cast<std::uint32_t>(displacement) & 0x00ffffffu} << 20) |
        (std::uint64_t{extended} << 45) | (4ull << 48);
}

inline StageImage image(const Configuration& config = {}) {
    StageImage result;
    result.stage = Stage::compute;
    auto append = [&](std::uint64_t word) {
        for (unsigned byte = 0; byte < 8; ++byte)
            result.bytes.push_back(std::byte((word >> (8 * byte)) & 255));
    };
    auto emit = [&](std::uint64_t word) {
        if (result.bytes.size() % 32 == 0) append(0);
        append(word);
    };
    auto address_word = [&](unsigned reg, AddressSource source, bool high) {
        if (source == AddressSource::literal_zero) emit(literal(reg, 0));
        else if (source == AddressSource::literal_nonzero) emit(literal(reg, 7));
        else emit(constant(reg, (source == AddressSource::other_descriptor ?
            output_descriptor_offset : input_descriptor_offset) + (high ? 4 : 0)));
    };
    address_word(0, config.low_source, false);
    address_word(1, config.high_source, true);
    emit(constant(2, output_descriptor_offset));
    emit(config.forge_store_high ? literal(3, 0) : constant(3, output_descriptor_offset + 4));
    if (config.low_addend) {
        emit(literal(6, config.low_addend));
        emit(add(0, 0, 6));
    }
    if (config.high_addend) {
        emit(literal(6, config.high_addend));
        emit(add(1, 1, 6));
    }
    if (config.ambiguous_pair) {
        emit(add(0, 0, 2));
        emit(add(1, 1, 3));
    }
    if (config.mixed_high) {
        emit(literal(6, 7));
        // ISETP compares a runtime descriptor word; predicating the following
        // move creates two possible high-word definitions at the memory access.
        emit(pattern("010110110110") | 7ull | (7ull << 16) |
            (6ull << 20) | (7ull << 39) | (1ull << 49));
        emit(copy(1, 3) & ~(7ull << 16));
    }
    if (config.branch_pair) {
        emit(literal(6, 7));
        emit(pattern("010110110110") | 7ull | (7ull << 16) |
            (6ull << 20) | (7ull << 39) | (1ull << 49));
        emit(copy(0, 2) & ~(7ull << 16));
        emit(copy(1, 3) & ~(7ull << 16));
    }
    unsigned input_pair = 0;
    if (config.copy_pair) {
        emit(copy(8, 0));
        emit(copy(9, 1));
        input_pair = 8;
    }
    if (config.shared_table) {
        // Stage the address pair in shared memory at word slots 0 and 1, then
        // reload it for the global access (LDS offsets are byte displacements).
        // R8 keeps the table base disjoint from the direct store's R2:R3 pair.
        emit(literal(8, 0));
        emit(shared_access(true, 0, 8, 0));
        emit(shared_access(true, 1, 8, 4));
        emit(shared_access(false, 4, 8, 0));
        emit(shared_access(false, 5, 8, 4));
        emit(global(false, 6, 4, config.load_displacement, config.extended));
        emit(shared_access(true, 6, 8, 8));
        emit(shared_access(false, 7, 8, 8));
        emit(global(true, 7, 2, config.store_displacement, config.extended));
        emit(pattern("111000110000") | (7ull << 16) | 15);
        return result;
    }
    emit(global(false, 4, input_pair, config.load_displacement, config.extended));
    emit(global(true, 4, 2, config.store_displacement, config.extended));
    emit(pattern("111000110000") | (7ull << 16) | 15);
    return result;
}

inline Result<LegalProgram> legalize_image(const Configuration& config = {}) {
    Facts source(config);
    FactReader facts(source);
    auto decoded = decode(image(config), facts, {});
    if (const auto* error = std::get_if<Diagnostic>(&decoded)) return *error;
    auto registers = emit_ir(std::get<DecodedProgram>(decoded), facts, {});
    if (const auto* error = std::get_if<Diagnostic>(&registers)) return *error;
    auto ssa = construct_ssa(std::get<RegisterProgram>(std::move(registers)), {});
    if (const auto* error = std::get_if<Diagnostic>(&ssa)) return *error;
    return legalize(std::get<SsaProgram>(std::move(ssa)), facts, profile());
}

inline MslArtifact artifact(const Configuration& config = {}) {
    auto structured = take(structure(take(legalize_image(config)), {}));
    return take(emit_msl(structured, take(plan_resources(structured, profile())), {}));
}
} // namespace armigo::shader::test::global_memory
