// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Independently authored instruction streams exercise the attribute contract;
// all fixture words and shader programs are synthetic.

#pragma once

#include "pipeline/phases.h"

#include <bit>

namespace armigo::shader::test::vertex_builtins {
using namespace armigo::shader::pipeline;

// A point covering a 4x4 target exports numeric VertexId/InstanceId in generic
// attribute zero's X/Y components. Each requested ID crosses every production
// phase, including unsigned conversion, rather than starting from hand-built IR.
inline Result<MslArtifact> guest_vertex(unsigned ids = 3) {
    struct Facts final : FactSource {
        Knowledge<FactValue> lookup(FactKey key) const override {
            if (key == FactKey{FactKind::depth_convention})
                return Known<FactValue>{DepthConvention::zero_to_one,
                    {OriginKind::explicit_option, "synthetic-vertex-builtins"}};
            return Unknown{"Synthetic vertex program has no draw-state facts"};
        }
    } source;
    auto take = []<class T>(Result<T> result) -> T {
        if (const auto* error = std::get_if<Diagnostic>(&result)) throw *error;
        return std::get<T>(std::move(result));
    };
    try {
        StageImage image;
        image.stage = Stage::vertex;
        auto append = [&](std::uint64_t word) {
            for (unsigned byte = 0; byte < 8; ++byte)
                image.bytes.push_back(std::byte((word >> (8 * byte)) & 255));
        };
        auto emit = [&](std::uint64_t word) {
            if (image.bytes.size() % 32 == 0) append(0);
            append(word);
        };
        auto literal = [&](unsigned reg, float value) {
            emit(0x0100000000070000ull | reg |
                 (std::uint64_t{std::bit_cast<std::uint32_t>(value)} << 20));
        };
        auto store = [&](unsigned reg, unsigned offset, unsigned words = 1) {
            emit(0xeff000000007ff00ull | reg | (std::uint64_t{offset} << 20) |
                 (std::uint64_t{words - 1} << 47));
        };
        for (unsigned id = 0; id < 2; ++id) if (ids & (1u << id)) {
            const unsigned reg = 4 + id;
            const unsigned offset = id == 0 ? 0x2fc : 0x2f8;
            emit(0xefd800000007ff00ull | reg | (std::uint64_t{offset} << 20));
            // I2F.F32.U32 keeps ID values numeric through the float varying.
            emit(0x5cb8000000070a00ull | reg | (std::uint64_t{reg} << 20));
            store(reg, 0x80 + 4 * id);
        }
        literal(0, 0.0f);
        literal(1, 0.0f);
        literal(2, 0.0f);
        literal(3, 1.0f);
        store(0, 0x70, 4);
        literal(6, 8.0f);
        store(6, 0x6c);
        emit(0xe30000000007000full);
        FactReader facts(source);
        const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        auto decoded = take(decode(image, facts, {}));
        auto registers = take(emit_ir(decoded, facts, {}));
        auto ssa = take(construct_ssa(std::move(registers), {}));
        auto legal = take(legalize(std::move(ssa), facts, profile));
        auto structured = take(structure(std::move(legal), {}));
        auto plan = take(plan_resources(structured, profile));
        return emit_msl(structured, plan, {});
    } catch (const Diagnostic& error) {
        return error;
    }
}

// A point centred on pixel (1, 1) of the 4x4 target exports its size through
// the guest point-size output (attribute 0x6c) only when shader_size is set;
// otherwise the translated stage writes the armigo_point_size function
// constant. Clip (-0.25, 0.25) maps to window (1.5, 1.5), so a size-4 sprite
// covers the pixel centres 0.5..2.5 and a size-1 sprite covers one pixel.
inline Result<MslArtifact> point_vertex(bool shader_size, float size = 4.0f) {
    struct Facts final : FactSource {
        Knowledge<FactValue> lookup(FactKey key) const override {
            if (key == FactKey{FactKind::depth_convention})
                return Known<FactValue>{DepthConvention::zero_to_one,
                    {OriginKind::explicit_option, "synthetic-point-vertex"}};
            return Unknown{"Synthetic vertex program has no draw-state facts"};
        }
    } source;
    auto take = []<class T>(Result<T> result) -> T {
        if (const auto* error = std::get_if<Diagnostic>(&result)) throw *error;
        return std::get<T>(std::move(result));
    };
    try {
        StageImage image;
        image.stage = Stage::vertex;
        auto append = [&](std::uint64_t word) {
            for (unsigned byte = 0; byte < 8; ++byte)
                image.bytes.push_back(std::byte((word >> (8 * byte)) & 255));
        };
        auto emit = [&](std::uint64_t word) {
            if (image.bytes.size() % 32 == 0) append(0);
            append(word);
        };
        auto literal = [&](unsigned reg, float value) {
            emit(0x0100000000070000ull | reg |
                 (std::uint64_t{std::bit_cast<std::uint32_t>(value)} << 20));
        };
        auto store = [&](unsigned reg, unsigned offset, unsigned words = 1) {
            emit(0xeff000000007ff00ull | reg | (std::uint64_t{offset} << 20) |
                 (std::uint64_t{words - 1} << 47));
        };
        literal(0, -0.25f);
        literal(1, 0.25f);
        literal(2, 0.0f);
        literal(3, 1.0f);
        store(0, 0x70, 4);
        if (shader_size) {
            literal(6, size);
            store(6, 0x6c);
        }
        emit(0xe30000000007000full);
        FactReader facts(source);
        const MetalProfile profile{"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
        auto decoded = take(decode(image, facts, {}));
        auto registers = take(emit_ir(decoded, facts, {}));
        auto ssa = take(construct_ssa(std::move(registers), {}));
        auto legal = take(legalize(std::move(ssa), facts, profile));
        auto structured = take(structure(std::move(legal), {}));
        auto plan = take(plan_resources(structured, profile));
        return emit_msl(structured, plan, {});
    } catch (const Diagnostic& error) {
        return error;
    }
}
} // namespace armigo::shader::test::vertex_builtins
