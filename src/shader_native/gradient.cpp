// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Sampling with explicit derivatives.
//
// The bound form names its handle with a constant-bank word. The register
// form carries the handle in the first coordinate-block register. That block
// then holds the coordinates and, when the shape is an array or texel offsets
// are enabled, one parameter word. The derivative block holds, for each
// coordinate component, the partial along the first screen axis and then the
// partial along the second. The parameter word's low 16 bits are the array
// layer; four-bit signed texel offsets occupy bits 16 onward, one per axis.
// Results fill the first destination and then the second, two components each.

#include "builder.h"
#include <bit>

namespace armigo::shader_native::detail {
namespace {
constexpr ir::NumericMode float_mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode signed_word{ir::Domain::signed_integer, ir::ScalarType::s32};
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
} // namespace

bool Builder::texture_gradient(Fields& fields) {
    const bool bound = opcode_ == 0xb6c;
    const bool register_handle = opcode_ == 0x36d;
    if (!bound && !register_handle) return false;
    enter(fields, false);
    const auto d0 = static_cast<unsigned>(fields.take(16, 8));
    const auto a = static_cast<unsigned>(fields.take(24, 8));
    const auto b = static_cast<unsigned>(fields.take(32, 8));
    std::optional<ir::TextureKey> key;
    if (bound) {
        const auto word_index = static_cast<std::uint32_t>(fields.take(40, 14));
        const auto bank = static_cast<unsigned>(fields.take(54, 5));
        if (fields.flag(59)) fail("bound gradient sample names a register handle");
        if (bank >= 18) fail("texture handle bank exceeds the established IR binding contract");
        key = ir::TextureKey{static_cast<std::uint8_t>(bank), word_index * 4};
    } else {
        if (fields.take(40, 19)) fail("register-handle gradient sample field is not admitted");
        if (!fields.flag(59)) fail("register-handle gradient sample lacks its handle selector");
    }
    if (fields.flag(60)) fail("packed gradient operands are not admitted");
    const auto dimension = static_cast<unsigned>(fields.take(61, 3));
    const auto d1 = static_cast<unsigned>(fields.take(64, 8));
    const auto mask = static_cast<unsigned>(fields.take(72, 4));
    const bool offset = fields.flag(76);
    if (fields.flag(77)) fail("non-divergent gradient mode is not admitted");
    if (fields.take(81, 3) != 7) fail("texture fault predicate is not admitted");
    fields.take(84, 3); // cache eviction priority
    fields.flag(90); // no-dependency scheduling hint
    if (!mask) fail("texture component mask is empty");
    // 1 = 2D, 2 = 3D, 5 = 2D array. One-dimensional shapes, cubes and cube
    // arrays have no admitted lowering here.
    if (dimension != 1 && dimension != 2 && dimension != 5)
        fail("gradient texture dimension selector is not admitted");
    const bool array = dimension == 5;
    const unsigned axes = dimension == 2 ? 3u : 2u;
    const ir::TextureShape shape{static_cast<ir::TextureDimension>(dimension & 3), array, false};

    auto sequence = [&](unsigned first, unsigned count) {
        std::vector<ir::Operand> values;
        if (count && first != 255 && first + count > 255) fail("texture register sequence crosses RZ");
        for (unsigned i = 0; i < count; ++i) values.push_back(first == 255 ? word(0) : reg(first + i));
        return values;
    };
    const unsigned parameter = array || offset ? 1u : 0u;
    auto coordinates = sequence(a, (register_handle ? 1u : 0u) + axes + parameter);
    auto derivatives = sequence(b, axes * 2);
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");

    ir::TextureInfo info;
    info.mode = ir::TextureMode::sample_gradients;
    info.descriptor = derived_texture(shape, false);
    info.component_mask = static_cast<std::uint8_t>(mask);
    std::vector<ir::Operand> inputs;
    auto input = [&](ir::Operand value) {
        const ir::InputIndex index{static_cast<std::uint16_t>(inputs.size())};
        inputs.push_back(std::move(value));
        return index;
    };
    unsigned cursor = 0;
    if (register_handle) info.texture = ir::DynamicHandle{input(coordinates[cursor++])};
    else info.texture = *key;
    for (unsigned i = 0; i < axes; ++i) info.coordinates.push_back(input(coordinates[cursor++]));
    if (parameter) {
        const auto packed = coordinates[cursor++];
        if (array) info.array_layer = input(bits(ir::Opcode::bit_and, {packed, word(0xffffu)}, unsigned_word));
        if (offset) {
            for (unsigned i = 0; i < axes; ++i)
                info.offsets.push_back(input(bits(ir::Opcode::bit_extract, {packed, word(16u + i * 4u), word(4)}, signed_word)));
        }
    }
    for (unsigned i = 0; i < axes; ++i) {
        info.gradients_x.push_back(input(derivatives[i * 2]));
        info.gradients_y.push_back(input(derivatives[i * 2 + 1]));
    }
    const auto count = static_cast<unsigned>(std::popcount(mask));
    std::vector<ir::Operand> outputs;
    for (unsigned i = 0; i < count; ++i) {
        if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        outputs.push_back(ir::ValueId{static_cast<std::uint32_t>(p_.values.size())});
        p_.values.push_back({ir::ValueType::word32});
    }
    operation({{}, ir::Opcode::texture_sample, {}, float_mode, inputs, outputs, info});
    if (d1 == 255 && outputs.size() > 2) fail("more than two texture components need a second destination");
    for (unsigned i = 0; i < outputs.size(); ++i) {
        const unsigned target = i < 2 ? (d0 == 255 ? 255 : d0 + i) : (d1 == 255 ? 255 : d1 + i - 2);
        if (target != 255 && target > 254) fail("texture destination pair crosses RZ");
        write(target, outputs[i]);
    }
    leave(fields);
    return true;
}
} // namespace armigo::shader_native::detail
