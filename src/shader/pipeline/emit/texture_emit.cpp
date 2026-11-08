// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Query result sizing follows the guest operation shape. Bindless forms stage
// the handle register as a dynamic input operand in guest source order; the
// constant-buffer origin rules that resolve it live in legalization.

#include "pipeline/dispatch.h"
#include "pipeline/decode/resource_fields.h"
#include "io_memory_shared.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>

namespace armigo::shader::pipeline {
namespace {
constexpr NumericMode float_mode{Domain::floating, ScalarType::f32, Rounding::nearest_even};
constexpr NumericMode signed_mode{Domain::signed_integer, ScalarType::s32};
constexpr NumericMode unsigned_mode{Domain::unsigned_integer, ScalarType::u32};
Operand word(std::uint32_t value) { return LiteralBits{value, ValueType::word32}; }

// Unwind only this instruction expansion; the public callback returns the exact
// diagnostic and the driver discards the failed phase's partial program.
template<class T> T require(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) throw *error;
    return std::move(std::get<T>(result));
}

class TextureEmitter {
public:
    explicit TextureEmitter(EmitterContext& context) : c(context), inst(context.instruction()) {}
    void run();
private:
    EmitterContext& c;
    const DecodedInstruction& inst;
    TextureInfo info;
    TextureDescriptor descriptor;
    NumericMode numeric;
    std::vector<Operand> inputs;
    unsigned a{}, b{}, dest{}, dest2{255}, mask{};
    bool compact{}, half{}, bindless{};

    [[noreturn]] void fail(std::string message, std::string modifier = {},
                          DiagnosticCode code = DiagnosticCode::unsupported_modifier) const {
        throw c.error(code, std::move(message), std::move(modifier));
    }
    unsigned field(DecodedField key, unsigned maximum) const {
        const auto value = require(decoded_field(inst, key));
        if (value > maximum) fail("Texture selector is outside its encoding range");
        return static_cast<unsigned>(value);
    }
    bool flag(DecodedField key) const { return field(key, 1) != 0; }
    unsigned source(SourceRole role) const;
    unsigned destination(RegisterRole role) const;
    Operand read(unsigned& cursor) {
        if (cursor > 255) fail("Texture register sequence exceeds R255", "register_sequence");
        auto value = require(c.read({RegisterFile::gpr, static_cast<std::uint16_t>(cursor++)}));
        // Snapshot before any destination store: query expansions can alias inputs.
        return require(c.unary(Opcode::copy, value, {}, ValueType::word32));
    }
    Operand ra() { return read(a); }
    Operand rb() { return read(b); }
    InputIndex input(Operand value) {
        const auto index = InputIndex{static_cast<std::uint16_t>(inputs.size())};
        inputs.push_back(std::move(value));
        return index;
    }
    void coord(Operand value) { info.coordinates.push_back(input(std::move(value))); }
    void level(Operand value) { info.level_or_bias = input(std::move(value)); }
    void compare(Operand value) { info.compare = input(std::move(value)); }
    void offsets(Operand packed, unsigned count, unsigned stride, unsigned width, unsigned start = 0) {
        for (unsigned i = 0; i < count; ++i) {
            auto extracted = require(c.ternary(Opcode::bit_extract, packed, word(start + i * stride),
                                              word(width), signed_mode, ValueType::word32));
            info.offsets.push_back(input(extracted));
        }
    }
    void shape(unsigned dimension, bool array, bool multisample, bool shadow, bool allow_buffer = false) const;
    unsigned destination_slot(RegisterRole role) const;
    void initialize();
    void regular();
    void texs();
    void tlds();
    void gather_compact();
    void query();
    void lod_query();
    void gradients();
    void emit_sample();
    void write(unsigned base, unsigned lane, Operand value) {
        // Components beyond the last physical destination are discarded, but the
        // texture operation retains its complete mask and all explicit results.
        const auto reg = static_cast<std::uint16_t>(std::min(base + lane, 255U));
        require(c.write({RegisterFile::gpr, reg}, value));
    }
};

unsigned TextureEmitter::source(SourceRole role) const {
    const auto decoded = require(decoded_source(inst, role));
    const auto* reg = std::get_if<Register>(&decoded.value);
    if (!reg || reg->file != RegisterFile::gpr || reg->index > 255 ||
        decoded.modifiers.absolute || decoded.modifiers.negate || decoded.modifiers.subword != Subword::full)
        fail("Texture operands require unmodified general registers", "source");
    return reg->index;
}
unsigned TextureEmitter::destination(RegisterRole role) const {
    const auto result = destination_slot(role);
    if (result == 255) fail("Texture destination register is absent", "destination");
    return result;
}
unsigned TextureEmitter::destination_slot(RegisterRole role) const {
    // RZ destinations are physical discards: the decoder omits the register
    // and the slot reports 255. Duplicates and invalid slots stay errors.
    std::optional<unsigned> found;
    for (const auto& access : inst.registers) {
        if (access.role != role || access.lane != 0) continue;
        if (found || access.reg.file != RegisterFile::gpr || access.reg.index > 255 ||
            access.access == Access::read)
            fail("Texture destination has an invalid or duplicate register role", "destination");
        found = access.reg.index;
    }
    return found.value_or(255);
}
void TextureEmitter::shape(unsigned dimension, bool array, bool multisample, bool shadow, bool allow_buffer) const {
    const auto actual = descriptor.shape;
    const bool buffer = allow_buffer && dimension == 0 && actual.dimension == TextureDimension::buffer;
    if ((!buffer && actual.dimension != static_cast<TextureDimension>(dimension)) || actual.array != array ||
        actual.multisample != multisample || (info.mode != TextureMode::fetch && descriptor.comparison != shadow))
        fail("Texture instruction shape conflicts with the descriptor fact", "texture_shape", DiagnosticCode::conflicting_fact);
}
void TextureEmitter::initialize() {
    bindless = inst.encoding == Encoding::texture_bindless;
    if (inst.encoding != Encoding::texture_bound && !bindless)
        fail("Texture instruction requires a decoded texture encoding", "encoding", DiagnosticCode::malformed_input);
    const bool bindless_name = inst.name == InstructionName::tex_b || inst.name == InstructionName::tld_b ||
        inst.name == InstructionName::tld4_b || inst.name == InstructionName::tmml_b ||
        inst.name == InstructionName::txd_b || inst.name == InstructionName::txq_b;
    if (bindless_name != bindless)
        fail("Bindless texture names and encodings must agree", "encoding", DiagnosticCode::malformed_input);
    for (const auto& access : inst.registers) {
        if (access.access != Access::read && access.reg.file == RegisterFile::predicate && access.reg.index != 7)
            fail("Texture residency predicate output is not implemented", "residency_predicate");
    }
    dest = compact ? destination_slot(RegisterRole::destination) : destination(RegisterRole::destination);
    a = source(SourceRole::a);
    if (inst.name != InstructionName::txq && inst.name != InstructionName::txq_b) b = source(SourceRole::b);
    dest2 = compact ? destination_slot(RegisterRole::second_destination) : 255;
    mask = inst.name == InstructionName::tld4s || inst.name == InstructionName::tld4s_f16 ? 4 :
        field(DecodedField::texture_component_mask, compact ? 7 : 15);
    if (compact) {
        // One destination pair exposes up to two components; two pairs expose
        // three or four. Reserved two-pair selectors must not become empty work.
        constexpr std::array<unsigned, 8> single{1, 2, 4, 8, 3, 9, 10, 12};
        constexpr std::array<unsigned, 5> dual{7, 11, 13, 14, 15};
        if (dest2 == 255) mask = single[mask];
        else {
            if (mask >= dual.size()) fail("Reserved compact texture component mask", "component_mask");
            mask = dual[mask];
        }
    }
    if (mask == 0) fail("Texture component mask is empty", "component_mask");
    if (!bindless) {
        const auto byte_offset = field(DecodedField::texture_handle_byte_offset, UINT32_MAX);
        if (byte_offset % 4) fail("Texture handle byte offset is not word aligned", "handle_byte_offset");
        // A supplied slot is already concrete. An absent slot means the default
        // handle buffer must be obtained through the dependency-recording reader.
        bool explicit_slot = false;
        for (const auto& value : inst.fields) explicit_slot |= value.field == DecodedField::texture_cbuf_slot;
        const auto slot = explicit_slot ? field(DecodedField::texture_cbuf_slot, 17) :
            require(c.facts().require(FactQuery<FactKind::texture_handle_cbuf>{}, Phase::emit_ir, {inst.pc, inst.name}));
        if (slot >= 18) fail("Default texture handle buffer slot is invalid", "texture_handle_cbuf", DiagnosticCode::invalid_fact);
        TextureKey key{static_cast<std::uint8_t>(slot), byte_offset};
        info.texture = key;
        descriptor = require(c.facts().require(fact_query::texture(key), Phase::emit_ir, {inst.pc, inst.name}));
        info.descriptor = descriptor;
        if (descriptor.scalar != ScalarType::f32 && descriptor.scalar != ScalarType::s32 && descriptor.scalar != ScalarType::u32)
            fail("Texture descriptor result type is unsupported", "texture_scalar");
        const bool size_query = inst.name == InstructionName::txq;
        const bool lod_query = inst.name == InstructionName::tmml;
        const auto result_scalar = size_query ? ScalarType::u32 : lod_query ? ScalarType::f32 : descriptor.scalar;
        numeric = inst.numeric.value_or(NumericMode{scalar_domain(result_scalar), result_scalar,
            result_scalar == ScalarType::f32 ? Rounding::nearest_even : Rounding::not_applicable});
        if (!valid_numeric_mode(numeric) || numeric.scalar != result_scalar)
            fail("Texture numeric mode conflicts with its descriptor", "numeric");
        if ((size_query && numeric != unsigned_mode) || (lod_query && numeric != float_mode))
            fail("Texture query numeric modifiers are unsupported", "numeric");
    } else {
        // The descriptor becomes known only when legalization resolves the
        // handle register to a constant-buffer word. Query results keep their
        // instruction-fixed numeric interpretation; descriptor-typed results
        // adopt the resolved component type there, so an instruction-level
        // result modifier could never survive resolution and is rejected now.
        const bool size_query = inst.name == InstructionName::txq_b;
        const bool lod_query = inst.name == InstructionName::tmml_b;
        if (size_query || lod_query) {
            const auto result_scalar = size_query ? ScalarType::u32 : ScalarType::f32;
            numeric = inst.numeric.value_or(NumericMode{scalar_domain(result_scalar), result_scalar,
                result_scalar == ScalarType::f32 ? Rounding::nearest_even : Rounding::not_applicable});
            if (!valid_numeric_mode(numeric) || numeric.scalar != result_scalar ||
                (size_query && numeric != unsigned_mode) || (lod_query && numeric != float_mode))
                fail("Texture query numeric modifiers are unsupported", "numeric");
        } else {
            if (inst.numeric) fail("Bindless sample results derive numeric from the descriptor fact", "numeric");
            numeric = float_mode;
        }
    }
    if (half && numeric.scalar != ScalarType::f32)
        fail("Compact half output requires floating texture components", "half_output");
    info.component_mask = static_cast<std::uint8_t>(mask);
}

void TextureEmitter::regular() {
    const auto dimension = field(DecodedField::texture_dimension, 7);
    const auto count = dimension / 2 == 0 ? 1U : dimension / 2 == 1 ? 2U : 3U;
    const bool array = dimension & 1;
    const bool fetch = inst.name == InstructionName::tld || inst.name == InstructionName::tld_b;
    const bool gather = inst.name == InstructionName::tld4 || inst.name == InstructionName::tld4_b;
    const bool shadow = fetch ? false : flag(DecodedField::texture_compare);
    const bool multisample = fetch && flag(DecodedField::texture_multisample);
    const auto offset = field(DecodedField::texture_offset, gather ? 2 : 1);
    const auto lod = gather ? 0 : field(DecodedField::texture_mode, fetch ? 1 : 7);
    if (fetch) {
        if (flag(DecodedField::texture_clamp)) fail("TLD clamp has no proven lowering", "clamp");
        info.mode = TextureMode::fetch;
    } else {
        if (flag(DecodedField::texture_lod_clamp)) fail("Texture LOD clamp has no proven lowering", "lod_clamp");
        if (flag(DecodedField::texture_ndv)) fail("Non-divergent texture modifier has no proven lowering", "ndv");
        if (gather) info.mode = TextureMode::gather;
        else switch (lod) {
        case 0: info.mode = TextureMode::sample_implicit; break;
        case 1: case 3: info.mode = TextureMode::sample_level; break;
        case 2: info.mode = TextureMode::sample_bias; break;
        // The .A modes are merged; retain uncertainty until the
        // modifier has a proven representation and lowering.
        default: fail("Unsupported texture LOD selector", "lod");
        }
    }
    const bool buffer_candidate = fetch && lod == 0 && !array && !multisample && !offset;
    // The shape check needs the descriptor fact, which bindless emission only
    // obtains when legalization resolves the handle; the operand consumption
    // below follows the instruction's dimension selector either way.
    if (!bindless) shape(dimension / 2, array, multisample, shadow, buffer_candidate);
    if (array) info.array_layer = input(ra());
    // The handle is the first Rb register; every other Rb consumer shifts.
    if (bindless) info.texture = DynamicHandle{input(rb())};
    for (unsigned i = 0; i < count; ++i) coord(ra());
    if (gather) {
        info.gather_component = static_cast<std::uint8_t>(field(DecodedField::gather_component, 3));
        if (shadow && info.gather_component != 0) fail("Shadow gather component must be zero", "gather_component");
        std::array<Operand, 2> packed;
        if (offset) packed[0] = rb();
        if (offset == 2) packed[1] = rb();
        if (shadow) compare(rb());
        for (unsigned i = 0; i < count * (offset == 2 ? 4 : offset); ++i)
            offsets(packed[(i / 4) & 1], 1, 8, 6, (i % 4) * 8);
    } else {
        const bool has_lod = fetch ? lod != 0 : lod > 1;
        if (multisample && has_lod) fail("Multisample fetch with explicit LOD is unsupported", "lod");
        Operand lod_value = has_lod ? rb() : word(0);
        std::optional<Operand> packed;
        if (offset) packed = rb();
        if (shadow) compare(rb());
        if (!multisample && descriptor.shape.dimension != TextureDimension::buffer &&
            (fetch || info.mode != TextureMode::sample_implicit)) level(lod_value);
        if (packed) offsets(*packed, count, 4, 4);
        if (multisample) info.sample_index = input(rb());
    }
    emit_sample();
}

void TextureEmitter::texs() {
    const auto target = field(DecodedField::texture_target, 13);
    const unsigned dimension = target == 0 ? 0 : target < 10 ? 1 : target < 12 ? 2 : 3;
    const bool array = target >= 7 && target <= 9;
    const bool shadow = target == 4 || target == 5 || target == 6 || target == 9;
    const bool explicit_lod = target == 3 || target == 5 || target == 13;
    const bool zero_lod = target == 0 || target == 2 || target == 6 || target == 8 || target == 9 || target == 11;
    info.mode = explicit_lod || zero_lod ? TextureMode::sample_level : TextureMode::sample_implicit;
    shape(dimension, array, false, shadow);
    if (array) {
        info.array_layer = input(ra());
        coord(ra()); coord(rb());
        if (shadow) compare(rb());
    } else if (target == 0) {
        coord(ra());
    } else if (target == 1 || target == 2) {
        coord(ra()); coord(rb());
    } else {
        coord(ra()); coord(ra());
        if (dimension >= 2) coord(rb());
        else if (shadow) compare(rb());
    }
    if (explicit_lod) level(rb());
    else if (zero_lod) level(word(0));
    emit_sample();
}

void TextureEmitter::tlds() {
    const auto target = field(DecodedField::texture_target, 15);
    info.mode = TextureMode::fetch;
    switch (target) {
    case 0: case 1:
        shape(0, false, false, false, target == 0);
        coord(ra());
        if (descriptor.shape.dimension != TextureDimension::buffer) level(target == 1 ? rb() : word(0));
        break;
    case 2:
        shape(1, false, false, false);
        coord(ra()); coord(rb()); level(word(0));
        break;
    case 4: case 5: case 6: case 12:
        shape(1, false, target == 6, false);
        coord(ra()); coord(ra());
        if (target == 6) info.sample_index = input(rb());
        else level(target == 4 ? word(0) : rb());
        if (target == 4 || target == 12) offsets(rb(), 2, 4, 4);
        break;
    case 7:
        shape(2, false, false, false);
        coord(ra()); coord(ra()); coord(rb()); level(word(0));
        break;
    case 8:
        shape(1, true, false, false);
        coord(rb()); coord(rb()); info.array_layer = input(ra()); level(word(0));
        break;
    default: fail("Reserved TLDS target encoding", "target");
    }
    emit_sample();
}

void TextureEmitter::gather_compact() {
    const bool shadow = flag(DecodedField::texture_compare);
    const bool offset = flag(DecodedField::texture_offset);
    info.mode = TextureMode::gather;
    shape(1, false, false, shadow);
    info.gather_component = static_cast<std::uint8_t>(field(DecodedField::gather_component, 3));
    if (shadow && info.gather_component != 0) fail("Shadow gather component must be zero", "gather_component");
    coord(ra());
    coord(shadow || offset ? ra() : rb());
    if (offset) offsets(rb(), 2, 8, 6);
    if (shadow) compare(rb());
    emit_sample();
}

void TextureEmitter::gradients() {
    if (flag(DecodedField::texture_lod_clamp)) fail("TXD LOD clamp is unsupported", "lod_clamp");
    const bool offset = flag(DecodedField::texture_offset);
    const auto dimension = field(DecodedField::texture_dimension, 7);
    const auto count = dimension / 2 == 0 ? 1U : dimension / 2 == 1 ? 2U : 3U;
    const bool array = dimension & 1;
    info.mode = TextureMode::sample_gradients;
    if (!bindless) shape(dimension / 2, array, false, false);
    // TXD_B names the handle in the first Ra register, before the coordinates.
    if (bindless) info.texture = DynamicHandle{input(ra())};
    for (unsigned i = 0; i < count; ++i) coord(ra());
    if (array || offset) {
        auto packed = ra();
        if (array) info.array_layer = input(require(c.binary(Opcode::bit_and, packed, word(0xffff),
                                                            unsigned_mode, ValueType::word32)));
        if (offset) offsets(packed, count, 4, 4, 16);
    }
    // The B registers hold, for each coordinate in turn, its derivative along
    // the first screen axis and then along the second.
    for (unsigned i = 0; i < count; ++i) {
        info.gradients_x.push_back(input(rb()));
        info.gradients_y.push_back(input(rb()));
    }
    emit_sample();
}

void TextureEmitter::emit_sample() {
    const std::vector<ValueType> output_types(std::popcount(mask), ValueType::word32);
    auto results = require(c.texture(Opcode::texture_sample, info, inputs, numeric, output_types));
    if (!half) {
        for (unsigned i = 0; i < results.size(); ++i)
            write(compact && i >= 2 ? dest2 : dest, compact ? i % 2 : i, results[i]);
        return;
    }
    for (unsigned pair = 0; pair < 2; ++pair) {
        std::array<Operand, 2> halves{word(0), word(0)};
        for (unsigned i = 0; i < 2; ++i) if (pair * 2 + i < results.size()) halves[i] = results[pair * 2 + i];
        const std::array<ValueType, 1> output{ValueType::word32};
        auto packed = require(c.emit(Opcode::pack_half2, numeric, halves, output,
                                    ConversionInfo{ScalarType::f32, ScalarType::f16}, Effects::none));
        write(pair ? dest2 : dest, 0, packed[0]);
    }
}

void TextureEmitter::lod_query() {
    if (flag(DecodedField::texture_ndv)) fail("TMML non-divergent modifier is unsupported", "ndv");
    if (mask & 12) fail("TMML Z/W results have no proven semantics", "component_mask");
    const auto dimension = field(DecodedField::texture_dimension, 7);
    const auto count = dimension / 2 == 0 ? 1U : dimension / 2 == 1 ? 2U : 3U;
    info.mode = TextureMode::query_lod;
    if (!bindless) shape(dimension / 2, dimension & 1, false, false);
    if (dimension & 1) info.array_layer = input(ra());
    if (bindless) info.texture = DynamicHandle{input(rb())};
    for (unsigned i = 0; i < count; ++i) coord(ra());
    const std::array<ValueType, 1> output{ValueType::word32};
    unsigned lane = 0;
    for (unsigned component = 0; component < 2; ++component) if (mask & (1U << component)) {
        info.component_mask = 1;
        info.query_component = static_cast<std::uint8_t>(component ^ 1);
        auto result = require(c.texture(Opcode::texture_query, info, inputs, float_mode, output));
        auto scaled = require(c.binary(Opcode::multiply, result[0], word(0x43800000), float_mode, ValueType::word32));
        // Keep the unflushed truncation explicit. Its integral result has no
        // subnormals, so the integer helper's input flush cannot change it.
        auto integral = require(c.unary(Opcode::truncate, scaled, float_mode, ValueType::word32));
        auto fixed = require(c.convert(integral, {ScalarType::f32, ScalarType::s32},
            {Domain::floating, ScalarType::f32, Rounding::toward_zero, true}, ValueType::word32));
        write(dest, lane++, fixed);
    }
}

void TextureEmitter::query() {
    const auto selector = field(DecodedField::texture_query, 63);
    if (selector != 1 && selector != 2) fail("TXQ query selector is unsupported", "query");
    if (selector == 2 && mask != 4) fail("TXQ texture-type query only supports the sample-count component", "component_mask");
    // TXQ_B names the handle in the first Ra register, before the LOD input.
    Operand handle_value;
    if (bindless) handle_value = ra();
    const auto lod = ra();
    const std::array<ValueType, 1> output{ValueType::word32};
    unsigned lane = 0;
    for (unsigned component = 0; component < 4; ++component) if (mask & (1U << component)) {
        inputs.clear();
        // Each expansion rebuilds the input list, so the dynamic handle role
        // is restaged with it.
        if (bindless) info.texture = DynamicHandle{input(handle_value)};
        info.level_or_bias.reset();
        info.component_mask = 1;
        info.query_component = static_cast<std::uint8_t>(component);
        if (selector == 2) {
            if (!bindless && !descriptor.shape.multisample)
                fail("Sample-count query requires a multisample descriptor", "query_samples");
            info.mode = TextureMode::query_samples;
            info.query_component = 0;
        } else if (component == 3) {
            if (!bindless && (descriptor.shape.multisample || descriptor.shape.dimension == TextureDimension::buffer))
                fail("Level-count query requires a mipmapped texture shape", "query_levels");
            info.mode = TextureMode::query_levels;
            info.query_component = 0;
        } else {
            const auto dimension = descriptor.shape.dimension;
            const unsigned spatial_axes = dimension == TextureDimension::d1 || dimension == TextureDimension::buffer ? 1 :
                                          dimension == TextureDimension::d3 ? 3 : 2;
            // For bindless the descriptor is unresolved at emission; the
            // component check reruns at legalization against the fact.
            if (!bindless && component >= spatial_axes + unsigned(descriptor.shape.array))
                fail("TXQ size component is absent from the texture shape", "query_component");
            info.mode = TextureMode::query_size;
            if (!descriptor.shape.multisample && descriptor.shape.dimension != TextureDimension::buffer) level(lod);
        }
        auto result = require(c.texture(Opcode::texture_query, info, inputs, unsigned_mode, output));
        write(dest, lane++, result[0]);
    }
}

void TextureEmitter::run() {
    compact = inst.name == InstructionName::texs || inst.name == InstructionName::texs_f16 ||
        inst.name == InstructionName::tlds || inst.name == InstructionName::tlds_f16 ||
        inst.name == InstructionName::tld4s || inst.name == InstructionName::tld4s_f16;
    half = inst.name == InstructionName::texs_f16 || inst.name == InstructionName::tlds_f16 || inst.name == InstructionName::tld4s_f16;
    initialize();
    switch (inst.name) {
    case InstructionName::tex: case InstructionName::tex_b:
    case InstructionName::tld: case InstructionName::tld_b:
    case InstructionName::tld4: case InstructionName::tld4_b: regular(); break;
    case InstructionName::texs: case InstructionName::texs_f16: texs(); break;
    case InstructionName::tlds: case InstructionName::tlds_f16: tlds(); break;
    case InstructionName::tld4s: case InstructionName::tld4s_f16: gather_compact(); break;
    case InstructionName::tmml: case InstructionName::tmml_b: lod_query(); break;
    case InstructionName::txd: case InstructionName::txd_b: gradients(); break;
    case InstructionName::txq: case InstructionName::txq_b: query(); break;
    default: fail("Instruction is outside the texture emitter family", {}, DiagnosticCode::unsupported_instruction);
    }
}
} // namespace

Status emit_texture_instruction(EmitterContext& context) {
    try {
        TextureEmitter(context).run();
        return std::monostate{};
    } catch (const Diagnostic& error) {
        return error;
    }
}

// SULD/SULD_B/SULD_D/SULD_DB/SUST/SUST_B/SUST_D/SUST_DB: surface (image)
// loads and stores. The sized forms read a shape selector; the plain forms
// carry an RGBA component mask. A bound handle resolves its descriptor fact
// like the sampled emitters; a
// bindless handle stages a dynamic input and legalization resolves it.
// A load whose destination is RZ is dead code and emits nothing. The encoded
// DestPred2 predicate is never written; a guest read of it later falls under
// the established undefined-read contract,
// and a store pads its unmasked texel components with zeros before the write
// so the emitted texel is always the complete four-component vector.
[[noreturn]] void surface_fail(EmitterContext& context, std::string message, std::string modifier = {},
    DiagnosticCode code = DiagnosticCode::unsupported_modifier) {
    throw context.error(code, std::move(message), std::move(modifier));
}
Status emit_surface_access(EmitterContext& context) {
    try {
        const DecodedInstruction& inst = context.instruction();        const auto fail = [&](std::string message, std::string modifier = {},
            DiagnosticCode code = DiagnosticCode::unsupported_modifier) {
            surface_fail(context, std::move(message), std::move(modifier), code);
        };
        const bool store = inst.name == InstructionName::sust || inst.name == InstructionName::sust_b ||
            inst.name == InstructionName::sust_d || inst.name == InstructionName::sust_db;
        const bool bindless = inst.encoding == Encoding::texture_bindless;
        const bool sized = inst.name == InstructionName::suld_d || inst.name == InstructionName::suld_db ||
            inst.name == InstructionName::sust_d || inst.name == InstructionName::sust_db;
        unsigned element_bytes = 4, elements = 1;
        if (sized) {
            const auto bytes = require(decoded_field(inst, DecodedField::memory_element_bytes));
            const auto count = require(decoded_field(inst, DecodedField::memory_elements));
            element_bytes = static_cast<unsigned>(bytes);
            elements = static_cast<unsigned>(count);
        }
        std::uint8_t component_mask = 0;
        if (sized) component_mask = static_cast<std::uint8_t>((1U << elements) - 1U);
        else {
            const auto mask = require(decoded_field(inst, DecodedField::texture_component_mask));
            if (mask == 0 || mask > 15) fail("Surface component mask is empty or invalid", "component_mask");
            component_mask = static_cast<std::uint8_t>(mask);
        }
        const auto dimension_field = require(decoded_field(inst, DecodedField::texture_dimension));
        const TextureDimension dimension = resource_decode::texture_dimension_of(dimension_field);
        const bool instruction_array = (dimension_field >> 3) != 0;
        const unsigned coordinate_count =
            dimension == TextureDimension::d1 || dimension == TextureDimension::buffer ? 1U :
            dimension == TextureDimension::d2 ? 2U : 3U;

        unsigned destination = 255;
        if (!store) {
            std::optional<unsigned> found;
            for (const auto& access : inst.registers) {
                if (access.role != RegisterRole::destination || access.lane != 0) continue;
                if (found || access.reg.file != RegisterFile::gpr || access.reg.index > 255 ||
                    access.access == Access::read)
                    fail("Surface destination has an invalid or duplicate register role", "destination");
                found = access.reg.index;
            }
            // A discarded destination makes the load dead code.
            if (!found) return std::monostate{};
            destination = *found;
        }
        auto value_register = store ? io_memory::store_source_register(inst) : Result<Register>{Register{}};
        if (const auto* error = std::get_if<Diagnostic>(&value_register)) return *error;

        TextureInfo info;
        info.mode = TextureMode::fetch;
        std::vector<Operand> inputs;
        unsigned cursor{};
        const auto next = [&]() -> Operand {
            if (cursor > 255) fail("Surface register sequence exceeds R255", "register_sequence");
            return require(context.read({RegisterFile::gpr, static_cast<std::uint16_t>(cursor++)}));
        };
        if (bindless) {
            const auto handle = require(decoded_source(inst, SourceRole::c));
            const auto* reg = std::get_if<Register>(&handle.value);
            if (!reg || reg->file != RegisterFile::gpr)
                fail("Surface bindless handle is not a general register", "handle");
            cursor = reg->index;
            info.texture = DynamicHandle{InputIndex{static_cast<std::uint16_t>(inputs.size())}};
            inputs.push_back(next());
            if (inst.numeric) fail("Bindless surface results derive numeric from the descriptor fact", "numeric");
        }
        const auto source_a = require(decoded_source(inst, SourceRole::a));
        const auto* coordinate_base = std::get_if<Register>(&source_a.value);
        if (!coordinate_base || coordinate_base->file != RegisterFile::gpr)
            fail("Surface coordinates are not general registers", "source_a");
        cursor = coordinate_base->index;
        std::vector<Operand> coordinates;
        for (unsigned i = 0; i < coordinate_count + (instruction_array ? 1U : 0U); ++i)
            coordinates.push_back(next());
        // The BA forms express the X coordinate in bytes of the element size.
        if (sized && inst.raw.bits<23, 1>()) {
            unsigned shift = 0;
            while ((1U << shift) < element_bytes) ++shift;
            if ((1U << shift) != element_bytes)
                fail("Byte-addressed surface access has no power-of-two element", "size");
            const auto shifted = require(context.binary(Opcode::shift_right, coordinates.front(),
                LiteralBits{shift, ValueType::word32},
                NumericMode{Domain::signed_integer, ScalarType::s32}, ValueType::word32));
            coordinates.front() = shifted;
        }
        // The instruction selector and the descriptor fact may disagree while
        // the derived descriptor is only an assumption; the runtime verifies
        // every consumed fact against the live TIC and stops the draw on any
        // mismatch. Emission therefore stages the axes the descriptor names,
        // padding an instruction-proven subset with zeros; an instruction that
        // proves more axes than the descriptor still fails.
        TextureDescriptor descriptor;
        NumericMode numeric;
        if (!bindless) {
            const auto byte_offset = require(decoded_field(inst, DecodedField::texture_handle_byte_offset));
            if (byte_offset % 4) fail("Surface handle byte offset is not word aligned", "handle_byte_offset");
            const auto slot = require(context.facts().require(FactQuery<FactKind::texture_handle_cbuf>{},
                Phase::emit_ir, {inst.pc, inst.name}));
            if (slot >= 18) fail("Default texture handle buffer slot is invalid", "texture_handle_cbuf",
                DiagnosticCode::invalid_fact);
            const TextureKey key{static_cast<std::uint8_t>(slot), static_cast<std::uint32_t>(byte_offset)};
            info.texture = key;
            descriptor = require(context.facts().require(fact_query::texture(key), Phase::emit_ir,
                {inst.pc, inst.name}));
            info.descriptor = descriptor;
            numeric = NumericMode{scalar_domain(descriptor.scalar), descriptor.scalar,
                descriptor.scalar == ScalarType::f32 ? Rounding::nearest_even : Rounding::not_applicable};
            if (!valid_numeric_mode(numeric)) fail("Surface descriptor numeric mode is invalid", "numeric");
            if (descriptor.shape.multisample)
                fail("Multisample surface access has no proven lowering", "shape",
                    DiagnosticCode::unsupported_capability);
        }
        const unsigned staged_dimensions = bindless ? coordinate_count :
            descriptor.shape.dimension == TextureDimension::d1 ||
                descriptor.shape.dimension == TextureDimension::buffer ? 1U :
            descriptor.shape.dimension == TextureDimension::d2 ? 2U : 3U;
        const bool staged_array = bindless ? instruction_array : descriptor.shape.array;
        if (staged_dimensions < coordinate_count || (staged_array && !instruction_array))
            fail("Surface instruction shape conflicts with the descriptor fact", "texture_shape",
                DiagnosticCode::conflicting_fact);
        for (unsigned i = 0; i < staged_dimensions; ++i)
            info.coordinates.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())}),
                inputs.push_back(i < coordinates.size() ? coordinates[i] :
                    Operand(LiteralBits{0, ValueType::word32}));
        if (staged_array) {
            info.array_layer = InputIndex{static_cast<std::uint16_t>(inputs.size())};
            inputs.push_back(coordinates.back());
        }
        if (store) {
            // The complete texel is written: unmasked components
            // become zero constants, so the mask always covers four components.
            info.component_mask = 15;
            const auto value_base = std::get<Register>(value_register).index;
            for (unsigned i = 0; i < 4; ++i) {
                if ((component_mask & (1U << i)) == 0) {
                    info.data.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())});
                    inputs.push_back(LiteralBits{0, ValueType::word32});
                    continue;
                }
                const auto lane = std::popcount(component_mask & ((1U << i) - 1U));
                const auto source_index = static_cast<std::uint16_t>(
                    value_base + lane < 255 ? value_base + lane : 255);
                info.data.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())});
                inputs.push_back(require(context.read({RegisterFile::gpr, source_index})));
            }
            const std::vector<ValueId> no_outputs;
            auto stored = context.texture(Opcode::image_store, info, inputs, numeric,
                std::span<const ValueType>{});
            if (const auto* error = std::get_if<Diagnostic>(&stored)) return *error;
            return std::monostate{};
        }
        const std::vector<ValueType> output_types(std::popcount(component_mask), ValueType::word32);
        auto results = require(context.texture(Opcode::image_load, info, inputs, numeric, output_types));
        for (unsigned i = 0; i < results.size(); ++i) {
            if (destination + i >= 255) break;
            require(context.write({RegisterFile::gpr, static_cast<std::uint16_t>(destination + i)},
                Operand{results[i]}));
        }
        return std::monostate{};
    } catch (const Diagnostic& error) {
        return error;
    }
}

// SUATOM/SUATOM_B/SUATOM_B2/SUATOM_CAS/SUATOM_CAS_B: surface (image) atomics.
// The integer 32-bit forms lower to the image atomic operation with the
// encoded kind; CAS keeps the expected/
// replacement order and returns the old value like every other surface form.
// A bound handle resolves its descriptor fact like the load/store emitters; a
// bindless handle stages a dynamic input and legalization resolves it. The
// unwritten CAS predicate destination follows the established DestPred2
// treatment. Floating and 64-bit atomic formats remain a structured rejection
// instead of a substituted format.
Status emit_surface_atomic(EmitterContext& context) {
    try {
        const DecodedInstruction& inst = context.instruction();
        const bool bindless = inst.encoding == Encoding::texture_bindless;
        const bool cas = inst.name == InstructionName::suatom_cas ||
            inst.name == InstructionName::suatom_cas_b;
        const auto operation_field = decoded_field(inst, DecodedField::atomic_operation);
        if (const auto* error = std::get_if<Diagnostic>(&operation_field)) return *error;
        const auto operation = std::get<std::uint64_t>(operation_field);
        constexpr AtomicKind kinds[] = {AtomicKind::add, AtomicKind::minimum, AtomicKind::maximum,
            AtomicKind::increment, AtomicKind::decrement, AtomicKind::bit_and, AtomicKind::bit_or,
            AtomicKind::bit_xor, AtomicKind::exchange};
        if (operation >= std::size(kinds))
            surface_fail(context, "Surface atomic operation selector is undefined", "atomic_operation");
        const auto bytes = require(decoded_field(inst, DecodedField::memory_element_bytes));
        const auto count = require(decoded_field(inst, DecodedField::memory_elements));
        if (bytes != 4 || count != 1)
            surface_fail(context, "Surface atomics require a 32-bit integer element", "size",
                DiagnosticCode::unsupported_capability);
        const auto dimension_field = require(decoded_field(inst, DecodedField::texture_dimension));
        const TextureDimension dimension = resource_decode::texture_dimension_of(dimension_field);
        const bool instruction_array = (dimension_field >> 3) != 0;
        const unsigned coordinate_count =
            dimension == TextureDimension::d1 || dimension == TextureDimension::buffer ? 1U :
            dimension == TextureDimension::d2 ? 2U : 3U;
        std::optional<unsigned> destination;
        for (const auto& access : inst.registers) {
            if (access.role != RegisterRole::destination || access.lane != 0) continue;
            if (destination || access.reg.file != RegisterFile::gpr || access.reg.index > 255 ||
                access.access != Access::write)
                surface_fail(context, "Surface atomic destination has an invalid or duplicate register role",
                    "destination");
            destination = access.reg.index;
        }
        if (!destination)
            surface_fail(context, "Surface atomic is missing its destination register", "destination");
        TextureInfo info;
        info.mode = TextureMode::fetch;
        info.atomic = cas ? AtomicKind::compare_exchange : kinds[operation];
        std::vector<Operand> inputs;
        unsigned cursor{};
        const auto next = [&]() -> Operand {
            if (cursor > 255) surface_fail(context, "Surface register sequence exceeds R255", "register_sequence");
            return require(context.read({RegisterFile::gpr, static_cast<std::uint16_t>(cursor++)}));
        };
        if (bindless) {
            const auto handle = require(decoded_source(inst, SourceRole::c));
            const auto* reg = std::get_if<Register>(&handle.value);
            if (!reg || reg->file != RegisterFile::gpr)
                surface_fail(context, "Surface bindless handle is not a general register", "handle");
            cursor = reg->index;
            info.texture = DynamicHandle{InputIndex{static_cast<std::uint16_t>(inputs.size())}};
            inputs.push_back(next());
            if (inst.numeric) surface_fail(context, "Bindless surface results derive numeric from the descriptor fact", "numeric");
        }
        const auto source_a = require(decoded_source(inst, SourceRole::a));
        const auto* coordinate_base = std::get_if<Register>(&source_a.value);
        if (!coordinate_base || coordinate_base->file != RegisterFile::gpr)
            surface_fail(context, "Surface coordinates are not general registers", "source_a");
        cursor = coordinate_base->index;
        std::vector<Operand> coordinates;
        for (unsigned i = 0; i < coordinate_count + (instruction_array ? 1U : 0U); ++i)
            coordinates.push_back(next());
        // The BA form expresses the X coordinate in bytes of the element size
        //.
        if (inst.raw.bits<28, 1>()) {
            const auto shifted = require(context.binary(Opcode::shift_right, coordinates.front(),
                LiteralBits{2, ValueType::word32},
                NumericMode{Domain::signed_integer, ScalarType::s32}, ValueType::word32));
            coordinates.front() = shifted;
        }
        TextureDescriptor descriptor;
        NumericMode numeric;
        if (!bindless) {
            const auto byte_offset = require(decoded_field(inst, DecodedField::texture_handle_byte_offset));
            if (byte_offset % 4) surface_fail(context, "Surface handle byte offset is not word aligned", "handle_byte_offset");
            const auto slot = require(context.facts().require(FactQuery<FactKind::texture_handle_cbuf>{},
                Phase::emit_ir, {inst.pc, inst.name}));
            if (slot >= 18) surface_fail(context, "Default texture handle buffer slot is invalid", "texture_handle_cbuf",
                DiagnosticCode::invalid_fact);
            const TextureKey key{static_cast<std::uint8_t>(slot), static_cast<std::uint32_t>(byte_offset)};
            info.texture = key;
            descriptor = require(context.facts().require(fact_query::texture(key), Phase::emit_ir,
                {inst.pc, inst.name}));
            info.descriptor = descriptor;
            numeric = NumericMode{scalar_domain(descriptor.scalar), descriptor.scalar,
                descriptor.scalar == ScalarType::f32 ? Rounding::nearest_even : Rounding::not_applicable};
            if (!valid_numeric_mode(numeric)) surface_fail(context, "Surface descriptor numeric mode is invalid", "numeric");
            if (descriptor.shape.multisample)
                surface_fail(context, "Multisample surface access has no proven lowering", "shape",
                    DiagnosticCode::unsupported_capability);
            if (descriptor.scalar != ScalarType::u32 && descriptor.scalar != ScalarType::s32)
                surface_fail(context, "Surface atomics require an integer descriptor component", "format",
                    DiagnosticCode::unsupported_capability);
        }
        const unsigned staged_dimensions = bindless ? coordinate_count :
            descriptor.shape.dimension == TextureDimension::d1 ||
                descriptor.shape.dimension == TextureDimension::buffer ? 1U :
            descriptor.shape.dimension == TextureDimension::d2 ? 2U : 3U;
        const bool staged_array = bindless ? instruction_array : descriptor.shape.array;
        if (staged_dimensions < coordinate_count || (staged_array && !instruction_array))
            surface_fail(context, "Surface instruction shape conflicts with the descriptor fact", "texture_shape",
                DiagnosticCode::conflicting_fact);
        for (unsigned i = 0; i < staged_dimensions; ++i)
            info.coordinates.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())}),
                inputs.push_back(i < coordinates.size() ? coordinates[i] :
                    Operand(LiteralBits{0, ValueType::word32}));
        if (staged_array) {
            info.array_layer = InputIndex{static_cast<std::uint16_t>(inputs.size())};
            inputs.push_back(coordinates.back());
        }
        // Data follows guest source order: CAS reads [expected, Rb]
        // then Rb again; a plain atomic reads Rb once.
        const auto data_register = require(decoded_source(inst, SourceRole::b));
        const auto* data_base = std::get_if<Register>(&data_register.value);
        if (!data_base || data_base->file != RegisterFile::gpr || data_base->index > 255)
            surface_fail(context, "Surface atomic data is not a general register", "source_b");
        const auto read_data = [&](unsigned lane) -> Operand {
            const auto index = static_cast<std::uint16_t>(data_base->index + lane < 255 ? data_base->index + lane : 255);
            return require(context.read({RegisterFile::gpr, index}));
        };
        if (cas) {
            info.data.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())});
            inputs.push_back(read_data(0));
            info.data.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())});
            inputs.push_back(read_data(0));
        } else {
            info.data.push_back(InputIndex{static_cast<std::uint16_t>(inputs.size())});
            inputs.push_back(read_data(0));
        }
        info.component_mask = 1;
        const std::vector<ValueType> output_types{ValueType::word32};
        auto results = require(context.texture(Opcode::image_atomic, info, inputs, numeric, output_types));
        require(context.write({RegisterFile::gpr, static_cast<std::uint16_t>(*destination)},
            Operand{results[0]}));
        return std::monostate{};
    } catch (const Diagnostic& error) {
        return error;
    }
}
} // namespace armigo::shader::pipeline
