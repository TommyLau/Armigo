// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Native texture sampling, gather and queries.
//
// A texture handle word holds the texture ID in bits 0-19 and the sampler ID
// in bits 20-31, the layout of the existing constant-buffer handle key. Bound
// forms name the handle's constant bank and word; the register form passes
// the handle among the operands and the uniform form in a uniform register.
// Sampling and gather operands form one list: the array layer, the
// coordinates, a register handle, the LOD or bias, the texel offsets and the
// depth reference. The packed operand mode splits that list between
// the A and B register ranges, A taking the larger half, as the Maxwell
// compact forms do; packed lists longer than four stay refused. Unpacked,
// a sample keeps the layer and coordinates in A and the rest in B; unpacked
// fetch and gather stay refused. Results fill the first
// destination pair and then the second, or the first destination's
// consecutive registers when only two components are written. A gather
// packs the same binary16 pairs as a sample when its result format asks
// for halves.

#include "builder.h"
#include <bit>
#include <functional>
#include <set>

namespace armigo::shader_native::detail {
namespace {
constexpr ir::NumericMode float_mode{ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::nearest_even};
constexpr ir::NumericMode unsigned_word{ir::Domain::unsigned_integer, ir::ScalarType::u32};
constexpr ir::NumericMode signed_word{ir::Domain::signed_integer, ir::ScalarType::s32};
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
// A bound U32 image. Storage images are not sampled, so coordinates stay
// unnormalized and there is no comparison sampler.
ir::TextureDescriptor integer_surface() {
    ir::TextureDescriptor descriptor;
    descriptor.shape = {ir::TextureDimension::d2, false, false};
    descriptor.scalar = ir::ScalarType::u32;
    descriptor.format = ir::ComponentFormat::unsigned_integer;
    descriptor.format_namespace = "armigo-derived-v1";
    return descriptor;
}
} // namespace

ir::TextureDescriptor derived_texture(ir::TextureShape shape, bool comparison) {
    ir::TextureDescriptor descriptor;
    descriptor.shape = shape;
    descriptor.scalar = ir::ScalarType::f32;
    descriptor.format = comparison ? ir::ComponentFormat::depth : ir::ComponentFormat::floating;
    descriptor.normalized_coordinates = true;
    descriptor.comparison = comparison;
    descriptor.format_namespace = "armigo-derived-v1";
    return descriptor;
}

bool Builder::texture(Fields& fields) {
    const unsigned base = opcode_ & 0x0ff;
    const unsigned family = opcode_ >> 8;
    const bool sample = base == 0x60 || base == 0x61, gather = base == 0x63 || base == 0x64;
    const bool lod_query = base == 0x69 || base == 0x6a, size_query = base == 0x6f || base == 0x70;
    const bool fetch = base == 0x66 || base == 0x67;
    if (!sample && !gather && !lod_query && !size_query && !fetch) return false;
    // Bound forms are 0xb.., register handles 0x3.. and uniform handles 0xf...
    const bool bound = family == 0xb, register_handle = family == 0x3, uniform_handle = family == 0xf;
    if (!bound && !register_handle && !(uniform_handle && (sample || gather))) return false;
    if (bound && base != 0x60 && base != 0x63 && base != 0x66 && base != 0x69 && base != 0x6f) return false;
    if (register_handle && base != 0x61 && base != 0x64 && base != 0x67 && base != 0x6a && base != 0x70) return false;
    if (uniform_handle && base != 0x60 && base != 0x63) return false;
    enter(fields, false);
    const auto d0 = static_cast<unsigned>(fields.take(16, 8)), a = static_cast<unsigned>(fields.take(24, 8));
    const auto b = size_query ? 255u : static_cast<unsigned>(fields.take(32, 8));
    const auto d1 = static_cast<unsigned>(fields.take(64, 8));
    const auto mask = static_cast<unsigned>(fields.take(72, 4));
    if (!mask) fail("texture component mask is empty");
    fields.take(84, 3); // cache eviction priority
    fields.take(90, 1); // no-dependency scheduling hint
    std::optional<ir::TextureKey> key;
    std::optional<unsigned> uniform_index;
    if (bound) {
        const auto offset = static_cast<std::uint32_t>(fields.take(40, 14)) * 4, slot = static_cast<unsigned>(fields.take(54, 5));
        if (slot >= 18) fail("texture handle bank exceeds the established IR binding contract");
        key = ir::TextureKey{static_cast<std::uint8_t>(slot), offset};
    } else if (register_handle) {
        if (fields.take(40, 19)) fail("register-handle texture field is not admitted");
        if (!fields.flag(59)) fail("register-handle texture lacks its handle selector");
    } else {
        uniform_index = static_cast<unsigned>(fields.take(40, 6));
        if (fields.take(46, 13)) fail("uniform-handle texture field is not admitted");
        if (!fields.flag(91)) fail("uniform-handle texture lacks its uniform selector");
    }

    ir::TextureInfo info;
    std::vector<ir::Operand> inputs;
    auto input = [&](ir::Operand value) {
        const ir::InputIndex index{static_cast<std::uint16_t>(inputs.size())};
        inputs.push_back(std::move(value)); return index;
    };
    auto sequence = [&](unsigned first, unsigned count) {
        std::vector<ir::Operand> values;
        if (count && first != 255 && first + count > 255) fail("texture register sequence crosses RZ");
        for (unsigned i = 0; i < count; ++i) values.push_back(first == 255 ? word(0) : reg(first + i));
        return values;
    };
    std::vector<ir::Operand> outputs;
    auto results = [&](unsigned count) {
        outputs.clear();
        for (unsigned i = 0; i < count; ++i) {
            if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
            outputs.push_back(ir::ValueId{static_cast<std::uint32_t>(p_.values.size())}); p_.values.push_back({ir::ValueType::word32});
        }
    };
    // Components fill the first destination pair, then the second; with no
    // second destination only a pair can be written.
    auto write_results = [&](const std::vector<ir::Operand>& values) {
        if (d1 == 255 && values.size() > 2) fail("more than two texture components need a second destination");
        for (unsigned i = 0; i < values.size(); ++i) {
            const unsigned target = i < 2 ? (d0 == 255 ? 255 : d0 + i) : (d1 == 255 ? 255 : d1 + i - 2);
            if (target != 255 && target > 254) fail("texture destination pair crosses RZ");
            write(target, values[i]);
        }
    };

    if (size_query) {
        // TXQ: dimension query per component; the LOD follows a register handle.
        if (fields.take(60, 4) & 3) fail("texture query packing or dimension field is not admitted");
        const auto query = static_cast<unsigned>(fields.raw().bits(62, 2));
        if (query != 0) fail("texture query other than dimensions is not admitted");
        auto list = sequence(a, register_handle ? 2 : 1);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        std::vector<ir::Operand> values;
        for (unsigned component = 0; component < 4; ++component) if (mask & (1u << component)) {
            inputs.clear(); info = {};
            if (register_handle) info.texture = ir::DynamicHandle{input(list[0])};
            else info.texture = *key;
            info.component_mask = 1;
            if (component == 3) info.mode = ir::TextureMode::query_levels;
            else {
                info.mode = ir::TextureMode::query_size;
                info.query_component = static_cast<std::uint8_t>(component);
                info.level_or_bias = input(list.back());
            }
            results(1);
            operation({{}, ir::Opcode::texture_query, {}, unsigned_word, inputs, outputs, info});
            values.push_back(outputs[0]);
        }
        write_results(values);
        leave(fields); return true;
    }

    const bool packed = fields.flag(60);
    bool half_results = false;
    const auto dimension = static_cast<unsigned>(fields.take(61, 3));
    if (dimension == 6) fail("texture dimension selector is not admitted");
    const bool array = dimension >= 4;
    const unsigned axes = (dimension & 3) == 0 ? 1 : (dimension & 3) == 1 ? 2 : 3;
    const ir::TextureShape shape{static_cast<ir::TextureDimension>(dimension & 3), array, false};
    // A gather's bit 77 belongs to its offset mode (76-77).
    if (!gather && fields.flag(77)) fail("non-divergent texture derivative mode is not admitted");
    // Sampling and gather carry a fault predicate at 81-83 (PT when unused);
    // TMML has no such field, so those bits must be clear.
    if (!lod_query && fields.take(81, 3) != 7) fail("texture fault predicate is not admitted");

    if (lod_query) {
        // TMML: array layer and coordinates in A, a register handle in B.
        if (packed) fail("packed LOD query operands are not admitted");
        if (mask & 12) fail("LOD query components beyond the pair are not admitted");
        auto list = sequence(a, axes + (array ? 1 : 0));
        if (register_handle) info.texture = ir::DynamicHandle{input(sequence(b, 1)[0])};
        else info.texture = *key;
        if (array) info.array_layer = input(list[0]);
        for (unsigned i = 0; i < axes; ++i) info.coordinates.push_back(input(list[i + (array ? 1 : 0)]));
        info.mode = ir::TextureMode::query_lod;
        info.descriptor = derived_texture(shape, false);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        // Results are 8.8 fixed point, the host LOD pair in reverse order.
        std::vector<ir::Operand> values;
        auto flushing = float_mode; flushing.flush_inputs = flushing.flush_output = true;
        for (unsigned component = 0; component < 2; ++component) if (mask & (1u << component)) {
            info.component_mask = 1;
            info.query_component = static_cast<std::uint8_t>(component ^ 1);
            results(1);
            operation({{}, ir::Opcode::texture_query, {}, float_mode, inputs, outputs, info});
            auto scaled = temporary(ir::Opcode::multiply, {outputs[0], word(0x43800000u)}, flushing);
            auto integral = temporary(ir::Opcode::truncate, {scaled}, {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::toward_zero, true, true});
            values.push_back(temporary(ir::Opcode::convert, {integral}, {ir::Domain::floating, ir::ScalarType::f32, ir::Rounding::toward_zero, true},
                                       ir::ConversionInfo{ir::ScalarType::f32, ir::ScalarType::s32}));
        }
        write_results(values);
        leave(fields); return true;
    }

    // Sampling, gather and texel fetch.
    bool comparison = false, offsets = false;
    unsigned level_mode = 0, component = 0, gather_offsets = 0;
    if (fetch) {
        // TLD: integer texel coordinates at level zero (LZ) or an explicit
        // level (LL) that follows them in the operand list, then for a bound
        // 2D fetch its texel offsets (bit 76) in one register, as a sample's.
        // The register-handle fetch does not carry that offset.
        offsets = fields.flag(76);
        if (offsets && opcode_ != 0xb66) fail("texel fetch offsets are admitted only on the bound form");
        if (offsets && (shape.dimension != ir::TextureDimension::d2 || array)) fail("texel fetch offsets other than 2D are not admitted");
        if (fields.flag(78)) fail("multisample texel fetch is not admitted");
        if (fields.take(79, 2)) fail("half-precision texture results are not admitted");
        level_mode = static_cast<unsigned>(fields.take(87, 3));
        if (level_mode != 1 && level_mode != 3) fail("texel fetch level mode is not admitted");
        if (shape.dimension == ir::TextureDimension::cube) fail("cube texel fetch is not admitted");
    } else if (gather) {
        // Offsets (bits 76-77): 1, one register after the coordinates with a
        // signed 6-bit field an axis at bits 0 and 8; 2, per-texel offsets in
        // two registers, each texel's x and y in consecutive bytes (6 signed
        // bits each), texel k taking the footprint corner at its offset.
        gather_offsets = static_cast<unsigned>(fields.take(76, 2));
        comparison = fields.flag(78);
        // .F16 (bit 79): the four gathered components pack into two registers.
        // Bit 80 is not a gather result format.
        const auto result_format = fields.take(79, 2);
        if (result_format > 1) fail("texture result format is not admitted");
        half_results = result_format == 1;
        component = static_cast<unsigned>(fields.take(87, 2));
        if (fields.flag(89)) fail("gather component field extension is not admitted");
        if (comparison && component) fail("depth-compare gather selects component zero");
        if (gather_offsets == 3) fail("gather offset mode is not admitted");
        if (gather_offsets && shape.dimension == ir::TextureDimension::cube) fail("cube gather offsets are not admitted");
        if (gather_offsets == 2 && (shape.dimension != ir::TextureDimension::d2 || mask != 15 || comparison))
            fail("per-texel gather offsets need a 2D four-component gather without comparison");
    } else {
        // Texel offsets (bit 76): one register after the level, four signed
        // bits an axis from bit 0, as the guest packs them.
        offsets = fields.flag(76);
        if (offsets && shape.dimension == ir::TextureDimension::cube) fail("cube texture offsets are not admitted");
        comparison = fields.flag(78);
        // .F16 (bit 79): two results to a register, packed as halves.
        const auto result_format = fields.take(79, 2);
        if (result_format > 1) fail("texture result format is not admitted");
        half_results = result_format == 1;
        level_mode = static_cast<unsigned>(fields.take(87, 3));
        if (level_mode > 3) fail("texture LOD clamp modes are not admitted");
    }
    // The operand list, in its documented order. A packed list (bit 60)
    // splits it in halves, the larger in A; otherwise A holds the layer and
    // coordinates and B the rest.
    enum Role { layer, coordinate, handle, level, offset, reference };
    std::vector<Role> roles;
    if (array) roles.push_back(layer);
    for (unsigned i = 0; i < axes; ++i) roles.push_back(coordinate);
    if (register_handle) roles.push_back(handle);
    if (fetch ? level_mode == 3 : level_mode >= 2) roles.push_back(level);
    if (offsets || gather_offsets == 1) roles.push_back(offset);
    if (gather_offsets == 2) { roles.push_back(offset); roles.push_back(offset); }
    if (comparison) roles.push_back(reference);
    if (!packed && (fetch || gather)) fail("unpacked texel fetch and gather operands are not admitted");
    if (packed && roles.size() > 4) fail("packed texture operand list longer than four is not admitted");
    // Half gather results need a bound, packed, four-component gather with
    // both destinations and without a depth comparison. One shared offset
    // register is a different operand list; per-texel offsets stay on the
    // ordinary gather path and pack afterwards.
    if (gather && half_results && (!bound || mask != 15 || gather_offsets == 1 || comparison || d0 == 255 || d1 == 255))
        fail("half-precision gather results are not admitted");
    const unsigned in_a = packed ? static_cast<unsigned>(roles.size() + 1) / 2 : axes + (array ? 1u : 0u);
    auto first = sequence(a, in_a), second = sequence(b, static_cast<unsigned>(roles.size()) - in_a);
    std::vector<ir::Operand> list(first);
    list.insert(list.end(), second.begin(), second.end());
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
    if (uniform_index) info.texture = ir::DynamicHandle{input(uniform(*uniform_index))};
    else if (key) info.texture = *key;
    for (unsigned i = 0; i < roles.size(); ++i) {
        switch (roles[i]) {
        case layer: info.array_layer = input(list[i]); break;
        case coordinate: info.coordinates.push_back(input(list[i])); break;
        case handle: info.texture = ir::DynamicHandle{input(list[i])}; break;
        case level: info.level_or_bias = input(list[i]); break;
        case offset:
            if (gather_offsets == 2)
                for (unsigned byte = 0; byte < 4; ++byte)
                    info.offsets.push_back(input(bits(ir::Opcode::bit_extract, {list[i], word(byte * 8), word(6)}, signed_word)));
            else
                for (unsigned axis = 0; axis < axes; ++axis)
                    info.offsets.push_back(input(bits(ir::Opcode::bit_extract, {list[i], word(axis * (gather ? 8 : 4)), word(gather ? 6 : 4)}, signed_word)));
            break;
        case reference: info.compare = input(list[i]); break;
        }
    }
    if (fetch) {
        info.mode = ir::TextureMode::fetch;
        if (level_mode == 1) info.level_or_bias = input(word(0));
    } else if (gather) {
        info.mode = ir::TextureMode::gather;
        info.gather_component = static_cast<std::uint8_t>(component);
    } else switch (level_mode) {
    case 0: info.mode = ir::TextureMode::sample_implicit; break;
    case 1: info.mode = ir::TextureMode::sample_level; info.level_or_bias = input(word(0)); break;
    case 2: info.mode = ir::TextureMode::sample_bias; break;
    default: info.mode = ir::TextureMode::sample_level; break;
    }
    info.descriptor = derived_texture(shape, comparison);
    info.component_mask = static_cast<std::uint8_t>(mask);
    results(static_cast<unsigned>(std::popcount(mask)));
    operation({{}, ir::Opcode::texture_sample, {}, float_mode, inputs, outputs, info});
    if (half_results) {
        // Results pair up as binary16 halves rounded to nearest even, the
        // first pair to the first destination and the second to the second
        // destination (or the register after the first); an odd count pads
        // the last high half with zero.
        std::vector<ir::Operand> packed_results;
        for (std::size_t pair = 0; pair * 2 < outputs.size(); ++pair)
            packed_results.push_back(temporary(ir::Opcode::pack_half2, {outputs[pair * 2], pair * 2 + 1 < outputs.size() ? outputs[pair * 2 + 1] : word(0)},
                                               float_mode, ir::ConversionInfo{ir::ScalarType::f32, ir::ScalarType::f16}));
        // RZ discards only its own destination, as for unpacked results.
        for (std::size_t i = 0; i < packed_results.size(); ++i) {
            const unsigned target = i == 0 ? d0 : d1 != 255 ? d1 : d0 == 255 ? 255 : d0 + 1;
            if (target != 255 && target > 254) fail("texture destination crosses RZ");
            write(target, packed_results[i]);
        }
    } else write_results(outputs);
    leave(fields); return true;
}

// Descriptor evidence: every texture operation states the shape and compare
// mode its instruction proves. A handle resolved to a constant-bank word
// through copies, constant loads and agreeing phis takes that evidence; any
// disagreement for one key leaves it unknown.
namespace {
// The constant-bank handle word a texture operation reads, through copies,
// constant loads and agreeing phis.
class HandleTracer {
public:
    explicit HandleTracer(const ir::Program& program) : program_(program) {
        for (const auto& op : program.operations)
            for (const auto& output : op.outputs) if (const auto* id = std::get_if<ir::ValueId>(&output)) definitions_[id->value] = &op;
        for (const auto& block : program.blocks) for (const auto& phi : block.phis) phis_[phi.output.value] = &phi;
    }
    std::optional<ir::TextureKey> key(const ir::Operation& op) const {
        const auto& info = std::get<ir::TextureInfo>(op.info);
        if (const auto* fixed = std::get_if<ir::TextureKey>(&info.texture)) return *fixed;
        return trace(op.inputs[std::get<ir::DynamicHandle>(info.texture).input.value], 0);
    }
private:
    std::optional<ir::TextureKey> trace(const ir::Operand& operand, unsigned depth) const {
        if (depth > 64) return std::nullopt;
        if (const auto* word = std::get_if<ir::ConstantWord>(&operand)) return ir::TextureKey{word->slot, word->byte_offset & ~3u};
        const auto* id = std::get_if<ir::ValueId>(&operand);
        if (!id) return std::nullopt;
        if (auto phi = phis_.find(id->value); phi != phis_.end()) {
            std::optional<ir::TextureKey> common;
            for (const auto& incoming : phi->second->inputs) {
                if (incoming.value == phi->second->output) continue;
                auto key = trace(incoming.value, depth + 1);
                if (!key || (common && *common != *key)) return std::nullopt;
                common = key;
            }
            return common;
        }
        auto found = definitions_.find(id->value);
        if (found == definitions_.end()) return std::nullopt;
        const auto& op = *found->second;
        if (op.opcode == ir::Opcode::copy) return trace(op.inputs[0], depth + 1);
        if (op.opcode == ir::Opcode::load) {
            const auto& m = std::get<ir::MemoryInfo>(op.info);
            if (m.space != ir::AddressSpace::constant_buffer || !m.constant_buffer_slot || m.dynamic_byte_offset || m.elements != 1) return std::nullopt;
            return ir::TextureKey{*m.constant_buffer_slot, m.byte_offset & ~3u};
        }
        return std::nullopt;
    }
    const ir::Program& program_;
    std::map<std::uint32_t, const ir::Operation*> definitions_;
    std::map<std::uint32_t, const ir::Phi*> phis_;
};
} // namespace

// An LOD query does not depend on the array layer, so the compiler may encode
// TMML of an array texture as plain 2D: its shape is weak evidence, used only
// for a handle no sampling, gather or fetch proves. Each LOD query then takes
// its handle's descriptor, with a zero layer when that is an array.
void fit_lod_queries(ir::Program& program, const TextureEvidence& evidence) {
    const HandleTracer tracer(program);
    for (auto& op : program.operations) {
        auto* info = std::get_if<ir::TextureInfo>(&op.info);
        if (!info || info->mode != ir::TextureMode::query_lod) continue;
        const auto key = tracer.key(op);
        if (!key) continue;
        const auto found = evidence.find(*key);
        if (found == evidence.end() || !found->second) continue;
        const auto& descriptor = *found->second;
        if (descriptor.shape.dimension != info->descriptor->shape.dimension ||
            (info->array_layer && !descriptor.shape.array))
            throw Failure{ir::diagnose(ir::Phase::legalize, ir::DiagnosticCode::conflicting_fact, {},
                "LOD query dimension disagrees with the handle's sampled shape")};
        if (descriptor.shape.array && !info->array_layer) {
            info->array_layer = ir::InputIndex{static_cast<std::uint16_t>(op.inputs.size())};
            op.inputs.push_back(ir::LiteralBits{0, ir::ValueType::word32});
        }
        info->descriptor = descriptor;
    }
}

void collect_texture_evidence(const ir::Program& program, TextureEvidence& evidence, std::set<ir::TextureKey>& assumed) {
    std::map<std::uint32_t, const ir::Operation*> definitions;
    std::map<std::uint32_t, const ir::Phi*> phis;
    for (const auto& op : program.operations)
        for (const auto& output : op.outputs) if (const auto* id = std::get_if<ir::ValueId>(&output)) definitions[id->value] = &op;
    for (const auto& block : program.blocks) for (const auto& phi : block.phis) phis[phi.output.value] = &phi;
    std::function<std::optional<ir::TextureKey>(const ir::Operand&, unsigned)> trace = [&](const ir::Operand& operand, unsigned depth) -> std::optional<ir::TextureKey> {
        if (depth > 64) return std::nullopt;
        if (const auto* word = std::get_if<ir::ConstantWord>(&operand)) return ir::TextureKey{word->slot, word->byte_offset & ~3u};
        const auto* id = std::get_if<ir::ValueId>(&operand);
        if (!id) return std::nullopt;
        if (auto phi = phis.find(id->value); phi != phis.end()) {
            std::optional<ir::TextureKey> common;
            for (const auto& incoming : phi->second->inputs) {
                if (incoming.value == phi->second->output) continue;
                auto key = trace(incoming.value, depth + 1);
                if (!key || (common && *common != *key)) return std::nullopt;
                common = key;
            }
            return common;
        }
        auto found = definitions.find(id->value);
        if (found == definitions.end()) return std::nullopt;
        const auto& op = *found->second;
        if (op.opcode == ir::Opcode::copy) return trace(op.inputs[0], depth + 1);
        if (op.opcode == ir::Opcode::load) {
            const auto& m = std::get<ir::MemoryInfo>(op.info);
            if (m.space != ir::AddressSpace::constant_buffer || !m.constant_buffer_slot || m.dynamic_byte_offset || m.elements != 1) return std::nullopt;
            return ir::TextureKey{*m.constant_buffer_slot, m.byte_offset & ~3u};
        }
        return std::nullopt;
    };
    std::set<ir::TextureKey> queried;
    std::map<ir::TextureKey, std::optional<ir::TextureDescriptor>> weak;
    for (const auto& op : program.operations) {
        const auto* info = std::get_if<ir::TextureInfo>(&op.info);
        if (!info) continue;
        std::optional<ir::TextureKey> key;
        if (const auto* fixed = std::get_if<ir::TextureKey>(&info->texture)) key = *fixed;
        else key = trace(op.inputs[std::get<ir::DynamicHandle>(info->texture).input.value], 0);
        if (!key) continue;
        if (!info->descriptor) { queried.insert(*key); continue; }
        auto& target = info->mode == ir::TextureMode::query_lod ? weak : evidence;
        auto [entry, inserted] = target.emplace(*key, info->descriptor);
        if (!inserted && entry->second && *entry->second != *info->descriptor) entry->second.reset();
    }
    for (const auto& [key, descriptor] : weak) evidence.emplace(key, descriptor);
    // A handle only dimension queries read proves no shape. As in the Maxwell
    // query-only contract, it is assumed a 2D, non-array, single-sample float
    // texture without comparison, which the host checks against the live
    // descriptor at draw time.
    for (const auto& key : queried) if (!evidence.count(key)) {
        evidence.emplace(key, derived_texture({ir::TextureDimension::d2, false, false}, false));
        assumed.insert(key);
    }
}

// SUST.P [Ra], Rb (0xb9b): a formatted store of an RGBA texel through a bound
// handle (bank bits 54-58, word bits 40-53). The dimension is bits 61-63 in
// the surface numbering (3 is 2D, 5 is 3D); the component mask bits 72-75
// picks the data registers from Rb in order, and unpicked components store
// zero, as the Maxwell surface contract writes a complete texel. SULD.P Rd,
// [Ra] (0xb97) is the formatted load through the same handle and dimension:
// the picked components land in Rd onward in order, and bits 81-84 hold its
// one admitted form (all set: no residency predicate). A 2D surface takes
// two coordinates from Ra and Ra+1; a 3D surface takes a third from Ra+2.
// Register 255 (RZ) reads zero for every coordinate. The image takes the
// derived float descriptor the host checks against the live descriptor. A
// load whose coordinate is outside the image on any axis reads zero; the
// third axis of a 3D image is its depth. SURED (0x59f) is a bound U32
// reduction of a 2D image: the old word is discarded. A buffer reduction
// and the wrapping increment and decrement stay refused.
bool Builder::surface(Fields& fields) {
    if (opcode_ == 0x59f) {
        enter(fields, false);
        fields.take(16, 8); // the reduction writes no register
        const auto coordinate = static_cast<unsigned>(fields.take(24, 8));
        const auto data = static_cast<unsigned>(fields.take(32, 8));
        const auto offset = fields.take(40, 14);
        const auto slot = static_cast<unsigned>(fields.take(54, 5));
        if (slot >= 18) fail("surface handle bank exceeds the established IR binding contract");
        if (fields.take(59, 2)) fail("surface handle or operand mode is not admitted");
        const auto dimension = static_cast<unsigned>(fields.take(61, 3));
        // Bits 87-89 are the operation. Every other bit of that word is the
        // fixed U32 bound form: bits 77-79 and 84 set, the rest clear.
        // The bound offset itself is thirteen bits; bit 53 stays clear.
        if ((fields.raw().words[2] & ~0x03800000u) != 0x0010e000u || offset > 0x1fff)
            fail("surface reduction mode is not admitted");
        fields.take(64, 23);
        const auto operation_kind = static_cast<unsigned>(fields.take(87, 3));
        fields.take(90, 6);
        if (dimension != 1 && dimension != 3) fail("surface reduction dimension other than a buffer or 2D is not admitted");
        if (dimension == 1) fail("surface reduction of a buffer is not admitted");
        if (coordinate > 253) fail("surface register sequence crosses RZ");
        // Increment and decrement wrap against the operand. The shared image
        // atomic path adds one instead, so those two stay refused.
        if (operation_kind == 3 || operation_kind == 4) fail("surface reduction increment or decrement is not admitted");
        static constexpr ir::AtomicKind kinds[]{ir::AtomicKind::add, ir::AtomicKind::minimum, ir::AtomicKind::maximum,
            ir::AtomicKind::increment, ir::AtomicKind::decrement, ir::AtomicKind::bit_and, ir::AtomicKind::bit_or, ir::AtomicKind::bit_xor};
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::TextureInfo info;
        info.mode = ir::TextureMode::fetch;
        info.atomic = kinds[operation_kind];
        info.texture = ir::TextureKey{static_cast<std::uint8_t>(slot), static_cast<std::uint32_t>(offset) * 4};
        info.descriptor = integer_surface();
        info.component_mask = 1;
        std::vector<ir::Operand> inputs;
        for (unsigned axis = 0; axis < 2; ++axis) {
            info.coordinates.push_back(ir::InputIndex{static_cast<std::uint16_t>(inputs.size())});
            inputs.push_back(reg(coordinate + axis));
        }
        info.data.push_back(ir::InputIndex{static_cast<std::uint16_t>(inputs.size())});
        inputs.push_back(data == 255 ? word(0) : reg(data));
        if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const ir::ValueId old{static_cast<std::uint32_t>(p_.values.size())};
        p_.values.push_back({ir::ValueType::word32});
        operation({{}, ir::Opcode::image_atomic, {}, unsigned_word, std::move(inputs), {old}, info});
        write(255, old);
        leave(fields); return true;
    }
    if (opcode_ != 0xb9b && opcode_ != 0xb97) return false;
    const bool load = opcode_ == 0xb97;
    enter(fields, false);
    const auto d = load ? static_cast<unsigned>(fields.take(16, 8)) : 255u;
    const auto a = static_cast<unsigned>(fields.take(24, 8)), b = load ? 255u : static_cast<unsigned>(fields.take(32, 8));
    const auto offset = static_cast<std::uint32_t>(fields.take(40, 14)) * 4, slot = static_cast<unsigned>(fields.take(54, 5));
    if (slot >= 18) fail("surface handle bank exceeds the established IR binding contract");
    if (fields.take(59, 2)) fail("surface handle or operand mode is not admitted");
    const auto dimension = static_cast<unsigned>(fields.take(61, 3));
    if (dimension != 3 && dimension != 5) fail("surface dimension other than 2D and 3D is not admitted");
    const auto mask = static_cast<unsigned>(fields.take(72, 4));
    if (!mask) fail("surface component mask is empty");
    if (load) {
        if (fields.take(81, 4) != 15) fail("surface load predicate or cache mode is not admitted");
    } else fields.take(84, 3); // cache eviction priority
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
    const auto count = static_cast<unsigned>(std::popcount(mask));
    // A 3D surface addresses with three registers; a 2D surface with two.
    // The sequence may end on R254 and must not include RZ.
    const unsigned axes = dimension == 5 ? 3u : 2u;
    if ((a != 255 && a + axes > 255) || (b != 255 && b + count > 255) || (d != 255 && d + count > 255))
        fail("surface register sequence crosses RZ");
    ir::TextureInfo info;
    info.mode = ir::TextureMode::fetch;
    info.texture = ir::TextureKey{static_cast<std::uint8_t>(slot), offset};
    info.descriptor = derived_texture({dimension == 5 ? ir::TextureDimension::d3 : ir::TextureDimension::d2, false, false}, false);
    info.component_mask = static_cast<std::uint8_t>(load ? mask : 15);
    std::vector<ir::Operand> inputs;
    for (unsigned i = 0; i < axes; ++i) {
        info.coordinates.push_back(ir::InputIndex{static_cast<std::uint16_t>(inputs.size())});
        inputs.push_back(a == 255 ? word(0) : reg(a + i));
    }
    if (load) {
        // A load to RZ discards every component: no image is read.
        if (d == 255) { leave(fields); return true; }
        const auto coordinates = inputs;
        std::vector<ir::Operand> outputs;
        for (unsigned i = 0; i < count; ++i) {
            if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
            outputs.push_back(ir::ValueId{static_cast<std::uint32_t>(p_.values.size())});
            p_.values.push_back({ir::ValueType::word32});
        }
        operation({{}, ir::Opcode::image_load, {}, float_mode, std::move(inputs), outputs, info});
        // An out-of-range coordinate reads zero in every component, as the
        // guest's surface load does; Metal reads a format without alpha as
        // alpha 1 there. The coordinates compare unsigned, so a negative one
        // is out of range too. A 3D image also compares its depth.
        std::optional<ir::Operand> inside;
        for (std::uint8_t axis = 0; axis < axes; ++axis) {
            ir::TextureInfo size;
            size.mode = ir::TextureMode::query_size;
            size.texture = info.texture;
            size.descriptor = info.descriptor;
            size.component_mask = 1;
            size.query_component = axis;
            size.level_or_bias = ir::InputIndex{0};
            auto extent = temporary(ir::Opcode::texture_query, {word(0)}, unsigned_word, size);
            auto within = temporary(ir::Opcode::compare, {coordinates[axis], extent}, unsigned_word,
                                    ir::ComparisonInfo{ir::Compare::less, false}, ir::ValueType::predicate);
            inside = inside ? temporary(ir::Opcode::logical_and, {*inside, within}, {ir::Domain::boolean, ir::ScalarType::boolean}, {},
                                        ir::ValueType::predicate)
                            : within;
        }
        for (unsigned i = 0; i < count; ++i)
            write(d + i, temporary(ir::Opcode::select, {*inside, outputs[i], word(0)}, unsigned_word));
        leave(fields); return true;
    }
    unsigned lane = 0;
    for (unsigned component = 0; component < 4; ++component) {
        info.data.push_back(ir::InputIndex{static_cast<std::uint16_t>(inputs.size())});
        inputs.push_back((mask & (1u << component)) && b != 255 ? reg(b + lane) : word(0));
        if (mask & (1u << component)) ++lane;
    }
    operation({{}, ir::Opcode::image_store, {}, float_mode, std::move(inputs), {}, info});
    leave(fields); return true;
}
} // namespace armigo::shader_native::detail
