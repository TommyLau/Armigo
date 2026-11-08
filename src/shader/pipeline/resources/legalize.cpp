// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "detail.h"
#include "pass_bridge.h"
#include <algorithm>
#include <array>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <map>
#include <set>

namespace armigo::shader::pipeline {
namespace {
using namespace resources;
struct Handle {
    TextureKey texture;
    std::optional<TextureKey> sampler;
    bool operator==(const Handle&) const = default;
};
class Resolver {
public:
    Resolver(Program& program, FactReader& reader, ResourceRequirements& requirements, const MetalProfile& profile)
        : p(program), facts(reader), r(requirements), profile(profile), definitions(p.values.size()), phis(p.values.size()) {
        for (std::uint32_t i = 0; i < p.operations.size(); ++i)
            for (const auto& output : p.operations[i].outputs)
                if (const auto* value = std::get_if<ValueId>(&output)) definitions[value->value] = OperationId{i};
        for (const auto& block : p.blocks)
            for (const auto& phi : block.phis) phis[phi.output.value] = &phi;
    }
    Status run();
private:
    Program& p;
    FactReader& facts;
    ResourceRequirements& r;
    const MetalProfile& profile;
    std::vector<std::optional<OperationId>> definitions;
    std::vector<const Phi*> phis;
    // shared_address_aligned through this program's definition index: one
    // step a lookup instead of a scan of every operation.
    bool aligned(const Operand& value, unsigned alignment, AlignmentMemo* memo = nullptr) {
        return shared_address_aligned_by(p, value, alignment, steps, [&](ValueId id, std::uint64_t& work) {
            ValueDefinitions found;
            ++work;
            if (id.value < definitions.size() && definitions[id.value]) found.operations.push_back(&p.operations[definitions[id.value]->value]);
            if (id.value < phis.size() && phis[id.value]) found.phis.push_back(phis[id.value]);
            return found;
        }, memo);
    }
    std::array<std::optional<ResourceId>, constant_buffer_slot_count> buffers;
    std::optional<Diagnostic> error;
    std::uint64_t steps{};
    SourceLocation source;
    // Descriptor words the guest stages through local/shared memory, keyed by
    // (space, base address value, folded byte constant) in program order. A
    // word written with
    // constant-buffer provenance and read back at the same location hands a
    // global access its storage descriptor; conflicting writers void the key.
    struct MemoryBase {
        TextureKey key{};
        std::optional<Operand> offset; // Guest displacement stored beside the base.
        std::uint32_t const_offset{};
    };
    std::map<std::tuple<unsigned, std::uint32_t, std::uint32_t>, std::optional<MemoryBase>> memory_bases;
    // Word alignment of the values stored at each location, and-ed across all
    // writers; a load from a recorded location inherits it after the write path
    // proves the displacement.
    std::map<std::tuple<unsigned, std::uint32_t, std::uint32_t>, bool> memory_alignment;
    bool fail(std::string message, DiagnosticCode code = DiagnosticCode::unresolved_resource) {
        if (!error) error = diagnose(Phase::legalize, code, source, std::move(message));
        return false;
    }
    bool step(unsigned depth = 0) {
        return (++steps <= p.limits.analysis_steps && depth <= std::min(p.limits.nesting_depth, 256U)) ||
            fail("Resource analysis budget exceeded", DiagnosticCode::limit_exceeded);
    }
    template <FactKind K> std::optional<typename FactTraits<K>::value_type> require(FactQuery<K> query) {
        if (!step()) return {};
        auto result = facts.require(query, Phase::legalize, source);
        if (const auto* e = std::get_if<Diagnostic>(&result)) { error = *e; return {}; }
        return std::get<typename FactTraits<K>::value_type>(result);
    }
    template <FactKind K> std::optional<Knowledge<typename FactTraits<K>::value_type>> read(FactQuery<K> query) {
        if (!step()) return {};
        auto result = facts.read(query, Phase::legalize, source);
        if (const auto* e = std::get_if<Diagnostic>(&result)) { error = *e; return {}; }
        return std::get<Knowledge<typename FactTraits<K>::value_type>>(result);
    }
    const Operation* definition(const Operand& operand) const {
        const auto* id = std::get_if<ValueId>(&operand);
        return id && id->value < definitions.size() && definitions[id->value] ?
            &p.operations[definitions[id->value]->value] : nullptr;
    }
    std::optional<std::uint32_t> exact(const Operand&, unsigned = 0);
    std::optional<std::uint32_t> upper(const Operand&, unsigned = 0);
    std::optional<Handle> handle(const Operand&, unsigned = 0);
    std::optional<TextureKey> masked(const Operand&, std::uint32_t, unsigned);
    std::optional<ResourceId> append(ResourceUse use) {
        if (r.uses.size() >= p.limits.resources) { fail("Resource count exceeds limit", DiagnosticCode::limit_exceeded); return {}; }
        ResourceId id{static_cast<std::uint32_t>(r.uses.size())};
        r.uses.push_back(std::move(use));
        return id;
    }
    std::optional<ResourceId> buffer(std::uint8_t, std::uint64_t);
    std::optional<ResourceId> guest_windows;
    bool texture(Operation&);
    bool pool_texture(Operation&, const std::optional<Handle>&, bool, InputIndex);
    // Every pooled sampler of the stage shares this one binding. Comparison
    // lives on the sampler state an ID names, not on a second pool.
    bool bind_stage_sampler_pool(TextureInfo&);
    void compact_texture_inputs(Operation&);
    bool memory(Operation&);
    std::vector<TextureKey> address_sources(const Operand&, unsigned = 0);
    std::vector<TextureKey> address_union(const Operand&, std::set<std::uint32_t>&, unsigned = 0);
    bool bind_storage(Operation&, TextureKey, ResourceId&, StorageAddress&);
    bool storage_words(const Operation&, const MemoryInfo&, TextureKey);
    bool storage_word_aligned(const Operand&, TextureKey, unsigned = 0);
    bool storage(Operation&);
    bool io(Operation&);
    bool lower_word(Operand&, std::vector<OperationId>&);
    // Descriptor origins propagated through local and shared memory.
    std::optional<std::pair<Operand, std::uint32_t>> strip_constants(const Operand&, unsigned = 0);
    std::optional<std::tuple<unsigned, std::uint32_t, std::uint32_t>> memory_key(
        const MemoryInfo&, const Operation&, unsigned = 0);
    struct BaseAddress {
        TextureKey key;
        std::optional<Operand> offset;
        std::uint32_t const_offset{};
    };
    bool pure_key(const Operand&, TextureKey&, unsigned = 0);
    bool displacement_aligned(const Operand&, unsigned = 0);
    std::optional<BaseAddress> base_address(const Operand&, unsigned = 0);
    const MemoryBase* memory_base_of(const MemoryInfo&, const Operation&, const Operand& result, unsigned = 0);
    void record_memory_stores();
    bool merge_dependencies_now() {
        auto merged = merge_dependencies(p.dependencies, facts.dependencies(), p.limits, Phase::legalize);
        if (const auto* e = std::get_if<Diagnostic>(&merged)) { error = *e; return false; }
        p.dependencies = std::get<std::vector<Dependency>>(std::move(merged));
        return true;
    }
};

std::optional<std::uint32_t> Resolver::exact(const Operand& operand, unsigned depth) {
    if (!step(depth)) return {};
    if (const auto* literal = std::get_if<LiteralBits>(&operand)) {
        if (literal->type == ValueType::word32 && literal->bits <= UINT32_MAX) return static_cast<std::uint32_t>(literal->bits);
        return {};
    }
    if (const auto* word = std::get_if<ConstantWord>(&operand)) {
        const auto byte_offset = word->byte_offset & ~std::uint32_t{3};
        if (byte_offset > constant_buffer_window_bytes - 4) return {};
        auto knowledge = read(fact_query::constant_word(word->slot, byte_offset));
        if (!knowledge) return {};
        if (const auto* known = std::get_if<Known<std::uint32_t>>(&*knowledge)) return known->value;
        return {};
    }
    const auto* op = definition(operand);
    if (!op || op->outputs.size() != 1 || op->numeric.scalar != ScalarType::u32 ||
        (op->numeric.domain != Domain::bits && op->numeric.domain != Domain::unsigned_integer)) return {};
    if (op->opcode == Opcode::copy) return exact(op->inputs[0], depth + 1);
    if (op->opcode == Opcode::load) {
        const auto& m = std::get<MemoryInfo>(op->info);
        if (m.space != AddressSpace::constant_buffer || m.element_bytes != 4 || m.elements != 1 ||
            m.dynamic_byte_offset || m.array_index || m.is_volatile || !m.constant_buffer_slot) return {};
        return exact(ConstantWord{*m.constant_buffer_slot, m.byte_offset}, depth + 1);
    }
    if (op->inputs.size() != 2) return {};
    const auto left = exact(op->inputs[0], depth + 1);
    if (!left) return {};
    const auto right = exact(op->inputs[1], depth + 1);
    if (!right) return {};
    switch (op->opcode) {
    case Opcode::bit_and: return *left & *right;
    case Opcode::bit_or: return *left | *right;
    case Opcode::bit_xor: return *left ^ *right;
    case Opcode::shift_left: return *left << (*right & 31);
    case Opcode::shift_right: return *left >> (*right & 31);
    case Opcode::add: return *left + *right;
    case Opcode::multiply: return *left * *right;
    case Opcode::minimum: return std::min(*left, *right);
    default: return {};
    }
}

std::optional<std::uint32_t> Resolver::upper(const Operand& operand, unsigned depth) {
    if (!step(depth)) return {};
    if (std::holds_alternative<LiteralBits>(operand) || std::holds_alternative<ConstantWord>(operand)) return exact(operand, depth + 1);
    const auto* op = definition(operand);
    if (!op || op->outputs.size() != 1 || op->numeric.scalar != ScalarType::u32 ||
        (op->numeric.domain != Domain::bits && op->numeric.domain != Domain::unsigned_integer)) return {};
    if (op->opcode == Opcode::copy) return upper(op->inputs[0], depth + 1);
    if (op->opcode == Opcode::load) return exact(operand, depth + 1);
    if (op->opcode == Opcode::load_io && r.compute) {
        const auto& io = std::get<IoInfo>(op->info);
        if (io.semantic == IoSemantic::subgroup_lane_id) return 31;
        if (io.semantic == IoSemantic::thread_id && !io.output_storage && std::has_single_bit(io.component_mask)) {
            const auto c = std::countr_zero(unsigned(io.component_mask));
            if (c < 3) return r.compute->local_size.dimensions[c] - 1;
        }
    }
    if (op->inputs.size() != 2) return {};
    // A literal mask or unsigned minimum establishes a range without specializing the other input.
    if (op->opcode == Opcode::bit_and || (op->opcode == Opcode::minimum && op->numeric.domain == Domain::unsigned_integer)) {
        for (unsigned side = 0; side != 2; ++side) {
            if (const auto* literal = std::get_if<LiteralBits>(&op->inputs[side]); literal && literal->type == ValueType::word32) {
                const auto other = upper(op->inputs[1 - side], depth + 1);
                return std::min(static_cast<std::uint32_t>(literal->bits), other.value_or(UINT32_MAX));
            }
        }
    }
    if ((op->opcode == Opcode::shift_left || op->opcode == Opcode::shift_right) &&
        std::holds_alternative<LiteralBits>(op->inputs[1])) {
        const auto a = upper(op->inputs[0], depth + 1);
        if (!a) return {};
        const auto count = std::get<LiteralBits>(op->inputs[1]).bits & 31;
        const auto bound = op->opcode == Opcode::shift_left ? std::uint64_t(*a) << count : *a >> count;
        if (bound <= UINT32_MAX) return static_cast<std::uint32_t>(bound);
        return {};
    }
    auto a = upper(op->inputs[0], depth + 1);
    auto b = upper(op->inputs[1], depth + 1);
    if (!a || !b) return {};
    if (op->opcode == Opcode::bit_and || op->opcode == Opcode::minimum) return std::min(*a, *b);
    std::uint64_t bound;
    if (op->opcode == Opcode::add) bound = std::uint64_t(*a) + *b;
    else if (op->opcode == Opcode::multiply) bound = std::uint64_t(*a) * *b;
    else return {};
    if (bound <= UINT32_MAX) return static_cast<std::uint32_t>(bound);
    return {};
}

std::optional<TextureKey> Resolver::masked(const Operand& operand, std::uint32_t mask, unsigned depth) {
    if (!step(depth)) return {};
    const auto* op = definition(operand);
    if (!op) return {};
    if (op->opcode == Opcode::copy) return masked(op->inputs[0], mask, depth + 1);
    if (op->opcode != Opcode::bit_and || op->numeric.scalar != ScalarType::u32) return {};
    for (unsigned side = 0; side != 2; ++side) {
        auto provenance = handle(op->inputs[side], depth + 1);
        if (!provenance || provenance->sampler) continue;
        auto bits = exact(op->inputs[1 - side], depth + 1);
        if (bits && *bits == mask) return provenance->texture;
        if (error) return {};
    }
    return {};
}

std::optional<Handle> Resolver::handle(const Operand& operand, unsigned depth) {
    if (!step(depth)) return {};
    if (const auto* word = std::get_if<ConstantWord>(&operand)) {
        if (word->byte_offset >= constant_buffer_window_bytes) return {};
        return Handle{{word->slot, word->byte_offset & ~std::uint32_t{3}}, {}};
    }
    const auto* op = definition(operand);
    if (!op) {
        const auto* value = std::get_if<ValueId>(&operand);
        const auto* phi = value && value->value < phis.size() ? phis[value->value] : nullptr;
        if (!phi) return {};
        std::optional<Handle> common;
        for (const auto& input : phi->inputs) {
            if (input.value == phi->output) continue;
            auto incoming = handle(input.value, depth + 1);
            if (!incoming || (common && *common != *incoming)) return {};
            common = incoming;
        }
        return common;
    }
    if (op->outputs.size() != 1 || op->numeric.scalar != ScalarType::u32 ||
        (op->numeric.domain != Domain::bits && op->numeric.domain != Domain::unsigned_integer)) return {};
    if (op->opcode == Opcode::copy) return handle(op->inputs[0], depth + 1);
    if (op->opcode == Opcode::load) {
        const auto& m = std::get<MemoryInfo>(op->info);
        if (m.space != AddressSpace::constant_buffer || !m.constant_buffer_slot || m.element_bytes != 4 ||
            m.elements != 1 || m.array_index || m.is_volatile) return {};
        std::uint64_t offset = m.byte_offset;
        if (m.dynamic_byte_offset) {
            auto dynamic = exact(op->inputs[m.dynamic_byte_offset->value], depth + 1);
            if (!dynamic) return {};
            offset += *dynamic;
        }
        if (offset >= constant_buffer_window_bytes) return {};
        return Handle{{*m.constant_buffer_slot, static_cast<std::uint32_t>(offset) & ~std::uint32_t{3}}, {}};
    }
    if (op->opcode == Opcode::bit_or) {
        for (unsigned side = 0; side != 2; ++side) {
            auto texture = masked(op->inputs[side], 0x000fffffU, depth + 1);
            if (!texture) continue;
            auto sampler = masked(op->inputs[1 - side], 0xfff00000U, depth + 1);
            if (sampler) return Handle{*texture, *sampler};
            if (error) return {};
        }
    }
    return {};
}

// Erases former handle inputs and any level role resolution dropped. Inputs
// shared with coordinate/data roles survive; every surviving role, the pool
// handle words among them, is remapped through the compacted input list.
void Resolver::compact_texture_inputs(Operation& op) {
    auto& t = std::get<TextureInfo>(op.info);
    std::vector<bool> live(op.inputs.size());
    auto mark = [&](InputIndex i) { live[i.value] = true; };
    for (const auto* list : {&t.coordinates, &t.gradients_x, &t.gradients_y, &t.offsets, &t.data})
        for (auto i : *list) mark(i);
    for (const auto i : {t.array_layer, t.array_index, t.level_or_bias, t.compare, t.sample_index, t.lod_clamp,
                         t.pool_texture, t.pool_sampler}) if (i) mark(*i);
    std::vector<std::uint16_t> remap(op.inputs.size());
    std::vector<Operand> inputs;
    for (std::size_t i = 0; i < live.size(); ++i) if (live[i]) {
        remap[i] = static_cast<std::uint16_t>(inputs.size()); inputs.push_back(op.inputs[i]);
    }
    for (auto* list : {&t.coordinates, &t.gradients_x, &t.gradients_y, &t.offsets, &t.data})
        for (auto& i : *list) i.value = remap[i.value];
    for (auto* i : {&t.array_layer, &t.array_index, &t.level_or_bias, &t.compare, &t.sample_index, &t.lod_clamp,
                    &t.pool_texture, &t.pool_sampler})
        if (*i) (*i)->value = remap[(*i)->value];
    op.inputs = std::move(inputs);
}

// A read-only texture handle no constant-buffer word or handle program
// resolves is read from the texture pool of its descriptor's element type,
// indexed by the handle word's texture ID. The sampler is the stage's one
// sampler pool when it is that same word, or a separate dynamic handle that
// does not resolve. A separate sampler that resolves to a constant-buffer
// key or a handle program, and is not itself a combined handle, stays an
// ordinary sampler binding with that key's fact checks
// (the project documentation). The descriptor is the one the frontend
// derived from the stage's own instructions.
bool Resolver::pool_texture(Operation& op, const std::optional<Handle>& sampler_resolved,
                            bool sampler_dynamic, InputIndex sampler_input) {
    auto& t = std::get<TextureInfo>(op.info);
    if (is_image(op) || t.atomic || t.array_index)
        return fail("Bindless texture handle has no proven constant-buffer provenance");
    if (!t.descriptor || !valid_descriptor(*t.descriptor) || t.descriptor->shape.dimension == TextureDimension::buffer)
        return fail("Pool texture has no admitted descriptor from its instructions", DiagnosticCode::missing_fact);
    const auto descriptor = *t.descriptor;
    const auto texture_input = std::get<DynamicHandle>(t.texture).input;
    enum class SamplerBinding { none, keyed, pool };
    auto binding = SamplerBinding::none;
    TextureKey keyed{};
    InputIndex pool_input{};
    if (needs_sampler(op)) {
        if (!t.sampler) {
            binding = SamplerBinding::pool;
            pool_input = texture_input;
        } else if (sampler_resolved && sampler_resolved->sampler) {
            return fail("Separate sampler handle has no unambiguous provenance");
        } else if (sampler_resolved) {
            binding = SamplerBinding::keyed;
            keyed = sampler_resolved->texture;
        } else if (sampler_dynamic) {
            binding = SamplerBinding::pool;
            pool_input = sampler_input;
        } else return fail("Separate sampler handle has no unambiguous provenance");
    } else if (t.sampler) return fail("Sampler handle is not used by this texture mode");
    // As for a dynamic handle resolved to a key: results take the
    // descriptor's interpretation, size-query components and the level role
    // follow its shape.
    if (t.mode < TextureMode::query_lod) {
        const auto scalar = descriptor.scalar;
        op.numeric = NumericMode{scalar_domain(scalar), scalar,
            scalar == ScalarType::f32 ? Rounding::nearest_even : Rounding::not_applicable};
    }
    if (op.opcode == Opcode::texture_query && t.mode == TextureMode::query_size) {
        const auto dimension = descriptor.shape.dimension;
        const unsigned spatial_axes = dimension == TextureDimension::d1 ? 1 : dimension == TextureDimension::d3 ? 3 : 2;
        if (t.query_component >= spatial_axes + unsigned(descriptor.shape.array))
            return fail("Texture size-query component is absent from the descriptor shape", DiagnosticCode::conflicting_fact);
    }
    const bool level_required = t.mode == TextureMode::sample_level || t.mode == TextureMode::sample_bias ||
        ((t.mode == TextureMode::fetch || t.mode == TextureMode::query_size) && !descriptor.shape.multisample);
    if (t.level_or_bias && !level_required) t.level_or_bias.reset();
    t.texture = TextureKey{texture_pool_slot, 0};
    t.pool_texture = texture_input;
    if (binding == SamplerBinding::keyed) t.sampler = keyed;
    else if (binding == SamplerBinding::pool) {
        t.sampler = TextureKey{texture_pool_slot, 0};
        t.pool_sampler = pool_input;
    } else t.sampler.reset();
    compact_texture_inputs(op);
    auto checked = check_texture(op, descriptor, profile, p.stage, Phase::legalize);
    if (auto* e = std::get_if<Diagnostic>(&checked)) { error = *e; return false; }
    // One texture pool per element type. Pooled samplers share one pool.
    for (std::uint32_t i = 0; i < r.uses.size() && !t.resource; ++i) {
        if (!step()) return false;
        auto* use = std::get_if<TextureUse>(&r.uses[i]);
        if (!use || !use->pool || use->descriptor != descriptor) continue;
        auto position = std::lower_bound(use->modes.begin(), use->modes.end(), t.mode);
        if (position == use->modes.end() || *position != t.mode) use->modes.insert(position, t.mode);
        t.resource = ResourceId{i};
    }
    if (!t.resource) {
        TextureUse use{ResourceTextureKind::sampled, false, false, TextureKey{texture_pool_slot, 0}, descriptor, Access::read, 1, false, {t.mode}};
        use.pool = true;
        t.resource = append(std::move(use));
        if (!t.resource) return false;
    }
    if (binding == SamplerBinding::keyed) {
        // The same fact checks a resolved separate sampler takes: the key's
        // descriptor must agree with the texture on comparison and normalized
        // coordinates, and the binding records that state.
        auto sampler_descriptor = require(fact_query::texture(keyed));
        if (!sampler_descriptor) return false;
        if (!valid_descriptor(*sampler_descriptor) || sampler_descriptor->comparison != descriptor.comparison ||
            sampler_descriptor->normalized_coordinates != descriptor.normalized_coordinates)
            return fail("Separate sampler state is incompatible with the texture descriptor", DiagnosticCode::invalid_fact);
        for (std::uint32_t i = 0; i < r.uses.size() && !t.sampler_resource; ++i) {
            if (!step()) return false;
            const auto* use = std::get_if<SamplerUse>(&r.uses[i]);
            if (use && !use->pool && use->key == keyed && use->array_length == 1 &&
                use->comparison == descriptor.comparison && use->normalized_coordinates == descriptor.normalized_coordinates)
                t.sampler_resource = ResourceId{i};
        }
        if (!t.sampler_resource) {
            t.sampler_resource = append(SamplerUse{keyed, descriptor.comparison, descriptor.normalized_coordinates, 1});
            if (!t.sampler_resource) return false;
        }
    } else if (binding == SamplerBinding::pool && !bind_stage_sampler_pool(t)) return false;
    return true;
}

bool Resolver::bind_stage_sampler_pool(TextureInfo& t) {
    for (std::uint32_t i = 0; i < r.uses.size() && !t.sampler_resource; ++i) {
        if (!step()) return false;
        const auto* use = std::get_if<SamplerUse>(&r.uses[i]);
        if (use && use->pool) t.sampler_resource = ResourceId{i};
    }
    if (!t.sampler_resource) {
        SamplerUse use{TextureKey{texture_pool_slot, 0}, false, true, 1};
        use.pool = true;
        t.sampler_resource = append(std::move(use));
        if (!t.sampler_resource) return false;
    }
    return true;
}

std::optional<ResourceId> Resolver::buffer(std::uint8_t slot, std::uint64_t extent) {
    if (slot >= constant_buffer_slot_count) {
        fail("Invalid constant-buffer slot", DiagnosticCode::invalid_ir);
        return {};
    }
    extent = std::clamp<std::uint64_t>(extent, 4, constant_buffer_window_bytes);
    if (!buffers[slot]) {
        buffers[slot] = append(BufferUse{ConstantBufferSource{slot}, Access::read,
            {SizeKind::constant_buffer_window, extent, 4, ConstantBufferPolicy::runtime_zero_words_v1}});
    } else {
        auto& use = std::get<BufferUse>(r.uses[buffers[slot]->value]);
        use.size.bytes = std::max(use.size.bytes, extent);
    }
    return buffers[slot];
}

bool Resolver::texture(Operation& op) {
    auto& t = std::get<TextureInfo>(op.info);
    // Implicit-LOD sampling has no host representation outside the fragment
    // stage: legalization turns it into an explicit-level sample, reusing the
    // bias operand as the level when one is encoded. Level-zero sampling keeps the
    // derivative effect recorded at emission ("optimizers may retain extra
    // effects"), and check_texture re-checks the remaining modes.
    if (p.stage != Stage::fragment && op.opcode == Opcode::texture_sample &&
        (t.mode == TextureMode::sample_implicit || t.mode == TextureMode::sample_bias)) {
        t.mode = TextureMode::sample_level;
        if (!t.level_or_bias) {
            t.level_or_bias = InputIndex{static_cast<std::uint16_t>(op.inputs.size())};
            op.inputs.push_back(LiteralBits{0, ValueType::word32});
        }
    }
    const bool dynamic_texture = std::holds_alternative<DynamicHandle>(t.texture);
    auto resolve = [&](const HandleSource& input) -> std::optional<Handle> {
        if (const auto* key = std::get_if<TextureKey>(&input)) return Handle{*key, {}};
        return handle(op.inputs[std::get<DynamicHandle>(input).input.value]);
    };
    auto resolved = resolve(t.texture);
    if (error) return false;
    // A constant-buffer key or a handle program is already resolved and is
    // never rewritten into a pool access. Only an unresolved dynamic texture
    // reads the texture pool, and only when pools are enabled.
    if (!resolved) {
        if (!dynamic_texture || !profile.texture_pools)
            return fail("Bindless texture handle has no proven constant-buffer provenance");
        std::optional<Handle> sampler_resolved;
        bool sampler_dynamic = false;
        InputIndex sampler_input{};
        if (t.sampler) {
            if (const auto* dynamic = std::get_if<DynamicHandle>(&*t.sampler)) {
                sampler_dynamic = true;
                sampler_input = dynamic->input;
                sampler_resolved = handle(op.inputs[sampler_input.value]);
                if (error) return false;
            } else sampler_resolved = Handle{std::get<TextureKey>(*t.sampler), {}};
        }
        return pool_texture(op, sampler_resolved, sampler_dynamic, sampler_input);
    }
    std::optional<Handle> sampler;
    bool sampler_dynamic = false;
    InputIndex sampler_input{};
    if (t.sampler) {
        if (const auto* dynamic = std::get_if<DynamicHandle>(&*t.sampler)) {
            sampler_dynamic = true;
            sampler_input = dynamic->input;
            sampler = handle(op.inputs[sampler_input.value]);
            if (error) return false;
        } else sampler = Handle{std::get<TextureKey>(*t.sampler), {}};
        // A resolved separate sampler that is itself a combined handle has no
        // single key. An unresolved dynamic sampler is the stage's sampler
        // pool when pools are enabled, and the old refusal otherwise.
        if (sampler && sampler->sampler) return fail("Separate sampler handle has no unambiguous provenance");
        if (!sampler && !profile.texture_pools) return fail("Separate sampler handle has no unambiguous provenance");
        if (sampler && resolved->sampler && *resolved->sampler != sampler->texture)
            return fail("Combined and explicit sampler keys disagree");
    }
    const bool pool_sampler = needs_sampler(op) && sampler_dynamic && !sampler;
    if (!needs_sampler(op) && (sampler || resolved->sampler || t.sampler)) return fail("Sampler handle is not used by this texture mode");
    t.texture = resolved->texture;
    if (needs_sampler(op)) {
        if (pool_sampler) {
            t.sampler = TextureKey{texture_pool_slot, 0};
            t.pool_sampler = sampler_input;
        } else t.sampler = sampler ? sampler->texture : resolved->sampler.value_or(resolved->texture);
    }
    auto descriptor = require(fact_query::texture(resolved->texture));
    if (!descriptor) return false;
    if (t.descriptor && *t.descriptor != *descriptor) return fail("Encoded texture descriptor conflicts with consumed fact", DiagnosticCode::conflicting_fact);
    t.descriptor = *descriptor;
    // Descriptor-typed results of a dynamic handle read their numeric
    // interpretation from the resolved descriptor, exactly as bound emission
    // derives it from the descriptor fact; query results keep the
    // instruction-fixed modes staged at emission.
    if (dynamic_texture && t.mode < TextureMode::query_lod) {
        const auto scalar = descriptor->scalar;
        op.numeric = NumericMode{scalar_domain(scalar), scalar,
            scalar == ScalarType::f32 ? Rounding::nearest_even : Rounding::not_applicable};
    }
    // Size-query components are validated against the resolved descriptor
    // shape; emission could not for a dynamic handle and stays authoritative
    // for a bound one.
    if (op.opcode == Opcode::texture_query && t.mode == TextureMode::query_size) {
        const auto dimension = descriptor->shape.dimension;
        const unsigned spatial_axes = dimension == TextureDimension::d1 || dimension == TextureDimension::buffer ? 1 :
                                      dimension == TextureDimension::d3 ? 3 : 2;
        if (t.query_component >= spatial_axes + unsigned(descriptor->shape.array))
            return fail("Texture size-query component is absent from the descriptor shape",
                DiagnosticCode::conflicting_fact);
    }
    // Level presence depends on the resolved shape: emission stages the guest
    // level operand whenever the instruction carries one, and resolution drops
    // it for shapes resolved without a level. A required but
    // unstaged level remains a structured rejection.
    const bool level_required = t.mode == TextureMode::sample_level || t.mode == TextureMode::sample_bias ||
        ((t.mode == TextureMode::fetch || t.mode == TextureMode::query_size) &&
         !descriptor->shape.multisample && descriptor->shape.dimension != TextureDimension::buffer);
    if (dynamic_texture && t.level_or_bias && !level_required) t.level_or_bias.reset();
    std::uint32_t length = 1;
    if (t.array_index) {
        auto count = require(FactQuery<FactKind::texture_array_length>{resolved->texture.cbuf_slot, resolved->texture.handle_byte_offset});
        if (!count) return false;
        if (*count == 0) return fail("Texture resource array is empty", DiagnosticCode::invalid_fact);
        auto maximum = upper(op.inputs[t.array_index->value]);
        if (!maximum || *maximum >= *count) return error ? false : fail("Texture resource-array index has no proven in-range bound");
        length = *count;
    }
    if (needs_sampler(op) && !pool_sampler) {
        const auto key = std::get<TextureKey>(*t.sampler);
        auto sampler_descriptor = require(fact_query::texture(key));
        if (!sampler_descriptor) return false;
        if (!valid_descriptor(*sampler_descriptor) || sampler_descriptor->comparison != descriptor->comparison ||
            sampler_descriptor->normalized_coordinates != descriptor->normalized_coordinates)
            return fail("Separate sampler state is incompatible with the texture descriptor", DiagnosticCode::invalid_fact);
        if (t.array_index && key != resolved->texture) {
            auto sampler_count = require(FactQuery<FactKind::texture_array_length>{key.cbuf_slot, key.handle_byte_offset});
            if (!sampler_count) return false;
            if (*sampler_count != length) return fail("Texture and sampler arrays do not share a proven range");
        }
    }
    compact_texture_inputs(op);
    auto checked = check_texture(op, *descriptor, profile, p.stage, Phase::legalize);
    if (auto* e = std::get_if<Diagnostic>(&checked)) { error = *e; return false; }
    const auto kind = is_image(op) ? ResourceTextureKind::storage_image : ResourceTextureKind::sampled;
    const auto access = op.opcode == Opcode::image_store ? Access::write : op.opcode == Opcode::image_atomic ? Access::read_write : Access::read;
    for (std::uint32_t i = 0; i < r.uses.size(); ++i) {
        if (!step()) return false;
        auto* use = std::get_if<TextureUse>(&r.uses[i]);
        if (use && use->kind == kind && use->key == resolved->texture && use->array_length == length) {
            if (use->descriptor != *descriptor) return fail("Resource descriptor changed", DiagnosticCode::conflicting_fact);
            use->access = combine(use->access, access);
            use->atomic |= t.atomic.has_value(); use->coherent |= t.coherent;
            use->dynamically_indexed |= t.array_index.has_value();
            auto position = std::lower_bound(use->modes.begin(), use->modes.end(), t.mode);
            if (position == use->modes.end() || *position != t.mode) use->modes.insert(position, t.mode);
            t.resource = ResourceId{i}; break;
        }
    }
    if (!t.resource) t.resource = append(TextureUse{kind, t.coherent, t.atomic.has_value(), resolved->texture,
        *descriptor, access, length, t.array_index.has_value(), {t.mode}});
    if (!t.resource) return false;
    if (needs_sampler(op) && pool_sampler) {
        if (!bind_stage_sampler_pool(t)) return false;
    } else if (needs_sampler(op)) {
        const auto key = std::get<TextureKey>(*t.sampler);
        for (std::uint32_t i = 0; i < r.uses.size(); ++i) {
            if (!step()) return false;
            const auto* use = std::get_if<SamplerUse>(&r.uses[i]);
            if (use && use->key == key && use->array_length == length && use->comparison == descriptor->comparison &&
                use->normalized_coordinates == descriptor->normalized_coordinates) { t.sampler_resource = ResourceId{i}; break; }
        }
        if (!t.sampler_resource) t.sampler_resource = append(SamplerUse{key, descriptor->comparison, descriptor->normalized_coordinates, length});
        if (!t.sampler_resource) return false;
    }
    return true;
}

// Track additive descriptor origins through SSA, retaining only origins present
// on every selected path. An origin repeated in an addition is not a base.
std::vector<TextureKey> Resolver::address_sources(const Operand& operand, unsigned depth) {
    if (!step(depth)) return {};
    if (const auto* word = std::get_if<ConstantWord>(&operand)) {
        if (word->byte_offset % 4 == 0 && word->byte_offset < constant_buffer_window_bytes)
            return {{word->slot, word->byte_offset}};
        return {};
    }
    auto intersect = [](std::vector<TextureKey> left, const std::vector<TextureKey>& right) {
        std::erase_if(left, [&](const auto& key) {
            return std::find(right.begin(), right.end(), key) == right.end();
        });
        return left;
    };
    const auto* operation = definition(operand);
    if (!operation) {
        const auto* value = std::get_if<ValueId>(&operand);
        const auto* phi = value && value->value < phis.size() ? phis[value->value] : nullptr;
        if (!phi) return {};
        std::optional<std::vector<TextureKey>> common;
        for (const auto& input : phi->inputs) {
            // A cyclic input needs an induction proof and contributes nothing:
            // the lowered access keeps its runtime descriptor bounds check, so
            // a loop-carried address is covered by the same membership test as
            // any other unproven path.
            if (input.value == phi->output) continue;
            auto incoming = address_sources(input.value, depth + 1);
            if (error) return {};
            // A control-flow input without provenance (for example a value
            // previously loaded through another storage binding) contributes
            // nothing: the lowered
            // access keeps its runtime descriptor bounds check, so addresses
            // the proof cannot explain read zero or drop the store exactly as
            // required by the runtime membership check. Two
            // provable descriptors that disagree still refuse to merge.
            if (incoming.empty()) continue;
            common = common ? intersect(std::move(*common), incoming) : std::move(incoming);
        }
        return common.value_or(std::vector<TextureKey>{});
    }
    const auto& op = *operation;
    if (op.numeric.scalar != ScalarType::u32 ||
        (op.numeric.domain != Domain::bits && op.numeric.domain != Domain::unsigned_integer)) return {};
    if (op.opcode == Opcode::load) {
        const auto& m = std::get<MemoryInfo>(op.info);
        if (m.space != AddressSpace::constant_buffer || !m.constant_buffer_slot ||
            m.element_bytes != 4 || m.array_index || m.is_volatile) {
            // A descriptor word staged through local/shared memory keeps the
            // provenance its matching store recorded; an unmatched load
            // contributes nothing.
            const auto* value = std::get_if<ValueId>(&operand);
            if (value && m.element_bytes == 4 && !m.array_index && !m.is_volatile &&
                (m.space == AddressSpace::shared || m.space == AddressSpace::local)) {
                if (const auto* entry = memory_base_of(m, op, *value, depth + 1)) return {entry->key};
            }
            return {};
        }
        std::uint64_t offset = m.byte_offset;
        if (m.dynamic_byte_offset) {
            const auto dynamic = exact(op.inputs[m.dynamic_byte_offset->value], depth + 1);
            if (!dynamic) return {};
            offset += *dynamic;
        }
        const auto* value = std::get_if<ValueId>(&operand);
        for (std::size_t i = 0; i < op.outputs.size(); ++i) {
            if (value && std::get_if<ValueId>(&op.outputs[i]) && std::get<ValueId>(op.outputs[i]) == *value) {
                offset = (offset & ~std::uint64_t{3}) + i * 4;
                if (offset < constant_buffer_window_bytes)
                    return {{*m.constant_buffer_slot, static_cast<std::uint32_t>(offset)}};
            }
        }
        return {};
    }
    if (op.outputs.size() != 1) return {};
    if (op.opcode == Opcode::copy && op.inputs.size() == 1)
        return address_sources(op.inputs[0], depth + 1);
    if (op.opcode == Opcode::select && op.inputs.size() == 3)
        return intersect(address_sources(op.inputs[1], depth + 1), address_sources(op.inputs[2], depth + 1));
    if ((op.opcode != Opcode::add && op.opcode != Opcode::subtract) || op.inputs.size() != 2) return {};
    auto left = address_sources(op.inputs[0], depth + 1);
    auto right = address_sources(op.inputs[1], depth + 1);
    auto result = left;
    std::erase_if(result, [&](const auto& key) { return std::find(right.begin(), right.end(), key) != right.end(); });
    if (op.opcode == Opcode::add) for (const auto& key : right)
        if (std::find(left.begin(), left.end(), key) == left.end()) result.push_back(key);
    return result;
}

constexpr std::size_t max_storage_targets = 8;

// The storage-target candidate walk looks through literal displacements, and
// each phi source contributes its own
// descriptor origins (nested phis are walked once). The union keeps every
// origin a path can supply; emission tests them at run time.
std::vector<TextureKey> Resolver::address_union(const Operand& operand, std::set<std::uint32_t>& visited,
                                                unsigned depth) {
    if (!step(depth)) return {};
    // A path contributes only a unique origin: an address built from two
    // descriptor words names none.
    const auto unique = [&](const Operand& source) {
        auto keys = address_sources(source, depth + 1);
        if (keys.size() > 1) keys.clear();
        return keys;
    };
    const auto merge = [](std::vector<TextureKey> into, const std::vector<TextureKey>& from) {
        for (const auto& key : from)
            if (std::find(into.begin(), into.end(), key) == into.end()) into.push_back(key);
        return into;
    };
    if (const auto* op = definition(operand); op && op->opcode == Opcode::add && op->inputs.size() == 2) {
        for (unsigned side = 0; side != 2; ++side)
            if (const auto* bits = std::get_if<LiteralBits>(&op->inputs[side]); bits && bits->type == ValueType::word32)
                return address_union(op->inputs[1 - side], visited, depth + 1);
    }
    // A predicated guest move reaches SSA as a select: both arms are paths.
    if (const auto* op = definition(operand); op && op->opcode == Opcode::select && op->inputs.size() == 3) {
        auto left = unique(op->inputs[1]);
        if (left.empty() && !error) left = address_union(op->inputs[1], visited, depth + 1);
        auto right = unique(op->inputs[2]);
        if (right.empty() && !error) right = address_union(op->inputs[2], visited, depth + 1);
        return error ? std::vector<TextureKey>{} : merge(std::move(left), right);
    }
    const auto* value = std::get_if<ValueId>(&operand);
    const auto* phi = value && !definition(operand) && value->value < phis.size() ? phis[value->value] : nullptr;
    if (!phi) return unique(operand);
    std::vector<TextureKey> result;
    if (!visited.insert(value->value).second) return result;
    for (const auto& input : phi->inputs) {
        if (input.value == phi->output) continue;
        auto incoming = unique(input.value);
        if (error) return {};
        if (incoming.empty()) incoming = address_union(input.value, visited, depth + 1);
        if (error) return {};
        for (const auto& key : incoming)
            if (std::find(result.begin(), result.end(), key) == result.end()) result.push_back(key);
    }
    return result;
}

// A storage write uses host words only when the descriptor-relative low
// address is proven word aligned. Descriptor alignment alone says nothing
// about a guest offset added to it.
bool Resolver::storage_word_aligned(const Operand& operand, TextureKey key, unsigned depth) {    if (!step(depth)) return false;
    if (const auto* word = std::get_if<ConstantWord>(&operand))
        return word->slot == key.cbuf_slot && word->byte_offset == key.handle_byte_offset;
    const auto* op = definition(operand);
    if (!op) {
        const auto* value = std::get_if<ValueId>(&operand);
        const auto* phi = value && value->value < phis.size() ? phis[value->value] : nullptr;
        if (!phi || phi->inputs.empty()) return false;
        for (const auto& input : phi->inputs)
            if (input.value == phi->output || !storage_word_aligned(input.value, key, depth + 1)) return false;
        return true;
    }
    if (op->opcode == Opcode::load) {
        const auto sources = address_sources(operand, depth + 1);
        if (sources.size() == 1 && sources.front() == key) return true;
        // A word routed through local/shared memory is aligned when the
        // displacement recorded beside its descriptor base is aligned.
        const auto& m = std::get<MemoryInfo>(op->info);
        if ((m.space == AddressSpace::shared || m.space == AddressSpace::local) &&
            m.element_bytes == 4 && !m.array_index && !m.is_volatile) {
            const auto* entry = memory_base_of(m, *op, operand, depth + 1);
            if (entry && entry->key == key)
                return entry->const_offset % 4 == 0 && (!entry->offset ||
                    aligned(*entry->offset, 4));
        }
        return false;
    }
    if (op->opcode == Opcode::copy && op->inputs.size() == 1)
        return storage_word_aligned(op->inputs[0], key, depth + 1);
    if (op->opcode == Opcode::select && op->inputs.size() == 3)
        return storage_word_aligned(op->inputs[1], key, depth + 1) &&
            storage_word_aligned(op->inputs[2], key, depth + 1);
    if ((op->opcode == Opcode::add || op->opcode == Opcode::subtract) && op->inputs.size() == 2) {
        if (storage_word_aligned(op->inputs[0], key, depth + 1) &&
            displacement_aligned(op->inputs[1], depth + 1)) return true;
        if (op->opcode == Opcode::add && storage_word_aligned(op->inputs[1], key, depth + 1) &&
            displacement_aligned(op->inputs[0], depth + 1)) return true;
    }
    return false;
}

// Folds literal additions off a memory address so a store and the load that
// reads its word name the same location by recovering the byte-address base
// from the lowered word index.
std::optional<std::pair<Operand, std::uint32_t>> Resolver::strip_constants(const Operand& operand, unsigned depth) {
    if (!step(depth)) return {};
    const auto* op = definition(operand);
    if (!op) {
        if (const auto* literal = std::get_if<LiteralBits>(&operand);
            literal && literal->type == ValueType::word32)
            return std::make_pair(Operand{Undefined{ValueType::word32}}, static_cast<std::uint32_t>(literal->bits));
        return std::make_pair(operand, 0u);
    }
    if (op->opcode == Opcode::copy && op->inputs.size() == 1) return strip_constants(op->inputs[0], depth + 1);
    if (op->opcode == Opcode::add && op->inputs.size() == 2) {
        const auto* literal = std::get_if<LiteralBits>(&op->inputs[0]);
        const auto* other = std::get_if<LiteralBits>(&op->inputs[1]);
        if (literal && !other) {
            if (auto base = strip_constants(op->inputs[1], depth + 1))
                return std::make_pair(base->first, base->second + static_cast<std::uint32_t>(literal->bits));
            return {};
        }
        if (other && !literal) {
            if (auto base = strip_constants(op->inputs[0], depth + 1))
                return std::make_pair(base->first, base->second + static_cast<std::uint32_t>(other->bits));
            return {};
        }
    }
    return std::make_pair(operand, 0u);
}

// The (space, base value, folded byte constant) identity of one local/shared
// word location. A literal address folds to the no-base form.
std::optional<std::tuple<unsigned, std::uint32_t, std::uint32_t>> Resolver::memory_key(
    const MemoryInfo& m, const Operation& op, unsigned depth) {
    if (!step(depth)) return {};
    const auto space = m.space == AddressSpace::shared ? 0u : 1u;
    std::uint32_t const_offset = m.byte_offset;
    if (!m.dynamic_byte_offset) return std::make_tuple(space, UINT32_MAX, const_offset);
    if (m.dynamic_byte_offset->value >= op.inputs.size()) return {};
    auto stripped = strip_constants(op.inputs[m.dynamic_byte_offset->value], depth + 1);
    if (!stripped) return {};
    const auto* id = std::get_if<ValueId>(&stripped->first);
    if (!id) return std::make_tuple(space, UINT32_MAX, const_offset + stripped->second);
    return std::make_tuple(space, id->value, const_offset + stripped->second);
}

// A word operand that is exactly a constant-buffer descriptor word. Words are
// recorded at four-byte granularity so a stored high word keeps the low
// word's descriptor agreement check resolvable. The storage contract requires
// low/high descriptor-origin agreement.
bool Resolver::pure_key(const Operand& operand, TextureKey& key, unsigned depth) {
    if (!step(depth)) return false;
    if (const auto* word = std::get_if<ConstantWord>(&operand)) {
        if (word->byte_offset % 4 != 0 || word->byte_offset > constant_buffer_window_bytes - 4) return false;
        key = TextureKey{word->slot, word->byte_offset};
        return true;
    }
    const auto* op = definition(operand);
    if (!op || op->opcode != Opcode::load || op->outputs.size() != 1) return false;
    const auto* memory = std::get_if<MemoryInfo>(&op->info);
    if (!memory) return false;
    const auto& m = *memory;
    if (m.space != AddressSpace::constant_buffer || !m.constant_buffer_slot ||
        m.element_bytes != 4 || m.elements != 1 || m.array_index || m.is_volatile) return false;
    std::uint64_t offset = m.byte_offset;
    if (m.dynamic_byte_offset) {
        const auto dynamic = exact(op->inputs[m.dynamic_byte_offset->value], depth + 1);
        if (!error && dynamic) offset += *dynamic;
        else return false;
    }
    const auto* value = std::get_if<ValueId>(&operand);
    for (std::size_t i = 0; i < op->outputs.size(); ++i) {
        if (value && std::get_if<ValueId>(&op->outputs[i]) && std::get<ValueId>(op->outputs[i]) == *value) {
            offset = (offset & ~std::uint64_t{3}) + i * 4;
            if (offset % 4 == 0 && offset <= constant_buffer_window_bytes - 4) {
                key = TextureKey{*m.constant_buffer_slot, static_cast<std::uint32_t>(offset)};
                return true;
            }
            return false;
        }
    }
    return false;
}

// A descriptor-relative displacement is word aligned when its chain proves it,
// or when it is a load from a location whose recorded stored values are.
bool Resolver::displacement_aligned(const Operand& operand, unsigned depth) {
    if (!step(depth)) return false;
    if (aligned(operand, 4)) return true;
    const auto* op = definition(operand);
    if (!op) return false;
    if (op->opcode == Opcode::copy && op->inputs.size() == 1)
        return displacement_aligned(op->inputs[0], depth + 1);
    if (op->opcode == Opcode::select && op->inputs.size() == 3)
        return displacement_aligned(op->inputs[1], depth + 1) &&
            displacement_aligned(op->inputs[2], depth + 1);
    if ((op->opcode == Opcode::add || op->opcode == Opcode::subtract) && op->inputs.size() == 2)
        return displacement_aligned(op->inputs[0], depth + 1) &&
            displacement_aligned(op->inputs[1], depth + 1);
    if (op->opcode != Opcode::load) return false;
    const auto* m = std::get_if<MemoryInfo>(&op->info);
    if (!m || m->element_bytes != 4 || m->array_index || m->is_volatile ||
        (m->space != AddressSpace::shared && m->space != AddressSpace::local)) return false;
    const auto key = memory_key(*m, *op, depth + 1);
    if (!key || error) return false;
    const auto found = memory_alignment.find(*key);
    return found != memory_alignment.end() && found->second;
}

// A stored word is a descriptor base when it is a constant-buffer word, an
// addition of one, or a word previously
// recorded for its local/shared location. The non-key side of an addition
// stays an opaque displacement operand.
std::optional<Resolver::BaseAddress> Resolver::base_address(const Operand& operand, unsigned depth) {
    if (!step(depth)) return {};
    TextureKey key;
    if (pure_key(operand, key, depth + 1)) return BaseAddress{key, {}, 0};
    const auto* op = definition(operand);
    if (!op) return {};
    if (op->opcode == Opcode::copy && op->inputs.size() == 1) return base_address(op->inputs[0], depth + 1);
    if (std::holds_alternative<LiteralBits>(operand)) return {};
    if (op->opcode == Opcode::load && op->outputs.size() == 1 && std::holds_alternative<ValueId>(op->outputs[0]) &&
        std::get<ValueId>(op->outputs[0]) == std::get<ValueId>(operand)) {
        const auto& m = std::get<MemoryInfo>(op->info);
        if ((m.space == AddressSpace::shared || m.space == AddressSpace::local) &&
            m.element_bytes == 4 && !m.array_index && !m.is_volatile)
            if (const auto* entry = memory_base_of(m, *op, std::get<ValueId>(operand), depth + 1))
                return BaseAddress{entry->key, entry->offset, entry->const_offset};
        return {};
    }
    if (op->opcode != Opcode::add || op->inputs.size() != 2) return {};
    for (unsigned side = 0; side != 2; ++side) {
        const auto& added = op->inputs[side];
        const auto& other = op->inputs[1 - side];
        if (const auto* bits = std::get_if<LiteralBits>(&added); bits && bits->type == ValueType::word32) {
            if (auto base = base_address(other, depth + 1)) {
                base->const_offset += static_cast<std::uint32_t>(bits->bits);
                return base;
            }
            continue;
        }
        TextureKey added_key;
        if (pure_key(added, added_key, depth + 1)) {
            if (error) return {};
            if (pure_key(other, key, depth + 1)) return {};
            if (error) return {};
            return BaseAddress{added_key, other, 0};
        }
        if (error) return {};
    }
    return {};
}

// Looks up the descriptor word recorded for the location this load reads.
const Resolver::MemoryBase* Resolver::memory_base_of(const MemoryInfo& m, const Operation& op,
                                                     const Operand& result, unsigned depth) {
    if (!std::holds_alternative<ValueId>(result)) return nullptr;
    const auto key = memory_key(m, op, depth);
    if (!key || error) return nullptr;
    const auto found = memory_bases.find(*key);
    if (found == memory_bases.end() || !found->second) return nullptr;
    return &*found->second;
}

// One ordered pass records the descriptor origin of every full-word
// local/shared store.
void Resolver::record_memory_stores() {
    // Stored words share most of their arithmetic: each value's alignment is
    // proven once.
    AlignmentMemo stored_alignment;
    for (auto& block : p.blocks) {
        for (const auto id : block.operations) {
            if (!step()) return;
            const auto& op = p.operations[id.value];
            const auto* m = std::get_if<MemoryInfo>(&op.info);
            if (!m || op.opcode != Opcode::store ||
                (m->space != AddressSpace::shared && m->space != AddressSpace::local) ||
                m->array_index || m->atomic || m->is_volatile ||
                m->elements != 1 || m->element_bytes != 4 || m->data.size() != 1 ||
                m->data[0].value >= op.inputs.size()) continue;
            const auto key = memory_key(*m, op);
            if (!key || error) continue;
            {
                const auto value_aligned = aligned(op.inputs[m->data[0].value], 4, &stored_alignment);
                auto [slot, inserted] = memory_alignment.try_emplace(*key, value_aligned);
                if (!inserted) slot->second = slot->second && value_aligned;
            }
            auto value = base_address(op.inputs[m->data[0].value]);
            if (error) return;
            if (!value) continue;
            auto& entry = memory_bases[*key];
            if (entry && entry->key != value->key)
                entry.reset(); // Conflicting writers void the location.
            else if (!entry)
                entry = MemoryBase{value->key, value->offset, value->const_offset};
        }
    }
}

bool Resolver::storage(Operation& op) {
    auto& m = std::get<MemoryInfo>(op.info);
    if (m.resource || m.storage_address) return fail("SSA storage contains preallocated resources", DiagnosticCode::invalid_ir);
    if (!m.dynamic_byte_offset || m.array_index || m.atomic ||
        (op.opcode != Opcode::load && op.opcode != Opcode::store))
        return fail("Global memory access has no resolvable descriptor address");
    auto candidates = address_sources(op.inputs[m.dynamic_byte_offset->value]);
    std::erase_if(candidates, [](const auto& key) {
        return key.handle_byte_offset % 8 != 0 || key.handle_byte_offset > constant_buffer_window_bytes - 12;
    });
    if (m.global_address_high) {
        const auto high = address_sources(op.inputs[m.global_address_high->value]);
        std::erase_if(candidates, [&](const auto& key) {
            return std::find(high.begin(), high.end(), TextureKey{key.cbuf_slot, key.handle_byte_offset + 4}) == high.end();
        });
    }
    if (error) return false;
    if (candidates.size() == 1) {
        const auto key = candidates.front();
        const bool aligned = storage_words(op, m, key);
        if (error) return false;
        if (op.opcode == Opcode::store && !aligned)
            return fail("Global storage write has no word-aligned descriptor-relative address proof",
                DiagnosticCode::unsupported_capability);
        ResourceId resource;
        StorageAddress address;
        if (!bind_storage(op, key, resource, address)) return false;
        auto& info = std::get<MemoryInfo>(op.info);
        info.resource = resource;
        info.storage_address = address;
        // Loads with the same proof read whole words, as stores write them.
        info.storage_word_aligned = aligned;
        return true;
    }
    // The address may come from several descriptors, so every origin a path
    // supplies becomes a candidate in packed (slot | offset << 16) order.
    // Candidate stores index host words; there is no per-path
    // alignment proof to take.
    std::set<std::uint32_t> visited;
    auto targets = address_union(op.inputs[m.dynamic_byte_offset->value], visited);
    std::erase_if(targets, [](const auto& key) {
        return key.handle_byte_offset % 8 != 0 || key.handle_byte_offset > constant_buffer_window_bytes - 12;
    });
    if (m.global_address_high) {
        visited.clear();
        const auto high = address_union(op.inputs[m.global_address_high->value], visited);
        std::erase_if(targets, [&](const auto& key) {
            return std::find(high.begin(), high.end(), TextureKey{key.cbuf_slot, key.handle_byte_offset + 4}) == high.end();
        });
    }
    if (error) return false;
    if (targets.empty())
        return fail("Global address low/high words have no constant-buffer storage descriptor provenance");
    // Each candidate adds three descriptor words to the access (opcode arity).
    if (targets.size() > max_storage_targets)
        return fail("Global address has more storage descriptor candidates than the access can carry",
            DiagnosticCode::limit_exceeded);
    if (op.opcode == Opcode::store && (m.element_bytes != 4 || m.byte_offset % 4 != 0))
        return fail("Multi-target global storage write is not a word store", DiagnosticCode::unsupported_capability);
    std::sort(targets.begin(), targets.end(), [](const auto& a, const auto& b) {
        return std::pair{a.handle_byte_offset, a.cbuf_slot} < std::pair{b.handle_byte_offset, b.cbuf_slot};
    });
    bool aligned = op.opcode == Opcode::load;
    for (const auto& target : targets) aligned = aligned && storage_words(op, m, target);
    if (error) return false;
    for (std::size_t i = 0; i != targets.size(); ++i) {
        ResourceId resource;
        StorageAddress address;
        if (!bind_storage(op, targets[i], resource, address)) return false;
        auto& info = std::get<MemoryInfo>(op.info);
        if (i == 0) { info.resource = resource; info.storage_address = address; }
        else info.storage_alternatives.push_back({resource, address});
        info.storage_word_aligned = aligned;
    }
    return true;
}

// Whether each 4-byte element of the access is one host word: its
// descriptor-relative address is proven word aligned for `key`. A load whose
// proof runs out of budget only loses the word fast path; it stays legal.
bool Resolver::storage_words(const Operation& op, const MemoryInfo& m, TextureKey key) {
    if (m.element_bytes != 4 || m.byte_offset % 4 != 0) return false;
    const auto spent = steps;
    const bool aligned = storage_word_aligned(op.inputs[m.dynamic_byte_offset->value], key);
    if (error && op.opcode != Opcode::store) {
        error.reset();
        steps = spent;
        return false;
    }
    return aligned;
}

// One storage binding per descriptor key, with the descriptor's base and size
// words appended as inputs of the access.
bool Resolver::bind_storage(Operation& op, TextureKey key, ResourceId& resource, StorageAddress& address) {
    const auto access = std::get<MemoryInfo>(op.info).access;
    std::optional<ResourceId> found;
    for (std::uint32_t i = 0; i < r.uses.size(); ++i) {
        if (!step()) return false;
        auto* use = std::get_if<BufferUse>(&r.uses[i]);
        const auto* source = use ? std::get_if<StorageBufferSource>(&use->source) : nullptr;
        if (source && source->descriptor == key) {
            use->access = combine(use->access, access);
            found = ResourceId{i};
            break;
        }
    }
    if (!found) found = append(BufferUse{StorageBufferSource{key}, access,
        {SizeKind::storage_descriptor, 4, 4}});
    if (!found) return false;
    auto word = [&](unsigned displacement) {
        const InputIndex index{static_cast<std::uint16_t>(op.inputs.size())};
        op.inputs.push_back(ConstantWord{key.cbuf_slot, key.handle_byte_offset + displacement});
        return index;
    };
    resource = *found;
    address = StorageAddress{key, word(0), word(4), word(8)};
    return true;
}

bool Resolver::memory(Operation& op) {
    auto& m = std::get<MemoryInfo>(op.info);
    if (op.opcode == Opcode::barrier || op.opcode == Opcode::memory_barrier) return true;
    if (m.space == AddressSpace::guest) {
        // Every guest access of a stage goes through one window table, which
        // also records the accesses that miss every window.
        if (!guest_windows) guest_windows = append(BufferUse{SupportBufferSource{std::string(guest_memory_schema),
            guest_memory_version}, Access::read_write, {SizeKind::accessed_extent, guest_memory_bytes, guest_memory_alignment}});
        m.resource = guest_windows;
        return m.resource.has_value();
    }
    if (m.space == AddressSpace::storage)
        return (m.resource && m.storage_address) || fail("Storage address has no resolved descriptor");
    if (m.array_index) return fail("Memory resource arrays require a source/stride contract");
    std::optional<std::uint32_t> maximum;
    if (m.dynamic_byte_offset) {
        maximum = upper(op.inputs[m.dynamic_byte_offset->value]);
        if (error) return false;
        if (maximum) m.dynamic_byte_offset_upper_bound = *maximum;
        else m.dynamic_byte_offset_upper_bound.reset();
    }
    if (m.space == AddressSpace::constant_buffer) {
        // Each component is backed by complete 32-bit words. Guest halfword
        // and word alignment is applied before selecting those host words;
        // B64 reads two independently aligned words at address and address+4.
        std::uint64_t extent = constant_buffer_window_bytes;
        if (!m.dynamic_byte_offset || maximum) {
            const std::uint64_t highest_base = std::uint64_t(m.byte_offset) + maximum.value_or(0);
            const std::uint64_t last_component = highest_base + std::uint64_t(m.element_bytes) * (m.elements - 1);
            const std::uint64_t last_word_address = m.element_bytes == 8 ?
                ((last_component + 4) & ~std::uint64_t{3}) : (last_component & ~std::uint64_t{3});
            extent = std::min<std::uint64_t>(last_word_address + 4, constant_buffer_window_bytes);
        }
        m.resource = buffer(*m.constant_buffer_slot, extent);
        return m.resource.has_value();
    }
    std::uint64_t extent = std::uint64_t(m.byte_offset) + std::uint64_t(m.element_bytes) * m.elements;
    if (m.dynamic_byte_offset && !maximum) {
        // armigo-local-guard-v1 and armigo-shared-guard-v1: local memory, and a
        // compute stage's shared memory, accept an unproven register offset; the
        // access stays runtime-checked and emission guards it, so no static
        // extent is accumulated here. Storage keeps strict proofs. The guard
        // protects the unbounded guest address at runtime.
        const bool guarded = m.space == AddressSpace::local ||
            (m.space == AddressSpace::shared && p.stage == Stage::compute && r.compute);
        if (!guarded) return fail("Dynamic memory offset has no proven bound");
    } else if (m.dynamic_byte_offset) {
        extent += *maximum;
    }
    if (m.space == AddressSpace::local) {
        auto size = require(FactQuery<FactKind::local_memory_bytes>{});
        if (!size) return false;
        if (m.dynamic_byte_offset && !maximum)
            return *size > 0 ||
                fail("Runtime-checked local access declares an empty allocation", DiagnosticCode::invalid_fact);
        return extent <= *size || fail("Local memory access exceeds proven size");
    }
    if (m.space == AddressSpace::shared && p.stage == Stage::compute && r.compute) {
        // A guarded access with an unproven register offset cannot prove word
        // alignment either; emission lowers it byte-wise beside the guard, so
        // only literal or proven offsets keep the natural-alignment proof.
        const bool guarded = m.dynamic_byte_offset && !maximum;
        if (!guarded && (m.byte_offset % m.element_bytes || (m.dynamic_byte_offset &&
            !aligned(op.inputs[m.dynamic_byte_offset->value], m.element_bytes))))
            return fail("Shared memory address has no natural-alignment proof",
                steps > p.limits.analysis_steps ? DiagnosticCode::limit_exceeded : DiagnosticCode::unsupported_capability);
        return extent <= r.compute->shared_memory_bytes || fail("Shared memory access exceeds launch allocation");
    }
    return fail("Shared memory requires a compute launch allocation", DiagnosticCode::unsupported_capability);
}

bool Resolver::io(Operation& op) {
    auto& info = std::get<IoInfo>(op.info);
    if (info.dynamic_byte_offset || info.vertex_index) return fail("Indirect IO or explicit vertex fetch requires a layout contract");
    if (info.semantic == IoSemantic::thread_id || info.semantic == IoSemantic::workgroup_id ||
        info.semantic == IoSemantic::subgroup_lane_id || info.semantic == IoSemantic::quad_lane_id) {
        const bool scalar_lane = info.semantic == IoSemantic::subgroup_lane_id ||
                                 info.semantic == IoSemantic::quad_lane_id;
        const auto components = static_cast<std::uint8_t>(scalar_lane ? 1 : 3);
        // The SIMD-group lane index also exists in fragment functions, where
        // warp shuffles read it; the quad position exists only in fragment
        // functions, where quad swizzle arithmetic reads it. Workgroup
        // builtins stay compute-only.
        const bool stage_ok = info.semantic == IoSemantic::quad_lane_id ? p.stage == Stage::fragment :
            info.semantic == IoSemantic::subgroup_lane_id ? p.stage != Stage::vertex :
            p.stage == Stage::compute;
        if (!stage_ok || op.opcode != Opcode::load_io || info.output_storage ||
            info.type != IoType{ScalarType::u32, components} || info.location || info.component_mask >= (1u << components))
            return fail("Invalid compute builtin IO", DiagnosticCode::invalid_ir);
    }
    IoUse use{info.semantic, info.location, info.component_mask, info.type, info.interpolation,
              (op.opcode == Opcode::store_io || info.output_storage) ? Access::write : Access::read, {}, {}, 0, 0, {}};
    if (p.stage == Stage::vertex && info.semantic == IoSemantic::position && use.access == Access::write) {
        // The host clip-depth adapter depends on state consumed with position IO.
        const auto depth = require(FactQuery<FactKind::depth_convention>{});
        if (!depth) return false;
        if (*depth > DepthConvention::negative_one_to_one)
            return fail("Vertex clip-depth convention is invalid", DiagnosticCode::invalid_fact);
    }
    if (p.stage == Stage::vertex && info.semantic == IoSemantic::generic && use.access == Access::read) {
        auto knowledge = read(FactQuery<FactKind::vertex_attribute_format>{info.location});
        if (!knowledge) return false;
        if (const auto* format = std::get_if<Known<AttributeFormat>>(&*knowledge)) {
            if (format->value.format > VertexFormat::f32 || format->value.components == 0 || format->value.components > 4)
                return fail("Vertex format fact is invalid", DiagnosticCode::invalid_fact);
            const auto vertex_scalar = format->value.format == VertexFormat::u8 || format->value.format == VertexFormat::u16 || format->value.format == VertexFormat::u32 ? ScalarType::u32 :
                format->value.format == VertexFormat::s8 || format->value.format == VertexFormat::s16 || format->value.format == VertexFormat::s32 ? ScalarType::s32 : ScalarType::f32;
            if (info.type.scalar != vertex_scalar) return fail("Vertex format/result conversion must be explicit", DiagnosticCode::unsupported_modifier);
            use.vertex_format = format->value;
        } else {
            // No pipeline-state fact: the declared class is the code-derived
            // property the post-SSA use analysis recorded on this load.
            if (op.opcode == Opcode::interpolate || info.dynamic_byte_offset)
                return fail("Dynamic or interpolated vertex attribute access requires a vertex format fact",
                    DiagnosticCode::unsupported_capability);
            const auto scalar = info.type.scalar;
            if (scalar != ScalarType::f32 && scalar != ScalarType::u32 && scalar != ScalarType::s32)
                return fail("Vertex generic input has neither a class analysis result nor a format fact");
            use.vertex_class = scalar == ScalarType::f32 ? VertexAttributeClass::float_class :
                scalar == ScalarType::s32 ? VertexAttributeClass::sint_class : VertexAttributeClass::uint_class;
        }
    }
    if (p.stage == Stage::fragment && info.semantic == IoSemantic::color && use.access == Access::write) {
        auto type = require(FactQuery<FactKind::fragment_output_type>{info.location});
        if (!type) return false;
        if (*type != info.type) return fail("Fragment output type disagrees with its fact", DiagnosticCode::invalid_fact);
    }
    if (p.stage == Stage::fragment && info.semantic == IoSemantic::generic && use.access == Access::read) {
        for (std::uint8_t c = 0; c < 4; ++c) if (info.component_mask & (1U << c)) {
            auto interpolation = require(fact_query::interpolation(info.location, c));
            if (!interpolation) return false;
            if (info.interpolation_location_override) {
                if (*info.interpolation_location_override != InterpolationLocation::center &&
                    *info.interpolation_location_override != InterpolationLocation::centroid)
                    return fail("Instruction interpolation location is unsupported", DiagnosticCode::unsupported_modifier);
                interpolation->location = *info.interpolation_location_override;
            }
            if (*interpolation != info.interpolation) return fail("Interpolation metadata disagrees with its fact", DiagnosticCode::invalid_fact);
        }
    }
    for (auto& prior : r.io) {
        if (!step()) return false;
        if (prior.semantic == use.semantic && prior.location == use.location && prior.access == use.access) {
            if (prior.type == use.type && prior.interpolation == use.interpolation &&
                prior.vertex_format == use.vertex_format && prior.vertex_class == use.vertex_class) {
                prior.mask |= use.mask; return true;
            }
            if (prior.mask & use.mask) return fail("Overlapping IO components have incompatible types or interpolation", DiagnosticCode::invalid_ir);
        }
    }
    if (r.io.size() >= p.limits.resources) return fail("IO reflection count exceeds limit", DiagnosticCode::limit_exceeded);
    r.io.push_back(use);
    return true;
}

bool Resolver::lower_word(Operand& operand, std::vector<OperationId>& before) {
    const auto* word = std::get_if<ConstantWord>(&operand);
    if (!word) return true;
    const auto binding = buffer(word->slot,
        std::min<std::uint64_t>((std::uint64_t(word->byte_offset) & ~std::uint64_t{3}) + 4,
            constant_buffer_window_bytes));
    if (!binding) return false;
    if (p.values.size() >= p.limits.values || p.operations.size() >= p.limits.operations)
        return fail("Constant-buffer load expansion exceeds IR limit", DiagnosticCode::limit_exceeded);
    MemoryInfo m;
    m.constant_buffer_slot = word->slot; m.byte_offset = word->byte_offset; m.alignment = 4; m.resource = *binding;
    const auto value = ValueId{static_cast<std::uint32_t>(p.values.size())};
    const auto operation = OperationId{static_cast<std::uint32_t>(p.operations.size())};
    p.values.push_back({ValueType::word32});
    p.operations.push_back({source, Opcode::load, Effects::read_memory, {}, {}, {value}, m});
    operand = value;
    before.push_back(operation);
    return true;
}

Status Resolver::run() {
    if (p.stage == Stage::compute) {
        auto local = require(FactQuery<FactKind::compute_local_size>{});
        if (!local) return *error;
        auto shared = require(FactQuery<FactKind::compute_shared_memory_bytes>{});
        if (!shared) return *error;
        r.compute = ComputeConfiguration{*local, *shared};
        if (!valid_compute_configuration(*r.compute))
            return diagnose(Phase::legalize, DiagnosticCode::unsupported_capability, {},
                "Compute launch exceeds supported dimensions or 32 KiB shared-memory allocation");
    }
    // Record local/shared descriptor staging before resolving provenance, in
    // program order, so a store and its matching load agree on one location.
    record_memory_stores();
    if (error) return *error;
    // Resolve provenance before inserting loads or replacing ConstantWord operands.
    for (auto& op : p.operations) {
        if (!step()) return *error;
        source = op.source;
        if (auto* t = std::get_if<TextureInfo>(&op.info)) {
            if (t->resource || t->sampler_resource) return diagnose(Phase::legalize, DiagnosticCode::invalid_ir, source, "SSA texture contains preallocated resources");
            if (!texture(op)) return *error;
        }
        if (const auto* m = std::get_if<MemoryInfo>(&op.info); m && m->space == AddressSpace::storage)
            if (!storage(op)) return *error;
    }
    // Definitions point into stable operation IDs. Only this second walk grows arenas.
    for (auto& block : p.blocks) {
        std::vector<OperationId> order;
        for (auto id : block.operations) {
            if (!step()) return *error;
            Operation op = p.operations[id.value];
            source = op.source;
            if (std::holds_alternative<MemoryInfo>(op.info) && !memory(op)) return *error;
            if (std::holds_alternative<IoInfo>(op.info) && !io(op)) return *error;
            for (auto& input : op.inputs) if (!lower_word(input, order)) return *error;
            p.operations[id.value] = std::move(op);
            order.push_back(id);
        }
        source = block.source;
        bool ok = std::visit([&](auto& term) {
            using T = std::decay_t<decltype(term)>;
            if constexpr (std::is_same_v<T, Branch>) return lower_word(term.condition, order);
            else if constexpr (std::is_same_v<T, Switch>) return lower_word(term.selector, order);
            else if constexpr (std::is_same_v<T, Return>) {
                for (auto& value : term.values) if (!lower_word(value, order)) return false;
            }
            return true;
        }, block.terminator);
        if (!ok) return *error;
        block.operations = std::move(order);
    }
    if (std::any_of(buffers.begin(), buffers.end(), [](const auto& buffer) { return buffer.has_value(); })) {
        if (!append(BufferUse{SupportBufferSource{std::string(constant_buffer_support_schema),
                constant_buffer_support_version}, Access::read,
                {SizeKind::accessed_extent, constant_buffer_support_bytes, 4}})) return *error;
    }
    if (!merge_dependencies_now()) return *error;
    return std::monostate{};
}
}

Result<LegalProgram> legalize(SsaProgram input, FactReader& facts, const MetalProfile& profile) {
    if (auto status = resources::check_profile(profile, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    if (auto status = validate(input.ir, IrForm::ssa, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    auto dependencies = merge_dependencies(input.ir.dependencies, facts.dependencies(), input.ir.limits, Phase::legalize);
    if (auto* error = std::get_if<Diagnostic>(&dependencies)) return *error;
    input.ir.dependencies = std::get<std::vector<Dependency>>(std::move(dependencies));
    if (auto status = passes::cleanup(input.ir); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    if (auto status = passes::legalize_nontexture(input.ir, facts, profile); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    if (auto status = validate(input.ir, IrForm::ssa, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    for (const auto& op : input.ir.operations) {
        auto supported = [&](ScalarType type) { return (type != ScalarType::f64 || profile.supports_fp64) &&
            (type != ScalarType::f16 || profile.supports_fp16); };
        const auto* conversion = std::get_if<ConversionInfo>(&op.info);
        if (!supported(op.numeric.scalar) || (conversion && (!supported(conversion->source) || !supported(conversion->destination))))
            return diagnose(Phase::legalize, DiagnosticCode::unsupported_capability, op.source, "Floating-point width is unsupported by the Metal profile");
    }
    // Derive the class of every fact-less generic vertex input from the IR
    // before resource resolution consumes the loads.
    if (auto status = classify_vertex_inputs(input.ir); std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    LegalProgram output{std::move(input.ir), {}, profile};
    Resolver resolver(output.ir, facts, output.requirements, profile);
    if (auto status = resolver.run(); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    if (auto status = validate(output.ir, IrForm::legal, Phase::legalize); std::holds_alternative<Diagnostic>(status)) return std::get<Diagnostic>(status);
    if (auto status = passes::hoist_guarded_samples(output.ir, profile.hoist_implicit_samples);
        std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    if (auto status = resources::check_requirements(output.ir, output.requirements, profile, Phase::legalize); std::holds_alternative<Diagnostic>(status))
        return std::get<Diagnostic>(status);
    return output;
}
} // namespace armigo::shader::pipeline
