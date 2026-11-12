// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "ir.h"
#include "facts.h"
#include "handle_program.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <set>
#include <utility>

namespace armigo::shader::pipeline {
Effects required_effects(const Operation& op) {
    const auto* sig = signature(op.opcode);
    Effects effects = sig ? sig->required_effects : Effects::none;
    if (op.opcode == Opcode::atomic || op.opcode == Opcode::image_atomic)
        effects = effects | Effects::atomic | Effects::read_memory | Effects::write_memory;
    if (const auto* texture = std::get_if<TextureInfo>(&op.info)) {
        effects = effects | Effects::texture;
        // Implicit-LOD and bias samples and LOD queries take their level of
        // detail from the quad's derivatives. A gather takes none: it reads the
        // 2x2 footprint of its coordinate in the base level. The guest's gather
        // instructions encode no level of detail, a SPIR-V or Metal gather has
        // no level or derivative operand, and SPIR-V admits gathers in every
        // stage while it restricts implicit-LOD sampling to fragments.
        if (texture->mode == TextureMode::sample_implicit || texture->mode == TextureMode::sample_bias ||
            texture->mode == TextureMode::query_lod) effects = effects | Effects::derivative;
    }
    return effects;
}
std::vector<BlockId> successors(const Terminator& terminator) {
    std::vector<BlockId> result;
    if (const auto* jump = std::get_if<Jump>(&terminator)) result.push_back(jump->target);
    if (const auto* branch = std::get_if<Branch>(&terminator)) {
        result.push_back(branch->when_true);
        result.push_back(branch->when_false);
    }
    if (const auto* select = std::get_if<Switch>(&terminator)) {
        for (const auto& item : select->cases) result.push_back(item.target);
        result.push_back(select->fallback);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
namespace {
struct Definition { BlockId block; std::size_t order{}; bool parameter{}; };
struct Use { ValueId value; BlockId block; std::size_t order{}; SourceLocation source; };

class Validator {
public:
    Validator(const Program& program, IrForm form, Phase phase)
        : p(program), form(form), phase(phase), block_owners(p.blocks.size()),
          operation_owners(p.operations.size()), definitions(p.values.size()),
          incoming(p.blocks.size()), outgoing(p.blocks.size()) {}
    Status run();
private:
    const Program& p;
    IrForm form;
    Phase phase;
    std::vector<unsigned> block_owners, operation_owners;
    std::vector<std::optional<Definition>> definitions;
    std::vector<std::vector<BlockId>> incoming, outgoing;
    std::vector<Use> uses;
    std::optional<Diagnostic> error;
    std::uint64_t steps{};
    bool fail(std::string message, SourceLocation source = {}, DiagnosticCode code = DiagnosticCode::invalid_ir) {
        if (!error) error = diagnose(phase, code, source, std::move(message));
        return false;
    }
    bool step() {
        if (++steps > p.limits.analysis_steps) return fail("IR validation work limit exceeded", {}, DiagnosticCode::limit_exceeded);
        return true;
    }
    bool valid_register(Register reg, bool output) {
        switch (reg.file) {
        case RegisterFile::gpr: return reg.index < (output ? 255 : 256);
        case RegisterFile::predicate: return reg.index < (output ? 7 : 8);
        case RegisterFile::condition_code: return reg.index < 4;
        }
        return false;
    }
    std::optional<ValueType> type(const Operand& operand) {
        if (const auto* id = std::get_if<ValueId>(&operand)) {
            if (id->value < p.values.size()) return p.values[id->value].type;
        } else if (const auto* reg = std::get_if<Register>(&operand)) {
            return reg->file == RegisterFile::gpr ? ValueType::word32 : ValueType::predicate;
        } else if (const auto* literal = std::get_if<LiteralBits>(&operand)) return literal->type;
        else if (std::holds_alternative<ConstantWord>(operand)) return ValueType::word32;
        return std::nullopt;
    }
    bool operand(const Operand&, BlockId, std::size_t order, SourceLocation, bool output = false);
    bool define(ValueId, Definition, SourceLocation);
    bool operation(const Operation&, BlockId, std::size_t);
    bool metadata(const Operation&);
    bool operand_types(const Operation&);
    bool numeric_mode(const Operation&);
    bool dominance(const Function&);
};

bool Validator::define(ValueId id, Definition definition, SourceLocation source) {
    if (id.value >= p.values.size()) return fail("Value ID is outside its arena", source);
    if (form == IrForm::registers && definitions[id.value] &&
        p.blocks[definitions[id.value]->block.value].function != p.blocks[definition.block.value].function)
        return fail("Compiler local is assigned by different functions", source);
    if (form != IrForm::registers && definitions[id.value]) return fail("SSA value has multiple definitions", source);
    definitions[id.value] = definition;
    return true;
}
bool Validator::operand(const Operand& operand_value, BlockId block, std::size_t order,
                        SourceLocation source, bool output) {
    if (!step()) return false;
    if (const auto* id = std::get_if<ValueId>(&operand_value)) {
        if (id->value >= p.values.size()) return fail("Value ID is outside its arena", source);
        if (output) return define(*id, {block, order, false}, source);
        uses.push_back({*id, block, order, source});
        return true;
    }
    if (const auto* reg = std::get_if<Register>(&operand_value)) {
        return (form == IrForm::registers && valid_register(*reg, output)) ||
               fail("Register is invalid or remains after SSA", source);
    }
    if (output) return fail("Operation destination is not writable", source);
    if (const auto* literal = std::get_if<LiteralBits>(&operand_value)) {
        if (literal->type == ValueType::word32 && literal->bits > UINT32_MAX) return fail("Word literal exceeds 32 bits", source);
        if (literal->type == ValueType::predicate && literal->bits > 1) return fail("Predicate literal is not Boolean", source);
        return true;
    }
    if (const auto* constant = std::get_if<ConstantWord>(&operand_value)) {
        if (constant->slot >= 18 || constant->byte_offset % 4 != 0) return fail("Invalid constant-buffer word address", source);
        if (form == IrForm::legal) return fail("Constant-buffer operand must become an explicit legal load", source);
        return true;
    }
    return fail("Undefined operand is read", source, DiagnosticCode::undefined_read);
}

bool Validator::metadata(const Operation& op) {
    std::vector<bool> referenced(op.inputs.size());
    auto index = [&](std::optional<InputIndex> i) {
        if (!i) return true;
        if (i->value >= op.inputs.size()) return fail("Metadata input index is outside operation inputs", op.source);
        referenced[i->value] = true;
        return true;
    };
    auto indices = [&](const std::vector<InputIndex>& values) {
        return std::all_of(values.begin(), values.end(), [&](auto value) { return index(value); });
    };
    if (const auto* m = std::get_if<MemoryInfo>(&op.info)) {
        if (!index(m->dynamic_byte_offset) || !index(m->array_index) ||
            !index(m->global_address_high) || !indices(m->data)) return false;
        if (m->storage_address && (!index(m->storage_address->base_low) ||
            !index(m->storage_address->base_high) || !index(m->storage_address->byte_size))) return false;
        for (const auto& target : m->storage_alternatives)
            if (!index(target.address.base_low) || !index(target.address.base_high) ||
                !index(target.address.byte_size)) return false;
        if (m->native_load_policy && (*m->native_load_policy != NativeLoadPolicy::aligned_v1 ||
            (m->space != AddressSpace::guest && m->space != AddressSpace::storage) ||
            op.opcode != Opcode::load || m->access != Access::read || m->atomic || m->is_volatile ||
            m->order != MemoryOrder::relaxed || (m->scope != MemoryScope::workgroup && m->scope != MemoryScope::device) ||
            (m->element_bytes != 1 && m->element_bytes != 2 && m->element_bytes != 4) ||
            !std::has_single_bit(unsigned(m->elements)) || m->elements > 4 ||
            (m->element_bytes < 4 && m->elements != 1) || m->alignment != m->element_bytes * m->elements || m->array_index))
            return fail("Native LDG policy requires an aligned non-RMW global load", op.source);
        if (m->native_shared_atomic_policy && !valid_native_shared_atomic(op, *m, p.stage))
            return fail("Native shared atomic policy requires a relaxed workgroup U32 addition", op.source);
        if (m->stage_buffer && (m->space != AddressSpace::shared ||
            !m->stage_buffer->valid_for(m->stage_buffer->allocation_bytes) ||
            (op.opcode != Opcode::load && op.opcode != Opcode::store) ||
            (m->element_bytes != 1 && m->element_bytes != 4) || m->elements != 1 ||
            m->byte_offset || !m->dynamic_byte_offset || m->atomic || m->is_volatile ||
            m->resource || m->constant_buffer_slot || m->array_index || m->global_address_high ||
            m->storage_address || !m->storage_alternatives.empty()))
            return fail("Stage-buffer policy requires a header-sized shared load or store", op.source);
        if (m->direct_storage && (m->space != AddressSpace::storage ||
            op.opcode != Opcode::load || m->atomic || m->is_volatile || m->element_bytes != 4 ||
            !std::has_single_bit(unsigned(m->elements)) || !m->dynamic_byte_offset ||
            m->global_address_high || m->storage_address || !m->storage_alternatives.empty() || m->array_index ||
            !valid_fact_key(FactQuery<FactKind::bound_storage_descriptor>{m->direct_storage->cbuf_slot,
                m->direct_storage->handle_byte_offset}.key())))
            return fail("Direct storage requires an ordinary descriptor-relative word load", op.source);
        if (!m->storage_alternatives.empty() && !m->storage_address)
            return fail("Storage alternatives require a primary storage address", op.source);
        if ((m->storage_address && m->space != AddressSpace::storage) ||
            (m->global_address_high && m->space != AddressSpace::storage && m->space != AddressSpace::guest) ||
            ((m->global_address_high || m->storage_address) && !m->dynamic_byte_offset))
            return fail("Global address metadata requires a storage address", op.source);
        // A guest access names the full 64-bit guest address: the low word
        // in the dynamic offset role and the high word, plus a displacement.
        if (m->space == AddressSpace::guest && (!m->dynamic_byte_offset || !m->global_address_high ||
            m->storage_address || !m->storage_alternatives.empty() || m->array_index))
            return fail("Guest memory access requires exactly a 64-bit guest address", op.source);
        if (m->storage_address && form == IrForm::ssa)
            return fail("Runtime storage descriptor appears before legalization", op.source);
        for (auto role : {m->dynamic_byte_offset, m->array_index, m->global_address_high})
            if (role && type(op.inputs[role->value]) != ValueType::word32)
                return fail("Memory address must use word32 storage", op.source);
        std::vector<const StorageAddress*> addresses;
        if (m->storage_address) addresses.push_back(&*m->storage_address);
        for (const auto& target : m->storage_alternatives) addresses.push_back(&target.address);
        for (const auto* address : addresses) {
            if (address->descriptor.cbuf_slot >= 18 || address->descriptor.handle_byte_offset % 8 != 0 ||
                address->descriptor.handle_byte_offset > 65536 - 12)
                return fail("Storage descriptor key is invalid", op.source);
            for (auto role : {address->base_low, address->base_high, address->byte_size})
                if (type(op.inputs[role.value]) != ValueType::word32)
                    return fail("Storage descriptor words must use word32 storage", op.source);
        }
        if (!std::has_single_bit(m->element_bytes) || m->element_bytes > 8 || m->elements == 0 || m->elements > 4 ||
            !std::has_single_bit(m->alignment)) return fail("Invalid memory width or alignment", op.source);
        if (m->space == AddressSpace::constant_buffer && (!m->constant_buffer_slot || *m->constant_buffer_slot >= 18 || m->access != Access::read))
            return fail("Invalid constant-buffer access", op.source);
        if (m->space != AddressSpace::constant_buffer && m->constant_buffer_slot)
            return fail("Nonconstant memory operation has a constant-buffer slot", op.source);
        if (m->space == AddressSpace::constant_buffer && m->resource && form != IrForm::legal)
            return fail("Register/SSA constant-buffer access must use its guest slot", op.source);
        if (form == IrForm::legal && (m->space == AddressSpace::storage || m->space == AddressSpace::constant_buffer ||
            m->space == AddressSpace::guest) &&
            op.opcode != Opcode::barrier && op.opcode != Opcode::memory_barrier && !m->resource)
            return fail("Legal buffer access has no resolved resource", op.source);
        if (op.opcode == Opcode::load && (m->access != Access::read || op.outputs.size() != m->elements || !m->data.empty()))
            return fail("Load signature disagrees with memory metadata", op.source);
        if (op.opcode == Opcode::store && (m->access == Access::read || m->data.size() != m->elements))
            return fail("Store signature disagrees with memory metadata", op.source);
        if (op.opcode == Opcode::atomic && (!m->atomic || m->access != Access::read_write ||
            !includes(op.effects, Effects::read_memory | Effects::write_memory)))
            return fail("Atomic access lacks read/write semantics", op.source);
        if ((m->space == AddressSpace::local || m->space == AddressSpace::shared) && m->resource)
            return fail("Invocation/workgroup memory cannot use a buffer resource", op.source);
        const auto memory_type = m->element_bytes == 8 ? ValueType::word64 : ValueType::word32;
        for (const auto& output : op.outputs) if (type(output) != memory_type)
            return fail("Memory result storage disagrees with element width", op.source);
        for (const auto data : m->data) if (type(op.inputs[data.value]) != memory_type)
            return fail("Memory data storage disagrees with element width", op.source);
        if (op.opcode == Opcode::atomic && m->data.size() != (*m->atomic == AtomicKind::compare_exchange ? 2U : 1U))
            return fail("Atomic data roles do not match its kind", op.source);
    }
    if (const auto* t = std::get_if<TextureInfo>(&op.info)) {
        auto handle = [&](const HandleSource& source) {
            if (const auto* dynamic = std::get_if<DynamicHandle>(&source)) {
                if (form == IrForm::legal) return fail("Unresolved dynamic texture handle in legal IR", op.source);
                return index(dynamic->input);
            }
            const auto& key = std::get<TextureKey>(source);
            // A key at handle_program_slot names a host-computed handle and one at
            // texture_pool_slot a pool read (handle_program.h).
            return ((key.cbuf_slot < 18 || key.cbuf_slot == handle_program_slot || key.cbuf_slot == texture_pool_slot) &&
                    key.handle_byte_offset % 4 == 0) ||
                   fail("Invalid texture handle address", op.source);
        };
        if (!handle(t->texture) || (t->sampler && !handle(*t->sampler)) || !indices(t->coordinates) ||
            !indices(t->gradients_x) || !indices(t->gradients_y) || !indices(t->offsets) || !indices(t->data) ||
            !index(t->array_layer) || !index(t->array_index) || !index(t->level_or_bias) ||
            !index(t->compare) || !index(t->sample_index) || !index(t->lod_clamp) ||
            !index(t->pool_texture) || !index(t->pool_sampler)) return false;
        const auto pooled = [](const std::optional<HandleSource>& source) {
            const auto* key = source ? std::get_if<TextureKey>(&*source) : nullptr;
            return key && key->cbuf_slot == texture_pool_slot;
        };
        if (pooled(t->texture) != t->pool_texture.has_value() || (t->sampler && pooled(t->sampler) != t->pool_sampler.has_value()) ||
            (!t->sampler && t->pool_sampler))
            return fail("Texture pool key and pool handle roles disagree", op.source);
        const bool image = op.opcode == Opcode::image_load || op.opcode == Opcode::image_store || op.opcode == Opcode::image_atomic;
        if ((image && t->mode != TextureMode::fetch) ||
            (op.opcode == Opcode::texture_sample && t->mode > TextureMode::fetch) ||
            (op.opcode == Opcode::texture_query && (t->mode < TextureMode::query_lod || t->mode > TextureMode::query_samples)))
            return fail("Texture mode does not match its opcode", op.source);
        if ((op.opcode == Opcode::texture_sample || op.opcode == Opcode::texture_query || op.opcode == Opcode::image_load) && !t->data.empty())
            return fail("Texture read/query cannot carry store data", op.source);
        if (op.opcode == Opcode::image_store && t->data.size() != static_cast<std::size_t>(std::popcount(t->component_mask)))
            return fail("Image store data count does not match component mask", op.source);
        if (op.opcode == Opcode::image_atomic && (!t->atomic || t->data.size() != (*t->atomic == AtomicKind::compare_exchange ? 2U : 1U)))
            return fail("Image atomic data roles do not match atomic kind", op.source);
        if (t->component_mask == 0 || t->component_mask > 15 || t->gather_component > 3 || t->query_component > 3)
            return fail("Invalid texture component selection", op.source);
        if (op.opcode != Opcode::image_store && op.outputs.size() != static_cast<std::size_t>(std::popcount(t->component_mask)))
            return fail("Texture result count does not match component mask", op.source);
        if ((t->mode == TextureMode::sample_implicit || t->mode == TextureMode::sample_bias || t->mode == TextureMode::query_lod) &&
            !includes(op.effects, Effects::derivative)) return fail("Implicit texture derivative effect is missing", op.source);
        if (form == IrForm::legal && (!t->descriptor || !t->resource)) return fail("Texture facts are unresolved in legal IR", op.source);
        if (op.opcode == Opcode::image_atomic && !t->atomic) return fail("Image atomic kind is absent", op.source);
        const bool needs_sampler = (op.opcode == Opcode::texture_sample && t->mode != TextureMode::fetch) ||
                                   (op.opcode == Opcode::texture_query && t->mode == TextureMode::query_lod);
        if (form == IrForm::legal && needs_sampler != t->sampler_resource.has_value())
            return fail("Texture sampler resource disagrees with operation mode", op.source);
        const auto texture_type = storage_type(op.numeric.scalar);
        for (const auto& output : op.outputs) if (type(output) != texture_type)
            return fail("Texture result storage disagrees with scalar type", op.source);
        for (const auto data : t->data) if (type(op.inputs[data.value]) != texture_type)
            return fail("Image data storage disagrees with scalar type", op.source);
    }
    if (const auto* io = std::get_if<IoInfo>(&op.info)) {
        if (!index(io->dynamic_byte_offset) || !index(io->vertex_index) || !index(io->sample_index) ||
            !index(io->reciprocal_w) || !index(io->offset_x) || !index(io->offset_y) || !indices(io->data)) return false;
        if (io->component_mask == 0 || io->component_mask > 15 || io->type.components == 0 || io->type.components > 4)
            return fail("Invalid IO component description", op.source);
        for (const auto& output : op.outputs) if (type(output) != storage_type(io->type.scalar))
            return fail("IO result storage disagrees with scalar type", op.source);
        for (const auto data : io->data) if (type(op.inputs[data.value]) != storage_type(io->type.scalar))
            return fail("IO data storage disagrees with scalar type", op.source);
        if (op.opcode != Opcode::store_io && !io->data.empty())
            return fail("IO read/interpolation cannot carry store data", op.source);
        const auto count = static_cast<std::size_t>(std::popcount(io->component_mask));
        if ((op.opcode == Opcode::store_io && io->data.size() != count) ||
            (op.opcode != Opcode::store_io && op.outputs.size() != count)) return fail("IO signature disagrees with component mask", op.source);
    }
    if (const auto* call = std::get_if<CallInfo>(&op.info)) {
        if (call->target.value >= p.functions.size()) return fail("Call target is outside function arena", op.source);
        const auto& target = p.functions[call->target.value];
        if (target.parameters.size() != op.inputs.size() || target.results.size() != op.outputs.size())
            return fail("Call signature does not match target", op.source);
        for (std::size_t i = 0; i < target.parameters.size(); ++i) {
            if (target.parameters[i].value >= p.values.size() || type(op.inputs[i]) != p.values[target.parameters[i].value].type)
                return fail("Call argument type does not match target", op.source);
        }
        for (std::size_t i = 0; i < target.results.size(); ++i) if (type(op.outputs[i]) != target.results[i])
            return fail("Call result type does not match target", op.source);
    }
    if ((std::holds_alternative<MemoryInfo>(op.info) || std::holds_alternative<TextureInfo>(op.info) ||
         std::holds_alternative<IoInfo>(op.info)) && std::find(referenced.begin(), referenced.end(), false) != referenced.end())
        return fail("Operation input has no metadata role", op.source);
    return true;
}
bool Validator::numeric_mode(const Operation& op) {
    const auto& mode = op.numeric;
    if (!valid_numeric_mode(mode)) return fail("Numeric domain, scalar or modifiers disagree", op.source);
    const auto schema = operand_schema(op.opcode);
    const bool floating = mode.domain == Domain::floating;
    if (schema == OperandSchema::conversion) {
        const auto& conversion = std::get<ConversionInfo>(op.info);
        const auto expected = scalar_domain(conversion.source) == Domain::floating ? conversion.source : conversion.destination;
        if (mode.scalar != expected || mode.domain != scalar_domain(expected))
            return fail("Conversion numeric mode does not describe its floating side or integer destination", op.source);
    }
    if ((schema == OperandSchema::numeric || schema == OperandSchema::comparison) && mode.domain == Domain::boolean)
        return fail("Boolean values require predicate logic, not numeric arithmetic", op.source);
    if (schema == OperandSchema::pack_half || schema == OperandSchema::unpack_half) {
        const auto& conversion = std::get<ConversionInfo>(op.info);
        const bool packing = schema == OperandSchema::pack_half;
        if (mode.scalar != ScalarType::f32 || conversion.source != (packing ? ScalarType::f32 : ScalarType::f16) ||
            conversion.destination != (packing ? ScalarType::f16 : ScalarType::f32))
            return fail("Half packing requires explicit f32/f16 conversion metadata", op.source);
    }
    if (schema == OperandSchema::bitcast && mode.domain != Domain::bits)
        return fail("Bitcast cannot perform numeric interpretation", op.source);
    if (schema == OperandSchema::carry && (mode.domain != Domain::unsigned_integer || mode.scalar != ScalarType::u32))
        return fail("Carry arithmetic requires unsigned word32 mode", op.source);
    if (schema == OperandSchema::predicate_logic && mode.domain != Domain::boolean)
        return fail("Logical arithmetic requires Boolean mode", op.source);
    switch (op.opcode) {
    case Opcode::fused_multiply_add: case Opcode::reciprocal: case Opcode::reciprocal_sqrt:
    case Opcode::square_root: case Opcode::sine: case Opcode::cosine: case Opcode::exp2:
    case Opcode::log2: case Opcode::range_reduce_sincos: case Opcode::range_reduce_exp2:
    case Opcode::ceiling: case Opcode::floor: case Opcode::round: case Opcode::truncate:
    case Opcode::is_nan: case Opcode::derivative_x: case Opcode::derivative_y:
    case Opcode::pack_half2: case Opcode::unpack_half2:
        if (!floating) return fail("Floating opcode requires a floating numeric mode", op.source);
        break;
    case Opcode::bit_and: case Opcode::bit_or: case Opcode::bit_xor: case Opcode::bit_not:
    case Opcode::bit_count: case Opcode::bit_reverse: case Opcode::find_lsb: case Opcode::find_msb:
    case Opcode::shift_left: case Opcode::shift_right: case Opcode::funnel_shift_left:
    case Opcode::funnel_shift_right: case Opcode::bit_extract: case Opcode::bit_insert:
    case Opcode::multiply_high: case Opcode::shuffle:
        if (floating || mode.domain == Domain::boolean) return fail("Integer opcode has a noninteger numeric mode", op.source);
        break;
    default: break;
    }
    if ((schema == OperandSchema::numeric || schema == OperandSchema::comparison) && mode.domain == Domain::bits) {
        switch (op.opcode) {
        case Opcode::bit_and: case Opcode::bit_or: case Opcode::bit_xor: case Opcode::bit_not:
        case Opcode::bit_count: case Opcode::bit_reverse: case Opcode::find_lsb: case Opcode::find_msb: break;
        default: return fail("Arithmetic and comparisons require an interpreted numeric domain", op.source);
        }
    }
    if ((schema == OperandSchema::comparison || schema == OperandSchema::is_nan) && (mode.flush_output || mode.saturate))
        return fail("Predicate result cannot have floating output modifiers", op.source);
    return true;
}
bool Validator::operand_types(const Operation& op) {
    const auto numeric = storage_type(op.numeric.scalar);
    auto in = [&](std::size_t i, ValueType wanted) { return type(op.inputs[i]) == wanted; };
    auto out = [&](std::size_t i, ValueType wanted) { return type(op.outputs[i]) == wanted; };
    auto all = [&](const std::vector<Operand>& operands, ValueType wanted) {
        return std::all_of(operands.begin(), operands.end(), [&](const Operand& value) { return type(value) == wanted; });
    };
    bool matches = true;
    switch (operand_schema(op.opcode)) {
    case OperandSchema::numeric: matches = all(op.inputs, numeric) && all(op.outputs, numeric); break;
    case OperandSchema::copy: matches = type(op.inputs[0]) == type(op.outputs[0]); break;
    case OperandSchema::comparison: matches = all(op.inputs, numeric) && out(0, ValueType::predicate); break;
    case OperandSchema::select: matches = in(0, ValueType::predicate) && type(op.inputs[1]) == type(op.inputs[2]) && type(op.inputs[1]) == type(op.outputs[0]); break;
    case OperandSchema::conversion: case OperandSchema::bitcast: {
        const auto& conversion = std::get<ConversionInfo>(op.info);
        matches = in(0, storage_type(conversion.source)) && out(0, storage_type(conversion.destination));
        if (op.opcode == Opcode::bitcast) matches = matches && scalar_bits(conversion.source) == scalar_bits(conversion.destination);
        break;
    }
    case OperandSchema::carry: matches = in(0, ValueType::word32) && in(1, ValueType::word32) && in(2, ValueType::predicate) && out(0, ValueType::word32) && out(1, ValueType::predicate); break;
    case OperandSchema::shift: matches = in(0, numeric) && in(1, ValueType::word32) && out(0, numeric); break;
    case OperandSchema::funnel: matches = in(0, numeric) && in(1, numeric) && in(2, ValueType::word32) && out(0, numeric); break;
    case OperandSchema::extract: matches = in(0, numeric) && in(1, ValueType::word32) && in(2, ValueType::word32) && out(0, numeric); break;
    case OperandSchema::insert: matches = in(0, numeric) && in(1, numeric) && in(2, ValueType::word32) && in(3, ValueType::word32) && out(0, numeric); break;
    case OperandSchema::pack_half: case OperandSchema::unpack_half: matches = all(op.inputs, ValueType::word32) && all(op.outputs, ValueType::word32); break;
    case OperandSchema::pack_word64: matches = all(op.inputs, ValueType::word32) && out(0, ValueType::word64); break;
    case OperandSchema::unpack_word64: matches = in(0, ValueType::word64) && all(op.outputs, ValueType::word32); break;
    case OperandSchema::predicate_logic: case OperandSchema::vote: matches = all(op.inputs, ValueType::predicate) && all(op.outputs, ValueType::predicate); break;
    case OperandSchema::predicate_to_mask: case OperandSchema::ballot: matches = in(0, ValueType::predicate) && out(0, ValueType::word32); break;
    case OperandSchema::mask_to_predicate: matches = in(0, ValueType::word32) && out(0, ValueType::predicate); break;
    case OperandSchema::is_nan: matches = in(0, numeric) && out(0, ValueType::predicate); break;
    case OperandSchema::shuffle: matches = all(op.inputs, ValueType::word32) && out(0, ValueType::word32) && out(1, ValueType::predicate); break;
    case OperandSchema::matrix: matches = all(op.inputs, ValueType::word32) && all(op.outputs, ValueType::word32); break;
    case OperandSchema::reduction: {
        const auto* info = std::get_if<SubgroupInfo>(&op.info);
        const auto kind = info ? info->reduction : std::nullopt;
        // A word's integer interpretation, unsigned or (for the extrema only)
        // signed, with no rounding, flush or saturation.
        const auto& n = op.numeric;
        const bool plain = n.rounding == Rounding::not_applicable && !n.flush_inputs && !n.flush_output && !n.saturate && !n.precise;
        const bool unsigned_mode = n.domain == Domain::unsigned_integer && n.scalar == ScalarType::u32;
        const bool signed_mode = n.domain == Domain::signed_integer && n.scalar == ScalarType::s32;
        const bool extremum = kind == AtomicKind::minimum || kind == AtomicKind::maximum;
        matches = all(op.inputs, ValueType::word32) && all(op.outputs, ValueType::word32) && plain &&
                  (kind == AtomicKind::add || extremum || kind == AtomicKind::bit_or || kind == AtomicKind::bit_xor) &&
                  (unsigned_mode || (signed_mode && extremum));
        break;
    }
    case OperandSchema::memory: case OperandSchema::texture: case OperandSchema::io: case OperandSchema::call: case OperandSchema::none: break;
    }
    return matches || fail("Operand storage types disagree with opcode schema", op.source);
}
bool Validator::operation(const Operation& op, BlockId block, std::size_t order) {
    const auto* sig = signature(op.opcode);
    if (!sig || !sig->inputs.accepts(op.inputs.size()) || !sig->outputs.accepts(op.outputs.size()))
        return fail("Opcode or operand arity is invalid", op.source);
    if (op.inputs.size() + op.outputs.size() > p.limits.operands_per_operation)
        return fail("Operation operand limit exceeded", op.source, DiagnosticCode::limit_exceeded);
    if (op.info.index() != static_cast<std::size_t>(sig->info)) return fail("Opcode metadata has the wrong kind", op.source);
    if (!includes(op.effects, required_effects(op))) return fail("Required operation effects are missing", op.source);
    if (op.numeric.domain == Domain::floating && op.numeric.rounding == Rounding::not_applicable)
        return fail("Floating operation has no explicit rounding mode", op.source);
    if (op.numeric.domain != Domain::floating && (op.numeric.flush_inputs || op.numeric.flush_output || op.numeric.saturate))
        return fail("Floating modifiers attached to a nonfloating operation", op.source);
    for (const auto& input : op.inputs) if (!operand(input, block, order, op.source)) return false;
    for (const auto& output : op.outputs) if (!operand(output, block, order, op.source, true)) return false;
    auto predicate = [&](const Operand& value) { return type(value) == ValueType::predicate; };
    if ((op.opcode == Opcode::compare || op.opcode == Opcode::is_nan || op.opcode == Opcode::mask_to_predicate) &&
        !predicate(op.outputs[0])) return fail("Operation requires a predicate result", op.source);
    if (op.opcode == Opcode::select && (!predicate(op.inputs[0]) || type(op.inputs[1]) != type(op.inputs[2]) ||
        type(op.inputs[1]) != type(op.outputs[0]))) return fail("Select input/result types disagree", op.source);
    if (op.opcode == Opcode::copy && type(op.inputs[0]) != type(op.outputs[0])) return fail("Copy changes storage type", op.source);
    if ((op.opcode == Opcode::add_carry || op.opcode == Opcode::subtract_borrow || op.opcode == Opcode::shuffle) &&
        !predicate(op.outputs[1])) return fail("Secondary result must be a predicate", op.source);
    return numeric_mode(op) && operand_types(op) && metadata(op);
}

// Cooper-style immediate dominators in reverse postorder. IDs never imply order.
bool Validator::dominance(const Function& function) {
    std::vector<bool> visited(p.blocks.size());
    std::vector<BlockId> postorder;
    std::vector<std::pair<BlockId, std::size_t>> stack{{function.entry, 0}};
    visited[function.entry.value] = true;
    while (!stack.empty()) {
        if (!step()) return false;
        auto& [block, next] = stack.back();
        if (next < outgoing[block.value].size()) {
            const auto child = outgoing[block.value][next++];
            if (!visited[child.value]) { visited[child.value] = true; stack.emplace_back(child, 0); }
        } else { postorder.push_back(block); stack.pop_back(); }
    }
    if (postorder.size() != function.blocks.size()) return fail("Function contains unreachable blocks");
    std::reverse(postorder.begin(), postorder.end());
    std::vector<std::size_t> rank(p.blocks.size());
    std::vector<BlockId> idom(p.blocks.size());
    for (std::size_t i = 0; i < postorder.size(); ++i) rank[postorder[i].value] = i;
    idom[function.entry.value] = function.entry;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 1; i < postorder.size(); ++i) {
            if (!step()) return false;
            const auto block = postorder[i];
            BlockId parent;
            for (auto predecessor : incoming[block.value]) {
                if (!idom[predecessor.value].valid()) continue;
                if (!parent.valid()) { parent = predecessor; continue; }
                auto other = predecessor;
                while (parent != other) {
                    if (!step()) return false;
                    if (rank[parent.value] > rank[other.value]) parent = idom[parent.value];
                    else other = idom[other.value];
                }
            }
            if (parent != idom[block.value]) { idom[block.value] = parent; changed = true; }
        }
    }
    for (const auto& use : uses) {
        if (p.blocks[use.block.value].function != p.blocks[function.entry.value].function) continue;
        if (!step()) return false;
        const auto& definition = definitions[use.value.value];
        if (!definition) return fail("SSA value has no definition", use.source, DiagnosticCode::undefined_read);
        if (p.blocks[definition->block.value].function != p.blocks[use.block.value].function)
            return fail("Value crosses a function boundary without an argument", use.source);
        if (definition->block == use.block) {
            if (!definition->parameter && definition->order >= use.order)
                return fail("Value is read before its definition", use.source);
        } else {
            auto cursor = use.block;
            while (cursor != definition->block && cursor != function.entry) {
                if (!step()) return false;
                cursor = idom[cursor.value];
            }
            if (cursor != definition->block) return fail("Value definition does not dominate its use", use.source);
        }
    }
    return true;
}

Status Validator::run() {
    if (p.api_mandated_early_fragment_tests && p.stage != Stage::fragment)
        return diagnose(phase, DiagnosticCode::invalid_ir, {}, "API early fragment tests require a fragment stage");
    if (p.functions.empty() || p.entry.value >= p.functions.size()) return diagnose(phase, DiagnosticCode::invalid_ir, {}, "Program entry function is invalid");
    for (const auto& value : p.values) {
        if (value.type > ValueType::predicate || (form != IrForm::registers && value.kind != ValueKind::temporary))
            return diagnose(phase, DiagnosticCode::invalid_ir, {}, "Invalid value type or mutable local in SSA IR");
    }
    std::uint64_t edge_count = 0, phi_count = 0;
    for (std::size_t f = 0; f < p.functions.size() && !error; ++f) {
        const auto& function = p.functions[f];
        if (function.entry.value >= p.blocks.size() || function.blocks.empty()) { fail("Function entry block is invalid"); break; }
        if (p.blocks[function.entry.value].function != FunctionId{static_cast<std::uint32_t>(f)}) { fail("Function entry has the wrong owner"); break; }
        for (auto id : function.blocks) {
            if (!step()) break;
            if (id.value >= p.blocks.size() || ++block_owners[id.value] != 1 || p.blocks[id.value].function.value != f) { fail("Block ownership is invalid"); break; }
        }
        for (auto parameter : function.parameters) if (!define(parameter, {function.entry, 0, true}, {})) break;
    }
    for (std::size_t b = 0; b < p.blocks.size() && !error; ++b) {
        const auto& block = p.blocks[b];
        if (block_owners[b] != 1) { fail("Block is not owned by exactly one function", block.source); break; }
        if (const auto* cases = std::get_if<Switch>(&block.terminator); cases && cases->cases.size() >= p.limits.edges) {
            fail("Switch target limit exceeded", block.source, DiagnosticCode::limit_exceeded); break;
        }
        if (block.predecessors.size() > p.limits.edges) { fail("Predecessor limit exceeded", block.source, DiagnosticCode::limit_exceeded); break; }
        outgoing[b] = successors(block.terminator);
        edge_count += outgoing[b].size();
        if (edge_count > p.limits.edges) { fail("CFG edge limit exceeded", block.source, DiagnosticCode::limit_exceeded); break; }
        for (auto target : outgoing[b]) {
            if (!step()) break;
            if (target.value >= p.blocks.size() || p.blocks[target.value].function != block.function) { fail("CFG target is invalid or crosses a function", block.source); break; }
            incoming[target.value].push_back(BlockId{static_cast<std::uint32_t>(b)});
        }
    }
    for (std::size_t b = 0; b < p.blocks.size() && !error; ++b) {
        const BlockId id{static_cast<std::uint32_t>(b)};
        const auto& block = p.blocks[b];
        auto predecessors = block.predecessors;
        std::sort(predecessors.begin(), predecessors.end());
        if (predecessors != incoming[b]) { fail("Predecessor labels disagree with CFG edges", block.source); break; }
        if (form == IrForm::registers && !block.phis.empty()) { fail("Register IR contains SSA phis", block.source); break; }
        for (const auto& phi : block.phis) {
            phi_count += phi.inputs.size();
            if (phi_count > p.limits.phi_inputs) { fail("Phi input limit exceeded", phi.source, DiagnosticCode::limit_exceeded); break; }
            if (phi.inputs.empty() || !define(phi.output, {id, 0, false}, phi.source)) { if (!error) fail("Phi has no inputs", phi.source); break; }
            std::vector<BlockId> labels;
            for (const auto& input : phi.inputs) {
                labels.push_back(input.predecessor);
                if (input.predecessor.value >= p.blocks.size() || !operand(input.value, input.predecessor,
                    std::numeric_limits<std::size_t>::max(), phi.source)) break;
                if (p.values[phi.output.value].type != p.values[input.value.value].type) { fail("Phi value types disagree", phi.source); break; }
            }
            std::sort(labels.begin(), labels.end());
            if (labels != predecessors) { fail("Phi must name each predecessor exactly once", phi.source); break; }
        }
        std::size_t order = 1;
        for (auto op : block.operations) {
            if (error) break;
            if (op.value >= p.operations.size() || ++operation_owners[op.value] != 1) { fail("Operation ownership is invalid", block.source); break; }
            if (!operation(p.operations[op.value], id, order++)) break;
        }
        if (error) break;
        if (const auto* branch = std::get_if<Branch>(&block.terminator)) {
            if (!operand(branch->condition, id, order, block.source) || type(branch->condition) != ValueType::predicate) fail("Branch requires a predicate", block.source);
        } else if (const auto* select = std::get_if<Switch>(&block.terminator)) {
            if (!operand(select->selector, id, order, block.source) || type(select->selector) != ValueType::word32) fail("Switch requires a word selector", block.source);
            std::set<std::uint32_t> cases;
            for (const auto& item : select->cases) if (!cases.insert(item.bits).second) { fail("Switch has duplicate case values", block.source); break; }
        } else if (const auto* result = std::get_if<Return>(&block.terminator)) {
            const auto& expected = p.functions[block.function.value].results;
            if (result->values.size() != expected.size()) fail("Return signature does not match function", block.source);
            else for (std::size_t i = 0; i < expected.size(); ++i) {
                if (!operand(result->values[i], id, order, block.source) || type(result->values[i]) != expected[i]) { fail("Return value type is invalid", block.source); break; }
            }
        } else if (std::holds_alternative<Unreachable>(block.terminator)) fail("Reachable block has no executable terminator", block.source);
    }
    if (!error && std::any_of(operation_owners.begin(), operation_owners.end(), [](auto n) { return n != 1; })) fail("Operation is not owned by exactly one block");
    if (!error && form == IrForm::registers) for (const auto& use : uses) {
        if (!definitions[use.value.value]) { fail("Compiler local has no definition", use.source, DiagnosticCode::undefined_read); break; }
        if (p.blocks[definitions[use.value.value]->block.value].function != p.blocks[use.block.value].function) {
            fail("Compiler local crosses a function boundary", use.source); break;
        }
    }
    if (!error && form != IrForm::registers) for (const auto& function : p.functions) if (!dominance(function)) break;
    if (error) return *error;
    return std::monostate{};
}
} // namespace
Status validate(const Program& program, IrForm form, Phase phase) {
    const auto& limit = program.limits;
    if (program.functions.size() > limit.functions || program.blocks.size() > limit.blocks ||
        program.operations.size() > limit.operations || program.values.size() > limit.values ||
        program.dependencies.size() > limit.dependencies) {
        return diagnose(phase, DiagnosticCode::limit_exceeded, {}, "IR arena limit exceeded");
    }
    auto dependencies = merge_dependencies(program.dependencies, {}, limit, phase);
    if (auto* error = std::get_if<Diagnostic>(&dependencies)) return *error;
    if (std::get<std::vector<Dependency>>(dependencies) != program.dependencies)
        return diagnose(phase, DiagnosticCode::invalid_ir, {}, "Program dependencies are not canonical");
    return Validator(program, form, phase).run();
}
} // namespace armigo::shader::pipeline
