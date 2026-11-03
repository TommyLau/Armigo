// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#include "pipeline/decode/resource_fields.h"

#include <array>
#include <string_view>
#include <utility>

namespace armigo::shader::pipeline {
namespace {
using namespace resource_decode;

// Field-list and access-list builders. Every DecodedField/SourceRole appears at
// most once per instruction; helpers keep that invariant by construction.
void add_field(DecodedInstruction& out, DecodedField field, std::uint64_t bits) {
    out.fields.push_back({field, bits});
}
void add_gpr(DecodedInstruction& out, RegisterRole role, unsigned base, unsigned lanes, Access access) {
    for (unsigned lane = 0; lane < lanes; ++lane) {
        out.registers.push_back({role, static_cast<std::uint8_t>(lane),
            Register{RegisterFile::gpr, static_cast<std::uint16_t>(base + lane)}, access});
    }
}
void add_predicate(DecodedInstruction& out, RegisterRole role, unsigned index, Access access) {
    out.registers.push_back({role, 0, Register{RegisterFile::predicate, static_cast<std::uint16_t>(index)}, access});
}
void add_source(DecodedInstruction& out, SourceRole role, unsigned index) {
    out.sources.push_back({role, Register{RegisterFile::gpr, static_cast<std::uint16_t>(index)}, {}});
}
// Predicate sources invert through the negate modifier; boolean inversion is a
// predicate's only meaningful modifier.
void add_predicate_source(DecodedInstruction& out, unsigned index, bool inverted) {
    DecodedSource source;
    source.role = SourceRole::predicate;
    source.value = Register{RegisterFile::predicate, static_cast<std::uint16_t>(index)};
    source.modifiers.negate = inverted;
    out.sources.push_back(std::move(source));
}

Result<DecodedInstruction> malformed(Pc pc, InstructionName name, std::string_view detail) {
    return diagnose(Phase::decode, DiagnosticCode::malformed_input, {pc, name}, std::string(detail));
}

DecodedInstruction begin(Pc pc, RawInstruction word, InstructionName name, Encoding encoding, bool predicated) {
    DecodedInstruction out;
    out.pc = pc;
    out.raw = word;
    out.name = name;
    out.encoding = encoding;
    out.predicate = instruction_predicate(word, predicated);
    return out;
}

unsigned component_count(std::uint64_t mask) {
    unsigned count = 0;
    for (unsigned bit = 0; bit < 4; ++bit) {
        count += static_cast<unsigned>((mask >> bit) & 1);
    }
    return count;
}

// --- Flow control ------------------------------------------------------------

Result<DecodedInstruction> decode_simple_flow(Pc pc, RawInstruction word, InstructionName name) {
    DecodedInstruction out = begin(pc, word, name, Encoding::none, true);
    add_field(out, DecodedField::condition, condition_selector(word));
    return out;
}

Result<DecodedInstruction> decode_bra(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::bra, Encoding::none, true);
    add_field(out, DecodedField::condition, condition_selector(word));
    add_field(out, DecodedField::branch_displacement, branch_displacement_value(relative_displacement(word)));
    return out;
}

// SSY/PBK/PCNT push a reconvergence/branch target and CAL pushes a call
// target; all four are unconditionally executed (NoPred encodings).
Result<DecodedInstruction> decode_push(Pc pc, RawInstruction word, InstructionName name) {
    DecodedInstruction out = begin(pc, word, name, Encoding::none, false);
    add_field(out, DecodedField::branch_displacement, branch_displacement_value(relative_displacement(word)));
    return out;
}

Result<DecodedInstruction> decode_register_branch(Pc pc, RawInstruction word, InstructionName name,
                                                  bool absolute_immediate) {
    DecodedInstruction out = begin(pc, word, name, Encoding::registers, true);
    add_field(out, DecodedField::condition, condition_selector(word));
    if (absolute_immediate) {
        // JMP/JMX combine a full 32-bit immediate with the register value; the
        // flow lane owns the combination, so the raw immediate is preserved.
        add_field(out, DecodedField::branch_absolute, 1);
        add_field(out, DecodedField::branch_displacement, word.bits<20, 32>());
    } else {
        add_field(out, DecodedField::branch_displacement, branch_displacement_value(relative_displacement(word)));
    }
    add_field(out, DecodedField::target_register, word.bits<8, 8>());
    add_source(out, SourceRole::a, static_cast<unsigned>(word.bits<8, 8>()));
    add_gpr(out, RegisterRole::source_a, static_cast<unsigned>(word.bits<8, 8>()), 1, Access::read);
    return out;
}

Result<DecodedInstruction> decode_nop(Pc pc, RawInstruction word) {
    return begin(pc, word, InstructionName::nop, Encoding::none, true);
}

// VOTE.VTG: a vote across the vertex, tessellation or geometry threads of a
// warp. It writes no register or predicate; only its guard predicate is read.
Result<DecodedInstruction> decode_votevtg(Pc pc, RawInstruction word) {
    return begin(pc, word, InstructionName::votevtg, Encoding::none, true);
}

Result<DecodedInstruction> decode_depbar(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::depbar, Encoding::none, true);
    add_field(out, DecodedField::barrier_id, word.bits<26, 3>());
    return out;
}

Result<DecodedInstruction> decode_bar(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::bar, Encoding::registers, true);
    add_field(out, DecodedField::barrier_kind, word.bits<32, 3>());
    add_field(out, DecodedField::barrier_id, word.bits<20, 12>());
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, 1, Access::read);
    const unsigned predicate = static_cast<unsigned>(word.bits<39, 3>());
    add_predicate_source(out, predicate, word.bits<42, 1>() != 0);
    add_predicate(out, RegisterRole::predicate_source, predicate, Access::read);
    return out;
}

Result<DecodedInstruction> decode_membar(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::membar, Encoding::none, true);
    // Raw selector: 0 = cta, 1 = gl, 2 = sys, 3 = vc; the memory lane maps it.
    add_field(out, DecodedField::memory_scope, word.bits<8, 2>());
    return out;
}

// --- Attribute load/store and interpolation ----------------------------------

Result<DecodedInstruction> decode_ald(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::ald, Encoding::registers, true);
    const MemorySize size = attribute_size(word.bits<47, 2>());
    add_field(out, DecodedField::io_byte_offset, word.bits<20, 11>());
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()),
        static_cast<unsigned>(size.elements), Access::write);
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, 1, Access::read);
    return out;
}

Result<DecodedInstruction> decode_ast(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::ast, Encoding::registers, true);
    const MemorySize size = attribute_size(word.bits<47, 2>());
    add_field(out, DecodedField::io_byte_offset, word.bits<20, 11>());
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, 1, Access::read);
    add_gpr(out, RegisterRole::source_b, static_cast<unsigned>(word.bits<0, 8>()),
        static_cast<unsigned>(size.elements), Access::read);
    const unsigned index = static_cast<unsigned>(word.bits<39, 8>());
    add_source(out, SourceRole::c, index);
    add_gpr(out, RegisterRole::source_c, index, 1, Access::read);
    return out;
}

Result<DecodedInstruction> decode_ipa(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::ipa, Encoding::registers, true);
    // Raw selectors: operation 0 = pass, 1 = multiply, 2 = constant, 3 = sc;
    // location 0..3 orders center, centroid, sample, offset. Saturation and
    // the indexed mode stay encoded-only (ipa_saturate and the Idx bit).
    add_field(out, DecodedField::io_byte_offset, word.bits<28, 10>());
    add_field(out, DecodedField::interpolation_mode, word.bits<54, 2>());
    add_field(out, DecodedField::interpolation_location, word.bits<52, 2>());
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), 1, Access::write);
    const unsigned index_a = static_cast<unsigned>(word.bits<8, 8>());
    const unsigned index_b = static_cast<unsigned>(word.bits<20, 8>());
    const unsigned index_c = static_cast<unsigned>(word.bits<39, 8>());
    add_source(out, SourceRole::a, index_a);
    add_gpr(out, RegisterRole::source_a, index_a, 1, Access::read);
    add_source(out, SourceRole::b, index_b);
    add_gpr(out, RegisterRole::source_b, index_b, 1, Access::read);
    add_source(out, SourceRole::c, index_c);
    add_gpr(out, RegisterRole::source_c, index_c, 1, Access::read);
    const unsigned predicate = static_cast<unsigned>(word.bits<47, 3>());
    add_predicate_source(out, predicate, word.bits<50, 1>() != 0);
    add_predicate(out, RegisterRole::predicate_source, predicate, Access::read);
    return out;
}

// --- Memory ------------------------------------------------------------------

Result<DecodedInstruction> decode_ldc(Pc pc, RawInstruction word) {
    const std::uint64_t selector = word.bits<48, 3>();
    if (selector > 5) {
        return malformed(pc, InstructionName::ldc, "LDC size selector must be 32- or 64-bit");
    }
    DecodedInstruction out = begin(pc, word, InstructionName::ldc, Encoding::constant_buffer, true);
    add_field(out, DecodedField::memory_space, static_cast<std::uint64_t>(AddressSpace::constant_buffer));
    add_field(out, DecodedField::memory_byte_offset, static_cast<std::uint64_t>(word.signed_bits<20, 16>()));
    const MemorySize size = ls_size(selector);
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()),
        static_cast<unsigned>(size.elements), Access::write);
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, 1, Access::read);
    // The constant-buffer slot stays encoded-only: ldc_cbuf_slot(word).
    return out;
}

// LDL/LDS loads and STL/STS stores; the address space differs by family.
Result<DecodedInstruction> decode_narrow_memory(Pc pc, RawInstruction word, InstructionName name, AddressSpace space,
                                                bool store) {
    const std::uint64_t selector = word.bits<48, 3>();
    if (selector > 6) {
        return malformed(pc, name, "Narrow memory size selector must not use the unordered 128-bit encoding");
    }
    DecodedInstruction out = begin(pc, word, name, Encoding::registers, true);
    add_field(out, DecodedField::memory_space, static_cast<std::uint64_t>(space));
    add_field(out, DecodedField::memory_byte_offset, static_cast<std::uint64_t>(word.signed_bits<20, 24>()));
    const MemorySize size = ls_size(selector);
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, 1, Access::read);
    if (store) {
        add_gpr(out, RegisterRole::source_b, static_cast<unsigned>(word.bits<0, 8>()),
            static_cast<unsigned>(size.elements), Access::read);
    } else {
        add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()),
            static_cast<unsigned>(size.elements), Access::write);
    }
    return out;
}

// LDG/STG: storage access with a signed byte offset and an optional extended
// 40-bit address pair (SrcA and SrcA + 1).
Result<DecodedInstruction> decode_storage(bool store, Pc pc, RawInstruction word) {
    const InstructionName name = store ? InstructionName::stg : InstructionName::ldg;
    DecodedInstruction out = begin(pc, word, name, Encoding::registers, true);
    add_field(out, DecodedField::memory_space, static_cast<std::uint64_t>(AddressSpace::storage));
    add_field(out, DecodedField::memory_byte_offset, static_cast<std::uint64_t>(word.signed_bits<20, 24>()));
    const MemorySize size = ls_size(word.bits<48, 3>());
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, ldg_extended_address(word) ? 2U : 1U, Access::read);
    if (store) {
        add_gpr(out, RegisterRole::source_b, static_cast<unsigned>(word.bits<0, 8>()),
            static_cast<unsigned>(size.elements), Access::read);
    } else {
        add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()),
            static_cast<unsigned>(size.elements), Access::write);
    }
    return out;
}

// --- Global and shared atomics ------------------------------------------------

Result<DecodedInstruction> decode_atom(Pc pc, RawInstruction word) {
    const std::uint64_t selector = atom_size_selector(word);
    if (selector > 5) {
        return malformed(pc, InstructionName::atom, "Atomic size selector is out of range");
    }
    const std::uint64_t operation = word.bits<52, 4>();
    if (operation > 8 && operation != 10) {
        return malformed(pc, InstructionName::atom, "Atomic operation selector is undefined");
    }
    DecodedInstruction out = begin(pc, word, InstructionName::atom, Encoding::registers, true);
    add_field(out, DecodedField::memory_space, static_cast<std::uint64_t>(AddressSpace::storage));
    add_field(out, DecodedField::memory_byte_offset, static_cast<std::uint64_t>(word.signed_bits<28, 20>()));
    const MemorySize size = atomic_size(selector);
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    add_field(out, DecodedField::atomic_operation, operation);
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, ld_st_extended_address(word) ? 2U : 1U, Access::read);
    const unsigned data = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, data);
    add_gpr(out, RegisterRole::source_b, data, 1, Access::read);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), 1, Access::write);
    return out;
}

Result<DecodedInstruction> decode_atom_cas(Pc pc, RawInstruction word) {
    DecodedInstruction out = begin(pc, word, InstructionName::atom_cas, Encoding::registers, true);
    add_field(out, DecodedField::memory_space, static_cast<std::uint64_t>(AddressSpace::storage));
    add_field(out, DecodedField::memory_byte_offset, static_cast<std::uint64_t>(word.signed_bits<28, 20>()));
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, ld_st_extended_address(word) ? 2U : 1U, Access::read);
    const unsigned expected = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, expected);
    add_gpr(out, RegisterRole::source_b, expected, 1, Access::read);
    // The destination supplies the replacement value and receives the old
    // value, so it is read and written. Size stays encoded-only
    // (atom_cas_selector(word)).
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), 1, Access::read_write);
    return out;
}

Result<DecodedInstruction> decode_shared_atomic(Pc pc, RawInstruction word, bool cas) {
    const InstructionName name = cas ? InstructionName::atoms_cas : InstructionName::atoms;
    if (!cas) {
        const std::uint64_t operation = word.bits<52, 4>();
        if (operation > 8 && operation != 10) {
            return malformed(pc, name, "Shared atomic operation selector is undefined");
        }
    }
    DecodedInstruction out = begin(pc, word, name, Encoding::registers, true);
    add_field(out, DecodedField::memory_space, static_cast<std::uint64_t>(AddressSpace::shared));
    if (!cas) {
        // Shared atomics address words: the guest divides the source register
        // by four, and the immediate is a word count normalized to bytes here.
        add_field(out, DecodedField::memory_byte_offset,
            static_cast<std::uint64_t>(word.signed_bits<30, 22>() * 4));
        const MemorySize size = atomic_size(atoms_size_selector(word));
        add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
        add_field(out, DecodedField::memory_elements, size.elements);
        add_field(out, DecodedField::atomic_operation, word.bits<52, 4>());
    }
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, 1, Access::read);
    const unsigned data = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, data);
    add_gpr(out, RegisterRole::source_b, data, 1, Access::read);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), 1,
        cas ? Access::read_write : Access::write);
    return out;
}

Result<DecodedInstruction> decode_red(Pc pc, RawInstruction word) {
    const std::uint64_t selector = red_size_selector(word);
    if (selector > 5) {
        return malformed(pc, InstructionName::red, "Reduction size selector is out of range");
    }
    DecodedInstruction out = begin(pc, word, InstructionName::red, Encoding::registers, true);
    add_field(out, DecodedField::memory_space, static_cast<std::uint64_t>(AddressSpace::storage));
    add_field(out, DecodedField::memory_byte_offset, static_cast<std::uint64_t>(word.signed_bits<28, 20>()));
    const MemorySize size = atomic_size(selector);
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    add_field(out, DecodedField::atomic_operation, word.bits<23, 3>());
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, ld_st_extended_address(word) ? 2U : 1U, Access::read);
    const unsigned data = static_cast<unsigned>(word.bits<0, 8>());
    add_source(out, SourceRole::b, data);
    add_gpr(out, RegisterRole::source_b, data, 1, Access::read);
    return out;
}

// --- Surface (image) load, store and atomic ----------------------------------

struct SurfaceTarget {
    TextureDimension dimension;
    bool array;
    unsigned coordinates;
};
[[nodiscard]] constexpr SurfaceTarget surface_target(std::uint64_t selector) {
    switch (selector) {
    case 0: return {TextureDimension::d1, false, 1};
    case 1: return {TextureDimension::buffer, false, 1};
    case 2: return {TextureDimension::d1, true, 2};
    case 3: return {TextureDimension::d2, false, 2};
    case 4: return {TextureDimension::d2, true, 3};
    default: return {TextureDimension::d3, false, 3}; // selector 5
    }
}

// Common image decode: coordinates from SrcA, optional predicate destination
// from the DestPred2 slot, bindless handle from SrcC on the _B encodings.
Result<DecodedInstruction> decode_surface(Pc pc, RawInstruction word, InstructionName name, bool store,
    bool bindless, bool sized, std::uint64_t selector, bool predicate_destination) {
    const std::uint64_t dimension = word.bits<33, 3>();
    if (dimension > 5) {
        return malformed(pc, name, "Surface dimension selector is undefined");
    }
    const SurfaceTarget target = surface_target(dimension);
    const unsigned lanes =
        sized ? static_cast<unsigned>(ls_size(selector).elements) : component_count(selector);
    if (!sized && selector == 0) {
        return malformed(pc, name, "Surface component mask must not be empty");
    }
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension,
        texture_dimension_value(target.dimension, target.array, false, false, false, false));
    add_field(out, DecodedField::texture_mode,
        texture_mode_value(TextureMode::fetch, static_cast<unsigned>(dimension), false, false));
    if (sized) {
        const MemorySize size = ls_size(selector);
        add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
        add_field(out, DecodedField::memory_elements, size.elements);
    } else {
        add_field(out, DecodedField::texture_component_mask, selector);
    }
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, target.coordinates, Access::read);
    if (bindless) {
        const unsigned handle = static_cast<unsigned>(word.bits<39, 8>());
        add_source(out, SourceRole::c, handle);
        add_gpr(out, RegisterRole::source_c, handle, 1, Access::read);
    } else {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    if (store) {
        add_gpr(out, RegisterRole::source_b, static_cast<unsigned>(word.bits<0, 8>()), lanes, Access::read);
    } else {
        add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), lanes, Access::write);
    }
    if (predicate_destination) {
        add_predicate(out, RegisterRole::predicate_destination, static_cast<unsigned>(word.bits<30, 3>()),
            Access::write);
    }
    return out;
}

Result<DecodedInstruction> decode_surface_atomic(Pc pc, RawInstruction word, InstructionName name, bool cas,
    bool bindless) {
    const std::uint64_t dimension = word.bits<33, 3>();
    if (dimension > 5) {
        return malformed(pc, name, "Surface dimension selector is undefined");
    }
    const std::uint64_t size_selector = bindless ? word.bits<36, 3>() : word.bits<51, 3>();
    const MemorySize size = atomic_size(size_selector);
    const SurfaceTarget target = surface_target(dimension);
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension,
        texture_dimension_value(target.dimension, target.array, false, false, false, false));
    add_field(out, DecodedField::texture_mode,
        texture_mode_value(TextureMode::fetch, static_cast<unsigned>(dimension), false, false));
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    if (!cas) {
        const std::uint64_t operation = word.bits<29, 4>();
        if (operation > 8) {
            return malformed(pc, name, "Surface atomic operation selector is undefined");
        }
        add_field(out, DecodedField::atomic_operation, operation);
    }
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, target.coordinates, Access::read);
    if (bindless) {
        const unsigned handle = static_cast<unsigned>(word.bits<39, 8>());
        add_source(out, SourceRole::c, handle);
        add_gpr(out, RegisterRole::source_c, handle, 1, Access::read);
    } else {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    const unsigned data = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, data);
    add_gpr(out, RegisterRole::source_b, data, 1, Access::read);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), 1,
        cas ? Access::read_write : Access::write);
    if (cas) {
        add_predicate(out, RegisterRole::predicate_destination, static_cast<unsigned>(word.bits<30, 3>()),
            Access::write);
    }
    return out;
}

Result<DecodedInstruction> decode_surface_reduction(Pc pc, RawInstruction word, bool bindless) {
    const InstructionName name = bindless ? InstructionName::sured_b : InstructionName::sured;
    const std::uint64_t dimension = word.bits<33, 3>();
    if (dimension > 5) {
        return malformed(pc, name, "Surface dimension selector is undefined");
    }
    const std::uint64_t size_selector = word.bits<20, 3>();
    if (size_selector > 5) {
        return malformed(pc, name, "Surface reduction size selector is out of range");
    }
    const MemorySize size = atomic_size(size_selector);
    const SurfaceTarget target = surface_target(dimension);
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension,
        texture_dimension_value(target.dimension, target.array, false, false, false, false));
    add_field(out, DecodedField::texture_mode,
        texture_mode_value(TextureMode::fetch, static_cast<unsigned>(dimension), false, false));
    add_field(out, DecodedField::memory_element_bytes, size.element_bytes);
    add_field(out, DecodedField::memory_elements, size.elements);
    add_field(out, DecodedField::atomic_operation, word.bits<24, 3>());
    const unsigned base = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base);
    add_gpr(out, RegisterRole::source_a, base, target.coordinates, Access::read);
    if (bindless) {
        const unsigned handle = static_cast<unsigned>(word.bits<39, 8>());
        add_source(out, SourceRole::c, handle);
        add_gpr(out, RegisterRole::source_c, handle, 1, Access::read);
    } else {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    const unsigned data = static_cast<unsigned>(word.bits<0, 8>());
    add_source(out, SourceRole::b, data);
    add_gpr(out, RegisterRole::source_b, data, 1, Access::read);
    return out;
}

// --- Sampled textures --------------------------------------------------------

// Short textures split coordinates across the Ra and Rb register groups in the
// guest order; the layer and first coordinates stay on Ra. Reserved rows
// report zero Ra lanes.
struct TexsSources {
    unsigned source_a_lanes;
    unsigned source_b_lanes;
};

// TEXS target table (TexsTarget selector at bits 53..56).
[[nodiscard]] constexpr TexsSources texs_sources(std::uint64_t selector) {
    switch (selector) {
    case 0: return {1, 0};   // 1D LodZero
    case 1: return {1, 1};   // 2D
    case 2: return {1, 1};   // 2D LodZero
    case 3: return {2, 1};   // 2D LodLevel
    case 4: return {2, 1};   // 2D DepthCompare
    case 5: return {2, 2};   // 2D LodLevelDepthCompare
    case 6: return {2, 1};   // 2D LodZeroDepthCompare
    case 7: return {2, 1};   // 2DArray
    case 8: return {2, 1};   // 2DArray LodZero
    case 9: return {2, 2};   // 2DArray LodZero DepthCompare
    case 10: return {2, 1};  // 3D
    case 11: return {2, 1};  // 3D LodZero
    case 12: return {2, 1};  // Cube
    case 13: return {2, 2};  // Cube LodLevel
    default: return {0, 0};  // 14 and 15 are undefined
    }
}

// TLDS target table (TldsTarget selector at bits 53..56).
[[nodiscard]] constexpr TexsSources tlds_sources(std::uint64_t selector) {
    switch (selector) {
    case 0x0: return {1, 0};  // 1D LodZero
    case 0x1: return {1, 1};  // 1D LodLevel
    case 0x2: return {1, 1};  // 2D LodZero
    case 0x4: return {2, 1};  // 2D LodZeroOffset
    case 0x5: return {2, 1};  // 2D LodLevel
    case 0x6: return {2, 1};  // 2D LodZeroMultisample
    case 0x7: return {2, 1};  // 3D LodZero
    case 0x8: return {1, 2};  // 2DArray LodZero (layer on Ra, coordinates on Rb)
    case 0xc: return {2, 2};  // 2D LodLevelOffset (level then packed offsets)
    default: return {0, 0};   // 0x3, 0x9..0xb and 0xd..0xf are undefined
    }
}

// TEXS/TLDS write-mask expansion: the 3-bit selector widens differently when
// the second destination register participates.
[[nodiscard]] constexpr std::uint64_t texs_component_mask(std::uint64_t selector, bool second) {
    constexpr std::uint64_t single[8] = {0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b1001, 0b1010, 0b1100};
    constexpr std::uint64_t dual[8] = {0b0111, 0b1011, 0b1101, 0b1110, 0b1111, 0, 0, 0};
    return (second ? dual : single)[selector];
}

// Short textures pack results sequentially: components 0..1 land in Dest lanes
// 0..1 and components 2..3 in Dest2 lanes 0..1. An RZ destination slot has no
// physical lanes and is not published.
void add_texs_results(DecodedInstruction& out, std::uint64_t mask, std::uint64_t destination,
    std::uint64_t second_destination) {
    const unsigned count = component_count(mask);
    const unsigned low_lanes = count < 2 ? count : 2U;
    const unsigned high_lanes = count > 2 ? count - 2 : 0U;
    if (destination != 255) {
        add_gpr(out, RegisterRole::destination, static_cast<unsigned>(destination), low_lanes, Access::write);
    }
    if (second_destination != 255 && high_lanes != 0) {
        add_gpr(out, RegisterRole::second_destination, static_cast<unsigned>(second_destination), high_lanes,
            Access::write);
    }
}

Result<DecodedInstruction> decode_texs(Pc pc, RawInstruction word, InstructionName name, bool fetch) {
    const std::uint64_t selector = word.bits<53, 4>();
    const TexsSources sources = fetch ? tlds_sources(selector) : texs_sources(selector);
    if (sources.source_a_lanes == 0) {
        return malformed(pc, name, "Short texture target selector is undefined");
    }
    const bool second = word.bits<28, 8>() != 255; // Dest2 encoded as RZ disables it.
    const std::uint64_t mask = texs_component_mask(word.bits<50, 3>(), second);
    if (mask == 0) {
        return malformed(pc, name, "Short texture write selector has no resolved components");
    }
    DecodedInstruction out = begin(pc, word, name, Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_target, selector);
    add_field(out, DecodedField::texture_component_mask, word.bits<50, 3>());
    add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    const unsigned base_a = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base_a);
    add_gpr(out, RegisterRole::source_a, base_a, sources.source_a_lanes, Access::read);
    // Rb is a formal operand slot even when the target consumes none of it.
    const unsigned base_b = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, base_b);
    if (sources.source_b_lanes != 0) {
        add_gpr(out, RegisterRole::source_b, base_b, sources.source_b_lanes, Access::read);
    }
    add_texs_results(out, mask, word.bits<0, 8>(), word.bits<28, 8>());
    return out;
}

Result<DecodedInstruction> decode_tld4s(Pc pc, RawInstruction word, InstructionName name) {
    const bool depth_compare = word.bits<50, 1>() != 0;
    const bool offset = word.bits<51, 1>() != 0;
    const bool second = word.bits<28, 8>() != 255;
    const std::uint64_t mask = texs_component_mask(4, second);
    if (mask == 0) {
        return malformed(pc, name, "Gather write selector has no resolved components");
    }
    const unsigned base_a = static_cast<unsigned>(word.bits<8, 8>());
    const unsigned a_lanes = depth_compare || offset ? 2U : 1U;
    const unsigned b_lanes =
        (!depth_compare && !offset ? 1U : 0U) + (offset ? 1U : 0U) + (depth_compare ? 1U : 0U);
    DecodedInstruction out = begin(pc, word, name, Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_compare, depth_compare ? 1 : 0);
    add_field(out, DecodedField::texture_offset, offset ? 1 : 0);
    add_field(out, DecodedField::gather_component, word.bits<52, 2>());
    add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    add_source(out, SourceRole::a, base_a);
    add_gpr(out, RegisterRole::source_a, base_a, a_lanes, Access::read);
    const unsigned base_b = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, base_b);
    if (b_lanes != 0) {
        add_gpr(out, RegisterRole::source_b, base_b, b_lanes, Access::read);
    }
    add_texs_results(out, mask, word.bits<0, 8>(), word.bits<28, 8>());
    return out;
}

// Long-form texture dimension handling: TexDim folds into the shared
// TextureDimension plus an array flag, and the coordinate count follows.
struct TextureShapeAccess {
    TextureDimension dimension;
    bool array;
    unsigned coordinates;
};
[[nodiscard]] constexpr TextureShapeAccess texture_shape(std::uint64_t selector) {
    switch (selector) {
    case 0: return {TextureDimension::d1, false, 1};
    case 1: return {TextureDimension::d1, true, 2};
    case 2: return {TextureDimension::d2, false, 2};
    case 3: return {TextureDimension::d2, true, 3};
    case 4: return {TextureDimension::d3, false, 3};
    case 5: return {TextureDimension::d3, true, 4};
    case 6: return {TextureDimension::cube, false, 3};
    default: return {TextureDimension::cube, true, 4}; // selector 7
    }
}

// Coordinates from SrcA; auxiliary operands (bindless handle, level/bias,
// offsets, depth-compare reference, sample index) consume consecutive
// registers from SrcB in guest order. Rb stays a formal operand slot
// even when the encoding consumes none of it, so emitters can stage cursors.
void add_texture_operands(DecodedInstruction& out, RawInstruction word, TextureShapeAccess shape,
    unsigned extra_a_lanes, unsigned b_lanes) {
    const unsigned base_a = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base_a);
    add_gpr(out, RegisterRole::source_a, base_a, shape.coordinates + extra_a_lanes, Access::read);
    const unsigned base_b = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, base_b);
    if (b_lanes != 0) {
        add_gpr(out, RegisterRole::source_b, base_b, b_lanes, Access::read);
    }
}

Result<DecodedInstruction> decode_tex(Pc pc, RawInstruction word, bool bindless) {
    const InstructionName name = bindless ? InstructionName::tex_b : InstructionName::tex;
    const std::uint64_t selector = word.bits<28, 3>();
    const bool depth_compare = word.bits<50, 1>() != 0;
    const bool offset = bindless ? word.bits<36, 1>() != 0 : word.bits<54, 1>() != 0;
    const std::uint64_t lod = bindless ? word.bits<37, 3>() : word.bits<55, 3>();
    // LOD selectors: 0 = none (implicit), 1 = zero, 2 = bias, 3 = level,
    // 6 = bias alternate, 7 = level alternate; 4 and 5 are undefined.
    if (lod == 4 || lod == 5) {
        return malformed(pc, name, "Texture LOD selector is undefined");
    }
    const TextureShapeAccess shape = texture_shape(selector);
    const unsigned b_lanes =
        (bindless ? 1U : 0U) + (lod > 1 ? 1U : 0U) + (offset ? 1U : 0U) + (depth_compare ? 1U : 0U);
    const std::uint64_t mask = word.bits<31, 4>();
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension, selector);
    add_field(out, DecodedField::texture_mode, lod);
    add_field(out, DecodedField::texture_compare, depth_compare ? 1 : 0);
    add_field(out, DecodedField::texture_ndv, word.bits<35, 1>() != 0 ? 1 : 0);
    add_field(out, DecodedField::texture_lod_clamp,
        (bindless ? word.bits<40, 1>() : word.bits<58, 1>()) != 0 ? 1 : 0);
    add_field(out, DecodedField::texture_component_mask, mask);
    if (!bindless) {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    if (offset) {
        add_field(out, DecodedField::texture_offset, 1);
    } else {
        add_field(out, DecodedField::texture_offset, 0);
    }
    add_texture_operands(out, word, shape, 0U, b_lanes);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), component_count(mask),
        Access::write);
    add_predicate(out, RegisterRole::predicate_destination, static_cast<unsigned>(word.bits<51, 3>()), Access::write);
    return out;
}

// TLD (long fetch): texture_mode carries the raw LL flag (0 or 1), the
// multisample flag selects fetches, and the CL bit is an explicit boolean.
Result<DecodedInstruction> decode_tld(Pc pc, RawInstruction word, bool bindless) {
    const InstructionName name = bindless ? InstructionName::tld_b : InstructionName::tld;
    const std::uint64_t selector = word.bits<28, 3>();
    const bool level = word.bits<55, 1>() != 0;
    const bool offset = word.bits<35, 1>() != 0;
    const bool multisample = word.bits<50, 1>() != 0;
    const TextureShapeAccess shape = texture_shape(selector);
    const unsigned b_lanes =
        (bindless ? 1U : 0U) + (level ? 1U : 0U) + (offset ? 1U : 0U) + (multisample ? 1U : 0U);
    const std::uint64_t mask = word.bits<31, 4>();
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension, selector);
    add_field(out, DecodedField::texture_mode, level ? 1 : 0);
    add_field(out, DecodedField::texture_multisample, multisample ? 1 : 0);
    add_field(out, DecodedField::texture_clamp, word.bits<54, 1>() != 0 ? 1 : 0);
    add_field(out, DecodedField::texture_component_mask, mask);
    if (!bindless) {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    add_field(out, DecodedField::texture_offset, offset ? 1 : 0);
    add_texture_operands(out, word, shape, 0U, b_lanes);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), component_count(mask),
        Access::write);
    add_predicate(out, RegisterRole::predicate_destination, static_cast<unsigned>(word.bits<51, 3>()), Access::write);
    return out;
}

// TLD4 (long gather): coordinates plus the gathered component, optional
// offsets and depth-compare reference.
Result<DecodedInstruction> decode_tld4(Pc pc, RawInstruction word, bool bindless) {
    const InstructionName name = bindless ? InstructionName::tld4_b : InstructionName::tld4;
    const std::uint64_t selector = word.bits<28, 3>();
    const std::uint64_t offset_selector = bindless ? word.bits<36, 2>() : word.bits<54, 2>();
    if (offset_selector > 2) {
        return malformed(pc, name, "Gather offset selector is undefined");
    }
    const bool depth_compare = word.bits<50, 1>() != 0;
    const TextureShapeAccess shape = texture_shape(selector);
    const unsigned offset_lanes = offset_selector == 2 ? 2U : offset_selector == 1 ? 1U : 0U;
    const unsigned b_lanes = (bindless ? 1U : 0U) + offset_lanes + (depth_compare ? 1U : 0U);
    const std::uint64_t mask = word.bits<31, 4>();
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension, selector);
    add_field(out, DecodedField::texture_compare, depth_compare ? 1 : 0);
    add_field(out, DecodedField::texture_ndv, word.bits<35, 1>() != 0 ? 1 : 0);
    add_field(out, DecodedField::texture_lod_clamp,
        (bindless ? word.bits<40, 1>() : word.bits<58, 1>()) != 0 ? 1 : 0);
    add_field(out, DecodedField::texture_component_mask, mask);
    add_field(out, DecodedField::gather_component, bindless ? word.bits<38, 2>() : word.bits<56, 2>());
    add_field(out, DecodedField::texture_offset, offset_selector);
    if (!bindless) {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    add_texture_operands(out, word, shape, 0U, b_lanes);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), component_count(mask),
        Access::write);
    add_predicate(out, RegisterRole::predicate_destination, static_cast<unsigned>(word.bits<51, 3>()), Access::write);
    return out;
}

// TMML: query the computed LOD pair. Rb stays a formal operand slot; the
// bindless form consumes it as the handle register.
Result<DecodedInstruction> decode_tmml(Pc pc, RawInstruction word, bool bindless) {
    const InstructionName name = bindless ? InstructionName::tmml_b : InstructionName::tmml;
    const TextureShapeAccess shape = texture_shape(word.bits<28, 3>());
    const std::uint64_t mask = word.bits<31, 4>();
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension, word.bits<28, 3>());
    add_field(out, DecodedField::texture_ndv, word.bits<35, 1>() != 0 ? 1 : 0);
    add_field(out, DecodedField::texture_component_mask, mask);
    if (!bindless) {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    const unsigned base_a = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base_a);
    add_gpr(out, RegisterRole::source_a, base_a, shape.coordinates, Access::read);
    const unsigned base_b = static_cast<unsigned>(word.bits<20, 8>());
    add_source(out, SourceRole::b, base_b);
    if (bindless) {
        add_gpr(out, RegisterRole::source_b, base_b, 1, Access::read);
    }
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), component_count(mask),
        Access::write);
    return out;
}

// TXQ: descriptor queries. The raw query selector is preserved; unsupported
// selectors fail at emission with a structured diagnostic.
Result<DecodedInstruction> decode_txq(Pc pc, RawInstruction word, bool bindless) {
    const InstructionName name = bindless ? InstructionName::txq_b : InstructionName::txq;
    const std::uint64_t mask = word.bits<31, 4>();
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_query, word.bits<22, 6>());
    add_field(out, DecodedField::texture_component_mask, mask);
    if (!bindless) {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    const unsigned base_a = static_cast<unsigned>(word.bits<8, 8>());
    add_source(out, SourceRole::a, base_a);
    // Bindless handles occupy the first source register, before the LOD input.
    add_gpr(out, RegisterRole::source_a, base_a, bindless ? 2U : 1U, Access::read);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), component_count(mask),
        Access::write);
    return out;
}

// TXD: explicit-gradient sample. The packed parameter register consumed after
// the coordinates carries the array layer and offsets; gradients consume two
// words per coordinate axis from the second source group.
Result<DecodedInstruction> decode_txd(Pc pc, RawInstruction word, bool bindless) {
    const InstructionName name = bindless ? InstructionName::txd_b : InstructionName::txd;
    const TextureShapeAccess shape = texture_shape(word.bits<28, 3>());
    const bool offset = word.bits<35, 1>() != 0;
    const unsigned a_lanes = (bindless ? 1U : 0U) + shape.coordinates + 1U;
    const unsigned b_lanes = 2U * shape.coordinates;
    const std::uint64_t mask = word.bits<31, 4>();
    DecodedInstruction out =
        begin(pc, word, name, bindless ? Encoding::texture_bindless : Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_dimension, word.bits<28, 3>());
    add_field(out, DecodedField::texture_lod_clamp, word.bits<50, 1>() != 0 ? 1 : 0);
    add_field(out, DecodedField::texture_component_mask, mask);
    if (!bindless) {
        add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    }
    add_field(out, DecodedField::texture_offset, offset ? 1 : 0);
    add_texture_operands(out, word, shape, a_lanes - shape.coordinates, b_lanes);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), component_count(mask),
        Access::write);
    add_predicate(out, RegisterRole::predicate_destination, static_cast<unsigned>(word.bits<51, 3>()), Access::write);
    return out;
}

// TXA: texture-with-animation sample with a bound handle. Its type comes from
// the descriptor, so no dimension selector exists in the encoding.
Result<DecodedInstruction> decode_txa(Pc pc, RawInstruction word) {
    const TextureShapeAccess shape = texture_shape(word.bits<28, 3>());
    DecodedInstruction out = begin(pc, word, InstructionName::txa, Encoding::texture_bound, true);
    add_field(out, DecodedField::texture_component_mask, word.bits<31, 4>());
    add_field(out, DecodedField::texture_handle_byte_offset, word.bits<36, 13>() * 4);
    add_texture_operands(out, word, shape, 0U, 0U);
    add_gpr(out, RegisterRole::destination, static_cast<unsigned>(word.bits<0, 8>()), 1, Access::write);
    return out;
}

// The SuatomCas prefix also covers Suatom, SuatomB2, SuatomCasB and Red,
// so the family expands into six disjoint rules that exclude those encodings
// while covering every remaining selector combination.
Result<DecodedInstruction> decode_suatom_cas(Pc pc, RawInstruction word) {
    return decode_surface_atomic(pc, word, InstructionName::suatom_cas, true, false);
}

std::span<const DecodeRule> resource_decode_rules_impl() {
    using N = InstructionName;
    using P = Pc;
    using R = RawInstruction;
    static const std::array<DecodeRule, 73> rules = {
        // Flow control.
        rule("111000100100xx", decode_bra),
        rule("111000100101xx", [](P pc, R w) { return decode_register_branch(pc, w, N::brx, false); }),
        rule("111000100001xx", [](P pc, R w) { return decode_register_branch(pc, w, N::jmp, true); }),
        rule("111000100000xx", [](P pc, R w) { return decode_register_branch(pc, w, N::jmx, true); }),
        rule("111000100110xx", [](P pc, R w) { return decode_push(pc, w, N::cal); }),
        rule("111000101001xx", [](P pc, R w) { return decode_push(pc, w, N::ssy); }),
        rule("111000101010xx", [](P pc, R w) { return decode_push(pc, w, N::pbk); }),
        rule("111000101011xx", [](P pc, R w) { return decode_push(pc, w, N::pcnt); }),
        rule("111000110100xx", [](P pc, R w) { return decode_simple_flow(pc, w, N::brk); }),
        rule("111000110101xx", [](P pc, R w) { return decode_simple_flow(pc, w, N::cont); }),
        rule("111000110000xx", [](P pc, R w) { return decode_simple_flow(pc, w, N::exit); }),
        rule("111000110011xx", [](P pc, R w) { return decode_simple_flow(pc, w, N::kil); }),
        rule("111000110010xx", [](P pc, R w) { return decode_simple_flow(pc, w, N::ret); }),
        rule("1111000011111x", [](P pc, R w) { return decode_simple_flow(pc, w, N::sync); }),
        rule("0101000010110x", decode_nop),
        rule("1111000011110x", decode_depbar),
        rule("0101000011100x", decode_votevtg),
        rule("1111000010101x", decode_bar),
        rule("1110111110011x", decode_membar),
        // Attribute load/store and interpolation.
        rule("1110111111011x", decode_ald),
        rule("1110111111110x", decode_ast),
        rule("11100000xxxxxx", decode_ipa),
        // Memory.
        rule("1110111110010x", decode_ldc),
        rule("1110111101000x", [](P pc, R w) {
            return decode_narrow_memory(pc, w, N::ldl, AddressSpace::local, false); }),
        rule("1110111101001x", [](P pc, R w) {
            return decode_narrow_memory(pc, w, N::lds, AddressSpace::shared, false); }),
        rule("1110111101010x", [](P pc, R w) {
            return decode_narrow_memory(pc, w, N::stl, AddressSpace::local, true); }),
        rule("1110111101011x", [](P pc, R w) {
            return decode_narrow_memory(pc, w, N::sts, AddressSpace::shared, true); }),
        rule("1110111011010x", [](P pc, R w) { return decode_storage(false, pc, w); }),
        rule("1110111011011x", [](P pc, R w) { return decode_storage(true, pc, w); }),
        rule("11101101xxxxxx", decode_atom),
        rule("111011101111xx", decode_atom_cas),
        rule("11101100xxxxxx", [](P pc, R w) { return decode_shared_atomic(pc, w, false); }),
        rule("111011100xxxxx", [](P pc, R w) { return decode_shared_atomic(pc, w, true); }),
        rule("1110101111111x", decode_red),
        // Surface load/store.
        rule("1110101100001x", [](P pc, R w) {
            return decode_surface(pc, w, N::suld, false, false, true, w.bits<20, 3>(), false); }),
        rule("1110101100000x", [](P pc, R w) {
            return decode_surface(pc, w, N::suld_b, false, true, false, w.bits<20, 4>(), false); }),
        rule("1110101100011x", [](P pc, R w) {
            return decode_surface(pc, w, N::suld_d, false, false, true, w.bits<20, 3>(), true); }),
        rule("1110101100010x", [](P pc, R w) {
            return decode_surface(pc, w, N::suld_db, false, true, true, w.bits<20, 3>(), true); }),
        rule("1110101100101x", [](P pc, R w) {
            return decode_surface(pc, w, N::sust, true, false, false, w.bits<20, 4>(), false); }),
        rule("1110101100100x", [](P pc, R w) {
            return decode_surface(pc, w, N::sust_b, true, true, false, w.bits<20, 4>(), false); }),
        rule("1110101100111x", [](P pc, R w) {
            return decode_surface(pc, w, N::sust_d, true, false, true, w.bits<20, 3>(), true); }),
        rule("1110101100110x", [](P pc, R w) {
            return decode_surface(pc, w, N::sust_db, true, true, true, w.bits<20, 3>(), false); }),
        // Surface atomics and reductions.
        rule("11101010x0xxxx", [](P pc, R w) {
            return decode_surface_atomic(pc, w, N::suatom, false, false); }),
        rule("1110101001110x", [](P pc, R w) {
            return decode_surface_atomic(pc, w, N::suatom_b, false, true); }),
        rule("1110101110xxxx", [](P pc, R w) {
            return decode_surface_atomic(pc, w, N::suatom_b2, false, true); }),
        rule("11101011110xxx", decode_suatom_cas),
        rule("111010111110xx", decode_suatom_cas),
        rule("1110101111110x", decode_suatom_cas),
        rule("11101010111xxx", decode_suatom_cas),
        rule("111010101100xx", decode_suatom_cas),
        rule("1110101011011x", decode_suatom_cas),
        rule("1110101011010x", [](P pc, R w) {
            return decode_surface_atomic(pc, w, N::suatom_cas_b, true, true); }),
        rule("1110101101011x", [](P pc, R w) { return decode_surface_reduction(pc, w, false); }),
        rule("1110101101010x", [](P pc, R w) { return decode_surface_reduction(pc, w, true); }),
        // Sampled textures, long form.
        rule("11000xxxxxxxxx", [](P pc, R w) { return decode_tex(pc, w, false); }),
        rule("1101111010xxxx", [](P pc, R w) { return decode_tex(pc, w, true); }),
        rule("11011100xxxxxx", [](P pc, R w) { return decode_tld(pc, w, false); }),
        rule("11011101xxxxxx", [](P pc, R w) { return decode_tld(pc, w, true); }),
        rule("110010xxxxxxxx", [](P pc, R w) { return decode_tld4(pc, w, false); }),
        rule("1101111011xxxx", [](P pc, R w) { return decode_tld4(pc, w, true); }),
        rule("1101111101011x", [](P pc, R w) { return decode_tmml(pc, w, false); }),
        rule("1101111101100x", [](P pc, R w) { return decode_tmml(pc, w, true); }),
        rule("1101111101001x", [](P pc, R w) { return decode_txq(pc, w, false); }),
        rule("1101111101010x", [](P pc, R w) { return decode_txq(pc, w, true); }),
        rule("1101111000xxxx", [](P pc, R w) { return decode_txd(pc, w, false); }),
        rule("1101111001xxxx", [](P pc, R w) { return decode_txd(pc, w, true); }),
        rule("1101111101000x", decode_txa),
        // Sampled textures, short form.
        rule("1101100xxxxxxx", [](P pc, R w) { return decode_texs(pc, w, N::texs, false); }),
        rule("1101000xxxxxxx", [](P pc, R w) { return decode_texs(pc, w, N::texs_f16, false); }),
        rule("1101101xxxxxxx", [](P pc, R w) { return decode_texs(pc, w, N::tlds, true); }),
        rule("1101001xxxxxxx", [](P pc, R w) { return decode_texs(pc, w, N::tlds_f16, true); }),
        rule("1101111100xxxx", [](P pc, R w) { return decode_tld4s(pc, w, N::tld4s); }),
        rule("1101111110xxxx", [](P pc, R w) { return decode_tld4s(pc, w, N::tld4s_f16); }),
    };
    return rules;
}

// The dispatch prefix width and example patterns are compile-time checked so
// field extraction and rule matching cannot silently drift.
static_assert(rule("11000xxxxxxxxx", nullptr).mask == 0xf800'0000'0000'0000ULL);
static_assert(rule("11000xxxxxxxxx", nullptr).value == 0xc000'0000'0000'0000ULL);
static_assert(rule("111000100100xx", nullptr).mask == 0xfff0'0000'0000'0000ULL);
} // namespace

std::span<const DecodeRule> resource_decode_rules() {
    return resource_decode_rules_impl();
}

static_assert(RawInstruction{0x1234'5678'9abc'def0ULL}.bits<20, 24>() == 0x6789'abULL);
static_assert(RawInstruction{0xf'ffff'8000'00ULL}.signed_bits<20, 24>() == -8);
} // namespace armigo::shader::pipeline
