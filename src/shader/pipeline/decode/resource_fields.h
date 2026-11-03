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

#pragma once
#include "pipeline/decoder.h"

#include <cstdint>

// The nested namespace is named resource_decode because the frozen pipeline
// contract already owns the function name decode in armigo::shader::pipeline.
namespace armigo::shader::pipeline::resource_decode {
// Shared accessors and documented field layouts for the resource/flow decode
// rules published by resource_decode_rules(). Emitter lanes must consume these
// helpers instead of re-deriving guest bit layouts from the raw word.
// Decoder rules use a 14-bit prefix dispatch width, so every pattern below
// covers bits 63..50 of the word.

inline constexpr unsigned decode_prefix_bits = 14;

// Builds one rule from a 14-character pattern of '1', '0' and 'x' (most
// significant bit first).
[[nodiscard]] constexpr DecodeRule rule(const char (&pattern)[15], DecodeInstruction decode) {
    std::uint64_t mask = 0;
    std::uint64_t value = 0;
    for (unsigned index = 0; index < decode_prefix_bits; ++index) {
        const unsigned bit = 63 - index;
        if (pattern[index] == '1') {
            mask |= std::uint64_t{1} << bit;
            value |= std::uint64_t{1} << bit;
        } else if (pattern[index] == '0') {
            mask |= std::uint64_t{1} << bit;
        }
    }
    return {mask, value, decode};
}

// Instruction predicate: 3-bit index at bit 16, inversion at bit 19. NoPred
// encodings report enabled == false regardless of the encoded bits.
[[nodiscard]] constexpr InstructionPredicate instruction_predicate(RawInstruction word, bool enabled) {
    return {static_cast<std::uint8_t>(word.bits<16, 3>()), word.bits<19, 1>() != 0, enabled};
}

// --- Branch fields -----------------------------------------------------------
// condition: raw 5-bit condition-code selector (bits 0..4); value 15 selects
// "always". Condition-code semantics belong to the flow lane.
[[nodiscard]] constexpr std::uint64_t condition_selector(RawInstruction word) { return word.bits<0, 5>(); }

// Branch flag bits (documented per family):
//   BRA/BRX/JMP/JMX/SSY/PBK/PCNT/CAL: bit 5 = cache/alignment hint (Ca),
//   BRA/JMP: bit 6 = limit (Lmt), bit 7 = uniform (U); CAL: bit 6 = increment.
//   EXIT: bit 5 = keep reference count. Consumers read them from the raw word
//   directly; the bit positions are the contract here.
// branch_displacement field value: 64-bit two's-complement byte displacement.
// Relative branches (BRA/BRX/CAL/SSY/PBK/PCNT): target = pc + displacement;
// the displacement already includes the +8 scheduling base,
// so the target is the target instruction word's own pc. BRX/JMX additionally
// add the unsigned source-register value to the same base.
// JMP/JMX: the raw 32-bit immediate (bits 20..51); the family treats it as
// absolute-with-register rather than pc-relative (branch_absolute field is set).
[[nodiscard]] constexpr std::uint64_t branch_displacement_value(std::int64_t bytes) {
    return static_cast<std::uint64_t>(bytes);
}
[[nodiscard]] constexpr std::int64_t relative_displacement(RawInstruction word) {
    return word.signed_bits<20, 24>() + 8;
}

// --- Texture field layouts ---------------------------------------------------
// Sampled-texture families (TEX, TEXS, TLD, TLDS, TLD4, TLD4S, TMML, TXQ, TXD
// and their bindless forms) publish exactly the frozen DecodedField encodings
// documented in pipeline/decoder.h: texture_dimension carries the raw TexDim
// selector (dimension * 2 + array), texture_mode carries the raw TEX Lod
// (0,1,2,3,6,7) or TLD Lod (0,1) selector, texture_target/texture_query keep
// their ISA selector values, compare/multisample/lod_clamp/ndv/clamp are
// explicit boolean fields, and TEXS/TLDS masks keep their encoded 3-bit
// selector. The accessors below serve the surface (image) families only,
// whose dimension is the SuDim selector rather than TexDim:
// texture_dimension bits: [2:0] TextureDimension, [3] array.
[[nodiscard]] constexpr std::uint64_t texture_dimension_value(TextureDimension dimension, bool array,
                                                             bool multisample, bool depth_compare, bool ndv,
                                                             bool level_clamp) {
    return static_cast<std::uint64_t>(dimension) | (array ? 1U << 3 : 0U) | (multisample ? 1U << 4 : 0U) |
           (depth_compare ? 1U << 5 : 0U) | (ndv ? 1U << 6 : 0U) | (level_clamp ? 1U << 7 : 0U);
}
[[nodiscard]] constexpr TextureDimension texture_dimension_of(std::uint64_t value) {
    return static_cast<TextureDimension>(value & 7);
}

// Surface texture_mode bits: [7:0] TextureMode, [15:8] raw SuDim selector.
[[nodiscard]] constexpr std::uint64_t texture_mode_value(TextureMode mode, unsigned raw_selector, bool level_zero,
                                                        bool lod_operand) {
    return static_cast<std::uint64_t>(mode) | (std::uint64_t{raw_selector} << 8) | (level_zero ? 1U << 16 : 0U) |
           (lod_operand ? 1U << 17 : 0U);
}
[[nodiscard]] constexpr TextureMode texture_mode_of(std::uint64_t value) {
    return static_cast<TextureMode>(value & 0xff);
}

// texture_offset follows decoder.h: TLD4 carries TexOffset (0 = none,
// 1 = aoffi, 2 = ptp); TEX, TLD, TXD and TLD4S carry a plain boolean.
// texture_component_mask: long forms keep their four component bits,
// bit 0 = x .. bit 3 = w; compact families keep the encoded selector.
// gather_component: raw 2-bit component selector (0 = x .. 3 = w).
// texture_cbuf_slot is never produced by decoding: Maxwell texture handles
// carry only a word offset into the default handle buffer, whose slot must be
// proven through the texture_handle_cbuf fact by the consumer.

// --- Memory sizes ------------------------------------------------------------
// memory_element_bytes is the scalar element width in bytes (1, 2 or 4);
// memory_elements is the vector element count (1, 2 or 4). Total access size
// is their product. Narrow elements use word32 containers with the
// sign/zero interpretation carried by the IR NumericMode, not by the decoder.
struct MemorySize {
    std::uint64_t element_bytes;
    std::uint64_t elements;
};
[[nodiscard]] constexpr bool valid(const MemorySize& size) {
    return (size.element_bytes == 1 || size.element_bytes == 2 || size.element_bytes == 4) &&
           (size.elements >= 1 && size.elements <= 4);
}

// LS load/store size selector (LDL/LDS/STL/STS and LDC use 0..5, LDG/STG
// allow 0..7): 0 = u8, 1 = s8, 2 = u16, 3 = s16, 4 = 32-bit, 5 = 64-bit,
// 6 = 128-bit, 7 = 128-bit unordered (LDG/STG only).
[[nodiscard]] constexpr MemorySize ls_size(std::uint64_t selector) {
    switch (selector) {
    case 0: case 1: return {1, 1};
    case 2: case 3: return {2, 1};
    case 4: return {4, 1};
    case 5: return {4, 2};
    default: return {4, 4};
    }
}

// Attribute/atomic size selectors (ALD/AST AlSize: element count is +1).
[[nodiscard]] constexpr MemorySize attribute_size(std::uint64_t selector) { return {4, selector + 1}; }
[[nodiscard]] constexpr MemorySize atomic_size(std::uint64_t selector) {
    switch (selector) {
    case 0: case 1: case 3: case 6: return {4, 1}; // u32, s32, f32, sd32
    case 2: case 5: case 7: return {4, 2};         // u64, s64, sd64
    default: return {2, 2};                        // f16x2
    }
}

// --- Per-family raw selectors without a frozen DecodedField key --------------
// These stay encoded-only; the owning family documents their meaning.
[[nodiscard]] constexpr std::uint64_t ldc_cbuf_slot(RawInstruction word) { return word.bits<36, 5>(); }
// LDC addressing mode: 0 = fixed slot, 1 = il, 2 = is, 3 = isl (1/2/3 derive
// the slot from the upper half of the source register).
[[nodiscard]] constexpr std::uint64_t ldc_address_mode(RawInstruction word) { return word.bits<44, 2>(); }
[[nodiscard]] constexpr std::uint64_t attribute_per_patch(RawInstruction word) { return word.bits<31, 1>(); }
// ALD only: bit 32 selects the output attribute space.
[[nodiscard]] constexpr bool attribute_output_space(RawInstruction word) { return word.bits<32, 1>() != 0; }
// ALD/AST physical addressing: P == 0, offset == 0, and a nonzero base register
// (SrcA for ALD, the address register at bit 8 for AST).
[[nodiscard]] constexpr bool attribute_physical(RawInstruction word) {
    return word.bits<31, 1>() == 0 && word.bits<20, 11>() == 0 && word.bits<8, 8>() != 0;
}
// IPA modifiers: raw operation selector (0 pass, 1 multiply, 2 constant, 3 sc),
// sample location selector, and saturation.
[[nodiscard]] constexpr bool ipa_saturate(RawInstruction word) { return word.bits<55, 1>() != 0; }
// DEPBAR: bit 29 = less-or-equal comparison, bits 20..25 = pending count,
// bits 0..5 = packed barrier/selector bits.
[[nodiscard]] constexpr bool depbar_le(RawInstruction word) { return word.bits<29, 1>() != 0; }
[[nodiscard]] constexpr std::uint64_t depbar_pending_count(RawInstruction word) { return word.bits<20, 6>(); }
// BAR fixed-function flags: bit 44 = a-fix, bit 43 = b-fix; the operation
// selector is decoded into the barrier_kind field and the unit into barrier_id.
[[nodiscard]] constexpr bool bar_a_fix(RawInstruction word) { return word.bits<44, 1>() != 0; }
[[nodiscard]] constexpr bool bar_b_fix(RawInstruction word) { return word.bits<43, 1>() != 0; }
// MEMBAR: raw scope selector in the memory_scope field (0 = cta, 1 = gl,
// 2 = sys, 3 = vc) and the invalidate selector at bits 0..1.
[[nodiscard]] constexpr std::uint64_t membar_invalidate(RawInstruction word) { return word.bits<0, 2>(); }
// SULD/SUST/SURED non-_D variants: bit 23 selects byte addressing; the SUATOM
// family carries its byte-address flag at bit 28 instead.
[[nodiscard]] constexpr bool surface_byte_address(RawInstruction word) { return word.bits<23, 1>() != 0; }
[[nodiscard]] constexpr bool suatom_byte_address(RawInstruction word) { return word.bits<28, 1>() != 0; }
// Cache operation selectors are kept raw (family-documented): LD/ST use bits
// 56..57, LDL/STL bits 44..45, LDG/STG bits 46..47, SULD/SUST bits 24..25.
[[nodiscard]] constexpr std::uint64_t ld_cache_op(RawInstruction word) { return word.bits<56, 2>(); }
[[nodiscard]] constexpr std::uint64_t ldl_cache_op(RawInstruction word) { return word.bits<44, 2>(); }
[[nodiscard]] constexpr std::uint64_t ldg_cache_op(RawInstruction word) { return word.bits<46, 2>(); }
[[nodiscard]] constexpr std::uint64_t suld_cache_op(RawInstruction word) { return word.bits<24, 2>(); }
// Extended 40-bit address flags: LDG/STG/ATOM/RED use bit 45, LD/ST bit 52.
[[nodiscard]] constexpr bool ldg_extended_address(RawInstruction word) { return word.bits<45, 1>() != 0; }
[[nodiscard]] constexpr bool ld_st_extended_address(RawInstruction word) { return word.bits<52, 1>() != 0; }
// Atomic size/operation selectors per family: ATOM size at 49..51 with the
// operation at 52..55, RED size at 20..22 with the operation at 23..25,
// shared ATOMS size at 28..29 with the operation at 52..55, ATOMCAS selector
// at 50..51 and shared ATOMSCAS selector at 28..29.
[[nodiscard]] constexpr std::uint64_t atom_size_selector(RawInstruction word) { return word.bits<49, 3>(); }
[[nodiscard]] constexpr std::uint64_t red_size_selector(RawInstruction word) { return word.bits<20, 3>(); }
[[nodiscard]] constexpr std::uint64_t atoms_size_selector(RawInstruction word) { return word.bits<28, 2>(); }
[[nodiscard]] constexpr std::uint64_t atom_cas_selector(RawInstruction word) { return word.bits<50, 2>(); }
[[nodiscard]] constexpr std::uint64_t atoms_cas_selector(RawInstruction word) { return word.bits<28, 2>(); }
} // namespace armigo::shader::pipeline::resource_decode
