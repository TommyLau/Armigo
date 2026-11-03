// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Executes relocated instructions on the host CPU. Each scenario takes an
// instruction encoded at an original address inside a private mapping,
// relocates it at least 4 MiB away (beyond every single-word reach except B,
// BL and ADRP), surrounds it with small independently written functions, and
// compares what the CPU computes with the original instruction's meaning.
// Scaffolding words were taken from the system assembler's output.

#include <armigo/aarch64/relocation.h>

#include "test_support.h"

#include <libkern/OSCacheControl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace {

using namespace armigo::aarch64;

constexpr std::uint32_t ret = 0xD65F03C0u;
constexpr std::uint32_t cmp_x0_x1 = 0xEB01001Fu;
constexpr std::uint32_t push_frame = 0xA9BF7BFDu;  // stp x29, x30, [sp, #-16]!
constexpr std::uint32_t pop_frame = 0xA8C17BFDu;   // ldp x29, x30, [sp], #16
constexpr std::uint32_t add_x0_1 = 0x91000400u;
constexpr std::uint32_t fmov_x0_d0 = 0x9E660000u;
constexpr std::uint32_t umov_x1_v0_d1 = 0x4E183C01u;
constexpr std::uint32_t eor_x0_x0_x1 = 0xCA010000u;
constexpr std::uint32_t movz_x0(std::uint32_t value) { return 0xD2800000u | (value << 5); }

constexpr std::size_t mapping_size = 8u << 20;
constexpr std::size_t far_offset = 4u << 20;

using Function = std::uint64_t (*)(std::uint64_t, std::uint64_t);

class Mapping {
public:
    Mapping() {
        memory_ = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (memory_ == MAP_FAILED) {
            memory_ = nullptr;
        }
    }
    ~Mapping() {
        if (memory_) {
            munmap(memory_, mapping_size);
        }
    }
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;

    bool valid() const { return memory_ != nullptr; }
    std::uint64_t address(std::size_t offset) const {
        return reinterpret_cast<std::uintptr_t>(memory_) + offset;
    }
    void put(std::size_t offset, std::uint32_t word) {
        std::memcpy(static_cast<char*>(memory_) + offset, &word, sizeof(word));
    }
    void put64(std::size_t offset, std::uint64_t value) {
        std::memcpy(static_cast<char*>(memory_) + offset, &value, sizeof(value));
    }
    // Writes a relocated sequence at offset and returns its length in bytes.
    std::size_t put(std::size_t offset, const RelocatedSequence& sequence) {
        for (std::size_t index = 0; index < sequence.count; ++index) {
            put(offset + 4 * index, sequence.words[index]);
        }
        return 4 * sequence.count;
    }
    bool seal() {
        if (mprotect(memory_, mapping_size, PROT_READ | PROT_EXEC) != 0) {
            return false;
        }
        sys_icache_invalidate(memory_, mapping_size);
        return true;
    }
    Function function(std::size_t offset) const {
        return reinterpret_cast<Function>(static_cast<std::uintptr_t>(address(offset)));
    }

private:
    void* memory_ = nullptr;
};

// Offsets: originals and targets live below far_offset, relocated code above.
struct Layout {
    std::size_t original;
    std::size_t target;
    std::size_t relocated;
};

constexpr Layout adr_case{0x10000, 0x10007, far_offset + 0x0000};
constexpr Layout adrp_case{0x11000, 0x32345, far_offset + 0x0100};
constexpr Layout ldr_x_scaled{0x20000, 0x20008, far_offset + 0x0200};
constexpr Layout ldr_x_unscaled{0x20010, 0x2001C, far_offset + 0x0300};
constexpr Layout ldrsw_case{0x20020, 0x20030, far_offset + 0x0400};
constexpr Layout ldr_w_case{0x20040, 0x20034, far_offset + 0x0500};
constexpr Layout ldr_q_case{0x20060, 0x20078, far_offset + 0x0600};
constexpr Layout cbz_case{0x30000, 0x30100, far_offset + 0x0700};
constexpr Layout tbnz_case{0x31000, 0x31100, far_offset + 0x0800};
constexpr Layout bcond_case{0x32000, 0x32100, far_offset + 0x0900};
constexpr Layout bl_preserve_case{0x40000, 0x40100, far_offset + 0x0A00};
constexpr Layout bl_plain_case{0x41000, 0x41100, far_offset + 0x0B00};
constexpr Layout blr_preserve_case{0x42000, 0x42100, far_offset + 0x0C00};

constexpr std::uint64_t literal_x = 0x0123456789ABCDEFull;
constexpr std::uint64_t literal_x2 = 0xFEDCBA9876543210ull;

RelocatedSequence relocate(Mapping& map, const Layout& layout, std::uint32_t word,
                           const SequenceOptions& options = {}) {
    const auto sequence = relocate_instruction_sequence(word, map.address(layout.original),
                                                        map.address(layout.relocated), options);
    CHECK(sequence.ok());
    return sequence;
}

std::uint32_t original_word(Encoded encoded) {
    CHECK(encoded.ok());
    return encoded.word;
}

// Leaf: <relocated adr/adrp/ldr x0> ; ret
void leaf(Mapping& map, const Layout& layout, std::uint32_t word, const SequenceOptions& options = {},
          std::initializer_list<std::uint32_t> tail = {}) {
    const auto sequence = relocate(map, layout, word, options);
    std::size_t at = layout.relocated + map.put(layout.relocated, sequence);
    for (const auto extra : tail) {
        map.put(at, extra);
        at += 4;
    }
    map.put(at, ret);
}

// Taken target returns 1, fall-through returns 2.
void two_way(Mapping& map, const Layout& layout, std::uint32_t word, std::size_t entry_prefix = 0) {
    const auto sequence = relocate(map, layout, word);
    CHECK(sequence.count == 2);
    const std::size_t end = layout.relocated + map.put(layout.relocated, sequence);
    map.put(end, movz_x0(2));
    map.put(end + 4, ret);
    map.put(layout.target, movz_x0(1));
    map.put(layout.target + 4, ret);
    if (entry_prefix) {
        map.put(layout.relocated - 4, entry_prefix);
    }
}

void build(Mapping& map) {
    const auto addr = [&](std::size_t offset) { return map.address(offset); };

    leaf(map, adr_case, original_word(encode_adr(0, addr(adr_case.original), addr(adr_case.target))));
    leaf(map, adrp_case, original_word(encode_adrp(0, addr(adrp_case.original), addr(adrp_case.target))));

    map.put64(ldr_x_scaled.target, literal_x);
    leaf(map, ldr_x_scaled, original_word(encode_load_literal(
        Opcode::ldr_literal_x, 0, addr(ldr_x_scaled.original), addr(ldr_x_scaled.target))));
    map.put64(ldr_x_unscaled.target, literal_x2);
    leaf(map, ldr_x_unscaled, original_word(encode_load_literal(
        Opcode::ldr_literal_x, 0, addr(ldr_x_unscaled.original), addr(ldr_x_unscaled.target))));
    map.put(ldrsw_case.target, 0x80000001u);
    leaf(map, ldrsw_case, original_word(encode_load_literal(
        Opcode::ldrsw_literal, 0, addr(ldrsw_case.original), addr(ldrsw_case.target))));
    map.put(ldr_w_case.target, 0xFFFFFFFEu);
    leaf(map, ldr_w_case, original_word(encode_load_literal(
        Opcode::ldr_literal_w, 0, addr(ldr_w_case.original), addr(ldr_w_case.target))));
    map.put64(ldr_q_case.target, literal_x);
    map.put64(ldr_q_case.target + 8, literal_x2);
    leaf(map, ldr_q_case,
         original_word(encode_load_literal(Opcode::ldr_literal_q, 0, addr(ldr_q_case.original),
                                           addr(ldr_q_case.target))),
         {.scratch_register = 9}, {fmov_x0_d0, umov_x1_v0_d1, eor_x0_x0_x1});

    two_way(map, cbz_case, original_word(encode_cbz(false, true, 0, addr(cbz_case.original),
                                                    addr(cbz_case.target))));
    two_way(map, tbnz_case, original_word(encode_tbz(true, 0, 40, addr(tbnz_case.original),
                                                     addr(tbnz_case.target))));
    two_way(map, bcond_case,
            original_word(encode_b_cond(Condition::lt, addr(bcond_case.original),
                                        addr(bcond_case.target))),
            cmp_x0_x1);

    // Callee returns 41; the continuation adds one and returns from the frame.
    for (const auto* layout : {&bl_preserve_case, &bl_plain_case, &blr_preserve_case}) {
        map.put(layout->target, movz_x0(41));
        map.put(layout->target + 4, ret);
        map.put(layout->relocated - 4, push_frame);
    }
    const auto continuation = [&](std::size_t at) {
        map.put(at, add_x0_1);
        map.put(at + 4, pop_frame);
        map.put(at + 8, ret);
    };
    const SequenceOptions preserve{.preserve_link_address = true};

    // Preserved link: the callee returns to the original stream at from + 4.
    auto sequence = relocate(map, bl_preserve_case,
                             original_word(encode_bl(addr(bl_preserve_case.original),
                                                     addr(bl_preserve_case.target))),
                             preserve);
    map.put(bl_preserve_case.relocated, sequence);
    map.put(bl_preserve_case.relocated + 4 * sequence.count, movz_x0(99));  // must not run
    continuation(bl_preserve_case.original + 4);

    // Plain relocation: the callee returns after the relocated BL.
    sequence = relocate(map, bl_plain_case,
                        original_word(encode_bl(addr(bl_plain_case.original),
                                                addr(bl_plain_case.target))));
    CHECK(sequence.link_value_changed);
    continuation(bl_plain_case.relocated + map.put(bl_plain_case.relocated, sequence));
    map.put(bl_plain_case.original + 4, movz_x0(99));  // must not run

    // BLR x1 with a preserved link address.
    sequence = relocate(map, blr_preserve_case, original_word(encode_blr(1)), preserve);
    map.put(blr_preserve_case.relocated, sequence);
    map.put(blr_preserve_case.relocated + 4 * sequence.count, movz_x0(99));
    continuation(blr_preserve_case.original + 4);
}

void run(const Mapping& map) {
    const auto call = [&](const Layout& layout, std::uint64_t a = 0, std::uint64_t b = 0,
                          std::size_t entry_back = 0) {
        return map.function(layout.relocated - entry_back)(a, b);
    };
    CHECK_EQ(call(adr_case), map.address(adr_case.target));
    CHECK_EQ(call(adrp_case), map.address(adrp_case.target) & ~std::uint64_t{0xFFF});
    CHECK_EQ(call(ldr_x_scaled), literal_x);
    CHECK_EQ(call(ldr_x_unscaled), literal_x2);
    CHECK_EQ(call(ldrsw_case), 0xFFFFFFFF80000001ull);
    CHECK_EQ(call(ldr_w_case, ~0ull), 0xFFFFFFFEull);
    CHECK_EQ(call(ldr_q_case), literal_x ^ literal_x2);

    CHECK_EQ(call(cbz_case, 0), 1);
    CHECK_EQ(call(cbz_case, 5), 2);
    CHECK_EQ(call(tbnz_case, std::uint64_t{1} << 40), 1);
    CHECK_EQ(call(tbnz_case, ~(std::uint64_t{1} << 40)), 2);
    CHECK_EQ(call(bcond_case, static_cast<std::uint64_t>(-3), 2, 4), 1);  // -3 < 2
    CHECK_EQ(call(bcond_case, 3, 2, 4), 2);

    CHECK_EQ(call(bl_preserve_case, 0, 0, 4), 42);
    CHECK_EQ(call(bl_plain_case, 0, 0, 4), 42);
    CHECK_EQ(call(blr_preserve_case, 0, map.address(blr_preserve_case.target), 4), 42);
}

void distances_require_expansion(const Mapping& map) {
    // Guard the premise: every non-B, non-ADRP case is out of single-word reach.
    for (const auto* layout : {&adr_case, &ldr_x_scaled, &ldr_q_case, &cbz_case, &tbnz_case,
                               &bcond_case}) {
        CHECK(map.address(layout->relocated) - map.address(layout->target) > (1u << 20));
    }
}

} // namespace

int main() {
    Mapping map;
    if (!CHECK(map.valid())) {
        return armigo::aarch64::test::finish("aarch64.relocation_execution");
    }
    build(map);
    distances_require_expansion(map);
    if (CHECK(map.seal())) {
        run(map);
    }
    return armigo::aarch64::test::finish("aarch64.relocation_execution");
}
