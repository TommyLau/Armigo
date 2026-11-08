// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Predicated texture samples, end to end: independently authored synthetic
// instruction words run through decode, emit_ir, SSA construction and
// legalization. A filtered sample whose inputs exist before its guard leaves
// the guarded block; texel fetches and samples that read values computed under
// the guard stay inside it. No game bytes, captures or dumps are used.

#include "pipeline/phases.h"
#include "pipeline/dispatch.h"

#include <iostream>
#include <vector>
#include <map>
#include <string_view>

namespace {
using namespace armigo::shader::pipeline;
int failures{};
void check(bool value, std::string_view label) {
    if (!value) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}
template<class T> T take(Result<T> result) {
    if (auto* error = std::get_if<Diagnostic>(&result)) {
        std::cerr << error->message << '\n';
        throw 1;
    }
    return std::get<T>(std::move(result));
}

TextureDescriptor descriptor() {
    return {{TextureDimension::d2, false, false}, ScalarType::f32,
        ComponentFormat::floating, true, false, 1, "synthetic-format-f32"};
}
class Facts final : public FactSource {
public:
    std::map<FactKey, FactValue> values;
    Facts() {
        values[{FactKind::constant_buffer_bound_mask}] = std::uint32_t{1};
        values[{FactKind::constant_buffer_size, 0}] = std::uint32_t{256};
        values[{FactKind::local_memory_bytes}] = std::uint32_t{64};
        values[fact_query::texture({0, 16}).key()] = descriptor();
    }
    Knowledge<FactValue> lookup(FactKey key) const override {
        if (auto it = values.find(key); it != values.end())
            return Known<FactValue>{it->second, {OriginKind::explicit_option, "synthetic-guarded-sample"}};
        return Unknown{"Synthetic fact unavailable"};
    }
};
MetalProfile profile() { return {"synthetic-metal", 1, 31, 32, 16, true, false, true, false, false, true}; }

std::uint64_t pattern(std::string_view bits) {
    std::uint64_t result{};
    for (std::size_t i = 0; i < bits.size(); ++i) if (bits[i] == '1') result |= std::uint64_t{1} << (63 - i);
    return result;
}
std::uint64_t place(unsigned first, unsigned width, std::uint64_t value) {
    return (value & ((std::uint64_t{1} << width) - 1)) << first;
}
// Every word takes an instruction predicate: 7 is the always-true PT.
// MOV32I: register := 32-bit literal.
std::uint64_t mov32i(unsigned dest, std::uint32_t literal, unsigned predicate = 7) {
    return pattern("000000010000") | dest | place(16, 3, predicate) | place(20, 32, literal);
}
// LDC: B32 load from a static constant-buffer byte address.
std::uint64_t ldc(unsigned dest, unsigned byte_offset, unsigned predicate = 7) {
    return pattern("1110111110010x") | dest | place(8, 8, 255) | place(16, 3, predicate) |
        place(20, 16, byte_offset) | place(48, 3, 4);
}
// ISETP: P0 := left < right (unsigned), combined with the always-true PT.
std::uint64_t isetp(unsigned left, unsigned right) {
    return pattern("010110110110") | place(8, 8, left) | place(16, 3, 7) |
        place(20, 8, right) | place(39, 3, 7) | place(49, 3, 1);
}
// ISETP into predicate `dest`, its second result discarded into PT.
std::uint64_t isetp_to(unsigned left, unsigned right, unsigned dest) {
    return isetp(left, right) | place(0, 3, 7) | place(3, 3, dest);
}
// The byte address of instruction `index`: a scheduling word opens every
// 32-byte group of three instructions.
unsigned at(std::size_t index) { return unsigned(32 * (index / 3) + 8 * (index % 3 + 1)); }
// BRA from the instruction at `from` to the one at `to`, under `predicate`.
std::uint64_t bra(std::size_t from, std::size_t to, unsigned predicate) {
    return pattern("111000100100") | 15 | place(16, 3, predicate) |
        place(20, 24, static_cast<std::uint64_t>(at(to) - at(from) - 8) & 0xFFFFFF);
}
// STL: store R<value> to local memory byte `offset` (keeps the sampled value
// live); a guarded store orders the samples after it.
std::uint64_t stl(unsigned value, unsigned offset = 0, unsigned predicate = 7) {
    return pattern("1110111101010x") | value | place(8, 8, 255) | place(16, 3, predicate) |
        place(20, 24, offset) | place(48, 3, 4);
}
std::uint64_t exit_word() { return pattern("111000110000") | place(16, 3, 7) | 15; }
// TEX_B: 2D sample, mask XY, coordinates on Ra, handle on Rb. LOD mode 1 is
// level zero, 0 the implicit LOD.
std::uint64_t texb(unsigned ra, unsigned rb, unsigned dest, unsigned predicate, unsigned lod = 1) {
    return pattern("1101111010xxxx") | place(28, 3, 2) | place(31, 4, 0b0011) | place(37, 3, lod) |
        place(16, 3, predicate) | place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TLD4_B: 2D gather of component 0, mask X. Offset selector 2 reads eight
// per-texel offsets from the two registers after the handle.
std::uint64_t tld4b(unsigned ra, unsigned rb, unsigned dest, unsigned predicate, unsigned offsets) {
    return pattern("1101111011xxxx") | place(28, 3, 2) | place(31, 4, 0b0001) | place(36, 2, offsets) |
        place(16, 3, predicate) | place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TLD_B: 2D integer fetch, mask XY.
std::uint64_t tldb(unsigned ra, unsigned rb, unsigned dest, unsigned predicate) {
    return pattern("11011101xxxxxx") | place(28, 3, 2) | place(31, 4, 0b0011) |
        place(16, 3, predicate) | place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}

StageImage image(const std::vector<std::uint64_t>& words) {
    StageImage result;
    result.stage = Stage::fragment;
    auto append = [&](std::uint64_t word) {
        for (unsigned i = 0; i < 8; ++i) result.bytes.push_back(std::byte((word >> (i * 8)) & 255));
    };
    append(0);
    for (auto word : words) {
        if (result.bytes.size() % 32 == 0) append(0);
        append(word);
    }
    if (result.bytes.size() % 32 == 0) append(0);
    return result;
}
LegalProgram legalized(const StageImage& input, bool hoist_implicit_samples = false) {
    Facts facts;
    FactReader reader(facts);
    auto registers = take(emit_ir(take(decode(input, reader, {})), reader, {}));
    auto host = profile();
    host.hoist_implicit_samples = hoist_implicit_samples;
    return take(legalize(take(construct_ssa(std::move(registers), {})), reader, host));
}

// Texture operations by where they ended up: in a block that tests a
// predicate (before its guard) or in a block such a test guards, which may
// itself test another predicate (guarded_guard counts those too). `gathers`
// splits the gathers out the same way.
struct Placement {
    unsigned before_guard{}, guarded{}, guarded_guard{}, other{};
    unsigned gathers_before_guard{}, gathers_guarded{};
    std::vector<TextureMode> modes;
};
Placement placement(const Program& p) {
    Placement result;
    for (std::uint32_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block = p.blocks[b];
        const bool guarded = block.predecessors.size() == 1 &&
            std::holds_alternative<Branch>(p.blocks[block.predecessors.front().value].terminator);
        const bool guard = std::holds_alternative<Branch>(block.terminator);
        for (const auto id : block.operations) {
            const auto* texture = std::get_if<TextureInfo>(&p.operations[id.value].info);
            if (!texture) continue;
            result.modes.push_back(texture->mode);
            const bool gather = texture->mode == TextureMode::gather;
            if (guarded) {
                ++result.guarded;
                if (guard) ++result.guarded_guard;
                if (gather) ++result.gathers_guarded;
            } else if (guard) {
                ++result.before_guard;
                if (gather) ++result.gathers_before_guard;
            } else {
                ++result.other;
            }
        }
    }
    return result;
}

// The coordinates R20/R21 and the handle R4 are ready before every guard.
std::vector<std::uint64_t> prologue() {
    return {mov32i(20, 0), mov32i(21, 0), ldc(4, 16), ldc(6, 0)};
}
// Tap k's own predicate, as a blur's per-tap bounds test is: P0 = cb[0] <
// cb[32 + 4k], unknown until run time.
void guard(std::vector<std::uint64_t>& words, unsigned k) {
    words.push_back(ldc(40 + k, 32 + 4 * k));
    words.push_back(isetp(6, 40 + k));
}
// Eight separate guarded taps writing R80..R95, enough for the function
// threshold.
void separate_taps(std::vector<std::uint64_t>& words) {
    for (unsigned k = 0; k < 8; ++k) { guard(words, k); words.push_back(texb(20, 4, 80 + 2 * k, 0)); }
}
// Stores the separate taps and the registers in `more`, then exits.
StageImage finish(std::vector<std::uint64_t> words, std::initializer_list<unsigned> more) {
    for (unsigned k = 0; k < 8; ++k) words.push_back(stl(80 + 2 * k, 4 * k));
    unsigned offset = 32;
    for (const auto value : more) { words.push_back(stl(value, offset)); offset += 4; }
    words.push_back(exit_word());
    return image(words);
}

// `count` taps, each under its own predicate. Every result is stored.
StageImage taps(unsigned count, bool fetch = false) {
    auto words = prologue();
    for (unsigned k = 0; k < count; ++k) {
        guard(words, k);
        words.push_back(fetch ? tldb(20, 4, 80 + 2 * k, 0) : texb(20, 4, 80 + 2 * k, 0));
    }
    for (unsigned k = 0; k < count; ++k) words.push_back(stl(80 + 2 * k, 4 * k));
    words.push_back(exit_word());
    return image(words);
}

void test_many_guarded_taps_leave_their_guards() {
    const auto where = placement(legalized(taps(8)).ir);
    check(where.modes.size() == 8 && where.modes.front() == TextureMode::sample_level, "Eight level-zero taps remain");
    check(where.before_guard == 8 && where.guarded == 0, "Eight guarded taps all run before their guards");
}

// An implicit-LOD sample takes its derivatives from its quad, and lanes whose
// guard is false would change them, so these taps keep their guards.
void test_implicit_lod_taps_stay_guarded() {
    auto words = prologue();
    for (unsigned k = 0; k < 8; ++k) { guard(words, k); words.push_back(texb(20, 4, 80 + 2 * k, 0, 0)); }
    for (unsigned k = 0; k < 8; ++k) words.push_back(stl(80 + 2 * k, 4 * k));
    words.push_back(exit_word());
    const auto where = placement(legalized(image(words)).ir);
    check(where.modes.size() == 8 && where.modes.front() == TextureMode::sample_implicit,
          "The predicated TEXs take the implicit LOD");
    check(where.guarded == 8, "Implicit-LOD samples stay under their predicates");
    const auto opted = placement(legalized(image(words), true).ir);
    check(opted.modes.size() == 8 && opted.before_guard == 8 && opted.guarded == 0,
          "A title's opt-in moves implicit-LOD samples before their guards");
}

void test_few_guarded_taps_stay() {
    const auto where = placement(legalized(taps(7)).ir);
    check(where.modes.size() == 7 && where.guarded == 7, "Below the tap threshold every sample keeps its guard");
}

void test_texel_fetches_stay_guarded() {
    const auto where = placement(legalized(taps(8, true)).ir);
    check(where.modes.size() == 8 && where.modes.front() == TextureMode::fetch, "The predicated TLDs are texel fetches");
    check(where.guarded == 8, "Texel fetches stay under their predicates");
}

// One guarded run of three taps among seven separate ones: the run moves at
// most four native accesses, two samples that may gain a border sample each.
void test_guarded_run_moves_at_most_the_access_cap() {
    auto words = prologue();
    guard(words, 0);
    for (unsigned k = 0; k < 3; ++k) words.push_back(texb(20, 4, 80 + 2 * k, 0));
    for (unsigned k = 3; k < 10; ++k) { guard(words, k); words.push_back(texb(20, 4, 80 + 2 * k, 0)); }
    for (unsigned k = 0; k < 10; ++k) words.push_back(stl(80 + 2 * k, 4 * k));
    words.push_back(exit_word());
    const auto where = placement(legalized(image(words)).ir);
    check(where.modes.size() == 10 && where.guarded == 1 && where.before_guard == 9,
          "A guarded run of three samples moves two and keeps the third");
}

// A guarded store before the tap in the same guarded run keeps it in place.
void test_store_orders_the_samples_after_it() {
    auto words = prologue();
    guard(words, 0);
    words.push_back(stl(20, 60, 0));
    words.push_back(texb(20, 4, 80, 0));
    for (unsigned k = 1; k < 9; ++k) { guard(words, k); words.push_back(texb(20, 4, 80 + 2 * k, 0)); }
    for (unsigned k = 0; k < 9; ++k) words.push_back(stl(80 + 2 * k, 4 * k));
    words.push_back(exit_word());
    const auto where = placement(legalized(image(words)).ir);
    check(where.modes.size() == 9 && where.guarded == 1 && where.before_guard == 8,
          "A sample after a guarded store stays guarded; the separate taps move");
}

// The coordinate is loaded under the same guard as the tap, so that tap cannot
// run before it; the eight separate taps still move.
void test_sample_of_guarded_value_stays_guarded() {
    auto words = prologue();
    guard(words, 9);
    words.push_back(ldc(22, 8, 0));
    words.push_back(texb(22, 4, 98, 0));
    for (unsigned k = 0; k < 8; ++k) { guard(words, k); words.push_back(texb(20, 4, 80 + 2 * k, 0)); }
    for (unsigned k = 0; k < 8; ++k) words.push_back(stl(80 + 2 * k, 4 * k));
    words.push_back(stl(98, 32));
    words.push_back(exit_word());
    const auto where = placement(legalized(image(words)).ir);
    check(where.modes.size() == 9 && where.guarded == 1 && where.before_guard == 8,
          "A sample of a coordinate computed under its guard stays guarded");
}
// A loop that may leave through a guarded exit: the exit's samples run once,
// so they must not move into the test inside the loop. P1 = cb[0] < cb[8].
void test_loop_exit_keeps_its_samples() {
    auto words = prologue();
    separate_taps(words);
    const auto loop = words.size();
    words.push_back(ldc(30, 8));
    words.push_back(isetp_to(6, 30, 1));
    const auto leave = words.size();
    words.push_back(0);
    words.push_back(bra(words.size(), loop, 7));
    const auto exit_block = words.size();
    words[leave] = bra(leave, exit_block, 1);
    words.push_back(texb(20, 4, 96, 7));
    words.push_back(texb(20, 4, 98, 7));
    const auto where = placement(legalized(finish(words, {96, 98})).ir);
    check(where.modes.size() == 10 && where.before_guard == 8 && where.guarded == 2,
          "A loop exit keeps its samples; the separate taps move");
}

// The same taps inside a loop still leave their own guards: each guarded
// block rejoins its guard's other side before the loop repeats.
void test_guarded_taps_in_a_loop_move() {
    auto words = prologue();
    const auto loop = words.size();
    separate_taps(words);
    words.push_back(ldc(30, 8));
    words.push_back(isetp_to(6, 30, 1));
    words.push_back(bra(words.size(), loop, 1));
    const auto where = placement(legalized(finish(words, {})).ir);
    check(where.modes.size() == 8 && where.before_guard == 8 && where.guarded == 0,
          "Guarded taps inside a loop move before their guards");
}

// Both arms of one branch hold two samples (four accesses each): the guard
// takes four accesses in all, one arm's pair.
void test_two_arms_share_one_budget() {
    auto words = prologue();
    separate_taps(words);
    words.push_back(ldc(30, 8));
    words.push_back(isetp_to(6, 30, 1));
    const auto branch = words.size();
    words.push_back(0);
    words.push_back(texb(20, 4, 96, 7));
    words.push_back(texb(20, 4, 98, 7));
    const auto jump = words.size();
    words.push_back(0);
    const auto other_arm = words.size();
    words.push_back(texb(20, 4, 100, 7));
    words.push_back(texb(20, 4, 102, 7));
    const auto join = words.size();
    words[branch] = bra(branch, other_arm, 1);
    words[jump] = bra(jump, join, 7);
    const auto where = placement(legalized(finish(words, {96, 98, 100, 102})).ir);
    check(where.modes.size() == 12 && where.before_guard == 10 && where.guarded == 2,
          "Two arms of one guard move four accesses between them");
}

// An outer guarded block that tests an inner predicate: its own sample stays
// (the block ends in a branch, not a jump), and the inner block's three samples
// move two into it, so nothing climbs past the inner guard.
void test_nested_guards_do_not_chain() {
    auto words = prologue();
    separate_taps(words);
    words.push_back(ldc(30, 8));
    words.push_back(isetp_to(6, 30, 1));
    const auto outer = words.size();
    words.push_back(0);
    words.push_back(texb(20, 4, 96, 7));
    words.push_back(isetp_to(30, 6, 2));
    const auto inner = words.size();
    words.push_back(0);
    words.push_back(texb(20, 4, 98, 7));
    words.push_back(texb(20, 4, 100, 7));
    words.push_back(texb(20, 4, 102, 7));
    const auto inner_join = words.size();
    words.push_back(mov32i(31, 0));
    const auto outer_join = words.size();
    words[outer] = bra(outer, outer_join, 1);
    words[inner] = bra(inner, inner_join, 2);
    const auto where = placement(legalized(finish(words, {96, 98, 100, 102})).ir);
    check(where.modes.size() == 12 && where.before_guard == 8 && where.guarded_guard == 3 && where.guarded == 4,
          "The inner guard takes two samples; the outer guard takes none");
}

// A gather with per-texel offsets is four native gathers: it fills a guard's
// budget, so the sample after it stays, and of two such gathers one moves.
// The handle is R8 so the packed offsets (zero) sit in R9 and R10.
void test_offset_gather_fills_the_budget() {
    auto words = prologue();
    words.push_back(ldc(8, 16));
    words.push_back(mov32i(9, 0));
    words.push_back(mov32i(10, 0));
    separate_taps(words);
    guard(words, 8);
    words.push_back(tld4b(20, 8, 96, 0, 2));
    words.push_back(texb(20, 4, 98, 0));
    guard(words, 9);
    words.push_back(tld4b(20, 8, 100, 0, 2));
    words.push_back(tld4b(20, 8, 102, 0, 2));
    const auto where = placement(legalized(finish(words, {96, 98, 100, 102})).ir);
    check(where.modes.size() == 12 && where.gathers_before_guard == 2 && where.gathers_guarded == 1 &&
              where.before_guard == 10 && where.guarded == 2,
          "An offset gather takes the whole budget: the sample after it and a second gather stay");
}
} // namespace

int main() {
    try {
        test_many_guarded_taps_leave_their_guards();
        test_implicit_lod_taps_stay_guarded();
        test_few_guarded_taps_stay();
        test_texel_fetches_stay_guarded();
        test_guarded_run_moves_at_most_the_access_cap();
        test_store_orders_the_samples_after_it();
        test_sample_of_guarded_value_stays_guarded();
        test_loop_exit_keeps_its_samples();
        test_guarded_taps_in_a_loop_move();
        test_two_arms_share_one_budget();
        test_nested_guards_do_not_chain();
        test_offset_gather_fills_the_budget();
    } catch (int) {
        ++failures;
    }
    if (failures) std::cerr << failures << " guarded sample checks failed\n";
    else std::cout << "All guarded sample checks passed\n";
    return failures ? 1 : 0;
}
