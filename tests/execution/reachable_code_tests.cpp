// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/execution/reachable_code.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

extern "C" const std::byte armigo_test_reachable_a_begin[];
extern "C" const std::byte armigo_test_reachable_a_end[];
extern "C" const std::byte armigo_test_reachable_b_begin[];
extern "C" const std::byte armigo_test_reachable_b_function[];
extern "C" const std::byte armigo_test_reachable_b_target[];
extern "C" const std::byte armigo_test_reachable_b_end[];
extern "C" const std::byte armigo_test_reachable_c_begin[];
extern "C" const std::byte armigo_test_reachable_c_end[];
extern "C" const std::byte armigo_test_reachable_got[];

namespace {
using namespace armigo;
using namespace armigo::execution;
namespace a64 = armigo::aarch64;

int failures = 0;

void expect(bool condition, const char* message, int line) {
    if (!condition) {
        std::fprintf(stderr, "line %d: %s\n", line, message);
        ++failures;
    }
}
#define EXPECT(condition) expect((condition), #condition, __LINE__)

std::uint32_t ok(a64::Encoded encoded) {
    if (!encoded.ok()) {
        std::fprintf(stderr, "encoding failed\n");
        ++failures;
    }
    return encoded.word;
}

constexpr std::uint32_t nop = 0xD503201Fu;
constexpr std::uint32_t udf = 0x00000000u;
constexpr std::uint32_t fadd_s = 0x1E7E0252u;           // FJCVTZS S0, W1: outside the decoder
constexpr std::uint32_t movn_reserved = 0xB2800000u;    // move wide with opc = 01
constexpr std::uint32_t ldr_post_overlap = 0xF8408400u; // LDR X0, [X0], #8
constexpr std::uint32_t retaa = 0xD65F0BFFu;
constexpr std::uint32_t eret = 0xD69F03E0u;
constexpr std::uint32_t paciasp = 0xD503233Fu;
constexpr std::uint32_t hlt = 0xD4400000u;

// A synthetic module: words are placed at absolute addresses.
struct Module {
    std::uint64_t base;
    std::vector<std::byte> bytes;

    Module(std::uint64_t base_address, std::size_t size) : base(base_address), bytes(size) {}

    void word(std::uint64_t address, std::uint32_t value) {
        std::memcpy(bytes.data() + (address - base), &value, sizeof(value));
    }
    void quad(std::uint64_t address, std::uint64_t value) {
        std::memcpy(bytes.data() + (address - base), &value, sizeof(value));
    }
    ReachableImage image(std::initializer_list<AddressRange> executable) const {
        return {base, bytes, executable};
    }
};

// Emits ADRP/LDR[/ADD]/BR reading the 8-byte slot.
void thunk(Module& module, std::uint64_t at, std::uint64_t slot, bool with_add,
           unsigned address_register = 16, unsigned target_register = 17) {
    const auto offset = static_cast<std::uint32_t>(slot & 0xFFF);
    module.word(at, ok(a64::encode_adrp(address_register, at, slot)));
    module.word(at + 4, ok(a64::encode_load_unsigned_offset(a64::Opcode::ldr_literal_x,
                                                            target_register, address_register,
                                                            offset)));
    if (with_add) {
        module.word(at + 8, ok(a64::encode_add_immediate(true, address_register,
                                                         address_register, offset)));
    }
    module.word(at + (with_add ? 12 : 8), ok(a64::encode_br(target_register)));
}

const ReachableSite* site_at(const ReachableCode& code, std::uint64_t address) {
    const auto it = std::lower_bound(
        code.sites.begin(), code.sites.end(), address,
        [](const ReachableSite& site, std::uint64_t a) { return site.address < a; });
    return it != code.sites.end() && it->address == address ? &*it : nullptr;
}

bool has_edge(const ReachableCode& code, std::uint64_t source, std::uint64_t target,
              EdgeKind kind, bool external) {
    return std::any_of(code.edges.begin(), code.edges.end(), [&](const ReachableEdge& edge) {
        return edge.source == source && edge.target == target && edge.kind == kind &&
               edge.external_leaf == external;
    });
}

bool only_failure(const ReachableCode& code, ReachableError error, std::uint64_t address) {
    return code.status == CollectionStatus::partial && code.failures.size() == 1 &&
           code.failures[0].error == error && code.failures[0].address == address;
}

constexpr std::uint64_t host_return = 0x0400'0000;
constexpr std::uint64_t host_abort = 0x0400'0010;

void cross_module_program() {
    constexpr std::uint64_t a = 0x10'0000;
    constexpr std::uint64_t b = 0x20'0000;
    Module ma(a, 0x3000);
    Module mb(b, 0x1000);

    ma.word(a + 0x00, nop);
    ma.word(a + 0x04, ok(a64::encode_cbz(false, true, 0, a + 0x04, a + 0x18)));
    ma.word(a + 0x08, ok(a64::encode_tbz(true, 1, 3, a + 0x08, a + 0x00)));
    ma.word(a + 0x0c, ok(a64::encode_bl(a + 0x0c, a + 0x100)));
    ma.word(a + 0x10, ok(a64::encode_bl(a + 0x10, a + 0x200)));
    ma.word(a + 0x14, ok(a64::encode_b_cond(a64::Condition::ne, a + 0x14, a + 0x04)));
    ma.word(a + 0x18, ok(a64::encode_bl(a + 0x18, host_return)));
    ma.word(a + 0x1c, ok(a64::encode_svc(1)));
    ma.word(a + 0x20, ok(a64::encode_bl(a + 0x20, a + 0x210)));
    ma.word(a + 0x24, ok(a64::encode_b_cond(a64::Condition::al, a + 0x24, a + 0x30)));
    ma.word(a + 0x28, udf);  // skipped by B.AL
    ma.word(a + 0x2c, udf);
    ma.word(a + 0x30, ok(a64::encode_bl(a + 0x30, host_abort)));
    ma.word(a + 0x34, udf);  // after a non-returning leaf

    ma.word(a + 0x100, ok(a64::encode_add_immediate(true, 0, 0, 1)));
    ma.word(a + 0x104, ok(a64::encode_b(a + 0x104, a + 0x10c)));
    ma.word(a + 0x108, udf);
    ma.word(a + 0x10c, ok(a64::encode_cbz(true, true, 0, a + 0x10c, a + 0x114)));
    ma.word(a + 0x110, ok(a64::encode_brk(1)));
    ma.word(a + 0x114, ok(a64::encode_ret()));

    thunk(ma, a + 0x200, a + 0x2000, true);
    thunk(ma, a + 0x210, a + 0x2008, false);
    ma.quad(a + 0x2000, b);
    ma.quad(a + 0x2008, host_return);

    mb.word(b + 0x00, ok(a64::encode_add_immediate(true, 2, 2, 4)));
    mb.word(b + 0x04, ok(a64::encode_tbz(false, 2, 40, b + 0x04, b + 0x0c)));
    mb.word(b + 0x08, ok(a64::encode_b(b + 0x08, b + 0x00)));
    mb.word(b + 0x0c, ok(a64::encode_bl(b + 0x0c, a + 0x100)));
    mb.word(b + 0x10, ok(a64::encode_ret()));

    const std::vector<ReachableImage> images{mb.image({{b, b + 0x800}}),
                                             ma.image({{a, a + 0x1000}})};
    const std::uint64_t entries[] = {a};
    const ExternalLeaf leaves[] = {{host_return, true}, {host_abort, false}};
    ReachableOptions options;
    options.traps = TrapPolicy::terminal;
    const ReachableCode code = collect_reachable_code(images, entries, leaves, options);

    EXPECT(code.complete());
    EXPECT(code.failures.empty());
    std::vector<std::uint64_t> expected;
    for (std::uint64_t offset = 0; offset <= 0x24; offset += 4) expected.push_back(a + offset);
    for (std::uint64_t offset : {0x30, 0x100, 0x104, 0x10c, 0x110, 0x114, 0x200, 0x204, 0x208,
                                 0x20c, 0x210, 0x214, 0x218}) {
        expected.push_back(a + offset);
    }
    for (std::uint64_t offset = 0; offset <= 0x10; offset += 4) expected.push_back(b + offset);
    std::vector<std::uint64_t> visited;
    for (const auto& site : code.sites) visited.push_back(site.address);
    EXPECT(visited == expected);

    EXPECT(site_at(code, a + 0x04)->kind == SiteKind::conditional_branch);
    EXPECT(site_at(code, a + 0x0c)->kind == SiteKind::call);
    EXPECT(site_at(code, a + 0x1c)->kind == SiteKind::supervisor_call);
    EXPECT(site_at(code, a + 0x24)->kind == SiteKind::branch);
    EXPECT(site_at(code, a + 0x110)->kind == SiteKind::trap);
    EXPECT(site_at(code, a + 0x114)->kind == SiteKind::return_);
    EXPECT(site_at(code, a + 0x200)->kind == SiteKind::thunk_address);
    EXPECT(site_at(code, a + 0x204)->kind == SiteKind::thunk_load);
    EXPECT(site_at(code, a + 0x208)->kind == SiteKind::thunk_add);
    EXPECT(site_at(code, a + 0x20c)->kind == SiteKind::thunk_branch);
    EXPECT(site_at(code, a + 0x218)->kind == SiteKind::thunk_branch);
    EXPECT(site_at(code, b)->image == 0 && site_at(code, a)->image == 1);
    EXPECT(site_at(code, a + 0x100)->effects.operation == a64::Operation::add_immediate);
    EXPECT(site_at(code, a + 0x10)->instruction.opcode == a64::Opcode::bl);

    EXPECT(code.thunks.size() == 2);
    if (code.thunks.size() == 2) {
        const ResolvedThunk& first = code.thunks[0];
        EXPECT(first.entry == a + 0x200 && first.branch == a + 0x20c && first.slot == a + 0x2000 &&
               first.target == b && first.has_add && !first.external_leaf &&
               first.address_register == 16 && first.target_register == 17);
        const ResolvedThunk& second = code.thunks[1];
        EXPECT(second.entry == a + 0x210 && second.branch == a + 0x218 &&
               second.target == host_return && second.external_leaf && !second.has_add);
    }
    EXPECT(has_edge(code, a + 0x200, b, EdgeKind::thunk, false));
    EXPECT(has_edge(code, a + 0x210, host_return, EdgeKind::thunk, true));
    EXPECT(has_edge(code, a + 0x18, host_return, EdgeKind::call, true));
    EXPECT(has_edge(code, a + 0x30, host_abort, EdgeKind::call, true));
    EXPECT(has_edge(code, b + 0x0c, a + 0x100, EdgeKind::call, false));
    EXPECT(has_edge(code, a + 0x08, a, EdgeKind::conditional_branch, false));
    EXPECT(has_edge(code, a + 0x24, a + 0x30, EdgeKind::branch, false));
    EXPECT(code.statistics.supervisor_calls == 1 && code.statistics.traps == 1 &&
           code.statistics.external_leaf_edges == 3 && code.statistics.assumed_sequential == 0);

    // Identical input gives identical output.
    const ReachableCode again = collect_reachable_code(images, entries, leaves, options);
    EXPECT(again.sites.size() == code.sites.size() && again.edges.size() == code.edges.size());

    // BRK is refused by the default trap policy and SVC under a failing policy.
    const ReachableCode strict = collect_reachable_code(images, entries, leaves);
    EXPECT(only_failure(strict, ReachableError::trap, a + 0x110));
    EXPECT(site_at(strict, a + 0x110)->kind == SiteKind::failed);
    options.supervisor_calls = SupervisorCallPolicy::fail;
    const ReachableCode no_svc = collect_reachable_code(images, entries, leaves, options);
    EXPECT(only_failure(no_svc, ReachableError::supervisor_call, a + 0x1c));
    EXPECT(site_at(no_svc, a + 0x20) == nullptr);

    // Without the B image the thunk target is not executable code.
    const std::vector<ReachableImage> only_a{ma.image({{a, a + 0x1000}})};
    const ReachableCode missing = collect_reachable_code(only_a, entries, leaves,
                                                         ReachableOptions{.traps = TrapPolicy::terminal});
    EXPECT(only_failure(missing, ReachableError::target_not_executable, a + 0x20c));
    if (!missing.failures.empty()) {
        EXPECT(missing.failures[0].target == b && missing.failures[0].slot == a + 0x2000 &&
               missing.failures[0].source == a + 0x200);
    }
    EXPECT(site_at(missing, a + 0x20c)->kind == SiteKind::failed);
}

void assembled_fixture() {
    const auto address = [](const std::byte* p) { return reinterpret_cast<std::uint64_t>(p); };
    const std::uint64_t a = address(armigo_test_reachable_a_begin);
    const std::uint64_t a_end = address(armigo_test_reachable_a_end);
    const std::uint64_t b = address(armigo_test_reachable_b_begin);
    const std::uint64_t b_end = address(armigo_test_reachable_b_end);
    const std::uint64_t c = address(armigo_test_reachable_c_begin);
    const std::uint64_t c_end = address(armigo_test_reachable_c_end);
    const std::uint64_t got = address(armigo_test_reachable_got);
    EXPECT(a_end == b && b_end == c);
    EXPECT(a_end - a == 13 * 4 && b_end - b == 7 * 4 && c_end - c == 4 * 4);

    const std::vector<ReachableImage> images{
        {a, {armigo_test_reachable_a_begin, a_end - a}, {{a, a_end}}},
        {b, {armigo_test_reachable_b_begin, b_end - b}, {{b, b_end}}},
        {c, {armigo_test_reachable_c_begin, c_end - c}, {{c, c_end}}},
        {got, {armigo_test_reachable_got, 16}, {}},
    };
    const std::uint64_t entries[] = {a};
    const ReachableCode code = collect_reachable_code(images, entries);
    EXPECT(code.complete());
    EXPECT(code.sites.size() == 20);
    EXPECT(code.thunks.size() == 1);
    if (code.thunks.size() == 1) {
        EXPECT(code.thunks[0].entry == a + 9 * 4 && code.thunks[0].slot == got &&
               code.thunks[0].target == address(armigo_test_reachable_b_target) &&
               code.thunks[0].has_add);
    }
    EXPECT(has_edge(code, a + 5 * 4, address(armigo_test_reachable_b_function), EdgeKind::call,
                    false));
    EXPECT(code.statistics.supervisor_calls == 1);

    const std::uint64_t indirect[] = {c};
    const ReachableCode stop = collect_reachable_code(images, indirect);
    EXPECT(only_failure(stop, ReachableError::indirect_call, c + 8));
    EXPECT(stop.sites.size() == 3 && stop.thunks.empty());
    ReachableOptions follow;
    follow.follow_indirect_call_return = true;
    const ReachableCode both = collect_reachable_code(images, indirect, {}, follow);
    EXPECT(both.status == CollectionStatus::partial && both.failures.size() == 2 &&
           both.failures[1].error == ReachableError::indirect_branch &&
           both.failures[1].address == c + 12 && both.failures[1].source == c + 8);
}

// A one-image program: code at base, executable [base, base + 0x400).
struct Small {
    static constexpr std::uint64_t base = 0x1'0000;
    Module module{base, 0x1000};
    std::vector<ReachableImage> images() const { return {module.image({{base, base + 0x400}})}; }
    ReachableCode run(std::initializer_list<std::uint64_t> entries = {base},
                      const ReachableOptions& options = {},
                      std::initializer_list<ExternalLeaf> leaves = {}) const {
        const auto views = images();
        const std::vector<std::uint64_t> e(entries);
        const std::vector<ExternalLeaf> l(leaves);
        return collect_reachable_code(views, e, l, options);
    }
};

void instruction_failures() {
    constexpr std::uint64_t s = Small::base;
    EXPECT(a64::decode_register_effects(fadd_s).status == a64::EffectStatus::unknown);
    EXPECT(a64::decode_register_effects(movn_reserved).status == a64::EffectStatus::reserved);
    EXPECT(a64::decode_register_effects(ldr_post_overlap).status ==
           a64::EffectStatus::unpredictable);

    {
        Small p;
        p.module.word(s, nop);
        p.module.word(s + 4, fadd_s);
        p.module.word(s + 8, ok(a64::encode_ret()));
        const ReachableCode strict = p.run();
        EXPECT(only_failure(strict, ReachableError::unknown_instruction, s + 4));
        EXPECT(strict.failures[0].source == s && site_at(strict, s + 8) == nullptr);
        const ReachableCode assumed =
            p.run({s}, {.unknown_instructions = UnknownInstructionPolicy::assume_sequential});
        EXPECT(assumed.complete() && assumed.sites.size() == 3 &&
               assumed.statistics.assumed_sequential == 1 &&
               site_at(assumed, s + 4)->kind == SiteKind::assumed_sequential);
    }
    for (const auto& [word, error] :
         {std::pair{movn_reserved, ReachableError::reserved_instruction},
          std::pair{ldr_post_overlap, ReachableError::unpredictable_instruction},
          std::pair{retaa, ReachableError::unsupported_control_flow},
          std::pair{eret, ReachableError::unsupported_control_flow},
          std::pair{ok(a64::encode_br(8)), ReachableError::indirect_branch},
          std::pair{ok(a64::encode_blr(8)), ReachableError::indirect_call},
          std::pair{hlt, ReachableError::trap}, std::pair{udf, ReachableError::trap}}) {
        Small p;
        p.module.word(s, word);
        p.module.word(s + 4, ok(a64::encode_ret()));
        const ReachableCode code =
            p.run({s}, {.unknown_instructions = UnknownInstructionPolicy::assume_sequential});
        EXPECT(only_failure(code, error, s));
        EXPECT(code.sites.size() == 1 && code.sites[0].kind == SiteKind::failed &&
               code.failures[0].word == word);
    }
    {
        Small p;  // HINT space continues even when unknown words fail.
        p.module.word(s, paciasp);
        p.module.word(s + 4, ok(a64::encode_ret()));
        EXPECT(p.run().complete());
    }
    {
        Small p;
        p.module.word(s, udf);
        const ReachableCode code = p.run({s}, {.traps = TrapPolicy::terminal});
        EXPECT(code.complete() && code.sites[0].kind == SiteKind::trap);
    }
}

void target_failures() {
    constexpr std::uint64_t s = Small::base;
    {
        Small p;  // Branch into the non-executable part of the image.
        p.module.word(s, ok(a64::encode_b(s, s + 0x800)));
        const ReachableCode code = p.run();
        EXPECT(only_failure(code, ReachableError::target_not_executable, s));
        EXPECT(code.failures[0].target == s + 0x800 && code.sites[0].kind == SiteKind::branch);
    }
    {
        Small p;  // A conditional branch keeps its valid fallthrough.
        p.module.word(s, ok(a64::encode_b_cond(a64::Condition::eq, s, s + 0x800)));
        p.module.word(s + 4, ok(a64::encode_ret()));
        const ReachableCode code = p.run();
        EXPECT(only_failure(code, ReachableError::target_not_executable, s));
        EXPECT(site_at(code, s + 4) != nullptr);
    }
    {
        Small p;  // The last executable word falls off its range.
        p.module.word(s + 0x3fc, nop);
        const ReachableCode code = p.run({s + 0x3fc});
        EXPECT(only_failure(code, ReachableError::fallthrough_not_executable, s + 0x3fc));
        EXPECT(code.failures[0].target == s + 0x400);
    }
    {
        Small p;  // A leaf inside code takes precedence and is never decoded.
        p.module.word(s, ok(a64::encode_bl(s, s + 0x100)));
        p.module.word(s + 4, udf);
        p.module.word(s + 0x100, udf);
        const ReachableCode code = p.run({s}, {}, {{s + 0x100, false}});
        EXPECT(code.complete() && code.sites.size() == 1 &&
               has_edge(code, s, s + 0x100, EdgeKind::call, true));
    }
    {
        Small p;  // Fallthrough into a leaf ends the path.
        p.module.word(s, nop);
        p.module.word(s + 4, udf);
        EXPECT(p.run({s}, {}, {{s + 4, true}}).sites.size() == 1);
    }
}

void thunk_failures() {
    constexpr std::uint64_t s = Small::base;
    constexpr std::uint64_t slot = s + 0x808;
    {
        Small p;  // GOT slot outside every supplied image.
        thunk(p.module, s, s + 0x10'0008, true);
        const ReachableCode code = p.run();
        EXPECT(only_failure(code, ReachableError::thunk_slot_unreadable, s + 12));
        EXPECT(code.failures[0].slot == s + 0x10'0008 && code.thunks.empty());
        EXPECT(site_at(code, s + 12)->kind == SiteKind::failed);
    }
    for (const auto& [value, error] :
         {std::pair{std::uint64_t{0}, ReachableError::target_not_executable},
          std::pair{s + 0x102, ReachableError::misaligned_target}}) {
        Small p;
        thunk(p.module, s, slot, false);
        p.module.quad(slot, value);
        const ReachableCode code = p.run();
        EXPECT(only_failure(code, error, s + 8));
        EXPECT(code.failures[0].target == value && code.failures[0].slot == slot);
    }
    {
        Small p;  // Control entering after the ADRP has no known slot address.
        thunk(p.module, s, slot, true);
        p.module.quad(slot, s + 0x100);
        p.module.word(s + 0x100, ok(a64::encode_ret()));
        const ReachableCode forward = p.run({s, s + 4});
        EXPECT(only_failure(forward, ReachableError::thunk_interior_entry, s + 4));
        const ReachableCode reverse = p.run({s + 4, s});
        EXPECT(only_failure(reverse, ReachableError::indirect_branch, s + 12));
    }
    struct Shape {
        unsigned target_register;
        unsigned branch_register;
        std::uint32_t add_word;  // zero: no ADD
    };
    const std::uint32_t add_mismatch = ok(a64::encode_add_immediate(true, 16, 16, 0x10));
    const std::uint32_t add_matching = ok(a64::encode_add_immediate(true, 16, 16, 0x8));
    constexpr std::uint32_t adds_matching = 0xB1002210u;  // ADDS X16, X16, #8
    // Each shape breaks one operand relation, so the BR stays unresolved.
    for (const Shape shape : {Shape{17, 16, 0}, Shape{17, 17, add_mismatch},
                              Shape{16, 16, add_matching}, Shape{17, 17, adds_matching}}) {
        Small p;
        p.module.word(s, ok(a64::encode_adrp(16, s, slot)));
        p.module.word(s + 4, ok(a64::encode_load_unsigned_offset(
                                 a64::Opcode::ldr_literal_x, shape.target_register, 16, 0x8)));
        std::uint64_t branch = s + 8;
        if (shape.add_word != 0) {
            p.module.word(s + 8, shape.add_word);
            branch = s + 12;
        }
        p.module.word(branch, ok(a64::encode_br(shape.branch_register)));
        p.module.quad(slot, s + 0x100);
        const ReachableCode code = p.run();
        EXPECT(only_failure(code, ReachableError::indirect_branch, branch));
        EXPECT(code.thunks.empty() && site_at(code, s)->kind == SiteKind::sequential);
    }
}

void terminal_thunk_calls() {
    constexpr auto s = Small::base;
    for (const bool with_add : {false, true}) {
        Small p;
        p.module.word(s, ok(a64::encode_bl(s, s + 0x100)));
        p.module.word(s + 4, ok(a64::encode_br(3)));
        thunk(p.module, s + 0x100, s + 0x808, with_add);
        p.module.quad(s + 0x808, host_abort);
        const auto stopped = p.run({s}, {}, {{host_abort, false}});
        EXPECT(stopped.complete() && stopped.thunks.size() == 1);
        EXPECT(site_at(stopped, s + 4) == nullptr);
        EXPECT(site_at(stopped, s + 0x100) != nullptr);
        const auto returns = p.run({s}, {}, {{host_abort, true}});
        EXPECT(only_failure(returns, ReachableError::indirect_branch, s + 4));
        // A similar-looking sequence with a different target register does
        // not inherit the GOT target's terminal contract.
        p.module.word(s + 0x100 + (with_add ? 12 : 8), ok(a64::encode_br(4)));
        const auto mismatch = p.run({s}, {}, {{host_abort, false}});
        EXPECT(!mismatch.complete() && site_at(mismatch, s + 4) != nullptr);
    }
}

void input_validation() {
    std::vector<std::byte> bytes(0x100);
    const std::span<const std::byte> view(bytes);
    const auto setup = [&](std::vector<ReachableImage> images, ReachableError error,
                           std::vector<ExternalLeaf> leaves = {},
                           std::vector<std::uint64_t> entries = {0x1000}) {
        const ReachableCode code = collect_reachable_code(images, entries, leaves);
        EXPECT(code.status == CollectionStatus::invalid_input && code.setup_error.error == error &&
               code.sites.empty());
    };
    setup({{0x1000, {}, {}}}, ReachableError::invalid_image);
    setup({{~std::uint64_t{0} - 0x10, view, {}}}, ReachableError::invalid_image);
    setup({{0x1000, view, {{0x1000, 0x1100}}}, {0x10f0, view, {}}},
          ReachableError::overlapping_images);
    setup({{0x1000, view, {{0x1002, 0x1100}}}}, ReachableError::invalid_executable_range);
    setup({{0x1000, view, {{0x1000, 0x1104}}}}, ReachableError::invalid_executable_range);
    setup({{0x1000, view, {{0x1000, 0x1000}}}}, ReachableError::invalid_executable_range);
    setup({{0x1000, view, {{0x1000, 0x1080}, {0x1040, 0x1100}}}},
          ReachableError::invalid_executable_range);
    const std::vector<ReachableImage> good{{0x1000, view, {{0x1000, 0x1080}}}};
    setup(good, ReachableError::invalid_leaf, {{0x2002, true}});
    setup(good, ReachableError::invalid_leaf, {{0x2000, true}, {0x2000, false}});
    setup(good, ReachableError::invalid_entry, {}, {0x1002});
    setup(good, ReachableError::invalid_entry, {}, {0x1080});
    setup(good, ReachableError::invalid_entry, {{0x1000, true}});
}

void bounds() {
    constexpr std::uint64_t s = Small::base;
    Small p;
    for (std::uint64_t offset = 0; offset < 40; offset += 4) p.module.word(s + offset, nop);
    p.module.word(s + 40, ok(a64::encode_ret()));
    const ReachableCode limited = p.run({s}, {.max_sites = 3});
    EXPECT(only_failure(limited, ReachableError::resource_limit, s + 12));
    EXPECT(limited.sites.size() == 3);

    Small q;
    q.module.word(s, ok(a64::encode_br(1)));
    q.module.word(s + 4, ok(a64::encode_br(2)));
    const ReachableCode truncated = q.run({s, s + 4}, {.max_failures = 1});
    EXPECT(truncated.failures.size() == 1 && truncated.failures_truncated);
}

} // namespace

int main() {
    cross_module_program();
    assembled_fixture();
    instruction_failures();
    target_failures();
    thunk_failures();
    terminal_thunk_calls();
    input_validation();
    bounds();
    const std::array<std::uint32_t, 1> unknown{0xffffffff};
    ReachableImage single{0x1000, std::as_bytes(std::span(unknown)), {{0x1000, 0x1004}}};
    const std::array<std::uint64_t, 1> entry{0x1000};
    const auto no_storage = collect_reachable_code(std::span(&single, 1), entry, {},
                                                    {.max_failures = 0});
    EXPECT(!no_storage.complete() && no_storage.failures_truncated);
    const std::array<std::uint32_t, 6> thunk{
        0x90000010, 0xf9400a11, 0xd61f0220, 0, 0x2000, 0};
    ReachableImage thunk_image{0x1000, std::as_bytes(std::span(thunk)), {{0x1000, 0x100c}}};
    const std::array leaves{ExternalLeaf{0x2000, false}};
    const auto one_site = collect_reachable_code(std::span(&thunk_image, 1), entry, leaves,
                                                 {.max_sites = 1});
    EXPECT(!one_site.complete() && one_site.sites.size() == 1);
    const std::array<std::uint32_t, 3> overflow{0xb0000010, 0xf9400a11, 0xd61f0220};
    constexpr std::uint64_t high = 0xfffffffffffff000;
    ReachableImage high_image{high, std::as_bytes(std::span(overflow)), {{high, high + 12}}};
    const std::array high_entry{high};
    const auto invalid_page = collect_reachable_code(std::span(&high_image, 1), high_entry);
    EXPECT(!invalid_page.complete() && !invalid_page.failures.empty() &&
           invalid_page.failures[0].error == ReachableError::target_address_overflow);
    if (failures != 0) {
        std::fprintf(stderr, "%d reachable-code checks failed\n", failures);
        return 1;
    }
    std::puts("reachable-code tests passed");
    return 0;
}
