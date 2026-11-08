// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/compat/observer.h>
#include <armigo/aarch64/instruction.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

namespace {
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)

using namespace armigo::compat;

// Registers and a sparse byte map standing in for guest memory.
class FakeState final : public ObserveState {
public:
    std::array<std::uint64_t, 31> registers{};
    std::uint64_t stack = 0;
    std::map<std::uint64_t, std::uint8_t> bytes;

    void put(std::uint64_t address, const void* data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) bytes[address + i] = static_cast<const std::uint8_t*>(data)[i];
    }
    std::uint64_t x(unsigned index) const noexcept override { return registers[index]; }
    std::uint64_t sp() const noexcept override { return stack; }
    bool read(std::uint64_t address, std::span<std::byte> out) const noexcept override {
        for (std::size_t i = 0; i < out.size(); ++i) {
            const auto found = bytes.find(address + i);
            if (found == bytes.end()) return false;
            out[i] = std::byte{found->second};
        }
        return true;
    }
};

std::optional<std::uint64_t> value_of(const char* text, const ObserveState& state) {
    auto parsed = parse_observe_points(std::string("0x10: ") + text);
    if (!std::holds_alternative<std::vector<ObservePoint>>(parsed)) return std::nullopt;
    return evaluate(std::get<std::vector<ObservePoint>>(parsed)[0].items[0].expression, state);
}

bool refused(const char* text) {
    return std::holds_alternative<std::string>(parse_observe_points(text));
}

void test_parse_and_evaluate() {
    auto parsed = parse_observe_points(
        "0x41230 if !(w0&1): x19, b[x26], s:[x19+0x260], [sp+0x60+b[x26]*16]; 0x5a0c4/8 if w1 != 1: x0, regs");
    CHECK(std::holds_alternative<std::vector<ObservePoint>>(parsed));
    if (!std::holds_alternative<std::vector<ObservePoint>>(parsed)) return;
    const auto& points = std::get<std::vector<ObservePoint>>(parsed);
    CHECK(points.size() == 2);
    CHECK(points[0].offset == 0x41230 && points[0].items.size() == 4 && points[0].limit == 64);
    CHECK(points[0].items[2].string && points[0].items[2].text == "s:[x19+0x260]");
    CHECK(points[1].offset == 0x5a0c4 && points[1].limit == 8 && points[1].items[1].registers);

    FakeState state;
    state.registers[0] = 0x100000002;
    state.registers[1] = 6;
    state.registers[19] = 0x5000;
    state.registers[26] = 0x1202;
    state.stack = 0x8000;
    const char name[] = "Npc";
    const std::uint64_t name_address = 0x6000, entry = 0xabcdef;
    state.put(0x5260, &name_address, 8);
    state.put(name_address, name, sizeof name);
    state.put(0x8000 + 0x60 + 2 * 16, &entry, 8);
    // The condition reads w0 only; bit 0 of x0 is clear.
    CHECK(evaluate(*points[0].condition, state) == 1u);
    CHECK(describe_hit(points[0], state) == "x19=0x5000 b[x26]=? s:[x19+0x260]='Npc' [sp+0x60+b[x26]*16]=?");
    state.put(0x1202, "\x02", 1);
    CHECK(describe_hit(points[0], state) ==
          "x19=0x5000 b[x26]=0x2 s:[x19+0x260]='Npc' [sp+0x60+b[x26]*16]=0xabcdef");
    state.registers[0] = 3;
    CHECK(!describe_hit(points[0], state));
    CHECK(describe_hit(points[1], state).has_value());
    state.registers[1] = 1;
    CHECK(!describe_hit(points[1], state));

    CHECK(value_of("1+2*3", state) == 7u);
    CHECK(value_of("(1+2)*3", state) == 9u);
    CHECK(value_of("x19>>4&0xff", state) == 0u);
    CHECK(value_of("1<<4|1", state) == 17u);
    CHECK(value_of("-1", state) == ~std::uint64_t{0});
    CHECK(value_of("~0 == -1", state) == 1u);
    CHECK(value_of("w0 != 3", state) == 0u);
    CHECK(value_of("2 < 3", state) == 1u);
    CHECK(value_of("w[0x5260]", state) == 0x6000u);
    CHECK(value_of("h[0x6000]", state) == 0x704eu);
    CHECK(value_of("[0x7000]", state) == std::nullopt);

    auto lines = parse_observe_points("0x10/2 if x0: x1,\n x2\n; 0x20 : x3 ");
    CHECK(std::holds_alternative<std::vector<ObservePoint>>(lines) &&
          std::get<std::vector<ObservePoint>>(lines).size() == 2 &&
          std::get<std::vector<ObservePoint>>(lines)[0].items.size() == 2);
    CHECK(refused("0x41231"));
    CHECK(refused("0x10: x31"));
    CHECK(refused("0x10: [x0"));
    CHECK(refused("0x10: x0 x1"));
    CHECK(refused("0x10/many: x0"));
    CHECK(refused("0x10: x0; 0x10: x1"));
    CHECK(refused(" ; "));
}

void test_hooks() {
    constexpr std::uint64_t trampoline = 0x5000000, gate = 0x5100000;
    const auto b = [](std::uint64_t from, std::uint64_t to) { return armigo::aarch64::encode_b(from, to).word; };

    // tbz w0, #0, +0x2c keeps its test against a branch to the original target.
    auto hook = make_observe_hook(0x36000160, 0x41230, trampoline, {gate, 64});
    CHECK(std::holds_alternative<ObserveHook>(hook));
    if (const auto* made = std::get_if<ObserveHook>(&hook)) {
        CHECK(made->replacement_instruction == b(0x41230, trampoline));
        CHECK(made->trampoline[0] == 0xf81f0ffeu && made->trampoline[2] == 0xf84107feu);
        CHECK(made->trampoline[1] == armigo::aarch64::encode_bl(trampoline + 4, gate).word);
        CHECK(made->trampoline[3] == 0x36000040u);
        CHECK(made->trampoline[4] == b(trampoline + 16, 0x41234));
        CHECK(made->trampoline[5] == b(trampoline + 20, 0x4125c));
        CHECK(made->site.return_address == trampoline + 8 && made->site.call_return_address == 0x41234 &&
              made->site.target_address == 0x41234);
    }
    // cbz x0, +0x900.
    hook = make_observe_hook(0xb4004800, 0x62000, trampoline, {gate, 64});
    if (const auto* made = std::get_if<ObserveHook>(&hook)) {
        CHECK(made->trampoline[3] == 0xb4000040u);
        CHECK(made->trampoline[5] == b(trampoline + 20, 0x62900));
    } else {
        CHECK(false);
    }
    // b.ne -0x40.
    hook = make_observe_hook(0x54fffe01, 0x1000, trampoline, {gate, 64});
    if (const auto* made = std::get_if<ObserveHook>(&hook)) {
        CHECK(made->trampoline[3] == 0x54000041u);
        CHECK(made->trampoline[5] == b(trampoline + 20, 0x1000 - 0x40));
    } else {
        CHECK(false);
    }
    // mov w14, w25 and stp x29, x30, [sp, #-0x20]! run as they are.
    for (const std::uint32_t word : {0x2a1903eeu, 0xa9be7bfdu}) {
        hook = make_observe_hook(word, 0x4122c, trampoline, {gate, 64});
        if (const auto* made = std::get_if<ObserveHook>(&hook)) {
            CHECK(made->trampoline[3] == word && made->trampoline[4] == b(trampoline + 16, 0x41230) &&
                  made->trampoline[5] == 0);
        } else {
            CHECK(false);
        }
    }
    // bl, b, adrp, ldr literal, ret, svc, mrs, ldxr.
    for (const std::uint32_t word : {0x94000010u, 0x14000010u, 0x90000000u, 0x58000040u, 0xd65f03c0u,
                                     0xd4000001u, 0xd53b4200u, 0xc85f7c00u})
        CHECK(std::holds_alternative<std::string>(make_observe_hook(word, 0x1000, trampoline, {gate, 64})));
    // Out of branch range, misaligned, overlapping.
    CHECK(std::holds_alternative<std::string>(make_observe_hook(0x2a1903ee, 0x1000, 0x40001000, {gate, 64})));
    CHECK(std::holds_alternative<std::string>(make_observe_hook(0x2a1903ee, 0x1002, trampoline, {gate, 64})));
    CHECK(std::holds_alternative<std::string>(make_observe_hook(0x2a1903ee, 0x1000, 0x1004, {gate, 64})));
}

} // namespace

int main() {
    test_parse_and_evaluate();
    test_hooks();
    if (failures) std::fprintf(stderr, "%u failures\n", failures);
    return failures ? 1 : 0;
}
