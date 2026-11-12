// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Synthetic lane masks and predicate results for native MATCH.ANY and
// MATCH.ALL.
#include "metal_harness.h"
#include <cstdio>
#include <initializer_list>
#include <tuple>

namespace sn = armigo::shader_native;
namespace ir = sn::ir;
using namespace sn::test;
namespace {
using Field = std::tuple<unsigned, unsigned, std::uint64_t>;
using Words = std::vector<std::uint32_t>;
constexpr std::uint64_t base = 0x7dfff00000ull;
constexpr std::uint32_t sentinel = 0xa5a5a5a5u;
sn::Instruction op(unsigned opcode, std::initializer_list<Field> fields) {
  auto instruction = raw(opcode);
  for (auto [first, count, value] : fields)
    field(instruction, first, count, value);
  return instruction;
}
sn::Instruction load(unsigned d, unsigned index) {
  return op(0xa02,
            {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}});
}
sn::Instruction s2r(unsigned d, unsigned selector) {
  return op(0x919, {{16, 8, d}, {72, 8, selector}});
}
sn::Instruction copy(unsigned d, unsigned source) {
  auto instruction = raw(0x202);
  field(instruction, 16, 8, d);
  instruction.words[1] = source;
  instruction.words[2] = 0xf00;
  return instruction;
}
sn::Instruction mask(unsigned d, unsigned source, std::uint32_t bits,
                     unsigned predicate = 7) {
  return op(0x812, {{16, 8, d},
                    {24, 8, source},
                    {32, 32, bits},
                    {64, 8, 255},
                    {72, 8, 0xc0},
                    {81, 3, predicate},
                    {87, 4, 15}});
}
sn::Instruction match(unsigned d, unsigned source, bool wide, bool any,
                      unsigned predicate, unsigned guard = 7) {
  auto instruction = op(0x3a1, {{16, 8, d},
                                {24, 8, source},
                                {73, 1, wide},
                                {79, 1, any},
                                {81, 3, predicate}});
  field(instruction, 12, 4, guard);
  return instruction;
}
Fixture fixture(bool wide, bool any, unsigned low_mask, unsigned high_mask,
                unsigned guard, unsigned destination, unsigned threads) {
  std::vector<sn::Instruction> code{load(8, 0),
                                    load(9, 1),
                                    load(16, 2),
                                    s2r(0, 0),
                                    s2r(1, 33),
                                    mask(2, 1, low_mask),
                                    mask(3, 1, high_mask),
                                    move(4, sentinel),
                                    mask(255, 0, 1, 0),
                                    op(0x20c, {{24, 8, 255},
                                               {32, 8, 255},
                                               {68, 4, 7},
                                               {76, 3, 7},
                                               {81, 3, 1},
                                               {84, 3, 7},
                                               {87, 4, 7}}),
                                    match(destination, 2, wide, any, 1, guard)};
  if (destination != 255 && destination != 2 && destination != 3 &&
      destination != 4)
    code.push_back(copy(4, destination));
  code.push_back(
      op(0x803, {{16, 8, 5}, {24, 8, 255}, {32, 32, 0x7f}, {76, 2, 0}}));
  for (unsigned component = 0; component < 4; ++component)
    code.push_back(copy(20 + component, 2 + component));
  // Each thread writes one independent 16-byte result record.
  code.push_back(op(0x225, {{16, 8, 10},
                            {24, 8, 1},
                            {32, 8, 16},
                            {64, 8, 8},
                            {81, 3, 7},
                            {87, 4, 15}}));
  code.push_back(op(0x386, {{24, 8, 10},
                            {32, 8, 20},
                            {72, 1, 1},
                            {73, 3, 6},
                            {77, 4, 5},
                            {84, 3, 1}}));
  code.push_back(exit_instruction());
  Fixture result(code, sn::Stage::compute);
  result.launch({threads, 1, 1}, 0);
  return result;
}
bool participates(unsigned lane, unsigned guard) {
  return guard == 7 || (guard == 0 && (lane & 1)) ||
         (guard == 8 && !(lane & 1));
}
void run(MetalHarness &gpu, bool wide, bool any, unsigned low_mask,
         unsigned high_mask, unsigned guard, unsigned destination,
         unsigned threads) {
  auto program =
      fixture(wide, any, low_mask, high_mask, guard, destination, threads);
  auto compiled = gpu.compile_compute(program);
  MetalHarness::GuestMemory guest{base, Words(threads * 4, 0xdeadbeefu), {}};
  gpu.dispatch(compiled, program,
               {std::uint32_t(base), std::uint32_t(base >> 32), 16}, guest);
  for (unsigned thread = 0; thread < threads; ++thread) {
    const unsigned first = (thread / 32) * 32, lane = thread % 32,
                   end = std::min(first + 32, threads);
    const auto low = thread & low_mask, high = thread & high_mask;
    std::uint32_t active = 0, same = 0;
    for (unsigned other = first; other < end; ++other)
      if (participates(other % 32, guard)) {
        active |= 1u << (other % 32);
        if ((other & low_mask) == low && (!wide || (other & high_mask) == high))
          same |= 1u << (other % 32);
      }
    const bool enabled = participates(lane, guard), all = same == active;
    const auto result = any ? same : all ? active : 0u;
    auto expected =
        std::array<std::uint32_t, 4>{low, high, sentinel, (lane & 1) | 2u};
    if (enabled) {
      if (destination == 2)
        expected[0] = result;
      else if (destination == 3)
        expected[1] = result;
      else if (destination != 255)
        expected[2] = result;
      if (!any && !all)
        expected[3] &= ~2u;
    }
    const auto *got = &guest.words[thread * 4];
    if (!std::equal(expected.begin(), expected.end(), got)) {
      std::fprintf(
          stderr,
          "MATCH wide=%u any=%u masks=%x/%x guard=%u destination=%u threads=%u "
          "thread=%u: expected %08x %08x %08x %08x, got %08x %08x %08x %08x\n",
          wide, any, low_mask, high_mask, guard, destination, threads, thread,
          expected[0], expected[1], expected[2], expected[3], got[0], got[1],
          got[2], got[3]);
      throw std::runtime_error("native match lane mask or predicate");
    }
  }
  check(guest.table[1] == 0, "match stores stay inside their guest window");
}
unsigned fragments(MetalHarness &gpu) {
  unsigned cases = 0;
  for (bool wide : {false, true})
    for (bool any : {false, true})
      for (unsigned low_mask : {0u, 3u, 31u}) {
        Fixture program(
            {s2r(2, 0), mask(10, 2, low_mask), mask(11, 2, 4),
             op(0x20c, {{24, 8, 255},
                        {32, 8, 255},
                        {68, 4, 7},
                        {76, 3, 7},
                        {81, 3, 1},
                        {84, 3, 7},
                        {87, 4, 7}}),
             match(0, 10, wide, any, 1),
             op(0x806, {{16, 8, 1}, {72, 2, 0}, {81, 3, 7}, {87, 4, 7}}),
             op(0x803, {{16, 8, 3}, {24, 8, 255}, {32, 32, 0x7f}, {76, 2, 0}}),
             exit_instruction()},
            sn::Stage::fragment);
        put(program.code, 0x48, 0xf);
        for (const auto &pixel : gpu.grid(gpu.compile(program), {}, 4, 4)) {
          const auto active = pixel[1], lane = pixel[2];
          check(lane < 32, "fragment match lane index");
          std::uint32_t same = 0;
          for (unsigned other = 0; other < 32; ++other)
            if ((active & (1u << other)) &&
                (other & low_mask) == (lane & low_mask) &&
                (!wide || (other & 4) == (lane & 4)))
              same |= 1u << other;
          const bool all = same == active;
          const auto result = any ? same : all ? active : 0u;
          check(pixel[0] == result && (pixel[3] & 2) == (any || all ? 2u : 0u),
                "fragment match follows its active lanes and raw values");
        }
        ++cases;
      }
  return cases;
}
void refused(sn::Instruction instruction, sn::Stage stage, const char *name) {
  Fixture program({instruction, exit_instruction()}, stage);
  check(std::holds_alternative<ir::Diagnostic>(
            sn::translate_native(program.program())),
        name);
}
} // namespace
int main() {
  @autoreleasepool {
    try {
      const auto known = match(4, 2, true, true, 1);
      check(known.words[0] == 0x020473a1u && known.words[1] == 0 &&
                known.words[2] == 0x00028200u && known.words[3] == 0,
            "independent match field encoding");
      MetalHarness gpu;
      unsigned cases = 0;
      for (bool wide : {false, true})
        for (bool any : {false, true})
          for (unsigned low_mask : {0u, 3u, 31u})
            for (unsigned guard : {7u, 0u, 8u}) {
              run(gpu, wide, any, low_mask, 4, guard, 4, 32);
              ++cases;
            }
      for (bool wide : {false, true})
        for (bool any : {false, true}) {
          for (unsigned threads : {1u, 16u, 31u, 64u}) {
            run(gpu, wide, any, 3, 4, 7, 4, threads);
            ++cases;
          }
          for (unsigned destination : {2u, 3u, 254u, 255u}) {
            run(gpu, wide, any, 3, 4, 7, destination, 32);
            ++cases;
          }
          run(gpu, wide, any, 0, 0, 0, 255, 32);
          ++cases;
          run(gpu, wide, any, 3, 4, 15, 4, 32);
          ++cases;
        }
      cases += fragments(gpu);
      auto bad = match(4, 2, false, true, 7);
      field(bad, 72, 1, 1);
      refused(bad, sn::Stage::compute, "match reserved mode is refused");
      bad = match(4, 2, false, false, 1);
      field(bad, 84, 1, 1);
      refused(bad, sn::Stage::compute, "match predicate extension is refused");
      bad = match(4, 2, false, true, 7);
      field(bad, 91, 1, 1);
      refused(bad, sn::Stage::compute, "match uniform datapath is refused");
      refused(match(4, 2, false, true, 7), sn::Stage::vertex,
              "match needs a fragment or workgroup stage");
      refused(match(4, 255, true, true, 7), sn::Stage::compute,
              "match source pair cannot overrun the register file");
      std::printf("Native match: %u GPU cases passed\n", cases);
      return 0;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
