// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Workgroup atomic addition, returned old values, and implicit active-lane
// increments.
#include "metal_harness.h"
#include <bit>
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
  auto i = raw(opcode);
  for (auto [first, count, value] : fields)
    field(i, first, count, value);
  return i;
}
sn::Instruction load(unsigned d, unsigned index) {
  return op(0xa02,
            {{16, 8, d}, {54, 5, 3}, {38, 16, index * 4}, {64, 32, 0xf00}});
}
sn::Instruction s2r(unsigned d, unsigned selector) {
  return op(0x919, {{16, 8, d}, {72, 8, selector}});
}
sn::Instruction mask(unsigned d, unsigned source, std::uint32_t value,
                     unsigned predicate = 7) {
  return op(0x812, {{16, 8, d},
                    {24, 8, source},
                    {32, 32, value},
                    {64, 8, 255},
                    {72, 8, 0xc0},
                    {81, 3, predicate},
                    {87, 4, 15}});
}
sn::Instruction atoms(bool one, unsigned destination, unsigned address,
                      unsigned data, std::uint32_t displacement,
                      unsigned stride = 0, unsigned guard = 7) {
  auto i = op(one ? 0xf8c : 0x38c, {{16, 8, destination},
                                    {24, 8, address},
                                    {32, 8, data},
                                    {40, 24, displacement & 0xffffff},
                                    {73, 3, 0},
                                    {78, 2, stride},
                                    {87, 4, one ? 10u : 0u}});
  if (one) {
    field(i, 64, 6, 63);
    field(i, 91, 1, 1);
  }
  field(i, 12, 4, guard);
  return i;
}
Fixture fixture(bool one, bool distributed, unsigned stride,
                std::int32_t displacement, unsigned guard, unsigned destination,
                unsigned threads, std::uint32_t initial, std::uint32_t addend,
                bool unaligned) {
  const unsigned scale = stride ? 1u << (stride + 1) : 1;
  const unsigned ra = unaligned ? 5 : 4;
  const unsigned first_address =
      distributed ? 64 : (ra * scale + std::uint32_t(displacement)) & ~3u;
  std::vector<sn::Instruction> code{load(8, 0),
                                    load(9, 1),
                                    load(16, 2),
                                    s2r(0, 33),
                                    s2r(7, 0),
                                    op(0x20c, {{24, 8, 0},
                                               {32, 8, 255},
                                               {68, 4, 7},
                                               {76, 3, 2},
                                               {81, 3, 2},
                                               {84, 3, 7},
                                               {87, 4, 7}})};
  for (unsigned component = 0; component < 4; ++component)
    code.push_back(move(20 + component, initial));
  auto initialize = op(
      0x388, {{24, 8, 255}, {32, 8, 20}, {40, 24, first_address}, {73, 3, 6}});
  field(initialize, 12, 4, 2);
  code.push_back(initialize);
  code.push_back(raw(0xb1d));
  if (distributed)
    code.push_back(mask(2, 7, 3));
  else
    code.push_back(move(2, ra));
  code.push_back(move(3, addend));
  code.push_back(move(4, sentinel));
  code.push_back(mask(255, 7, 1, 0));
  code.push_back(atoms(one, destination, 2, one ? 129 : 3,
                       distributed ? 64 : std::uint32_t(displacement), stride,
                       guard));
  code.push_back(raw(0xb1d));
  // The native atomic address is a word address: low two byte bits are
  // discarded.
  code.push_back(move(18, distributed ? 64 : std::uint32_t(displacement)));
  code.push_back(op(0x824, {{16, 8, 6},
                            {24, 8, 2},
                            {32, 32, scale},
                            {64, 8, 18},
                            {81, 3, 7},
                            {87, 4, 15}}));
  code.push_back(mask(6, 6, 0xfffffffcu));
  code.push_back(op(0x984, {{16, 8, 5}, {24, 8, 6}, {73, 3, 4}}));
  code.push_back(op(0x225, {{16, 8, 10},
                            {24, 8, 0},
                            {32, 8, 16},
                            {64, 8, 8},
                            {81, 3, 7},
                            {87, 4, 15}}));
  code.push_back(op(0x386, {{24, 8, 10},
                            {32, 8, 4},
                            {72, 1, 1},
                            {73, 3, 6},
                            {77, 4, 5},
                            {84, 3, 1}}));
  code.push_back(exit_instruction());
  Fixture f(code, sn::Stage::compute);
  f.launch({threads, 1, 1}, 1024);
  return f;
}
bool active(unsigned lane, unsigned guard) {
  return guard == 7 || (guard == 0 && (lane & 1)) ||
         (guard == 8 && !(lane & 1));
}
void run(MetalHarness &gpu, bool one, bool distributed, unsigned stride,
         std::int32_t displacement, unsigned guard, unsigned destination,
         unsigned threads, std::uint32_t initial, std::uint32_t addend,
         bool unaligned = false) {
  auto f = fixture(one, distributed, stride, displacement, guard, destination,
                   threads, initial, addend, unaligned);
  auto compiled = gpu.compile_compute(f);
  MetalHarness::GuestMemory guest{base, Words(threads * 4, 0xdeadbeefu), {}};
  gpu.dispatch(compiled, f,
               {std::uint32_t(base), std::uint32_t(base >> 32), 16}, guest);
  const unsigned bins = distributed ? 4 : 1,
                 scale = stride ? 1u << (stride + 1) : 1;
  std::array<unsigned, 4> counts{};
  std::array<Words, 4> got_old, wanted_old;
  for (unsigned t = 0; t < threads; ++t)
    if (active(t % 32, guard))
      ++counts[distributed ? t % 4 : 0];
  for (unsigned bin = 0; bin < bins; ++bin)
    for (unsigned i = 0; i < counts[bin]; ++i)
      wanted_old[bin].push_back(initial + i * (one ? 1 : addend));
  for (unsigned t = 0; t < threads; ++t) {
    const unsigned bin = distributed ? t % 4 : 0, lane = t % 32;
    const auto *got = &guest.words[t * 4];
    const auto expected = initial + counts[bin] * (one ? 1 : addend);
    const auto address =
        distributed
            ? 64 + bin * 4
            : (((unaligned ? 5u : 4u) * scale + std::uint32_t(displacement)) &
               ~3u);
    if (got[1] != expected || got[2] != address || got[3] != lane) {
      std::fprintf(
          stderr,
          "Shared atomic one=%u bins=%u stride=%u displacement=%d guard=%u "
          "thread=%u final=%08x expected=%08x address=%u expected=%u\n",
          one, bins, stride, displacement, guard, t, got[1], expected, got[2],
          address);
      throw std::runtime_error("native shared atomic result or address");
    }
    if (!one && destination != 255 && active(lane, guard))
      got_old[bin].push_back(got[0]);
    else
      check(
          got[0] == sentinel,
          "discarded or inactive atomic result leaves its register unchanged");
  }
  if (!one && destination != 255)
    for (unsigned bin = 0; bin < bins; ++bin) {
      std::sort(got_old[bin].begin(), got_old[bin].end());
      std::sort(wanted_old[bin].begin(), wanted_old[bin].end());
      check(got_old[bin] == wanted_old[bin],
            "contended atomic returns each old value exactly once");
    }
  check(guest.table[1] == 0, "atomic result records stay inside guest memory");
}
void histogram(MetalHarness &gpu, bool one, unsigned threads, unsigned guard) {
  std::vector<sn::Instruction> code{
      load(8, 0),
      load(9, 1),
      load(16, 2),
      s2r(0, 33),
      s2r(7, 0),
      op(0x388, {{24, 8, 0}, {32, 8, 255}, {73, 3, 4}, {78, 2, 1}}),
      raw(0xb1d),
      op(0x824, {{16, 8, 6},
                 {24, 8, 0},
                 {32, 32, 4},
                 {64, 8, 255},
                 {81, 3, 7},
                 {87, 4, 15}}),
      op(0xb82,
         {{16, 8, 12}, {24, 8, 6}, {38, 16, 16}, {54, 5, 3}, {73, 3, 4}}),
      op(0x305, {{16, 8, 2},
                 {32, 8, 12},
                 {75, 2, 2},
                 {77, 1, 1},
                 {78, 2, 3},
                 {84, 2, 2}}),
      op(0x824, {{16, 8, 2},
                 {24, 8, 2},
                 {32, 32, 4},
                 {64, 8, 255},
                 {81, 3, 7},
                 {87, 4, 15}}),
      move(3, 1),
      move(4, sentinel),
      mask(255, 7, 1, 0),
      atoms(one, one ? 255 : 4, 2, one ? 129 : 3, 0, 0, guard),
      raw(0xb1d),
      op(0x984, {{16, 8, 5}, {24, 8, 2}, {73, 3, 4}}),
      op(0x202, {{16, 8, 6}, {32, 8, 2}, {64, 32, 0xf00}}),
      op(0x225, {{16, 8, 10},
                 {24, 8, 0},
                 {32, 8, 16},
                 {64, 8, 8},
                 {81, 3, 7},
                 {87, 4, 15}}),
      op(0x386, {{24, 8, 10},
                 {32, 8, 4},
                 {72, 1, 1},
                 {73, 3, 6},
                 {77, 4, 5},
                 {84, 3, 1}}),
      exit_instruction()};
  Fixture f(code, sn::Stage::compute);
  f.launch({threads, 1, 1}, 512);
  auto compiled = gpu.compile_compute(f);
  check(compiled.artifact.source.find("native_shared_word_atomic_v1") !=
            std::string::npos,
        "native shared atomic policy is versioned in emitted source");
  Words input{std::uint32_t(base), std::uint32_t(base >> 32), 16, 0};
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    input.resize(4);
    std::array<unsigned, 32> counts{};
    for (unsigned t = 0; t < threads; ++t) {
      const unsigned bin = (t * 7 + repeat * 5) % 32;
      input.push_back(std::bit_cast<std::uint32_t>(float(bin) + 0.75f));
      if (active(t % 32, guard))
        ++counts[bin];
    }
    MetalHarness::GuestMemory guest{base, Words(threads * 4, 0xdeadbeefu), {}};
    gpu.dispatch(compiled, f, input, guest);
    std::array<Words, 32> old;
    for (unsigned t = 0; t < threads; ++t) {
      const unsigned bin = (t * 7 + repeat * 5) % 32;
      const auto *got = &guest.words[t * 4];
      check(got[1] == counts[bin] && got[2] == bin * 4 && got[3] == t % 32,
            "runtime float histogram counts and word addresses");
      if (!one && active(t % 32, guard))
        old[bin].push_back(got[0]);
      else
        check(
            got[0] == sentinel,
            "runtime implicit or inactive increment preserves result register");
    }
    if (!one)
      for (unsigned bin = 0; bin < 32; ++bin) {
        std::sort(old[bin].begin(), old[bin].end());
        Words expected;
        for (unsigned i = 0; i < counts[bin]; ++i)
          expected.push_back(i);
        check(old[bin] == expected,
              "runtime contended histogram returns every old value");
      }
    check(guest.table[1] == 0,
          "runtime histogram records stay inside guest memory");
  }
}
void refused(sn::Instruction i, const char *name,
             sn::Stage stage = sn::Stage::compute) {
  Fixture f({i, exit_instruction()}, stage);
  if (stage == sn::Stage::compute)
    f.launch({32, 1, 1}, 1024);
  check(
      std::holds_alternative<ir::Diagnostic>(sn::translate_native(f.program())),
      name);
}
} // namespace
int main() {
  @autoreleasepool {
    try {
      MetalHarness gpu;
      unsigned cases = 0;
      for (bool one : {false, true})
        for (unsigned guard : {7u, 0u, 8u}) {
          run(gpu, one, false, 0, 64, guard, one ? 255 : 4, 32, 17, 3);
          ++cases;
          run(gpu, one, true, 1, 64, guard, 255, 64, 0xfffffff0u, 1);
          ++cases;
        }
      for (unsigned stride : {0u, 1u, 2u, 3u}) {
        run(gpu, false, false, stride, 64, 7, 4, 32, 0xfffffff0u, 3);
        ++cases;
      }
      run(gpu, false, false, 0, -4, 7, 4, 32, 17, 0);
      ++cases;
      run(gpu, true, false, 0, 64, 0, 255, 32, 17, 0, true);
      ++cases;
      run(gpu, false, false, 0, 64, 7, 255, 1, 17, 5);
      ++cases;
      for (bool one : {false, true})
        for (unsigned threads : {32u, 64u})
          for (unsigned guard : {7u, 0u}) {
            histogram(gpu, one, threads, guard);
            ++cases;
          }
      auto bad = atoms(false, 4, 255, 3, 64);
      field(bad, 87, 4, 1);
      refused(bad, "unimplemented shared atomic operation is refused");
      bad = atoms(false, 4, 255, 3, 64);
      field(bad, 73, 3, 1);
      refused(bad, "unimplemented shared atomic width is refused");
      bad = atoms(false, 4, 255, 3, 64);
      field(bad, 91, 1, 1);
      refused(bad, "uniform regular shared atomic is refused");
      bad = atoms(true, 255, 255, 129, 64);
      field(bad, 87, 4, 0);
      refused(bad, "implicit increment operation field is checked");
      bad = atoms(true, 4, 255, 129, 64);
      refused(bad, "implicit increment cannot return an old value");
      bad = atoms(true, 255, 255, 129, 64);
      field(bad, 91, 1, 0);
      refused(bad, "implicit increment encoding bit is checked");
      bad = atoms(true, 255, 255, 129, 64);
      field(bad, 64, 6, 1);
      refused(bad, "unimplemented uniform increment address is refused");
      refused(atoms(false, 4, 255, 3, 64),
              "shared atomics require compute allocation", sn::Stage::fragment);
      refused(atoms(false, 4, 255, 3, 1024),
              "constant native atomic address cannot exceed allocation");
      std::printf("Native shared atomic: %u GPU cases passed\n", cases);
      return 0;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
}
