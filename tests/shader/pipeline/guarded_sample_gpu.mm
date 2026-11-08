// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Offscreen checks for guarded texture taps on a 4x6 Metal 4 target.
// Eight taps, each under its own freshly loaded predicate, are enough for
// the hoister to move explicit-LOD, gradient and gather taps. A reference
// program is the same words with a predicated local store opening every
// guarded run, so those taps stay inside the guard, and both forms must draw
// identical pixels. Implicit-LOD, bias and depth-compare samples take a
// coarser mip on a partial quad once they leave the guard, so the hoister
// must leave them in place. Gathers read the base level whatever the quad's
// slopes. A title's opt-in moves implicit-LOD and bias samples too; on a
// one-level texture whose minification and magnification filters match, that
// move leaves every pixel unchanged. Synthetic fragment words only.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipeline/phases.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace armigo::shader::pipeline;

constexpr NSUInteger kWidth = 4;
constexpr NSUInteger kHeight = 6;
constexpr NSUInteger kPixelBytes = 4 * sizeof(std::uint32_t);
constexpr unsigned kTaps = 8;
constexpr std::uint32_t kCbufBytes = 256;
// Bit x + y * kWidth of a 4x6 image. Quad occupancies, in row-major quad order:
// 0, 1, 2 (horizontal), 3, 4, 2 (vertical).
constexpr std::uint32_t kActiveMask = 0x00774F04u;
// 0.05 * 16 texels is rho 0.8. Implicit LOD rounds to level 0; a +1 bias
// rounds to level 1. The inactive step below still saturates the chain.
constexpr std::uint32_t kScaleBits = 0x3d4ccccdu;
constexpr std::uint32_t kBigBits = 0x40800000u; // 4, steep inactive slope
constexpr std::uint32_t kBiasBits = 0x3f800000u;  // 1.0
constexpr std::uint32_t kRefBits = 0x3f000000u;   // 0.5
constexpr std::uint32_t kConstant = 0x3e800000u;  // 0.25, texel-fetch false color
constexpr std::uint32_t kFetchColor = 0x3f000000u;

// Level colors are exact multiples of 1/8 so eight summed taps stay exact.
constexpr std::array<std::uint32_t, 5> kLevelBits{
    0x3e000000u, // 0.125
    0x3e800000u, // 0.25
    0x3ec00000u, // 0.375
    0x3f000000u, // 0.5
    0x3f800000u, // 1
};
constexpr std::array<float, 5> kLevelDepth{0.10f, 0.20f, 0.70f, 0.80f, 0.90f};

const char *vertex_source =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "vertex float4 armigo_vertex_main(uint vid [[vertex_id]]) {\n"
    "  const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};\n"
    "  return float4(positions[vid], 0.0, 1.0);\n"
    "}\n";

enum class Mode {
  level,
  implicit,
  bias,
  gradient,
  gather,
  compare,
  compare_gather,
  fetch,
};

const char *mode_name(Mode mode) {
  switch (mode) {
  case Mode::level: return "level zero";
  case Mode::implicit: return "implicit lod";
  case Mode::bias: return "bias";
  case Mode::gradient: return "gradient";
  case Mode::gather: return "gather";
  case Mode::compare: return "compare";
  case Mode::compare_gather: return "compare gather";
  case Mode::fetch: return "texel fetch";
  }
  return "unknown";
}

bool compares(Mode mode) { return mode == Mode::compare || mode == Mode::compare_gather; }
// The modes whose derivatives come from the quad stay under their guards.
bool hoists(Mode mode) { return mode != Mode::implicit && mode != Mode::bias && mode != Mode::compare; }

unsigned quad_occupancy(unsigned quad_x, unsigned quad_y) {
  unsigned count = 0;
  for (unsigned ly = 0; ly < 2; ++ly)
    for (unsigned lx = 0; lx < 2; ++lx) {
      const unsigned index = (quad_x * 2 + lx) + (quad_y * 2 + ly) * kWidth;
      count += (kActiveMask >> index) & 1u;
    }
  return count;
}

bool lane_active(NSUInteger x, NSUInteger y) {
  return ((kActiveMask >> (x + y * kWidth)) & 1u) != 0;
}

template <class T> T take(Result<T> result) {
  if (const auto *error = std::get_if<Diagnostic>(&result))
    throw std::runtime_error("phase=" + std::to_string(unsigned(error->phase)) +
                             " pc=" + std::to_string(error->pc.value_or(0)) + ": " + error->message);
  return std::get<T>(std::move(result));
}

TextureDescriptor color_descriptor() {
  return {{TextureDimension::d2, false, false}, ScalarType::f32, ComponentFormat::floating,
          true, false, 1, "synthetic-format-f32"};
}

TextureDescriptor depth_descriptor() {
  return {{TextureDimension::d2, false, false}, ScalarType::f32, ComponentFormat::depth,
          true, true, 0, "synthetic-shadow-f32"};
}

struct Facts final : FactSource {
  bool compare{false};
  bool local_memory{false};
  std::map<FactKey, FactValue> values;
  Facts(bool compare_in, bool local_in) : compare(compare_in), local_memory(local_in) {
    values[{FactKind::fragment_output_type}] = IoType{ScalarType::f32, 4};
    values[{FactKind::constant_buffer_bound_mask}] = std::uint32_t{1};
    values[{FactKind::constant_buffer_size, 0}] = kCbufBytes;
    if (local_memory) values[{FactKind::local_memory_bytes}] = std::uint32_t{64};
    values[fact_query::texture({0, 16}).key()] = compare ? depth_descriptor() : color_descriptor();
  }
  Knowledge<FactValue> lookup(FactKey key) const override {
    if (auto found = values.find(key); found != values.end())
      return Known<FactValue>{found->second, {OriginKind::explicit_option, "synthetic-guarded-sample-gpu"}};
    return Unknown{"Synthetic fact unavailable"};
  }
};

MetalProfile profile() {
  return {"synthetic-metal4", 1, 31, 128, 16, true, false, true, false, false, false};
}

std::uint64_t pattern(std::string_view bits) {
  std::uint64_t word{};
  for (unsigned i = 0; i < bits.size(); ++i)
    if (bits[i] == '1') word |= 1ull << (63 - i);
  return word;
}
std::uint64_t place(unsigned first, unsigned width, std::uint64_t value) {
  return (value & ((1ull << width) - 1)) << first;
}
std::uint64_t guarded(std::uint64_t word, unsigned predicate) {
  return (word & ~(15ull << 16)) | (std::uint64_t{predicate} << 16);
}
std::uint64_t mov32i(unsigned dest, std::uint32_t bits) {
  return guarded(0x0100000000000000ull | dest | (std::uint64_t{bits} << 20), 7);
}
std::uint64_t ldc(unsigned dest, unsigned byte_offset) {
  return pattern("1110111110010x") | dest | place(8, 8, 255) | place(16, 3, 7) |
         place(20, 16, byte_offset) | place(48, 3, 4);
}
std::uint64_t ipa_position(unsigned dest, unsigned axis) {
  return 0xe000000000070000ull | dest | (255ull << 8) | (255ull << 20) |
         (std::uint64_t{0x70 + axis * 4} << 28) | (255ull << 39) | (7ull << 47);
}
// P0 = left < right. The complement lands in P1.
std::uint64_t fsetp_less(unsigned left, unsigned right) {
  return pattern("010110111011") | place(0, 3, 1) | place(3, 3, 0) | place(8, 8, left) |
         place(16, 3, 7) | place(20, 8, right) | place(39, 3, 7) | place(45, 2, 3) |
         place(48, 4, 1);
}
// P0 = left < right (unsigned). The complement lands in P1, not over P0.
std::uint64_t isetp_less(unsigned left, unsigned right) {
  return pattern("010110110110") | place(0, 3, 1) | place(8, 8, left) | place(16, 3, 7) |
         place(20, 8, right) | place(39, 3, 7) | place(49, 3, 1);
}
std::uint64_t f2i_trunc(unsigned dest, unsigned src) {
  return pattern("0101110010110") | dest | place(8, 2, 2) | place(10, 2, 2) | place(16, 3, 7) |
         place(20, 8, src) | place(39, 2, 3);
}
std::uint64_t i2f(unsigned dest, unsigned src) {
  return pattern("0101110010111") | dest | place(8, 2, 2) | place(10, 2, 2) | place(16, 3, 7) |
         place(20, 8, src);
}
std::uint64_t fmul(unsigned dest, unsigned left, unsigned right) {
  return 0x5c68000000070000ull | dest | (std::uint64_t{left} << 8) | (std::uint64_t{right} << 20);
}
std::uint64_t fadd(unsigned dest, unsigned left, unsigned right, bool negate_right = false) {
  return pattern("0101110001011") | dest | (std::uint64_t{left} << 8) | (7ull << 16) |
         (std::uint64_t{right} << 20) | (std::uint64_t{negate_right} << 45);
}
std::uint64_t iadd(unsigned dest, unsigned left, unsigned right) {
  return pattern("0101110000010") | dest | (std::uint64_t{left} << 8) | (7ull << 16) |
         (std::uint64_t{right} << 20);
}
std::uint64_t imul(unsigned dest, unsigned left, unsigned right) {
  return pattern("0101110000111") | dest | (std::uint64_t{left} << 8) | (7ull << 16) |
         (std::uint64_t{right} << 20);
}
std::uint64_t shl_imm(unsigned dest, unsigned src, unsigned count) {
  return pattern("0011100001001") | dest | (std::uint64_t{src} << 8) | (7ull << 16) |
         (std::uint64_t{count} << 20) | (1ull << 39);
}
std::uint64_t shr(unsigned dest, unsigned src, unsigned count) {
  return pattern("0101110000101") | dest | (std::uint64_t{src} << 8) | (7ull << 16) |
         (std::uint64_t{count} << 20);
}
// LOP32I. operation 0 is AND, 2 is XOR. The predicate destination is PT.
std::uint64_t lop32i(unsigned dest, unsigned src, unsigned operation, std::uint32_t literal) {
  return pattern("000001") | dest | place(8, 8, src) | place(16, 3, 7) | place(20, 32, literal) |
         place(53, 2, operation);
}
// STL of one b32 word at a byte displacement from RZ.
std::uint64_t stl(unsigned value, unsigned offset, unsigned predicate) {
  return pattern("1110111101010x") | value | place(8, 8, 255) | place(16, 3, predicate) |
         place(20, 24, offset) | place(48, 3, 4);
}
// TEX_B, 2D, mask R. lod 0 is implicit, 2 is bias. The handle is Rb.
std::uint64_t texb(unsigned ra, unsigned rb, unsigned dest, unsigned predicate, unsigned lod,
                   bool compare) {
  return pattern("1101111010xxxx") | place(28, 3, 2) | place(31, 4, 0b0001) | place(37, 3, lod) |
         place(50, 1, compare ? 1u : 0u) | place(16, 3, predicate) | place(51, 3, 7) |
         place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TLD4_B, 2D, mask R, no offset, component 0. The handle is Rb.
std::uint64_t tld4b(unsigned ra, unsigned rb, unsigned dest, unsigned predicate, bool compare) {
  return pattern("1101111011xxxx") | place(28, 3, 2) | place(31, 4, 0b0001) |
         place(50, 1, compare ? 1u : 0u) | place(16, 3, predicate) | place(51, 3, 7) |
         place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
// TXD_B, 2D, mask R. The handle leads Ra and the gradients start at Rb.
std::uint64_t txdb(unsigned ra, unsigned rb, unsigned dest, unsigned predicate) {
  return pattern("1101111001xxxx") | place(28, 3, 2) | place(31, 4, 0b0001) |
         place(16, 3, predicate) | place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) |
         place(0, 8, dest);
}
std::uint64_t tldb(unsigned ra, unsigned rb, unsigned dest, unsigned predicate) {
  return pattern("11011101xxxxxx") | place(28, 3, 2) | place(31, 4, 0b0001) | place(16, 3, predicate) |
         place(51, 3, 7) | place(8, 8, ra) | place(20, 8, rb) | place(0, 8, dest);
}
std::uint64_t store_color(unsigned source, unsigned count) {
  return guarded(0xeff000000000ff00ull | source | (std::uint64_t{0x80} << 20) |
                     (std::uint64_t{count - 1} << 47),
                 7);
}
std::uint64_t exit_word() { return 0xe30000000007000full; }

StageImage image(const std::vector<std::uint64_t> &words) {
  StageImage result;
  result.stage = Stage::fragment;
  auto append = [&](std::uint64_t word) {
    for (unsigned i = 0; i < 8; ++i)
      result.bytes.push_back(std::byte((word >> (8 * i)) & 255));
  };
  for (const auto word : words) {
    if (result.bytes.size() % 32 == 0) append(0);
    append(word);
  }
  if (result.bytes.size() % 32 == 0) append(0);
  return result;
}

// Registers shared by every hoisted tap. Texture operands sit at R4 and R20,
// or at R40 for a gradient sample, clear of this prologue.
constexpr unsigned kPosX = 0;
constexpr unsigned kPosY = 1;
constexpr unsigned kPixX = 2;
constexpr unsigned kPixY = 3;
constexpr unsigned kHandle = 4;
constexpr unsigned kAux = 5;
constexpr unsigned kIndex = 8;
constexpr unsigned kMask = 9;
constexpr unsigned kShifted = 10;
constexpr unsigned kActive = 11;
constexpr unsigned kCode = 12;
constexpr unsigned kLaneX = 14;
constexpr unsigned kLaneY = 15;
constexpr unsigned kBumpX = 16;
constexpr unsigned kBumpY = 17;
constexpr unsigned kBumpFX = 18;
constexpr unsigned kBumpFY = 19;
constexpr unsigned kCoordX = 20;
constexpr unsigned kCoordY = 21;
constexpr unsigned kScale = 24;
constexpr unsigned kBig = 25;
constexpr unsigned kScaledX = 26;
constexpr unsigned kScaledY = 27;
constexpr unsigned kOffX = 28;
constexpr unsigned kOffY = 29;
constexpr unsigned kInactiveF = 33;
constexpr unsigned kDelta = 34;
constexpr unsigned kSpan = 35;
constexpr unsigned kTxd = 40;
constexpr unsigned kLoaded = 50;
constexpr unsigned kSum = 70;
constexpr unsigned kTap = 80;

// pixel_code is 0 on an active lane and 1 on an inactive lane. Inactive lanes
// add a 4-texel step on the odd edge of the quad, so the whole-quad slope is
// steep wherever a guard is only partly taken. Active lanes keep the gentle slope.
void append_coordinates(std::vector<std::uint64_t> &words) {
  words.push_back(ipa_position(kPosX, 0));
  words.push_back(ipa_position(kPosY, 1));
  words.push_back(f2i_trunc(kPixX, kPosX));
  words.push_back(f2i_trunc(kPixY, kPosY));
  words.push_back(shl_imm(kIndex, kPixY, 2));
  words.push_back(iadd(kIndex, kPixX, kIndex));
  words.push_back(mov32i(kMask, kActiveMask));
  words.push_back(shr(kShifted, kMask, kIndex));
  words.push_back(lop32i(kActive, kShifted, 0, 1));
  words.push_back(lop32i(kCode, kActive, 2, 1));
  words.push_back(lop32i(kLaneX, kPixX, 0, 1));
  words.push_back(lop32i(kLaneY, kPixY, 0, 1));
  words.push_back(imul(kBumpX, kLaneX, kCode));
  words.push_back(imul(kBumpY, kLaneY, kCode));
  words.push_back(i2f(kBumpFX, kBumpX));
  words.push_back(i2f(kBumpFY, kBumpY));
  words.push_back(mov32i(kScale, kScaleBits));
  words.push_back(mov32i(kBig, kBigBits));
  words.push_back(fmul(kScaledX, kPosX, kScale));
  words.push_back(fmul(kScaledY, kPosY, kScale));
  words.push_back(fmul(kOffX, kBumpFX, kBig));
  words.push_back(fmul(kOffY, kBumpFY, kBig));
  words.push_back(fadd(kCoordX, kScaledX, kOffX));
  words.push_back(fadd(kCoordY, kScaledY, kOffY));
}

std::uint64_t tap_word(Mode mode, unsigned dest, unsigned predicate) {
  switch (mode) {
  case Mode::level: return texb(kCoordX, kHandle, dest, predicate, 1, false);
  case Mode::implicit: return texb(kCoordX, kHandle, dest, predicate, 0, false);
  case Mode::bias: return texb(kCoordX, kHandle, dest, predicate, 2, false);
  case Mode::compare: return texb(kCoordX, kHandle, dest, predicate, 0, true);
  case Mode::gather: return tld4b(kCoordX, kHandle, dest, predicate, false);
  case Mode::compare_gather: return tld4b(kCoordX, kHandle, dest, predicate, true);
  case Mode::gradient: return txdb(kTxd, kTxd + 4, dest, predicate);
  case Mode::fetch: break;
  }
  throw std::runtime_error("no derivative encoding");
}

// `reference` opens each guarded run with a predicated store. The store is not
// read back; it only stops the hoister, which halts a run at the first store.
std::vector<std::uint64_t> derivative_program(Mode mode, bool reference) {
  std::vector<std::uint64_t> words;
  append_coordinates(words);
  if (mode == Mode::gradient) {
    words.push_back(i2f(kInactiveF, kCode));
    words.push_back(fadd(kDelta, kBig, kScale, true));
    words.push_back(fmul(kSpan, kInactiveF, kDelta));
    for (unsigned lane = 0; lane < 4; ++lane)
      words.push_back(fadd(kTxd + 4 + lane, kScale, kSpan));
    words.push_back(ldc(kTxd, 16));
    words.push_back(mov32i(kTxd + 1, 0));
    words.push_back(mov32i(kTxd + 2, 0));
    words.push_back(mov32i(kTxd + 3, 0));
    // Copies of the coordinates. The packed parameter register stays zero.
    words.push_back(mov32i(kSpan + 1, 0x3f800000u));
    words.push_back(fmul(kTxd + 1, kCoordX, kSpan + 1));
    words.push_back(fmul(kTxd + 2, kCoordY, kSpan + 1));
  } else {
    words.push_back(ldc(kHandle, 16));
    if (mode == Mode::bias) words.push_back(mov32i(kAux, kBiasBits));
    if (compares(mode)) words.push_back(mov32i(kAux, kRefBits));
  }
  for (unsigned tap = 0; tap < kTaps; ++tap) words.push_back(mov32i(kTap + tap, 0));
  for (unsigned tap = 0; tap < kTaps; ++tap) {
    // A fresh constant-buffer word keeps this predicate distinct from the others.
    words.push_back(ldc(kLoaded, 32 + 4 * tap));
    words.push_back(isetp_less(kCode, kLoaded));
    if (reference) words.push_back(stl(kCode, 4 * tap, 0));
    words.push_back(tap_word(mode, kTap + tap, 0));
  }
  for (unsigned lane = 0; lane < 4; ++lane) words.push_back(mov32i(kSum + lane, 0));
  for (unsigned tap = 0; tap < kTaps; ++tap) words.push_back(fadd(kSum, kSum, kTap + tap));
  words.push_back(store_color(kSum, 4));
  words.push_back(exit_word());
  return words;
}

// One predicated texel fetch. Columns 0 and 1 take it; the read stays guarded.
std::vector<std::uint64_t> fetch_program() {
  constexpr unsigned kThreshold = 3;
  constexpr unsigned kColor = 8;
  std::vector<std::uint64_t> words;
  words.push_back(ipa_position(kPosX, 0));
  words.push_back(mov32i(kThreshold, 0x40000000u));
  words.push_back(fsetp_less(kPosX, kThreshold));
  words.push_back(mov32i(kCoordX, 0));
  words.push_back(mov32i(kCoordY, 0));
  words.push_back(ldc(kHandle, 16));
  for (unsigned lane = 0; lane < 4; ++lane) words.push_back(mov32i(kColor + lane, kConstant));
  words.push_back(tldb(kCoordX, kHandle, kColor, 0));
  words.push_back(store_color(kColor, 4));
  words.push_back(exit_word());
  return words;
}

MslArtifact translate(const std::vector<std::uint64_t> &words, bool compare, bool local_memory,
                      bool hoist_implicit_samples = false) {
  Facts facts(compare, local_memory);
  FactReader reader(facts);
  auto host = profile();
  host.hoist_implicit_samples = hoist_implicit_samples;
  auto decoded = take(decode(image(words), reader, {}));
  auto registers = take(emit_ir(decoded, reader, {}));
  auto ssa = take(construct_ssa(std::move(registers), {}));
  auto legal = take(legalize(std::move(ssa), reader, host));
  auto structured = take(structure(std::move(legal), {}));
  auto plan = take(plan_resources(structured, host));
  return take(emit_msl(structured, plan, {}));
}

std::string fragment_body(const std::string &source) {
  const auto entry = source.find("armigo_fragment_main");
  if (entry == std::string::npos) return source.substr(0, std::min<std::size_t>(source.size(), 2000));
  return source.substr(entry, std::min<std::size_t>(source.size() - entry, 4000));
}

std::size_t matching_paren(const std::string &body, std::size_t open) {
  int depth = 0;
  for (std::size_t i = open; i < body.size(); ++i) {
    if (body[i] == '(') ++depth;
    else if (body[i] == ')') {
      if (--depth == 0) return i;
    }
  }
  return std::string::npos;
}

std::size_t close_brace(const std::string &body, std::size_t open) {
  int depth = 0;
  for (std::size_t i = open; i < body.size(); ++i) {
    if (body[i] == '{') ++depth;
    else if (body[i] == '}') {
      if (--depth == 0) return i;
    }
  }
  return std::string::npos;
}

struct Guard {
  std::size_t begin{};
  std::size_t open{};
  std::size_t close{};
};

std::vector<Guard> guard_ifs(const std::string &body) {
  std::vector<Guard> guards;
  std::size_t cursor = 0;
  while (cursor < body.size()) {
    const auto at = body.find("if (", cursor);
    if (at == std::string::npos) break;
    const auto paren = body.find('(', at);
    const auto end = paren == std::string::npos ? std::string::npos : matching_paren(body, paren);
    if (end == std::string::npos) break;
    const auto cond = body.substr(paren + 1, end - paren - 1);
    cursor = end + 1;
    if (cond.find("value") == std::string::npos || cond.find("armigo_") != std::string::npos) continue;
    const auto open = body.find('{', end);
    if (open == std::string::npos) break;
    const auto close = close_brace(body, open);
    if (close == std::string::npos) break;
    guards.push_back({at, open, close});
    cursor = open + 1;
  }
  return guards;
}

bool inside_guard(const std::vector<Guard> &guards, std::size_t pos) {
  for (const auto &guard : guards)
    if (pos > guard.open && pos < guard.close) return true;
  return false;
}

std::string_view call_at(const std::string &body, std::size_t pos) {
  static constexpr std::string_view kCalls[] = {
      ".sample_compare(", ".gather_compare(", ".sample(", ".gather(", ".read("};
  for (const auto call : kCalls)
    if (pos + call.size() <= body.size() && body.compare(pos, call.size(), call) == 0) return call;
  return {};
}

// Hoisted calls sit before a guard if. Reference calls sit inside one.
bool placement(const std::string &source, bool inside, std::string_view token,
               std::string_view required, std::string_view forbidden, unsigned minimum, std::string &why) {
  const auto entry = source.find("armigo_fragment_main");
  if (entry == std::string::npos) {
    why = "no fragment entry";
    return false;
  }
  const auto body = source.substr(entry);
  const auto guards = guard_ifs(body);
  if (guards.size() < minimum) {
    why = "guard if count " + std::to_string(guards.size());
    return false;
  }
  unsigned found = 0;
  for (std::size_t pos = 0; pos < body.size();) {
    const auto call = call_at(body, pos);
    if (call.empty()) {
      ++pos;
      continue;
    }
    if (call != token) {
      pos += call.size();
      continue;
    }
    const auto open = body.find('(', pos);
    const auto end = open == std::string::npos ? std::string::npos : matching_paren(body, open);
    if (end == std::string::npos) {
      why = "unclosed sample call";
      return false;
    }
    const auto text = body.substr(pos, end - pos + 1);
    if ((!required.empty() && text.find(required) == std::string::npos) ||
        (!forbidden.empty() && text.find(forbidden) != std::string::npos)) {
      why = "sample call has the wrong operands";
      return false;
    }
    const bool in = inside_guard(guards, pos);
    if (in != inside) {
      why = inside ? "sample call is outside the guard if" : "sample call is inside the guard if";
      return false;
    }
    if (!inside) {
      const bool precedes = std::any_of(guards.begin(), guards.end(),
                                        [&](const Guard &guard) { return guard.begin > pos; });
      if (!precedes) {
        why = "sample call does not precede a guard if";
        return false;
      }
    }
    ++found;
    pos = end + 1;
  }
  if (found < minimum) {
    why = "sample call count " + std::to_string(found);
    return false;
  }
  return true;
}

id<MTLLibrary> compile_library(id<MTLDevice> device, const std::string &source, const char *label) {
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion4_0;
  options.mathMode = MTLMathModeSafe;
  NSError *error = nil;
  id<MTLLibrary> result = [device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                               options:options
                                                 error:&error];
  if (!result)
    std::fprintf(stderr, "%s Metal compilation failed: %s\n%s\n", label,
                 error.localizedDescription.UTF8String, fragment_body(source).c_str());
  return result;
}

id<MTLTexture> color_texture(id<MTLDevice> device) {
  constexpr NSUInteger size = 16;
  auto *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                        width:size
                                                                       height:size
                                                                    mipmapped:YES];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (!texture || texture.mipmapLevelCount < kLevelBits.size()) return nil;
  for (NSUInteger level = 0; level < kLevelBits.size(); ++level) {
    const NSUInteger width = std::max<NSUInteger>(size >> level, 1);
    std::vector<std::uint32_t> pixels(width * width * 4, kLevelBits[level]);
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, width)
               mipmapLevel:level
                 withBytes:pixels.data()
               bytesPerRow:width * 4 * sizeof(std::uint32_t)];
  }
  return texture;
}

id<MTLTexture> depth_texture(id<MTLDevice> device) {
  constexpr NSUInteger size = 16;
  auto *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                        width:size
                                                                       height:size
                                                                    mipmapped:YES];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (!texture || texture.mipmapLevelCount < kLevelDepth.size()) return nil;
  for (NSUInteger level = 0; level < kLevelDepth.size(); ++level) {
    const NSUInteger width = std::max<NSUInteger>(size >> level, 1);
    std::vector<float> depths(width * width);
    for (NSUInteger y = 0; y < width; ++y)
      for (NSUInteger x = 0; x < width; ++x)
        depths[y * width + x] = kLevelDepth[level] + float(x + y * width) * (1.0f / 4096.0f);
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, width)
               mipmapLevel:level
                 withBytes:depths.data()
               bytesPerRow:width * sizeof(float)];
  }
  return texture;
}

// One level whose texels all differ, so a filtered sample depends on where
// it lands; the values are multiples of 1/256 and sum exactly.
id<MTLTexture> one_level_texture(id<MTLDevice> device) {
  constexpr NSUInteger size = 16;
  auto *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                        width:size
                                                                       height:size
                                                                    mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (!texture) return nil;
  std::vector<float> pixels(size * size * 4);
  for (NSUInteger i = 0; i < size * size; ++i)
    for (NSUInteger c = 0; c < 4; ++c) pixels[i * 4 + c] = float(i) / 256.0f;
  [texture replaceRegion:MTLRegionMake2D(0, 0, size, size)
             mipmapLevel:0
               withBytes:pixels.data()
             bytesPerRow:size * 4 * sizeof(float)];
  return texture;
}

id<MTLTexture> solid_texture(id<MTLDevice> device, std::uint32_t bits) {
  auto *descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                        width:4
                                                                       height:4
                                                                    mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  if (!texture) return nil;
  std::vector<std::uint32_t> pixels(4 * 4 * 4, bits);
  [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
             mipmapLevel:0
               withBytes:pixels.data()
             bytesPerRow:4 * 4 * sizeof(std::uint32_t)];
  return texture;
}

id<MTLSamplerState> make_sampler(id<MTLDevice> device, bool compare) {
  auto *sampler = [MTLSamplerDescriptor new];
  sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterNearest;
  sampler.mipFilter = MTLSamplerMipFilterNearest;
  sampler.sAddressMode = sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
  sampler.normalizedCoordinates = YES;
  sampler.lodMinClamp = 0;
  sampler.lodMaxClamp = 16;
  sampler.maxAnisotropy = 1;
  // Greater orders the reference above the stored depth, so level 0 (0.10)
  // passes and the coarse levels (0.70 and above) fail.
  sampler.compareFunction = compare ? MTLCompareFunctionGreater : MTLCompareFunctionNever;
  return [device newSamplerStateWithDescriptor:sampler];
}

// Linear minification and magnification, no mip filter, no anisotropy: the
// level of detail cannot select anything on a one-level texture.
id<MTLSamplerState> linear_sampler(id<MTLDevice> device) {
  auto *sampler = [MTLSamplerDescriptor new];
  sampler.minFilter = sampler.magFilter = MTLSamplerMinMagFilterLinear;
  sampler.mipFilter = MTLSamplerMipFilterNotMipmapped;
  sampler.sAddressMode = sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
  sampler.normalizedCoordinates = YES;
  sampler.maxAnisotropy = 1;
  return [device newSamplerStateWithDescriptor:sampler];
}

struct Frame {
  bool ok{false};
  std::vector<std::uint32_t> rgba;
};

Frame render(id<MTLDevice> device, id<MTL4Compiler> compiler, id<MTL4CommandQueue> queue,
             const char *name, const MslArtifact &fragment, id<MTLTexture> texture,
             id<MTLSamplerState> sampler) {
  Frame frame;
  const auto vertex_library = compile_library(device, vertex_source, "vertex");
  const auto fragment_library = compile_library(device, fragment.source, name);
  if (!vertex_library || !fragment_library) return frame;
  MTL4LibraryFunctionDescriptor *vertex_function = [MTL4LibraryFunctionDescriptor new];
  vertex_function.library = vertex_library;
  vertex_function.name = @"armigo_vertex_main";
  MTL4LibraryFunctionDescriptor *fragment_function = [MTL4LibraryFunctionDescriptor new];
  fragment_function.library = fragment_library;
  fragment_function.name = [NSString stringWithUTF8String:fragment.entry_point.c_str()];
  MTL4RenderPipelineDescriptor *pipeline_descriptor = [MTL4RenderPipelineDescriptor new];
  pipeline_descriptor.vertexFunctionDescriptor = vertex_function;
  pipeline_descriptor.fragmentFunctionDescriptor = fragment_function;
  pipeline_descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
  NSError *error = nil;
  id<MTLRenderPipelineState> pipeline =
      [compiler newRenderPipelineStateWithDescriptor:pipeline_descriptor
                               compilerTaskOptions:nil
                                             error:&error];
  if (!pipeline) {
    std::fprintf(stderr, "%s pipeline creation failed: %s\n%s\n", name,
                 error.localizedDescription.UTF8String, fragment_body(fragment.source).c_str());
    return frame;
  }

  std::uint32_t buffer_count = 1, texture_count = 0, sampler_count = 0;
  bool saw_constant = false, saw_sizes = false, saw_texture = false, saw_sampler = false;
  const auto grow = [](std::uint32_t &count, std::uint32_t index) {
    if (index + 1 > count) count = index + 1;
  };
  for (const auto &resource : fragment.reflection.resources) {
    const auto *range = std::get_if<MetalRange>(&resource.location);
    if (!range) throw std::runtime_error(std::string(name) + ": resource is not a direct binding");
    if (const auto *buffer = std::get_if<BufferUse>(&resource.use)) {
      grow(buffer_count, range->first);
      if (std::holds_alternative<ConstantBufferSource>(buffer->source)) saw_constant = true;
      else if (const auto *support = std::get_if<SupportBufferSource>(&buffer->source)) {
        if (support->schema == constant_buffer_support_schema) saw_sizes = true;
      }
    } else if (std::holds_alternative<TextureUse>(resource.use)) {
      grow(texture_count, range->first);
      saw_texture = true;
    } else if (std::holds_alternative<SamplerUse>(resource.use)) {
      grow(sampler_count, range->first);
      saw_sampler = true;
    }
  }
  if (fragment.reflection.custom_border) grow(buffer_count, fragment.reflection.custom_border->buffer_index);
  if (!saw_constant || !saw_sizes || !saw_texture)
    throw std::runtime_error(std::string(name) + ": translated stage is missing a texture binding");
  if (saw_sampler && !sampler) throw std::runtime_error(std::string(name) + ": sampler binding has no sampler");

  std::vector<std::uint32_t> constants(kCbufBytes / 4, 0);
  for (unsigned tap = 0; tap < kTaps; ++tap) constants[8 + tap] = 1u;
  id<MTLBuffer> constant_buffer =
      [device newBufferWithBytes:constants.data()
                          length:kCbufBytes
                         options:MTLResourceStorageModeShared];
  std::array<std::uint32_t, constant_buffer_slot_count> sizes{};
  sizes[0] = kCbufBytes;
  id<MTLBuffer> size_table = [device newBufferWithBytes:sizes.data()
                                                length:sizes.size() * sizeof(std::uint32_t)
                                               options:MTLResourceStorageModeShared];
  std::array<std::uint8_t, 64> border_bytes{};
  id<MTLBuffer> border = fragment.reflection.custom_border
                             ? [device newBufferWithBytes:border_bytes.data()
                                                   length:border_bytes.size()
                                                  options:MTLResourceStorageModeShared]
                             : nil;
  auto *target_descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                              width:kWidth
                                                                             height:kHeight
                                                                          mipmapped:NO];
  target_descriptor.storageMode = MTLStorageModePrivate;
  target_descriptor.usage = MTLTextureUsageRenderTarget;
  id<MTLTexture> target = [device newTextureWithDescriptor:target_descriptor];
  id<MTLBuffer> readback = [device newBufferWithLength:kWidth * kHeight * kPixelBytes
                                              options:MTLResourceStorageModeShared];
  MTL4ArgumentTableDescriptor *table_descriptor = [MTL4ArgumentTableDescriptor new];
  table_descriptor.maxBufferBindCount = buffer_count;
  table_descriptor.maxTextureBindCount = texture_count;
  table_descriptor.maxSamplerStateBindCount = sampler_count;
  id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
  MTLResidencySetDescriptor *residency_descriptor = [MTLResidencySetDescriptor new];
  residency_descriptor.initialCapacity = 8;
  id<MTLResidencySet> residency = [device newResidencySetWithDescriptor:residency_descriptor error:&error];
  id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
  id<MTL4CommandBuffer> commands = [device newCommandBuffer];
  if (!constant_buffer || !size_table || (fragment.reflection.custom_border && !border) || !target ||
      !readback || !table || !residency || !allocator || !commands) {
    std::fprintf(stderr, "%s Metal resource creation failed.\n", name);
    return frame;
  }
  for (const auto &resource : fragment.reflection.resources) {
    const auto *range = std::get_if<MetalRange>(&resource.location);
    if (const auto *buffer = std::get_if<BufferUse>(&resource.use)) {
      if (std::holds_alternative<ConstantBufferSource>(buffer->source))
        [table setAddress:constant_buffer.gpuAddress atIndex:range->first];
      else if (const auto *support = std::get_if<SupportBufferSource>(&buffer->source);
               support && support->schema == constant_buffer_support_schema)
        [table setAddress:size_table.gpuAddress atIndex:range->first];
    } else if (std::holds_alternative<TextureUse>(resource.use)) {
      [table setTexture:texture.gpuResourceID atIndex:range->first];
    } else if (std::holds_alternative<SamplerUse>(resource.use)) {
      [table setSamplerState:sampler.gpuResourceID atIndex:range->first];
    }
  }
  if (fragment.reflection.custom_border)
    [table setAddress:border.gpuAddress atIndex:fragment.reflection.custom_border->buffer_index];

  NSMutableArray *allocations = [@[constant_buffer, size_table, target, readback, texture] mutableCopy];
  if (border) [allocations addObject:border];
  for (id<MTLAllocation> allocation in allocations) [residency addAllocation:allocation];
  [residency commit];
  [commands beginCommandBufferWithAllocator:allocator];
  [commands useResidencySet:residency];
  MTL4RenderPassDescriptor *pass = [MTL4RenderPassDescriptor new];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
  id<MTL4RenderCommandEncoder> render_encoder = [commands renderCommandEncoderWithDescriptor:pass];
  if (!render_encoder) return frame;
  [render_encoder setRenderPipelineState:pipeline];
  [render_encoder setArgumentTable:table atStages:MTLRenderStageVertex | MTLRenderStageFragment];
  [render_encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [render_encoder endEncoding];
  id<MTL4ComputeCommandEncoder> copy = [commands computeCommandEncoder];
  if (!copy) return frame;
  [copy barrierAfterQueueStages:MTLStageFragment
                   beforeStages:MTLStageBlit
              visibilityOptions:MTL4VisibilityOptionDevice];
  [copy copyFromTexture:target
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(kWidth, kHeight, 1)
               toBuffer:readback
      destinationOffset:0
 destinationBytesPerRow:kWidth * kPixelBytes
destinationBytesPerImage:0];
  [copy endEncoding];
  [commands endCommandBuffer];
  NSMutableArray *retained = [@[vertex_library, fragment_library, pipeline, target, readback, residency,
                                allocator, table, constant_buffer, size_table, texture] mutableCopy];
  if (border) [retained addObject:border];
  if (sampler) [retained addObject:sampler];
  dispatch_semaphore_t completed = dispatch_semaphore_create(0);
  __block NSError *execution_error = nil;
  MTL4CommitOptions *options = [MTL4CommitOptions new];
  [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
    static_cast<void>(retained.count);
    execution_error = feedback.error;
    dispatch_semaphore_signal(completed);
  }];
  id<MTL4CommandBuffer> submitted[] = {commands};
  [queue commit:submitted count:1 options:options];
  if (dispatch_semaphore_wait(completed, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) != 0) {
    std::fprintf(stderr, "%s execution timed out.\n", name);
    return frame;
  }
  if (execution_error) {
    std::fprintf(stderr, "%s execution failed: %s\n", name, execution_error.localizedDescription.UTF8String);
    return frame;
  }
  const auto *pixels = static_cast<const std::uint32_t *>(readback.contents);
  frame.rgba.assign(pixels, pixels + kWidth * kHeight * 4);
  frame.ok = true;
  return frame;
}

void print_grid(const char *name, const std::vector<std::uint32_t> &rgba) {
  std::fprintf(stderr, "%s red grid:\n", name);
  for (NSUInteger y = 0; y < kHeight; ++y) {
    for (NSUInteger x = 0; x < kWidth; ++x) {
      const auto red = rgba[(y * kWidth + x) * 4];
      std::fprintf(stderr, " %c:%08x", lane_active(x, y) ? 'A' : '.', red);
    }
    std::fputc('\n', stderr);
  }
}

// Inactive lanes stay at the zero sum. Derivative modes must show more than
// one active color, which is the gentle quad against the steep partial quads.
bool sane_pixels(Mode mode, const std::vector<std::uint32_t> &rgba, std::string &why) {
  bool saw_active = false;
  bool varied = false;
  std::uint32_t first = 0;
  for (NSUInteger y = 0; y < kHeight; ++y) {
    for (NSUInteger x = 0; x < kWidth; ++x) {
      const auto base = (y * kWidth + x) * 4;
      const bool active = lane_active(x, y);
      if (!active) {
        if (rgba[base] != 0 || rgba[base + 1] != 0 || rgba[base + 2] != 0 || rgba[base + 3] != 0) {
          why = "inactive lane wrote a sample";
          return false;
        }
        continue;
      }
      if (!saw_active) {
        first = rgba[base];
        saw_active = true;
      } else if (rgba[base] != first) {
        varied = true;
      }
    }
  }
  if (!saw_active) {
    why = "no active lane";
    return false;
  }
  const bool expect_split = !hoists(mode);
  if (expect_split && !varied) {
    why = "active lanes share one color, so the quad slopes did not change the sample";
    return false;
  }
  if (!expect_split && first == 0 && !varied) {
    why = "active lanes stayed zero";
    return false;
  }
  return true;
}

// A gather reads the base level whatever the quad's slopes, steep inactive
// ones included: eight level-0 taps sum to 8 * 0.125 for a colour gather, and
// to eight passes for a comparison gather against the level-0 depths.
bool base_level_only(Mode mode, const std::vector<std::uint32_t> &rgba, std::string &why) {
  const std::uint32_t expected = mode == Mode::gather ? 0x3f800000u : 0x41000000u;
  for (NSUInteger y = 0; y < kHeight; ++y)
    for (NSUInteger x = 0; x < kWidth; ++x)
      if (lane_active(x, y) && rgba[(y * kWidth + x) * 4] != expected) {
        char text[96];
        std::snprintf(text, sizeof text, "lane (%lu,%lu) read %08x, not the base level's %08x",
                      static_cast<unsigned long>(x), static_cast<unsigned long>(y),
                      rgba[(y * kWidth + x) * 4], expected);
        why = text;
        return false;
      }
  return true;
}

struct CallSpec {
  std::string_view token;
  std::string_view required;
  std::string_view forbidden;
};

CallSpec call_spec(Mode mode) {
  switch (mode) {
  case Mode::level: return {".sample(", "level(", {}};
  case Mode::implicit: return {".sample(", {}, "bias("};
  case Mode::bias: return {".sample(", "bias(", {}};
  case Mode::gradient: return {".sample(", "gradient2d(", {}};
  case Mode::gather: return {".gather(", {}, {}};
  case Mode::compare: return {".sample_compare(", {}, {}};
  case Mode::compare_gather: return {".gather_compare(", {}, {}};
  case Mode::fetch: return {".read(", {}, {}};
  }
  return {};
}

int run_derivative(id<MTLDevice> device, id<MTL4Compiler> compiler, id<MTL4CommandQueue> queue, Mode mode,
                   id<MTLTexture> texture, id<MTLSamplerState> sampler) {
  const char *name = mode_name(mode);
  const auto spec = call_spec(mode);
  try {
    const auto hoisted = translate(derivative_program(mode, false), compares(mode), false);
    const auto guarded = translate(derivative_program(mode, true), compares(mode), true);
    std::string why;
    if (!placement(hoisted.source, !hoists(mode), spec.token, spec.required, spec.forbidden, kTaps, why)) {
      std::fprintf(stderr, "%s hoisted MSL: %s\n%s\n", name, why.c_str(), fragment_body(hoisted.source).c_str());
      std::printf("%s: placement fail\n", name);
      return 1;
    }
    if (!placement(guarded.source, true, spec.token, spec.required, spec.forbidden, kTaps, why)) {
      std::fprintf(stderr, "%s guarded MSL: %s\n%s\n", name, why.c_str(), fragment_body(guarded.source).c_str());
      std::printf("%s: placement fail\n", name);
      return 1;
    }
    const auto left = render(device, compiler, queue, name, hoisted, texture, sampler);
    const auto right_name = std::string(name) + " guarded";
    const auto right = render(device, compiler, queue, right_name.c_str(), guarded, texture, sampler);
    if (!left.ok || !right.ok) {
      std::printf("%s: equivalence fail\n", name);
      return 1;
    }
    if (left.rgba != right.rgba) {
      std::fprintf(stderr, "%s hoisted and guarded pixels differ\n", name);
      for (NSUInteger y = 0; y < kHeight; ++y) {
        for (NSUInteger x = 0; x < kWidth; ++x) {
          const auto index = (y * kWidth + x) * 4;
          if (!std::equal(left.rgba.begin() + index, left.rgba.begin() + index + 4, right.rgba.begin() + index))
            std::fprintf(stderr, "  (%lu,%lu) hoisted %08x %08x %08x %08x guarded %08x %08x %08x %08x\n",
                         static_cast<unsigned long>(x), static_cast<unsigned long>(y), left.rgba[index],
                         left.rgba[index + 1], left.rgba[index + 2], left.rgba[index + 3], right.rgba[index],
                         right.rgba[index + 1], right.rgba[index + 2], right.rgba[index + 3]);
        }
      }
      print_grid(name, left.rgba);
      std::printf("%s: placement pass, equivalence fail\n", name);
      return 1;
    }
    if (!sane_pixels(mode, left.rgba, why) ||
        ((mode == Mode::gather || mode == Mode::compare_gather) && !base_level_only(mode, left.rgba, why))) {
      std::fprintf(stderr, "%s: %s\n", name, why.c_str());
      print_grid(name, left.rgba);
      std::printf("%s: placement pass, equivalence fail\n", name);
      return 1;
    }
    std::printf("%s: placement pass, equivalence pass\n", name);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s: %s\n", name, error.what());
    std::printf("%s: not translated (%s)\n", name, error.what());
    return 1;
  }
}

// A title's opt-in: the implicit-LOD or bias taps leave their guards, and on a
// one-level texture with matching filters the partial quads keep their pixels.
int run_one_level(id<MTLDevice> device, id<MTL4Compiler> compiler, id<MTL4CommandQueue> queue, Mode mode,
                  id<MTLTexture> texture, id<MTLSamplerState> sampler) {
  const auto label = std::string(mode_name(mode)) + " opt-in, one level";
  const char *name = label.c_str();
  const auto spec = call_spec(mode);
  try {
    const auto hoisted = translate(derivative_program(mode, false), false, false, true);
    const auto guarded = translate(derivative_program(mode, true), false, true, true);
    std::string why;
    if (!placement(hoisted.source, false, spec.token, spec.required, spec.forbidden, kTaps, why) ||
        !placement(guarded.source, true, spec.token, spec.required, spec.forbidden, kTaps, why)) {
      std::fprintf(stderr, "%s MSL: %s\n%s\n", name, why.c_str(), fragment_body(hoisted.source).c_str());
      std::printf("%s: placement fail\n", name);
      return 1;
    }
    const auto left = render(device, compiler, queue, name, hoisted, texture, sampler);
    const auto right_name = label + " guarded";
    const auto right = render(device, compiler, queue, right_name.c_str(), guarded, texture, sampler);
    bool lit = false;
    for (NSUInteger i = 0; left.ok && i < kWidth * kHeight; ++i) lit |= left.rgba[i * 4] != 0;
    if (!left.ok || !right.ok || left.rgba != right.rgba || !lit) {
      if (left.ok && right.ok) {
        print_grid(name, left.rgba);
        print_grid(right_name.c_str(), right.rgba);
      }
      std::printf("%s: placement pass, equivalence fail\n", name);
      return 1;
    }
    std::printf("%s: placement pass, equivalence pass\n", name);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s: %s\n", name, error.what());
    std::printf("%s: not translated (%s)\n", name, error.what());
    return 1;
  }
}

int run_fetch(id<MTLDevice> device, id<MTL4Compiler> compiler, id<MTL4CommandQueue> queue,
              id<MTLTexture> texture) {
  const char *name = "texel fetch";
  try {
    const auto artifact = translate(fetch_program(), false, false);
    std::string why;
    if (!placement(artifact.source, true, ".read(", {}, {}, 1, why)) {
      std::fprintf(stderr, "%s MSL: %s\n%s\n", name, why.c_str(), fragment_body(artifact.source).c_str());
      std::printf("%s: placement fail\n", name);
      return 1;
    }
    const auto drawn = render(device, compiler, queue, name, artifact, texture, nil);
    if (!drawn.ok) {
      std::printf("%s: pixels fail\n", name);
      return 1;
    }
    int failures = 0;
    for (NSUInteger y = 0; y < kHeight; ++y) {
      for (NSUInteger x = 0; x < kWidth; ++x) {
        const bool taken = x < 2;
        const auto base = (y * kWidth + x) * 4;
        const auto expected = taken ? kFetchColor : kConstant;
        if (drawn.rgba[base] != expected || drawn.rgba[base + 1] != kConstant ||
            drawn.rgba[base + 2] != kConstant || drawn.rgba[base + 3] != kConstant) {
          std::fprintf(stderr, "%s pixel (%lu,%lu) guard %s red %08x expected %08x\n", name,
                       static_cast<unsigned long>(x), static_cast<unsigned long>(y), taken ? "true" : "false",
                       drawn.rgba[base], expected);
          failures = 1;
        }
      }
    }
    if (failures) {
      std::printf("%s: placement pass, pixels fail\n", name);
      return 1;
    }
    std::printf("%s: placement pass, pixels pass\n", name);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s: %s\n", name, error.what());
    std::printf("%s: not translated (%s)\n", name, error.what());
    return 1;
  }
}

} // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  @autoreleasepool {
    bool occupancies[5] = {};
    for (unsigned quad_y = 0; quad_y < kHeight / 2; ++quad_y)
      for (unsigned quad_x = 0; quad_x < kWidth / 2; ++quad_x) {
        const auto count = quad_occupancy(quad_x, quad_y);
        if (count < 5) occupancies[count] = true;
        std::printf("quad (%u,%u) active lanes %u\n", quad_x, quad_y, count);
      }
    for (unsigned count = 0; count < 5; ++count) {
      if (!occupancies[count]) {
        std::fprintf(stderr, "active mask misses occupancy %u\n", count);
        return 1;
      }
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::fputs("No Metal device.\n", stderr);
      return 1;
    }
    NSError *error = nil;
    id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
    id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
    if (!compiler || !queue) {
      std::fputs("Metal 4 compiler or queue unavailable.\n", stderr);
      return 1;
    }
    id<MTLTexture> color = color_texture(device);
    id<MTLTexture> depth = depth_texture(device);
    id<MTLTexture> fetch = solid_texture(device, kFetchColor);
    id<MTLSamplerState> color_sampler = make_sampler(device, false);
    id<MTLSamplerState> depth_sampler = make_sampler(device, true);
    if (!color || !depth || !fetch || !color_sampler || !depth_sampler) {
      std::fputs("Texture setup failed.\n", stderr);
      return 1;
    }
    int failures = 0;
    const Mode color_modes[] = {Mode::level, Mode::implicit, Mode::bias, Mode::gradient, Mode::gather};
    for (const auto mode : color_modes)
      failures += run_derivative(device, compiler, queue, mode, color, color_sampler);
    const Mode depth_modes[] = {Mode::compare, Mode::compare_gather};
    for (const auto mode : depth_modes)
      failures += run_derivative(device, compiler, queue, mode, depth, depth_sampler);
    failures += run_fetch(device, compiler, queue, fetch);
    id<MTLTexture> one_level = one_level_texture(device);
    id<MTLSamplerState> linear = linear_sampler(device);
    if (!one_level || !linear) {
      std::fputs("One-level texture setup failed.\n", stderr);
      return 1;
    }
    for (const auto mode : {Mode::implicit, Mode::bias})
      failures += run_one_level(device, compiler, queue, mode, one_level, linear);
    if (failures == 0) std::printf("All guarded sample GPU checks passed\n");
    else std::printf("%d guarded sample GPU cases failed\n", failures);
    return failures == 0 ? 0 : 1;
  }
}
