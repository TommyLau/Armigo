// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <armigo/shader_native/frontend.h>
#include <armigo/shader_native/instruction.h>
#include "stage_memory.h"
#include <array>
#include <deque>
#include <map>
#include <tuple>
#include <set>
#include <vector>

namespace armigo::shader_native::detail {
struct Failure { ir::Diagnostic diagnostic; };
std::string opcode_name(unsigned opcode);

// Strict field reader. Every bit a handler interprets is marked as read, and
// any set bit no handler read refuses the instruction, so an unvalidated
// modifier never passes silently. The opcode, guard and scheduling fields are
// read on construction.
class Fields {
public:
    explicit Fields(Instruction raw);
    std::uint64_t take(unsigned first, unsigned count);
    bool flag(unsigned bit) { return take(bit, 1) != 0; }
    bool unread() const;
    const Instruction& raw() const { return raw_; }
private:
    Instruction raw_;
    std::array<std::uint32_t, 4> read_{};
};

// Descriptor evidence per constant-bank handle word: the shape and compare
// mode the program's own texture instructions prove, or nothing when two
// instructions disagree.
using TextureEvidence = std::map<ir::TextureKey, std::optional<ir::TextureDescriptor>>;
ir::TextureDescriptor derived_texture(ir::TextureShape shape, bool comparison);
// Handles only dimension queries read are assumed 2D and listed in assumed.
void collect_texture_evidence(const ir::Program& program, TextureEvidence& evidence, std::set<ir::TextureKey>& assumed);
// Gives each LOD query its handle's proven descriptor (see texture.cpp).
void fit_lod_queries(ir::Program& program, const TextureEvidence& evidence);

// A decoded ALU source. Sign-modifier positions depend on the operand slot;
// the handler reads them only where the operation defines them.
struct AluSource {
    ir::Operand value;
    std::optional<unsigned> absolute, negate;
};

// Whether an instruction never executes: its guard is !PT, or !UPT for a
// uniform guard (bits 12-15 all set). Only that guard is constant false.
inline bool never_executes(const Instruction& raw) { return raw.bits(12, 4) == 0xf; }

// A BRX jump table: the bank-1 byte offset its selector load reads from and
// the case values taken from the program's constants there.
struct JumpTable {
    std::uint32_t offset{};
    std::vector<std::uint32_t> cases;
    // RET: no table; the register must hold one of the call return offsets
    // the program writes, as literals reaching it through copies and phis.
    bool returns{};
};
// Proves that every multiway branch selects on the bank-1 load its table came
// from, after SSA has connected each selector to its definition.
void verify_jump_tables(const ir::Program& program, const std::vector<JumpTable>& tables);

class Builder {
public:
    // raw_vertex_inputs: the host fetches vertex attributes in the shader,
    // as the guest attribute register words.
    Builder(const Program& source, ir::Limits limits, PixelRateState rate = PixelRateState::unknown, bool raw_vertex_inputs = false);
    ir::RegisterProgram run();
    bool requires_disabled_rate() const { return requires_disabled_rate_; }
    const std::vector<JumpTable>& jump_tables() const { return jump_tables_; }

private:
    const Program& source_;
    PixelRateState rate_;
    bool raw_vertex_inputs_{};
    // Set while lowering a guest form whose non-FTZ floating arithmetic is
    // admitted with Metal's own subnormal flushing (FSWZADD, FFMA), as the
    // Switch 1 translation emits it.
    bool metal_flush_{};
    bool requires_disabled_rate_{};
    ir::Program p_;
    // Each call site lowers its callee as its own copy: blocks are keyed by
    // instruction and call context, and a context records where its RET goes.
    struct CallContext { std::uint32_t parent{}, callee{}, return_pc{}; };
    std::vector<CallContext> contexts_{CallContext{}};
    std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>, std::uint32_t> context_ids_;
    std::uint32_t context_{};
    std::map<std::pair<std::uint32_t, std::uint32_t>, ir::BlockId> blocks_;
    std::deque<std::pair<std::uint32_t, std::uint32_t>> pending_;
    ir::BlockId block_in(std::uint32_t pc, std::uint32_t context);
    std::uint32_t call_context(std::uint32_t callee, std::uint32_t return_pc);
    ir::BlockId current_;
    std::uint32_t pc_{};
    unsigned opcode_{};
    // Uniform registers and predicates are warp-wide values. Every
    // invocation of a warp computes them from the same uniform sources, so
    // each invocation keeps its own copy as a mutable local of the IR.
    std::array<std::optional<ir::ValueId>, 63> uniform_registers_{};
    std::array<std::optional<ir::ValueId>, 7> uniform_predicates_{};
    // Task and mesh stages: where their stage buffer lies in workgroup memory.
    std::optional<StageMemory> stage_memory_;
    std::optional<ir::BlockId> next_;
    std::vector<JumpTable> jump_tables_;
    // Bank-1 offsets every BRX of the program reads its table from; a table
    // ends where the next one starts.
    std::set<std::uint32_t> table_starts_;
    std::optional<std::uint32_t> table_offset(std::uint32_t pc) const;

    [[noreturn]] void fail(std::string message, ir::DiagnosticCode code = ir::DiagnosticCode::unsupported_instruction) const;
    ir::BlockId extra();
    ir::BlockId block(std::uint32_t pc);
    ir::Operand reg(unsigned index) const;
    ir::Operand temporary(ir::Opcode opcode, std::vector<ir::Operand> inputs,
                          ir::NumericMode mode = {}, ir::OperationInfo info = {}, ir::ValueType type = ir::ValueType::word32);
    void operation(ir::Operation op);
    void write(unsigned destination, ir::Operand value);
    ir::Operand predicate(Predicate guard);
    ir::Operand condition(const DecodedInstruction& instruction);
    void guard(ir::Operand condition, ir::BlockId yes, ir::BlockId no);
    ir::Operand operand(const Operand& input);
    ir::Operand constant(unsigned bank, unsigned byte_offset);

    // Uniform state.
    ir::ValueId local(std::optional<ir::ValueId>& slot, ir::ValueType type);
    ir::Operand uniform(unsigned index);
    void write_uniform(unsigned index, ir::Operand value);
    ir::Operand uniform_predicate(unsigned index, bool inverted);
    void write_uniform_predicate(unsigned index, ir::Operand value);
    void write_predicate(unsigned index, ir::Operand value);
    ir::Operand predicate_source(Fields& fields, unsigned first, bool uniform);
    // Field-checked instructions enter their guard, with a uniform predicate
    // for uniform-datapath operations, and leave once every set bit is read.
    void enter(const Fields& fields, bool uniform_guard);
    // A transfer that always leaves the instruction (RET, BRX): only a guard
    // other than PT gives it a fall-through block.
    void enter_transfer(const Fields& fields);
    void leave(const Fields& fields);

    // Word helpers for integer lowering.
    ir::Operand bits(ir::Opcode opcode, std::vector<ir::Operand> inputs, ir::NumericMode mode = {ir::Domain::bits, ir::ScalarType::u32});
    ir::Operand logic(ir::Opcode opcode, std::vector<ir::Operand> inputs);
    // Sum and carry-out of left + right + carry.
    std::array<ir::Operand, 2> add_with_carry(ir::Operand left, ir::Operand right, ir::Operand carry);
    ir::Operand lookup(unsigned table, const std::array<ir::Operand, 3>& sources, bool boolean);
    std::array<AluSource, 3> alu_sources(Fields& fields, unsigned count, bool uniform);
    ir::Operand modified_integer(Fields& fields, const AluSource& source, bool allow_negate);

    void exports();
    void attribute(Instruction raw);
    void arithmetic(Instruction raw);
    void constant_load(Instruction raw);
    void interpolate(Instruction raw);
    ir::Operand interpolated(unsigned address, unsigned sample, bool constant);
    // Field-checked handlers. Each returns false when the opcode is not its
    // family and otherwise lowers the instruction or fails.
    bool integer(Fields& fields);
    bool uniform_operation(Fields& fields);
    bool conversion(Fields& fields);
    bool flow(Fields& fields);
    bool texture(Fields& fields);
    bool global_memory(Fields& fields);
    // system.cpp: special registers (S2R), votes (VOTE, VOTEU), R2UR, P2R.
    bool system_value(Fields& fields);
    // local.cpp: per-invocation local memory (LDL, STL).
    bool local_memory(Fields& fields);
    // gradient.cpp: sampling with explicit derivatives (TXD).
    bool texture_gradient(Fields& fields);
    // shared.cpp: workgroup shared memory (LDS, STS) and barriers (BAR).
    bool shared_memory(Fields& fields);
    // half.cpp: packed binary16 arithmetic (HADD2, HFMA2, HMNMX2) and HSETP2.
    bool half_precision(Fields& fields);
    // texture.cpp: formatted surface loads and stores, and U32 surface reductions.
    bool surface(Fields& fields);
    // tensor.cpp: warp matrix instructions (HMMA, LDSM, MOVM).
    bool tensor(Fields& fields);
    // stage_buffer.cpp: the task and mesh stage buffer (ISBERD, ISBEWR).
    bool stage_buffer(Fields& fields);
    // Compute, task and mesh stages run as workgroups of SIMD-groups.
    bool workgroup_stage() const {
        return source_.stage == Stage::compute || source_.stage == Stage::task || source_.stage == Stage::mesh;
    }
    void instruction(Instruction raw);
    void merge_chains();
};
// handle_program.cpp: texture and sampler handles the stage reads from guest
// memory at constant-buffer-computed addresses move to handle programs the
// host evaluates; their keys become handle_program_slot keys. Returns the
// programs, by key byte offset / 4.
std::vector<std::vector<std::uint32_t>> build_handle_programs(ir::Program& program);
} // namespace armigo::shader_native::detail
