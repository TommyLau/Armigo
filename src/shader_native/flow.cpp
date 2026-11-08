// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Native control and synchronization instructions with per-invocation
// meaning, and uniform constant loads.
//
// Convergence barriers (BSSY, BSYNC, BREAK) and WARPSYNC only regroup the
// invocations of a warp: BSSY records a barrier and its reconvergence point,
// BSYNC waits for the barrier's other members and falls through, and BREAK
// leaves the barrier. None changes an invocation's registers, predicates or
// control flow, so each lowers to a fall-through. DEPBAR waits on scoreboards;
// the shared IR orders every memory result before its uses. Subgroup
// operations stay refused here, so no admitted program depends on a warp's
// grouping.

#include "builder.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <set>

namespace armigo::shader_native::detail {
namespace {
ir::Operand word(std::uint32_t value) { return ir::LiteralBits{value, ir::ValueType::word32}; }
// Signed 50-bit displacement from the following instruction.
std::optional<std::uint64_t> relative_target(const Instruction& raw, std::uint64_t pc, std::uint64_t extent) {
    const auto encoded = raw.bits(32, 50);
    const auto relative = encoded & (1ull << 49) ? static_cast<std::int64_t>(encoded) - (1ll << 50) : static_cast<std::int64_t>(encoded);
    const auto target = static_cast<std::int64_t>(pc) + 16 + relative;
    if (target < 0 || target % 16 || static_cast<std::uint64_t>(target) >= extent) return std::nullopt;
    return static_cast<std::uint64_t>(target);
}
} // namespace

bool Builder::flow(Fields& fields) {
    auto fall_through = [&] {
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        p_.blocks[current_.value].terminator = ir::Jump{block(pc_ + 16)};
    };
    auto true_predicate = [&](unsigned first) {
        if (fields.take(first, 4) != 7) fail("predicate operand other than PT is not admitted");
    };
    switch (opcode_) {
    case 0x945: // BSSY Bn, reconvergence target
        fields.take(16, 4);
        if (!relative_target(fields.raw(), pc_, source_.instructions.size()))
            fail("convergence target is not an instruction boundary inside the program");
        fields.take(32, 50);
        true_predicate(87);
        fall_through(); return true;
    case 0x941: // BSYNC Bn
        fields.take(16, 4); true_predicate(87);
        fall_through(); return true;
    case 0x942: // BREAK [!]Pn, Bn: the predicate selects the leaving invocations.
        fields.take(16, 4); fields.take(87, 4);
        fall_through(); return true;
    case 0x948: // WARPSYNC mask
        fields.take(32, 32); true_predicate(87);
        fall_through(); return true;
    case 0x91a: // DEPBAR scoreboard wait
        fields.take(32, 32);
        fall_through(); return true;
    case 0x805: { // CS2R Rd, SRZ: zero one register or a register pair.
        enter(fields, false);
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        if (fields.take(72, 8) != 0xff) fail("special register other than SRZ is not admitted");
        const bool pair = fields.flag(80);
        if (pair && d != 255 && d > 253) fail("special register pair crosses RZ");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        write(d, ir::LiteralBits{0, ir::ValueType::word32});
        if (pair && d != 255) write(d + 1, ir::LiteralBits{0, ir::ValueType::word32});
        leave(fields); return true;
    }
    case 0x95b: { // KILL: discard the fragment under the guard.
        if (source_.stage != Stage::fragment) fail("fragment discard outside a fragment stage");
        enter(fields, false);
        true_predicate(87);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        operation({{}, ir::Opcode::discard, {}, {}, {}, {}, {}});
        leave(fields); return true;
    }
    case 0x944: { // CALL.REL.NOINC: a guarded transfer into the callee.
        // The return address of a native call is an ordinary register the
        // program sets itself, so the call transfers control only. Each call
        // site enters its own copy of the callee (a call context), whose RET
        // returns to the instruction after this CALL.
        const auto target = relative_target(fields.raw(), pc_, source_.instructions.size());
        if (!target) fail("call target is not an instruction boundary inside the program");
        fields.take(32, 50);
        if (fields.take(86, 1) != 1) fail("call without the no-increment form is not admitted");
        true_predicate(87);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        const auto bits = fields.raw().bits(12, 4);
        auto test = predicate({PredicateFile::general, static_cast<std::uint8_t>(bits & 7), (bits & 8) != 0});
        // A guarded transfer over the next instruction, a loop's back-edge
        // BRA, leaves the loop at its reconvergence point: a forward branch,
        // not a call. A call writes its return offset just before it (MOV of
        // this CALL's next offset), which keeps such a CALL a call.
        const auto after = [&](std::uint32_t at) { return Instruction::read(source_.instructions.subspan(at, 16)); };
        const bool linked = pc_ >= 16 && (after(pc_ - 16).words[0] & ~0x00ff0000u) == 0x00007802u &&
                            after(pc_ - 16).bits(16, 8) < 255 && after(pc_ - 16).words[1] == pc_ + 16 && after(pc_ - 16).words[2] == 0xf00u;
        const auto back = pc_ + 32 < source_.instructions.size() && after(pc_ + 16).bits(0, 12) == 0x947
            ? relative_target(after(pc_ + 16), pc_ + 16, source_.instructions.size()) : std::nullopt;
        if (bits != 7 && *target == pc_ + 32 && !linked && back && *back <= pc_ + 16) {
            if (const auto* literal = std::get_if<ir::LiteralBits>(&test))
                p_.blocks[current_.value].terminator = ir::Jump{block(literal->bits ? pc_ + 32 : pc_ + 16)};
            else guard(test, block(pc_ + 32), block(pc_ + 16));
            return true;
        }
        const auto callee = block_in(static_cast<std::uint32_t>(*target), call_context(static_cast<std::uint32_t>(*target), pc_ + 16));
        if (const auto* literal = std::get_if<ir::LiteralBits>(&test))
            p_.blocks[current_.value].terminator = literal->bits ? ir::Terminator{ir::Jump{callee}} : ir::Terminator{ir::Jump{block(pc_ + 16)}};
        else guard(test, callee, block(pc_ + 16));
        return true;
    }
    case 0x389: case 0x589: case 0x989: case 0xf89: { // SHFL Rd, Pd, Ra, lane, bounds
        // The warp shuffle of the Maxwell contract: the bounds word holds the
        // clamp (bits 0-4) and the segment mask (bits 8-12), the mode (58-59)
        // selects index, up, down or butterfly, and Pd reports an in-range
        // source lane. The lane is a register (32-39) or an immediate
        // (53-57), the bounds a register (64-71) or an immediate (40-52).
        // Metal offers SIMD-group functions to fragment and compute functions only.
        // The transfer runs before the guard on every lane that reaches the
        // instruction, so a lane the guard turns off still offers its
        // register, as the guest's parked register; only the results are
        // written under the guard.
        if (source_.stage != Stage::fragment && !workgroup_stage())
            fail("warp shuffle outside a fragment or workgroup stage is not admitted");
        const auto d = static_cast<unsigned>(fields.take(16, 8));
        const auto value = reg(static_cast<unsigned>(fields.take(24, 8)));
        const bool lane_immediate = opcode_ == 0x989 || opcode_ == 0xf89, bounds_immediate = opcode_ == 0x589 || opcode_ == 0xf89;
        const auto lane = lane_immediate ? word(static_cast<std::uint32_t>(fields.take(53, 5))) : reg(static_cast<unsigned>(fields.take(32, 8)));
        const auto bounds = bounds_immediate ? word(static_cast<std::uint32_t>(fields.take(40, 13))) : reg(static_cast<unsigned>(fields.take(64, 8)));
        const auto mode = static_cast<unsigned>(fields.take(58, 2));
        const auto in_range = static_cast<unsigned>(fields.take(81, 3));
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::IoInfo lane_id; lane_id.semantic = ir::IoSemantic::subgroup_lane_id; lane_id.type = {ir::ScalarType::u32, 1}; lane_id.component_mask = 1;
        temporary(ir::Opcode::load_io, {}, {ir::Domain::unsigned_integer, ir::ScalarType::u32}, lane_id);
        constexpr ir::ShuffleKind kinds[]{ir::ShuffleKind::index, ir::ShuffleKind::up, ir::ShuffleKind::down, ir::ShuffleKind::butterfly};
        if (p_.values.size() + 2 > p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const ir::ValueId result{static_cast<std::uint32_t>(p_.values.size())}, valid{result.value + 1};
        p_.values.push_back({ir::ValueType::word32}); p_.values.push_back({ir::ValueType::predicate});
        operation({{}, ir::Opcode::shuffle, {}, {ir::Domain::bits, ir::ScalarType::u32}, {value, lane, bounds}, {result, valid},
                   ir::SubgroupInfo{kinds[mode], 32}});
        enter(fields, false);
        write(d, result);
        write_predicate(in_range, valid);
        leave(fields); return true;
    }
    case 0x950: { // RET.REL.NODEC Ra, base: return to base + Ra.
        // In a call context the return goes to the instruction after that
        // context's CALL, in the caller's context. The compiler writes that
        // offset into Ra before the CALL; the one-case switch keeps the
        // register as its selector, and verify_jump_tables proves after SSA
        // that Ra holds exactly that literal.
        if (!context_) fail("return outside a native call is not admitted");
        enter_transfer(fields);
        const auto ra = static_cast<unsigned>(fields.take(24, 8));
        const auto base = relative_target(fields.raw(), pc_, source_.instructions.size());
        if (!base) fail("return base is not an instruction boundary inside the program");
        fields.take(32, 50);
        if (fields.take(86, 1) != 1) fail("return without the no-decrement form is not admitted");
        true_predicate(87);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        const auto& call = contexts_[context_];
        if (ra == 255 || call.return_pc < *base) fail("return register or base is not admitted");
        const auto value = static_cast<std::uint32_t>(call.return_pc - *base);
        const auto target = block_in(call.return_pc, call.parent);
        jump_tables_.push_back(JumpTable{0, {value}, true});
        p_.blocks[current_.value].terminator = ir::Switch{reg(ra), {ir::SwitchCase{value, target}}, target};
        return true;
    }
    case 0x949: case 0x958: { // BRX Ra or BRXU URa, base: branch to base + Ra.
        // The compiler loads Ra from a table of program-relative byte
        // offsets in bank 1, the program's own constants (BRXU: URa by an
        // indexed ULDC, its form bit 91 set). Each table entry up to the next
        // table becomes a case; verify_jump_tables proves after SSA that the
        // selector is that load. An index outside the table is undefined
        // guest behaviour and takes the last entry.
        enter_transfer(fields);
        const bool uniform_selector = opcode_ == 0x958;
        const auto ra = static_cast<unsigned>(fields.take(24, uniform_selector ? 6 : 8));
        if (uniform_selector && (fields.take(30, 2) || !fields.flag(91))) fail("uniform indirect branch register form is not admitted");
        const auto base = relative_target(fields.raw(), pc_, source_.instructions.size());
        if (!base) fail("indirect branch base is not an instruction boundary inside the program");
        fields.take(32, 50);
        true_predicate(87);
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        const auto offset = table_offset(pc_);
        if (!offset || ra == (uniform_selector ? 63u : 255u)) fail("indirect branch register is not loaded from a program constant table");
        const auto end = table_starts_.upper_bound(*offset);
        const std::size_t limit = std::min<std::size_t>(source_.constants.size(), end == table_starts_.end() ? source_.constants.size() : *end);
        JumpTable table{*offset, {}};
        std::vector<ir::SwitchCase> cases;
        for (std::size_t at = *offset; at + 4 <= limit && cases.size() < 256; at += 4) {
            const auto value = static_cast<std::uint32_t>(little_endian(source_.constants, at, 4));
            const auto target = *base + value;
            if (value % 16 || target >= source_.instructions.size()) break;
            table.cases.push_back(value);
            if (std::none_of(cases.begin(), cases.end(), [&](const auto& c) { return c.bits == value; }))
                cases.push_back({value, block(static_cast<std::uint32_t>(target))});
        }
        if (cases.empty()) fail("indirect branch table has no instruction target");
        jump_tables_.push_back(std::move(table));
        const auto fallback = cases.back().target;
        p_.blocks[current_.value].terminator = ir::Switch{uniform_selector ? uniform(ra) : reg(ra), std::move(cases), fallback};
        return true;
    }
    default: return false;
    }
}

namespace {
// Which register of its destination an instruction writes uniform register
// `index` as (0, or 1 for the high half of a pair), or nothing: the admitted
// uniform ALU forms that write a register (UISETP and UPLOP3 write only
// predicates; UIMAD.WIDE a pair), UMOV, UP2UR, VOTEU, ULDC (.64 a pair),
// R2UR, S2UR and REDUX, all with the destination at bits 16-21. A write this
// misses leaves verify_jump_tables to refuse the branch.
std::optional<unsigned> uniform_write(const Instruction& raw, unsigned index) {
    const auto opcode = static_cast<unsigned>(raw.bits(0, 12)), base = opcode & 0x1ff;
    static constexpr unsigned alu[]{0x087, 0x090, 0x091, 0x092, 0x096, 0x099, 0x09a, 0x0a4, 0x0a5, 0x0bd, 0x0be, 0x0bf};
    const bool writer = std::find(std::begin(alu), std::end(alu), base) != std::end(alu) || opcode == 0x882 || opcode == 0xc82 ||
                        opcode == 0x883 || opcode == 0x886 || opcode == 0xab9 || opcode == 0xabb || opcode == 0x3c2 ||
                        opcode == 0x9c3 || opcode == 0x3c4;
    if (!writer) return std::nullopt;
    const auto d = static_cast<unsigned>(raw.bits(16, 6));
    const bool pair = base == 0x0a5 || ((opcode == 0xab9 || opcode == 0xabb) && raw.bits(73, 3) == 5);
    if (d == index) return 0u;
    if (pair && d != 63 && d + 1 == index) return 1u;
    return std::nullopt;
}
} // namespace

// The bank-1 offset of the 32-bit indexed LDC (BRXU: ULDC) that last writes
// the BRX register in the instructions before it, without crossing control
// flow. Instructions that never execute neither write the register nor
// transfer control.
std::optional<std::uint32_t> Builder::table_offset(std::uint32_t pc) const {
    const auto branch = Instruction::read(source_.instructions.subspan(pc, 16));
    const bool uniform_selector = branch.bits(0, 12) == 0x958;
    const auto ra = branch.bits(24, uniform_selector ? 6 : 8);
    for (std::uint32_t at = pc, steps = 0; at >= 16 && steps < 64; ++steps) {
        at -= 16;
        const auto raw = Instruction::read(source_.instructions.subspan(at, 16));
        if (never_executes(raw)) continue;
        const auto opcode = raw.bits(0, 12);
        if (opcode == 0x947 || opcode == 0x949 || opcode == 0x958 || opcode == 0x94d || opcode == 0x944 || opcode == 0x945 ||
            opcode == 0x941 || opcode == 0x942 || opcode == 0x950) return std::nullopt;
        std::uint32_t half = 0;
        if (uniform_selector) {
            const auto written = uniform_write(raw, static_cast<unsigned>(ra));
            if (!written) continue;
            // A 32-bit indexed ULDC, or either half of a 64-bit one (the high
            // half's table starts a word later).
            const auto width = raw.bits(73, 3);
            if (opcode != 0xabb || raw.bits(54, 5) != 1 || (width != 4 && width != 5) ||
                raw.words[2] != ((1u << 27) | (static_cast<std::uint32_t>(width) << 9)) ||
                (raw.words[1] & ~0x07ffffc0u) || raw.bits(22, 2) || raw.bits(24, 6) == 63 || raw.bits(30, 2)) return std::nullopt;
            half = *written;
        } else {
            if (raw.bits(16, 8) != ra) continue;
            if (opcode != 0xb82 || raw.bits(54, 5) != 1 || raw.words[2] != (4u << 9) || (raw.words[1] & ~0x07ffffc0u) ||
                raw.bits(24, 8) == 255) return std::nullopt;
        }
        const auto displacement = static_cast<std::uint32_t>(raw.bits(38, 16));
        if (displacement & 0x8000 || displacement % 4) return std::nullopt;
        return displacement + 4 * half;
    }
    return std::nullopt;
}

void verify_jump_tables(const ir::Program& program, const std::vector<JumpTable>& tables) {
    auto definition = [&](ir::ValueId value) -> const ir::Operation* {
        for (const auto& op : program.operations)
            for (const auto& output : op.outputs)
                if (const auto* id = std::get_if<ir::ValueId>(&output); id && id->value == value.value) return &op;
        return nullptr;
    };
    auto refuse = [](const char* message) {
        throw Failure{ir::diagnose(ir::Phase::construct_ssa, ir::DiagnosticCode::unsupported_instruction, {}, message)};
    };
    std::map<std::uint32_t, const ir::Phi*> phis;
    for (const auto& block : program.blocks) for (const auto& phi : block.phis) phis[phi.output.value] = &phi;
    // The literal values an operand may hold, through copies and phis.
    std::function<bool(const ir::Operand&, std::set<std::uint32_t>&, unsigned)> literals =
        [&](const ir::Operand& operand, std::set<std::uint32_t>& out, unsigned depth) -> bool {
        if (depth > 64) return false;
        if (const auto* literal = std::get_if<ir::LiteralBits>(&operand)) { out.insert(static_cast<std::uint32_t>(literal->bits)); return true; }
        const auto* id = std::get_if<ir::ValueId>(&operand);
        if (!id) return false;
        if (auto phi = phis.find(id->value); phi != phis.end()) {
            for (const auto& input : phi->second->inputs)
                if (input.value != phi->second->output && !literals(input.value, out, depth + 1)) return false;
            return true;
        }
        const auto* op = definition(*id);
        return op && op->opcode == ir::Opcode::copy && op->inputs.size() == 1 && literals(op->inputs[0], out, depth + 1);
    };
    for (const auto& block : program.blocks) {
        const auto* select = std::get_if<ir::Switch>(&block.terminator);
        if (!select) continue;
        std::vector<std::uint32_t> bits;
        for (const auto& item : select->cases) bits.push_back(item.bits);
        const bool returned = std::any_of(tables.begin(), tables.end(), [&](const JumpTable& table) {
            if (!table.returns || table.cases != bits) return false;
            std::set<std::uint32_t> held;
            return literals(select->selector, held, 0) &&
                std::all_of(held.begin(), held.end(), [&](std::uint32_t value) { return std::find(bits.begin(), bits.end(), value) != bits.end(); });
        });
        if (returned) continue;
        const auto* selector = std::get_if<ir::ValueId>(&select->selector);
        std::optional<ir::ValueId> value = selector ? std::optional<ir::ValueId>{*selector} : std::nullopt;
        const ir::Operation* op = value ? definition(*value) : nullptr;
        while (op && op->opcode == ir::Opcode::copy && op->inputs.size() == 1 && std::holds_alternative<ir::ValueId>(op->inputs[0])) {
            value = std::get<ir::ValueId>(op->inputs[0]);
            op = definition(*value);
        }
        const auto* memory = op && op->opcode == ir::Opcode::load ? std::get_if<ir::MemoryInfo>(&op->info) : nullptr;
        if (!memory || memory->space != ir::AddressSpace::constant_buffer || memory->constant_buffer_slot != 1 ||
            memory->elements < 1 || memory->elements > 2 || memory->element_bytes != 4 || !memory->dynamic_byte_offset)
            refuse("indirect branch selector is not a bank-1 table load");
        // The selector is one word of the load; the second word of a pair
        // reads one word further into the table.
        std::uint32_t element = 0;
        for (std::uint32_t k = 0; k < op->outputs.size(); ++k)
            if (const auto* output = std::get_if<ir::ValueId>(&op->outputs[k]); output && output->value == value->value) element = k;
        std::uint32_t offset = memory->byte_offset + 4 * element;
        if (const auto* address = std::get_if<ir::ValueId>(&op->inputs[memory->dynamic_byte_offset->value])) {
            const auto* add = definition(*address);
            if (add && add->opcode == ir::Opcode::add && add->inputs.size() == 2)
                if (const auto* literal = std::get_if<ir::LiteralBits>(&add->inputs[1])) offset += static_cast<std::uint32_t>(literal->bits);
        }
        const bool proven = std::any_of(tables.begin(), tables.end(), [&](const JumpTable& table) {
            if (table.returns || table.offset != offset) return false;
            std::vector<std::uint32_t> distinct;
            for (auto value : table.cases) if (std::find(distinct.begin(), distinct.end(), value) == distinct.end()) distinct.push_back(value);
            return distinct == bits;
        });
        if (!proven) refuse("indirect branch selector does not load its table's offset");
    }
}

bool Builder::uniform_operation(Fields& fields) {
    if (opcode_ != 0xab9 && opcode_ != 0xabb) return false;
    // ULDC URd, c[bank][offset]: widths 0-3 read U8, S8, U16 and S16
    // (zero- or sign-extended, as LDC's sized loads), 4 one word and 5 two
    // consecutive words. 0xabb indexes the bank with a uniform register
    // (24-29) plus the signed offset, its form bit 91 set.
    enter(fields, true);
    const auto d = static_cast<unsigned>(fields.take(16, 6));
    if (fields.take(22, 2)) fail("uniform destination field extension is not admitted");
    const auto width = static_cast<unsigned>(fields.take(73, 3));
    if (width > 5) fail("uniform constant load width is not admitted");
    constexpr ir::ScalarType narrow[]{ir::ScalarType::u8, ir::ScalarType::s8, ir::ScalarType::u16, ir::ScalarType::s16};
    const std::uint8_t bytes = width < 2 ? 1 : width < 4 ? 2 : 4;
    const ir::NumericMode load_mode = width < 4
        ? ir::NumericMode{width % 2 ? ir::Domain::signed_integer : ir::Domain::unsigned_integer, narrow[width]}
        : ir::NumericMode{ir::Domain::unsigned_integer, ir::ScalarType::u32};
    if (fields.take(32, 6)) fail("uniform constant load index field is not admitted");
    if (opcode_ == 0xabb) {
        const auto index = static_cast<unsigned>(fields.take(24, 6));
        if (fields.take(30, 2)) fail("uniform index field extension is not admitted");
        if (!fields.flag(91)) fail("indexed uniform constant load lacks its form bit");
        const auto encoded = static_cast<std::uint32_t>(fields.take(38, 16));
        const auto displacement = encoded & 0x8000 ? encoded | 0xffff0000u : encoded;
        const auto bank = static_cast<unsigned>(fields.take(54, 5));
        if (bank >= 18) fail("constant bank exceeds the established IR binding contract");
        const unsigned elements = width == 5 ? 2 : 1;
        if (elements == 2 && d != 63 && d > 61) fail("paired uniform constant load register range is not admitted");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        const ir::NumericMode mode{ir::Domain::unsigned_integer, ir::ScalarType::u32};
        ir::MemoryInfo info; info.space = ir::AddressSpace::constant_buffer; info.access = ir::Access::read;
        info.constant_buffer_slot = bank; info.element_bytes = bytes; info.elements = elements; info.alignment = bytes;
        info.dynamic_byte_offset = ir::InputIndex{0};
        auto address = displacement ? temporary(ir::Opcode::add, {uniform(index), word(displacement)}, mode) : uniform(index);
        std::vector<ir::Operand> outputs;
        for (unsigned i = 0; i < elements; ++i) {
            if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
            outputs.push_back(ir::ValueId{static_cast<std::uint32_t>(p_.values.size())}); p_.values.push_back({ir::ValueType::word32});
        }
        operation({{}, ir::Opcode::load, {}, load_mode, {address}, outputs, info});
        for (unsigned i = 0; i < elements; ++i) if (d != 63) write_uniform(d + i, outputs[i]);
        leave(fields); return true;
    }
    const auto offset = static_cast<unsigned>(fields.take(38, 16)), bank = static_cast<unsigned>(fields.take(54, 5));
    if (width < 4) {
        if (offset % bytes) fail("narrow uniform constant load is not naturally aligned");
        if (bank >= 18) fail("constant bank exceeds the established IR binding contract");
        if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
        ir::MemoryInfo info; info.space = ir::AddressSpace::constant_buffer; info.access = ir::Access::read;
        info.constant_buffer_slot = bank; info.element_bytes = bytes; info.elements = 1; info.alignment = bytes;
        info.byte_offset = offset;
        if (p_.values.size() >= p_.limits.values) fail("value limit exceeded", ir::DiagnosticCode::limit_exceeded);
        const ir::ValueId loaded{static_cast<std::uint32_t>(p_.values.size())}; p_.values.push_back({ir::ValueType::word32});
        operation({{}, ir::Opcode::load, {}, load_mode, {}, {loaded}, info});
        write_uniform(d, loaded);
        leave(fields); return true;
    }
    const bool pair = width == 5;
    if (pair && (offset % 8 || (d != 63 && d > 61))) fail("paired uniform constant load alignment or register range is not admitted");
    auto low = constant(bank, offset);
    auto high = pair ? constant(bank, offset + 4) : ir::Operand{};
    if (fields.unread()) fail("instruction mode or reserved bits are not admitted");
    write_uniform(d, low);
    if (pair && d != 63) write_uniform(d + 1, high);
    leave(fields); return true;
}
} // namespace armigo::shader_native::detail
