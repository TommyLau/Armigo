// ============================================================================
// Armigo Execution - Native Guest Execution
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Bounded static collection of reachable A64 instructions.
//
// collect_reachable_code() walks linked, read-only image views from known
// entry addresses with the bounded classifier (instruction.h) and register
// effects (register_effects.h). It follows fallthrough, direct B/BL, both
// outcomes of conditional branches, and baseline ADRP/LDR[/ADD]/BR GOT thunks
// whose slot and target it validates. RET ends a path. Every other indirect
// transfer, every word it cannot classify, and every target outside the
// supplied executable ranges or declared external leaves is a structured
// failure; nothing is assumed safe.
//
// This is a reachability inventory only. It does not prove that any
// instruction may run natively, does not identify exclusive-monitor windows,
// does not recover jump tables, and never executes or modifies its input.
// A complete collection ends every path at RET, a leaf or a terminal trap.
// Recorded failures leave the sites a lower bound; invalid input collects nothing.

#pragma once

#include <armigo/aarch64/instruction.h>
#include <armigo/aarch64/register_effects.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace armigo::execution {

// Half-open guest address interval [begin, end).
struct AddressRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

// One linked module (or any readable region) as it will be at execution time.
// bytes is caller-owned and must neither change nor be released during the
// call; every byte is readable data for GOT slots. executable lists absolute,
// 4-byte aligned, non-overlapping intervals inside [base, base + size) that
// hold instructions. A data-only view has no executable ranges.
struct ReachableImage {
    std::uint64_t base = 0;
    std::span<const std::byte> bytes;
    std::vector<AddressRange> executable;
};

// A declared address outside the walk, such as a host entry point or a guest
// function the caller handles separately. A leaf is never decoded, even when
// it lies inside an executable range. returns selects whether a BL to the leaf
// continues at the following instruction.
struct ExternalLeaf {
    std::uint64_t address = 0;
    bool returns = true;
};

enum class SupervisorCallPolicy : std::uint8_t {
    fail,         // SVC is a failure
    fallthrough,  // SVC is collected and the walk continues after it
};

// BRK, HLT, HVC, SMC and UDF. None of them returns to the next instruction
// architecturally at EL0; terminal ends the path without claiming one.
enum class TrapPolicy : std::uint8_t {
    fail,
    terminal,
};

// Words register_effects.h reports as EffectStatus::unknown outside the
// branch, exception and system encoding group. Such words cannot transfer
// control architecturally if allocated, but the decoder cannot show that they
// are allocated. Unknown words inside that group always fail.
enum class UnknownInstructionPolicy : std::uint8_t {
    fail,
    assume_sequential,  // collected with SiteKind::assumed_sequential
};

struct ReachableOptions {
    SupervisorCallPolicy supervisor_calls = SupervisorCallPolicy::fallthrough;
    TrapPolicy traps = TrapPolicy::fail;
    UnknownInstructionPolicy unknown_instructions = UnknownInstructionPolicy::fail;
    // After an unresolved BLR, also collect the return continuation. The BLR
    // failure is still reported and the result stays partial.
    bool follow_indirect_call_return = false;
    std::size_t max_sites = 1'000'000;
    // Collection stops once this many failures are recorded.
    std::size_t max_failures = 256;
};

enum class SiteKind : std::uint8_t {
    sequential,          // decoded, classified or HINT; continues
    assumed_sequential,  // unknown word continued by policy
    branch,              // B, or B.cond/BC.cond with AL or NV
    conditional_branch,  // B.cond, BC.cond, CBZ, CBNZ, TBZ, TBNZ
    call,                // BL
    return_,             // RET
    supervisor_call,     // SVC collected with fallthrough
    trap,                // BRK, HLT, HVC, SMC, UDF ending the path by policy
    thunk_address,       // ADRP of a resolved thunk
    thunk_load,          // LDR of the GOT slot
    thunk_add,           // optional ADD of the slot offset
    thunk_branch,        // BR through the loaded slot
    failed,              // a failure was raised here; see failures
};

struct ReachableSite {
    std::uint64_t address = 0;
    std::uint32_t word = 0;
    std::uint32_t image = 0;  // index into the supplied images
    SiteKind kind = SiteKind::sequential;
    aarch64::Instruction instruction;
    aarch64::RegisterEffects effects;
};

enum class EdgeKind : std::uint8_t {
    branch,
    conditional_branch,
    call,
    thunk,  // ADRP address of a thunk to its resolved target
};

struct ReachableEdge {
    std::uint64_t source = 0;
    std::uint64_t target = 0;
    EdgeKind kind = EdgeKind::branch;
    bool external_leaf = false;
};

struct ResolvedThunk {
    std::uint64_t entry = 0;   // ADRP
    std::uint64_t branch = 0;  // BR
    std::uint64_t slot = 0;    // 8-byte GOT slot read from the supplied image
    std::uint64_t target = 0;
    bool external_leaf = false;
    bool has_add = false;
    std::uint8_t address_register = 0;  // ADRP Rd, LDR Rn, ADD Rd/Rn
    std::uint8_t target_register = 0;   // LDR Rt, BR Rn
};

enum class ReachableError : std::uint8_t {
    none,

    // Input validation; no inventory is produced.
    invalid_image,             // empty bytes or base + size overflows
    overlapping_images,
    invalid_executable_range,  // empty, misaligned, outside its image or overlapping
    invalid_leaf,              // misaligned, or declared twice with different behavior
    invalid_entry,             // misaligned, not executable, or a leaf

    // Collection failures; the inventory is partial.
    unknown_instruction,        // EffectStatus::unknown under UnknownInstructionPolicy::fail
    reserved_instruction,       // EffectStatus::reserved (UNDEFINED)
    unpredictable_instruction,  // EffectStatus::unpredictable
    unsupported_control_flow,   // unknown word in the branch/exception/system group
    indirect_branch,            // BR outside a validated thunk
    indirect_call,              // BLR
    supervisor_call,            // SVC under SupervisorCallPolicy::fail
    trap,                       // BRK/HLT/HVC/SMC/UDF under TrapPolicy::fail
    target_address_overflow,    // a PC-relative target leaves the address space
    misaligned_target,
    target_not_executable,      // neither in an executable range nor a leaf
    fallthrough_not_executable, // the next instruction leaves its executable range
    thunk_slot_misaligned,
    thunk_slot_unreadable,      // the 8-byte slot is not inside one supplied image
    thunk_interior_entry,       // control reaches a thunk other than at its ADRP
    resource_limit,             // more than max_sites sites
};

const char* describe(ReachableError error) noexcept;

struct ReachableFailure {
    ReachableError error = ReachableError::none;
    // Instruction whose handling failed; for input errors, the offending
    // address or zero.
    std::uint64_t address = 0;
    std::uint32_t word = 0;
    // Instruction that led to address (branch source or preceding
    // instruction), or address itself for an entry.
    std::uint64_t source = 0;
    // Offending branch or thunk target, or zero.
    std::uint64_t target = 0;
    // GOT slot for thunk failures, or zero.
    std::uint64_t slot = 0;
};

enum class CollectionStatus : std::uint8_t {
    complete,       // no failure; every path ended at RET, a leaf or a terminal trap
    partial,        // failures recorded; the inventory is a lower bound
    invalid_input,  // setup_error; nothing was collected
};

struct ReachableStatistics {
    std::uint64_t assumed_sequential = 0;
    std::uint64_t supervisor_calls = 0;
    std::uint64_t traps = 0;
    std::uint64_t external_leaf_edges = 0;
};

struct ReachableCode {
    CollectionStatus status = CollectionStatus::invalid_input;
    ReachableFailure setup_error;           // set only for invalid_input
    std::vector<ReachableSite> sites;       // sorted by address, unique
    std::vector<ReachableEdge> edges;       // sorted and unique
    std::vector<ResolvedThunk> thunks;      // sorted by entry
    std::vector<ReachableFailure> failures; // sorted by address, then error
    bool failures_truncated = false;        // stopped at max_failures
    ReachableStatistics statistics;

    bool complete() const noexcept { return status == CollectionStatus::complete; }
};

// Collects instructions reachable from entries. Images must not overlap.
// Every entry must be a 4-byte aligned executable address that is not a leaf.
// The result is deterministic for identical inputs.
ReachableCode collect_reachable_code(std::span<const ReachableImage> images,
                                     std::span<const std::uint64_t> entries,
                                     std::span<const ExternalLeaf> leaves = {},
                                     const ReachableOptions& options = {});

} // namespace armigo::execution
