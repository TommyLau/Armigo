// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Private instruction-family registry of the graphics lowerer. Families that
// live in their own translation unit register one immutable entry here; the
// default case of the Builder dispatch consults the registry and otherwise
// fails exactly as before. A normal return from a lowerer means the
// instruction was lowered completely; an unsupported instruction must throw
// the structured error through FamilyLower::fail, never return silently.
#pragma once

#include <armigo/shader/graphics.h>

#include <initializer_list>
#include <span>

namespace armigo::shader::graphics::detail {

// Non-owning facade over the in-progress Builder. Exposes only the checked
// node primitives and the structured error path; every primitive applies to
// the instruction currently being lowered. Member definitions live in
// graphics.cpp, after Builder is complete.
// Operand of a registered pure operation: a scalar or a predicate node.
struct OperationOperand {
    OperationOperand(ScalarId id) noexcept : index(id.index) {}
    OperationOperand(PredicateId id) noexcept : index(id.index) {}
    std::uint32_t index;
};

class FamilyLower {
public:
    explicit FamilyLower(Builder& builder) noexcept;

    // Validation-only facade used by forward metadata admission: the
    // lowerer runs its payload, modifier and immediate checks and fails
    // exactly as it would when lowering, but nothing is appended or
    // written, register and predicate definedness is not consulted (a
    // guard-false instruction is admitted on metadata alone), and every
    // returned identity is a placeholder that must not be inspected.
    struct ValidationOnly {};
    FamilyLower(const Builder& context, std::uint32_t pc, ValidationOnly) noexcept;
    [[nodiscard]] bool validating() const noexcept { return builder_ == nullptr; }

    [[nodiscard]] std::uint32_t pc() const noexcept;
    [[nodiscard]] shader_input::Stage stage() const noexcept;
    [[nodiscard]] ScalarId read(maxwell::RegisterOperand reg);
    void write(maxwell::RegisterOperand reg, ScalarId value);
    [[nodiscard]] ScalarId literal(std::uint32_t bits);
    [[nodiscard]] ScalarId arithmetic(FloatArithmetic arithmetic);
    [[nodiscard]] ScalarId word_and(ScalarId operand, std::uint32_t mask);
    [[nodiscard]] ScalarId saturate(ScalarId operand);
    [[nodiscard]] ScalarId constant(maxwell::ConstantBufferOperand operand);

    // Registered pure operations (graphics_operations.h). The kind's result
    // must be Raw32 for operation() and Predicate for predicate_operation();
    // operand count and types must match its signature.
    [[nodiscard]] ScalarId operation(PureKind kind,
                                     std::initializer_list<OperationOperand> operands,
                                     std::uint32_t immediate = 0);
    [[nodiscard]] PredicateId predicate_operation(
        PureKind kind, std::initializer_list<OperationOperand> operands,
        std::uint32_t immediate = 0);

    // Predicate registers P0..P6; index 7 is PT. Reading PT yields a true
    // literal and writing PT discards. Reading an unwritten predicate fails
    // as an uninitialized register. Inversion is an explicit node.
    [[nodiscard]] PredicateId read_predicate(std::uint8_t index);
    void write_predicate(std::uint8_t index, PredicateId value);
    [[nodiscard]] PredicateId predicate_literal(bool value);
    [[nodiscard]] PredicateId negate_predicate(PredicateId operand);

    // Throws the structured lowering error attributed to the current pc.
    [[noreturn]] void fail(Error::Code code, std::uint32_t detail = 0);

private:
    Builder* builder_ = nullptr;
    const Builder* context_ = nullptr;
    std::uint32_t pc_ = 0;
};

// One registered lowering for a decoder family.
struct LowerEntry {
    maxwell::OpcodeFamily family;
    void (*lower)(const maxwell::Instruction& instruction, FamilyLower& lower);
};

// Immutable, append-only table of registered family lowerers.
[[nodiscard]] std::span<const LowerEntry> registered_lowerers() noexcept;

// Registered lowerer for the family, or null.
[[nodiscard]] inline const LowerEntry* find_registered_lowerer(
    maxwell::OpcodeFamily family) noexcept {
    for (const LowerEntry& entry : registered_lowerers()) {
        if (entry.family == family) {
            return &entry;
        }
    }
    return nullptr;
}

// Unit lowerers; each is defined beside the family it serves and declared
// here so the table below stays the only registration point.
void lower_mov_register(const maxwell::Instruction& instruction,
                        FamilyLower& lower);

void lower_mov_constant(const maxwell::Instruction& instruction,
                        FamilyLower& lower);
void lower_fma_register_constant(const maxwell::Instruction& instruction, FamilyLower& lower);
void lower_fma_constant_negated(const maxwell::Instruction& instruction, FamilyLower& lower);
void lower_word_and_immediate(const maxwell::Instruction& instruction, FamilyLower& lower);
void lower_fmul_saturate_register(const maxwell::Instruction& instruction,
                                  FamilyLower& lower);
void lower_ffma_register(const maxwell::Instruction& instruction,
                         FamilyLower& lower);
void lower_fmul_immediate32(const maxwell::Instruction& instruction,
                            FamilyLower& lower);
void lower_fadd_immediate32(const maxwell::Instruction& instruction,
                            FamilyLower& lower);
void lower_range_reduction(const maxwell::Instruction& instruction,
                           FamilyLower& lower);
void lower_dependency_barrier(const maxwell::Instruction& instruction,
                              FamilyLower& lower);

void lower_select_by_predicate(const maxwell::Instruction& instruction,
                               FamilyLower& lower);

void lower_word_logic(const maxwell::Instruction& instruction, FamilyLower& lower);
void lower_float_min_max(const maxwell::Instruction& instruction,
                         FamilyLower& lower);

void lower_float_to_integer(const maxwell::Instruction& instruction,
                            FamilyLower& lower);
void lower_float_set_predicate(const maxwell::Instruction& instruction,
                               FamilyLower& lower);
void lower_ssy(const maxwell::Instruction& instruction, FamilyLower& lower);

void lower_integer_to_float(const maxwell::Instruction& instruction, FamilyLower& lower);

void lower_float_round_integral(const maxwell::Instruction& instruction,
                                FamilyLower& lower);

}  // namespace armigo::shader::graphics::detail
