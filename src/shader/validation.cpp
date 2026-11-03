// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/validation.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace armigo::shader {
namespace {

// Words that may not name a kernel because C++ or the Metal Shading Language
// reserves them, or because they name a global type or namespace that the
// generated source refers to. Sorted for binary search.
constexpr std::string_view kReservedWords[] = {
    "alignas", "alignof", "and", "and_eq", "as_type", "asm", "auto",
    "bfloat", "bitand", "bitor", "bool", "break", "case", "catch", "char",
    "char16_t", "char32_t", "char8_t", "class", "co_await", "co_return",
    "co_yield", "compl", "concept", "const", "const_cast", "constant",
    "consteval", "constexpr", "constinit", "continue", "decltype",
    "default", "delete", "device", "do", "double", "dynamic_cast", "else",
    "enum", "explicit", "export", "extern", "false", "float", "for",
    "fragment", "friend", "goto", "half", "if", "inline", "int", "kernel",
    "long", "main", "mesh", "metal", "mutable", "namespace", "new",
    "noexcept", "not", "not_eq", "nullptr", "object", "object_data",
    "operator", "or", "or_eq", "private", "protected", "ptrdiff_t",
    "public", "ray_data", "register", "reinterpret_cast", "requires",
    "return", "short", "signed", "size_t", "sizeof", "static",
    "static_assert", "static_cast", "stitchable", "struct", "switch",
    "template", "this", "thread", "thread_local", "threadgroup",
    "threadgroup_imageblock", "throw", "true", "try", "typedef", "typeid",
    "typename", "uchar", "uint", "ulong", "union", "unsigned", "ushort",
    "using", "vertex", "virtual", "visible", "void", "volatile", "wchar_t",
    "while", "xor", "xor_eq",
};

// Scalar element names that form global vector and matrix type names, such as
// float4 or half3x3.
constexpr std::string_view kVectorElementNames[] = {
    "bfloat", "bool", "char",  "double", "float", "half", "int",
    "long",   "short", "uchar", "uint",   "ulong", "ushort",
};

static_assert(std::is_sorted(std::begin(kReservedWords),
                             std::end(kReservedWords)),
              "kReservedWords must stay sorted for binary search");

constexpr std::string_view kGeneratedPrefix = "armigo_";

bool is_ascii_letter(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool is_ascii_digit(char c) noexcept { return c >= '0' && c <= '9'; }

bool is_vector_dimension(char c) noexcept { return c >= '2' && c <= '4'; }

// Matches <element>N and <element>NxM for N, M in [2, 4].
bool is_vector_or_matrix_type_name(std::string_view name) {
    for (std::string_view element : kVectorElementNames) {
        if (!name.starts_with(element)) {
            continue;
        }
        std::string_view suffix = name.substr(element.size());
        if (suffix.size() == 1 && is_vector_dimension(suffix[0])) {
            return true;
        }
        if (suffix.size() == 3 && is_vector_dimension(suffix[0]) &&
            suffix[1] == 'x' && is_vector_dimension(suffix[2])) {
            return true;
        }
    }
    return false;
}

bool is_known_type(ScalarType type) noexcept {
    return type == ScalarType::f32 || type == ScalarType::u32;
}

bool is_known_opcode(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::constant:
    case Opcode::load_parameter:
    case Opcode::load_input:
    case Opcode::fadd:
    case Opcode::fmul:
    case Opcode::store_output:
        return true;
    }
    return false;
}

bool is_finite_f32_bits(std::uint32_t bits) noexcept {
    // An all-ones exponent encodes infinity or NaN.
    return (bits & 0x7f800000u) != 0x7f800000u;
}

std::string type_name(ScalarType type) {
    if (is_known_type(type)) {
        return to_string(type);
    }
    return "type #" + std::to_string(static_cast<unsigned>(type));
}

// Validation state of each instruction's result.
enum class ValueState : std::uint8_t {
    none,      // Instruction defines no value (store_output).
    defined,   // Valid value of the recorded type.
    poisoned,  // Instruction was invalid; uses produce no further diagnostics.
};

struct ValueInfo {
    ValueState state = ValueState::none;
    ScalarType type = ScalarType::f32;
};

class Validator {
public:
    explicit Validator(const Program& program) : program_(program) {}

    std::vector<Diagnostic> run() {
        check_declarations();
        if (program_.body.size() > kMaxInstructions) {
            // Do not walk an oversized body; the limit bounds validation cost.
            return std::move(diagnostics_);
        }
        values_.resize(program_.body.size());
        stored_.assign(program_.outputs.size(), false);
        for (std::size_t i = 0; i < program_.body.size(); ++i) {
            check_instruction(static_cast<std::uint32_t>(i));
        }
        for (std::size_t binding = 0; binding < stored_.size(); ++binding) {
            if (!stored_[binding]) {
                report(DiagnosticCode::output_never_stored,
                       Diagnostic::kNoInstruction,
                       "output " + std::to_string(binding) +
                           " is never stored");
            }
        }
        return std::move(diagnostics_);
    }

private:
    void report(DiagnosticCode code, std::uint32_t instruction,
                std::string message) {
        diagnostics_.push_back({code, instruction, std::move(message)});
        ++instruction_errors_;
    }

    void check_declarations() {
        if (!is_valid_entry_point(program_.entry_point)) {
            report(DiagnosticCode::invalid_entry_point,
                   Diagnostic::kNoInstruction,
                   "entry point name is not a valid, unreserved identifier");
        }
        if (program_.outputs.empty()) {
            report(DiagnosticCode::no_outputs, Diagnostic::kNoInstruction,
                   "program declares no outputs");
        }
        if (program_.parameters.size() > kMaxParameters) {
            report(DiagnosticCode::too_many_parameters,
                   Diagnostic::kNoInstruction,
                   std::to_string(program_.parameters.size()) +
                       " parameters exceed the limit of " +
                       std::to_string(kMaxParameters));
        }
        const std::size_t io = program_.inputs.size() + program_.outputs.size();
        if (io > kMaxIoBindings) {
            report(DiagnosticCode::too_many_bindings,
                   Diagnostic::kNoInstruction,
                   std::to_string(io) +
                       " input and output bindings exceed the limit of " +
                       std::to_string(kMaxIoBindings));
        }
        if (program_.body.size() > kMaxInstructions) {
            report(DiagnosticCode::too_many_instructions,
                   Diagnostic::kNoInstruction,
                   std::to_string(program_.body.size()) +
                       " instructions exceed the limit of " +
                       std::to_string(kMaxInstructions));
        }
        check_declared_types("parameter slot", program_.parameters);
        check_declared_types("input", program_.inputs);
        check_declared_types("output", program_.outputs);
    }

    void check_declared_types(const char* what,
                              const std::vector<ScalarType>& types) {
        for (std::size_t i = 0; i < types.size(); ++i) {
            if (!is_known_type(types[i])) {
                report(DiagnosticCode::unknown_type, Diagnostic::kNoInstruction,
                       std::string(what) + " " + std::to_string(i) +
                           " has unknown " + type_name(types[i]));
            }
        }
    }

    // Returns the referenced value's info, or nullptr after reporting the
    // reference as invalid. Poisoned values are returned without a report so
    // that callers can skip dependent checks silently.
    const ValueInfo* check_operand(std::uint32_t at, ValueId operand,
                                   const char* role) {
        if (operand.is_none()) {
            report(DiagnosticCode::operand_missing, at,
                   std::string(role) + " operand is missing");
            return nullptr;
        }
        if (operand.index >= program_.body.size()) {
            report(DiagnosticCode::operand_out_of_range, at,
                   std::string(role) + " operand refers to instruction " +
                       std::to_string(operand.index) +
                       ", which does not exist");
            return nullptr;
        }
        if (operand.index >= at) {
            report(DiagnosticCode::operand_not_dominating, at,
                   std::string(role) + " operand refers to instruction " +
                       std::to_string(operand.index) +
                       ", which does not precede its use");
            return nullptr;
        }
        const ValueInfo& info = values_[operand.index];
        if (info.state == ValueState::none) {
            report(DiagnosticCode::operand_has_no_value, at,
                   std::string(role) + " operand refers to instruction " +
                       std::to_string(operand.index) +
                       ", which defines no value");
            return nullptr;
        }
        return &info;
    }

    void require_no_operand(std::uint32_t at, ValueId operand,
                            const char* role) {
        if (!operand.is_none()) {
            report(DiagnosticCode::unexpected_field, at,
                   std::string(role) + " operand must be empty for " +
                       to_string(program_.body[at].opcode));
        }
    }

    void require_zero_immediate(std::uint32_t at) {
        if (program_.body[at].immediate != 0) {
            report(DiagnosticCode::unexpected_field, at,
                   std::string("immediate must be zero for ") +
                       to_string(program_.body[at].opcode));
        }
    }

    void check_instruction(std::uint32_t at) {
        const Instruction& inst = program_.body[at];
        instruction_errors_ = 0;
        ValueInfo& result = values_[at];

        if (!is_known_opcode(inst.opcode)) {
            report(DiagnosticCode::unknown_opcode, at,
                   "opcode #" +
                       std::to_string(static_cast<unsigned>(inst.opcode)) +
                       " is not defined");
            result.state = ValueState::poisoned;
            return;
        }
        if (!is_known_type(inst.type)) {
            report(DiagnosticCode::unknown_type, at,
                   std::string(to_string(inst.opcode)) + " has unknown " +
                       type_name(inst.type));
            if (inst.opcode == Opcode::store_output) {
                result.state = ValueState::none;
                // Treat the binding as stored to avoid a cascading
                // output_never_stored diagnostic.
                if (inst.immediate < stored_.size()) {
                    stored_[inst.immediate] = true;
                }
            } else {
                result.state = ValueState::poisoned;
            }
            return;
        }

        switch (inst.opcode) {
        case Opcode::constant:
            check_constant(at, inst);
            break;
        case Opcode::load_parameter:
            check_load(at, inst, program_.parameters,
                       DiagnosticCode::parameter_out_of_range,
                       "parameter slot");
            break;
        case Opcode::load_input:
            check_load(at, inst, program_.inputs,
                       DiagnosticCode::input_out_of_range, "input binding");
            break;
        case Opcode::fadd:
        case Opcode::fmul:
            check_binary_f32(at, inst);
            break;
        case Opcode::store_output:
            check_store(at, inst);
            result.state = ValueState::none;
            return;
        }

        result.type = inst.type;
        result.state = instruction_errors_ == 0 ? ValueState::defined
                                                : ValueState::poisoned;
    }

    void check_constant(std::uint32_t at, const Instruction& inst) {
        require_no_operand(at, inst.operand0, "first");
        require_no_operand(at, inst.operand1, "second");
        if (inst.type == ScalarType::f32 &&
            !is_finite_f32_bits(inst.immediate)) {
            report(DiagnosticCode::nonfinite_constant, at,
                   "f32 constant is an infinity or NaN");
        }
    }

    void check_load(std::uint32_t at, const Instruction& inst,
                    const std::vector<ScalarType>& declarations,
                    DiagnosticCode out_of_range, const char* what) {
        require_no_operand(at, inst.operand0, "first");
        require_no_operand(at, inst.operand1, "second");
        if (inst.immediate >= declarations.size()) {
            report(out_of_range, at,
                   std::string(what) + " " + std::to_string(inst.immediate) +
                       " is not declared");
            return;
        }
        const ScalarType declared = declarations[inst.immediate];
        if (is_known_type(declared) && declared != inst.type) {
            report(DiagnosticCode::result_type_mismatch, at,
                   std::string(what) + " " + std::to_string(inst.immediate) +
                       " is declared " + type_name(declared) +
                       " but loaded as " + type_name(inst.type));
        }
    }

    void check_binary_f32(std::uint32_t at, const Instruction& inst) {
        require_zero_immediate(at);
        if (inst.type != ScalarType::f32) {
            report(DiagnosticCode::result_type_mismatch, at,
                   std::string(to_string(inst.opcode)) +
                       " produces f32, not " + type_name(inst.type));
        }
        const std::array<std::pair<ValueId, const char*>, 2> operands = {{
            {inst.operand0, "first"},
            {inst.operand1, "second"},
        }};
        for (const auto& [operand, role] : operands) {
            const ValueInfo* info = check_operand(at, operand, role);
            if (info != nullptr && info->state == ValueState::poisoned) {
                // Suppress the cascade but keep this result unusable.
                ++instruction_errors_;
            } else if (info != nullptr && info->type != ScalarType::f32) {
                report(DiagnosticCode::operand_type_mismatch, at,
                       std::string(role) + " operand of " +
                           to_string(inst.opcode) + " is " +
                           type_name(info->type) + ", expected f32");
            }
        }
    }

    void check_store(std::uint32_t at, const Instruction& inst) {
        require_no_operand(at, inst.operand1, "second");
        const ValueInfo* info = check_operand(at, inst.operand0, "first");
        if (info != nullptr && info->state == ValueState::defined &&
            info->type != inst.type) {
            report(DiagnosticCode::operand_type_mismatch, at,
                   "stored value is " + type_name(info->type) +
                       " but the store is typed " + type_name(inst.type));
        }
        if (inst.immediate >= program_.outputs.size()) {
            report(DiagnosticCode::output_out_of_range, at,
                   "output binding " + std::to_string(inst.immediate) +
                       " is not declared");
            return;
        }
        const ScalarType declared = program_.outputs[inst.immediate];
        if (is_known_type(declared) && declared != inst.type) {
            report(DiagnosticCode::output_type_mismatch, at,
                   "output binding " + std::to_string(inst.immediate) +
                       " is declared " + type_name(declared) +
                       " but the store is typed " + type_name(inst.type));
        }
        if (stored_[inst.immediate]) {
            report(DiagnosticCode::output_stored_twice, at,
                   "output binding " + std::to_string(inst.immediate) +
                       " is already stored");
        }
        stored_[inst.immediate] = true;
    }

    const Program& program_;
    std::vector<Diagnostic> diagnostics_;
    std::vector<ValueInfo> values_;
    std::vector<bool> stored_;
    std::size_t instruction_errors_ = 0;
};

}  // namespace

bool is_valid_entry_point(const std::string& name) {
    if (name.empty() || name.size() > kMaxEntryPointLength) {
        return false;
    }
    if (!is_ascii_letter(name[0])) {
        return false;
    }
    for (char c : name) {
        if (!is_ascii_letter(c) && !is_ascii_digit(c) && c != '_') {
            return false;
        }
    }
    // Avoid reserved identifiers in both logical and generated symbols.
    if (name.find("__") != std::string::npos) {
        return false;
    }
    if (std::string_view(name).starts_with(kGeneratedPrefix)) {
        return false;
    }
    if (std::binary_search(std::begin(kReservedWords), std::end(kReservedWords),
                           std::string_view(name))) {
        return false;
    }
    return !is_vector_or_matrix_type_name(name);
}

std::vector<Diagnostic> validate(const Program& program) {
    return Validator(program).run();
}

}  // namespace armigo::shader
