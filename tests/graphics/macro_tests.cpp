// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/macro.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using armigo::graphics::execute_macro;
using armigo::graphics::MacroCallbacks;
using armigo::graphics::MacroResult;
using armigo::graphics::MacroResultCode;

unsigned failures = 0;

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (false)

constexpr std::uint32_t value_word(std::uint32_t operation,
                                   std::uint32_t assignment,
                                   std::uint32_t destination,
                                   std::uint32_t source_a,
                                   std::uint32_t source_b,
                                   std::uint32_t alu_operation = 0,
                                   bool exit = false)
{
    return operation | (assignment << 4) | (static_cast<std::uint32_t>(exit) << 7) |
           (destination << 8) | (source_a << 11) | (source_b << 14) | (alu_operation << 17);
}

constexpr std::uint32_t immediate_word(std::uint32_t operation,
                                       std::uint32_t assignment,
                                       std::uint32_t destination,
                                       std::uint32_t source_a,
                                       std::int32_t immediate,
                                       bool exit = false)
{
    return operation | (assignment << 4) | (static_cast<std::uint32_t>(exit) << 7) |
           (destination << 8) | (source_a << 11) |
           ((static_cast<std::uint32_t>(immediate) & 0x3ffffU) << 14);
}

constexpr std::uint32_t bitfield_word(std::uint32_t operation,
                                      std::uint32_t assignment,
                                      std::uint32_t destination,
                                      std::uint32_t source_a,
                                      std::uint32_t source_b,
                                      std::uint32_t source_bit,
                                      std::uint32_t width,
                                      std::uint32_t destination_bit,
                                      bool exit = false)
{
    return value_word(operation, assignment, destination, source_a, source_b, source_bit, exit) |
           (width << 22) | (destination_bit << 27);
}

constexpr std::uint32_t branch_word(std::uint32_t source,
                                    bool branch_on_nonzero,
                                    bool annul,
                                    std::int32_t displacement,
                                    bool exit = false)
{
    return 7U | (static_cast<std::uint32_t>(branch_on_nonzero) << 4) |
           (static_cast<std::uint32_t>(annul) << 5) | (static_cast<std::uint32_t>(exit) << 7) |
           (source << 11) | ((static_cast<std::uint32_t>(displacement) & 0x3ffffU) << 14);
}

constexpr std::uint32_t nop(bool exit = false)
{
    return value_word(1, 1, 0, 0, 0, 0, exit);
}

struct Sink {
    std::array<std::uint32_t, 64> registers{};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> emitted;
    std::uint32_t stopped_method = std::numeric_limits<std::uint32_t>::max();
    bool fail_read = false;
    bool fail_emit = false;

    static MacroResult read(void* raw, std::uint32_t method, std::uint32_t& value) noexcept
    {
        auto& self = *static_cast<Sink*>(raw);
        if (self.fail_read || method >= self.registers.size()) {
            return {MacroResultCode::ReadFailed, "synthetic register read failed", 0};
        }
        value = self.registers[method];
        return {};
    }

    static MacroResult emit(void* raw, std::uint32_t method, std::uint32_t value) noexcept
    {
        auto& self = *static_cast<Sink*>(raw);
        if (method == self.stopped_method) {
            return {MacroResultCode::CallbackStopped, "synthetic rendering boundary", 0};
        }
        if (self.fail_emit) {
            return {MacroResultCode::EmitFailed, "synthetic method sink failed", 0};
        }
        self.emitted.emplace_back(method, value);
        return {};
    }

    MacroCallbacks callbacks() noexcept { return {read, emit, this}; }
};

MacroResult run(std::span<const std::uint32_t> code,
                std::span<const std::uint32_t> arguments,
                Sink& sink,
                std::uint32_t entry = 0,
                std::uint32_t budget = 1024)
{
    return execute_macro(code, entry, arguments, sink.callbacks(), budget);
}

void check_emit(const Sink& sink, std::size_t index, std::uint32_t method, std::uint32_t value)
{
    CHECK(index < sink.emitted.size());
    if (index < sink.emitted.size()) {
        CHECK(sink.emitted[index].first == method);
        CHECK(sink.emitted[index].second == value);
    }
}

void test_validation_and_budget()
{
    Sink sink;
    const std::array one_argument{0U};
    CHECK(run({}, one_argument, sink).code == MacroResultCode::InvalidEntry);

    const std::array code{nop(true), nop()};
    CHECK(run(code, {}, sink).code == MacroResultCode::MissingArgument);
    CHECK(run(code, one_argument, sink, 0, 0).code == MacroResultCode::InstructionBudgetExceeded);
    CHECK(run(code, one_argument, sink, 2).code == MacroResultCode::InvalidEntry);

    const std::array no_exit{nop()};
    CHECK(run(no_exit, one_argument, sink).code == MacroResultCode::ProgramCounterOutOfRange);

    const std::array invalid_operation{value_word(6, 1, 1, 0, 0)};
    CHECK(run(invalid_operation, one_argument, sink).code == MacroResultCode::InvalidOperation);
    const std::array invalid_alu{value_word(0, 1, 1, 0, 0, 4)};
    CHECK(run(invalid_alu, one_argument, sink).code == MacroResultCode::InvalidAluOperation);

    const std::array loop{branch_word(0, false, true, 0)};
    const MacroResult exhausted = run(loop, one_argument, sink, 0, 5);
    CHECK(exhausted.code == MacroResultCode::InstructionBudgetExceeded);
    CHECK(exhausted.pc == 0);
}

void test_argument_policy_and_zero_register()
{
    Sink sink;
    const std::array one_argument{0x44U};
    const std::array two_arguments{0x44U, 0x55U};

    const std::array no_fetch{nop(true), nop()};
    CHECK(run(no_fetch, one_argument, sink));
    CHECK(run(no_fetch, two_arguments, sink).code == MacroResultCode::SurplusArguments);

    const std::array missing_fetch{immediate_word(1, 0, 2, 0, 0), nop(true), nop()};
    CHECK(run(missing_fetch, one_argument, sink).code == MacroResultCode::MissingArgument);

    const std::array zero_register{
        immediate_word(1, 1, 0, 0, 7),
        immediate_word(1, 2, 7, 0, 0x140),
        value_word(0, 4, 2, 0, 0, 8, true),
        nop(),
    };
    CHECK(run(zero_register, one_argument, sink));
    CHECK(sink.emitted.size() == 1);
    check_emit(sink, 0, 0x140, 0);
}

void test_add_and_carry()
{
    Sink sink;
    const std::array arguments{0xffffffffU, 1U};
    const std::array code{
        immediate_word(1, 0, 2, 0, 0),
        immediate_word(1, 2, 7, 0, 0x2100),
        value_word(0, 4, 3, 1, 2, 0),
        value_word(0, 4, 4, 0, 0, 1, true),
        nop(),
    };
    CHECK(run(code, arguments, sink));
    CHECK(sink.emitted.size() == 2);
    check_emit(sink, 0, 0x100, 0);
    check_emit(sink, 1, 0x102, 1);
}

void test_subtract_and_borrow()
{
    Sink sink;
    const std::array arguments{0U, 1U};
    const std::array code{
        immediate_word(1, 0, 2, 0, 0),
        immediate_word(1, 2, 7, 0, 0x1120),
        value_word(0, 4, 6, 1, 2, 2),
        immediate_word(1, 1, 3, 0, 5),
        immediate_word(1, 1, 4, 0, 3),
        value_word(0, 4, 5, 3, 4, 3),
        value_word(0, 4, 5, 3, 4, 3, true),
        nop(),
    };
    CHECK(run(code, arguments, sink));
    CHECK(sink.emitted.size() == 3);
    check_emit(sink, 0, 0x120, 0xffffffffU);
    check_emit(sink, 1, 0x121, 1);
    check_emit(sink, 2, 0x122, 2);
}

void test_carry_preservation()
{
    Sink sink;
    sink.registers[3] = 0xa5a5a5a5U;
    const std::array arguments{0xffffffffU, 1U};
    const std::array code{
        immediate_word(1, 0, 2, 0, 0),
        value_word(0, 1, 3, 1, 2, 0),
        value_word(0, 1, 4, 1, 2, 8),
        immediate_word(1, 1, 4, 4, 7),
        bitfield_word(2, 1, 4, 4, 2, 0, 8, 4),
        immediate_word(5, 1, 4, 0, 3),
        immediate_word(1, 2, 7, 0, 0x150),
        value_word(0, 4, 5, 0, 0, 1, true),
        nop(),
    };
    CHECK(run(code, arguments, sink));
    CHECK(sink.emitted.size() == 1);
    check_emit(sink, 0, 0x150, 1);
}

void test_immediates_and_logic()
{
    Sink immediate_sink;
    const std::array one_argument{0U};
    const std::array immediates{
        immediate_word(1, 2, 7, 0, 0x1250),
        immediate_word(1, 4, 2, 0, -131072),
        immediate_word(1, 4, 2, 0, 131071, true),
        nop(),
    };
    CHECK(run(immediates, one_argument, immediate_sink));
    CHECK(immediate_sink.emitted.size() == 2);
    check_emit(immediate_sink, 0, 0x250, 0xfffe0000U);
    check_emit(immediate_sink, 1, 0x251, 0x1ffffU);

    Sink logic_sink;
    const std::array arguments{0xf0f000ffU, 0x0ff00f0fU};
    const std::array logic{
        immediate_word(1, 0, 2, 0, 0),
        immediate_word(1, 2, 7, 0, 0x1260),
        value_word(0, 4, 3, 1, 2, 8),
        value_word(0, 4, 3, 1, 2, 9),
        value_word(0, 4, 3, 1, 2, 10),
        value_word(0, 4, 3, 1, 2, 11),
        value_word(0, 4, 3, 1, 2, 12, true),
        nop(),
    };
    CHECK(run(logic, arguments, logic_sink));
    CHECK(logic_sink.emitted.size() == 5);
    check_emit(logic_sink, 0, 0x260, 0xff000ff0U);
    check_emit(logic_sink, 1, 0x261, 0xfff00fffU);
    check_emit(logic_sink, 2, 0x262, 0x00f0000fU);
    check_emit(logic_sink, 3, 0x263, 0xf00000f0U);
    check_emit(logic_sink, 4, 0x264, 0xff0ffff0U);
}

void test_bitfield_operations()
{
    Sink sink;
    const std::array arguments{0xffff0000U, 0x12345678U};
    const std::array code{
        immediate_word(1, 0, 2, 0, 0),
        immediate_word(1, 2, 7, 0, 0x1160),
        bitfield_word(2, 4, 6, 1, 2, 8, 8, 4),
        immediate_word(1, 1, 3, 0, 8),
        bitfield_word(3, 4, 6, 3, 2, 0, 8, 4),
        immediate_word(1, 1, 3, 0, 4),
        bitfield_word(4, 4, 6, 3, 2, 8, 8, 0, true),
        nop(),
    };
    CHECK(run(code, arguments, sink));
    CHECK(sink.emitted.size() == 3);
    check_emit(sink, 0, 0x160, 0xffff0560U);
    check_emit(sink, 1, 0x161, 0x560U);
    check_emit(sink, 2, 0x162, 0x560U);

    Sink boundary_sink;
    const std::array shift_arguments{32U, 1U};
    const std::array unsupported{
        immediate_word(1, 0, 2, 0, 0),
        bitfield_word(3, 1, 3, 1, 2, 0, 1, 0),
    };
    CHECK(run(unsupported, shift_arguments, boundary_sink).code ==
          MacroResultCode::UnsupportedVariableShift);

    Sink width_sink;
    const std::array width_arguments{0U, 0xffffffffU};
    const std::array widths{
        immediate_word(1, 0, 2, 0, 0),
        immediate_word(1, 2, 7, 0, 0x1270),
        bitfield_word(2, 4, 3, 1, 2, 0, 0, 0),
        bitfield_word(2, 4, 3, 1, 2, 0, 31, 0, true),
        nop(),
    };
    CHECK(run(widths, width_arguments, width_sink));
    CHECK(width_sink.emitted.size() == 2);
    check_emit(width_sink, 0, 0x270, 0);
    check_emit(width_sink, 1, 0x271, 0x7fffffffU);
}

void test_assignment_operations_and_method_increment()
{
    Sink sink;
    const std::array arguments{0U, 0xaaU, 0xbbU, 0xccU, 0xddU};
    const std::array code{
        immediate_word(1, 0, 2, 0, 0),
        immediate_word(1, 2, 7, 0, 0x2170),
        value_word(0, 3, 3, 2, 0, 9),
        value_word(0, 4, 4, 3, 0, 9),
        immediate_word(1, 5, 5, 0, 0x3180),
        value_word(0, 4, 6, 5, 0, 9),
        immediate_word(1, 6, 6, 0, 0x1190),
        immediate_word(1, 7, 6, 0, 0x21a0, true),
        nop(),
    };
    CHECK(run(code, arguments, sink));
    CHECK(sink.emitted.size() == 5);
    check_emit(sink, 0, 0x170, 0xaaU);
    check_emit(sink, 1, 0x172, 0xbbU);
    check_emit(sink, 2, 0x180, 0xccU);
    check_emit(sink, 3, 0x190, 0xddU);
    check_emit(sink, 4, 0x1a0, 2U);

    Sink overflow_sink;
    const std::array overflow{
        immediate_word(1, 2, 7, 0, 0x1fff),
        immediate_word(1, 4, 3, 0, 1),
    };
    CHECK(run(overflow, std::span<const std::uint32_t>(arguments).first(1), overflow_sink).code ==
          MacroResultCode::UnsupportedMethodAddressOverflow);
    CHECK(overflow_sink.emitted.empty());

    Sink fixed_sink;
    const std::array fixed{
        immediate_word(1, 2, 7, 0, 0x240),
        immediate_word(1, 4, 3, 0, 1),
        immediate_word(1, 4, 3, 0, 2, true),
        nop(),
    };
    CHECK(run(fixed, std::span<const std::uint32_t>(arguments).first(1), fixed_sink));
    CHECK(fixed_sink.emitted.size() == 2);
    check_emit(fixed_sink, 0, 0x240, 1);
    check_emit(fixed_sink, 1, 0x240, 2);
}

void test_register_read_and_callback_failures()
{
    Sink sink;
    sink.registers[3] = 0x76543210U;
    const std::array arguments{5U};
    const std::array code{
        immediate_word(1, 2, 7, 0, 0x1b0),
        immediate_word(5, 4, 2, 1, -2, true),
        nop(),
    };
    CHECK(run(code, arguments, sink));
    CHECK(sink.emitted.size() == 1);
    check_emit(sink, 0, 0x1b0, 0x76543210U);

    Sink read_failure;
    read_failure.fail_read = true;
    MacroResult failed = run(code, arguments, read_failure);
    CHECK(failed.code == MacroResultCode::ReadFailed);
    CHECK(failed.detail == std::string_view("synthetic register read failed"));
    CHECK(failed.pc == 1);

    Sink stopped;
    stopped.stopped_method = 0x1b0;
    MacroResult stop = run(code, arguments, stopped);
    CHECK(stop.code == MacroResultCode::CallbackStopped);
    CHECK(stop.detail == std::string_view("synthetic rendering boundary"));
    CHECK(stop.pc == 1);
    CHECK(stopped.emitted.empty());

    Sink emit_failure;
    emit_failure.fail_emit = true;
    failed = run(code, arguments, emit_failure);
    CHECK(failed.code == MacroResultCode::EmitFailed);
    CHECK(failed.pc == 1);

    CHECK(execute_macro(code, 0, arguments, MacroCallbacks{nullptr, Sink::emit, &sink}, 100).code ==
          MacroResultCode::ReadCallbackUnavailable);
    CHECK(execute_macro(code, 0, arguments, MacroCallbacks{Sink::read, nullptr, &sink}, 100).code ==
          MacroResultCode::EmitCallbackUnavailable);
}

void test_branches_and_exit_delay()
{
    const std::array arguments{0U};
    const auto program = [](bool annul) {
        return std::array{
            immediate_word(1, 2, 7, 0, 0x1200),
            branch_word(0, false, annul, 3, true),
            immediate_word(1, 4, 1, 0, 0x11, true),
            immediate_word(1, 4, 1, 0, 0x22),
            immediate_word(1, 4, 1, 0, 0x33, true),
            immediate_word(1, 4, 1, 0, 0x44),
        };
    };

    Sink delayed_sink;
    const auto delayed = program(false);
    CHECK(run(delayed, arguments, delayed_sink));
    CHECK(delayed_sink.emitted.size() == 3);
    check_emit(delayed_sink, 0, 0x200, 0x11);
    check_emit(delayed_sink, 1, 0x201, 0x33);
    check_emit(delayed_sink, 2, 0x202, 0x44);

    Sink annulled_sink;
    const auto annulled = program(true);
    CHECK(run(annulled, arguments, annulled_sink));
    CHECK(annulled_sink.emitted.size() == 2);
    check_emit(annulled_sink, 0, 0x200, 0x33);
    check_emit(annulled_sink, 1, 0x201, 0x44);

    Sink not_taken_sink;
    const std::array not_taken{
        immediate_word(1, 2, 7, 0, 0x220),
        branch_word(0, true, false, 3, true),
        immediate_word(1, 4, 1, 0, 0x55),
        immediate_word(1, 4, 1, 0, 0x66),
        nop(),
    };
    CHECK(run(not_taken, arguments, not_taken_sink));
    CHECK(not_taken_sink.emitted.size() == 1);
    check_emit(not_taken_sink, 0, 0x220, 0x55);

    Sink invalid_sink;
    const std::array invalid_target{branch_word(0, false, true, -1)};
    CHECK(run(invalid_target, arguments, invalid_sink).code ==
          MacroResultCode::InvalidBranchTarget);

    const std::array branch_in_delay{
        nop(true),
        branch_word(0, false, true, 0),
    };
    CHECK(run(branch_in_delay, arguments, invalid_sink).code == MacroResultCode::BranchInDelaySlot);

    Sink backward_sink;
    const std::array backward_arguments{2U};
    const std::array backward{
        immediate_word(1, 2, 7, 0, 0x1280),
        value_word(0, 4, 2, 1, 0, 9),
        immediate_word(1, 1, 1, 1, -1),
        branch_word(1, true, true, -2),
        value_word(0, 4, 2, 1, 0, 9, true),
        nop(),
    };
    CHECK(run(backward, backward_arguments, backward_sink));
    CHECK(backward_sink.emitted.size() == 3);
    check_emit(backward_sink, 0, 0x280, 2);
    check_emit(backward_sink, 1, 0x281, 1);
    check_emit(backward_sink, 2, 0x282, 0);
}

void test_entry_word()
{
    Sink sink;
    const std::array arguments{0U};
    const std::array code{
        value_word(6, 1, 0, 0, 0),
        immediate_word(1, 2, 7, 0, 0x230),
        immediate_word(1, 4, 1, 0, 0x77, true),
        nop(),
    };
    CHECK(run(code, arguments, sink, 1));
    CHECK(sink.emitted.size() == 1);
    check_emit(sink, 0, 0x230, 0x77);
}

}  // namespace

int main()
{
    test_validation_and_budget();
    test_argument_policy_and_zero_register();
    test_add_and_carry();
    test_subtract_and_borrow();
    test_carry_preservation();
    test_immediates_and_logic();
    test_bitfield_operations();
    test_assignment_operations_and_method_increment();
    test_register_read_and_callback_failures();
    test_branches_and_exit_delay();
    test_entry_word();
    return failures == 0 ? 0 : 1;
}
