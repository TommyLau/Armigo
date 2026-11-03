// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/maxwell_forward_program.h>

#include "forward_reconvergence.h"
#include "maxwell_decode_helpers.h"
#include "maxwell_forward_decode.h"

#include <armigo/shader/maxwell_flow.h>

#include <map>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace armigo::shader::maxwell {
namespace {

// The first instruction slot of the program region: every 32-byte scheduling
// group starts with an 8-byte scheduling word, so the entry slot sits at 8.
constexpr std::uint32_t kEntryPc = 8;

bool is_pt(Predicate predicate) {
    return predicate.index == 7 && !predicate.inverted;
}

bool is_inverted_pt(Predicate predicate) {
    return predicate.index == 7 && predicate.inverted;
}

// Next instruction-slot pc after a fallthrough, skipping the scheduling word
// that starts each group. Returns false when the path would leave the
// program region without a full instruction word.
bool fallthrough_pc(std::uint32_t pc, std::uint32_t program_size,
                    std::uint32_t& next) {
    std::uint64_t wide = static_cast<std::uint64_t>(pc) +
                         shader_input::format::instruction_word_size;
    if (wide % shader_input::format::scheduling_group_size == 0) {
        wide += shader_input::format::instruction_word_size;
    }
    if (wide + shader_input::format::instruction_word_size > program_size) {
        return false;
    }
    next = static_cast<std::uint32_t>(wide);
    return true;
}

// Always-true flow test: the only condition the branch helper admits.
constexpr std::uint64_t kAlwaysFlowTest = 15;

// Maps a rejected branch candidate to a structured control-flow error. The
// branch helper's word-shape verdict never reaches this function: a
// not_branch word continues into the guarded instruction decoders. The
// helper reports an unsupported field only as a raw detail value, so the
// failing field is re-derived here from the word in the helper's own check
// order: flow test, constant-buffer target mode, then the conservative
// zero-required reserved ranges. The helper can report unsupported_field
// only from those four checks, so a word that passes the first two carries
// nonzero reserved bits, and the low range is reported first exactly as the
// helper does. Other rejections keep DecodeField::None and the helper's
// detail verbatim.
DecodeError branch_failure(std::uint64_t word,
                           const ForwardBranchError& branch_error) {
    if (branch_error.code != ForwardBranchErrorCode::unsupported_field) {
        return detail::failure(DecodeError::Code::UnsupportedControlFlow,
                               branch_error.pc, OpcodeFamily::Branch,
                               DecodeField::None, branch_error.detail, 0);
    }
    const std::uint64_t flow_test = detail::field(word, 0, 5);
    if (flow_test != kAlwaysFlowTest) {
        return detail::failure(DecodeError::Code::UnsupportedControlFlow,
                               branch_error.pc, OpcodeFamily::Branch,
                               DecodeField::FlowTest, flow_test,
                               kAlwaysFlowTest);
    }
    const std::uint64_t address_mode = detail::field(word, 5, 1);
    if (address_mode != 0) {
        return detail::failure(DecodeError::Code::UnsupportedControlFlow,
                               branch_error.pc, OpcodeFamily::Branch,
                               DecodeField::AddressMode, address_mode, 0);
    }
    const std::uint64_t reserved_low = detail::field(word, 6, 10);
    return detail::failure(DecodeError::Code::UnsupportedControlFlow,
                           branch_error.pc, OpcodeFamily::Branch,
                           DecodeField::ReservedBits,
                           reserved_low != 0 ? word & detail::bit_mask(10, 6)
                                             : word & detail::bit_mask(4, 44),
                           0);
}

}  // namespace

ForwardDecodeResult decode_forward_program(shader_input::ByteSpan control,
                                           shader_input::ByteSpan code,
                                           DecodeLimits limits) {
    const shader_input::ParseResult parsed_result =
        shader_input::parse_shader({control, code, {}});
    if (const auto* parse_error =
            std::get_if<shader_input::ParseError>(&parsed_result)) {
        DecodeError error = detail::failure(DecodeError::Code::ShaderParse, 0);
        error.parse_error = *parse_error;
        return error;
    }
    const shader_input::ParsedShader& parsed =
        std::get<shader_input::ParsedShader>(parsed_result);
    if (parsed.stage != shader_input::Stage::Vertex &&
        parsed.stage != shader_input::Stage::Fragment) {
        return detail::failure(DecodeError::Code::UnsupportedStage, 0,
                               OpcodeFamily::Unknown, DecodeField::None,
                               static_cast<std::uint32_t>(parsed.stage), 0);
    }

    try {
        ForwardProgram result;
        result.stage = parsed.stage;
        std::copy_n(reinterpret_cast<const std::byte*>(parsed.sph.data),
                    result.sph.size(), result.sph.begin());
        result.program_size = parsed.program_size;

        std::map<std::uint32_t, ForwardInstruction> nodes;
        detail::ReconvergenceStacks stacks;
        std::map<std::uint32_t, detail::StackId> incoming{{kEntryPc, 0}};
        std::set<std::uint32_t> worklist{kEntryPc};
        const auto enqueue = [&](std::uint32_t target, detail::StackId stack)
            -> std::optional<DecodeError> {
            const auto [entry, inserted] = incoming.emplace(target, stack);
            if (!inserted && entry->second != stack) {
                return detail::failure(DecodeError::Code::UnsupportedControlFlow,
                                       target, OpcodeFamily::ReconvergenceSync,
                                       DecodeField::ReconvergenceStack, stack,
                                       entry->second);
            }
            if (inserted) worklist.insert(target);
            return std::nullopt;
        };

        while (!worklist.empty()) {
            // All admitted edges are forward, so every predecessor reaches a
            // slot before it is decoded. Join checks also compare full stacks.
            const std::uint32_t pc = *worklist.begin();
            worklist.erase(worklist.begin());
            const detail::StackId stack = incoming.at(pc);
            if (static_cast<std::uint64_t>(pc) +
                    shader_input::format::instruction_word_size >
                parsed.program_size) {
                // A reachable path leaves the program region without a PT
                // EXIT; that is the forward-profile missing-exit condition.
                return detail::failure(DecodeError::Code::MissingExit, pc);
            }
            if (nodes.size() == limits.max_instructions) {
                return detail::failure(DecodeError::Code::InstructionLimit,
                                       pc, OpcodeFamily::Unknown,
                                       DecodeField::None, nodes.size() + 1,
                                       limits.max_instructions);
            }
            const std::uint64_t word =
                detail::read_word(parsed.code.data + parsed.program_offset +
                                  pc);

            const auto enqueue_fallthrough =
                [&](detail::StackId next_stack) -> std::optional<DecodeError> {
                std::uint32_t next = 0;
                if (!fallthrough_pc(pc, parsed.program_size, next)) {
                    return detail::failure(DecodeError::Code::MissingExit, pc);
                }
                return enqueue(next, next_stack);
            };

            const auto opcode = detail::reconvergence_opcode(word);
            if (opcode == detail::ReconvergenceOpcode::UnsupportedBreak) {
                return detail::failure(DecodeError::Code::UnsupportedControlFlow,
                                       pc, OpcodeFamily::Branch);
            }
            if (opcode == detail::ReconvergenceOpcode::Ssy) {
                auto decoded = detail::decode_ssy(word, pc, parsed.program_size);
                if (const auto* error = std::get_if<DecodeError>(&decoded)) return *error;
                auto instruction = std::get<Instruction>(std::move(decoded));
                const auto target = std::get<ReconvergenceSetup>(instruction.payload).target_pc;
                nodes.emplace(pc, std::move(instruction));
                const auto pushed = stacks.push(stack, detail::TokenKind::Ssy, target);
                if (auto error = enqueue_fallthrough(pushed)) return *error;
                continue;
            }

            ForwardBranchResult branch;
            detail::StackId taken_stack = stack;
            if (opcode == detail::ReconvergenceOpcode::Sync) {
                if (auto error = detail::validate_sync(word, pc)) return *error;
                const auto popped = stacks.pop(stack, detail::TokenKind::Ssy);
                if (!popped) {
                    return detail::failure(DecodeError::Code::UnsupportedControlFlow,
                                           pc, OpcodeFamily::ReconvergenceSync,
                                           DecodeField::ReconvergenceStack, 0, 1);
                }
                branch = resolve_forward_target(
                    popped->target_pc, pc, parsed.program_size,
                    Predicate{static_cast<std::uint8_t>(detail::field(word, 16, 3)),
                              detail::field(word, 19, 1) != 0});
                if (const auto* error = std::get_if<ForwardBranchError>(&branch)) {
                    return detail::reconvergence_target_error(
                        *error, OpcodeFamily::ReconvergenceSync);
                }
                taken_stack = popped->remaining;
            } else {
                // Only not_branch falls through to the production decoders.
                branch = decode_forward_branch(word, pc, parsed.program_size);
                if (const auto* error = std::get_if<ForwardBranchError>(&branch);
                    error != nullptr && error->code != ForwardBranchErrorCode::not_branch) {
                    return branch_failure(word, *error);
                }
            }
            const auto* forward = std::get_if<ForwardBranch>(&branch);

            if (forward != nullptr) {
                if (nodes.emplace(pc, *forward).second) {
                    // An always-PT branch admits only its target; an
                    // inverted-PT branch only its fallthrough; any other
                    // preserved guard admits both edges.
                    if (!is_inverted_pt(forward->predicate)) {
                        if (auto error = enqueue(forward->target_pc, taken_stack)) {
                            return *error;
                        }
                    }
                    if (!is_pt(forward->predicate)) {
                        if (auto error = enqueue_fallthrough(stack)) {
                            return *error;
                        }
                    }
                }
                continue;
            }

            auto decoded = decode_guarded_instruction(word, pc);
            if (auto* error = std::get_if<DecodeError>(&decoded)) {
                return *error;
            }
            Instruction instruction = std::get<Instruction>(std::move(decoded));
            const bool terminal =
                instruction.family == OpcodeFamily::Exit &&
                is_pt(instruction.predicate);
            nodes.emplace(instruction.pc, std::move(instruction));
            // Only an unconditional PT EXIT ends a path; a guarded or
            // inverted-PT EXIT also continues into its fallthrough.
            if (!terminal) {
                if (auto error = enqueue_fallthrough(stack)) {
                    return *error;
                }
            }
        }

        result.instructions.reserve(nodes.size());
        for (auto& entry : nodes) {
            result.instructions.push_back(std::move(entry.second));
        }
        return ForwardDecodeResult(std::move(result));
    } catch (const std::bad_alloc&) {
        return detail::failure(DecodeError::Code::AllocationFailure, 0);
    } catch (const std::length_error&) {
        return detail::failure(DecodeError::Code::AllocationFailure, 0);
    }
}

}  // namespace armigo::shader::maxwell
