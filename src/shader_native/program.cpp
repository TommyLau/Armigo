// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/program.h>
#include <algorithm>

namespace armigo::shader_native {
ProgramResult parse_program(const Record& record) {
    const auto c = record.control, code = record.code;
    auto fail = [&](ProgramError error, std::string message, std::uint8_t selector = 0) -> ProgramResult {
        return ProgramFailure{error, selector, std::move(message)};
    };
    if (c.size() < 0xd8) return fail(ProgramError::truncated_control, "Native control is truncated");
    if (little_endian(c, 0, 4) != 0x98761234 || little_endian(c, 4, 4) != 0xc9 || little_endian(c, 8, 4) != 20)
        return fail(ProgramError::control_family, "Unsupported native control family or version");
    const auto selector = static_cast<std::uint8_t>(little_endian(c, 0x56, 1));
    if (selector != 0 && selector != 1 && selector != 5)
        return fail(ProgramError::unsupported_stage,
                    selector == 6 || selector == 7 ? "Native stage 6/7 semantics are not established" : "Native stage is not implemented", selector);
    if (record.slot) {
        const unsigned expected = selector == 0 ? 0 : selector == 1 ? 4 : 5;
        if (*record.slot != expected) return fail(ProgramError::stage_slot, "Native stage disagrees with BNSH slot", selector);
    }
    const auto extent = little_endian(c, 0x30, 4);
    if ((record.kind == ContainerKind::bnsh ? extent != c.size() : extent && extent != c.size()) ||
        (record.kind == ContainerKind::bahs && little_endian(c, 0xd4, 4) != c.size()) ||
        little_endian(c, 0x44, 4) != code.size())
        return fail(ProgramError::payload_size, "Native control and payload extents disagree", selector);
    const auto end = little_endian(c, 0x38, 4), constant_size = little_endian(c, 0x3c, 4);
    const auto constant_offset = little_endian(c, 0x40, 4);
    const std::size_t header_size = selector == 5 ? 0 : 128;
    if (end <= header_size || end > constant_offset || constant_offset > code.size() || constant_size > code.size() - constant_offset)
        return fail(ProgramError::program_bounds, "Native program, padding or constant bounds are invalid", selector);
    if ((end - header_size) % 16)
        return fail(ProgramError::instruction_alignment, "Native instruction extent is not a multiple of 16 bytes", selector);
    auto padding = code.subspan(end, constant_offset - end);
    auto trailing_padding = code.subspan(constant_offset + constant_size);
    const auto nonzero = [](std::byte b) { return b != std::byte{}; };
    if (std::any_of(padding.begin(), padding.end(), nonzero) ||
        std::any_of(trailing_padding.begin(), trailing_padding.end(), nonzero))
        return fail(ProgramError::nonzero_padding, "Unclassified nonzero bytes follow the native program image", selector);
    Program p;
    p.stage = static_cast<Stage>(selector); p.control = c; p.payload = code;
    p.image = code.first(end); p.header = code.first(header_size);
    p.instructions = code.subspan(header_size, end - header_size); p.padding = padding;
    p.constants = code.subspan(constant_offset, constant_size); p.trailing_padding = trailing_padding;
    if (p.stage == Stage::compute)
        p.compute_metadata = ComputeMetadata{{static_cast<std::uint32_t>(little_endian(c, 0x58, 4)),
                                               static_cast<std::uint32_t>(little_endian(c, 0x5c, 4)),
                                               static_cast<std::uint32_t>(little_endian(c, 0x60, 4))}};
    p.image_digest = digest(p.image); p.control_digest = digest(c); p.payload_digest = digest(code);
    const auto identity = std::string("armigo-stage/ampere128/nvn2-c9-v20/1\n") + std::to_string(selector) + "\n" +
                          p.control_digest + "\n" + p.payload_digest + "\n";
    p.stage_identity = digest(std::as_bytes(std::span(identity)));
    return p;
}
} // namespace armigo::shader_native
