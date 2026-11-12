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
    // The driver's version-20 reader consumes control bytes [0, 0xe0) and is
    // given no length, so a runtime control is exactly that span.
    if (record.kind == ContainerKind::runtime && c.size() != runtime_control_bytes)
        return fail(ProgramError::truncated_control, "Runtime native control must be exactly the version-20 reader span");
    // Version 17 carries every field read below at the version-20 offsets
    // (provisional: one compute stage shows it; the trailing words after
    // +0xbb, which nothing reads, differ).
    const auto version = little_endian(c, 8, 4);
    if (little_endian(c, 0, 4) != 0x98761234 || little_endian(c, 4, 4) != 0xc9 || (version != 20 && version != 17))
        return fail(ProgramError::control_family, "Unsupported native control family or version");
    const auto selector = static_cast<std::uint8_t>(little_endian(c, 0x56, 1));
    if (selector != 0 && selector != 1 && selector != 5 && selector != 6 && selector != 7)
        return fail(ProgramError::unsupported_stage, "Native stage is not implemented", selector);
    if (record.kind == ContainerKind::runtime && record.slot)
        return fail(ProgramError::stage_slot, "Runtime native records have no container slot", selector);
    if (record.slot) {
        const unsigned expected = selector == 0 ? 0 : selector == 1 ? 4 : selector == 5 ? 5 : selector + 1u;
        if (*record.slot != expected) return fail(ProgramError::stage_slot, "Native stage disagrees with BNSH slot", selector);
    }
    // Container extents at +0x30 (BNSH) and +0xd4 (BAHS) bound file records;
    // runtime records carry none, and their caller bounds the control.
    const auto extent = little_endian(c, 0x30, 4);
    if ((record.kind == ContainerKind::bnsh && extent != c.size()) ||
        (record.kind == ContainerKind::bahs && ((extent && extent != c.size()) || little_endian(c, 0xd4, 4) != c.size())) ||
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
    if (p.stage == Stage::fragment)
        p.early_fragment_tests = little_endian(c, 0x5b, 1) != 0 &&
            (little_endian(p.header, 19 * 4, 4) & 2u) == 0;
    if (p.stage == Stage::compute) {
        ComputeLaunchFacts launch{{static_cast<std::uint32_t>(little_endian(c, 0x58, 4)),
                                   static_cast<std::uint32_t>(little_endian(c, 0x5c, 4)),
                                   static_cast<std::uint32_t>(little_endian(c, 0x60, 4))},
                                  static_cast<std::uint32_t>(little_endian(c, 0x64, 4))};
        if (std::any_of(launch.local_size.begin(), launch.local_size.end(), [](auto n) { return n == 0 || n > 1024; }) ||
            std::uint64_t{launch.local_size[0]} * launch.local_size[1] * launch.local_size[2] > 1024)
            return fail(ProgramError::program_bounds, "Native compute thread group size is invalid", selector);
        p.launch = launch;
    }
    if (p.stage == Stage::task || p.stage == Stage::mesh) {
        MeshLaunchFacts launch;
        launch.threads = static_cast<std::uint32_t>(little_endian(c, 0x58, 2));
        launch.payload_bytes = static_cast<std::uint32_t>(little_endian(c, 0x5a, 2));
        if (launch.threads == 0 || launch.threads > 1024)
            return fail(ProgramError::program_bounds, "Native task or mesh workgroup size is invalid", selector);
        if (p.stage == Stage::mesh) {
            launch.task = little_endian(c, 0x5e, 1) != 0;
            const auto vertices = static_cast<std::uint8_t>(little_endian(c, 0x5f, 1));
            launch.primitive_vertices = vertices ? vertices : 3;
            launch.max_vertices = static_cast<std::uint32_t>(little_endian(c, 0x60, 2));
            launch.max_primitives = static_cast<std::uint32_t>(little_endian(c, 0x62, 2));
            if (launch.primitive_vertices > 3)
                return fail(ProgramError::program_bounds, "Native mesh primitive topology is invalid", selector);
            if (little_endian(c, 0x68, 1) & 0x80) {
                const auto at = static_cast<std::uint32_t>(little_endian(c, 0x64, 4));
                if (at % 16 || at < header_size || std::uint64_t{at} + 128 > end)
                    return fail(ProgramError::program_bounds, "Native mesh per-primitive header is outside the program", selector);
                launch.primitive_header = at;
            }
        }
        p.mesh_launch = launch;
    }
    p.image_digest = digest(p.image); p.control_digest = digest(c); p.payload_digest = digest(code);
    // The identity covers the canonical control: the reader span [0, 0xe0)
    // with the container extents zeroed. A file record and the runtime
    // control a driver receives for the same program therefore share it;
    // file-only bytes and extents stay outside.
    Buffer canonical(c.begin(), c.begin() + std::min(c.size(), runtime_control_bytes));
    canonical.resize(runtime_control_bytes);
    for (const std::size_t at : {std::size_t{0x30}, std::size_t{0xd4}})
        std::fill(canonical.begin() + at, canonical.begin() + at + 4, std::byte{});
    p.canonical_control_digest = digest(canonical);
    const auto identity = "armigo-stage/ampere128/nvn2-c9-v" + std::to_string(version) + "/2\n" + std::to_string(selector) + "\n" +
                          p.canonical_control_digest + "\n" + p.payload_digest + "\n";
    p.stage_identity = digest(std::as_bytes(std::span(identity)));
    return p;
}
} // namespace armigo::shader_native
