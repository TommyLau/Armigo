// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_input/container.h>

#include <array>
#include <cstring>
#include <set>
#include <utility>

namespace armigo::shader_input {
namespace {

constexpr std::array<Stage, 6> slot_stages = {
    Stage::Vertex, Stage::TessellationControl, Stage::TessellationEvaluation,
    Stage::Geometry, Stage::Fragment, Stage::Compute,
};

// Readers are called only after the enclosing range has been checked.
std::uint32_t u32(ByteSpan bytes, std::size_t at) {
    std::uint32_t value = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
        value |= static_cast<std::uint32_t>(bytes.data[at + byte]) << (8 * byte);
    }
    return value;
}

std::uint64_t u64(ByteSpan bytes, std::size_t at) {
    return u32(bytes, at) | (static_cast<std::uint64_t>(u32(bytes, at + 4)) << 32);
}

bool covers(ByteSpan bytes, std::uint64_t offset, std::uint64_t size) {
    return offset <= bytes.size && size <= bytes.size - offset;
}

ContainerError failure(ContainerError::Code code, std::uint64_t offset = 0,
                       std::uint64_t value = 0, std::uint64_t bound = 0) {
    return {code, offset, value, bound, std::nullopt};
}

std::optional<ContainerError> header(ByteSpan bytes, const char* magic,
                                     std::size_t magic_size,
                                     std::size_t minimum) {
    if (bytes.data == nullptr && bytes.size != 0) {
        return failure(ContainerError::Code::InvalidSpan);
    }
    if (bytes.size < minimum) {
        return failure(ContainerError::Code::Truncated, 0, bytes.size, minimum);
    }
    if (std::memcmp(bytes.data, magic, magic_size) != 0) {
        return failure(ContainerError::Code::Magic);
    }
    return std::nullopt;
}

}  // namespace

const char* ContainerError::message() const noexcept {
    switch (code) {
        case Code::InvalidSpan: return "container span has a null data pointer";
        case Code::Truncated: return "container header is truncated";
        case Code::Magic: return "container magic does not match";
        case Code::MissingResourceBlock: return "BNSH resource block is missing";
        case Code::Range: return "container field points outside its bounded region";
        case Code::VariationLimit: return "BNSH variation count exceeds the limit";
        case Code::StageLimit: return "container stage count exceeds the limit";
        case Code::ScanLimit: return "BAHS table scan exceeds the byte limit";
        case Code::CandidateLimit: return "BAHS candidate count exceeds the limit";
        case Code::ControlSizeMismatch: return "descriptor and control sizes differ";
        case Code::CodeSizeMismatch: return "descriptor and shader code sizes differ";
        case Code::SlotStageMismatch: return "BNSH slot and shader stages differ";
        case Code::ShaderInvalid: return "container shader failed structural validation";
    }
    return "unknown shader container error";
}

BnshResult parse_bnsh(ByteSpan input, ContainerLimits limits) {
    if (const auto error = header(input, "BNSH\0\0\0\0", 8, 0x88)) {
        return *error;
    }
    if (std::memcmp(input.data + 0x60, "grsc", 4) != 0) {
        return failure(ContainerError::Code::MissingResourceBlock, 0x60);
    }
    BnshContainer result;
    result.variation_count = u32(input, 0x7c);
    if (result.variation_count > limits.max_variations) {
        return failure(ContainerError::Code::VariationLimit, 0x7c,
                       result.variation_count, limits.max_variations);
    }
    const auto table = u64(input, 0x80);
    const auto table_size = std::uint64_t{0x40} * result.variation_count;
    if (!covers(input, table, table_size)) {
        return failure(ContainerError::Code::Range, 0x80, table, input.size);
    }
    for (std::uint32_t variation = 0; variation < result.variation_count;
         ++variation) {
        const auto row = static_cast<std::size_t>(table) + 0x40ull * variation;
        const auto program = u64(input, row + 0x10);
        if (program == 0) {
            continue;
        }
        if (!covers(input, program, 0x38)) {
            return failure(ContainerError::Code::Range, row + 0x10,
                           program, input.size);
        }
        for (std::size_t slot = 0; slot < slot_stages.size(); ++slot) {
            const auto pointer_at = static_cast<std::size_t>(program) + 8 + slot * 8;
            const auto descriptor = u64(input, pointer_at);
            if (descriptor == 0) {
                continue;
            }
            if (result.stages.size() == limits.max_stages) {
                return failure(ContainerError::Code::StageLimit, pointer_at,
                               result.stages.size() + 1, limits.max_stages);
            }
            if (!covers(input, descriptor, 0x20)) {
                return failure(ContainerError::Code::Range, pointer_at,
                               descriptor, input.size);
            }
            const auto at = static_cast<std::size_t>(descriptor);
            const auto control_offset = u64(input, at + 8);
            const auto code_offset = u64(input, at + 16);
            const auto code_size = u32(input, at + 24);
            const auto control_size = u32(input, at + 28);
            if (!covers(input, control_offset, control_size)) {
                return failure(ContainerError::Code::Range, at + 8,
                               control_offset, input.size);
            }
            if (!covers(input, code_offset, code_size)) {
                return failure(ContainerError::Code::Range, at + 16,
                               code_offset, input.size);
            }
            const auto parsed = parse_shader({
                {input.data + control_offset, control_size},
                {input.data + code_offset, code_size}, {}});
            if (const auto* error = std::get_if<ParseError>(&parsed)) {
                auto detail = failure(ContainerError::Code::ShaderInvalid, at);
                detail.shader_error = *error;
                return detail;
            }
            const auto& shader = std::get<ParsedShader>(parsed);
            if (shader.control_size != control_size) {
                return failure(ContainerError::Code::ControlSizeMismatch,
                               at + 28, control_size, shader.control_size);
            }
            if (shader.code_size != code_size) {
                return failure(ContainerError::Code::CodeSizeMismatch,
                               at + 24, code_size, shader.code_size);
            }
            if (shader.stage != slot_stages[slot]) {
                return failure(ContainerError::Code::SlotStageMismatch,
                               pointer_at, static_cast<std::uint32_t>(shader.stage),
                               static_cast<std::uint32_t>(slot_stages[slot]));
            }
            result.stages.push_back({shader, at,
                static_cast<std::size_t>(control_offset),
                static_cast<std::size_t>(code_offset), variation,
                static_cast<StageSlot>(slot)});
        }
    }
    return result;
}

BahsResult parse_bahs(ByteSpan input, ContainerLimits limits) {
    if (const auto error = header(input, "BAHS", 4, 0x20)) {
        return *error;
    }
    const auto region_size = u32(input, 0x18);
    const auto region_start = u32(input, 0x1c);
    if (region_start < 0x20 || !covers(input, region_start, region_size)) {
        return failure(ContainerError::Code::Range, 0x1c, region_start, input.size);
    }
    if (region_start > limits.max_scan_bytes) {
        return failure(ContainerError::Code::ScanLimit, 0x1c,
                       region_start, limits.max_scan_bytes);
    }
    BahsContainer result;
    std::set<std::pair<std::size_t, std::size_t>> accepted;
    const ByteSpan tables{input.data, region_start};
    for (std::size_t entry = 0; entry <= tables.size - 16; entry += 4) {
        ++result.scanned_positions;
        if (u32(input, entry + 4) != 0) {
            continue;
        }
        const auto code_offset = u32(input, entry);
        const auto control_size = u32(input, entry + 8);
        if (code_offset % 0x100 != 0 ||
            control_size < format::control_min_size ||
            control_size > limits::control_max_size) {
            continue;
        }
        // Decode two's-complement displacement without implementation-defined
        // unsigned-to-signed narrowing. It is relative to the entry's start.
        const auto displacement_bits = u32(input, entry + 12);
        const std::int64_t displacement = displacement_bits < 0x80000000u
            ? static_cast<std::int64_t>(displacement_bits)
            : static_cast<std::int64_t>(displacement_bits) - 0x100000000ll;
        const auto control_position = static_cast<std::int64_t>(entry) + displacement;
        if (control_position < 0 ||
            !covers(tables, static_cast<std::uint64_t>(control_position),
                    control_size)) {
            continue;
        }
        const auto control_offset = static_cast<std::size_t>(control_position);
        if (u32(input, control_offset) != format::control_magic ||
            u32(input, control_offset + format::control_size_offset) != control_size) {
            continue;
        }
        if (result.candidate_count == limits.max_candidates) {
            return failure(ContainerError::Code::CandidateLimit, entry,
                           result.candidate_count + 1, limits.max_candidates);
        }
        ++result.candidate_count;
        const auto code_size = u32(input, control_offset + format::code_size_field);
        if (code_offset > region_size || code_size > region_size - code_offset) {
            ++result.rejected_candidates;
            continue;
        }
        const auto absolute_code = static_cast<std::size_t>(region_start) + code_offset;
        const auto parsed = parse_shader({
            {input.data + control_offset, control_size},
            {input.data + absolute_code, code_size}, {}});
        if (std::holds_alternative<ParseError>(parsed)) {
            ++result.rejected_candidates;
            continue;
        }
        const auto key = std::pair{control_offset, absolute_code};
        if (accepted.contains(key)) {
            ++result.duplicate_candidates;
            continue;
        }
        if (result.stages.size() == limits.max_stages) {
            return failure(ContainerError::Code::StageLimit, entry,
                           result.stages.size() + 1, limits.max_stages);
        }
        accepted.insert(key);
        result.stages.push_back({std::get<ParsedShader>(parsed), entry,
            control_offset, absolute_code, std::nullopt, std::nullopt});
    }
    return result;
}

}  // namespace armigo::shader_input
