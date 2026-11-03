// ============================================================================
// Armigo Horizon - Compatibility Metadata
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once
#include <armigo/horizon/startup_queries.h>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::horizon {

// Caller-owned records for the exact-pair host profile, not a full SVC model.
struct StackAliasRecord {
    std::uint64_t destination = 0, source = 0, size = 0;
    MemoryType source_type = MemoryType::normal;
};

struct StackAliasUnmapPlan {
    std::vector<MemoryBlock> blocks;
    std::size_t alias_index = 0;
};

enum class StackAliasError {
    InvalidRange, UnregisteredAlias, InvalidMemoryMap, InvalidSource,
    InvalidDestination, AdmissionLimit, AllocationFailure
};
struct StackAliasFailure {
    StackAliasError code;
    std::string_view detail;
};
using StackAliasUnmapResult = std::variant<StackAliasUnmapPlan, StackAliasFailure>;

// Plans metadata only. Inputs remain unchanged. The caller retains backing,
// serializes map/record access, commits bytes and metadata, then erases the
// selected record. Partial unmaps are outside this host profile. Configured
// process limits and fallible commit handling remain the caller's responsibility.
StackAliasUnmapResult plan_stack_alias_unmap(
    std::span<const MemoryBlock> blocks,
    std::span<const StackAliasRecord> aliases,
    std::uint64_t destination, std::uint64_t source, std::uint64_t size);

}  // namespace armigo::horizon
