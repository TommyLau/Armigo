// ============================================================================
// Armigo Linker - Guest Dynamic Linking
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace armigo::linker {

enum class DynamicError {
    none,
    invalid_header,
    invalid_table,
    invalid_symbol,
    unsupported_relocation,
    resource_limit,
};

struct DynamicLimits {
    std::size_t max_entries = 4096;
    std::size_t max_symbols = 1'000'000;
    std::size_t max_relocations = 1'000'000;
    std::size_t max_relocation_records = 1'000'000;
    std::size_t max_name_bytes = 4096;
    std::size_t max_total_name_bytes = 64 * 1024 * 1024;
};

struct Symbol {
    std::string name;
    std::uint64_t value = 0;
    std::uint64_t size = 0;
    std::uint16_t section = 0;
    std::uint8_t info = 0;
    std::uint8_t visibility = 0;

    bool defined() const noexcept { return section != 0; }
};

enum class RelocationKind : std::uint32_t {
    absolute64 = 257,
    global_data = 1025,
    jump_slot = 1026,
    relative = 1027,
};

struct Relocation {
    std::uint64_t target = 0;
    RelocationKind kind = RelocationKind::relative;
    std::uint32_t symbol = 0;
    std::int64_t addend = 0;
};

struct DynamicModule {
    std::vector<Symbol> symbols;
    std::vector<std::string> dependencies;
    std::vector<Relocation> relocations;
    // RELR addends remain in the image at these validated target offsets.
    std::vector<std::uint64_t> packed_relative_targets;
    std::uint64_t init = 0;
    std::uint64_t fini = 0;
    std::uint64_t init_array = 0;
    std::uint64_t init_array_size = 0;
    std::uint64_t fini_array = 0;
    std::uint64_t fini_array_size = 0;
};

struct DynamicResult {
    DynamicError error = DynamicError::none;
    std::string message;
    std::optional<DynamicModule> module;
    bool ok() const noexcept { return module.has_value(); }
};

// Reads a zero-based, materialized little-endian ARM64 module image.
// This never modifies the image, performs relocation, or executes guest code.
DynamicResult read_dynamic_module(std::span<const std::byte> image,
                                  const DynamicLimits& limits = {});

} // namespace armigo::linker
