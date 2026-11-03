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

#include <armigo/linker/dynamic_module.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::linker {

// A module-relative byte range, [offset, offset + size).
struct ModuleRange {
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

// One module in load order. The image is the materialized, zero-based module
// content used to read relocation preimages; it is never modified by planning.
// The module occupies [base, base + span_size) in the guest address space, which
// may extend beyond the image to cover zero-initialized storage.
struct LinkModule {
    std::string_view name;
    std::span<const std::byte> image;
    const DynamicModule* metadata = nullptr;
    std::uint64_t base = 0;
    std::uint64_t span_size = 0;
    // Module-relative ranges that relocations are permitted to write. Every
    // eight-byte target must lie within their union; no ranges permits no
    // writes. Ranges must lie within the image.
    std::span<const ModuleRange> writable;
};

// An explicit host-supplied symbol address.
struct HostBinding {
    std::string_view name;
    std::uint64_t address = 0;
};

// How a defined, default-visibility symbol reference binds when its own module
// also defines it. Horizon's exact rule is not yet independently verified.
enum class PreemptionModel : std::uint8_t {
    // ELF default: search module definitions in load order; the first wins.
    load_order,
    // Bind to the referring module's own definition before searching others.
    own_definition_first,
};

struct LinkOptions {
    // Consulted first for preemptible references; overrides module definitions.
    std::span<const HostBinding> forced;
    // Consulted only when no module exports the referenced name.
    std::span<const HostBinding> fallback;
    PreemptionModel preemption = PreemptionModel::load_order;
    // Reject a plan when a forced binding is never used by any relocation.
    bool require_forced_bindings_used = true;
};

struct LinkLimits {
    std::size_t max_modules = 64;
    std::size_t max_writable_ranges = 4096;
    std::size_t max_host_bindings = 65536;
    std::size_t max_host_name_bytes = 4096;
    std::size_t max_total_symbols = 8'000'000;
    std::size_t max_writes = 4'000'000;
    std::size_t max_reported_unresolved = 256;
};

enum class LinkError {
    none,
    invalid_argument,
    invalid_layout,
    aliased_images,
    invalid_metadata,
    invalid_symbol,
    unsupported_symbol,
    unsupported_relocation,
    forbidden_target,
    conflicting_write,
    address_overflow,
    duplicate_host_binding,
    unresolved_symbol,
    unused_forced_binding,
    resource_limit,
};

// Where a planned value came from.
enum class WriteSource : std::uint8_t {
    relative,        // RELATIVE: module base plus addend.
    packed_relative, // RELR: module base plus the preimage word.
    module_symbol,   // A definition in a linked module (provider = module index).
    absolute_symbol, // An absolute definition (provider = module index).
    forced_host,     // A forced host binding (provider = forced binding index).
    host_fallback,   // An ordinary host binding (provider = fallback index).
    weak_unresolved, // An unresolved weak reference; the symbol value is zero.
    null_symbol,     // A symbolic relocation using symbol index zero.
};

struct PlannedWrite {
    std::uint64_t offset = 0;   // Module-relative, eight-byte aligned target.
    std::uint64_t expected = 0; // Little-endian preimage required before writing.
    std::uint64_t value = 0;    // Little-endian value written by apply.
    RelocationKind kind = RelocationKind::relative;
    WriteSource source = WriteSource::relative;
    std::uint32_t symbol = 0;   // Referenced symbol index in this module.
    std::uint32_t provider = 0; // Module index for relative writes; otherwise source-specific.
    std::uint32_t provider_symbol = 0; // Defining symbol index for module sources.
};

struct UnresolvedSymbol {
    std::size_t module = 0;
    std::uint32_t symbol = 0;
    std::string name;
};

enum class ApplyError {
    none,
    module_count_mismatch,
    image_size_mismatch,
    aliased_images,
    aliased_descriptors,
    image_mismatch,
    preimage_mismatch,
};

struct ApplyResult {
    ApplyError error = ApplyError::none;
    std::size_t module = 0;
    std::uint64_t offset = 0;
    const char* message = "";
    bool ok() const noexcept { return error == ApplyError::none; }
};

namespace detail {
class LinkPlanner;
}

// An immutable, validated set of relocation writes for an ordered module list.
// The plan retains no reference to its inputs.
class LinkPlan {
public:
    LinkPlan(const LinkPlan&) = delete;
    LinkPlan& operator=(const LinkPlan&) = delete;
    LinkPlan(LinkPlan&& other) noexcept;
    LinkPlan& operator=(LinkPlan&& other) noexcept;

    std::size_t module_count() const noexcept { return modules_.size(); }
    std::uint64_t base(std::size_t module) const { return modules_.at(module).base; }
    std::uint64_t span_size(std::size_t module) const { return modules_.at(module).span_size; }
    std::uint64_t image_size(std::size_t module) const { return modules_.at(module).image_size; }
    std::span<const PlannedWrite> writes(std::size_t module) const;
    // For each LinkOptions::forced entry, in input order, the number of
    // referring symbol entries bound to it.
    std::span<const std::uint64_t> forced_binding_uses() const noexcept { return forced_uses_; }

    // Applies every planned write to images supplied in plan order. All
    // preconditions and preimages are validated before the first write. On
    // failure no byte is modified. Descriptor storage must be separate from
    // destination images. This function never allocates.
    ApplyResult apply(std::span<const std::span<std::byte>> images) const noexcept;

private:
    friend class detail::LinkPlanner;

    struct ModuleState {
        std::uint64_t base = 0;
        std::uint64_t span_size = 0;
        std::uint64_t image_size = 0;
        std::uint64_t fingerprint = 0;
        std::size_t first_write = 0;
        std::size_t write_count = 0;
    };

    LinkPlan() = default;

    std::vector<ModuleState> modules_;
    std::vector<PlannedWrite> writes_;
    std::vector<std::uint64_t> forced_uses_;
};

struct PlanResult {
    LinkError error = LinkError::none;
    std::string message;
    // Unresolved strong references, bounded by LinkLimits::max_reported_unresolved.
    std::vector<UnresolvedSymbol> unresolved;
    std::size_t unresolved_count = 0;
    // Forced binding names never used, when that is rejected.
    std::vector<std::string> unused_forced;
    std::optional<LinkPlan> plan;
    bool ok() const noexcept { return plan.has_value(); }
};

// Validates modules and bindings and computes every relocation value without
// modifying any input or executing guest code.
PlanResult plan_link(std::span<const LinkModule> modules, const LinkOptions& options = {},
                     const LinkLimits& limits = {});

} // namespace armigo::linker
