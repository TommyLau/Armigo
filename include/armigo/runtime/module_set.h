// ============================================================================
// Armigo Runtime - Native Module Runtime
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/linker/link_plan.h>
#include <armigo/loader/nso_module.h>
#include <armigo/platform/host_module.h>

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace armigo::runtime {

struct NsoSource {
    std::string_view name;
    std::span<const std::byte> file;
    // The image already decoded from `file` with the default reader options, if
    // the caller has one; loading takes it instead of decoding the file again.
    loader::NsoModuleImage* decoded = nullptr;
};

struct ModulePreparationView {
    std::size_t module_index;
    std::string_view module_name;
    std::uintptr_t base_address;
    // Original logical image. Host-link mode has applied relocations; guest
    // bootstrap mode preserves the source relocation targets unchanged.
    std::span<std::byte> image;
    std::span<std::byte> appended_code;
    std::size_t appended_code_offset;
    const linker::DynamicModule& metadata;
    // Load-order bases of the complete set, available only during this callback.
    // This permits cross-module fixups without retaining borrowed module views.
    std::span<const std::uintptr_t> module_bases{};
    // Optional page-aligned RW storage between the original image and code.
    std::span<std::byte> appended_data{};
    std::size_t appended_data_offset = 0;
};

// Trusted host preparation callback, called after the whole set is mapped and
// the selected relocation mode is handled, but before any module is sealed.
// Return false to reject the load. The callback must not execute code, retain
// borrowed spans/references, or publish addresses for execution before load_nso
// succeeds. It must verify instruction preimages, overlap and execution
// contracts for its own preparation plan. It must not let another thread access
// the module views while preparation is in progress.
using ModulePrepareCallback = bool (*)(const ModulePreparationView&, void*) noexcept;

struct ModulePreparation {
    // An optional zero-filled area after the original image, aligned to a host
    // page and sealed read/execute separately from the original writable data.
    std::size_t appended_code_bytes = 0;
    ModulePrepareCallback callback = nullptr;
    void* context = nullptr;
    // Sparse offline preparation opts in and supplies metadata-root entries.
    // Interior covered words and appendix code are not independent entries.
    bool restrict_entry_points = false;
    std::span<const std::uint64_t> admitted_entry_offsets{};
    // False when the host never runs this module's code: no host page of it is
    // executable, while the guest memory map keeps each segment's own
    // permissions. A native page larger than the guest page may then hold its
    // code and its writable data together.
    bool host_executes = true;
    // Must be a native-page multiple. This storage stays RW after sealing;
    // it is outside the original image and all dynamic relocation ranges.
    std::size_t appended_data_bytes = 0;
    // Optional prepared backing, retained by the caller through load_nso.
    // The callback must bind data/host metadata only; code is already RX.
    std::span<const platform::EmbeddedModulePiece> embedded_pieces{};
};

enum class ModuleRelocationMode {
    host_link,
    guest_bootstrap,
};

struct ModuleLoadOptions {
    loader::NsoReadOptions reader;
    linker::DynamicLimits dynamic_limits;
    linker::LinkLimits link_limits;
    // A module's own definitions take precedence during its relocation pass.
    linker::LinkOptions bindings{
        .preemption = linker::PreemptionModel::own_definition_first};
    std::size_t max_total_image_bytes = 2ull * 1024 * 1024 * 1024;
    std::size_t max_module_name_bytes = 256;
    // Empty keeps ordinary loading behavior. Otherwise exactly one entry per
    // source; a null callback is allowed only with no appended code requested.
    // Callbacks and contexts are borrowed only for this synchronous load.
    std::span<const ModulePreparation> preparation;
    // Runs the preparation callbacks of different modules at the same time. They
    // must then be safe to run concurrently, must not throw, and every callback
    // runs even after another has rejected the load.
    bool parallel_preparation = false;
    // Guest bootstrap mode maps and prepares the original images without
    // applying host relocations. The guest loader owns relocation afterward.
    ModuleRelocationMode relocation_mode = ModuleRelocationMode::host_link;
};

enum class ModuleLoadError {
    none,
    invalid_sources,
    resource_limit,
    invalid_container,
    invalid_dynamic_metadata,
    mapping_failed,
    linking_failed,
    application_failed,
    sealing_failed,
    preparation_failed,
};

struct ModuleSetResult;

// Owns a complete sealed set of module images. Host-link mode links before
// sealing; guest bootstrap mode deliberately preserves relocation targets.
// Loading never calls initializers or guest entry points. Arbitrary guest
// execution still requires instruction preparation and runtime services.
class ModuleSet {
public:
    static ModuleSetResult load_nso(std::span<const NsoSource> sources,
                                   const ModuleLoadOptions& options = {});
    ModuleSet(const ModuleSet&) = delete;
    ModuleSet& operator=(const ModuleSet&) = delete;
    ModuleSet(ModuleSet&&) noexcept = default;
    ModuleSet& operator=(ModuleSet&&) noexcept = default;

    std::size_t module_count() const noexcept { return modules_.size(); }
    std::size_t relocation_count() const noexcept;
    std::string_view module_name(std::size_t index) const;
    const platform::HostModule& mapping(std::size_t index) const;
    const linker::DynamicModule& metadata(std::size_t index) const;
    // Appended preparation code is not part of the original module's symbol
    // or relocation address range. Empty when no area was requested.
    std::optional<linker::ModuleRange> appended_code(std::size_t index) const;
    // Trusted legacy loaders admit executable ownership. Sparse preparation
    // additionally restricts guest thread entry to its retained metadata roots.
    bool permits_guest_entry(std::uint64_t address) const noexcept;

    // Finds a defined function in the selected module, not a preempted import.
    // The result proves mapped executable placement, not host ABI compatibility.
    std::optional<std::uintptr_t> entry_address(
        std::size_t index, std::string_view symbol) const noexcept;

private:
    ModuleSet() = default;
    struct Module {
        std::string name;
        linker::DynamicModule metadata;
        platform::HostModule mapping;
        std::vector<linker::ModuleRange> relocatable_ranges;
        std::size_t original_image_size = 0;
        std::size_t original_mapping_size = 0;
        std::size_t appended_code_bytes = 0;
        std::size_t appended_data_bytes = 0;
        bool restrict_entry_points = false;
        std::vector<std::uint64_t> admitted_entry_offsets;
    };
    std::vector<Module> modules_;
};

struct ModuleSetResult {
    ModuleLoadError error = ModuleLoadError::none;
    // Empty for success or a failure that cannot be attributed to one module.
    std::optional<std::size_t> module_index;
    // May be empty when reporting allocation failure.
    std::string message;
    std::size_t unresolved_count = 0;
    std::vector<linker::UnresolvedSymbol> unresolved;
    std::optional<ModuleSet> modules;
    bool ok() const noexcept { return modules.has_value(); }
};

} // namespace armigo::runtime
