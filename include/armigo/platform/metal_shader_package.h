// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/platform/metal_draw.h>

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <variant>

namespace armigo::platform {

struct MetalShaderPackageOptions {
    // Supplied by trusted application/profile configuration, never by the package.
    std::string expected_chain_id;
    bool allow_development_subset = false;
    // Development runs: an entry whose reflection is valid for the package but
    // outside the runtime profile does not reject the package; looking it up
    // returns a structured Unsupported result naming the reason instead.
    bool defer_unsupported_entries = false;
    // Development runs: opening checks only the manifest, and indexes the
    // program identities it records. Each stage's directory, program,
    // reflection and library are checked when a lookup first selects it, so a
    // defect in a stage the title never draws with cannot delay or reject
    // startup; a defect in a stage that is selected is reported by that lookup,
    // naming the stage (a program that differs from the manifest included).
    bool lazy_entries = false;
    // With lazy_entries: background threads that prepare every entry after
    // opening, so the work is done by the time draws select stages.
    unsigned prefetch_threads = 0;
};

// What the background preparation did: entries prepared, entries that failed
// (each is reported again by the lookup that selects it) and the first failure.
struct MetalShaderPackagePrefetch {
    std::size_t prepared = 0;
    std::size_t failed = 0;
    std::string first_failure;
};

struct MetalShaderPackageError {
    enum class Code {
        InvalidOptions,
        Input,
        Manifest,
        Compatibility,
        Integrity,
        Reflection,
        Identity,
        Allocation,
    };
    Code code = Code::Manifest;
    std::string detail;
};

// A stage translated at run time, built from the files a package stage holds
// besides its Metal library: the program (its identity bytes), the reflection
// and the Metal source. The reflection is checked and parsed as a package's is
// (its source version, program digest and size and stage must match); the
// stage carries the source instead of a library. Given in memory, the source
// must match the reflection's source digest now; given as a file
// (source_file, with an empty source), it stays on disk and is checked when
// stage_source reads it.
using SourceStageResult = std::variant<std::shared_ptr<const CompiledGraphicsStage>, MetalShaderPackageError>;
[[nodiscard]] SourceStageResult make_source_stage(shader_input::Stage stage,
                                                  std::span<const std::byte> program,
                                                  std::span<const std::byte> reflection,
                                                  std::string source,
                                                  std::string source_file = {});
// The Metal source of a stage built by make_source_stage: held in memory, or
// read from its file and checked against the reflection's source digest.
[[nodiscard]] std::variant<std::string, MetalShaderPackageError> stage_source(const CompiledGraphicsStage& stage);

class MetalShaderPackage {
public:
    using Options = MetalShaderPackageOptions;
    using OpenResult = std::variant<std::shared_ptr<MetalShaderPackage>, MetalShaderPackageError>;

    [[nodiscard]] static OpenResult open(const std::filesystem::path& directory,
                                         Options options);
    ~MetalShaderPackage();

    MetalShaderPackage(const MetalShaderPackage&) = delete;
    MetalShaderPackage& operator=(const MetalShaderPackage&) = delete;

    [[nodiscard]] CompiledGraphicsLookupResult lookup(
        const graphics::RenderStop& snapshot,
        const graphics::GpuAddressSpace& memory) noexcept;
    [[nodiscard]] static CompiledGraphicsLookupResult callback(
        void* context, const graphics::RenderStop& snapshot,
        const graphics::GpuAddressSpace& memory) noexcept;
    // Compute dispatch lookup over the same package: resolves the program by
    // identity, then selects the variant whose local size and shared memory
    // equal the decoded queue meta data. A miss is a structured CommandResult
    // naming the program and both sides' sizes.
    [[nodiscard]] CompiledComputeStageResult lookup_compute(
        const graphics::RenderStop& snapshot,
        const graphics::GpuAddressSpace& memory) noexcept;
    [[nodiscard]] static CompiledComputeStageResult compute_callback(
        void* context, const graphics::RenderStop& snapshot,
        const graphics::GpuAddressSpace& memory) noexcept;
    // Identifies a stage once from immutable SPH/program bytes. Render-target
    // output specialization remains a pipeline decision; no package search is
    // needed when the resulting artifact is bound by the NVN driver.
    [[nodiscard]] CompiledComputeStageResult lookup_stage(
        shader_input::Stage stage, std::span<const std::byte> program,
        shader::graphics_msl::VertexDepthConvention depth) noexcept;
    // Identifies a stage from the program at a host-addressable guest address,
    // reading as many bytes as each candidate's recorded length needs (the
    // command-stream path reads programs the same way).
    [[nodiscard]] CompiledComputeStageResult lookup_stage_at(
        shader_input::Stage stage, std::uint64_t program_address,
        shader::graphics_msl::VertexDepthConvention depth) noexcept;
    // The shared stage a manifest names by CompiledGraphicsStage::identity, loaded
    // like a lookup would load it; null when the package has no such stage.
    // Safe to call from any thread, concurrently with lookups.
    [[nodiscard]] std::shared_ptr<const CompiledGraphicsStage> stage_by_identity(
        std::string_view identity) noexcept;
    // Tells the background preparation which stages the title selects first, by
    // identity in the order it first selects them, so they are prepared before
    // the rest instead of when the guest asks for them. Identities the package
    // lacks are ignored. Only the first call has an effect; without prefetch
    // threads it does nothing. Safe to call from any thread.
    void prefetch_first(std::span<const std::string> identities) noexcept;
    // Waits for the background preparation to end and reports its outcome;
    // without prefetch threads it returns at once with zero counts.
    [[nodiscard]] MetalShaderPackagePrefetch finish_prefetch() noexcept;
    // The package's compiler-chain ID.
    [[nodiscard]] const std::string& chain_id() const noexcept;
    // The SHA-256 of the package manifest (hexadecimal): which stages, with
    // which libraries, the package holds. Two packages from one chain differ here.
    [[nodiscard]] const std::string& content_id() const noexcept;
    // The package and callback context outlive every channel that registers it.
    // Returned diagnostic views remain valid for the package lifetime.

private:
    struct Impl;
    explicit MetalShaderPackage(std::unique_ptr<Impl>) noexcept;
    std::unique_ptr<Impl> impl_;
};

}  // namespace armigo::platform
