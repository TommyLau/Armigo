// ============================================================================
// Armigo Shader - Native Shader Translation
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
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::shader_native {

// A native shader section stored beside an empty armigo-metal-package-v1
// shell. The Maxwell loader does not read this section. Stage identity is the
// digest a file record and a run-time control share for the same program.
inline constexpr std::string_view native_section_schema = "armigo-native-shader-section-v1";

enum class NativeStageKind { vertex, fragment, compute, task, mesh };
enum class NativeDepth { stage, minus_one_to_one, zero_to_one };

struct NativeCounts {
    std::uint64_t enumerated = 0;
    std::uint64_t unique = 0;
    std::uint64_t translated = 0;
    std::uint64_t compiled = 0;
    std::uint64_t duplicates = 0;
    std::map<std::string, std::uint64_t> by_stage;
    std::map<std::string, std::uint64_t> translated_by_stage;
    std::map<std::string, std::uint64_t> compiled_by_stage;
    std::map<std::string, std::uint64_t> refusals;
};

struct NativeVariant {
    NativeDepth depth = NativeDepth::stage;
    std::string reflection_digest;
    std::string source_digest;
    std::string library_digest;
    std::uint64_t reflection_size = 0;
    std::uint64_t source_size = 0;
    std::uint64_t library_size = 0;
};

struct NativeEntry {
    std::string stage_identity;
    NativeStageKind stage = NativeStageKind::fragment;
    std::string program_digest;
    std::uint64_t program_size = 0;
    std::vector<NativeVariant> variants;
};

// Draw-state facts the section was built with. A lookup whose facts differ is
// not a hit: the library would not match the stage the resolver would translate.
struct NativeSpecialization {
    bool pixel_rate_disabled = true;
    bool vertex_fetch = false;
    bool fragment_position_matches = true;
    bool vertex_depth_minus_one_to_one = true;
    bool vertex_depth_zero_to_one = true;
};

struct NativeSection {
    NativeSpecialization specialization;
    NativeCounts counts;
    std::vector<NativeEntry> entries;
};

// The largest Metal source a native variant may store as source.metal; add()
// refuses more when it writes sources.
inline constexpr std::size_t maximum_native_source_bytes = 4 * 1024 * 1024;
// A variant that stores no source.metal records only the source's digest and
// size; that size is bounded like a library.
inline constexpr std::size_t maximum_unstored_source_bytes = 64 * 1024 * 1024;

struct NativeVariantWrite {
    NativeDepth depth = NativeDepth::stage;
    std::vector<std::byte> reflection;
    std::string source;
    std::vector<std::byte> library;
};

struct NativeStageWrite {
    std::string stage_identity;
    NativeStageKind stage = NativeStageKind::fragment;
    std::vector<std::byte> program;
    std::vector<NativeVariantWrite> variants;
};

struct NativeLookup {
    NativeStageKind stage = NativeStageKind::fragment;
    bool pixel_rate_disabled = true;
    bool vertex_fetch = false;
    bool vertex_depth_zero_to_one = false;
    bool fragment_position_matches = true;
};

// Run-time stage identity: `<stage number>.<program sha256>.<suffix>`.
// 0 vertex, 1 fragment, 3 task, 4 mesh, 5 compute. A vertex or mesh suffix is
// 1 (zero-to-one) or 2 (minus-one-to-one). Any other stage number, or any
// other vertex or mesh suffix, is not this text.
struct RuntimeStageIdentity {
    NativeStageKind stage = NativeStageKind::fragment;
    std::string program_digest;
    bool vertex_depth_zero_to_one = false;
};

std::optional<RuntimeStageIdentity> parse_runtime_stage_identity(std::string_view text);

struct NativeStageFiles {
    std::vector<std::byte> program;
    std::vector<std::byte> reflection;
    std::string source;
    std::vector<std::byte> library;
};

struct NativeMiss {
    enum class Reason { absent, specialization, corrupt };
    Reason reason = Reason::absent;
    std::string detail;
};

// Writes native/ under a package directory. add is not safe to call from two
// threads at once. finish writes native/manifest.json once.
class NativeSectionWriter {
public:
    // write_source false omits source.metal. The manifest still records that
    // source's digest and size.
    static std::variant<std::string, NativeSectionWriter> create(const std::filesystem::path& package_root,
                                                                  bool write_source = true);
    std::variant<std::string, std::monostate> add(const NativeStageWrite& stage);
    std::variant<std::string, NativeSection> finish(NativeCounts counts, NativeSpecialization specialization);

    NativeSectionWriter(NativeSectionWriter&&) noexcept;
    NativeSectionWriter& operator=(NativeSectionWriter&&) noexcept;
    ~NativeSectionWriter();

private:
    struct State;
    explicit NativeSectionWriter(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

// Opens the section index only. find reads one stage and checks its digests.
class NativePackage {
public:
    static std::variant<std::string, NativePackage> open(const std::filesystem::path& package_root);
    std::variant<NativeStageFiles, NativeMiss> find(std::string_view stage_identity, const NativeLookup& lookup) const;
    // The packaged stage identity for a program image of this stage kind.
    // The index is built on the first call from the opened section.
    std::optional<std::string> identity_for_program(std::string_view program_digest_hex, NativeStageKind stage) const;
    const NativeSection& section() const;

    NativePackage(NativePackage&&) noexcept;
    NativePackage& operator=(NativePackage&&) noexcept;
    ~NativePackage();

private:
    struct State;
    explicit NativePackage(std::unique_ptr<State> state);
    std::unique_ptr<State> state_;
};

} // namespace armigo::shader_native
