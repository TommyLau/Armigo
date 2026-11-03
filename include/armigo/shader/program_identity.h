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

#include <armigo/shader_input/shader_input.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace armigo::shader::program_identity {

using Digest = std::array<std::byte, 32>;

inline constexpr std::size_t prefix_size = 112;
inline constexpr std::size_t maximum_program_size = 4u * 1024u * 1024u;
inline constexpr std::size_t maximum_identities = 1000000;
inline constexpr std::size_t maximum_bucket_size = 64;
inline constexpr std::uint64_t maximum_build_hash_bytes = std::uint64_t{64} << 30;
inline constexpr std::size_t maximum_lookup_work_bytes = 32u * 1024u * 1024u;

struct ProgramInput {
    shader_input::Stage stage;
    std::span<const std::byte> program;
};

struct HashProvider {
    std::optional<Digest> (*function)(void*, std::span<const std::byte>) noexcept = nullptr;
    void* context = nullptr;
};

struct MemoryReader {
    bool (*function)(void*, std::uint64_t, std::span<std::byte>) noexcept = nullptr;
    void* context = nullptr;
};

struct Identity {
    shader_input::Stage stage;
    std::size_t length;
    Digest prefix_digest;
    Digest full_digest;
};

struct Error {
    enum class Code : std::uint8_t {
        UnsupportedStage,
        ProgramTooShort,
        ProgramTooLarge,
        HeaderStageMismatch,
        IdentityLimit,
        BucketLimit,
        BuildHashLimit,
        LookupWorkLimit,
        PrefixRelation,
        HashFailure,
        AddressOverflow,
        ReadFailure,
        Miss,
        Ambiguous,
        Allocation,
    };
    Code code;
    std::size_t input = 0;
    std::uint64_t detail = 0;
    [[nodiscard]] const char* message() const noexcept;
};

class Index {
public:
    Index() = delete;
    Index(const Index&) = default;
    Index(Index&&) noexcept = default;
    Index& operator=(const Index&) = default;
    Index& operator=(Index&&) noexcept = default;

    [[nodiscard]] std::span<const Identity> identities() const noexcept {
        return identities_;
    }

private:
    explicit Index(std::vector<Identity> identities) noexcept
        : identities_(std::move(identities)) {}
    std::vector<Identity> identities_;

    friend std::variant<Index, Error> build(std::span<const ProgramInput>, HashProvider);
    friend std::variant<Identity, Error> lookup(const Index&, shader_input::Stage,
                                                std::uint64_t, MemoryReader,
                                                HashProvider);
};

using BuildResult = std::variant<Index, Error>;
using LookupResult = std::variant<Identity, Error>;

// Input bytes remain immutable until build returns. Lookup requires memory
// stability across its prefix and candidate reads; callers provide locking
// or a stable reader snapshot when guest memory can change concurrently.
[[nodiscard]] BuildResult build(std::span<const ProgramInput> inputs,
                                HashProvider hash);
[[nodiscard]] LookupResult lookup(const Index& index, shader_input::Stage stage,
                                  std::uint64_t address, MemoryReader reader,
                                  HashProvider hash);

}  // namespace armigo::shader::program_identity
