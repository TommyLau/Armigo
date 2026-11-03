// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader/program_identity.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <stdexcept>

namespace armigo::shader::program_identity {
namespace {

using Stage = shader_input::Stage;

bool supported(Stage stage) noexcept {
    return stage == Stage::Vertex || stage == Stage::Fragment || stage == Stage::Compute;
}

std::uint32_t word(std::span<const std::byte> bytes) noexcept {
    return std::uint32_t{std::to_integer<unsigned char>(bytes[0])} |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[1])} << 8) |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[2])} << 16) |
           (std::uint32_t{std::to_integer<unsigned char>(bytes[3])} << 24);
}

bool header_matches(Stage stage, std::span<const std::byte> bytes) noexcept {
    // Compute programs carry no SPH header; their bytes are matched by digest
    // alone, like the identity build for the same stage.
    if (stage == Stage::Compute) return true;
    return ((word(bytes) >> shader_input::format::sph_stage_shift) &
            shader_input::format::sph_stage_mask) ==
           shader_input::sph_stage_type_for(stage);
}

bool byte_less(const ProgramInput& lhs, const ProgramInput& rhs) noexcept {
    if (lhs.stage != rhs.stage) return lhs.stage < rhs.stage;
    return std::lexicographical_compare(lhs.program.begin(), lhs.program.end(),
                                        rhs.program.begin(), rhs.program.end());
}

bool same_bytes(std::span<const std::byte> lhs,
                std::span<const std::byte> rhs) noexcept {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

bool proper_prefix(std::span<const std::byte> shorter,
                   std::span<const std::byte> longer) noexcept {
    return shorter.size() < longer.size() &&
           std::equal(shorter.begin(), shorter.end(), longer.begin());
}

bool identity_less(const Identity& lhs, const Identity& rhs) noexcept {
    if (lhs.stage != rhs.stage) return lhs.stage < rhs.stage;
    if (lhs.prefix_digest != rhs.prefix_digest)
        return lhs.prefix_digest < rhs.prefix_digest;
    if (lhs.length != rhs.length) return lhs.length < rhs.length;
    return lhs.full_digest < rhs.full_digest;
}

struct Failure { Error error; };
[[noreturn]] void fail(Error::Code code, std::size_t input = 0,
                       std::uint64_t detail = 0) {
    throw Failure{{code, input, detail}};
}

}  // namespace

const char* Error::message() const noexcept {
    switch (code) {
        case Code::UnsupportedStage: return "program identity stage is unsupported";
        case Code::ProgramTooShort: return "program identity span is shorter than the discovery prefix";
        case Code::ProgramTooLarge: return "program identity span exceeds the size policy";
        case Code::HeaderStageMismatch: return "program SPH stage differs from the bound stage";
        case Code::IdentityLimit: return "program identity count exceeds the policy";
        case Code::BucketLimit: return "program identity prefix bucket exceeds the policy";
        case Code::BuildHashLimit: return "program identity build hash work exceeds the policy";
        case Code::LookupWorkLimit: return "program identity lookup work exceeds the policy";
        case Code::PrefixRelation: return "one same-stage program identity is a byte prefix of another";
        case Code::HashFailure: return "SHA-256 provider failed";
        case Code::AddressOverflow: return "program identity memory range overflows";
        case Code::ReadFailure: return "program identity memory read failed";
        case Code::Miss: return "bound program is absent from the identity index";
        case Code::Ambiguous: return "bound program matches multiple identities";
        case Code::Allocation: return "program identity allocation failed";
    }
    return "unknown program identity error";
}

BuildResult build(std::span<const ProgramInput> inputs, HashProvider hash) {
    try {
        if (inputs.size() > maximum_identities)
            fail(Error::Code::IdentityLimit, inputs.size(), maximum_identities);
        if (hash.function == nullptr) fail(Error::Code::HashFailure);
        std::uint64_t work = 0;
        std::vector<ProgramInput> ordered;
        ordered.reserve(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& input = inputs[i];
            if (!supported(input.stage)) fail(Error::Code::UnsupportedStage, i);
            if (input.program.size() < prefix_size)
                fail(Error::Code::ProgramTooShort, i, input.program.size());
            if (input.program.size() > maximum_program_size)
                fail(Error::Code::ProgramTooLarge, i, input.program.size());
            if (!header_matches(input.stage, input.program))
                fail(Error::Code::HeaderStageMismatch, i);
            const auto amount = static_cast<std::uint64_t>(prefix_size + input.program.size());
            if (amount > maximum_build_hash_bytes - work)
                fail(Error::Code::BuildHashLimit, i, work);
            work += amount;
            ordered.push_back(input);
        }
        std::sort(ordered.begin(), ordered.end(), byte_less);
        std::vector<ProgramInput> unique;
        unique.reserve(ordered.size());
        for (const auto& input : ordered) {
            if (!unique.empty() && unique.back().stage == input.stage) {
                if (same_bytes(unique.back().program, input.program)) continue;
                if (proper_prefix(unique.back().program, input.program))
                    fail(Error::Code::PrefixRelation);
            }
            unique.push_back(input);
        }
        std::vector<Identity> records;
        records.reserve(unique.size());
        for (std::size_t i = 0; i < unique.size(); ++i) {
            const auto prefix = unique[i].program.first(prefix_size);
            const auto prefix_digest = hash.function(hash.context, prefix);
            if (!prefix_digest) fail(Error::Code::HashFailure, i);
            const auto full_digest = hash.function(hash.context, unique[i].program);
            if (!full_digest) fail(Error::Code::HashFailure, i);
            records.push_back({unique[i].stage, unique[i].program.size(),
                               *prefix_digest, *full_digest});
        }
        std::sort(records.begin(), records.end(), identity_less);
        std::size_t bucket_begin = 0;
        while (bucket_begin < records.size()) {
            std::size_t bucket_end = bucket_begin + 1;
            while (bucket_end < records.size() &&
                   records[bucket_end].stage == records[bucket_begin].stage &&
                   records[bucket_end].prefix_digest == records[bucket_begin].prefix_digest)
                ++bucket_end;
            if (bucket_end - bucket_begin > maximum_bucket_size)
                fail(Error::Code::BucketLimit, bucket_begin, bucket_end - bucket_begin);
            bucket_begin = bucket_end;
        }
        return Index{std::move(records)};
    } catch (const Failure& failure) {
        return failure.error;
    } catch (const std::bad_alloc&) {
        return Error{Error::Code::Allocation};
    } catch (const std::length_error&) {
        return Error{Error::Code::Allocation};
    }
}

LookupResult lookup(const Index& index, Stage stage, std::uint64_t address,
                    MemoryReader reader, HashProvider hash) {
    try {
        if (!supported(stage)) fail(Error::Code::UnsupportedStage);
        if (reader.function == nullptr) fail(Error::Code::ReadFailure);
        if (hash.function == nullptr) fail(Error::Code::HashFailure);
        if (address > std::numeric_limits<std::uint64_t>::max() - prefix_size)
            fail(Error::Code::AddressOverflow, 0, prefix_size);
        std::array<std::byte, prefix_size> prefix{};
        if (!reader.function(reader.context, address, prefix))
            fail(Error::Code::ReadFailure, 0, prefix_size);
        if (!header_matches(stage, prefix)) fail(Error::Code::HeaderStageMismatch);
        const auto prefix_digest = hash.function(hash.context, prefix);
        if (!prefix_digest) fail(Error::Code::HashFailure);
        const Identity needle{stage, 0, *prefix_digest, {}};
        const auto begin = std::lower_bound(index.identities_.begin(), index.identities_.end(),
            needle, [](const Identity& lhs, const Identity& rhs) {
                if (lhs.stage != rhs.stage) return lhs.stage < rhs.stage;
                return lhs.prefix_digest < rhs.prefix_digest;
            });
        auto end = begin;
        while (end != index.identities_.end() && end->stage == stage &&
               end->prefix_digest == *prefix_digest) ++end;
        if (begin == end) fail(Error::Code::Miss);
        if (static_cast<std::size_t>(end - begin) > maximum_bucket_size)
            fail(Error::Code::BucketLimit);
        std::size_t work = prefix_size;
        std::size_t previous_length = 0;
        for (auto it = begin; it != end; ++it) {
            if (it != begin && it->length == previous_length) continue;
            if (it->length > maximum_lookup_work_bytes - work)
                fail(Error::Code::LookupWorkLimit, 0, work);
            work += it->length;
            previous_length = it->length;
        }
        std::optional<Identity> match;
        for (auto group = begin; group != end;) {
            const std::size_t length = group->length;
            auto group_end = group;
            while (group_end != end && group_end->length == length) ++group_end;
            Digest digest{};
            if (length == prefix_size) {
                digest = *prefix_digest;
            } else {
                if (address > std::numeric_limits<std::uint64_t>::max() - length)
                    fail(Error::Code::AddressOverflow, 0, length);
                std::vector<std::byte> bytes(length);
                if (!reader.function(reader.context, address, bytes))
                    fail(Error::Code::ReadFailure, 0, length);
                const auto hashed = hash.function(hash.context, bytes);
                if (!hashed) fail(Error::Code::HashFailure, 0, length);
                digest = *hashed;
            }
            for (auto candidate = group; candidate != group_end; ++candidate) {
                if (candidate->full_digest != digest) continue;
                if (match) fail(Error::Code::Ambiguous);
                match = *candidate;
            }
            group = group_end;
        }
        if (!match) fail(Error::Code::Miss);
        return *match;
    } catch (const Failure& failure) {
        return failure.error;
    } catch (const std::bad_alloc&) {
        return Error{Error::Code::Allocation};
    } catch (const std::length_error&) {
        return Error{Error::Code::Allocation};
    }
}

}  // namespace armigo::shader::program_identity
