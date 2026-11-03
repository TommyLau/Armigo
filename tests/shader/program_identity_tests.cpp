// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "test_support.h"

#include "sha256.h"
#include <armigo/shader/program_identity.h>

#include <algorithm>
#include <limits>

namespace armigo::shader::test {
namespace {

namespace pi = program_identity;
using Stage = shader_input::Stage;

struct HashState {
    std::size_t calls = 0;
    std::size_t bytes = 0;
    std::size_t fail_call = 0;
    bool collide = false;
};

std::optional<pi::Digest> hash(void* context,
                               std::span<const std::byte> bytes) noexcept {
    auto& state = *static_cast<HashState*>(context);
    ++state.calls;
    state.bytes += bytes.size();
    if (state.fail_call == state.calls) return std::nullopt;
    if (state.collide) return pi::Digest{};
    return loader::detail::sha256(bytes);
}

struct ReadState {
    std::uint64_t base = 0x10000;
    std::span<const std::byte> bytes;
    std::size_t calls = 0;
    std::size_t read_bytes = 0;
    std::size_t fail_call = 0;
};

bool read(void* context, std::uint64_t address,
          std::span<std::byte> destination) noexcept {
    auto& state = *static_cast<ReadState*>(context);
    ++state.calls;
    state.read_bytes += destination.size();
    if (state.fail_call == state.calls || address != state.base ||
        destination.size() > state.bytes.size()) return false;
    std::copy_n(state.bytes.begin(), destination.size(), destination.begin());
    return true;
}

std::vector<std::byte> program(Stage stage, std::size_t size,
                               unsigned seed = 1) {
    std::vector<std::byte> result(size);
    for (std::size_t i = 4; i < size; ++i)
        result[i] = std::byte((i * 37 + seed * 53) & 255);
    const std::uint32_t sph = shader_input::sph_stage_type_for(stage) <<
                              shader_input::format::sph_stage_shift;
    for (unsigned i = 0; i < 4 && i < size; ++i)
        result[i] = std::byte(sph >> (i * 8));
    return result;
}

pi::HashProvider provider(HashState& state) { return {hash, &state}; }
pi::MemoryReader reader(ReadState& state) { return {read, &state}; }

template <typename T>
const pi::Error* rejected(const std::variant<T, pi::Error>& result,
                          pi::Error::Code code) {
    const auto* error = std::get_if<pi::Error>(&result);
    ARMIGO_CHECK(error != nullptr);
    if (error) ARMIGO_CHECK(error->code == code);
    return error;
}

const pi::Index* accepted(const pi::BuildResult& result) {
    const auto* index = std::get_if<pi::Index>(&result);
    ARMIGO_CHECK(index != nullptr);
    return index;
}

}  // namespace

ARMIGO_TEST(program_identity, deduplicates_and_resolves_full_span) {
    auto a = program(Stage::Vertex, 160, 3);
    auto b = a;
    b[143] ^= std::byte{0x5a};
    auto fragment = program(Stage::Fragment, 176, 7);
    const std::array inputs{
        pi::ProgramInput{Stage::Vertex, a}, pi::ProgramInput{Stage::Vertex, b},
        pi::ProgramInput{Stage::Vertex, a},
        pi::ProgramInput{Stage::Fragment, fragment}};
    HashState build_hash;
    const auto built = pi::build(inputs, provider(build_hash));
    const pi::Index* index = accepted(built);
    if (!index) return;
    ARMIGO_CHECK_EQ(index->identities().size(), 3u);
    ARMIGO_CHECK_EQ(build_hash.calls, 6u);

    ReadState memory{0x24000, b};
    HashState lookup_hash;
    const auto found = pi::lookup(*index, Stage::Vertex, memory.base,
                                  reader(memory), provider(lookup_hash));
    const auto* identity = std::get_if<pi::Identity>(&found);
    ARMIGO_CHECK(identity != nullptr);
    if (identity) {
        ARMIGO_CHECK(identity->stage == Stage::Vertex);
        ARMIGO_CHECK_EQ(identity->length, b.size());
        ARMIGO_CHECK(identity->full_digest == *loader::detail::sha256(b));
    }
    ARMIGO_CHECK_EQ(memory.calls, 2u);
    ARMIGO_CHECK_EQ(memory.read_bytes, pi::prefix_size + b.size());
    ARMIGO_CHECK_EQ(lookup_hash.calls, 2u);

    b[159] ^= std::byte{1};
    ReadState changed{0x24000, b};
    HashState changed_hash;
    rejected(pi::lookup(*index, Stage::Vertex, changed.base, reader(changed),
                        provider(changed_hash)), pi::Error::Code::Miss);
    ReadState wrong_stage{0x24000, a};
    HashState wrong_hash;
    rejected(pi::lookup(*index, Stage::Fragment, wrong_stage.base,
                        reader(wrong_stage), provider(wrong_hash)),
             pi::Error::Code::HeaderStageMismatch);
}

ARMIGO_TEST(program_identity, rejects_invalid_inputs_prefixes_and_build_limits) {
    HashState hashes;
    auto short_program = program(Stage::Vertex, pi::prefix_size - 1);
    rejected(pi::build(std::array{pi::ProgramInput{Stage::Vertex, short_program}},
                       provider(hashes)), pi::Error::Code::ProgramTooShort);
    auto too_large = program(Stage::Vertex, pi::maximum_program_size + 1);
    rejected(pi::build(std::array{pi::ProgramInput{Stage::Vertex, too_large}},
                       provider(hashes)), pi::Error::Code::ProgramTooLarge);
    auto mismatch = program(Stage::Fragment, 128);
    rejected(pi::build(std::array{pi::ProgramInput{Stage::Vertex, mismatch}},
                       provider(hashes)), pi::Error::Code::HeaderStageMismatch);
    auto unsupported = program(Stage::Geometry, 128);
    rejected(pi::build(std::array{pi::ProgramInput{Stage::Geometry, unsupported}},
                       provider(hashes)), pi::Error::Code::UnsupportedStage);

    auto prefix = program(Stage::Vertex, 128, 9);
    auto extension = prefix;
    extension.resize(144, std::byte{0x77});
    rejected(pi::build(std::array{pi::ProgramInput{Stage::Vertex, prefix},
                                  pi::ProgramInput{Stage::Vertex, extension}},
                       provider(hashes)), pi::Error::Code::PrefixRelation);

    std::vector<pi::ProgramInput> too_many(pi::maximum_identities + 1,
                                           {Stage::Vertex, {}});
    rejected(pi::build(too_many, provider(hashes)), pi::Error::Code::IdentityLimit);

    auto maximum = program(Stage::Vertex, pi::maximum_program_size, 11);
    std::vector<pi::ProgramInput> excessive_work(16384,
                                                 {Stage::Vertex, maximum});
    const auto calls_before = hashes.calls;
    rejected(pi::build(excessive_work, provider(hashes)),
             pi::Error::Code::BuildHashLimit);
    ARMIGO_CHECK_EQ(hashes.calls, calls_before);

    auto common = program(Stage::Vertex, 129, 13);
    std::vector<std::vector<std::byte>> storage;
    std::vector<pi::ProgramInput> bucket;
    for (unsigned i = 0; i < pi::maximum_bucket_size + 1; ++i) {
        storage.push_back(common);
        storage.back()[128] = std::byte(i);
        bucket.push_back({Stage::Vertex, storage.back()});
    }
    HashState bucket_hash;
    rejected(pi::build(bucket, provider(bucket_hash)), pi::Error::Code::BucketLimit);
}

ARMIGO_TEST(program_identity, reports_callback_address_miss_and_ambiguity_failures) {
    auto a = program(Stage::Vertex, 160, 21);
    HashState build_hash;
    const auto built = pi::build(std::array{pi::ProgramInput{Stage::Vertex, a}},
                                 provider(build_hash));
    const pi::Index* index = accepted(built);
    if (!index) return;

    ReadState failed_read{0x31000, a, 0, 0, 1};
    HashState ordinary_hash;
    rejected(pi::lookup(*index, Stage::Vertex, failed_read.base,
                        reader(failed_read), provider(ordinary_hash)),
             pi::Error::Code::ReadFailure);
    ReadState failed_full{0x31000, a, 0, 0, 2};
    rejected(pi::lookup(*index, Stage::Vertex, failed_full.base,
                        reader(failed_full), provider(ordinary_hash)),
             pi::Error::Code::ReadFailure);
    ReadState memory{0x31000, a};
    HashState failed_hash{0, 0, 1};
    rejected(pi::lookup(*index, Stage::Vertex, memory.base, reader(memory),
                        provider(failed_hash)), pi::Error::Code::HashFailure);
    ReadState overflow{std::numeric_limits<std::uint64_t>::max() - 31, a};
    rejected(pi::lookup(*index, Stage::Vertex, overflow.base, reader(overflow),
                        provider(ordinary_hash)), pi::Error::Code::AddressOverflow);
    ReadState full_overflow{std::numeric_limits<std::uint64_t>::max() - 120, a};
    rejected(pi::lookup(*index, Stage::Vertex, full_overflow.base,
                        reader(full_overflow), provider(ordinary_hash)),
             pi::Error::Code::AddressOverflow);

    HashState construction_failure{0, 0, 1};
    rejected(pi::build(std::array{pi::ProgramInput{Stage::Vertex, a}},
                       provider(construction_failure)), pi::Error::Code::HashFailure);

    auto b = a;
    b[150] ^= std::byte{7};
    HashState collision_hash;
    collision_hash.collide = true;
    const auto collision_index = pi::build(
        std::array{pi::ProgramInput{Stage::Vertex, a},
                   pi::ProgramInput{Stage::Vertex, b}}, provider(collision_hash));
    const pi::Index* ambiguous_index = accepted(collision_index);
    if (ambiguous_index) {
        ReadState collision_memory{0x41000, a};
        HashState lookup_collision;
        lookup_collision.collide = true;
        rejected(pi::lookup(*ambiguous_index, Stage::Vertex, collision_memory.base,
                            reader(collision_memory), provider(lookup_collision)),
                 pi::Error::Code::Ambiguous);
    }
}

ARMIGO_TEST(program_identity, preflights_lookup_aggregate_work) {
    constexpr std::size_t count = 9;
    constexpr std::size_t first_length = 3800000;
    std::vector<std::vector<std::byte>> storage;
    std::vector<pi::ProgramInput> inputs;
    storage.reserve(count);
    inputs.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        storage.push_back(program(Stage::Vertex, first_length + i, 31));
        std::fill(storage.back().begin() + pi::prefix_size, storage.back().end(),
                  std::byte(0x20 + i));
        inputs.push_back({Stage::Vertex, storage.back()});
    }
    HashState build_hash;
    const auto built = pi::build(inputs, provider(build_hash));
    const pi::Index* index = accepted(built);
    if (!index) return;
    ReadState memory{0x51000, storage.front()};
    HashState lookup_hash;
    rejected(pi::lookup(*index, Stage::Vertex, memory.base, reader(memory),
                        provider(lookup_hash)), pi::Error::Code::LookupWorkLimit);
    ARMIGO_CHECK_EQ(memory.calls, 1u);
    ARMIGO_CHECK_EQ(lookup_hash.calls, 1u);
}

}  // namespace armigo::shader::test
