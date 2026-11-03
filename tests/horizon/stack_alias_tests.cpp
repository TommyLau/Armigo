// ============================================================================
// Armigo Horizon - Compatibility Metadata
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/stack_alias.h>
#include "../../src/horizon/process_memory.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
using namespace armigo::horizon;
namespace {
unsigned failures = 0;
#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr, "Check failed at %d: %s\n", __LINE__, #__VA_ARGS__); ++failures; } } while (false)
constexpr std::uint64_t source = 0x10002000, destination = 0x20002000, size = 0x2000;
std::vector<MemoryBlock> fixture(MemoryType type = MemoryType::normal) {
    return {{{source - 0x2000, 0x6000}, type, 1, 0},
            {{destination - 0x2000, 0x6000}, MemoryType::stack, 0, 3},
            {{0x30000000, 0x1000}, MemoryType::shared, 0x18, 0x1b, 2, 4}};
}
bool equal(std::span<const MemoryBlock> a, std::span<const MemoryBlock> b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i]; const auto& y = b[i];
        if (x.range.address != y.range.address || x.range.size != y.range.size ||
            x.type != y.type || x.attributes != y.attributes || x.permissions != y.permissions ||
            x.ipc_references != y.ipc_references || x.device_references != y.device_references ||
            x.merge_disable != y.merge_disable)
            return false;
    }
    return true;
}
void rejected(const std::vector<MemoryBlock>& blocks,
              const std::vector<StackAliasRecord>& aliases, StackAliasError expected,
              std::uint64_t dest = destination, std::uint64_t src = source,
              std::uint64_t count = size) {
    const auto before = blocks;
    const auto result = plan_stack_alias_unmap(blocks, aliases, dest, src, count);
    const auto* error = std::get_if<StackAliasFailure>(&result);
    CHECK(error && error->code == expected && !error->detail.empty());
    CHECK(equal(blocks, before));
}
void restoration() {
    for (const auto type : {MemoryType::normal, MemoryType::code_data}) {
        const auto input = fixture(type);
        const std::vector<StackAliasRecord> aliases{
            {0x50000000, 0x40000000, size, type},
            {destination, source, size, type}};
        auto result = plan_stack_alias_unmap(input, aliases, destination, source, size);
        const auto* plan = std::get_if<StackAliasUnmapPlan>(&result);
        CHECK(plan);
        if (!plan) continue;
        CHECK(plan->alias_index == 1 && plan->blocks.size() == 7);
        auto expected = std::vector<MemoryBlock>{
            {{source - 0x2000, 0x2000}, type, 1, 0},
            {{source, size}, type, 0, 3},
            {{source + size, 0x2000}, type, 1, 0},
            {{destination - 0x2000, 0x2000}, MemoryType::stack, 0, 3},
            {{destination, size}, MemoryType::free, 0, 0},
            {{destination + size, 0x2000}, MemoryType::stack, 0, 3},
            input.back()};
        CHECK(equal(plan->blocks, expected));
        CHECK(equal(input, fixture(type)));
        CHECK(aliases[1].source == source && aliases.size() == 2);
        ProcessConfiguration config;
        config.heap_address = 0x1000000000; config.alias_address = 0x2000000000;
        config.stack_address = 0x6000000000; config.allowed_core_mask = 1;
        const auto created = ProcessRecord::create(config, {}, input);
        CHECK(created.ok());
        if (created.ok()) {
            CHECK(created.process->replace_memory_map(plan->blocks) == ProcessStateError::none);
            CHECK(created.process->query_memory(source).permissions == 3);
            CHECK(created.process->query_memory(destination).type == MemoryType::free);
        }
        rejected(plan->blocks, aliases, StackAliasError::InvalidSource);
    }
    auto input = fixture();
    input[0].range.size = 0x3000;
    input.insert(input.begin() + 1, {{source + 0x1000, 0x3000}, MemoryType::normal, 1, 0});
    const std::array aliases{StackAliasRecord{destination, source, size}};
    const auto result = plan_stack_alias_unmap(input, aliases, destination, source, size);
    CHECK(std::holds_alternative<StackAliasUnmapPlan>(result));
    if (const auto* plan = std::get_if<StackAliasUnmapPlan>(&result)) {
        const std::vector<MemoryBlock> expected{
            {{source - 0x2000, 0x2000}, MemoryType::normal, 1, 0},
            {{source, 0x1000}, MemoryType::normal, 0, 3},
            {{source + 0x1000, 0x1000}, MemoryType::normal, 0, 3},
            {{source + size, 0x2000}, MemoryType::normal, 1, 0},
            {{destination - 0x2000, 0x2000}, MemoryType::stack, 0, 3},
            {{destination, size}, MemoryType::free, 0, 0},
            {{destination + size, 0x2000}, MemoryType::stack, 0, 3}, input.back()};
        CHECK(equal(plan->blocks, expected));
    }
    // Reverse source/destination address order without changing their roles.
    auto reverse = fixture();
    reverse[0].range.address = destination - 0x2000;
    reverse[1].range.address = source - 0x2000;
    std::swap(reverse[0], reverse[1]);
    const std::array reversed_alias{StackAliasRecord{source, destination, size}};
    const auto reversed = plan_stack_alias_unmap(reverse, reversed_alias, source, destination, size);
    CHECK(std::holds_alternative<StackAliasUnmapPlan>(reversed));
    if (const auto* plan = std::get_if<StackAliasUnmapPlan>(&reversed)) {
        const std::vector<MemoryBlock> expected{
            {{source - 0x2000, 0x2000}, MemoryType::stack, 0, 3},
            {{source, size}, MemoryType::free, 0, 0},
            {{source + size, 0x2000}, MemoryType::stack, 0, 3},
            {{destination - 0x2000, 0x2000}, MemoryType::normal, 1, 0},
            {{destination, size}, MemoryType::normal, 0, 3},
            {{destination + size, 0x2000}, MemoryType::normal, 1, 0}, reverse.back()};
        CHECK(equal(plan->blocks, expected));
    }
}
void negative_cases() {
    const std::vector<StackAliasRecord> aliases{{destination, source, size}};
    const auto input = fixture();
    rejected(input, {}, StackAliasError::UnregisteredAlias);
    rejected(input, {aliases[0], aliases[0]}, StackAliasError::UnregisteredAlias);
    rejected(input, aliases, StackAliasError::UnregisteredAlias, destination, source, 0x1000);
    rejected(input, aliases, StackAliasError::UnregisteredAlias, destination + 0x1000);
    for (const auto count : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{0x2001}})
        rejected(input, aliases, StackAliasError::InvalidRange, destination, source, count);
    rejected(input, aliases, StackAliasError::InvalidRange, source, source);
    rejected(input, aliases, StackAliasError::InvalidRange, destination + 1);
    rejected(input, aliases, StackAliasError::InvalidRange, destination,
             std::numeric_limits<std::uint64_t>::max() - 4095, 0x2000);
    rejected(input, aliases, StackAliasError::InvalidRange, ProcessRecord::query_end);
    for (unsigned field = 0; field < 5; ++field) {
        auto bad = input;
        if (field == 0) bad[0].permissions = 3;
        if (field == 1) bad[0].attributes = 0;
        if (field == 2) bad[0].ipc_references = 1;
        if (field == 3) bad[0].device_references = 1;
        if (field == 4) bad[0].type = MemoryType::code;
        rejected(bad, aliases, StackAliasError::InvalidSource);
        bad = input;
        if (field == 0) bad[1].permissions = 1;
        if (field == 1) bad[1].attributes = 1;
        if (field == 2) bad[1].ipc_references = 1;
        if (field == 3) bad[1].device_references = 1;
        if (field == 4) bad[1].type = MemoryType::normal;
        rejected(bad, aliases, StackAliasError::InvalidDestination);
    }
    auto bad = input;
    bad[0].range.size = 0x2000;  // A hole at the selected source.
    rejected(bad, aliases, StackAliasError::InvalidSource);
    bad = input; bad[1].range.size = 0x2000;
    rejected(bad, aliases, StackAliasError::InvalidDestination);
    bad = input; std::swap(bad[0], bad[1]);
    rejected(bad, aliases, StackAliasError::InvalidMemoryMap);
    bad = input; bad[0].range.size = 0;
    rejected(bad, aliases, StackAliasError::InvalidMemoryMap);
    bad = input; bad[1].range.address = source;
    rejected(bad, aliases, StackAliasError::InvalidMemoryMap);
    bad = input; bad[2].type = static_cast<MemoryType>(255);
    rejected(bad, aliases, StackAliasError::InvalidMemoryMap);
    bad = input; bad[2] = {{0x30000000, 0x1000}, MemoryType::free, 1, 0};
    rejected(bad, aliases, StackAliasError::InvalidMemoryMap);
    rejected(input, {{destination, source, size, MemoryType::stack}}, StackAliasError::InvalidSource);
    rejected(std::vector<MemoryBlock>(65537), aliases, StackAliasError::AdmissionLimit);
    rejected(input, std::vector<StackAliasRecord>(4097), StackAliasError::AdmissionLimit);
    std::vector<MemoryBlock> many;
    for (std::uint64_t i = 0; i < 65534; ++i)
        many.push_back({{i * 4096, 4096}, MemoryType::free, 0, 0});
    many.push_back({{0x40000000, 0x4000}, MemoryType::normal, 1, 0});
    many.push_back({{0x50000000, 0x4000}, MemoryType::stack, 0, 3});
    rejected(many, {{0x50001000, 0x40001000, 0x2000}}, StackAliasError::AdmissionLimit,
             0x50001000, 0x40001000, 0x2000);
}
void merge_boundaries() {
    namespace pm = process_memory;
    constexpr std::uint64_t base = 0x40000000, page = guest_page_size;
    const MemoryBlock shared{{base, page}, MemoryType::shared, 0, 1};
    auto map = pm::replace({}, shared.range, shared, pm::merge_disable_mapping);
    map = pm::replace(map, {base + page, page}, shared, pm::merge_disable_mapping);
    CHECK(map.size() == 2 && map[0].range.size == page && map[1].range.size == page);
    CHECK(map[0].merge_disable == 1 && map[1].merge_disable == 1);

    // An unrelated update must not erase either shared-object boundary.
    map = pm::replace(map, {base + 4 * page, page},
        MemoryBlock{{}, MemoryType::normal, 0, 3});
    CHECK(map.size() == 3 && map[0].range.size == page && map[1].range.size == page);
    CHECK(pm::uniform(map, {base, 2 * page}).has_value());
    auto read_write = shared;
    read_write.permissions = 3;
    map = pm::replace(map, {base, 2 * page}, read_write);
    CHECK(map.size() == 3 && map[0].merge_disable == 1 && map[1].merge_disable == 1);
    CHECK(map[0].permissions == 3 && map[1].permissions == 3);

    // Only an explicit clear permits a former mapping boundary to disappear.
    const auto retained = pm::replace(map, {base, 2 * page}, std::nullopt);
    CHECK(retained.size() == 3 && retained[0].type == MemoryType::free &&
          retained[0].merge_disable == 1 && retained[1].merge_disable == 1);
    map = pm::replace(map, {base, 2 * page}, std::nullopt, 0, pm::merge_disable_mapping);
    CHECK(map.size() == 1 && map[0].range.address == base + 4 * page);

    MemoryBlock normal{{base, 3 * page}, MemoryType::normal, 0, 3, 0, 0, 0x21};
    const auto left = pm::fragment(normal, {base, page});
    const auto middle = pm::fragment(normal, {base + page, page});
    const auto right = pm::fragment(normal, {base + 2 * page, page});
    CHECK(left.merge_disable == 1 && middle.merge_disable == 0 && right.merge_disable == 0x20);
    auto changed = normal;
    changed.permissions = 1;
    map = pm::replace(std::span{&normal, 1}, middle.range, changed);
    CHECK(map.size() == 3 && map[0].merge_disable == 1 &&
          map[1].merge_disable == 0 && map[2].merge_disable == 0x20);
    map = pm::replace(map, middle.range, normal);
    CHECK(map.size() == 1 && map[0].range.size == 3 * page && map[0].merge_disable == 0x21);

    // Lock barriers may differ without changing range-state uniformity.
    std::array locked{
        MemoryBlock{{base, page}, MemoryType::normal, 1, 0, 0, 0, 8},
        MemoryBlock{{base + page, page}, MemoryType::normal, 1, 0, 0, 0, 9}};
    CHECK(pm::uniform(locked, {base, 2 * page}).has_value());
    map = pm::replace(locked, {base, 2 * page}, normal, 0, pm::merge_disable_locked);
    CHECK(map.size() == 2 && map[0].merge_disable == 0 && map[1].merge_disable == 1);

    // Unmapping a stack pair clears only its own boundary kind.
    std::array pair{
        MemoryBlock{{source, size}, MemoryType::normal, 1, 0, 0, 0, 9},
        MemoryBlock{{destination, size}, MemoryType::stack, 0, 3, 0, 0, 9}};
    const std::array aliases{StackAliasRecord{destination, source, size}};
    const auto planned = plan_stack_alias_unmap(pair, aliases, destination, source, size);
    const auto* plan = std::get_if<StackAliasUnmapPlan>(&planned);
    CHECK(plan && plan->blocks.size() == 2);
    if (plan && plan->blocks.size() == 2)
        CHECK(plan->blocks[0].merge_disable == 1 && plan->blocks[1].merge_disable == 8);

    // Internal barriers must not change the public forty-byte encoding.
    auto public_normal = normal;
    public_normal.merge_disable = 0;
    CHECK(encode_memory_info(normal) == encode_memory_info(public_normal));
}
}  // namespace
int main() {
    restoration(); negative_cases(); merge_boundaries();
    std::printf("Stack alias verification: %u failures\n", failures);
    return failures ? 1 : 0;
}
