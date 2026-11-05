// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/graphics/presentation_hints.h>
#include <armigo/execution/guest_dispatch.h>
#include <armigo/platform/thread_slot.h>
#include <armigo/runtime/guest_memory.h>
#include <armigo/runtime/launch_plan.h>
#include <armigo/runtime/preparation_format.h>

#include "preparation_fixture.h"
#include "../../src/runtime/prepared_artifact.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace {
using namespace armigo;
using namespace armigo::execution;
using namespace armigo::runtime;

unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Check failed at line %u: %s\n", __LINE__, #condition); \
    ++failures; \
} } while (false)

template<class T>
const T* value(const LaunchResult<T>& result) {
    return std::get_if<T>(&result);
}

template<class T>
bool failed(const LaunchResult<T>& result, LaunchError code) {
    const auto* failure = std::get_if<LaunchFailure>(&result);
    return failure != nullptr && failure->code == code &&
           !failure->operation.empty() && !failure->detail.empty();
}

PreparationOptions preparation_options(PreparationMode mode = PreparationMode::synthetic_test) {
    return {
        .required_preparation_identity = installed_preparation_identity(),
        .max_artifact_bytes = default_prepared_artifact_limit,
        .mode = mode,
    };
}

LaunchProfile synthetic_profile() {
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.is_64_bit = true;
    return profile;
}

LaunchProfile original_profile() {
    auto profile = synthetic_profile();
    profile.origin = ProfileOrigin::npdm;
    return profile;
}

std::uint64_t read_wide(const platform::HostModule& mapping, std::uint64_t offset) {
    const auto bytes = mapping.readable_bytes(offset, 8);
    if (!bytes) return 0;
    std::uint64_t result = 0;
    std::memcpy(&result, bytes->data(), sizeof(result));
    return result;
}

std::uint32_t read_word(const platform::HostModule& mapping, std::uint64_t offset) {
    const auto bytes = mapping.readable_bytes(offset, 4);
    if (!bytes) return 0;
    std::uint32_t result = 0;
    std::memcpy(&result, bytes->data(), sizeof(result));
    return result;
}

std::size_t module_index(const ModuleSet& modules, std::string_view name) {
    for (std::size_t index = 0; index < modules.module_count(); ++index) {
        if (modules.module_name(index) == name) return index;
    }
    return modules.module_count();
}

struct SupervisorState {
    std::uint64_t calls = 0;
    bool correct = true;
};

GateStatus supervisor(std::uint16_t immediate, GateCall& call,
                      GuestThreadContext&, void* raw) {
    auto& state = *static_cast<SupervisorState*>(raw);
    ++state.calls;
    state.correct &= immediate == 0x42 && call.x(0) != 0;
    return GateStatus::resume;
}

void derives_dependency_order_and_rejects_unsupported_profiles() {
    const auto input = test::two_module_input();
    const auto artifacts = prepare_modules(input, preparation_options());
    CHECK(value(artifacts) != nullptr);
    if (!value(artifacts)) return;
    const auto plan = plan_launch(input, synthetic_profile(), *value(artifacts));
    CHECK(value(plan) != nullptr);
    if (value(plan)) {
        CHECK(value(plan)->module_order == std::vector<std::size_t>({1, 0}));
        CHECK(value(plan)->entry.module_index == 0);
        CHECK(value(plan)->entry.image_offset == 0x100);
        CHECK(value(plan)->entry_abi == LaunchEntryAbi::synthetic_exact_sp);
        CHECK(value(plan)->relocation_owner == LaunchOwnership::host_loader);
        CHECK(value(plan)->initialization_owner == LaunchOwnership::guest_bootstrap);
    }

    auto original = synthetic_profile();
    original.origin = ProfileOrigin::npdm;
    CHECK(failed(plan_launch(input, original, *value(artifacts)), LaunchError::artifact_mismatch));
    auto narrow = synthetic_profile();
    narrow.is_64_bit = false;
    CHECK(failed(plan_launch(input, narrow, *value(artifacts)), LaunchError::unsupported));

    const auto missing_entry_input = test::two_module_input("xstart");
    const auto missing_artifacts = prepare_modules(missing_entry_input, preparation_options());
    CHECK(value(missing_artifacts) != nullptr);
    if (value(missing_artifacts)) {
        CHECK(failed(plan_launch(missing_entry_input, synthetic_profile(),
                                 *value(missing_artifacts)), LaunchError::unsupported));
    }
    const auto ambiguous_input = test::two_module_input("_start", true);
    const auto ambiguous_artifacts = prepare_modules(ambiguous_input, preparation_options());
    CHECK(value(ambiguous_artifacts) != nullptr);
    if (value(ambiguous_artifacts)) {
        CHECK(failed(plan_launch(ambiguous_input, synthetic_profile(),
                                 *value(ambiguous_artifacts)), LaunchError::unsupported));
    }
}

void rejects_invalid_plans_before_publication() {
    const auto input = test::two_module_input();
    const auto artifacts = prepare_modules(input, preparation_options());
    const auto planned = value(artifacts) ?
        plan_launch(input, synthetic_profile(), *value(artifacts)) :
        LaunchResult<LaunchPlan>{LaunchFailure{}};
    CHECK(value(artifacts) != nullptr && value(planned) != nullptr);
    if (!value(artifacts) || !value(planned)) return;
    auto key = std::make_shared<platform::ThreadSlotKey>();
    CHECK(key->valid());
    if (!key->valid()) return;

    auto rejects = [&](LaunchPlan plan, LaunchError error) {
        CHECK(failed(load_prepared_launch(input, *value(artifacts), synthetic_profile(),
                                          plan, key), error));
    };
    auto plan = *value(planned);
    plan.module_order = {0, 0};
    rejects(plan, LaunchError::invalid_input);
    plan = *value(planned);
    plan.module_order = {1};
    rejects(plan, LaunchError::invalid_input);
    plan = *value(planned);
    plan.module_order = {1, 2};
    rejects(plan, LaunchError::invalid_input);
    plan = *value(planned);
    plan.entry.module_index = 2;
    rejects(plan, LaunchError::invalid_input);
    plan = *value(planned);
    plan.entry.image_offset = test::image_size;
    rejects(plan, LaunchError::invalid_input);
    plan = *value(planned);
    plan.entry.image_offset += 1;
    rejects(plan, LaunchError::invalid_input);
    plan = *value(planned);
    plan.entry_abi = LaunchEntryAbi::original_process;
    rejects(plan, LaunchError::unsupported);
    plan = *value(planned);
    plan.relocation_owner = LaunchOwnership::guest_bootstrap;
    rejects(plan, LaunchError::unsupported);
    plan = *value(planned);
    plan.initialization_owner = LaunchOwnership::host_loader;
    rejects(plan, LaunchError::unsupported);
    CHECK(failed(load_prepared_launch(input, *value(artifacts), synthetic_profile(),
                                      *value(planned), nullptr), LaunchError::invalid_input));
    ModuleLoadOptions guest_override;
    guest_override.relocation_mode = ModuleRelocationMode::guest_bootstrap;
    CHECK(failed(load_prepared_launch(input, *value(artifacts), synthetic_profile(),
                                      *value(planned), key, guest_override),
                 LaunchError::unsupported));
}

void loads_rootless_companion_without_admitting_entries() {
    const auto input = test::rootless_companion_input();
    const auto artifacts = prepare_modules(input, preparation_options());
    CHECK(value(artifacts) != nullptr);
    if (!value(artifacts)) return;
    const auto planned = plan_launch(input, synthetic_profile(), *value(artifacts));
    CHECK(value(planned) != nullptr);
    if (!value(planned)) return;
    auto key = std::make_shared<platform::ThreadSlotKey>();
    CHECK(key->valid());
    if (!key->valid()) return;
    const auto loaded = load_prepared_launch(input, *value(artifacts), synthetic_profile(),
                                             *value(planned), key);
    CHECK(value(loaded) != nullptr);
    if (!value(loaded)) return;
    const auto& launch = *value(loaded);
    const auto companion = module_index(*launch.modules, "sdk");
    CHECK(companion < launch.modules->module_count());
    if (companion >= launch.modules->module_count()) return;
    CHECK(launch.modules->permits_guest_entry(launch.entry_address));
    CHECK(!launch.modules->permits_guest_entry(
        launch.modules->mapping(companion).base_address() + 0x100));
}

void plans_and_loads_original_process_entry() {
    const auto input = test::original_process_input();
    const auto artifacts = prepare_modules(input, preparation_options(PreparationMode::automatic));
    CHECK(value(artifacts) != nullptr);
    if (!value(artifacts)) return;
    const auto planned = plan_launch(input, original_profile(), *value(artifacts));
    CHECK(value(planned) != nullptr);
    if (!value(planned)) return;
    CHECK(value(planned)->module_order == std::vector<std::size_t>({1, 0, 2}));
    CHECK(value(planned)->entry.module_index == 1);
    CHECK(value(planned)->entry.image_offset == 0);
    CHECK(value(planned)->entry_abi == LaunchEntryAbi::original_process);
    CHECK(value(planned)->relocation_owner == LaunchOwnership::guest_bootstrap);
    CHECK(value(planned)->initialization_owner == LaunchOwnership::guest_bootstrap);

    std::vector<preparation::ModulePayload> payloads;
    for (const auto& module : value(artifacts)->modules) {
        const auto payload = preparation::decode_payload(module.payload);
        CHECK(value(payload) != nullptr);
        if (!value(payload)) return;
        payloads.push_back(*value(payload));
    }
    auto key = std::make_shared<platform::ThreadSlotKey>();
    CHECK(key->valid());
    if (!key->valid()) return;
    const auto loaded = load_prepared_launch(input, *value(artifacts), original_profile(),
                                             *value(planned), key);
    CHECK(value(loaded) != nullptr);
    if (!value(loaded)) return;
    const auto& launch = *value(loaded);
    CHECK(launch.entry_address != 0 && launch.process_entry_address != 0);
    CHECK(launch.entry_address != launch.process_entry_address);
    CHECK(std::is_sorted(launch.original_entry_landings.begin(),
        launch.original_entry_landings.end(), [](const auto& left, const auto& right) {
            return left.logical_root < right.logical_root;
        }));
    const auto expected_landing_count = std::accumulate(payloads.begin(), payloads.end(),
        std::size_t{}, [](std::size_t count, const auto& payload) {
            return count + payload.roots.size();
        });
    CHECK(launch.original_entry_landings.size() == expected_landing_count);
    const auto rtld_index = module_index(*launch.modules, "rtld");
    const auto main_index = module_index(*launch.modules, "main");
    CHECK(rtld_index == 0 && main_index == 1);
    if (rtld_index >= launch.modules->module_count() ||
        main_index >= launch.modules->module_count()) return;
    const auto& rtld = launch.modules->mapping(rtld_index);
    const auto& main = launch.modules->mapping(main_index);
    CHECK(launch.entry_address == rtld.base_address());
    CHECK(launch.modules->permits_guest_entry(launch.entry_address));
    CHECK(!launch.modules->permits_guest_entry(launch.process_entry_address));
    CHECK(read_wide(main, test::original_pointer_slot_offset) == 0);
    for (std::size_t index = 1; index < launch.modules->module_count(); ++index) {
        const auto& previous = launch.modules->mapping(index - 1);
        const auto& current = launch.modules->mapping(index);
        CHECK(previous.base_address() + previous.mapping_size() <= current.base_address());
    }
    for (std::size_t loaded_index = 0;
         loaded_index < launch.modules->module_count(); ++loaded_index) {
        const auto manifest_index = launch.plan.module_order[loaded_index];
        const auto& mapping = launch.modules->mapping(loaded_index);
        const auto appendix = launch.modules->appended_code(loaded_index);
        CHECK(appendix.has_value());
        if (!appendix) return;
        for (std::size_t root_index = 0;
             root_index < payloads[manifest_index].roots.size(); ++root_index) {
            const auto root = mapping.base_address() +
                payloads[manifest_index].roots[root_index];
            const auto expected_landing = mapping.base_address() + appendix->offset +
                payloads[manifest_index].image.appendix.size() + root_index * 8;
            const auto found = std::lower_bound(launch.original_entry_landings.begin(),
                launch.original_entry_landings.end(), root,
                [](const auto& candidate, std::uint64_t value) {
                    return candidate.logical_root < value;
                });
            CHECK(found != launch.original_entry_landings.end());
            if (found == launch.original_entry_landings.end()) continue;
            CHECK(found->logical_root == root && found->landing == expected_landing);
            CHECK(!launch.modules->permits_guest_entry(found->landing));
            CHECK(read_word(mapping, found->landing - mapping.base_address()) ==
                  0xaa1f03f0u);
            const auto branch = aarch64::encode_b(found->landing + 4, root);
            CHECK(branch.ok());
            if (branch.ok()) {
                CHECK(read_word(mapping,
                    found->landing - mapping.base_address() + 4) == branch.word);
            }
        }
    }
    const auto main_landing = std::lower_bound(launch.original_entry_landings.begin(),
        launch.original_entry_landings.end(), launch.entry_address,
        [](const auto& candidate, std::uint64_t value) {
            return candidate.logical_root < value;
        });
    CHECK(main_landing != launch.original_entry_landings.end());
    if (main_landing != launch.original_entry_landings.end())
        CHECK(launch.process_entry_address == main_landing->landing);

    auto forged = *value(planned);
    forged.relocation_owner = LaunchOwnership::host_loader;
    CHECK(failed(load_prepared_launch(input, *value(artifacts), original_profile(),
                                      forged, key), LaunchError::unsupported));

    const auto main_input = test::original_process_input(false);
    const auto main_artifacts = prepare_modules(main_input, preparation_options(PreparationMode::automatic));
    CHECK(value(main_artifacts) != nullptr);
    if (value(main_artifacts)) {
        const auto main_plan = plan_launch(
            main_input, original_profile(), *value(main_artifacts));
        CHECK(value(main_plan) != nullptr);
        if (value(main_plan)) {
            CHECK(value(main_plan)->module_order ==
                  std::vector<std::size_t>({0}));
            CHECK(value(main_plan)->entry.module_index == 0);
            CHECK(value(main_plan)->entry.image_offset == 0);
        }
    }
    auto missing_rtld = input;
    missing_rtld.modules.erase(missing_rtld.modules.begin() + 1);
    CHECK(failed(prepare_modules(missing_rtld, preparation_options(PreparationMode::automatic)), LaunchError::unsupported));
}

void detached_metadata_and_ignored_orphans_preserve_manifest_indices() {
    auto input = test::original_process_input();
    const auto metadata = input.process_metadata;
    input.process_metadata.reset();
    // An ignored file has deliberately invalid bytes and precedes every live
    // manifest entry. It must neither be decoded nor retain an artifact index.
    input.modules.insert(input.modules.begin(), {"subsdk1", {std::byte{0xff}}});
    const auto artifacts = prepare_modules(input, preparation_options(PreparationMode::automatic));
    CHECK(value(artifacts) != nullptr);
    if (!value(artifacts)) return;
    CHECK(value(artifacts)->modules.size() == 3);
    CHECK(value(artifacts)->ignored_modules == std::vector<std::string>{"subsdk1"});
    CHECK(value(artifacts)->modules[0].module_name == "main");
    CHECK(value(artifacts)->modules[1].module_name == "rtld");
    CHECK(value(artifacts)->modules[2].module_name == "sdk");
    input.process_metadata = metadata;
    const auto plan = plan_launch(input, original_profile(), *value(artifacts));
    CHECK(value(plan) != nullptr);
    if (!value(plan)) return;
    CHECK(value(plan)->module_order == std::vector<std::size_t>({1, 0, 2}));
    CHECK(value(plan)->entry.module_index == 1 && value(plan)->entry.image_offset == 0);
    auto key = std::make_shared<platform::ThreadSlotKey>();
    const auto loaded = load_prepared_launch(input, *value(artifacts), original_profile(),
                                             *value(plan), key);
    CHECK(value(loaded) != nullptr);
    if (value(loaded)) {
        CHECK(value(loaded)->modules->module_count() == 3);
        CHECK(module_index(*value(loaded)->modules, "rtld") == 0);
        CHECK(value(loaded)->entry_address == value(loaded)->modules->mapping(0).base_address());
    }
}

void preparation_mode_is_bound_to_the_artifact() {
    test::ModuleSpec module;
    module.symbols = {{"_start", 0, 4, true}};
    test::put_word(module, 0, aarch64::encode_ret().word);
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", test::build_module(std::move(module))});
    const auto synthetic = prepare_modules(input, preparation_options(PreparationMode::synthetic_test));
    const auto original = prepare_modules(input, preparation_options(PreparationMode::automatic));
    CHECK(value(synthetic) != nullptr && value(original) != nullptr);
    if (!value(synthetic) || !value(original)) return;
    CHECK(value(plan_launch(input, synthetic_profile(), *value(synthetic))) != nullptr);
    CHECK(failed(plan_launch(input, synthetic_profile(), *value(original)), LaunchError::artifact_mismatch));
    input.process_metadata = std::vector<std::byte>{std::byte{1}};
    CHECK(value(plan_launch(input, original_profile(), *value(original))) != nullptr);
    CHECK(failed(plan_launch(input, original_profile(), *value(synthetic)), LaunchError::artifact_mismatch));
}

void original_roster_ignores_dynamic_dependency_names() {
    loader::ExeFsDirectoryInput input;
    for (const auto name : {"sdk", "subsdk0", "main", "rtld"}) {
        test::ModuleSpec module;
        module.name = name;
        module.dependencies = {"fictional.library.name", "not-an-exefs-file"};
        module.symbols = {{std::string(name) + "_body", 0x100, 4, true}};
        test::put_word(module, 0, aarch64::encode_b(0, 0x100).word);
        test::put_word(module, 0x100, aarch64::encode_ret().word);
        input.modules.push_back({name, test::build_module(std::move(module))});
    }
    const auto artifacts = prepare_modules(input, preparation_options(PreparationMode::automatic));
    CHECK(value(artifacts) != nullptr);
    if (!value(artifacts)) return;
    input.process_metadata = std::vector<std::byte>{std::byte{1}};
    const auto plan = plan_launch(input, original_profile(), *value(artifacts));
    CHECK(value(plan) != nullptr);
    if (value(plan)) CHECK(value(plan)->module_order == std::vector<std::size_t>({3, 2, 1, 0}));
    for (std::size_t index = 0; index < value(artifacts)->modules.size(); ++index) {
        const auto payload = preparation::decode_payload(value(artifacts)->modules[index].payload);
        CHECK(value(payload) != nullptr);
        if (value(payload)) CHECK(value(payload)->load_rank == 3 - index);
    }
    input.process_metadata.reset();
    CHECK(failed(prepare_modules(input, preparation_options(PreparationMode::synthetic_test)),
                 LaunchError::unsupported));
}

void original_and_synthetic_preemption_select_distinct_providers() {
    test::ModuleSpec main;
    main.dependencies = {"rtld"};
    main.symbols = {{"_start", 0x100, 12, true}, {"shared_export", 0x300, 4, true}};
    main.relocations = {{test::pointer_slot_offset, 2, 257, 0}};
    test::put_word(main, 0x100, aarch64::encode_adr(9, 0x100, test::pointer_slot_offset).word);
    test::put_word(main, 0x104, test::load64(9, 9));
    const auto indirect = aarch64::encode_br(9).word;
    test::put_word(main, 0x108, indirect);
    test::put_word(main, 0x300, aarch64::encode_ret().word);

    test::ModuleSpec rtld;
    rtld.symbols = {{"bootstrap_body", 0x100, 4, true}, {"shared_export", 0x200, 4, true}};
    test::put_word(rtld, 0, aarch64::encode_b(0, 0x100).word);
    test::put_word(rtld, 0x100, aarch64::encode_ret().word);
    test::put_word(rtld, 0x200, aarch64::encode_ret().word);
    loader::ExeFsDirectoryInput input;
    input.modules = {{"main", test::build_module(std::move(main))},
                     {"rtld", test::build_module(std::move(rtld))}};

    // The read-only slot isolates offline relocation provenance. This fixture
    // is not loaded as an original process, whose relocation writes are guest-owned.
    for (const auto mode : {PreparationMode::original_process, PreparationMode::synthetic_test}) {
        const auto artifacts = prepare_modules(input, preparation_options(mode));
        CHECK(value(artifacts) != nullptr);
        if (!value(artifacts)) continue;
        const auto decoded = preparation::decode_payload(value(artifacts)->modules[0].payload);
        CHECK(value(decoded) != nullptr);
        if (!value(decoded)) continue;
        const auto& payload = *value(decoded);
        // Both modes load rtld before main; only symbol preemption differs.
        CHECK(payload.load_rank == 1);
        const auto site = std::find_if(payload.image.gate_sites.begin(), payload.image.gate_sites.end(),
            [&](const auto& candidate) { return candidate.instruction == indirect; });
        CHECK(site != payload.image.gate_sites.end());
        if (site == payload.image.gate_sites.end()) continue;
        CHECK(site->allowed_indirect_targets.size() == 1);
        if (site->allowed_indirect_targets.size() != 1) continue;
        const auto target = site->allowed_indirect_targets.front();
        const bool original = mode == PreparationMode::original_process;
        CHECK(target.module_index == (original ? 1u : 0u));
        CHECK(target.image_offset == (original ? 0x200u : 0x300u));
    }
}

void admits_complete_appendix_pages_without_new_entries(bool original = false) {
    const auto page = platform::HostModule::native_page_size();
    CHECK(page.has_value());
    if (!page) return;
    test::ModuleSpec module;
    module.name = "main";
    const auto segment_bytes = std::max(*page, std::size_t{0x4000});
    module.text.resize(segment_bytes);
    module.rodata.resize(segment_bytes);
    module.data.resize(segment_bytes);
    module.symbols = {{"_start", 0x100, 8, true}};
    if (original) test::put_word(module, 0, aarch64::encode_b(0, 0x100).word);
    test::put_word(module, 0x100, aarch64::encode_svc(0x42).word);
    test::put_word(module, 0x104, aarch64::encode_ret().word);
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", test::build_module(std::move(module))});
    if (original) input.process_metadata = std::vector<std::byte>{std::byte{1}};
    const auto artifacts = prepare_modules(input, preparation_options(original ? PreparationMode::automatic : PreparationMode::synthetic_test));
    CHECK(value(artifacts) != nullptr);
    if (!value(artifacts)) return;
    const auto payload = preparation::decode_payload(value(artifacts)->modules.front().payload);
    CHECK(value(payload) != nullptr);
    if (!value(payload)) return;
    const auto& image = value(payload)->image;
    CHECK(!image.appendix.empty() && image.appendix.size() % *page != 0);

    auto profile = synthetic_profile();
    if (original) profile.origin = ProfileOrigin::npdm;
    profile.process.allowed_core_mask = 1;
    profile.process.heap_address = 0;
    profile.process.alias_address = 0;
    profile.process.stack_address = 0;
    profile.process.heap_size = 8 * 1024 * 1024;
    profile.process.base_alias_size = 2 * 1024 * 1024;
    profile.process.stack_size = 2 * 1024 * 1024;
    profile.permissions.handle_table_capacity = 64;
    const auto plan = plan_launch(input, profile, *value(artifacts));
    CHECK(value(plan) != nullptr);
    if (!value(plan)) return;
    auto key = std::make_shared<platform::ThreadSlotKey>();
    const auto loaded = load_prepared_launch(input, *value(artifacts), profile, *value(plan), key);
    CHECK(value(loaded) != nullptr);
    if (!value(loaded)) return;
    const auto& launch = *value(loaded);
    const auto& mapping = launch.modules->mapping(0);
    const auto appendix = launch.modules->appended_code(0);
    CHECK(appendix.has_value());
    if (!appendix) return;
    CHECK(appendix->offset == image.appendix_offset);
    const auto payload_size = image.appendix.size() +
        (original ? value(payload)->roots.size() * 8 : 0);
    CHECK(appendix->size == ((payload_size + *page - 1) & ~(*page - 1)));
    CHECK(appendix->size > image.appendix.size());
    const auto tail = mapping.readable_bytes(image.appendix_offset + payload_size,
                                            appendix->size - payload_size);
    CHECK(tail.has_value());
    if (tail) CHECK(std::all_of(tail->begin(), tail->end(), [](std::byte byte) { return byte == std::byte{}; }));
    CHECK(launch.modules->permits_guest_entry(launch.entry_address));
    for (std::uint64_t offset = appendix->offset; offset < appendix->offset + appendix->size; offset += 4)
        CHECK(!launch.modules->permits_guest_entry(mapping.base_address() + offset));

    const auto memory = create_guest_memory(profile, {16 * 1024 * 1024}, launch.modules,
        std::make_shared<horizon::HandleTable>(profile.permissions.handle_table_capacity));
    CHECK(value(memory) != nullptr);
    if (value(memory)) {
        const auto block = (*value(memory))->process()->query_memory(
            mapping.base_address() + appendix->offset + appendix->size - 1);
        CHECK(block.type == (original ? horizon::MemoryType::code_data : horizon::MemoryType::code));
        CHECK(block.permissions == 5);
    }
}

void loads_and_executes_prepared_multimodule_code() {
    const auto input = test::two_module_input();
    const auto artifacts = prepare_modules(input, preparation_options());
    const auto planned = value(artifacts) ?
        plan_launch(input, synthetic_profile(), *value(artifacts)) :
        LaunchResult<LaunchPlan>{LaunchFailure{}};
    auto key = std::make_shared<platform::ThreadSlotKey>();
    CHECK(value(artifacts) != nullptr && value(planned) != nullptr && key->valid());
    if (!value(artifacts) || !value(planned) || !key->valid()) return;

    const auto loaded = load_prepared_launch(input, *value(artifacts), synthetic_profile(),
                                             *value(planned), key);
    CHECK(value(loaded) != nullptr);
    if (!value(loaded)) return;
    const auto& launch = *value(loaded);
    CHECK(launch.thread_slot == key);
    CHECK(launch.modules != nullptr && launch.sites != nullptr && launch.entry_address != 0 &&
          launch.process_entry_address == 0 && launch.original_entry_landings.empty());
    if (!launch.modules || !launch.sites || launch.entry_address == 0) return;
    const auto main_index = module_index(*launch.modules, "main");
    const auto sdk_index = module_index(*launch.modules, "sdk");
    CHECK(main_index < launch.modules->module_count());
    CHECK(sdk_index < launch.modules->module_count());
    if (main_index >= launch.modules->module_count() ||
        sdk_index >= launch.modules->module_count()) return;

    const auto& main = launch.modules->mapping(main_index);
    const auto& sdk = launch.modules->mapping(sdk_index);
    CHECK(launch.modules->permits_guest_entry(launch.entry_address));
    CHECK(!launch.modules->permits_guest_entry(launch.entry_address + 4));
    CHECK(!launch.modules->permits_guest_entry(main.base_address() + test::unreachable_word_offset));
    const auto appendix = launch.modules->appended_code(main_index);
    CHECK(appendix.has_value());
    if (appendix) CHECK(!launch.modules->permits_guest_entry(main.base_address() + appendix->offset));
    CHECK(read_wide(main, test::pointer_slot_offset) == sdk.base_address() + 0x100);
    CHECK(read_wide(main, test::packed_relative_offset) ==
          main.base_address() + test::packed_relative_addend);
    CHECK(read_wide(main, test::literal_offset) == test::literal_value);
    CHECK(read_word(main, test::unreachable_word_offset) ==
          test::unreachable_instruction_word);

    alignas(16) std::array<std::uint64_t, 7> output{};
    CHECK(output[0] == 0);
    GuestThreadContext context;
    context.set_guest_x18(10);
    context.set_thread_pointer(0x12345678abcdef00ull);
    SupervisorState supervisor_state;
    GuestDispatchScope dispatch(launch.sites, context,
                                {supervisor, &supervisor_state});
    CHECK(dispatch.setup_error() == GuestDispatchFailureKind::none);
    platform::ThreadSlotBinding binding(*launch.thread_slot,
                                        context.bound_register_cell());
    CHECK(binding.native_access().has_value());
    if (dispatch.setup_error() != GuestDispatchFailureKind::none ||
        !binding.native_access()) return;
    using Entry = std::uint64_t* (*)(std::uint64_t*);
    const auto entry = reinterpret_cast<Entry>(launch.entry_address);
    CHECK(entry(output.data()) == output.data());
    CHECK(output[0] == 1);
    CHECK(output[1] == 1);
    CHECK(output[2] == 1);
    CHECK(output[3] == 0x12345678abcdef00ull);
    CHECK(output[4] == 15);
    CHECK(output[6] == test::literal_value);
    CHECK(supervisor_state.correct && supervisor_state.calls == 1);
    CHECK(dispatch.failure_count() == 0);

    const auto first_output = output;
    const auto first_x18 = context.guest_x18();
    bool second_correct = false;
    std::thread second([&] {
        alignas(16) std::array<std::uint64_t, 7> second_output{};
        GuestThreadContext second_context;
        second_context.set_guest_x18(20);
        second_context.set_thread_pointer(0x0fedcba987654321ull);
        SupervisorState second_supervisor;
        GuestDispatchScope second_dispatch(launch.sites, second_context,
                                           {supervisor, &second_supervisor});
        platform::ThreadSlotBinding second_binding(*launch.thread_slot,
                                                   second_context.bound_register_cell());
        if (second_dispatch.setup_error() != GuestDispatchFailureKind::none ||
            !second_binding.native_access()) return;
        const bool returned = entry(second_output.data()) == second_output.data();
        second_correct = returned && second_output[0] == 1 && second_output[1] == 1 &&
            second_output[2] == 1 && second_output[3] == 0x0fedcba987654321ull &&
            second_output[4] == 25 && second_output[6] == test::literal_value &&
            second_supervisor.correct && second_supervisor.calls == 1 &&
            second_dispatch.failure_count() == 0;
    });
    second.join();
    CHECK(second_correct);
    CHECK(output == first_output);
    CHECK(context.bound() && context.guest_x18() == first_x18 &&
          context.thread_pointer() == 0x12345678abcdef00ull);
    CHECK(launch.thread_slot->active_bindings() == 1);
}

struct CompatibilityMemory {
    std::array<std::byte, 32> object{};
    std::int64_t period = 320000;
    const compat::InstalledPatch* installed = nullptr;
    unsigned callbacks = 0;
    std::thread::id thread = std::this_thread::get_id();
    static bool read(void* raw, std::uint64_t address, std::span<std::byte> bytes) noexcept {
        auto& self = *static_cast<CompatibilityMemory*>(raw);
        const auto object = reinterpret_cast<std::uintptr_t>(self.object.data());
        if (address >= object && address - object <= self.object.size() &&
            bytes.size() <= self.object.size() - (address - object)) {
            std::memcpy(bytes.data(), self.object.data() + address - object, bytes.size());
            return true;
        }
        const auto framework = reinterpret_cast<std::uintptr_t>(&self.period);
        if (address == self.installed->module_base + test::data_offset && bytes.size() == 8) {
            std::memcpy(bytes.data(), &framework, 8); return true;
        }
        if (address == framework && bytes.size() == 8) {
            std::memcpy(bytes.data(), &self.period, 8); return true;
        }
        return false;
    }
    static bool write(void* raw, std::uint64_t address, std::span<const std::byte> bytes) noexcept {
        auto& self = *static_cast<CompatibilityMemory*>(raw);
        const auto object = reinterpret_cast<std::uintptr_t>(self.object.data());
        if (address < object || address - object > self.object.size() ||
            bytes.size() > self.object.size() - (address - object)) return false;
        std::memcpy(self.object.data() + address - object, bytes.data(), bytes.size());
        return true;
    }
    static GateStatus handle(const CallHookSite&, GateCall& call, GuestThreadContext&, void* raw) {
        auto& self = *static_cast<CompatibilityMemory*>(raw);
        CHECK(self.thread == std::this_thread::get_id());
        ++self.callbacks;
        return compat::handle_patch(*self.installed, call, {read, write, &self});
    }
};

void rejects_unsupported_and_filtered_prepared_rates() {
    test::ModuleSpec module;
    module.name = "main";
    module.symbols = {{"_start", 0x100, 8, true}};
    test::put_word(module, 0x100, 0xd2800020u); // MOV X0, #1.
    test::put_word(module, 0x104, aarch64::encode_ret().word);
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", test::build_module(std::move(module))});
    auto profile = synthetic_profile();
    profile.title_id = 0x123;
    const auto artifacts = prepare_modules(input, preparation_options());
    CHECK(value(artifacts)); if (!value(artifacts)) return;
    const auto plan = plan_launch(input, profile, *value(artifacts));
    CHECK(value(plan)); if (!value(plan)) return;
    const auto image = loader::read_nso_module(input.modules[0].bytes);
    CHECK(image.ok()); if (!image.ok()) return;
    compat::PatchDescription patch;
    patch.identity = {profile.title_id, image.image->build_id};
    patch.frame_rate = compat::FrameRateSettings{30, 60, {30, 120}};
    patch.words = {{0x100, 0xd2800020u, 0xd2800060u, false, "Synthetic private mode.", 120}};
    const auto load = [&] {
        return load_prepared_launch(input, *value(artifacts), profile, *value(plan),
            std::make_shared<platform::ThreadSlotKey>(), {}, std::span(&patch, 1));
    };
    const auto check_word = [](const LaunchResult<PreparedLaunch>& result, std::uint32_t expected) {
        CHECK(value(result));
        if (const auto* loaded = value(result)) {
            const auto bytes = loaded->modules->mapping(0).readable_bytes(0x100, 4);
            CHECK(bytes.has_value());
            std::uint32_t word = 0;
            if (bytes) std::memcpy(&word, bytes->data(), 4);
            CHECK(word == expected);
        }
    };
    const auto check_refusal = [](const LaunchResult<PreparedLaunch>& result) {
        CHECK(failed(result, LaunchError::invalid_input));
        if (const auto* failure = std::get_if<LaunchFailure>(&result))
            CHECK(failure->operation == "compatibility-frame-rate");
    };
    const auto environment = [](const char* name) -> std::optional<std::string> {
        if (const char* setting = std::getenv(name)) return std::string(setting);
        return {};
    };
    const auto old_rate = environment("ARMIGO_COMPAT_FRAME_RATE");
    const auto old_driver = environment("ARMIGO_NVN");
    setenv("ARMIGO_NVN", "stream", 1);
    setenv("ARMIGO_COMPAT_FRAME_RATE", "60", 1);
    check_refusal(load());
    setenv("ARMIGO_COMPAT_FRAME_RATE", "120", 1);
    check_word(load(), 0xd2800060u);
    patch.words.front().nvn_driver_only = true;
    check_refusal(load());
    setenv("ARMIGO_COMPAT_FRAME_RATE", "30", 1);
    check_word(load(), 0xd2800020u);
    if (old_rate) setenv("ARMIGO_COMPAT_FRAME_RATE", old_rate->c_str(), 1);
    else unsetenv("ARMIGO_COMPAT_FRAME_RATE");
    if (old_driver) setenv("ARMIGO_NVN", old_driver->c_str(), 1);
    else unsetenv("ARMIGO_NVN");
}

void loads_identity_selected_call_hooks() {
    test::ModuleSpec module;
    module.name = "main";
    module.symbols = {{"_start", 0x100, 16, true}, {"callee", 0x200, 24, true},
                      {"second", 0x300, 16, true}};
    test::put_word(module, 0x100, 0xf81f0ffeu); // STR X30,[SP,#-16]!
    test::put_word(module, 0x104, aarch64::encode_bl(0x104, 0x200).word);
    test::put_word(module, 0x108, 0xf84107feu); // LDR X30,[SP],#16
    test::put_word(module, 0x10c, aarch64::encode_ret().word);
    test::put_word(module, 0x300, 0xf81f0ffeu); // STR X30,[SP,#-16]!
    test::put_word(module, 0x304, aarch64::encode_bl(0x304, 0x200).word);
    test::put_word(module, 0x308, 0xf84107feu); // LDR X30,[SP],#16
    test::put_word(module, 0x30c, aarch64::encode_ret().word);
    test::put_word(module, 0x200, 0xb9400402u); // LDR W2,[X0,#4]
    test::put_word(module, 0x204, 0xb9001402u); // STR W2,[X0,#20]
    test::put_word(module, 0x208, 0xb9401002u); // LDR W2,[X0,#16]
    test::put_word(module, 0x20c, 0x11000442u); // ADD W2,W2,#1
    test::put_word(module, 0x210, 0xb9001002u); // STR W2,[X0,#16]
    test::put_word(module, 0x214, aarch64::encode_ret().word);
    loader::ExeFsDirectoryInput input;
    input.modules.push_back({"main", test::build_module(std::move(module))});
    for (std::size_t i = 0; i < 32; ++i) input.modules[0].bytes[0x40 + i] = std::byte(i + 1);
    auto profile = synthetic_profile(); profile.title_id = 0x123;
    const auto artifacts = prepare_modules(input, preparation_options());
    if (const auto* error = std::get_if<LaunchFailure>(&artifacts))
        std::fprintf(stderr, "Compatibility preparation failed: %s %s\n", error->operation.c_str(), error->detail.c_str());
    CHECK(value(artifacts)); if (!value(artifacts)) return;
    const auto plan = plan_launch(input, profile, *value(artifacts));
    CHECK(value(plan)); if (!value(plan)) return;
    const auto image = loader::read_nso_module(input.modules[0].bytes);
    CHECK(image.ok()); if (!image.ok()) return;
    compat::PatchDescription patch;
    patch.identity = {profile.title_id, image.image->build_id};
    patch.module = "frame-rate";
    patch.original_callee = 0x200;
    patch.calls = {{0x104, aarch64::encode_bl(0x104, 0x200).word},
                   {0x304, aarch64::encode_bl(0x304, 0x200).word}};
    patch.layout = {test::data_offset, 0, 0, 4, 8, 12};
    auto key = std::make_shared<platform::ThreadSlotKey>();
    const auto load = [&](const compat::PatchDescription* selected) {
        return load_prepared_launch(input, *value(artifacts), profile, *value(plan), key, {},
            selected ? std::span{selected, 1} : std::span<const compat::PatchDescription>{});
    };
    auto unchanged = load(nullptr);
    CHECK(value(unchanged) && !value(unchanged)->call_hooks);
    auto mismatch = patch; ++mismatch.identity.title_id;
    auto unmatched = load(&mismatch);
    CHECK(value(unmatched) && !value(unmatched)->compatibility_patch);
    mismatch = patch; mismatch.identity.main_build_id.back() ^= std::byte{1};
    unmatched = load(&mismatch);
    CHECK(value(unmatched) && !value(unmatched)->call_hooks);
    mismatch = patch; mismatch.calls[1].expected_instruction ^= 1;
    CHECK(failed(load(&mismatch), LaunchError::artifact_mismatch));
    mismatch = patch; mismatch.calls[1].offset = 0x300;
    CHECK(failed(load(&mismatch), LaunchError::artifact_mismatch));
    const auto loaded = load(&patch);
    if (const auto* error = std::get_if<LaunchFailure>(&loaded))
        std::fprintf(stderr, "Compatibility load failed: %s %s\n", error->operation.c_str(), error->detail.c_str());
    CHECK(value(loaded)); if (!value(loaded)) return;
    const auto& launch = *value(loaded);
    CHECK(launch.call_hooks && launch.call_hooks->sites().size() == 2);
    CHECK(launch.compatibility_patch);
    if (!launch.call_hooks || !launch.compatibility_patch) return;
    CompatibilityMemory memory;
    memory.installed = launch.compatibility_patch.get();
    std::uint32_t base = 2;
    std::memcpy(memory.object.data() + 12, &base, 4);
    GuestThreadContext context;
    GuestDispatchScope dispatch(launch.sites, context, {}, GateFailurePolicy::trap,
                                launch.call_hooks, {CompatibilityMemory::handle, &memory});
    CHECK(dispatch.setup_error() == GuestDispatchFailureKind::none);
    platform::ThreadSlotBinding binding(*key, context.bound_register_cell());
    CHECK(binding.native_access());
    if (dispatch.setup_error() != GuestDispatchFailureKind::none || !binding.native_access()) return;
    using Entry = void (*)(void*, std::uint32_t);
    const auto second = launch.modules->entry_address(0, "second");
    CHECK(second); if (!second) return;
    reinterpret_cast<Entry>(launch.entry_address)(memory.object.data(), 2);
    reinterpret_cast<Entry>(*second)(memory.object.data(), 2);
    std::uint32_t calls = 0, interval = 0; float observed = 0;
    std::memcpy(&calls, memory.object.data() + 16, 4);
    std::memcpy(&observed, memory.object.data() + 20, 4);
    std::memcpy(&interval, memory.object.data() + 8, 4);
    CHECK(memory.callbacks == 2 && calls == 2 && observed == 0.5f && interval == 2);
    CHECK(graphics::presentation_scheduling_period(std::chrono::nanoseconds::zero()) ==
          std::chrono::nanoseconds(3'333'333));
    CHECK(dispatch.failure_count() == 0);
    memory.object[0] = std::byte{1}; memory.period = 640000;
    reinterpret_cast<Entry>(launch.entry_address)(memory.object.data(), 3);
    reinterpret_cast<Entry>(*second)(memory.object.data(), 3);
    std::memcpy(&observed, memory.object.data() + 20, 4);
    std::memcpy(&interval, memory.object.data() + 8, 4);
    std::memcpy(&calls, memory.object.data() + 16, 4);
    CHECK(observed == 0.5f && interval == 2 && memory.callbacks == 4 && calls == 4);
    CHECK(graphics::presentation_scheduling_period(std::chrono::nanoseconds::zero()) ==
          std::chrono::nanoseconds::zero());
}

} // namespace

int main() {
    loads_identity_selected_call_hooks();
    rejects_unsupported_and_filtered_prepared_rates();
    derives_dependency_order_and_rejects_unsupported_profiles();
    rejects_invalid_plans_before_publication();
    loads_rootless_companion_without_admitting_entries();
    plans_and_loads_original_process_entry();
    detached_metadata_and_ignored_orphans_preserve_manifest_indices();
    original_roster_ignores_dynamic_dependency_names();
    preparation_mode_is_bound_to_the_artifact();
    original_and_synthetic_preemption_select_distinct_providers();
    admits_complete_appendix_pages_without_new_entries();
    admits_complete_appendix_pages_without_new_entries(true);
    loads_and_executes_prepared_multimodule_code();
    if (failures != 0) return 1;
    std::puts("Launch planning and prepared multi-module native execution passed.");
    return 0;
}
