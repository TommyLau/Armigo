// ============================================================================
// Armigo NVN - Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "redirect.h"
#include "../runtime/preparation_fixture.h"
#include <armigo/runtime/launch_plan.h>
#include <armigo/runtime/preparation_format.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int armigo_nvn_guest_resolve_integer(std::uintptr_t loader, const char* name,
                                                  std::uint32_t property, int* value);

namespace {
using namespace armigo;
unsigned failures = 0;
#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #value); ++failures; } } while (false)
template<class T> const T* value(const runtime::LaunchResult<T>& result) { return std::get_if<T>(&result); }

void launch(bool title_enabled, const char* override_value, bool known, bool expected_driver,
            bool matching_identity = true, bool export_bootstrap = true, bool expected_failure = false) {
    using namespace runtime;
    test::ModuleSpec main;
    main.name = "main";
    main.symbols = {{"_start", 0x100, 4, true}};
    test::put_word(main, 0x100, aarch64::encode_ret().word);
    test::ModuleSpec sdk;
    sdk.name = "sdk";
    sdk.symbols = {{export_bootstrap ? "nvnBootstrapLoader" : "syntheticLoader", 0x100, 20, true}};
    for (std::uint64_t at = 0x100; at < 0x110; at += 4) test::put_word(sdk, at, 0xd503201f);
    test::put_word(sdk, 0x110, aarch64::encode_ret().word);
    const std::string_view marker = known ? "NintendoSDK_NVN-20_5_22-Release" : "NintendoSDK_NVN-00_00_00-Release";
    std::memcpy(sdk.rodata.data() + 0x2000, marker.data(), marker.size());
    loader::ExeFsDirectoryInput input;
    input.modules = {{"main", test::build_module(std::move(main))}, {"sdk", test::build_module(std::move(sdk))}};
    const auto source_main = input.modules[0].bytes;
    const auto source_sdk = input.modules[1].bytes;
    const auto original_main = loader::read_nso_module(input.modules.front().bytes);
    CHECK(original_main.ok());
    if (!original_main.ok()) return;
    LaunchProfile profile;
    profile.origin = ProfileOrigin::synthetic_test;
    profile.is_64_bit = true;
    profile.title_id = 1;
    compat::PatchDescription description;
    description.identity = {profile.title_id, original_main.image->build_id};
    description.nvn_driver = title_enabled;
    if (!matching_identity) ++description.identity.title_id;
    const PreparationOptions options{.required_preparation_identity = installed_preparation_identity(),
        .mode = PreparationMode::synthetic_test};
    const auto artifacts = prepare_modules(input, options);
    CHECK(value(artifacts));
    if (!value(artifacts)) return;
    const auto plan = plan_launch(input, profile, *value(artifacts));
    CHECK(value(plan));
    if (!value(plan)) return;
    if (override_value) setenv("ARMIGO_NVN", override_value, 1); else unsetenv("ARMIGO_NVN");
    const auto loaded = load_prepared_launch(input, *value(artifacts), profile, *value(plan),
        std::make_shared<platform::ThreadSlotKey>(), {}, std::span(&description, 1));
    if (!expected_failure) if (const auto* failure = std::get_if<LaunchFailure>(&loaded))
        std::fprintf(stderr, "%s: %s\n", failure->operation.c_str(), failure->detail.c_str());
    CHECK(input.modules[0].bytes == source_main && input.modules[1].bytes == source_sdk);
    if (expected_failure) {
        CHECK(std::holds_alternative<LaunchFailure>(loaded));
        return;
    }
    CHECK(value(loaded));
    if (!value(loaded)) return;
    const auto& modules = *value(loaded)->modules;
    std::size_t sdk_index = 0;
    while (sdk_index < modules.module_count() && modules.module_name(sdk_index) != "sdk") ++sdk_index;
    CHECK(sdk_index < modules.module_count());
    if (sdk_index == modules.module_count()) return;
    const auto entry = modules.entry_address(sdk_index, "nvnBootstrapLoader");
    CHECK(entry.has_value());
    if (!entry) return;
    std::uint32_t first_word = 0;
    std::memcpy(&first_word, reinterpret_cast<const void*>(*entry), 4);
    CHECK(first_word == (expected_driver ? 0x58000050u : 0xd503201fu));
    if (expected_driver) {
        using Loader = nvn::GenericFunction (*)(const char*);
        const auto function = reinterpret_cast<Loader>(*entry)("nvnDeviceGetInteger");
        CHECK(function != nullptr);
        int major_version = 0;
        CHECK(armigo_nvn_guest_resolve_integer(*entry, "nvnDeviceGetInteger", 0, &major_version) == 1);
        CHECK(major_version == 55);
    }
}
}
int main() {
    setenv("ARMIGO_NVN_COUNT", "1", 1);
    launch(false, nullptr, true, false);
    launch(true, nullptr, true, true);
    launch(false, "driver", true, true);
    launch(true, "stream", true, false);
    launch(true, "driver", false, false);
    launch(true, nullptr, true, false, false);
    launch(true, "invalid", true, false, true, true, true);
    launch(true, nullptr, true, false, true, false, true);
    unsetenv("ARMIGO_NVN");
    std::printf("NVN prepared launch failures: %u\n", failures);
    return failures ? 1 : 0;
}
