// ============================================================================
// Armigo Tests - NVN Entry Point Table
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "commands.h"
#include "entry_table.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

using namespace armigo::nvn;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

int device_integer_calls = 0;
void device_get_integer(const NVNdevice*, NVNdeviceInfo, int* out) {
    ++device_integer_calls;
    *out = 55;
}

} // namespace

int main() {
    // Nothing is handed out before an API version is selected.
    CHECK(armigo_nvn_bootstrap_loader("nvnDeviceGetInteger") == nullptr);
    select_api_version({55, 15});
    CHECK(find_entry("nvnDeviceGetInteger").has_value());
    CHECK(entry_name(*find_entry("nvnQueueSubmitCommands")) == "nvnQueueSubmitCommands");
    CHECK(!find_entry("nvnNotAnEntryPoint").has_value());
    CHECK(armigo_nvn_bootstrap_loader("nvnNotAnEntryPoint") == nullptr);

    // A known entry point without an implementation returns zero.
    const auto stub = reinterpret_cast<EntrySignature<Entry::nvnDeviceInitialize>::type>(
        armigo_nvn_bootstrap_loader("nvnDeviceInitialize"));
    CHECK(stub != nullptr && stub(nullptr, nullptr) == 0);

    install<Entry::nvnDeviceGetInteger>(&device_get_integer);
    const auto loaded = reinterpret_cast<void (*)(const NVNdevice*, NVNdeviceInfo, int*)>(
        armigo_nvn_bootstrap_loader("nvnDeviceGetInteger"));
    int value = 0;
    CHECK(loaded != nullptr);
    if (loaded) loaded(nullptr, NVNdeviceInfo{}, &value);
    CHECK(value == 55 && device_integer_calls == 1);
    CHECK(installed_entry(Entry::nvnDeviceGetInteger) == reinterpret_cast<GenericFunction>(&device_get_integer));

    // Driver package from a synthetic image with an SDK middleware marker.
    const std::string image = std::string("xx\0SDK MW+Vendor+NintendoSDK_NVN-20_5_22-Release\0yy", 52);
    const auto package = driver_package(std::as_bytes(std::span(image.data(), image.size())));
    CHECK(package && *package == "20_5_22");
    CHECK(known_api_version("20_5_22") == (ApiVersion{55, 15}));
    CHECK(known_api_version("13_3_2") == (ApiVersion{55, 12}));
    select_api_version({55, 12});
    CHECK(selected_api_version() == (ApiVersion{55, 12}));
    CHECK(armigo_nvn_bootstrap_loader("nvnDeviceGetInteger") != nullptr);
    select_api_version({55, 13});
    CHECK(!selected_api_version() && armigo_nvn_bootstrap_loader("nvnDeviceGetInteger") == nullptr);
    select_api_version({55, 15});
    CHECK(selected_api_version() == (ApiVersion{55, 15}));
    CHECK(!known_api_version("1_0_0").has_value());
    const std::string none = "no marker here";
    CHECK(!driver_package(std::as_bytes(std::span(none.data(), none.size()))).has_value());

    const auto marker = std::string("NintendoSDK_NVN-20_5_22-Release");
    const auto duplicate = marker + marker;
    CHECK(driver_package(std::as_bytes(std::span(duplicate.data(), duplicate.size()))) == "20_5_22");
    const auto conflicting = marker + "NintendoSDK_NVN-99_0_0-Release";
    CHECK(!driver_package(std::as_bytes(std::span(conflicting.data(), conflicting.size()))));

    // Handles keep the texture ID in 20 bits and the sampler ID in the next 12.
    const auto handle = combined_handle(0x12345, 0x678);
    CHECK(handle_texture_id(handle) == 0x12345 && handle_sampler_id(handle) == 0x678);
    std::printf("NVN entry table failures: %u\n", failures);
    return failures ? 1 : 0;
}
