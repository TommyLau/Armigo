// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Launch profile derivation from synthetic NPDM bytes. Two authored profiles
// must admit with different valid permissions, cores, priorities, and stack
// or resource settings; malformed sections, requests outside
// their ACID restrictions, unknown capability kinds, absent metadata, mapping
// or interrupt requests, and the authenticated trust policy must each fail
// with a structured result before any execution. No fixture derives from a
// real title's metadata.

#include <armigo/runtime/launch_profile.h>

#include "npdm.h"
#include "npdm_fixture.h"
#include "test_support.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <variant>
#include <vector>

namespace {

namespace fixture = npdm_fixture;

using armigo::runtime::LaunchError;
using armigo::runtime::LaunchFailure;
using armigo::runtime::LaunchProfile;
using armigo::runtime::LaunchResult;
using armigo::runtime::LaunchStage;
using armigo::runtime::NpdmTrustPolicy;
using fixture::Bytes;

LaunchProfile expect_success(const Bytes& image) {
    const LaunchResult<LaunchProfile> result = armigo::runtime::derive_launch_profile(
        image, NpdmTrustPolicy::structurally_validated_extracted_input);
    ARMIGO_CHECK(std::holds_alternative<LaunchProfile>(result));
    if (std::holds_alternative<LaunchFailure>(result)) {
        const LaunchFailure& failure = std::get<LaunchFailure>(result);
        std::fprintf(stderr, "unexpected failure: %s: %s\n", failure.operation.c_str(),
                     failure.detail.c_str());
    }
    return std::holds_alternative<LaunchProfile>(result)
               ? std::get<LaunchProfile>(result)
               : LaunchProfile{};
}

LaunchFailure expect_failure(const Bytes& image, LaunchError code) {
    const LaunchResult<LaunchProfile> result = armigo::runtime::derive_launch_profile(
        image, NpdmTrustPolicy::structurally_validated_extracted_input);
    ARMIGO_CHECK(std::holds_alternative<LaunchFailure>(result));
    if (!std::holds_alternative<LaunchFailure>(result)) {
        return LaunchFailure{};
    }
    const LaunchFailure& failure = std::get<LaunchFailure>(result);
    ARMIGO_CHECK(failure.stage == LaunchStage::profile);
    ARMIGO_CHECK(failure.code == code);
    ARMIGO_CHECK(!failure.operation.empty());
    ARMIGO_CHECK(!failure.detail.empty());
    return failure;
}

fixture::NpdmImage build(fixture::Parts parts) {
    return fixture::build_npdm(parts);
}

void test_two_profiles_admit_with_different_settings() {
    const LaunchProfile a = expect_success(fixture::profile_image_a().bytes);
    const LaunchProfile b = expect_success(fixture::profile_image_b().bytes);

    ARMIGO_CHECK(a.origin == armigo::runtime::ProfileOrigin::npdm);
    ARMIGO_CHECK(b.origin == armigo::runtime::ProfileOrigin::npdm);
    ARMIGO_CHECK(a.permissions.intended_kernel_version == fixture::cap_kernel_version(20, 5));
    ARMIGO_CHECK(a.permissions.program_type == 1);
    ARMIGO_CHECK(a.permissions.debug_flags == 0);
    ARMIGO_CHECK(a.title_id == fixture::kProgramId);
    ARMIGO_CHECK(b.title_id == fixture::kProgramId);
    ARMIGO_CHECK(a.is_64_bit);
    ARMIGO_CHECK(b.is_64_bit);

    // Admitted supervisor calls differ and are sorted and unique.
    const std::vector<std::uint16_t> expected_a_calls{0, 4, 5, 25};
    const std::vector<std::uint16_t> expected_b_calls{54};
    ARMIGO_CHECK(a.permissions.supervisor_calls == expected_a_calls);
    ARMIGO_CHECK(b.permissions.supervisor_calls == expected_b_calls);

    // Service ACL bytes transport the ACI0 request stream verbatim.
    const Bytes expected_a_sac = fixture::service_access_control(
        {{"fsp-srv", false}, {"acc:u0", false}});
    const Bytes expected_b_sac = fixture::service_access_control(
        {{"vi:m", false}, {"nvdrv", false}, {"lm", false}});
    ARMIGO_CHECK(a.permissions.service_access_control == expected_a_sac);
    ARMIGO_CHECK(b.permissions.service_access_control == expected_b_sac);

    // Effective file-system permissions are the ACI0 mask inside the ACID.
    ARMIGO_CHECK(a.permissions.file_system_access_control.size() == 0x1C);
    ARMIGO_CHECK(b.permissions.file_system_access_control.size() == 0x1C);
    ARMIGO_CHECK(a.permissions.file_system_access_control[4] == std::byte{1});
    ARMIGO_CHECK(a.permissions.file_system_access_control[5] == std::byte{0});
    ARMIGO_CHECK(b.permissions.file_system_access_control[4] == std::byte{9});
    ARMIGO_CHECK(b.permissions.file_system_access_control[5] == std::byte{0});

    // Handle table, priority bounds, cores, resources and stacks differ.
    ARMIGO_CHECK(a.permissions.handle_table_capacity == 512);
    ARMIGO_CHECK(b.permissions.handle_table_capacity == 1000);
    ARMIGO_CHECK(a.permissions.highest_priority == 28);
    ARMIGO_CHECK(a.permissions.lowest_priority == 59);
    ARMIGO_CHECK(b.permissions.highest_priority == 16);
    ARMIGO_CHECK(b.permissions.lowest_priority == 47);
    ARMIGO_CHECK(a.process.allowed_core_mask == 0b111);
    ARMIGO_CHECK(b.process.allowed_core_mask == 0b1110);
    ARMIGO_CHECK(a.process.private_system_resource_bytes == 0x200000);
    ARMIGO_CHECK(b.process.private_system_resource_bytes == 0x600000);
    ARMIGO_CHECK(a.main_thread.stack_size == 0x40000);
    ARMIGO_CHECK(b.main_thread.stack_size == 0x80000);
    ARMIGO_CHECK(a.main_thread.priority == 44);
    ARMIGO_CHECK(b.main_thread.priority == 20);
    ARMIGO_CHECK(a.main_thread.ideal_core == 1);
    ARMIGO_CHECK(b.main_thread.ideal_core == 3);

    // META flag bits steer only the fields they own.
    ARMIGO_CHECK(!a.process.alias_extra_enabled);
    ARMIGO_CHECK(b.process.alias_extra_enabled);
    ARMIGO_CHECK(a.process.address_bits == 39);
    ARMIGO_CHECK(b.process.address_bits == 39);
}

void test_authenticated_policy_is_unsupported() {
    const fixture::NpdmImage image = fixture::profile_image_a();
    const LaunchResult<LaunchProfile> result = armigo::runtime::derive_launch_profile(
        image.bytes, NpdmTrustPolicy::require_authenticated);
    ARMIGO_CHECK(std::holds_alternative<LaunchFailure>(result));
    const LaunchFailure& failure = std::get<LaunchFailure>(result);
    ARMIGO_CHECK(failure.stage == LaunchStage::profile);
    ARMIGO_CHECK(failure.code == LaunchError::unsupported);
    ARMIGO_CHECK(failure.operation == "npdm.trust_policy");
}

void test_structural_failures_surface_through_the_profile() {
    fixture::NpdmImage image = fixture::profile_image_a();
    fixture::put_u8(image.bytes, 0x00, 'X');
    expect_failure(image.bytes, LaunchError::invalid_input);

    fixture::NpdmImage overlap = fixture::profile_image_a();
    fixture::put_u32(overlap.bytes, fixture::kMetaAcidOffset, 0x90);
    expect_failure(overlap.bytes, LaunchError::invalid_input);
}

void test_acid_flag_admission() {
    fixture::Parts parts;
    parts.acid_flags = 0;
    expect_failure(build(parts).bytes, LaunchError::unsupported);
    for (std::uint32_t flags : {3u, 0x41u, 0x80000001u}) {
        parts.acid_flags = flags;
        expect_success(build(parts).bytes);
    }
    for (std::uint32_t pool : {1u, 2u, 3u, 4u, 15u}) {
        parts.acid_flags = 1 | (pool << 2);
        expect_failure(build(parts).bytes,
                       pool <= 3 ? LaunchError::unsupported : LaunchError::invalid_input);
    }
}

void test_program_id_must_sit_inside_the_acid_range() {
    fixture::Parts parts;
    parts.program_id = fixture::kProgramIdMax + 1;
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
    parts = fixture::Parts{};
    parts.program_id_min = fixture::kProgramIdMax;
    parts.program_id_max = fixture::kProgramIdMin;
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
}

void test_meta_flag_admission() {
    fixture::Parts parts;
    parts.meta_flags = 0x01; // 32-bit address space
    expect_failure(build(parts).bytes, LaunchError::unsupported);

    parts.meta_flags = 0x87; // Loader rejects bit seven before profile admission.
    expect_failure(build(parts).bytes, LaunchError::invalid_input);

    parts.meta_flags = 0x17; // optimize-memory-allocation is an accepted no-op
    expect_success(build(parts).bytes);
}

void test_size_admission() {
    fixture::Parts parts;
    for (std::uint32_t size : {0u, 0x200000u, 0x1FE00000u}) {
        parts.system_resource = size;
        ARMIGO_CHECK(expect_success(build(parts).bytes).process.private_system_resource_bytes == size);
    }
    for (std::uint32_t size : {0x1000u, 0x200001u, 0x20000000u}) {
        parts.system_resource = size;
        ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::invalid_input).operation ==
                     "npdm.system_resource");
    }
    parts = fixture::Parts{};
    for (std::uint32_t size : {0u, 0x40001u}) {
        parts.stack_size = size;
        expect_failure(build(parts).bytes, LaunchError::invalid_input);
    }
}

void test_file_system_access_restriction() {
    fixture::Parts parts;
    parts.aci0_fs_permissions = ~std::uint64_t{0};
    parts.acid_fs_permissions = ~std::uint64_t{0};
    const auto profile = expect_success(build(parts).bytes);
    ARMIGO_CHECK(profile.permissions.file_system_access_control[4] == std::byte{0xFF});
    ARMIGO_CHECK(profile.permissions.file_system_access_control[11] == std::byte{0x3F});
    parts.aci0_fs_permissions = 0xF;
    parts.acid_fs_permissions = 3;
    ARMIGO_CHECK(expect_success(build(parts).bytes).permissions.file_system_access_control[4] ==
                 std::byte{3});
}

void test_file_system_access_version_semantics() {
    for (std::uint8_t version : {0, 2}) {
        fixture::NpdmImage image = fixture::profile_image_a();
        fixture::put_u8(image.bytes, 0xC0, version);
        const auto profile = expect_success(image.bytes);
        for (std::size_t i = 4; i < 12; ++i) {
            ARMIGO_CHECK(profile.permissions.file_system_access_control[i] == std::byte{0});
        }
    }
    fixture::Parts parts;
    parts.fs_version = 255;
    ARMIGO_CHECK(expect_success(build(parts).bytes).permissions.file_system_access_control[4] ==
                 std::byte{1});
    auto image = build(parts);
    fixture::put_u8(image.bytes, 0xC1, 0xA5); // Reserved byte is not part of the version.
    expect_success(image.bytes);
}

void test_file_system_access_content_owners_are_unsupported() {
    for (std::size_t pair : {0xCu, 0x14u}) {
        fixture::NpdmImage image = fixture::profile_image_a();
        fixture::put_u32(image.bytes, 0xC0 + pair, 0);
        fixture::put_u32(image.bytes, 0xC0 + pair + 4, 1);
        ARMIGO_CHECK(expect_failure(image.bytes, LaunchError::unsupported).operation ==
                     "npdm.file_system_access");
        fixture::put_u32(image.bytes, 0xC0 + pair, 0x1C);
        expect_failure(image.bytes, LaunchError::invalid_input);
        fixture::put_u32(image.bytes, 0xC0 + pair + 4, 0);
        expect_success(image.bytes);
        fixture::put_u32(image.bytes, 0xC0 + pair, 0);
        expect_success(image.bytes);
        fixture::put_u32(image.bytes, 0xC0 + pair, 0xFFFFFFFFu);
        expect_failure(image.bytes, LaunchError::invalid_input);
    }
}

void test_file_system_access_owner_offset_bounds() {
    fixture::NpdmImage image = fixture::profile_image_a();
    fixture::put_u32(image.bytes, 0xC0 + 0x0C, 0x20);
    expect_failure(image.bytes, LaunchError::invalid_input);
}

void test_acid_file_system_header_minimum_size() {
    const auto parsed = armigo::loader::parse_npdm(fixture::profile_image_a().bytes);
    if (!std::holds_alternative<armigo::loader::ParsedNpdm>(parsed)) {
        ARMIGO_CHECK(false);
        return;
    }
    fixture::NpdmImage image = fixture::profile_image_a();
    const std::uint32_t acid_offset =
        std::get<armigo::loader::ParsedNpdm>(parsed).acid.offset;
    fixture::put_u32(image.bytes, acid_offset + 0x224, 0x1D);
    expect_failure(image.bytes, LaunchError::invalid_input);
}

void test_acid_file_system_owner_grants() {
    fixture::NpdmImage image = fixture::profile_image_a();
    const auto parsed = std::get<armigo::loader::ParsedNpdm>(armigo::loader::parse_npdm(image.bytes));
    fixture::put_u64(image.bytes, parsed.acid_fs_access_control.offset + 0xC, 1);
    expect_success(image.bytes); // Unrequested owner ranges are not process requests.
    fixture::put_u8(image.bytes, parsed.acid_fs_access_control.offset + 1, 1);
    expect_failure(image.bytes, LaunchError::invalid_input); // List bytes are missing.
    fixture::put_u32(image.bytes, parsed.acid.offset + 0x224, 0x34);
    expect_success(image.bytes); // Bounded owner grants require no requested-owner state.
}

void test_service_restriction() {
    fixture::Parts parts;
    parts.acid_services = {{"fsp-srv", false}}; // acc:u0 not granted
    expect_failure(build(parts).bytes, LaunchError::permission_denied);

    parts.acid_services = {{"fsp*", false}, {"acc*", false}}; // wildcard grants
    expect_success(build(parts).bytes);
}

void test_service_host_flag_must_match() {
    fixture::Parts parts;
    parts.aci0_services = {{"fsp-srv", true}};
    parts.acid_services = {{"fsp-srv", false}};
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
}

void test_service_entry_truncation() {
    fixture::Parts parts;
    parts.aci0_services = {{"a", false}, {"b", false}};
    parts.acid_services = {{"a", false}};
    fixture::NpdmImage image = build(parts);
    const auto parsed = std::get<armigo::loader::ParsedNpdm>(armigo::loader::parse_npdm(image.bytes));
    fixture::put_u8(image.bytes, parsed.aci0_service_access_control.offset + 2, 7);
    expect_success(image.bytes); // The incomplete second request is ignored.
    fixture::put_u32(image.bytes, parsed.acid.offset + 0x22C, 3);
    fixture::put_u8(image.bytes, parsed.acid_service_access_control.offset + 2, 7);
    expect_success(image.bytes); // The incomplete restriction tail is also ignored.
}

void test_capability_restriction_requests() {
    fixture::Parts parts;
    parts.aci0_capabilities[1] = fixture::cap_syscall_mask(0, 0x30); // ACID grants 0x31.
    expect_success(build(parts).bytes);

    parts = fixture::Parts{};
    parts.acid_capabilities[0] = fixture::cap_core_priority(59, 28, 0, 1); // request is 0..2
    expect_failure(build(parts).bytes, LaunchError::permission_denied);

    parts = fixture::Parts{};
    parts.aci0_capabilities[0] = fixture::cap_core_priority(59, 27, 0, 2);
    expect_failure(build(parts).bytes, LaunchError::permission_denied);

    parts = fixture::Parts{};
    parts.acid_capabilities[3] = fixture::cap_handle_table(256); // request is 512
    expect_failure(build(parts).bytes, LaunchError::permission_denied);

    parts = fixture::Parts{};
    parts.aci0_capabilities[4] = fixture::cap_kernel_version(20, 4);
    expect_failure(build(parts).bytes, LaunchError::permission_denied);

    parts = fixture::Parts{};
    parts.acid_capabilities[5] = fixture::cap_program_type(2);
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
}

void test_debug_flag_restriction() {
    fixture::Parts parts;
    parts.aci0_capabilities.push_back(fixture::cap_debug_flags(true, false, false));
    parts.acid_capabilities.push_back(fixture::cap_debug_flags(true, false, false));
    ARMIGO_CHECK(expect_success(build(parts).bytes).permissions.debug_flags == 1);

    parts.acid_capabilities.back() = fixture::cap_debug_flags(false, false, false);
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
}

void test_unknown_capability_kinds() {
    fixture::Parts parts;
    parts.aci0_capabilities.push_back(0x00000001u);
    expect_failure(build(parts).bytes, LaunchError::invalid_input);

    parts.aci0_capabilities.back() = 0x00000000u;
    expect_failure(build(parts).bytes, LaunchError::invalid_input);
}

void test_duplicate_initialize_once_capabilities() {
    for (std::size_t index : {0u, 1u, 3u, 4u, 6u}) {
        fixture::Parts parts;
        parts.aci0_capabilities.push_back(parts.aci0_capabilities[index]);
        ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::invalid_input).operation ==
                     "npdm.kernel_access_control");
    }
    fixture::Parts parts;
    const auto debug = fixture::cap_debug_flags(true, false, false);
    parts.aci0_capabilities.insert(parts.aci0_capabilities.end(), {debug, debug});
    parts.acid_capabilities.push_back(debug);
    ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::invalid_input).operation ==
                 "npdm.kernel_access_control");
}

void test_invalid_capability_fields() {
    fixture::Parts parts;
    parts.aci0_capabilities[0] = fixture::cap_core_priority(59, 28, 2, 1);
    ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::permission_denied).operation ==
                 "npdm.capability_restriction");
    // Valid descriptor identities and matching ACID grants reach kernel checks.
    for (const auto [index, malformed] : {
             std::pair<std::size_t, std::uint32_t>{3, fixture::cap_handle_table(512) | (1u << 26)},
             {4, fixture::cap_kernel_version(0, 5)},
             {6, fixture::cap_program_type(1) | (1u << 17)}}) {
        parts = fixture::Parts{};
        parts.aci0_capabilities[index] = malformed;
        parts.acid_capabilities[index == 6 ? 5 : index] = malformed;
        ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::invalid_input).operation ==
                     "npdm.kernel_access_control");
    }
    for (const auto malformed : {fixture::cap_debug_flags(true, true, false),
                                fixture::cap_debug_flags(true, false, false) | (1u << 20),
                                fixture::cap_map_region(4, 0, 0)}) {
        parts = fixture::Parts{};
        parts.aci0_capabilities.push_back(malformed);
        parts.acid_capabilities.push_back(malformed);
        ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::invalid_input).operation ==
                     "npdm.kernel_access_control");
    }
}

void test_mapping_requests_are_unsupported() {
    for (const auto descriptor : {fixture::cap_map_io_page(0x100),
                                 fixture::cap_map_region(1, 0, 0),
                                 fixture::cap_interrupt_pair(0x30, 0x3FF)}) {
        fixture::Parts parts;
        parts.aci0_capabilities.push_back(descriptor);
        expect_failure(build(parts).bytes, LaunchError::permission_denied);
        parts.acid_capabilities.push_back(descriptor);
        ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::unsupported).operation ==
                     "npdm.capability_support");
    }
    fixture::Parts parts;
    parts.aci0_capabilities.push_back(fixture::cap_map_range_address(0x40, false));
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
    parts.aci0_capabilities.push_back(fixture::cap_map_range_size(4, true));
    parts.acid_capabilities.push_back(fixture::cap_map_range_address(0x20, false));
    parts.acid_capabilities.push_back(fixture::cap_map_range_size(0x40, true));
    expect_failure(build(parts).bytes, LaunchError::unsupported);
}

void test_idle_mapping_and_interrupt_descriptors_admit() {
    fixture::Parts parts;
    parts.aci0_capabilities.push_back(fixture::cap_map_region(0, 0, 0));
    parts.aci0_capabilities.push_back(fixture::cap_interrupt_pair(0x3FF, 0x3FF));
    expect_success(build(parts).bytes);
}

void test_absent_metadata_fails() {
    fixture::Parts parts;
    parts.aci0_capabilities = {
        fixture::cap_syscall_mask(0, 0x31),
        fixture::cap_handle_table(512),
        fixture::cap_kernel_version(20, 5),
    };
    expect_failure(build(parts).bytes, LaunchError::invalid_input);

    parts.aci0_capabilities = {
        fixture::cap_core_priority(59, 28, 0, 2),
        fixture::cap_kernel_version(20, 5),
    };
    expect_failure(build(parts).bytes, LaunchError::invalid_input);

    parts.aci0_capabilities = {
        fixture::cap_core_priority(59, 28, 0, 2),
        fixture::cap_handle_table(512),
    };
    expect_failure(build(parts).bytes, LaunchError::invalid_input);

    parts.aci0_capabilities = {};
    expect_failure(build(parts).bytes, LaunchError::invalid_input);
}

void test_non_application_program_type_is_unsupported() {
    fixture::Parts parts;
    parts.aci0_capabilities[6] = fixture::cap_program_type(2);
    parts.acid_capabilities[5] = fixture::cap_program_type(2);
    expect_failure(build(parts).bytes, LaunchError::unsupported);
    parts.aci0_capabilities.pop_back();
    expect_failure(build(parts).bytes, LaunchError::unsupported);
}

void test_main_thread_must_sit_inside_admitted_ranges() {
    fixture::Parts parts;
    parts.main_priority = 10;
    expect_failure(build(parts).bytes, LaunchError::invalid_input);

    parts = fixture::Parts{};
    parts.main_priority = 60;
    expect_failure(build(parts).bytes, LaunchError::invalid_input);

    parts = fixture::Parts{};
    parts.main_core = 3;
    expect_failure(build(parts).bytes, LaunchError::invalid_input);
}

void test_priority_and_virtual_core_boundaries() {
    for (std::uint32_t priority : {3u, 4u, 15u, 63u}) {
        fixture::Parts parts;
        parts.main_priority = static_cast<std::uint8_t>(priority);
        parts.aci0_capabilities[0] = fixture::cap_core_priority(63, priority, 0, 2);
        parts.acid_capabilities[0] = fixture::cap_core_priority(63, 0, 0, 63);
        if (priority == 3) {
            ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::invalid_input).operation ==
                         "npdm.kernel_access_control");
        } else {
            ARMIGO_CHECK(expect_success(build(parts).bytes).permissions.highest_priority == priority);
        }
    }
    for (std::uint32_t core : {3u, 4u, 63u, 64u}) {
        fixture::Parts parts;
        parts.aci0_capabilities[0] = fixture::cap_core_priority(59, 28, 0, core);
        parts.acid_capabilities[0] = fixture::cap_core_priority(59, 28, 0, 255);
        if (core == 3) {
            ARMIGO_CHECK(expect_success(build(parts).bytes).process.allowed_core_mask == 15);
        } else {
            const auto failure = expect_failure(build(parts).bytes,
                core < 64 ? LaunchError::unsupported : LaunchError::invalid_input);
            ARMIGO_CHECK(failure.operation ==
                         (core < 64 ? "npdm.capability_support" : "npdm.kernel_access_control"));
        }
    }
}

void test_raw_restriction_catalogue() {
    fixture::Parts parts;
    parts.acid_capabilities.insert(parts.acid_capabilities.end(),
        {0, 1, fixture::cap_padding(), fixture::cap_core_priority(1, 60, 255, 0),
         fixture::cap_handle_table(0)});
    expect_success(build(parts).bytes); // Unused entries and later duplicates are not decoded.
    parts.acid_capabilities.insert(parts.acid_capabilities.begin(), fixture::cap_handle_table(1));
    expect_failure(build(parts).bytes, LaunchError::permission_denied); // First singleton wins.

    parts = fixture::Parts{};
    parts.acid_capabilities[1] = fixture::cap_syscall_mask(0, 0x10);
    parts.acid_capabilities.push_back(fixture::cap_syscall_mask(0, 0x21));
    expect_failure(build(parts).bytes, LaunchError::permission_denied); // Separate grants do not OR.
    parts.acid_capabilities.push_back(fixture::cap_syscall_mask(0, 0xF1));
    expect_success(build(parts).bytes); // Any single covering grant suffices.

    parts = fixture::Parts{};
    parts.acid_capabilities[1] = fixture::cap_syscall_mask(2, 0xFFFFFF);
    parts.aci0_capabilities[1] = fixture::cap_syscall_mask(0, 0);
    expect_failure(build(parts).bytes, LaunchError::permission_denied); // Even zero needs its slot.
    parts.acid_capabilities.push_back(fixture::cap_syscall_mask(0, 0));
    expect_success(build(parts).bytes);

    parts = fixture::Parts{};
    parts.aci0_capabilities.push_back(fixture::cap_debug_flags(false, true, false));
    parts.acid_capabilities.push_back(fixture::cap_debug_flags(true, true, true));
    ARMIGO_CHECK(expect_success(build(parts).bytes).permissions.debug_flags == 2);
}

void test_restrictions_precede_kernel_and_service_validation() {
    fixture::Parts parts;
    parts.aci0_capabilities[3] = fixture::cap_handle_table(900) | (1u << 26);
    parts.acid_capabilities[3] = fixture::cap_handle_table(512);
    parts.acid_services.clear();
    ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::permission_denied).operation ==
                 "npdm.capability_restriction");
    parts.acid_capabilities[3] = fixture::cap_handle_table(1000);
    ARMIGO_CHECK(expect_failure(build(parts).bytes, LaunchError::invalid_input).operation ==
                 "npdm.kernel_access_control");
}

void test_mapping_restriction_boundaries() {
    for (std::uint32_t pages : {0u, 1u, 2u}) {
        fixture::Parts parts;
        const auto address = fixture::cap_map_range_address(0xFFFFFF, false);
        const auto size = fixture::cap_map_range_size(pages, true);
        parts.aci0_capabilities.insert(parts.aci0_capabilities.end(), {address, size});
        parts.acid_capabilities.insert(parts.acid_capabilities.end(), {address, size});
        const auto failure = expect_failure(build(parts).bytes,
            pages == 1 ? LaunchError::unsupported : LaunchError::invalid_input);
        ARMIGO_CHECK(failure.operation ==
                     (pages == 1 ? "npdm.capability_support" : "npdm.kernel_access_control"));
    }
    fixture::Parts parts;
    parts.aci0_capabilities.insert(parts.aci0_capabilities.end(),
        {fixture::cap_map_range_address(0x40, false), fixture::cap_map_range_size(4, true)});
    parts.acid_capabilities.insert(parts.acid_capabilities.end(),
        {fixture::cap_map_range_address(0x40, true), fixture::cap_map_range_size(4, true)});
    expect_failure(build(parts).bytes, LaunchError::permission_denied); // Address flags must agree.
    parts.acid_capabilities[parts.acid_capabilities.size() - 2] =
        fixture::cap_map_range_address(0x40, false);
    parts.acid_capabilities.back() = fixture::cap_map_range_size(4, false);
    expect_failure(build(parts).bytes, LaunchError::permission_denied); // Size flags must agree.

    parts = fixture::Parts{};
    const auto rw = fixture::cap_map_region(1, 0, 0);
    const auto ro = rw | (1u << 17);
    parts.aci0_capabilities.push_back(ro);
    parts.acid_capabilities.push_back(rw);
    expect_failure(build(parts).bytes, LaunchError::unsupported);
    parts.aci0_capabilities.back() = rw;
    parts.acid_capabilities.back() = ro;
    expect_failure(build(parts).bytes, LaunchError::permission_denied);

    for (std::uint32_t interrupt : {1019u, 1020u, 1022u}) {
        parts = fixture::Parts{};
        parts.aci0_capabilities.push_back(fixture::cap_interrupt_pair(interrupt, 0x3FF));
        parts.acid_capabilities.push_back(fixture::cap_interrupt_pair(0x3FF, 0x3FF));
        expect_failure(build(parts).bytes,
                       interrupt == 1019 ? LaunchError::unsupported : LaunchError::invalid_input);
    }
}

void test_mapping_catalogue_pair_order() {
    fixture::Parts parts;
    const auto address = fixture::cap_map_range_address(0x40, false);
    const auto size = fixture::cap_map_range_size(4, true);
    parts.aci0_capabilities.insert(parts.aci0_capabilities.end(), {address, size});
    parts.acid_capabilities.insert(parts.acid_capabilities.end(),
                                  {address, fixture::cap_padding(), address, size});
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
    parts.acid_capabilities.erase(parts.acid_capabilities.end() - 4,
                                 parts.acid_capabilities.end() - 2);
    parts.acid_capabilities.insert(parts.acid_capabilities.end(),
                                  {address, fixture::cap_padding()});
    expect_failure(build(parts).bytes, LaunchError::unsupported);
}

void test_partial_capability_word_is_not_a_descriptor() {
    auto image = fixture::profile_image_a();
    const auto parsed = std::get<armigo::loader::ParsedNpdm>(armigo::loader::parse_npdm(image.bytes));
    // ACI0 may overlap the next parent by one byte; only complete words count.
    fixture::put_u32(image.bytes, fixture::kMetaAci0Size, parsed.aci0.size + 1);
    fixture::put_u32(image.bytes, parsed.aci0.offset + 0x34,
                     parsed.aci0_kernel_access_control.size + 1);
    image.bytes.push_back(std::byte{0});
    fixture::put_u32(image.bytes, fixture::kMetaAcidSize, parsed.acid.size + 1);
    fixture::put_u32(image.bytes, parsed.acid.offset + 0x204, parsed.acid.size + 1 - 0x100);
    fixture::put_u32(image.bytes, parsed.acid.offset + 0x234,
                     parsed.acid_kernel_access_control.size + 1);
    expect_success(image.bytes);
}

void test_service_request_size_limit() {
    fixture::Parts parts;
    parts.aci0_services.assign(256, {"a", false});
    parts.acid_services = {{"a", false}};
    auto image = build(parts);
    expect_success(image.bytes); // Exactly 0x200 bytes are stored by SM.
    const auto parsed = std::get<armigo::loader::ParsedNpdm>(armigo::loader::parse_npdm(image.bytes));
    fixture::put_u32(image.bytes, parsed.aci0.offset + 0x2C, 0x201);
    ARMIGO_CHECK(expect_failure(image.bytes, LaunchError::invalid_input).operation ==
                 "npdm.service_access_control");
}

void test_service_name_matching() {
    fixture::Parts parts;
    parts.acid_services = {{"*", false}};
    expect_success(build(parts).bytes);
    parts.aci0_services = {{"ab*", false}};
    parts.acid_services = {{"a*", false}};
    expect_failure(build(parts).bytes, LaunchError::permission_denied);
    parts.acid_services = parts.aci0_services;
    expect_success(build(parts).bytes);

    parts.aci0_services = {{"a", false}};
    parts.acid_services = {{"abc", false}};
    auto image = build(parts);
    const auto parsed = std::get<armigo::loader::ParsedNpdm>(armigo::loader::parse_npdm(image.bytes));
    const auto request = parsed.aci0_service_access_control.offset;
    const auto restriction = parsed.acid_service_access_control.offset;
    fixture::put_u8(image.bytes, request, 0x78); // Control bits 3..6 do not change the entry.
    fixture::put_u8(image.bytes, restriction + 2, 0);
    fixture::put_u8(image.bytes, restriction + 3, '*');
    expect_success(image.bytes); // ['a', 0, '*'] matches the zero-padded request ['a'].
    fixture::put_u8(image.bytes, restriction, 1);
    fixture::put_u32(image.bytes, parsed.acid.offset + 0x22C, 3);
    expect_success(image.bytes); // ['a', 0] and ['a'] are the same padded name.
    fixture::put_u8(image.bytes, request + 1, 0xFF);
    fixture::put_u8(image.bytes, restriction + 1, 0xFF);
    expect_success(image.bytes); // Name bytes are not restricted to printable ASCII.
}

void test_fs_version_short_circuit_and_header_bounds() {
    auto image = fixture::profile_image_a();
    const auto parsed = std::get<armigo::loader::ParsedNpdm>(armigo::loader::parse_npdm(image.bytes));
    fixture::put_u8(image.bytes, 0xC0, 0);
    fixture::put_u32(image.bytes, 0xCC, 0xFFFFFFFFu);
    fixture::put_u8(image.bytes, parsed.acid_fs_access_control.offset + 1, 255);
    expect_success(image.bytes); // Empty version access precedes owner validation.
    fixture::put_u32(image.bytes, 0x80 + 0x24, 0x1B);
    expect_failure(image.bytes, LaunchError::invalid_input); // Fixed headers still must fit.
}

} // namespace

int main() {
    test_two_profiles_admit_with_different_settings();
    test_authenticated_policy_is_unsupported();
    test_structural_failures_surface_through_the_profile();
    test_acid_flag_admission();
    test_program_id_must_sit_inside_the_acid_range();
    test_meta_flag_admission();
    test_size_admission();
    test_file_system_access_restriction();
    test_file_system_access_version_semantics();
    test_file_system_access_content_owners_are_unsupported();
    test_file_system_access_owner_offset_bounds();
    test_acid_file_system_header_minimum_size();
    test_acid_file_system_owner_grants();
    test_service_restriction();
    test_service_host_flag_must_match();
    test_service_entry_truncation();
    test_capability_restriction_requests();
    test_debug_flag_restriction();
    test_unknown_capability_kinds();
    test_duplicate_initialize_once_capabilities();
    test_invalid_capability_fields();
    test_mapping_requests_are_unsupported();
    test_idle_mapping_and_interrupt_descriptors_admit();
    test_absent_metadata_fails();
    test_non_application_program_type_is_unsupported();
    test_main_thread_must_sit_inside_admitted_ranges();
    test_priority_and_virtual_core_boundaries();
    test_raw_restriction_catalogue();
    test_restrictions_precede_kernel_and_service_validation();
    test_mapping_restriction_boundaries();
    test_service_name_matching();
    test_service_request_size_limit();
    test_mapping_catalogue_pair_order();
    test_partial_capability_word_is_not_a_descriptor();
    test_fs_version_short_circuit_and_header_bounds();
    return armigo::loader::test::finish("runtime.launch_profile");
}
