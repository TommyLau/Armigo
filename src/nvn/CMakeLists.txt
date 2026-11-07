# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# NVN graphics driver (fast path). Subsystems add their sources here.
add_library(armigo_nvn STATIC entry_table.cpp null_device.cpp abi.S redirect.cpp guest_wait.cpp
    device.cpp memory.cpp textures.cpp samplers.cpp programs.cpp state_objects.cpp command_buffers.cpp)
target_include_directories(armigo_nvn PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(armigo_nvn PUBLIC cxx_std_20)
target_link_libraries(armigo_nvn PUBLIC armigo_horizon armigo_graphics armigo_execution)
target_link_libraries(armigo_runtime PRIVATE armigo_nvn)

add_library(armigo_metal_nvn_resources STATIC
    "${PROJECT_SOURCE_DIR}/src/platform/metal/nvn_resources.mm"
    "${PROJECT_SOURCE_DIR}/src/platform/metal/nvn_deswizzle.mm")
target_include_directories(armigo_metal_nvn_resources PUBLIC "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src" PRIVATE "${PROJECT_SOURCE_DIR}/src/platform/metal")
target_link_libraries(armigo_metal_nvn_resources PUBLIC armigo_nvn armigo_metal_texture_format
    "-framework Metal" "-framework Foundation")

if(BUILD_TESTING)
    add_executable(armigo_nvn_timeline_tests "${PROJECT_SOURCE_DIR}/tests/nvn/timeline_tests.cpp" timeline.cpp)
    target_include_directories(armigo_nvn_timeline_tests PRIVATE "${PROJECT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_features(armigo_nvn_timeline_tests PRIVATE cxx_std_20)
    target_link_libraries(armigo_nvn_timeline_tests PRIVATE Threads::Threads)
    add_test(NAME nvn.timeline COMMAND armigo_nvn_timeline_tests)
    set_tests_properties(nvn.timeline PROPERTIES TIMEOUT 30)
    foreach(test_name IN ITEMS abi counting null_device)
        add_executable(armigo_nvn_${test_name}_tests "${PROJECT_SOURCE_DIR}/tests/nvn/${test_name}_tests.cpp")
        target_include_directories(armigo_nvn_${test_name}_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
        target_link_libraries(armigo_nvn_${test_name}_tests PRIVATE armigo_nvn)
        add_test(NAME nvn.${test_name} COMMAND armigo_nvn_${test_name}_tests)
    endforeach()
    add_executable(armigo_nvn_launch_tests "${PROJECT_SOURCE_DIR}/tests/nvn/launch_tests.cpp")
    target_include_directories(armigo_nvn_launch_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn" "${PROJECT_SOURCE_DIR}/src/loader")
    target_link_libraries(armigo_nvn_launch_tests PRIVATE armigo_runtime armigo_nvn)
    add_test(NAME nvn.launch COMMAND armigo_nvn_launch_tests)
    add_executable(armigo_nvn_guest_wait_tests "${PROJECT_SOURCE_DIR}/tests/nvn/guest_wait_tests.cpp")
    target_include_directories(armigo_nvn_guest_wait_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
    target_link_libraries(armigo_nvn_guest_wait_tests PRIVATE armigo_nvn)
    add_test(NAME nvn.guest_wait COMMAND armigo_nvn_guest_wait_tests)
    set_tests_properties(nvn.guest_wait PROPERTIES TIMEOUT 10)
    add_executable(armigo_nvn_compatibility_tests "${PROJECT_SOURCE_DIR}/tests/nvn/compatibility_tests.cpp")
    target_link_libraries(armigo_nvn_compatibility_tests PRIVATE armigo_compat)
    add_test(NAME nvn.compatibility COMMAND armigo_nvn_compatibility_tests)
    add_executable(armigo_nvn_redirect_tests "${PROJECT_SOURCE_DIR}/tests/nvn/redirect_tests.cpp")
    target_include_directories(armigo_nvn_redirect_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
    target_link_libraries(armigo_nvn_redirect_tests PRIVATE armigo_nvn)
    add_test(NAME nvn.redirect COMMAND armigo_nvn_redirect_tests)
    add_executable(armigo_nvn_flat_map_tests "${PROJECT_SOURCE_DIR}/tests/nvn/flat_map_tests.cpp")
    target_include_directories(armigo_nvn_flat_map_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
    add_test(NAME nvn.flat_map COMMAND armigo_nvn_flat_map_tests)
    add_executable(armigo_nvn_entry_table_tests "${PROJECT_SOURCE_DIR}/tests/nvn/entry_table_tests.cpp")
    target_include_directories(armigo_nvn_entry_table_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
    target_link_libraries(armigo_nvn_entry_table_tests PRIVATE armigo_nvn)
    add_test(NAME nvn.entry_table COMMAND armigo_nvn_entry_table_tests)
    foreach(kind memory textures samplers)
        add_executable(armigo_nvn_resources_${kind}_tests
            "${PROJECT_SOURCE_DIR}/tests/nvn/resources_${kind}.cpp")
        target_include_directories(armigo_nvn_resources_${kind}_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
        target_link_libraries(armigo_nvn_resources_${kind}_tests PRIVATE armigo_nvn)
        add_test(NAME nvn.resources_${kind} COMMAND armigo_nvn_resources_${kind}_tests)
    endforeach()
    add_executable(armigo_metal_nvn_resources_tests "${PROJECT_SOURCE_DIR}/tests/metal/nvn_resources.mm")
    target_include_directories(armigo_metal_nvn_resources_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(armigo_metal_nvn_resources_tests PRIVATE armigo_metal_nvn_resources)
    add_test(NAME metal.nvn_resources COMMAND armigo_metal_nvn_resources_tests)
    set_tests_properties(metal.nvn_resources PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
endif()

target_sources(armigo_nvn PRIVATE queues.cpp sync.cpp windows.cpp timeline.cpp)
target_link_libraries(armigo_nvn PUBLIC armigo_services_vi)

add_library(armigo_nvn_present_metal STATIC ../platform/metal/nvn_present.mm)
target_link_libraries(armigo_nvn_present_metal PUBLIC armigo_nvn
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

if(BUILD_TESTING)
    foreach(suite sync window)
        add_executable(armigo_nvn_present_${suite}_tests "${PROJECT_SOURCE_DIR}/tests/nvn/present_${suite}_tests.cpp")
        target_include_directories(armigo_nvn_present_${suite}_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
        target_link_libraries(armigo_nvn_present_${suite}_tests PRIVATE armigo_nvn)
        add_test(NAME nvn.present_${suite} COMMAND armigo_nvn_present_${suite}_tests)
    endforeach()
endif()

if(BUILD_TESTING AND ARMIGO_ENABLE_GPU_TESTS)
    add_executable(armigo_nvn_present_metal_tests "${PROJECT_SOURCE_DIR}/tests/nvn/present_metal_tests.mm")
    target_link_libraries(armigo_nvn_present_metal_tests PRIVATE armigo_nvn_present_metal armigo_platform_presenter
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME nvn.present_metal COMMAND armigo_nvn_present_metal_tests)
    set_tests_properties(nvn.present_metal PROPERTIES SKIP_RETURN_CODE 77)
endif()

if(BUILD_TESTING)
    add_executable(armigo_nvn_replay_program_tests "${PROJECT_SOURCE_DIR}/tests/nvn/replay_program_tests.cpp")
    target_include_directories(armigo_nvn_replay_program_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
    target_link_libraries(armigo_nvn_replay_program_tests PRIVATE armigo_nvn)
    add_test(NAME nvn.replay_programs COMMAND armigo_nvn_replay_program_tests)
endif()

add_library(armigo_nvn_metal STATIC "${PROJECT_SOURCE_DIR}/src/platform/metal/nvn_replay_programs.mm")
target_include_directories(armigo_nvn_metal PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn" "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(armigo_nvn_metal PUBLIC armigo_nvn armigo_metal_shader_package armigo_shader_input
    armigo_shader_translation)

if(BUILD_TESTING)
    target_link_libraries(armigo_metal_shader_package_tests PRIVATE armigo_nvn_metal)
endif()

if(BUILD_TESTING)
    add_executable(armigo_nvn_replay_state_tests "${PROJECT_SOURCE_DIR}/tests/nvn/replay_state_tests.cpp")
    target_include_directories(armigo_nvn_replay_state_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
    target_link_libraries(armigo_nvn_replay_state_tests PRIVATE armigo_nvn)
    add_test(NAME nvn.replay_state COMMAND armigo_nvn_replay_state_tests)
endif()

if(BUILD_TESTING)
    add_executable(armigo_nvn_replay_commands_tests "${PROJECT_SOURCE_DIR}/tests/nvn/replay_commands_tests.cpp")
    target_include_directories(armigo_nvn_replay_commands_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/nvn")
    target_link_libraries(armigo_nvn_replay_commands_tests PRIVATE armigo_nvn)
    add_test(NAME nvn.replay_commands COMMAND armigo_nvn_replay_commands_tests)
endif()

target_sources(armigo_nvn_metal PRIVATE "${PROJECT_SOURCE_DIR}/src/platform/metal/nvn_replay.mm"
    "${PROJECT_SOURCE_DIR}/src/platform/metal/nvn_install.mm"
    "${PROJECT_SOURCE_DIR}/src/platform/metal/pso_manifest.cpp"
    "${PROJECT_SOURCE_DIR}/src/platform/metal/pso_archive_build.mm")
target_link_libraries(armigo_nvn_metal PUBLIC armigo_metal_clear_encoder armigo_metal_texture_format armigo_metal_nvn_resources
    armigo_nvn_present_metal armigo_services_vi)
if(BUILD_TESTING)
    add_executable(armigo_metal_nvn_replay_tests "${PROJECT_SOURCE_DIR}/tests/metal/nvn_replay_tests.mm")
    target_include_directories(armigo_metal_nvn_replay_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/nvn" "${PROJECT_SOURCE_DIR}/src/platform/metal")
    target_link_libraries(armigo_metal_nvn_replay_tests PRIVATE armigo_nvn_metal)
    # Keep the unbatched path covered alongside default and explicit-on variants.
    add_test(NAME nvn.metal_replay COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_BATCH_FINALIZED_QUERIES=0")
    # The same cases with the worker making its Metal calls itself.
    add_test(NAME nvn.metal_replay_worker_encoding COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_worker_encoding PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_ENCODE_THREAD=0;ARMIGO_NVN_BATCH_FINALIZED_QUERIES=0")
    add_test(NAME nvn.metal_replay_conditional_contract COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_conditional_contract PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_CONDITIONAL_CONTRACT_TESTS=1")
    # The same suites under Metal API validation: every encoder barrier must
    # use stages its encoder admits (a render encoder waits only on vertex work).
    add_test(NAME nvn.metal_replay_validation COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_validation PROPERTIES ENVIRONMENT "MTL_DEBUG_LAYER=1")
    add_test(NAME nvn.metal_replay_conditional_contract_validation COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_conditional_contract_validation PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_CONDITIONAL_CONTRACT_TESTS=1;MTL_DEBUG_LAYER=1")
    add_test(NAME nvn.metal_replay_worker_encoding_validation COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_worker_encoding_validation PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_ENCODE_THREAD=0;MTL_DEBUG_LAYER=1")
    # Exercise finalized-query reuse, write guards and restored draw state in CI.
    add_test(NAME nvn.metal_replay_finalized_queries COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_finalized_queries PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_BATCH_FINALIZED_QUERIES=1;ARMIGO_NVN_CONDITIONAL_CONTRACT_TESTS=1;MTL_DEBUG_LAYER=1")
    add_test(NAME nvn.metal_replay_finalized_queries_worker COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_finalized_queries_worker PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_BATCH_FINALIZED_QUERIES=1;ARMIGO_NVN_CONDITIONAL_CONTRACT_TESTS=1;ARMIGO_NVN_ENCODE_THREAD=0;MTL_DEBUG_LAYER=1")
    add_test(NAME nvn.metal_unused_colors COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_unused_colors PROPERTIES ENVIRONMENT
        "ARMIGO_NVN_UNUSED_COLOR_CONTRACT_TESTS=1;ARMIGO_NVN_OMIT_UNUSED_COLORS=0;ARMIGO_NVN_ENCODE_THREAD=1;MTL_DEBUG_LAYER=1")
    add_test(NAME nvn.metal_unused_colors_worker COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_unused_colors_worker PROPERTIES ENVIRONMENT
        "ARMIGO_NVN_UNUSED_COLOR_CONTRACT_TESTS=1;ARMIGO_NVN_OMIT_UNUSED_COLORS=0;ARMIGO_NVN_ENCODE_THREAD=0;MTL_DEBUG_LAYER=1")
    # Conditional-render predicates as the compute kernel before each block's pass.
    add_test(NAME nvn.metal_replay_predicate_kernel COMMAND armigo_metal_nvn_replay_tests)
    set_tests_properties(nvn.metal_replay_predicate_kernel PROPERTIES
        ENVIRONMENT "ARMIGO_NVN_PREDICATE_IN_PASS=0")
endif()

if(BUILD_TESTING)
    add_executable(armigo_metal_nvn_replay_ordering_tests
        "${PROJECT_SOURCE_DIR}/tests/metal/nvn_replay_ordering_tests.mm")
    target_include_directories(armigo_metal_nvn_replay_ordering_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/nvn" "${PROJECT_SOURCE_DIR}/src/platform/metal")
    target_link_libraries(armigo_metal_nvn_replay_ordering_tests PRIVATE armigo_nvn_metal)
    add_test(NAME nvn.metal_replay_ordering COMMAND armigo_metal_nvn_replay_ordering_tests)
    add_executable(armigo_metal_nvn_encode_ring_tests
        "${PROJECT_SOURCE_DIR}/tests/metal/nvn_encode_ring_tests.cpp")
    target_include_directories(armigo_metal_nvn_encode_ring_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/platform/metal")
    add_test(NAME nvn.metal_encode_ring COMMAND armigo_metal_nvn_encode_ring_tests)
    add_executable(armigo_metal_nvn_write_ranges_tests
        "${PROJECT_SOURCE_DIR}/tests/metal/nvn_write_ranges_tests.cpp")
    target_include_directories(armigo_metal_nvn_write_ranges_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/platform/metal")
    target_compile_features(armigo_metal_nvn_write_ranges_tests PRIVATE cxx_std_20)
    add_test(NAME nvn.metal_write_ranges COMMAND armigo_metal_nvn_write_ranges_tests)
    add_executable(armigo_metal_pso_manifest_tests
        "${PROJECT_SOURCE_DIR}/tests/metal/pso_manifest_tests.cpp"
        "${PROJECT_SOURCE_DIR}/src/platform/metal/pso_manifest.cpp")
    target_include_directories(armigo_metal_pso_manifest_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/platform/metal")
    target_compile_features(armigo_metal_pso_manifest_tests PRIVATE cxx_std_20)
    add_test(NAME nvn.metal_pso_manifest COMMAND armigo_metal_pso_manifest_tests)
    add_executable(armigo_metal_pso_archive_tests "${PROJECT_SOURCE_DIR}/tests/metal/pso_archive_tests.mm"
        "${PROJECT_SOURCE_DIR}/src/platform/metal/pso_manifest.cpp")
    target_include_directories(armigo_metal_pso_archive_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/platform/metal")
    target_link_libraries(armigo_metal_pso_archive_tests PRIVATE "-framework Metal" "-framework Foundation")
    add_test(NAME nvn.metal_pso_archive COMMAND armigo_metal_pso_archive_tests)
    set_tests_properties(nvn.metal_pso_archive PROPERTIES LABELS "gpu;metal4" SKIP_RETURN_CODE 77 TIMEOUT 60)
    add_executable(armigo_metal_nvn_attachment_actions_tests
        "${PROJECT_SOURCE_DIR}/tests/metal/nvn_attachment_actions_tests.cpp")
    target_include_directories(armigo_metal_nvn_attachment_actions_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/platform/metal")
    add_test(NAME nvn.metal_attachment_actions COMMAND armigo_metal_nvn_attachment_actions_tests)
endif()

if(BUILD_TESTING)
    set_tests_properties(nvn.metal_replay nvn.metal_replay_predicate_kernel nvn.metal_replay_ordering
        PROPERTIES TIMEOUT 90 LABELS "nvn;metal")
endif()
