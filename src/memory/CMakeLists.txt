# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Guest memory metadata model library and the host page write tracker.
#
# The root build integrates this directory with
# add_subdirectory(src/memory). Source and include paths are resolved
# relative to this file so the target works from any including project.

set(ARMIGO_MEMORY_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(armigo_memory guest_memory_model.cpp write_tracker.cpp)
add_library(armigo::memory ALIAS armigo_memory)

target_include_directories(armigo_memory
    PUBLIC
        "${ARMIGO_MEMORY_SOURCE_ROOT}/include"
)

target_compile_features(armigo_memory PUBLIC cxx_std_20)

if(BUILD_TESTING)
    add_executable(armigo_memory_tests
        "${ARMIGO_MEMORY_SOURCE_ROOT}/tests/memory/test_guest_memory_model.cpp"
    )
    target_compile_features(armigo_memory_tests PRIVATE cxx_std_20)
    target_link_libraries(armigo_memory_tests PRIVATE armigo::memory)

    add_test(
        NAME armigo_memory_tests
        COMMAND armigo_memory_tests
    )

    add_executable(armigo_memory_failure_tests
        "${ARMIGO_MEMORY_SOURCE_ROOT}/tests/memory/failure_contracts.cpp")
    target_link_libraries(armigo_memory_failure_tests PRIVATE armigo::memory)
    add_test(NAME memory.failure_contracts COMMAND armigo_memory_failure_tests)

    add_executable(armigo_memory_write_tracker_tests
        "${ARMIGO_MEMORY_SOURCE_ROOT}/tests/memory/write_tracker_tests.cpp")
    target_compile_features(armigo_memory_write_tracker_tests PRIVATE cxx_std_20)
    find_package(Threads REQUIRED)
    target_link_libraries(armigo_memory_write_tracker_tests PRIVATE armigo::memory
        Threads::Threads)
    add_test(NAME memory.write_tracker COMMAND armigo_memory_write_tracker_tests)
    set_tests_properties(memory.write_tracker PROPERTIES TIMEOUT 120)

    # The registry contract test exercises the optional CPU write stamp
    # surface through a registry created with tracking enabled.
    add_executable(armigo_memory_write_tracker_registry_tests
        "${ARMIGO_MEMORY_SOURCE_ROOT}/tests/memory/write_tracker_registry_tests.cpp")
    target_compile_features(armigo_memory_write_tracker_registry_tests PRIVATE cxx_std_20)
    target_link_libraries(armigo_memory_write_tracker_registry_tests PRIVATE
        armigo::graphics armigo::memory)
    add_test(NAME memory.write_tracker_registry
             COMMAND armigo_memory_write_tracker_registry_tests)
    set_tests_properties(memory.write_tracker_registry PROPERTIES TIMEOUT 120)
endif()
