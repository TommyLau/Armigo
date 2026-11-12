# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Guest module reader library.
#
# The root build integrates this directory with add_subdirectory(src/loader).
# Source and include paths are resolved relative to this file so the target
# also works from an out-of-tree harness that adds this directory directly.

set(ARMIGO_LOADER_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

find_library(ARMIGO_SECURITY_FRAMEWORK Security REQUIRED)
find_library(ARMIGO_COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/nso_component.cmake")
armigo_add_nso_component()

add_library(armigo_loader STATIC
    $<TARGET_OBJECTS:armigo_loader_nso>
    exefs_directory.cpp
    zbic.cpp
    npdm.cpp
    nup.cpp
    container/keys.cpp
    container/crypto.cpp
    container/nca.cpp
    container/container.cpp
)
add_library(armigo::loader ALIAS armigo_loader)

# All player configurations use the same pinned decoder dependency.
if(NOT TARGET zstd::libzstd_static)
    include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/RuntimeDependencies.cmake")
endif()
target_link_libraries(armigo_loader PRIVATE zstd::libzstd_static)
target_compile_definitions(armigo_loader PRIVATE ARMIGO_HAS_ZSTD=1)


target_include_directories(armigo_loader
    PUBLIC "${ARMIGO_LOADER_SOURCE_ROOT}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_features(armigo_loader PUBLIC cxx_std_20)
# Container signature verification uses Security.framework RSA-PSS checks
# with CoreFoundation data types; CommonCrypto's AES primitives live in
# libSystem and need no link input.
target_link_libraries(armigo_loader
    PUBLIC "${ARMIGO_SECURITY_FRAMEWORK}" "${ARMIGO_COREFOUNDATION_FRAMEWORK}")

if(BUILD_TESTING)
    if(TARGET zstd::libzstd_static)
        add_executable(armigo_loader_zbic_tests
            "${ARMIGO_LOADER_SOURCE_ROOT}/tests/loader/zbic_tests.cpp")
        target_include_directories(armigo_loader_zbic_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(armigo_loader_zbic_tests PRIVATE armigo::loader)
        add_test(NAME loader.zbic COMMAND armigo_loader_zbic_tests)
    endif()
    add_executable(armigo_loader_nup_tests
        "${ARMIGO_LOADER_SOURCE_ROOT}/tests/loader/nup_tests.cpp")
    target_include_directories(armigo_loader_nup_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_loader_nup_tests PRIVATE armigo::loader)
    add_test(NAME loader.nup COMMAND armigo_loader_nup_tests)

    add_executable(armigo_loader_lz4_tests
        "${ARMIGO_LOADER_SOURCE_ROOT}/tests/loader/test_lz4_block.cpp")
    target_link_libraries(armigo_loader_lz4_tests PRIVATE armigo::loader)
    add_test(NAME loader.lz4_block COMMAND armigo_loader_lz4_tests)

    add_executable(armigo_loader_nso_tests
        "${ARMIGO_LOADER_SOURCE_ROOT}/tests/loader/test_nso_module.cpp")
    # The fixture builder hashes synthetic segments with the internal helper.
    target_include_directories(armigo_loader_nso_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_loader_nso_tests PRIVATE armigo::loader)
    add_test(NAME loader.nso_module COMMAND armigo_loader_nso_tests)

    add_executable(armigo_loader_exefs_tests
        "${ARMIGO_LOADER_SOURCE_ROOT}/tests/loader/exefs_directory_tests.cpp")
    target_link_libraries(armigo_loader_exefs_tests PRIVATE armigo::loader)
    add_test(NAME loader.exefs_directory COMMAND armigo_loader_exefs_tests)

    add_executable(armigo_loader_npdm_tests
        "${ARMIGO_LOADER_SOURCE_ROOT}/tests/loader/npdm_tests.cpp")
    # The structural reader header is internal to this directory.
    target_include_directories(armigo_loader_npdm_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_loader_npdm_tests PRIVATE armigo::loader)
    add_test(NAME loader.npdm COMMAND armigo_loader_npdm_tests)
    set_tests_properties(loader.npdm PROPERTIES LABELS "loader;runtime" TIMEOUT 30)

    add_executable(armigo_loader_container_tests
        "${ARMIGO_LOADER_SOURCE_ROOT}/tests/loader/container/test_container.cpp")
    # The container primitives header is internal to this directory.
    target_include_directories(armigo_loader_container_tests
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_loader_container_tests PRIVATE armigo::loader)
    add_test(NAME loader.container COMMAND armigo_loader_container_tests)
    set_tests_properties(loader.container PROPERTIES LABELS "loader" TIMEOUT 60)
endif()

# Development inspection tool; it summarizes modules without executing them.
add_executable(armigo-nso-inspect tools/nso_inspect.cpp)
target_link_libraries(armigo-nso-inspect PRIVATE armigo::loader)
if(BUILD_TESTING)
    add_test(NAME loader.nso_inspect_help COMMAND armigo-nso-inspect --help)
endif()

add_executable(armigo-nup-inspect tools/nup_inspect.cpp)
target_link_libraries(armigo-nup-inspect PRIVATE armigo::loader)
