# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Armigo native shader input parser: validates the control and code
# containers and enumerates scheduling groups and instruction word slots for
# the independent Maxwell decoder.
#
# The parent project adds this directory with add_subdirectory(src/shader_input).
# Paths are resolved relative to this file so the directory also builds when
# added from an out-of-tree harness.

cmake_path(SET ARMIGO_SHADER_INPUT_REPOSITORY_ROOT NORMALIZE
           "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(ARMIGO_SHADER_INPUT_INCLUDE_DIR
    "${ARMIGO_SHADER_INPUT_REPOSITORY_ROOT}/include")
set(ARMIGO_SHADER_INPUT_TEST_DIR
    "${ARMIGO_SHADER_INPUT_REPOSITORY_ROOT}/tests/shader_input")

add_library(armigo_shader_input
    archive.cpp
    container.cpp
    gfd_cache.cpp
    glslc_output.cpp
    shader_input.cpp
)
add_library(armigo::shader_input ALIAS armigo_shader_input)

target_include_directories(armigo_shader_input PUBLIC
    "$<BUILD_INTERFACE:${ARMIGO_SHADER_INPUT_INCLUDE_DIR}>"
)
target_compile_features(armigo_shader_input PUBLIC cxx_std_20)
set_target_properties(armigo_shader_input PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(armigo_shader_input PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(armigo_shader_input PRIVATE /W4 /permissive-)
endif()

# Default the option exactly as include(CTest) would, so the tests also
# build under harnesses that only call enable_testing().
if(NOT DEFINED BUILD_TESTING)
    option(BUILD_TESTING "Build the testing tree." ON)
endif()

if(BUILD_TESTING)
    add_executable(armigo_shader_input_tests
        "${ARMIGO_SHADER_INPUT_TEST_DIR}/archive_tests.cpp"
        "${ARMIGO_SHADER_INPUT_TEST_DIR}/container_tests.cpp"
        "${ARMIGO_SHADER_INPUT_TEST_DIR}/gfd_cache_tests.cpp"
        "${ARMIGO_SHADER_INPUT_TEST_DIR}/glslc_output_tests.cpp"
        "${ARMIGO_SHADER_INPUT_TEST_DIR}/shader_input_tests.cpp"
        "${ARMIGO_SHADER_INPUT_TEST_DIR}/test_main.cpp"
    )
    target_include_directories(armigo_shader_input_tests PRIVATE
        "${ARMIGO_SHADER_INPUT_TEST_DIR}"
    )
    target_link_libraries(armigo_shader_input_tests PRIVATE armigo_shader_input)
    set_target_properties(armigo_shader_input_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(armigo_shader_input_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()

    foreach(group IN ITEMS control code structure metadata yaz0 sarc bnsh bahs gfd glslc)
        add_test(NAME shader_input.${group}
                 COMMAND armigo_shader_input_tests ${group})
    endforeach()
endif()
