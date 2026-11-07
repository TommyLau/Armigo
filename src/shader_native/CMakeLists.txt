# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Native input and ISA components deliberately do not join the Maxwell source identity.
find_package(zstd CONFIG QUIET)
if(NOT TARGET zstd::libzstd_static)
    message(STATUS "Native shader inventory disabled: static Zstandard package unavailable")
    return()
endif()
add_library(armigo_shader_native STATIC input.cpp program.cpp instruction.cpp opcode_inventory.cpp)
target_include_directories(armigo_shader_native PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(armigo_shader_native PUBLIC armigo_shader_input PRIVATE zstd::libzstd_static)
if(NOT ARMIGO_IOS)
    add_executable(armigo-native-shader-inventory inventory.cpp)
    target_link_libraries(armigo-native-shader-inventory PRIVATE armigo_shader_native armigo_loader)
endif()
if(BUILD_TESTING)
    add_executable(armigo_shader_native_input_tests "${PROJECT_SOURCE_DIR}/tests/shader_native/input_tests.cpp")
    target_link_libraries(armigo_shader_native_input_tests PRIVATE armigo_shader_native zstd::libzstd_static)
    add_test(NAME shader_native.input COMMAND armigo_shader_native_input_tests)
    add_executable(armigo_shader_native_program_tests "${PROJECT_SOURCE_DIR}/tests/shader_native/program_tests.cpp")
    target_link_libraries(armigo_shader_native_program_tests PRIVATE armigo_shader_native)
    add_test(NAME shader_native.program COMMAND armigo_shader_native_program_tests)
    add_executable(armigo_shader_native_instruction_tests "${PROJECT_SOURCE_DIR}/tests/shader_native/instruction_tests.cpp")
    target_link_libraries(armigo_shader_native_instruction_tests PRIVATE armigo_shader_native)
    add_test(NAME shader_native.instruction COMMAND armigo_shader_native_instruction_tests)
endif()

# A direct native frontend reuses backend contracts without modifying their sources.
add_library(armigo_shader_native_frontend STATIC frontend.cpp arithmetic.cpp)
target_link_libraries(armigo_shader_native_frontend PUBLIC armigo_shader_native armigo_shader_pipeline)
if(BUILD_TESTING)
    add_executable(armigo_shader_native_frontend_tests "${PROJECT_SOURCE_DIR}/tests/shader_native/frontend_tests.cpp")
    target_link_libraries(armigo_shader_native_frontend_tests PRIVATE armigo_shader_native_frontend)
    add_test(NAME shader_native.frontend COMMAND armigo_shader_native_frontend_tests)
endif()

if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    add_executable(armigo_shader_native_numeric_gpu_tests "${PROJECT_SOURCE_DIR}/tests/shader_native/numeric_gpu.mm")
    target_link_libraries(armigo_shader_native_numeric_gpu_tests PRIVATE armigo_shader_native_frontend "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader_native.numeric_gpu COMMAND armigo_shader_native_numeric_gpu_tests)
    set_tests_properties(shader_native.numeric_gpu PROPERTIES LABELS "shader;gpu;metal4" TIMEOUT 120)
endif()
