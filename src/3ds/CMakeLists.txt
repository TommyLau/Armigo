# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(Armigo3DS LANGUAGES CXX)
    include(CTest)
endif()

get_filename_component(ARMIGO_3DS_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_library(armigo_3ds_memory STATIC memory/guest_memory.cpp)
target_compile_features(armigo_3ds_memory PUBLIC cxx_std_20)
set_target_properties(armigo_3ds_memory PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(armigo_3ds_memory PUBLIC "${ARMIGO_3DS_SOURCE_ROOT}/include")
target_compile_options(armigo_3ds_memory PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>")

if(ANDROID AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    enable_language(ASM)
    target_sources(armigo_3ds_memory PRIVATE
        platform/android/guest_memory.cpp platform/android/map_memory.S)
endif()

if(BUILD_TESTING)
    add_executable(armigo_3ds_memory_tests
        "${ARMIGO_3DS_SOURCE_ROOT}/tests/3ds/guest_memory_tests.cpp")
    target_link_libraries(armigo_3ds_memory_tests PRIVATE armigo_3ds_memory)
    target_compile_options(armigo_3ds_memory_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME armigo_3ds_guest_memory COMMAND armigo_3ds_memory_tests)

    if(ANDROID AND CMAKE_SIZEOF_VOID_P EQUAL 4)
        add_executable(armigo_3ds_android_memory_tests
            "${ARMIGO_3DS_SOURCE_ROOT}/tests/3ds/android_memory_tests.cpp")
        target_link_libraries(armigo_3ds_android_memory_tests PRIVATE armigo_3ds_memory)
        target_compile_options(armigo_3ds_android_memory_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME armigo_3ds_android_memory COMMAND armigo_3ds_android_memory_tests)
    endif()
endif()
