# ============================================================================
# Armigo Tests - Native Runtime Verification
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED MODULE)
    message(FATAL_ERROR "SOURCE_DIR and MODULE are required")
endif()
set(root "$ENV{TMPDIR}/armigo-ios-embedded-cache")
cmake_path(NORMAL_PATH root)
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/sources-a" "${root}/sources-b")
file(WRITE "${root}/sources-a/embedded.c" "void embedded_a(void) {}\n")
file(WRITE "${root}/sources-b/embedded.c" "void embedded_b(void) {}\n")
set(build "${root}/build")

function(configure_player directory)
    if(directory STREQUAL "")
        unset(ENV{ARMIGO_IOS_EMBEDDED_SOURCES_DIR})
    else()
        set(ENV{ARMIGO_IOS_EMBEDDED_SOURCES_DIR} "${directory}")
    endif()
    set(unset_argument -UARMIGO_IOS_EMBEDDED_SOURCES_DIR)
    if(ARGC GREATER 1 AND ARGV1 STREQUAL "NO_UNSET")
        set(unset_argument)
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -G Ninja -S "${SOURCE_DIR}" -B "${build}"
                "-DMODULE=${MODULE}" ${unset_argument}
        RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "Player configure failed for '${directory}'")
    endif()
endfunction()

function(expect_selected expected)
    file(READ "${build}/selected.txt" selected)
    string(STRIP "${selected}" selected)
    if(NOT selected STREQUAL expected)
        message(FATAL_ERROR "Selected '${selected}', expected '${expected}'")
    endif()
endfunction()

function(expect_cache_clear)
    if(NOT EXISTS "${build}/CMakeCache.txt")
        message(FATAL_ERROR "CMakeCache.txt is missing")
    endif()
    file(STRINGS "${build}/CMakeCache.txt" entries REGEX "^ARMIGO_IOS_EMBEDDED_SOURCES_DIR:")
    if(entries)
        message(FATAL_ERROR "CMake cache still records embedded sources: ${entries}")
    endif()
endfunction()

function(expect_graph directory present)
    file(READ "${build}/build.ninja" graph)
    string(FIND "${graph}" "${directory}/embedded.c" found)
    if(present AND found EQUAL -1)
        message(FATAL_ERROR "Build graph omitted ${directory}")
    endif()
    if(NOT present AND NOT found EQUAL -1)
        message(FATAL_ERROR "Build graph still contains ${directory}")
    endif()
endfunction()

configure_player("${root}/sources-a")
expect_selected("${root}/sources-a")
expect_cache_clear()
expect_graph("${root}/sources-a" YES)
expect_graph("${root}/sources-b" NO)

configure_player("${root}/sources-b")
expect_selected("${root}/sources-b")
expect_cache_clear()
expect_graph("${root}/sources-b" YES)
expect_graph("${root}/sources-a" NO)

configure_player("")
expect_selected("")
expect_cache_clear()
expect_graph("${root}/sources-a" NO)
expect_graph("${root}/sources-b" NO)

file(APPEND "${build}/CMakeCache.txt" "ARMIGO_IOS_EMBEDDED_SOURCES_DIR:PATH=${root}/sources-a\n")
configure_player("" NO_UNSET)
expect_selected("")
expect_cache_clear()
expect_graph("${root}/sources-a" NO)

unset(ENV{ARMIGO_IOS_EMBEDDED_SOURCES_DIR})
execute_process(
    COMMAND "${CMAKE_COMMAND}" -G Ninja -S "${SOURCE_DIR}" -B "${build}"
            "-DMODULE=${MODULE}"
            "-DARMIGO_IOS_EMBEDDED_SOURCES_DIR=${root}/sources-b"
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Configure with a cache assignment failed")
endif()
expect_selected("")
expect_cache_clear()
expect_graph("${root}/sources-b" NO)
file(REMOVE_RECURSE "${root}")
