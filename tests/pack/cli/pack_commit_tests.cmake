# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

if(NOT DEFINED SCRIPT)
    message(FATAL_ERROR "SCRIPT is required")
endif()
set(root "$ENV{TMPDIR}/armigo-pack-commit-test")
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/repository")
execute_process(COMMAND git init -q "${root}/repository" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${root}/repository" config user.name "Synthetic Test" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${root}/repository" config user.email "synthetic@example.invalid" COMMAND_ERROR_IS_FATAL ANY)
file(WRITE "${root}/repository/input.txt" "first\n")
execute_process(COMMAND git -C "${root}/repository" add input.txt COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${root}/repository" commit -q -m "test: first" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${root}/repository" rev-parse HEAD
    OUTPUT_VARIABLE first OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(header "${root}/pack_commit.h")
execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${root}/repository" "-DOUTPUT=${header}"
    -P "${SCRIPT}" COMMAND_ERROR_IS_FATAL ANY)
file(READ "${header}" content)
if(NOT content MATCHES "#define ARMIGO_PACK_COMMIT \"${first}\"")
    message(FATAL_ERROR "The first build-time commit was not generated")
endif()
file(WRITE "${root}/repository/main.cpp"
    "#include \"pack_commit.h\"\n#include <cstdio>\nint main() { std::puts(ARMIGO_PACK_COMMIT); }\n")
file(WRITE "${root}/repository/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.30)
project(pack_commit_test LANGUAGES CXX)
set(header "${CMAKE_CURRENT_BINARY_DIR}/pack_commit.h")
add_custom_target(refresh
    COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}" "-DOUTPUT=${header}" -P "${PACK_SCRIPT}"
    BYPRODUCTS "${header}" VERBATIM)
add_executable(show main.cpp)
add_dependencies(show refresh)
set_source_files_properties(main.cpp PROPERTIES OBJECT_DEPENDS "${header}")
target_include_directories(show PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
]=])
execute_process(COMMAND "${CMAKE_COMMAND}" -G Ninja -S "${root}/repository" -B "${root}/build" "-DPACK_SCRIPT=${SCRIPT}"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${root}/build" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${root}/build/show" OUTPUT_VARIABLE actual_first
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT actual_first STREQUAL first)
    message(FATAL_ERROR "The first compiled commit was stale")
endif()
file(WRITE "${root}/repository/input.txt" "second\n")
execute_process(COMMAND git -C "${root}/repository" add input.txt COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${root}/repository" commit -q -m "test: second" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git -C "${root}/repository" rev-parse HEAD
    OUTPUT_VARIABLE second OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(first STREQUAL second)
    message(FATAL_ERROR "Synthetic commits did not change HEAD")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${root}/repository" "-DOUTPUT=${header}"
    -P "${SCRIPT}" COMMAND_ERROR_IS_FATAL ANY)
file(READ "${header}" content)
if(NOT content MATCHES "#define ARMIGO_PACK_COMMIT \"${second}\"")
    message(FATAL_ERROR "The build-time commit remained stale after a new commit")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${root}/build" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${root}/build/show" OUTPUT_VARIABLE actual_second
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT actual_second STREQUAL second)
    message(FATAL_ERROR "The compiled commit remained stale without reconfiguration")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${root}" "-DOUTPUT=${header}"
    -P "${SCRIPT}" COMMAND_ERROR_IS_FATAL ANY)
file(READ "${header}" content)
if(NOT content MATCHES "#define ARMIGO_PACK_COMMIT \"unknown\"")
    message(FATAL_ERROR "A non-repository did not produce the unknown fallback")
endif()
file(REMOVE_RECURSE "${root}")
