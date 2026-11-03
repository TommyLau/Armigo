# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Emits the synthetic MSL fixtures and compiles each one as Metal 4.0.
#
# Required variables: TEST_EXECUTABLE, XCRUN, OUTPUT_DIR.

foreach(variable IN ITEMS TEST_EXECUTABLE XCRUN OUTPUT_DIR)
    if(NOT DEFINED ${variable})
        message(FATAL_ERROR "${variable} is not set")
    endif()
endforeach()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

execute_process(
    COMMAND "${TEST_EXECUTABLE}" --emit-fixtures "${OUTPUT_DIR}"
    RESULT_VARIABLE emit_result)
if(NOT emit_result EQUAL 0)
    message(FATAL_ERROR "fixture emission failed")
endif()

file(GLOB fixtures "${OUTPUT_DIR}/*.metal")
list(SORT fixtures)
list(LENGTH fixtures fixture_count)
if(fixture_count EQUAL 0)
    message(FATAL_ERROR "no fixtures were emitted")
endif()

foreach(fixture IN LISTS fixtures)
    get_filename_component(stem "${fixture}" NAME_WE)
    execute_process(
        COMMAND "${XCRUN}" -sdk macosx metal -std=metal4.0 -Werror
                -c "${fixture}" -o "${OUTPUT_DIR}/${stem}.air"
        RESULT_VARIABLE compile_result
        ERROR_VARIABLE compile_errors)
    if(NOT compile_result EQUAL 0)
        message(FATAL_ERROR "Metal compilation of ${stem} failed:\n${compile_errors}")
    endif()
    message(STATUS "compiled ${stem}")
endforeach()
