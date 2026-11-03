# ============================================================================
# Armigo Tests - Native Runtime Verification
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Emits the forward-lowered graph MSL fixture and compiles it as Metal 4.0.
#
# Required variables: TEST_EXECUTABLE, XCRUN, OUTPUT_DIR.

foreach(variable IN ITEMS TEST_EXECUTABLE XCRUN OUTPUT_DIR)
    if(NOT DEFINED ${variable})
        message(FATAL_ERROR "${variable} is not set")
    endif()
endforeach()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

set(ENV{ARMIGO_FORWARD_MSL_DIR} "${OUTPUT_DIR}")
execute_process(
    COMMAND "${TEST_EXECUTABLE}" forward_lower_msl
    RESULT_VARIABLE emit_result
    OUTPUT_QUIET ERROR_VARIABLE emit_errors)
unset(ENV{ARMIGO_FORWARD_MSL_DIR})
if(NOT emit_result EQUAL 0)
    message(FATAL_ERROR "forward fixture emission failed:\n${emit_errors}")
endif()

file(GLOB fixtures "${OUTPUT_DIR}/*.metal")
list(SORT fixtures)
list(LENGTH fixtures fixture_count)
if(fixture_count EQUAL 0)
    message(FATAL_ERROR "no forward fixtures were emitted")
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
