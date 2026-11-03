# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

add_library(armigo_diagnostics STATIC diagnostics.cpp)
target_include_directories(armigo_diagnostics PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(armigo_diagnostics PUBLIC cxx_std_20)

if(BUILD_TESTING)
    add_executable(armigo_diagnostics_tests
        "${PROJECT_SOURCE_DIR}/tests/common/diagnostics_tests.cpp")
    target_link_libraries(armigo_diagnostics_tests PRIVATE armigo_diagnostics)
    add_test(NAME common.diagnostics COMMAND armigo_diagnostics_tests)
    set_tests_properties(common.diagnostics PROPERTIES TIMEOUT 60 LABELS "common")
endif()
