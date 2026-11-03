# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

add_library(armigo_linker STATIC dynamic_module.cpp link_plan.cpp)
target_include_directories(armigo_linker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_compile_features(armigo_linker PUBLIC cxx_std_20)

if(BUILD_TESTING)
    add_executable(armigo_dynamic_tests ../../tests/linker/dynamic_module_tests.cpp)
    target_link_libraries(armigo_dynamic_tests PRIVATE armigo_linker)
    add_test(NAME linker.dynamic_module COMMAND armigo_dynamic_tests)

    add_executable(armigo_link_plan_tests ../../tests/linker/link_plan_tests.cpp)
    target_link_libraries(armigo_link_plan_tests PRIVATE armigo_linker)
    add_test(NAME linker.link_plan COMMAND armigo_link_plan_tests)
endif()
