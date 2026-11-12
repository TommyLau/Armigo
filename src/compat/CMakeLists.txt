# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

add_library(armigo_compat STATIC
    "${PROJECT_SOURCE_DIR}/src/titles/frame_rate.cpp"
    "${PROJECT_SOURCE_DIR}/src/titles/resolution.cpp"
    observer.cpp patches.mm title_plan.cpp title_settings.mm)
target_include_directories(armigo_compat PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(armigo_compat PUBLIC armigo_horizon armigo_loader armigo_aarch64
    armigo_execution armigo_execution_call_hook armigo_execution_gate_patch "${ARMIGO_FOUNDATION_FRAMEWORK}")
target_link_libraries(armigo_runtime PUBLIC armigo_compat)

if(BUILD_TESTING)
    add_executable(armigo_compat_patches_tests "${PROJECT_SOURCE_DIR}/tests/compat/patches_tests.cpp")
    target_link_libraries(armigo_compat_patches_tests PRIVATE armigo_compat)
    add_test(NAME compat.patches COMMAND armigo_compat_patches_tests)
    add_executable(armigo_compat_observer_tests "${PROJECT_SOURCE_DIR}/tests/compat/observer_tests.cpp")
    target_link_libraries(armigo_compat_observer_tests PRIVATE armigo_compat)
    add_test(NAME compat.observer COMMAND armigo_compat_observer_tests)
    # Checks every title folder in the repository, and the lookup on synthetic folders.
    add_executable(armigo_title_settings_tests "${PROJECT_SOURCE_DIR}/tests/compat/title_settings_tests.cpp"
        "${PROJECT_SOURCE_DIR}/src/platform/metal/pso_manifest.cpp")
    target_include_directories(armigo_title_settings_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/platform/metal")
    target_link_libraries(armigo_title_settings_tests PRIVATE armigo_compat)
    add_test(NAME compat.title_settings COMMAND armigo_title_settings_tests "${PROJECT_SOURCE_DIR}/titles")
    add_executable(armigo_title_frame_rate_tests "${PROJECT_SOURCE_DIR}/tests/titles/frame_rate_tests.cpp")
    target_link_libraries(armigo_title_frame_rate_tests PRIVATE armigo_compat)
    add_test(NAME title.frame_rate COMMAND armigo_title_frame_rate_tests)
    # Pins every title's selected words, calls, hooks and resolution words by digest.
    add_executable(armigo_title_plan_golden_tests "${PROJECT_SOURCE_DIR}/tests/titles/plan_golden_tests.cpp")
    target_link_libraries(armigo_title_plan_golden_tests PRIVATE armigo_compat)
    add_test(NAME title.plan_golden COMMAND armigo_title_plan_golden_tests "${PROJECT_SOURCE_DIR}/titles"
        "${PROJECT_SOURCE_DIR}/tests/titles/plan_golden.txt")
    include("${PROJECT_SOURCE_DIR}/tests/titles/titles.cmake")
    add_executable(armigo_title_resolution_tests "${PROJECT_SOURCE_DIR}/tests/titles/resolution_tests.cpp")
    target_link_libraries(armigo_title_resolution_tests PRIVATE armigo_compat)
    add_test(NAME title.resolution COMMAND armigo_title_resolution_tests)
endif()
