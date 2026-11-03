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
    "${PROJECT_SOURCE_DIR}/src/titles/01007EF00011E000/frame_rate.cpp"
    "${PROJECT_SOURCE_DIR}/src/titles/01007EF00011E000/resolution.cpp"
    patches.mm title_settings.mm)
target_include_directories(armigo_compat PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(armigo_compat PUBLIC armigo_horizon armigo_loader armigo_aarch64
    armigo_execution "${ARMIGO_FOUNDATION_FRAMEWORK}")
target_link_libraries(armigo_runtime PUBLIC armigo_compat)

if(BUILD_TESTING)
    add_executable(armigo_compat_patches_tests "${PROJECT_SOURCE_DIR}/tests/compat/patches_tests.cpp")
    target_link_libraries(armigo_compat_patches_tests PRIVATE armigo_compat)
    add_test(NAME compat.patches COMMAND armigo_compat_patches_tests)
    # Checks every title folder in the repository, and the lookup on synthetic folders.
    add_executable(armigo_title_settings_tests "${PROJECT_SOURCE_DIR}/tests/compat/title_settings_tests.cpp"
        "${PROJECT_SOURCE_DIR}/src/platform/metal/pso_manifest.cpp")
    target_include_directories(armigo_title_settings_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/platform/metal")
    target_link_libraries(armigo_title_settings_tests PRIVATE armigo_compat)
    add_test(NAME compat.title_settings COMMAND armigo_title_settings_tests "${PROJECT_SOURCE_DIR}/titles")
    add_executable(armigo_01007EF00011E000_frame_rate_tests
        "${PROJECT_SOURCE_DIR}/tests/titles/01007EF00011E000/frame_rate_tests.cpp")
    target_link_libraries(armigo_01007EF00011E000_frame_rate_tests PRIVATE armigo_compat)
    add_test(NAME title.01007EF00011E000.frame_rate COMMAND armigo_01007EF00011E000_frame_rate_tests)
    add_executable(armigo_01007EF00011E000_resolution_tests
        "${PROJECT_SOURCE_DIR}/tests/titles/01007EF00011E000/resolution_tests.cpp")
    target_link_libraries(armigo_01007EF00011E000_resolution_tests PRIVATE armigo_compat)
    add_test(NAME title.01007EF00011E000.resolution COMMAND armigo_01007EF00011E000_resolution_tests)
endif()
