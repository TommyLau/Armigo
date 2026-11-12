# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Title-owned test targets. Generic build files include this list.

add_executable(armigo_01007EF00011E000_frame_rate_tests
    "${PROJECT_SOURCE_DIR}/tests/titles/01007EF00011E000/frame_rate_tests.cpp")
target_link_libraries(armigo_01007EF00011E000_frame_rate_tests PRIVATE armigo_compat)
add_test(NAME title.01007EF00011E000.frame_rate COMMAND armigo_01007EF00011E000_frame_rate_tests)

add_executable(armigo_01004B10157F2000_settings_tests
    "${PROJECT_SOURCE_DIR}/tests/titles/01004B10157F2000/settings_tests.cpp")
target_link_libraries(armigo_01004B10157F2000_settings_tests PRIVATE armigo_compat)
add_test(NAME title.01004B10157F2000.settings
    COMMAND armigo_01004B10157F2000_settings_tests "${PROJECT_SOURCE_DIR}/titles")

add_executable(armigo_01007EF00011E000_settings_tests
    "${PROJECT_SOURCE_DIR}/tests/titles/01007EF00011E000/settings_tests.cpp")
target_link_libraries(armigo_01007EF00011E000_settings_tests PRIVATE armigo_compat)
add_test(NAME title.01007EF00011E000.settings
    COMMAND armigo_01007EF00011E000_settings_tests "${PROJECT_SOURCE_DIR}/titles")
