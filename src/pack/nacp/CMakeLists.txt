# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Control data naming for the packer: parse the title table (including the
# compressed one newer control data uses), choose the app title and icon,
# and build a file-system-safe app name.

add_library(armigo_pack_nacp STATIC nacp.cpp)
target_include_directories(armigo_pack_nacp PUBLIC "${PROJECT_SOURCE_DIR}/include")
# Raw DEFLATE for the compressed title area; the system zlib ships with the
# macOS and iOS SDKs.
target_link_libraries(armigo_pack_nacp PRIVATE z)

if(BUILD_TESTING)
    add_executable(armigo_pack_nacp_tests "${PROJECT_SOURCE_DIR}/tests/pack/nacp/nacp_tests.cpp")
    target_link_libraries(armigo_pack_nacp_tests PRIVATE armigo_pack_nacp z)
    add_test(NAME pack.nacp COMMAND armigo_pack_nacp_tests)
endif()
