# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# The packer's libraries. The root build adds this directory only.

add_subdirectory(nacp)
add_subdirectory(inputs)

if(NOT ARMIGO_IOS)
    add_library(armigo_pack_ios STATIC ios_link.cpp)
    target_link_libraries(armigo_pack_ios PUBLIC armigo_runtime)
    target_compile_features(armigo_pack_ios PUBLIC cxx_std_20)
    if(BUILD_TESTING)
        add_subdirectory("${CMAKE_SOURCE_DIR}/tests/pack/ios" "${CMAKE_CURRENT_BINARY_DIR}/ios-tests")
    endif()
endif()
