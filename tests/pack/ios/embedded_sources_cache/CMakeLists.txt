# ============================================================================
# Armigo Tests - Native Runtime Verification
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

cmake_minimum_required(VERSION 3.25)
project(embedded_sources_cache LANGUAGES C)
if(NOT DEFINED MODULE)
    message(FATAL_ERROR "MODULE is required")
endif()
include("${MODULE}")
file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${ARMIGO_IOS_EMBEDDED_SOURCES_DIR}")
file(WRITE "${CMAKE_BINARY_DIR}/empty.c" "void player_empty(void) {}\n")
add_library(player STATIC "${CMAKE_BINARY_DIR}/empty.c")
if(NOT ARMIGO_IOS_EMBEDDED_SOURCES_DIR STREQUAL "")
    set(embedded_source "${ARMIGO_IOS_EMBEDDED_SOURCES_DIR}/embedded.c")
    if(NOT EXISTS "${embedded_source}")
        message(FATAL_ERROR "Missing generated iOS embedded source: ${embedded_source}")
    endif()
    target_sources(player PRIVATE "${embedded_source}")
endif()
