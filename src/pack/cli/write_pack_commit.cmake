# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED OUTPUT)
    message(FATAL_ERROR "SOURCE_DIR and OUTPUT are required")
endif()

execute_process(
    COMMAND git -C "${SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE status)
if(NOT status EQUAL 0 OR NOT commit MATCHES "^[0-9a-fA-F]+$")
    set(commit "unknown")
endif()

set(temporary "${OUTPUT}.tmp")
file(WRITE "${temporary}" "#pragma once\n#define ARMIGO_PACK_COMMIT \"${commit}\"\n")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${temporary}" "${OUTPUT}")
file(REMOVE "${temporary}")
