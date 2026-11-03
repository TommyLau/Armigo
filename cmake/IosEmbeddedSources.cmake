# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# The packer sets ARMIGO_IOS_EMBEDDED_SOURCES_DIR in the environment of one
# configure. A cache entry cannot be distinguished from a value left by an
# earlier pack, so this include drops any cache entry and keeps the directory
# in a normal variable for this configure only.
unset(ARMIGO_IOS_EMBEDDED_SOURCES_DIR CACHE)
set(ARMIGO_IOS_EMBEDDED_SOURCES_DIR "")
if(DEFINED ENV{ARMIGO_IOS_EMBEDDED_SOURCES_DIR} AND NOT "$ENV{ARMIGO_IOS_EMBEDDED_SOURCES_DIR}" STREQUAL "")
    set(ARMIGO_IOS_EMBEDDED_SOURCES_DIR "$ENV{ARMIGO_IOS_EMBEDDED_SOURCES_DIR}")
endif()
