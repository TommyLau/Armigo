# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

include_guard(GLOBAL)
include(FetchContent)
enable_language(C)

set(ARMIGO_DEPENDENCY_CACHE "$ENV{HOME}/Library/Caches/Armigo/dependencies" CACHE PATH
    "Archive and source cache outside the source tree")
cmake_path(ABSOLUTE_PATH ARMIGO_DEPENDENCY_CACHE NORMALIZE)
cmake_path(IS_PREFIX CMAKE_SOURCE_DIR "${ARMIGO_DEPENDENCY_CACHE}" NORMALIZE cache_in_source)
if(cache_in_source)
    message(FATAL_ERROR "ARMIGO_DEPENDENCY_CACHE must be outside the source tree.")
endif()

function(armigo_dependency_archive name version url hash output)
    string(TOUPPER "${name}" upper)
    set(ARMIGO_${upper}_ARCHIVE "" CACHE FILEPATH "Offline pinned ${name} release archive")
    file(MAKE_DIRECTORY "${ARMIGO_DEPENDENCY_CACHE}/archives")
    file(LOCK "${ARMIGO_DEPENDENCY_CACHE}/archives/${name}-${version}.lock" GUARD FUNCTION TIMEOUT 120)
    if(ARMIGO_${upper}_ARCHIVE)
        get_filename_component(archive "${ARMIGO_${upper}_ARCHIVE}" ABSOLUTE)
        if(NOT EXISTS "${archive}")
            message(FATAL_ERROR "The offline ${name} archive is missing: ${archive}")
        endif()
    else()
        set(archive "${ARMIGO_DEPENDENCY_CACHE}/archives/${name}-${version}.tar.gz")
        if(NOT EXISTS "${archive}")
            file(DOWNLOAD "${url}" "${archive}" EXPECTED_HASH "SHA256=${hash}" TLS_VERIFY ON STATUS status)
            list(GET status 0 code)
            if(NOT code EQUAL 0)
                message(FATAL_ERROR "Cannot download required ${name} archive: ${status}")
            endif()
        endif()
    endif()
    file(SHA256 "${archive}" actual)
    if(NOT actual STREQUAL hash)
        message(FATAL_ERROR "The ${name} archive SHA-256 does not match the pinned release.")
    endif()
    set(${output} "${archive}" PARENT_SCOPE)
endfunction()

set(ARMIGO_OPUS_VERSION 1.6.1)
set(ARMIGO_OPUS_SHA256 6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1)
set(ARMIGO_ZSTD_VERSION 1.5.7)
set(ARMIGO_ZSTD_SHA256 eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3)
armigo_dependency_archive(opus "${ARMIGO_OPUS_VERSION}"
    "https://downloads.xiph.org/releases/opus/opus-${ARMIGO_OPUS_VERSION}.tar.gz" "${ARMIGO_OPUS_SHA256}" opus_archive)
armigo_dependency_archive(zstd "${ARMIGO_ZSTD_VERSION}"
    "https://github.com/facebook/zstd/releases/download/v${ARMIGO_ZSTD_VERSION}/zstd-${ARMIGO_ZSTD_VERSION}.tar.gz"
    "${ARMIGO_ZSTD_SHA256}" zstd_archive)
string(SHA256 build_identity "${CMAKE_BINARY_DIR}")
set(source_cache "${ARMIGO_DEPENDENCY_CACHE}/sources/${build_identity}")

function(armigo_build_runtime_dependencies)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    # Apple arm64 requires NEON on both supported platforms. iOS leaves the
    # generic processor name empty, so use the validated target architecture.
    set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
    set(OPUS_USE_NEON ON)
    set(OPUS_MAY_HAVE_NEON ON)
    set(OPUS_PRESUME_NEON ON)
    set(BUILD_TESTING OFF)
    set(BUILD_SHARED_LIBS OFF)
    set(OPUS_BUILD_TESTING OFF)
    set(OPUS_BUILD_PROGRAMS OFF)
    set(OPUS_BUILD_SHARED_LIBRARY OFF)
    set(OPUS_BUILD_FRAMEWORK OFF)
    set(OPUS_INSTALL_PKG_CONFIG_MODULE OFF)
    set(OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF)
    set(OPUS_FIXED_POINT OFF)
    set(OPUS_FLOAT_APPROX OFF)
    set(OPUS_DISABLE_INTRINSICS OFF)
    set(OPUS_DRED OFF)
    set(OPUS_OSCE OFF)
    set(OPUS_DEEP_PLC OFF)
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_CONTRIB OFF)
    set(ZSTD_BUILD_TESTS OFF)
    set(ZSTD_BUILD_STATIC ON)
    set(ZSTD_BUILD_SHARED OFF)
    set(ZSTD_MULTITHREAD_SUPPORT ON)
    set(ZSTD_LEGACY_SUPPORT ON)
    set(ZSTD_BUILD_COMPRESSION ON)
    set(ZSTD_BUILD_DECOMPRESSION ON)
    set(ZSTD_BUILD_DICTBUILDER ON)
    foreach(dependency opus zstd)
        string(TOUPPER "${dependency}" upper)
        if(FETCHCONTENT_SOURCE_DIR_ARMIGO_${upper}_SOURCE)
            message(FATAL_ERROR "Source-directory overrides bypass archive verification; supply ARMIGO_${upper}_ARCHIVE instead.")
        endif()
    endforeach()
    FetchContent_Declare(armigo_opus_source URL "${opus_archive}" URL_HASH "SHA256=${ARMIGO_OPUS_SHA256}"
        SOURCE_DIR "${source_cache}/opus-${ARMIGO_OPUS_VERSION}" DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_Declare(armigo_zstd_source URL "${zstd_archive}" URL_HASH "SHA256=${ARMIGO_ZSTD_SHA256}"
        SOURCE_DIR "${source_cache}/zstd-${ARMIGO_ZSTD_VERSION}" SOURCE_SUBDIR build/cmake DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(armigo_opus_source armigo_zstd_source)
    set(ARMIGO_OPUS_SOURCE "${armigo_opus_source_SOURCE_DIR}" PARENT_SCOPE)
    set(ARMIGO_ZSTD_SOURCE "${armigo_zstd_source_SOURCE_DIR}" PARENT_SCOPE)
endfunction()
armigo_build_runtime_dependencies()
if(NOT TARGET opus OR NOT TARGET libzstd_static)
    message(FATAL_ERROR "A required player dependency is missing.")
endif()
add_library(zstd::libzstd_static ALIAS libzstd_static)
add_library(armigo_opus INTERFACE)
target_link_libraries(armigo_opus INTERFACE Opus::opus)

# Notices are generated from verified downloaded sources, never stored here.
file(READ "${ARMIGO_OPUS_SOURCE}/COPYING" ARMIGO_OPUS_NOTICE)
file(READ "${ARMIGO_ZSTD_SOURCE}/LICENSE" ARMIGO_ZSTD_NOTICE)
configure_file("${CMAKE_CURRENT_LIST_DIR}/dependency_notices.h"
    "${CMAKE_BINARY_DIR}/generated/armigo/pack/dependency_notices.h" @ONLY)
