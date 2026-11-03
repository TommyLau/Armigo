# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

add_executable(armigo_ios_link_fixture ios_link_tests.cpp)
target_link_libraries(armigo_ios_link_fixture PRIVATE armigo_pack_ios)
set(fixture_dir "${CMAKE_CURRENT_BINARY_DIR}/fixture")
add_custom_command(
    OUTPUT "${fixture_dir}/embedded.S" "${fixture_dir}/embedded.cpp"
    COMMAND armigo_ios_link_fixture --generate "${fixture_dir}"
    DEPENDS armigo_ios_link_fixture
    VERBATIM)
add_executable(armigo_ios_link_tests ios_link_tests.cpp
    "${fixture_dir}/embedded.S" "${fixture_dir}/embedded.cpp")
target_compile_definitions(armigo_ios_link_tests PRIVATE ARMIGO_TEST_LINKED=1)
target_link_libraries(armigo_ios_link_tests PRIVATE armigo_pack_ios)
add_test(NAME pack.ios_link COMMAND armigo_ios_link_tests "${fixture_dir}/images")
set_tests_properties(pack.ios_link PROPERTIES TIMEOUT 60)
add_test(NAME pack.ios_embedded_cache
    COMMAND "${CMAKE_COMMAND}"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/embedded_sources_cache"
            "-DMODULE=${CMAKE_SOURCE_DIR}/cmake/IosEmbeddedSources.cmake"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/embedded_sources_cache_tests.cmake")
set_tests_properties(pack.ios_embedded_cache PROPERTIES TIMEOUT 60)
