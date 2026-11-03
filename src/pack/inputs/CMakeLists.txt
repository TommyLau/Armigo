# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

set(ARMIGO_PACK_INPUTS_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

add_library(armigo_pack_inputs STATIC inputs.cpp)
add_library(armigo::pack_inputs ALIAS armigo_pack_inputs)
target_include_directories(armigo_pack_inputs
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${ARMIGO_PACK_INPUTS_SOURCE_ROOT}/src/loader")
target_compile_features(armigo_pack_inputs PUBLIC cxx_std_20)
target_link_libraries(armigo_pack_inputs PUBLIC armigo::loader)

add_executable(armigo-pack-inputs-probe probe.cpp)
target_link_libraries(armigo-pack-inputs-probe PRIVATE armigo::pack_inputs)

if(BUILD_TESTING)
    add_executable(armigo_pack_inputs_tests
        "${ARMIGO_PACK_INPUTS_SOURCE_ROOT}/tests/pack/inputs/inputs_tests.cpp")
    target_include_directories(armigo_pack_inputs_tests
        PRIVATE "${ARMIGO_PACK_INPUTS_SOURCE_ROOT}/tests/loader/container")
    target_link_libraries(armigo_pack_inputs_tests PRIVATE armigo::pack_inputs)
    add_test(NAME pack.inputs COMMAND armigo_pack_inputs_tests)
    set_tests_properties(pack.inputs PROPERTIES LABELS "pack;loader" TIMEOUT 120)
endif()
