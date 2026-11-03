# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Offline AArch64 instruction preparation primitives. The library is portable
# C++20 and does not execute guest code; only the execution test needs an
# arm64 host.
#
# The root build integrates this directory with add_subdirectory(src/aarch64).

set(ARMIGO_AARCH64_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(ARMIGO_AARCH64_HOST_ARCHS "${CMAKE_OSX_ARCHITECTURES}")
else()
    set(ARMIGO_AARCH64_HOST_ARCHS "${CMAKE_SYSTEM_PROCESSOR}")
endif()
# Assembler fixtures use Mach-O syntax and the execution test runs arm64 code.
if(APPLE AND ARMIGO_AARCH64_HOST_ARCHS MATCHES "^(arm64|aarch64)$")
    set(ARMIGO_AARCH64_NATIVE_TESTS ON)
else()
    set(ARMIGO_AARCH64_NATIVE_TESTS OFF)
endif()

add_library(armigo_aarch64 STATIC
    instruction.cpp
    encode.cpp
    relocation.cpp
    analysis.cpp
    register_effects.cpp
    register_substitution.cpp
)
add_library(armigo::aarch64 ALIAS armigo_aarch64)
target_include_directories(armigo_aarch64 PUBLIC "${ARMIGO_AARCH64_SOURCE_ROOT}/include")
target_compile_features(armigo_aarch64 PUBLIC cxx_std_20)

if(BUILD_TESTING)
    set(ARMIGO_AARCH64_TESTS "${ARMIGO_AARCH64_SOURCE_ROOT}/tests/aarch64")

    add_executable(armigo_aarch64_instruction_tests "${ARMIGO_AARCH64_TESTS}/instruction_tests.cpp")
    target_link_libraries(armigo_aarch64_instruction_tests PRIVATE armigo::aarch64)
    add_test(NAME aarch64.instruction COMMAND armigo_aarch64_instruction_tests)

    add_executable(armigo_aarch64_relocation_tests "${ARMIGO_AARCH64_TESTS}/relocation_tests.cpp")
    target_link_libraries(armigo_aarch64_relocation_tests PRIVATE armigo::aarch64)
    add_test(NAME aarch64.relocation COMMAND armigo_aarch64_relocation_tests)

    add_executable(armigo_aarch64_analysis_tests "${ARMIGO_AARCH64_TESTS}/analysis_tests.cpp")
    target_link_libraries(armigo_aarch64_analysis_tests PRIVATE armigo::aarch64)
    add_test(NAME aarch64.analysis COMMAND armigo_aarch64_analysis_tests)

    add_executable(armigo_aarch64_register_effects_tests
        "${ARMIGO_AARCH64_TESTS}/register_effects_tests.cpp")
    target_link_libraries(armigo_aarch64_register_effects_tests PRIVATE armigo::aarch64)
    add_test(NAME aarch64.register_effects COMMAND armigo_aarch64_register_effects_tests)

    add_executable(armigo_aarch64_register_substitution_tests
        "${ARMIGO_AARCH64_TESTS}/register_substitution_tests.cpp")
    target_link_libraries(armigo_aarch64_register_substitution_tests PRIVATE armigo::aarch64)
    add_test(NAME aarch64.register_substitution
        COMMAND armigo_aarch64_register_substitution_tests)

    set_tests_properties(aarch64.instruction aarch64.relocation aarch64.analysis
        aarch64.register_effects aarch64.register_substitution
        PROPERTIES LABELS "aarch64" TIMEOUT 60)

    if(ARMIGO_AARCH64_NATIVE_TESTS)
        enable_language(ASM)
        add_executable(armigo_aarch64_assembler_tests
            "${ARMIGO_AARCH64_TESTS}/assembler_fixture.S"
            "${ARMIGO_AARCH64_TESTS}/assembler_fixture_tests.cpp")
        target_link_libraries(armigo_aarch64_assembler_tests PRIVATE armigo::aarch64)
        add_test(NAME aarch64.assembler_fixture COMMAND armigo_aarch64_assembler_tests)

        add_executable(armigo_aarch64_relocation_execution_tests
            "${ARMIGO_AARCH64_TESTS}/relocation_execution_tests.cpp")
        target_link_libraries(armigo_aarch64_relocation_execution_tests PRIVATE armigo::aarch64)
        add_test(NAME aarch64.relocation_execution COMMAND armigo_aarch64_relocation_execution_tests)

        add_executable(armigo_aarch64_register_effects_fixture_tests
            "${ARMIGO_AARCH64_TESTS}/register_effects_fixture.S"
            "${ARMIGO_AARCH64_TESTS}/register_effects_fixture_tests.cpp")
        target_link_libraries(armigo_aarch64_register_effects_fixture_tests
            PRIVATE armigo::aarch64)
        add_test(NAME aarch64.register_effects_fixture
            COMMAND armigo_aarch64_register_effects_fixture_tests)

        add_executable(armigo_aarch64_register_substitution_fixture_tests
            "${ARMIGO_AARCH64_TESTS}/register_substitution_fixture.S"
            "${ARMIGO_AARCH64_TESTS}/register_substitution_fixture_tests.cpp")
        target_link_libraries(armigo_aarch64_register_substitution_fixture_tests
            PRIVATE armigo::aarch64)
        add_test(NAME aarch64.register_substitution_fixture
            COMMAND armigo_aarch64_register_substitution_fixture_tests)

        set_tests_properties(aarch64.assembler_fixture aarch64.relocation_execution
            aarch64.register_effects_fixture aarch64.register_substitution_fixture
            PROPERTIES LABELS "aarch64;arm64" TIMEOUT 60)
    endif()
endif()
