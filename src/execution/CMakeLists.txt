# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Native execution library: the AArch64 host gate and its dispatcher.
#
# The root build integrates this directory with add_subdirectory(src/execution).
# Paths are resolved relative to this file so the target also works from an
# external harness that adds this directory on its own.

set(ARMIGO_EXECUTION_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(ARMIGO_EXECUTION_ARCHS "${CMAKE_OSX_ARCHITECTURES}")
else()
    set(ARMIGO_EXECUTION_ARCHS "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(ARMIGO_EXECUTION_ARCHS MATCHES "^(arm64|aarch64)$" AND APPLE)
    set(ARMIGO_EXECUTION_HOST_GATE ON)
else()
    set(ARMIGO_EXECUTION_HOST_GATE OFF)
    message(STATUS "armigo_execution: host gate requires Apple arm64; "
                   "only the frame layout is built for '${ARMIGO_EXECUTION_ARCHS}'.")
endif()

if(ARMIGO_EXECUTION_HOST_GATE)
    enable_language(ASM)
    add_library(armigo_execution gate_entry.S gate_dispatch.cpp gate_code.cpp)
else()
    add_library(armigo_execution gate_frame_layout_check.cpp)
endif()
add_library(armigo::execution ALIAS armigo_execution)

add_library(armigo_execution_reachable_code reachable_code.cpp)
add_library(armigo::execution_reachable_code ALIAS armigo_execution_reachable_code)
target_link_libraries(armigo_execution_reachable_code PUBLIC armigo::aarch64)

target_include_directories(armigo_execution
    PUBLIC
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/include"
)
target_compile_features(armigo_execution PUBLIC cxx_std_20)
target_compile_definitions(armigo_execution
    PUBLIC ARMIGO_EXECUTION_HOST_GATE=$<BOOL:${ARMIGO_EXECUTION_HOST_GATE}>)

# Prepared gate-site dispatch for guest system registers and supervisor calls.
# It builds on the host gate and the bounded A64 decoder (src/aarch64).
if(ARMIGO_EXECUTION_HOST_GATE)
    add_library(armigo_execution_call_hook call_hook.cpp)
    add_library(armigo::execution_call_hook ALIAS armigo_execution_call_hook)
    target_compile_features(armigo_execution_call_hook PUBLIC cxx_std_20)
    target_link_libraries(armigo_execution_call_hook
        PUBLIC armigo::execution armigo::aarch64)
    add_library(armigo_execution_guest_dispatch guest_dispatch.cpp)
    add_library(armigo::execution_guest_dispatch ALIAS armigo_execution_guest_dispatch)
    target_compile_features(armigo_execution_guest_dispatch PUBLIC cxx_std_20)
    target_link_libraries(armigo_execution_guest_dispatch
        PUBLIC armigo::execution armigo::aarch64 armigo::execution_call_hook)
    add_library(armigo_execution_gate_patch gate_patch.cpp gate_site_template.S)
    add_library(armigo::execution_gate_patch ALIAS armigo_execution_gate_patch)
    target_link_libraries(armigo_execution_gate_patch PUBLIC armigo::execution_guest_dispatch)
    add_library(armigo_execution_indirect_gate_patch indirect_gate_patch.cpp)
    target_link_libraries(armigo_execution_indirect_gate_patch PUBLIC armigo::execution_guest_dispatch)
    add_library(armigo_execution_scalar_x18_patch scalar_x18_patch.cpp)
    target_link_libraries(armigo_execution_scalar_x18_patch PUBLIC armigo::aarch64)
    add_library(armigo_execution_conditional_x18_patch conditional_x18_patch.cpp)
    target_link_libraries(armigo_execution_conditional_x18_patch PUBLIC armigo::aarch64)
    add_library(armigo_execution_indirect_x18_patch indirect_x18_patch.cpp)
    target_link_libraries(armigo_execution_indirect_x18_patch PUBLIC armigo::aarch64)
    add_library(armigo_execution_embedded_patch embedded_patch.cpp)
    target_link_libraries(armigo_execution_embedded_patch PUBLIC armigo_execution
        armigo_execution_scalar_x18_patch armigo_execution_conditional_x18_patch
        armigo_execution_indirect_x18_patch)
    add_library(armigo_execution_prepared_image prepared_image.cpp)
    target_link_libraries(armigo_execution_prepared_image PUBLIC
        armigo_execution_gate_patch armigo_execution_indirect_gate_patch
        armigo_execution_scalar_x18_patch armigo_execution_conditional_x18_patch
        armigo_execution_indirect_x18_patch)
endif()

if(BUILD_TESTING AND ARMIGO_EXECUTION_HOST_GATE)
    add_executable(armigo_indirect_x18_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/indirect_x18_tests.cpp"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/indirect_x18_tests.S")
    target_link_libraries(armigo_indirect_x18_tests PRIVATE
        armigo_execution_indirect_x18_patch armigo_execution_guest_dispatch
        armigo_host_memory armigo_thread_slot)
    add_test(NAME execution.indirect_x18 COMMAND armigo_indirect_x18_tests)
    set_tests_properties(execution.indirect_x18 PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_conditional_x18_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/conditional_x18_tests.cpp"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/conditional_x18_tests.S")
    target_link_libraries(armigo_conditional_x18_tests PRIVATE
        armigo_execution_conditional_x18_patch armigo_execution_embedded_patch armigo_execution_guest_dispatch
        armigo_host_memory armigo_thread_slot)
    add_test(NAME execution.conditional_x18 COMMAND armigo_conditional_x18_tests)
    set_tests_properties(execution.conditional_x18 PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_indirect_gate_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/indirect_gate_tests.cpp"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/indirect_gate_tests.S")
    target_link_libraries(armigo_indirect_gate_tests
        PRIVATE armigo_execution_indirect_gate_patch armigo_host_memory)
    add_test(NAME execution.indirect_gate COMMAND armigo_indirect_gate_tests)
    set_tests_properties(execution.indirect_gate PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_call_hook_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/call_hook_tests.cpp"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/call_hook_fixture.S")
    target_link_libraries(armigo_call_hook_tests
        PRIVATE armigo_execution_guest_dispatch armigo_host_memory)
    add_test(NAME execution.call_hook COMMAND armigo_call_hook_tests)
    set_tests_properties(execution.call_hook PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_execution_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/fixtures.S"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/gate_tests.cpp"
    )
    target_compile_features(armigo_execution_tests PRIVATE cxx_std_20)
    target_link_libraries(armigo_execution_tests PRIVATE armigo::execution)
    add_test(NAME execution.host_gate COMMAND armigo_execution_tests)
    set_tests_properties(execution.host_gate PROPERTIES LABELS "execution;arm64" TIMEOUT 120)

    add_executable(armigo_execution_guest_dispatch_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/guest_dispatch_fixtures.S"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/guest_dispatch_tests.cpp"
    )
    target_compile_features(armigo_execution_guest_dispatch_tests PRIVATE cxx_std_20)
    target_link_libraries(armigo_execution_guest_dispatch_tests
        PRIVATE armigo::execution_guest_dispatch)
    add_test(NAME execution.guest_dispatch COMMAND armigo_execution_guest_dispatch_tests)
    set_tests_properties(execution.guest_dispatch
        PROPERTIES LABELS "execution;arm64" TIMEOUT 120)
endif()

if(BUILD_TESTING AND ARMIGO_EXECUTION_HOST_GATE)
    add_executable(armigo_host_callback_stack_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/host_callback_stack_tests.cpp"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/host_callback_stack_fixture.S")
    target_include_directories(armigo_host_callback_stack_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_host_callback_stack_tests PRIVATE armigo_execution
        armigo_guest_stack armigo_host_memory)
    add_test(NAME execution.host_callback_stack COMMAND armigo_host_callback_stack_tests)
    set_tests_properties(execution.host_callback_stack PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_guest_exit_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/guest_exit_tests.cpp"
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/guest_exit_fixture.S")
    target_link_libraries(armigo_guest_exit_tests PRIVATE armigo_execution_guest_entry
        armigo_execution_gate_patch armigo_host_memory armigo_thread_slot)
    add_test(NAME execution.guest_exit COMMAND armigo_guest_exit_tests)
    set_tests_properties(execution.guest_exit PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
endif()

if(BUILD_TESTING AND ARMIGO_EXECUTION_HOST_GATE)
    add_executable(armigo_embedded_patch_tests
        "${ARMIGO_EXECUTION_SOURCE_ROOT}/tests/execution/embedded_patch_tests.cpp")
    target_link_libraries(armigo_embedded_patch_tests PRIVATE
        armigo_execution_embedded_patch armigo_execution_guest_dispatch
        armigo_host_memory armigo_thread_slot)
    add_test(NAME execution.embedded_patch COMMAND armigo_embedded_patch_tests)
    set_tests_properties(execution.embedded_patch PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
endif()
