# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

set(ARMIGO_HORIZON_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_library(armigo_horizon STATIC handle_table.cpp startup_queries.cpp ipc.cpp event.cpp
    arbitration.cpp process_wide_key.cpp shared_memory.cpp stack_alias.cpp thread_lifecycle.cpp)
target_include_directories(armigo_horizon PUBLIC "${ARMIGO_HORIZON_SOURCE_ROOT}/include")
target_compile_features(armigo_horizon PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(armigo_horizon PUBLIC Threads::Threads)

if(BUILD_TESTING)
    add_executable(armigo_stack_alias_tests
        "${ARMIGO_HORIZON_SOURCE_ROOT}/tests/horizon/stack_alias_tests.cpp")
    target_link_libraries(armigo_stack_alias_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.stack_alias COMMAND armigo_stack_alias_tests)
    set_tests_properties(horizon.stack_alias PROPERTIES LABELS "horizon" TIMEOUT 30)
    add_executable(armigo_event_tests
        "${ARMIGO_HORIZON_SOURCE_ROOT}/tests/horizon/event_tests.cpp")
    target_link_libraries(armigo_event_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.startup_event COMMAND armigo_event_tests)
    set_tests_properties(horizon.startup_event PROPERTIES LABELS "horizon" TIMEOUT 30)
    add_executable(armigo_ipc_tests
        "${ARMIGO_HORIZON_SOURCE_ROOT}/tests/horizon/ipc_tests.cpp")
    target_link_libraries(armigo_ipc_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.ipc COMMAND armigo_ipc_tests)
    set_tests_properties(horizon.ipc PROPERTIES LABELS "horizon" TIMEOUT 30)
    add_executable(armigo_handle_table_tests
        "${ARMIGO_HORIZON_SOURCE_ROOT}/tests/horizon/handle_table_tests.cpp")
    target_link_libraries(armigo_handle_table_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.handle_table COMMAND armigo_handle_table_tests)
    set_tests_properties(horizon.handle_table PROPERTIES LABELS "horizon" TIMEOUT 30)
    add_executable(armigo_startup_query_tests
        "${ARMIGO_HORIZON_SOURCE_ROOT}/tests/horizon/startup_queries_tests.cpp")
    target_link_libraries(armigo_startup_query_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.startup_queries COMMAND armigo_startup_query_tests)
    set_tests_properties(horizon.startup_queries PROPERTIES LABELS "horizon" TIMEOUT 30)
endif()
