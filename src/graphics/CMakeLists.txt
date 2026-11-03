# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

cmake_path(SET ARMIGO_GRAPHICS_REPOSITORY_ROOT NORMALIZE
           "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(ARMIGO_GRAPHICS_INCLUDE_DIR "${ARMIGO_GRAPHICS_REPOSITORY_ROOT}/include")
set(ARMIGO_GRAPHICS_TEST_DIR "${ARMIGO_GRAPHICS_REPOSITORY_ROOT}/tests/graphics")

add_library(armigo_graphics memory.cpp macro.cpp channel.cpp engines.cpp compute_qmd.cpp upload.cpp color_clear.cpp twod_blit.cpp vertex_input.cpp draw_plan.cpp texture.cpp texture_layout.cpp dma_copy.cpp image_readback.cpp render_target.cpp)
target_sources(armigo_graphics PRIVATE texture_format.cpp render_submission.cpp resource_memory.cpp render_state.cpp texture_cache.cpp buffer_cache.cpp texflush.cpp)
add_library(armigo::graphics ALIAS armigo_graphics)
find_package(Threads REQUIRED)
# The coherence registry consults the process write tracker.
target_link_libraries(armigo_graphics PUBLIC Threads::Threads armigo_memory PRIVATE armigo_host_memory)
target_include_directories(armigo_graphics PUBLIC
    "$<BUILD_INTERFACE:${ARMIGO_GRAPHICS_INCLUDE_DIR}>")
target_compile_features(armigo_graphics PUBLIC cxx_std_20)
set_target_properties(armigo_graphics PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(armigo_graphics PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(armigo_graphics PRIVATE /W4 /permissive-)
endif()

if(NOT DEFINED BUILD_TESTING)
    option(BUILD_TESTING "Build the testing tree." ON)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_texture_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/texture_tests.cpp")
    target_link_libraries(armigo_graphics_texture_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.texture COMMAND armigo_graphics_texture_tests)
    set_tests_properties(graphics.texture PROPERTIES TIMEOUT 60)

    add_executable(armigo_graphics_vertex_input_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/vertex_input_tests.cpp")
    target_link_libraries(armigo_graphics_vertex_input_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.vertex_input COMMAND armigo_graphics_vertex_input_tests)
    set_tests_properties(graphics.vertex_input PROPERTIES TIMEOUT 60)

    add_executable(armigo_graphics_draw_plan_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/draw_plan_tests.cpp")
    target_link_libraries(armigo_graphics_draw_plan_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.draw_plan COMMAND armigo_graphics_draw_plan_tests)
    set_tests_properties(graphics.draw_plan PROPERTIES TIMEOUT 60)

    add_executable(armigo_graphics_color_clear_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/color_clear_tests.cpp")
    target_link_libraries(armigo_graphics_color_clear_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.color_clear COMMAND armigo_graphics_color_clear_tests)
    set_tests_properties(graphics.color_clear PROPERTIES TIMEOUT 60)

    add_executable(armigo_graphics_twod_blit_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/twod_blit_tests.cpp")
    target_link_libraries(armigo_graphics_twod_blit_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.twod_blit COMMAND armigo_graphics_twod_blit_tests)
    set_tests_properties(graphics.twod_blit PROPERTIES TIMEOUT 60)

    add_executable(armigo_graphics_render_target_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/render_target_tests.cpp")
    target_link_libraries(armigo_graphics_render_target_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.render_target COMMAND armigo_graphics_render_target_tests)
    set_tests_properties(graphics.render_target PROPERTIES TIMEOUT 60)

    add_executable(armigo_graphics_memory_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/memory_tests.cpp")
    target_link_libraries(armigo_graphics_memory_tests PRIVATE armigo_graphics)
    set_target_properties(armigo_graphics_memory_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(armigo_graphics_memory_tests PRIVATE
            -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME graphics.memory COMMAND armigo_graphics_memory_tests)
    set_tests_properties(graphics.memory PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_macro_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/macro_tests.cpp")
    target_link_libraries(armigo_graphics_macro_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.macro COMMAND armigo_graphics_macro_tests)
    set_tests_properties(graphics.macro PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_channel_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/channel_tests.cpp")
    target_link_libraries(armigo_graphics_channel_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.channel COMMAND armigo_graphics_channel_tests)
    set_tests_properties(graphics.channel PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_engine_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/engines_tests.cpp")
    target_link_libraries(armigo_graphics_engine_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.engines COMMAND armigo_graphics_engine_tests)
    set_tests_properties(graphics.engines PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_compute_qmd_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/compute_qmd_tests.cpp")
    target_include_directories(armigo_graphics_compute_qmd_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_graphics_compute_qmd_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.compute_qmd COMMAND armigo_graphics_compute_qmd_tests)
    set_tests_properties(graphics.compute_qmd PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_compute_launch_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/compute_launch_tests.cpp")
    target_link_libraries(armigo_graphics_compute_launch_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.compute_launch COMMAND armigo_graphics_compute_launch_tests)
    set_tests_properties(graphics.compute_launch PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_upload_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/upload_tests.cpp")
    target_link_libraries(armigo_graphics_upload_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.upload COMMAND armigo_graphics_upload_tests)
    set_tests_properties(graphics.upload PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_image_readback_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/image_readback_tests.cpp")
    target_link_libraries(armigo_graphics_image_readback_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.image_readback COMMAND armigo_graphics_image_readback_tests)
    set_tests_properties(graphics.image_readback PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_texture_layout_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/texture_layout_tests.cpp")
    target_link_libraries(armigo_graphics_texture_layout_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.texture_layout COMMAND armigo_graphics_texture_layout_tests)
    set_tests_properties(graphics.texture_layout PROPERTIES TIMEOUT 60)
    add_executable(armigo_graphics_dma_copy_tests "${ARMIGO_GRAPHICS_TEST_DIR}/dma_copy_tests.cpp")
    target_link_libraries(armigo_graphics_dma_copy_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.dma_copy COMMAND armigo_graphics_dma_copy_tests)
    set_tests_properties(graphics.dma_copy PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_renderer_contracts_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/renderer_contracts_tests.cpp")
    target_link_libraries(armigo_graphics_renderer_contracts_tests PRIVATE
        armigo_graphics Threads::Threads)
    add_test(NAME graphics.renderer_contracts COMMAND armigo_graphics_renderer_contracts_tests)
    set_tests_properties(graphics.renderer_contracts PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_render_state_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/render_state_tests.cpp")
    target_link_libraries(armigo_graphics_render_state_tests PRIVATE armigo_graphics)
    add_test(NAME graphics.render_state COMMAND armigo_graphics_render_state_tests)
    set_tests_properties(graphics.render_state PROPERTIES TIMEOUT 60)
    add_executable(armigo_graphics_resource_memory_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/resource_memory_tests.cpp")
    target_link_libraries(armigo_graphics_resource_memory_tests PRIVATE
        armigo_graphics Threads::Threads)
    add_test(NAME graphics.resource_memory COMMAND armigo_graphics_resource_memory_tests)
    set_tests_properties(graphics.resource_memory PROPERTIES TIMEOUT 60)
    add_executable(armigo_graphics_resource_memory_scale_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/resource_memory_scale_tests.cpp")
    target_link_libraries(armigo_graphics_resource_memory_scale_tests PRIVATE
        armigo_graphics Threads::Threads)
    add_test(NAME graphics.resource_memory_scale
        COMMAND armigo_graphics_resource_memory_scale_tests)
    set_tests_properties(graphics.resource_memory_scale PROPERTIES TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_graphics_buffer_cache_tests
        "${ARMIGO_GRAPHICS_TEST_DIR}/buffer_cache_tests.cpp")
    target_include_directories(armigo_graphics_buffer_cache_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_graphics_buffer_cache_tests PRIVATE
        armigo_graphics Threads::Threads)
    add_test(NAME graphics.buffer_cache COMMAND armigo_graphics_buffer_cache_tests)
    set_tests_properties(graphics.buffer_cache PROPERTIES TIMEOUT 60)
endif()
