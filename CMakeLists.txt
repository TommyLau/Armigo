# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum Apple OS version")
endif()
if(NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architecture")
endif()

project(Armigo VERSION 0.1.0 LANGUAGES CXX OBJCXX ASM)

if(NOT APPLE OR NOT CMAKE_SYSTEM_NAME MATCHES "^(Darwin|iOS)$")
    message(FATAL_ERROR "Armigo requires an Apple macOS or iOS target.")
endif()
set(ARMIGO_IOS OFF)
set(ARMIGO_PLATFORM_DIRECTORY macos)
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(ARMIGO_IOS ON)
    set(ARMIGO_PLATFORM_DIRECTORY ios)
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET VERSION_EQUAL "26.0")
        message(FATAL_ERROR "The iOS player deployment target must be 26.0.")
    endif()
endif()
if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "26.0")
    message(FATAL_ERROR "Armigo requires macOS or iOS 26.0 or later.")
endif()
if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    message(FATAL_ERROR "The initial Armigo host targets Apple Silicon arm64.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic -Werror=unguarded-availability-new)
add_compile_options("$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
add_compile_options(
    "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-ffile-prefix-map=${CMAKE_BINARY_DIR}=./build")

include(CTest)
if(ARMIGO_IOS AND BUILD_TESTING)
    message(FATAL_ERROR "Device builds require BUILD_TESTING=OFF; run synthetic tests on macOS.")
endif()
option(ARMIGO_ENABLE_GPU_TESTS "Run tests that require an available Metal 4 GPU" OFF)

add_subdirectory(src/common)
link_libraries(armigo_diagnostics)

add_subdirectory(src/shader_input)
add_subdirectory(src/shader)
add_subdirectory(src/graphics)
add_subdirectory(src/memory)
add_subdirectory(src/linker)
add_subdirectory(src/loader)
add_subdirectory(src/execution)
add_subdirectory(src/aarch64)
add_subdirectory(src/horizon)
add_subdirectory(src/audio)
add_subdirectory(src/pack)

if(ARMIGO_IOS)
    set_property(TARGET armigo-nso-inspect PROPERTY EXCLUDE_FROM_ALL TRUE)
    set(ARMIGO_HOST_SHADER_TRANSLATE "" CACHE FILEPATH "Built macOS armigo-shader-translate executable")
    if(NOT EXISTS "${ARMIGO_HOST_SHADER_TRANSLATE}")
        message(FATAL_ERROR "Set ARMIGO_HOST_SHADER_TRANSLATE to a built macOS translator.")
    endif()
    add_executable(armigo-shader-translate IMPORTED GLOBAL)
    set_target_properties(armigo-shader-translate PROPERTIES IMPORTED_LOCATION "${ARMIGO_HOST_SHADER_TRANSLATE}")
else()
    add_executable(armigo-shader-extract src/shader_input/tools/extract.cpp)
    target_include_directories(armigo-shader-extract PRIVATE src/loader)
    target_link_libraries(armigo-shader-extract PRIVATE armigo_shader_input armigo_loader)
    target_compile_features(armigo-shader-extract PRIVATE cxx_std_20)

    add_executable(armigo-shader-translate src/shader/tools/translate.cpp)
    target_include_directories(armigo-shader-translate PRIVATE src/loader)
    target_link_libraries(armigo-shader-translate PRIVATE armigo_shader armigo_loader)
    # Lane 15: the ported pipeline path of the translator CLI.
    target_link_libraries(armigo-shader-translate PRIVATE armigo_shader_pipeline)
    target_link_libraries(armigo-shader-translate PRIVATE armigo_shader_translation)
    target_compile_features(armigo-shader-translate PRIVATE cxx_std_20)

    add_library(armigo_shader_ios_package STATIC src/shader/package_ios.cpp)
    target_include_directories(armigo_shader_ios_package PUBLIC include)
    target_link_libraries(armigo_shader_ios_package
        PRIVATE armigo_shader_package armigo_shader_translation)
    target_compile_features(armigo_shader_ios_package PUBLIC cxx_std_20)

    add_executable(armigo-shader-package-ios src/shader/tools/package_ios.cpp)
    target_link_libraries(armigo-shader-package-ios PRIVATE armigo_shader_ios_package)
    target_compile_features(armigo-shader-package-ios PRIVATE cxx_std_20)

    if(BUILD_TESTING)
        add_executable(armigo_shader_ios_package_tests tests/shader/package/ios_package_tests.cpp)
        target_link_libraries(armigo_shader_ios_package_tests
            PRIVATE armigo_shader_ios_package armigo_shader_translation)
        add_test(NAME shader.ios_package COMMAND armigo_shader_ios_package_tests)
        set_tests_properties(shader.ios_package PROPERTIES LABELS "shader;pack" TIMEOUT 60)
    endif()

endif()

find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)

add_library(armigo_metal_probe STATIC src/platform/metal/headless_probe.mm)
target_include_directories(armigo_metal_probe PUBLIC include)
target_link_libraries(armigo_metal_probe
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_color_clear STATIC src/platform/metal/color_clear.mm)
target_include_directories(armigo_metal_color_clear PUBLIC include)
target_link_libraries(armigo_metal_color_clear PUBLIC armigo_graphics
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_draw STATIC src/platform/metal/draw.mm)
target_include_directories(armigo_metal_draw PUBLIC include)
target_link_libraries(armigo_metal_draw PUBLIC armigo_graphics armigo_shader
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_buffer_cache STATIC src/platform/metal/buffer_cache.mm)
target_include_directories(armigo_metal_buffer_cache PUBLIC include PRIVATE src/platform/metal)
target_link_libraries(armigo_metal_buffer_cache PUBLIC armigo_graphics
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_texture_format STATIC src/platform/metal/texture_format.mm)
target_include_directories(armigo_metal_texture_format PUBLIC include)
target_link_libraries(armigo_metal_texture_format PUBLIC armigo_graphics
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
target_link_libraries(armigo_metal_draw PUBLIC armigo_metal_texture_format)

add_library(armigo_metal_attachments STATIC src/platform/metal/attachments.mm)
target_include_directories(armigo_metal_attachments PUBLIC include src/platform/metal)
target_link_libraries(armigo_metal_attachments PUBLIC armigo_graphics
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_clear_encoder STATIC src/platform/metal/clear_encoder.mm)
target_include_directories(armigo_metal_clear_encoder PUBLIC include src/platform/metal)
target_link_libraries(armigo_metal_clear_encoder PUBLIC armigo_metal_attachments
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

# Lane: GPU block-linear texture swizzle pipelines shared by texture transfers.
add_library(armigo_metal_texture_swizzle STATIC src/platform/metal/texture_swizzle_gpu.mm)
target_include_directories(armigo_metal_texture_swizzle PUBLIC include src/platform/metal)
target_link_libraries(armigo_metal_texture_swizzle PUBLIC armigo_graphics
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_twod_blit STATIC src/platform/metal/twod_blit.mm)
target_include_directories(armigo_metal_twod_blit PUBLIC include src/platform/metal)
target_link_libraries(armigo_metal_twod_blit PUBLIC armigo_graphics
    armigo_metal_attachments
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_texture_cache STATIC src/platform/metal/texture_cache.mm)
target_include_directories(armigo_metal_texture_cache PUBLIC include)
target_link_libraries(armigo_metal_texture_cache PUBLIC armigo_graphics armigo_metal_texture_format
    armigo_metal_texture_swizzle
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_pipeline_state STATIC src/platform/metal/pipeline_state.mm)
target_include_directories(armigo_metal_pipeline_state PUBLIC include src/platform/metal)
target_link_libraries(armigo_metal_pipeline_state PUBLIC armigo_graphics
    armigo_metal_texture_format armigo_metal_attachments
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_compute_dispatch STATIC
    src/platform/metal/compute_dispatch.mm)
target_include_directories(armigo_metal_compute_dispatch PUBLIC include
    PRIVATE src/platform/metal src/graphics)
target_link_libraries(armigo_metal_compute_dispatch PUBLIC armigo_graphics armigo_metal_texture_format
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_metal_shader_package STATIC src/platform/metal/shader_package.mm)
target_include_directories(armigo_metal_shader_package PUBLIC include PRIVATE src/loader)
target_link_libraries(armigo_metal_shader_package PUBLIC armigo_graphics armigo_shader
    PRIVATE armigo_loader "${ARMIGO_FOUNDATION_FRAMEWORK}")

add_library(armigo_metal_renderer STATIC
    src/platform/metal/device_context.mm
    src/platform/metal/renderer.mm
    src/platform/metal/submission.mm
    src/platform/metal/bindings.mm
    src/platform/metal/draw_encoding.mm)
target_include_directories(armigo_metal_renderer PUBLIC include src/platform/metal)
target_link_libraries(armigo_metal_renderer PUBLIC armigo_graphics
    armigo_metal_buffer_cache armigo_metal_texture_cache armigo_metal_clear_encoder
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

find_library(ARMIGO_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
if(ARMIGO_IOS)
    find_library(ARMIGO_UI_FRAMEWORK UIKit REQUIRED)
else()
    find_library(ARMIGO_APPKIT_FRAMEWORK AppKit REQUIRED)
    set(ARMIGO_UI_FRAMEWORK "${ARMIGO_APPKIT_FRAMEWORK}")
endif()

# Lane H: Metal presenter (headless readback and CAMetalLayer drawable present).
add_library(armigo_platform_presenter STATIC src/platform/metal/presenter.mm)
target_include_directories(armigo_platform_presenter PUBLIC include)
target_link_libraries(armigo_platform_presenter
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}"
    "${ARMIGO_QUARTZCORE_FRAMEWORK}")

# Apple application owning the CAMetalLayer-backed window.
add_library(armigo_platform_application STATIC src/platform/${ARMIGO_PLATFORM_DIRECTORY}/application.mm
    src/platform/macos/display_refresh.mm)
if(NOT ARMIGO_IOS)
    find_library(ARMIGO_GAME_CONTROLLER_FRAMEWORK GameController REQUIRED)
    target_sources(armigo_platform_application PRIVATE src/platform/macos/key_guide.mm)
    target_link_libraries(armigo_platform_application PRIVATE "${ARMIGO_GAME_CONTROLLER_FRAMEWORK}")
endif()
if(ARMIGO_IOS)
    target_sources(armigo_platform_application PRIVATE src/platform/ios/touch_controls.mm)
    target_link_libraries(armigo_platform_application PRIVATE armigo_host_input)
endif()
target_include_directories(armigo_platform_application PUBLIC include)
target_link_libraries(armigo_platform_application PUBLIC armigo_platform_presenter
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_UI_FRAMEWORK}"
    "${ARMIGO_QUARTZCORE_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
add_library(armigo_metal_engine_renderer STATIC
    src/platform/metal/engine_renderer.mm
    src/platform/metal/engine_pipeline.mm)
target_include_directories(armigo_metal_engine_renderer PUBLIC include PRIVATE src/platform/metal)
target_link_libraries(armigo_metal_engine_renderer PUBLIC armigo_metal_renderer
    armigo_metal_pipeline_state
    armigo_metal_compute_dispatch armigo_metal_draw armigo_metal_color_clear armigo_metal_twod_blit
    PRIVATE "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")

add_library(armigo_host_memory STATIC src/platform/host_module.cpp src/platform/process_memory.cpp)

target_include_directories(armigo_host_memory PUBLIC include)

add_library(armigo_thread_slot STATIC src/platform/thread_slot.cpp)
target_include_directories(armigo_thread_slot PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(armigo_thread_slot PUBLIC Threads::Threads)

add_library(armigo_guest_stack STATIC src/platform/guest_stack.cpp)
target_include_directories(armigo_guest_stack PUBLIC include)

add_library(armigo_execution_guest_entry STATIC
    src/execution/guest_entry.cpp src/execution/guest_preempt.cpp src/execution/guest_entry.S)
target_link_libraries(armigo_execution_guest_entry PUBLIC armigo_execution armigo_guest_stack)

add_library(armigo_services_time STATIC src/services/time.cpp)
target_include_directories(armigo_services_time PUBLIC include)
target_link_libraries(armigo_services_time PUBLIC Threads::Threads)

add_library(armigo_services_sm STATIC
    src/services/service_manager.cpp src/services/service_manager_session.cpp)
target_include_directories(armigo_services_sm PUBLIC include)
target_link_libraries(armigo_services_sm PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_lm STATIC src/services/log_manager.cpp)
target_include_directories(armigo_services_lm PUBLIC include)
target_link_libraries(armigo_services_lm PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_apm STATIC src/services/performance.cpp)
add_library(armigo_services_hid STATIC src/services/hid.cpp src/services/hid_shared_memory.cpp
    src/services/hid_motion.cpp)
target_include_directories(armigo_services_hid PUBLIC include)
target_link_libraries(armigo_services_hid PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_nfp STATIC src/services/nfp.cpp)
target_include_directories(armigo_services_nfp PUBLIC include)
target_link_libraries(armigo_services_nfp PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_audren STATIC src/services/audio_renderer.cpp)
target_include_directories(armigo_services_audren PUBLIC include)
target_link_libraries(armigo_services_audren PUBLIC armigo_horizon armigo_audio_renderer Threads::Threads)
target_include_directories(armigo_services_apm PUBLIC include)
target_link_libraries(armigo_services_apm PUBLIC armigo_horizon Threads::Threads)

# Host PCM output: the default output device through a CoreAudio AudioUnit.
find_library(ARMIGO_AUDIOTOOLBOX_FRAMEWORK AudioToolbox REQUIRED)
add_library(armigo_platform_pcm_sink STATIC src/platform/macos/pcm_sink.mm)
target_include_directories(armigo_platform_pcm_sink PUBLIC include)
target_link_libraries(armigo_platform_pcm_sink PUBLIC armigo_audio_renderer
    PRIVATE "${ARMIGO_AUDIOTOOLBOX_FRAMEWORK}")
if(ARMIGO_IOS)
    find_library(ARMIGO_AVFAUDIO_FRAMEWORK AVFAudio REQUIRED)
    target_link_libraries(armigo_platform_pcm_sink PRIVATE "${ARMIGO_AVFAUDIO_FRAMEWORK}")
endif()

add_library(armigo_services_am STATIC src/services/applet.cpp src/services/software_keyboard.cpp)
target_include_directories(armigo_services_am PUBLIC include)
target_link_libraries(armigo_services_am PUBLIC armigo_horizon armigo_services_settings Threads::Threads)

add_library(armigo_services_ro STATIC src/services/loader.cpp)
target_include_directories(armigo_services_ro PUBLIC include)
target_link_libraries(armigo_services_ro PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_vi STATIC src/services/video_interface.cpp
    src/services/vsync_source.cpp)
target_include_directories(armigo_services_vi PUBLIC include)
target_link_libraries(armigo_services_vi PUBLIC armigo_horizon armigo_runtime
    armigo_platform_display Threads::Threads)

# Host VP9 decoding for the video decoder engine (VideoToolbox).
add_library(armigo_platform_video STATIC src/platform/video/videotoolbox_vp9.mm)
target_include_directories(armigo_platform_video PUBLIC include)
target_link_libraries(armigo_platform_video PUBLIC "-framework VideoToolbox" "-framework CoreMedia"
    "-framework CoreVideo" "-framework CoreFoundation")
# Multimedia engines (video decoder, video image compositor) and their command streams.
add_library(armigo_host1x STATIC src/host1x/host1x.cpp src/host1x/decoder_engine.cpp
    src/host1x/compositor_engine.cpp src/host1x/vp9.cpp)
target_include_directories(armigo_host1x PUBLIC include)
target_link_libraries(armigo_host1x PUBLIC armigo_platform_video armigo_graphics Threads::Threads)
add_library(armigo_services_nvdrv STATIC src/services/nvdrv.cpp)
target_include_directories(armigo_services_nvdrv PUBLIC include)
target_link_libraries(armigo_services_nvdrv PUBLIC armigo_horizon armigo_graphics armigo_host1x Threads::Threads)

add_library(armigo_services_acc STATIC src/services/account.cpp)
target_include_directories(armigo_services_acc PUBLIC include)
target_link_libraries(armigo_services_acc PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_aoc STATIC src/services/addon_content.cpp)
target_include_directories(armigo_services_aoc PUBLIC include)
target_link_libraries(armigo_services_aoc PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_context_writer STATIC src/services/context_writer.cpp)
target_include_directories(armigo_services_context_writer PUBLIC include)
target_link_libraries(armigo_services_context_writer PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_pctl STATIC src/services/parental_control.cpp)
target_include_directories(armigo_services_pctl PUBLIC include)
target_link_libraries(armigo_services_pctl PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_audout STATIC src/services/audio_out.cpp)
target_include_directories(armigo_services_audout PUBLIC include)
target_link_libraries(armigo_services_audout PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_bsd STATIC src/services/bsd.cpp)
target_include_directories(armigo_services_bsd PUBLIC include)
target_link_libraries(armigo_services_bsd PUBLIC armigo_horizon Threads::Threads)
add_library(armigo_services_nifm STATIC src/services/nifm.cpp)
target_include_directories(armigo_services_nifm PUBLIC include)
target_link_libraries(armigo_services_nifm PUBLIC armigo_horizon Threads::Threads)
add_library(armigo_services_mm STATIC src/services/mm.cpp)
target_include_directories(armigo_services_mm PUBLIC include)
target_link_libraries(armigo_services_mm PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_settings STATIC src/services/settings.cpp)
target_include_directories(armigo_services_settings PUBLIC include)
target_link_libraries(armigo_services_settings PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_shared_font STATIC src/services/shared_font.cpp)
target_include_directories(armigo_services_shared_font PUBLIC include)
target_link_libraries(armigo_services_shared_font PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_prepo STATIC src/services/play_report.cpp
    src/services/play_report_validation.cpp)
target_include_directories(armigo_services_prepo PUBLIC include)
target_link_libraries(armigo_services_prepo PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_time_session STATIC src/services/time_session.cpp)
target_include_directories(armigo_services_time_session PUBLIC include)
target_link_libraries(armigo_services_time_session PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_file_system STATIC src/services/file_system.cpp)
target_include_directories(armigo_services_file_system PUBLIC include)
target_link_libraries(armigo_services_file_system PUBLIC armigo_horizon Threads::Threads)

add_library(armigo_services_shared_clock STATIC src/services/shared_clock.cpp)
target_include_directories(armigo_services_shared_clock PUBLIC include)

add_library(armigo_runtime STATIC src/runtime/module_set.cpp src/runtime/kernel_queries.cpp
    src/runtime/launch_profile.cpp src/runtime/guest_clock.cpp src/runtime/presentation_clock.cpp
    src/runtime/frame_rate_governor.cpp src/runtime/guest_memory.cpp src/runtime/thread_runtime.cpp src/runtime/presentation.cpp
    src/runtime/prepared_modules.cpp src/runtime/preparation_analysis.cpp src/runtime/launch_plan.cpp
    src/runtime/process_runtime.cpp src/runtime/service_registry.cpp src/runtime/addon_content.cpp
    src/runtime/native_preparation.cpp src/runtime/native_launch.cpp src/runtime/embedded_modules.cpp src/runtime/host_controller.cpp)
target_sources(armigo_horizon PRIVATE src/horizon/process_memory.cpp)
target_include_directories(armigo_runtime PUBLIC include)
# The launch profile consumes the loader's internal NPDM structural reader.
target_include_directories(armigo_runtime PRIVATE src/loader)
target_link_libraries(armigo_runtime PUBLIC armigo_loader armigo_linker armigo_host_memory
    armigo_horizon armigo_execution_guest_dispatch armigo_graphics armigo_execution_prepared_image)
target_link_libraries(armigo_runtime PRIVATE Threads::Threads armigo_execution_guest_entry armigo_thread_slot
    armigo_aarch64 armigo_execution_gate_patch armigo_execution_embedded_patch armigo_execution_scalar_x18_patch
    armigo_execution_conditional_x18_patch armigo_nvn_metal)
target_link_libraries(armigo_runtime PUBLIC armigo_host_input armigo_services_hid)
target_link_libraries(armigo_runtime PRIVATE armigo_romfs armigo_metal_engine_renderer
    armigo_metal_shader_package armigo_services_sm armigo_services_lm armigo_services_apm
    armigo_services_hid armigo_services_nfp armigo_services_audren armigo_services_am
    armigo_services_ro armigo_services_vi armigo_services_nvdrv armigo_services_acc
    armigo_services_aoc armigo_services_context_writer armigo_services_pctl armigo_services_settings armigo_services_audout armigo_services_bsd armigo_services_nifm armigo_services_mm armigo_services_prepo
    armigo_services_time_session armigo_services_file_system armigo_services_shared_clock
    armigo_services_shared_font armigo_platform_pcm_sink)

# Build identity is derived from the installed compiler chain, never a package.
# Utility-only builds remain available when the optional compiler tools are absent.
find_program(ARMIGO_SHADER_XCRUN xcrun)
set(armigo_shader_chain_tools_available TRUE)
set(ARMIGO_SHADER_SDK macosx)
set(ARMIGO_SHADER_CHAIN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/src/shader/tools/chain.py")
if(ARMIGO_IOS)
    set(ARMIGO_SHADER_SDK iphoneos)
    set(ARMIGO_SHADER_CHAIN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/ios/shader_chain.py")
endif()
foreach(armigo_shader_chain_tool IN ITEMS metal metallib)
    if(ARMIGO_SHADER_XCRUN)
        execute_process(
            COMMAND "${ARMIGO_SHADER_XCRUN}" --sdk "${ARMIGO_SHADER_SDK}" --find
                "${armigo_shader_chain_tool}"
            RESULT_VARIABLE armigo_shader_chain_tool_result
            OUTPUT_VARIABLE armigo_shader_chain_tool_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    else()
        set(armigo_shader_chain_tool_result 1)
    endif()
    if(NOT armigo_shader_chain_tool_result EQUAL 0 OR
       NOT EXISTS "${armigo_shader_chain_tool_path}")
        set(armigo_shader_chain_tools_available FALSE)
    endif()
endforeach()

if(armigo_shader_chain_tools_available)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(ARMIGO_EXPECTED_SHADER_CHAIN
        "${CMAKE_CURRENT_BINARY_DIR}/generated/armigo/shader/expected_metal_chain.h")
    add_custom_command(OUTPUT "${ARMIGO_EXPECTED_SHADER_CHAIN}"
        COMMAND "${Python3_EXECUTABLE}" "${ARMIGO_SHADER_CHAIN_SCRIPT}"
            --translate-tool "$<TARGET_FILE:armigo-shader-translate>"
            --output "${ARMIGO_EXPECTED_SHADER_CHAIN}"
        DEPENDS armigo-shader-translate "${ARMIGO_SHADER_CHAIN_SCRIPT}" src/shader/tools/chain.py
        VERBATIM)
    target_sources(armigo_runtime PRIVATE "${ARMIGO_EXPECTED_SHADER_CHAIN}")
    target_include_directories(armigo_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
else()
    if(ARMIGO_IOS)
        message(FATAL_ERROR "The iOS player requires the iPhoneOS Metal 4 compiler tools.")
    endif()
    message(STATUS
        "Armigo runtime: Metal compiler tools not found; trusted shader packages disabled")
endif()

target_compile_definitions(armigo_runtime PRIVATE ARMIGO_HAS_RUNTIME_PRESENTATION=1)

if(NOT ARMIGO_IOS)
    add_executable(armigo-module-check tools/module_check/main.cpp)
    target_link_libraries(armigo-module-check PRIVATE armigo_runtime)
endif()

add_library(armigo_romfs STATIC src/romfs/romfs_virtual.cpp)
target_include_directories(armigo_romfs PUBLIC include)

if(NOT ARMIGO_IOS)
    add_executable(armigo-romfs-build tools/romfs_build/main.cpp)
    target_include_directories(armigo-romfs-build PRIVATE include)
endif()

if(NOT ARMIGO_IOS)
    add_executable(armigo-headless tools/headless/main.mm)
    target_link_libraries(armigo-headless
        PRIVATE armigo_metal_probe armigo_runner_shell "${ARMIGO_FOUNDATION_FRAMEWORK}")
endif()

# Title settings (titles/<TitleID>/) installed next to the executables, where
# the runtime looks for them (compat::find_title_settings).
file(GLOB_RECURSE ARMIGO_TITLE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/titles/*")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/titles.stamp"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${CMAKE_CURRENT_BINARY_DIR}/titles"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/titles"
            "${CMAKE_CURRENT_BINARY_DIR}/titles"
    COMMAND "${CMAKE_COMMAND}" -E touch "${CMAKE_CURRENT_BINARY_DIR}/titles.stamp"
    DEPENDS ${ARMIGO_TITLE_FILES}
    COMMENT "Installing title settings")
add_custom_target(armigo_titles DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/titles.stamp")
if(TARGET armigo-headless)
    add_dependencies(armigo-headless armigo_titles)
endif()

# Lane I: runner shell seam shared by the headless and windowed tools.
add_library(armigo_runner_shell STATIC
    tools/headless/input_script.cpp
    tools/headless/runner_shell.cpp
    tools/headless/pso_archive_command.mm
    tools/headless/report_json.mm
    tools/windowed/windowed_session.cpp
    tools/windowed/application_host.cpp
    tools/windowed/frame_rate_controller.cpp
    tools/windowed/player_settings.cpp
    tools/windowed/player_choices.mm)
target_include_directories(armigo_runner_shell PUBLIC tools/headless tools/windowed)
target_link_libraries(armigo_runner_shell
    PUBLIC armigo_runtime armigo_platform_presenter
    PRIVATE armigo_platform_application armigo_services_file_system "${ARMIGO_FOUNDATION_FRAMEWORK}")

# Lane I: windowed runner application on the AppKit main thread.
# It is an application bundle so macOS classifies it as a game (Game Mode) from
# its Info.plist; the executable sits at Armigo.app/Contents/MacOS/Armigo.
if(ARMIGO_IOS)
    add_executable(armigo-ios-player tools/ios/main.mm)
    target_link_libraries(armigo-ios-player PRIVATE armigo_runner_shell "${ARMIGO_UI_FRAMEWORK}")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/IosEmbeddedSources.cmake")
    if(NOT ARMIGO_IOS_EMBEDDED_SOURCES_DIR STREQUAL "")
        foreach(source embedded.S embedded.cpp)
            if(NOT EXISTS "${ARMIGO_IOS_EMBEDDED_SOURCES_DIR}/${source}")
                message(FATAL_ERROR "Missing generated iOS embedded source: ${ARMIGO_IOS_EMBEDDED_SOURCES_DIR}/${source}")
            endif()
            target_sources(armigo-ios-player PRIVATE "${ARMIGO_IOS_EMBEDDED_SOURCES_DIR}/${source}")
        endforeach()
    else()
        # An unpacked player links no guest code: its weak reference to the
        # generated provider stays unbound, and loading a title reports that
        # the player has no linked guest code.
        target_link_options(armigo-ios-player PRIVATE "LINKER:-U,_armigo_linked_embedded_bundle")
    endif()
else()
    add_executable(armigo MACOSX_BUNDLE tools/windowed/main.mm)
    target_link_libraries(armigo PRIVATE armigo_runner_shell)
    set_target_properties(armigo PROPERTIES
        OUTPUT_NAME Armigo
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/tools/windowed/Info.plist.in"
        MACOSX_BUNDLE_GUI_IDENTIFIER "org.armigo.Armigo"
        MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
        MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
    add_dependencies(armigo armigo_titles)
    # The player carries only the linker's ad-hoc signature: Armigo signs
    # nothing. A packed app copies the executable alone, from Armigo.linker
    # (the bare name `armigo` in this build directory is the packer); the
    # title settings are copied into the bundle's resources.
    add_custom_command(TARGET armigo POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:armigo>"
                "${CMAKE_CURRENT_BINARY_DIR}/Armigo.linker"
        COMMAND "${CMAKE_COMMAND}" -E rm -rf "$<TARGET_BUNDLE_CONTENT_DIR:armigo>/Resources/titles"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/titles"
                "$<TARGET_BUNDLE_CONTENT_DIR:armigo>/Resources/titles"
        VERBATIM)
endif()

if(BUILD_TESTING)
    target_compile_definitions(armigo_runtime PRIVATE ARMIGO_RUNTIME_TESTING=1)
    add_executable(armigo_process_outcome_tests tests/runtime/process_outcome_tests.cpp
        tests/runtime/process_outcome_fixture.S)
    target_include_directories(armigo_process_outcome_tests PRIVATE tests/loader src/loader src/runtime)
    target_link_libraries(armigo_process_outcome_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.process_outcomes COMMAND armigo_process_outcome_tests)
    add_executable(armigo_process_mutex_stop_tests tests/runtime/process_mutex_stop_tests.cpp
        tests/runtime/process_runtime_fixture.S)
    target_include_directories(armigo_process_mutex_stop_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_process_mutex_stop_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.process_mutex_stop COMMAND armigo_process_mutex_stop_tests)
    add_executable(armigo_process_runtime_tests tests/runtime/process_runtime_tests.cpp
        tests/runtime/process_runtime_fixture.S)
    target_include_directories(armigo_process_runtime_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_process_runtime_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.process_runtime COMMAND armigo_process_runtime_tests)
    add_executable(armigo_runtime_addon_content_tests tests/runtime/addon_content_tests.cpp)
    target_link_libraries(armigo_runtime_addon_content_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.addon_content COMMAND armigo_runtime_addon_content_tests)

    add_executable(armigo_service_bootstrap_tests tests/runtime/service_bootstrap_tests.cpp)
    target_include_directories(armigo_service_bootstrap_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_service_bootstrap_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.service_bootstrap COMMAND armigo_service_bootstrap_tests)

    add_executable(armigo_service_registry_tests tests/runtime/service_registry_tests.cpp)
    target_include_directories(armigo_service_registry_tests PRIVATE src/runtime)
    target_link_libraries(armigo_service_registry_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.service_registry COMMAND armigo_service_registry_tests)
    add_test(NAME runtime.service_registry_legacy_memory_release
        COMMAND armigo_service_registry_tests --legacy-memory-release)
    set_tests_properties(runtime.service_registry_legacy_memory_release PROPERTIES
        ENVIRONMENT "ARMIGO_METAL_RENDERER=legacy" TIMEOUT 30)
    if(ARMIGO_ENABLE_GPU_TESTS)
        add_test(NAME runtime.service_registry_gpu_memory_release
            COMMAND armigo_service_registry_tests --gpu-memory-release)
        set_tests_properties(runtime.service_registry_gpu_memory_release PROPERTIES
            ENVIRONMENT "ARMIGO_METAL_RENDERER=persistent"
            LABELS "gpu;metal4;runtime" TIMEOUT 60)
    endif()

    add_executable(armigo_guest_memory_tests tests/runtime/guest_memory_tests.cpp)
    target_link_libraries(armigo_guest_memory_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.guest_memory COMMAND armigo_guest_memory_tests)

    add_executable(armigo_guest_memory_failure_tests tests/runtime/guest_memory_failure_tests.cpp)
    target_link_libraries(armigo_guest_memory_failure_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.guest_memory_failures COMMAND armigo_guest_memory_failure_tests)

    add_executable(armigo_guest_memory_concurrency_tests tests/runtime/guest_memory_concurrency_tests.cpp)
    target_link_libraries(armigo_guest_memory_concurrency_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.guest_memory_concurrency COMMAND armigo_guest_memory_concurrency_tests)
    set_tests_properties(runtime.guest_memory_concurrency PROPERTIES LABELS "runtime;concurrency" TIMEOUT 60)

    add_executable(armigo_guest_memory_module_tests tests/runtime/guest_memory_module_tests.cpp)
    target_include_directories(armigo_guest_memory_module_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_guest_memory_module_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.guest_memory_modules COMMAND armigo_guest_memory_module_tests)

    add_executable(armigo_guest_memory_dispatch_tests tests/runtime/guest_memory_dispatch_tests.cpp
        tests/execution/guest_dispatch_fixtures.S)
    target_link_libraries(armigo_guest_memory_dispatch_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.guest_memory_dispatch COMMAND armigo_guest_memory_dispatch_tests)

    add_executable(armigo_process_memory_tests tests/platform/process_memory_tests.cpp)
    target_include_directories(armigo_process_memory_tests PRIVATE src/platform)
    target_link_libraries(armigo_process_memory_tests PRIVATE armigo_host_memory)
    add_test(NAME platform.process_memory COMMAND armigo_process_memory_tests)

    add_executable(armigo_launch_contracts_tests tests/runtime/launch_contracts_tests.cpp)
    target_link_libraries(armigo_launch_contracts_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.launch_contracts COMMAND armigo_launch_contracts_tests)
    set_tests_properties(runtime.launch_contracts PROPERTIES LABELS "runtime;contracts" TIMEOUT 30)
    add_executable(armigo_launch_profile_tests tests/runtime/launch_profile_tests.cpp)
    target_include_directories(armigo_launch_profile_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_launch_profile_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.launch_profile COMMAND armigo_launch_profile_tests)
    set_tests_properties(runtime.launch_profile PROPERTIES LABELS "runtime;profile" TIMEOUT 30)
    add_executable(armigo_guest_clock_tests tests/runtime/guest_clock_tests.cpp)
    target_link_libraries(armigo_guest_clock_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.guest_clock COMMAND armigo_guest_clock_tests)
    set_tests_properties(runtime.guest_clock PROPERTIES LABELS "runtime" TIMEOUT 30)
    add_executable(armigo_presentation_clock_tests tests/runtime/presentation_clock_tests.cpp)
    target_link_libraries(armigo_presentation_clock_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.presentation_clock COMMAND armigo_presentation_clock_tests)
    set_tests_properties(runtime.presentation_clock PROPERTIES LABELS "runtime" TIMEOUT 30)
    add_executable(armigo_frame_rate_governor_tests tests/runtime/frame_rate_governor_tests.cpp)
    target_link_libraries(armigo_frame_rate_governor_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.frame_rate_governor COMMAND armigo_frame_rate_governor_tests)
    set_tests_properties(runtime.frame_rate_governor PROPERTIES LABELS "runtime" TIMEOUT 30)
    add_executable(armigo_player_settings_tests tests/runtime/player_settings_tests.cpp)
    target_link_libraries(armigo_player_settings_tests PRIVATE armigo_runner_shell)
    add_test(NAME runtime.player_settings COMMAND armigo_player_settings_tests)
    set_tests_properties(runtime.player_settings PROPERTIES LABELS "runtime" TIMEOUT 30)
    add_executable(armigo_periodic_thread_budget_tests tests/runtime/periodic_thread_budget_tests.cpp)
    target_include_directories(armigo_periodic_thread_budget_tests PRIVATE include)
    add_test(NAME runtime.periodic_thread_budget COMMAND armigo_periodic_thread_budget_tests)
    set_tests_properties(runtime.periodic_thread_budget PROPERTIES LABELS "runtime;platform" TIMEOUT 30)
    add_executable(armigo_time_session_tests tests/services/time_session_tests.cpp)
    target_link_libraries(armigo_time_session_tests PRIVATE armigo_services_time_session
        armigo_services_shared_clock)
    add_test(NAME services.time_session COMMAND armigo_time_session_tests)
    set_tests_properties(services.time_session PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_audio_out_tests tests/services/audio_out_tests.cpp)
    target_link_libraries(armigo_audio_out_tests PRIVATE armigo_services_audout)
    add_test(NAME services.audout COMMAND armigo_audio_out_tests)
    set_tests_properties(services.audout PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_bsd_tests tests/services/bsd_tests.cpp)
    target_link_libraries(armigo_bsd_tests PRIVATE armigo_services_bsd)
    add_test(NAME services.bsd COMMAND armigo_bsd_tests)
    set_tests_properties(services.bsd PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_host1x_tests tests/host1x/host1x_tests.cpp)
    target_link_libraries(armigo_host1x_tests PRIVATE armigo_host1x)
    add_test(NAME host1x.engines COMMAND armigo_host1x_tests)
    set_tests_properties(host1x.engines PROPERTIES LABELS "host1x" TIMEOUT 60)
    add_executable(armigo_nifm_tests tests/services/nifm_tests.cpp)
    target_link_libraries(armigo_nifm_tests PRIVATE armigo_services_nifm)
    add_test(NAME services.nifm COMMAND armigo_nifm_tests)
    set_tests_properties(services.nifm PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_mm_tests tests/services/mm_tests.cpp)
    target_link_libraries(armigo_mm_tests PRIVATE armigo_services_mm)
    add_test(NAME services.mm COMMAND armigo_mm_tests)
    set_tests_properties(services.mm PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_settings_tests tests/services/settings_tests.cpp)
    target_link_libraries(armigo_settings_tests PRIVATE armigo_services_settings)
    add_test(NAME services.set COMMAND armigo_settings_tests)
    set_tests_properties(services.set PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_shared_font_tests tests/services/shared_font_tests.cpp)
    target_link_libraries(armigo_shared_font_tests PRIVATE armigo_services_shared_font)
    add_test(NAME services.shared_font COMMAND armigo_shared_font_tests)
    set_tests_properties(services.shared_font PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_pctl_tests tests/services/parental_control_tests.cpp)
    target_link_libraries(armigo_pctl_tests PRIVATE armigo_services_pctl)
    add_test(NAME services.pctl COMMAND armigo_pctl_tests)
    set_tests_properties(services.pctl PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_prepo_tests tests/services/play_report_tests.cpp)
    target_link_libraries(armigo_prepo_tests PRIVATE armigo_services_prepo)
    add_test(NAME services.prepo COMMAND armigo_prepo_tests)
    set_tests_properties(services.prepo PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_context_writer_tests tests/services/context_writer_tests.cpp)
    target_link_libraries(armigo_context_writer_tests PRIVATE armigo_services_context_writer)
    add_test(NAME services.context_writer COMMAND armigo_context_writer_tests)
    set_tests_properties(services.context_writer PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_file_system_tests tests/services/file_system_tests.cpp)
    target_link_libraries(armigo_file_system_tests PRIVATE armigo_services_file_system armigo_romfs)
    add_test(NAME services.file_system COMMAND armigo_file_system_tests)
    set_tests_properties(services.file_system PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_romfs_virtual_tests tests/romfs/romfs_virtual_tests.cpp)
    target_link_libraries(armigo_romfs_virtual_tests PRIVATE armigo_romfs)
    add_test(NAME romfs.virtual_storage COMMAND armigo_romfs_virtual_tests)
    set_tests_properties(romfs.virtual_storage PROPERTIES LABELS "romfs" TIMEOUT 30)

    add_executable(armigo_romfs_image_tests tests/tools/romfs_image_tests.cpp)
    target_include_directories(armigo_romfs_image_tests PRIVATE include)
    add_test(NAME tools.romfs_image COMMAND armigo_romfs_image_tests)
    set_tests_properties(tools.romfs_image PROPERTIES LABELS "tools" TIMEOUT 30)
    add_executable(armigo_process_memory_covered_tests tests/horizon/process_memory_covered_tests.cpp)
    target_link_libraries(armigo_process_memory_covered_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.process_memory_covered COMMAND armigo_process_memory_covered_tests)
    set_tests_properties(horizon.process_memory_covered PROPERTIES LABELS "horizon")
    add_executable(armigo_arbitration_tests tests/horizon/arbitration_tests.cpp)
    target_link_libraries(armigo_arbitration_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.arbitration COMMAND armigo_arbitration_tests)
    set_tests_properties(horizon.arbitration PROPERTIES LABELS "horizon")
    add_executable(armigo_address_synchronization_tests tests/horizon/address_synchronization_tests.cpp)
    target_link_libraries(armigo_address_synchronization_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.address_synchronization COMMAND armigo_address_synchronization_tests)
    set_tests_properties(horizon.address_synchronization PROPERTIES LABELS "horizon" TIMEOUT 30)
    add_executable(armigo_process_wide_key_tests tests/horizon/process_wide_key_tests.cpp)
    target_link_libraries(armigo_process_wide_key_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.process_wide_key COMMAND armigo_process_wide_key_tests)
    set_tests_properties(horizon.process_wide_key PROPERTIES LABELS "horizon" TIMEOUT 30)
    add_executable(armigo_mutex_cv_tests tests/horizon/mutex_cv_tests.cpp)
    target_link_libraries(armigo_mutex_cv_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.mutex_cv COMMAND armigo_mutex_cv_tests)
    set_tests_properties(horizon.mutex_cv PROPERTIES LABELS "horizon" TIMEOUT 30)
    add_executable(armigo_shared_memory_tests tests/horizon/shared_memory_tests.cpp)
    target_link_libraries(armigo_shared_memory_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.shared_memory COMMAND armigo_shared_memory_tests)
    set_tests_properties(horizon.shared_memory PROPERTIES LABELS "horizon" TIMEOUT 30)

    add_executable(armigo_ro_tests tests/services/loader_tests.cpp)
    target_link_libraries(armigo_ro_tests PRIVATE armigo_services_ro)
    add_test(NAME services.ro COMMAND armigo_ro_tests)
    set_tests_properties(services.ro PROPERTIES LABELS "services")

    add_executable(armigo_vi_tests tests/services/video_interface_tests.cpp)
    target_link_libraries(armigo_vi_tests PRIVATE armigo_services_vi)
    add_test(NAME services.vi COMMAND armigo_vi_tests)
    set_tests_properties(services.vi PROPERTIES LABELS "services")

    add_executable(armigo_presentation_tests tests/runtime/presentation_tests.cpp)
    target_link_libraries(armigo_presentation_tests PRIVATE armigo_services_vi)
    add_test(NAME runtime.presentation COMMAND armigo_presentation_tests)
    set_tests_properties(runtime.presentation PROPERTIES LABELS "runtime;services" TIMEOUT 30)

    add_executable(armigo_vsync_source_tests tests/services/vsync_source_tests.cpp)
    target_link_libraries(armigo_vsync_source_tests PRIVATE armigo_services_vi)
    add_test(NAME services.vsync_source COMMAND armigo_vsync_source_tests)
    set_tests_properties(services.vsync_source PROPERTIES LABELS "services" TIMEOUT 30)

    add_executable(armigo_vi_presentation_tests tests/services/video_interface_presentation_tests.cpp)
    target_link_libraries(armigo_vi_presentation_tests PRIVATE armigo_services_vi)
    add_test(NAME services.vi_presentation COMMAND armigo_vi_presentation_tests)
    set_tests_properties(services.vi_presentation PROPERTIES LABELS "runtime;services" TIMEOUT 30)

    if(ARMIGO_ENABLE_GPU_TESTS)
        add_executable(armigo_vi_presentation_gpu_tests tests/services/video_interface_presentation_tests.cpp)
        target_compile_definitions(armigo_vi_presentation_gpu_tests PRIVATE ARMIGO_LANE_G_GPU_TESTS=1)
        target_link_libraries(armigo_vi_presentation_gpu_tests PRIVATE armigo_services_vi armigo_platform_presenter)
        add_test(NAME services.vi_presentation_gpu COMMAND armigo_vi_presentation_gpu_tests)
        set_tests_properties(services.vi_presentation_gpu PROPERTIES LABELS "runtime;services;gpu;metal4" TIMEOUT 60)
    endif()

    add_executable(armigo_nvdrv_gpu_tests tests/services/nvdrv_gpu_tests.cpp)
    target_link_libraries(armigo_nvdrv_gpu_tests PRIVATE armigo_services_nvdrv)
    add_test(NAME services.nvdrv_gpu COMMAND armigo_nvdrv_gpu_tests)
    set_tests_properties(services.nvdrv_gpu PROPERTIES LABELS "services" TIMEOUT 60)

    add_executable(armigo_nvdrv_tests tests/services/nvdrv_tests.cpp)
    target_link_libraries(armigo_nvdrv_tests PRIVATE armigo_services_nvdrv)
    add_test(NAME services.nvdrv COMMAND armigo_nvdrv_tests)
    set_tests_properties(services.nvdrv PROPERTIES LABELS "services")

    add_executable(armigo_aoc_tests tests/services/addon_content_tests.cpp)
    target_link_libraries(armigo_aoc_tests PRIVATE armigo_services_aoc)
    add_test(NAME services.aoc COMMAND armigo_aoc_tests)
    set_tests_properties(services.aoc PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_applet_tests tests/services/applet_tests.cpp)
    target_link_libraries(armigo_applet_tests PRIVATE armigo_services_am)
    add_test(NAME services.applet COMMAND armigo_applet_tests)
    set_tests_properties(services.applet PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_account_tests tests/services/account_tests.cpp)
    target_link_libraries(armigo_account_tests PRIVATE armigo_services_acc)
    add_test(NAME services.account COMMAND armigo_account_tests)
    set_tests_properties(services.account PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_hid_tests tests/services/hid_tests.cpp)
    target_link_libraries(armigo_hid_tests PRIVATE armigo_services_hid)
    add_test(NAME services.hid COMMAND armigo_hid_tests)
    add_executable(armigo_hid_motion_tests tests/services/hid_motion_tests.cpp)
    target_include_directories(armigo_hid_motion_tests PRIVATE src/services)
    target_link_libraries(armigo_hid_motion_tests PRIVATE armigo_services_hid)
    add_test(NAME services.hid_motion COMMAND armigo_hid_motion_tests)
    add_executable(armigo_hid_seven_axis_tests tests/services/hid_seven_axis_tests.cpp)
    target_link_libraries(armigo_hid_seven_axis_tests PRIVATE armigo_services_hid)
    add_test(NAME services.hid_seven_axis COMMAND armigo_hid_seven_axis_tests)

    add_executable(armigo_nfp_tests tests/services/nfp_tests.cpp)
    target_link_libraries(armigo_nfp_tests PRIVATE armigo_services_nfp)
    add_test(NAME services.nfp COMMAND armigo_nfp_tests)

    add_executable(armigo_audio_renderer_tests tests/services/audio_renderer_tests.cpp)
    target_link_libraries(armigo_audio_renderer_tests PRIVATE armigo_services_audren)
    add_test(NAME services.audio_renderer COMMAND armigo_audio_renderer_tests)
    set_tests_properties(services.audio_renderer PROPERTIES LABELS "services;audio" TIMEOUT 60)
    # The host PCM sink with its AudioToolbox calls replaced by the test's mocks.
    add_executable(armigo_pcm_sink_tests tests/platform/pcm_sink_tests.cpp
        src/platform/macos/pcm_sink.mm)
    target_compile_definitions(armigo_pcm_sink_tests PRIVATE ARMIGO_PCM_SINK_MOCK_AUDIO_UNIT)
    target_link_libraries(armigo_pcm_sink_tests PRIVATE armigo_audio_renderer
        "${ARMIGO_AUDIOTOOLBOX_FRAMEWORK}")
    add_test(NAME platform.pcm_sink COMMAND armigo_pcm_sink_tests)
    set_tests_properties(platform.pcm_sink PROPERTIES LABELS "platform;audio" TIMEOUT 60)
    add_executable(armigo_performance_tests tests/services/performance_tests.cpp)
    target_link_libraries(armigo_performance_tests PRIVATE armigo_services_apm)
    add_test(NAME services.apm COMMAND armigo_performance_tests)
    set_tests_properties(services.apm PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_log_manager_tests tests/services/log_manager_tests.cpp)
    target_link_libraries(armigo_log_manager_tests PRIVATE armigo_services_lm)
    add_test(NAME services.lm COMMAND armigo_log_manager_tests)
    set_tests_properties(services.lm PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_shared_clock_tests tests/services/shared_clock_tests.cpp)
    target_link_libraries(armigo_shared_clock_tests PRIVATE armigo_services_shared_clock)
    add_test(NAME services.shared_clock COMMAND armigo_shared_clock_tests)
    set_tests_properties(services.shared_clock PROPERTIES LABELS "services" TIMEOUT 30)
    add_executable(armigo_sm_tests tests/services/service_manager_tests.cpp)
    target_link_libraries(armigo_sm_tests PRIVATE armigo_services_sm)
    add_test(NAME services.sm COMMAND armigo_sm_tests)
    add_executable(armigo_sm_session_tests tests/services/service_manager_session_tests.cpp)
    target_link_libraries(armigo_sm_session_tests PRIVATE armigo_services_sm)
    add_test(NAME services.sm_session COMMAND armigo_sm_session_tests)
    set_tests_properties(services.sm services.sm_session PROPERTIES LABELS "services;horizon" TIMEOUT 30)
    add_executable(armigo_reachable_code_tests
        tests/execution/reachable_code_tests.cpp tests/execution/reachable_code_fixture.S)
    target_link_libraries(armigo_reachable_code_tests PRIVATE armigo_execution_reachable_code)
    add_test(NAME execution.reachable_code COMMAND armigo_reachable_code_tests)
    set_tests_properties(execution.reachable_code PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_time_service_tests tests/services/time_tests.cpp)
    target_link_libraries(armigo_time_service_tests PRIVATE armigo_services_time)
    add_test(NAME services.time COMMAND armigo_time_service_tests)
    set_tests_properties(services.time PROPERTIES LABELS "services" TIMEOUT 30)
    add_executable(armigo_guest_stack_tests tests/platform/guest_stack_tests.cpp)
    target_link_libraries(armigo_guest_stack_tests PRIVATE armigo_guest_stack)
    add_test(NAME platform.guest_stack COMMAND armigo_guest_stack_tests)
    add_executable(armigo_guest_entry_tests
        tests/execution/guest_entry_tests.cpp tests/execution/guest_entry_fixture.S)
    target_link_libraries(armigo_guest_entry_tests PRIVATE armigo_execution_guest_entry)
    add_test(NAME execution.guest_entry COMMAND armigo_guest_entry_tests)
    set_tests_properties(platform.guest_stack execution.guest_entry
        PROPERTIES LABELS "execution;arm64" TIMEOUT 60)
    add_executable(armigo_guest_preempt_tests tests/execution/guest_preempt_tests.cpp)
    target_link_libraries(armigo_guest_preempt_tests PRIVATE armigo_execution_guest_entry)
    target_include_directories(armigo_guest_preempt_tests PRIVATE src/execution)
    add_test(NAME execution.guest_preempt COMMAND armigo_guest_preempt_tests)
    set_tests_properties(execution.guest_preempt PROPERTIES LABELS "execution;arm64" TIMEOUT 60)
    add_executable(armigo_scalar_x18_tests
        tests/execution/scalar_x18_tests.cpp tests/execution/scalar_x18_fixture.S)
    target_link_libraries(armigo_scalar_x18_tests PRIVATE armigo_execution_scalar_x18_patch
        armigo_execution_guest_dispatch armigo_host_memory armigo_thread_slot)
    add_test(NAME execution.scalar_x18_patch COMMAND armigo_scalar_x18_tests)
    set_tests_properties(execution.scalar_x18_patch PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_thread_register_patch_tests
        tests/execution/thread_register_patch_tests.cpp)
    target_link_libraries(armigo_thread_register_patch_tests PRIVATE armigo_aarch64
        armigo_execution_guest_dispatch armigo_host_memory armigo_thread_slot)
    add_test(NAME execution.thread_register_patch COMMAND armigo_thread_register_patch_tests)
    set_tests_properties(execution.thread_register_patch PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_x18_forms_tests
        tests/execution/x18_forms_tests.cpp tests/execution/x18_forms_fixture.S)
    target_link_libraries(armigo_x18_forms_tests PRIVATE armigo_execution_scalar_x18_patch
        armigo_execution_guest_dispatch armigo_host_memory armigo_thread_slot)
    add_test(NAME execution.x18_forms COMMAND armigo_x18_forms_tests)
    set_tests_properties(execution.x18_forms PROPERTIES LABELS "execution;arm64" TIMEOUT 60)
    add_executable(armigo_native_register_tests
        tests/runtime/native_register_tests.cpp
        tests/runtime/native_register_fixture.S
        tests/platform/thread_slot_fixture.S)
    target_link_libraries(armigo_native_register_tests
        PRIVATE armigo_execution_guest_dispatch armigo_thread_slot armigo_execution_guest_entry)
    add_test(NAME runtime.native_register_cell COMMAND armigo_native_register_tests)
    set_tests_properties(runtime.native_register_cell PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_thread_slot_tests
        tests/platform/thread_slot_tests.cpp tests/platform/thread_slot_fixture.S)
    target_link_libraries(armigo_thread_slot_tests PRIVATE armigo_thread_slot)
    add_test(NAME platform.thread_slot COMMAND armigo_thread_slot_tests)
    set_tests_properties(platform.thread_slot PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_startup_query_gate_tests
        tests/horizon/startup_query_gate_tests.cpp tests/execution/guest_dispatch_fixtures.S)
    target_link_libraries(armigo_startup_query_gate_tests PRIVATE armigo_runtime)
    add_test(NAME horizon.native_query_gate COMMAND armigo_startup_query_gate_tests)
    set_tests_properties(horizon.native_query_gate PROPERTIES LABELS "execution;horizon;arm64" TIMEOUT 30)
    add_executable(armigo_gate_patch_tests
        tests/execution/gate_patch_tests.cpp tests/execution/gate_patch_fixture.S)
    target_link_libraries(armigo_gate_patch_tests
        PRIVATE armigo_execution_gate_patch armigo_host_memory)
    add_test(NAME execution.gate_patch COMMAND armigo_gate_patch_tests)
    set_tests_properties(execution.gate_patch PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_local_gate_tests
        tests/execution/local_gate_tests.cpp tests/execution/local_gate_fixture.S)
    target_link_libraries(armigo_local_gate_tests PRIVATE armigo_execution_embedded_patch armigo_execution armigo_host_memory)
    add_test(NAME execution.local_gate COMMAND armigo_local_gate_tests)
    set_tests_properties(execution.local_gate PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_exclusive_monitor_tests
        tests/execution/exclusive_monitor_tests.cpp tests/execution/exclusive_monitor_fixture.S)
    target_link_libraries(armigo_exclusive_monitor_tests PRIVATE
        armigo_execution_prepared_image armigo_execution_guest_entry armigo_host_memory)
    add_test(NAME execution.exclusive_monitor COMMAND armigo_exclusive_monitor_tests)
    set_tests_properties(execution.exclusive_monitor PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_executable(armigo_module_set_tests
        tests/runtime/module_set_tests.cpp tests/runtime/module_fixture.S)
    target_include_directories(armigo_module_set_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_module_set_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.module_set COMMAND armigo_module_set_tests)
    set_tests_properties(runtime.module_set PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME shader_input.extract
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/tools/shader_extract_tests.py"
            "$<TARGET_FILE:armigo-shader-extract>")
    add_test(NAME shader.translate
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/tools/shader_translate_tests.py"
            "$<TARGET_FILE:armigo-shader-translate>")
    set_tests_properties(shader.translate PROPERTIES TIMEOUT 60 LABELS "shader;tools")
    add_test(NAME shader.package_build
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/tools/shader_package_build_tests.py")
    set_tests_properties(shader.package_build PROPERTIES TIMEOUT 60 LABELS "shader;tools"
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    # Header composition and fail-closed mocks need no Metal device at runtime.
    add_executable(armigo_metal_renderer_contracts_tests tests/metal/renderer_contracts.mm)
    target_include_directories(armigo_metal_renderer_contracts_tests PRIVATE src)
    target_link_libraries(armigo_metal_renderer_contracts_tests PRIVATE
        armigo_graphics "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME graphics.metal_renderer_contracts COMMAND armigo_metal_renderer_contracts_tests)
    set_tests_properties(graphics.metal_renderer_contracts PROPERTIES TIMEOUT 60)
    add_executable(armigo_metal_shader_package_tests tests/metal/shader_package.mm)
    target_include_directories(armigo_metal_shader_package_tests PRIVATE src/loader)
    target_link_libraries(armigo_metal_shader_package_tests PRIVATE
        armigo_metal_shader_package armigo_loader armigo_shader_translation "${ARMIGO_FOUNDATION_FRAMEWORK}")
    add_test(NAME shader.metal_package COMMAND armigo_metal_shader_package_tests)

    add_executable(armigo_metal_compute_shader_package_tests
        tests/metal/compute_shader_package.mm)
    target_include_directories(armigo_metal_compute_shader_package_tests PRIVATE
        src/loader src/platform/metal src/graphics)
    target_link_libraries(armigo_metal_compute_shader_package_tests PRIVATE
        armigo_metal_shader_package armigo_graphics
        "${ARMIGO_FOUNDATION_FRAMEWORK}")
    add_test(NAME shader.metal_compute_package
             COMMAND armigo_metal_compute_shader_package_tests)
    set_tests_properties(shader.metal_package PROPERTIES TIMEOUT 60 LABELS "shader")
    set_tests_properties(shader_input.extract PROPERTIES TIMEOUT 60)
    add_test(NAME runtime.module_check_cli
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/tools/module_check_tests.py"
            "$<TARGET_FILE:armigo-module-check>")
    set_tests_properties(runtime.module_check_cli PROPERTIES TIMEOUT 90)
    add_test(NAME repository.privacy_policy
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/tools/repository_policy_tests.py")
    add_test(NAME build.optional_metal_tools
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/tests/tools/optional_metal_tools_build_tests.py")
    set_tests_properties(build.optional_metal_tools PROPERTIES
        LABELS "build;tools" TIMEOUT 600 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    add_executable(armigo_host_module_tests
        tests/platform/host_module_tests.cpp tests/platform/mapping_fixture.S)
    target_link_libraries(armigo_host_module_tests PRIVATE armigo_host_memory)
    add_test(NAME platform.host_module COMMAND armigo_host_module_tests)
    set_tests_properties(platform.host_module PROPERTIES LABELS "execution;arm64" TIMEOUT 30)
    add_test(NAME headless.help COMMAND armigo-headless --help)
    add_executable(armigo_runtime_launch_tests tests/tools/runtime_launch_tests.cpp)
    target_link_libraries(armigo_runtime_launch_tests PRIVATE armigo_runner_shell)
    add_test(NAME tools.runtime_launch COMMAND armigo_runtime_launch_tests)
    set_tests_properties(tools.runtime_launch PROPERTIES LABELS "tools" TIMEOUT 60)
    add_executable(armigo_container_options_tests tests/tools/container_options_tests.cpp)
    target_link_libraries(armigo_container_options_tests PRIVATE armigo_runner_shell)
    add_test(NAME tools.container_options COMMAND armigo_container_options_tests)
    set_tests_properties(tools.container_options PROPERTIES LABELS "tools" TIMEOUT 60)
    add_executable(armigo_windowed_tests tests/tools/windowed_tests.cpp)
    target_link_libraries(armigo_windowed_tests PRIVATE armigo_runner_shell)
    add_test(NAME tools.windowed COMMAND armigo_windowed_tests)
    set_tests_properties(tools.windowed PROPERTIES LABELS "tools" TIMEOUT 60)
    add_executable(armigo_input_script_tests tests/tools/input_script_tests.cpp)
    target_link_libraries(armigo_input_script_tests PRIVATE armigo_runner_shell)
    add_test(NAME tools.input_script COMMAND armigo_input_script_tests)
    set_tests_properties(tools.input_script PROPERTIES LABELS "tools" TIMEOUT 30)
    add_executable(armigo_runner_results_tests tests/tools/runner_results_tests.cpp)
    target_link_libraries(armigo_runner_results_tests PRIVATE armigo_runner_shell)
    add_test(NAME tools.runner_results COMMAND armigo_runner_results_tests)
    set_tests_properties(tools.runner_results PROPERTIES LABELS "tools" TIMEOUT 30)
    add_test(NAME tools.runtime_launch_cli
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/tools/runtime_launch_cli_tests.py"
            "$<TARGET_FILE:armigo-headless>" "$<TARGET_FILE:armigo>")
    set_tests_properties(tools.runtime_launch_cli PROPERTIES LABELS "tools" TIMEOUT 90)
    if(ARMIGO_ENABLE_GPU_TESTS)
        add_executable(armigo_metal_engine_pipeline_tests tests/metal/engine_pipeline_tests.mm)
        target_include_directories(armigo_metal_engine_pipeline_tests PRIVATE src)
        target_link_libraries(armigo_metal_engine_pipeline_tests PRIVATE armigo_metal_engine_renderer)
        add_test(NAME graphics.metal4_engine_pipeline COMMAND armigo_metal_engine_pipeline_tests)
        set_tests_properties(graphics.metal4_engine_pipeline
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_engine_state_tests tests/metal/engine_state_tests.mm)
        target_link_libraries(armigo_metal_engine_state_tests PRIVATE
            armigo_metal_engine_renderer armigo_shader_pipeline)
        add_test(NAME graphics.metal4_engine_state COMMAND armigo_metal_engine_state_tests)
        set_tests_properties(graphics.metal4_engine_state
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 120)
        add_executable(armigo_metal_renderer_tests tests/metal/renderer_tests.mm)
        target_link_libraries(armigo_metal_renderer_tests PRIVATE armigo_metal_renderer)
        add_test(NAME graphics.metal4_renderer COMMAND armigo_metal_renderer_tests)
        set_tests_properties(graphics.metal4_renderer
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 120)
        add_executable(armigo_metal_bindings_tests tests/metal/bindings.mm)
        target_include_directories(armigo_metal_bindings_tests PRIVATE src)
        target_link_libraries(armigo_metal_bindings_tests PRIVATE armigo_metal_renderer)
        add_test(NAME graphics.metal4_bindings COMMAND armigo_metal_bindings_tests)
        add_executable(armigo_metal_submission_tests tests/metal/submission.mm)
        target_link_libraries(armigo_metal_submission_tests PRIVATE armigo_metal_renderer)
        add_test(NAME graphics.metal4_submission COMMAND armigo_metal_submission_tests)
        add_executable(armigo_metal_draw_encoding_tests tests/metal/draw_encoding_tests.mm)
        target_link_libraries(armigo_metal_draw_encoding_tests PRIVATE armigo_metal_renderer)
        add_test(NAME graphics.metal4_draw_encoding COMMAND armigo_metal_draw_encoding_tests)
        set_tests_properties(graphics.metal4_bindings graphics.metal4_submission
            graphics.metal4_draw_encoding
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 120)
        add_executable(armigo_color_clear_gpu_tests tests/metal/color_clear.mm)
        target_link_libraries(armigo_color_clear_gpu_tests PRIVATE armigo_metal_color_clear)
        add_test(NAME graphics.metal4_color_clear COMMAND armigo_color_clear_gpu_tests)
        set_tests_properties(graphics.metal4_color_clear
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_buffer_cache_gpu_tests tests/metal/buffer_cache_tests.mm)
        target_include_directories(armigo_metal_buffer_cache_gpu_tests
            PRIVATE src/platform/metal)
        target_link_libraries(armigo_metal_buffer_cache_gpu_tests PRIVATE
            armigo_metal_buffer_cache)
        add_test(NAME graphics.metal4_buffer_cache
            COMMAND armigo_metal_buffer_cache_gpu_tests)
        set_tests_properties(graphics.metal4_buffer_cache
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_attachments_gpu_tests tests/metal/attachments_tests.mm)
        target_link_libraries(armigo_metal_attachments_gpu_tests PRIVATE
            armigo_metal_attachments)
        add_test(NAME graphics.metal4_attachments
            COMMAND armigo_metal_attachments_gpu_tests)
        set_tests_properties(graphics.metal4_attachments
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_clear_encoder_gpu_tests
            tests/metal/clear_encoder_tests.mm)
        target_link_libraries(armigo_metal_clear_encoder_gpu_tests PRIVATE
            armigo_metal_clear_encoder)
        add_test(NAME graphics.metal4_clear_encoder
            COMMAND armigo_metal_clear_encoder_gpu_tests)
        set_tests_properties(graphics.metal4_clear_encoder
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_texture_cache_gpu_tests tests/metal/texture_cache.mm)
        target_include_directories(armigo_metal_texture_cache_gpu_tests PRIVATE src/platform/metal)
        target_link_libraries(armigo_metal_texture_cache_gpu_tests PRIVATE armigo_metal_texture_cache)
        add_test(NAME graphics.metal4_texture_cache COMMAND armigo_metal_texture_cache_gpu_tests)
        set_tests_properties(graphics.metal4_texture_cache
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 120)
        add_executable(armigo_metal_texture_swizzle_gpu_tests tests/metal/texture_swizzle_gpu.mm)
        target_include_directories(armigo_metal_texture_swizzle_gpu_tests PRIVATE src/platform/metal)
        target_link_libraries(armigo_metal_texture_swizzle_gpu_tests PRIVATE
            armigo_metal_texture_swizzle)
        add_test(NAME graphics.metal4_texture_swizzle COMMAND armigo_metal_texture_swizzle_gpu_tests)
        set_tests_properties(graphics.metal4_texture_swizzle
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 120)
        add_executable(armigo_metal_texture_format_gpu_tests tests/metal/texture_format.mm)
        target_include_directories(armigo_metal_texture_format_gpu_tests
            PRIVATE src/platform/metal)
        target_link_libraries(armigo_metal_texture_format_gpu_tests PRIVATE
            armigo_metal_texture_format)
        add_test(NAME graphics.metal4_texture_format
            COMMAND armigo_metal_texture_format_gpu_tests)
        set_tests_properties(graphics.metal4_texture_format
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_sampler_gpu_tests
            tests/metal/sampler_execution.mm)
        target_include_directories(armigo_metal_sampler_gpu_tests
            PRIVATE src/platform/metal)
        target_link_libraries(armigo_metal_sampler_gpu_tests PRIVATE
            armigo_metal_texture_format)
        add_test(NAME graphics.metal4_sampler
            COMMAND armigo_metal_sampler_gpu_tests)
        set_tests_properties(graphics.metal4_sampler
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_custom_border_gpu_tests
            tests/metal/custom_border_execution.mm)
        target_include_directories(armigo_metal_custom_border_gpu_tests PRIVATE
            src/platform/metal src/shader)
        target_link_libraries(armigo_metal_custom_border_gpu_tests PRIVATE
            armigo_metal_texture_format)
        add_test(NAME graphics.metal4_custom_border
            COMMAND armigo_metal_custom_border_gpu_tests)
        set_tests_properties(graphics.metal4_custom_border PROPERTIES
            ENVIRONMENT "MTL_DEBUG_LAYER=1"
            LABELS "gpu;metal4;graphics;shader" TIMEOUT 120)
        add_executable(armigo_metal_pipeline_state_gpu_tests
            tests/metal/pipeline_state_tests.mm)
        target_link_libraries(armigo_metal_pipeline_state_gpu_tests PRIVATE
            armigo_metal_pipeline_state)
        add_test(NAME graphics.metal4_pipeline_state
            COMMAND armigo_metal_pipeline_state_gpu_tests)
        set_tests_properties(graphics.metal4_pipeline_state
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 120)
        add_executable(armigo_metal_fragment_output_types_gpu_tests
            tests/metal/fragment_output_types.mm)
        target_link_libraries(armigo_metal_fragment_output_types_gpu_tests PRIVATE
            armigo_shader_pipeline "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME graphics.metal4_fragment_output_types
            COMMAND armigo_metal_fragment_output_types_gpu_tests)
        set_tests_properties(graphics.metal4_fragment_output_types
            PROPERTIES LABELS "gpu;metal4;graphics;shader" TIMEOUT 120)
        add_executable(armigo_metal_fragment_depth_gpu_tests
            tests/metal/fragment_depth_execution.mm)
        target_link_libraries(armigo_metal_fragment_depth_gpu_tests PRIVATE
            armigo_shader_pipeline "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME graphics.metal4_fragment_depth
            COMMAND armigo_metal_fragment_depth_gpu_tests)
        set_tests_properties(graphics.metal4_fragment_depth
            PROPERTIES LABELS "gpu;metal4;graphics;shader" TIMEOUT 120)
        add_executable(armigo_metal_fragment_output_conversions_gpu_tests
            tests/metal/fragment_output_conversions.mm)
        target_link_libraries(armigo_metal_fragment_output_conversions_gpu_tests PRIVATE
            armigo_metal_pipeline_state)
        add_test(NAME graphics.metal4_fragment_output_conversions
            COMMAND armigo_metal_fragment_output_conversions_gpu_tests)
        set_tests_properties(graphics.metal4_fragment_output_conversions
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 240)
        add_executable(armigo_metal_draw_gpu_tests tests/metal/draw.mm)
        target_compile_options(armigo_metal_draw_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_include_directories(armigo_metal_draw_gpu_tests PRIVATE src/platform/metal)
        target_link_libraries(armigo_metal_draw_gpu_tests PRIVATE armigo_metal_engine_renderer)
        add_test(NAME graphics.metal4_draw
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/tools/metal_draw_tests.py"
                "$<TARGET_FILE:armigo_metal_draw_gpu_tests>")
        set_tests_properties(graphics.metal4_draw
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 60)
        add_executable(armigo_metal_twod_blit_gpu_tests tests/metal/twod_blit.mm)
        target_include_directories(armigo_metal_twod_blit_gpu_tests PRIVATE src)
        target_link_libraries(armigo_metal_twod_blit_gpu_tests PRIVATE
            armigo_metal_twod_blit armigo_metal_renderer armigo_metal_engine_renderer)
        add_test(NAME graphics.metal4_twod_blit COMMAND armigo_metal_twod_blit_gpu_tests)
        set_tests_properties(graphics.metal4_twod_blit
            PROPERTIES LABELS "gpu;metal4;graphics" TIMEOUT 120)
        add_test(NAME metal4.probe COMMAND armigo-headless --probe)
        add_test(NAME metal4.offscreen COMMAND armigo-headless --self-test)
        set_tests_properties(metal4.probe metal4.offscreen
            PROPERTIES LABELS "gpu;metal4" TIMEOUT 30)

        # Lane H: presenter and application tests. The windowed tests create a
        # real window when a display session exists and skip cleanly otherwise.
        add_executable(armigo_metal_presenter_gpu_tests tests/metal/presenter.mm)
        target_link_libraries(armigo_metal_presenter_gpu_tests PRIVATE armigo_platform_presenter)
        add_test(NAME platform.metal4_presenter COMMAND armigo_metal_presenter_gpu_tests)
        set_tests_properties(platform.metal4_presenter
            PROPERTIES LABELS "gpu;metal4;platform" TIMEOUT 60)
        add_executable(armigo_platform_presenter_layer_tests tests/platform/presenter_layer.mm)
        target_link_libraries(armigo_platform_presenter_layer_tests PRIVATE armigo_platform_presenter)
        add_test(NAME platform.presenter_layer COMMAND armigo_platform_presenter_layer_tests)
        set_tests_properties(platform.presenter_layer
            PROPERTIES LABELS "gpu;metal4;platform" TIMEOUT 60)
        add_executable(armigo_platform_presenter_windowed_tests
            tests/platform/presenter_windowed.mm)
        target_link_libraries(armigo_platform_presenter_windowed_tests PRIVATE
            armigo_platform_application)
        add_test(NAME platform.presenter_windowed
            COMMAND armigo_platform_presenter_windowed_tests)
        set_tests_properties(platform.presenter_windowed
            PROPERTIES LABELS "gpu;metal4;platform" TIMEOUT 120)
        add_executable(armigo_platform_application_tests tests/platform/application.mm)
        target_link_libraries(armigo_platform_application_tests PRIVATE
            armigo_platform_application)
        add_test(NAME platform.application COMMAND armigo_platform_application_tests)
        set_tests_properties(platform.application
            PROPERTIES LABELS "gpu;metal4;platform" TIMEOUT 120)
        # The main screen's display link; creates no window.
        add_executable(armigo_platform_display_refresh_tests tests/platform/display_refresh.mm)
        target_include_directories(armigo_platform_display_refresh_tests PRIVATE src/platform/macos)
        target_link_libraries(armigo_platform_display_refresh_tests PRIVATE armigo_platform_application
            "${ARMIGO_APPKIT_FRAMEWORK}" "${ARMIGO_QUARTZCORE_FRAMEWORK}")
        add_test(NAME platform.display_refresh COMMAND armigo_platform_display_refresh_tests)
        set_tests_properties(platform.display_refresh
            PROPERTIES LABELS "platform" TIMEOUT 30 SKIP_RETURN_CODE 77)

        add_executable(armigo_shader_gpu_tests
            tests/metal/shader_execution.mm tests/shader/fixtures.cpp)
        target_include_directories(armigo_shader_gpu_tests PRIVATE tests/shader)
        target_link_libraries(armigo_shader_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_execution COMMAND armigo_shader_gpu_tests)
        set_tests_properties(shader.metal4_execution
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 90)

        add_executable(armigo_rtz_multiply_gpu_tests
            tests/metal/rtz_multiply.mm tests/shader/rtz_multiply_tests.cpp)
        target_compile_definitions(armigo_rtz_multiply_gpu_tests PRIVATE ARMIGO_RTZ_ORACLE_ONLY)
        target_compile_options(armigo_rtz_multiply_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_rtz_multiply_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_rtz_multiply COMMAND armigo_rtz_multiply_gpu_tests)
        set_tests_properties(shader.metal4_rtz_multiply
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_reciprocal_gpu_tests
            tests/metal/reciprocal.mm tests/shader/reciprocal_tests.cpp)
        target_compile_definitions(armigo_reciprocal_gpu_tests PRIVATE ARMIGO_RECIPROCAL_ORACLE_ONLY)
        target_compile_options(armigo_reciprocal_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_reciprocal_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_reciprocal COMMAND armigo_reciprocal_gpu_tests)
        set_tests_properties(shader.metal4_reciprocal
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_rne_multiply_gpu_tests
            tests/metal/rne_multiply.mm tests/shader/rne_multiply_tests.cpp)
        target_compile_definitions(armigo_rne_multiply_gpu_tests PRIVATE ARMIGO_RNE_MULTIPLY_ORACLE_ONLY)
        target_compile_options(armigo_rne_multiply_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_rne_multiply_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_rne_multiply COMMAND armigo_rne_multiply_gpu_tests)
        set_tests_properties(shader.metal4_rne_multiply
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_saturate_gpu_tests
            tests/metal/saturate.mm tests/shader/saturate_tests.cpp)
        target_compile_definitions(armigo_saturate_gpu_tests PRIVATE ARMIGO_SATURATE_ORACLE_ONLY)
        target_link_libraries(armigo_saturate_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_saturate COMMAND armigo_saturate_gpu_tests)
        set_tests_properties(shader.metal4_saturate
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_fma_gpu_tests
            tests/metal/fma.mm tests/shader/fma_tests.cpp)
        target_compile_definitions(armigo_fma_gpu_tests PRIVATE ARMIGO_FMA_ORACLE_ONLY)
        target_compile_options(armigo_fma_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_fma_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_fma COMMAND armigo_fma_gpu_tests)
        set_tests_properties(shader.metal4_fma
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_graphics_shader_gpu_tests tests/metal/graphics_execution.mm)
        target_compile_options(armigo_graphics_shader_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_graphics_shader_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_graphics COMMAND armigo_graphics_shader_gpu_tests)
        set_tests_properties(shader.metal4_graphics
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_metal_compute_execution_tests tests/metal/compute_execution.mm)
        target_include_directories(armigo_metal_compute_execution_tests PRIVATE
            src/platform/metal src/graphics)
        target_link_libraries(armigo_metal_compute_execution_tests PRIVATE
            armigo_metal_compute_dispatch armigo_metal_engine_renderer armigo_graphics
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME metal.compute_execution COMMAND armigo_metal_compute_execution_tests)
        set_tests_properties(metal.compute_execution
            PROPERTIES LABELS "gpu;metal4;compute" TIMEOUT 120)

        add_executable(armigo_graph_msl_gpu_tests tests/metal/graph_execution.mm)
        target_compile_options(armigo_graph_msl_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_graph_msl_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_graph_execution COMMAND armigo_graph_msl_gpu_tests)
        set_tests_properties(shader.metal4_graph_execution
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_graph_texture_gpu_tests tests/metal/graph_texture_execution.mm)
        target_compile_options(armigo_graph_texture_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_graph_texture_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_graph_texture COMMAND armigo_graph_texture_gpu_tests)
        set_tests_properties(shader.metal4_graph_texture
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        # Registered pure operations: CPU reference against real GPU bits.
        add_executable(armigo_operation_gpu_tests tests/metal/operation_execution.mm)
        target_compile_options(armigo_operation_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_operation_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_operations COMMAND armigo_operation_gpu_tests)
        set_tests_properties(shader.metal4_operations
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_word_logic_gpu_tests tests/metal/word_logic_execution.mm)
        target_compile_options(armigo_word_logic_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_word_logic_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_word_logic COMMAND armigo_word_logic_gpu_tests)
        set_tests_properties(shader.metal4_word_logic
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_float_set_predicate_gpu_tests tests/metal/float_set_predicate_execution.mm)
        target_compile_options(armigo_float_set_predicate_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_float_set_predicate_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_float_set_predicate COMMAND armigo_float_set_predicate_gpu_tests)
        set_tests_properties(shader.metal4_float_set_predicate
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_float_to_integer_gpu_tests tests/metal/float_to_integer_execution.mm)
        target_compile_options(armigo_float_to_integer_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_float_to_integer_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_float_to_integer COMMAND armigo_float_to_integer_gpu_tests)
        set_tests_properties(shader.metal4_float_to_integer
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_float_min_max_gpu_tests tests/metal/float_min_max_execution.mm)
        target_compile_options(armigo_float_min_max_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_float_min_max_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_float_min_max COMMAND armigo_float_min_max_gpu_tests)
        set_tests_properties(shader.metal4_float_min_max
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_integer_to_float_gpu_tests
            tests/metal/integer_to_float_execution.mm)
        target_compile_options(armigo_integer_to_float_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_integer_to_float_gpu_tests PRIVATE armigo_shader
            "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_integer_to_float
                 COMMAND armigo_integer_to_float_gpu_tests)
        set_tests_properties(shader.metal4_integer_to_float
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)

        add_executable(armigo_float_round_integral_gpu_tests
            tests/metal/float_round_integral_execution.mm)
        target_compile_options(armigo_float_round_integral_gpu_tests PRIVATE
            -ffp-contract=off -fno-fast-math -frounding-math)
        target_link_libraries(armigo_float_round_integral_gpu_tests PRIVATE
            armigo_shader "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
        add_test(NAME shader.metal4_float_round_integral
                 COMMAND armigo_float_round_integral_gpu_tests)
        set_tests_properties(shader.metal4_float_round_integral
            PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)
    endif()
endif()

add_library(armigo_host_input STATIC
    src/platform/host_input.cpp src/platform/touch_controls.cpp
    src/platform/apple_input.mm src/platform/apple_rumble.mm)
add_library(armigo_platform_display STATIC
    src/platform/display_capabilities.cpp src/platform/${ARMIGO_PLATFORM_DIRECTORY}/display_capabilities.mm)
target_include_directories(armigo_platform_display PUBLIC include)
find_library(ARMIGO_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
target_link_libraries(armigo_platform_display PRIVATE
    "${ARMIGO_UI_FRAMEWORK}" "${ARMIGO_COREGRAPHICS_FRAMEWORK}"
    "${ARMIGO_FOUNDATION_FRAMEWORK}")
target_include_directories(armigo_host_input PUBLIC include)
find_library(ARMIGO_GAME_CONTROLLER_FRAMEWORK GameController REQUIRED)
find_library(ARMIGO_CORE_HAPTICS_FRAMEWORK CoreHaptics REQUIRED)
target_link_libraries(armigo_host_input PRIVATE
    "${ARMIGO_GAME_CONTROLLER_FRAMEWORK}" "${ARMIGO_CORE_HAPTICS_FRAMEWORK}" "${ARMIGO_FOUNDATION_FRAMEWORK}")

if(BUILD_TESTING)
    add_executable(armigo_host_input_tests tests/platform/host_input_tests.cpp)
    target_link_libraries(armigo_host_input_tests PRIVATE armigo_host_input)
    add_test(NAME platform.host_input COMMAND armigo_host_input_tests)
    add_executable(armigo_host_controller_tests tests/runtime/host_controller_tests.cpp)
    target_link_libraries(armigo_host_controller_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.host_controller COMMAND armigo_host_controller_tests)
    add_executable(armigo_native_preparation_tests tests/runtime/native_preparation_tests.cpp)
    target_include_directories(armigo_native_preparation_tests PRIVATE src/loader)
    target_link_libraries(armigo_native_preparation_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.native_preparation COMMAND armigo_native_preparation_tests)
    set_tests_properties(platform.host_input PROPERTIES LABELS "platform" TIMEOUT 30)
    add_executable(armigo_apple_input_tests tests/platform/apple_input.mm)
    target_link_libraries(armigo_apple_input_tests PRIVATE armigo_host_input
        "${ARMIGO_GAME_CONTROLLER_FRAMEWORK}" "${ARMIGO_FOUNDATION_FRAMEWORK}")
    add_test(NAME platform.apple_input COMMAND armigo_apple_input_tests)
    set_tests_properties(platform.apple_input PROPERTIES LABELS "platform" TIMEOUT 30)
endif()

if(BUILD_TESTING)
    add_executable(armigo_thread_lifecycle_tests tests/horizon/thread_lifecycle_tests.cpp)
    target_link_libraries(armigo_thread_lifecycle_tests PRIVATE armigo_horizon)
    add_test(NAME horizon.thread_lifecycle COMMAND armigo_thread_lifecycle_tests)
    add_executable(armigo_thread_runtime_tests tests/runtime/thread_runtime_tests.cpp
        tests/runtime/thread_runtime_fixture.S)
    target_include_directories(armigo_thread_runtime_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_thread_runtime_tests PRIVATE armigo_runtime
        armigo_execution_gate_patch armigo_thread_slot armigo_execution_guest_entry)
    add_test(NAME runtime.thread_runtime COMMAND armigo_thread_runtime_tests)
    set_tests_properties(horizon.thread_lifecycle runtime.thread_runtime
        PROPERTIES LABELS "runtime;horizon;arm64" TIMEOUT 60)
endif()

if(BUILD_TESTING)
    foreach(kind IN ITEMS arbitration mutex_cv event routing)
        add_executable(armigo_wait_cancellation_${kind}_tests
            tests/horizon/wait_cancellation_${kind}_tests.cpp)
        target_link_libraries(armigo_wait_cancellation_${kind}_tests PRIVATE armigo_horizon)
        add_test(NAME horizon.wait_cancellation_${kind}
            COMMAND armigo_wait_cancellation_${kind}_tests)
        set_tests_properties(horizon.wait_cancellation_${kind}
            PROPERTIES LABELS "horizon;threads" TIMEOUT 60)
    endforeach()
endif()

# Lane B: deterministic offline preparation and synthetic launch ownership.
if(NOT ARMIGO_IOS)
    add_executable(armigo-prepare tools/prepare/main.cpp)
    target_link_libraries(armigo-prepare PRIVATE armigo_runtime)
endif()
if(BUILD_TESTING)
    add_executable(armigo_preparation_analysis_tests tests/runtime/preparation_analysis_tests.cpp)
    target_include_directories(armigo_preparation_analysis_tests PRIVATE src/runtime)
    target_link_libraries(armigo_preparation_analysis_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.preparation_analysis COMMAND armigo_preparation_analysis_tests)
    add_executable(armigo_prepared_modules_tests tests/runtime/prepared_modules_tests.cpp)
    target_include_directories(armigo_prepared_modules_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_prepared_modules_tests PRIVATE armigo_runtime)
    add_test(NAME runtime.prepared_modules COMMAND armigo_prepared_modules_tests $<TARGET_FILE:armigo-prepare>)
    add_executable(armigo_launch_plan_tests tests/runtime/launch_plan_tests.cpp)
    target_include_directories(armigo_launch_plan_tests PRIVATE tests/loader src/loader)
    target_link_libraries(armigo_launch_plan_tests PRIVATE armigo_runtime armigo_thread_slot)
    add_test(NAME runtime.launch_plan COMMAND armigo_launch_plan_tests)
    add_executable(armigo_prepared_image_tests tests/execution/prepared_image_tests.cpp)
    target_link_libraries(armigo_prepared_image_tests PRIVATE armigo_execution_prepared_image)
    add_test(NAME execution.prepared_image COMMAND armigo_prepared_image_tests)
    set_tests_properties(runtime.preparation_analysis runtime.prepared_modules
        runtime.launch_plan execution.prepared_image PROPERTIES LABELS "runtime;preparation;arm64" TIMEOUT 60)
endif()

add_subdirectory(src/compat)
add_subdirectory(src/nvn)
if(NOT ARMIGO_IOS)
    # The packer runs on macOS only.
    add_subdirectory(src/pack/cli)
endif()
