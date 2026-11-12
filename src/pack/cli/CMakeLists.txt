# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# The packer is a separate target from the player. The player target is named
# `armigo` and produces Armigo.app; this executable is also named `armigo` and
# is written at the build root, beside Armigo.linker, titles/ and the shader
# tools. A fallback that launches a bare `armigo` when the bundle executable
# is missing would launch this packer.

armigo_find_framework(ARMIGO_PACK_FOUNDATION_FRAMEWORK Foundation)
armigo_find_framework(ARMIGO_PACK_IMAGEIO_FRAMEWORK ImageIO)
armigo_find_framework(ARMIGO_PACK_COREGRAPHICS_FRAMEWORK CoreGraphics)
armigo_find_framework(ARMIGO_PACK_UNIFORMTYPEIDENTIFIERS_FRAMEWORK UniformTypeIdentifiers)
armigo_find_framework(ARMIGO_PACK_CORESERVICES_FRAMEWORK CoreServices)

add_library(armigo_pack STATIC
    naming.cpp
    command.cpp
    run.cpp
    space.cpp
    settings.cpp
    bundle.cpp
    firmware.cpp
    icon.cpp
    icon.mm
    ios.cpp
    package_tools.cpp
    launch_services.mm)
target_include_directories(armigo_pack
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/shader"
        "${CMAKE_BINARY_DIR}/generated")
target_link_libraries(armigo_pack
    PUBLIC
        armigo_pack_nacp
        armigo_pack_inputs
        armigo_compat
        armigo_shader_package
        "${ARMIGO_PACK_FOUNDATION_FRAMEWORK}"
        "${ARMIGO_PACK_IMAGEIO_FRAMEWORK}"
        "${ARMIGO_PACK_COREGRAPHICS_FRAMEWORK}"
        "${ARMIGO_PACK_UNIFORMTYPEIDENTIFIERS_FRAMEWORK}"
        "${ARMIGO_PACK_CORESERVICES_FRAMEWORK}")
target_compile_features(armigo_pack PUBLIC cxx_std_20)

if(BUILD_TESTING)
    add_executable(armigo_pack_cli_tests
        "${CMAKE_SOURCE_DIR}/tests/pack/cli/pack_cli_tests.cpp"
        pack_flow.cpp
        "${CMAKE_SOURCE_DIR}/src/platform/metal/pso_manifest.cpp")
    target_include_directories(armigo_pack_cli_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/src/platform/metal")
    target_link_libraries(armigo_pack_cli_tests PRIVATE armigo_pack)
    add_executable(armigo_pack_pipeline_warmup_tests
        "${CMAKE_SOURCE_DIR}/tests/pack/cli/pipeline_warmup_tests.cpp" pipelines.cpp)
    target_link_libraries(armigo_pack_pipeline_warmup_tests PRIVATE armigo_pack armigo_nvn_metal)
    add_test(NAME pack.pipeline_warmup COMMAND armigo_pack_pipeline_warmup_tests)
    set_tests_properties(pack.pipeline_warmup PROPERTIES LABELS pack TIMEOUT 60)
    add_executable(armigo_pack_localization_tests
        "${CMAKE_SOURCE_DIR}/tests/pack/cli/bundle_localization_tests.mm")
    target_link_libraries(armigo_pack_localization_tests PRIVATE armigo_pack)
    add_test(NAME pack.localizations COMMAND armigo_pack_localization_tests)
    set_tests_properties(pack.localizations PROPERTIES LABELS pack TIMEOUT 60)
    add_test(NAME pack.cli COMMAND armigo_pack_cli_tests)
    set_tests_properties(pack.cli PROPERTIES LABELS pack TIMEOUT 60)
    add_test(NAME pack.commit
        COMMAND "${CMAKE_COMMAND}"
                "-DSCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/write_pack_commit.cmake"
                -P "${CMAKE_SOURCE_DIR}/tests/pack/cli/pack_commit_tests.cmake")
    set_tests_properties(pack.commit PROPERTIES LABELS pack TIMEOUT 60)
endif()

set(ARMIGO_PACK_COMMIT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/generated/pack_commit.h")
add_custom_target(armigo_pack_commit
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${CMAKE_COMMAND}"
            "-DSOURCE_DIR=${CMAKE_SOURCE_DIR}"
            "-DOUTPUT=${ARMIGO_PACK_COMMIT_HEADER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/write_pack_commit.cmake"
    BYPRODUCTS "${ARMIGO_PACK_COMMIT_HEADER}"
    VERBATIM)

add_executable(armigo_pack_cli
    main.cpp
    pack_flow.cpp
    shaders.cpp
    pipelines.cpp
    native_shaders.cpp)
set_source_files_properties(main.cpp PROPERTIES OBJECT_DEPENDS "${ARMIGO_PACK_COMMIT_HEADER}")
set_target_properties(armigo_pack_cli PROPERTIES
    OUTPUT_NAME armigo
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_include_directories(armigo_pack_cli PRIVATE
    "${CMAKE_SOURCE_DIR}/src/loader"
    "${CMAKE_SOURCE_DIR}/src/platform/metal"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(armigo_pack_cli
    PRIVATE
        armigo_pack
        armigo_pack_inputs
        armigo_pack_ios
        armigo_shader_package
        armigo_shader_ios_package
        armigo_shader_translation
        armigo_shader_native_frontend
        armigo_shader_native_package
        armigo_loader
        armigo_runtime
        armigo_nvn_metal)
if(TARGET armigo_shader_native_package)
    add_executable(armigo-native-shader-package native_tool.cpp native_shaders.cpp)
    target_include_directories(armigo-native-shader-package PRIVATE
        "${CMAKE_SOURCE_DIR}/src/platform/metal"
        "${CMAKE_SOURCE_DIR}/src/loader")
    target_link_libraries(armigo-native-shader-package PRIVATE
        armigo_pack
        armigo_shader_native_frontend
        armigo_shader_native_package
        armigo_shader_package
        armigo_loader
        armigo_nvn_metal)
    set_target_properties(armigo-native-shader-package PROPERTIES
        OUTPUT_NAME armigo-native-shader-package
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    # The title settings beside the tool select a .nup title's shader inputs.
    add_dependencies(armigo-native-shader-package armigo_titles)
endif()
if(BUILD_TESTING AND TARGET armigo_shader_native_package)
    # Pins the stages each shader reader yields from synthetic containers.
    add_executable(armigo_pack_shader_golden_tests "${CMAKE_SOURCE_DIR}/tests/pack/cli/shader_golden_tests.cpp"
        shaders.cpp native_shaders.cpp)
    target_include_directories(armigo_pack_shader_golden_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/src/loader"
        "${CMAKE_SOURCE_DIR}/src/platform/metal"
        "${CMAKE_SOURCE_DIR}/tests/loader"
        "${CMAKE_SOURCE_DIR}/tests/shader_input"
        "${CMAKE_SOURCE_DIR}/tests/shader_native")
    target_link_libraries(armigo_pack_shader_golden_tests PRIVATE
        armigo_pack
        armigo_shader_package
        armigo_shader_translation
        armigo_shader_native_frontend
        armigo_shader_native_package
        armigo_loader
        armigo_nvn_metal)
    add_test(NAME pack.shader_golden COMMAND armigo_pack_shader_golden_tests
        "${CMAKE_SOURCE_DIR}/tests/pack/shader_golden.txt")
    set_tests_properties(pack.shader_golden PROPERTIES LABELS pack TIMEOUT 120)
endif()
add_dependencies(armigo_pack_cli armigo armigo-headless armigo_titles armigo-shader-translate armigo-prepare
    armigo_pack_commit)
