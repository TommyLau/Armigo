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

find_library(ARMIGO_PACK_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ARMIGO_PACK_IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(ARMIGO_PACK_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(ARMIGO_PACK_UNIFORMTYPEIDENTIFIERS_FRAMEWORK UniformTypeIdentifiers REQUIRED)

add_library(armigo_pack STATIC
    naming.cpp
    command.cpp
    space.cpp
    settings.cpp
    bundle.cpp
    firmware.cpp
    icon.cpp
    icon.mm
    ios.cpp)
target_include_directories(armigo_pack
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(armigo_pack
    PUBLIC
        armigo_pack_nacp
        armigo_pack_inputs
        "${ARMIGO_PACK_FOUNDATION_FRAMEWORK}"
        "${ARMIGO_PACK_IMAGEIO_FRAMEWORK}"
        "${ARMIGO_PACK_COREGRAPHICS_FRAMEWORK}"
        "${ARMIGO_PACK_UNIFORMTYPEIDENTIFIERS_FRAMEWORK}")
target_compile_features(armigo_pack PUBLIC cxx_std_20)

if(BUILD_TESTING)
    add_executable(armigo_pack_cli_tests
        "${CMAKE_SOURCE_DIR}/tests/pack/cli/pack_cli_tests.cpp")
    target_link_libraries(armigo_pack_cli_tests PRIVATE armigo_pack)
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
    shaders.cpp
    pipelines.cpp)
set_source_files_properties(main.cpp PROPERTIES OBJECT_DEPENDS "${ARMIGO_PACK_COMMIT_HEADER}")
set_target_properties(armigo_pack_cli PROPERTIES
    OUTPUT_NAME armigo
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_include_directories(armigo_pack_cli PRIVATE
    "${CMAKE_SOURCE_DIR}/src/loader"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(armigo_pack_cli
    PRIVATE
        armigo_pack
        armigo_pack_inputs
        armigo_pack_ios
        armigo_shader_package
        armigo_shader_ios_package
        armigo_shader_translation
        armigo_runtime
        armigo_nvn_metal)
add_dependencies(armigo_pack_cli armigo armigo_titles armigo-shader-translate armigo-prepare armigo_pack_commit)
