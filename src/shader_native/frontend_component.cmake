# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Compilation and fingerprinting share this declared production manifest.
# Input/ISA implementation tools and tests remain in the parent build file.
function(armigo_native_frontend_production_files output)
    cmake_path(SET native_root NORMALIZE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    cmake_path(SET native_repository NORMALIZE "${native_root}/../..")
    set(native_files)
    foreach(file IN ITEMS
            frontend.cpp arithmetic.cpp integer.cpp conversion.cpp flow.cpp
            texture.cpp memory.cpp system.cpp local.cpp gradient.cpp shared.cpp
            half.cpp tensor.cpp stage_buffer.cpp stage_memory.cpp mesh_stage.cpp
            handle_program.cpp builder.h arithmetic.h stage_memory.h mesh_stage.h)
        list(APPEND native_files "${native_root}/${file}")
    endforeach()
    foreach(file IN ITEMS frontend.h program.h instruction.h input.h)
        list(APPEND native_files "${native_repository}/include/armigo/shader_native/${file}")
    endforeach()
    set(${output} ${native_files} PARENT_SCOPE)
endfunction()

function(armigo_add_native_frontend_component)
    armigo_native_frontend_production_files(native_sources)
    add_library(armigo_shader_native_frontend STATIC ${native_sources})
    target_link_libraries(armigo_shader_native_frontend PUBLIC armigo_shader_native armigo_shader_pipeline)
endfunction()
