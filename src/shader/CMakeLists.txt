# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# Armigo native shader foundation: typed scalar IR, validation, Metal Shading
# Language emission, and a host reference evaluator.
#
# The parent project adds this directory with add_subdirectory(src/shader).
# Paths are resolved relative to this file so the directory also builds when
# added from an out-of-tree harness.

cmake_path(SET ARMIGO_SHADER_REPOSITORY_ROOT NORMALIZE
           "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(ARMIGO_SHADER_INCLUDE_DIR "${ARMIGO_SHADER_REPOSITORY_ROOT}/include")
set(ARMIGO_SHADER_TEST_DIR "${ARMIGO_SHADER_REPOSITORY_ROOT}/tests/shader")

if(NOT TARGET armigo_shader_input)
    add_subdirectory("${ARMIGO_SHADER_REPOSITORY_ROOT}/src/shader_input"
                     "${CMAKE_CURRENT_BINARY_DIR}/shader_input")
endif()

# Registered pure graphics operations: one translation unit per kind.
# New kinds append their unit here and one immutable registry entry.
set(ARMIGO_SHADER_OPERATION_SOURCES
    operations/float_sign.cpp
    operations/integer_to_float.cpp
    operations/predicate_logic.cpp
    operations/select.cpp
    operations/word_logic.cpp
    operations/float_to_integer.cpp
    operations/float_min_max.cpp
    operations/float_set_predicate.cpp
    operations/float_round_integral.cpp
)

add_library(armigo_shader
    diagnostics.cpp
    ${ARMIGO_SHADER_OPERATION_SOURCES}
    graphics_operations.cpp
    graphics_operations.h
    families/float_min_max_decode.cpp
    families/float_min_max_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/float_min_max.h"
    families/word_and_immediate_decode.cpp
    families/word_and_immediate_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/word_and_immediate.h"
    families/fma_register_constant_decode.cpp
    families/fma_register_constant_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/fma_register_constant.h"
    families/fma_constant_negated_decode.cpp
    families/fma_constant_negated_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/fma_constant_negated.h"
    families/mov_constant_decode.cpp
    families/mov_constant_lower.cpp
    families/mov_register_decode.cpp
    families/mov_register_lower.cpp
    families/isetp_register_decode.cpp
    families/predicate_logic_decode.cpp
    families/fmul_saturate_register_decode.cpp
    families/fmul_saturate_register_lower.cpp
    families/ffma_register_decode.cpp
    families/ffma_register_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/ffma_register.h"
    families/fmul_immediate32_decode.cpp
    families/fmul_immediate32_lower.cpp
    families/fadd_immediate32_decode.cpp
    families/fadd_immediate32_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/float_immediate32.h"
    families/word_logic_decode.cpp
    families/word_logic_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/word_logic.h"
    families/float_round_integral_decode.cpp
    families/float_round_integral_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/float_round_integral.h"
    families/select_by_predicate_decode.cpp
    families/select_by_predicate_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/select_by_predicate.h"
    families/integer_to_float_decode.cpp
    families/integer_to_float_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/integer_to_float.h"
    families/range_reduction_decode.cpp
    families/range_reduction_lower.cpp
    families/dependency_barrier_decode.cpp
    families/dependency_barrier_lower.cpp
    families/float_to_integer_decode.cpp
    families/float_to_integer_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/float_to_integer.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/predicate_logic.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/range_reduction.h"
    families/float_set_predicate_decode.cpp
    families/float_set_predicate_lower.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/float_set_predicate.h"
    family_registry_validation.h
    fma.cpp
    forward_branch.cpp
    forward_metadata.cpp
    forward_metadata.h
    forward_lower.cpp
    forward_state.cpp
    forward_state.h
    forward_program.cpp
    forward_reconvergence.cpp
    forward_reconvergence.h
    families/ssy_lower.cpp
    graphics.cpp
    graphics_builder.h
    graphics_evaluate.cpp
    graphics_control_flow.cpp
    graphics_validation.h
    graphics_families.cpp
    graphics_families.h
    graphics_msl.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/graphics_control_flow.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/graphics_forward.h"
    ir.cpp
    maxwell.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/families/mov_constant.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/maxwell_flow.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/maxwell_forward_program.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/maxwell_operands.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/maxwell_predicates.h"
    maxwell_decode_helpers.h
    maxwell_forward_decode.h
    maxwell_families.cpp
    maxwell_families.h
    msl_emitter.cpp
    program_identity.cpp
    reciprocal.cpp
    reference_evaluator.cpp
    rne_multiply.cpp
    rtz_multiply.cpp
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/saturate.h"
    saturate.cpp
    validation.cpp
)
add_library(armigo::shader ALIAS armigo_shader)
target_link_libraries(armigo_shader PUBLIC armigo_shader_input)

# Parallel shader port contracts. This target is intentionally not linked into
# the existing translator until its phase implementations pass integration.
add_library(armigo_shader_pipeline STATIC
    pipeline/types.h
    pipeline/instruction_name.h
    pipeline/opcode.h
    pipeline/fact_types.h
    pipeline/diagnostic.h
    pipeline/facts.h
    pipeline/facts.cpp
    pipeline/ir.h
    pipeline/validation.cpp
    pipeline/decoder.h
    pipeline/resources.h
    pipeline/structured.h
    pipeline/emitter_context.h
    pipeline/phases.h
    pipeline/msl_context.h
    pipeline/emitter_registry.h
    pipeline/msl_registry.h
    pipeline/emit/integer_emit.h
    pipeline/emit/integer_arithmetic.cpp
    pipeline/emit/integer_comparison.cpp
    pipeline/emit/integer_logical.cpp
    pipeline/emit/integer_shift_bitfield.cpp
    pipeline/emit/integer_move_predicate.cpp
    pipeline/dispatch.h
    pipeline/dispatch.cpp
    pipeline/flow/flow.h
    pipeline/flow/flow_decode.cpp
    pipeline/flow/flow_emit.cpp
    pipeline/emit/float_helpers.h
    pipeline/emit/float_helpers.cpp
    pipeline/emit/float_arithmetic.cpp
    pipeline/emit/float_comparison.cpp
    pipeline/emit/float_conversion.cpp
    pipeline/emit/float_half.cpp
    pipeline/emit/float_min_max.cpp
    pipeline/emit/float_multifunction.cpp
    pipeline/emit/float_swizzle_add.cpp
    pipeline/emit/warp_shuffle.cpp
    pipeline/decode/resource_fields.h
    pipeline/decode/resource_decode.cpp
    pipeline/ssa/dominance.h
    pipeline/ssa/dominance.cpp
    pipeline/ssa/ssa.cpp
    pipeline/decode/alu_decode.cpp
)
add_library(armigo::shader_pipeline ALIAS armigo_shader_pipeline)

# One stage's translation through the ported pipeline, shared by the offline
# translator and the run-time preparation.
add_library(armigo_shader_translation STATIC translation/stage_translation.cpp
    translation/stage_cache.cpp)
target_include_directories(armigo_shader_translation PRIVATE
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/loader")
target_link_libraries(armigo_shader_translation PUBLIC armigo_shader armigo_shader_input
    PRIVATE armigo_shader_pipeline armigo_loader)
target_compile_features(armigo_shader_translation PUBLIC cxx_std_20)
target_include_directories(armigo_shader_pipeline PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_compile_features(armigo_shader_pipeline PUBLIC cxx_std_20)
set_target_properties(armigo_shader_pipeline PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(armigo_shader_pipeline PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(BUILD_TESTING)
    add_executable(armigo_shader_pipeline_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/contracts_tests.cpp")
    target_link_libraries(armigo_shader_pipeline_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_contracts COMMAND armigo_shader_pipeline_tests)
    set_tests_properties(shader.pipeline_contracts PROPERTIES LABELS "shader" TIMEOUT 30)
    add_executable(armigo_pipeline_flow_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/flow_tests.cpp")
    target_link_libraries(armigo_pipeline_flow_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_flow COMMAND armigo_pipeline_flow_tests)
    set_tests_properties(shader.pipeline_flow PROPERTIES LABELS "shader" TIMEOUT 60)

    add_executable(armigo_pipeline_emit_integer_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/emit_integer_tests.cpp")
    target_link_libraries(armigo_pipeline_emit_integer_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_emit_integer COMMAND armigo_pipeline_emit_integer_tests)
    set_tests_properties(shader.pipeline_emit_integer PROPERTIES LABELS "shader" TIMEOUT 60)
    add_executable(armigo_pipeline_emit_float_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/emit_float_tests.cpp")
    target_link_libraries(armigo_pipeline_emit_float_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_emit_float COMMAND armigo_pipeline_emit_float_tests)
    set_tests_properties(shader.pipeline_emit_float PROPERTIES LABELS "shader" TIMEOUT 30)

    # Lane-owned pipeline decode tests: one executable per lane table.
    add_executable(armigo_pipeline_decode_resource_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/decode_resource_tests.cpp")
    target_link_libraries(armigo_pipeline_decode_resource_tests PRIVATE armigo_shader_pipeline)
    set_target_properties(armigo_pipeline_decode_resource_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(armigo_pipeline_decode_resource_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME shader.pipeline_decode_resource COMMAND armigo_pipeline_decode_resource_tests)
    set_tests_properties(shader.pipeline_decode_resource PROPERTIES LABELS "shader" TIMEOUT 30)
    add_executable(armigo_pipeline_ssa_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/ssa_tests.cpp")
    target_link_libraries(armigo_pipeline_ssa_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_ssa COMMAND armigo_pipeline_ssa_tests)
    set_tests_properties(shader.pipeline_ssa PROPERTIES LABELS "shader" TIMEOUT 30)
    add_executable(armigo_shader_translation_tests
        "${ARMIGO_SHADER_TEST_DIR}/translation/stage_translation_tests.cpp")
    target_link_libraries(armigo_shader_translation_tests PRIVATE armigo_shader_translation)
    add_test(NAME shader.stage_translation COMMAND armigo_shader_translation_tests)
    set_tests_properties(shader.stage_translation PROPERTIES LABELS "shader" TIMEOUT 60)
    add_executable(armigo_pipeline_decode_alu_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/decode_alu_tests.cpp")
    target_link_libraries(armigo_pipeline_decode_alu_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_decode_alu COMMAND armigo_pipeline_decode_alu_tests)
    set_tests_properties(shader.pipeline_decode_alu PROPERTIES LABELS "shader" TIMEOUT 30)

    add_executable(armigo_shader_package_tests
        "${ARMIGO_SHADER_TEST_DIR}/package/package_tests.cpp")
    target_link_libraries(armigo_shader_package_tests PRIVATE armigo_shader_package)
    add_test(NAME shader.package COMMAND armigo_shader_package_tests)
    set_tests_properties(shader.package PROPERTIES LABELS "shader;pack" TIMEOUT 60)
endif()

add_library(armigo_shader_package STATIC
    package.cpp
    package_json.cpp
    package_validation.cpp)
target_include_directories(armigo_shader_package PUBLIC "${ARMIGO_SHADER_INCLUDE_DIR}")
target_compile_features(armigo_shader_package PUBLIC cxx_std_20)
set_target_properties(armigo_shader_package PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(armigo_shader_package PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Fingerprint declared production sources, so unrelated or unfinished files in
# a shared checkout do not silently become part of the translation chain. The
# root build file is left out: it declares other targets, and the shader
# libraries' own sources and build file are listed here.
set(ARMIGO_SHADER_VERSION_FILES
    "${CMAKE_CURRENT_LIST_FILE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/source_version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/translate.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/chain.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_package.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/package.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/package_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/package_validation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/package_internal.h"
    "${ARMIGO_SHADER_INCLUDE_DIR}/armigo/shader/package.h"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/shader_input/CMakeLists.txt"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/shader_input/tools/extract.cpp"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/loader/CMakeLists.txt"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/include/armigo/loader/nso_module.h"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/include/armigo/loader/lz4_block.h"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/loader/nso_module.cpp"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/loader/lz4_block.cpp"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/loader/sha256.h"
    "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/loader/sha256_commoncrypto.cpp")
foreach(component IN ITEMS shader shader_input shader_pipeline shader_translation)
    get_target_property(component_sources armigo_${component} SOURCES)
    get_target_property(component_directory armigo_${component} SOURCE_DIR)
    foreach(component_source IN LISTS component_sources)
        cmake_path(ABSOLUTE_PATH component_source BASE_DIRECTORY "${component_directory}"
                   NORMALIZE OUTPUT_VARIABLE component_path)
        list(APPEND ARMIGO_SHADER_VERSION_FILES "${component_path}")
        cmake_path(GET component_source STEM component_stem)
        set(component_header "${ARMIGO_SHADER_REPOSITORY_ROOT}/include/armigo/${component}/${component_stem}.h")
        if(EXISTS "${component_header}")
            list(APPEND ARMIGO_SHADER_VERSION_FILES "${component_header}")
        endif()
    endforeach()
endforeach()
set(armigo_shader_normalized_version_files)
foreach(version_file IN LISTS ARMIGO_SHADER_VERSION_FILES)
    cmake_path(NORMAL_PATH version_file OUTPUT_VARIABLE normalized_version_file)
    list(APPEND armigo_shader_normalized_version_files "${normalized_version_file}")
endforeach()
set(ARMIGO_SHADER_VERSION_FILES ${armigo_shader_normalized_version_files})
list(REMOVE_DUPLICATES ARMIGO_SHADER_VERSION_FILES)
list(SORT ARMIGO_SHADER_VERSION_FILES)
set(armigo_shader_fingerprint "armigo-shader-source-v1\n")
foreach(version_file IN LISTS ARMIGO_SHADER_VERSION_FILES)
    file(RELATIVE_PATH version_label "${ARMIGO_SHADER_REPOSITORY_ROOT}" "${version_file}")
    file(SHA256 "${version_file}" version_digest)
    string(APPEND armigo_shader_fingerprint "${version_label}:${version_digest}\n")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ARMIGO_SHADER_VERSION_FILES})
string(SHA256 ARMIGO_SHADER_SOURCE_ID "${armigo_shader_fingerprint}")
configure_file(source_version.h.in generated/armigo/shader/source_version.h @ONLY)

target_include_directories(armigo_shader PUBLIC
    "$<BUILD_INTERFACE:${ARMIGO_SHADER_INCLUDE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>"
)
target_compile_features(armigo_shader PUBLIC cxx_std_20)
set_target_properties(armigo_shader PROPERTIES CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(armigo_shader PRIVATE -Wall -Wextra -Wpedantic)
    # The reference evaluator defines results as separately rounded binary32
    # operations; forbid contraction into fused multiply-add.
    set_source_files_properties(reference_evaluator.cpp PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
    set_source_files_properties(graphics_evaluate.cpp PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math;-frounding-math")
    # Operation units are CPU references evaluated inside the same floating
    # scope, so they follow the evaluator's floating contract.
    set_source_files_properties(${ARMIGO_SHADER_OPERATION_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math;-frounding-math")
elseif(MSVC)
    target_compile_options(armigo_shader PRIVATE /W4 /permissive-)
    set_source_files_properties(reference_evaluator.cpp graphics_evaluate.cpp
        ${ARMIGO_SHADER_OPERATION_SOURCES} PROPERTIES COMPILE_OPTIONS "/fp:strict")
endif()

if(BUILD_TESTING)
    add_executable(armigo_shader_tests
        "${ARMIGO_SHADER_TEST_DIR}/emitter_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/fixtures.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/fma_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/fma_register_constant_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/ffma_register_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/forward_branch_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/forward_reconvergence_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/forward_lower_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/fma_constant_negated_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/forward_metadata_tests.cpp"
    "${ARMIGO_SHADER_TEST_DIR}/forward_program_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/forward_registered_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/select_by_predicate_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/word_logic_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/integer_to_float_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/fmul_saturate_register_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/fmul_saturate_register_lower_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/float_immediate32_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/float_to_integer_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/float_round_integral_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/word_and_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graphics_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graphics_builder_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graphics_control_flow_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graph_cpu_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graph_cpu_execution_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graph_msl_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graphics_msl_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/graphics_saturate_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/isetp_register_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/predicate_logic_decode_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/maxwell_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/mov_constant_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/operation_registry_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/range_reduction_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/float_min_max_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/float_set_predicate_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/program_identity_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/reference_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/reciprocal_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/rne_multiply_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/rtz_multiply_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/saturate_tests.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/test_main.cpp"
        "${ARMIGO_SHADER_TEST_DIR}/validation_tests.cpp"
    )
    target_include_directories(armigo_shader_tests PRIVATE
        "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/loader"
        "${ARMIGO_SHADER_REPOSITORY_ROOT}/src/shader")
    target_link_libraries(armigo_shader_tests PRIVATE armigo_shader armigo_loader)
    set_target_properties(armigo_shader_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(armigo_shader_tests PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_source_files_properties("${ARMIGO_SHADER_TEST_DIR}/rtz_multiply_tests.cpp"
            "${ARMIGO_SHADER_TEST_DIR}/reciprocal_tests.cpp"
            "${ARMIGO_SHADER_TEST_DIR}/rne_multiply_tests.cpp"
            "${ARMIGO_SHADER_TEST_DIR}/graphics_tests.cpp"
            "${ARMIGO_SHADER_TEST_DIR}/fma_tests.cpp"
            PROPERTIES COMPILE_OPTIONS "-frounding-math")
    endif()

    # One test group per line: new families append their own line.
    foreach(group IN ITEMS
            validation
            emission
            reference
            rounding
            maxwell
            mov_constant
            reciprocal
            graphics
            rne_multiply
            fma
            fma_rc
            word_and
            graphics_msl
            program_identity
            isetp_register
            saturate
            graphics_control_flow
            forward_branch
            forward_reconvergence
            predicate_logic
            fma_cn
            forward_metadata
            graph_cpu_execution
            forward_program
            graphics_builder
            fmul_saturate_register
            graphics_saturate
            fmul_saturate_register_lower
            forward_lower
            forward_lower_msl
            operation_registry
            forward_registered
            select_by_predicate
            word_logic
            integer_to_float
            ffma_register
            float_immediate32
            range_reduction
            float_to_integer
            float_min_max
            float_set_predicate
            float_round_integral
    )
        add_test(NAME shader.${group}
                 COMMAND armigo_shader_tests ${group})
    endforeach()

    # Compile every emitted fixture with the Metal toolchain when available.
    # Generated sources and AIR files stay in the build tree.
    if(APPLE)
        find_program(ARMIGO_XCRUN xcrun)
        if(ARMIGO_XCRUN)
            execute_process(
                COMMAND "${ARMIGO_XCRUN}" -sdk macosx -f metal
                RESULT_VARIABLE armigo_metal_lookup
                OUTPUT_QUIET ERROR_QUIET)
            if(armigo_metal_lookup EQUAL 0)
                add_test(NAME shader.msl_compile
                    COMMAND "${CMAKE_COMMAND}"
                        "-DTEST_EXECUTABLE=$<TARGET_FILE:armigo_shader_tests>"
                        "-DXCRUN=${ARMIGO_XCRUN}"
                        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/msl_fixtures"
                        -P "${ARMIGO_SHADER_TEST_DIR}/compile_msl_fixtures.cmake")
                add_test(NAME shader.forward_lower_msl_compile
                    COMMAND "${CMAKE_COMMAND}"
                        "-DTEST_EXECUTABLE=$<TARGET_FILE:armigo_shader_tests>"
                        "-DXCRUN=${ARMIGO_XCRUN}"
                        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/forward_msl_fixtures"
                        -P "${ARMIGO_SHADER_TEST_DIR}/compile_forward_msl_fixtures.cmake")
            else()
                message(STATUS
                    "Armigo shader: Metal compiler not found; "
                    "shader.msl_compile test disabled")
            endif()
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# Ported shader pipeline lanes. Each lane appends its
# sources and test executable here; the consolidated fingerprint refresh at the
# end of this file hashes every armigo_shader_pipeline source.

# Lane 4: concrete EmitterContext and the emit_ir phase.
target_sources(armigo_shader_pipeline PRIVATE
    pipeline/emit/context_ir.h
    pipeline/emit/context_ir.cpp
    pipeline/ir/emit_ir.h
    pipeline/ir/emit_ir.cpp)
if(BUILD_TESTING)
    add_executable(armigo_pipeline_ir_context_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/ir_context_tests.cpp")
    target_link_libraries(armigo_pipeline_ir_context_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_ir_context COMMAND armigo_pipeline_ir_context_tests)
    set_tests_properties(shader.pipeline_ir_context PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Lane 10: resources, legalize orchestration and plan_resources.
target_sources(armigo_shader_pipeline PRIVATE
    pipeline/resources/detail.h
    pipeline/resources/pass_bridge.h
    pipeline/resources/validation.cpp
    pipeline/resources/legalize.cpp
    pipeline/resources/plan.cpp
    pipeline/resources/vertex_class.cpp)
if(BUILD_TESTING)
    add_executable(armigo_pipeline_resources_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/resources_tests.cpp")
    target_link_libraries(armigo_pipeline_resources_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_resources COMMAND armigo_pipeline_resources_tests)
    set_tests_properties(shader.pipeline_resources PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Lane 12: structured control flow and edge-local phi destruction.
target_sources(armigo_shader_pipeline PRIVATE
    pipeline/structured/internal.h
    pipeline/structured/phi.cpp
    pipeline/structured/regions.cpp
    pipeline/structured/structure.cpp
    pipeline/structured/validation.cpp)
if(BUILD_TESTING)
    add_executable(armigo_pipeline_structured_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/structured_tests.cpp")
    target_link_libraries(armigo_pipeline_structured_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_structured COMMAND armigo_pipeline_structured_tests)
    set_tests_properties(shader.pipeline_structured PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Lane 14: structured MSL stage, resource, IO and texture emission.
target_sources(armigo_shader_pipeline PRIVATE
    pipeline/msl/stage_internal.h
    pipeline/msl/custom_border.h
    pipeline/msl/stage_emit.cpp
    pipeline/msl/resources_declarations.cpp
    pipeline/msl/resources_memory.cpp
    pipeline/msl/resources_texture.cpp
    pipeline/msl/vertex_fetch.cpp)
target_link_libraries(armigo_shader_pipeline PRIVATE armigo_shader)
if(BUILD_TESTING)
    add_executable(armigo_shader_pipeline_msl_stage_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/msl_stage_tests.cpp")
    target_link_libraries(armigo_shader_pipeline_msl_stage_tests PRIVATE armigo_shader_pipeline armigo_shader)
    add_test(NAME shader.pipeline_msl_stage COMMAND armigo_shader_pipeline_msl_stage_tests)
    set_tests_properties(shader.pipeline_msl_stage PROPERTIES LABELS "shader" TIMEOUT 60)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_XCRUN AND armigo_metal_lookup EQUAL 0)
    add_test(NAME shader.pipeline_msl_stage_compile
        COMMAND armigo_shader_pipeline_msl_stage_tests
            "${CMAKE_CURRENT_BINARY_DIR}/pipeline_stage_fixtures" --metal)
    set_tests_properties(shader.pipeline_msl_stage_compile PROPERTIES LABELS "shader" TIMEOUT 60)
endif()

# Lane 9: texture instruction expansion and independently authored synthetic tests.
target_sources(armigo_shader_pipeline PRIVATE pipeline/emit/texture_emit.cpp)
if(BUILD_TESTING)
    add_executable(armigo_pipeline_emit_texture_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/emit_texture_tests.cpp")
    target_link_libraries(armigo_pipeline_emit_texture_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_emit_texture COMMAND armigo_pipeline_emit_texture_tests)
    set_tests_properties(shader.pipeline_emit_texture PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Texture decode/emit alignment: synthetic guest words run through the resource
# decode rules and the registered texture emitter into the recording context.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_texture_decode_emit_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/texture_decode_emit_tests.cpp")
    target_link_libraries(armigo_pipeline_texture_decode_emit_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_texture_decode_emit COMMAND armigo_pipeline_texture_decode_emit_tests)
    set_tests_properties(shader.pipeline_texture_decode_emit PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Emit lane 8: attribute, interpolation, memory and barrier instruction
# emitters, registered by emitter_registry.h rows.
target_sources(armigo_shader_pipeline PRIVATE
    pipeline/emit/io_memory_shared.h
    pipeline/emit/io_memory_attribute.cpp
    pipeline/emit/io_memory_constant.cpp
    pipeline/emit/io_memory_local.cpp
    pipeline/emit/io_memory_global.cpp
    pipeline/emit/io_memory_barrier.cpp)
if(BUILD_TESTING)
    add_executable(armigo_pipeline_emit_io_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/emit_io_tests.cpp")
    target_link_libraries(armigo_pipeline_emit_io_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_emit_io COMMAND armigo_pipeline_emit_io_tests)
    set_tests_properties(shader.pipeline_emit_io PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Lane 11: mandatory cleanup and nontexture legalization.
target_sources(armigo_shader_pipeline PRIVATE
    pipeline/passes/passes.h
    pipeline/passes/internal.h
    pipeline/passes/folding.cpp
    pipeline/passes/cleanup.cpp
    pipeline/passes/nontexture.cpp)
if(BUILD_TESTING)
    add_executable(armigo_pipeline_passes_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/passes_tests.cpp")
    target_link_libraries(armigo_pipeline_passes_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_passes COMMAND armigo_pipeline_passes_tests)
    set_tests_properties(shader.pipeline_passes PROPERTIES LABELS "shader" TIMEOUT 60)
endif()

# Lane 13: MSL expression and numeric generation for pure IR operations.
target_sources(armigo_shader_pipeline PRIVATE
    pipeline/msl/expressions.h
    pipeline/msl/expressions.cpp
    pipeline/msl/numeric.cpp)
if(BUILD_TESTING)
    add_executable(armigo_pipeline_msl_expr_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/msl_expr_tests.cpp")
    target_link_libraries(armigo_pipeline_msl_expr_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_msl_expr COMMAND armigo_pipeline_msl_expr_tests)
    set_tests_properties(shader.pipeline_msl_expr PROPERTIES LABELS "shader" TIMEOUT 60)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_XCRUN AND armigo_metal_lookup EQUAL 0)
    add_test(NAME shader.pipeline_msl_expr_compile
        COMMAND armigo_pipeline_msl_expr_tests
            "${CMAKE_CURRENT_BINARY_DIR}/pipeline_msl_expr_fixtures" --metal)
    set_tests_properties(shader.pipeline_msl_expr_compile PROPERTIES LABELS "shader" TIMEOUT 120)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    # The parent finds these frameworks after adding this directory; look them
    # up here too so a fresh configure links the GPU test.
    find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)
    target_sources(armigo_pipeline_msl_expr_tests PRIVATE
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/msl_expr_gpu.mm")
    target_compile_definitions(armigo_pipeline_msl_expr_tests PRIVATE ARMIGO_MSL_EXPR_GPU)
    set_source_files_properties("${ARMIGO_SHADER_TEST_DIR}/pipeline/msl_expr_gpu.mm" PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math;-frounding-math")
    target_compile_options(armigo_pipeline_msl_expr_tests PRIVATE -ffp-contract=off -fno-fast-math)
    target_include_directories(armigo_pipeline_msl_expr_tests PRIVATE "${ARMIGO_SHADER_INCLUDE_DIR}")
    target_link_libraries(armigo_pipeline_msl_expr_tests PRIVATE
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_msl_expr_gpu
        COMMAND armigo_pipeline_msl_expr_tests --gpu)
    set_tests_properties(shader.pipeline_msl_expr_gpu PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 120)
endif()

# ALU decode/emit alignment: synthetic guest words decoded through the lane-1
# rules and dispatched into the integer/float emitter families.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_alu_decode_emit_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/alu_decode_emit_tests.cpp")
    target_link_libraries(armigo_pipeline_alu_decode_emit_tests PRIVATE armigo_shader_pipeline)
    set_target_properties(armigo_pipeline_alu_decode_emit_tests PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(armigo_pipeline_alu_decode_emit_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME shader.pipeline_alu_decode_emit COMMAND armigo_pipeline_alu_decode_emit_tests)
    set_tests_properties(shader.pipeline_alu_decode_emit PROPERTIES LABELS "shader" TIMEOUT 60)
endif()

# IO, memory and flow review regressions through the real decoder.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_memory_review_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/memory_review_tests.cpp")
    target_link_libraries(armigo_pipeline_memory_review_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_memory_review COMMAND armigo_pipeline_memory_review_tests)
    set_tests_properties(shader.pipeline_memory_review PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Vertex input class analysis: code-derived float/uint/sint classification
# replaces the vertex_attribute_format requirement for fact-less packages,
# from legalize through the emitted MSL declarations.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_vertex_class_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/vertex_class_tests.cpp")
    target_link_libraries(armigo_pipeline_vertex_class_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_vertex_class COMMAND armigo_pipeline_vertex_class_tests)
    set_tests_properties(shader.pipeline_vertex_class PROPERTIES LABELS "shader" TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_pipeline_complementary_guard_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/complementary_guard_tests.cpp")
    target_link_libraries(armigo_pipeline_complementary_guard_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_complementary_guards COMMAND armigo_pipeline_complementary_guard_tests)
    set_tests_properties(shader.pipeline_complementary_guards PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# ---------------------------------------------------------------------------
if(BUILD_TESTING)
    add_executable(armigo_pipeline_guarded_transfer_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/guarded_transfer_tests.cpp")
    target_link_libraries(armigo_pipeline_guarded_transfer_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_guarded_transfers COMMAND armigo_pipeline_guarded_transfer_tests)
    set_tests_properties(shader.pipeline_guarded_transfers PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# Bindless texture handle resolution: synthetic programs run the real decode,
# emit, SSA and legalize phases against proven constant-buffer provenance.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_bindless_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/bindless_tests.cpp")
    target_link_libraries(armigo_pipeline_bindless_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_bindless COMMAND armigo_pipeline_bindless_tests)
    set_tests_properties(shader.pipeline_bindless PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

target_sources(armigo_shader_pipeline PRIVATE pipeline/ssa/guard_paths.cpp)

if(BUILD_TESTING)
    add_executable(armigo_pipeline_correlated_guard_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/correlated_guard_tests.cpp")
    target_link_libraries(armigo_pipeline_correlated_guard_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_correlated_guards COMMAND armigo_pipeline_correlated_guard_tests)
    set_tests_properties(shader.pipeline_correlated_guards PROPERTIES LABELS "shader" TIMEOUT 30)
endif()

# ALU review regressions: encoded instructions through the full pipeline.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_alu_review_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/alu_review_tests.cpp")
    target_link_libraries(armigo_pipeline_alu_review_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_alu_review COMMAND armigo_pipeline_alu_review_tests)
    set_tests_properties(shader.pipeline_alu_review PROPERTIES LABELS "shader" TIMEOUT 60)
    if(APPLE AND ARMIGO_XCRUN AND armigo_metal_lookup EQUAL 0)
        add_test(NAME shader.pipeline_alu_review_compile COMMAND armigo_pipeline_alu_review_tests
            "${CMAKE_CURRENT_BINARY_DIR}/pipeline_alu_review_fixtures")
        set_tests_properties(shader.pipeline_alu_review_compile PROPERTIES LABELS "shader" TIMEOUT 120)
    endif()
endif()

# Runtime-sized constant-buffer windows: generated MSL contracts, Metal 4
# compilation, and optional offscreen execution against short and empty backing.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_cbuf_msl_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/cbuf_msl_tests.cpp")
    target_link_libraries(armigo_pipeline_cbuf_msl_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_cbuf_msl COMMAND armigo_pipeline_cbuf_msl_tests)
    set_tests_properties(shader.pipeline_cbuf_msl PROPERTIES LABELS "shader" TIMEOUT 60)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_XCRUN AND armigo_metal_lookup EQUAL 0)
    add_test(NAME shader.pipeline_cbuf_msl_compile
        COMMAND armigo_pipeline_cbuf_msl_tests
            "${CMAKE_CURRENT_BINARY_DIR}/pipeline_cbuf_msl_fixtures" --metal)
    set_tests_properties(shader.pipeline_cbuf_msl_compile PROPERTIES LABELS "shader;metal4" TIMEOUT 120)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)
    target_sources(armigo_pipeline_cbuf_msl_tests PRIVATE
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/cbuf_msl_gpu.mm")
    target_compile_definitions(armigo_pipeline_cbuf_msl_tests PRIVATE ARMIGO_CBUF_MSL_GPU)
    set_source_files_properties("${ARMIGO_SHADER_TEST_DIR}/pipeline/cbuf_msl_gpu.mm" PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math;-frounding-math")
    target_include_directories(armigo_pipeline_cbuf_msl_tests PRIVATE "${ARMIGO_SHADER_INCLUDE_DIR}")
    target_link_libraries(armigo_pipeline_cbuf_msl_tests PRIVATE
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_cbuf_msl_gpu COMMAND armigo_pipeline_cbuf_msl_tests --gpu)
    set_tests_properties(shader.pipeline_cbuf_msl_gpu PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 120)
endif()

# Vertex output initialization policy (armigo-output-init-v1): structured and
# decoded early-exit vertex programs, Metal 4 compilation of the emitted
# initialization, and offscreen execution observing the entry values.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_output_init_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/output_init_tests.cpp")
    target_link_libraries(armigo_pipeline_output_init_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_output_init COMMAND armigo_pipeline_output_init_tests)
    set_tests_properties(shader.pipeline_output_init PROPERTIES LABELS "shader" TIMEOUT 60)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_XCRUN AND armigo_metal_lookup EQUAL 0)
    add_test(NAME shader.pipeline_output_init_compile
        COMMAND armigo_pipeline_output_init_tests
            "${CMAKE_CURRENT_BINARY_DIR}/pipeline_output_init_fixtures" --metal)
    set_tests_properties(shader.pipeline_output_init_compile PROPERTIES LABELS "shader;metal4" TIMEOUT 120)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)
    target_sources(armigo_pipeline_output_init_tests PRIVATE
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/output_init_gpu.mm")
    target_compile_definitions(armigo_pipeline_output_init_tests PRIVATE ARMIGO_OUTPUT_INIT_GPU)
    target_include_directories(armigo_pipeline_output_init_tests PRIVATE "${ARMIGO_SHADER_INCLUDE_DIR}")
    target_link_libraries(armigo_pipeline_output_init_tests PRIVATE
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_output_init_gpu
        COMMAND armigo_pipeline_output_init_tests --gpu)
    set_tests_properties(shader.pipeline_output_init_gpu PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 120)
endif()
# Shader-side vertex fetch against Metal's vertex descriptor fetch, for every
# format the replay maps and every input class.
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)
    add_executable(armigo_pipeline_vertex_fetch_gpu_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/vertex_fetch_gpu.mm")
    target_include_directories(armigo_pipeline_vertex_fetch_gpu_tests PRIVATE "${ARMIGO_SHADER_INCLUDE_DIR}")
    target_link_libraries(armigo_pipeline_vertex_fetch_gpu_tests PRIVATE armigo_shader_pipeline
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_vertex_fetch_gpu COMMAND armigo_pipeline_vertex_fetch_gpu_tests)
    set_tests_properties(shader.pipeline_vertex_fetch_gpu PROPERTIES
        LABELS "gpu;metal4;shader" SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
# FSWZADD quad swizzle add: synthetic fragment programs through the full
# pipeline, the MSL quad-position entry contract, and offscreen Metal 4
# execution against the CPU model of the FSWZADD semantics.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_fswzadd_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/fswzadd_tests.cpp")
    target_link_libraries(armigo_pipeline_fswzadd_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_fswzadd COMMAND armigo_pipeline_fswzadd_tests)
    set_tests_properties(shader.pipeline_fswzadd PROPERTIES LABELS "shader" TIMEOUT 60)
    if(APPLE AND ARMIGO_XCRUN AND armigo_metal_lookup EQUAL 0)
        add_test(NAME shader.pipeline_fswzadd_compile
            COMMAND armigo_pipeline_fswzadd_tests
                "${CMAKE_CURRENT_BINARY_DIR}/pipeline_fswzadd_fixtures" --metal)
        set_tests_properties(shader.pipeline_fswzadd_compile PROPERTIES LABELS "shader;metal4" TIMEOUT 120)
    endif()
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)
    target_sources(armigo_pipeline_fswzadd_tests PRIVATE
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/fswzadd_gpu.mm")
    target_compile_definitions(armigo_pipeline_fswzadd_tests PRIVATE ARMIGO_FSWZADD_GPU)
    set_source_files_properties("${ARMIGO_SHADER_TEST_DIR}/pipeline/fswzadd_gpu.mm" PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math;-frounding-math")
    target_include_directories(armigo_pipeline_fswzadd_tests PRIVATE "${ARMIGO_SHADER_INCLUDE_DIR}")
    target_link_libraries(armigo_pipeline_fswzadd_tests PRIVATE
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_fswzadd_gpu
        COMMAND armigo_pipeline_fswzadd_tests --gpu)
    set_tests_properties(shader.pipeline_fswzadd_gpu PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 120)
endif()

# Compute stages: synthetic GPU dispatch and CLI coverage.
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)
    add_executable(armigo_pipeline_compute_gpu_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/compute_gpu_tests.mm")
    target_compile_options(armigo_pipeline_compute_gpu_tests PRIVATE -fobjc-arc)
    target_include_directories(armigo_pipeline_compute_gpu_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_pipeline_compute_gpu_tests PRIVATE armigo_shader_pipeline
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_compute_gpu COMMAND armigo_pipeline_compute_gpu_tests)
    set_tests_properties(shader.pipeline_compute_gpu PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)
endif()
if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME shader.pipeline_compute_cli
        COMMAND "${Python3_EXECUTABLE}" "${ARMIGO_SHADER_TEST_DIR}/pipeline/compute_cli_tests.py"
            "${CMAKE_BINARY_DIR}/armigo-shader-translate")
    set_tests_properties(shader.pipeline_compute_cli PROPERTIES LABELS "shader" TIMEOUT 60)
endif()

# Per-thread local memory (LDL/STL): allocation declaration, legalization's
# guarded/bounded split, MSL text contracts, and offscreen Metal 4 execution
# of stores, loads and the out-of-range guard against a CPU model.
if(BUILD_TESTING)
    add_executable(armigo_pipeline_local_memory_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/local_memory_tests.cpp")
    target_link_libraries(armigo_pipeline_local_memory_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_local_memory COMMAND armigo_pipeline_local_memory_tests)
    set_tests_properties(shader.pipeline_local_memory PROPERTIES LABELS "shader" TIMEOUT 60)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_XCRUN AND armigo_metal_lookup EQUAL 0)
    add_test(NAME shader.pipeline_local_memory_compile
        COMMAND armigo_pipeline_local_memory_tests
            "${CMAKE_CURRENT_BINARY_DIR}/pipeline_local_memory_fixtures" --metal)
    set_tests_properties(shader.pipeline_local_memory_compile PROPERTIES LABELS "shader;metal4" TIMEOUT 120)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    find_library(ARMIGO_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(ARMIGO_METAL_FRAMEWORK Metal REQUIRED)
    add_executable(armigo_pipeline_local_memory_gpu_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/local_memory_gpu.mm")
    target_compile_options(armigo_pipeline_local_memory_gpu_tests PRIVATE -fobjc-arc)
    target_include_directories(armigo_pipeline_local_memory_gpu_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(armigo_pipeline_local_memory_gpu_tests PRIVATE armigo_shader_pipeline
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_local_memory_gpu COMMAND armigo_pipeline_local_memory_gpu_tests)
    set_tests_properties(shader.pipeline_local_memory_gpu PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 60)
endif()

if(BUILD_TESTING)
    add_executable(armigo_pipeline_global_memory_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/global_memory_tests.cpp")
    target_link_libraries(armigo_pipeline_global_memory_tests PRIVATE armigo_shader_pipeline)
    add_test(NAME shader.pipeline_global_memory COMMAND armigo_pipeline_global_memory_tests)
endif()
if(BUILD_TESTING AND APPLE AND ARMIGO_ENABLE_GPU_TESTS)
    add_executable(armigo_pipeline_global_memory_gpu_tests
        "${ARMIGO_SHADER_TEST_DIR}/pipeline/global_memory_gpu.mm")
    target_compile_options(armigo_pipeline_global_memory_gpu_tests PRIVATE -fobjc-arc)
    target_link_libraries(armigo_pipeline_global_memory_gpu_tests PRIVATE armigo_shader_pipeline
        "${ARMIGO_FOUNDATION_FRAMEWORK}" "${ARMIGO_METAL_FRAMEWORK}")
    add_test(NAME shader.pipeline_global_memory_gpu COMMAND armigo_pipeline_global_memory_gpu_tests)
    set_tests_properties(shader.pipeline_global_memory_gpu PROPERTIES LABELS "gpu;metal4;shader" TIMEOUT 120)
endif()

# Consolidated fingerprint refresh (keep last): every armigo_shader_pipeline
# source joins the translation-chain fingerprint, so the chain changes whenever
# the ported pipeline changes.
get_target_property(armigo_pipeline_sources armigo_shader_pipeline SOURCES)
foreach(pipeline_source IN LISTS armigo_pipeline_sources)
    cmake_path(ABSOLUTE_PATH pipeline_source BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
               NORMALIZE OUTPUT_VARIABLE pipeline_path)
    list(APPEND ARMIGO_SHADER_VERSION_FILES "${pipeline_path}")
endforeach()

list(REMOVE_DUPLICATES ARMIGO_SHADER_VERSION_FILES)
list(SORT ARMIGO_SHADER_VERSION_FILES)
set(armigo_shader_fingerprint "armigo-shader-source-v1\n")
foreach(version_file IN LISTS ARMIGO_SHADER_VERSION_FILES)
    file(RELATIVE_PATH version_label "${ARMIGO_SHADER_REPOSITORY_ROOT}" "${version_file}")
    file(SHA256 "${version_file}" version_digest)
    string(APPEND armigo_shader_fingerprint "${version_label}:${version_digest}\n")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ARMIGO_SHADER_VERSION_FILES})
string(SHA256 ARMIGO_SHADER_SOURCE_ID "${armigo_shader_fingerprint}")
configure_file(source_version.h.in generated/armigo/shader/source_version.h @ONLY)

