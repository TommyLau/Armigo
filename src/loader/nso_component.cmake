# ============================================================================
# Armigo Build - Native Build and Automation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

# This leaf owns the NSO/LZ4/SHA production sources and compilation policy.
# The shader fingerprint reads the manifest before loader targets are created.
# Keep container formats, optional codecs, tools and tests in the parent file.
function(armigo_nso_production_files output)
    cmake_path(SET nso_root NORMALIZE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    cmake_path(SET nso_repository NORMALIZE "${nso_root}/../..")
    set(${output}
        "${nso_root}/nso_module.cpp"
        "${nso_repository}/include/armigo/loader/nso_module.h"
        "${nso_root}/lz4_block.cpp"
        "${nso_repository}/include/armigo/loader/lz4_block.h"
        "${nso_root}/sha256_commoncrypto.cpp"
        "${nso_root}/sha256.h"
        PARENT_SCOPE)
endfunction()

function(armigo_add_nso_component)
    armigo_nso_production_files(nso_sources)
    cmake_path(SET nso_root NORMALIZE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    cmake_path(SET nso_repository NORMALIZE "${nso_root}/../..")
    add_library(armigo_loader_nso OBJECT ${nso_sources})
    target_include_directories(armigo_loader_nso PRIVATE
        "${nso_repository}/include" "${nso_root}")
    target_compile_features(armigo_loader_nso PRIVATE cxx_std_20)
endfunction()
