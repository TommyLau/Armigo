#!/usr/bin/env python3
# ============================================================================
# Armigo Tools - Development and Verification Tools
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Trusted iOS Metal 4 package identity, using the installed host translator."""

import argparse
import importlib.util
from pathlib import Path

_chain_path = Path(__file__).resolve().parents[2] / "src/shader/tools/chain.py"
_spec = importlib.util.spec_from_file_location("armigo_shader_chain", _chain_path)
chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chain)


def current_profile(translate_tool):
    profile = chain.current_profile(translate_tool)
    profile.update({
        "platform": "ios-metal4",
        "compile_options": ["-std=metal4.0", "-mios-version-min=26.0", "-Werror"],
        "metal_binary_sha256": chain._sha256_file(chain._run(["xcrun", "--sdk", "iphoneos", "--find", "metal"])),
        "metallib_binary_sha256": chain._sha256_file(chain._run(["xcrun", "--sdk", "iphoneos", "--find", "metallib"])),
        "metal_version": chain._first_line(chain._run(["xcrun", "--sdk", "iphoneos", "metal", "--version"]), "metal"),
        "sdk_build": chain._run(["xcrun", "--sdk", "iphoneos", "--show-sdk-build-version"]),
        "sdk_version": chain._run(["xcrun", "--sdk", "iphoneos", "--show-sdk-version"]),
    })
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translate-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    chain.emit_expected_header(chain.canonical_json(current_profile(arguments.translate_tool)), arguments.output)
