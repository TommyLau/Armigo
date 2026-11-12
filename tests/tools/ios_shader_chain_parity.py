#!/usr/bin/env python3
# ============================================================================
# Armigo Tests - Native Runtime Verification
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Compare a C++ package with the player's profile using the same tool fixture."""
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: ios_shader_chain_parity.py TRANSLATOR MANIFEST")
    script = ROOT / "tools/ios/shader_chain.py"
    spec = importlib.util.spec_from_file_location("armigo_ios_chain", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads(Path(sys.argv[2]).read_text())
    profile = module.current_profile(Path(sys.argv[1]))
    profile_text = module.chain.canonical_json(profile)
    assert json.loads(manifest["profile_text"]) == profile, "C++ iOS package and player compiler profiles differ"
    assert manifest["profile_text"] == profile_text, "iOS profile canonicalization differs"
    expected = module.chain.chain_id(profile_text)
    assert manifest["chain_id"] == expected, "C++ iOS package and player chain IDs differ"
    header = Path(sys.argv[2]).parent.parent / "expected_metal_chain.h"
    module.chain.emit_expected_header(profile_text, header)
    match = re.search(r'expected_metal_chain_id\[\] = "([0-9a-f]{64})"', header.read_text())
    assert match and match[1] == manifest["chain_id"], "Generated player header rejects the package"
    print("C++ iOS package profile and generated player chain match")


if __name__ == "__main__":
    main()
