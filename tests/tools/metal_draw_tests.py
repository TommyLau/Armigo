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

"""Compile synthetic fixtures offline before executing the Metal draw backend."""

from pathlib import Path
import subprocess
import sys
import tempfile


def run(arguments, timeout=20):
    subprocess.run(arguments, check=True, timeout=timeout)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: metal_draw_tests.py TEST_EXECUTABLE")
    executable = str(Path(sys.argv[1]).resolve(strict=True))
    with tempfile.TemporaryDirectory(prefix="armigo-metal-draw-") as temporary:
        directory = Path(temporary)
        run([executable, "--emit-fixtures", str(directory)])
        for stage in ("vertex", "fragment", "depth_fragment"):
            source = directory / f"{stage}.metal"
            air = directory / f"{stage}.air"
            library = directory / f"{stage}.metallib"
            run(["xcrun", "--sdk", "macosx", "metal", "-std=metal4.0",
                 "-mmacosx-version-min=26.0", "-Werror", "-c", str(source),
                 "-o", str(air)])
            run(["xcrun", "--sdk", "macosx", "metallib", str(air), "-o", str(library)])
        run([executable, str(directory)], timeout=60)


if __name__ == "__main__":
    main()
