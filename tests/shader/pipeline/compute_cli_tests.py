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
# All fixture words and expected outcomes are independently authored.

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def main():
    translator = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="armigo-compute-input-") as inputs, \
            tempfile.TemporaryDirectory(prefix="armigo-compute-output-") as outputs:
        inputs, outputs = Path(inputs), Path(outputs)
        control = bytearray(0x718)
        for offset, value in [(0, 0x98761234), (4, 1), (8, 14), (0x20, len(control)),
                              (0x700, 16), (0x704, 16), (0x708, 0), (0x714, 5)]:
            struct.pack_into("<I", control, offset, value)
        (inputs / "control.bin").write_bytes(control)
        (inputs / "code.bin").write_bytes(struct.pack("<QQ", 0, 0xE30000000007000F))
        common = [translator, "--translator", "ported", "--control", str(inputs / "control.bin"),
                  "--code", str(inputs / "code.bin")]
        launch = ["--compute-local-size", "4,4,2", "--compute-shared-memory", "0"]
        cases = [
            ("unknown-local", [], "No --compute-local-size"),
            ("unknown-shared", launch[:2], "No --compute-shared-memory"),
            ("zero-dimension", ["--compute-local-size", "0,4,2"] + launch[2:], "Compute launch exceeds"),
            ("oversized-launch", ["--compute-local-size", "32,32,2"] + launch[2:], "Compute launch exceeds"),
            ("unaligned-shared", launch[:2] + ["--compute-shared-memory", "2"], "Compute launch exceeds"),
            ("duplicate-zero", launch + launch[2:], "Duplicate --compute-shared-memory"),
            ("graphics-option", launch + ["--vertex-depth", "zero-to-one"], "Graphics options"),
            ("explicit", launch, None),
        ]
        for name, options, expected in cases:
            destination = outputs / name
            result = subprocess.run(common + ["--output", str(destination)] + options,
                                    capture_output=True, text=True, timeout=30)
            if expected:
                assert result.returncode != 0 and expected in result.stderr, (name, result.stderr)
                assert not destination.exists(), name
            else:
                assert result.returncode == 0, result.stderr
                reflection = json.loads((destination / "reflection.json").read_text())
                assert reflection["schema"] == "armigo-shader-stage-v3"
                assert reflection["profile"] == {"identity": "armigo-macos-metal4", "version": 3}
                assert reflection["argument_buffer"] is None
                assert reflection["stage"] == "compute"
                assert reflection["compute"] == {"local_size": [4, 4, 2], "shared_memory_bytes": 0}
                facts = {item["kind"]: item for item in reflection["dependencies"]}
                assert facts["compute_local_size"]["value"] == [4, 4, 2]
                assert facts["compute_shared_memory_bytes"]["value"] == 0
                assert facts["compute_local_size"]["origin"] == "explicit_option"
                assert "kernel void armigo_compute_main" in (destination / "source.metal").read_text()
        print("Compute CLI launch facts and v2 reflection passed.")


if __name__ == "__main__":
    main()
