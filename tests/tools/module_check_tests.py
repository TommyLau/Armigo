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

"""Exercise the module-check command with independently authored containers."""

import hashlib
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = str(Path(sys.argv.pop(1)).resolve())


def fixture():
    text, ro, data = (bytearray(4096) for _ in range(3))
    # A deliberately trapping initializer proves validation does not run code.
    struct.pack_into("<I", text, 4, 0x4000)
    struct.pack_into("<I", text, 0x100, 0xD4200000)
    struct.pack_into("<7I", ro, 0, 0x30444F4D, 0x40, 0x5000, 0x5000,
                     0xF00, 0xF00, 0x4100)
    tags = [(4, 0x4180), (5, 0x4200), (10, 1), (6, 0x4300),
            (11, 24), (12, 0x100), (0, 0)]
    for index, tag in enumerate(tags):
        struct.pack_into("<QQ", ro, 0x40 + index * 16, *tag)
    struct.pack_into("<II", ro, 0x180, 1, 1)
    result = bytearray(256)
    result[:4] = b"NSO0"
    struct.pack_into("<I", result, 12, 0x38)
    for index, segment in enumerate((text, ro, data)):
        struct.pack_into("<III", result, 0x10 + index * 16,
                         len(result), index * 0x4000, len(segment))
        struct.pack_into("<I", result, 0x60 + index * 4, len(segment))
        result[0xA0 + index * 32:0xC0 + index * 32] = hashlib.sha256(segment).digest()
        result.extend(segment)
    return result


class ModuleCheckTests(unittest.TestCase):
    def run_tool(self, *arguments, cwd=None):
        return subprocess.run([EXECUTABLE, *map(str, arguments)], cwd=cwd,
                              capture_output=True, text=True, timeout=20)

    def test_argument_contract(self):
        self.assertEqual(self.run_tool("--help").returncode, 0)
        for args in ((), ("--",), ("--execute",), ("--help", "extra")):
            self.assertEqual(self.run_tool(*args).returncode, 2)

    def test_loads_without_execution_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "-synthetic-input"
            content = fixture()
            path.write_bytes(content)
            result = self.run_tool("--", path.name, path.name, cwd=directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated 2 modules", result.stdout)
            self.assertIn("no module code executed", result.stdout)
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(list(Path(directory).iterdir()), [path])
            self.assertNotIn(directory, result.stdout + result.stderr)

    def test_input_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-input"
            missing = self.run_tool(path)
            self.assertEqual(missing.returncode, 1)
            self.assertIn("Input 1", missing.stderr)
            path.write_bytes(b"invalid")
            malformed = self.run_tool(path)
            self.assertEqual(malformed.returncode, 1)
            self.assertIn("at input 1", malformed.stderr)
            content = fixture()
            content[-1] ^= 1
            path.write_bytes(content)
            corrupt = self.run_tool(path)
            self.assertEqual(corrupt.returncode, 1)
            self.assertNotIn(directory, missing.stderr + malformed.stderr + corrupt.stderr)

    def test_nonregular_inputs_and_resource_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = Path(directory) / "pipe"
            os.mkfifo(pipe)
            for path in (pipe, directory):
                result = self.run_tool(path)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("regular file", result.stderr)
            oversized = Path(directory) / "oversized"
            with oversized.open("wb") as stream:
                stream.truncate(512 * 1024 * 1024 + 1)
            result = self.run_tool(oversized)
            self.assertEqual(result.returncode, 1)
            self.assertIn("input limit", result.stderr)
            self.assertEqual(self.run_tool(*([oversized] * 65)).returncode, 2)


if __name__ == "__main__":
    unittest.main()
