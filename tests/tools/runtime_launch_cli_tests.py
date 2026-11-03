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

"""Exercise the preserved runner shell command-line paths.

The windowed application is only exercised through its help and argument
validation paths so a real window is never created in tests.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HEADLESS = str(Path(sys.argv.pop(1)).resolve())
WINDOWED = str(Path(sys.argv.pop(1)).resolve())


def run_tool(executable, *arguments):
    return subprocess.run([executable, *arguments], capture_output=True, text=True,
                          timeout=120)


class HeadlessPreservedPaths(unittest.TestCase):

    def test_help_is_preserved(self):
        result = run_tool(HEADLESS, "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--probe | --self-test | --help", result.stdout)
        self.assertIn("armigo-headless run --exefs DIR --romfs DIR [--shaders DIR]",
                      result.stdout)

        run_help = run_tool(HEADLESS, "run", "--help")
        self.assertEqual(run_help.returncode, 0)
        self.assertIn("--operation-mode MODE", run_help.stdout)

    def test_missing_arguments_still_report_usage(self):
        result = run_tool(HEADLESS)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Expected --probe, --self-test, --help, run, or pso-archive.", result.stderr)

    def test_unknown_flag_still_reports_usage(self):
        result = run_tool(HEADLESS, "--bogus")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Expected --probe, --self-test, --help, run, or pso-archive.", result.stderr)

    def test_probe_keeps_its_schema(self):
        result = run_tool(HEADLESS, "--probe")
        self.assertIn(result.returncode, (0, 1))
        report = json.loads(result.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["mode"], "probe")
        self.assertFalse(report["guest_execution_implemented"])
        for key in ("device", "metal4_supported", "self_test_ran", "self_test_passed",
                    "verified_buffer_bytes", "verified_pixels", "error"):
            self.assertIn(key, report)


class HeadlessRunMode(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        for name in ("exefs", "romfs", "shaders"):
            (root / name).mkdir()

    def path(self, name):
        return str(Path(self.temporary.name) / name)

    def test_validation_errors_are_usage_errors(self):
        result = run_tool(HEADLESS, "run", "--exefs", self.path("missing"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("--exefs", result.stderr)

        result = run_tool(HEADLESS, "run", "--exefs", self.path("exefs"),
                          "--romfs", self.path("romfs"), "--frames", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--frames", result.stderr)

        result = run_tool(HEADLESS, "run", "--operation-mode", "portable")
        self.assertEqual(result.returncode, 2)
        self.assertIn("console or handheld", result.stderr)

    def test_run_reports_the_structured_unsupported_result(self):
        report_path = self.path("report.json")
        result = run_tool(HEADLESS, "run", "--exefs", self.path("exefs"),
                          "--romfs", self.path("romfs"), "--shaders",
                          self.path("shaders"), "--frames", "2",
                          "--operation-mode", "handheld",
                          "--report", report_path)
        self.assertIn(result.returncode, (3, 4))
        self.assertLess(result.returncode, 5)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["mode"], "run-headless")
        self.assertEqual(report["outcome"], "unsupported" if result.returncode == 3
                         else "failure")
        self.assertEqual(report["exit_code"], result.returncode)
        self.assertEqual(report["metadata_signature"], "not_verified")
        self.assertFalse(report["quiescent"])
        self.assertIn("failure", report)
        if result.returncode == 3:
            self.assertIn("not available in this build", report["failure"]["detail"])
            self.assertEqual(report["failure"]["code"], "unsupported")
            written = json.loads(Path(report_path).read_text())
            self.assertEqual(written["outcome"], "unsupported")


class WindowedCommandLine(unittest.TestCase):

    def test_help_describes_the_run_mode(self):
        result = run_tool(WINDOWED, "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("armigo run --exefs DIR --romfs DIR [--shaders DIR]", result.stdout)
        self.assertIn("Exit codes:", result.stdout)

        run_help = run_tool(WINDOWED, "run", "--help")
        self.assertEqual(run_help.returncode, 0)
        self.assertIn("--operation-mode MODE", run_help.stdout)

    def test_missing_subcommand_is_a_usage_error(self):
        result = run_tool(WINDOWED)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Expected --help, run, or pso-archive.", result.stderr)

    def test_argument_validation_precedes_any_window(self):
        result = run_tool(WINDOWED, "run", "--operation-mode", "portable")
        self.assertEqual(result.returncode, 2)
        self.assertIn("console or handheld", result.stderr)

        result = run_tool(WINDOWED, "run", "--exefs", "missing", "--romfs", "missing",
                          "--shaders", "missing")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--exefs", result.stderr)


if __name__ == "__main__":
    unittest.main()
