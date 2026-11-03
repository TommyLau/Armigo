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

"""Verify headless utility modes build without optional Metal compiler tools."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class OptionalMetalToolsBuildTests(unittest.TestCase):
    def run_command(self, arguments, environment, timeout=300):
        result = subprocess.run(arguments, cwd=ROOT, env=environment, check=False,
                                capture_output=True, text=True, timeout=timeout)
        self.assertEqual(
            result.returncode, 0,
            f"command failed: {' '.join(str(value) for value in arguments)}\n"
            f"stdout:\n{result.stdout[-8000:]}\nstderr:\n{result.stderr[-8000:]}")
        return result

    def test_headless_utility_modes_build_without_metal_compilers(self):
        cmake = shutil.which("cmake")
        ninja = shutil.which("ninja")
        real_xcrun = shutil.which("xcrun")
        self.assertIsNotNone(cmake)
        self.assertIsNotNone(ninja)
        self.assertIsNotNone(real_xcrun)

        with tempfile.TemporaryDirectory(prefix="armigo-optional-metal-") as temporary:
            temporary_root = Path(temporary)
            fake_bin = temporary_root / "bin"
            build = temporary_root / "build"
            fake_bin.mkdir()
            log = temporary_root / "xcrun.log"
            wrapper = fake_bin / "xcrun"
            wrapper.write_text(
                "#!/usr/bin/env python3\n"
                "import os\n"
                "from pathlib import Path\n"
                "import sys\n"
                "arguments = sys.argv[1:]\n"
                "with Path(os.environ['ARMIGO_XCRUN_LOG']).open('a') as stream:\n"
                "    stream.write(' '.join(arguments) + '\\n')\n"
                "lookup = ('--find' in arguments or '-f' in arguments)\n"
                "if lookup and arguments[-1] in ('metal', 'metallib'):\n"
                "    raise SystemExit(72)\n"
                "if 'macosx' in arguments:\n"
                "    index = arguments.index('macosx') + 1\n"
                "    if index < len(arguments) and arguments[index] in ('metal', 'metallib'):\n"
                "        raise SystemExit(72)\n"
                "os.execv(os.environ['ARMIGO_REAL_XCRUN'],\n"
                "         [os.environ['ARMIGO_REAL_XCRUN'], *arguments])\n",
                encoding="utf-8")
            wrapper.chmod(0o755)

            environment = os.environ.copy()
            environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
            environment["ARMIGO_REAL_XCRUN"] = real_xcrun
            environment["ARMIGO_XCRUN_LOG"] = str(log)
            environment["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"

            for tool in ("metal", "metallib"):
                missing = subprocess.run(
                    [str(wrapper), "--sdk", "macosx", "--find", tool],
                    env=environment, check=False, capture_output=True, text=True)
                self.assertEqual(missing.returncode, 72)

            configured = self.run_command([
                cmake, "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
                f"-DCMAKE_MAKE_PROGRAM={ninja}",
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DCMAKE_OSX_ARCHITECTURES=arm64",
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0",
                "-DBUILD_TESTING=OFF",
            ], environment)
            self.assertIn("Metal compiler tools not found", configured.stdout)
            self.run_command(
                [cmake, "--build", str(build), "--target", "armigo-headless"],
                environment, timeout=540)

            expected_header = (
                build / "generated" / "armigo" / "shader" / "expected_metal_chain.h")
            self.assertFalse(expected_header.exists())
            help_result = self.run_command(
                [str(build / "armigo-headless"), "--help"], environment)
            self.assertIn("--probe", help_result.stdout)
            self.assertIn("--self-test", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
