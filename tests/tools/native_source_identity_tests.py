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

"""Native-only changes invalidate real same-input stage-cache entries."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeSourceIdentityTests(unittest.TestCase):
    def command(self, args):
        result = subprocess.run([str(x) for x in args], text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0,
                         f"command failed: {args}\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
        return result.stdout.strip()

    def test_native_source_and_header_invalidate_same_stage(self):
        with tempfile.TemporaryDirectory(prefix="armigo-native-identity-") as temporary:
            root = Path(temporary)
            source = root / "source"
            shutil.copytree(ROOT, source, ignore=shutil.ignore_patterns(
                ".git", "build", "__pycache__", ".DS_Store", ".orca", ".cache"))
            build = root / "build"
            cache = root / "stage-cache"
            probe = root / "cache-probe"
            cmake = shutil.which("cmake")
            compiler = shutil.which("c++")
            self.assertIsNotNone(cmake)
            self.assertIsNotNone(compiler)

            def configure_and_compile():
                self.command([cmake, "-S", source, "-B", build, "-G", "Ninja",
                              "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
                              "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0"])
                self.command([compiler, "-std=c++20", "-I" + str(source / "include"),
                              "-I" + str(source / "src/loader"),
                              "-I" + str(build / "src/shader/generated"),
                              source / "tests/shader/translation/stage_cache_identity_probe.cpp",
                              source / "src/shader/translation/stage_cache.cpp",
                              source / "src/loader/sha256_commoncrypto.cpp", "-o", probe])

            configure_and_compile()
            first = self.command([probe, cache, "miss"])
            self.assertEqual(self.command([probe, cache, "hit"]), first)
            versions = {first.split()[0]}
            for relative in ("src/shader_native/system.cpp", "src/shader_native/builder.h"):
                with (source / relative).open("a") as stream:
                    stream.write("\n// Synthetic production-input fingerprint change.\n")
                configure_and_compile()
                changed = self.command([probe, cache, "miss"])
                self.assertEqual(changed.split()[1], first.split()[1], "input and options key stays identical")
                self.assertNotIn(changed.split()[0], versions)
                versions.add(changed.split()[0])
                self.assertEqual(self.command([probe, cache, "hit"]), changed)
            # Unlisted work must not accidentally enter the declared production identity.
            (source / "src/shader_native/unlisted.cpp").write_text(
                "// Copyright (c) 2026 Tommy Lau.\n"
                "// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary\n"
                "// Unlisted synthetic file.\n")
            configure_and_compile()
            self.assertEqual(self.command([probe, cache, "hit"]), changed)


if __name__ == "__main__":
    unittest.main()
