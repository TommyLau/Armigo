# ============================================================================
# Armigo Tests - Native Runtime Verification
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Read real bundle metadata and preference overrides in a synthetic player."""

import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import uuid


with tempfile.TemporaryDirectory(prefix="armigo-player-choices-") as directory:
    contents = Path(directory) / "Choices.app" / "Contents"
    executable = contents / "MacOS" / "Choices"
    executable.parent.mkdir(parents=True)
    shutil.copy2(sys.argv[1], executable)
    base = {
        "CFBundleExecutable": "Choices",
        "CFBundleIdentifier": "com.armigo.tests.choices." + uuid.uuid4().hex,
        "CFBundlePackageType": "APPL",
    }
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ARMIGO_")}

    def check(policy, expected, *arguments):
        metadata = dict(base)
        if policy is not None:
            metadata["ArmigoTitleFrameRate"] = policy
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(metadata, stream)
        result = subprocess.run([str(executable), *arguments], env=environment,
                                text=True, capture_output=True, check=True)
        assert result.stdout.strip() == expected, (result.stdout, result.stderr, expected)

    check(None, "60 0 0 unset 60")
    policy = {"Default": 30, "DisplayRefresh": 60}
    check(policy, "30 1 60 30 60")
    check(policy, "60 1 60 60 60", "-frame_rate", "60")
    check(policy, "0 1 60 0 60", "-frame_rate", "0")
    check(policy, "60 1 60 60 60", "-frame_rate", "120")
    check({"Default": "30", "DisplayRefresh": 60}, "60 0 0 unset 60")
    environment["ARMIGO_COMPAT_FRAME_RATE"] = "30"
    check(policy, "60 1 60 30 60", "-frame_rate", "60")
    environment["ARMIGO_VI_REFRESH_HZ"] = "30"
    check(policy, "60 1 60 30 30", "-frame_rate", "60")

print("Player title frame-rate preference checks passed.")
