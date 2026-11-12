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

    def check(policy, expected, *arguments, clock=60, fallback=0, options="0,30,60,",
              language=None, language_fallback=0, supported="en-US,ja,fr,"):
        metadata = dict(base)
        if policy is not None:
            metadata["ArmigoTitleFrameRate"] = policy
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(metadata, stream)
        result = subprocess.run([str(executable), *arguments], env=environment,
                                text=True, capture_output=True, check=True)
        expected += f"\nclock={clock} fallback={fallback} options={options}"
        if language is not None:
            expected += f"\nlanguage={language} fallback={language_fallback} supported={supported}"
        assert result.stdout.strip() == expected, (result.stdout, result.stderr, expected)

    check(None, "60 0 0 unset 60")
    policy = {"Default": 30, "DisplayRefresh": 60}
    check(policy, "30 1 60 30 60", "-expect_default", "30")
    check({"Default": 60, "DisplayRefresh": 60}, "60 1 60 60 60", "-expect_default", "60")
    check(policy, "60 1 60 60 60", "-frame_rate", "60")
    check(policy, "0 1 60 0 60", "-frame_rate", "0")
    check(policy, "60 1 60 60 60", "-frame_rate", "120", fallback=1)
    high = dict(policy, SupportedRates=[30, 60, 120])
    check(high, "60 1 60 60 60", "-frame_rate", "120", fallback=1)
    check(high, "120 1 120 120 120", "-frame_rate", "120", "-display_hz", "120",
          clock=120, options="0,30,60,120,")
    check(high, "0 1 60 0 60", "-frame_rate", "0", "-display_hz", "120", options="0,30,60,120,")
    botw = {"Default": 60, "DisplayRefresh": 60, "SupportedRates": [30, 60, 120], "ExperimentalRates": [120]}
    for hz in (60, 90, 119.88):
        check(botw, "60 1 60 60 60", "-frame_rate", "120", "-display_hz", str(hz),
              "-expect_experimental", "120", fallback=1)
    for hz in (120, 144, 240):
        check(botw, "120 1 120 120 120", "-frame_rate", "120", "-display_hz", str(hz),
              "-expect_experimental", "120", clock=120, options="0,30,60,120,")
        check(botw, "0 1 60 0 60", "-frame_rate", "0", "-display_hz", str(hz), options="0,30,60,120,")
    check(dict(policy, SupportedRates=[30, 120]), "30 1 60 30 60", "-frame_rate", "120",
          fallback=1, options="0,30,")
    check(None, "60 0 0 unset 60", "-frame_rate", "120", "-display_hz", "120", fallback=1)
    check({"Default": "30", "DisplayRefresh": 60}, "60 0 0 unset 60")

    language_args = ("-language_key", "game_language", "-languages", "en-US,ja,fr")
    check(None, "60 0 0 unset 60", *language_args, language="auto")
    check(None, "60 0 0 unset 60", "-write_language", "game_language", "fr",
          *language_args, language="fr")
    check(None, "60 0 0 unset 60", *language_args, language="fr")
    check(None, "60 0 0 unset 60", "-write_language", "game_language", "auto",
          *language_args, language="auto")
    check(None, "60 0 0 unset 60", "-write_language", "game_language", "es",
          *language_args, language="auto", language_fallback=1)
    check(None, "60 0 0 unset 60", *language_args, language="auto", language_fallback=1)
    check(None, "60 0 0 unset 60", "-write_language", "game_language", "auto",
          "-language_key", "game_language", "-languages", "en-US,not-a-language,fr,en-US",
          language="auto", supported="en-US,fr,")

    first_title = "game_language.0200000000000A42"
    second_title = "game_language.0200000000000A43"
    check(None, "60 0 0 unset 60", "-write_language", first_title, "fr",
          "-write_language", second_title, "ja", "-language_key", first_title,
          "-languages", "en-US,ja,fr", "-expect_language", second_title, "ja", language="fr")
    check(None, "60 0 0 unset 60", "-language_key", second_title, "-languages", "en-US,ja,fr",
          "-expect_language", first_title, "fr", language="ja")

    migration_args = (
        "-write_language", "game_language", "fr", "-write_frame_rate", "frame_rate", "120",
        "-language", "en-US", "-region", "1", "-fps_overlay", "1", "-display_hz", "120",
        *language_args,
    )
    check(high, "120 1 120 120 120", *migration_args, clock=120, options="0,30,60,120,", language="fr")
    check(high, "120 1 120 120 120", "-display_hz", "120", *language_args,
          clock=120, options="0,30,60,120,", language="fr")
    check(high, "120 1 120 120 120", "-write_language", "game_language", "es",
          "-display_hz", "120", *language_args, clock=120, options="0,30,60,120,",
          language="auto", language_fallback=1)

    environment["ARMIGO_COMPAT_FRAME_RATE"] = "30"
    check(policy, "60 1 60 30 60", "-frame_rate", "60")
    environment["ARMIGO_VI_REFRESH_HZ"] = "30"
    check(policy, "60 1 60 30 30", "-frame_rate", "60")

print("Player display and game-language preference checks passed.")
