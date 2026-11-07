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

"""Check the generated Info.plist of the windowed player bundle."""

import plistlib
import sys


def main():
    with open(sys.argv[1], "rb") as source:
        info = plistlib.load(source)
    failures = []
    # macOS classifies the app as a game only from this exact category, and
    # Game Mode applies to games that also declare their support.
    category = info.get("LSApplicationCategoryType")
    if category != "public.app-category.games":
        failures.append(f"LSApplicationCategoryType is {category!r}, expected 'public.app-category.games'")
    if info.get("LSSupportsGameMode") is not True:
        failures.append("LSSupportsGameMode is not true")
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
