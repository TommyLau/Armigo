#!/usr/bin/env python3
# ============================================================================
# Armigo Tools - Development and Verification Tools
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Compare actual configured player features across the two Apple platforms."""

import argparse
import copy
import json
from pathlib import Path
import subprocess


PLATFORMS = {
    "macos": {"window": "AppKit", "audio_device": "HAL", "code_loading": "native_or_embedded"},
    "ios": {"window": "UIKit", "audio_device": "RemoteIO", "code_loading": "embedded"},
}


def compare(macos, ios):
    for name, manifest in (("macos", macos), ("ios", ios)):
        if set(manifest) != {"core", "dependencies", "platform"}:
            raise ValueError(f"Unexpected {name} feature categories")
        if manifest["platform"] != PLATFORMS[name]:
            raise ValueError(f"Undocumented {name} platform features: {manifest['platform']}")
        for feature in ("hwopus", "zbic", "native_shader"):
            if manifest["core"].get(feature) is not True:
                raise ValueError(f"Required {name} feature disabled: {feature}")
    for category in ("core", "dependencies"):
        if macos[category] != ios[category]:
            left = json.dumps(macos[category], sort_keys=True, indent=2).splitlines()
            right = json.dumps(ios[category], sort_keys=True, indent=2).splitlines()
            import difflib
            difference = "\n".join(difflib.unified_diff(left, right, fromfile="macos", tofile="ios"))
            raise ValueError(f"Player {category} parity failed:\n{difference}")


def check_rejections(macos, ios):
    mutations = [
        lambda data: data["core"].update(hwopus=False),
        lambda data: data["core"].update(horizon_services=[]),
        lambda data: data["dependencies"]["zstd"].update(version="different"),
        lambda data: data["platform"].update(extra_service="unsupported"),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(ios)
        mutate(changed)
        try:
            compare(macos, changed)
        except ValueError:
            continue
        raise ValueError("The parity check accepted a deliberately incompatible configuration")


def check_archive_rejections(args):
    fixture = Path(args.build) / "archive-check"
    fixture.mkdir(parents=True, exist_ok=True)
    module = Path(args.source) / "cmake/RuntimeDependencies.cmake"
    (fixture / "CMakeLists.txt").write_text(
        'cmake_minimum_required(VERSION 3.25)\nproject(ArchiveCheck LANGUAGES CXX)\n'
        f'include("{module}")\n')
    bad = fixture / "wrong.tar.gz"
    bad.write_bytes(b"independently authored invalid dependency archive")
    for archive, expected in ((bad, "SHA-256 does not match"),
                              (fixture / "missing.tar.gz", "archive is missing")):
        result = subprocess.run(
            ["cmake", "-S", str(fixture), "-B", str(fixture / "build"),
             f"-DARMIGO_DEPENDENCY_CACHE={args.dependency_cache}",
             f"-DARMIGO_OPUS_ARCHIVE={archive}"], capture_output=True, text=True, check=False)
        if not result.returncode or expected not in result.stdout + result.stderr:
            raise ValueError("The dependency configure accepted an absent or modified archive")


def main():
    parser = argparse.ArgumentParser()
    for argument in ("source", "build", "translator", "dependency-cache", "opus-archive", "zstd-archive"):
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    manifests = {}
    for platform in PLATFORMS:
        output = Path(args.build) / platform
        command = ["cmake", "-S", args.source, "-B", str(output), "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF", "-DARMIGO_ENABLE_GPU_TESTS=OFF",
                   "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0",
                   f"-DARMIGO_DEPENDENCY_CACHE={args.dependency_cache}",
                   f"-DARMIGO_OPUS_ARCHIVE={args.opus_archive}", f"-DARMIGO_ZSTD_ARCHIVE={args.zstd_archive}"]
        if platform == "ios":
            command += ["-DCMAKE_SYSTEM_NAME=iOS", "-DCMAKE_OSX_SYSROOT=iphoneos",
                        f"-DARMIGO_HOST_SHADER_TRANSLATE={args.translator}"]
        else:
            command += ["-DCMAKE_OSX_SYSROOT=macosx"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(f"{platform} configure failed:\n{result.stdout}\n{result.stderr}")
        manifests[platform] = json.loads((output / "armigo-player-features.json").read_text())
    compare(manifests["macos"], manifests["ios"])
    check_rejections(manifests["macos"], manifests["ios"])
    check_archive_rejections(args)
    print("macOS and iOS player services, decoders, dependency versions and compiler options match.")


if __name__ == "__main__":
    main()
