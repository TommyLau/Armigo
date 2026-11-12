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

"""Deterministic shader-chain fingerprints for trusted offline builds."""

import hashlib
import json
from pathlib import Path
import subprocess


TIMEOUT_SECONDS = 60
HEX_LENGTH = 64


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _run(arguments):
    try:
        result = subprocess.run(arguments, check=False, capture_output=True, text=True,
                                timeout=TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot query shader tool: {arguments[0]}") from error
    if result.returncode != 0:
        detail = result.stderr.strip()[:4096]
        raise RuntimeError(f"Shader tool query failed: {' '.join(arguments)}: {detail}")
    return result.stdout.strip()


def _first_line(text, label):
    lines = text.splitlines()
    if not lines or not lines[0].strip():
        raise RuntimeError(f"{label} returned no version")
    return lines[0].strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise RuntimeError("Cannot hash Apple shader tool") from error
    return digest.hexdigest()


def current_profile(translate_tool):
    source_version = _first_line(_run([str(translate_tool), "--version"]),
                                 "translator")
    if len(source_version) != HEX_LENGTH or any(c not in "0123456789abcdef" for c in source_version):
        raise RuntimeError("Translator source version must be 64 lowercase hexadecimal digits")
    metal_path = _first_line(_run(["xcrun", "--sdk", "macosx", "--find", "metal"]),
                             "xcrun --find metal")
    metallib_path = _first_line(_run(["xcrun", "--sdk", "macosx", "--find", "metallib"]),
                                "xcrun --find metallib")
    return {
        "compile_options": ["-std=metal4.0", "-mmacosx-version-min=26.0", "-fwrapv", "-Werror"],
        "deployment_target": "26.0",
        "identity": "span-prefix112-v1",
        "metal_binary_sha256": _sha256_file(metal_path),
        "metal_language": "metal4.0",
        "metal_version": _first_line(
            _run(["xcrun", "--sdk", "macosx", "metal", "--version"]), "metal"),
        "metallib_binary_sha256": _sha256_file(metallib_path),
        "platform": "macos-metal4",
        "sdk_build": _first_line(
            _run(["xcrun", "--sdk", "macosx", "--show-sdk-build-version"]), "macOS SDK"),
        "sdk_version": _first_line(
            _run(["xcrun", "--sdk", "macosx", "--show-sdk-version"]), "macOS SDK"),
        "source_version": source_version,
        "translation_profile": "bounded-float3-clip-depth-positionw-v1",
    }


def chain_id(profile_text):
    return hashlib.sha256(b"armigo-metal-chain-v1\n" + profile_text.encode("utf-8")).hexdigest()


def emit_expected_header(profile_text, path):
    try:
        profile = json.loads(profile_text)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Expected-chain profile is invalid JSON") from error
    if canonical_json(profile) != profile_text:
        raise RuntimeError("Expected-chain profile is not canonical JSON")
    value = chain_id(profile_text)
    content = f"""// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// Generated from the trusted shader tool profile. Do not edit.
// ============================================================================
#pragma once
namespace armigo::shader {{
inline constexpr char expected_metal_chain_id[] = \"{value}\";
}}  // namespace armigo::shader
"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(destination)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translate-tool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    emit_expected_header(canonical_json(current_profile(arguments.translate_tool)),
                         arguments.output)
