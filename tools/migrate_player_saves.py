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

"""Import explicitly selected saves without traversing or changing legacy data."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import unicodedata


class ImportFailure(ValueError):
    pass


def selected_files(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        raise ImportFailure("The import description must contain schema_version and files.")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ImportFailure("The import description version is unsupported.")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ImportFailure("Select at least one saved-data file.")
    paths = set()
    result = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ImportFailure("Each selected file needs its path, size and SHA-256.")
        name = item["path"]
        if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
                or PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ImportFailure("A selected path escapes or ambiguously names the source directory.")
        key = unicodedata.normalize("NFC", name).casefold()
        if key in paths or any(key.startswith(old + "/") or old.startswith(key + "/") for old in paths):
            raise ImportFailure("Selected paths collide.")
        if type(item["size"]) is not int or item["size"] < 0:
            raise ImportFailure("A selected file size is invalid.")
        if not isinstance(item["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", item["sha256"]):
            raise ImportFailure("A selected file SHA-256 is invalid.")
        paths.add(key)
        result.append(item)
    return result


def reject_symlink_ancestors(path):
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ImportFailure("Symbolic links are not allowed in import locations.")


def open_selected(root, name):
    directory = os.dup(root)
    try:
        parts = name.split("/")
        for part in parts[:-1]:
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)


def publish_exclusively(staged, destination):
    """Publish a directory atomically without replacing even an empty directory."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(staged), os.fsencode(destination), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(staged), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise ImportFailure("This host cannot publish imported saves without replacement.")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def import_saves(source, destination, manifest):
    files = selected_files(manifest)
    source = Path(os.path.abspath(source))
    destination = Path(os.path.abspath(destination))
    reject_symlink_ancestors(source)
    reject_symlink_ancestors(destination)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ImportFailure("Source and destination save directories must be separate.")
    if os.path.lexists(destination):
        raise ImportFailure("The destination already exists; no data was replaced.")
    staged = destination.with_name(destination.name + ".import")
    reject_symlink_ancestors(staged)
    if os.path.lexists(staged):
        raise ImportFailure("An earlier import staging directory exists; preserve and inspect it before retrying.")
    root = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_ancestors(destination.parent)
        staged.mkdir()
        total = 0
        for item in files:
            descriptor = open_selected(root, item["path"])
            with os.fdopen(descriptor, "rb") as incoming:
                before = os.fstat(incoming.fileno())
                if not stat.S_ISREG(before.st_mode) or before.st_size != item["size"]:
                    raise ImportFailure("Selected source file type or size does not match: " + item["path"])
                target = staged / item["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                copied = 0
                with target.open("xb") as outgoing:
                    while copied < item["size"]:
                        chunk = incoming.read(min(1 << 20, item["size"] - copied))
                        if not chunk:
                            raise ImportFailure("Selected source file ended early: " + item["path"])
                        outgoing.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                after = os.fstat(incoming.fileno())
                if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                        or digest.hexdigest() != item["sha256"]):
                    raise ImportFailure("Selected source file changed or its SHA-256 does not match: " + item["path"])
                total += copied
        # Only the staging tree is traversed, never the legacy source tree.
        for directory, _, _ in os.walk(staged, topdown=False):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        publish_exclusively(staged, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {"schema_version": 1, "outcome": "imported", "files": len(files), "bytes": total}
    finally:
        os.close(root)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImportFailure("The import description contains a duplicate field.")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        with args.manifest.open(encoding="utf-8") as stream:
            manifest = json.load(stream, object_pairs_hook=unique_object)
        receipt = import_saves(args.source, args.destination, manifest)
    except (ImportFailure, OSError, json.JSONDecodeError) as error:
        print("Save import failed: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
