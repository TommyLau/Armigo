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

"""Selected imports preserve every legacy file and never replace existing saves."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "migrate_player_saves.py"
SPEC = importlib.util.spec_from_file_location("player_save_import", SCRIPT)
IMPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORT)


class SaveImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.source = self.base / "legacy"
        self.source.mkdir()
        self.destination = self.base / "new" / "selected"
        self.payloads = {"nested/arbitrary.save-data": b"\x00\xff\x80ABC\x00",
                         "unknown-format": bytes(range(256)), "empty": b""}
        for name, payload in self.payloads.items():
            path = self.source / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
        (self.source / "unselected-unknown").write_bytes(b"keep this genuine or unknown file")
        (self.source / "diagnostics").mkdir()
        (self.source / "diagnostics" / "image").write_bytes(b"retain debugging evidence")
        self.manifest = {"schema_version": 1, "files": [
            {"path": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in self.payloads.items()]}
        self.original = {str(p.relative_to(self.source)): p.read_bytes()
                         for p in self.source.rglob("*") if p.is_file()}

    def assert_legacy_preserved(self):
        actual = {str(p.relative_to(self.source)): p.read_bytes()
                  for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(actual, self.original)

    def test_selected_nested_binary_files_only_without_source_traversal(self):
        walk = os.walk
        def guarded_walk(path, *args, **kwargs):
            self.assertNotEqual(Path(path), self.source)
            self.assertNotIn(self.source, Path(path).parents)
            return walk(path, *args, **kwargs)
        with mock.patch.object(IMPORT.os, "walk", side_effect=guarded_walk):
            result = IMPORT.import_saves(self.source, self.destination, self.manifest)
        self.assertEqual(result["files"], 3)
        self.assertEqual(result["bytes"], sum(map(len, self.payloads.values())))
        actual = {str(p.relative_to(self.destination)): p.read_bytes()
                  for p in self.destination.rglob("*") if p.is_file()}
        self.assertEqual(actual, self.payloads)
        self.assert_legacy_preserved()

    def test_size_mismatch_preserves_staging_and_source(self):
        self.manifest["files"][0]["size"] += 1
        with self.assertRaises(IMPORT.ImportFailure):
            IMPORT.import_saves(self.source, self.destination, self.manifest)
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.destination.with_name("selected.import").is_dir())
        self.assert_legacy_preserved()

    def test_hash_mismatch_does_not_publish(self):
        self.manifest["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(IMPORT.ImportFailure):
            IMPORT.import_saves(self.source, self.destination, self.manifest)
        self.assertFalse(self.destination.exists())
        self.assert_legacy_preserved()

    def test_escape_and_ambiguous_paths_rejected_before_staging(self):
        for name in ("../escape", "/absolute", "a/../b", "a//b", "a/./b", "a\\b", "bad\0name", ""):
            with self.subTest(name=name):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0]["path"] = name
                with self.assertRaises(IMPORT.ImportFailure):
                    IMPORT.import_saves(self.source, self.destination, manifest)
                self.assertFalse(self.destination.parent.exists())

    def test_case_and_file_directory_collisions_rejected(self):
        for name in ("NESTED/ARBITRARY.SAVE-DATA", "nested", "nested/arbitrary.save-data/child"):
            with self.subTest(name=name):
                manifest = copy.deepcopy(self.manifest)
                item = dict(manifest["files"][0], path=name)
                manifest["files"].append(item)
                with self.assertRaises(IMPORT.ImportFailure):
                    IMPORT.import_saves(self.source, self.destination, manifest)

    def test_file_and_directory_symlinks_are_not_followed(self):
        outside = self.base / "outside"
        outside.mkdir()
        (outside / "file").write_bytes(b"secret")
        for name, target in (("link", outside / "file"), ("directory", outside)):
            (self.source / name).symlink_to(target)
        for name in ("link", "directory/file"):
            with self.subTest(name=name):
                manifest = {"schema_version": 1, "files": [
                    {"path": name, "size": 6, "sha256": hashlib.sha256(b"secret").hexdigest()}]}
                destination = self.base / ("out-" + name.replace("/", "-"))
                with self.assertRaises(OSError):
                    IMPORT.import_saves(self.source, destination, manifest)
                self.assertFalse(destination.exists())
        self.assertEqual((outside / "file").read_bytes(), b"secret")

    def test_source_and_destination_ancestor_symlinks_rejected(self):
        source_alias = self.base / "source-alias"
        source_alias.symlink_to(self.source)
        destination_alias = self.base / "destination-alias"
        destination_alias.symlink_to(self.base)
        for source, destination in ((source_alias, self.destination),
                                    (self.source, destination_alias / "new-save")):
            with self.assertRaises(IMPORT.ImportFailure):
                IMPORT.import_saves(source, destination, self.manifest)

    def test_existing_destination_is_never_overwritten(self):
        self.destination.mkdir(parents=True)
        (self.destination / "owner-save").write_bytes(b"original")
        with self.assertRaises(IMPORT.ImportFailure):
            IMPORT.import_saves(self.source, self.destination, self.manifest)
        self.assertEqual((self.destination / "owner-save").read_bytes(), b"original")
        self.assert_legacy_preserved()

    def test_racing_empty_destination_cannot_be_replaced(self):
        publish = IMPORT.publish_exclusively
        def racing_publish(staged, destination):
            destination.mkdir()
            publish(staged, destination)
        with mock.patch.object(IMPORT, "publish_exclusively", side_effect=racing_publish):
            with self.assertRaises(FileExistsError):
                IMPORT.import_saves(self.source, self.destination, self.manifest)
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertTrue(self.destination.with_name("selected.import").is_dir())
        self.assert_legacy_preserved()

    def test_interrupted_staging_is_not_deleted_or_reused(self):
        staged = self.destination.with_name("selected.import")
        staged.mkdir(parents=True)
        (staged / "partial").write_bytes(b"preserve pending import")
        with self.assertRaises(IMPORT.ImportFailure):
            IMPORT.import_saves(self.source, self.destination, self.manifest)
        self.assertFalse(self.destination.exists())
        self.assertEqual((staged / "partial").read_bytes(), b"preserve pending import")
        self.assert_legacy_preserved()

    def test_nested_destination_and_malformed_description_rejected(self):
        with self.assertRaises(IMPORT.ImportFailure):
            IMPORT.import_saves(self.source, self.source / "new", self.manifest)
        for manifest in ({"schema_version": True, "files": []},
                         {"schema_version": 1, "files": []},
                         {"schema_version": 2, "files": self.manifest["files"]}):
            with self.assertRaises(IMPORT.ImportFailure):
                IMPORT.import_saves(self.source, self.destination, manifest)

    def test_cli_reports_success_and_rejects_duplicate_fields(self):
        manifest_path = self.base / "selected.json"
        manifest_path.write_text(json.dumps(self.manifest))
        command = [sys.executable, str(SCRIPT), "--source", str(self.source),
                   "--destination", str(self.destination), "--manifest", str(manifest_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["outcome"], "imported")
        manifest_path.write_text('{"schema_version":1,"schema_version":1,"files":[]}')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate field", result.stderr)
        self.assert_legacy_preserved()


if __name__ == "__main__":
    unittest.main()
