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

"""End-to-end extraction checks using independently authored synthetic bytes."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = str(Path(sys.argv[1]).resolve())


def shader_container(stage=0, c1_bytes=0):
    control = bytearray(0x718)
    program_start = 0 if stage == 5 else 0x80
    code_offset = 0 if stage == 5 else 0x30
    c1_offset = program_start + 32
    code = bytearray(c1_offset + c1_bytes)
    for offset, value in ((0, 0x98761234), (4, 1), (8, 14), (0x20, len(control)),
                          (0x700, c1_offset), (0x704, len(code)), (0x708, code_offset),
                          (0x714, stage)):
        struct.pack_into("<I", control, offset, value)
    if stage != 5:
        struct.pack_into("<I", code, 0, 0x12345678)
        struct.pack_into("<I", code, 0x30, 1 << 10)
    code[program_start:c1_offset] = bytes(range(32))
    code[c1_offset:] = bytes([0xCE]) * c1_bytes
    container = bytearray(0xC00)
    container[:8] = b"BNSH\0\0\0\0"
    container[0x60:0x64] = b"grsc"
    struct.pack_into("<I", container, 0x7C, 1)
    for offset, value in ((0x80, 0x100), (0x110, 0x180), (0x188 + (40 if stage == 5 else 0), 0x200),
                          (0x208, 0x300), (0x210, 0xB00)):
        struct.pack_into("<Q", container, offset, value)
    struct.pack_into("<II", container, 0x218, len(code), len(control))
    container[0x300:0x300 + len(control)] = control
    container[0xB00:0xB00 + len(code)] = code
    return bytes(container), bytes(control), bytes(code)


def sarc(payload):
    archive = bytearray(0x60) + payload
    archive[:4] = b"SARC"
    struct.pack_into("<HHII", archive, 4, 0x14, 0xFEFF, len(archive), 0x60)
    archive[0x14:0x18] = b"SFAT"
    struct.pack_into("<HHI", archive, 0x18, 12, 1, 0x65)
    struct.pack_into("<IIII", archive, 0x20, 0, 1 << 24, 0, len(payload))
    archive[0x30:0x34] = b"SFNT"
    struct.pack_into("<H", archive, 0x34, 8)
    archive[0x38:0x43] = b"shader.bin\0"
    return bytes(archive)


def literal_yaz0(payload):
    encoded = bytearray(b"Yaz0" + struct.pack(">I", len(payload)) + bytes(8))
    for offset in range(0, len(payload), 8):
        encoded.append(0xFF)
        encoded += payload[offset:offset + 8]
    return bytes(encoded)


def literal_lz4(payload):
    if len(payload) < 15:
        return bytes([len(payload) << 4]) + payload
    encoded = bytearray([0xF0])
    remaining = len(payload) - 15
    while remaining >= 255:
        encoded.append(255)
        remaining -= 255
    encoded.append(remaining)
    return bytes(encoded) + payload


def nso_wrapper(container, compressed=False, empty_data=False, module_header=True):
    """An authored NSO with shader containers in distinct image segments."""
    segments = [bytearray(0x1000), bytearray(0x1000),
                bytearray() if empty_data else bytearray(0x1000)]
    image_offsets = (0, 0x4000, 0x8000)
    for segment in segments:
        if segment:
            segment[0x100:0x100 + len(container)] = container
    bss_size = 0x1800
    if module_header:
        struct.pack_into("<I", segments[0], 4, 0x80)
        segments[0][0x80:0x84] = b"MOD0"
        bss_start = image_offsets[2] + len(segments[2])
        relative = (0x4000 - 0x80, bss_start - 0x80,
                    bss_start + bss_size - 0x80, 0, 0, 0x8000 - 0x80)
        struct.pack_into("<6i", segments[0], 0x84, *relative)
    header = bytearray(0x100)
    header[:4] = b"NSO0"
    mask = sum(1 << i for i, segment in enumerate(segments) if compressed and segment)
    struct.pack_into("<I", header, 0xC, 0x38 | mask)
    struct.pack_into("<I", header, 0x3C, bss_size)
    body = bytearray()
    for index, segment in enumerate(segments):
        stored = literal_lz4(segment) if mask & (1 << index) else bytes(segment)
        file_offset = 0x100 + len(body) if stored else 0xFFFFFFFF
        struct.pack_into("<III", header, 0x10 + index * 0x10,
                         file_offset, image_offsets[index], len(segment))
        struct.pack_into("<I", header, 0x60 + index * 4, len(stored))
        header[0xA0 + index * 32:0xC0 + index * 32] = hashlib.sha256(segment).digest()
        body.extend(stored)
    return bytes(header + body)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="armigo-extract-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input.bin"
        self.container, self.control, self.code = shader_container()
        self.source.write_bytes(self.container)

    def run_tool(self, output, *arguments):
        return subprocess.run([EXECUTABLE, "--output", str(output), *map(str, arguments)],
                              text=True, capture_output=True, timeout=30)

    def test_nested_archives_identity_dedup_and_readonly_inputs(self):
        wrapped = self.root / "nested.bin"
        wrapped.write_bytes(literal_yaz0(sarc(self.container)))
        original = wrapped.read_bytes()
        observed = self.root / "bound.bin"
        observed.write_bytes(self.code[0x30:] + bytes(16))
        output = self.root / "result"
        result = self.run_tool(output, "--compare-program", observed, self.source, wrapped)
        self.assertEqual(result.returncode, 0, result.stderr)
        inventory = json.loads((output / "inventory.json").read_text())
        self.assertFalse(inventory["runnable"])
        self.assertEqual(inventory["unique_inputs"], 1)
        self.assertEqual(inventory["unique_programs"], 1)
        self.assertEqual(len(inventory["occurrences"]), 2)
        expected = hashlib.sha256(self.code[0x30:]).hexdigest()
        for occurrence in inventory["occurrences"]:
            self.assertEqual(occurrence["program_key"], expected)
            self.assertEqual(occurrence["program_bytes"], len(self.code) - 0x30)
            directory = output / "stages" / occurrence["input"]
            self.assertEqual((directory / "control.bin").read_bytes(), self.control)
            self.assertEqual((directory / "code.bin").read_bytes(), self.code)
            self.assertEqual(occurrence["variation"], 0)
            self.assertEqual(occurrence["stage"], 0)
        self.assertEqual(inventory["observations"][str(observed)], 2)
        self.assertEqual(wrapped.read_bytes(), original)
        self.assertEqual(self.source.read_bytes(), self.container)

    def test_graphics_and_compute_keys_exclude_c1(self):
        for stage in (0, 5):
            container, control, code = shader_container(stage, 24)
            self.source.write_bytes(container)
            output = self.root / f"stage-{stage}"
            result = self.run_tool(output, self.source)
            self.assertEqual(result.returncode, 0, result.stderr)
            inventory = json.loads((output / "inventory.json").read_text())
            occurrence = inventory["occurrences"][0]
            start = 0 if stage == 5 else 0x30
            expected = hashlib.sha256(code[start:-24]).hexdigest()
            self.assertEqual(occurrence["program_key"], expected)
            self.assertEqual(occurrence["program_bytes"], len(code) - start - 24)
            self.assertEqual(occurrence["stage"], stage)

    def test_nso_raw_lz4_image_offsets_and_output_bytes(self):
        inventories = []
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                self.source.write_bytes(nso_wrapper(self.container, compressed))
                output = self.root / ("lz4" if compressed else "raw")
                result = self.run_tool(output, self.source)
                self.assertEqual(result.returncode, 0, result.stderr)
                inventory = json.loads((output / "inventory.json").read_text())
                self.assertEqual(inventory["unique_inputs"], 1)
                self.assertEqual(inventory["unique_programs"], 1)
                self.assertEqual([row["container_offset"] for row in inventory["occurrences"]],
                                 [0x100, 0x4100, 0x8100])
                for row in inventory["occurrences"]:
                    directory = output / "stages" / row["input"]
                    self.assertEqual((directory / "control.bin").read_bytes(), self.control)
                    self.assertEqual((directory / "code.bin").read_bytes(), self.code)
                    self.assertEqual(row["program_key"], hashlib.sha256(self.code[0x30:]).hexdigest())
                inventories.append(inventory)
        self.assertEqual(inventories[0], inventories[1])

    def test_nso_empty_segment_and_zero_bss_are_not_shader_inputs(self):
        self.source.write_bytes(nso_wrapper(self.container, compressed=True,
                                            empty_data=True, module_header=False))
        output = self.root / "empty-data"
        result = self.run_tool(output, self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        inventory = json.loads((output / "inventory.json").read_text())
        self.assertEqual([row["container_offset"] for row in inventory["occurrences"]],
                         [0x100, 0x4100])
        self.assertEqual(inventory["containers"], 2)
        self.assertEqual(inventory["unique_inputs"], 1)

    def test_nso_invalid_inputs_refuse_before_scanning_shader_bytes(self):
        raw = nso_wrapper(self.container)
        cases = []
        corrupt = bytearray(raw)
        corrupt[0xA0] ^= 1
        cases.append(("hash", corrupt, "SHA-256"))
        corrupt = bytearray(raw)
        struct.pack_into("<I", corrupt, 0x20, len(corrupt) + 1)
        cases.append(("range", corrupt, "file range"))
        corrupt = bytearray(raw)
        struct.pack_into("<I", corrupt, 0x24, 0x1000 - 1)
        cases.append(("layout", corrupt, "layout"))
        corrupt = bytearray(raw)
        struct.pack_into("<I", corrupt, 0xC, 0x138)
        cases.append(("flags", corrupt, "unsupported flag bits"))
        # Keep the segment hash valid so the MOD0 range check is the refusal.
        corrupt = bytearray(raw)
        struct.pack_into("<I", corrupt, 0x100 + 0x84, 0x7FFFFFFF)
        corrupt[0xA0:0xC0] = hashlib.sha256(corrupt[0x100:0x1100]).digest()
        cases.append(("mod0", corrupt, "module header"))
        corrupt = bytearray(nso_wrapper(self.container, compressed=True))
        ro_file_offset = struct.unpack_from("<I", corrupt, 0x20)[0]
        corrupt[ro_file_offset] = 0
        cases.append(("lz4", corrupt, "LZ4"))
        cases.append(("truncated", raw[:24], "header"))
        for name, payload, diagnostic in cases:
            with self.subTest(name=name):
                self.source.write_bytes(payload)
                output = self.root / name
                result = self.run_tool(output, self.source)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
                self.assertEqual(list((output / "stages").rglob("*.bin")), [])

    def test_output_cannot_be_nested_under_input_directory(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "asset.bin").write_bytes(self.container)
        output = inputs / "generated" / "result"
        result = self.run_tool(output, inputs)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inside an input directory", result.stderr)
        self.assertFalse(output.exists())
        link = self.root / "input-alias"
        link.symlink_to(inputs, target_is_directory=True)
        result = self.run_tool(link / "generated", inputs)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((inputs / "generated").exists())

    def test_overlapping_archive_members_have_aggregate_work_limit(self):
        def repeated_archive(payload):
            count = 65535
            data_offset = 0x20 + count * 16 + 8
            archive = bytearray(data_offset) + payload
            archive[:4] = b"SARC"
            struct.pack_into("<HHII", archive, 4, 0x14, 0xFEFF, len(archive), data_offset)
            archive[0x14:0x18] = b"SFAT"
            struct.pack_into("<HHI", archive, 0x18, 12, count, 0x65)
            for index in range(count):
                struct.pack_into("<IIII", archive, 0x20 + index * 16, 0, 0, 0, len(payload))
            archive[data_offset - 8:data_offset - 4] = b"SFNT"
            struct.pack_into("<H", archive, data_offset - 4, 8)
            return bytes(archive)
        self.source.write_bytes(repeated_archive(repeated_archive(b"")))
        result = self.run_tool(self.root / "bounded-result", self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("traversal exceeds", result.stderr)

    def test_recognized_invalid_bahs_is_not_successful_extraction(self):
        archive = bytearray(0xA00)
        archive[:4] = b"BAHS"
        struct.pack_into("<II", archive, 0x18, 0x100, 0x900)
        archive[0x100:0x100 + len(self.control)] = self.control
        # The candidate is recognizable but its code cannot fit the region.
        struct.pack_into("<IIIi", archive, 0x40, 0x100, 0, len(self.control), 0xC0)
        self.source.write_bytes(archive)
        output = self.root / "rejected-result"
        result = self.run_tool(output, self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed structural validation", result.stderr)
        inventory = json.loads((output / "inventory.json").read_text())
        self.assertEqual(inventory["rejected_candidates"], 1)
        self.assertFalse(inventory["runnable"])

    def test_program_mismatch_fails(self):
        observed = self.root / "unmatched.bin"
        observed.write_bytes(bytes(len(self.code)))
        result = self.run_tool(self.root / "result", "--compare-program", observed, self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no extracted match", result.stderr)

    def test_worktree_directory_file_and_symlink_are_refused(self):
        for name, marker in (("normal", "directory"), ("linked", "file")):
            repository = self.root / name
            repository.mkdir()
            if marker == "directory":
                (repository / ".git").mkdir()
            else:
                (repository / ".git").write_text("gitdir: ../metadata\n")
            output = repository / "new" / "result"
            result = self.run_tool(output, self.source)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside every Git work tree", result.stderr)
            self.assertFalse(output.exists())
        alias = self.root / "alias"
        alias.symlink_to(self.root / "normal", target_is_directory=True)
        result = self.run_tool(alias / "result", self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((alias / "result").exists())

    def test_existing_output_is_preserved(self):
        output = self.root / "result"
        output.mkdir()
        marker = output / "existing.txt"
        marker.write_text("preserve this file")
        result = self.run_tool(output, self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "preserve this file")
        self.assertFalse((output / "inventory.json").exists())

    def test_invalid_container_and_oversized_input_fail(self):
        malformed = bytearray(self.container)
        struct.pack_into("<Q", malformed, 0x80, (1 << 64) - 1)
        self.source.write_bytes(malformed)
        result = self.run_tool(self.root / "bad-result", self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Shader extraction failed", result.stderr)
        huge = self.root / "oversized.bin"
        with huge.open("wb") as stream:
            stream.truncate(512 * 1024 * 1024 + 1)
        result = self.run_tool(self.root / "large-result", huge)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("file-size policy", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
