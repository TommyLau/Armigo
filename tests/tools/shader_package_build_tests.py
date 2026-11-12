#!/usr/bin/env python3
# ============================================================================
# Armigo Tests - Offline Shader Translation
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Synthetic checks for deterministic offline Metal package construction."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "src" / "shader" / "tools"
SCRIPT = TOOLS / "build_package.py"
SOURCE_VERSION = "3a" * 32


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class PackageBuildTests(unittest.TestCase):
    def test_direct_storage_descriptor_without_gpu_constant_window(self):
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        binding = {"cbuf_slot": 0, "descriptor_byte_offset": 0x510,
                   "metal_buffer_index": 0, "access": "read", "alignment": 4}
        fact = {"kind": "bound_storage_descriptor", "index": 0,
                "byte_offset": 0x510, "component": 0,
                "origin": "validated_header", "value": True}
        self.assertEqual(module.validate_storage_buffers_v3([binding], [], [fact]), {0})
        for dependencies in ([], [dict(fact, value=False)], [dict(fact, byte_offset=0x520)]):
            with self.subTest(dependencies=dependencies), self.assertRaises(RuntimeError):
                module.validate_storage_buffers_v3([binding], [], dependencies)
        module.validate_dependencies([fact], 0)
        with self.assertRaises(RuntimeError):
            module.validate_dependencies([dict(fact, value=1)], 0)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="armigo-package-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.inputs = self.root / "inputs"
        self.bin.mkdir()
        self.inputs.mkdir()
        self.log = self.root / "xcrun.log"
        self.translate = self.bin / "translate"
        self.translate.write_text(
            "#!/usr/bin/env python3\nimport sys\n"
            f"print('{SOURCE_VERSION}') if sys.argv[1:] == ['--version'] else sys.exit(2)\n")
        self.translate.chmod(0o755)
        (self.bin / "metal-tool").write_bytes(b"synthetic metal executable")
        (self.bin / "metallib-tool").write_bytes(b"synthetic metallib executable")
        xcrun = self.bin / "xcrun"
        xcrun.write_text("""#!/usr/bin/env python3
import hashlib, json, os
from pathlib import Path
import sys, time
a=sys.argv[1:]
root=Path(__file__).parent
if a == ['--sdk','macosx','--find','metal']: print(root/'metal-tool'); raise SystemExit
if a == ['--sdk','macosx','--find','metallib']: print(root/'metallib-tool'); raise SystemExit
if a == ['--sdk','macosx','metal','--version']: print('Apple metal synthetic 4.0'); raise SystemExit
if a == ['--sdk','macosx','--show-sdk-version']: print('26.1'); raise SystemExit
if a == ['--sdk','macosx','--show-sdk-build-version']: print('26B77'); raise SystemExit
tool=a[2] if a[:2] == ['--sdk','macosx'] else ''
with open(os.environ['FAKE_XCRUN_LOG'],'a') as f: f.write(json.dumps(a)+'\\n')
if os.environ.get('FAKE_FAIL') == tool: print('synthetic failure',file=sys.stderr); raise SystemExit(7)
if os.environ.get('FAKE_SLEEP') == tool: time.sleep(1)
out=Path(a[a.index('-o')+1])
source=Path(a[a.index('-c')+1]) if tool == 'metal' else Path(a[3])
prefix=b'AIR' if tool == 'metal' else b'LIB'
out.write_bytes(prefix+hashlib.sha256(source.read_bytes()).digest())
""")
        xcrun.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment["PATH"] = str(self.bin) + os.pathsep + self.environment.get("PATH", "")
        self.environment["FAKE_XCRUN_LOG"] = str(self.log)

    def stage(self, name, stage="vertex", size=160, seed=1, options=None):
        root = self.inputs / name
        root.mkdir()
        program = bytearray((index * 29 + seed * 17) & 255 for index in range(size))
        struct.pack_into("<I", program, 0, (1 if stage == "vertex" else 5) << 10)
        source = ("#include <metal_stdlib>\n// " + name + "\n").encode()
        if options is None:
            options = ({"vertex_depth": "minus-one-to-one",
                        "fragment_position": "unspecified",
                        "vertex_attributes": [{"generic_slot": 0, "attribute_index": 2,
                                               "format": "float32x3"}]}
                       if stage == "vertex" else
                       {"vertex_depth": "unspecified",
                        "fragment_position": "reciprocal-clip-w",
                        "vertex_attributes": []})
        metadata = {
            "schema": "armigo-shader-stage-v1", "source_version": SOURCE_VERSION,
            "program_key": hashlib.sha256(program).hexdigest(), "program_size": len(program),
            "program_file": "program.bin", "source_file": "source.metal",
            "source_digest": hashlib.sha256(source).hexdigest(), "stage": stage,
            "entry_point": "armigo_vertex_main" if stage == "vertex" else "armigo_fragment_main",
            **options, "constant_buffers": [], "textures": [],
            "interface": {"position_input_mask": 0 if stage == "vertex" else 8,
                          "position_output_mask": 15 if stage == "vertex" else 0,
                          "color_output_mask": 0 if stage == "vertex" else 15,
                          "generic_inputs": ([{"slot": 0, "component_mask": 7,
                                               "interpolation": "screen-linear"}]
                                             if stage == "vertex" else
                                             [{"slot": 0, "component_mask": 3,
                                               "interpolation": "perspective"}]),
                          "generic_outputs": ([{"slot": 0, "component_mask": 3,
                                                "interpolation": "screen-linear"}]
                                              if stage == "vertex" else [])},
        }
        (root / "program.bin").write_bytes(program)
        (root / "source.metal").write_bytes(source)
        (root / "reflection.json").write_text(canonical(metadata) + "\n")
        return root

    def stage_v2(self, name, stage="vertex", size=160, seed=1, mutate=None):
        """A synthetic armigo-shader-stage-v2 stage in the ported pipeline's
        admitted runtime subset (float32x3 streams, one constant buffer)."""
        root = self.inputs / name
        root.mkdir()
        program = bytearray((index * 31 + seed * 13) & 255 for index in range(size))
        struct.pack_into("<I", program, 0, (1 if stage == "vertex" else 5) << 10)
        source = ("#include <metal_stdlib>\n// " + name + "\n").encode()
        c1 = None
        metadata = {
            "schema": "armigo-shader-stage-v2", "source_version": SOURCE_VERSION,
            "program_key": hashlib.sha256(program).hexdigest(), "program_size": len(program),
            "program_file": "program.bin", "source_file": "source.metal",
            "source_digest": hashlib.sha256(source).hexdigest(), "stage": stage,
            "entry_point": "armigo_vertex_main" if stage == "vertex" else "armigo_fragment_main",
            "vertex_depth": "minus-one-to-one" if stage == "vertex" else "unspecified",
            "fragment_position": "unspecified" if stage == "vertex" else "reciprocal-clip-w",
            "c1_digest": c1,
            "profile": {"identity": "armigo-macos-metal4", "version": 1},
            "constant_buffers": [{"guest_binding": 3,
                                  "size_kind": "constant_buffer_window",
                                  "window_bytes": 64, "alignment": 4,
                                  "policy": "runtime_zero_words_v1",
                                  "metal_buffer_index": 0}],
            "support_buffers": [{"schema": "armigo-cb-sizes", "version": 1,
                                 "byte_extent": 72, "alignment": 4,
                                 "metal_buffer_index": 1}],
            "textures": [],
            "vertex_streams": ([] if stage == "fragment" else
                               [{"slot": 0, "attribute_index": 2, "format": "f32",
                                 "components": 3, "byte_stride": 12}]),
            "vertex_inputs": [],
            "interface": {"position_input_mask": 0 if stage == "vertex" else 8,
                          "position_output_mask": 15 if stage == "vertex" else 0,
                          "color_output_mask": 0 if stage == "vertex" else 15,
                          "color_outputs": ([] if stage == "vertex" else
                                            [{"location": 0, "component_mask": 15,
                                              "scalar": "f32", "components": 4}]),
                          "writes_fragment_depth": False,
                          "writes_sample_mask": False,
                          "generic_inputs": ([{"slot": 0, "component_mask": 7,
                                               "scalar": "f32", "components": 3,
                                               "interpolation": "linear"}]
                                             if stage == "vertex" else
                                             [{"slot": 0, "component_mask": 3,
                                               "scalar": "f32", "components": 4,
                                               "interpolation": "perspective"}]),
                          "generic_outputs": ([{"slot": 0, "component_mask": 3,
                                                "scalar": "f32", "components": 4,
                                                "interpolation": "linear"}]
                                              if stage == "vertex" else [])},
            "dependencies": [{"kind": "depth_convention", "index": 0, "byte_offset": 0,
                              "component": 0, "origin": "explicit_option",
                              "value": "minus-one-to-one"}],
        }
        if stage == "fragment":
            metadata["dependencies"].append(
                {"kind": "fragment_output_type", "index": 0, "byte_offset": 0,
                 "component": 0, "origin": "derived",
                 "value": {"scalar": "f32", "components": 4}})
        if stage == "vertex":
            metadata["dependencies"].append(
                {"kind": "vertex_attribute_format", "index": 0, "byte_offset": 0,
                 "component": 0, "origin": "explicit_option",
                 "value": {"format": "f32", "components": 3}})
        if mutate is not None:
            mutate(metadata)
        (root / "program.bin").write_bytes(program)
        (root / "source.metal").write_bytes(source)
        (root / "reflection.json").write_text(canonical(metadata) + "\n")
        return root

    def stage_v3(self, name, resources, argument_buffer=None, mutate=None, seed=1,
                 stage="vertex"):
        root = self.stage_v2(name, stage=stage, seed=seed)
        metadata = json.loads((root / "reflection.json").read_text())
        metadata["schema"] = "armigo-shader-stage-v3"
        metadata["profile"]["version"] = 3
        metadata["argument_buffer"] = argument_buffer
        metadata["textures"] = resources
        if mutate is not None:
            mutate(metadata)
        (root / "reflection.json").write_text(canonical(metadata) + "\n")
        return root

    @staticmethod
    def v3_texture(resource_id, first=None, byte_offset=None, array_length=1):
        location = ({"kind": "direct", "namespace": "texture",
                     "first": first, "count": array_length}
                    if first is not None else
                    {"kind": "argument_buffer_member", "byte_offset": byte_offset,
                     "count": array_length, "stride": 8})
        return {"cbuf_slot": 3, "handle_byte_offset": resource_id * 4,
                "kind": "texture", "dimension": "d2", "array": False,
                "multisample": False, "scalar": "f32",
                "component_format": "floating", "normalized_coordinates": True,
                "comparison": False, "resource_id": resource_id,
                "array_length": array_length, "location": location,
                "resource_kind": "sampled", "access": "read"}

    @staticmethod
    def atomic_image(scalar, component, **changes):
        image = {"cbuf_slot": 3, "handle_byte_offset": 0, "kind": "texture",
                 "dimension": "d2", "array": False, "multisample": False,
                 "scalar": scalar, "component_format": component,
                 "normalized_coordinates": False, "comparison": False,
                 "resource_id": 0, "array_length": 1,
                 "location": {"kind": "direct", "namespace": "texture",
                              "first": 0, "count": 1},
                 "resource_kind": "storage_image", "access": "read_write"}
        image.update(changes)
        return image

    @staticmethod
    def v3_storage(**changes):
        binding = {"cbuf_slot": 3, "descriptor_byte_offset": 8,
                   "metal_buffer_index": 2, "access": "read_write",
                   "alignment": 4}
        binding.update(changes)
        return binding

    def test_accepts_v2_stages_alongside_v1_and_specializes_on_facts(self):
        vertex_v2 = self.stage_v2("ported-vertex")
        fragment_v1 = self.stage("ported-fragment", "fragment", 176, 5)
        output = self.root / "mixed"
        result = self.run_tool(output, (fragment_v1, vertex_v2))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["entries"]), 2)
        reflection = json.loads(
            (output / "stages" / manifest["entries"][0]["directory"].split("/")[1] /
             "reflection.json").read_text())
        self.assertIn(reflection["schema"],
                      ("armigo-shader-stage-v1", "armigo-shader-stage-v2"))
        # Two v2 stages differing only in a consumed fact value must receive
        # different variant identities, matching contents each.
        depth = self.stage_v2("ported-depth", mutate=lambda value:
                              value["dependencies"].__setitem__(0, {
                                  **value["dependencies"][0], "value": "zero-to-one"}))
        self.assertNotEqual(
            (vertex_v2 / "reflection.json").read_text(),
            (depth / "reflection.json").read_text())
        both = self.root / "both-depths"
        result = self.run_tool(both, (vertex_v2, depth))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((both / "manifest.json").read_text())["entries"]), 2)

    def test_accepts_v3_direct_resources_beyond_legacy_texture_limit(self):
        resources = [self.v3_texture(index, first=index) for index in range(17)]
        resources[-1]["handle_byte_offset"] = 65532
        stage = self.stage_v3("v3-direct-17", resources)
        output = self.root / "v3-direct-17-out"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        packaged = json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["schema"], "armigo-shader-stage-v3")
        self.assertEqual(packaged["profile"]["version"], 3)
        self.assertEqual(packaged["textures"][-1]["location"]["first"], 16)

        indirect = self.stage_v3(
            "v3-indirect-order",
            [self.v3_texture(7, byte_offset=8), self.v3_texture(2, byte_offset=0)],
            {"abi_version": 1, "buffer_index": 2,
             "byte_extent": 16, "alignment": 8}, seed=19)
        indirect_output = self.root / "v3-indirect-order-out"
        result = self.run_tool(indirect_output, (indirect,))
        self.assertEqual(result.returncode, 0, result.stderr)

    @staticmethod
    def texture_pool_record(first, cbuf=None):
        record = {"kind": "texture_pool", "dimension": "d2", "array": False,
                  "multisample": False, "scalar": "f32", "component_format": "floating",
                  "comparison": False,
                  "resource_kind": "sampled", "access": "read", "resource_id": 0,
                  "array_length": 1,
                  "location": {"kind": "direct", "namespace": "buffer",
                               "first": first, "count": 1}}
        if cbuf is not None:
            record["cbuf_slot"] = cbuf
            record["handle_byte_offset"] = 0
        return record

    @staticmethod
    def sampler_pool_record(first, resource_id, cbuf=None, state=False):
        record = {"kind": "sampler_pool", "resource_id": resource_id, "array_length": 1,
                  "location": {"kind": "direct", "namespace": "buffer",
                               "first": first, "count": 1}}
        if cbuf is not None:
            record["cbuf_slot"] = cbuf
            record["handle_byte_offset"] = 4
        if cbuf is not None or state:
            record["comparison"] = False
            record["normalized_coordinates"] = True
        return record

    def test_accepts_texture_and_sampler_pools(self):
        bare = self.stage_v3(
            "pools-bare",
            [self.texture_pool_record(2), self.sampler_pool_record(3, 1)],
            stage="fragment", seed=42)
        bare_output = self.root / "pools-bare-out"
        result = self.run_tool(bare_output, (bare,))
        self.assertEqual(result.returncode, 0, result.stderr)
        bare_manifest = json.loads((bare_output / "manifest.json").read_text())
        bare_packaged = json.loads((bare_output / bare_manifest["entries"][0]["directory"] /
                                    "reflection.json").read_text())
        self.assertEqual([item["kind"] for item in bare_packaged["textures"]],
                         ["texture_pool", "sampler_pool"])
        self.assertNotIn("cbuf_slot", bare_packaged["textures"][0])
        self.assertNotIn("normalized_coordinates", bare_packaged["textures"][0])
        self.assertNotIn("comparison", bare_packaged["textures"][1])
        self.assertEqual(bare_packaged["textures"][1]["location"]["namespace"], "buffer")

        keyed = self.stage_v3(
            "pools-keyed",
            [self.texture_pool_record(2, 30), self.sampler_pool_record(3, 1, 30)],
            stage="fragment", seed=41)
        self.failure(self.root / "pools-keyed-out", (keyed,),
                     "do not carry constant-buffer keys")
        stated = self.stage_v3(
            "pools-sampler-state",
            [self.texture_pool_record(2), self.sampler_pool_record(3, 1, state=True)],
            stage="fragment", seed=43)
        self.failure(self.root / "pools-sampler-state-out", (stated,),
                     "do not carry sampler state")

        second = self.stage_v3(
            "pools-two-samplers",
            [self.texture_pool_record(2), self.sampler_pool_record(3, 1),
             self.sampler_pool_record(4, 2)],
            stage="fragment", seed=44)
        self.failure(self.root / "pools-two-samplers-out", (second,), "one sampler pool")
        bad_slot = self.stage_v3(
            "pools-bad-slot",
            [self.texture_pool_record(2, 17), self.sampler_pool_record(3, 1, 30)],
            stage="fragment", seed=45)
        self.failure(self.root / "pools-bad-slot-out", (bad_slot,),
                     "do not carry constant-buffer keys")
        texture = self.v3_texture(0, first=0)
        texture["cbuf_slot"] = 30
        non_pool = self.stage_v3("pools-texture-slot-30", [texture], stage="fragment", seed=46)
        self.failure(self.root / "pools-texture-slot-30-out", (non_pool,),
                     "texture provenance reflection is invalid")
        v2 = self.stage_v2("pools-v2", "fragment", mutate=lambda value: value.update(
            textures=[{"kind": "texture_pool"}]))
        self.failure(self.root / "pools-v2-out", (v2,), "texture kind is unsupported")

    def test_custom_border_capability_is_versioned_and_changes_variant(self):
        texture = self.v3_texture(0, first=0)
        sampler = {"kind": "sampler", "resource_id": 1, "array_length": 1,
                   "cbuf_slot": 3, "handle_byte_offset": 0,
                   "comparison": False, "normalized_coordinates": True,
                   "location": {"kind": "direct", "namespace": "sampler",
                                "first": 0, "count": 1}}
        border = {"abi": "coverage-v1", "buffer_index": 2,
                  "pairs": [{"texture": 0, "sampler": 1}]}
        stage = self.stage_v3("custom-border", [texture, sampler], stage="fragment",
                              mutate=lambda value: value.update(custom_border=border))
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        loaded = module.load_stage(stage)
        without_border = json.loads(canonical(loaded["metadata"]))
        without_border.pop("custom_border")
        self.assertNotEqual(module.variant_data(loaded["metadata"]),
                            module.variant_data(without_border))
        module.validate_reflection_v2(without_border, (stage / "program.bin").read_bytes(),
                                     (stage / "source.metal").read_bytes())
        output = self.root / "custom-border-output"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        packaged = json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["custom_border"], border)
        original = json.loads((stage / "reflection.json").read_text())
        mutations = (
            lambda b: b.update(abi="coverage-v2"),
            lambda b: b.update(buffer_index=0),
            lambda b: b.update(buffer_index=31),
            lambda b: b.update(pairs=[]),
            lambda b: b["pairs"][0].update(texture=99),
            lambda b: b["pairs"][0].update(sampler=0),
            lambda b: b["pairs"].append(dict(b["pairs"][0])),
        )
        for index, mutate in enumerate(mutations):
            metadata = json.loads(canonical(original))
            mutate(metadata["custom_border"])
            (stage / "reflection.json").write_text(canonical(metadata))
            self.failure(self.root / ("bad-border-" + str(index)), (stage,), "custom border")

    def test_custom_border_supports_vertex_stages_and_reserved_streams(self):
        texture = self.v3_texture(0, first=0)
        sampler = {"kind": "sampler", "resource_id": 1, "array_length": 1,
                   "cbuf_slot": 3, "handle_byte_offset": 0,
                   "comparison": False, "normalized_coordinates": True,
                   "location": {"kind": "direct", "namespace": "sampler",
                                "first": 0, "count": 1}}
        border = {"abi": "coverage-v1", "buffer_index": 2,
                  "pairs": [{"texture": 0, "sampler": 1}]}
        stage = self.stage_v3("custom-border-vertex", [texture, sampler],
                              mutate=lambda value: value.update(custom_border=border))
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        metadata = json.loads((stage / "reflection.json").read_text())
        module.validate_reflection_v2(metadata, (stage / "program.bin").read_bytes(),
                                      (stage / "source.metal").read_bytes())
        output = self.root / "custom-border-vertex-output"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        packaged = json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["custom_border"], border)
        # The vertex stream reserves buffer 18; the support buffer must not.
        colliding = json.loads(canonical(metadata))
        colliding["custom_border"]["buffer_index"] = 18
        (stage / "reflection.json").write_text(canonical(colliding))
        self.failure(self.root / "bad-border-vertex-stream", (stage,),
                     "custom border support overlaps")

    def test_v3_draw_support_is_versioned_and_changes_variant(self):
        support = {"schema": "armigo-draw-parameters", "version": 1,
                   "byte_extent": 4, "alignment": 4, "metal_buffer_index": 2}
        stage = self.stage_v3("draw-support", [], mutate=lambda value:
                              value.update(draw_support=support))
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        metadata = json.loads((stage / "reflection.json").read_text())
        program = (stage / "program.bin").read_bytes()
        source = (stage / "source.metal").read_bytes()
        module.validate_reflection_v2(metadata, program, source)
        legacy = json.loads(canonical(metadata))
        legacy.pop("draw_support")
        module.validate_reflection_v2(legacy, program, source)
        self.assertNotEqual(module.variant_data(metadata), module.variant_data(legacy))
        for key, invalid in (("schema", "future"), ("version", 2),
                             ("byte_extent", 8), ("alignment", 8),
                             ("metal_buffer_index", 31), ("metal_buffer_index", 0),
                             ("metal_buffer_index", 1), ("metal_buffer_index", 18)):
            broken = json.loads(canonical(metadata))
            broken["draw_support"][key] = invalid
            with self.subTest(key=key, value=invalid), self.assertRaises(RuntimeError):
                module.validate_reflection_v2(broken, program, source)
        output = self.root / "draw-support-out"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        packaged = json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["draw_support"], support)

    def test_v3_storage_buffers_round_trip_and_remain_optional(self):
        binding = self.v3_storage()
        stage = self.stage_v3(
            "v3-storage", [], mutate=lambda value:
            value.update(storage_buffers=[binding]))
        output = self.root / "v3-storage-out"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        packaged = json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["storage_buffers"], [binding])

        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        program = (stage / "program.bin").read_bytes()
        source = (stage / "source.metal").read_bytes()
        omitted = json.loads((stage / "reflection.json").read_text())
        omitted.pop("storage_buffers")
        empty = json.loads(canonical(omitted))
        empty["storage_buffers"] = []
        module.validate_reflection_v2(omitted, program, source)
        module.validate_reflection_v2(empty, program, source)
        self.assertEqual(module.variant_data(omitted), module.variant_data(empty))

        for schema, factory in (("v1", self.stage), ("v2", self.stage_v2)):
            legacy = factory("storage-schema-" + schema)
            metadata = json.loads((legacy / "reflection.json").read_text())
            metadata["storage_buffers"] = []
            with self.subTest(schema=schema), self.assertRaisesRegex(
                    RuntimeError, "unexpected schema"):
                module.validate_reflection(
                    metadata, (legacy / "program.bin").read_bytes(),
                    (legacy / "source.metal").read_bytes())

    def test_rejects_malformed_v3_storage_buffers_before_apple_tools(self):
        mutations = (
            ("not-array", lambda value: value.update(storage_buffers={})),
            ("missing-field", lambda value: value["storage_buffers"][0].pop("access")),
            ("extra-field", lambda value: value["storage_buffers"][0].update(extra=0)),
            ("slot", lambda value: value["storage_buffers"][0].update(cbuf_slot=18)),
            ("missing-window", lambda value: value["storage_buffers"][0].update(cbuf_slot=4)),
            ("misaligned-offset", lambda value:
             value["storage_buffers"][0].update(descriptor_byte_offset=4)),
            ("window-overrun", lambda value:
             value["storage_buffers"][0].update(descriptor_byte_offset=56)),
            ("offset-limit", lambda value:
             value["storage_buffers"][0].update(descriptor_byte_offset=65528)),
            ("metal-limit", lambda value:
             value["storage_buffers"][0].update(metal_buffer_index=31)),
            ("constant-collision", lambda value:
             value["storage_buffers"][0].update(metal_buffer_index=0)),
            ("support-collision", lambda value:
             value["storage_buffers"][0].update(metal_buffer_index=1)),
            ("stream-collision", lambda value:
             value["storage_buffers"][0].update(metal_buffer_index=18)),
            ("access", lambda value: value["storage_buffers"][0].update(access="atomic")),
            ("alignment", lambda value: value["storage_buffers"][0].update(alignment=8)),
            ("duplicate-descriptor", lambda value: value["storage_buffers"].append(
                {**value["storage_buffers"][0], "metal_buffer_index": 3})),
            ("duplicate-metal", lambda value: value["storage_buffers"].append(
                {**value["storage_buffers"][0], "descriptor_byte_offset": 16})),
        )
        for index, (name, mutate) in enumerate(mutations):
            def apply(value, change=mutate):
                value["storage_buffers"] = [self.v3_storage()]
                change(value)
            stage = self.stage_v3("malformed-v3-storage-" + name, [],
                                  mutate=apply, seed=index + 50)
            self.failure(self.root / ("malformed-v3-storage-out-" + name),
                         (stage,), "armigo-shader-build-package")

        argument = self.stage_v3(
            "malformed-v3-storage-argument", [self.v3_texture(0, byte_offset=0)],
            {"abi_version": 1, "buffer_index": 2,
             "byte_extent": 8, "alignment": 8},
            lambda value: value.update(storage_buffers=[self.v3_storage()]), seed=90)
        self.failure(self.root / "malformed-v3-storage-argument-out", (argument,),
                     "armigo-shader-build-package")

        def depth_collision(value):
            value["constant_buffers"] = [{
                "guest_binding": 3, "size_kind": "constant_buffer_window",
                "window_bytes": 64, "alignment": 4,
                "policy": "runtime_zero_words_v1", "metal_buffer_index": 0}]
            value["support_buffers"] = [{
                "schema": "armigo-cb-sizes", "version": 1,
                "byte_extent": 72, "alignment": 4, "metal_buffer_index": 1}]
            value["storage_buffers"] = [self.v3_storage(metal_buffer_index=30)]
            value["depth_bias_mode"] = "shader-v1"
        depth = self.stage_v3("malformed-v3-storage-depth", [],
                              mutate=depth_collision, seed=91, stage="fragment")
        self.failure(self.root / "malformed-v3-storage-depth-out", (depth,),
                     "armigo-shader-build-package")

    def test_v3_arrays_and_separate_sampler_dependencies_round_trip(self):
        texture = self.v3_texture(4, byte_offset=0, array_length=17)
        texture["handle_byte_offset"] = 0
        scalar = self.v3_texture(2, byte_offset=136)
        scalar["handle_byte_offset"] = 0
        sampler = {"kind": "sampler", "resource_id": 8, "cbuf_slot": 4,
                   "handle_byte_offset": 128, "array_length": 17,
                   "comparison": False, "normalized_coordinates": True,
                   "location": {"kind": "argument_buffer_member", "byte_offset": 144,
                                "count": 17, "stride": 8}}
        descriptor = {key: texture[key] for key in (
            "dimension", "array", "multisample", "scalar", "component_format",
            "normalized_coordinates", "comparison")}
        dependencies = [
            {"kind": "texture_descriptor", "index": 4, "byte_offset": 128,
             "component": 0, "origin": "derived", "value": descriptor},
            {"kind": "texture_array_length", "index": 3, "byte_offset": 0,
             "component": 0, "origin": "derived", "value": 17},
            {"kind": "texture_array_length", "index": 4, "byte_offset": 128,
             "component": 0, "origin": "derived", "value": 17}]
        stage = self.stage_v3(
            "v3-array-provenance", [sampler, scalar, texture],
            {"abi_version": 1, "buffer_index": 2, "byte_extent": 280, "alignment": 8},
            lambda value: value["dependencies"].extend(dependencies))
        output = self.root / "v3-array-provenance-out"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        reflected = json.loads((output / manifest["entries"][0]["directory"] /
                                "reflection.json").read_text())
        self.assertEqual(reflected["textures"], [sampler, scalar, texture])
        # A consumed array fact cannot contradict its reflected extent.
        metadata = json.loads((stage / "reflection.json").read_text())
        metadata["dependencies"][-1]["value"] = 16
        (stage / "reflection.json").write_text(canonical(metadata))
        self.failure(self.root / "v3-array-mismatch-out", (stage,), "array-length")

    def test_rejects_malformed_v3_resource_layouts_before_apple_tools(self):
        layout = {"abi_version": 1, "buffer_index": 2,
                  "byte_extent": 16, "alignment": 8}
        mutations = (
            ("misaligned", lambda value:
             value["textures"][1]["location"].update(byte_offset=9)),
            ("overlap", lambda value:
             value["textures"][1]["location"].update(byte_offset=0)),
            ("overflow", lambda value:
             value["textures"][1]["location"].update(byte_offset=0xfffffff8)),
            ("extent", lambda value: value["argument_buffer"].update(byte_extent=24)),
            ("count", lambda value: value["textures"][1]["location"].update(count=2)),
            ("provenance", lambda value:
             value["textures"][1].update(handle_byte_offset=2)),
            ("boolean-abi", lambda value: value["argument_buffer"].update(abi_version=True)),
            ("float-alignment", lambda value: value["argument_buffer"].update(alignment=8.0)),
            ("boolean-count", lambda value: value["textures"][1]["location"].update(count=True)),
            ("float-stride", lambda value: value["textures"][1]["location"].update(stride=8.0)),
            ("duplicate-id", lambda value: value["textures"][1].update(resource_id=0)),
            ("sampled-write", lambda value: value["textures"][1].update(access="write")),
            ("truncated-array", lambda value: (
                value["textures"][1].update(handle_byte_offset=65532, array_length=2),
                value["textures"][1]["location"].update(count=2),
                value["argument_buffer"].update(byte_extent=24))),
        )
        for index, (name, mutate) in enumerate(mutations):
            resources = [self.v3_texture(0, byte_offset=0),
                         self.v3_texture(1, byte_offset=8)]
            stage = self.stage_v3("malformed-v3-" + name, resources,
                                  dict(layout), mutate, seed=index + 20)
            self.failure(self.root / ("malformed-v3-out-" + name), (stage,),
                         "armigo-shader-build-package")

    def test_rejects_malformed_v2_reflections_before_apple_tools(self):
        # The builder validates well-formedness only; runtime admission (the
        # profile identity, float32x3 streams, admitted dependency kinds) is
        # the shader package loader's decision and is covered there.
        mutations = (
            ("texture", lambda value: value.update(textures=[
                {"cbuf_slot": 2, "handle_byte_offset": 16, "kind": "storage",
                 "dimension": "d2", "array": False, "multisample": False,
                 "scalar": "f32", "component_format": "floating",
                 "normalized_coordinates": True, "comparison": False,
                 "texture_index": 0}])),
            ("extent", lambda value: value["constant_buffers"][0].update(window_bytes=3)),
            ("c1", lambda value: value.update(c1_digest="zz")),
        )
        for name, mutate in mutations:
            stage = self.stage_v2("malformed-v2-" + name, mutate=mutate)
            self.failure(self.root / ("v2-out-" + name), (stage,),
                         "armigo-shader-build-package")
        self.assertFalse(self.log.exists())

    def class_input_v2(self, name, seed=1, mutate=None):
        """A synthetic v2 vertex stage consuming class-typed vertex inputs:
        the form the ported translator emits when no per-slot stream format
        is proven. Mirrors a minimal flat/perspective output interface."""
        def apply(value):
            value["vertex_streams"] = []
            value["vertex_inputs"] = [{"location": 0, "class": "float"}]
            value["interface"]["generic_inputs"] = [
                {"slot": 0, "component_mask": 15, "scalar": "f32",
                 "components": 4, "interpolation": "flat"}]
            value["interface"]["generic_outputs"] = [
                {"slot": 0, "component_mask": 15, "scalar": "f32",
                 "components": 4, "interpolation": "flat"},
                {"slot": 1, "component_mask": 3, "scalar": "f32",
                 "components": 4, "interpolation": "perspective"}]
            # Attribute-format facts describe declared streams; a class-input
            # stage keeps only its explicit options.
            value["dependencies"] = [item for item in value["dependencies"]
                                     if item["kind"] != "vertex_attribute_format"]
            if mutate is not None:
                mutate(value)
        return self.stage_v2(name, seed=seed, mutate=apply)

    def test_builds_widened_v2_vertex_interfaces(self):
        # Flat and perspective vertex outputs and class-typed vertex inputs
        # are inside the widened profile, and both stages build alongside the
        # linear-output form.
        stage = self.class_input_v2("widened-vertex", seed=3)
        linear = self.class_input_v2(
            "widened-vertex-linear", seed=5,
            mutate=lambda value: [item.update(interpolation="linear")
                                  for item in value["interface"]["generic_outputs"]])
        output = self.root / "widened"
        result = self.run_tool(output, (stage, linear))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["entries"]), 2)
        self.assertTrue(all(entry["stage"] == "vertex" for entry in manifest["entries"]))

    def test_builds_v2_vertex_stage_with_partial_position(self):
        # A vertex stage may declare only position X/Y/Z: the translation
        # supplies the hardware default W. No declared position still fails.
        partial = self.stage_v2(
            "partial-position", seed=7,
            mutate=lambda value: value["interface"].update(position_output_mask=7))
        output = self.root / "partial-position-out"
        result = self.run_tool(output, (partial,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["entries"]), 1)
        missing = self.stage_v2(
            "missing-position", seed=9,
            mutate=lambda value: value["interface"].update(position_output_mask=0))
        self.failure(self.root / "missing-position-out", (missing,),
                     "vertex interface is outside the supported profile")

    def test_builds_v2_vertex_stage_textures(self):
        # The draw backend binds vertex-stage texture/sampler pairs through
        # the same guest descriptor path as fragment stages, so the builder
        # keeps them and specializes on the consumed handle-buffer fact.
        def mutate(value):
            value["textures"] = [dict(self.TEXTURE_ENTRY), dict(self.SAMPLER_ENTRY)]
            value["dependencies"].append(
                {"kind": "texture_handle_cbuf", "index": 0, "byte_offset": 0,
                 "component": 0, "origin": "explicit_option", "value": 2})
        stage = self.stage_v2("vertex-textured", mutate=mutate)
        output = self.root / "vertex-textured"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["entries"][0]["stage"], "vertex")
        other = self.stage_v2("vertex-textured-slot", mutate=lambda value: (
            value.update(textures=[{**self.TEXTURE_ENTRY, "cbuf_slot": 4},
                                   {**self.SAMPLER_ENTRY, "cbuf_slot": 4}]),
            value["dependencies"].append(
                {"kind": "texture_handle_cbuf", "index": 0, "byte_offset": 0,
                 "component": 0, "origin": "explicit_option", "value": 4})))
        self.assertNotEqual((stage / "reflection.json").read_text(),
                            (other / "reflection.json").read_text())
        both = self.root / "vertex-textured-both"
        result = self.run_tool(both, (stage, other))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((both / "manifest.json").read_text())["entries"]), 2)

    def test_v2_colour_output_records(self):
        # Typed per-target colour records must rebuild the summary mask, stay
        # inside their declared components and restate the type their
        # fragment_output_type fact proved. Multi-target and integer-typed
        # reflections build; disagreements fail before any Apple tool runs.
        def output_fact(target, scalar="f32", components=4):
            return {"kind": "fragment_output_type", "index": target, "byte_offset": 0,
                    "component": 0, "origin": "derived",
                    "value": {"scalar": scalar, "components": components}}

        def sparse_mrt(value):
            value["interface"]["color_outputs"] = [
                {"location": 0, "component_mask": 15, "scalar": "f32", "components": 4},
                {"location": 2, "component_mask": 5, "scalar": "f32", "components": 4}]
            value["interface"]["color_output_mask"] = 15 | (5 << 8)
            value["dependencies"].append(output_fact(2))

        good = self.stage_v2("colour-mrt", stage="fragment", mutate=sparse_mrt)
        output = self.root / "colour-mrt"
        result = self.run_tool(output, (good,))
        self.assertEqual(result.returncode, 0, result.stderr)

        def unsigned(value):
            value["interface"]["color_outputs"] = [
                {"location": 0, "component_mask": 15, "scalar": "u32", "components": 4}]
            value["dependencies"][1] = output_fact(0, "u32")

        unsigned_stage = self.stage_v2("colour-uint", stage="fragment", mutate=unsigned)
        output = self.root / "colour-uint"
        result = self.run_tool(output, (unsigned_stage,))
        self.assertEqual(result.returncode, 0, result.stderr)

        mutations = (
            ("summary", "fragment", lambda v: v["interface"].update(color_output_mask=0xf0f)),
            ("duplicate", "fragment", lambda v: v["interface"]["color_outputs"].append(
                {"location": 0, "component_mask": 15, "scalar": "f32", "components": 4})),
            ("mask-outside", "fragment", lambda v: (
                v["interface"]["color_outputs"][0].update(components=2),
                v["dependencies"][1].update(value={"scalar": "f32", "components": 2}))),
            ("scalar", "fragment", lambda v: (
                v["interface"]["color_outputs"][0].update(scalar="f16"),
                v["dependencies"][1].update(value={"scalar": "f16", "components": 4}))),
            ("type-mismatch", "fragment", lambda v: v["dependencies"][1].update(
                value={"scalar": "u32", "components": 4})),
            ("missing-fact", "fragment", lambda v: v["dependencies"].pop(1)),
            ("extra-fact", "fragment", lambda v: v["dependencies"].append(output_fact(3))),
            ("depth-not-boolean",
             "fragment", lambda v: v["interface"].update(writes_fragment_depth=1)),
            ("vertex-colour", "vertex", lambda v: (
                v["interface"].update(color_output_mask=15),
                v["interface"]["color_outputs"].append(
                    {"location": 0, "component_mask": 15, "scalar": "f32",
                     "components": 4}),
                v["dependencies"].append(output_fact(0)))),
            ("vertex-depth-flag",
             "vertex", lambda v: v["interface"].update(writes_fragment_depth=True)),
        )
        for label, stage, mutate in mutations:
            broken = self.stage_v2("colour-" + label, stage=stage, mutate=mutate)
            self.failure(self.root / ("colour-out-" + label), (broken,),
                         "armigo-shader-build-package")

    def test_v2_adaptive_fragment_output_contract(self):
        def adaptive(value):
            value["fragment_output_mode"] = "render-target-v1"
            value["interface"]["color_outputs"][0].update(
                scalar="u32", components=4)
            value["dependencies"][1]["value"] = {"scalar": "u32", "components": 4}

        stage = self.stage_v2("adaptive-colour", stage="fragment", mutate=adaptive)
        output = self.root / "adaptive-colour-output"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["entries"]), 1)

        def adaptive_without_colours(value):
            value["fragment_output_mode"] = "render-target-v1"
            value["interface"].update(color_output_mask=0, color_outputs=[])
            value["dependencies"] = [item for item in value["dependencies"]
                                     if item["kind"] != "fragment_output_type"]

        empty_stage = self.stage_v2("adaptive-no-colour", stage="fragment",
                                    mutate=adaptive_without_colours)
        empty_output = self.root / "adaptive-no-colour-output"
        result = self.run_tool(empty_output, (empty_stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(
            (empty_output / "manifest.json").read_text())["entries"]), 1)

        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        adaptive_metadata = json.loads((stage / "reflection.json").read_text())
        static_metadata = json.loads(canonical(adaptive_metadata))
        static_metadata.pop("fragment_output_mode")
        self.assertNotEqual(module.variant_data(adaptive_metadata),
                            module.variant_data(static_metadata))

        mutations = (
            ("unknown-mode", "fragment", lambda v: (
                adaptive(v), v.update(fragment_output_mode="future-mode"))),
            ("null-mode", "fragment", lambda v:
                v.update(fragment_output_mode=None)),
            ("vertex-mode", "vertex", lambda v:
                v.update(fragment_output_mode="render-target-v1")),
            ("float-backing", "fragment", lambda v:
                v.update(fragment_output_mode="render-target-v1")),
            ("narrow-backing", "fragment", lambda v: (
                adaptive(v),
                v["interface"]["color_outputs"][0].update(component_mask=7, components=3),
                v["interface"].update(color_output_mask=7),
                v["dependencies"][1].update(
                    value={"scalar": "u32", "components": 3}))),
        )
        for label, shader_stage, mutate in mutations:
            broken = self.stage_v2("adaptive-" + label, stage=shader_stage,
                                   mutate=mutate)
            self.failure(self.root / ("adaptive-out-" + label), (broken,),
                         "armigo-shader-build-package")

    def test_v2_depth_bias_capability_survives_packaging_and_changes_variant(self):
        stage = self.stage_v2("depth-bias", stage="fragment", mutate=lambda value:
                              value.update(depth_bias_mode="shader-v1"))
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        loaded = module.load_stage(stage)
        self.assertEqual(loaded["metadata"]["depth_bias_mode"], "shader-v1")
        without_bias = json.loads(canonical(loaded["metadata"]))
        without_bias.pop("depth_bias_mode")
        self.assertNotEqual(module.variant_data(loaded["metadata"]),
                            module.variant_data(without_bias))

        output = self.root / "depth-bias-output"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["entries"]), 1)
        packaged = json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["depth_bias_mode"], "shader-v1")
        self.assertEqual(packaged["interface"], without_bias["interface"])

    def test_rejects_invalid_v2_depth_bias_capabilities_before_apple_tools(self):
        mutations = (
            ("unknown", "fragment", lambda value:
             value.update(depth_bias_mode="future-mode"), "depth bias mode is invalid"),
            ("null", "fragment", lambda value:
             value.update(depth_bias_mode=None), "depth bias mode is invalid"),
            ("vertex", "vertex", lambda value: None, "depth bias mode is invalid"),
            ("constant-collision", "fragment", lambda value:
             value["constant_buffers"][0].update(metal_buffer_index=30),
             "depth bias buffer overlaps"),
            ("support-collision", "fragment", lambda value:
             value["support_buffers"][0].update(metal_buffer_index=30),
             "depth bias buffer overlaps"),
            ("guest-depth", "fragment", lambda value:
             value["interface"].update(writes_fragment_depth=True),
             "cannot replace a guest fragment-depth export"),
        )
        for label, shader_stage, mutate, diagnostic in mutations:
            with self.subTest(label=label):
                def apply(value):
                    value["depth_bias_mode"] = "shader-v1"
                    mutate(value)
                stage = self.stage_v2("depth-bias-" + label,
                                      stage=shader_stage, mutate=apply)
                self.failure(self.root / ("depth-bias-output-" + label),
                             (stage,), diagnostic)
        self.assertFalse(self.log.exists())

    def test_rejects_malformed_v2_vertex_inputs(self):
        def base(name, stage, mutate):
            def apply(value):
                value["vertex_streams"] = []
                value["vertex_inputs"] = [{"location": 0, "class": "float"}]
                value["interface"]["generic_inputs"] = [
                    {"slot": 0, "component_mask": 15, "scalar": "f32",
                     "components": 4, "interpolation": "flat"}]
                value["dependencies"] = [item for item in value["dependencies"]
                                         if item["kind"] != "vertex_attribute_format"]
                mutate(value)
            return self.stage_v2(name, stage=stage, mutate=apply)
        mutations = (
            ("bad-class", "vertex", lambda v: v["vertex_inputs"][0].update(
                {"class": "signed"})),
            ("duplicate-location", "vertex", lambda v: v["vertex_inputs"].append(
                {"location": 0, "class": "float"})),
            ("fragment-inputs", "fragment", lambda v: None),
            ("mixed-streams", "vertex", lambda v: v["vertex_streams"].append(
                {"slot": 1, "attribute_index": 3, "format": "f32",
                 "components": 3, "byte_stride": 12})),
            ("uncovered-input", "vertex", lambda v: v["vertex_inputs"].append(
                {"location": 5, "class": "float"})),
        )
        for name, stage, mutate in mutations:
            broken = base("vertex-inputs-" + name, stage, mutate)
            self.failure(self.root / ("inputs-out-" + name), (broken,),
                         "armigo-shader-build-package")
        self.assertFalse(self.log.exists())

    def test_v2_runtime_window_and_support_validation(self):
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        stage = self.stage_v2("runtime-window-validation")
        original = json.loads((stage / "reflection.json").read_text())
        program = (stage / "program.bin").read_bytes()
        source = (stage / "source.metal").read_bytes()

        def validate(value):
            module.validate_reflection_v2(value, program, source)

        for extent in (4, 65536):
            value = json.loads(canonical(original))
            value["constant_buffers"][0].update(window_bytes=extent, metal_buffer_index=30)
            value["support_buffers"][0].update(metal_buffer_index=29)
            validate(value)
        empty = json.loads(canonical(original))
        empty.update(constant_buffers=[], support_buffers=[])
        validate(empty)
        overflow = json.loads(canonical(empty))
        overflow["vertex_streams"] = [
            {**original["vertex_streams"][0], "slot": index, "attribute_index": index}
            for index in range(14)]
        with self.assertRaisesRegex(RuntimeError, "exceed the Metal buffer namespace"):
            validate(overflow)
        mutations = [
            ("missing-support", lambda v: v.pop("support_buffers")),
            ("empty-support", lambda v: v.update(support_buffers=[])),
            ("extra-support", lambda v: v["support_buffers"].append(v["support_buffers"][0])),
            ("orphan-support", lambda v: v.update(constant_buffers=[])),
            ("cb-stream-collision", lambda v: v["constant_buffers"][0].update(
                metal_buffer_index=18)),
            ("support-stream-collision", lambda v: v["support_buffers"][0].update(
                metal_buffer_index=18)),
            ("duplicate-cb", lambda v: v["constant_buffers"].append(v["constant_buffers"][0])),
            ("duplicate-metal", lambda v: v["constant_buffers"].append(
                {**v["constant_buffers"][0], "guest_binding": 4})),
        ]
        for field, bad_values in {
                "guest_binding": (-1, 18, True),
                "window_bytes": (0, 3, 6, 65540, True, 4.0),
                "metal_buffer_index": (-1, 31, True),
                "alignment": (1, 8, True),
                "size_kind": ("accessed_extent", None),
                "policy": ("none", None),
                "byte_extent": (64,),
        }.items():
            for bad in bad_values:
                mutations.append((f"cb-{field}-{bad}", lambda v, f=field, b=bad:
                                  v["constant_buffers"][0].update({f: b})))
        for field, bad_values in {
                "schema": ("unknown", None), "version": (0, 2, True),
                "byte_extent": (68, 76, True), "alignment": (1, 8, True),
                "metal_buffer_index": (0, -1, 31, True), "extra": (0,),
        }.items():
            for bad in bad_values:
                mutations.append((f"support-{field}-{bad}", lambda v, f=field, b=bad:
                                  v["support_buffers"][0].update({f: b})))
        for name, mutate in mutations:
            with self.subTest(name=name):
                value = json.loads(canonical(original))
                mutate(value)
                with self.assertRaises(RuntimeError):
                    validate(value)

    def test_v2_window_and_support_metadata_affect_variant_identity(self):
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        stage = self.stage_v2("runtime-window-identity")
        original = json.loads((stage / "reflection.json").read_text())
        identity = canonical(module.variant_data(original))
        for section, field, replacement in (
                ("constant_buffers", "guest_binding", 4),
                ("constant_buffers", "window_bytes", 128),
                ("constant_buffers", "metal_buffer_index", 30),
                ("support_buffers", "metal_buffer_index", 29)):
            with self.subTest(section=section, field=field):
                value = json.loads(canonical(original))
                value[section][0][field] = replacement
                self.assertNotEqual(canonical(module.variant_data(value)), identity)
        self.assertEqual(module.variant_data(original)["constant_buffers"],
                         original["constant_buffers"])
        self.assertEqual(module.variant_data(original)["support_buffers"],
                         original["support_buffers"])

    def textured_v2(self, name, textures, dependency=2):
        """A synthetic v2 fragment stage whose bound texture and sampler share
        one handle word at constant buffer 2, offset 32."""
        def mutate(value):
            value["textures"] = textures
            if dependency is not None:
                value["dependencies"].append(
                    {"kind": "texture_handle_cbuf", "index": 0, "byte_offset": 0,
                     "component": 0, "origin": "explicit_option", "value": dependency})
        return self.stage_v2(name, "fragment", 176, 11, mutate=mutate)


    TEXTURE_ENTRY = {"cbuf_slot": 2, "handle_byte_offset": 32, "kind": "texture",
                     "dimension": "d2", "array": False, "multisample": False,
                     "scalar": "f32", "component_format": "unorm",
                     "normalized_coordinates": True, "comparison": False,
                     "texture_index": 3}
    SAMPLER_ENTRY = {"cbuf_slot": 2, "handle_byte_offset": 32, "kind": "sampler",
                     "comparison": False, "normalized_coordinates": True,
                     "sampler_index": 1}

    def test_builds_v2_bound_texture_with_shared_handle_word(self):
        # One handle word binds both halves: the texture and its sampler carry
        # the same provenance, and the consumed handle-buffer fact records the
        # slot. The builder must keep the pair and specialize on the fact.
        stage = self.textured_v2("ported-texture",
                                 [dict(self.TEXTURE_ENTRY), dict(self.SAMPLER_ENTRY)])
        output = self.root / "textured"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["entries"]), 1)
        packaged = json.loads((output / "stages" / manifest["entries"][0]["directory"].split("/")[1] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["textures"], [dict(self.TEXTURE_ENTRY),
                                                dict(self.SAMPLER_ENTRY)])
        other = self.textured_v2("ported-texture-slot", [
            {**self.TEXTURE_ENTRY, "cbuf_slot": 4},
            {**self.SAMPLER_ENTRY, "cbuf_slot": 4}], dependency=4)
        self.assertNotEqual((stage / "reflection.json").read_text(),
                            (other / "reflection.json").read_text())
        both = self.root / "both-slots"
        result = self.run_tool(both, (stage, other))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((both / "manifest.json").read_text())["entries"]), 2)

    def test_rejects_malformed_v2_texture_entries_before_apple_tools(self):
        mutations = (
            ("duplicate-index", [dict(self.TEXTURE_ENTRY),
                                 {**self.TEXTURE_ENTRY, "handle_byte_offset": 36},
                                 {**self.SAMPLER_ENTRY},
                                 {**self.SAMPLER_ENTRY, "handle_byte_offset": 36,
                                  "sampler_index": 2}]),
            ("duplicate-sampler-index", [dict(self.TEXTURE_ENTRY),
                                         {**self.TEXTURE_ENTRY, "handle_byte_offset": 36},
                                         dict(self.SAMPLER_ENTRY),
                                         {**self.SAMPLER_ENTRY, "handle_byte_offset": 36}]),
            ("unaligned-handle", [{**self.TEXTURE_ENTRY, "handle_byte_offset": 34},
                                  {**self.SAMPLER_ENTRY, "handle_byte_offset": 34}]),
            ("unknown-kind", [{"cbuf_slot": 2, "handle_byte_offset": 32,
                               "kind": "storage"}]),
            ("bad-slot", [{**self.TEXTURE_ENTRY, "cbuf_slot": 18},
                          {**self.SAMPLER_ENTRY, "cbuf_slot": 18}]),
        )
        for name, textures in mutations:
            stage = self.textured_v2("malformed-texture-" + name, textures)
            self.failure(self.root / ("texture-out-" + name), (stage,),
                         "armigo-shader-build-package")
        self.assertFalse(self.log.exists())

    DESCRIPTOR_FACT = {"kind": "texture_descriptor", "index": 2, "byte_offset": 32,
                       "component": 0, "origin": "derived",
                       "value": {"dimension": "d2", "array": False,
                                 "multisample": False, "scalar": "f32",
                                 "component_format": "unorm",
                                 "normalized_coordinates": True,
                                 "comparison": False}}

    def descriptor_stage(self, name, slot=2, offset=32, texture_index=3):
        """A v2 fragment stage with a bound texture pair and the agreeing
        derived descriptor fact at one handle word."""
        def mutate(value):
            value["textures"].extend([
                {**self.TEXTURE_ENTRY, "cbuf_slot": slot,
                 "handle_byte_offset": offset, "texture_index": texture_index},
                {**self.SAMPLER_ENTRY, "cbuf_slot": slot,
                 "handle_byte_offset": offset}])
            value["dependencies"].extend([
                {"kind": "texture_handle_cbuf", "index": 0, "byte_offset": 0,
                 "component": 0, "origin": "explicit_option", "value": slot},
                {**self.DESCRIPTOR_FACT, "index": slot, "byte_offset": offset}])
        return self.stage_v2(name, "fragment", 176, 11, mutate=mutate)

    def test_builds_v2_texture_descriptor_dependency_and_specializes(self):
        # The derived descriptor fact travels as a consumed dependency naming
        # the bound handle it was proven from, and the fact key/value
        # participates in the variant identity.
        stage = self.descriptor_stage("descriptor")
        output = self.root / "descriptor"
        result = self.run_tool(output, (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["entries"]), 1)
        packaged = json.loads((output / "stages" /
                               manifest["entries"][0]["directory"].split("/")[1] /
                               "reflection.json").read_text())
        self.assertIn(dict(self.DESCRIPTOR_FACT), packaged["dependencies"])
        other = self.descriptor_stage("descriptor-other", slot=4, offset=36,
                                      texture_index=4)
        self.assertNotEqual((stage / "reflection.json").read_text(),
                            (other / "reflection.json").read_text())
        both = self.root / "both-descriptors"
        result = self.run_tool(both, (stage, other))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((both / "manifest.json").read_text())["entries"]), 2)

    def test_builds_v2_depth_compare_descriptor_dependency(self):
        # A depth-compare fact carries the depth format class and agrees with
        # a comparison texture/sampler pair at the same handle word.
        def mutate(value):
            value["textures"].extend([
                {**self.TEXTURE_ENTRY, "component_format": "depth", "comparison": True},
                {**self.SAMPLER_ENTRY, "comparison": True}])
            value["dependencies"].extend([
                {"kind": "texture_handle_cbuf", "index": 0, "byte_offset": 0,
                 "component": 0, "origin": "explicit_option", "value": 2},
                {**self.DESCRIPTOR_FACT,
                 "value": {**self.DESCRIPTOR_FACT["value"],
                           "component_format": "depth", "comparison": True}}])
        stage = self.stage_v2("depth-descriptor", "fragment", 176, 11, mutate=mutate)
        result = self.run_tool(self.root / "depth-descriptor-out", (stage,))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_builds_widened_and_deferred_v2_texture_descriptor_profiles(self):
        profiles = (
            ("d1", {"dimension": "d1"}, True),
            ("d1-array", {"dimension": "d1", "array": True}, True),
            ("d2-array-depth", {"array": True, "component_format": "depth",
                                "comparison": True}, True),
            ("d3", {"dimension": "d3", "component_format": "snorm"}, True),
            ("cube", {"dimension": "cube", "component_format": "floating"}, True),
            ("cube-array", {"dimension": "cube", "array": True,
                            "component_format": "depth", "comparison": True}, True),
            ("uint-fetch", {"scalar": "u32",
                            "component_format": "unsigned_integer"}, False),
            ("sint-fetch", {"dimension": "d3", "scalar": "s32",
                            "component_format": "signed_integer"}, False),
            ("rectangle-fetch", {"normalized_coordinates": False}, False),
            # The generic offline schema preserves profiles that runtime L4
            # still rejects, allowing deferred package entries to diagnose at lookup.
            ("multisample-deferred", {"multisample": True}, False),
            ("buffer-deferred", {"dimension": "buffer"}, False),
        )
        stages = []
        for index, (name, changes, sampled) in enumerate(profiles):
            texture = {**self.TEXTURE_ENTRY, **changes}
            sampler = {**self.SAMPLER_ENTRY,
                       "comparison": texture["comparison"],
                       "normalized_coordinates": texture["normalized_coordinates"]}
            descriptor = {name: texture[name] for name in
                          ("dimension", "array", "multisample", "scalar",
                           "component_format", "normalized_coordinates", "comparison")}

            def mutate(value, texture=texture, sampler=sampler,
                       descriptor=descriptor, sampled=sampled):
                value["textures"].append(texture)
                if sampled:
                    value["textures"].append(sampler)
                value["dependencies"].extend([
                    {"kind": "texture_handle_cbuf", "index": 0, "byte_offset": 0,
                     "component": 0, "origin": "explicit_option", "value": 2},
                    {**self.DESCRIPTOR_FACT, "value": descriptor}])

            stages.append(self.stage_v2("widened-" + name, "fragment", 176,
                                        20 + index, mutate=mutate))
        output = self.root / "widened-descriptors"
        result = self.run_tool(output, stages)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((output / "manifest.json").read_text())["entries"]),
                         len(profiles))

    def test_rejects_malformed_v2_texture_descriptor_dependencies(self):
        def stage_with(name, dependency):
            def mutate(value):
                value["textures"].extend([dict(self.TEXTURE_ENTRY),
                                          dict(self.SAMPLER_ENTRY)])
                value["dependencies"].extend([
                    {"kind": "texture_handle_cbuf", "index": 0, "byte_offset": 0,
                     "component": 0, "origin": "explicit_option", "value": 2},
                    dependency])
            return self.stage_v2(name, "fragment", 176, 11, mutate=mutate)
        mutations = (
            ("unbound-handle", {**self.DESCRIPTOR_FACT, "byte_offset": 36}),
            ("slot-mismatch", {**self.DESCRIPTOR_FACT, "index": 4}),
            ("disagreeing-shape", {**self.DESCRIPTOR_FACT,
                                   "value": {**self.DESCRIPTOR_FACT["value"],
                                             "dimension": "cube"}}),
            ("disagreeing-array", {**self.DESCRIPTOR_FACT,
                                   "value": {**self.DESCRIPTOR_FACT["value"],
                                             "array": True}}),
            ("disagreeing-comparison", {**self.DESCRIPTOR_FACT,
                                        "value": {**self.DESCRIPTOR_FACT["value"],
                                                  "component_format": "depth",
                                                  "comparison": True}}),
            ("depth-without-comparison", {**self.DESCRIPTOR_FACT,
                                          "value": {**self.DESCRIPTOR_FACT["value"],
                                                    "component_format": "depth"}}),
            ("integer-scalar", {**self.DESCRIPTOR_FACT,
                                "value": {**self.DESCRIPTOR_FACT["value"],
                                          "scalar": "u32",
                                          "component_format": "unsigned_integer"}}),
            ("disagreeing-normalized", {**self.DESCRIPTOR_FACT,
                                        "value": {**self.DESCRIPTOR_FACT["value"],
                                                  "normalized_coordinates": False}}),
            ("disagreeing-format", {**self.DESCRIPTOR_FACT,
                                    "value": {**self.DESCRIPTOR_FACT["value"],
                                              "component_format": "floating"}}),
            ("multisample", {**self.DESCRIPTOR_FACT,
                             "value": {**self.DESCRIPTOR_FACT["value"],
                                       "multisample": True}}),
            ("unaligned-offset", {**self.DESCRIPTOR_FACT, "byte_offset": 34}),
            ("bad-component", {**self.DESCRIPTOR_FACT, "component": 2}),
            ("missing-value-field", {**self.DESCRIPTOR_FACT,
                                     "value": {key: item for key, item in
                                               self.DESCRIPTOR_FACT["value"].items()
                                               if key != "comparison"}}),
        )
        for name, dependency in mutations:
            stage = stage_with("malformed-descriptor-" + name, dependency)
            self.failure(self.root / ("descriptor-out-" + name), (stage,),
                         "armigo-shader-build-package")
        self.assertFalse(self.log.exists())

    def test_atomic_image_flag_is_optional_and_restricted(self):
        def packaged(name, resources, seed):
            stage = self.stage_v3(name, resources, stage="fragment", seed=seed)
            output = self.root / (name + "-out")
            result = self.run_tool(output, (stage,))
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            return json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())

        unsigned = packaged("atomic-u32",
                            [self.atomic_image("u32", "unsigned_integer", atomic=True)], 41)
        self.assertIs(unsigned["textures"][0]["atomic"], True)
        signed = packaged("atomic-s32",
                          [self.atomic_image("s32", "signed_integer", atomic=True)], 42)
        self.assertIs(signed["textures"][0]["atomic"], True)
        absent = packaged("atomic-absent",
                          [self.atomic_image("u32", "unsigned_integer")], 43)
        self.assertNotIn("atomic", absent["textures"][0])

        sampled = self.v3_texture(0, first=0)
        sampled["atomic"] = True
        self.failure(self.root / "atomic-sampled-out",
                     (self.stage_v3("atomic-sampled", [sampled], stage="fragment", seed=44),),
                     "texture atomic flag is invalid")
        floating = self.atomic_image("f32", "floating", atomic=True)
        self.failure(self.root / "atomic-float-out",
                     (self.stage_v3("atomic-float", [floating], stage="fragment", seed=45),),
                     "texture atomic flag is invalid")
        denied = self.atomic_image("u32", "unsigned_integer", atomic=False)
        self.failure(self.root / "atomic-false-out",
                     (self.stage_v3("atomic-false", [denied], stage="fragment", seed=46),),
                     "texture atomic flag is invalid")

    def run_tool(self, output, stages, *extra, environment=None):
        command = [sys.executable, str(SCRIPT), "--translate-tool", str(self.translate),
                   "--output", str(output), "--jobs", "2"]
        for stage in stages:
            command += ["--stage", str(stage)]
        command += list(extra)
        return subprocess.run(command, capture_output=True, text=True, timeout=15,
                              env=environment or self.environment)

    def failure(self, output, stages, text, environment=None):
        result = self.run_tool(output, stages, environment=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(text, result.stderr)
        self.assertTrue(not output.exists() or not any(output.iterdir()))
        self.assertFalse(list(output.parent.glob(".armigo-package-*")))
        return result

    def test_deterministic_manifest_commands_and_artifacts(self):
        vertex = self.stage("vertex")
        fragment = self.stage("fragment", "fragment", 176, 9)
        outputs = [self.root / "package-a", self.root / "package-b"]
        for output, order in zip(outputs, ((vertex, fragment), (fragment, vertex))):
            result = self.run_tool(output, order)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((outputs[0] / "manifest.json").read_bytes(),
                         (outputs[1] / "manifest.json").read_bytes())
        manifest = json.loads((outputs[0] / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "armigo-metal-package-v1")
        self.assertEqual(manifest["source_version"], SOURCE_VERSION)
        self.assertFalse(manifest["complete_inventory"])
        self.assertFalse(manifest["runnable"])
        self.assertEqual(manifest["platform"], "macos-metal4")
        self.assertEqual(manifest["entries"], sorted(manifest["entries"],
            key=lambda item: (item["stage"], item["program_key"], item["variant_id"])))
        profile = json.loads(manifest["profile_text"])
        self.assertEqual(canonical(profile), manifest["profile_text"])
        self.assertEqual(profile["source_version"], manifest["source_version"])
        self.assertEqual(profile["platform"], manifest["platform"])
        self.assertNotIn(str(self.root), manifest["profile_text"])
        expected_chain = hashlib.sha256(
            b"armigo-metal-chain-v1\n" + manifest["profile_text"].encode()).hexdigest()
        self.assertEqual(manifest["chain_id"], expected_chain)
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        chain_module = importlib.import_module("chain")
        header = self.root / "expected_chain.h"
        chain_module.emit_expected_header(manifest["profile_text"], header)
        self.assertIn(expected_chain, header.read_text())
        for entry in manifest["entries"]:
            directory = outputs[0] / entry["directory"]
            self.assertEqual(sorted(path.name for path in directory.iterdir()),
                             ["program.bin", "reflection.json", "shader.metallib"])
            library = (directory / "shader.metallib").read_bytes()
            self.assertEqual(hashlib.sha256(library).hexdigest(), entry["library_digest"])
            self.assertEqual(len(library), entry["library_size"])
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        metal = [item for item in commands if item[2] == "metal"]
        self.assertTrue(metal)
        self.assertEqual(metal[0][3:11], ["-std=metal4.0", "-mmacosx-version-min=26.0", "-fwrapv",
                                          "-Werror", "-Wno-unused-variable",
                                          "-Wno-sometimes-uninitialized",
                                          "-fbracket-depth=1024", "-c"])

    def test_optional_compile_cache_reuses_identical_sources(self):
        # With the developer cache enabled, an identical source compiled by the
        # same tools is not recompiled, and the packaged library is identical.
        stage = self.stage("cached")
        environment = dict(self.environment)
        environment["ARMIGO_SHADER_COMPILE_CACHE"] = str(self.root / "compile-cache")
        first = self.root / "cache-first"
        result = self.run_tool(first, (stage,), environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        compiled = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertTrue(any(item[2] == "metal" for item in compiled))
        self.assertEqual(len(list((self.root / "compile-cache").glob("*.metallib"))), 1)
        self.log.unlink()
        second = self.root / "cache-second"
        result = self.run_tool(second, (stage,), environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        recompiled = [json.loads(line) for line in self.log.read_text().splitlines()] \
            if self.log.exists() else []
        self.assertFalse(any(item[2] in ("metal", "metallib") for item in recompiled))
        self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())

    def test_deduplicates_identical_variant_and_rejects_conflicts_and_prefixes(self):
        stage = self.stage("one")
        output = self.root / "dedup"
        result = self.run_tool(output, (stage, stage))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads((output / "manifest.json").read_text())["entries"]), 1)
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(sum(item[2] == "metal" for item in commands), 1)

        conflict = self.stage("conflict")
        metadata = json.loads((conflict / "reflection.json").read_text())
        program = (stage / "program.bin").read_bytes()
        (conflict / "program.bin").write_bytes(program)
        source = (conflict / "source.metal").read_bytes()
        metadata["program_key"] = hashlib.sha256(program).hexdigest()
        metadata["program_size"] = len(program)
        metadata["source_digest"] = hashlib.sha256(source).hexdigest()
        (conflict / "reflection.json").write_text(canonical(metadata) + "\n")
        self.failure(self.root / "conflict-out", (stage, conflict), "conflicting contents")

        shorter = self.stage("shorter", size=128, seed=17)
        longer = self.stage("longer", size=144, seed=18)
        short_bytes = (shorter / "program.bin").read_bytes()
        long_bytes = short_bytes + bytes(16)
        (longer / "program.bin").write_bytes(long_bytes)
        long_meta = json.loads((longer / "reflection.json").read_text())
        long_meta["program_key"] = hashlib.sha256(long_bytes).hexdigest()
        long_meta["program_size"] = len(long_bytes)
        (longer / "reflection.json").write_text(canonical(long_meta) + "\n")
        self.failure(self.root / "prefix-out", (shorter, longer), "byte prefix")

    def test_rejects_hash_stage_version_and_output_boundaries(self):
        stage = self.stage("invalid")
        metadata = json.loads((stage / "reflection.json").read_text())
        metadata["program_key"] = "0" * 64
        (stage / "reflection.json").write_text(canonical(metadata) + "\n")
        self.failure(self.root / "bad-hash", (stage,), "program identity")

        stage = self.stage("mismatch", "fragment")
        program = bytearray((stage / "program.bin").read_bytes())
        struct.pack_into("<I", program, 0, 1 << 10)
        (stage / "program.bin").write_bytes(program)
        metadata = json.loads((stage / "reflection.json").read_text())
        metadata["program_key"] = hashlib.sha256(program).hexdigest()
        (stage / "reflection.json").write_text(canonical(metadata) + "\n")
        self.failure(self.root / "bad-stage", (stage,), "SPH stage")

        good = self.stage("good")
        self.failure(good / "nested", (good,), "input directory")
        repository = self.root / "repository"
        repository.mkdir()
        (repository / ".git").write_text("gitdir: synthetic\n")
        self.failure(repository / "package", (good,), "Git work tree")
        alias = self.root / "repository-alias"
        alias.symlink_to(repository, target_is_directory=True)
        self.failure(alias / "package", (good,), "Git work tree")

        wrong_translate = self.bin / "wrong-translate"
        wrong_translate.write_text("#!/bin/sh\necho " + "4b" * 32 + "\n")
        wrong_translate.chmod(0o755)
        original = self.translate
        self.translate = wrong_translate
        self.failure(self.root / "version-out", (good,), "source version")
        self.translate = original

    def test_rejects_source_digest_short_program_options_and_large_bucket(self):
        changed_source = self.stage("changed-source")
        (changed_source / "source.metal").write_bytes(b"different validated source\n")
        self.failure(self.root / "source-out", (changed_source,), "source digest")

        short = self.stage("short", size=111)
        self.failure(self.root / "short-out", (short,), "identity prefix")

        options = self.stage("options")
        metadata = json.loads((options / "reflection.json").read_text())
        metadata["vertex_depth"] = "unspecified"
        (options / "reflection.json").write_text(canonical(metadata) + "\n")
        self.failure(self.root / "options-out", (options,), "explicit options")

        bucket = []
        base = None
        for index in range(65):
            item = self.stage(f"bucket-{index}", size=160, seed=71)
            program = bytearray((item / "program.bin").read_bytes())
            if base is None:
                base = bytes(program[:112])
            program[:112] = base
            program[159] = index
            (item / "program.bin").write_bytes(program)
            metadata = json.loads((item / "reflection.json").read_text())
            metadata["program_key"] = hashlib.sha256(program).hexdigest()
            (item / "reflection.json").write_text(canonical(metadata) + "\n")
            bucket.append(item)
        self.failure(self.root / "bucket-out", bucket, "prefix bucket")

    def test_rejects_malformed_nested_reflection_before_apple_tools(self):
        mutations = (
            ("guest", lambda value: value.update(constant_buffers=[
                {"guest_binding": 99, "minimum_size": 4, "metal_buffer_index": 0}])),
            ("minimum", lambda value: value.update(constant_buffers=[
                {"guest_binding": 0, "minimum_size": -4, "metal_buffer_index": 0}])),
            ("metal", lambda value: value.update(constant_buffers=[
                {"guest_binding": 0, "minimum_size": 4, "metal_buffer_index": 99}])),
            ("handle", lambda value: value.update(textures=[
                {"handle_byte_offset": 3, "texture_index": 0, "sampler_index": 0}])),
            ("interface", lambda value: value.update(interface={})),
        )
        for name, mutate in mutations:
            stage = self.stage("malformed-" + name)
            metadata = json.loads((stage / "reflection.json").read_text())
            mutate(metadata)
            (stage / "reflection.json").write_text(canonical(metadata) + "\n")
            self.failure(self.root / ("malformed-out-" + name), (stage,),
                         "armigo-shader-build-package")
        self.assertFalse(self.log.exists())

    def test_canonical_input_dedup_and_patched_aggregate_budgets(self):
        stage = self.stage("bounded")
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        original_read = module.read_regular
        with mock.patch.object(module, "read_regular", wraps=original_read) as reads:
            with mock.patch.dict(os.environ, self.environment, clear=True):
                result = module.main(["--translate-tool", str(self.translate), "--output",
                                      str(self.root / "dedup-read"), "--stage", str(stage),
                                      "--stage", str(stage.resolve()), "--jobs", "1"])
        self.assertEqual(result, 0)
        self.assertEqual(reads.call_count, 3)

        input_size = sum((stage / name).stat().st_size
                         for name in ("program.bin", "source.metal", "reflection.json"))
        with mock.patch.object(module, "MAX_INPUT_BYTES", input_size - 1):
            with mock.patch.dict(os.environ, self.environment, clear=True):
                result = module.main(["--translate-tool", str(self.translate), "--output",
                                      str(self.root / "input-budget"), "--stage", str(stage)])
        self.assertEqual(result, 1)
        self.assertFalse((self.root / "input-budget").exists())

        static_size = (stage / "program.bin").stat().st_size + (stage / "reflection.json").stat().st_size
        with mock.patch.object(module, "MAX_OUTPUT_ARTIFACT_BYTES", static_size + 1):
            with mock.patch.dict(os.environ, self.environment, clear=True):
                result = module.main(["--translate-tool", str(self.translate), "--output",
                                      str(self.root / "artifact-budget"), "--stage", str(stage),
                                      "--jobs", "1"])
        self.assertEqual(result, 1)
        self.assertFalse((self.root / "artifact-budget").exists())
        self.assertFalse(list(self.root.glob(".armigo-package-*")))

    def test_admits_handle_program_slot_and_refuses_a_bad_program(self):
        # Slot 31 names program (handle_byte_offset / 4). The binding's
        # array length selects that many consecutive programs. A descriptor
        # fact at the same slot needs the one program at its own offset.
        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")

        def bytes_of(stage):
            return ((stage / "program.bin").read_bytes(),
                    (stage / "source.metal").read_bytes())

        def accept(stage):
            metadata = json.loads((stage / "reflection.json").read_text())
            program, source = bytes_of(stage)
            module.validate_reflection(metadata, program, source)
            return metadata, program, source

        def v2(value):
            value["handle_programs"] = [[0xffffffff, 0]]
            value["textures"] = [
                {**self.TEXTURE_ENTRY, "cbuf_slot": 31, "handle_byte_offset": 0},
                {**self.SAMPLER_ENTRY, "cbuf_slot": 31, "handle_byte_offset": 0}]
            value["dependencies"].append(
                {**self.DESCRIPTOR_FACT, "index": 31, "byte_offset": 0})

        v2_stage = self.stage_v2("handle-v2", "fragment", 176, 31, mutate=v2)
        metadata, program, source = accept(v2_stage)
        output = self.root / "handle-v2-out"
        result = self.run_tool(output, (v2_stage,))
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        packaged = json.loads((output / manifest["entries"][0]["directory"] /
                               "reflection.json").read_text())
        self.assertEqual(packaged["handle_programs"], [[0xffffffff, 0]])
        self.assertEqual(packaged["textures"][0]["cbuf_slot"], 31)
        bare = json.loads(canonical(metadata))
        bare.pop("handle_programs")
        self.assertNotEqual(module.variant_data(metadata), module.variant_data(bare))

        texture = self.v3_texture(0, first=0, array_length=2)
        texture.update(cbuf_slot=31, handle_byte_offset=0)
        sampler = {"kind": "sampler", "resource_id": 1, "array_length": 1,
                   "cbuf_slot": 31, "handle_byte_offset": 0,
                   "comparison": False, "normalized_coordinates": True,
                   "location": {"kind": "direct", "namespace": "sampler",
                                "first": 0, "count": 1}}

        def v3(value):
            value["handle_programs"] = [[1, 2, 3], []]
            value["dependencies"].append({
                "kind": "texture_descriptor", "index": 31, "byte_offset": 0,
                "component": 0, "origin": "derived",
                "value": {"dimension": "d2", "array": False, "multisample": False,
                          "scalar": "f32", "component_format": "floating",
                          "normalized_coordinates": True, "comparison": False}})

        v3_stage = self.stage_v3("handle-v3", [texture, sampler], mutate=v3)
        v3_metadata, v3_program, v3_source = accept(v3_stage)
        v3_output = self.root / "handle-v3-out"
        result = self.run_tool(v3_output, (v3_stage,))
        self.assertEqual(result.returncode, 0, result.stderr)

        # The last of 4096 programs, and a program of 4096 words, are in range.
        # An unused program list does not disturb a constant-buffer binding.
        last = json.loads(canonical(metadata))
        last["handle_programs"] = [[] for _ in range(4096)]
        for item in last["textures"]:
            item["handle_byte_offset"] = 16380
        for item in last["dependencies"]:
            if item["kind"] == "texture_descriptor":
                item["byte_offset"] = 16380
        module.validate_reflection(last, program, source)
        words = json.loads(canonical(metadata))
        words["handle_programs"] = [list(range(4096))]
        module.validate_reflection(words, program, source)
        unused = self.stage_v2("handle-unused", mutate=lambda value:
                               value.update(handle_programs=[[1]]))
        accept(unused)

        def refuse(base, program_bytes, source_bytes, message, mutate):
            broken = json.loads(canonical(base))
            mutate(broken)
            with self.assertRaisesRegex(RuntimeError, message):
                module.validate_reflection(broken, program_bytes, source_bytes)

        def shift_handles(value, offset):
            for item in value["textures"]:
                item["handle_byte_offset"] = offset

        refuse(metadata, program, source, "texture provenance",
               lambda value: shift_handles(value, 4))
        refuse(metadata, program, source, "texture provenance",
               lambda value: value["textures"][1].update(handle_byte_offset=4))
        refuse(metadata, program, source, "texture provenance",
               lambda value: shift_handles(value, 2))
        refuse(metadata, program, source, "texture provenance",
               lambda value: value.pop("handle_programs"))
        refuse(metadata, program, source, "texture provenance",
               lambda value: value.update(handle_programs=[]))
        refuse(metadata, program, source, "texture provenance",
               lambda value: value["textures"][0].update(cbuf_slot=18))
        refuse(metadata, program, source, "texture provenance",
               lambda value: value["textures"][0].update(cbuf_slot=30))
        def descriptor(value):
            return next(item for item in value["dependencies"]
                        if item["kind"] == "texture_descriptor")

        refuse(metadata, program, source, "texture descriptor dependency key",
               lambda value: descriptor(value).update(byte_offset=4))
        refuse(metadata, program, source, "texture descriptor dependency key",
               lambda value: descriptor(value).update(index=18))
        refuse(v3_metadata, v3_program, v3_source, "texture provenance",
               lambda value: value.update(handle_programs=[[1]]))
        refuse(v3_metadata, v3_program, v3_source, "storage-buffer reflection is invalid",
               lambda value: value.update(storage_buffers=[self.v3_storage(cbuf_slot=31)]))
        refuse(metadata, program, source, "constant-buffer reflection is invalid",
               lambda value: value["constant_buffers"][0].update(guest_binding=31))
        refuse(metadata, program, source, "handle_programs must be an array",
               lambda value: value.update(handle_programs={}))
        refuse(metadata, program, source, "handle_programs must be an array",
               lambda value: value.update(handle_programs="words"))
        refuse(metadata, program, source, "handle program must be an array",
               lambda value: value.update(handle_programs=[1]))
        refuse(metadata, program, source, "handle program word is invalid",
               lambda value: value.update(handle_programs=[[1.5]]))
        refuse(metadata, program, source, "handle program word is invalid",
               lambda value: value.update(handle_programs=[[True]]))
        refuse(metadata, program, source, "handle program word is invalid",
               lambda value: value.update(handle_programs=[["1"]]))
        refuse(metadata, program, source, "handle program word is invalid",
               lambda value: value.update(handle_programs=[[-1]]))
        refuse(metadata, program, source, "handle program word is invalid",
               lambda value: value.update(handle_programs=[[2 ** 32]]))
        refuse(metadata, program, source, "handle_programs exceeds its count policy",
               lambda value: value.update(handle_programs=[[] for _ in range(4097)]))
        refuse(metadata, program, source, "handle program exceeds its count policy",
               lambda value: value.update(handle_programs=[list(range(4097))]))
        legacy = self.stage("handle-v1")
        legacy_metadata = json.loads((legacy / "reflection.json").read_text())
        legacy_metadata["handle_programs"] = [[1]]
        with self.assertRaisesRegex(RuntimeError, "unexpected schema"):
            module.validate_reflection(legacy_metadata, *bytes_of(legacy))

    def test_tool_failure_timeout_and_cleanup(self):
        stage = self.stage("tool")
        failed_environment = self.environment.copy()
        failed_environment["FAKE_FAIL"] = "metal"
        self.failure(self.root / "failed-tool", (stage,), "tool failed", failed_environment)

        sys.path.insert(0, str(TOOLS))
        self.addCleanup(lambda: sys.path.remove(str(TOOLS)))
        module = importlib.import_module("build_package")
        previous = module.COMMAND_TIMEOUT_SECONDS
        module.COMMAND_TIMEOUT_SECONDS = 0.05
        self.addCleanup(setattr, module, "COMMAND_TIMEOUT_SECONDS", previous)
        sleepy = self.environment.copy()
        sleepy["FAKE_SLEEP"] = "metal"
        with mock.patch.dict(os.environ, sleepy, clear=True):
            result = module.main(["--translate-tool", str(self.translate), "--output",
                                  str(self.root / "timeout"), "--stage", str(stage),
                                  "--jobs", "1"])
        self.assertEqual(result, 1)
        self.assertFalse(list(self.root.glob(".armigo-package-*")))


if __name__ == "__main__":
    unittest.main()
