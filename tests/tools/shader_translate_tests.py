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

"""CLI checks using independently assembled synthetic instruction streams."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = str(Path(sys.argv.pop(1)).resolve())


def field(word, start, width, value):
    mask = ((1 << width) - 1) << start
    return (word & ~mask) | ((value << start) & mask)


def instruction(opcode):
    return (opcode << 48) | (7 << 16)


def literal(register, value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return field(field(field(instruction(0x0100), 0, 8, register), 12, 4, 15), 20, 32, bits)


def attribute(opcode, register, component):
    word = field(instruction(opcode), 0, 8, register)
    word = field(word, 8, 8, 255)
    word = field(word, 20, 10, component * 4)
    return field(word, 39, 8, 255)


def interpolate(register, component):
    word = field(instruction(0xe000), 0, 8, register)
    word = field(word, 8, 8, 255)
    word = field(word, 20, 8, 255)
    word = field(word, 30, 8, component)
    return field(word, 39, 11, 0x7ff)


def multiply(register, lhs, rhs):
    word = field(instruction(0x5c68), 0, 8, register)
    word = field(word, 8, 8, lhs)
    word = field(word, 20, 8, rhs)
    word = field(word, 39, 2, 3)
    return field(word, 44, 2, 1)


def containers(stage=0, unknown=False, two_attributes=False, word_transform=None):
    sph = bytearray(0x50)
    struct.pack_into("<I", sph, 0, (1 if stage == 0 else 5) << 10)
    if stage == 0:
        sph[0x18] = 7
        sph[0x35] = 0xf0
        sph[0x36] = 3
        words = [attribute(0xefd8, reg, 31 + reg) for reg in (1, 2, 3)]
        add = field(instruction(0x4c58), 0, 8, 1)
        add = field(add, 8, 8, 1)
        add = field(add, 20, 14, 4)
        add = field(add, 34, 5, 3)
        words += [field(add, 44, 1, 1), literal(4, 1.0)]
        words += [attribute(0xeff0, reg, 27 + reg) for reg in (1, 2, 3, 4)]
        words += [attribute(0xeff0, 1, 32), attribute(0xeff0, 2, 33)]
        if two_attributes:
            sph[0x18] |= 1 << 4
            sph[0x36] |= 4
            words += [attribute(0xefd8, 5, 36), attribute(0xeff0, 5, 34)]
    else:
        sph[0x17] = 0x80
        sph[0x18] = 0x0a
        struct.pack_into("<I", sph, 0x48, 15)
        words = [interpolate(4, 32), interpolate(5, 33), interpolate(6, 31)]
        reciprocal = field(field(instruction(0x5080), 0, 8, 7), 8, 8, 6)
        words += [field(reciprocal, 20, 4, 4), multiply(8, 4, 7), multiply(9, 5, 7)]
        texture = field(instruction(0xd000), 0, 8, 0)
        for start, width, value in ((8, 8, 8), (20, 8, 9), (28, 8, 2),
                                    (36, 13, 8), (49, 1, 1), (50, 3, 4),
                                    (53, 4, 1), (59, 1, 1)):
            texture = field(texture, start, width, value)
        words += [texture]
    words += [field(instruction(0xe300), 0, 5, 15)]
    if unknown:
        words[0] = instruction(0x1234)
    if word_transform is not None:
        words = word_transform(words)
    stream = bytearray()
    for index, word in enumerate(words):
        if index % 3 == 0:
            stream += bytes(8)
        stream += struct.pack("<Q", word)
    c1_offset = 0x80 + len(stream)
    code = bytearray(0x30) + sph + stream + bytes([0x6d] * 16)
    struct.pack_into("<I", code, 0, 0x12345678)
    control = bytearray(0x718)
    for offset, value in ((0, 0x98761234), (4, 1), (8, 14), (0x20, len(control)),
                          (0x700, c1_offset), (0x704, len(code)), (0x708, 0x30),
                          (0x714, stage)):
        struct.pack_into("<I", control, offset, value)
    return bytes(control), bytes(code), bytes(code[0x30:c1_offset])


def guarded_vertex_words(_):
    words = [attribute(0xefd8, reg, 31 + reg) for reg in (1, 2, 3)]
    compare = field(instruction(0x5b60), 0, 3, 1)
    for start, width, value in ((8, 8, 1), (20, 8, 255), (39, 3, 7), (49, 3, 5)):
        compare = field(compare, start, width, value)
    words += [compare, field(literal(1, 0.25), 16, 4, 0),
              field(literal(2, 0.75), 16, 4, 9), literal(4, 1.0)]
    words += [attribute(0xeff0, reg, 27 + reg) for reg in (1, 2, 3, 4)]
    words += [attribute(0xeff0, 1, 32), attribute(0xeff0, 2, 33)]
    return words + [field(instruction(0xe300), 0, 5, 15)]


def branch_word(displacement):
    return field(field(instruction(0xe240), 0, 5, 15), 20, 24,
                 displacement & 0xffffff)


def pattern_word(pattern, fields):
    bits = 0
    for index, character in enumerate(pattern):
        if character == '1':
            bits |= 1 << (63 - index)
    bits |= 7 << 16  # P7, the always-true predicate every encoding carries.
    for start, width, value in fields:
        bits = field(bits, start, width, value)
    return bits


def ported_attribute_load(register, byte_offset, words=1):
    # ALD dest, srcA, offset(bytes), size(words-1); pipeline resource encoding.
    return pattern_word("1110111111011x",
                        ((0, 8, register), (8, 8, 255), (20, 11, byte_offset),
                         (39, 8, 255), (47, 2, words - 1)))


def ported_interpolate(register, byte_offset):
    # IPA dest, offset(bytes); the SPH interpolation map supplies the mode.
    # Sources B/C and the instruction predicate stay at RZ/P7: the encoder
    # normalizes them away and SSA must not see phantom R0/P0 reads.
    return pattern_word("11100000000000",
                        ((0, 8, register), (8, 8, 255), (20, 8, 255),
                         (28, 10, byte_offset), (39, 8, 255), (47, 3, 7),
                         (52, 2, 0), (54, 2, 1)))


def ported_attribute_store(register, byte_offset, words=1):
    # AST value(D slot), offset(bytes), size(words-1).
    return pattern_word("1110111111110x",
                        ((0, 8, register), (8, 8, 255), (20, 11, byte_offset),
                         (39, 8, 255), (47, 2, words - 1)))


def ported_exit():
    return pattern_word("111000110000", ((0, 4, 15),))


def ported_texs(register, ra, rb, handle_word=8, target=2):
    # TEXS dest, Ra(x), Rb(y); selected TEXS target, two components, bound
    # handle at handle_word*4 bytes. The encoding carries no constant-buffer
    # slot, so emission consumes the texture_handle_cbuf fact for the handle
    # buffer. Target 2 is 2D LodZero; 8 is 2D-array LodZero; 4 is 2D
    # depth-compare.
    return pattern_word("1101100xxxxxxx",
                        ((0, 8, register), (8, 8, ra), (20, 8, rb), (28, 8, 255),
                         (36, 13, handle_word), (50, 3, 4), (53, 4, target)))


def ported_depth_texture(register, ra, rb, gather=False):
    # Bound 2D TEX/TLD4 with depth comparison and all four result components.
    return pattern_word("110010xxxxxxxx" if gather else "11000xxxxxxxxx",
                        ((0, 8, register), (8, 8, ra), (20, 8, rb),
                         (28, 3, 2), (31, 4, 15), (36, 13, 8),
                         (50, 1, 1), (51, 3, 7)))


def ported_txq(register, ra, query=1, mask=3, handle_word=8):
    # TXQ size query (selector 1) with the selected component mask; the LOD
    # input comes from Ra and the handle carries no Rb operand.
    return pattern_word("1101111101001x",
                        ((0, 8, register), (8, 8, ra), (22, 6, query),
                         (31, 4, mask), (36, 13, handle_word)))


def ported_tlds(register, ra, rb, handle_word=8, target=6, mask=4):
    # TLDS texel fetch; target 6 is the 2D multisample fetch with its sample
    # index in Rb. The compact mask selector 4 selects three components.
    return pattern_word("1101101xxxxxxx",
                        ((0, 8, register), (8, 8, ra), (20, 8, rb), (28, 8, 255),
                         (36, 13, handle_word), (50, 3, mask), (53, 4, target)))


def ported_position_store(register):
    # AST of four words at output offset 0x70: the vertex position export.
    return ported_attribute_store(register, 0x70, words=4)


def ported_containers(stage, words, interpolation_map=0, export_mask=0,
                      local_memory=(0, 0, 0), extra=0):
    sph = bytearray(0x50)
    struct.pack_into("<I", sph, 0, (1 if stage == 0 else 5) << 10)
    for offset, value in zip((4, 8, 12), local_memory):
        struct.pack_into("<I", sph, offset, value)
    if isinstance(interpolation_map, int):
        sph[0x18] = interpolation_map
    else:
        sph[0x18:0x18 + len(interpolation_map)] = bytes(interpolation_map)
    # The fragment target mask exports four consecutive registers per active
    # target, so mask 3 exports the first TEXS result registers R0/R1.
    struct.pack_into("<I", sph, 0x48, export_mask)
    # Bit 0 of the extra word exports the sample mask, bit 1 the fragment
    # depth, after the colour targets' registers.
    struct.pack_into("<I", sph, 0x4c, extra)
    stream = bytearray()
    for index, word in enumerate(words):
        if index % 3 == 0:
            stream += bytes(8)
        stream += struct.pack("<Q", word)
    c1_offset = 0x80 + len(stream)
    code = bytearray(0x30) + sph + stream + bytes([0x6d] * 16)
    struct.pack_into("<I", code, 0, 0x12345678)
    control = bytearray(0x718)
    for offset, value in ((0, 0x98761234), (4, 1), (8, 14), (0x20, len(control)),
                          (0x700, c1_offset), (0x704, len(code)), (0x708, 0x30),
                          (0x714, stage)):
        struct.pack_into("<I", control, offset, value)
    return bytes(control), bytes(code), bytes(code[0x30:c1_offset])


class TranslationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="armigo-translate-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.control = self.inputs / 'control.bin'
        self.code = self.inputs / 'code.bin'
        self.install()

    def install(self, stage=0, unknown=False):
        self.control_bytes, self.code_bytes, self.program = containers(stage, unknown)
        self.control.write_bytes(self.control_bytes)
        self.code.write_bytes(self.code_bytes)

    def run_tool(self, output, *options, control=None, code=None):
        return subprocess.run([EXECUTABLE, '--control', str(control or self.control),
                               '--code', str(code or self.code), '--output', str(output), *options],
                              capture_output=True, text=True, timeout=30)

    def vertex_options(self):
        return ('--vertex-float3', '0:2', '--vertex-depth', 'minus-one-to-one')

    def check_failure(self, output, *options, contains=None, **paths):
        result = self.run_tool(output, *options, **paths)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        if contains:
            self.assertIn(contains, result.stderr)
        self.assertTrue(not output.exists() or not any(output.iterdir()))
        self.assertFalse(list(output.parent.glob('.armigo-stage-*')))
        return result

    def test_vertex_reflection_identity_and_determinism(self):
        result_dir = self.root / 'first'
        second_dir = self.root / 'second'
        for output in (result_dir, second_dir):
            result = self.run_tool(output, *self.vertex_options())
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(p.name for p in result_dir.iterdir()),
                         ['program.bin', 'reflection.json', 'source.metal'])
        for path in result_dir.iterdir():
            self.assertEqual(path.read_bytes(), (second_dir / path.name).read_bytes())
        metadata = json.loads((result_dir / 'reflection.json').read_text())
        self.assertEqual(metadata['schema'], 'armigo-shader-stage-v1')
        version = subprocess.run([EXECUTABLE, '--version'], check=True, capture_output=True,
                                 text=True, timeout=5).stdout.strip()
        self.assertEqual(metadata['source_version'], version)
        self.assertEqual(metadata['stage'], 'vertex')
        self.assertEqual(metadata['entry_point'], 'armigo_vertex_main')
        self.assertEqual(metadata['vertex_depth'], 'minus-one-to-one')
        self.assertEqual(metadata['vertex_attributes'],
                         [{'generic_slot': 0, 'attribute_index': 2, 'format': 'float32x3'}])
        self.assertEqual(metadata['constant_buffers'],
                         [{'guest_binding': 3, 'minimum_size': 20, 'metal_buffer_index': 0}])
        self.assertEqual(metadata['textures'], [])
        self.assertEqual(metadata['interface']['position_output_mask'], 15)
        self.assertEqual(metadata['interface']['generic_inputs'][0]['component_mask'], 7)
        self.assertEqual(metadata['interface']['generic_outputs'][0]['component_mask'], 3)
        self.assertEqual(metadata['program_file'], 'program.bin')
        self.assertEqual(metadata['program_size'], len(self.program))
        self.assertEqual(metadata['program_key'], hashlib.sha256(self.program).hexdigest())
        self.assertEqual((result_dir / 'program.bin').read_bytes(), self.program)
        self.assertEqual(metadata['source_file'], 'source.metal')
        self.assertEqual(metadata['source_digest'],
                         hashlib.sha256((result_dir / 'source.metal').read_bytes()).hexdigest())
        self.assertNotIn(str(self.root), (result_dir / 'reflection.json').read_text())
        self.assertEqual(self.control.read_bytes(), self.control_bytes)
        self.assertEqual(self.code.read_bytes(), self.code_bytes)

    def test_fragment_explicit_position_and_resources(self):
        self.install(stage=1)
        output = self.root / 'fragment'
        result = self.run_tool(output, '--fragment-position', 'reciprocal-clip-w')
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['stage'], 'fragment')
        self.assertEqual(metadata['entry_point'], 'armigo_fragment_main')
        self.assertEqual(metadata['vertex_depth'], 'unspecified')
        self.assertEqual(metadata['fragment_position'], 'reciprocal-clip-w')
        self.assertEqual(metadata['interface']['position_input_mask'], 8)
        self.assertEqual(metadata['interface']['color_output_mask'], 15)
        self.assertEqual(metadata['interface']['generic_inputs'],
                         [{'slot': 0, 'component_mask': 3, 'interpolation': 'perspective'}])
        self.assertEqual(metadata['textures'],
                         [{'handle_byte_offset': 32, 'texture_index': 0, 'sampler_index': 0}])
        self.assertEqual((output / 'program.bin').read_bytes(), self.program)
        self.assertIn('.sample(', (output / 'source.metal').read_text())
        self.check_failure(self.root / 'missing-position', contains='emit code=')
        self.check_failure(self.root / 'wrong-depth', '--vertex-depth', 'zero-to-one',
                           contains='not fragment-stage')
        self.check_failure(self.root / 'wrong-attribute', '--vertex-float3', '0:0',
                           contains='not fragment-stage')

    def test_guarded_vertex_uses_forward_lowering_and_original_identity(self):
        control, code, program = containers(word_transform=guarded_vertex_words)
        self.control.write_bytes(control)
        self.code.write_bytes(code)
        output = self.root / 'guarded'
        result = self.run_tool(output, *self.vertex_options())
        self.assertEqual(result.returncode, 0, result.stderr)
        source = (output / 'source.metal').read_text()
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertIn('if (a0)', source)
        self.assertIn('!=', source)
        self.assertEqual(metadata['program_key'], hashlib.sha256(program).hexdigest())
        self.assertEqual((output / 'program.bin').read_bytes(), program)
        self.assertEqual(metadata['interface']['generic_inputs'],
                         [{'slot': 0, 'component_mask': 7, 'interpolation': 'screen-linear'}])

    def test_forward_branch_prunes_dead_words_without_normalizing_identity(self):
        keys = []
        sources = []
        # Raw targets 32 (a scheduler word) and 40 both normalize to PC 40.
        # Their input identities must still differ.
        for displacement in (16, 24):
            def words(original):
                return [branch_word(displacement), instruction(0x1234),
                        instruction(0x1234)] + original
            control, code, program = containers(word_transform=words)
            self.control.write_bytes(control)
            self.code.write_bytes(code)
            output = self.root / ('branch-' + str(displacement))
            result = self.run_tool(output, *self.vertex_options())
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output / 'reflection.json').read_text())
            self.assertEqual(metadata['program_key'], hashlib.sha256(program).hexdigest())
            self.assertEqual((output / 'program.bin').read_bytes(), program)
            keys.append(metadata['program_key'])
            sources.append((output / 'source.metal').read_bytes())
        self.assertNotEqual(*keys)
        self.assertEqual(*sources)

    def test_forward_failures_publish_no_partial_stage(self):
        control, code, _ = containers(word_transform=lambda words:
                                     [branch_word(-16)] + words)
        self.control.write_bytes(control)
        self.code.write_bytes(code)
        self.check_failure(self.root / 'backedge', *self.vertex_options(),
                           contains='decode code=')
        control, code, _ = containers(word_transform=lambda words:
                                     [field(instruction(0x1234), 16, 4, 15)] + words)
        self.control.write_bytes(control)
        self.code.write_bytes(code)
        self.check_failure(self.root / 'false-unknown', *self.vertex_options(),
                           contains='pc=8')
        # A skipped EXIT forces the graph route; implicit-LOD sampling in graph
        # programs is emitted under the documented host-derivative policy.
        control, code, _ = containers(stage=1, word_transform=lambda words:
            [field(field(instruction(0xe300), 0, 5, 15), 16, 4, 15)] + words)
        self.control.write_bytes(control)
        self.code.write_bytes(code)
        result = self.run_tool(self.root / 'graph-texture', '--fragment-position',
                               'reciprocal-clip-w')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'graph-texture' / 'source.metal').is_file())
        self.assertIn('.sample(', (self.root / 'graph-texture' / 'source.metal').read_text())

    def test_vertex_options_are_required_and_stage_specific(self):
        self.check_failure(self.root / 'missing-depth', '--vertex-float3', '0:2',
                           contains='emit code=')
        self.check_failure(self.root / 'missing-format', '--vertex-depth', 'zero-to-one',
                           contains='emit code=')
        self.check_failure(self.root / 'duplicate-format', *self.vertex_options(),
                           '--vertex-float3', '0:2', contains='emit code=')
        self.check_failure(self.root / 'wrong-stage', *self.vertex_options(),
                           '--fragment-position', 'reciprocal-clip-w', contains='not a vertex-stage')
        for invalid in ('-1:0', '32:0', '0:31', '0:0:1', ':0', '0:x'):
            self.check_failure(self.root / 'invalid-option', '--vertex-float3', invalid)
        output = self.root / 'zero-depth'
        result = self.run_tool(output, '--vertex-float3', '0:2', '--vertex-depth', 'zero-to-one')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / 'reflection.json').read_text())['vertex_depth'], 'zero-to-one')

    def test_malformed_and_unsupported_inputs_have_no_output(self):
        self.install(unknown=True)
        result = self.check_failure(self.root / 'unsupported', *self.vertex_options(), contains='decode code=')
        self.assertIn('pc=8', result.stderr)
        self.assertIn('field=', result.stderr)
        self.install()
        self.control.write_bytes(self.control_bytes[:20])
        self.check_failure(self.root / 'malformed', *self.vertex_options(), contains='parse code=')
        self.install()
        self.code.write_bytes(self.code_bytes[:0x40])
        empty = self.root / 'empty-on-error'
        empty.mkdir()
        self.check_failure(empty, *self.vertex_options(), contains='parse code=')
        self.install()
        undeclared = bytearray(self.code_bytes)
        undeclared[0x30 + 0x18] = 0
        self.code.write_bytes(undeclared)
        self.check_failure(self.root / 'lower-failure', *self.vertex_options(), contains='lower code=')

    def test_repeated_vertex_options_are_order_independent(self):
        control, code, _ = containers(two_attributes=True)
        self.control.write_bytes(control)
        self.code.write_bytes(code)
        left, right = self.root / 'left', self.root / 'right'
        for output, formats in ((left, ('0:2', '1:3')), (right, ('1:3', '0:2'))):
            result = self.run_tool(output, '--vertex-float3', formats[0],
                                   '--vertex-float3', formats[1], '--vertex-depth', 'zero-to-one')
            self.assertEqual(result.returncode, 0, result.stderr)
        for filename in ('source.metal', 'reflection.json', 'program.bin'):
            self.assertEqual((left / filename).read_bytes(), (right / filename).read_bytes())

    def test_input_limits_checked_before_reading(self):
        self.control.write_bytes(bytes(65537))
        self.check_failure(self.root / 'large-control', *self.vertex_options(), contains='size limit')
        self.install()
        with self.code.open('wb') as stream:
            stream.truncate(512 * 1024 * 1024 + 1)
        self.check_failure(self.root / 'large-code', *self.vertex_options(), contains='size limit')

    def test_new_or_empty_output_and_preservation(self):
        empty = self.root / 'empty'
        empty.mkdir()
        result = self.run_tool(empty, *self.vertex_options())
        self.assertEqual(result.returncode, 0, result.stderr)
        before = {p.name: p.read_bytes() for p in empty.iterdir()}
        result = self.run_tool(empty, *self.vertex_options())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('new or empty', result.stderr)
        self.assertEqual(before, {p.name: p.read_bytes() for p in empty.iterdir()})

    def test_output_git_markers_and_symlinks(self):
        for marker_is_file in (False, True):
            worktree = self.root / ('linked' if marker_is_file else 'repo')
            worktree.mkdir()
            marker = worktree / '.git'
            if marker_is_file:
                marker.write_text('gitdir: /unavailable/synthetic/worktree\n')
            else:
                marker.mkdir()
            self.check_failure(worktree / 'deep' / 'output', *self.vertex_options(), contains='Git work tree')
            alias = self.root / ('linked-alias' if marker_is_file else 'repo-alias')
            alias.symlink_to(worktree, target_is_directory=True)
            self.check_failure(alias / 'output', *self.vertex_options(), contains='Git work tree')

    def test_output_input_directory_overlap_and_aliases(self):
        self.check_failure(self.inputs / 'output', *self.vertex_options(), contains='input directory')
        alias = self.root / 'input-alias'
        alias.symlink_to(self.inputs, target_is_directory=True)
        self.check_failure(alias / 'output', *self.vertex_options(), contains='input directory')
        spelled = self.root / 'spelled-inputs'
        spelled.mkdir()
        link = spelled / 'control.bin'
        link.symlink_to(self.control)
        self.check_failure(spelled / 'output', *self.vertex_options(), control=link,
                           contains='input directory')

    def test_program_key_excludes_control_wrapper_and_c1(self):
        first = self.root / 'original'
        self.assertEqual(self.run_tool(first, *self.vertex_options()).returncode, 0)
        changed = bytearray(self.code_bytes)
        changed[0x20:0x30] = bytes([0x91] * 16)
        changed[-16:] = bytes([0xe3] * 16)
        self.code.write_bytes(changed)
        second = self.root / 'changed'
        result = self.run_tool(second, *self.vertex_options())
        self.assertEqual(result.returncode, 0, result.stderr)
        for filename in ('program.bin', 'reflection.json', 'source.metal'):
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())


class PortedTranslationTests(unittest.TestCase):
    """--translator ported: structured per-phase diagnostics on synthetic
    containers built from the ported pipeline's resource encodings."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="armigo-translate-ported-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.vertex_options = ('--vertex-float3', '0:2', '--vertex-float3', '1:3',
                               '--vertex-depth', 'minus-one-to-one')

    def run_ported(self, stage, words, *options, interpolation_map=0, export_mask=0,
                   local_memory=(0, 0, 0), extra=0):
        control, code, _ = ported_containers(stage, words, interpolation_map, export_mask,
                                             local_memory, extra)
        directory = self.root / 'in'
        directory.mkdir(exist_ok=True)
        (directory / 'control.bin').write_bytes(control)
        (directory / 'code.bin').write_bytes(code)
        output = self.root / ('out-%d' % len(list(self.root.glob('out-*'))))
        result = subprocess.run([EXECUTABLE, '--control', str(directory / 'control.bin'),
                                 '--code', str(directory / 'code.bin'),
                                 '--output', str(output), '--translator', 'ported',
                                 *options], capture_output=True, text=True, timeout=30)
        return result, output

    def run_arguments(self, *options):
        # A raw run over a valid container for option-parsing failures, which
        # abort before any input is read.
        directory = self.root / 'in'
        directory.mkdir(exist_ok=True)
        control, code, _ = ported_containers(0, [ported_exit()])
        (directory / 'control.bin').write_bytes(control)
        (directory / 'code.bin').write_bytes(code)
        output = self.root / ('out-%d' % len(list(self.root.glob('out-*'))))
        result = subprocess.run([EXECUTABLE, '--control', str(directory / 'control.bin'),
                                 '--code', str(directory / 'code.bin'),
                                 '--output', str(output), *options],
                                capture_output=True, text=True, timeout=30)
        return result, output

    def test_storage_descriptor_reflection_matches_generated_binding(self):
        # Two runtime descriptor words feed one extended four-word load.
        words = [pattern_word("1110111110010",
                              ((0, 8, reg), (8, 8, 255), (20, 16, offset),
                               (36, 5, 2), (48, 3, 4)))
                 for reg, offset in ((10, 16), (11, 20))]
        words += [pattern_word("1110111011010",
                               ((0, 8, 0), (8, 8, 10), (45, 1, 1), (48, 3, 6))),
                  ported_position_store(0), ported_exit()]
        result, output = self.run_ported(0, words, *self.vertex_options)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['storage_buffers'], [{
            'cbuf_slot': 2, 'descriptor_byte_offset': 16,
            'metal_buffer_index': 1, 'access': 'read', 'alignment': 4}])
        window = next(item for item in metadata['constant_buffers']
                      if item['guest_binding'] == 2)
        self.assertGreaterEqual(window['window_bytes'], 28)
        source = (output / 'source.metal').read_text()
        self.assertIn('const device uint* resource', source)
        self.assertIn('[[buffer(1)]]', source)
        self.assertIn('storage_base', source)
        self.assertIn('storage_size', source)

    def test_translator_option_is_explicit_and_default_stays_current(self):
        # The default translator keeps emitting v1 stages; --translator must be
        # an explicit, validated selection.
        control, code, _ = containers()
        directory = self.root / 'in'
        directory.mkdir()
        (directory / 'control.bin').write_bytes(control)
        (directory / 'code.bin').write_bytes(code)
        for arguments in ([], ['--translator', 'current']):
            output = self.root / ('current-%s' % '-'.join(arguments or ['default']))
            result = subprocess.run([EXECUTABLE, '--control', str(directory / 'control.bin'),
                                     '--code', str(directory / 'code.bin'),
                                     '--output', str(output), *arguments,
                                     '--vertex-float3', '0:2', '--vertex-depth',
                                     'minus-one-to-one'],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output / 'reflection.json').read_text())
            self.assertEqual(metadata['schema'], 'armigo-shader-stage-v1')
        for invalid in ('fast', 'CURRENT', ''):
            output = self.root / ('invalid-%s' % invalid)
            result = subprocess.run([EXECUTABLE, '--control', str(directory / 'control.bin'),
                                     '--code', str(directory / 'code.bin'),
                                     '--output', str(output),
                                     '--vertex-float3', '0:2', '--vertex-depth',
                                     'minus-one-to-one', '--translator', invalid],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
        duplicate = self.run_ported(0, [ported_exit()], *self.vertex_options,
                                    '--translator', 'ported')
        self.assertIn('Duplicate --translator', duplicate[0].stderr)

    def test_ported_failures_publish_no_partial_stage(self):
        # Malformed container: the parse diagnostic keeps the census format.
        control, code, _ = ported_containers(0, [ported_exit()])
        directory = self.root / 'truncated'
        directory.mkdir()
        (directory / 'control.bin').write_bytes(control[:20])
        (directory / 'code.bin').write_bytes(code)
        output = self.root / 'parse'
        result = subprocess.run([EXECUTABLE, '--control', str(directory / 'control.bin'),
                                 '--code', str(directory / 'code.bin'),
                                 '--output', str(output), '--translator', 'ported',
                                 *self.vertex_options],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('parse code=', result.stderr)
        self.assertFalse(output.exists())

        # Unknown instruction encoding: decode rejects instead of dropping.
        result, output = self.run_ported(0, [pattern_word("1111000011111", ((0, 4, 15),)),
                                             ported_exit()],
                                         *self.vertex_options)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('decode code=', result.stderr)
        self.assertIn('pc=8', result.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(list(output.parent.glob('.armigo-stage-*')))

    def test_ported_missing_facts_are_structured_diagnostics(self):
        # A fragment interpolation without its SPH mode stays Unknown and must
        # surface as a missing-fact diagnostic, never a default.
        result, output = self.run_ported(1, [ported_interpolate(4, 0x80), ported_exit()])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('emit_ir code=', result.stderr)
        self.assertIn('fact=', result.stderr)
        self.assertFalse(output.exists())
        # A vertex stage that exports a position needs a depth convention.
        result, output = self.run_ported(0, [ported_position_store(0), ported_exit()],
                                         '--vertex-float3', '0:2', '--vertex-float3', '1:3')
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output.exists())

    def test_ported_constant_buffer_windows_and_support_reflection(self):
        # A synthetic U32 LDC writes the sole exported color component. No
        # captured binding state is available to this translation.
        ldc = pattern_word("1110111110010x",
                           ((0, 8, 0), (8, 8, 255), (20, 16, 12),
                            (36, 5, 17), (48, 3, 4)))
        result, output = self.run_ported(1, [ldc, ported_exit()], export_mask=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['schema'], 'armigo-shader-stage-v3')
        self.assertEqual(metadata['profile'], {'identity': 'armigo-macos-metal4', 'version': 3})
        self.assertIsNone(metadata['argument_buffer'])
        self.assertEqual(metadata['constant_buffers'], [{
            'guest_binding': 17, 'size_kind': 'constant_buffer_window',
            'window_bytes': 16, 'alignment': 4,
            'policy': 'runtime_zero_words_v1', 'metal_buffer_index': 0}])
        self.assertEqual(metadata['support_buffers'], [{
            'schema': 'armigo-cb-sizes', 'version': 1, 'byte_extent': 72,
            'alignment': 4, 'metal_buffer_index': 1}])
        self.assertFalse(any(item['kind'] in ('constant_buffer_size',
                                             'constant_buffer_bound_mask')
                             for item in metadata['dependencies']))
        self.assertNotIn('byte_extent', metadata['constant_buffers'][0])

    def test_ported_fragment_depth_bias_reserves_binding_and_preserves_guest_io(self):
        ldc = pattern_word("1110111110010x",
                           ((0, 8, 0), (8, 8, 255), (20, 16, 12),
                            (36, 5, 17), (48, 3, 4)))
        result, output = self.run_ported(1, [ldc, ported_exit()], export_mask=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['depth_bias_mode'], 'shader-v1')
        self.assertEqual(metadata['interface']['position_input_mask'], 0)
        self.assertFalse(metadata['interface']['writes_fragment_depth'])
        buffer_indices = [item['metal_buffer_index'] for key in
                          ('constant_buffers', 'support_buffers')
                          for item in metadata[key]]
        self.assertNotIn(30, buffer_indices)
        self.assertEqual(len(buffer_indices), len(set(buffer_indices)))
        source = (output / 'source.metal').read_text()
        self.assertIn('constant float4& armigo_depth_bias [[buffer(30)]]', source)
        self.assertIn('float4 armigo_position [[position]]', source)
        self.assertIn('constant bool armigo_has_depth [[function_constant(8)]];', source)
        self.assertIn('float armigo_depth [[depth(any), function_constant(armigo_has_depth)]]', source)
        self.assertLess(source.index('if (armigo_has_depth) {'),
                        source.index('out.armigo_depth ='))
        self.assertIn('dfdx(armigo_z)', source)
        self.assertIn('dfdy(armigo_z)', source)
        for index in buffer_indices:
            self.assertIn(f'[[buffer({index})]]', source)

    def test_ported_guest_depth_export_does_not_claim_depth_bias_capability(self):
        result, output = self.run_ported(
            1, [ported_interpolate(1, 0x80), ported_exit()],
            interpolation_map=1, extra=2)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertTrue(metadata['interface']['writes_fragment_depth'])
        self.assertNotIn('depth_bias_mode', metadata)
        self.assertNotIn('armigo_depth_bias', (output / 'source.metal').read_text())

    def test_ported_local_memory_uses_sph_sizes_for_ldl_and_stl(self):
        # LDL/STL use the validated SPH sizes: low + high + CRS/32 bytes per
        # thread, matching the guest warp accounting rule.
        ldl = pattern_word("1110111101000x",
                           ((0, 8, 0), (8, 8, 255), (20, 24, 4), (48, 3, 4)))
        stl = pattern_word("1110111101010x",
                           ((0, 8, 255), (8, 8, 255), (20, 24, 8), (48, 3, 4)))
        result, output = self.run_ported(
            1, [ldl, stl, ported_exit()], export_mask=1,
            local_memory=(4, 4, 128))
        # The allocation fact reaches legalization and sizes the emitted
        # per-thread array: 128 bytes become 32 zero-initialized words.
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertTrue(any(entry['kind'] == 'local_memory_bytes' and
                            entry['value'] == 12 and entry['index'] == 0
                            for entry in metadata['dependencies']),
                        metadata['dependencies'])
        source = (output / 'source.metal').read_text()
        self.assertIn('thread uint armigo_local_words[3] = {};', source)

    def test_ported_stage_identity_excludes_c1(self):
        # Captures of one program with different trailing C1 bytes must yield
        # one package variant, so the stage publishes identical files.
        words = [ported_interpolate(0, 0x80), ported_exit()]
        control, code, _ = ported_containers(1, words, 1, 1, (0, 0, 0), 0)
        outputs = []
        for index, tail in enumerate((0x6d, 0xe3)):
            directory = self.root / ('c1-%d' % index)
            directory.mkdir()
            changed = bytearray(code)
            changed[-16:] = bytes([tail] * 16)
            (directory / 'control.bin').write_bytes(control)
            (directory / 'code.bin').write_bytes(changed)
            output = self.root / ('c1-out-%d' % index)
            result = subprocess.run([EXECUTABLE, '--control', str(directory / 'control.bin'),
                                     '--code', str(directory / 'code.bin'),
                                     '--output', str(output), '--translator', 'ported'],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(output)
        for filename in ('program.bin', 'reflection.json', 'source.metal'):
            self.assertEqual((outputs[0] / filename).read_bytes(),
                             (outputs[1] / filename).read_bytes())
        self.assertIsNone(json.loads((outputs[0] / 'reflection.json').read_text())['c1_digest'])

    def test_ported_stage_without_constant_buffers_has_no_support_buffer(self):
        result, output = self.run_ported(1, [ported_interpolate(0, 0x80), ported_exit()],
                                         interpolation_map=1, export_mask=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['constant_buffers'], [])
        self.assertEqual(metadata['support_buffers'], [])

    def test_ported_colour_mask_packs_one_nibble_per_target(self):
        # RT0 plus RT2 packs to 0x00000f0f, each target's component mask in its
        # own nibble, with one typed record and one fragment_output_type fact
        # per written target.
        words = [ported_interpolate(0, 0x80), ported_interpolate(1, 0x84),
                 ported_interpolate(2, 0x88), ported_interpolate(3, 0x8c),
                 ported_interpolate(4, 0x80), ported_interpolate(5, 0x84),
                 ported_interpolate(6, 0x88), ported_interpolate(7, 0x8c),
                 ported_exit()]
        result, output = self.run_ported(1, words, interpolation_map=0xff,
                                         export_mask=0x0f0f)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['interface']['color_output_mask'], 0x0f0f)
        self.assertEqual(metadata['interface']['color_outputs'],
                         [{'location': 0, 'component_mask': 15, 'scalar': 'u32',
                           'components': 4},
                          {'location': 2, 'component_mask': 15, 'scalar': 'u32',
                           'components': 4}])
        self.assertEqual(metadata['fragment_output_mode'], 'render-target-v1')
        facts = sorted(item['index'] for item in metadata['dependencies']
                       if item['kind'] == 'fragment_output_type')
        self.assertEqual(facts, [0, 2])
        self.assertFalse(metadata['interface']['writes_fragment_depth'])
        self.assertFalse(metadata['interface']['writes_sample_mask'])
        source = (output / 'source.metal').read_text()
        for location in (0, 2):
            self.assertIn(
                f'constant uint armigo_color{location}_type [[function_constant({location})]];',
                source)
            self.assertIn(f'uint4 armigo_color{location}_raw = uint4(0);', source)
            self.assertIn(
                f'out.color{location}_f32 = as_type<float4>(armigo_color{location}_raw);',
                source)
            self.assertIn(
                f'out.color{location}_s32 = as_type<int4>(armigo_color{location}_raw);',
                source)

    def test_ported_high_colour_targets_stay_in_range(self):
        # Targets 4 and 7 shift by 16 and 28 bits; the previous eight-bit
        # packing overflowed the 32-bit mask for them.
        words = [ported_interpolate(0, 0x80), ported_interpolate(1, 0x84),
                 ported_interpolate(2, 0x88), ported_interpolate(3, 0x8c),
                 ported_exit()]
        for export_mask, expected in ((0x000f0000, 0xf0000), (0xf0000000, 0xf0000000)):
            result, output = self.run_ported(1, words, interpolation_map=0xff,
                                             export_mask=export_mask)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output / 'reflection.json').read_text())
            self.assertEqual(metadata['interface']['color_output_mask'], expected)
            self.assertEqual(metadata['interface']['color_outputs'],
                             [{'location': 4 if expected == 0xf0000 else 7,
                               'component_mask': 15, 'scalar': 'u32', 'components': 4}])

    def test_ported_sparse_colour_masks_reflect_written_components(self):
        # Mask 0x5 exports components X and Z of target 0 from registers R0/R2.
        words = [ported_interpolate(0, 0x80), ported_interpolate(2, 0x88), ported_exit()]
        result, output = self.run_ported(1, words, interpolation_map=0xff, export_mask=0x5)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['interface']['color_output_mask'], 5)
        self.assertEqual(metadata['interface']['color_outputs'],
                         [{'location': 0, 'component_mask': 5, 'scalar': 'u32',
                           'components': 4}])

    def test_ported_fragment_depth_and_sample_mask_are_reflected(self):
        # The extra SPH word exports the sample mask and then the fragment
        # depth after the colour targets' registers.
        words = [ported_interpolate(0, 0x80), ported_interpolate(1, 0x84),
                 ported_interpolate(2, 0x88), ported_interpolate(3, 0x8c),
                 literal(4, 1.0), literal(5, 0.5), ported_exit()]
        result, output = self.run_ported(1, words, interpolation_map=0xff,
                                         export_mask=0xf, extra=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual(metadata['interface']['color_output_mask'], 15)
        self.assertTrue(metadata['interface']['writes_fragment_depth'])
        self.assertTrue(metadata['interface']['writes_sample_mask'])
        source = (output / 'source.metal').read_text()
        self.assertIn('[[depth(any)]]', source)
        self.assertIn('[[sample_mask]]', source)
        # Without the extra exports the flags stay unset.
        result, output = self.run_ported(1, words[:5] + [ported_exit()],
                                         interpolation_map=0xff, export_mask=0xf)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertFalse(metadata['interface']['writes_fragment_depth'])
        self.assertFalse(metadata['interface']['writes_sample_mask'])

    def test_ported_indexed_constant_buffers_have_room_for_size_table(self):
        # An interpolated selector leaves every supported CB slot reachable.
        ldc = pattern_word("1110111110010x",
                           ((0, 8, 0), (8, 8, 3), (44, 2, 2), (48, 3, 4)))
        result, output = self.run_ported(
            1, [ported_interpolate(3, 0x80), ldc, ported_exit()],
            interpolation_map=1, export_mask=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / 'reflection.json').read_text())
        self.assertEqual([entry['guest_binding'] for entry in metadata['constant_buffers']],
                         list(range(18)))
        self.assertEqual([entry['metal_buffer_index'] for entry in metadata['constant_buffers']],
                         list(range(18)))
        self.assertEqual(metadata['support_buffers'][0]['metal_buffer_index'], 18)

    def test_ported_interpolation_map_is_one_byte_per_attribute(self):
        # Generic 1 declares only its Y component as perspective (byte 0x19,
        # bits 3:2). Reading Y and the undeclared Z both resolve the mode from
        # that byte; generic 2, which declares nothing, stays a missing fact.
        for byte_offset in (0x94, 0x98):
            result, _ = self.run_ported(1, [ported_interpolate(4, byte_offset), ported_exit()],
                                        interpolation_map=[0x00, 0x08])
            self.assertNotIn('fact=9:', result.stderr, result.stderr)
        result, output = self.run_ported(1, [ported_interpolate(4, 0xa0), ported_exit()],
                                         interpolation_map=[0x00, 0x08])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fact=9:2:', result.stderr, result.stderr)
        self.assertFalse(output.exists())

    def test_ported_explicit_and_implicit_fragment_w_reads_merge(self):
        # Shaders commonly read fragment W explicitly (IPA a[0x7c]) and also
        # interpolate a perspective varying, which reads W implicitly. Both
        # describe one built-in input and must not conflict.
        pass_w = pattern_word("11100000000000",
                              ((0, 8, 3), (8, 8, 255), (28, 10, 0x7c), (52, 2, 0), (54, 2, 0)))
        result, _ = self.run_ported(1, [pass_w, ported_interpolate(4, 0x80), ported_exit()],
                                    interpolation_map=0x02)
        self.assertNotIn('Overlapping IO', result.stderr, result.stderr)

    def test_ported_vertex_depth_is_consumed_with_position_output(self):
        words = [literal(0, 0.5), literal(1, 1.0)]
        words += [ported_attribute_store(0, 0x70 + c * 4) for c in range(3)]
        words += [ported_attribute_store(1, 0x7c), ported_exit()]
        result, output = self.run_ported(0, words)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('legalize code=3', result.stderr)
        self.assertIn('fact=11:0:0:0', result.stderr)
        self.assertFalse(output.exists())
        for convention in ('minus-one-to-one', 'zero-to-one'):
            result, output = self.run_ported(0, words, '--vertex-depth', convention)
            self.assertEqual(result.returncode, 0, result.stderr)
            reflection = (output / 'reflection.json').read_text()
            depths = [dependency for dependency in json.loads(reflection)['dependencies']
                      if dependency['kind'] == 'depth_convention']
            self.assertEqual(len(depths), 1)
            self.assertEqual(depths[0]['origin'], 'explicit_option')
            self.assertEqual(depths[0]['value'], convention)
            source = (output / 'source.metal').read_text()
            self.assertEqual('armigo_depth_sum * 0.5f' in source,
                             convention == 'minus-one-to-one')

    def test_ported_fragment_position_option_reaches_emission(self):
        for words, interpolation_map in (
                ([ported_interpolate(0, 0x7c), ported_exit()], 0),
                ([ported_interpolate(0, 0x80), ported_exit()], 2)):
            result, output = self.run_ported(1, words, interpolation_map=interpolation_map,
                                             export_mask=1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('explicit host position convention', result.stderr)
            self.assertFalse(output.exists())
            result, output = self.run_ported(1, words, '--fragment-position',
                                             'reciprocal-clip-w',
                                             interpolation_map=interpolation_map, export_mask=1)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output / 'reflection.json').read_text())
            self.assertEqual(metadata['fragment_position'], 'reciprocal-clip-w')
            self.assertIn('in.position.w', (output / 'source.metal').read_text())

    def test_ported_tmml_uses_metal_lod_queries(self):
        # TEXS on the same handle proves its descriptor; TMML supplies no
        # comparison-format evidence. Coordinates stay live fragment inputs.
        for mask, methods in ((1, ('calculate_unclamped_lod',)),
                              (2, ('calculate_clamped_lod',)),
                              (3, ('calculate_clamped_lod', 'calculate_unclamped_lod'))):
            tmml = pattern_word('1101111101011x',
                                ((0, 8, 0), (8, 8, 4), (28, 3, 2),
                                 (31, 4, mask), (36, 13, 8)))
            words = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84),
                     ported_texs(6, 4, 5), tmml, ported_exit()]
            result, output = self.run_ported(1, words, '--texture-handle-cbuf', '2',
                                             '--texture-descriptors', 'derived',
                                             interpolation_map=5,
                                             export_mask=3 if mask == 3 else 1)
            self.assertEqual(result.returncode, 0, result.stderr)
            source = (output / 'source.metal').read_text()
            for method in ('calculate_clamped_lod', 'calculate_unclamped_lod'):
                self.assertEqual(method in source, method in methods)

    def test_ported_vertex_io_and_position_requirements(self):
        # A vertex program without outputs emits a void vertex function whose
        # reflection requires rasterizer discard.
        for words in ([ported_attribute_load(1, 0x80), ported_exit()], [ported_exit()]):
            result, output = self.run_ported(0, words, *self.vertex_options)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output / 'reflection.json').read_text())
            self.assertEqual(metadata.get('rasterizer_discard'), 'required-v1')
            self.assertIn('vertex void', (output / 'source.metal').read_text())
        # Outputs without a position fail late, at emission: a program-wide
        # emit_msl diagnostic.
        result, output = self.run_ported(0, [ported_attribute_load(1, 0x80),
                                             ported_attribute_store(1, 0x80), ported_exit()],
                                         *self.vertex_options)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr.startswith('armigo-shader-translate: emit_msl code='),
                        result.stderr)
        self.assertIn('position', result.stderr)
        self.assertFalse(output.exists())

    def test_ported_vertex_inputs_classify_by_use_without_format_facts(self):
        # A vertex attribute load without any --vertex-float3 option must
        # classify by use: the load legalizes (no missing
        # vertex_attribute_format fact) and the program reaches MSL emission,
        # which stops on the synthetic program's missing position export
        # instead. Supplying the option keeps the explicit-format path.
        program = [ported_attribute_load(1, 0x80), ported_attribute_store(1, 0x80), ported_exit()]
        result, _ = self.run_ported(0, program, '--vertex-depth', 'minus-one-to-one')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('emit_msl code=', result.stderr, result.stderr)
        self.assertNotIn('fact=', result.stderr, result.stderr)
        result, _ = self.run_ported(0, program, *self.vertex_options)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('emit_msl code=', result.stderr, result.stderr)

    def test_texture_handle_cbuf_option_is_validated_and_translator_scoped(self):
        for index, invalid in enumerate(('18', '-1', 'x', '2:0', '')):
            result, output = self.run_arguments('--translator', 'ported',
                                                '--texture-handle-cbuf', invalid)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
        result, output = self.run_arguments('--translator', 'ported',
                                            '--texture-handle-cbuf', '2',
                                            '--texture-handle-cbuf', '3')
        self.assertIn('Duplicate --texture-handle-cbuf', result.stderr)
        self.assertFalse(output.exists())
        # Only the ported pipeline consumes facts; the current translator must
        # reject the option instead of ignoring it.
        result, output = self.run_arguments('--texture-handle-cbuf', '2')
        self.assertIn('requires --translator ported', result.stderr)
        self.assertFalse(output.exists())

    def test_texture_handle_cbuf_answers_bound_texture_emission(self):
        # A fragment TEXS with an encoded handle word but no constant-buffer
        # slot consumes texture_handle_cbuf during emit_ir. Without the option
        # the fact stays Unknown and the stage stops with a missing-fact
        # diagnostic naming the texture_handle_cbuf key.
        words = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84),
                 ported_texs(6, 4, 5), ported_exit()]
        result, output = self.run_ported(1, words, '--fragment-position', 'reciprocal-clip-w',
                                         interpolation_map=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('emit_ir code=', result.stderr)
        self.assertIn('fact=4:0:0:0', result.stderr)
        self.assertFalse(output.exists())
        # With the option the recorded slot is the provenance; emission moves
        # on to the next missing evidence, the descriptor at that exact key.
        result, output = self.run_ported(1, words, '--fragment-position', 'reciprocal-clip-w',
                                         '--texture-handle-cbuf', '2', interpolation_map=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('emit_ir code=', result.stderr)
        self.assertIn('fact=5:2:32:0', result.stderr)
        self.assertNotIn('fact=4:', result.stderr)
        self.assertFalse(output.exists())

    def test_texture_descriptors_option_is_validated_and_translator_scoped(self):
        # Value validation, duplicate rejection and translator scoping mirror
        # the other ported-path evidence options.
        for invalid in ('captured', 'DERIVED', '2', ''):
            result, output = self.run_arguments('--translator', 'ported',
                                                '--texture-descriptors', invalid)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
        result, output = self.run_arguments('--translator', 'ported',
                                            '--texture-descriptors', 'derived',
                                            '--texture-descriptors', 'derived')
        self.assertIn('Duplicate --texture-descriptors', result.stderr)
        self.assertFalse(output.exists())
        # Only the ported pipeline consumes the facts; the current translator
        # must reject the option instead of ignoring it.
        result, output = self.run_arguments('--texture-descriptors', 'derived')
        self.assertIn('requires --translator ported', result.stderr)
        self.assertFalse(output.exists())

    def test_derived_texture_descriptors_answer_agreeing_handles(self):
        # A TEXS whose only descriptor evidence is its own encoding: with
        # --texture-descriptors derived the emit-ir descriptor fact is answered
        # from the pre-scan. Constant-buffer reads no longer need binding
        # facts, so the stage translates and records the derived dependency.
        # 2D and 2D-array targets each derive their shape; the compact write
        # selector writes XY, and only R0 is exported.
        base = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84)]
        for target in (2, 8):
            words = base + [ported_texs(0, 4, 5, target=target), ported_exit()]
            result, output = self.run_ported(
                1, words, '--texture-handle-cbuf', '2',
                '--texture-descriptors', 'derived',
                interpolation_map=[5, 5], export_mask=1)
            self.assertEqual(result.returncode, 0, result.stderr)
            reflection = (output / 'reflection.json').read_text()
            self.assertIn('texture_descriptor', reflection)
            self.assertIn('"kind":"texture_descriptor","index":2,"byte_offset":32,"component":0,"origin":"derived"', reflection)
        # Without the option the same stage still stops on the descriptor.
        words = base + [ported_texs(0, 4, 5), ported_exit()]
        result, output = self.run_ported(1, words, '--texture-handle-cbuf', '2',
                                         interpolation_map=[5, 5], export_mask=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fact=5:2:32:0', result.stderr)
        self.assertFalse(output.exists())

    def test_ported_v3_resource_thresholds_and_locations(self):
        # Each result feeds the next instruction, keeping all synthetic
        # resources live through lowering and SSA.
        for count, sampled in ((16, True), (17, True), (128, False), (129, False)):
            with self.subTest(count=count, sampled=sampled):
                seed = (pattern_word("1110111110010x",
                                     ((0, 8, 0), (8, 8, 255), (20, 16, 12),
                                      (36, 5, 17), (48, 3, 4)))
                        if count == 17 else ported_interpolate(0, 0x80))
                words = [seed, ported_interpolate(5, 0x84)]
                for index in range(count):
                    words.append(ported_texs(0, 0, 5, handle_word=8 + index) if sampled
                                 else ported_txq(0, 0, handle_word=8 + index))
                words.append(ported_exit())
                result, output = self.run_ported(
                    1, words, '--texture-handle-cbuf', '2',
                    '--texture-descriptors', 'derived',
                    interpolation_map=[5, 5], export_mask=1)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads((output / 'reflection.json').read_text())
                self.assertEqual(metadata['schema'], 'armigo-shader-stage-v3')
                self.assertEqual(metadata['profile']['version'], 3)
                # Validate real translator output through the package boundary,
                # including arena IDs and member order that differ from byte order.
                tools = Path(__file__).resolve().parents[2] / 'src' / 'shader' / 'tools'
                sys.path.insert(0, str(tools))
                try:
                    import build_package
                    build_package.validate_reflection(
                        metadata, (output / 'program.bin').read_bytes(),
                        (output / 'source.metal').read_bytes())
                finally:
                    sys.path.pop(0)
                resources = metadata['textures']
                self.assertEqual(len(resources), count * (2 if sampled else 1))
                self.assertEqual(len({item['resource_id'] for item in resources}), len(resources))
                indirect = count > (16 if sampled else 128)
                source = (output / 'source.metal').read_text()
                layout = metadata['argument_buffer']
                if indirect:
                    self.assertEqual(layout['abi_version'], 1)
                    self.assertEqual(layout['alignment'], 8)
                    self.assertEqual(layout['byte_extent'], len(resources) * 8)
                    self.assertNotEqual(layout['buffer_index'], 30)
                    self.assertIn('const device ArmigoStageResources&', source)
                    self.assertIn(f"[[buffer({layout['buffer_index']})]]", source)
                    self.assertNotIn('[[texture(', source)
                    self.assertNotIn('[[sampler(', source)
                    self.assertEqual(sorted(item['location']['byte_offset'] for item in resources),
                                     list(range(0, layout['byte_extent'], 8)))
                else:
                    self.assertIsNone(layout)
                    self.assertNotIn('ArmigoStageResources', source)
                for item in resources:
                    self.assertEqual(item['cbuf_slot'], 2)
                    self.assertEqual(item['array_length'], 1)
                    self.assertNotIn('texture_index', item)
                    self.assertNotIn('sampler_index', item)
                    location = item['location']
                    self.assertEqual(location['count'], 1)
                    self.assertEqual(location['kind'], 'argument_buffer_member' if indirect else 'direct')
                    if indirect:
                        self.assertEqual(location['stride'], 8)
                    else:
                        self.assertEqual(location['namespace'], item['kind'])

    def test_derived_depth_comparison_descriptors_reach_msl(self):
        base = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84),
                ported_interpolate(6, 0x88)]
        forms = [(f'texs-{target}', ported_texs(0, 4, 5, target=target),
                  'sample_compare', target == 9) for target in (4, 5, 6, 9)]
        forms += [('tex', ported_depth_texture(0, 4, 6), 'sample_compare', False),
                  ('tld4', ported_depth_texture(0, 4, 6, gather=True),
                   'gather_compare', False)]
        for label, form, method, array in forms:
            with self.subTest(form=label):
                result, output = self.run_ported(
                    1, base + [form, ported_exit()], '--texture-handle-cbuf', '2',
                    '--texture-descriptors', 'derived',
                    interpolation_map=[5, 5], export_mask=3)
                self.assertEqual(result.returncode, 0, result.stderr)
                source = (output / 'source.metal').read_text()
                self.assertIn('depth2d_array<float>' if array else 'depth2d<float>', source)
                self.assertIn(f'.{method}(', source)
                reflection = json.loads((output / 'reflection.json').read_text())
                resources = reflection['textures']
                texture = next(item for item in resources if item['kind'] == 'texture')
                sampler = next(item for item in resources if item['kind'] == 'sampler')
                self.assertTrue(texture['comparison'])
                self.assertEqual(texture['component_format'], 'depth')
                self.assertTrue(sampler['comparison'])
                self.assertEqual(texture['cbuf_slot'], sampler['cbuf_slot'])
                self.assertEqual(texture['handle_byte_offset'], sampler['handle_byte_offset'])

    def test_conflicting_depth_comparison_uses_remain_unknown(self):
        words = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84),
                 ported_texs(0, 4, 5, target=2), ported_texs(2, 4, 5, target=4),
                 ported_exit()]
        result, output = self.run_ported(
            1, words, '--texture-handle-cbuf', '2', '--texture-descriptors', 'derived',
            interpolation_map=[5, 5], export_mask=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fact=5:2:32:0', result.stderr)
        self.assertIn('disagree', result.stderr)
        self.assertFalse(output.exists())

    def test_disagreeing_texture_uses_keep_the_missing_fact(self):
        # One handle sampled as plain 2D by one instruction and as a 2D array
        # by another: the uses disagree, so the descriptor stays Unknown and
        # the stage stops with the same structured missing fact.
        words = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84),
                 ported_texs(0, 4, 5, target=2), ported_texs(2, 4, 5, target=8),
                 ported_exit()]
        result, output = self.run_ported(1, words, '--texture-handle-cbuf', '2',
                                         '--texture-descriptors', 'derived',
                                         interpolation_map=[5, 5], export_mask=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('emit_ir code=', result.stderr)
        self.assertIn('fact=5:2:32:0', result.stderr)
        self.assertIn('disagree', result.stderr)
        self.assertFalse(output.exists())

    def test_derived_query_only_handles_assume_a_runtime_verified_2d(self):
        # A TXQ size query proves no sampled shape: with --texture-descriptors
        # derived the handle assumes the plain 2D descriptor the runtime
        # verifier admits (evidence label code:assumed-query-only-2d) and the
        # stage translates. The label is recorded evidence, not a serialized
        # default: every draw re-checks the live texture against the
        # assumption. A size component a 2D shape cannot answer still fails
        # its own structured diagnostic.
        base = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84)]
        words = base + [ported_txq(0, 4), ported_txq(2, 4),
                        ported_position_store(0), ported_exit()]
        result, output = self.run_ported(0, words, *self.vertex_options,
                                         '--texture-handle-cbuf', '2',
                                         '--texture-descriptors', 'derived')
        self.assertEqual(result.returncode, 0, result.stderr)
        reflection = (output / 'reflection.json').read_text()
        self.assertIn('"kind":"texture_descriptor","index":2,"byte_offset":32,'
                      '"component":0,"origin":"derived"', reflection)
        words = base + [ported_txq(0, 4, mask=4), ported_exit()]
        result, output = self.run_ported(0, words, *self.vertex_options,
                                         '--texture-handle-cbuf', '2',
                                         '--texture-descriptors', 'derived')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('emit_ir code=', result.stderr)
        self.assertIn('absent from the texture shape', result.stderr)
        self.assertFalse(output.exists())

    def test_derived_fetch_only_handles_assume_no_comparison(self):
        # TXQ plus a multisample texel fetch on one handle: the fetch proves
        # the 2D shape but no instruction constrains the sampler compare mode,
        # so the descriptor assumes comparison off (evidence label
        # code:assumed-fetch-no-compare). The draw-time verifier keeps
        # checking the live sampler state against that assumption.
        words = [ported_interpolate(4, 0x80), ported_interpolate(5, 0x84),
                 ported_txq(6, 4), ported_tlds(0, 4, 5), ported_exit()]
        result, output = self.run_ported(
            1, words, '--fragment-position', 'reciprocal-clip-w',
            '--texture-handle-cbuf', '2', '--texture-descriptors', 'derived',
            interpolation_map=[5, 5], export_mask=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('.read(', (output / 'source.metal').read_text())
        reflection = json.loads((output / 'reflection.json').read_text())
        texture = next(item for item in reflection['textures']
                       if item.get('kind') == 'texture')
        self.assertFalse(texture['comparison'])


if __name__ == '__main__':
    unittest.main()
