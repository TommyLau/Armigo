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

"""Build a bounded development subset of offline Metal shader libraries."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import uuid

import chain


PREFIX_SIZE = 112
MAX_PROGRAM_SIZE = 4 * 1024 * 1024
MAX_SOURCE_SIZE = 4 * 1024 * 1024
MAX_REFLECTION_SIZE = 1024 * 1024
MAX_LIBRARY_SIZE = 64 * 1024 * 1024
# Translated sources are inputs only (the package carries compiled libraries),
# and a whole-title package spans tens of thousands of stages.
MAX_INPUT_BYTES = 8 * 1024 * 1024 * 1024
MAX_OUTPUT_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_IDENTITIES = 1_000_000
MAX_BUCKET = 64
COMMAND_TIMEOUT_SECONDS = 60
STAGE_TYPES = {"vertex": 1, "fragment": 5}
HEX = set("0123456789abcdef")
SCALARS = {"u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64",
           "f16", "f32", "f64", "boolean"}
VERTEX_FORMATS = {"u8", "s8", "unorm8", "snorm8", "u16", "s16", "unorm16",
                  "snorm16", "f16", "u32", "s32", "f32"}
FACT_KINDS = {"local_memory_bytes", "constant_buffer_bound_mask",
              "constant_buffer_size", "constant_buffer_word", "texture_handle_cbuf",
              "texture_descriptor", "texture_array_length", "vertex_attribute_format",
              "fragment_output_type", "input_interpolation", "topology",
              "depth_convention", "origin_convention", "negate_y", "alpha_test",
              "early_z", "alpha_to_coverage", "dual_source_blend", "draw_parameters",
              "compute_local_size", "compute_shared_memory_bytes"}
FACT_ORIGINS = {"validated_header", "captured_state", "explicit_option", "derived"}
TEXTURE_DIMENSIONS = {"d1", "d2", "d3", "cube", "buffer"}
COMPONENT_FORMATS = {"unorm", "snorm", "unsigned_integer", "signed_integer",
                     "floating", "depth"}


def valid_texture_component_contract(value):
    scalar, component = value["scalar"], value["component_format"]
    compatible = ((scalar in {"f16", "f32"} and
                   component in {"floating", "unorm", "snorm", "depth"}) or
                  (scalar == "u32" and component == "unsigned_integer") or
                  (scalar == "s32" and component == "signed_integer"))
    return compatible and (not value["comparison"] or
                           (scalar == "f32" and component == "depth"))


def texture_descriptor_value(value, label):
    require_keys(value, ("dimension", "array", "multisample", "scalar",
                         "component_format", "normalized_coordinates",
                         "comparison"), label)
    require(value["dimension"] in TEXTURE_DIMENSIONS and
            value["scalar"] in SCALARS and
            value["component_format"] in COMPONENT_FORMATS and
            all(isinstance(value[name], bool) for name in
                ("array", "multisample", "normalized_coordinates", "comparison")),
            f"{label} value is invalid")
    require(valid_texture_component_contract(value),
            f"{label} has incompatible scalar and component format")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(data):
    return json.loads(data.decode("utf-8"), object_pairs_hook=_object_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Invalid JSON constant: {value}")))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def is_uint(value, maximum):
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum


def require_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys),
            f"{label} has an unexpected schema")


def read_regular(path, maximum, label):
    require(path.is_file() and not path.is_symlink(), f"{label} must be a regular file")
    size = path.stat().st_size
    require(size <= maximum, f"{label} exceeds its size policy")
    data = path.read_bytes()
    require(len(data) <= maximum, f"{label} changed beyond its size policy")
    return data


def validate_attributes(items):
    require(isinstance(items, list), "vertex_attributes must be an array")
    previous = -1
    attributes = set()
    for item in items:
        require_keys(item, ("generic_slot", "attribute_index", "format"), "vertex attribute")
        require(is_uint(item["generic_slot"], 31) and item["generic_slot"] > previous,
                "vertex attributes must have unique sorted slots")
        require(is_uint(item["attribute_index"], 30) and
                item["attribute_index"] not in attributes and
                item["format"] == "float32x3",
                "vertex attribute option is unsupported")
        previous = item["generic_slot"]
        attributes.add(item["attribute_index"])


def validate_constant_buffers(items):
    require(isinstance(items, list) and len(items) <= 18,
            "constant_buffers must be an array of at most 18 entries")
    guests, metals = set(), set()
    for item in items:
        require_keys(item, ("guest_binding", "minimum_size", "metal_buffer_index"),
                     "constant buffer")
        guest, size, metal = (item["guest_binding"], item["minimum_size"],
                              item["metal_buffer_index"])
        require(is_uint(guest, 17) and is_uint(size, 65536) and size > 0 and size % 4 == 0 and
                is_uint(metal, 17) and guest not in guests and metal not in metals,
                "constant-buffer reflection is invalid")
        guests.add(guest)
        metals.add(metal)


def validate_textures(items):
    require(isinstance(items, list) and len(items) <= 16,
            "textures must be an array of at most 16 entries")
    handles, textures, samplers = set(), set(), set()
    for item in items:
        require_keys(item, ("handle_byte_offset", "texture_index", "sampler_index"), "texture")
        handle, texture, sampler = (item["handle_byte_offset"], item["texture_index"],
                                    item["sampler_index"])
        require(is_uint(handle, 32764) and handle % 4 == 0 and is_uint(texture, 15) and
                is_uint(sampler, 15) and handle not in handles and texture not in textures and
                sampler not in samplers, "texture reflection is invalid")
        handles.add(handle)
        textures.add(texture)
        samplers.add(sampler)


def validate_generic(items, label):
    require(isinstance(items, list) and len(items) <= 32,
            f"{label} must be an array of at most 32 entries")
    slots = set()
    for item in items:
        require_keys(item, ("slot", "component_mask", "interpolation"), label)
        require(is_uint(item["slot"], 31) and is_uint(item["component_mask"], 15) and
                item["component_mask"] != 0 and item["slot"] not in slots and
                item["interpolation"] in ("constant", "perspective", "screen-linear"),
                f"{label} reflection is invalid")
        slots.add(item["slot"])


def validate_generic_v2(items, label):
    require(isinstance(items, list) and len(items) <= 32,
            f"{label} must be an array of at most 32 entries")
    slots = set()
    for item in items:
        require_keys(item, ("slot", "component_mask", "scalar", "components",
                            "interpolation"), label)
        require(is_uint(item["slot"], 31) and is_uint(item["component_mask"], 15) and
                item["component_mask"] != 0 and item["slot"] not in slots and
                item["interpolation"] in ("flat", "perspective", "linear") and
                item["scalar"] in SCALARS and is_uint(item["components"], 4) and
                item["components"] != 0,
                f"{label} reflection is invalid")
        slots.add(item["slot"])


def validate_interface(value, stage, attributes, textures):
    keys = ("position_input_mask", "position_output_mask", "color_output_mask",
            "generic_inputs", "generic_outputs")
    require_keys(value, keys, "interface")
    require(is_uint(value["position_input_mask"], 15) and
            is_uint(value["position_output_mask"], 15) and
            is_uint(value["color_output_mask"], 0xffffffff),
            "interface masks are invalid")
    validate_generic(value["generic_inputs"], "generic_inputs")
    validate_generic(value["generic_outputs"], "generic_outputs")
    if stage == "vertex":
        require(value["position_input_mask"] == 0 and value["position_output_mask"] == 15 and
                value["color_output_mask"] == 0 and not textures and
                len(value["generic_inputs"]) == len(attributes),
                "vertex interface is outside the supported profile")
        inputs = {item["slot"]: item for item in value["generic_inputs"]}
        for option in attributes:
            require(option["generic_slot"] in inputs and
                    inputs[option["generic_slot"]]["component_mask"] & ~7 == 0,
                    "vertex attributes do not cover the reflected inputs")
        require(all(item["interpolation"] == "screen-linear"
                    for item in value["generic_outputs"]),
                "vertex output interpolation is invalid")
    else:
        perspective = any(item["interpolation"] == "perspective"
                          for item in value["generic_inputs"])
        require(value["position_output_mask"] == 0 and value["color_output_mask"] == 15 and
                not value["generic_outputs"] and value["position_input_mask"] in (0, 8),
                "fragment interface is outside the supported profile")
        return perspective
    return False


def validate_reflection(metadata, program, source):
    if metadata.get("schema") == "armigo-shader-stage-compute-v1":
        return validate_reflection_compute(metadata, program, source)
    if metadata.get("schema") in ("armigo-shader-stage-v2", "armigo-shader-stage-v3"):
        return validate_reflection_v2(metadata, program, source)
    keys = ("schema", "source_version", "program_key", "program_size", "program_file",
            "source_file", "source_digest", "stage", "entry_point", "vertex_depth",
            "fragment_position", "vertex_attributes", "constant_buffers", "textures",
            "interface")
    require_keys(metadata, keys, "reflection")
    require(metadata["schema"] == "armigo-shader-stage-v1", "reflection schema is unsupported")
    for name in ("source_version", "program_key", "source_digest"):
        require(isinstance(metadata[name], str) and len(metadata[name]) == 64 and
                all(c in HEX for c in metadata[name]), f"reflection {name} is invalid")
    require(metadata["program_file"] == "program.bin" and
            metadata["source_file"] == "source.metal", "reflection filenames are invalid")
    require(metadata["program_size"] == len(program) and
            metadata["program_key"] == digest(program), "program identity does not match reflection")
    require(metadata["source_digest"] == digest(source), "source digest does not match reflection")
    stage = metadata["stage"]
    require(stage in STAGE_TYPES, "reflection stage is unsupported")
    require(len(program) >= PREFIX_SIZE, "program is shorter than the identity prefix")
    require(len(program) <= MAX_PROGRAM_SIZE, "program exceeds the identity size policy")
    sph = struct.unpack_from("<I", program)[0]
    require(((sph >> 10) & 15) == STAGE_TYPES[stage], "program SPH stage mismatch")
    validate_attributes(metadata["vertex_attributes"])
    if stage == "vertex":
        require(metadata["entry_point"] == "armigo_vertex_main" and
                metadata["vertex_depth"] in ("minus-one-to-one", "zero-to-one") and
                metadata["fragment_position"] == "unspecified",
                "vertex explicit options are invalid")
    else:
        require(metadata["entry_point"] == "armigo_fragment_main" and
                metadata["vertex_depth"] == "unspecified" and
                metadata["fragment_position"] in ("unspecified", "reciprocal-clip-w") and
                metadata["vertex_attributes"] == [], "fragment explicit options are invalid")
    validate_constant_buffers(metadata["constant_buffers"])
    validate_textures(metadata["textures"])
    perspective = validate_interface(metadata["interface"], stage,
                                     metadata["vertex_attributes"], metadata["textures"])
    if stage == "fragment":
        needs_position = metadata["interface"]["position_input_mask"] == 8 or perspective
        require((needs_position and metadata["fragment_position"] == "reciprocal-clip-w") or
                (not needs_position and metadata["fragment_position"] == "unspecified"),
                "fragment position option does not match its interface")
    return stage


def validate_profile(value):
    require_keys(value, ("identity", "version"), "profile")
    require(isinstance(value["identity"], str) and value["identity"] != "" and
            all(ord(c) < 0x80 for c in value["identity"]) and
            is_uint(value["version"], 0xffffffff), "profile reflection is invalid")


def validate_constant_buffers_v2(items):
    require(isinstance(items, list) and len(items) <= 18,
            "constant_buffers must be an array of at most 18 entries")
    guests, metals = set(), set()
    for item in items:
        require_keys(item, ("guest_binding", "size_kind", "window_bytes", "alignment",
                            "policy", "metal_buffer_index"), "constant buffer")
        guest, extent, metal = (item["guest_binding"], item["window_bytes"],
                                item["metal_buffer_index"])
        require(is_uint(guest, 17) and is_uint(extent, 65536) and extent >= 4 and
                extent % 4 == 0 and is_uint(metal, 30) and guest not in guests and
                metal not in metals and item["size_kind"] == "constant_buffer_window" and
                is_uint(item["alignment"], 4) and item["alignment"] == 4 and
                item["policy"] == "runtime_zero_words_v1",
                "constant-buffer reflection is invalid")
        guests.add(guest)
        metals.add(metal)
    return metals


def validate_support_buffers_v2(items, constant_buffer_indices):
    require(isinstance(items, list) and len(items) == bool(constant_buffer_indices),
            "support_buffers must contain one entry exactly when constant buffers exist")
    for item in items:
        require_keys(item, ("schema", "version", "byte_extent", "alignment",
                            "metal_buffer_index"), "support buffer")
        require(item["schema"] == "armigo-cb-sizes" and
                is_uint(item["version"], 1) and item["version"] == 1 and
                is_uint(item["byte_extent"], 72) and item["byte_extent"] == 72 and
                is_uint(item["alignment"], 4) and item["alignment"] == 4 and
                is_uint(item["metal_buffer_index"], 30) and
                item["metal_buffer_index"] not in constant_buffer_indices,
                "support-buffer reflection is invalid")


def validate_storage_buffers_v3(items, constant_buffers):
    require(isinstance(items, list) and len(items) <= 31,
            "storage_buffers must be an array of at most 31 entries")
    windows = {item["guest_binding"]: item["window_bytes"]
               for item in constant_buffers}
    descriptors, metals = set(), set()
    for item in items:
        require_keys(item, ("cbuf_slot", "descriptor_byte_offset",
                            "metal_buffer_index", "access", "alignment"),
                     "storage buffer")
        slot = item["cbuf_slot"]
        offset = item["descriptor_byte_offset"]
        metal = item["metal_buffer_index"]
        descriptor = (slot, offset)
        require(is_uint(slot, 17) and is_uint(offset, 65524) and
                offset % 8 == 0 and slot in windows and
                offset + 12 <= windows[slot] and
                is_uint(metal, 30) and metal not in metals and
                descriptor not in descriptors and
                item["access"] in ("read", "write", "read_write") and
                is_uint(item["alignment"], 4) and item["alignment"] == 4,
                "storage-buffer reflection is invalid")
        descriptors.add(descriptor)
        metals.add(metal)
    return metals


def validate_textures_v2(items):
    require(isinstance(items, list) and len(items) <= 32,
            "textures must be an array of at most 32 entries")
    # A texture and its sampler come from the same handle word, so the offset
    # is unique within each kind, not across kinds.
    texture_handles, sampler_handles, textures, samplers = set(), set(), set(), set()
    for item in items:
        require(isinstance(item, dict) and "kind" in item, "texture entry is invalid")
        kind = item["kind"]
        require(kind in ("texture", "sampler"), "texture kind is unsupported")
        if kind == "texture":
            require_keys(item, ("cbuf_slot", "handle_byte_offset", "kind", "dimension",
                                "array", "multisample", "scalar", "component_format",
                                "normalized_coordinates", "comparison",
                                "texture_index"), "texture")
        else:
            require_keys(item, ("cbuf_slot", "handle_byte_offset", "kind", "comparison",
                                "normalized_coordinates", "sampler_index"), "sampler")
        handle = item["handle_byte_offset"]
        handles = texture_handles if kind == "texture" else sampler_handles
        require(is_uint(item["cbuf_slot"], 17) and is_uint(handle, 32764) and
                handle % 4 == 0 and handle not in handles,
                "texture provenance reflection is invalid")
        handles.add(handle)
        if kind == "texture":
            require(item["dimension"] in TEXTURE_DIMENSIONS and
                    item["scalar"] in SCALARS and
                    item["component_format"] in COMPONENT_FORMATS and
                    isinstance(item["array"], bool) and
                    isinstance(item["multisample"], bool) and
                    isinstance(item["normalized_coordinates"], bool) and
                    isinstance(item["comparison"], bool) and
                    is_uint(item["texture_index"], 15) and
                    item["texture_index"] not in textures,
                    "texture reflection is invalid")
            textures.add(item["texture_index"])
        else:
            require(isinstance(item["comparison"], bool) and
                    isinstance(item["normalized_coordinates"], bool) and
                    is_uint(item["sampler_index"], 15) and
                    item["sampler_index"] not in samplers,
                    "sampler reflection is invalid")
            samplers.add(item["sampler_index"])


def validate_argument_buffer_v3(value):
    if value is None:
        return None
    require_keys(value, ("abi_version", "buffer_index", "byte_extent", "alignment"),
                 "argument_buffer")
    require(is_uint(value["abi_version"], 1) and value["abi_version"] == 1 and
            is_uint(value["buffer_index"], 30) and
            is_uint(value["byte_extent"], 0xffffffff) and value["byte_extent"] > 0 and
            value["byte_extent"] % 8 == 0 and
            is_uint(value["alignment"], 8) and value["alignment"] == 8,
            "argument-buffer layout is invalid")
    return value["buffer_index"]


def validate_textures_v3(items, argument_buffer):
    require(isinstance(items, list) and len(items) <= 4096,
            "textures must be an array of at most 4096 entries")
    resource_ids = [item.get("resource_id") if isinstance(item, dict) else None
                    for item in items]
    require(all(is_uint(resource_id, 0xffffffff) for resource_id in resource_ids) and
            len(set(resource_ids)) == len(resource_ids),
            "resource IDs must be unique uint32 values")
    direct_ranges = {"texture": [], "sampler": []}
    member_ranges = []
    for item in items:
        require(isinstance(item, dict) and "kind" in item, "texture entry is invalid")
        kind = item["kind"]
        require(kind in ("texture", "sampler"), "texture kind is unsupported")
        common = ("cbuf_slot", "handle_byte_offset", "kind", "resource_id",
                  "array_length", "location")
        if kind == "texture":
            require_keys(item, common + ("dimension", "array", "multisample", "scalar",
                         "component_format", "normalized_coordinates", "comparison",
                         "resource_kind", "access"), "texture")
        else:
            require_keys(item, common + ("comparison", "normalized_coordinates"), "sampler")
        length = item["array_length"]
        offset = item["handle_byte_offset"]
        require(is_uint(item["cbuf_slot"], 17) and is_uint(offset, 65532) and
                offset % 4 == 0 and is_uint(length, 0xffffffff) and length > 0 and
                offset + length * 4 <= 65536,
                "texture provenance reflection is invalid")
        require(isinstance(item["comparison"], bool) and
                isinstance(item["normalized_coordinates"], bool),
                f"{kind} reflection is invalid")
        if kind == "texture":
            require(item["dimension"] in TEXTURE_DIMENSIONS and
                    item["scalar"] in SCALARS and
                    item["component_format"] in COMPONENT_FORMATS and
                    isinstance(item["array"], bool) and
                    isinstance(item["multisample"], bool) and
                    item["resource_kind"] in ("sampled", "storage_image") and
                    item["access"] in ("read", "write", "read_write") and
                    (item["resource_kind"] != "sampled" or item["access"] == "read"),
                    "texture reflection is invalid")
            texture_descriptor_value({name: item[name] for name in
                                      ("dimension", "array", "multisample", "scalar",
                                       "component_format", "normalized_coordinates",
                                       "comparison")}, "texture")
        location = item["location"]
        require(isinstance(location, dict) and "kind" in location,
                "resource location is invalid")
        if argument_buffer is None:
            require_keys(location, ("kind", "namespace", "first", "count"),
                         "resource location")
            namespace = "texture" if kind == "texture" else "sampler"
            first, count = location["first"], location["count"]
            capacity = 128 if namespace == "texture" else 16
            require(location["kind"] == "direct" and
                    location["namespace"] == namespace and
                    is_uint(first, 0xffffffff) and is_uint(count, 0xffffffff) and count == length and
                    first + count <= capacity,
                    "direct resource location is invalid")
            ranges = direct_ranges[namespace]
            require(all(first + count <= start or first >= end for start, end in ranges),
                    "direct resource locations overlap")
            ranges.append((first, first + count))
        else:
            require_keys(location, ("kind", "byte_offset", "count", "stride"),
                         "resource location")
            byte_offset = location["byte_offset"]
            require(location["kind"] == "argument_buffer_member" and
                    is_uint(byte_offset, 0xffffffff) and byte_offset % 8 == 0 and
                    is_uint(location["count"], 0xffffffff) and location["count"] == length and
                    is_uint(location["stride"], 8) and location["stride"] == 8,
                    "argument-buffer member location is invalid")
            member_end = byte_offset + length * 8
            require(member_end <= 0xffffffff and member_end <= argument_buffer["byte_extent"] and
                    all(member_end <= start or byte_offset >= end
                        for start, end in member_ranges),
                    "argument-buffer member exceeds its layout")
            member_ranges.append((byte_offset, member_end))
    if argument_buffer is not None:
        cursor = 0
        for start, end in sorted(member_ranges):
            require(start == cursor, "argument-buffer members must be dense")
            cursor = end
        require(cursor == argument_buffer["byte_extent"],
                "argument-buffer extent does not match its dense members")


def validate_vertex_streams(items):
    require(isinstance(items, list) and len(items) <= 32,
            "vertex_streams must be an array of at most 32 entries")
    slots, attributes = set(), set()
    for item in items:
        require_keys(item, ("slot", "attribute_index", "format", "components",
                            "byte_stride"), "vertex stream")
        require(is_uint(item["slot"], 31) and item["slot"] not in slots and
                is_uint(item["attribute_index"], 30) and
                item["attribute_index"] not in attributes and
                item["format"] in VERTEX_FORMATS and
                is_uint(item["components"], 4) and item["components"] != 0 and
                is_uint(item["byte_stride"], 64) and item["byte_stride"] != 0,
                "vertex stream reflection is invalid")
        slots.add(item["slot"])
        attributes.add(item["attribute_index"])


def validate_dependencies(items):
    require(isinstance(items, list) and len(items) <= 1024,
            "dependencies must be an array of at most 1024 entries")
    keys = set()
    for item in items:
        require_keys(item, ("kind", "index", "byte_offset", "component", "origin",
                            "value"), "dependency")
        require(item["kind"] in FACT_KINDS and item["origin"] in FACT_ORIGINS and
                is_uint(item["index"], 0xffffffff) and
                is_uint(item["byte_offset"], 0xffffffff) and
                is_uint(item["component"], 3), "dependency reflection is invalid")
        if item["kind"] == "texture_descriptor":
            require(is_uint(item["index"], 17) and is_uint(item["byte_offset"], 32764) and
                    item["byte_offset"] % 4 == 0 and item["component"] == 0,
                    "texture descriptor dependency key is invalid")
            texture_descriptor_value(item["value"], "texture descriptor dependency")
        identity = (item["kind"], item["index"], item["byte_offset"], item["component"])
        require(identity not in keys, "dependencies must be deduplicated")
        keys.add(identity)


def validate_texture_descriptor_dependencies(items, textures):
    # Every consumed descriptor fact must name a reflected bound texture at
    # the recorded slot and agree with the plan's descriptor, mirroring the
    # runtime admission.
    bound = {(item["cbuf_slot"], item["handle_byte_offset"]): item
             for item in textures if item["kind"] == "texture"}
    for item in items:
        if item["kind"] != "texture_descriptor":
            continue
        entry = bound.get((item["index"], item["byte_offset"]))
        require(entry is not None,
                "texture descriptor dependency names no bound texture")
        value = item["value"]
        descriptor_fields = ("dimension", "array", "multisample", "scalar",
                             "component_format", "normalized_coordinates", "comparison")
        require(all(value[name] == entry[name] for name in descriptor_fields),
                "texture descriptor dependency disagrees with the bound texture")


def validate_resource_dependencies_v3(dependencies, resources):
    by_key = {}
    for resource in resources:
        by_key.setdefault((resource["cbuf_slot"], resource["handle_byte_offset"]), []).append(resource)
    descriptor_fields = ("dimension", "array", "multisample", "scalar",
                         "component_format", "normalized_coordinates", "comparison")
    for dependency in dependencies:
        kind = dependency["kind"]
        if kind not in ("texture_descriptor", "texture_array_length"):
            continue
        entries = by_key.get((dependency["index"], dependency["byte_offset"]), [])
        require(entries and dependency["component"] == 0,
                "resource dependency names no bound resource")
        value = dependency["value"]
        if kind == "texture_descriptor":
            for entry in entries:
                fields = descriptor_fields if entry["kind"] == "texture" else (
                    "comparison", "normalized_coordinates")
                require(all(value[name] == entry[name] for name in fields),
                        "resource descriptor dependency disagrees with its provenance")
        else:
            require(is_uint(value, 0xffffffff) and value > 0 and
                    any(entry["array_length"] == value for entry in entries) and
                    all(entry["array_length"] in (1, value) for entry in entries),
                    "resource array-length dependency disagrees with its provenance")


def validate_vertex_inputs_v2(value, streams, stage):
    # Class-typed vertex inputs (location and float/uint/sint class) replace
    # per-slot stream formats; a stage uses one form or the other.
    require(isinstance(value, list) and len(value) <= 32, "vertex inputs are invalid")
    seen = set()
    for item in value:
        require_keys(item, ("location", "class"), "vertex input")
        require(is_uint(item["location"], 31) and item["location"] not in seen and
                item["class"] in ("float", "uint", "sint"), "vertex input is invalid")
        seen.add(item["location"])
    require(not value or (stage == "vertex" and not streams),
            "vertex inputs are only valid for vertex stages without stream formats")


# Render targets are written as float or as raw 32-bit integers; narrower or
# floating-64 scalars have no admitted render-target class.
COLOR_OUTPUT_SCALARS = {"f32", "u32", "s32"}


def validate_color_outputs(items, label):
    require(isinstance(items, list) and len(items) <= 8,
            f"{label} must be an array of at most 8 entries")
    locations = set()
    for item in items:
        require_keys(item, ("location", "component_mask", "scalar", "components"), label)
        require(is_uint(item["location"], 7) and item["location"] not in locations and
                is_uint(item["component_mask"], 15) and item["component_mask"] != 0 and
                item["scalar"] in COLOR_OUTPUT_SCALARS and
                is_uint(item["components"], 4) and item["components"] != 0 and
                (item["component_mask"] >> item["components"]) == 0,
                f"{label} reflection is invalid")
        locations.add(item["location"])
    return locations


def validate_interface_v2(value, stage, streams, textures, dependencies,
                          fragment_output_mode=None):
    keys = ("position_input_mask", "position_output_mask", "color_output_mask",
            "color_outputs", "writes_fragment_depth", "writes_sample_mask",
            "generic_inputs", "generic_outputs")
    require_keys(value, keys, "interface")
    require(is_uint(value["position_input_mask"], 15) and
            is_uint(value["position_output_mask"], 15) and
            is_uint(value["color_output_mask"], 0xffffffff),
            "interface masks are invalid")
    validate_generic_v2(value["generic_inputs"], "generic_inputs")
    validate_generic_v2(value["generic_outputs"], "generic_outputs")
    color_locations = validate_color_outputs(value["color_outputs"], "color_outputs")
    require(isinstance(value["writes_fragment_depth"], bool) and
            isinstance(value["writes_sample_mask"], bool),
            "fragment depth and sample-mask writes must be Booleans")
    # The summary mask is exactly the union of the per-target records, packed
    # at four bits per render target like the SPH export mask.
    summary = 0
    for item in value["color_outputs"]:
        summary |= item["component_mask"] << (4 * item["location"])
    require(summary == value["color_output_mask"],
            "color outputs disagree with the summary mask")
    if stage == "compute":
        require(value["position_input_mask"] == 0 and value["position_output_mask"] == 0 and
                value["color_output_mask"] == 0 and not value["color_outputs"] and
                not value["writes_fragment_depth"] and not value["writes_sample_mask"] and
                not value["generic_inputs"] and not value["generic_outputs"],
                "compute interface has graphics inputs or outputs")
        return False
    if stage == "vertex":
        # The runtime binds vertex-stage texture/sampler pairs like fragment
        # ones, and links every generic output against the consuming fragment
        # inputs, so flat, perspective and linear outputs are all admissible.
        require(value["position_input_mask"] == 0 and value["position_output_mask"] == 15 and
                value["color_output_mask"] == 0 and not value["color_outputs"] and
                not value["writes_fragment_depth"] and not value["writes_sample_mask"] and
                len(value["generic_inputs"]) == len(streams),
                "vertex interface is outside the supported profile")
        inputs = {item["slot"]: item for item in value["generic_inputs"]}
        for stream in streams:
            slot = stream["slot"] if "slot" in stream else stream["location"]
            require(slot in inputs and
                    ("slot" not in stream or inputs[slot]["component_mask"] & ~7 == 0),
                    "vertex streams do not cover the reflected inputs")
    else:
        perspective = any(item["interpolation"] == "perspective"
                          for item in value["generic_inputs"])
        require(value["position_output_mask"] == 0 and not value["generic_outputs"] and
                value["position_input_mask"] in (0, 8),
                "fragment interface is outside the supported profile")
        # Every written target consumed its fragment_output_type fact exactly
        # once, and each record must restate the type that fact proved.
        facts = {}
        for item in dependencies:
            if item["kind"] == "fragment_output_type":
                require(item["index"] not in facts and item["byte_offset"] == 0 and
                        item["component"] == 0,
                        "fragment_output_type dependencies must be unique per target")
                facts[item["index"]] = item
        require(set(facts) == color_locations,
                "written colour targets and fragment_output_type facts disagree")
        for item in value["color_outputs"]:
            fact = facts[item["location"]]
            require(fact["value"] == {"scalar": item["scalar"],
                                      "components": item["components"]},
                    "colour output disagrees with its fragment_output_type fact")
            if fragment_output_mode == "render-target-v1":
                require(item["scalar"] == "u32" and item["components"] == 4,
                        "render-target-v1 requires uint4 raw colour outputs")
        return perspective
    return False


def validate_reflection_v2(metadata, program, source):
    v3 = metadata.get("schema") == "armigo-shader-stage-v3"
    keys = ("schema", "source_version", "program_key", "program_size", "program_file",
            "source_file", "source_digest", "stage", "entry_point", "vertex_depth",
            "fragment_position", "c1_digest", "profile", "constant_buffers",
            "support_buffers", "textures", "vertex_streams", "vertex_inputs", "interface",
            "dependencies")
    required = set(keys) | ({"argument_buffer"} if v3 else set())
    optional = {"fragment_output_mode", "depth_bias_mode", "y_direction", "point_size", "point_coord"} | (
        {"storage_buffers", "custom_border", "draw_support", "vertex_fetch", "compute"} if v3 else set())
    require(isinstance(metadata, dict) and required <= set(metadata) and
            set(metadata) <= required | optional,
            "reflection has an unexpected schema")
    require(metadata["schema"] in ("armigo-shader-stage-v2", "armigo-shader-stage-v3"),
            "reflection schema is unsupported")
    for name in ("source_version", "program_key", "source_digest"):
        require(isinstance(metadata[name], str) and len(metadata[name]) == 64 and
                all(c in HEX for c in metadata[name]), f"reflection {name} is invalid")
    require(metadata["c1_digest"] is None or
            (isinstance(metadata["c1_digest"], str) and len(metadata["c1_digest"]) == 64 and
             all(c in HEX for c in metadata["c1_digest"])), "reflection c1_digest is invalid")
    require(metadata["program_file"] == "program.bin" and
            metadata["source_file"] == "source.metal", "reflection filenames are invalid")
    require(metadata["program_size"] == len(program) and
            metadata["program_key"] == digest(program), "program identity does not match reflection")
    require(metadata["source_digest"] == digest(source), "source digest does not match reflection")
    stage = metadata["stage"]
    require(stage in STAGE_TYPES or (v3 and stage == "compute"),
            "reflection stage is unsupported")
    require(("compute" in metadata) == (stage == "compute"),
            "compute launch reflection does not match the stage")
    fragment_output_mode = metadata.get("fragment_output_mode")
    if "fragment_output_mode" in metadata:
        require(stage == "fragment" and fragment_output_mode == "render-target-v1",
                "fragment output mode is invalid")
    if "depth_bias_mode" in metadata:
        require(stage == "fragment" and metadata["depth_bias_mode"] == "shader-v1",
                "depth bias mode is invalid")
    if "y_direction" in metadata:
        require(metadata["y_direction"] == "function-constant-v1",
                "y-direction mode is invalid")
    if "point_size" in metadata:
        require(stage == "vertex" and metadata["point_size"] in ("fixed-one-v1", "program-v1"),
                "point-size mode is invalid")
    if "point_coord" in metadata:
        require(stage == "fragment" and metadata["point_coord"] == "attribute-v1",
                "point-coordinate mode is invalid")
    require(len(program) >= PREFIX_SIZE, "program is shorter than the identity prefix")
    require(len(program) <= MAX_PROGRAM_SIZE, "program exceeds the identity size policy")
    # Compute programs carry no SPH header; their stage comes from the QMD.
    if stage != "compute":
        sph = struct.unpack_from("<I", program)[0]
        require(((sph >> 10) & 15) == STAGE_TYPES[stage], "program SPH stage mismatch")
    validate_profile(metadata["profile"])
    if v3:
        # Profile version 3 adds the armigo_y_negate function constant for
        # stages that read S2R SR_YDIRECTION.
        require(metadata["profile"] == {"identity": "armigo-macos-metal4", "version": 3},
                "reflection host profile is invalid")
    if stage == "vertex":
        require(metadata["entry_point"] == "armigo_vertex_main" and
                metadata["vertex_depth"] in ("minus-one-to-one", "zero-to-one") and
                metadata["fragment_position"] == "unspecified",
                "vertex explicit options are invalid")
    elif stage == "compute":
        # The launch specialization (local size, static shared memory) is part
        # of the artifact identity and must match the queue meta data exactly.
        launch = metadata["compute"]
        require(isinstance(launch, dict) and set(launch) == {"local_size", "shared_memory_bytes"},
                "compute launch reflection has an unexpected schema")
        local_size = launch["local_size"]
        require(isinstance(local_size, list) and len(local_size) == 3 and
                all(is_uint(value, 1024) and value >= 1 for value in local_size) and
                is_uint(launch["shared_memory_bytes"], 49152),
                "compute launch reflection is invalid")
        require(metadata["entry_point"] == "armigo_compute_main" and
                metadata["vertex_depth"] == "unspecified" and
                metadata["fragment_position"] == "unspecified" and
                metadata["vertex_streams"] == [] and metadata["vertex_inputs"] == [],
                "compute explicit options are invalid")
    else:
        require(metadata["entry_point"] == "armigo_fragment_main" and
                metadata["vertex_depth"] == "unspecified" and
                metadata["fragment_position"] in ("unspecified", "reciprocal-clip-w") and
                metadata["vertex_streams"] == [],
                "fragment explicit options are invalid")
    constant_buffer_indices = validate_constant_buffers_v2(metadata["constant_buffers"])
    validate_support_buffers_v2(metadata["support_buffers"], constant_buffer_indices)
    storage_buffer_indices = (validate_storage_buffers_v3(
        metadata.get("storage_buffers", []), metadata["constant_buffers"])
        if v3 else set())
    argument_buffer_index = (validate_argument_buffer_v3(metadata["argument_buffer"])
                             if v3 else None)
    if v3:
        validate_textures_v3(metadata["textures"], metadata["argument_buffer"])
    else:
        validate_textures_v2(metadata["textures"])
    validate_vertex_streams(metadata["vertex_streams"])
    validate_vertex_inputs_v2(metadata["vertex_inputs"], metadata["vertex_streams"], stage)
    vertex_buffer_indices = set(range(18, 18 + len(metadata["vertex_streams"])))
    require(all(index < 31 for index in vertex_buffer_indices),
            "vertex stream buffer indices exceed the Metal buffer namespace")
    support_buffer_indices = {item["metal_buffer_index"]
                              for item in metadata["support_buffers"]}
    reserved_buffer_indices = (constant_buffer_indices | support_buffer_indices |
                               storage_buffer_indices | vertex_buffer_indices)
    if "draw_support" in metadata:
        support = metadata["draw_support"]
        require(isinstance(support, dict) and set(support) == {
            "schema", "version", "byte_extent", "alignment", "metal_buffer_index"},
            "draw support has an unexpected schema")
        index = support["metal_buffer_index"]
        require(stage == "vertex" and support["schema"] == "armigo-draw-parameters" and
                type(support["version"]) is int and support["version"] == 1 and
                type(support["byte_extent"]) is int and support["byte_extent"] == 4 and
                type(support["alignment"]) is int and support["alignment"] == 4 and
                type(index) is int and 0 <= index < 31,
                "draw support ABI is invalid")
        require(index not in reserved_buffer_indices,
                "draw support overlaps a reflected buffer")
        reserved_buffer_indices.add(index)
    if "vertex_fetch" in metadata:
        fetch = metadata["vertex_fetch"]
        require(isinstance(fetch, dict) and set(fetch) == {
            "schema", "version", "byte_extent", "alignment", "metal_buffer_index"},
            "vertex fetch has an unexpected schema")
        index = fetch["metal_buffer_index"]
        require(stage == "vertex" and fetch["schema"] == "armigo-vertex-fetch" and
                type(fetch["version"]) is int and fetch["version"] == 1 and
                type(fetch["byte_extent"]) is int and fetch["byte_extent"] == 384 and
                type(fetch["alignment"]) is int and fetch["alignment"] == 8 and
                type(index) is int and 0 <= index < 31 and not vertex_buffer_indices,
                "vertex fetch ABI is invalid")
        require(index not in reserved_buffer_indices,
                "vertex fetch table overlaps a reflected buffer")
        reserved_buffer_indices.add(index)
    require(not storage_buffer_indices.intersection(
                constant_buffer_indices | support_buffer_indices | vertex_buffer_indices),
            "storage buffer overlaps a reserved buffer index")
    if argument_buffer_index is not None:
        require(argument_buffer_index not in reserved_buffer_indices and
                not ("depth_bias_mode" in metadata and argument_buffer_index == 30),
                "argument buffer overlaps a reserved buffer index")
    if "depth_bias_mode" in metadata:
        require(30 not in reserved_buffer_indices,
                "depth bias buffer overlaps a reflected buffer")
    if "custom_border" in metadata:
        border = metadata["custom_border"]
        require(stage in ("fragment", "vertex") and isinstance(border, dict) and
                set(border) == {"abi", "buffer_index", "pairs"} and
                border["abi"] == "coverage-v1" and
                type(border["buffer_index"]) is int and 0 <= border["buffer_index"] < 31,
                "custom border support ABI is invalid")
        require(border["buffer_index"] not in reserved_buffer_indices and
                border["buffer_index"] != argument_buffer_index and
                not ("depth_bias_mode" in metadata and border["buffer_index"] == 30),
                "custom border support overlaps a reflected buffer")
        resources = {item["resource_id"]: item for item in metadata["textures"]}
        require(isinstance(border["pairs"], list) and 0 < len(border["pairs"]) <= 4096,
                "custom border pairs are invalid")
        seen = set()
        for pair in border["pairs"]:
            require(isinstance(pair, dict) and set(pair) == {"texture", "sampler"} and
                    type(pair["texture"]) is int and type(pair["sampler"]) is int,
                    "custom border pair is invalid")
            texture, sampler = resources.get(pair["texture"], {}), resources.get(pair["sampler"], {})
            require(texture.get("kind") == "texture" and texture.get("resource_kind") == "sampled" and
                    texture.get("dimension") == "d2" and not texture.get("array") and
                    not texture.get("multisample") and texture.get("scalar") == "f32" and
                    texture.get("component_format") != "depth" and
                    sampler.get("kind") == "sampler" and
                    all(item.get("array_length") == 1 and item.get("normalized_coordinates") and
                        not item.get("comparison") for item in (texture, sampler)) and
                    (texture.get("cbuf_slot"), texture.get("handle_byte_offset")) ==
                    (sampler.get("cbuf_slot"), sampler.get("handle_byte_offset")) and
                    pair["texture"] not in seen,
                    "custom border pair is outside the admitted sampling profile")
            seen.add(pair["texture"])
    require(not vertex_buffer_indices.intersection(
                constant_buffer_indices | support_buffer_indices),
            "constant/support buffer indices overlap vertex streams")
    validate_dependencies(metadata["dependencies"])
    for item in metadata["dependencies"]:
        # Launch facts belong to compute stages and restate their launch reflection.
        if item["kind"] == "compute_local_size":
            require(stage == "compute" and item["value"] == metadata["compute"]["local_size"],
                    "compute local size dependency disagrees with the launch reflection")
        if item["kind"] == "compute_shared_memory_bytes":
            require(stage == "compute" and
                    item["value"] == metadata["compute"]["shared_memory_bytes"],
                    "compute shared memory dependency disagrees with the launch reflection")
    if v3:
        validate_resource_dependencies_v3(metadata["dependencies"], metadata["textures"])
    else:
        validate_texture_descriptor_dependencies(
            metadata["dependencies"],
            [item for item in metadata["textures"] if item["kind"] == "texture"])
    perspective = validate_interface_v2(metadata["interface"], stage,
                                        metadata["vertex_streams"] or metadata["vertex_inputs"],
                                        [item for item in metadata["textures"]
                                         if item["kind"] == "texture"],
                                        metadata["dependencies"], fragment_output_mode)
    if "depth_bias_mode" in metadata:
        require(not metadata["interface"]["writes_fragment_depth"],
                "shader depth bias cannot replace a guest fragment-depth export")
    if stage == "fragment":
        needs_position = metadata["interface"]["position_input_mask"] == 8 or perspective
        require((needs_position and metadata["fragment_position"] == "reciprocal-clip-w") or
                (not needs_position and metadata["fragment_position"] == "unspecified"),
                "fragment position option does not match its interface")
    return stage


def validate_reflection_compute(metadata, program, source):
    # Compute stages carry no SPH header: identity is the raw program bytes.
    # Constant buffers are not reflected - the queue meta data publishes the
    # eight enabled slots with address and size, and slot i binds Metal buffer
    # i. Only the bindings derived offline are reflected here.
    keys = ("schema", "source_version", "program_key", "program_size", "program_file",
            "source_file", "source_digest", "stage", "entry_point", "local_size",
            "shared_memory_bytes", "profile", "storage_buffers", "textures")
    require_keys(metadata, keys, "reflection")
    require(metadata["schema"] == "armigo-shader-stage-compute-v1",
            "reflection schema is unsupported")
    for name in ("source_version", "program_key", "source_digest"):
        require(isinstance(metadata[name], str) and len(metadata[name]) == 64 and
                all(c in HEX for c in metadata[name]), f"reflection {name} is invalid")
    require(metadata["program_file"] == "program.bin" and
            metadata["source_file"] == "source.metal", "reflection filenames are invalid")
    require(metadata["program_size"] == len(program) and
            metadata["program_key"] == digest(program),
            "program identity does not match reflection")
    require(metadata["source_digest"] == digest(source), "source digest does not match reflection")
    require(metadata["stage"] == "compute", "compute reflection stage is invalid")
    require(metadata["entry_point"] == "armigo_compute_main",
            "compute entry point is invalid")
    require(metadata["profile"] == {"identity": "armigo-macos-metal4", "version": 3},
            "compute host profile is invalid")
    local_size = metadata["local_size"]
    require(isinstance(local_size, list) and len(local_size) == 3 and
            all(is_uint(value, 65535) and value >= 1 for value in local_size),
            "compute local size is invalid")
    require(is_uint(metadata["shared_memory_bytes"], 32768),
            "compute shared memory size is invalid")
    # Storage-buffer descriptors live inside queue-meta-data constant buffers
    # (slots 0..7); fit against the live size is a runtime duty.
    require(isinstance(metadata["storage_buffers"], list) and
            len(metadata["storage_buffers"]) <= 31, "storage_buffers must be an array")
    metals, descriptors = set(), set()
    for item in metadata["storage_buffers"]:
        require_keys(item, ("cbuf_slot", "descriptor_byte_offset", "metal_buffer_index",
                            "access", "alignment"), "storage buffer")
        slot, offset, metal = (item["cbuf_slot"], item["descriptor_byte_offset"],
                               item["metal_buffer_index"])
        require(is_uint(slot, 3) and is_uint(offset, 65524) and offset % 8 == 0 and
                is_uint(metal, 30) and metal not in metals and metal >= 8 and
                (slot, offset) not in descriptors and
                item["access"] in ("read", "write", "read_write") and
                is_uint(item["alignment"], 4) and item["alignment"] == 4,
                "compute storage-buffer reflection is invalid")
        metals.add(metal)
        descriptors.add((slot, offset))
    require(isinstance(metadata["textures"], list) and len(metadata["textures"]) <= 32,
            "textures must be an array of at most 32 entries")
    texture_handles, sampler_handles = set(), set()
    for item in metadata["textures"]:
        require(isinstance(item, dict) and "kind" in item, "texture entry is invalid")
        kind = item["kind"]
        require(kind in ("texture", "sampler"), "texture kind is unsupported")
        require(is_uint(item["cbuf_slot"], 3) and item["handle_byte_offset"] % 4 == 0 and
                is_uint(item["handle_byte_offset"], 65532),
                "compute texture handle location is invalid")
        if kind == "texture":
            require_keys(item, ("cbuf_slot", "handle_byte_offset", "kind", "dimension",
                                "array", "multisample", "scalar", "component_format",
                                "normalized_coordinates", "comparison",
                                "texture_index"), "texture")
            require(is_uint(item["texture_index"], 15) and
                    (item["cbuf_slot"], item["handle_byte_offset"]) not in texture_handles,
                    "compute texture binding is invalid")
            texture_handles.add((item["cbuf_slot"], item["handle_byte_offset"]))
        else:
            require_keys(item, ("cbuf_slot", "handle_byte_offset", "kind", "comparison",
                                "normalized_coordinates", "sampler_index"), "sampler")
            require(is_uint(item["sampler_index"], 15) and
                    (item["cbuf_slot"], item["handle_byte_offset"]) not in sampler_handles,
                    "compute sampler binding is invalid")
            sampler_handles.add((item["cbuf_slot"], item["handle_byte_offset"]))
    return "compute"


def variant_data(metadata):
    if metadata["schema"] == "armigo-shader-stage-compute-v1":
        # The variant identity carries the launch specialization: two variants
        # of one program differing in local size or shared memory are distinct.
        return {
            "schema": metadata["schema"],
            "program_key": metadata["program_key"],
            "stage": metadata["stage"],
            "entry_point": metadata["entry_point"],
            "profile": metadata["profile"],
            "local_size": metadata["local_size"],
            "shared_memory_bytes": metadata["shared_memory_bytes"],
            "storage_buffers": metadata["storage_buffers"],
            "textures": metadata["textures"],
        }
    if metadata["schema"] in ("armigo-shader-stage-v2", "armigo-shader-stage-v3"):
        # Ported stages specialize on their consumed facts, host profile,
        # streams, bindings and C1 use; evidence labels stay out of the
        # identity.
        result = {
            "fragment_position": metadata["fragment_position"],
            "program_key": metadata["program_key"],
            "stage": metadata["stage"],
            "vertex_depth": metadata["vertex_depth"],
            "schema": metadata["schema"],
            "c1_digest": metadata["c1_digest"],
            "profile": metadata["profile"],
            "vertex_streams": metadata["vertex_streams"],
            "constant_buffers": metadata["constant_buffers"],
            "support_buffers": metadata["support_buffers"],
            "vertex_inputs": metadata["vertex_inputs"],
            "dependencies": [{key: item[key] for key in
                              ("kind", "index", "byte_offset", "component", "value")}
                             for item in metadata["dependencies"]],
        }
        if "fragment_output_mode" in metadata:
            result["fragment_output_mode"] = metadata["fragment_output_mode"]
        if "depth_bias_mode" in metadata:
            result["depth_bias_mode"] = metadata["depth_bias_mode"]
        if "y_direction" in metadata:
            result["y_direction"] = metadata["y_direction"]
        if "point_size" in metadata:
            result["point_size"] = metadata["point_size"]
        if "point_coord" in metadata:
            result["point_coord"] = metadata["point_coord"]
        if "custom_border" in metadata:
            result["custom_border"] = metadata["custom_border"]
        if "draw_support" in metadata:
            result["draw_support"] = metadata["draw_support"]
        if "vertex_fetch" in metadata:
            result["vertex_fetch"] = metadata["vertex_fetch"]
        if "compute" in metadata:
            result["compute"] = metadata["compute"]
        if metadata["schema"] == "armigo-shader-stage-v3":
            result["argument_buffer"] = metadata["argument_buffer"]
            result["storage_buffers"] = metadata.get("storage_buffers", [])
            result["textures"] = metadata["textures"]
        return result
    return {
        "fragment_position": metadata["fragment_position"],
        "program_key": metadata["program_key"],
        "stage": metadata["stage"],
        "vertex_attributes": metadata["vertex_attributes"],
        "vertex_depth": metadata["vertex_depth"],
    }


def load_stage(path):
    root = Path(path)
    require(root.is_dir(), "--stage must name a directory")
    program = read_regular(root / "program.bin", MAX_PROGRAM_SIZE, "program.bin")
    source = read_regular(root / "source.metal", MAX_SOURCE_SIZE, "source.metal")
    reflection_bytes = read_regular(root / "reflection.json", MAX_REFLECTION_SIZE,
                                    "reflection.json")
    try:
        metadata = load_json(reflection_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError("reflection.json is invalid JSON") from error
    stage = validate_reflection(metadata, program, source)
    variant_text = canonical_json(variant_data(metadata))
    return {"root": root, "program": program, "source": source,
            "reflection": reflection_bytes, "metadata": metadata, "stage": stage,
            "program_key": digest(program), "prefix_key": digest(program[:PREFIX_SIZE]),
            "variant_id": digest(variant_text.encode("utf-8"))}


def canonical_stage_roots(paths):
    require(len(paths) <= MAX_IDENTITIES, "stage count exceeds the identity policy")
    roots = {}
    total = 0
    for path in paths:
        root = Path(path).resolve(strict=True)
        require(root.is_dir(), "--stage must name a directory")
        roots[str(root)] = root
    for root in roots.values():
        for name, maximum in (("program.bin", MAX_PROGRAM_SIZE),
                              ("source.metal", MAX_SOURCE_SIZE),
                              ("reflection.json", MAX_REFLECTION_SIZE)):
            path = root / name
            require(path.is_file() and not path.is_symlink(), f"{name} must be a regular file")
            size = path.stat().st_size
            require(size <= maximum, f"{name} exceeds its size policy")
            total += size
            require(total <= MAX_INPUT_BYTES, "captured stage inputs exceed the aggregate policy")
    return list(roots.values())


def validate_inventory(stages):
    unique = {}
    for stage in stages:
        old = unique.get(stage["variant_id"])
        if old is None:
            unique[stage["variant_id"]] = stage
        else:
            require(old["program"] == stage["program"] and old["source"] == stage["source"] and
                    old["reflection"] == stage["reflection"],
                    "duplicate variant ID has conflicting contents")
    values = list(unique.values())
    identities = {(item["stage"], item["program_key"], len(item["program"]))
                  for item in values}
    require(len(identities) <= MAX_IDENTITIES, "identity count exceeds the policy")
    by_stage = sorted(values, key=lambda item: (item["stage"], item["program"]))
    for left, right in zip(by_stage, by_stage[1:]):
        if left["stage"] == right["stage"] and len(left["program"]) < len(right["program"]):
            require(not right["program"].startswith(left["program"]),
                    "same-stage program is a byte prefix of another")
    buckets = {}
    for item in values:
        key = (item["stage"], item["prefix_key"])
        buckets.setdefault(key, set()).add((item["program_key"], len(item["program"])))
        require(len(buckets[key]) <= MAX_BUCKET, "identity prefix bucket exceeds the policy")
    return sorted(values, key=lambda item: item["variant_id"])


def is_within(path, root):
    return path == root or root in path.parents


def validate_output(output, input_roots):
    supplied = Path(output)
    require(not supplied.is_symlink(), "output cannot be a symbolic link")
    if supplied.exists():
        require(supplied.is_dir() and not any(supplied.iterdir()),
                "output must be new or empty")
    resolved = supplied.resolve(strict=False)
    for root in input_roots:
        require(not is_within(resolved, root), "output cannot be inside an input directory")
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / ".git").exists():
            raise RuntimeError("output cannot be inside a Git work tree")
    return resolved


def run_tool(arguments):
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    try:
        result = subprocess.run(arguments, check=False, capture_output=True, text=True,
                                timeout=COMMAND_TIMEOUT_SECONDS, env=environment)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Apple shader tool timed out") from error
    except OSError as error:
        raise RuntimeError("Cannot execute Apple shader tool") from error
    if result.returncode != 0:
        raise RuntimeError("Apple shader tool failed: " + result.stderr.strip()[:4096])


# The translator predeclares SSA temporaries and proves every read defined,
# including reads correlated with the guarding predicate of their definition;
# Clang's path-insensitive uninitialized-use heuristic cannot see that
# correlation, so it is not a gate here.
METAL_COMPILE_FLAGS = ("-std=metal4.0", "-mmacosx-version-min=26.0", "-Werror",
                       "-Wno-unused-variable", "-Wno-sometimes-uninitialized")
_compiler_identity = None
_compiler_identity_lock = threading.Lock()


def compiler_identity():
    """Digest of the Metal compiler and linker binaries and the compiler version."""
    global _compiler_identity
    with _compiler_identity_lock:
        if _compiler_identity is None:
            hasher = hashlib.sha256()
            for tool in ("metal", "metallib"):
                found = subprocess.run(["xcrun", "--sdk", "macosx", "--find", tool],
                                       capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS)
                require(found.returncode == 0, "Apple shader tool lookup failed")
                hasher.update(Path(found.stdout.strip()).read_bytes())
            version = subprocess.run(["xcrun", "--sdk", "macosx", "metal", "--version"],
                                     capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS)
            require(version.returncode == 0, "Apple shader tool version query failed")
            hasher.update(version.stdout.encode("utf-8"))
            _compiler_identity = hasher.digest()
        return _compiler_identity


def compile_cache_path(source):
    """Optional developer cache of compiled libraries, keyed by everything that
    determines the compiler output: tool binaries and version, flags and source."""
    root = os.environ.get("ARMIGO_SHADER_COMPILE_CACHE")
    if not root:
        return None
    hasher = hashlib.sha256(compiler_identity())
    hasher.update("\0".join(METAL_COMPILE_FLAGS).encode("utf-8") + b"\0")
    hasher.update(source)
    return Path(root) / (hasher.hexdigest() + ".metallib")


def compile_stage(item, work_root):
    work = work_root / item["variant_id"]
    work.mkdir()
    source = work / "source.metal"
    air = work / "shader.air"
    library = work / "shader.metallib"
    source.write_bytes(item["source"])
    cached = compile_cache_path(item["source"])
    if cached is not None and cached.is_file() and not cached.is_symlink() and \
            0 < cached.stat().st_size <= MAX_LIBRARY_SIZE:
        shutil.copyfile(cached, library)
        return item, library, library.stat().st_size, file_digest(library)
    run_tool(["xcrun", "--sdk", "macosx", "metal", *METAL_COMPILE_FLAGS,
              "-c", str(source), "-o", str(air)])
    require(air.is_file() and not air.is_symlink() and
            0 < air.stat().st_size <= MAX_LIBRARY_SIZE,
            "Metal AIR output violates the size policy")
    run_tool(["xcrun", "--sdk", "macosx", "metallib", str(air), "-o", str(library)])
    require(library.is_file() and not library.is_symlink(), "metallib output is missing")
    require(0 < library.stat().st_size <= MAX_LIBRARY_SIZE,
            "metallib output violates the size policy")
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_name(cached.name + "." + uuid.uuid4().hex + ".tmp")
        shutil.copyfile(library, temporary)
        os.replace(temporary, cached)
    return item, library, library.stat().st_size, file_digest(library)


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def publish(arguments):
    roots = canonical_stage_roots(arguments.stage)
    loaded = []
    actual_input_bytes = 0
    for root in roots:
        item = load_stage(root)
        actual_input_bytes += len(item["program"]) + len(item["source"]) + len(item["reflection"])
        require(actual_input_bytes <= MAX_INPUT_BYTES,
                "captured stage inputs exceed the aggregate policy")
        loaded.append(item)
    stages = validate_inventory(loaded)
    profile = chain.current_profile(arguments.translate_tool)
    require(all(item["metadata"]["source_version"] == profile["source_version"]
                for item in stages), "stage source version differs from translator")
    profile_text = canonical_json(profile)
    chain_value = chain.chain_id(profile_text)
    output = validate_output(arguments.output, roots)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (".armigo-package-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        work = staging / ".work"
        work.mkdir()
        artifact_bytes = sum(len(item["program"]) + len(item["reflection"]) for item in stages)
        require(artifact_bytes <= MAX_OUTPUT_ARTIFACT_BYTES,
                "package artifacts exceed the aggregate policy")
        entries = []
        destinations = {}
        for item in stages:
            directory = Path("stages") / item["variant_id"]
            destination = staging / directory
            destination.mkdir(parents=True)
            (destination / "program.bin").write_bytes(item["program"])
            (destination / "reflection.json").write_bytes(item["reflection"])
            destinations[item["variant_id"]] = (directory, destination)
        with ThreadPoolExecutor(max_workers=arguments.jobs) as executor:
            iterator = iter(stages)
            pending = set()
            for _ in range(min(len(stages), arguments.jobs * 2)):
                pending.add(executor.submit(compile_stage, next(iterator), work))
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    item, library_path, library_size, library_digest = future.result()
                    artifact_bytes += library_size
                    require(artifact_bytes <= MAX_OUTPUT_ARTIFACT_BYTES,
                            "package artifacts exceed the aggregate policy")
                    variant_id = item["variant_id"]
                    directory, destination = destinations[variant_id]
                    library_path.replace(destination / "shader.metallib")
                    entries.append({
                        "directory": directory.as_posix(), "library_digest": library_digest,
                        "library_size": library_size, "prefix_key": item["prefix_key"],
                        "program_key": item["program_key"], "program_size": len(item["program"]),
                        "reflection_digest": digest(item["reflection"]), "stage": item["stage"],
                        "variant_id": variant_id,
                    })
                    try:
                        pending.add(executor.submit(compile_stage, next(iterator), work))
                    except StopIteration:
                        pass
        entries.sort(key=lambda item: (item["stage"], item["program_key"], item["variant_id"]))
        manifest = {"schema": "armigo-metal-package-v1",
                    "source_version": profile["source_version"], "chain_id": chain_value,
                    "platform": "macos-metal4", "profile_text": profile_text,
                    "complete_inventory": False, "runnable": False, "entries": entries}
        manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
        require(len(manifest_bytes) <= MAX_OUTPUT_ARTIFACT_BYTES - artifact_bytes,
                "package artifacts exceed the aggregate policy")
        (staging / "manifest.json").write_bytes(manifest_bytes)
        shutil.rmtree(work)
        require(validate_output(arguments.output, roots) == output,
                "output boundary changed during package build")
        if output.exists():
            output.rmdir()
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", required=True, type=Path)
    parser.add_argument("--translate-tool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    result = parser.parse_args(argv)
    require(1 <= result.jobs <= 8, "--jobs must be between 1 and 8")
    return result


def main(argv=None):
    try:
        publish(parse_arguments(argv))
        return 0
    except (OSError, RuntimeError) as error:
        print(f"armigo-shader-build-package: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
