#!/usr/bin/env python3
# ============================================================================
# Armigo - Frame timeline summary
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================
"""Summarize driver pass samples from replay stderr, selecting submission fences."""

import argparse
import collections


def read_passes(path, first, frames):
    passes = []
    submissions = {}
    with open(path, encoding="utf-8", errors="replace") as source:
        for line in source:
            if not line.startswith(("nvn_pass ", "nvn_pass_frame ")):
                continue
            fields = dict(word.split("=", 1) for word in line.split()[1:] if "=" in word)
            fence = int(fields["frame"])
            if first <= fence < first + frames:
                if line.startswith("nvn_pass_frame "):
                    submissions[fence] = fields
                else:
                    passes.append(fields)
    return passes, submissions


def summarize(passes, frame_count, by_index=False):
    groups = collections.defaultdict(list)
    for p in passes:
        key = (p["type"], p["width"], p["height"], p["samples"], p["colors"], p["depth"], p["attachments"])
        if by_index:
            key = (p["pass"],) + key
        groups[key].append(p)
    for key, rows in sorted(groups.items(),
                            key=lambda item: sum(float(r["gpu_ms"]) for r in item[1]
                                                 if r["valid"] == "1"), reverse=True):
        valid = [r for r in rows if r["valid"] == "1"]
        total = sum(float(r["gpu_ms"]) for r in valid)
        draws = sum(int(r["draws"]) for r in rows)
        state = " ".join(f"{name}={sum(int(r[name]) for r in rows) / frame_count:.2f}"
                         for name in ("pipeline_binds", "depth_binds", "blend_changes",
                                      "blended_draws", "depth_draws"))
        label = "/".join(key)
        print(f"{label:42s} passes/frame={len(rows)/frame_count:6.2f} "
              f"draws/frame={draws/frame_count:8.2f} valid={len(valid)}/{len(rows)} "
              f"stage_ms/frame={total/frame_count:.6f} "
              f"stage_ms/pass={total/len(valid) if valid else float('nan'):.6f} {state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--first", type=int, required=True, help="first submission fence, not game frame")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    if args.first < 0 or args.frames <= 0:
        parser.error("first must be nonnegative and frames must be positive")
    passes, submissions = read_passes(args.log, args.first, args.frames)
    if not submissions:
        parser.exit(1, "no pass samples in the requested fence window\n")
    expected = sum(int(s["passes"]) for s in submissions.values())
    dropped = sum(int(s["dropped"]) for s in submissions.values())
    if expected != len(passes) or any(int(p["frame"]) not in submissions for p in passes):
        parser.exit(1, "incomplete pass records in the requested window; retain the complete log\n")
    print(f"observed submissions={len(submissions)} requested={args.frames} dropped_passes={dropped}; "
          "times are sampled stage spans, not exclusive GPU cost")
    print("by type/width/height/samples/colors/depth/attachments:")
    summarize(passes, len(submissions))
    print("by pass index/type/width/height/samples/colors/depth/attachments:")
    summarize(passes, len(submissions), by_index=True)
    print("Zero-draw clear/resolve timings are unmeasured. Invalid samples are excluded, not zero-cost.")


if __name__ == "__main__":
    main()
