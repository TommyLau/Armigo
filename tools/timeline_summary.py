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
"""Summarize guest submissions or presentations without treating segments as frames.

GPU execution intervals come only from driver feedback, never callback latency.
A presentation groups guest fences since the preceding successful presentation
on the single replay queue. Multi-window attribution needs a separate identity.
"""

import argparse
import collections


def read_trace(path):
    events = collections.defaultdict(list)
    first = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            time_ns, name, key, value = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            if name == "timeline_start":
                continue
            first = time_ns if first is None else min(first, time_ns)
            events[name].append((time_ns, key, value))
    return events, first


def single(events, name):
    return {key: (time_ns, value) for time_ns, key, value in events[name]}


def execution_intervals(events):
    """Return complete interval lists by guest fence, and segment counts.

    None means at least one segment lacks usable feedback. An older trace's
    single GPU interval remains supported; commit-to-callback is not a fallback.
    """
    begin = single(events, "segment_gpu_begin")
    end = single(events, "segment_gpu_end")
    intervals = collections.defaultdict(list)
    counts = collections.Counter()
    for _, segment, fence in events["segment_owner"]:
        counts[fence] += 1
        if segment not in begin or segment not in end or end[segment][0] < begin[segment][0]:
            intervals[fence] = None
        elif intervals[fence] is not None:
            intervals[fence].append((begin[segment][0], end[segment][0]))
    for time_ns, fence, span in events["gpu_end"]:
        if fence not in counts and span > 0 and time_ns >= span:
            intervals[fence] = [(time_ns - span, time_ns)]
            counts[fence] = 1
    return intervals, counts


def interval_totals(intervals):
    if not intervals:
        return 0, 0
    ordered = sorted(intervals)
    begin, end = ordered[0]
    total = 0
    for next_begin, next_end in ordered[1:]:
        if next_begin > end:
            total += end - begin
            begin = next_begin
        end = max(end, next_end)
    return total + end - begin, max(end for _, end in ordered) - ordered[0][0]


def presentation_boundaries(events):
    calls = collections.defaultdict(collections.deque)
    result = []
    records = [(time, 0, fence, value) for time, fence, value in events["present_call"]]
    records += [(time, 1, fence, value) for time, fence, value in events["present_queued"]]
    for time, kind, fence, value in sorted(records):
        if kind == 0:
            calls[fence].append(time)
        elif calls[fence]:
            called = calls[fence].popleft()
            if value:
                result.append((called, fence))
    return sorted(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--frames", type=int, default=60, help="maximum guest units to report")
    parser.add_argument("--from-seconds", type=float, default=0.0)
    parser.add_argument("--presentations", action="store_true",
                        help="group guest submissions by successful single-queue presentations")
    args = parser.parse_args()
    events, first = read_trace(args.trace)
    submits = sorted(events["submit"])
    if not submits:
        print("no submit events in the trace")
        return 1
    submit_time = {fence: time for time, fence, _ in submits}
    intervals, counts = execution_intervals(events)
    first_commit = single(events, "first_commit")
    final_commit = single(events, "commit")
    encode_begin = single(events, "encode_begin")
    encode_end = single(events, "encode_end")
    units = []
    assigned = set()
    if args.presentations:
        previous_fence, previous_time = 0, submits[0][0]
        for time, fence in presentation_boundaries(events):
            if fence < previous_fence:
                print("nonmonotonic presentation fences: explicit queue/window identity required")
                return 1
            fences = [key for key in submit_time if previous_fence < key <= fence]
            units.append((time, previous_time, time, fence, fences))
            assigned.update(fences)
            previous_fence, previous_time = fence, time
        label = "guest presentations"
    else:
        for index, (time, fence, _) in enumerate(submits[:-1]):
            units.append((time, time, submits[index + 1][0], fence, [fence]))
        label = "guest submissions"
    threshold = (first or 0) + int(args.from_seconds * 1e9)
    units = [unit for unit in units if unit[0] >= threshold][:args.frames]
    if not units:
        print("no complete guest units in the selected window")
        return 1
    print(f"unit: {label}; GPU columns are feedback execution intervals, not hardware busy cycles")
    print(f"{'fence':>7} {'cadence_ms':>10} {'encode_ms':>10} {'first_ms':>9} "
          f"{'final_ms':>9} {'segments':>8} {'gpu_union':>10} {'gpu_span':>10}")
    sums = collections.Counter()
    timed = 0
    for _, start, end, fence, fences in units:
        origin = min((submit_time[key] for key in fences), default=start)
        firsts = [first_commit.get(key, final_commit.get(key, (0, 0)))[0] for key in fences]
        finals = [final_commit.get(key, (0, 0))[0] for key in fences]
        first_ms = (min(firsts) - origin) / 1e6 if firsts and all(firsts) else None
        final_ms = (max(finals) - origin) / 1e6 if finals and all(finals) else None
        encode_ms = sum(max(0, encode_end[key][0] - encode_begin[key][0]) / 1e6
                        for key in fences if key in encode_begin and key in encode_end)
        segment_count = sum(counts[key] for key in fences)
        complete = all(key in intervals and intervals[key] is not None for key in fences)
        union_ms = span_ms = None
        if complete:
            union, span = interval_totals([item for key in fences for item in intervals[key]])
            union_ms, span_ms = union / 1e6, span / 1e6
            sums["gpu_union"] += union_ms
            timed += 1
        cadence_ms = (end - start) / 1e6
        sums["cadence"] += cadence_ms
        sums["encode"] += encode_ms
        sums["segments"] += segment_count
        def display(value):
            return f"{value:.3f}" if value is not None else "n/a"
        print(f"{fence:7d} {cadence_ms:10.3f} {encode_ms:10.3f} {display(first_ms):>9} "
              f"{display(final_ms):>9} {segment_count:8d} {display(union_ms):>10} {display(span_ms):>10}")
    n = len(units)
    gpu = f"{sums['gpu_union'] / timed:.3f} ms" if timed else "n/a"
    print(f"average over {n} {label}: cadence {sums['cadence']/n:.3f} ms, "
          f"encode {sums['encode']/n:.3f} ms, segments {sums['segments']/n:.2f}, "
          f"GPU interval union {gpu} ({timed}/{n} units with complete timing)")
    if args.presentations:
        print(f"unattributed guest submissions in trace: {len(set(submit_time) - assigned)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
