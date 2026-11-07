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
"""Loss markers must never produce apparently complete timing summaries."""

import pathlib
import subprocess
import sys
import tempfile

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "tools" / "timeline_summary.py"
TRACE = """0\ttimeline_start\t0\t0
1000000000\tsubmit\t1\t0
1000000010\tencode_begin\t1\t0
1000000020\tencode_end\t1\t0
2000000000\tsubmit\t2\t0
2000000010\tencode_begin\t2\t0
2000000020\tencode_end\t2\t0
3000000000\tsubmit\t3\t0
"""


def run(path):
    return subprocess.run([sys.executable, str(SCRIPT), str(path), "--from-seconds", "0.5"],
                          capture_output=True, text=True, check=False)


def main():
    with tempfile.TemporaryDirectory(prefix="armigo-timeline-reader-") as directory:
        path = pathlib.Path(directory) / "trace.tsv"
        path.write_text(TRACE, encoding="utf-8")
        valid = run(path)
        assert valid.returncode == 0, valid.stderr
        assert "average over 1 guest submissions" in valid.stdout, valid.stdout
        for marker in ("timeline_overflow", "timeline_incomplete", "timeline_symbols_dropped", "timeline_unquiesced"):
            path.write_text(TRACE + f"0\t{marker}\t1\t0\n", encoding="utf-8")
            damaged = run(path)
            assert damaged.returncode != 0, marker
            assert "incomplete trace" in damaged.stderr and marker in damaged.stderr, damaged.stderr
            assert "average over" not in damaged.stdout and "unit:" not in damaged.stdout, damaged.stdout
    print("Timeline reader loss handling passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
