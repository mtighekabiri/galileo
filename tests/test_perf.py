"""Opt-in performance probes for the paths the app leans on hardest.

These are measurements, not assertions: each test times a hot path and prints
a ``PERF`` line, so a change's cost or win can be read off a before/after run.
They make no timing assertions because wall time on shared CI is noise; the
regression safety net is the functional suite, and these numbers are for a
human comparing two runs on the same machine.

Skipped unless explicitly requested, so the ordinary suite stays fast:

    GALILEO_PERF=1 python -m pytest tests/test_perf.py -s -q
"""

import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import galileo_blend as blend
import galileo_core as core
import galileo_deflicker as deflicker

from conftest import make_background, make_texture

pytestmark = pytest.mark.skipif(os.environ.get("GALILEO_PERF") != "1",
                                reason="set GALILEO_PERF=1 to run perf probes")

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "Galileo_Insertion_Tool_1.0.0.py")


def timed(fn, reps=5, warmup=1):
    """Best-of-N wall time in seconds; best-of because the floor is the signal."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def report(name, seconds, extra=""):
    suffix = f"  ({extra})" if extra else ""
    print(f"\nPERF {name}: {seconds * 1000:.2f} ms{suffix}")


def test_module_import_time():
    """Cold import of the app module in a subprocess, side effects included."""
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('app', {APP_PATH!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")

    def run():
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    report("app module import (subprocess)", timed(run, reps=3))


def test_photometric_match_large_creative():
    """The blend pipeline at a print-resolution creative, default settings."""
    rng = np.random.default_rng(3)
    creative = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    base = make_background(1280, 720, seed=4)
    quad = np.float32([[300, 200], [900, 220], [880, 560], [320, 540]])
    settings = blend.BlendSettings()

    run = lambda: blend.photometric_match(creative, base, quad, settings)
    report("photometric_match 1920x1080 creative", timed(run),
           f"defaults: colour={settings.colour}, lighting={settings.lighting}, "
           f"sharpness={settings.sharpness}")


def test_full_composite_with_grain(logo_bgra):
    """One preview frame's worth of insert work: match, warp, grain."""
    base = make_background(1280, 720, seed=5)
    quad = np.float32([[300, 200], [900, 220], [880, 560], [320, 540]])
    region = core.Region(quad)
    settings = blend.BlendSettings()
    creative = cv2.resize(logo_bgra, (960, 640), interpolation=cv2.INTER_CUBIC)

    def run():
        styled = blend.photometric_match(creative, base, quad, settings)
        out = core.composite_region(base, styled, region)
        sigma = blend.grain_sigma_for(base, quad, settings)
        if sigma > 0.1:
            height, width = out.shape[:2]
            mask = core.quad_to_mask(quad, width, height)
            out = blend.add_grain(out, sigma, mask, seed=7)
        return out

    report("composite frame (match+warp+grain)", timed(run))


def test_deflicker_scan(clip_video):
    """The synchronous sample that runs on every video open."""
    path, _ = clip_video
    run = lambda: deflicker.scan(path)
    report("deflicker.scan", timed(run, reps=3),
           f"{deflicker.SCAN_FRAMES}-frame window requested")


def test_project_history_json():
    """Serialise a 900-frame tracking history the way a project save does."""
    from conftest import quad_path
    quads = quad_path(900)
    history = {i: np.asarray(q, np.float64) for i, q in enumerate(quads)}

    as_lists = core.history_to_lists(history)
    pretty = json.dumps(as_lists, indent=2)
    compact = json.dumps(as_lists, separators=(",", ":"))

    run_dump = lambda: json.dumps(as_lists, indent=2)
    run_load = lambda: json.loads(pretty)
    report("project history json.dumps", timed(run_dump),
           f"pretty={len(pretty):,} B vs compact={len(compact):,} B")
    report("project history json.loads", timed(run_load))


def test_region_geometry():
    """The geometry recomputed inside paint loops: boundary and homography."""
    quad = np.float32([[300, 200], [900, 220], [880, 560], [320, 540]])
    region = core.Region(quad, curved=True)
    region.set_control_point(0, 0, [480, 170])

    run_boundary = lambda: core.region_boundary(region, samples=32)
    report("region_boundary samples=32 x12 tiles",
           timed(lambda: [run_boundary() for _ in range(12)]))
    report("Region.homography x100",
           timed(lambda: [region.homography() for _ in range(100)]))
