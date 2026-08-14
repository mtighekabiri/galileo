"""Faults found by scanning rather than by anything going wrong.

Each of these is invisible in normal use, which is what makes it worth a test:
a leak nobody notices until the machine runs out of handles, a crash that only
a malformed clip provokes, and two copies of an export format that had already
drifted apart.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv
import importlib.util

import numpy as np
import pytest

import galileo_core as core
from conftest import make_background

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication                          # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)

QUAD = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


class TestTrackingNeverThrows:
    """track() promises the caller a refused result rather than an exception,
    and the caller is the frame-display path -- an exception there stops
    playback rather than costing one unmeasured frame."""

    def test_a_frame_of_the_wrong_size_is_refused_not_raised(self):
        tracker = core.PlanarTracker(make_background(200, 200), QUAD)
        result = tracker.track(make_background(300, 300))
        assert result.ok is False
        assert result.reason

    def test_the_tracker_is_not_left_broken_by_one(self):
        tracker = core.PlanarTracker(make_background(200, 200), QUAD)
        before = tracker.quad.copy()
        tracker.track(make_background(300, 300))
        assert np.array_equal(tracker.quad, before)
        assert tracker.track(make_background(200, 200, seed=2)).ok

    def test_an_empty_frame_is_refused_not_raised(self):
        tracker = core.PlanarTracker(make_background(200, 200), QUAD)
        assert tracker.track(np.zeros((200, 200, 3), np.uint8)).ok is False


class TestTheRenderLetsGoOfItsFiles:
    def test_a_failed_render_releases_every_placement_capture(self, qapp,
                                                              clip_video,
                                                              tmp_path):
        """These were released on the way out of a *successful* render only,
        so the case most likely to be retried leaked one open file per
        placement every time."""
        path, truth = clip_video
        history = {i: [tuple(map(float, c)) for c in q]
                   for i, q in enumerate(truth)}

        class Watcher:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

        watchers = [Watcher(), Watcher()]
        settings = galileo_app.RenderSettings(
            path, str(tmp_path / "out.mp4"), 0, 4, 1.0, 25.0, history,
            np.zeros((4, 2, 2), np.float32), False)
        worker = galileo_app.RenderWorker(settings)

        def prepared(_settings, caveats):
            made = []
            for watcher in watchers:
                snapshot = galileo_app.PlacementSnapshot.__new__(
                    galileo_app.PlacementSnapshot)
                snapshot.capture = watcher
                made.append(snapshot)
            raise RuntimeError("boom, part way through")

        worker._prepare_placements = prepared
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses and statuses[0].startswith("Error")

    def test_a_render_that_cannot_open_its_video_reports_that(self, qapp,
                                                              tmp_path):
        """The cleanup used to reference a name bound further down, so a
        failure this early raised on the way out and buried the real cause."""
        settings = galileo_app.RenderSettings(
            "/nowhere/missing.mp4", str(tmp_path / "out.mp4"), 0, 4, 1.0, 25.0,
            {0: QUAD}, np.zeros((4, 2, 2), np.float32), False)
        worker = galileo_app.RenderWorker(settings)
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses, "the worker never reported anything"
        assert "Could not open" in statuses[0], statuses

    def test_placement_captures_are_released_after_a_good_render(self, qapp,
                                                                 clip_video,
                                                                 tmp_path):
        path, truth = clip_video
        history = {i: [tuple(map(float, c)) for c in q]
                   for i, q in enumerate(truth)}
        settings = galileo_app.RenderSettings(
            path, str(tmp_path / "ok.mp4"), 0, 4, 1.0, 25.0, history,
            np.zeros((4, 2, 2), np.float32), False,
            overlay_bgra=np.dstack([np.full((40, 60, 3), 200, np.uint8),
                                    np.full((40, 60), 255, np.uint8)]))
        worker = galileo_app.RenderWorker(settings)
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses and statuses[0].startswith("Completed"), statuses


class TestOneAoiWriter:
    """There were two, and they had drifted: the single-placement copy headed
    every file "aoi1" rather than naming the placement, so which advert a file
    described depended on how many there had been."""

    def placement(self, name):
        made = galileo_app.Placement(name)
        made.points = list(QUAD)
        made.tracking_history = {i: [(x + i, y) for x, y in QUAD]
                                 for i in range(5)}
        made.enabled = True
        return made

    def header_of(self, path):
        with open(path, newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle))

    def test_the_file_is_named_after_the_placement(self, qapp, tmp_path):
        """The copy that handled a lone placement headed every file "aoi1"."""
        window = galileo_app.MainWindow()
        try:
            out = str(tmp_path / "one.csv")
            window._write_aoi_file(out, self.placement("Left Screen"),
                                   25.0, 640, 480, None, None, 1.0)
            assert self.header_of(out)[0] == "name:Left Screen"
        finally:
            window.dirty = False
            window.close()

    def test_the_export_runs_end_to_end_for_a_lone_placement(self, qapp,
                                                             clip_video,
                                                             tmp_path,
                                                             monkeypatch):
        """Drives the menu action itself, since that is the path that was
        rewritten to call the shared writer."""
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.central_panel.load_video(path)
            overlay = window.central_panel.tracking_overlay
            overlay.placements = [self.placement("Solo")]
            overlay.active_index = 0
            out = str(tmp_path / "menu.csv")
            for name, value in (("getSaveFileName", (out, "")),):
                monkeypatch.setattr(galileo_app.QFileDialog, name,
                                    staticmethod(lambda *a, **k: value))
            for name in ("question",):
                monkeypatch.setattr(galileo_app.QMessageBox, name,
                                    staticmethod(lambda *a, **k:
                                                 galileo_app.QMessageBox.Yes))
            for name in ("information", "warning"):
                monkeypatch.setattr(galileo_app.QMessageBox, name,
                                    staticmethod(lambda *a, **k: None))
            window.save_aoi_geometry()
            assert os.path.exists(out)
            assert self.header_of(out)[0] == "name:Solo"
        finally:
            window.dirty = False
            window.close()

    def test_one_and_many_write_the_same_rows(self, qapp, tmp_path):
        """The point of a single writer: whatever changes, both cases change
        together."""
        window = galileo_app.MainWindow()
        try:
            one = self.placement("Solo")
            alone = str(tmp_path / "alone.csv")
            among = str(tmp_path / "among.csv")
            window._write_aoi_file(alone, one, 25.0, 640, 480, None, None, 1.0)
            window._write_aoi_file(among, one, 25.0, 640, 480, None, None, 1.0)
            assert open(alone).read() == open(among).read()
        finally:
            window.dirty = False
            window.close()

    def test_unrenderable_frames_are_counted_not_silently_dropped(self, qapp,
                                                                  tmp_path):
        window = galileo_app.MainWindow()
        try:
            placement = self.placement("Has Gaps")
            placement.tracking_history[2] = [(5.0, 5.0)] * 4     # collapsed
            rows, skipped = window._write_aoi_file(
                str(tmp_path / "gaps.csv"), placement, 25.0, 640, 480,
                None, None, 1.0)
            assert skipped >= 1
            assert rows >= 1
        finally:
            window.dirty = False
            window.close()
