"""Several insertions tracked and rendered at once.

A concourse or mall shot routinely has more than one screen worth filling, and
an A/B test wants different creatives in them. Each placement carries its own
shape, tracking, creative and tracker.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util

import cv2
import numpy as np
import pytest

import lumen_core as core
import lumen_blend as blend
from conftest import quad_path

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

_spec = importlib.util.spec_from_file_location(
    "lumen_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "LUMEN_Insertion_Tool_1.0.0.py"))
lumen_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lumen_app)


def creative(colour):
    bgr = np.full((120, 180, 3), colour, np.uint8)
    return np.dstack([bgr, np.full((120, 180), 255, np.uint8)])


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def app(qapp, clip_video):
    path, truth = clip_video
    window = lumen_app.MainWindow()
    window.show()
    window.central_panel.load_video(path)
    yield window, window.central_panel, window.central_panel.tracking_overlay, truth
    window.dirty = False
    window.close()


@pytest.fixture
def two(app):
    """Two placements on offset paths, each with its own creative."""
    window, panel, overlay, truth = app
    first = overlay.active
    first.name = "Left"
    first.overlay_bgra = creative((0, 0, 220))
    first.tracking_history = {i: [tuple(map(float, c)) for c in q]
                              for i, q in enumerate(truth)}
    first.points = list(first.tracking_history[0])

    second = overlay.add_placement("Right")
    second.overlay_bgra = creative((220, 120, 0))
    second.tracking_history = {
        i: [(x + 150.0, y + 20.0) for x, y in first.tracking_history[i]]
        for i in first.tracking_history}
    second.points = list(second.tracking_history[0])
    overlay.set_active(0)
    return window, panel, overlay, truth, first, second


class TestPlacementModel:
    def test_starts_with_exactly_one(self, app):
        window, panel, overlay, truth = app
        assert len(overlay.placements) == 1

    def test_adding_makes_the_new_one_active(self, app):
        window, panel, overlay, truth = app
        added = overlay.add_placement("Second")
        assert overlay.active is added
        assert overlay.active_index == 1

    def test_each_keeps_its_own_shape(self, two):
        window, panel, overlay, truth, first, second = two
        assert first.points != second.points

    def test_each_keeps_its_own_history(self, two):
        window, panel, overlay, truth, first, second = two
        assert first.tracking_history[3] != second.tracking_history[3]

    def test_the_last_placement_cannot_be_removed(self, app):
        window, panel, overlay, truth = app
        overlay.remove_placement(0)
        assert len(overlay.placements) == 1

    def test_removing_releases_its_video(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.remove_placement(1)
        assert len(overlay.placements) == 1
        assert overlay.active is first

    def test_active_index_is_clamped_after_removal(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        overlay.remove_placement(1)
        assert overlay.active_index == 0
        assert overlay.active is first

    def test_only_ready_placements_count(self, two):
        window, panel, overlay, truth, first, second = two
        assert len(overlay.ready_placements()) == 2
        second.enabled = False
        assert [p.name for p in overlay.ready_placements()] == ["Left"]

    def test_a_placement_without_a_creative_is_not_ready(self, app):
        window, panel, overlay, truth = app
        overlay.active.points = [tuple(map(float, p)) for p in truth[0]]
        assert overlay.ready_placements() == []


class TestDelegation:
    """The GUI was written against one insertion; delegation keeps it working."""

    def test_overlay_attributes_follow_the_active_placement(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        assert overlay.points == second.points
        overlay.set_active(0)
        assert overlay.points == first.points

    def test_writing_through_the_overlay_hits_the_active_placement(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        overlay.brightness = 42
        assert second.brightness == 42
        assert first.brightness != 42

    def test_panel_history_follows_the_active_placement(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        assert panel.tracking_history is second.tracking_history

    def test_tracker_state_is_per_placement(self, two):
        window, panel, overlay, truth, first, second = two
        panel.tracking_mode = True
        panel.read_frame()
        assert first.tracker is not None
        assert second.tracker is not None
        assert first.tracker is not second.tracker

    def test_feature_source_is_per_placement(self, two, monkeypatch):
        """One shot can hold a printed poster and a digital screen."""
        window, panel, overlay, truth, first, second = two
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        overlay.set_active(1)
        window.toggle_digital_screen(True)
        assert second.feature_source == core.PlanarTracker.SURROUND
        assert first.feature_source == core.PlanarTracker.INTERIOR


class TestTracking:
    def test_all_placements_advance_together(self, two):
        window, panel, overlay, truth, first, second = two
        for placement in (first, second):
            placement.tracking_history = {0: placement.tracking_history[0]}
        panel.tracking_mode = True
        for _ in range(4):
            panel.read_frame()
        assert len(first.tracking_history) >= 3
        assert len(second.tracking_history) >= 3

    def test_one_failing_does_not_disturb_the_other(self, two):
        """Independent trackers: a lost surface must not take the rest down."""
        window, panel, overlay, truth, first, second = two
        second.points = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
        panel.tracking_mode = True
        for _ in range(3):
            panel.read_frame()
        index = panel.get_current_frame_index()
        assert index in first.tracking_history


class TestRendering:
    def test_both_creatives_reach_the_file(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        out = str(tmp_path / "both.mp4")
        settings = window.build_render_settings(0, 9, 1.0, out)
        assert [p.name for p in settings.placements] == ["Left", "Right"]

        worker = lumen_app.RenderWorker(settings)
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses and statuses[0].startswith("Completed"), statuses

        capture = cv2.VideoCapture(out)
        for _ in range(5):
            ok, frame = capture.read()
        capture.release()
        assert ok

        for placement, channel in ((first, 2), (second, 0)):
            quad = np.float32(placement.tracking_history[4])
            mask = core.quad_to_mask(quad, frame.shape[1], frame.shape[0]) > 0
            mean = frame[mask].mean(axis=0)
            assert mean[channel] > 120, (
                f"{placement.name} creative missing from the render: {mean}")

    def test_a_disabled_placement_is_left_out(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        second.enabled = False
        settings = window.build_render_settings(0, 5, 1.0, str(tmp_path / "one.mp4"))
        assert [p.name for p in settings.placements] == ["Left"]

    def test_snapshots_carry_no_qt_objects(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        settings = window.build_render_settings(0, 5, 1.0, str(tmp_path / "x.mp4"))
        for snapshot in settings.placements:
            assert isinstance(snapshot.creative, np.ndarray)
            assert snapshot.creative is not first.overlay_bgra

    def test_untracked_placements_are_skipped_not_fatal(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        second.tracking_history = {}
        out = str(tmp_path / "skip.mp4")
        settings = window.build_render_settings(0, 5, 1.0, out)
        worker = lumen_app.RenderWorker(settings)
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses and statuses[0].startswith("Completed")


class TestProjectRoundTrip:
    def test_all_placements_survive_a_save_and_load(self, two, tmp_path, monkeypatch):
        window, panel, overlay, truth, first, second = two
        second.curved_enabled = True
        second.brightness = 33

        project = str(tmp_path / "multi.json")
        monkeypatch.setattr(lumen_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        monkeypatch.setattr(lumen_app.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        for name in ("information", "warning"):
            monkeypatch.setattr(lumen_app.QMessageBox, name,
                                staticmethod(lambda *a, **k: None))
        window.save_project()

        import json
        with open(project) as handle:
            data = json.load(handle)
        assert len(data["placements"]) == 2
        assert [p["name"] for p in data["placements"]] == ["Left", "Right"]

        window.load_project()
        restored = window.central_panel.tracking_overlay.placements
        assert [p.name for p in restored] == ["Left", "Right"]
        assert len(restored[1].tracking_history) == len(second.tracking_history)
        assert restored[1].brightness == 33
        assert restored[1].curved_enabled is True


class TestAoiPerPlacement:
    def test_one_file_per_placement(self, two, tmp_path, monkeypatch):
        """Eye-tracking has to tell the adverts apart."""
        window, panel, overlay, truth, first, second = two
        out = tmp_path / "aoi.csv"
        window.last_render = {
            "out_path": str(tmp_path / "stim.mp4"), "start_frame": 0,
            "end_frame": 9, "scale_factor": 1.0, "fps": 25.0,
            "width": 640, "height": 480}
        monkeypatch.setattr(lumen_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.save_aoi_geometry()

        produced = sorted(f.name for f in tmp_path.glob("aoi_*.csv"))
        assert produced == ["aoi_Left.csv", "aoi_Right.csv"]

    def test_each_file_names_its_placement_and_has_rows(self, two, tmp_path,
                                                        monkeypatch):
        window, panel, overlay, truth, first, second = two
        out = tmp_path / "aoi.csv"
        window.last_render = {
            "out_path": str(tmp_path / "stim.mp4"), "start_frame": 0,
            "end_frame": 9, "scale_factor": 1.0, "fps": 25.0,
            "width": 640, "height": 480}
        monkeypatch.setattr(lumen_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.save_aoi_geometry()

        import csv
        with open(tmp_path / "aoi_Right.csv", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows[0][0] == "name:Right"
        assert len(rows) > 2
        # The two placements are offset by 150px, so their AOIs must differ.
        with open(tmp_path / "aoi_Left.csv", newline="") as handle:
            left = list(csv.reader(handle))
        assert rows[2][-1] != left[2][-1]
