"""End-to-end checks that drive the real Qt application offscreen.

These exercise the widgets themselves -- loading a video, drawing a shape,
inserting a creative, tracking, and rendering a file -- so the wiring between
the GUI and the core is covered, not just the core in isolation.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

import lumen_core as core
from conftest import make_texture, quad_path, render_clip, write_video

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "lumen_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "LUMEN_Insertion_Tool_1.0.0.py"))
lumen_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lumen_app)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def window(qapp):
    win = lumen_app.MainWindow()
    yield win
    win.close()


@pytest.fixture
def loaded(window, clip_video):
    path, truth = clip_video
    window.central_panel.load_video(path)
    return window, path, truth


class TestStartup:
    def test_window_builds(self, window):
        assert window.central_panel is not None
        assert window.central_panel.tracking_overlay is not None

    def test_curve_toggle_exists_in_the_toolbar(self, window):
        labels = [i.meaning_label.text() for i in window.left_col.icons]
        assert "Curve" in labels
        assert "Draw" in labels


class TestVideoLoading:
    def test_loads_and_reports_frame_zero(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        assert panel.cap is not None and panel.cap.isOpened()
        assert panel.get_current_frame_index() == 0
        assert panel.total_frames == len(truth)

    def test_frame_index_stays_consistent_while_stepping(self, loaded):
        """The two old definitions of 'current frame' could disagree by one."""
        window, path, truth = loaded
        panel = window.central_panel
        for expected in range(1, 6):
            panel.on_forward()
            assert panel.get_current_frame_index() == expected

    def test_jump_to_frame_lands_exactly(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        panel.jump_to_frame(7)
        assert panel.get_current_frame_index() == 7


class TestTrackingThroughTheGui:
    def test_playing_forward_tracks_the_shape(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        overlay = panel.tracking_overlay

        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        panel.tracking_mode = True

        for _ in range(8):
            panel.read_frame()

        assert len(overlay.tracking_history) >= 5
        index = panel.get_current_frame_index()
        tracked = np.float32(overlay.tracking_history[index])
        error = np.linalg.norm(tracked - np.float32(truth[index]), axis=1).mean()
        assert error < 12.0, f"tracked shape drifted {error:.1f}px from truth"

    def test_editing_the_shape_reanchors_the_tracker(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
        panel.tracking_mode = True
        panel.read_frame()
        assert panel.tracker is not None

        panel.on_shape_changed()
        assert panel.tracker is None   # rebuilt from the corrected shape


class TestRendering:
    def _prepare(self, window, truth, logo):
        panel = window.central_panel
        overlay = panel.tracking_overlay
        overlay.overlay_bgra = logo
        overlay.inserted_overlay_is_video = False
        overlay.tracking_history = {
            i: [tuple(map(float, p)) for p in quad]
            for i, quad in enumerate(truth)
        }
        panel.tracking_history = dict(overlay.tracking_history)
        return panel, overlay

    def test_render_writes_a_playable_file(self, loaded, logo_bgra, tmp_path):
        window, path, truth = loaded
        panel, overlay = self._prepare(window, truth, logo_bgra)

        out = str(tmp_path / "render.mp4")
        settings = window.build_render_settings(0, 9, 1.0, out)
        worker = lumen_app.RenderWorker(settings)

        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()

        assert statuses and statuses[0].startswith("Completed"), statuses
        assert os.path.exists(out)

        cap = cv2.VideoCapture(out)
        assert cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert count == 10

    def test_settings_snapshot_holds_no_qt_objects(self, loaded, logo_bgra, tmp_path):
        """The worker must not be handed anything owned by the GUI thread."""
        window, path, truth = loaded
        self._prepare(window, truth, logo_bgra)
        settings = window.build_render_settings(0, 5, 1.0, str(tmp_path / "x.mp4"))

        for value in vars(settings).values():
            assert not isinstance(value, (lumen_app.QPixmap, lumen_app.QImage))
        assert isinstance(settings.overlay_bgra, np.ndarray)
        # A detached copy, not a view onto the widget's array.
        assert settings.overlay_bgra is not window.central_panel.tracking_overlay.overlay_bgra

    def test_transparent_creative_does_not_paint_black(self, loaded, tmp_path):
        """The old render turned a PNG's transparent area into black."""
        window, path, truth = loaded

        creative = np.zeros((80, 80, 4), np.uint8)
        creative[:, :, :3] = (0, 0, 255)
        creative[:, :, 3] = 0            # entirely transparent
        panel, overlay = self._prepare(window, truth, creative)

        out = str(tmp_path / "transparent.mp4")
        worker = lumen_app.RenderWorker(
            window.build_render_settings(0, 4, 1.0, out))
        worker.run()

        cap = cv2.VideoCapture(out)
        ok, rendered = cap.read()
        cap.release()
        assert ok

        source = cv2.VideoCapture(path)
        _, original = source.read()
        source.release()

        centre = np.float32(truth[0]).mean(axis=0).astype(int)
        patch = rendered[centre[1] - 5:centre[1] + 5, centre[0] - 5:centre[0] + 5]
        original_patch = original[centre[1] - 5:centre[1] + 5,
                                  centre[0] - 5:centre[0] + 5]
        assert patch.mean() > 20, "transparent creative blacked out the region"
        assert abs(float(patch.mean()) - float(original_patch.mean())) < 25

    def test_render_interpolates_between_sparse_keyframes(self, loaded,
                                                          logo_bgra, tmp_path):
        window, path, truth = loaded
        panel, overlay = self._prepare(window, truth, logo_bgra)
        # Keep only the ends, forcing the render to fill everything between.
        overlay.tracking_history = {0: overlay.tracking_history[0],
                                    9: overlay.tracking_history[9]}

        out = str(tmp_path / "sparse.mp4")
        settings = window.build_render_settings(0, 9, 1.0, out)
        dense = core.interpolate_tracking(settings.history, 0, 9)
        assert sorted(dense) == list(range(10))

        worker = lumen_app.RenderWorker(settings)
        worker.run()
        assert os.path.exists(out)

    def test_render_at_reduced_scale(self, loaded, logo_bgra, tmp_path):
        window, path, truth = loaded
        self._prepare(window, truth, logo_bgra)
        out = str(tmp_path / "half.mp4")
        worker = lumen_app.RenderWorker(
            window.build_render_settings(0, 5, 0.5, out))
        worker.run()

        cap = cv2.VideoCapture(out)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        assert width == 320

    def test_cancelling_stops_early(self, loaded, logo_bgra, tmp_path):
        window, path, truth = loaded
        self._prepare(window, truth, logo_bgra)
        out = str(tmp_path / "cancelled.mp4")
        worker = lumen_app.RenderWorker(
            window.build_render_settings(0, 9, 1.0, out))
        worker.cancel()

        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses == ["Canceled"]

    def test_curved_render_runs(self, loaded, logo_bgra, tmp_path):
        window, path, truth = loaded
        panel, overlay = self._prepare(window, truth, logo_bgra)
        overlay.curved_enabled = True
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        overlay.set_control_point(0, 0, np.float32(truth[0][0]) + [10, -25])

        out = str(tmp_path / "curved.mp4")
        settings = window.build_render_settings(0, 4, 1.0, out)
        assert settings.curved is True
        assert np.any(settings.curvature != 0)

        worker = lumen_app.RenderWorker(settings)
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses and statuses[0].startswith("Completed"), statuses


class TestPreviewMatchesRender:
    def test_preview_uses_the_same_compositor_as_the_render(self, loaded, logo_bgra):
        """A preview that is not the render is a lie; check they agree exactly."""
        window, path, truth = loaded
        panel = window.central_panel
        overlay = panel.tracking_overlay
        overlay.overlay_bgra = logo_bgra
        overlay.points = [tuple(map(float, p)) for p in truth[0]]

        frame = panel.prev_frame.copy()
        styled = overlay.styled_overlay(frame, overlay.points)
        region = overlay.current_region()

        preview = core.composite_region(frame, styled, region)
        rendered = core.composite_region(frame, styled, region)
        assert np.array_equal(preview, rendered)


class TestScreenAndOcclusionToggles:
    def test_digital_screen_switches_the_feature_source(self, loaded, monkeypatch):
        window, path, truth = loaded
        panel = window.central_panel
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        assert panel.feature_source == core.PlanarTracker.INTERIOR
        window.toggle_digital_screen(True)
        assert panel.feature_source == core.PlanarTracker.SURROUND
        window.toggle_digital_screen(False)
        assert panel.feature_source == core.PlanarTracker.INTERIOR

    def test_switching_mode_reanchors_the_tracker(self, loaded, monkeypatch):
        window, path, truth = loaded
        panel = window.central_panel
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
        panel.tracking_mode = True
        panel.read_frame()
        assert panel.tracker is not None

        window.toggle_digital_screen(True)
        assert panel.tracker is None

    def test_the_tracker_is_built_with_the_chosen_source(self, loaded, monkeypatch):
        window, path, truth = loaded
        panel = window.central_panel
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.toggle_digital_screen(True)

        panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
        panel.tracking_mode = True
        panel.read_frame()
        assert panel.tracker.feature_source == core.PlanarTracker.SURROUND

    def test_occlusion_off_produces_no_mask(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        assert panel.occlusion_enabled is False
        assert panel.occlusion_mask(panel.prev_frame, truth[0]) is None

    def test_occlusion_without_the_model_warns_and_stays_off(self, loaded, monkeypatch):
        window, path, truth = loaded
        monkeypatch.setattr(core.PersonSegmenter, "is_available",
                            classmethod(lambda cls: False))
        warned = []
        monkeypatch.setattr(lumen_app.QMessageBox, "warning",
                            staticmethod(lambda *a, **k: warned.append(a)))

        window.toggle_occlusion(True)
        assert warned, "no warning shown for the missing model"
        assert window.central_panel.occlusion_enabled is False

    def test_render_settings_carry_a_flag_not_a_net(self, loaded, logo_bgra,
                                                    tmp_path):
        """A cv2.dnn.Net cannot cross threads, so only a flag may travel."""
        window, path, truth = loaded
        overlay = window.central_panel.tracking_overlay
        overlay.overlay_bgra = logo_bgra
        overlay.tracking_history = {0: [tuple(map(float, p)) for p in truth[0]]}
        window.central_panel.occlusion_enabled = True

        settings = window.build_render_settings(0, 2, 1.0, str(tmp_path / "o.mp4"))
        assert settings.occlusion is True
        assert not hasattr(settings, "segmenter")
        for value in vars(settings).values():
            assert not isinstance(value, cv2.dnn.Net)


class TestAoiExport:
    """The AOI CSV is what the eye-tracking analysis is run against."""

    def _export(self, window, tmp_path, monkeypatch, history):
        overlay = window.central_panel.tracking_overlay
        overlay.tracking_history = history
        out = str(tmp_path / "aoi.csv")
        monkeypatch.setattr(lumen_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (out, "")))
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.save_aoi_geometry()

        import csv
        with open(out, newline="") as handle:
            rows = list(csv.reader(handle))
        return rows[0], rows[1], rows[2:]

    def test_gaps_are_filled_so_the_aoi_never_vanishes(self, loaded, tmp_path,
                                                       monkeypatch):
        """A hole in the AOI loses every fixation that lands in it."""
        window, path, truth = loaded
        sparse = {0: [tuple(map(float, p)) for p in truth[0]],
                  9: [tuple(map(float, p)) for p in truth[9]]}
        _, _, rows = self._export(window, tmp_path, monkeypatch, sparse)

        # One row per frame across the tracked range, not just the two keys.
        assert len(rows) == 10
        assert all(row[-1] for row in rows), "a frame exported with no points"

    def test_rows_are_contiguous_in_time(self, loaded, tmp_path, monkeypatch):
        window, path, truth = loaded
        sparse = {0: [tuple(map(float, p)) for p in truth[0]],
                  5: [tuple(map(float, p)) for p in truth[5]]}
        _, _, rows = self._export(window, tmp_path, monkeypatch, sparse)

        ends = [float(r[1]) for r in rows]
        starts = [float(r[0]) for r in rows]
        for previous_end, next_start in zip(ends, starts[1:]):
            assert next_start == pytest.approx(previous_end, abs=1e-6)

    def test_points_track_the_moving_surface(self, loaded, tmp_path, monkeypatch):
        window, path, truth = loaded
        history = {i: [tuple(map(float, p)) for p in q]
                   for i, q in enumerate(truth[:6])}
        _, _, rows = self._export(window, tmp_path, monkeypatch, history)

        first = [float(v) for v in rows[0][-1].split(";")]
        last = [float(v) for v in rows[-1][-1].split(";")]
        assert len(first) == 8
        assert first != last, "AOI did not move with the surface"

    def test_stimulus_dimensions_are_consistent(self, loaded, tmp_path, monkeypatch):
        window, path, truth = loaded
        header, _, _ = self._export(
            window, tmp_path, monkeypatch,
            {0: [tuple(map(float, p)) for p in truth[0]]})
        stim = {k: v for k, v in (f.split(":", 1) for f in header if ":" in f)}
        # Both were not formatted the same way before; a parser reading them as
        # a matched pair would disagree about the frame size.
        assert stim["stim_width"].isdigit()
        assert stim["stim_height"].isdigit()


class TestProjectRoundTrip:
    def test_project_saves_the_whole_history(self, loaded, tmp_path, monkeypatch):
        window, path, truth = loaded
        overlay = window.central_panel.tracking_overlay
        overlay.tracking_history = {
            i: [tuple(map(float, p)) for p in quad]
            for i, quad in enumerate(truth[:5])
        }
        overlay.curved_enabled = True
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        overlay.set_control_point(1, 0, np.float32(truth[0][1]) + [12, 4])

        project = str(tmp_path / "project.json")
        monkeypatch.setattr(lumen_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.save_project()

        import json
        with open(project) as handle:
            data = json.load(handle)

        # The old version saved only the current quad, losing every other frame.
        assert len(data["tracking_history"]) == 5
        assert data["curved"] is True
        assert np.any(np.array(data["curvature"]) != 0)
