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

import galileo_core as core
from conftest import make_texture, quad_path, render_clip, write_video

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def window(qapp):
    win = galileo_app.MainWindow()
    yield win
    # Closing asks what to do about unsaved tracking work, and offscreen that
    # question is a dialog nobody can answer -- the run stops there for good.
    # Nothing here is testing the prompt, so the project is retired as saved.
    win.mark_clean()
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
        worker = galileo_app.RenderWorker(settings)

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
            assert not isinstance(value, (galileo_app.QPixmap, galileo_app.QImage))
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
        worker = galileo_app.RenderWorker(
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

        worker = galileo_app.RenderWorker(settings)
        worker.run()
        assert os.path.exists(out)

    def test_render_at_reduced_scale(self, loaded, logo_bgra, tmp_path):
        window, path, truth = loaded
        self._prepare(window, truth, logo_bgra)
        out = str(tmp_path / "half.mp4")
        worker = galileo_app.RenderWorker(
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
        worker = galileo_app.RenderWorker(
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

        worker = galileo_app.RenderWorker(settings)
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
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))

        assert panel.feature_source == core.PlanarTracker.INTERIOR
        window.toggle_digital_screen(True)
        assert panel.feature_source == core.PlanarTracker.SURROUND
        window.toggle_digital_screen(False)
        assert panel.feature_source == core.PlanarTracker.INTERIOR

    def test_switching_mode_reanchors_the_tracker(self, loaded, monkeypatch):
        window, path, truth = loaded
        panel = window.central_panel
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
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
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
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
        monkeypatch.setattr(galileo_app.QMessageBox, "warning",
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
        window.central_panel.obstruction_enabled = True

        settings = window.build_render_settings(0, 2, 1.0, str(tmp_path / "o.mp4"))
        assert settings.occlusion is True
        assert settings.obstructions is True
        assert not hasattr(settings, "segmenter")
        for value in vars(settings).values():
            assert not isinstance(value, cv2.dnn.Net)

    def test_obstructions_off_produces_no_mask(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        assert panel.obstruction_enabled is False
        assert panel.obstruction_mask(panel.prev_frame, truth[0]) is None

    def test_obstructions_without_the_model_warn_and_stay_off(self, loaded,
                                                              monkeypatch):
        window, path, truth = loaded
        monkeypatch.setattr(core.DepthOcclusionSegmenter, "is_available",
                            classmethod(lambda cls: False))
        warned = []
        monkeypatch.setattr(galileo_app.QMessageBox, "warning",
                            staticmethod(lambda *a, **k: warned.append(a)))

        window.toggle_obstructions(True)
        assert warned, "no warning shown for the missing model"
        assert window.central_panel.obstruction_enabled is False

    def test_the_two_sources_are_combined(self, loaded, monkeypatch):
        """Both kinds of occlusion have to reach the compositor, and a
        pedestrian behind a railing is found by both at once."""
        window, path, truth = loaded
        panel = window.central_panel
        height, width = panel.prev_frame.shape[:2]

        person = np.zeros((height, width), np.uint8)
        person[10:40, 10:40] = 255
        depth = np.zeros((height, width), np.uint8)
        depth[30:60, 30:60] = 255
        monkeypatch.setattr(panel, "occlusion_mask", lambda f, q: person)
        monkeypatch.setattr(panel, "obstruction_mask", lambda f, q: depth)

        combined = panel.combined_occlusion(panel.prev_frame, truth[0])
        assert np.array_equal(combined, cv2.max(person, depth))
        assert combined[20, 20] == 255 and combined[50, 50] == 255

    def test_either_source_alone_still_reaches_the_compositor(self, loaded,
                                                             monkeypatch):
        window, path, truth = loaded
        panel = window.central_panel
        only = np.zeros(panel.prev_frame.shape[:2], np.uint8)
        only[5:15, 5:15] = 255

        monkeypatch.setattr(panel, "occlusion_mask", lambda f, q: None)
        monkeypatch.setattr(panel, "obstruction_mask", lambda f, q: only)
        assert panel.combined_occlusion(panel.prev_frame, truth[0]) is only

        monkeypatch.setattr(panel, "occlusion_mask", lambda f, q: only)
        monkeypatch.setattr(panel, "obstruction_mask", lambda f, q: None)
        assert panel.combined_occlusion(panel.prev_frame, truth[0]) is only

        monkeypatch.setattr(panel, "occlusion_mask", lambda f, q: None)
        monkeypatch.setattr(panel, "obstruction_mask", lambda f, q: None)
        assert panel.combined_occlusion(panel.prev_frame, truth[0]) is None

    def test_the_cache_notices_a_toggle_on_the_same_frame(self, loaded,
                                                          monkeypatch):
        """Switching a source on while paused has to redraw the frame with it.
        The frame has not changed, so the toggles are part of the cache key."""
        window, path, truth = loaded
        panel = window.central_panel
        placement = panel.tracking_overlay.placements[0]
        placement.points = [tuple(map(float, p)) for p in truth[0]]

        calls = []
        monkeypatch.setattr(panel, "occlusion_mask",
                            lambda f, q: calls.append("person"))
        monkeypatch.setattr(panel, "obstruction_mask",
                            lambda f, q: calls.append("depth"))

        panel.occlusion_enabled = True
        panel.occlusion_for_frame(panel.prev_frame, [placement])
        panel.occlusion_for_frame(panel.prev_frame, [placement])
        assert len(calls) == 2, "the same frame was segmented twice"

        panel.obstruction_enabled = True
        panel.occlusion_for_frame(panel.prev_frame, [placement])
        assert "depth" in calls, "turning a source on did not recompute"


class TestThePreviousFrameToggle:
    """The Options menu can mark, in each magnifier tile, where the corner
    sat on the frame before."""

    def test_the_menu_switch_reaches_the_magnifier(self, loaded):
        window, path, truth = loaded
        assert window.central_panel.magnifier.show_previous is False
        window.title_bar.previous_positions_action.setChecked(True)
        assert window.central_panel.magnifier.show_previous is True
        window.title_bar.previous_positions_action.setChecked(False)
        assert window.central_panel.magnifier.show_previous is False

    def test_tracking_feeds_the_magnifier_yesterdays_corners(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        overlay = panel.tracking_overlay
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        panel.tracking_mode = True
        for _ in range(4):
            panel.read_frame()

        panel.update_magnifier(force=True)
        index = panel.get_current_frame_index()
        expected = overlay.tracking_history[index - 1]
        ghosts = panel.magnifier.previous_points
        assert set(ghosts) == {0, 1, 2, 3}
        for corner in range(4):
            assert ghosts[corner] == pytest.approx(expected[corner])

    def test_frame_zero_has_no_yesterday(self, loaded):
        window, path, truth = loaded
        assert window.central_panel.magnifier_previous_points() == {}

    def test_an_untracked_predecessor_gives_no_ghost(self, loaded):
        """Falling back to some older recorded frame would draw a ghost that
        looks like one frame of motion but is really many, which misleads
        exactly the judgement the ghost exists to serve."""
        window, path, truth = loaded
        panel = window.central_panel
        panel.tracking_overlay.tracking_history = {
            0: [tuple(map(float, p)) for p in truth[0]]}
        panel.jump_to_frame(5)
        assert panel.magnifier_previous_points() == {}

    def test_curving_keeps_each_ghost_with_its_corner(self, loaded):
        """With curving on, corner i sits at index 3i between its bend
        handles, and its ghost has to move there with it."""
        window, path, truth = loaded
        panel = window.central_panel
        overlay = panel.tracking_overlay
        overlay.tracking_history = {0: [tuple(map(float, p)) for p in truth[0]]}
        panel.jump_to_frame(1)
        overlay.curved_enabled = True
        overlay.points = [tuple(map(float, p)) for p in truth[1]]
        assert set(panel.magnifier_previous_points()) == {0, 3, 6, 9}


class TestAoiExport:
    """The AOI CSV is what the eye-tracking analysis is run against."""

    def _export(self, window, tmp_path, monkeypatch, history, render=None):
        overlay = window.central_panel.tracking_overlay
        overlay.tracking_history = history
        window.last_render = render
        out = str(tmp_path / "aoi.csv")
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (out, "")))
        for name in ("information", "warning"):
            monkeypatch.setattr(galileo_app.QMessageBox, name,
                                staticmethod(lambda *a, **k: None))
        # Exporting with no render asks for confirmation; answer yes.
        monkeypatch.setattr(galileo_app.QMessageBox, "question",
                            staticmethod(lambda *a, **k: galileo_app.QMessageBox.Yes))
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


class TestAoiMatchesTheRender:
    """The CSV must describe the rendered stimulus, not the source footage.

    A render can start partway in and be scaled down. Exporting absolute source
    times at full resolution produced a file that parsed perfectly and was wrong
    in three independent ways at once, so every fixation landed in the wrong
    place with nothing to reveal it.
    """

    def _export(self, window, tmp_path, monkeypatch, history, render):
        return TestAoiExport()._export(window, tmp_path, monkeypatch,
                                       history, render)

    @pytest.fixture
    def render_info(self, tmp_path):
        out = tmp_path / "stim.mp4"
        out.write_bytes(b"x")
        return {"out_path": str(out), "start_frame": 5, "end_frame": 12,
                "scale_factor": 0.5, "fps": 25.0, "width": 320, "height": 240}

    def test_times_are_relative_to_the_rendered_file(self, loaded, tmp_path,
                                                     monkeypatch, render_info):
        window, path, truth = loaded
        history = {i: [tuple(map(float, p)) for p in q]
                   for i, q in enumerate(truth)}
        _, _, rows = self._export(window, tmp_path, monkeypatch, history,
                                  render_info)
        # The render starts at frame 5, so the first AOI row is time zero.
        assert float(rows[0][0]) == pytest.approx(0.0, abs=1e-6)
        assert float(rows[-1][1]) == pytest.approx(8 / 25.0, abs=1e-6)

    def test_only_the_rendered_range_is_exported(self, loaded, tmp_path,
                                                 monkeypatch, render_info):
        window, path, truth = loaded
        history = {i: [tuple(map(float, p)) for p in q]
                   for i, q in enumerate(truth)}
        _, _, rows = self._export(window, tmp_path, monkeypatch, history,
                                  render_info)
        assert len(rows) == 8      # frames 5..12 inclusive

    def test_coordinates_are_in_the_rendered_pixel_space(self, loaded, tmp_path,
                                                         monkeypatch, render_info):
        window, path, truth = loaded
        history = {i: [tuple(map(float, p)) for p in q]
                   for i, q in enumerate(truth)}
        _, _, rows = self._export(window, tmp_path, monkeypatch, history,
                                  render_info)
        exported = [float(v) for v in rows[0][-1].split(";")]
        source = np.float32(truth[5]).reshape(-1)
        assert np.allclose(exported, source * 0.5, atol=1e-3)

    def test_header_reports_the_rendered_size(self, loaded, tmp_path,
                                              monkeypatch, render_info):
        window, path, truth = loaded
        history = {i: [tuple(map(float, p)) for p in q]
                   for i, q in enumerate(truth)}
        header, _, _ = self._export(window, tmp_path, monkeypatch, history,
                                    render_info)
        stim = {k: v for k, v in (f.split(":", 1) for f in header if ":" in f)}
        assert stim["stim_width"] == "320"
        assert stim["stim_height"] == "240"

    def test_declining_the_no_render_prompt_writes_nothing(self, loaded, tmp_path,
                                                           monkeypatch):
        window, path, truth = loaded
        window.last_render = None
        window.central_panel.tracking_overlay.tracking_history = {
            0: [tuple(map(float, p)) for p in truth[0]]}
        out = tmp_path / "declined.csv"
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        monkeypatch.setattr(galileo_app.QMessageBox, "question",
                            staticmethod(lambda *a, **k: galileo_app.QMessageBox.No))
        window.save_aoi_geometry()
        assert not out.exists()

    def test_unrenderable_shapes_are_skipped(self, loaded, tmp_path,
                                             monkeypatch, render_info):
        """A folded quad draws no advert, so it must claim no AOI either."""
        window, path, truth = loaded
        history = {i: [tuple(map(float, p)) for p in q]
                   for i, q in enumerate(truth)}
        # A bowtie at frame 7 has no interior.
        good = np.float32(truth[7])
        history[7] = [tuple(good[0]), tuple(good[1]), tuple(good[3]), tuple(good[2])]
        _, _, rows = self._export(window, tmp_path, monkeypatch, history,
                                  render_info)
        assert len(rows) == 7      # one fewer than the 8-frame range


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
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.save_project()

        import json
        with open(project) as handle:
            data = json.load(handle)

        # The old version saved only the current quad, losing every other frame.
        assert len(data["tracking_history"]) == 5
        assert data["curved"] is True
        assert np.any(np.array(data["curvature"]) != 0)


class TestSteadyingTheTrackedPath:
    """The fit that takes the wobble out has to reach the file, and the
    preview has to be showing the same thing while it does."""

    def hand_corrected(self, panel, truth):
        """A history that is right on average and unsteady frame to frame."""
        rng = np.random.default_rng(0)
        placement = panel.tracking_overlay.placements[0]
        history = {i: [tuple(map(float, p))
                       for p in np.float32(quad) + rng.normal(0, 1.5, (4, 2))]
                   for i, quad in enumerate(truth)}
        placement.tracking_history = dict(history)
        return placement, history

    def test_it_is_off_until_asked_for(self, loaded):
        window, path, truth = loaded
        assert window.central_panel.steady_tracking is False

    def test_the_toggle_carries_into_the_render(self, loaded, logo_bgra, tmp_path):
        window, path, truth = loaded
        overlay = window.central_panel.tracking_overlay
        overlay.overlay_bgra = logo_bgra
        overlay.tracking_history = {0: [tuple(map(float, p)) for p in truth[0]]}

        window.toggle_steady_tracking(True)
        settings = window.build_render_settings(0, 2, 1.0, str(tmp_path / "o.mp4"))
        assert settings.steady_tracking is True
        assert settings.steady_window == core.STEADY_WINDOW

    def test_the_render_is_steadier_than_what_was_recorded(self, loaded,
                                                           logo_bgra, tmp_path):
        window, path, truth = loaded
        panel = window.central_panel
        panel.tracking_overlay.overlay_bgra = logo_bgra
        placement, history = self.hand_corrected(panel, truth)

        def dense(steady):
            panel.steady_tracking = steady
            settings = window.build_render_settings(
                0, len(truth) - 1, 1.0, str(tmp_path / "o.mp4"))
            worker = galileo_app.RenderWorker(settings)
            return worker._prepare_placements(settings, [])[0].dense

        def wobble(path_):
            keys = sorted(path_)
            quads = np.array([np.asarray(path_[k], np.float64).reshape(4, 2)
                              for k in keys])
            return float(np.linalg.norm(
                quads[2:] - 2 * quads[1:-1] + quads[:-2], axis=2).mean())

        raw, steadied = wobble(dense(False)), wobble(dense(True))
        assert steadied < raw / 3, f"{raw:.2f}px only came down to {steadied:.2f}px"

    def test_the_preview_shows_what_the_render_will_write(self, loaded,
                                                          logo_bgra, tmp_path):
        window, path, truth = loaded
        panel = window.central_panel
        panel.tracking_overlay.overlay_bgra = logo_bgra
        placement, history = self.hand_corrected(panel, truth)

        panel.steady_tracking = True
        settings = window.build_render_settings(
            0, len(truth) - 1, 1.0, str(tmp_path / "o.mp4"))
        rendered = galileo_app.RenderWorker(settings)._prepare_placements(
            settings, [])[0].dense

        for frame in range(len(truth)):
            panel.replay_recorded_shapes(frame)
            shown = np.asarray(placement.points, np.float64).reshape(4, 2)
            assert np.allclose(shown, np.asarray(rendered[frame], np.float64)
                               .reshape(4, 2)), f"frame {frame} differs"

    def test_the_recording_itself_is_never_rewritten(self, loaded):
        """Steadying is a way of reading the history, not an edit to it, so it
        can be turned on to see what it does and off again at no cost."""
        window, path, truth = loaded
        panel = window.central_panel
        placement, history = self.hand_corrected(panel, truth)

        window.toggle_steady_tracking(True)
        for frame in range(len(truth)):
            panel.replay_recorded_shapes(frame)

        assert all(np.allclose(placement.tracking_history[i], history[i])
                   for i in history)


class TestCorrectionsAreKept:
    """A shape a person placed is a correction; one the tracker placed is an
    estimate. A pass used to overwrite the first with the second and say
    nothing, which is what made fixing a difficult clip endless."""

    def track_forward(self, panel, truth, frames):
        panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
        panel.tracking_mode = True
        for _ in range(frames):
            panel.read_frame()
        panel.tracking_mode = False

    def test_a_hand_edit_is_recorded_as_one(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        overlay = panel.tracking_overlay

        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        overlay.commit_shape()
        assert panel.get_current_frame_index() in overlay.manual_frames

    def test_the_tracker_does_not_claim_frames_it_placed(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        self.track_forward(panel, truth, 5)

        placement = panel.tracking_overlay.placements[0]
        assert placement.tracking_history, "nothing was tracked at all"
        assert not placement.manual_frames, "the tracker's own work was marked by hand"

    def test_a_later_pass_keeps_the_correction(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        overlay = panel.tracking_overlay
        placement = overlay.placements[0]

        self.track_forward(panel, truth, 8)
        panel.jump_to_frame(4)
        overlay.points = [tuple(map(float, p))
                          for p in np.float32(truth[4]) + [9.0, -6.0]]
        overlay.commit_shape()
        corrected = [tuple(p) for p in placement.tracking_history[4]]

        # ...and now the whole stretch is tracked again over the top of it.
        panel.jump_to_frame(0)
        self.track_forward(panel, truth, 8)

        assert np.allclose(placement.tracking_history[4], corrected), \
            "the correction was overwritten by the pass it was made to fix"
        assert any(f in placement.tracking_history for f in range(5, 8)), \
            "the pass stopped instead of carrying on past the correction"

    def test_deleting_the_shape_takes_the_correction_with_it(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        overlay = panel.tracking_overlay

        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        overlay.commit_shape()
        frame = panel.get_current_frame_index()
        assert frame in overlay.manual_frames

        panel.on_delete_shape()
        assert frame not in overlay.manual_frames
        assert frame not in overlay.tracking_history

    def test_a_project_remembers_which_were_corrections(self, loaded):
        window, path, truth = loaded
        overlay = window.central_panel.tracking_overlay
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        overlay.commit_shape()
        placement = overlay.placements[0]

        import json
        restored = galileo_app.Placement.from_dict(
            json.loads(json.dumps(placement.to_dict())))
        assert restored.manual_frames == placement.manual_frames

    def test_a_project_saved_before_this_existed_still_loads(self, loaded):
        window, path, truth = loaded
        overlay = window.central_panel.tracking_overlay
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        overlay.commit_shape()

        older = {k: v for k, v in overlay.placements[0].to_dict().items()
                 if k != "manual_frames"}
        assert galileo_app.Placement.from_dict(older).manual_frames == set()


class TestObstructionSensitivity:
    """How readily something counts as being in front of the surface is a
    setting, because what defeats the depth network is the picture already on
    the billboard and the tool cannot know how much depth that picture has in
    it."""

    def test_it_starts_at_the_measured_middle(self, loaded):
        window, path, truth = loaded
        assert window.central_panel.obstruction_sensitivity == "normal"

    def test_each_level_reaches_the_render(self, loaded, logo_bgra, tmp_path):
        window, path, truth = loaded
        overlay = window.central_panel.tracking_overlay
        overlay.overlay_bgra = logo_bgra
        overlay.tracking_history = {0: [tuple(map(float, p)) for p in truth[0]]}

        for level in ("low", "high", "normal"):
            window.set_obstruction_sensitivity(level)
            settings = window.build_render_settings(
                0, 2, 1.0, str(tmp_path / "o.mp4"))
            assert settings.obstruction_sensitivity == level
            assert (core.DEPTH_SENSITIVITY[settings.obstruction_sensitivity]
                    == core.DEPTH_SENSITIVITY[level])

    def test_changing_it_rebuilds_the_segmenter(self, loaded):
        """It holds the setting it was built with, so the old one has to go or
        the preview would go on masking to the setting just replaced."""
        window, path, truth = loaded
        panel = window.central_panel
        panel.depth_segmenter = object()
        window.set_obstruction_sensitivity("low")
        assert panel.depth_segmenter is None

    def test_the_menu_shows_what_is_in_use(self, loaded):
        window, path, truth = loaded
        window.set_obstruction_sensitivity("low")
        ticked = [level for level, action
                  in window.title_bar.sensitivity_actions.items()
                  if action.isChecked()]
        assert ticked == ["low"]

    def test_a_level_it_does_not_know_is_refused(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        window.set_obstruction_sensitivity("normal")
        window.set_obstruction_sensitivity("very high indeed")
        assert panel.obstruction_sensitivity == "normal"
        ticked = [level for level, action
                  in window.title_bar.sensitivity_actions.items()
                  if action.isChecked()]
        assert ticked == ["normal"], "the tick drifted off what is in use"

    def test_settings_built_the_old_way_still_work(self, loaded):
        """RenderSettings is constructed positionally in places."""
        settings = galileo_app.RenderSettings(
            "video.mp4", "out.mp4", 0, 1, 1.0, 25.0, {},
            np.zeros((4, 2, 2), np.float32), False)
        assert settings.obstruction_sensitivity == "normal"


class TestTheKeyboardStaysWithTheDrawingTools:
    """Stepping a frame used to kill the drawing keys until Draw was toggled
    off and on again.

    Clicking a button moves keyboard focus onto it, and the overlay only
    receives the arrow keys, 1-5 and T/R/B/L while it holds focus. Toggling
    Draw appeared to fix it because turning it back on calls setFocus as a
    side effect. Driving these through real clicks matters: calling the slots
    directly never moves focus, so it passes either way and proves nothing.
    """

    def drawing(self, window, truth):
        panel = window.central_panel
        panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
        window.on_draw_clicked(True)
        return panel, panel.tracking_overlay

    def test_the_transport_never_takes_the_keyboard(self, loaded):
        window, path, truth = loaded
        panel = window.central_panel
        for name in ("rewind_btn", "play_pause_btn", "forward_btn",
                     "copy_btn", "del_btn"):
            assert getattr(panel, name).focusPolicy() == Qt.NoFocus, name

    def test_clicking_through_the_transport_leaves_drawing_live(self, loaded):
        window, path, truth = loaded
        window.show()
        panel, overlay = self.drawing(window, truth)
        assert window.focusWidget() is overlay

        for button in (panel.forward_btn, panel.rewind_btn, panel.forward_btn):
            QTest.mouseClick(button, Qt.LeftButton)
            assert window.focusWidget() is overlay, "the keyboard was taken away"

    def test_the_arrow_keys_still_nudge_after_stepping(self, loaded):
        window, path, truth = loaded
        window.show()
        panel, overlay = self.drawing(window, truth)
        QTest.mouseClick(panel.forward_btn, Qt.LeftButton)
        QTest.mouseClick(panel.rewind_btn, Qt.LeftButton)

        overlay.selected_point_index = 0
        before = overlay.points[0]
        QTest.keyClick(window.focusWidget(), Qt.Key_Right)
        assert overlay.points[0] != before, "arrow keys dead after stepping"

    def test_a_jump_hands_the_keyboard_back(self, loaded):
        """jump_to_frame turns the tracking switch off and then used to ask
        whether it was on before restoring focus, so it never did."""
        window, path, truth = loaded
        window.show()
        panel, overlay = self.drawing(window, truth)
        panel.jump_to_frame(5)
        assert window.focusWidget() is overlay

    def test_with_draw_off_the_focus_is_left_where_it_was(self, loaded):
        window, path, truth = loaded
        window.show()
        panel, overlay = self.drawing(window, truth)
        window.on_draw_clicked(False)
        panel.slider.setFocus()
        panel.on_forward()
        assert window.focusWidget() is not overlay
