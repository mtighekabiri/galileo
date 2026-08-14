"""Finding every instance of a target, and searching from a clip.

A concourse often carries the same poster on several panels, and a researcher
may have a short clip of the target rather than a clean photograph of it.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util

import cv2
import numpy as np
import pytest

import galileo_core as core
from conftest import make_background, make_texture, quad_path, render_clip, write_video

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication

# Loaded once, like the other GUI test modules. Re-executing it inside a
# function-scoped fixture makes a fresh copy of every Qt class per test, which
# crashes the interpreter.
_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)

WIDTH, HEIGHT = 900, 600
PANELS = [
    np.float32([[60, 80], [330, 70], [338, 265], [68, 272]]),
    np.float32([[420, 120], [640, 140], [628, 300], [408, 286]]),
    np.float32([[690, 330], [860, 340], [854, 470], [684, 462]]),
]


def plant(texture, quads, seed=17):
    """Put the same poster on several panels of one frame."""
    frame = make_background(WIDTH, HEIGHT, seed=seed)
    th, tw = texture.shape[:2]
    source = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    for quad in quads:
        matrix = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
        warped = cv2.warpPerspective(texture, matrix, (WIDTH, HEIGHT))
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        frame[mask > 0] = warped[mask > 0]
    return frame


@pytest.fixture(scope="session")
def qapp():
    """Held for the session: a discarded QApplication is collected, and the
    next QWidget then aborts the interpreter."""
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def poster():
    return make_texture(300, 200, seed=31)


@pytest.fixture(scope="module")
def three_panels(poster):
    return plant(poster, PANELS)


def nearest_error(detection, quads=PANELS):
    return min(float(np.linalg.norm(detection.quad - q, axis=1).mean())
               for q in quads)


class TestLocateAll:
    def test_finds_every_planted_instance(self, poster, three_panels):
        found = core.ReferenceMatcher(poster).locate_all(three_panels)
        assert len(found) == len(PANELS), f"found {len(found)} of {len(PANELS)}"

    def test_each_lands_on_a_real_panel(self, poster, three_panels):
        for detection in core.ReferenceMatcher(poster).locate_all(three_panels):
            assert nearest_error(detection) < 6.0

    def test_results_are_distinct_panels(self, poster, three_panels):
        found = core.ReferenceMatcher(poster).locate_all(three_panels)
        centres = [tuple(np.float32(d.quad).mean(axis=0).round(0)) for d in found]
        assert len(set(centres)) == len(found), "the same panel came back twice"

    def test_strongest_first(self, poster, three_panels):
        found = core.ReferenceMatcher(poster).locate_all(three_panels)
        assert found == sorted(found, key=lambda d: d.n_inliers, reverse=True)

    def test_max_results_is_respected(self, poster, three_panels):
        found = core.ReferenceMatcher(poster).locate_all(three_panels, max_results=2)
        assert len(found) == 2

    def test_no_false_positives_on_unrelated_footage(self, poster):
        rng = np.random.default_rng(3)
        for frame in (rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8),
                      make_background(WIDTH, HEIGHT, seed=44)):
            assert core.ReferenceMatcher(poster).locate_all(frame) == []

    def test_a_different_poster_matches_nothing(self, three_panels):
        other = make_texture(300, 200, seed=77)
        assert core.ReferenceMatcher(other).locate_all(three_panels) == []

    def test_single_instance_still_returns_one(self, poster):
        frame = plant(poster, PANELS[:1])
        found = core.ReferenceMatcher(poster).locate_all(frame)
        assert len(found) == 1
        assert nearest_error(found[0]) < 4.0

    def test_locate_still_returns_the_strongest_single_hit(self, poster, three_panels):
        matcher = core.ReferenceMatcher(poster)
        one = matcher.locate(three_panels)
        best = matcher.locate_all(three_panels)[0]
        assert one is not None
        assert np.allclose(one.quad, best.quad, atol=2.0)


class TestOverlapRejection:
    def test_near_duplicate_quads_are_collapsed(self):
        quad = PANELS[0]
        a = core.Detection(quad, 0.9, 40, 50)
        b = core.Detection(quad + 3.0, 0.8, 30, 50)     # essentially the same panel
        assert len(core._drop_overlapping([a, b])) == 1

    def test_separate_quads_both_survive(self):
        a = core.Detection(PANELS[0], 0.9, 40, 50)
        b = core.Detection(PANELS[2], 0.8, 30, 50)
        assert len(core._drop_overlapping([a, b])) == 2

    def test_the_stronger_duplicate_is_kept(self):
        strong = core.Detection(PANELS[0], 0.9, 40, 50)
        weak = core.Detection(PANELS[0] + 2.0, 0.5, 15, 50)
        assert core._drop_overlapping([weak, strong])[0] is strong


class TestVideoReference:
    @staticmethod
    @pytest.fixture(scope="class")
    def reference_clip(tmp_path_factory):
        """A handheld clip *framed on* the target, as the docs require.

        Each frame fills with the poster, shifting and rotating a little, which
        is what someone walking up to a billboard and filming it produces.
        """
        texture = make_texture(300, 200, seed=31)
        frames = []
        for index in range(8):
            angle = (index - 4) * 1.2
            scale = 1.0 + index * 0.004
            matrix = cv2.getRotationMatrix2D((150, 100), angle, scale)
            frames.append(cv2.warpAffine(texture, matrix, (300, 200),
                                         borderMode=cv2.BORDER_REPLICATE))
        path = tmp_path_factory.mktemp("ref") / "target.mp4"
        return write_video(frames, path)

    @staticmethod
    @pytest.fixture(scope="class")
    def wide_reference_clip(tmp_path_factory):
        """A clip where the target is only part of the picture -- the mistake."""
        texture = make_texture(300, 200, seed=31)
        frames = render_clip(quad_path(8), texture=texture)
        path = tmp_path_factory.mktemp("wideref") / "wide.mp4"
        return write_video(frames, path)

    def test_open_reference_picks_the_video_matcher(self, reference_clip):
        matcher = core.open_reference(reference_clip)
        assert isinstance(matcher, core.VideoReferenceMatcher)

    def test_open_reference_still_handles_images(self, tmp_path, poster):
        path = str(tmp_path / "poster.png")
        cv2.imwrite(path, poster)
        assert isinstance(core.open_reference(path), core.ReferenceMatcher)

    def test_samples_several_views(self, reference_clip):
        matcher = core.VideoReferenceMatcher(reference_clip, samples=5)
        assert 2 <= len(matcher.views) <= 5

    def test_locates_the_target_from_a_clip(self, reference_clip, three_panels):
        """A clip lands close, but not as exactly as a cropped still.

        Each sampled view's own rectangle is what gets mapped, and a handheld
        view is never framed on the target to the pixel, so the quad comes back
        a little off -- around 11 px on a 270 px panel here. That is a starting
        point to nudge, which is what the workflow expects, rather than the
        sub-pixel result a properly cropped photograph gives.
        """
        detection = core.VideoReferenceMatcher(reference_clip).locate(three_panels)
        assert detection is not None
        assert nearest_error(detection) < 20.0

    def test_finds_all_instances_from_a_clip(self, reference_clip, three_panels):
        found = core.VideoReferenceMatcher(reference_clip).locate_all(three_panels)
        assert len(found) >= 2
        for detection in found:
            assert nearest_error(detection) < 20.0

    def test_views_of_the_same_panel_are_not_returned_twice(self, reference_clip,
                                                            three_panels):
        """Six views all matching one panel must not yield six placements."""
        found = core.VideoReferenceMatcher(reference_clip).locate_all(three_panels)
        centres = [tuple(np.float32(d.quad).mean(axis=0).round(-1)) for d in found]
        assert len(set(centres)) == len(centres)

    def test_scans_a_clip_for_the_target(self, reference_clip, tmp_path, poster):
        frames = [plant(poster, PANELS[:1], seed=17) for _ in range(4)]
        haystack = write_video(frames, tmp_path / "haystack.mp4")
        detection = core.VideoReferenceMatcher(reference_clip).scan_video(
            haystack, step=2)
        assert detection is not None

    def test_a_wide_reference_covers_far_too_much(self, wide_reference_clip,
                                                  three_panels):
        """Documents the constraint, and the signal that catches breaking it.

        Each sampled frame is taken to *be* the target, so a clip in which the
        poster is only part of the picture matches confidently and returns the
        corners of the whole scene. frame_coverage is how a caller notices.
        """
        detection = core.VideoReferenceMatcher(wide_reference_clip).locate(
            three_panels)
        assert detection is not None, "it still matches; that is the trap"
        assert nearest_error(detection) > 50.0
        assert core.frame_coverage(detection, three_panels.shape) > 0.5

    def test_a_properly_framed_reference_covers_a_sensible_area(
            self, reference_clip, three_panels):
        detection = core.VideoReferenceMatcher(reference_clip).locate(three_panels)
        assert core.frame_coverage(detection, three_panels.shape) < 0.3

    def test_unreadable_video_raises(self):
        with pytest.raises(IOError):
            core.VideoReferenceMatcher("/nonexistent/clip.mp4")

    def test_a_featureless_clip_is_rejected_with_advice(self, tmp_path):
        blank = [np.full((240, 320, 3), 200, np.uint8) for _ in range(4)]
        path = write_video(blank, tmp_path / "blank.mp4")
        with pytest.raises(ValueError, match="holds still"):
            core.VideoReferenceMatcher(path)


class TestDetectionsBecomePlacements:
    """The intended flow: several matches turn into several placements."""

    @pytest.fixture
    def window(self, qapp, tmp_path, poster):
        frames = [plant(poster, PANELS) for _ in range(4)]
        path = write_video(frames, tmp_path / "panels.mp4")

        win = galileo_app.MainWindow()
        win.show()
        win.central_panel.load_video(path)
        yield win, galileo_app, path
        win.dirty = False
        win.close()

    def test_each_match_becomes_its_own_placement(self, window, poster, tmp_path,
                                                  monkeypatch):
        win, app_module, path = window
        ref = str(tmp_path / "ref.png")
        cv2.imwrite(ref, poster)

        monkeypatch.setattr(app_module.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (ref, "")))
        monkeypatch.setattr(app_module.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        # Accept every offered match.
        monkeypatch.setattr(app_module.MatchChoiceDialog, "exec_",
                            lambda self: app_module.QDialog.Accepted)

        win.detect_from_reference()

        placements = win.central_panel.tracking_overlay.placements
        assert len(placements) == len(PANELS)
        for placement in placements:
            assert len(placement.points) == 4
            assert placement.tracking_history, "no keyframe recorded"

    def test_declining_the_dialog_creates_nothing(self, window, poster, tmp_path,
                                                  monkeypatch):
        win, app_module, path = window
        ref = str(tmp_path / "ref.png")
        cv2.imwrite(ref, poster)
        monkeypatch.setattr(app_module.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (ref, "")))
        monkeypatch.setattr(app_module.MatchChoiceDialog, "exec_",
                            lambda self: app_module.QDialog.Rejected)

        before = len(win.central_panel.tracking_overlay.placements)
        win.detect_from_reference()
        assert len(win.central_panel.tracking_overlay.placements) == before
