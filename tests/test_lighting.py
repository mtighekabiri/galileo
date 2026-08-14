"""Footage whose brightness will not sit still.

Optical flow rests on one assumption: that a point keeps its brightness from
one frame to the next. A camera panning across a dim concourse onto a bright
screen breaks it on every frame, because its automatic exposure is hunting the
whole time. That is not an edge case for this tool -- it is the ordinary
condition of the footage it is given.

The second half of the file is the other thing these clips make possible:
telling a screen playing its own picture from a poster stuck on a panel, which
is the most damaging thing that can quietly go wrong with a marked area.
"""

import cv2
import numpy as np
import pytest

import galileo_core as core
from conftest import make_background, make_texture


WIDTH, HEIGHT, FRAMES = 900, 500, 40
SCENE_WIDTH = 4200
PANEL = np.float32([[900, 140], [1220, 132], [1228, 352], [908, 360]])
FLAT = np.float32([[0, 0], [320, 0], [320, 220], [0, 220]])


def scene_showing(picture):
    scene = make_background(SCENE_WIDTH, HEIGHT, seed=21)
    cv2.warpPerspective(picture, cv2.getPerspectiveTransform(FLAT, PANEL),
                        (SCENE_WIDTH, HEIGHT), dst=scene,
                        borderMode=cv2.BORDER_TRANSPARENT)
    return scene


def pan(picture_for, lighting=None, step=8.0, count=FRAMES):
    """A camera panning along the street, optionally misbehaving."""
    frames, quads = [], []
    for index in range(count):
        offset = 700 + step * index
        scene = scene_showing(picture_for(index))
        frame = scene[:, int(offset):int(offset) + WIDTH].copy()
        if lighting is not None:
            frame = lighting(frame, index)
        frames.append(frame)
        quads.append(PANEL - np.float32([offset, 0]))
    return frames, quads


POSTER = make_texture(320, 220, seed=5)


def printed(index):
    return POSTER


def scrolling_advert(index):
    """A screen whose picture slides across it, independently of the panel."""
    picture = np.full((220, 320, 3), (40, 36, 46), np.uint8)
    cv2.rectangle(picture, (0, 0), (319, 219), (200, 200, 210), 6)
    offset = (index * 17) % 320
    for repeat in range(-1, 3):
        cv2.rectangle(picture, (offset + repeat * 120 - 40, 40),
                      (offset + repeat * 120 + 40, 180), (30, 180, 240), -1)
    return picture


def changing_slides(index):
    """A screen cycling through adverts: perfect between changes, and a
    completely different picture at each one."""
    rng = np.random.default_rng(index // 4)
    picture = make_texture(320, 220, seed=int(rng.integers(0, 999)))
    cv2.rectangle(picture, (0, 0), (319, 219), (200, 200, 210), 6)
    return picture


def hunting(amount):
    def apply(frame, index):
        gain = 1.0 + amount * np.sin(index * 0.9)
        return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return apply


def a_light_coming_on(frame, index):
    return np.clip(frame.astype(np.float32) * (0.55 + 0.02 * index),
                   0, 255).astype(np.uint8)


def a_person_crossing(frame, index):
    x = int(60 + index * 22)
    cv2.ellipse(frame, (x, 300), (50, 155), 0, 0, 360, (38, 34, 40), -1)
    cv2.circle(frame, (x, 128), 36, (52, 46, 50), -1)
    return frame


def final_error(frames, quads, **kwargs):
    tracker = core.PlanarTracker(frames[0], quads[0], **kwargs)
    error = 0.0
    for frame, truth in zip(frames[1:], quads[1:]):
        tracker.track(frame)
        error = float(np.linalg.norm(tracker.quad - truth, axis=1).mean())
    return error


class TestLevellingAFrame:
    def test_it_undoes_a_change_of_exposure(self):
        gray = cv2.cvtColor(make_background(400, 300, seed=4), cv2.COLOR_BGR2GRAY)
        brighter = np.clip(gray.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
        assert np.abs(core.level_gray(gray).astype(int)
                      - core.level_gray(brighter).astype(int)).mean() < 4.0

    def test_it_keeps_the_detail(self):
        gray = cv2.cvtColor(make_background(400, 300, seed=4), cv2.COLOR_BGR2GRAY)
        levelled = core.level_gray(gray)
        assert np.corrcoef(gray.ravel(), levelled.ravel())[0, 1] > 0.99

    def test_a_blank_frame_is_left_alone(self):
        blank = np.full((80, 80), 130, np.uint8)
        assert np.array_equal(core.level_gray(blank), blank)

    def test_the_statistics_describe_the_lighting_not_the_contents(self):
        """A large bright object crossing the frame moves a mean and a
        standard deviation enough to fake an exposure change. The median
        barely notices, which is the whole reason it is used."""
        gray = np.full((300, 400), 90, np.uint8)
        gray[:, :40] = 220
        crowded = gray.copy()
        cv2.circle(crowded, (200, 150), 90, 255, -1)

        median_shift = abs(float(np.median(gray)) - float(np.median(crowded)))
        mean_shift = abs(float(gray.mean()) - float(crowded.mean()))
        assert median_shift < mean_shift / 4


class TestFootageWhoseBrightnessMoves:
    def test_a_hunting_exposure_no_longer_drags_the_shape_off(self):
        frames, quads = pan(printed, hunting(0.40))
        assert final_error(frames, quads) < 1.0

    def test_it_is_far_better_than_leaving_it(self):
        frames, quads = pan(printed, hunting(0.40))
        levelled = final_error(frames, quads)
        plain = final_error(frames, quads, normalise_illumination=False)
        assert plain > 5.0, f"expected the fault, got {plain:.2f}px"
        assert levelled * 20 < plain, f"{levelled:.2f}px vs {plain:.2f}px"

    def test_a_severe_swing_is_handled_too(self):
        frames, quads = pan(printed, hunting(0.70))
        assert final_error(frames, quads) < 1.0

    def test_a_light_coming_on_is_handled(self):
        frames, quads = pan(printed, a_light_coming_on)
        assert final_error(frames, quads) < 1.0

    def test_steady_footage_pays_almost_nothing_for_it(self):
        frames, quads = pan(printed)
        levelled = final_error(frames, quads)
        plain = final_error(frames, quads, normalise_illumination=False)
        assert levelled < plain + 0.3, f"{levelled:.2f}px vs {plain:.2f}px"

    def test_it_can_be_switched_off(self):
        frames, quads = pan(printed, hunting(0.40))
        assert final_error(frames, quads, normalise_illumination=False) > 5.0


class TestTellingAScreenFromAPoster:
    """Following the texture inside a digital screen follows the advert being
    played rather than the panel playing it, and the tracker reports no
    failure while it does, so the export is confidently wrong."""

    def verdict(self, frames, quads):
        return core.looks_like_a_playing_screen(frames, quads)

    def test_a_printed_poster_is_left_alone(self):
        assert not self.verdict(*pan(printed)).independent

    def test_a_scrolling_advert_is_caught(self):
        assert self.verdict(*pan(scrolling_advert)).independent

    def test_a_screen_changing_slides_is_caught(self):
        """It agrees perfectly between changes and wildly at each one, so an
        average buries exactly the frames that matter."""
        assert self.verdict(*pan(changing_slides)).independent

    def test_a_screen_showing_one_still_picture_is_not_a_problem(self):
        """Nothing is moving independently, so the inside tracks it fine and
        there is nothing to warn about."""
        still = make_texture(320, 220, seed=7)
        assert not self.verdict(*pan(lambda i: still)).independent

    def test_an_exposure_change_is_not_mistaken_for_one(self):
        """Without levelling this is exactly what it was mistaken for, and an
        ordinary poster was condemned as a screen."""
        assert not self.verdict(*pan(printed, hunting(0.40))).independent

    def test_a_person_crossing_is_not_mistaken_for_one(self):
        assert not self.verdict(*pan(printed, a_person_crossing)).independent

    def test_a_fast_pan_is_not_mistaken_for_one(self):
        assert not self.verdict(*pan(printed, step=30.0)).independent

    def test_it_still_catches_a_screen_through_the_confounders(self):
        assert self.verdict(*pan(changing_slides, a_person_crossing)).independent
        assert self.verdict(*pan(scrolling_advert, hunting(0.40))).independent

    def test_the_evidence_is_reported(self):
        verdict = self.verdict(*pan(scrolling_advert))
        assert verdict.checked > 5
        assert verdict.exceeded >= 1
        assert verdict.worst_gap > 2.0
        assert "independent=True" in repr(verdict)

    def test_nothing_to_go_on_is_not_an_accusation(self):
        frames, quads = pan(printed)
        assert not core.looks_like_a_playing_screen(frames[:1], quads[:1]).independent


class TestWhatItSavesTheUser:
    def test_the_right_choice_is_worth_hundreds_of_pixels(self):
        """The whole point: this is the difference between a usable export and
        one whose insert has slid off the panel, with nothing to warn of it."""
        frames, quads = pan(scrolling_advert)
        inside = final_error(frames, quads)
        around = final_error(frames, quads,
                             feature_source=core.PlanarTracker.SURROUND)
        assert inside > 100, f"expected the fault, got {inside:.1f}px"
        assert around < 1.0, f"surround should be sound, got {around:.1f}px"
        assert core.looks_like_a_playing_screen(frames, quads).independent

    def test_reporting_trouble_is_not_the_same_as_recovering(self):
        """Some frames do report a failed step -- 25 of 39 on a scrolling
        advert, 9 on one changing slides. It makes no difference: the shape
        still ends hundreds of pixels off the panel, because a refused step
        only leaves the tracker where it was while the surface keeps moving.
        Checking this is why the verdict is worth having up front, rather than
        waiting for the tracker to complain."""
        for picture in (scrolling_advert, changing_slides):
            frames, quads = pan(picture)
            tracker = core.PlanarTracker(frames[0], quads[0])
            reported, error = 0, 0.0
            for frame, truth in zip(frames[1:], quads[1:]):
                if not tracker.track(frame).ok:
                    reported += 1
                error = float(np.linalg.norm(tracker.quad - truth, axis=1).mean())
            assert error > 100, f"{picture.__name__}: only {error:.0f}px off"
            assert reported < len(frames), "it cannot refuse every single step"


# --------------------------------------------------------------------------
# In the application
# --------------------------------------------------------------------------

import os                                                          # noqa: E402
import sys                                                         # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util                                              # noqa: E402

pytest.importorskip("PyQt5.QtWidgets")
from PyQt5.QtWidgets import QApplication, QMessageBox              # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def clip_of(picture_for, tmp_path, name):
    from conftest import write_video
    frames, quads = pan(picture_for, count=16)
    return write_video(frames, tmp_path / name), quads


@pytest.fixture
def window(qapp):
    win = galileo_app.MainWindow()
    yield win
    win.dirty = False
    win.close()


class TestTheApplicationAsksAboutIt:
    def mark_and_check(self, window, path, quad, monkeypatch, answer):
        asked = {}

        def fake(parent, title, text, *args, **kwargs):
            asked["title"], asked["text"] = title, text
            return answer

        monkeypatch.setattr(galileo_app.QMessageBox, "question", staticmethod(fake))
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.central_panel.load_video(path)
        window.central_panel.tracking_overlay.points = [
            tuple(map(float, p)) for p in quad]
        window.check_for_a_playing_screen()
        return asked

    def test_marking_a_screen_offers_the_change(self, window, tmp_path,
                                                monkeypatch):
        path, quads = clip_of(scrolling_advert, tmp_path, "screen.mp4")
        asked = self.mark_and_check(window, path, quads[0], monkeypatch,
                                    QMessageBox.Yes)
        assert "Screen" in asked.get("title", "")
        assert window.central_panel.feature_source == \
            core.PlanarTracker.SURROUND

    def test_declining_leaves_it_as_it_was(self, window, tmp_path, monkeypatch):
        path, quads = clip_of(scrolling_advert, tmp_path, "screen.mp4")
        self.mark_and_check(window, path, quads[0], monkeypatch, QMessageBox.No)
        assert window.central_panel.feature_source == \
            core.PlanarTracker.INTERIOR

    def test_marking_a_poster_says_nothing(self, window, tmp_path, monkeypatch):
        path, quads = clip_of(printed, tmp_path, "poster.mp4")
        asked = self.mark_and_check(window, path, quads[0], monkeypatch,
                                    QMessageBox.Yes)
        assert asked == {}
        assert window.central_panel.feature_source == \
            core.PlanarTracker.INTERIOR

    def test_it_does_not_nag_once_the_mode_is_already_on(self, window, tmp_path,
                                                        monkeypatch):
        path, quads = clip_of(scrolling_advert, tmp_path, "screen.mp4")
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        window.central_panel.load_video(path)
        window.central_panel.feature_source = core.PlanarTracker.SURROUND
        window.central_panel.tracking_overlay.points = [
            tuple(map(float, p)) for p in quads[0]]
        asked = {}
        monkeypatch.setattr(galileo_app.QMessageBox, "question",
                            staticmethod(lambda *a, **k: asked.setdefault("x", 1)))
        window.check_for_a_playing_screen()
        assert asked == {}

    def test_reading_ahead_leaves_playback_where_it_was(self, window, tmp_path):
        path, quads = clip_of(printed, tmp_path, "poster.mp4")
        window.central_panel.load_video(path)
        before = window.central_panel.get_current_frame_index()
        assert len(window.read_ahead(6)) == 6
        assert window.central_panel.get_current_frame_index() == before

    def test_an_unmarked_area_is_not_judged(self, window, tmp_path, monkeypatch):
        path, quads = clip_of(scrolling_advert, tmp_path, "screen.mp4")
        window.central_panel.load_video(path)
        asked = {}
        monkeypatch.setattr(galileo_app.QMessageBox, "question",
                            staticmethod(lambda *a, **k: asked.setdefault("x", 1)))
        window.central_panel.tracking_overlay.points = [(10.0, 10.0)]
        window.check_for_a_playing_screen()
        assert asked == {}
