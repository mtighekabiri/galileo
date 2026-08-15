"""Taking flicker out of footage shot under failing office lighting.

Two kinds need different corrections and the wrong one does nothing at all, so
most of what is worth testing here is that the right one gets chosen.
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import galileo_deflicker as deflicker      # noqa: E402
import conftest                            # noqa: E402

W, H, FPS, N = 320, 240, 25.0, 120


def office_frames(count=N):
    """A static office with something moving through it.

    The movement matters: it is what a brightness measurement can mistake for
    the lights changing, and what a banding measurement can mistake for bands.
    """
    plate = conftest.make_background(W, H, seed=5)
    texture = conftest.make_texture(seed=2)
    base = np.float32([[90, 70], [220, 66], [224, 160], [94, 164]])
    frames = []
    for i in range(count):
        frame = plate.copy()
        quad = base + np.float32([0.28 * i, 0.11 * i])
        th, tw = texture.shape[:2]
        matrix = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), quad)
        warped = cv2.warpPerspective(texture, matrix, (W, H))
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        frame[mask > 0] = warped[mask > 0]
        x = int(15 + 2.2 * i)
        cv2.rectangle(frame, (x, 150), (x + 35, 235), (40, 45, 60), -1)
        frames.append(frame)
    return frames


def with_whole_frame_flicker(frames, amplitude=0.18, hz=7.3):
    return [np.clip(f.astype(np.float32)
                    * (1.0 + amplitude * np.sin(2 * np.pi * hz * i / FPS)),
                    0, 255).astype(np.uint8)
            for i, f in enumerate(frames)]


def with_banding(frames, amplitude=0.16, cycles=3.0, hz=6.0):
    rows = np.arange(H, dtype=np.float32)[:, None]
    out = []
    for i, f in enumerate(frames):
        gain = 1.0 + amplitude * np.sin(2 * np.pi * cycles * rows / H
                                        + 2 * np.pi * hz * i / FPS)
        out.append(np.clip(f.astype(np.float32) * gain[:, :, None],
                           0, 255).astype(np.uint8))
    return out


def clip(tmp_path, frames, name):
    return conftest.write_video(frames, str(tmp_path / name), fps=FPS)


def flicker_of(frames):
    lum = [np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    return deflicker.whole_frame_amount(lum)


def banding_of(frames):
    rows = np.float64([np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), axis=1)
                       for f in frames])
    return deflicker.banding_amount(rows)


@pytest.fixture(scope="module")
def steady():
    return office_frames()


@pytest.fixture(scope="module")
def flickering(steady):
    return with_whole_frame_flicker(steady)


@pytest.fixture(scope="module")
def banded(steady):
    return with_banding(steady)


class TestTellingTheKindsApart:
    """Choosing wrong is not a near miss: the whole-frame correction leaves
    banding entirely untouched."""

    def test_steady_footage_is_left_alone(self, steady):
        rows = np.float64([np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), axis=1)
                           for f in steady])
        lum = [np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in steady]
        assert deflicker.classify(lum, rows, FPS).kind == "steady"

    def test_whole_frame_flicker_is_named(self, flickering):
        rows = np.float64([np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), axis=1)
                           for f in flickering])
        lum = [np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in flickering]
        assert deflicker.classify(lum, rows, FPS).kind == "frame"

    def test_banding_is_named(self, banded):
        rows = np.float64([np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), axis=1)
                           for f in banded])
        lum = [np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in banded]
        assert deflicker.classify(lum, rows, FPS).kind == "rows"

    def test_whole_frame_flicker_is_not_mistaken_for_banding(self, flickering):
        """Every row does brighten together, so the frame's own level has to
        be divided out before the rows are judged."""
        size, shape = banding_of(flickering)
        assert shape < deflicker.BAND_SHAPE, (
            f"{shape:.0f}% of the row variation read as one crawling band")

    def test_a_moving_figure_is_not_mistaken_for_banding(self, steady):
        size, shape = banding_of(steady)
        assert not (size > deflicker.BAND_SIZE and shape > deflicker.BAND_SHAPE)

    def test_real_banding_is_concentrated_in_one_peak(self, banded):
        size, shape = banding_of(banded)
        assert size > deflicker.BAND_SIZE and shape > deflicker.BAND_SHAPE


class TestScanning:
    def test_a_short_look_finds_whole_frame_flicker(self, tmp_path, flickering):
        report = deflicker.scan(clip(tmp_path, flickering, "flicker.mp4"))
        assert report.kind == "frame"
        assert report.whole > deflicker.FLICKER_SIZE

    def test_a_short_look_finds_banding(self, tmp_path, banded):
        report = deflicker.scan(clip(tmp_path, banded, "bands.mp4"))
        assert report.kind == "rows"

    def test_it_says_nothing_is_wrong_with_steady_footage(self, tmp_path, steady):
        report = deflicker.scan(clip(tmp_path, steady, "steady.mp4"))
        assert report.kind == "steady"

    def test_it_reads_far_fewer_frames_than_the_clip_has(self, tmp_path,
                                                         flickering):
        """This runs on every video that is opened."""
        report = deflicker.scan(clip(tmp_path, flickering, "f2.mp4"), sample=32)
        assert report.frames <= 32 < N

    def test_a_missing_file_is_not_an_exception(self, tmp_path):
        assert deflicker.scan(str(tmp_path / "nothing.mp4")) is None

    def test_it_names_the_beat_it_found(self, tmp_path, flickering):
        report = deflicker.scan(clip(tmp_path, flickering, "f3.mp4"))
        assert report.hz is not None and 5.0 < report.hz < 10.0


class TestCorrecting:
    def apply_all(self, frames, fixer):
        return [fixer.apply(f, i) for i, f in enumerate(frames)]

    def test_whole_frame_flicker_is_taken_out(self, tmp_path, flickering, steady):
        fixer = deflicker.Deflicker.measure(
            clip(tmp_path, flickering, "a.mp4"))
        assert fixer.mode == "frame"
        before, after = flicker_of(flickering), flicker_of(self.apply_all(flickering, fixer))
        assert before > 15.0
        assert after < 2.0, f"{before:.2f}% -> {after:.2f}%"

    def test_banding_is_taken_out(self, tmp_path, banded, steady):
        fixer = deflicker.Deflicker.measure(clip(tmp_path, banded, "b.mp4"))
        assert fixer.mode == "rows"
        before, _ = banding_of(banded)
        after, _ = banding_of(self.apply_all(banded, fixer))
        floor, _ = banding_of(steady)
        assert before > 8.0
        assert after <= floor * 1.15, f"{before:.2f}% -> {after:.2f}% (floor {floor:.2f}%)"

    def test_a_whole_frame_gain_does_nothing_for_banding(self, tmp_path, banded):
        """The reason the kind is worked out rather than offered as a choice."""
        fixer = deflicker.Deflicker.measure(
            clip(tmp_path, banded, "c.mp4"), mode="frame")
        before, _ = banding_of(banded)
        after, _ = banding_of(self.apply_all(banded, fixer))
        assert abs(after - before) < 1.0, (
            f"expected no change, got {before:.2f}% -> {after:.2f}%")

    def test_the_per_row_correction_also_fixes_whole_frame_flicker(
            self, tmp_path, flickering):
        """A row's own level carries the global flicker too, so 'rows'
        subsumes 'frame' -- 'frame' exists only because it is cheaper."""
        fixer = deflicker.Deflicker.measure(
            clip(tmp_path, flickering, "d.mp4"), mode="rows")
        after = flicker_of(self.apply_all(flickering, fixer))
        assert after < 2.0

    def test_steady_footage_yields_no_corrector(self, tmp_path, steady):
        assert deflicker.Deflicker.measure(clip(tmp_path, steady, "e.mp4")) is None

    def test_it_does_not_wreck_the_picture(self, tmp_path, flickering, steady):
        """Brightness is levelled; the scene has to survive intact."""
        fixer = deflicker.Deflicker.measure(clip(tmp_path, flickering, "f.mp4"))
        fixed = self.apply_all(flickering, fixer)
        error = np.abs(np.float32(fixed[60]) - np.float32(steady[60])).mean()
        assert error < 12.0, f"mean error {error:.1f} against the unflickered frame"

    def test_frames_past_the_measurement_are_untouched(self, tmp_path, flickering):
        fixer = deflicker.Deflicker.measure(clip(tmp_path, flickering, "g.mp4"))
        frame = flickering[0]
        assert fixer.apply(frame, len(fixer) + 5) is frame
        assert fixer.apply(frame, -1) is frame

    def test_a_gain_never_runs_away(self, tmp_path, flickering):
        fixer = deflicker.Deflicker.measure(clip(tmp_path, flickering, "h.mp4"),
                                            max_gain=1.4)
        assert fixer.gains.max() <= 1.4 + 1e-6
        assert fixer.gains.min() >= 1.0 / 1.4 - 1e-6

    def test_row_gains_are_refused_for_footage_of_another_height(
            self, tmp_path, banded):
        """Stretching them across a different frame would stripe it."""
        fixer = deflicker.Deflicker.measure(clip(tmp_path, banded, "i.mp4"))
        odd = cv2.resize(banded[0], (W // 2, H // 2))
        assert fixer.apply(odd, 0) is odd

    def test_cancelling_the_measurement_gives_nothing_back(self, tmp_path,
                                                           flickering):
        path = clip(tmp_path, flickering, "j.mp4")
        assert deflicker.Deflicker.measure(
            path, progress=lambda done, total: done < 10) is None

    def test_the_ends_of_the_clip_are_not_darkened(self, tmp_path, flickering):
        """A zero-padded reference used to drag the first and last frames
        towards black, and the correction then over-brightened them."""
        fixer = deflicker.Deflicker.measure(clip(tmp_path, flickering, "k.mp4"))
        edges = np.concatenate([fixer.gains[:5].ravel(), fixer.gains[-5:].ravel()])
        assert 0.75 < float(edges.mean()) < 1.35


# --------------------------------------------------------------------------
# Wired into the app
# --------------------------------------------------------------------------

import importlib.util                                    # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication                  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def app(qapp, tmp_path, flickering):
    path = clip(tmp_path, flickering, "loaded.mp4")
    window = galileo_app.MainWindow()
    window.central_panel.load_video(path)
    yield window, window.central_panel, path
    window.dirty = False
    window.close()


class TestWiredIntoTheApp:
    def test_opening_a_clip_looks_for_flicker(self, app):
        window, panel, path = app
        assert panel.flicker_report is not None
        assert panel.flicker_report.kind == "frame"

    def test_the_menu_says_what_it_found(self, app):
        window, panel, path = app
        window.refresh_deflicker_action()
        tip = window.title_bar.deflicker_action.toolTip()
        assert "Flicker found" in tip, tip

    def test_it_is_off_until_asked_for(self, app):
        window, panel, path = app
        assert panel.deflicker is None

    def test_switching_it_on_steadies_the_preview(self, app):
        window, panel, path = app
        rough = [np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                 for f in read_preview(panel)]
        panel.set_deflicker(True)
        assert panel.deflicker is not None
        steadied = [np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                    for f in read_preview(panel)]
        before = deflicker.whole_frame_amount(rough)
        after = deflicker.whole_frame_amount(steadied)
        assert before > 15.0 and after < 2.0, f"{before:.2f}% -> {after:.2f}%"

    def test_switching_it_off_puts_the_footage_back(self, app):
        window, panel, path = app
        panel.set_deflicker(True)
        panel.set_deflicker(False)
        assert panel.deflicker is None
        lum = [np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
               for f in read_preview(panel)]
        assert deflicker.whole_frame_amount(lum) > 15.0

    def test_the_correction_happens_before_tracking_sees_the_frame(self, app):
        """The whole point of correcting the plate first."""
        window, panel, path = app
        panel.set_deflicker(True)
        panel.cap.set(cv2.CAP_PROP_POS_FRAMES, 40)
        panel.read_frame()
        corrected = panel.deflicker.apply(raw_frame(path, 40), 40)
        assert np.array_equal(panel.prev_frame, corrected)

    def test_it_reaches_the_render(self, app, tmp_path):
        window, panel, path = app
        panel.set_deflicker(True)
        settings = window.build_render_settings(0, 40, 1.0,
                                                str(tmp_path / "out.mp4"))
        assert settings.deflicker is panel.deflicker

    def test_the_rendered_file_is_steady_too(self, app, tmp_path):
        """A preview steadied and a file that still pulses would be worse
        than not offering this at all."""
        window, panel, path = app
        overlay = panel.tracking_overlay
        overlay.overlay_bgra = np.dstack([
            np.full((80, 120, 3), (40, 40, 210), np.uint8),
            np.full((80, 120), 255, np.uint8)])
        corners = [(90.0, 70.0), (220.0, 66.0), (224.0, 160.0), (94.0, 164.0)]
        overlay.tracking_history = {
            i: [(x + 0.28 * i, y + 0.11 * i) for x, y in corners]
            for i in range(N)}
        overlay.points = list(overlay.tracking_history[0])

        panel.set_deflicker(True)
        out = str(tmp_path / "steady.mp4")
        settings = window.build_render_settings(0, N - 1, 1.0, out)
        worker = galileo_app.RenderWorker(settings)
        status = []
        worker.finished.connect(status.append)
        worker.run()
        assert status and not status[0].startswith("Error"), status

        capture = cv2.VideoCapture(out)
        lum = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            lum.append(np.median(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        capture.release()
        assert deflicker.whole_frame_amount(lum) < 3.0

    def test_the_setting_survives_a_save_and_load(self, app, tmp_path,
                                                  monkeypatch):
        window, panel, path = app
        panel.set_deflicker(True)
        project = str(tmp_path / "flick.json")
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        monkeypatch.setattr(galileo_app.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        for name in ("information", "warning", "question"):
            monkeypatch.setattr(galileo_app.QMessageBox, name,
                                staticmethod(lambda *a, **k: None))
        window.save_project()

        import json
        with open(project) as handle:
            assert json.load(handle)["deflicker"] == "frame"

        panel.set_deflicker(False)
        window.dirty = False
        window.load_project()
        assert panel.deflicker is not None
        assert panel.deflicker.mode == "frame"

    def test_steady_footage_offers_nothing_to_correct(self, qapp, tmp_path,
                                                      steady, monkeypatch):
        seen = []
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: seen.append(a)))
        window = galileo_app.MainWindow()
        try:
            window.central_panel.load_video(clip(tmp_path, steady, "ok.mp4"))
            assert window.central_panel.flicker_report.kind == "steady"
            window.toggle_deflicker(True)
            assert window.central_panel.deflicker is None
            assert window.title_bar.deflicker_action.isChecked() is False
            assert seen, "the user was told nothing"
        finally:
            window.dirty = False
            window.close()


def read_preview(panel, count=60):
    """Step the panel through frames and collect what it put on screen."""
    panel.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for _ in range(count):
        panel.read_frame()
        frames.append(panel.prev_frame.copy())
    return frames


def raw_frame(path, index):
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    capture.release()
    assert ok
    return frame
