"""Saving and reloading has to give back exactly what was there.

Tracking data is the expensive part of using this tool -- a clip is tracked
once and the result reused, reloaded, handed to someone else and exported
against. A save that quietly rounds is the worst kind of fault, because the
file looks fine and nothing ever reports a problem.
"""

import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util

import numpy as np
import pytest

import galileo_core as core

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication                          # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


#: Values chosen to break careless handling: long decimals, a number that is
#: not representable in binary, one that rounds differently in float32, a
#: negative and an off-screen corner, and something very close to zero.
AWKWARD = [(123.456789012345, 67.89012345678901),
           (-4.7000000000000002, 1919.9999999999998),
           (0.1 + 0.2, 1e-7),
           (1234.5678901234567, -890.12345678)]


def moved(points, dx, dy):
    return [(x + dx, y + dy) for x, y in points]


@pytest.fixture
def placement(qapp):
    made = galileo_app.Placement("Awkward")
    made.points = list(AWKWARD)
    made.tracking_history = {0: list(AWKWARD),
                             7: moved(AWKWARD, 1 / 3, -2 / 3),
                             999: moved(AWKWARD, -0.000001, 0.000001)}
    made.curvature = np.array(
        [[[0.123456789, -0.987654321], [1e-7, 0.5]]] * 4, np.float32)
    made.curved_enabled = True
    return made


def round_trip(placement):
    return galileo_app.Placement.from_dict(
        json.loads(json.dumps(placement.to_dict())))


class TestNothingIsLostOnTheWayToDisk:
    def test_the_corners_come_back_bit_for_bit(self, placement):
        restored = round_trip(placement)
        assert restored.points == [tuple(p) for p in placement.points]

    def test_every_tracked_frame_comes_back_bit_for_bit(self, placement):
        """This went through a float32 coercion on the way out, which rounded
        a hand-placed corner before it ever reached the file."""
        restored = round_trip(placement)
        assert set(restored.tracking_history) == set(placement.tracking_history)
        for frame, corners in placement.tracking_history.items():
            assert restored.tracking_history[frame] == [tuple(c) for c in corners]

    def test_the_curvature_comes_back_bit_for_bit(self, placement):
        restored = round_trip(placement)
        assert np.array_equal(restored.curvature, placement.curvature)
        assert restored.curvature.dtype == placement.curvature.dtype

    def test_saving_repeatedly_does_not_drift(self, placement):
        """A file opened, edited a little and saved again, over and over, is
        the ordinary way this is used."""
        current = placement.to_dict()
        for _ in range(10):
            current = round_trip(
                galileo_app.Placement.from_dict(current)).to_dict()
        assert current["tracking_history"] == placement.to_dict()["tracking_history"]
        assert current["points"] == placement.to_dict()["points"]

    def test_the_frame_numbers_stay_themselves(self, placement):
        restored = round_trip(placement)
        assert sorted(restored.tracking_history) == [0, 7, 999]

    def test_the_settings_come_back_too(self, placement):
        placement.brightness = 37
        placement.contrast = 1.234567
        placement.colourise_enabled = True
        placement.morph.yaw = 12.5
        placement.morph.bow_h = 0.6789
        placement.blend.colour = 0.4321
        restored = round_trip(placement)
        assert restored.brightness == 37
        assert restored.contrast == 1.234567
        assert restored.colourise_enabled is True
        assert restored.morph.yaw == 12.5
        assert restored.morph.bow_h == 0.6789
        assert restored.blend.colour == 0.4321


class TestTheHistoryConverter:
    def test_it_keeps_whatever_precision_it_was_given(self):
        history = {4: AWKWARD}
        out = core.history_to_lists(history)
        assert [tuple(c) for c in out[4]] == AWKWARD

    def test_a_float32_history_survives_untouched(self):
        """What the tracker itself produces; it must not be widened and then
        come back different."""
        history = {3: np.array(AWKWARD, np.float32)}
        out = core.history_to_lists(history)
        assert np.array_equal(np.array(out[3], np.float32), history[3])

    def test_it_drops_frames_that_are_not_a_quad(self):
        assert core.history_to_lists({1: None, 2: AWKWARD[:2], 3: AWKWARD}) \
            .keys() == {3}

    def test_the_keys_come_out_as_numbers(self):
        assert list(core.history_to_lists({"12": AWKWARD})) == [12]


class TestTheStandaloneTrackingFile:
    def test_it_writes_and_reads_the_same_numbers(self, qapp, tmp_path,
                                                  monkeypatch):
        window = galileo_app.MainWindow()
        try:
            overlay = window.central_panel.tracking_overlay
            overlay.tracking_history = {0: list(AWKWARD),
                                        5: moved(AWKWARD, 1 / 7, -1 / 11)}
            original = {k: [tuple(c) for c in v]
                        for k, v in overlay.tracking_history.items()}

            path = str(tmp_path / "tracking.json")
            monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                                staticmethod(lambda *a, **k: (path, "")))
            monkeypatch.setattr(galileo_app.QFileDialog, "getOpenFileName",
                                staticmethod(lambda *a, **k: (path, "")))
            monkeypatch.setattr(galileo_app.QMessageBox, "information",
                                staticmethod(lambda *a, **k: None))
            window.save_tracking_points()

            overlay.tracking_history = {}
            window.dirty = False
            window.load_tracking_points()

            reloaded = {k: [tuple(c) for c in v]
                        for k, v in overlay.tracking_history.items()}
            assert reloaded == original
        finally:
            window.dirty = False
            window.close()
