"""Guarding the user's tracking work.

Every test here corresponds to a way the app used to silently destroy or fail
to record what the operator had marked. Tracking a clip is the expensive part
of using this tool, so these are correctness tests, not polish.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util

import numpy as np
import pytest

import lumen_core as core
from conftest import quad_path, render_clip, write_video

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent

_spec = importlib.util.spec_from_file_location(
    "lumen_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "LUMEN_Insertion_Tool_1.0.0.py"))
lumen_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lumen_app)

KEY_PRESS = 6   # QEvent.KeyPress


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def app(qapp, clip_video):
    path, truth = clip_video
    window = lumen_app.MainWindow()
    window.show()
    window.central_panel.load_video(path)
    panel = window.central_panel
    panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
    panel.current_frame_index = 0
    yield window, panel, panel.tracking_overlay, truth
    window.dirty = False
    window.close()


def press(widget, key, modifiers=Qt.NoModifier):
    event = QKeyEvent(KEY_PRESS, key, modifiers)
    widget.keyPressEvent(event)
    return event


class TestSingleHistory:
    def test_panel_and_overlay_share_one_store(self, app):
        window, panel, overlay, truth = app
        assert panel.tracking_history is overlay.tracking_history

    def test_writing_through_either_name_is_the_same_data(self, app):
        window, panel, overlay, truth = app
        panel.tracking_history[3] = [(1.0, 2.0)] * 4
        assert overlay.tracking_history[3] == [(1.0, 2.0)] * 4

    def test_assignment_still_works(self, app):
        """load_project assigns wholesale; the property must accept that."""
        window, panel, overlay, truth = app
        panel.tracking_history = {7: [(0.0, 0.0)] * 4}
        assert overlay.tracking_history == {7: [(0.0, 0.0)] * 4}


class TestNothingIsSilentlyDestroyed:
    def test_turning_tracking_off_keeps_the_shape(self, app):
        """Rewind, backward scrub and Go-to-Frame all switch tracking off."""
        window, panel, overlay, truth = app
        overlay.commit_shape()
        panel.tracking_mode = True
        panel.on_tracking_toggled(False)
        assert len(overlay.points) == 4

    def test_jumping_to_an_untracked_frame_keeps_the_shape(self, app):
        window, panel, overlay, truth = app
        overlay.commit_shape()
        panel.jump_to_frame(11)
        assert len(overlay.points) == 4, "shape vanished on an untracked frame"

    def test_jumping_to_a_tracked_frame_loads_that_shape(self, app):
        window, panel, overlay, truth = app
        marker = [(9.0, 9.0), (19.0, 9.0), (19.0, 19.0), (9.0, 19.0)]
        overlay.tracking_history[6] = marker
        panel.jump_to_frame(6)
        assert overlay.points == marker

    def test_rewinding_keeps_the_shape(self, app):
        window, panel, overlay, truth = app
        overlay.commit_shape()
        panel.jump_to_frame(4)
        panel.on_rewind()
        assert len(overlay.points) == 4


class TestEditsArePersisted:
    def test_arrow_nudge_reaches_the_history(self, app):
        """The documented precision tool used to change only what was drawn."""
        window, panel, overlay, truth = app
        overlay.tracking_history.clear()
        overlay.selected_point_index = 0
        before = overlay.points[0]

        press(overlay, Qt.Key_Right)

        assert 0 in overlay.tracking_history
        assert overlay.tracking_history[0][0][0] == pytest.approx(before[0] + 1.0)

    def test_shift_gives_a_coarse_step(self, app):
        window, panel, overlay, truth = app
        overlay.selected_point_index = 1
        before = overlay.points[1][0]
        press(overlay, Qt.Key_Right, Qt.ShiftModifier)
        assert overlay.points[1][0] == pytest.approx(before + 10.0)

    def test_ctrl_gives_a_subpixel_step(self, app):
        window, panel, overlay, truth = app
        overlay.selected_point_index = 1
        before = overlay.points[1][0]
        press(overlay, Qt.Key_Right, Qt.ControlModifier)
        assert overlay.points[1][0] == pytest.approx(before + 0.25)

    def test_nudge_moves_only_the_selected_corner(self, app):
        window, panel, overlay, truth = app
        original = list(overlay.points)
        overlay.selected_point_index = 2
        press(overlay, Qt.Key_Right)
        moved = [i for i in range(4) if overlay.points[i] != original[i]]
        assert moved == [2]

    def test_selecting_the_fifth_slot_moves_the_whole_shape(self, app):
        window, panel, overlay, truth = app
        original = list(overlay.points)
        overlay.selected_point_index = 4
        press(overlay, Qt.Key_Right)
        assert all(overlay.points[i][0] == pytest.approx(original[i][0] + 1.0)
                   for i in range(4))

    def test_an_incomplete_shape_clears_its_history_entry(self, app):
        window, panel, overlay, truth = app
        overlay.commit_shape()
        assert 0 in overlay.tracking_history
        overlay.points = overlay.points[:2]
        overlay.commit_shape()
        assert 0 not in overlay.tracking_history


class TestKeyboardReachesTheRightPlace:
    def test_d_and_c_fall_through_to_the_panel(self, app):
        """The overlay holds focus almost always and used to swallow these."""
        window, panel, overlay, truth = app
        for key in (Qt.Key_D, Qt.Key_C):
            assert not press(overlay, key).isAccepted(), f"{key} was swallowed"

    def test_number_keys_select_a_corner(self, app):
        window, panel, overlay, truth = app
        press(overlay, Qt.Key_3)
        assert overlay.selected_point_index == 2

    def test_arrow_keys_are_consumed(self, app):
        window, panel, overlay, truth = app
        overlay.selected_point_index = 0
        assert press(overlay, Qt.Key_Left).isAccepted()


class TestTrackingSwitchGuard:
    def test_switch_is_disabled_without_a_shape(self, app):
        window, panel, overlay, truth = app
        overlay.points = []
        panel.update_tracking_availability()
        assert not panel.switch.isEnabled()

    def test_switch_is_enabled_once_the_shape_is_complete(self, app):
        window, panel, overlay, truth = app
        panel.update_tracking_availability()
        assert panel.switch.isEnabled()

    def test_enabling_without_a_shape_is_refused(self, app, monkeypatch):
        """It used to go green and record nothing for the whole clip."""
        window, panel, overlay, truth = app
        monkeypatch.setattr(lumen_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        overlay.points = []
        panel.on_tracking_toggled(True)
        assert panel.tracking_mode is False


class TestUnsavedWork:
    def test_editing_marks_the_project_dirty(self, app):
        window, panel, overlay, truth = app
        window.dirty = False
        overlay.commit_shape()
        assert window.dirty is True

    def test_title_shows_the_unsaved_marker(self, app):
        window, panel, overlay, truth = app
        window.mark_dirty()
        assert window.title_bar.title_label.text().endswith("*")

    def test_clean_project_needs_no_confirmation(self, app):
        window, panel, overlay, truth = app
        window.mark_clean()
        assert window.confirm_discard("x") is True

    def test_cancelling_the_prompt_blocks_the_action(self, app, monkeypatch):
        window, panel, overlay, truth = app
        window.mark_dirty()
        monkeypatch.setattr(lumen_app.QMessageBox, "warning",
                            staticmethod(lambda *a, **k: lumen_app.QMessageBox.Cancel))
        assert window.confirm_discard("x") is False

    def test_discard_allows_the_action(self, app, monkeypatch):
        window, panel, overlay, truth = app
        window.mark_dirty()
        monkeypatch.setattr(lumen_app.QMessageBox, "warning",
                            staticmethod(lambda *a, **k: lumen_app.QMessageBox.Discard))
        assert window.confirm_discard("x") is True

    def test_title_names_the_loaded_video(self, app):
        window, panel, overlay, truth = app
        assert "moving.mp4" in window.title_bar.title_label.text()


class TestDeadCodeIsGone:
    @pytest.mark.parametrize("name", ["set_tracking_toggle_state",
                                      "auto_enable_tracking_if_shape_ready",
                                      "save_tracking_points_for_frame"])
    def test_removed(self, app, name):
        """Each referenced attributes that do not exist, so calling one raised."""
        window, panel, overlay, truth = app
        assert not hasattr(panel, name)
