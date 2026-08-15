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

import galileo_core as core
import galileo_blend as blend
from conftest import quad_path

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)


def creative(colour):
    bgr = np.full((120, 180, 3), colour, np.uint8)
    return np.dstack([bgr, np.full((120, 180), 255, np.uint8)])


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def app(qapp, clip_video):
    path, truth = clip_video
    window = galileo_app.MainWindow()
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
        left_tracker = first.tracker

        overlay.set_active(1)
        panel.read_frame()
        assert second.tracker is not None
        assert second.tracker is not left_tracker

    def test_feature_source_is_per_placement(self, two, monkeypatch):
        """One shot can hold a printed poster and a digital screen."""
        window, panel, overlay, truth, first, second = two
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        overlay.set_active(1)
        window.toggle_digital_screen(True)
        assert second.feature_source == core.PlanarTracker.SURROUND
        assert first.feature_source == core.PlanarTracker.INTERIOR


class TestTracking:
    """A pass follows the placement being worked on, and only that one.

    Two screens in a concourse shot get tracked one at a time: mark the first,
    play through, correct it by hand where it slipped; then add the second and
    play through again. When every placement was tracked on every frame, that
    second pass re-tracked the first one as well and overwrote its history from
    wherever its corners happened to sit -- discarding the hand corrections,
    and looking right on screen because the shape had just been redrawn from
    the same bad guess that replaced them.
    """

    def test_the_active_placement_is_tracked(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        second.tracking_history = {0: second.tracking_history[0]}
        panel.tracking_mode = True
        for _ in range(4):
            panel.read_frame()
        assert len(second.tracking_history) >= 3

    def test_a_pass_leaves_the_other_placements_history_alone(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        before = {index: shape[:] for index, shape in first.tracking_history.items()}
        panel.tracking_mode = True
        for _ in range(4):
            panel.read_frame()
        assert first.tracking_history == before

    def test_hand_corrections_survive_a_pass_over_another_placement(self, two):
        """The case that loses real work: corrected corners must stay put."""
        window, panel, overlay, truth, first, second = two
        corrected = [(x + 7.0, y - 5.0) for x, y in first.tracking_history[2]]
        first.tracking_history[2] = corrected

        overlay.set_active(1)
        panel.tracking_mode = True
        for _ in range(4):
            panel.read_frame()
        assert first.tracking_history[2] == corrected

    def test_an_untracked_placement_still_follows_its_own_record(self, two):
        """Not tracking it must not mean freezing it in the frame."""
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        panel.tracking_mode = True
        for _ in range(4):
            panel.read_frame()
        index = panel.get_current_frame_index()
        assert first.points == first.tracking_history[index]

    def test_a_placement_with_nothing_recorded_here_is_left_where_it_is(self, two):
        window, panel, overlay, truth, first, second = two
        first.tracking_history = {}
        held = first.points[:]
        overlay.set_active(1)
        panel.tracking_mode = True
        for _ in range(3):
            panel.read_frame()
        assert first.points == held
        assert first.tracking_history == {}

    def test_the_skipped_placement_drops_its_stale_tracker(self, two):
        """Coming back to it must re-anchor, not resume from an old frame."""
        window, panel, overlay, truth, first, second = two
        panel.tracking_mode = True
        panel.read_frame()
        assert first.tracker is not None

        overlay.set_active(1)
        for _ in range(3):
            panel.read_frame()
        assert first.tracker is None
        assert first.kalman_filters == []

    def test_one_failing_does_not_disturb_the_other(self, two):
        """A lost surface must not take the rest of the shot down with it."""
        window, panel, overlay, truth, first, second = two
        second.points = [(5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0)]
        overlay.set_active(1)
        before = {index: shape[:] for index, shape in first.tracking_history.items()}
        panel.tracking_mode = True
        for _ in range(3):
            panel.read_frame()
        assert first.tracking_history == before

    def test_with_tracking_off_every_placement_follows_its_record(self, two):
        """Scrubbing used to move the active placement and freeze the rest."""
        window, panel, overlay, truth, first, second = two
        panel.tracking_mode = False
        panel.jump_to_frame(5)
        assert first.points == first.tracking_history[5]
        assert second.points == second.tracking_history[5]

    def test_the_label_names_the_placement_being_tracked(self, two):
        window, panel, overlay, truth, first, second = two
        overlay.set_active(1)
        panel.on_tracking_toggled(True)
        assert "Right" in panel.tracking_label.text()
        overlay.set_active(0)
        panel.update_tracking_availability()
        assert "Left" in panel.tracking_label.text()

    def test_the_label_stays_plain_with_a_single_placement(self, app):
        window, panel, overlay, truth = app
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        panel.on_tracking_toggled(True)
        assert panel.tracking_label.text() == "Tracking on"

    def test_a_long_placement_name_is_shortened_not_cut_off(self, two):
        """Real names are long: "Ryanair concourse portrait by gate 42"."""
        window, panel, overlay, truth, first, second = two
        second.name = "Ryanair concourse portrait screen by gate 42"
        overlay.set_active(1)
        panel.on_tracking_toggled(True)
        panel.status_bar.layout().activate()

        label = panel.tracking_label
        assert label.width() >= label.sizeHint().width(), "text is being clipped"
        assert "…" in label.text()
        assert second.name in label.toolTip()


class TestAccountingForAPass:
    """A pass writes into the history; the rest of the app has to notice.

    Nothing did. The placement list went on showing the count from before the
    pass -- and with a pass now covering one placement, that list is the only
    place you can see how far each has been followed. Worse, the unsaved-work
    flag stayed clear, so tracking a whole clip and closing the window threw
    the lot away with no prompt: a single corner nudged by hand marked the
    project dirty, two hundred tracked frames did not.
    """

    def test_the_count_in_the_list_follows_a_pass(self, two):
        window, panel, overlay, truth, first, second = two
        first.tracking_history = {0: first.tracking_history[0]}
        window.refresh_placement_list()
        assert "1 frame" in window.placement_list.item(0).text()

        panel.on_tracking_toggled(True)
        for _ in range(5):
            panel.read_frame()
        panel.on_tracking_toggled(False)
        assert f"{len(first.tracking_history)} frames" in \
            window.placement_list.item(0).text()

    def test_a_pass_counts_as_unsaved_work(self, two):
        window, panel, overlay, truth, first, second = two
        window.dirty = False
        panel.on_tracking_toggled(True)
        for _ in range(4):
            panel.read_frame()
        panel.on_tracking_toggled(False)
        assert window.dirty is True

    def test_running_off_the_end_of_the_clip_ends_the_pass(self, two):
        window, panel, overlay, truth, first, second = two
        window.dirty = False
        panel.on_tracking_toggled(True)
        for _ in range(len(truth) + 5):
            panel.read_frame()
        assert window.dirty is True

    def test_pausing_ends_the_pass(self, two):
        window, panel, overlay, truth, first, second = two
        window.dirty = False
        panel.on_tracking_toggled(True)
        panel.playing = True
        for _ in range(4):
            panel.read_frame()
        panel.toggle_play_pause()
        assert window.dirty is True

    def test_toggling_with_nothing_tracked_invents_no_unsaved_work(self, two):
        window, panel, overlay, truth, first, second = two
        window.dirty = False
        panel.on_tracking_toggled(True)
        panel.on_tracking_toggled(False)
        assert window.dirty is False


class TestRendering:
    def test_both_creatives_reach_the_file(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        out = str(tmp_path / "both.mp4")
        settings = window.build_render_settings(0, 9, 1.0, out)
        assert [p.name for p in settings.placements] == ["Left", "Right"]

        worker = galileo_app.RenderWorker(settings)
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
        worker = galileo_app.RenderWorker(settings)
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
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        monkeypatch.setattr(galileo_app.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        for name in ("information", "warning"):
            monkeypatch.setattr(galileo_app.QMessageBox, name,
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
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
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
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))
        monkeypatch.setattr(galileo_app.QMessageBox, "information",
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


class TestOneCreativeInSeveralPlacements:
    """The library describes one placement at a time -- the active one. It
    used to describe whichever had been filled last, so a card could read
    "Insert" over a creative that was in the video, and filling a second
    placement from the same card took two clicks with a lie in between."""

    def cards(self, window):
        return list(window.library_cards())

    def button(self, card):
        from PyQt5.QtWidgets import QPushButton
        return card.findChild(QPushButton, "selectBtn")

    def sources(self, overlay):
        return [p.overlay_source_path for p in overlay.placements]

    def test_the_same_creative_can_fill_two_placements(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        art = str(tmp_path / "one.png")
        cv2.imwrite(art, np.full((80, 120, 3), 200, np.uint8))
        window.add_overlay(art)
        card = self.cards(window)[0]

        overlay.set_active(0)
        window.refresh_library_cards()
        self.button(card).click()
        assert overlay.placements[0].overlay_source_path == art

        overlay.set_active(1)
        window.refresh_library_cards()
        assert self.button(card).text() == "Insert", (
            "it should not claim to be in a placement it is not in")
        self.button(card).click()
        assert self.sources(overlay) == [art, art]

    def test_the_labels_follow_the_active_placement(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        red = str(tmp_path / "red.png")
        blue = str(tmp_path / "blue.png")
        cv2.imwrite(red, np.full((80, 120, 3), (30, 30, 220), np.uint8))
        cv2.imwrite(blue, np.full((80, 120, 3), (220, 120, 30), np.uint8))
        window.add_overlay(red)
        window.add_overlay(blue)
        first_card, second_card = self.cards(window)

        overlay.set_active(0)
        window.refresh_library_cards()
        self.button(first_card).click()
        overlay.set_active(1)
        window.refresh_library_cards()
        self.button(second_card).click()

        overlay.set_active(0)
        window.refresh_library_cards()
        assert [self.button(c).text() for c in (first_card, second_card)] == \
            ["Inserted", "Insert"]
        overlay.set_active(1)
        window.refresh_library_cards()
        assert [self.button(c).text() for c in (first_card, second_card)] == \
            ["Insert", "Inserted"]

    def test_filling_one_placement_leaves_the_other_alone(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        red = str(tmp_path / "red.png")
        blue = str(tmp_path / "blue.png")
        cv2.imwrite(red, np.full((80, 120, 3), (30, 30, 220), np.uint8))
        cv2.imwrite(blue, np.full((80, 120, 3), (220, 120, 30), np.uint8))
        window.add_overlay(red)
        window.add_overlay(blue)
        cards = self.cards(window)

        overlay.set_active(0)
        window.refresh_library_cards()
        self.button(cards[0]).click()
        overlay.set_active(1)
        window.refresh_library_cards()
        self.button(cards[1]).click()
        assert self.sources(overlay) == [red, blue]

    def test_taking_a_creative_out_clears_it_everywhere(self, two, tmp_path):
        """Removing it from the library cannot leave a placement holding it."""
        window, panel, overlay, truth, first, second = two
        art = str(tmp_path / "one.png")
        cv2.imwrite(art, np.full((80, 120, 3), 200, np.uint8))
        window.add_overlay(art)
        card = self.cards(window)[0]

        for row in (0, 1):
            overlay.set_active(row)
            window.refresh_library_cards()
            self.button(card).click()
        assert self.sources(overlay) == [art, art]

        overlay.set_active(0)
        window.refresh_library_cards()
        self.button(card).click()          # a click on "Inserted" removes it
        assert overlay.placements[0].overlay_source_path is None

    def test_both_are_rendered(self, two, tmp_path):
        window, panel, overlay, truth, first, second = two
        art = str(tmp_path / "one.png")
        cv2.imwrite(art, np.full((80, 120, 3), 200, np.uint8))
        window.add_overlay(art)
        card = self.cards(window)[0]
        for row in (0, 1):
            overlay.set_active(row)
            window.refresh_library_cards()
            self.button(card).click()
        assert len(overlay.ready_placements()) == 2
