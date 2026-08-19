"""Inserting a *video* creative rather than a still one.

A still fills its area on every frame the area is tracked. A video has to be
told which of its own frames belongs on which frame of the base clip, and that
mapping is what these cover: where the creative starts, that it advances one
frame per frame, that a render of part of the clip picks up in the right place,
and that a creative shorter than the shot does not leave the footage untouched
without saying so.

The creative here is a grey ramp -- frame *i* is a flat ``20 + 20*i`` -- so the
frame that landed can be read straight back out of the render. The step is wide
enough that neither the mp4v round trip nor the warp can move a level far
enough to be mistaken for its neighbour.
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
import galileo_morph as morphlib
from conftest import make_texture, quad_path, render_clip, write_video

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtWidgets import QApplication, QPushButton

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)

FRAMES = 12
STEP = 20          # grey levels between one creative frame and the next
FLOOR = 20         # the level of its first frame


def creative_level(index: int) -> int:
    return FLOOR + STEP * index


def creative_clip(path, count=FRAMES, fps=25.0):
    """A creative whose every frame states which frame it is."""
    frames = [np.full((120, 200, 3), creative_level(i), np.uint8)
              for i in range(count)]
    return write_video(frames, path, fps=fps)


def landed(frame, quad):
    """Which creative frame is showing inside ``quad``, by its grey level.

    Read from the middle of the area: the outer ring is a blend of the creative
    and the footage under it, and the mp4v round trip moves a flat level by a
    few, neither of which is worth being sensitive to.
    """
    height, width = frame.shape[:2]
    mask = core.quad_to_mask(np.float32(quad), width, height)
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))
    if not cv2.countNonZero(mask):
        return None
    level = frame[mask > 0].mean()
    return int(round((level - FLOOR) / STEP))


def is_untouched(rendered, original, quad):
    """True if the area still holds the footage rather than a creative."""
    height, width = rendered.shape[:2]
    mask = cv2.erode(core.quad_to_mask(np.float32(quad), width, height),
                     np.ones((15, 15), np.uint8))
    difference = cv2.absdiff(rendered, original)[mask > 0].mean()
    return difference < 8.0


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="session")
def base_clip(tmp_path_factory):
    """A base video and the ground-truth path of the surface in it."""
    quads = quad_path(FRAMES)
    frames = render_clip(quads, texture=make_texture())
    directory = tmp_path_factory.mktemp("base")
    return write_video(frames, directory / "base.mp4"), quads, frames


@pytest.fixture(scope="session")
def creative_video(tmp_path_factory):
    return creative_clip(tmp_path_factory.mktemp("creative") / "advert.mp4")


class _Placement:
    """The fields PlacementSnapshot reads, without a widget behind them."""

    def __init__(self, video_path, history, insert_frame=0, name="Screen"):
        self.name = name
        self.tracking_history = history
        self.curvature = np.zeros((4, 2, 2), np.float32)
        self.curved_enabled = False
        self.overlay_bgra = None
        self.inserted_overlay_is_video = True
        self.overlay_video_path = video_path
        self.inserted_overlay_start_frame = insert_frame
        self.brightness = 0
        self.contrast = 1.0
        self.colourise_enabled = False
        # Off, so what comes back out of the file is the creative's own level
        # rather than one matched to the footage around it.
        self.blend = blend.BlendSettings.off()
        self.morph = morphlib.Morph()


def render(base, snapshot, out_path, start=0, end=FRAMES - 1, fps=25.0):
    """Run a render to completion and return (status, frames)."""
    settings = galileo_app.RenderSettings(
        base, str(out_path), start, end, 1.0, fps, snapshot.history,
        np.zeros((4, 2, 2), np.float32), False,
        placements=[snapshot], include_audio=False)
    worker = galileo_app.RenderWorker(settings)
    statuses = []
    worker.finished.connect(statuses.append)
    worker.run()

    frames = []
    capture = cv2.VideoCapture(str(out_path))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return (statuses[0] if statuses else ""), frames


def full_history(quads):
    return {i: [tuple(map(float, c)) for c in quad]
            for i, quad in enumerate(quads)}


class TestWhereTheCreativeStarts:
    def test_the_playhead_at_insert_time_does_not_decide_it(self, base_clip,
                                                            creative_video):
        """The reported fault: mark, track the clip through, *then* insert.

        The playhead is at the end of the shot by then, and that used to become
        the creative's first frame -- so the render was the base video back,
        with the advert only in its final moments.
        """
        base, quads, _ = base_clip
        history = full_history(quads)
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(creative_video, history, insert_frame=FRAMES - 1), 1.0)
        assert snapshot.start_frame == 0

    def test_an_untracked_placement_falls_back_to_the_insert_frame(self,
                                                                   creative_video):
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(creative_video, {}, insert_frame=7), 1.0)
        assert snapshot.start_frame == 7

    def test_it_is_the_first_tracked_frame_not_the_first_of_the_clip(self,
                                                                    base_clip,
                                                                    creative_video):
        base, quads, _ = base_clip
        history = {i: [tuple(map(float, c)) for c in quads[i]]
                   for i in range(5, FRAMES)}
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(creative_video, history, insert_frame=0), 1.0)
        assert snapshot.start_frame == 5

    def test_tracking_done_after_the_insert_moves_it(self, qapp, creative_video):
        """The anchor is derived, so an area tracked later still gets the
        creative from its first frame rather than partway through."""
        placement = galileo_app.Placement("Screen")
        placement.inserted_overlay_is_video = True
        placement.inserted_overlay_start_frame = 0
        assert placement.creative_frame_for(40) == 40
        placement.tracking_history = {i: [(0.0, 0.0)] * 4
                                      for i in range(40, 60)}
        assert placement.creative_frame_for(40) == 0
        assert placement.creative_frame_for(47) == 7

    def test_a_frame_before_the_start_holds_the_first_one(self, qapp):
        """The area can be carried to a frame with no tracking of its own; the
        creative goes with it rather than blinking out."""
        placement = galileo_app.Placement("Screen")
        placement.inserted_overlay_is_video = True
        placement.tracking_history = {i: [(0.0, 0.0)] * 4 for i in range(10, 20)}
        assert placement.creative_frame_for(3) == 0


class TestRenderingAVideoCreative:
    def test_inserting_after_tracking_fills_the_whole_range(self, qapp,
                                                            base_clip,
                                                            creative_video,
                                                            tmp_path):
        base, quads, plain = base_clip
        history = full_history(quads)
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(creative_video, history, insert_frame=FRAMES - 1), 1.0)
        status, frames = render(base, snapshot, tmp_path / "late.mp4")

        assert status.startswith("Completed"), status
        assert len(frames) == FRAMES
        untouched = [i for i, frame in enumerate(frames)
                     if is_untouched(frame, plain[i], quads[i])]
        assert not untouched, f"the creative never reached frames {untouched}"

    def test_each_frame_carries_the_next_frame_of_the_creative(self, qapp,
                                                               base_clip,
                                                               creative_video,
                                                               tmp_path):
        base, quads, _ = base_clip
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(creative_video, full_history(quads)), 1.0)
        status, frames = render(base, snapshot, tmp_path / "advance.mp4")

        assert status.startswith("Completed"), status
        assert [landed(frame, quads[i]) for i, frame in enumerate(frames)] \
            == list(range(FRAMES))

    def test_a_sub_range_render_picks_up_where_the_preview_was(self, qapp,
                                                              base_clip,
                                                              creative_video,
                                                              tmp_path):
        """Frame 9 of the clip carries frame 9 of the creative whether the
        render started at the top of the clip or halfway down it."""
        base, quads, _ = base_clip
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(creative_video, full_history(quads)), 1.0)
        status, frames = render(base, snapshot, tmp_path / "part.mp4",
                                start=9, end=FRAMES - 1)

        assert status.startswith("Completed"), status
        assert [landed(frame, quads[9 + i]) for i, frame in enumerate(frames)] \
            == list(range(9, FRAMES))

    def test_a_creative_slower_than_the_base_clip_plays_at_its_own_speed(
            self, qapp, base_clip, tmp_path):
        """Half the frame rate, so it advances one frame for every two of the
        base's -- mapped by time, not by counting one off against the other."""
        base, quads, _ = base_clip
        slow = creative_clip(tmp_path / "slow.mp4", count=FRAMES, fps=12.5)
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(slow, full_history(quads)), 1.0)
        status, frames = render(base, snapshot, tmp_path / "slow_out.mp4")

        assert status.startswith("Completed"), status
        seen = [landed(frame, quads[i]) for i, frame in enumerate(frames)]
        assert seen[0] == 0
        assert seen == sorted(seen), f"the creative ran backwards: {seen}"
        assert max(b - a for a, b in zip(seen, seen[1:])) <= 1, seen
        assert abs(seen[-1] - (FRAMES - 1) / 2) <= 1, seen


class TestACreativeThatRunsOut:
    def test_its_last_frame_is_held_rather_than_the_insert_dropped(self, qapp,
                                                                   base_clip,
                                                                   tmp_path):
        base, quads, plain = base_clip
        short = creative_clip(tmp_path / "short.mp4", count=4)
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(short, full_history(quads)), 1.0)
        status, frames = render(base, snapshot, tmp_path / "short_out.mp4")

        assert status.startswith("Completed"), status
        seen = [landed(frame, quads[i]) for i, frame in enumerate(frames)]
        assert seen == [0, 1, 2, 3] + [3] * (FRAMES - 4)
        assert not any(is_untouched(frame, plain[i], quads[i])
                       for i, frame in enumerate(frames))

    def test_it_is_reported_rather_than_left_to_be_noticed(self, qapp,
                                                           base_clip, tmp_path):
        base, quads, _ = base_clip
        short = creative_clip(tmp_path / "brief.mp4", count=4)
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(short, full_history(quads), name="Concourse"), 1.0)
        status, _frames = render(base, snapshot, tmp_path / "brief_out.mp4")

        assert "Concourse" in status and "shorter" in status, status

    def test_one_that_yields_no_frames_at_all_says_so(self, qapp, base_clip,
                                                      creative_video, tmp_path):
        """A creative that opens but decodes nothing left the render reporting
        an unqualified success over a file with no insert in it."""
        base, quads, _ = base_clip
        snapshot = galileo_app.PlacementSnapshot(
            _Placement(creative_video, full_history(quads), name="Board"), 1.0)

        class Dead:
            def isOpened(self):
                return True

            def get(self, prop):
                return 25.0

            def set(self, prop, value):
                return False

            def read(self):
                return False, None

            def release(self):
                pass

        worker_settings = galileo_app.RenderSettings(
            base, str(tmp_path / "dead.mp4"), 0, 5, 1.0, 25.0, snapshot.history,
            np.zeros((4, 2, 2), np.float32), False,
            placements=[snapshot], include_audio=False)
        worker = galileo_app.RenderWorker(worker_settings)
        prepare = worker._prepare_placements

        def with_a_dead_capture(settings, caveats):
            ready = prepare(settings, caveats)
            for placement in ready:
                placement.capture.release()
                placement.capture = Dead()
            return ready

        worker._prepare_placements = with_a_dead_capture
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()

        assert statuses and statuses[0].startswith("Completed"), statuses
        assert "Board" in statuses[0] and "not inserted" in statuses[0], statuses[0]


class TestThePreviewAgreesWithTheRender:
    """The preview is meant to be a true preview, not a lookalike."""

    @pytest.fixture
    def app(self, qapp, base_clip, creative_video):
        base, quads, plain = base_clip
        window = galileo_app.MainWindow()
        window.central_panel.load_video(base)
        yield window, window.central_panel, base, quads, plain
        window.dirty = False
        window.close()

    def test_the_frame_previewed_is_the_frame_written(self, app,
                                                      creative_video, tmp_path):
        window, panel, base, quads, _ = app
        overlay = panel.tracking_overlay
        placement = overlay.active
        placement.tracking_history = full_history(quads)
        placement.points = list(placement.tracking_history[0])
        placement.blend = blend.BlendSettings.off()
        # Inserted from the end of the shot, as it is after a tracking pass.
        panel.jump_to_frame(FRAMES - 1)
        overlay.insert_video_overlay(FRAMES - 1, creative_video)

        previewed = []
        for index in (0, 4, 9, FRAMES - 1):
            panel.jump_to_frame(index)
            previewed.append(landed(panel.display_frame, quads[index]))

        snapshot = galileo_app.PlacementSnapshot(placement, 1.0)
        _status, frames = render(base, snapshot, tmp_path / "parity.mp4")
        written = [landed(frames[i], quads[i]) for i in (0, 4, 9, FRAMES - 1)]

        assert previewed == written == [0, 4, 9, FRAMES - 1]

    def test_a_short_creative_holds_its_last_frame_on_screen_too(self, app,
                                                                  tmp_path):
        """Whatever the render writes, the preview shows -- including the held
        frame of a creative that ends before the shot does."""
        window, panel, base, quads, _ = app
        overlay = panel.tracking_overlay
        placement = overlay.active
        placement.tracking_history = full_history(quads)
        placement.points = list(placement.tracking_history[0])
        placement.blend = blend.BlendSettings.off()
        # Parked away from the frame the sweep starts on, so every jump below
        # genuinely decodes rather than being served the frame already shown.
        panel.jump_to_frame(FRAMES - 1)
        overlay.insert_video_overlay(0, creative_clip(tmp_path / "brief.mp4",
                                                      count=3))

        seen = []
        for index in range(FRAMES):
            panel.jump_to_frame(index)
            seen.append(landed(panel.display_frame, quads[index]))
        assert seen == [0, 1, 2] + [2] * (FRAMES - 3)

    def test_a_held_frame_is_not_decoded_again_every_frame(self, app, tmp_path):
        """Re-seeking a long-GOP creative per frame is the cost the sequential
        cursor exists to avoid; holding a frame must not bring it back."""
        window, panel, base, quads, _ = app
        overlay = panel.tracking_overlay
        placement = overlay.active
        placement.tracking_history = full_history(quads)
        placement.points = list(placement.tracking_history[0])
        overlay.insert_video_overlay(0, creative_clip(tmp_path / "three.mp4",
                                                      count=3))

        class Counting:
            def __init__(self, capture):
                self.capture = capture
                self.reads = 0

            def read(self):
                self.reads += 1
                return self.capture.read()

            def __getattr__(self, name):
                return getattr(self.capture, name)

        counter = Counting(placement.overlay_video_cap)
        placement.overlay_video_cap = counter
        for index in range(FRAMES):
            panel.jump_to_frame(index)
        assert counter.reads <= 4, (
            f"the creative was decoded {counter.reads} times over "
            f"{FRAMES} frames of a three-frame creative")

    def test_inserting_one_puts_it_on_screen_there_and_then(self, app,
                                                             tmp_path):
        """A still appeared the moment it was inserted; a video did not show
        up until the playhead happened to move."""
        window, panel, base, quads, _ = app
        overlay = panel.tracking_overlay
        overlay.active.tracking_history = full_history(quads)
        overlay.points = list(overlay.active.tracking_history[0])
        panel.jump_to_frame(4)
        before = panel.display_frame.copy()

        window.add_overlay(creative_clip(tmp_path / "now.mp4"))
        card = next(iter(window.library_cards()))
        card.findChild(QPushButton, "selectBtn").click()

        assert not is_untouched(panel.display_frame, before, quads[4]), (
            "inserting the creative changed nothing on screen")

    def test_a_reopened_project_previews_its_creative(self, app, tmp_path,
                                                      monkeypatch):
        """The load recorded the path and stopped there, leaving no capture to
        decode from: the creative rendered but previewed as nothing at all."""
        window, panel, base, quads, _ = app
        overlay = panel.tracking_overlay
        placement = overlay.active
        placement.tracking_history = full_history(quads)
        placement.points = list(placement.tracking_history[0])
        placement.blend = blend.BlendSettings.off()
        overlay.insert_video_overlay(0, creative_clip(tmp_path / "saved.mp4"))

        project = str(tmp_path / "project.json")
        monkeypatch.setattr(galileo_app.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        monkeypatch.setattr(galileo_app.QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (project, "")))
        for name in ("information", "warning"):
            monkeypatch.setattr(galileo_app.QMessageBox, name,
                                staticmethod(lambda *a, **k: None))
        window.save_project()
        panel.jump_to_frame(6)
        window.load_project()

        restored = window.central_panel.tracking_overlay.active
        assert restored.inserted_overlay_is_video
        assert restored.overlay_video_cap is not None, "no capture was opened"
        assert restored.overlay_bgra is not None, "no frame was decoded"
        assert landed(window.central_panel.display_frame,
                      quads[panel.get_current_frame_index()]) is not None

    def test_every_placement_advances_not_only_the_active_one(self, app,
                                                              tmp_path):
        """Each placement renders its own creative, so each has to preview
        its own too."""
        window, panel, base, quads, _ = app
        overlay = panel.tracking_overlay
        first = overlay.active
        first.name = "Left"
        second = overlay.add_placement("Right")
        for placement, name in ((first, "left.mp4"), (second, "right.mp4")):
            placement.tracking_history = full_history(quads)
            placement.points = list(placement.tracking_history[0])
            placement.blend = blend.BlendSettings.off()
            overlay.set_active(overlay.placements.index(placement))
            overlay.insert_video_overlay(0, creative_clip(tmp_path / name))
        overlay.set_active(0)

        panel.jump_to_frame(7)
        assert first.overlay_video_shown == 7
        assert second.overlay_video_shown == 7, (
            "the placement not being edited froze on its old frame")

    def test_the_whole_way_through_from_the_library_card(self, app, tmp_path):
        """The user's route: load the creative, track, insert, render."""
        window, panel, base, quads, plain = app
        overlay = panel.tracking_overlay
        window.add_overlay(creative_clip(tmp_path / "card.mp4"))

        overlay.points = [tuple(map(float, c)) for c in quads[0]]
        panel.tracking_mode = True
        for _ in range(FRAMES - 1):
            panel.read_frame()
        panel.tracking_mode = False
        overlay.active.blend = blend.BlendSettings.off()

        card = next(iter(window.library_cards()))
        card.findChild(QPushButton, "selectBtn").click()
        assert overlay.inserted_overlay_is_video

        out = str(tmp_path / "card_out.mp4")
        settings = window.build_render_settings(0, FRAMES - 1, 1.0, out)
        worker = galileo_app.RenderWorker(settings)
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses and statuses[0].startswith("Completed"), statuses

        tracked = sorted(overlay.active.tracking_history)
        capture = cv2.VideoCapture(out)
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()

        drawn = [i for i in tracked if i < len(frames)]
        assert drawn, "nothing was tracked to render"
        assert not any(is_untouched(frames[i], plain[i],
                                    overlay.active.tracking_history[i])
                       for i in drawn), "the creative never reached the file"
