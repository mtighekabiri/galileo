"""Surfaces that leave the shot.

A researcher films a concourse and the camera pans past the screen they are
filling. Following only the screen's own texture stops the moment the screen
does: the quad parks against the edge of frame and the insert bunches up
against it, instead of carrying on out of shot and coming back on the panel
when the camera swings round again.

These clips are built by sliding a window over a wide scene, so the camera's
motion and the surface's position are both known exactly.
"""

import cv2
import numpy as np
import pytest

import galileo_core as core
from conftest import make_background, make_texture


WIDTH, HEIGHT = 800, 450
PANEL = np.float32([[1400, 120], [1660, 115], [1666, 330], [1406, 338]])


def wide_scene(width=3000):
    """A long street with one textured panel standing in it."""
    scene = make_background(width, HEIGHT, seed=11)
    poster = make_texture(300, 200, seed=3)
    source = np.float32([[0, 0], [300, 0], [300, 200], [0, 200]])
    cv2.warpPerspective(poster, cv2.getPerspectiveTransform(source, PANEL),
                        (width, HEIGHT), dst=scene,
                        borderMode=cv2.BORDER_TRANSPARENT)
    return scene


def pan(offsets):
    """Frames and ground-truth quads for a camera panning along the scene."""
    scene = wide_scene()
    frames = [scene[:, int(o):int(o) + WIDTH].copy() for o in offsets]
    quads = [PANEL - np.float32([o, 0]) for o in offsets]
    return frames, quads


def follow(frames, quads, **kwargs):
    """Run the tracker, returning its own error and the smoothed error."""
    tracker = core.PlanarTracker(frames[0], quads[0], **kwargs)
    filters = core.make_filters(quads[0])
    tracked, smoothed = [], []
    for frame, truth in zip(frames[1:], quads[1:]):
        result = tracker.track(frame)
        quad = core.smooth_quad(filters, result.quad if result.ok else None)
        tracked.append(float(np.linalg.norm(tracker.quad - truth, axis=1).mean()))
        smoothed.append(float(np.linalg.norm(quad - truth, axis=1).mean()))
    return np.array(tracked), np.array(smoothed), tracker


@pytest.fixture(scope="module")
def leaves():
    """The panel is panned off the left-hand edge and stays gone."""
    return pan(np.linspace(1200, 2100, 50))


@pytest.fixture(scope="module")
def returns():
    """Panned off, held off for a while, then panned back onto it."""
    return pan(list(np.linspace(1200, 2100, 45)) + list(np.linspace(2100, 1200, 45)))


@pytest.fixture(scope="module")
def drives_away():
    """A static camera, and an advert leaving the shot under its own steam."""
    poster = make_texture(300, 200, seed=3)
    background = make_background(WIDTH, HEIGHT, seed=11)
    source = np.float32([[0, 0], [300, 0], [300, 200], [0, 200]])
    quads = [np.float32([[300 + 12.0 * i, 120], [560 + 12.0 * i, 115],
                         [566 + 12.0 * i, 330], [306 + 12.0 * i, 338]])
             for i in range(60)]
    frames = []
    for quad in quads:
        frame = background.copy()
        warped = cv2.warpPerspective(
            poster, cv2.getPerspectiveTransform(source, quad), (WIDTH, HEIGHT))
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        frame[mask > 0] = warped[mask > 0]
        frames.append(frame)
    return frames, quads


class TestMeasuringHowMuchIsInShot:
    def test_a_region_wholly_in_frame(self):
        frame = make_background(WIDTH, HEIGHT)
        quad = np.float32([[100, 100], [400, 100], [400, 300], [100, 300]])
        assert core.PlanarTracker(frame, quad).visible_fraction() == \
            pytest.approx(1.0, abs=0.01)

    def test_a_region_half_out_of_frame(self):
        frame = make_background(WIDTH, HEIGHT)
        quad = np.float32([[-150, 100], [150, 100], [150, 300], [-150, 300]])
        assert core.PlanarTracker(frame, quad).visible_fraction() == \
            pytest.approx(0.5, abs=0.02)

    def test_a_region_wholly_outside(self):
        frame = make_background(WIDTH, HEIGHT)
        quad = np.float32([[-500, 100], [-200, 100], [-200, 300], [-500, 300]])
        assert core.PlanarTracker(frame, quad).visible_fraction() == 0.0


class TestTheOriginalComplaint:
    def test_without_this_the_quad_parks_at_the_edge_of_frame(self, leaves):
        """Documents the fault: it stops dead where the picture stops."""
        frames, quads = leaves
        _, _, tracker = follow(frames, quads, offscreen_below=0.0)
        travelled = abs(tracker.quad[0][0] - quads[0][0][0])
        expected = abs(quads[-1][0][0] - quads[0][0][0])
        assert travelled < expected * 0.6, (
            f"expected it to stall, but it moved {travelled:.0f} of {expected:.0f}px")

    def test_it_keeps_going_once_the_surface_is_gone(self, leaves):
        frames, quads = leaves
        tracked, _, tracker = follow(frames, quads)
        assert tracker.camera_frames > 10, "never started following the camera"
        assert tracked.max() < 5.0, f"worst error {tracked.max():.1f}px"

    def test_it_is_far_better_than_stalling(self, leaves):
        frames, quads = leaves
        stalled, _, _ = follow(frames, quads, offscreen_below=0.0)
        carried, _, _ = follow(frames, quads)
        assert carried.max() * 20 < stalled.max(), (
            f"carried {carried.max():.1f}px vs stalled {stalled.max():.1f}px")


class TestComingBack:
    def test_the_surface_is_picked_up_again_where_it_belongs(self, returns):
        frames, quads = returns
        tracked, _, tracker = follow(frames, quads)
        assert tracked[-10:].mean() < 1.0, (
            f"off by {tracked[-10:].mean():.1f}px once back in view")

    def test_it_turns_round_when_the_camera_does(self, returns):
        """A prediction that only coasts cannot reverse, and runs away by
        hundreds of pixels while the camera comes back the other way."""
        frames, quads = returns
        tracked, _, _ = follow(frames, quads)
        assert tracked.max() < 5.0, f"worst error {tracked.max():.1f}px"

    def test_the_measured_position_beats_coasting_by_a_wide_margin(self, returns):
        frames, quads = returns
        coasted, _, _ = follow(frames, quads, offscreen_below=0.0)
        carried, _, _ = follow(frames, quads)
        assert carried.max() * 50 < coasted.max(), (
            f"carried {carried.max():.1f}px vs coasted {coasted.max():.1f}px")

    def test_it_goes_back_to_reading_the_surface_itself(self, returns):
        frames, quads = returns
        _, _, tracker = follow(frames, quads)
        assert tracker.following_camera is False
        assert tracker.visible_fraction() > 0.9


class TestSurfacesThatMoveOnTheirOwn:
    def test_an_advert_driving_out_of_shot_keeps_moving(self, drives_away):
        """The camera is still, so following it alone would park the insert
        where the vehicle was last seen while the vehicle carries on."""
        frames, quads = drives_away
        tracked, _, _ = follow(frames, quads)
        assert tracked[-1] < 5.0, f"off by {tracked[-1]:.1f}px at the end"

    def test_it_is_better_than_leaving_it_to_the_smoothing(self, drives_away):
        frames, quads = drives_away
        _, coasted, _ = follow(frames, quads, offscreen_below=0.0)
        _, carried, _ = follow(frames, quads)
        assert carried[-1] < coasted[-1] / 5, (
            f"carried {carried[-1]:.1f}px vs coasted {coasted[-1]:.1f}px")

    def test_a_fixed_billboard_is_not_dragged_about_by_noise(self, returns):
        """What is left over for a fixed surface after the camera is accounted
        for is measurement error. Carried for fifty frames it walks the shape
        off the panel, so below a floor it is taken to be nothing."""
        frames, quads = returns
        tracker = core.PlanarTracker(frames[0], quads[0])
        for frame in frames[1:]:
            tracker.track(frame)
            if tracker.following_camera and tracker._offscreen_drift is not None:
                assert np.all(tracker._offscreen_drift == 0)
                return
        pytest.fail("never went off screen")

    def test_a_surface_of_its_own_gets_a_drift_worth_carrying(self, drives_away):
        frames, quads = drives_away
        tracker = core.PlanarTracker(frames[0], quads[0])
        for frame in frames[1:]:
            tracker.track(frame)
            if tracker.following_camera and tracker._offscreen_drift is not None:
                assert np.linalg.norm(tracker._offscreen_drift) > 8.0
                return
        pytest.fail("never went off screen")


class TestItDoesNotDisturbOrdinaryTracking:
    def test_a_region_in_full_view_reads_its_own_texture(self, returns):
        frames, quads = returns
        tracker = core.PlanarTracker(frames[0], quads[0])
        for frame in frames[1:8]:
            tracker.track(frame)
        assert tracker.following_camera is False
        assert tracker.camera_frames == 0

    def test_a_region_still_mostly_in_shot_is_left_alone(self, returns):
        """Switching over early would throw away good measurements."""
        frames, quads = returns
        tracker = core.PlanarTracker(frames[0], quads[0])
        for frame, truth in zip(frames[1:], quads[1:]):
            visible = tracker.visible_fraction()
            tracker.track(frame)
            if visible > 0.5:
                assert not tracker.following_camera, f"switched at {visible:.2f}"

    def test_accuracy_in_view_is_unchanged(self, returns):
        frames, quads = returns
        with_it, _, _ = follow(frames[:18], quads[:18])
        without, _, _ = follow(frames[:18], quads[:18], offscreen_below=0.0)
        assert with_it.mean() == pytest.approx(without.mean(), abs=0.05)


class TestWhatTheCallerIsTold:
    def test_the_result_says_when_it_is_following_the_camera(self, leaves):
        frames, quads = leaves
        tracker = core.PlanarTracker(frames[0], quads[0])
        states = [tracker.track(frame).following_camera for frame in frames[1:]]
        assert states[0] is False
        assert states[-1] is True

    def test_these_steps_count_as_tracked_not_failed(self, leaves):
        """They are measurements, not guesses, so the caller should take them
        rather than fall back on prediction."""
        frames, quads = leaves
        tracker = core.PlanarTracker(frames[0], quads[0])
        carried = [tracker.track(frame) for frame in frames[1:]]
        following = [r for r in carried if r.following_camera]
        assert len(following) > 10
        assert all(r.ok for r in following)
        assert all(r.quad is not None for r in following)

    def test_how_long_it_spent_off_screen_is_recorded(self, leaves):
        frames, quads = leaves
        _, _, tracker = follow(frames, quads)
        assert 0 < tracker.camera_frames < len(frames)
