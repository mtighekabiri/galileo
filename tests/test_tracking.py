"""Tracking accuracy against footage with a known ground-truth path."""

import cv2
import numpy as np
import pytest

import galileo_core as core
from conftest import bend_edge, make_texture, quad_path, render_clip


def corner_error(a, b):
    """Mean distance between corresponding corners, in pixels."""
    return float(np.linalg.norm(np.float32(a) - np.float32(b), axis=1).mean())


class TestPlanarTracker:
    def test_follows_a_known_path(self, clip):
        frames, truth = clip
        tracker = core.PlanarTracker(frames[0], truth[0])

        errors = []
        for frame, expected in zip(frames[1:], truth[1:]):
            result = tracker.track(frame)
            assert result.ok, result.reason
            errors.append(corner_error(result.quad, expected))

        # Sub-pixel through the middle, and no runaway by the end.
        assert np.mean(errors) < 2.0, f"mean corner error {np.mean(errors):.2f}px"
        assert errors[-1] < 4.0, f"final corner error {errors[-1]:.2f}px"

    def test_does_not_drift_away_over_the_clip(self, clip):
        """Error must stay bounded rather than growing every frame."""
        frames, truth = clip
        tracker = core.PlanarTracker(frames[0], truth[0])
        errors = []
        for frame, expected in zip(frames[1:], truth[1:]):
            result = tracker.track(frame)
            errors.append(corner_error(result.quad, expected))

        first_half = np.mean(errors[:len(errors) // 2])
        second_half = np.mean(errors[len(errors) // 2:])
        assert second_half < first_half + 2.5

    def test_beats_tracking_the_bare_corners(self, clip):
        """The old approach, reproduced, should be measurably worse.

        This is the whole reason for the rewrite: four hand-placed corners are
        poor features, so following them directly accumulates error that a
        many-point RANSAC fit does not.
        """
        frames, truth = clip

        # Old behaviour: LK straight on the four corner points.
        points = np.float32(truth[0]).reshape(-1, 1, 2)
        prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        naive_errors = []
        for frame, expected in zip(frames[1:], truth[1:]):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev, gray, points, None, **core.LK_PARAMS)
            if points is None or status.sum() < 4:
                naive_errors.append(float("inf"))
                break
            naive_errors.append(corner_error(points.reshape(-1, 2), expected))
            prev = gray

        tracker = core.PlanarTracker(frames[0], truth[0])
        new_errors = [corner_error(tracker.track(f).quad, e)
                      for f, e in zip(frames[1:], truth[1:])]

        assert np.mean(new_errors) < np.mean(naive_errors)

    def test_reports_failure_on_an_untrackable_region(self):
        """A flat, textureless patch has nothing to track; say so, don't guess."""
        blank = np.full((240, 320, 3), 128, np.uint8)
        quad = np.float32([[60, 60], [200, 60], [200, 180], [60, 180]])
        tracker = core.PlanarTracker(blank, quad)
        result = tracker.track(np.full((240, 320, 3), 128, np.uint8))
        assert not result.ok
        assert "features" in result.reason

    def test_state_survives_a_failed_step(self, clip):
        """A failure must not corrupt the tracker; it should recover."""
        frames, truth = clip
        tracker = core.PlanarTracker(frames[0], truth[0])
        tracker.track(frames[1])
        before = tracker.quad.copy()

        garbage = np.random.default_rng(3).integers(
            0, 255, frames[0].shape, dtype=np.uint8)
        result = tracker.track(garbage)
        if not result.ok:
            assert np.allclose(tracker.quad, before)

        recovered = tracker.track(frames[2])
        assert recovered.ok
        assert corner_error(recovered.quad, truth[2]) < 8.0

    def test_rejects_an_implausible_jump(self, clip):
        frames, truth = clip
        tracker = core.PlanarTracker(frames[0], truth[0])
        # A completely unrelated frame should not be accepted as one step.
        far = render_clip([truth[0] + np.float32([300, 200])],
                          texture=make_texture(seed=9))[0]
        result = tracker.track(far)
        assert not result.ok

    def test_reset_reanchors_to_a_new_quad(self, clip):
        frames, truth = clip
        tracker = core.PlanarTracker(frames[0], truth[0])
        tracker.track(frames[1])

        nudged = np.float32(truth[2]) + np.float32([3, -2])
        tracker.reset(frames[2], nudged)
        assert np.allclose(tracker.quad, nudged)
        assert np.allclose(tracker.cumulative_h, np.eye(3), atol=1e-9)

        result = tracker.track(frames[3])
        assert result.ok

    def test_map_from_anchor_carries_extra_points(self, clip):
        """Curved-edge control points must ride along with the surface."""
        frames, truth = clip
        tracker = core.PlanarTracker(frames[0], truth[0])

        centre = np.float32(truth[0]).mean(axis=0, keepdims=True)
        for frame in frames[1:6]:
            tracker.track(frame)

        moved = tracker.map_from_anchor(centre)
        expected = np.float32(tracker.quad).mean(axis=0)
        assert np.linalg.norm(moved[0] - expected) < 3.0

    def test_region_curvature_rides_the_tracked_surface(self, clip):
        """A bend must rotate and scale with the surface it belongs to.

        Held as screen-space offsets, a curve would stay stubbornly axis-aligned
        while the shot rotated and slide off the thing it was shaped around.
        Held in the edge's own frame it turns with the plane, which is what
        these assertions pin down.
        """
        frames, truth = clip
        region = core.Region(truth[0], curved=True)
        bend_edge(region, 0, [0, -30])
        original_offset = region.control_offsets()[0, 0].copy()

        tracker = core.PlanarTracker(frames[0], truth[0])
        for frame in frames[1:8]:
            tracker.track(frame)

        moved = region.with_corners(tracker.quad)
        assert not moved.is_straight()

        # The coefficients are unchanged: the bend is the same bend.
        assert np.allclose(moved.curvature, region.curvature, atol=1e-6)

        # Its pixel offset has followed the edge's own rotation and scale.
        def edge_vector(quad):
            return np.float32(quad)[1] - np.float32(quad)[0]

        before, after = edge_vector(truth[0]), edge_vector(tracker.quad)
        scale = np.linalg.norm(after) / np.linalg.norm(before)
        angle = (np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0]))

        new_offset = moved.control_offsets()[0, 0]
        assert np.linalg.norm(new_offset) == pytest.approx(
            np.linalg.norm(original_offset) * scale, rel=0.02)

        rotation = np.float32([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        assert np.allclose(new_offset, rotation @ original_offset * scale, atol=0.2)


class TestKalman:
    def test_converges_onto_a_steady_measurement(self):
        f = core.SimpleKalmanFilter(0.0, 0.0)
        for _ in range(40):
            f.predict()
            out = f.update([100.0, 50.0])
        assert out[0] == pytest.approx(100.0, abs=1.0)
        assert out[1] == pytest.approx(50.0, abs=1.0)

    def test_coasts_along_when_a_measurement_is_missing(self):
        filters = core.make_filters([[0, 0], [10, 0], [10, 10], [0, 10]])
        for i in range(1, 15):
            core.smooth_quad(filters, np.float32(
                [[i, 0], [10 + i, 0], [10 + i, 10], [i, 10]]))
        coasted = core.smooth_quad(filters, None)
        # Velocity was +1px/frame in x, so prediction should keep moving.
        assert coasted[0][0] > 13.0

    def test_tracks_a_measurement_without_much_lag(self):
        """Over-damping shows up as the insert trailing the camera."""
        filters = core.make_filters([[0, 0], [10, 0], [10, 10], [0, 10]])
        for i in range(1, 30):
            quad = np.float32([[i * 2, 0], [10 + i * 2, 0],
                               [10 + i * 2, 10], [i * 2, 10]])
            out = core.smooth_quad(filters, quad)
        assert abs(out[0][0] - 58.0) < 2.0
