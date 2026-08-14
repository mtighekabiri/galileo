"""Drift: the shape sliding off its surface during a long camera move.

Composing a homography per frame carries every step's small error forward for
the rest of the clip, so the insert creeps off the panel as the camera pans --
tracking the whole time, just steadily offset. These tests measure that against
footage whose motion is known exactly.
"""

import cv2
import numpy as np
import pytest

import lumen_core as core
from conftest import make_background, make_texture, quad_path


def render_on(quads, texture, size, background):
    """Warp a poster onto each given quad."""
    width, height = size
    th, tw = texture.shape[:2]
    source = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    frames = []
    for quad in quads:
        frame = background.copy()
        matrix = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
        warped = cv2.warpPerspective(texture, matrix, (width, height))
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        frame[mask > 0] = warped[mask > 0]
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def pan():
    """A long, steady pan: the case where drift shows up plainly."""
    width, height, count = 1200, 540, 120
    quads = [np.float32([[40 + 7.0 * i, 150], [340 + 7.0 * i, 140],
                         [348 + 7.0 * i, 360], [108 + 7.0 * i, 370]])
             for i in range(count)]
    frames = render_on(quads, make_texture(280, 180, seed=5), (width, height),
                       make_background(width, height, seed=21))
    return frames, quads


@pytest.fixture(scope="module")
def swing():
    """A 62-degree orbit: the anchor's own appearance changes a great deal."""
    width, height, count = 800, 600, 40
    focal = 700.0
    camera = np.float32([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    plane = np.float32([[-1.6, -1.0, 0], [1.6, -1.0, 0], [1.6, 1.0, 0], [-1.6, 1.0, 0]])

    def project(degrees):
        angle = np.deg2rad(degrees)
        rotation = np.float32([[np.cos(angle), 0, np.sin(angle)],
                               [0, 1, 0],
                               [-np.sin(angle), 0, np.cos(angle)]])
        points = plane @ rotation.T + np.float32([0, 0, 6.0])
        projected = (camera @ points.T).T
        return (projected[:, :2] / projected[:, 2:3]).astype(np.float32)

    quads = [project(a) for a in np.linspace(0, 62, count)]
    frames = render_on(quads, make_texture(400, 250, seed=21), (width, height),
                       make_background(width, height, seed=8))
    return frames, quads


def track_errors(frames, quads, **kwargs):
    tracker = core.PlanarTracker(frames[0], quads[0], **kwargs)
    errors = []
    for frame, expected in zip(frames[1:], quads[1:]):
        tracker.track(frame)
        errors.append(float(np.linalg.norm(tracker.quad - expected, axis=1).mean()))
    return np.array(errors), tracker


class TestDriftDuringAPan:
    def test_incremental_tracking_alone_drifts(self, pan):
        """Documents the fault: error grows steadily with distance travelled."""
        frames, quads = pan
        errors, _ = track_errors(frames, quads, anchor_interval=0)
        rate = np.polyfit(np.arange(len(errors)), errors, 1)[0]
        assert rate > 0.005, f"expected accumulating drift, got {rate:+.4f} px/frame"
        assert errors[-1] > 1.5

    def test_anchor_correction_removes_it(self, pan):
        frames, quads = pan
        errors, tracker = track_errors(frames, quads)
        rate = np.polyfit(np.arange(len(errors)), errors, 1)[0]

        assert tracker.anchor_corrections > 5, "corrections never fired"
        assert abs(rate) < 0.003, f"still drifting at {rate:+.4f} px/frame"
        assert errors[-1] < 0.5, f"final error {errors[-1]:.2f}px"

    def test_it_is_much_better_than_incremental_alone(self, pan):
        frames, quads = pan
        without, _ = track_errors(frames, quads, anchor_interval=0)
        with_anchor, _ = track_errors(frames, quads)
        assert with_anchor[-1] * 5 < without[-1], (
            f"anchored {with_anchor[-1]:.2f}px vs incremental {without[-1]:.2f}px")

    def test_the_error_does_not_grow_with_clip_length(self, pan):
        """The point of anchoring: accuracy stops depending on elapsed time."""
        frames, quads = pan
        errors, _ = track_errors(frames, quads)
        early = errors[10:30].mean()
        late = errors[-20:].mean()
        assert late < early + 0.3, f"early {early:.2f}px, late {late:.2f}px"


class TestCorrectionIsGated:
    def test_a_diverging_viewpoint_does_not_make_things_worse(self, swing):
        """Under a big orbit the anchor stops matching; corrections must not
        drag the shape off a chain that is doing better without them."""
        frames, quads = swing
        without, _ = track_errors(frames, quads, anchor_interval=0)
        with_anchor, _ = track_errors(frames, quads)
        assert with_anchor.mean() <= without.mean() + 0.05, (
            f"anchored {with_anchor.mean():.2f}px vs incremental {without.mean():.2f}px")

    def test_a_poorly_matched_anchor_is_refused(self, swing):
        """Demanding a near-perfect match should block nearly everything here."""
        frames, quads = swing
        _, strict = track_errors(frames, quads, anchor_min_ratio=0.99)
        _, lenient = track_errors(frames, quads, anchor_min_ratio=0.0)
        assert strict.anchor_corrections < lenient.anchor_corrections

    def test_the_ratio_is_recorded_for_inspection(self, pan):
        frames, quads = pan
        _, tracker = track_errors(frames, quads)
        assert 0.0 < tracker.anchor_ratio <= 1.0

    def test_a_huge_correction_is_refused(self, pan):
        """A drift fix moves the shape a little; a big jump means the anchor
        has stopped matching and must not be followed."""
        frames, quads = pan
        _, tracker = track_errors(frames, quads, anchor_max_correction=0.0)
        assert tracker.anchor_corrections == 0

    def test_disabling_it_falls_back_cleanly(self, pan):
        frames, quads = pan
        errors, tracker = track_errors(frames, quads, anchor_interval=0)
        assert tracker.anchor_corrections == 0
        assert np.all(np.isfinite(errors))


class TestBackwardFlowNeedsAGuess:
    def test_corrections_keep_firing_as_the_target_travels(self, pan):
        """The consistency check runs anchor -> now -> anchor, and the return
        trip has to be given a starting point. Without one it must find points
        most of a frame away, fails for every point, and no correction is ever
        accepted after the first few frames."""
        frames, quads = pan
        _, tracker = track_errors(frames, quads)
        expected = (len(frames) - 1) // tracker.anchor_interval
        assert tracker.anchor_corrections >= expected - 1, (
            f"only {tracker.anchor_corrections} of about {expected} fired")
