"""Steadying a tracked path, and keeping the corrections made to it.

Two complaints sit behind this file, and they are the same complaint at two
ends of the job. Tracking a difficult surface leaves the shape twitching frame
to frame, so the corners get corrected by hand, one frame at a time. But a hand
is not steady to the pixel either, so the insert still fidgets in the finished
video -- and the corrections themselves used to be overwritten by the next
tracking pass, which is what made it hours of work rather than minutes.

Accuracy is measured the way the rest of the suite measures it: mean corner
distance from a known path, in pixels. Steadiness needs its own measure, so
these use the mean size of the second difference of the corners --
``q[t+1] - 2q[t] + q[t-1]`` -- which is what the shape's own acceleration
across the frame comes to. A camera move has almost none of it; a wobble is
made of it.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import galileo_core as core
from conftest import quad_path


def fidget(path: dict) -> float:
    """How much the shape accelerates frame to frame, in pixels."""
    keys = sorted(path)
    quads = np.array([np.asarray(path[k], np.float64).reshape(4, 2) for k in keys])
    if len(quads) < 3:
        return 0.0
    return float(np.linalg.norm(quads[2:] - 2 * quads[1:-1] + quads[:-2],
                                axis=2).mean())


def corner_error(path: dict, truth) -> float:
    return float(np.mean([
        np.linalg.norm(np.asarray(path[k], np.float64).reshape(4, 2) - truth[k],
                       axis=1).mean()
        for k in sorted(path)]))


@pytest.fixture(scope="module")
def truth():
    """A smooth camera move: the path a tracker is trying to recover."""
    return quad_path(120)


@pytest.fixture
def corrected(truth):
    """That path as a person would have left it, correcting frame by frame.

    Right on average and unsteady in between, which is exactly what clicking a
    corner into place on every frame produces.
    """
    rng = np.random.default_rng(0)
    return {i: truth[i] + rng.normal(0, 1.5, (4, 2)) for i in range(len(truth))}


class TestWhatIsBeingFixed:
    def test_a_camera_move_is_smooth_and_a_corrected_path_is_not(self, truth,
                                                                 corrected):
        """The size of the problem, stated once so the rest has a scale.

        The movement being followed barely accelerates at all. The recording of
        it, once corrected by hand, does so hundreds of times more -- and that
        difference is the fidget seen in the finished video.
        """
        smooth = fidget({i: truth[i] for i in range(len(truth))})
        assert smooth < 0.05, f"the path itself should be smooth, got {smooth:.3f}"
        assert fidget(corrected) > 3.0, "the corrected path should be visibly unsteady"


class TestSteadying:
    def test_it_takes_the_wobble_out(self, corrected):
        before = fidget(corrected)
        after = fidget(core.smooth_tracking(corrected))
        assert after < before / 5, f"{before:.2f}px only came down to {after:.2f}px"

    def test_it_also_lands_closer_to_the_truth(self, truth, corrected):
        """Not a trade of accuracy for steadiness. Most of what is removed was
        never movement, so taking it out leaves the shape nearer where it
        belongs as well as calmer."""
        before = corner_error(corrected, truth)
        after = corner_error(core.smooth_tracking(corrected), truth)
        assert after < before, f"accuracy went from {before:.2f}px to {after:.2f}px"

    def test_real_movement_passes_straight_through(self, truth):
        """A pan, a zoom and a roll all at once, with nothing wrong with it.
        The fit has to leave that alone or the insert slides off the surface."""
        path = {i: truth[i] for i in range(len(truth))}
        assert corner_error(core.smooth_tracking(path), truth) < 0.01

    def test_a_deliberate_move_is_not_smeared(self, truth):
        """Somewhere in the clip the shape is picked up and put somewhere else
        -- a cut, or a correction that big. Fitting across it would drag the
        frames either side towards a position neither of them was ever in."""
        moved = {i: truth[i].copy() for i in range(len(truth))}
        for i in range(60, len(truth)):
            moved[i] = moved[i] + np.float32([40, 25])

        steadied = core.smooth_tracking(moved)
        for frame in (58, 59, 60, 61):
            shifted = np.linalg.norm(
                np.asarray(steadied[frame]).reshape(4, 2) - moved[frame],
                axis=1).mean()
            assert shifted < 0.5, f"frame {frame} was pulled {shifted:.1f}px"

    def test_gaps_are_not_filled_in(self, truth):
        """Untracked stretches stay untracked. Interpolation across them is a
        separate decision, made later by interpolate_tracking."""
        sparse = {i: truth[i] for i in list(range(0, 30)) + list(range(70, 100))}
        assert sorted(core.smooth_tracking(sparse)) == sorted(sparse)

    def test_strength_scales_it(self, corrected):
        full = fidget(core.smooth_tracking(corrected, strength=1.0))
        part = fidget(core.smooth_tracking(corrected, strength=0.5))
        none = fidget(core.smooth_tracking(corrected, strength=0.0))
        assert full < part < none

    def test_too_few_frames_are_left_alone(self, truth):
        few = {0: truth[0], 1: truth[1]}
        steadied = core.smooth_tracking(few)
        assert np.allclose(steadied[0], truth[0])
        assert np.allclose(steadied[1], truth[1])

    def test_it_is_deterministic(self, corrected):
        first = core.smooth_tracking(corrected)
        for _ in range(3):
            again = core.smooth_tracking(corrected)
            assert all(np.array_equal(first[k], again[k]) for k in first)


class TestOneFrameAtATime:
    """The preview asks for the frame on screen; the render fits the whole
    clip. They have to agree to the last decimal or the file is not what was
    approved -- which is the same reason preview and render composite through
    one function."""

    def test_asking_for_one_frame_matches_asking_for_all(self, corrected):
        whole = core.smooth_tracking(corrected)
        for frame in sorted(corrected):
            one = core.smooth_tracking(corrected, only=frame)
            assert np.array_equal(one[frame], whole[frame]), f"frame {frame}"

    def test_they_agree_across_gaps_and_jumps_too(self, truth):
        rng = np.random.default_rng(3)
        mixed = {}
        for i in range(0, 40):
            mixed[i] = truth[i] + rng.normal(0, 1.2, (4, 2))
        for i in range(40, 80):                      # a jump partway through
            mixed[i] = truth[i] + np.float32([60, 40]) + rng.normal(0, 1.2, (4, 2))
        for i in range(95, 120):                     # and a gap before the end
            mixed[i] = truth[i] + rng.normal(0, 1.2, (4, 2))

        whole = core.smooth_tracking(mixed)
        for frame in sorted(mixed):
            one = core.smooth_tracking(mixed, only=frame)
            assert np.array_equal(one[frame], whole[frame]), f"frame {frame}"

    def test_a_frame_that_was_never_recorded_gives_nothing(self, corrected):
        assert core.smooth_tracking(corrected, only=10_000) == {}
