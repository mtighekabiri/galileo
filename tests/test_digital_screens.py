"""Tracking a digital screen, whose picture moves independently of the panel.

This is the case the tool is mostly pointed at: an OOH screen in an airport,
mall or high street, already playing an advert. Features found on the picture
follow that advert rather than the screen showing it, so the insert wanders
off the panel entirely. These tests pin down both the failure and the fix.
"""

import cv2
import numpy as np
import pytest

import galileo_core as core
from conftest import make_background, make_texture, quad_path


WIDTH, HEIGHT = 640, 480


def screen_content(index):
    """An advert playing on the screen: it scrolls under its own steam."""
    img = np.full((240, 320, 3), 15, np.uint8)
    offset = (index * 14) % 320
    for repeat in range(-1, 3):
        x = offset + repeat * 160
        cv2.rectangle(img, (x, 40), (x + 110, 200), (40, 180, 255), -1)
        cv2.putText(img, "SALE", (x + 8, 140), cv2.FONT_HERSHEY_DUPLEX, 1.4,
                    (255, 255, 255), 3)
    return img


def build_clip(n_frames=30, animated=True):
    """A wall-mounted screen surrounded by fixed detail (bezel, wall, posters)."""
    truth = quad_path(n_frames, WIDTH, HEIGHT)
    surround_texture = make_texture(320, 240, seed=11)
    static_content = screen_content(0)

    frames = []
    for index in range(n_frames):
        frame = make_background(WIDTH, HEIGHT, seed=4)
        quad = truth[index].astype(np.float32)
        outer = (quad - quad.mean(axis=0)) * 1.45 + quad.mean(axis=0)

        content = screen_content(index) if animated else static_content
        for texture, destination in ((surround_texture, outer), (content, quad)):
            th, tw = texture.shape[:2]
            matrix = cv2.getPerspectiveTransform(
                np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]),
                destination.astype(np.float32))
            warped = cv2.warpPerspective(texture, matrix, (WIDTH, HEIGHT))
            mask = np.zeros((HEIGHT, WIDTH), np.uint8)
            cv2.fillPoly(mask, [destination.astype(np.int32)], 255)
            frame[mask > 0] = warped[mask > 0]
        frames.append(frame)
    return frames, truth


def run_tracker(frames, truth, **kwargs):
    tracker = core.PlanarTracker(frames[0], truth[0], **kwargs)
    errors = []
    for frame, expected in zip(frames[1:], truth[1:]):
        tracker.track(frame)
        errors.append(float(np.linalg.norm(
            tracker.quad - np.float32(expected), axis=1).mean()))
    return errors


@pytest.fixture(scope="module")
def animated_clip():
    return build_clip(animated=True)


@pytest.fixture(scope="module")
def static_clip():
    return build_clip(animated=False)


class TestFeatureSource:
    def test_interior_features_lose_an_animated_screen(self, animated_clip):
        """Documents the failure: tracking the picture follows the picture."""
        frames, truth = animated_clip
        errors = run_tracker(frames, truth,
                             feature_source=core.PlanarTracker.INTERIOR)
        assert np.mean(errors) > 20, (
            "expected interior tracking to drift badly on an animated screen; "
            f"got {np.mean(errors):.1f}px")

    def test_surround_features_hold_an_animated_screen(self, animated_clip):
        frames, truth = animated_clip
        errors = run_tracker(frames, truth,
                             feature_source=core.PlanarTracker.SURROUND)
        assert np.mean(errors) < 4.0, f"mean {np.mean(errors):.1f}px"
        assert errors[-1] < 6.0, f"final {errors[-1]:.1f}px"

    def test_surround_is_dramatically_better_here(self, animated_clip):
        frames, truth = animated_clip
        interior = np.mean(run_tracker(
            frames, truth, feature_source=core.PlanarTracker.INTERIOR))
        surround = np.mean(run_tracker(
            frames, truth, feature_source=core.PlanarTracker.SURROUND))
        assert surround * 10 < interior, (
            f"surround {surround:.1f}px vs interior {interior:.1f}px")

    def test_interior_remains_the_right_choice_for_static_artwork(self, static_clip):
        """A printed billboard's own artwork is rigid, so use it."""
        frames, truth = static_clip
        errors = run_tracker(frames, truth,
                             feature_source=core.PlanarTracker.INTERIOR)
        assert np.mean(errors) < 4.0, f"mean {np.mean(errors):.1f}px"

    def test_both_mode_runs(self, static_clip):
        frames, truth = static_clip
        errors = run_tracker(frames, truth,
                            feature_source=core.PlanarTracker.BOTH)
        assert np.mean(errors) < 6.0

    def test_default_is_interior(self, static_clip):
        frames, truth = static_clip
        tracker = core.PlanarTracker(frames[0], truth[0])
        assert tracker.feature_source == core.PlanarTracker.INTERIOR


class TestFeatureMask:
    def test_surround_mask_excludes_the_region_itself(self, static_clip):
        frames, truth = static_clip
        tracker = core.PlanarTracker(frames[0], truth[0],
                                     feature_source=core.PlanarTracker.SURROUND)
        mask = tracker._feature_mask()
        centre = np.int32(np.float32(truth[0]).mean(axis=0))
        assert mask[centre[1], centre[0]] == 0, "surround band covers the screen"
        assert cv2.countNonZero(mask) > 0, "surround band is empty"

    def test_interior_mask_covers_the_region(self, static_clip):
        frames, truth = static_clip
        tracker = core.PlanarTracker(frames[0], truth[0],
                                     feature_source=core.PlanarTracker.INTERIOR)
        mask = tracker._feature_mask()
        centre = np.int32(np.float32(truth[0]).mean(axis=0))
        assert mask[centre[1], centre[0]] == 255

    def test_surround_features_land_outside_the_region(self, static_clip):
        frames, truth = static_clip
        tracker = core.PlanarTracker(frames[0], truth[0],
                                     feature_source=core.PlanarTracker.SURROUND)
        assert len(tracker.features) > 0
        inside = core.quad_to_mask(truth[0], WIDTH, HEIGHT)
        for x, y in tracker.features.reshape(-1, 2):
            assert inside[int(y), int(x)] == 0
