"""Locating a target in footage from an uploaded reference image."""

import cv2
import numpy as np
import pytest

import lumen_core as core
from conftest import make_texture, quad_path, render_clip, write_video


def corner_error(a, b):
    return float(np.linalg.norm(np.float32(a) - np.float32(b), axis=1).mean())


class TestReferenceMatcher:
    def test_finds_the_target_it_was_given(self, clip, texture):
        """The reference is the poster itself, planted in the frame."""
        frames, truth = clip
        matcher = core.ReferenceMatcher(texture)
        detection = matcher.locate(frames[0])

        assert detection is not None
        assert corner_error(detection.quad, truth[0]) < 6.0
        assert detection.n_inliers >= 12

    def test_corner_order_matches_the_reference(self, clip, texture):
        """The creative's top-left must land on the target's top-left."""
        frames, truth = clip
        detection = core.ReferenceMatcher(texture).locate(frames[0])
        # Corner-for-corner, not merely the same set of points.
        for found, expected in zip(detection.quad, truth[0]):
            assert np.linalg.norm(found - expected) < 10.0

    def test_finds_it_again_after_it_has_moved(self, clip, texture):
        frames, truth = clip
        matcher = core.ReferenceMatcher(texture)
        detection = matcher.locate(frames[-1])
        assert detection is not None
        assert corner_error(detection.quad, truth[-1]) < 8.0

    def test_absent_target_is_reported_as_missing(self, texture):
        """No false positive on footage that does not contain the target."""
        rng = np.random.default_rng(11)
        noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        assert core.ReferenceMatcher(texture).locate(noise) is None

    def test_a_different_poster_is_not_matched(self, clip):
        frames, _ = clip
        other = make_texture(seed=99)
        assert core.ReferenceMatcher(other).locate(frames[0]) is None

    def test_featureless_reference_is_rejected_up_front(self):
        blank = np.full((100, 100, 3), 200, np.uint8)
        with pytest.raises(ValueError, match="too little detail"):
            core.ReferenceMatcher(blank)

    def test_detection_converts_to_a_region(self, clip, texture):
        frames, _ = clip
        detection = core.ReferenceMatcher(texture).locate(frames[0])
        region = detection.to_region(curved=True)
        assert isinstance(region, core.Region)
        assert region.curved is True
        assert region.is_straight()  # curved on, but not yet bent

    def test_orb_backend_also_works(self, clip, texture):
        frames, truth = clip
        matcher = core.ReferenceMatcher(texture, detector="orb")
        assert matcher.detector_name == "orb"
        detection = matcher.locate(frames[0])
        assert detection is not None
        assert corner_error(detection.quad, truth[0]) < 12.0


class TestScanVideo:
    def test_scans_a_clip_and_finds_the_target(self, clip_video, texture):
        path, truth = clip_video
        detection = core.ReferenceMatcher(texture).scan_video(path, step=3)
        assert detection is not None
        assert detection.frame_index is not None
        assert corner_error(detection.quad, truth[detection.frame_index]) < 10.0

    def test_finds_a_target_that_only_appears_later(self, tmp_path, texture):
        """The target is absent at the start, so frame zero is not enough."""
        rng = np.random.default_rng(5)
        empty = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
                 for _ in range(6)]
        quads = quad_path(8)
        frames = empty + render_clip(quads, texture=texture)

        path = write_video(frames, tmp_path / "late.mp4")
        detection = core.ReferenceMatcher(texture).scan_video(path, step=2)

        assert detection is not None
        assert detection.frame_index >= 6

    def test_returns_none_when_the_target_is_never_present(self, tmp_path):
        rng = np.random.default_rng(7)
        frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
                  for _ in range(6)]
        path = write_video(frames, tmp_path / "absent.mp4")
        assert core.ReferenceMatcher(make_texture(seed=42)).scan_video(
            path, step=2) is None

    def test_missing_video_raises(self, texture):
        with pytest.raises(IOError):
            core.ReferenceMatcher(texture).scan_video("/nonexistent/clip.mp4")


class TestDetectionFeedsTracking:
    def test_detect_then_track_end_to_end(self, clip, texture):
        """The intended flow: auto-locate the target, then track it."""
        frames, truth = clip
        detection = core.ReferenceMatcher(texture).locate(frames[0])
        assert detection is not None

        tracker = core.PlanarTracker(frames[0], detection.quad)
        for frame in frames[1:]:
            result = tracker.track(frame)
            assert result.ok, result.reason

        assert corner_error(tracker.quad, truth[-1]) < 8.0
