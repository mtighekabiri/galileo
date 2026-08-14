"""Occlusion: drawing the insert behind whatever is in front of the surface."""

import cv2
import numpy as np
import pytest

import lumen_core as core


QUAD = np.float32([[100, 80], [400, 90], [390, 300], [110, 290]])


@pytest.fixture
def base():
    return np.full((360, 640, 3), 60, np.uint8)


@pytest.fixture
def opaque():
    creative = np.zeros((100, 100, 4), np.uint8)
    creative[:, :, :3] = (0, 0, 255)
    creative[:, :, 3] = 255
    return creative


class TestOcclusionMask:
    """The compositing side, driven by a mask made by hand.

    Kept independent of the model so the behaviour is pinned down exactly and
    the suite still runs without the weights downloaded.
    """

    def test_no_mask_paints_everything(self, base, opaque):
        out = core.composite_overlay(base, opaque, QUAD)
        assert not np.allclose(out[190, 250], 60)

    def test_masked_pixels_are_left_alone(self, base, opaque):
        blocker = np.zeros(base.shape[:2], np.uint8)
        blocker[150:250, 200:300] = 255
        out = core.composite_overlay(base, opaque, QUAD, occlusion=blocker)

        assert np.allclose(out[200, 250], 60, atol=1), "insert painted over the blocker"
        assert not np.allclose(out[110, 150], 60), "insert missing where nothing blocks"

    def test_a_full_mask_suppresses_the_insert_entirely(self, base, opaque):
        blocker = np.full(base.shape[:2], 255, np.uint8)
        out = core.composite_overlay(base, opaque, QUAD, occlusion=blocker)
        assert np.array_equal(out, base)

    def test_partial_mask_blends_proportionally(self, base, opaque):
        blocker = np.full(base.shape[:2], 128, np.uint8)
        out = core.composite_overlay(base, opaque, QUAD, occlusion=blocker)
        # Half the creative's opacity, so halfway between base and creative.
        assert np.all(np.abs(out[190, 250].astype(int)
                             - np.array([30, 30, 157])) < 4)

    def test_occlusion_combines_with_creative_alpha(self, base):
        creative = np.zeros((100, 100, 4), np.uint8)
        creative[:, :, :3] = 255
        creative[:, :, 3] = 128                    # already half transparent
        blocker = np.zeros(base.shape[:2], np.uint8)
        blocker[150:250, 200:300] = 255
        out = core.composite_overlay(base, creative, QUAD, occlusion=blocker)
        assert np.allclose(out[200, 250], 60, atol=1)

    def test_works_on_curved_regions_too(self, base, opaque):
        region = core.Region(QUAD, curved=True)
        region.set_control_point(0, 0, region.controls[0, 0] + np.float32([0, -30]))
        region.set_control_point(0, 1, region.controls[0, 1] + np.float32([0, -30]))

        blocker = np.zeros(base.shape[:2], np.uint8)
        blocker[150:250, 200:300] = 255
        out = core.composite_region(base, opaque, region, occlusion=blocker)
        assert np.allclose(out[200, 250], 60, atol=1)

    def test_mask_of_the_wrong_size_is_resized(self, base, opaque):
        """A mask produced at a different scale must still line up."""
        small = np.zeros((180, 320), np.uint8)
        small[75:125, 100:150] = 255
        out = core.composite_overlay(base, opaque, QUAD, occlusion=small)
        assert out.shape == base.shape


model_available = pytest.mark.skipif(
    not core.PersonSegmenter.is_available(),
    reason="segmentation model not downloaded; run fetch_model.py")


@model_available
class TestPersonSegmenter:
    def test_reports_availability(self):
        assert core.PersonSegmenter.is_available() is True

    def test_empty_scene_yields_an_empty_mask(self):
        """No false positives, or the insert would be eaten away at random."""
        segmenter = core.PersonSegmenter()
        blank = np.full((360, 640, 3), 128, np.uint8)
        assert int((segmenter.mask(blank) > 127).sum()) == 0

    def test_mask_is_frame_sized_and_binary_ish(self):
        segmenter = core.PersonSegmenter()
        frame = np.full((240, 320, 3), 90, np.uint8)
        mask = segmenter.mask(frame)
        assert mask.shape == frame.shape[:2]
        assert mask.dtype == np.uint8

    def test_region_crop_restricts_the_mask(self):
        """Running on a crop keeps distant pedestrians big enough to segment."""
        segmenter = core.PersonSegmenter()
        frame = np.full((360, 640, 3), 128, np.uint8)
        quad = np.float32([[50, 50], [150, 50], [150, 150], [50, 150]])
        mask = segmenter.mask(frame, quad)
        # Nothing outside the padded crop can be marked.
        assert int(mask[300:, 500:].sum()) == 0

class TestModelDiscovery:
    def test_missing_model_raises_a_useful_error(self, monkeypatch):
        """The error has to say how to fix it; users will hit this first."""
        monkeypatch.setattr(core, "find_model", lambda name: None)
        with pytest.raises(FileNotFoundError, match="fetch_model.py"):
            core.PersonSegmenter()

    def test_availability_is_false_without_the_model(self, monkeypatch):
        monkeypatch.setattr(core, "find_model", lambda name: None)
        assert core.PersonSegmenter.is_available() is False

    def test_find_model_returns_none_when_absent(self):
        assert core.find_model("definitely-not-a-real-model.onnx") is None

    def test_registered_directory_is_searched(self, tmp_path):
        marker = tmp_path / "fake_model.onnx"
        marker.write_bytes(b"x")
        core.register_model_dir(str(tmp_path))
        assert core.find_model("fake_model.onnx") == str(marker)

    def test_loading_a_bogus_model_raises(self, tmp_path):
        bogus = tmp_path / "bogus.onnx"
        bogus.write_bytes(b"not an onnx file")
        with pytest.raises(IOError):
            core.PersonSegmenter(model_path=str(bogus))
