"""Alpha handling, masking and the curved warp."""

import cv2
import numpy as np
import pytest

import lumen_core as core
from conftest import bend_edge


QUAD = np.float32([[100, 80], [400, 90], [390, 300], [110, 290]])


@pytest.fixture
def base():
    return np.full((360, 640, 3), 60, np.uint8)


class TestAlpha:
    def test_transparent_pixels_leave_the_base_alone(self, base):
        """The old render masked by quad and added, painting alpha=0 black."""
        overlay = np.zeros((100, 100, 4), np.uint8)
        overlay[:, :, :3] = 255          # white, but fully transparent
        out = core.composite_overlay(base, overlay, QUAD)
        assert np.array_equal(out, base)

    def test_opaque_pixels_replace_the_base(self, base):
        overlay = np.zeros((100, 100, 4), np.uint8)
        overlay[:, :, :3] = (10, 20, 200)
        overlay[:, :, 3] = 255
        out = core.composite_overlay(base, overlay, QUAD)
        centre = out[190, 250]
        assert np.allclose(centre, (10, 20, 200), atol=2)

    def test_partial_alpha_blends(self, base):
        overlay = np.zeros((100, 100, 4), np.uint8)
        overlay[:, :, :3] = 255
        overlay[:, :, 3] = 128
        out = core.composite_overlay(base, overlay, QUAD)
        centre = out[190, 250].astype(int)
        # Halfway between the base's 60 and the overlay's 255.
        assert np.all(np.abs(centre - 157) < 4)

    def test_a_transparent_hole_shows_the_base_through(self, base):
        overlay = np.zeros((100, 100, 4), np.uint8)
        overlay[:, :, :3] = (0, 0, 255)
        overlay[:, :, 3] = 255
        overlay[35:65, 35:65, 3] = 0          # punch a hole in the middle
        out = core.composite_overlay(base, overlay, QUAD)
        assert np.allclose(out[190, 250], 60, atol=2)      # hole -> base shows
        assert np.allclose(out[110, 250], (0, 0, 255), atol=6)  # rim -> overlay

    def test_bgr_overlay_without_alpha_is_treated_as_opaque(self, base):
        overlay = np.full((100, 100, 3), (0, 255, 0), np.uint8)
        out = core.composite_overlay(base, overlay, QUAD)
        assert np.allclose(out[190, 250], (0, 255, 0), atol=2)

    def test_opacity_scales_the_blend(self, base):
        overlay = np.zeros((100, 100, 4), np.uint8)
        overlay[:, :, :3] = 255
        overlay[:, :, 3] = 255
        out = core.composite_overlay(base, overlay, QUAD, opacity=0.5)
        assert np.all(np.abs(out[190, 250].astype(int) - 157) < 4)


class TestMasking:
    def test_nothing_is_drawn_outside_the_quad(self, base):
        overlay = np.full((100, 100, 4), 255, np.uint8)
        out = core.composite_overlay(base, overlay, QUAD)
        assert np.array_equal(out[10, 10], base[10, 10])
        assert np.array_equal(out[350, 630], base[350, 630])

    def test_base_is_not_mutated_by_default(self, base):
        before = base.copy()
        overlay = np.full((100, 100, 4), 255, np.uint8)
        core.composite_overlay(base, overlay, QUAD)
        assert np.array_equal(base, before)

    def test_in_place_writes_through(self, base):
        overlay = np.full((100, 100, 4), 255, np.uint8)
        out = core.composite_overlay(base, overlay, QUAD, in_place=True)
        assert out is base
        assert not np.allclose(base[190, 250], 60)

    def test_degenerate_quad_is_a_no_op(self, base):
        overlay = np.full((100, 100, 4), 255, np.uint8)
        bowtie = np.float32([[100, 80], [400, 90], [110, 290], [390, 300]])
        assert np.array_equal(core.composite_overlay(base, overlay, bowtie), base)

    def test_offscreen_quad_is_a_no_op(self, base):
        overlay = np.full((100, 100, 4), 255, np.uint8)
        away = np.float32([[-400, -400], [-300, -400], [-300, -300], [-400, -300]])
        assert np.array_equal(core.composite_overlay(base, overlay, away), base)

    def test_partially_offscreen_quad_still_draws(self, base):
        overlay = np.full((100, 100, 4), 255, np.uint8)
        edge = np.float32([[-50, 100], [150, 100], [150, 250], [-50, 250]])
        out = core.composite_overlay(base, overlay, edge)
        assert not np.allclose(out[175, 60], 60)


class TestCurved:
    def test_default_controls_match_the_straight_path(self, base):
        """Toggling curvature on must not move a region that isn't bent."""
        overlay = np.full((120, 120, 4), 200, np.uint8)
        overlay[:, :, 3] = 255

        straight = core.composite_overlay(base, overlay, QUAD)
        region = core.Region(QUAD, curved=True)
        curved = core.composite_region(base, overlay, region)
        assert np.array_equal(straight, curved)

    def test_bending_moves_pixels_outside_the_straight_quad(self, base):
        overlay = np.full((120, 120, 4), 255, np.uint8)
        region = core.Region(QUAD, curved=True)
        bend_edge(region, 0, [0, -45])        # bow the top edge upwards

        out = core.composite_region(base, overlay, region)
        straight_mask = core.quad_to_mask(QUAD, 640, 360)
        painted = np.any(out != base, axis=2)
        # Something got painted above the straight top edge.
        assert np.any(painted & (straight_mask == 0))

    def test_curved_warp_preserves_perspective(self):
        """A bent region must still foreshorten, not fall back to bilinear.

        The quad is a strong trapezoid, so a checkerboard mapped into it should
        show visibly narrower squares along the short edge. A bilinear patch
        would space them evenly and lose that.
        """
        board = np.zeros((240, 240, 4), np.uint8)
        board[:, :, 3] = 255
        for i in range(12):
            for j in range(12):
                if (i + j) % 2 == 0:
                    board[i * 20:(i + 1) * 20, j * 20:(j + 1) * 20, :3] = 255

        trapezoid = np.float32([[80, 60], [560, 60], [420, 300], [220, 300]])
        base = np.zeros((360, 640, 3), np.uint8)

        region = core.Region(trapezoid, curved=True)
        bend_edge(region, 1, [8, 0])          # a slight bend on one side
        out = core.composite_region(base, overlay=board, region=region)

        # Count edges along a scanline near the wide top vs the narrow bottom.
        def transitions(row):
            line = out[row, :, 0] > 128
            inside = core.quad_to_mask(trapezoid, 640, 360)[row] > 0
            line = line[inside]
            return int(np.count_nonzero(np.diff(line.astype(np.int8))))

        assert transitions(80) >= transitions(280)

    def test_inverse_iteration_is_stable_for_strong_curves(self, base):
        overlay = np.full((120, 120, 4), 255, np.uint8)
        region = core.Region(QUAD, curved=True)
        bend_edge(region, 0, [0, -70])
        bend_edge(region, 2, [0, 70])
        out = core.composite_region(base, overlay, region)
        assert np.all(np.isfinite(out))
        assert out.dtype == np.uint8


class TestColour:
    def test_brightness_leaves_alpha_untouched(self):
        img = np.zeros((10, 10, 4), np.uint8)
        img[:, :, :3] = 100
        img[:, :, 3] = 77
        out = core.apply_brightness_contrast(img, brightness=50)
        assert np.all(out[:, :, 3] == 77)
        assert np.all(out[:, :, 0] == 150)

    def test_identity_adjustment_returns_the_input(self):
        img = np.zeros((4, 4, 3), np.uint8)
        assert core.apply_brightness_contrast(img, 0, 1.0) is img

    def test_colour_transfer_moves_towards_the_target(self):
        source = np.full((50, 50, 3), (200, 200, 200), np.uint8)
        target = np.full((50, 50, 3), (30, 60, 90), np.uint8)
        out = core.color_transfer(source, target)
        assert abs(int(out.mean()) - int(target.mean())) < 30

    def test_colourise_keeps_the_alpha_channel(self, base):
        overlay = np.zeros((60, 60, 4), np.uint8)
        overlay[:, :, :3] = 220
        overlay[:, :, 3] = 128
        out = core.apply_colourise(overlay, base, QUAD)
        assert out.shape[2] == 4
        assert np.all(out[:, :, 3] == 128)
