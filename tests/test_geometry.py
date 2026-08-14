"""Quad validation and curved-region geometry."""

import numpy as np
import pytest

import lumen_core as core
from conftest import bend_edge


SQUARE = [[10, 10], [110, 10], [110, 110], [10, 110]]


class TestQuadValidation:
    def test_area_of_a_known_square(self):
        assert core.quad_area(SQUARE) == pytest.approx(10000.0)

    def test_area_ignores_winding_direction(self):
        assert core.quad_area(SQUARE[::-1]) == pytest.approx(10000.0)

    def test_a_plain_square_is_simple(self):
        assert core.is_simple_quad(SQUARE)
        assert core.is_valid_quad(SQUARE)

    def test_bowtie_is_rejected(self):
        # Swapping two corners crosses the edges over each other.
        bowtie = [[10, 10], [110, 10], [10, 110], [110, 110]]
        assert not core.is_simple_quad(bowtie)
        assert not core.is_valid_quad(bowtie)

    def test_collapsed_quad_is_rejected(self):
        assert not core.is_valid_quad([[5, 5], [5, 5], [5, 5], [5, 5]])

    def test_collinear_quad_is_rejected(self):
        assert not core.is_valid_quad([[0, 0], [10, 0], [20, 0], [30, 0]])

    def test_non_finite_is_rejected(self):
        assert not core.is_valid_quad([[0, 0], [np.nan, 0], [10, 10], [0, 10]])

    def test_concave_quad_is_still_valid(self):
        # An arrowhead is concave but has a perfectly good interior; only
        # self-intersection is disqualifying.
        arrow = [[0, 0], [100, 0], [50, 40], [0, 100]]
        assert core.is_valid_quad(arrow)

    def test_bounds_are_clipped_to_the_frame(self):
        box = core.quad_bounds([[-50, -50], [50, -50], [50, 50], [-50, 50]], 640, 480)
        assert box[0] == 0 and box[1] == 0
        assert box[2] <= 640 and box[3] <= 480

    def test_bounds_are_none_when_fully_offscreen(self):
        assert core.quad_bounds([[-90, -90], [-50, -90], [-50, -50], [-90, -50]],
                                640, 480) is None


class TestMask:
    def test_concave_quad_fills_correctly(self):
        """fillConvexPoly would wrongly fill the notch of a concave shape."""
        arrow = np.float32([[10, 10], [110, 10], [60, 50], [10, 110]])
        mask = core.quad_to_mask(arrow, 128, 128)
        # A point in the notch, outside the true shape but inside its hull.
        assert mask[30, 100] == 0
        # A point solidly inside.
        assert mask[20, 30] == 255


class TestRegion:
    def test_defaults_are_straight(self):
        region = core.Region(SQUARE, curved=True)
        assert region.is_straight()
        assert np.allclose(region.control_offsets(), 0, atol=1e-4)

    def test_curved_flag_off_forces_straight(self):
        region = core.Region(SQUARE, curved=False)
        bend_edge(region, 0, [0, -40], slots=(0,))
        assert region.is_straight()

    def test_bending_an_edge_makes_it_curved(self):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -40], slots=(0,))
        assert not region.is_straight()

    def test_boundary_of_a_straight_region_is_its_corners(self):
        region = core.Region(SQUARE, curved=True)
        assert np.allclose(core.region_boundary(region), np.float32(SQUARE))

    def test_forward_map_hits_the_corners(self):
        region = core.Region(SQUARE, curved=True)
        for (u, v), expected in zip([(0, 0), (1, 0), (1, 1), (0, 1)], SQUARE):
            got = core.region_forward(region, np.float64(u), np.float64(v))
            assert np.allclose(got, expected, atol=1e-4)

    def test_curving_leaves_the_corners_pinned(self):
        """Bending an edge must not drag the corners with it."""
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -50])
        for (u, v), expected in zip([(0, 0), (1, 0), (1, 1), (0, 1)], SQUARE):
            got = core.region_forward(region, np.float64(u), np.float64(v))
            assert np.allclose(got, expected, atol=1e-4)

    def test_curved_edge_actually_bulges(self):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -50])  # pull the top edge upwards
        midpoint = core.region_forward(region, np.float64(0.5), np.float64(0.0))
        straight_mid = np.float32([60, 10])
        assert midpoint[1] < straight_mid[1] - 20  # bulged up, well clear

    @pytest.mark.parametrize("bend", [5, 15, 30])
    def test_inverse_recovers_forward(self, bend):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -bend])
        bend_edge(region, 1, [bend * 0.8, 0])

        u = np.array([0.2, 0.5, 0.8, 0.35])
        v = np.array([0.3, 0.5, 0.1, 0.75])
        points = core.region_forward(region, u, v)
        recovered = core._region_inverse(region, points)

        assert np.allclose(recovered[..., 0], u, atol=1e-5)
        assert np.allclose(recovered[..., 1], v, atol=1e-5)

    def test_inverse_is_exact_over_the_whole_patch(self):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -25])
        bend_edge(region, 2, [10, 15])

        t = np.linspace(0.02, 0.98, 20)
        u, v = np.meshgrid(t, t)
        points = core.region_forward(region, u, v)
        recovered = core._region_inverse(region, points)

        assert np.abs(recovered[..., 0] - u).max() < 1e-5
        assert np.abs(recovered[..., 1] - v).max() < 1e-5


class TestFolding:
    """A bend can go far enough that the surface turns inside out."""

    def test_gentle_curves_are_not_folded(self):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -30])
        assert not core.region_is_folded(region)
        assert core.is_valid_region(region)

    def test_extreme_curves_are_detected_as_folded(self):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -80])
        bend_edge(region, 1, [64, 0])
        assert core.region_is_folded(region)
        assert not core.is_valid_region(region)

    def test_a_straight_region_is_never_folded(self):
        assert not core.region_is_folded(core.Region(SQUARE, curved=True))

    def test_a_folded_region_is_not_composited(self):
        base = np.full((200, 200, 3), 50, np.uint8)
        overlay = np.full((60, 60, 4), 255, np.uint8)
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -90])
        bend_edge(region, 1, [72, 0])
        assert np.array_equal(
            core.composite_region(base, overlay, region), base)

    def test_bad_corners_make_the_region_invalid(self):
        bowtie = [[10, 10], [110, 10], [10, 110], [110, 110]]
        assert not core.is_valid_region(core.Region(bowtie))


class TestRegionState:
    """Moving a region and round-tripping it through a project file."""

    def test_with_corners_carries_the_curve(self):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 0, [0, -40])
        offsets_before = region.control_offsets()

        moved = region.with_corners(np.float32(SQUARE) + [200, 50])
        assert np.allclose(moved.control_offsets(), offsets_before, atol=1e-3)
        assert not moved.is_straight()

    def test_round_trips_through_a_dict(self):
        region = core.Region(SQUARE, curved=True)
        bend_edge(region, 2, [12, -7])
        restored = core.Region.from_dict(region.to_dict())

        assert np.allclose(restored.corners, region.corners)
        assert np.allclose(restored.controls, region.controls)
        assert restored.curved is True

    def test_reads_a_legacy_bare_quad(self):
        restored = core.Region.from_dict(SQUARE)
        assert np.allclose(restored.corners, np.float32(SQUARE))
        assert restored.curved is False
