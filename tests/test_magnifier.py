"""The magnifier: placing handles where single pixels decide the outcome.

Marking the area is the one part of the job where a pixel matters. The tracker
seeds its features from whatever is enclosed, so a corner left a little inside
a screen throws away the bezel detail that tracks best, and one left a little
outside picks up the wall behind. Neither is visible at video scale, which is
what the magnifier is for -- so these tests check that what it shows is true,
not merely that it draws something.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import importlib.util

import cv2
import numpy as np
import pytest

import galileo_core as core

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5.QtCore import QRect                                   # noqa: E402
from PyQt5.QtWidgets import QApplication                         # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)

Point = galileo_app.MagnifierPoint
CORNERS = [(150.0, 110.0), (470.0, 110.0), (470.0, 300.0), (150.0, 300.0)]


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def screen_frame(width=640, height=400):
    """Footage with a hard-edged screen in it, the thing corners must land on."""
    frame = np.full((height, width, 3), 70, np.uint8)
    cv2.rectangle(frame, (150, 110), (470, 300), (28, 28, 32), -1)
    cv2.rectangle(frame, (150, 110), (470, 300), (215, 215, 225), 3)
    return frame


def grab(widget):
    """Render the widget to BGR. Format_RGB32 is already BGR in memory."""
    image = widget.grab().toImage().convertToFormat(4)
    bits = image.bits()
    bits.setsize(image.byteCount())
    array = np.frombuffer(bits, np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4)
    return array[:, :image.width(), :3].copy()


@pytest.fixture
def magnifier(qapp):
    widget = galileo_app.MagnifierWidget()
    widget.resize(320, 320)
    return widget


def curved_region(bend=22.0):
    region = core.Region(np.float32(CORNERS), curved=True)
    for slot in (0, 1):
        default = np.asarray(
            region.default_controls(np.float32(CORNERS))[0, slot], np.float32)
        region.set_control_point(0, slot, default + np.float32([0, -bend]))
    return region


def handle_points(region):
    """The twelve handles, in the order the panel builds them."""
    names = ("Top", "Right", "Bottom", "Left")
    points = []
    for edge in range(4):
        points.append(Point(*CORNERS[edge], str(edge + 1), Point.CORNER,
                            f"Corner {edge + 1}"))
        for slot in range(2):
            cx, cy = region.controls[edge, slot]
            points.append(Point(cx, cy, f"{names[edge][0]}{slot + 1}",
                                Point.HANDLE,
                                f"{names[edge]} edge, bend {slot + 1}"))
    return points


class TestTheViewTellsTheTruth:
    def test_the_crosshair_marks_the_handle_exactly(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS, zoom_factor=8)
        for rect, index in magnifier.tiles():
            point = magnifier.points[index]
            centre_x = rect.x() + rect.width() / 2.0
            centre_y = rect.y() + rect.height() / 2.0
            under = magnifier.source_at(rect, point, centre_x, centre_y)
            assert under == pytest.approx((point.x, point.y), abs=1e-6)

    def test_it_still_tells_the_truth_at_the_edge_of_frame(self, magnifier):
        """The old view slid back inside the frame but drew the crosshair at
        the middle of the tile regardless, so within half a view of any border
        it pointed at the wrong pixel -- exactly where placement is fiddliest."""
        frame = screen_frame()
        edges = [(2.0, 2.0), (637.0, 3.0), (0.0, 399.0), (320.0, 1.0)]
        magnifier.setData(frame, edges, zoom_factor=12)
        for rect, index in magnifier.tiles():
            point = magnifier.points[index]
            under = magnifier.source_at(rect, point,
                                        rect.x() + rect.width() / 2.0,
                                        rect.y() + rect.height() / 2.0)
            assert under == pytest.approx((point.x, point.y), abs=1e-6)

    def test_a_handle_outside_the_frame_is_still_centred(self, magnifier):
        """Tracking carries handles off screen during a pan; the view must not
        lurch back inside and start lying about where they are."""
        magnifier.setData(screen_frame(), [(-40.0, 200.0)], zoom_factor=8)
        rect, index = magnifier.tiles()[0]
        point = magnifier.points[index]
        under = magnifier.source_at(rect, point,
                                    rect.x() + rect.width() / 2.0,
                                    rect.y() + rect.height() / 2.0)
        assert under == pytest.approx((-40.0, 200.0), abs=1e-6)

    def test_sub_pixel_positions_are_not_rounded_away(self, magnifier):
        """At 16x, rounding a handle to the nearest whole pixel would put the
        mark 16 screen pixels from where the handle really is."""
        magnifier.setData(screen_frame(), [(200.4, 150.7)], zoom_factor=16)
        rect, index = magnifier.tiles()[0]
        scale, offset_x, offset_y = magnifier.tile_mapping(
            rect, magnifier.points[index])
        assert 200.4 * scale + offset_x == pytest.approx(
            rect.x() + rect.width() / 2.0, abs=1e-6)
        assert 150.7 * scale + offset_y == pytest.approx(
            rect.y() + rect.height() / 2.0, abs=1e-6)


class TestItShowsRealPixels:
    def test_high_zoom_does_not_blur_the_pixels_together(self, magnifier):
        """The question being asked is which pixel the edge falls on. An
        interpolated enlargement answers it with a gradient, which is no
        answer -- so past a point it stops interpolating."""
        checker = np.zeros((80, 80, 3), np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        magnifier.setData(checker, [(40.0, 40.0)],
                          zoom_factor=magnifier.CRISP_ABOVE + 2)
        magnifier.focus_index = 0
        shown = grab(magnifier)[40:280, 40:280].reshape(-1, 3).mean(axis=1)
        muddled = np.mean((shown > 60) & (shown < 195))
        assert muddled < 0.25, f"{muddled:.0%} of the view is neither pixel"

    def test_the_footage_is_shown_at_its_own_brightness(self, magnifier):
        """Nothing may wash the picture over. The label's translucent backing
        was left as the current brush, so the outline drawn round each tile
        filled it instead, quietly dimming every view to a third -- which
        looks like footage that is simply dark, not like a fault."""
        magnifier.setData(np.full((80, 80, 3), 255, np.uint8), [(40.0, 40.0)],
                          zoom_factor=8)
        magnifier.focus_index = 0
        shown = grab(magnifier)
        assert np.median(shown) == 255, f"median brightness {np.median(shown)}"

    def test_a_dark_shot_is_not_lifted_either(self, magnifier):
        magnifier.setData(np.full((80, 80, 3), 40, np.uint8), [(40.0, 40.0)],
                          zoom_factor=8)
        magnifier.focus_index = 0
        assert np.median(grab(magnifier)) == 40

    def test_low_zoom_still_smooths(self, magnifier):
        """Below that, smoothing is the better picture: nobody is counting
        pixels while looking for a handle."""
        assert magnifier.CRISP_ABOVE > magnifier.MIN_ZOOM

    def test_the_zoom_is_bounded(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS, zoom_factor=500)
        assert magnifier.zoom_factor == magnifier.MAX_ZOOM
        magnifier.setData(screen_frame(), CORNERS, zoom_factor=0.01)
        assert magnifier.zoom_factor == magnifier.MIN_ZOOM


class TestLayout:
    def test_four_corners_keep_the_familiar_two_by_two(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS)
        assert magnifier.grid_shape() == (2, 2)
        assert len(magnifier.tiles()) == 4

    def test_each_corner_appears_where_it_sits_on_the_video(self, magnifier):
        """Corners are clicked round the shape, but a grid reads left to right,
        so without a swap the bottom two would be the wrong way about."""
        magnifier.setData(screen_frame(), CORNERS)
        placed = {index: rect for rect, index in magnifier.tiles()}
        middle_x, middle_y = magnifier.width() / 2, magnifier.height() / 2
        for index, (x, y) in enumerate(CORNERS):
            rect = placed[index]
            on_the_right = rect.center().x() > middle_x
            near_the_bottom = rect.center().y() > middle_y
            assert on_the_right == (x > 300), f"corner {index + 1} horizontally"
            assert near_the_bottom == (y > 200), f"corner {index + 1} vertically"

    def test_curving_on_shows_every_handle(self, magnifier):
        """The eight bend handles are the ones that most needed magnifying and
        were the only ones the magnifier never showed."""
        region = curved_region()
        magnifier.setData(screen_frame(), handle_points(region), region=region)
        assert len(magnifier.tiles()) == 12
        assert magnifier.grid_shape() == (3, 4)

    def test_a_row_is_one_edge_of_the_area(self, magnifier):
        region = curved_region()
        magnifier.setData(screen_frame(), handle_points(region), region=region)
        rows = {}
        for rect, index in magnifier.tiles():
            rows.setdefault(rect.y(), []).append(magnifier.points[index].label)
        assert sorted(rows.values()) == sorted(
            [["1", "T1", "T2"], ["2", "R1", "R2"],
             ["3", "B1", "B2"], ["4", "L1", "L2"]])

    def test_dragging_one_handle_gives_it_the_whole_widget(self, magnifier):
        region = curved_region()
        points = handle_points(region)
        magnifier.setData(screen_frame(), points, region=region, focus_index=4)
        tiles = magnifier.tiles()
        assert len(tiles) == 1
        rect, index = tiles[0]
        assert index == 4
        assert rect.width() > magnifier.width() * 0.9

    def test_a_focus_beyond_the_handles_falls_back_to_all_of_them(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS, focus_index=99)
        assert len(magnifier.tiles()) == 4


class TestWhatItDrawsOverThePicture:
    def test_the_areas_outline_is_drawn_through_each_view(self, magnifier):
        """Without it a bend handle cannot be judged at all: what matters is
        where the curve ends up, and that is often nowhere near the handle."""
        region = core.Region(np.float32(CORNERS))
        magnifier.setData(screen_frame(), CORNERS, region=region, zoom_factor=8)
        with_outline = grab(magnifier)
        magnifier.setData(screen_frame(), CORNERS, region=None, zoom_factor=8)
        without = grab(magnifier)

        def greenness(image):
            return int(((image[:, :, 1].astype(int)
                         - image[:, :, 2].astype(int) > 60)).sum())

        assert greenness(with_outline) > greenness(without) + 200

    def test_a_curved_outline_is_drawn_curved(self, magnifier):
        straight = core.Region(np.float32(CORNERS))
        magnifier.setData(screen_frame(), CORNERS, region=straight, zoom_factor=6)
        flat = grab(magnifier)
        magnifier.setData(screen_frame(), CORNERS, region=curved_region(40),
                          zoom_factor=6)
        bent = grab(magnifier)
        assert not np.array_equal(flat, bent)

    def test_a_half_drawn_shape_does_not_stop_it_drawing(self, magnifier):
        """Corners are placed one at a time, so it has to cope with three."""
        magnifier.setData(screen_frame(), CORNERS[:3], region=None)
        assert grab(magnifier).shape[:2] == (320, 320)

    def test_nothing_to_show_says_so_rather_than_crashing(self, magnifier):
        magnifier.setData(None, [])
        assert grab(magnifier).shape[:2] == (320, 320)
        magnifier.setData(screen_frame(), [])
        assert grab(magnifier).shape[:2] == (320, 320)


class TestOlderCallers:
    def test_bare_coordinate_pairs_are_still_accepted(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS)
        assert [p.label for p in magnifier.points] == ["1", "2", "3", "4"]
        assert all(p.kind == Point.CORNER for p in magnifier.points)

    def test_the_plain_coordinates_are_still_readable(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS)
        assert magnifier.overlay_points == CORNERS

    def test_the_buffer_argument_is_tolerated(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS, buffer=25)
        assert len(magnifier.tiles()) == 4


class TestWiredIntoThePanel:
    @pytest.fixture
    def app(self, qapp, clip_video, logo_bgra):
        path, truth = clip_video
        window = galileo_app.MainWindow()
        window.central_panel.load_video(path)
        overlay = window.central_panel.tracking_overlay
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        yield window, window.central_panel, overlay
        window.dirty = False
        window.close()

    def test_four_corners_when_the_area_is_straight(self, app):
        window, panel, overlay = app
        points, selected, focus = panel.magnifier_points()
        assert len(points) == 4
        assert focus is None

    def test_twelve_handles_once_curving_is_on(self, app):
        window, panel, overlay = app
        overlay.curved_enabled = True
        points, _, _ = panel.magnifier_points()
        assert len(points) == 12
        assert sum(p.kind == Point.HANDLE for p in points) == 8

    def test_the_handles_are_where_the_region_says_they_are(self, app):
        window, panel, overlay = app
        overlay.curved_enabled = True
        overlay.set_control_point(1, 0, (overlay.points[1][0] + 14,
                                         overlay.points[1][1] + 9))
        points, _, _ = panel.magnifier_points()
        expected = overlay.current_region().controls[1, 0]
        shown = points[1 * 3 + 0 + 1]
        assert (shown.x, shown.y) == pytest.approx(tuple(expected), abs=1e-4)

    def test_dragging_a_bend_handle_focuses_that_handle(self, app):
        """Corner i moved to position 3i once the bend handles were
        interleaved, so every index has to be carried across with it."""
        window, panel, overlay = app
        overlay.curved_enabled = True
        overlay.drag_control = (2, 1)
        points, selected, focus = panel.magnifier_points()
        assert focus == selected == 2 * 3 + 1 + 1
        assert points[focus].kind == Point.HANDLE
        assert points[focus].label == "B2"

    def test_dragging_a_corner_focuses_that_corner(self, app):
        window, panel, overlay = app
        overlay.curved_enabled = True
        overlay.drag_index = 3
        points, _, focus = panel.magnifier_points()
        assert points[focus].kind == Point.CORNER
        assert points[focus].label == "4"

    def test_the_selected_corner_survives_the_reordering(self, app):
        window, panel, overlay = app
        overlay.curved_enabled = True
        overlay.selected_point_index = 2
        points, selected, _ = panel.magnifier_points()
        assert points[selected].label == "3"

    def test_the_region_is_handed_over_so_the_outline_can_be_drawn(self, app):
        window, panel, overlay = app
        panel.update_magnifier(force=True)
        assert panel.magnifier.region is not None
        assert len(panel.magnifier.points) == 4
