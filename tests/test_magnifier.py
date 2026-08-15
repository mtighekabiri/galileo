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

from PyQt5.QtCore import (QEvent, QPoint, QPointF, QRect,        # noqa: E402
                          Qt)
from PyQt5.QtGui import QMouseEvent, QWheelEvent                # noqa: E402
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


class TestHowMuchItMagnifies:
    """A fixed magnification means wildly different things in a tile 50 pixels
    across and one 500 across. At 8x a default-sized tile held nine pixels of
    footage: nothing to align against, and blocks so large it read as mush."""

    def span(self, magnifier):
        """How much footage the first tile shows across its shorter side."""
        rect, _ = magnifier.tiles()[0]
        return min(rect.width(), rect.height()) / magnifier.zoom_for(rect)

    def test_a_small_tile_still_shows_useful_context(self, magnifier):
        magnifier.resize(150, 150)
        magnifier.setData(screen_frame(), CORNERS)
        assert self.span(magnifier) > 20

    def test_the_same_footage_is_shown_whatever_the_size(self, magnifier):
        spans = []
        for size in ((150, 150), (320, 320), (700, 500)):
            magnifier.resize(*size)
            magnifier.setData(screen_frame(), CORNERS)
            spans.append(self.span(magnifier))
        assert max(spans) - min(spans) < 3, spans

    def test_a_bigger_tile_earns_more_magnification(self, magnifier):
        magnifier.resize(150, 150)
        magnifier.setData(screen_frame(), CORNERS)
        small = magnifier.zoom_for(magnifier.tiles()[0][0])
        magnifier.resize(760, 560)
        big = magnifier.zoom_for(magnifier.tiles()[0][0])
        assert big > small * 3

    def test_a_stage_sized_view_is_crisp_enough_to_count_pixels(self, magnifier):
        magnifier.resize(940, 530)
        magnifier.setData(screen_frame(), handle_points(curved_region()),
                          region=curved_region())
        assert magnifier.zoom_for(magnifier.tiles()[0][0]) >= magnifier.CRISP_ABOVE

    def test_scrolling_takes_the_choice_over(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS)
        assert magnifier.auto_zoom is True
        magnifier.wheelEvent(QWheelEvent(
            QPointF(60, 60), QPointF(60, 60), QPoint(0, 120), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        assert magnifier.auto_zoom is False
        assert magnifier.zoom_factor > magnifier.zoom_for(magnifier.tiles()[0][0]) - 1

    def test_an_explicit_zoom_is_kept(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS, zoom_factor=16)
        assert magnifier.auto_zoom is False
        assert magnifier.zoom_for(magnifier.tiles()[0][0]) == 16

    def test_clicking_the_badge_hands_the_choice_back(self, magnifier):
        """Otherwise a zoom scrolled somewhere unhelpful can only be undone by
        guessing at the number it started from."""
        magnifier.setData(screen_frame(), CORNERS, zoom_factor=30)
        badge = magnifier.zoom_badge_rect().center()
        magnifier.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, badge, magnifier.mapToGlobal(badge),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        assert magnifier.auto_zoom is True


class TestFillingTheStage:
    """Twelve handles in a floating box a couple of hundred pixels wide leaves
    each one smaller than the thumbnail it replaced."""

    def test_it_starts_floating(self, magnifier):
        assert magnifier.expanded is False

    def test_the_button_toggles_it(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS)
        centre = magnifier.expand_button_rect().center()
        for expected in (True, False):
            magnifier.mousePressEvent(QMouseEvent(
                QEvent.MouseButtonPress, centre, magnifier.mapToGlobal(centre),
                Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
            assert magnifier.expanded is expected

    def test_double_clicking_toggles_it_too(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS)
        middle = QPoint(magnifier.width() // 2, magnifier.height() // 2)
        magnifier.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, middle, magnifier.mapToGlobal(middle),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        assert magnifier.expanded is True

    def test_a_double_click_on_the_grip_resizes_instead(self, magnifier):
        """The grip is a drag target; toggling on it would fight the drag."""
        magnifier.setData(screen_frame(), CORNERS)
        corner = QPoint(magnifier.width() - 5, magnifier.height() - 5)
        magnifier.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, corner, magnifier.mapToGlobal(corner),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        assert magnifier.expanded is False

    def test_it_says_when_it_changes(self, magnifier):
        seen = []
        magnifier.expandToggled.connect(seen.append)
        magnifier.set_expanded(True)
        magnifier.set_expanded(True)      # no change, no repeat
        magnifier.set_expanded(False)
        assert seen == [True, False]

    def test_the_button_is_offered_even_with_nothing_to_show(self, magnifier):
        """Expanding and then clearing the shape would otherwise leave the
        stage covered with no way back."""
        magnifier.set_expanded(True)
        magnifier.setData(None, [])
        box = magnifier.expand_button_rect()
        drawn = grab(magnifier)[box.y():box.bottom(), box.x():box.right()]
        assert drawn.max() > 120

    def test_the_panel_fills_the_stage_and_puts_it_back(self, qapp, clip_video):
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.resize(1280, 800)
            panel = window.central_panel
            panel.load_video(path)
            panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
            panel.layout_video_stage()
            floating = panel.magnifier.geometry()

            panel.magnifier.set_expanded(True)
            assert panel.magnifier.geometry() == panel.tracking_overlay.geometry()

            panel.magnifier.set_expanded(False)
            # Back to floating in the same corner. Not necessarily the same
            # size: it re-applies whatever the current handle count needs,
            # which is the auto-sizing doing its job.
            assert panel.magnifier.pos() == floating.topLeft()
            assert panel.magnifier.geometry() != panel.tracking_overlay.geometry()
        finally:
            window.dirty = False
            window.close()

    def test_a_relayout_keeps_it_filling_the_stage(self, qapp, clip_video):
        """Anything that re-lays the video out must not drop it back into the
        corner while it is meant to be filling the stage."""
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.resize(1280, 800)
            panel = window.central_panel
            panel.load_video(path)
            panel.magnifier.set_expanded(True)
            panel.layout_video_stage()
            assert panel.magnifier.geometry() == panel.tracking_overlay.geometry()
        finally:
            window.dirty = False
            window.close()

    def test_the_panel_does_not_resize_it_while_it_fills_the_stage(self, qapp,
                                                                   clip_video):
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.resize(1280, 800)
            panel = window.central_panel
            panel.load_video(path)
            panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
            panel.magnifier.set_expanded(True)
            filled = panel.magnifier.geometry()
            panel.update_magnifier_dimensions()
            assert panel.magnifier.geometry() == filled
        finally:
            window.dirty = False
            window.close()


class TestDraggingItBigger:
    """The corner has always resized it, but only until the panel sized it
    once -- which happens as soon as two corners exist, so in practice the
    handle was dead for the whole of a real session."""

    def drag_the_corner(self, widget, dx, dy):
        start = QPoint(widget.width() - 5, widget.height() - 5)
        widget.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, start, widget.mapToGlobal(start),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        end = start + QPoint(dx, dy)
        widget.mouseMoveEvent(QMouseEvent(
            QEvent.MouseMove, end, widget.mapToGlobal(end),
            Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
        widget.mouseReleaseEvent(QMouseEvent(
            QEvent.MouseButtonRelease, end, widget.mapToGlobal(end),
            Qt.LeftButton, Qt.NoButton, Qt.NoModifier))
        return widget.width(), widget.height()

    def test_dragging_the_corner_makes_it_bigger(self, magnifier):
        assert self.drag_the_corner(magnifier, 60, 40) == (380, 360)

    def test_it_still_works_after_the_size_has_been_pinned(self, magnifier):
        """setFixedSize leaves the maximum equal to the minimum, and then the
        drag's own resize() does nothing at all -- no error, no movement, just
        a handle that seems dead. Whatever pinned it, the grip must win."""
        magnifier.setFixedSize(200, 150)
        assert self.drag_the_corner(magnifier, 60, 60) == (260, 210)

    def test_it_will_not_shrink_below_its_minimum(self, magnifier):
        assert self.drag_the_corner(magnifier, -900, -900) == (150, 150)

    def test_dragging_it_stops_the_panel_resizing_it_again(self, magnifier):
        assert magnifier.user_resized is False
        self.drag_the_corner(magnifier, 30, 30)
        assert magnifier.user_resized is True

    def test_a_click_away_from_the_corner_is_not_a_resize(self, magnifier):
        press = QPoint(20, 20)
        magnifier.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, press, magnifier.mapToGlobal(press),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        assert magnifier.resizing is False
        assert magnifier.user_resized is False

    def test_the_corner_is_marked_so_it_can_be_found(self, magnifier):
        """It had no mark of any kind, so the only way to find it was to
        already know it was there."""
        magnifier.setData(np.zeros((80, 80, 3), np.uint8), [(40.0, 40.0)])
        corner = grab(magnifier)[-20:, -20:]
        assert corner.max() > 120, "nothing is drawn in the resize corner"

    def test_the_panels_auto_size_leaves_it_resizable(self, qapp, clip_video):
        """The fault came in through the panel, so it is checked there too."""
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.central_panel.load_video(path)
            overlay = window.central_panel.tracking_overlay
            overlay.points = [tuple(map(float, p)) for p in truth[0]]
            window.central_panel.update_magnifier_dimensions()
            magnifier = window.central_panel.magnifier
            assert magnifier.maximumWidth() > magnifier.minimumWidth()
            before = magnifier.width()
            assert self.drag_the_corner(magnifier, 50, 50)[0] == before + 50
        finally:
            window.dirty = False
            window.close()


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


class TestTheBendHandlesAreDrawnOnTheCurve:
    """A bend handle is placed correctly when the curve it produces sits on
    the screen's edge. Showing the curve without what is pulling it leaves
    that unjudgeable."""

    def test_a_neighbouring_handle_appears_in_the_view(self, magnifier):
        region = curved_region(40)
        centre = Point(*region.controls[0, 0], "T1", Point.HANDLE)
        # Wide enough for the next handle along to fall inside the view: they
        # sit a third of an edge apart, which is 213px once magnified.
        magnifier.resize(700, 500)
        magnifier.setData(screen_frame(), [centre], region=region,
                          focus_index=0, zoom_factor=2)
        shown = grab(magnifier)

        rect, _ = magnifier.tiles()[0]
        scale, offset_x, offset_y = magnifier.tile_mapping(rect, centre)
        other = region.controls[0, 1]
        x = int(other[0] * scale + offset_x)
        y = int(other[1] * scale + offset_y)
        assert rect.contains(x, y), "fixture should put the other handle in view"

        patch = shown[y - 8:y + 8, x - 8:x + 8].astype(int)
        bluish = (patch[:, :, 0] > patch[:, :, 2] + 25).sum()
        assert bluish > 12, "no handle drawn where the other bend sits"

    def test_the_corners_are_drawn_too(self, magnifier):
        region = core.Region(np.float32(CORNERS))
        centre = Point(*CORNERS[0], "1", Point.CORNER)
        magnifier.resize(500, 900)          # tall enough to reach corner 4
        magnifier.setData(screen_frame(), [centre], region=region,
                          focus_index=0, zoom_factor=2)
        shown = grab(magnifier)
        rect, _ = magnifier.tiles()[0]
        scale, offset_x, offset_y = magnifier.tile_mapping(rect, centre)
        x = int(CORNERS[3][0] * scale + offset_x)
        y = int(CORNERS[3][1] * scale + offset_y)
        assert rect.contains(x, y), "fixture should put corner 4 in view"
        patch = shown[y - 8:y + 8, x - 8:x + 8].astype(int)
        assert (patch[:, :, 2] > patch[:, :, 0] + 25).sum() > 12

    def test_the_picked_handle_is_marked_out(self, magnifier):
        region = curved_region(40)
        centre = Point(*region.controls[0, 0], "T1", Point.HANDLE)
        magnifier.resize(700, 500)
        magnifier.setData(screen_frame(), [centre], region=region,
                          focus_index=0, zoom_factor=2, selected_control=None)
        plain = grab(magnifier)
        magnifier.setData(screen_frame(), [centre], region=region,
                          focus_index=0, zoom_factor=2, selected_control=(0, 1))
        picked = grab(magnifier)
        assert not np.array_equal(plain, picked)

    def test_a_straight_area_draws_no_bend_handles(self, magnifier):
        """They do not exist until curving is switched on."""
        straight = core.Region(np.float32(CORNERS))
        centre = Point(*CORNERS[0], "1", Point.CORNER)
        magnifier.setData(screen_frame(), [centre], region=straight,
                          focus_index=0, zoom_factor=2)
        shown = grab(magnifier).astype(int)
        bluish = ((shown[:, :, 0] > shown[:, :, 2] + 40)
                  & (shown[:, :, 0] > 150)).sum()
        assert bluish < 200, "something blue is drawn on a straight area"


class TestPickingABendHandleWithTheKeys:
    """Keys 1-4 have always picked the corners. The eight bend handles could
    only ever be dragged, so there was no way to nudge one by a pixel -- which
    matters more there than on a corner, a handle being a lever."""

    @pytest.fixture
    def app(self, qapp, clip_video):
        path, truth = clip_video
        window = galileo_app.MainWindow()
        panel = window.central_panel
        panel.load_video(path)
        overlay = panel.tracking_overlay
        overlay.points = [tuple(map(float, p)) for p in truth[0]]
        overlay.curved_enabled = True
        yield window, panel, overlay
        window.dirty = False
        window.close()

    def press(self, overlay, key, modifier=Qt.NoModifier):
        from PyQt5.QtGui import QKeyEvent
        overlay.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, modifier))

    def test_each_edge_key_picks_that_edges_bend(self, app):
        window, panel, overlay = app
        for key, edge in ((Qt.Key_T, 0), (Qt.Key_R, 1),
                          (Qt.Key_B, 2), (Qt.Key_L, 3)):
            overlay.selected_control = None
            self.press(overlay, key)
            assert overlay.selected_control == (edge, 0)

    def test_pressing_it_again_steps_to_the_other_one(self, app):
        window, panel, overlay = app
        self.press(overlay, Qt.Key_T)
        assert overlay.selected_control == (0, 0)
        self.press(overlay, Qt.Key_T)
        assert overlay.selected_control == (0, 1)
        self.press(overlay, Qt.Key_T)
        assert overlay.selected_control == (0, 0)

    def test_the_arrow_keys_move_it(self, app):
        window, panel, overlay = app
        self.press(overlay, Qt.Key_T)
        before = np.array(overlay.current_region().controls[0, 0])
        self.press(overlay, Qt.Key_Right)
        after = np.array(overlay.current_region().controls[0, 0])
        assert after[0] == pytest.approx(before[0] + 1.0, abs=0.01)
        assert after[1] == pytest.approx(before[1], abs=0.01)

    def test_the_modifiers_work_the_same_as_on_a_corner(self, app):
        window, panel, overlay = app
        self.press(overlay, Qt.Key_T)
        before = np.array(overlay.current_region().controls[0, 0])
        self.press(overlay, Qt.Key_Right, Qt.ShiftModifier)
        assert overlay.current_region().controls[0, 0][0] == \
            pytest.approx(before[0] + 10.0, abs=0.01)
        self.press(overlay, Qt.Key_Left, Qt.ControlModifier)
        assert overlay.current_region().controls[0, 0][0] == \
            pytest.approx(before[0] + 9.75, abs=0.01)

    def test_picking_a_corner_lets_the_bend_go(self, app):
        window, panel, overlay = app
        self.press(overlay, Qt.Key_T)
        self.press(overlay, Qt.Key_2)
        assert overlay.selected_control is None
        assert overlay.selected_point_index == 1

    def test_the_keys_do_nothing_until_curving_is_on(self, app):
        window, panel, overlay = app
        overlay.curved_enabled = False
        self.press(overlay, Qt.Key_T)
        assert overlay.selected_control is None

    def test_the_magnifier_follows_the_pick(self, app):
        window, panel, overlay = app
        self.press(overlay, Qt.Key_B)
        self.press(overlay, Qt.Key_B)
        points, selected, focus = panel.magnifier_points()
        assert points[selected].label == "B2"
        assert focus is None, "picking with a key should not hide the others"

    def test_a_nudge_is_recorded_for_the_frame(self, app):
        """An edit that is drawn but never stored is the fault this whole
        path was written to avoid."""
        window, panel, overlay = app
        self.press(overlay, Qt.Key_T)
        before = np.array(overlay.curvature).copy()
        self.press(overlay, Qt.Key_Up)
        assert not np.array_equal(before, np.array(overlay.curvature))


class TestItIsBigEnoughToBeUseful:
    def test_twelve_handles_get_more_room_than_four(self, qapp, clip_video):
        """Sized for the shape alone, switching curved edges on divided the
        same 150 pixels twelve ways instead of four, leaving each handle a
        50x37 thumbnail -- smaller than the thing it was meant to magnify."""
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            panel = window.central_panel
            panel.load_video(path)
            overlay = panel.tracking_overlay
            overlay.points = [tuple(map(float, p)) for p in truth[0]]
            panel.update_magnifier(force=True)
            flat = panel.magnifier.size()

            overlay.curved_enabled = True
            panel.update_magnifier(force=True)
            curved = panel.magnifier.size()
            assert curved.height() > flat.height()

            rect, _ = panel.magnifier.tiles()[0]
            assert min(rect.width(), rect.height()) >= 80, (
                f"each view is only {rect.width()}x{rect.height()}")
        finally:
            window.dirty = False
            window.close()

    def test_a_hand_resize_is_still_respected(self, qapp, clip_video):
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            panel = window.central_panel
            panel.load_video(path)
            panel.tracking_overlay.points = [tuple(map(float, p)) for p in truth[0]]
            panel.magnifier.user_resized = True
            panel.magnifier.resize(210, 210)
            panel.tracking_overlay.curved_enabled = True
            panel.update_magnifier(force=True)
            assert panel.magnifier.size().width() == 210
        finally:
            window.dirty = False
            window.close()


class TestOneViewOfTheWholeArea:
    """A grid shows each handle closely but never shows the shape they make.
    This shows the shape, which is what tells you whether the outline is
    following the screen."""

    def whole(self, magnifier, region=None):
        region = region if region is not None else curved_region()
        magnifier.setData(screen_frame(), handle_points(region), region=region,
                          selected_index=4, focus_index=None)
        magnifier.set_whole_area(True)
        return region

    def test_it_shows_a_single_view(self, magnifier):
        region = curved_region()
        magnifier.setData(screen_frame(), handle_points(region), region=region,
                          focus_index=None)
        assert len(magnifier.tiles()) == 12
        magnifier.set_whole_area(True)
        assert len(magnifier.tiles()) == 1

    def test_that_view_gets_the_whole_widget(self, magnifier):
        self.whole(magnifier)
        rect, _ = magnifier.tiles()[0]
        assert rect.width() > magnifier.width() * 0.9

    def test_every_handle_is_inside_it(self, magnifier):
        """The point of the mode: all twelve at once, none cut off."""
        region = self.whole(magnifier)
        rect, index = magnifier.tiles()[0]
        point = magnifier.point_at(index)
        scale, offset_x, offset_y = magnifier.tile_mapping(rect, point)
        for handle in magnifier.points:
            x = handle.x * scale + offset_x
            y = handle.y * scale + offset_y
            assert rect.contains(int(x), int(y)), (
                f"{handle.label} falls outside the view")

    def test_it_frames_the_outline_not_just_the_corners(self, magnifier):
        """A bend can swing wide of the corners; cutting it off would hide
        the very thing being judged."""
        magnifier.set_whole_area(True)
        far = curved_region(90)
        magnifier.setData(screen_frame(), handle_points(far), region=far,
                          focus_index=None)
        bounds = magnifier.area_bounds()
        outline = core.region_boundary(far, samples=32)
        assert bounds[1] <= float(np.min(outline[:, 1])) + 0.01

    def test_it_fits_whatever_size_it_is_given(self, magnifier):
        """An area larger than the widget cannot be both shown whole and
        magnified, and showing it whole is the point."""
        magnifier.resize(200, 200)
        self.whole(magnifier)
        small = magnifier.zoom_for(magnifier.tiles()[0][0])
        magnifier.resize(900, 700)
        big = magnifier.zoom_for(magnifier.tiles()[0][0])
        assert big > small * 2, f"{small:.2f} then {big:.2f}"

    def test_enlarging_it_buys_real_magnification(self, magnifier):
        magnifier.resize(1100, 800)
        self.whole(magnifier)
        assert magnifier.zoom_for(magnifier.tiles()[0][0]) > 1.5

    def test_no_crosshair_is_planted_in_the_middle(self, magnifier):
        """Nothing is at the centre of this view, so marking it would point
        at a handle that is not there."""
        region = self.whole(magnifier)
        shown = grab(magnifier)
        middle = shown[magnifier.height() // 2 - 3:magnifier.height() // 2 + 3,
                       magnifier.width() // 2 - 3:magnifier.width() // 2 + 3]
        assert middle.std() < 60, "something is drawn dead centre"

    def test_the_picked_handle_stands_out(self, magnifier):
        region = curved_region()
        magnifier.set_whole_area(True)
        magnifier.setData(screen_frame(), handle_points(region), region=region,
                          selected_index=0, focus_index=None)
        first = grab(magnifier)
        magnifier.setData(screen_frame(), handle_points(region), region=region,
                          selected_index=3, focus_index=None)
        assert not np.array_equal(first, grab(magnifier))

    def test_the_button_toggles_it(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS, focus_index=None)
        centre = magnifier.single_button_rect().center()
        for expected in (True, False):
            magnifier.mousePressEvent(QMouseEvent(
                QEvent.MouseButtonPress, centre, magnifier.mapToGlobal(centre),
                Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
            assert magnifier.whole_area is expected

    def test_it_does_not_ask_for_a_grid_worth_of_room(self, magnifier):
        region = self.whole(magnifier)
        assert magnifier.grid_shape() == (1, 1)

    def test_its_button_does_not_sit_on_the_other_one(self, magnifier):
        assert not magnifier.single_button_rect().intersects(
            magnifier.expand_button_rect())

    def test_double_clicking_a_button_does_not_also_expand(self, magnifier):
        magnifier.setData(screen_frame(), CORNERS, focus_index=None)
        centre = magnifier.single_button_rect().center()
        magnifier.mouseDoubleClickEvent(QMouseEvent(
            QEvent.MouseButtonDblClick, centre, magnifier.mapToGlobal(centre),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        assert magnifier.expanded is False

    def test_dragging_still_overrides_it(self, magnifier):
        """A drag wants the handle under the mouse as close as it can get."""
        region = curved_region()
        magnifier.set_whole_area(True)
        magnifier.setData(screen_frame(), handle_points(region), region=region,
                          selected_index=1, focus_index=5)
        assert magnifier.tiles()[0][1] == 5

    def test_it_copes_with_nothing_marked(self, magnifier):
        magnifier.set_whole_area(True)
        magnifier.setData(screen_frame(), [])
        assert grab(magnifier).shape[:2] == (320, 320)
