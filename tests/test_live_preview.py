"""Changes have to show the moment they are made.

Nearly everything the user adjusts changes how the creative *looks* rather
than where it is, and the creative is drawn into the frame by the panel, not
by the overlay widget that carries the outline. Asking only the overlay to
repaint leaves the change invisible until the video happens to decode another
frame -- which reads as the control being broken.
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

from PyQt5.QtWidgets import QApplication                          # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "galileo_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Galileo_Insertion_Tool_1.0.0.py"))
galileo_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(galileo_app)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def creative(colour=(30, 60, 220), width=320, height=200):
    bgr = np.full((height, width, 3), colour, np.uint8)
    cv2.putText(bgr, "AD", (70, 130), cv2.FONT_HERSHEY_DUPLEX, 3.0,
                (255, 255, 255), 8)
    return np.dstack([bgr, np.full((height, width), 255, np.uint8)])


@pytest.fixture
def loaded(qapp, clip_video):
    path, truth = clip_video
    window = galileo_app.MainWindow()
    panel = window.central_panel
    panel.load_video(path)
    overlay = panel.tracking_overlay
    overlay.points = [tuple(map(float, p)) for p in truth[0]]
    panel.refresh_display()
    yield window, panel, overlay
    window.dirty = False
    window.close()


def on_screen(panel):
    """What the video label is actually showing."""
    pixmap = panel.video_label.pixmap()
    assert pixmap is not None, "nothing has been drawn at all"
    image = pixmap.toImage().convertToFormat(4)
    bits = image.bits()
    bits.setsize(image.byteCount())
    array = np.frombuffer(bits, np.uint8).reshape(
        image.height(), image.bytesPerLine() // 4, 4)
    return array[:, :image.width(), :3].copy()


class TestItShowsWithoutSteppingAFrame:
    def test_inserting_a_creative(self, loaded):
        window, panel, overlay = loaded
        before = on_screen(panel)
        overlay.insert_image_overlay_from_array = None      # not a real method
        overlay.overlay_bgra = creative()
        overlay.refresh_preview()
        assert not np.array_equal(before, on_screen(panel))

    def test_taking_it_out_again(self, loaded):
        window, panel, overlay = loaded
        overlay.overlay_bgra = creative()
        overlay.refresh_preview()
        inserted = on_screen(panel)
        overlay.remove_inserted_overlay()
        assert not np.array_equal(inserted, on_screen(panel))

    def test_brightness(self, loaded):
        window, panel, overlay = loaded
        overlay.overlay_bgra = creative()
        overlay.refresh_preview()
        before = on_screen(panel)
        overlay.set_brightness(90)
        assert not np.array_equal(before, on_screen(panel))

    def test_contrast(self, loaded):
        window, panel, overlay = loaded
        overlay.overlay_bgra = creative()
        overlay.refresh_preview()
        before = on_screen(panel)
        overlay.set_contrast(1.8)
        assert not np.array_equal(before, on_screen(panel))

    def test_colourise(self, loaded):
        window, panel, overlay = loaded
        overlay.overlay_bgra = creative()
        overlay.refresh_preview()
        before = on_screen(panel)
        overlay.toggle_colourise(True)
        assert not np.array_equal(before, on_screen(panel))

    def test_the_overlay_is_repainted_as_well(self, loaded):
        """Both halves, not one: the outline lives on the overlay widget and
        the creative in the frame beneath it."""
        window, panel, overlay = loaded
        painted = []
        original = overlay.update
        overlay.update = lambda *a, **k: (painted.append(1), original(*a, **k))[1]
        overlay.set_brightness(20)
        assert painted, "the overlay was never asked to repaint"

    def test_it_survives_having_no_frame_yet(self, qapp):
        """Adjusting before a video is loaded must not raise."""
        window = galileo_app.MainWindow()
        try:
            window.central_panel.tracking_overlay.set_brightness(30)
        finally:
            window.dirty = False
            window.close()


class TestTheShapeEditorPreviewsOnTheFootage:
    """Showing the artwork alone says what the shape is but not whether it
    suits the surface, and the surface is the only thing that can answer."""

    def preview_of(self, panel, overlay):
        def preview():
            frame = panel.prev_frame
            if frame is None or len(overlay.points) != 4:
                return None
            composited = panel.composite_placements(frame)
            height, width = composited.shape[:2]
            bounds = core.quad_bounds(overlay.points, width, height, pad=60)
            if bounds is None:
                return None
            x0, y0, x1, y1 = bounds
            return composited[y0:y1, x0:x1]
        return preview

    def thumbnail(self, dialog):
        pixmap = dialog.thumbnail.pixmap()
        if pixmap is None:
            return None
        image = pixmap.toImage().convertToFormat(4)
        bits = image.bits()
        bits.setsize(image.byteCount())
        return np.frombuffer(bits, np.uint8).reshape(
            image.height(), image.bytesPerLine() // 4, 4)[:, :image.width(), :3].copy()

    def test_the_preview_updates_as_a_slider_moves(self, loaded):
        window, panel, overlay = loaded
        overlay.overlay_bgra = creative()
        panel.refresh_display()
        dialog = galileo_app.MorphDialog(
            overlay.morph, overlay.overlay_bgra, panel.refresh_display,
            parent=window, preview=self.preview_of(panel, overlay))
        before = self.thumbnail(dialog)
        dialog.sliders["yaw"].setValue(30)
        assert before is not None
        assert not np.array_equal(before, self.thumbnail(dialog))

    def test_it_shows_the_footage_and_not_a_checkerboard(self, loaded):
        window, panel, overlay = loaded
        overlay.overlay_bgra = creative()
        panel.refresh_display()
        plain = galileo_app.MorphDialog(
            overlay.morph, overlay.overlay_bgra, lambda: None, parent=window)
        on_shot = galileo_app.MorphDialog(
            overlay.morph, overlay.overlay_bgra, lambda: None, parent=window,
            preview=self.preview_of(panel, overlay))
        assert not np.array_equal(self.thumbnail(plain),
                                  self.thumbnail(on_shot))

    def test_the_video_behind_is_updated_too(self, loaded):
        window, panel, overlay = loaded
        overlay.overlay_bgra = creative()
        panel.refresh_display()
        asked = []
        dialog = galileo_app.MorphDialog(
            overlay.morph, overlay.overlay_bgra, lambda: asked.append(1),
            parent=window, preview=self.preview_of(panel, overlay))
        dialog.sliders["bow_h"].setValue(60)
        assert asked

    def test_without_a_preview_it_still_shows_the_artwork(self, loaded):
        window, panel, overlay = loaded
        dialog = galileo_app.MorphDialog(
            overlay.morph, creative(), lambda: None, parent=window)
        assert self.thumbnail(dialog) is not None

    def test_a_preview_that_fails_does_not_stop_the_editing(self, loaded):
        """A shape is still worth setting even if the shot cannot be drawn."""
        window, panel, overlay = loaded

        def broken():
            raise RuntimeError("no")

        dialog = galileo_app.MorphDialog(
            overlay.morph, creative(), lambda: None, parent=window,
            preview=broken)
        dialog.sliders["yaw"].setValue(20)
        assert overlay.morph.yaw == 20
        assert self.thumbnail(dialog) is not None, "fell back to nothing"

    def test_a_preview_with_nothing_to_show_falls_back(self, loaded):
        window, panel, overlay = loaded
        dialog = galileo_app.MorphDialog(
            overlay.morph, creative(), lambda: None, parent=window,
            preview=lambda: None)
        assert self.thumbnail(dialog) is not None


class TestTheLibraryCardReads:
    def test_the_creative_name_is_not_clipped(self, qapp, tmp_path):
        """A round 18px height cut the descenders, so "ad.png" read "ad.pnq"."""
        from PyQt5.QtWidgets import QLabel
        path = tmp_path / "ad.png"
        cv2.imwrite(str(path), np.full((40, 60, 3), 200, np.uint8))
        window = galileo_app.MainWindow()
        try:
            window.add_overlay(str(path))
            card = window.overlay_layout.itemAt(
                window.overlay_layout.count() - 2).widget()
            labels = [w for w in card.findChildren(QLabel) if w.text()]
            assert labels, "the card shows no name at all"
            for label in labels:
                assert label.height() >= label.fontMetrics().height(), (
                    f"{label.text()!r} is given less height than its font needs")
        finally:
            window.dirty = False
            window.close()

    def test_no_placeholder_word_where_a_logo_would_go(self, qapp):
        """With no logo.png beside it, the word "Logo" beside the title reads
        as something broken rather than as nothing at all."""
        window = galileo_app.MainWindow()
        try:
            label = window.title_bar.logo_label
            assert not (label.isVisible() and label.text())
        finally:
            window.dirty = False
            window.close()


class TestTheWindowWorksAtItsSmallest:
    """1024x640 is the smallest the app allows, so everything has to fit
    there -- not only on the machine it was written on."""

    def test_every_tool_is_reachable(self, qapp, clip_video):
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.show()
            window.setGeometry(0, 0, window.minimumWidth(),
                               window.minimumHeight())
            window.central_panel.load_video(path)
            for _ in range(4):
                QApplication.processEvents()
            hidden = [icon.meaning_label.text() for icon in window.left_col.icons
                      if icon.visibleRegion().boundingRect().height() < icon.height()]
            assert not hidden, f"{hidden} fall off the bottom of the toolbar"
        finally:
            window.dirty = False
            window.close()

    def test_the_last_tool_is_the_one_that_ends_the_job(self, qapp):
        window = galileo_app.MainWindow()
        try:
            order = [icon.meaning_label.text() for icon in window.left_col.icons]
            assert order[-1] == "Render", order
            assert order.index("Draw") < order.index("Curve") < order.index("Render")
        finally:
            window.dirty = False
            window.close()


class TestTheMagnifierKeepsOutOfTheWay:
    def test_it_takes_whichever_corner_the_area_reaches_into_least(self, qapp,
                                                                   clip_video):
        """Sized to hold twelve views it is large enough to cover the very
        area being worked on, which is the one thing it must not hide. A big
        area on a small stage cannot always be missed entirely, so what is
        promised is the least bad corner -- and that is what is checked."""
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.show()
            window.setGeometry(0, 0, 1400, 880)
            panel = window.central_panel
            panel.load_video(path)
            overlay = panel.tracking_overlay
            panel.magnifier.show()

            for marked in ([(60., 50.), (360., 50.), (360., 260.), (60., 260.)],
                           [(600., 50.), (900., 50.), (900., 260.), (600., 260.)],
                           [(60., 300.), (360., 300.), (360., 500.), (60., 500.)]):
                overlay.points = marked
                panel.update_magnifier(force=True)
                panel.layout_video_stage()
                QApplication.processEvents()

                stage = overlay.geometry()
                shown = [overlay.raw_to_display(x, y) for x, y in marked]
                area = (min(p[0] for p in shown) + stage.x(),
                        min(p[1] for p in shown) + stage.y(),
                        max(p[0] for p in shown) + stage.x(),
                        max(p[1] for p in shown) + stage.y())

                box = panel.magnifier
                chosen = panel._overlap(
                    (box.x(), box.y(), box.x() + box.width(),
                     box.y() + box.height()), area)

                margin = 10
                every = []
                for left in (stage.x() + margin,
                             stage.x() + stage.width() - box.width() - margin):
                    for top in (stage.y() + margin,
                                stage.y() + stage.height() - box.height() - margin):
                        every.append(panel._overlap(
                            (left, top, left + box.width(), top + box.height()),
                            area))
                assert chosen <= min(every) + 1, (
                    f"it covers {chosen:.0f}px where {min(every):.0f} was available")
                assert chosen < 0.25 * ((area[2] - area[0]) * (area[3] - area[1]))
        finally:
            window.dirty = False
            window.close()

    def test_it_still_has_somewhere_to_go_with_no_area_marked(self, qapp,
                                                              clip_video):
        path, truth = clip_video
        window = galileo_app.MainWindow()
        try:
            window.central_panel.load_video(path)
            window.central_panel.layout_video_stage()
            assert window.central_panel.magnifier.x() >= 0
        finally:
            window.dirty = False
            window.close()
