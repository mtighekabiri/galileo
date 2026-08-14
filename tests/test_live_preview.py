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
