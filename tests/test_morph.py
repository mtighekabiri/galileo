"""Shaping the creative: bending and tilting the artwork itself.

The tracked quad says where the surface is and is left exactly as the tracker
measured it. This is the other half -- how the artwork sits on that surface --
so every test here checks the creative's own canvas, never the tracking.
"""

import os
import sys

import cv2
import numpy as np
import pytest

import lumen_core as core
import lumen_morph as morph


def striped(width=400, height=200, spacing=40):
    """Evenly spaced stripes: a ruler for seeing where the sheet was stretched."""
    img = np.zeros((height, width, 4), np.uint8)
    img[:, :, :3] = 30
    img[:, :, 3] = 255
    for x in range(spacing // 2, width, spacing):
        img[:, x - 1:x + 2, :3] = 255
    return img


def stripe_gaps(image):
    """Distance between neighbouring stripes, left to right."""
    column = image[:, :, 0].mean(axis=0)
    lit = np.where(column > 100)[0]
    runs = np.split(lit, np.where(np.diff(lit) > 1)[0] + 1)
    centres = np.array([run.mean() for run in runs if len(run)])
    return np.diff(centres)


def end_compression(image):
    """How much harder the ends are squeezed than the middle.

    Above 1 the ends hold more of the sheet per pixel than the centre does,
    which is what a surface curving away at its edges looks like.
    """
    gaps = stripe_gaps(image)
    return gaps[len(gaps) // 2] / ((gaps[0] + gaps[-1]) / 2)


@pytest.fixture
def poster():
    bgr = np.full((200, 400, 3), (30, 50, 200), np.uint8)
    cv2.putText(bgr, "AD", (120, 140), cv2.FONT_HERSHEY_DUPLEX, 4.0,
                (255, 255, 255), 10)
    return np.dstack([bgr, np.full((200, 400), 255, np.uint8)])


class TestDoingNothing:
    def test_a_fresh_morph_is_identity(self):
        assert morph.Morph().is_identity()

    def test_identity_returns_the_very_same_array(self, poster):
        """Not merely equal: the creative must not be copied for nothing."""
        assert morph.apply_morph(poster, morph.Morph()) is poster

    def test_switching_it_off_ignores_the_settings(self, poster):
        shaped = morph.Morph(yaw=30, bow_h=0.8, enabled=False)
        assert shaped.is_identity()
        assert morph.apply_morph(poster, shaped) is poster

    def test_no_morph_at_all_is_safe(self, poster):
        assert morph.apply_morph(poster, None) is poster
        assert morph.apply_morph(None, morph.Morph(yaw=10)) is None


class TestBow:
    def test_a_convex_bow_squeezes_the_ends(self):
        assert end_compression(morph.apply_bow(striped(), bow_h=0.6)) > 1.2

    def test_bowing_harder_squeezes_harder(self):
        """A slider that stops responding halfway along is worse than none."""
        measured = [end_compression(morph.apply_bow(striped(), bow_h=a))
                    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        assert measured == sorted(measured), measured
        assert measured[-1] > 2.0, measured

    def test_convex_and_concave_differ_the_right_way_round(self):
        """Both curvatures turn their ends away by the same angle, so they look
        more alike than one might expect -- but a middle bulging towards the
        viewer is magnified, and so always spreads more than a hollow one."""
        for amount in (0.2, 0.4, 0.6, 0.8, 1.0):
            convex = end_compression(morph.apply_bow(striped(), bow_h=amount))
            concave = end_compression(morph.apply_bow(striped(), bow_h=-amount))
            assert convex > concave, f"at {amount}: {convex:.3f} vs {concave:.3f}"

    def test_the_sheet_never_folds_over_itself(self):
        """The lookup that inverts the bow is only valid while it increases;
        past that the near end overtakes its neighbour and the artwork tears."""
        for amount in np.linspace(-1.0, 1.0, 81):
            profile = morph._bow_profile(600, float(amount))
            assert np.all(np.diff(profile) > 0), f"folded at bow {amount:.3f}"

    def test_the_ends_stay_pinned_to_the_canvas(self):
        """However hard it is bowed the creative still fills its area, because
        that area is what gets warped into the tracked quad."""
        for amount in (-1.0, -0.5, 0.5, 1.0):
            profile = morph._bow_profile(600, amount)
            assert profile[0] == pytest.approx(0.0, abs=0.01)
            assert profile[-1] == pytest.approx(599.0, abs=0.01)

    def test_it_is_symmetric_about_the_middle(self):
        """Whatever happens on the left happens to the right in mirror image.

        Checked on the profile rather than on a picture: any test image would
        have to be perfectly symmetric itself for the comparison to mean
        anything, and the asymmetry of the picture would hide that of the bow.
        """
        for amount in (-0.8, -0.3, 0.3, 0.8):
            profile = morph._bow_profile(601, amount)
            assert np.allclose(profile + profile[::-1], 600.0, atol=0.05), amount

    def test_the_two_axes_are_independent(self):
        across = morph.apply_bow(striped(), bow_h=0.6)
        down = morph.apply_bow(striped(), bow_v=0.6)
        assert not np.array_equal(across, down)
        assert np.array_equal(stripe_gaps(down), stripe_gaps(striped()))

    def test_the_canvas_keeps_its_size(self, poster):
        assert morph.apply_bow(poster, 0.9, 0.4).shape == poster.shape


class TestTilt:
    def test_turning_the_right_edge_away_makes_it_shorter(self):
        corners = morph.tilt_corners(400, 200, pitch=0, yaw=30, roll=0,
                                     perspective=1.0)
        left = np.linalg.norm(corners[3] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        assert right < left * 0.9, f"left {left:.1f}, right {right:.1f}"

    def test_tipping_the_top_away_makes_it_shorter(self):
        corners = morph.tilt_corners(400, 200, pitch=30, yaw=0, roll=0,
                                     perspective=1.0)
        top = np.linalg.norm(corners[1] - corners[0])
        bottom = np.linalg.norm(corners[2] - corners[3])
        assert top < bottom * 0.9, f"top {top:.1f}, bottom {bottom:.1f}"

    def test_the_two_directions_mirror_each_other(self):
        one = morph.tilt_corners(400, 200, 0, 25, 0)
        other = morph.tilt_corners(400, 200, 0, -25, 0)
        assert np.allclose(np.sort(one[:, 1]), np.sort(other[:, 1]), atol=0.5)
        assert not np.allclose(one, other)

    def test_more_perspective_means_more_foreshortening(self):
        """The control has to do something across its whole travel."""
        def taper(strength):
            corners = morph.tilt_corners(400, 200, 0, 30, 0, strength)
            return (np.linalg.norm(corners[2] - corners[1])
                    / np.linalg.norm(corners[3] - corners[0]))

        measured = [taper(s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert measured == sorted(measured, reverse=True), measured
        assert measured[0] > 0.95, "lowest setting should be nearly a flat squash"
        assert measured[-1] < 0.8, "highest setting should look clearly deep"

    def test_a_roll_turns_the_artwork_clockwise(self):
        """Positive is clockwise on screen, as a rotation means everywhere else."""
        corners = morph.tilt_corners(400, 200, 0, 0, 20)
        assert corners[0][0] > 200 * 0.2, "top-left should have swung right"

    def test_the_result_stays_inside_its_canvas(self):
        for pitch, yaw, roll in [(40, 0, 0), (0, 45, 0), (0, 0, 30), (30, 30, 20)]:
            corners = morph.tilt_corners(400, 200, pitch, yaw, roll, 1.0)
            assert corners[:, 0].min() >= -0.5 and corners[:, 0].max() <= 400.5
            assert corners[:, 1].min() >= -0.5 and corners[:, 1].max() <= 200.5

    def test_the_canvas_keeps_its_size(self, poster):
        assert morph.apply_tilt(poster, 20, 20, 10).shape == poster.shape


class TestWhatTheCompositorSees:
    def test_a_tilt_leaves_transparency_not_black(self, poster):
        """The compositor keys on alpha. Black corners would paint a black
        wedge over the footage instead of letting it show through."""
        tilted = morph.apply_morph(poster, morph.Morph(yaw=35, perspective=0.8))
        assert tilted[0, 0, 3] == 0
        assert tilted[100, 200, 3] == 255

    def test_the_footage_shows_through_where_the_tilt_vacated(self, poster):
        scene = np.full((300, 500, 3), 90, np.uint8)
        quad = np.float32([[40, 30], [460, 30], [460, 270], [40, 270]])
        flat = core.composite_overlay(scene, poster, quad)
        tilted = core.composite_overlay(
            scene, morph.apply_morph(poster, morph.Morph(pitch=35, perspective=0.9)),
            quad)
        untouched = np.abs(tilted.astype(int) - 90).max(axis=2) < 3
        assert untouched.sum() > 0.05 * untouched.size, (
            "a strong tilt should leave part of the area showing the shot")
        assert not np.array_equal(flat, tilted)

    def test_a_three_channel_creative_comes_back_with_alpha(self):
        bgr = np.full((80, 120, 3), (10, 20, 30), np.uint8)
        shaped = morph.apply_morph(bgr, morph.Morph(yaw=20))
        assert shaped.shape == (80, 120, 4)

    def test_bow_and_tilt_together_do_both(self, poster):
        both = morph.apply_morph(poster, morph.Morph(yaw=25, bow_h=0.7))
        bow_only = morph.apply_morph(poster, morph.Morph(bow_h=0.7))
        tilt_only = morph.apply_morph(poster, morph.Morph(yaw=25))
        assert not np.array_equal(both, bow_only)
        assert not np.array_equal(both, tilt_only)


class TestSettingsRoundTrip:
    def test_it_survives_being_written_out_and_read_back(self):
        original = morph.Morph(pitch=12, yaw=-8, roll=3, bow_h=0.4,
                               bow_v=-0.2, perspective=0.65)
        restored = morph.Morph.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_a_copy_is_independent(self):
        original = morph.Morph(yaw=20)
        clone = original.copy()
        clone.yaw = -20
        assert original.yaw == 20

    def test_missing_and_unknown_keys_are_tolerated(self):
        """Settings files written by an older or newer build must still load."""
        assert morph.Morph.from_dict(None).is_identity()
        assert morph.Morph.from_dict({"yaw": 15, "nonsense": 4}).yaw == 15


# --------------------------------------------------------------------------
# In the application
# --------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5.QtWidgets")

import importlib.util                                            # noqa: E402

from PyQt5.QtWidgets import QApplication                         # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "lumen_app",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "LUMEN_Insertion_Tool_1.0.0.py"))
lumen_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lumen_app)


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def app(qapp, clip_video, logo_bgra):
    path, truth = clip_video
    window = lumen_app.MainWindow()
    window.central_panel.load_video(path)
    overlay = window.central_panel.tracking_overlay
    overlay.overlay_bgra = logo_bgra
    overlay.points = [tuple(map(float, p)) for p in truth[0]]
    overlay.tracking_history = {i: [tuple(map(float, c)) for c in q]
                                for i, q in enumerate(truth)}
    yield window, window.central_panel, overlay, truth
    window.dirty = False
    window.close()


class TestInTheApplication:
    def test_a_new_placement_starts_flat(self, app):
        window, panel, overlay, truth = app
        assert overlay.morph.is_identity()

    def test_the_preview_shows_the_shape(self, app):
        window, panel, overlay, truth = app
        frame = panel.prev_frame
        flat = overlay.styled_overlay(frame, overlay.points)
        overlay.morph.yaw = 30
        shaped = overlay.styled_overlay(frame, overlay.points)
        assert not np.array_equal(flat, shaped)

    def test_shaping_happens_before_the_footage_is_matched_to(self, app):
        """Photometric matching measures the creative against the shot, so it
        has to look at the shape that will actually be laid down."""
        window, panel, overlay, truth = app
        overlay.morph.yaw = 35
        overlay.blend.enabled = True
        styled = overlay.styled_overlay(panel.prev_frame, overlay.points)
        assert styled[0, 0, 3] == 0, "the tilt's transparent corner was filled in"

    def test_each_placement_keeps_its_own_shape(self, app):
        window, panel, overlay, truth = app
        overlay.morph.yaw = 20
        second = overlay.add_placement("Second")
        assert second.morph.is_identity()
        assert overlay.placements[0].morph.yaw == 20

    def test_the_shape_is_saved_and_reloaded(self, app, tmp_path):
        window, panel, overlay, truth = app
        overlay.morph.yaw = 18
        overlay.morph.bow_h = 0.35
        restored = lumen_app.Placement.from_dict(overlay.active.to_dict())
        assert restored.morph.yaw == 18
        assert restored.morph.bow_h == pytest.approx(0.35)

    def test_the_render_draws_the_same_shape_as_the_preview(self, app, tmp_path):
        window, panel, overlay, truth = app
        overlay.morph.yaw = 30
        settings = window.build_render_settings(
            0, 4, 1.0, str(tmp_path / "shaped.mp4"))
        carried = settings.placements[0].morph if settings.placements else settings.morph
        assert carried.yaw == 30

    def test_a_render_of_a_shaped_creative_completes(self, app, tmp_path):
        window, panel, overlay, truth = app
        overlay.morph.yaw = 25
        overlay.morph.bow_h = 0.5
        out = str(tmp_path / "shaped.mp4")
        worker = lumen_app.RenderWorker(
            window.build_render_settings(0, 4, 1.0, out))
        statuses = []
        worker.finished.connect(statuses.append)
        worker.run()
        assert statuses and statuses[0].startswith("Completed"), statuses
        assert os.path.exists(out)

    def test_the_dialog_writes_through_to_the_placement(self, app):
        window, panel, overlay, truth = app
        dialog = lumen_app.MorphDialog(overlay.morph, overlay.overlay_bgra,
                                       lambda: None, parent=window)
        dialog.sliders["yaw"].setValue(22)
        dialog.sliders["bow_h"].setValue(40)
        assert overlay.morph.yaw == 22
        assert overlay.morph.bow_h == pytest.approx(0.40)

    def test_cancelling_the_dialog_puts_it_back(self, app):
        window, panel, overlay, truth = app
        overlay.morph.yaw = 10
        dialog = lumen_app.MorphDialog(overlay.morph, overlay.overlay_bgra,
                                       lambda: None, parent=window)
        dialog.sliders["yaw"].setValue(40)
        dialog._revert_and_reject()
        assert overlay.morph.yaw == 10
