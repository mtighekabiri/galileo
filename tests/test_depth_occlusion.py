"""Occlusion by depth: drawing the insert behind anything at all in front of it.

The person model handles one kind of obstruction. This handles the rest --
railings, lampposts, signs -- by asking how far away things are rather than
what they are, and holding the creative back wherever the answer is "nearer
than the surface".

Most of what follows never loads the model. What the model produces is a depth
map; what turns one into a mask is :func:`plane_deviation_mask`, and that is
plain arithmetic over an array, so the behaviour worth pinning down can be
stated exactly with depth maps built by hand. The cases are the ones a real
shot actually presents: an obstruction across the middle, a surface seen at an
angle, sky showing past a loosely marked edge.
"""

import cv2
import numpy as np
import pytest

import galileo_core as core


WIDTH, HEIGHT = 320, 240
QUAD = np.float32([[40, 30], [280, 40], [270, 200], [50, 190]])


@pytest.fixture(scope="module")
def interior():
    """The pixels inside the marked area, which is what the claims are about."""
    return core.quad_to_mask(QUAD, WIDTH, HEIGHT) > 127


def coordinates():
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    return xs / (WIDTH - 1), ys / (HEIGHT - 1)


def surface(tilt_x=0.6, tilt_y=-0.3, offset=2.0):
    """A flat surface as the network sees it: inverse depth, affine in x and y.

    Any plane looks like this from any angle, which is the whole reason the
    detector fits one rather than thresholding depth outright.
    """
    xs, ys = coordinates()
    return (tilt_x * xs + tilt_y * ys + offset).astype(np.float32)


def region(rows, columns):
    patch = np.zeros((HEIGHT, WIDTH), bool)
    patch[rows, columns] = True
    return patch


def marked(mask, where, interior):
    """The fraction of ``where``, inside the marked area, the mask holds back."""
    selected = where & interior
    return float((mask[selected] > 127).mean()) if selected.any() else 0.0


class TestNothingInFront:
    """No obstruction must mean no mask. A detector that nibbles at the
    creative on clean frames is worse than none: the damage is spread over
    every frame instead of confined to the ones with something in the way."""

    def test_a_clean_surface_is_left_alone(self):
        assert core.plane_deviation_mask(surface(), QUAD).sum() == 0

    def test_noise_alone_is_not_an_obstruction(self):
        rng = np.random.default_rng(0)
        noisy = (surface() + rng.normal(0, 0.01, (HEIGHT, WIDTH))).astype(np.float32)
        assert core.plane_deviation_mask(noisy, QUAD).sum() == 0

    def test_a_surface_the_model_bends_slightly_is_tolerated(self):
        """Depth estimates bow gently over a real wall; the threshold is set
        against the scene's own depth range so that bowing stays under it."""
        xs, ys = coordinates()
        flat = surface()
        span = float(flat.max() - flat.min())
        bowed = flat + (0.02 * span * ((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / 0.5)
        assert core.plane_deviation_mask(bowed.astype(np.float32), QUAD).sum() == 0

    def test_a_featureless_map_yields_nothing(self):
        """A model given nothing to work with returns something shapeless.
        Whatever it is, it is not an obstruction."""
        assert core.plane_deviation_mask(np.full((HEIGHT, WIDTH), 7.0, np.float32),
                                         QUAD).sum() == 0


class TestSomethingInFront:
    def test_an_object_in_front_is_masked(self, interior):
        blob = region(slice(90, 150), slice(120, 180))
        depth = surface()
        depth[blob] += 0.8

        mask = core.plane_deviation_mask(depth, QUAD)
        assert marked(mask, blob, interior) > 0.9
        assert marked(mask, ~blob, interior) < 0.01, "surface eaten away"

    def test_a_thin_bar_survives(self, interior):
        """A railing is a few pixels wide by the time the model sees it, and
        the speckle filter it has to get past is the same size. This is the
        narrowest thing the detector claims to find, so it is pinned here."""
        bar = region(slice(None), slice(150, 153))
        depth = surface()
        depth[bar] += 0.8

        assert marked(core.plane_deviation_mask(depth, QUAD), bar, interior) > 0.8

    def test_a_surface_seen_at_an_angle_is_not_confused(self, interior):
        """Steep foreshortening spreads the surface's own depth over a wide
        range -- wider, here, than the obstruction stands off it. Fitting a
        plane is what separates the two; a fixed depth cut could not."""
        xs, ys = coordinates()
        blob = region(slice(90, 150), slice(120, 180))
        depth = (4.0 * xs - 3.0 * ys + 2.0).astype(np.float32)
        depth[blob] += 0.8

        mask = core.plane_deviation_mask(depth, QUAD)
        assert marked(mask, blob, interior) > 0.9
        assert marked(mask, ~blob, interior) < 0.01

    def test_obstructions_up_to_half_the_area_are_found(self, interior):
        covering = region(slice(60, 170), slice(60, 200))
        depth = surface()
        depth[covering] += 0.8

        mask = core.plane_deviation_mask(depth, QUAD)
        assert marked(mask, covering, interior) > 0.85
        assert marked(mask, ~covering, interior) < 0.02

    def test_a_covering_obstruction_fails_soft(self, interior):
        """Past about half the area the obstruction becomes the majority of
        what was marked, and the fit takes it for the surface. What matters is
        the direction of the failure: nothing is marked, so the creative is
        painted over as it would be with the feature switched off -- rather
        than the surface being marked and the creative deleted from the shot."""
        covering = region(slice(60, 200), slice(60, 230))
        depth = surface()
        depth[covering] += 0.8

        mask = core.plane_deviation_mask(depth, QUAD)
        assert marked(mask, ~covering, interior) < 0.02, "the surface was masked"


class TestThingsBehind:
    """Nothing behind the surface may ever be masked. This is what lets the
    detector run on a digital screen: an advert of a landscape reads as
    receding, and receding is ignored, so the picture the screen is playing
    cannot eat into the creative replacing it."""

    def test_content_beyond_the_surface_is_ignored(self, interior):
        blob = region(slice(90, 150), slice(120, 180))
        depth = surface()
        depth[blob] -= 0.8

        mask = core.plane_deviation_mask(depth, QUAD)
        assert marked(mask, np.ones((HEIGHT, WIDTH), bool), interior) < 0.01

    @pytest.mark.parametrize("rows, columns", [
        (slice(90, 150), slice(120, 180)),      # a tenth of the area
        (slice(60, 180), slice(80, 220)),       # nearly half of it
    ])
    def test_distance_showing_through_the_edge_does_not_mask_the_surface(
            self, rows, columns, interior):
        """A quad marked loosely around a billboard catches sky past its edge.
        The sky is genuinely far away, and a detector that preferred distance
        would call the billboard an obstruction standing in front of it and
        mask the creative away completely."""
        beyond = region(rows, columns)
        depth = surface()
        depth[beyond] -= 0.8

        mask = core.plane_deviation_mask(depth, QUAD)
        assert marked(mask, ~beyond, interior) < 0.02


class TestDegenerateInput:
    def test_a_tiny_quad_yields_an_empty_mask(self):
        tiny = np.float32([[5, 5], [7, 5], [7, 7], [5, 7]])
        assert core.plane_deviation_mask(surface(), tiny).sum() == 0

    def test_a_quad_off_the_map_yields_an_empty_mask(self):
        away = np.float32([[-900, -900], [-880, -900], [-880, -880], [-900, -880]])
        assert core.plane_deviation_mask(surface(), away).sum() == 0

    def test_the_mask_matches_the_map_it_came_from(self):
        mask = core.plane_deviation_mask(surface(), QUAD)
        assert mask.shape == (HEIGHT, WIDTH)
        assert mask.dtype == np.uint8


class TestTheFitItself:
    def test_it_recovers_a_plane_it_is_given(self):
        interior = core.quad_to_mask(QUAD, WIDTH, HEIGHT) > 127
        (a, b, c), scale = core._fit_plane_robust(surface(0.6, -0.3, 2.0), interior)

        assert a == pytest.approx(0.6, abs=1e-3)
        assert b == pytest.approx(-0.3, abs=1e-3)
        assert c == pytest.approx(2.0, abs=1e-3)
        assert scale == pytest.approx(0.0, abs=1e-4)

    def test_an_obstruction_does_not_drag_the_fit_forward(self):
        """The fit has to stay on the surface, not split the difference with
        whatever is in front of it -- everything measured afterwards is
        measured from this plane."""
        interior = core.quad_to_mask(QUAD, WIDTH, HEIGHT) > 127
        depth = surface(0.6, -0.3, 2.0)
        depth[region(slice(80, 150), slice(100, 200))] += 0.8

        (a, b, c), _ = core._fit_plane_robust(depth, interior)
        assert a == pytest.approx(0.6, abs=1e-2)
        assert b == pytest.approx(-0.3, abs=1e-2)
        assert c == pytest.approx(2.0, abs=1e-2)

    def test_it_is_deterministic(self):
        """The preview jumps between frames and the renderer walks them in
        order; both have to produce the same mask for the same frame, so
        nothing here may be sampled at random."""
        interior = core.quad_to_mask(QUAD, WIDTH, HEIGHT) > 127
        rng = np.random.default_rng(5)
        depth = (surface() + rng.normal(0, 0.02, (HEIGHT, WIDTH))).astype(np.float32)
        depth[region(slice(90, 150), slice(120, 180))] += 0.8

        first = core._fit_plane_robust(depth, interior)
        assert all(core._fit_plane_robust(depth, interior) == first
                   for _ in range(3))


model_available = pytest.mark.skipif(
    not core.DepthOcclusionSegmenter.is_available(),
    reason="depth model not downloaded; run fetch_model.py")


def scene_with_a_panel():
    """A street-ish scene with a flat panel on a wall and a receding floor."""
    height, width = 540, 960
    frame = np.full((height, width, 3), (150, 140, 130), np.uint8)
    for y in range(300, height):
        shade = int(60 + (y - 300) * 0.35)
        frame[y, :] = (shade, shade, shade + 8)
    panel = np.float32([[250, 120], [700, 120], [700, 380], [250, 380]])
    cv2.rectangle(frame, (250, 120), (700, 380), (60, 90, 200), -1)
    cv2.rectangle(frame, (250, 120), (700, 380), (20, 20, 20), 6)
    for i in range(5):
        cv2.putText(frame, "SALE", (280, 180 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame, panel


@model_available
class TestTheModel:
    def test_it_reports_being_available(self):
        assert core.DepthOcclusionSegmenter.is_available() is True

    def test_the_mask_covers_the_frame_and_is_a_mask(self):
        segmenter = core.DepthOcclusionSegmenter()
        frame, panel = scene_with_a_panel()

        mask = segmenter.mask(frame, panel)
        assert mask.shape == frame.shape[:2]
        assert mask.dtype == np.uint8
        assert mask.max() <= 255

    def test_an_unobstructed_panel_is_barely_touched(self):
        """Run end to end through the real model: the whole point is that an
        ordinary frame comes back clean."""
        segmenter = core.DepthOcclusionSegmenter()
        frame, panel = scene_with_a_panel()
        inside = core.quad_to_mask(panel, frame.shape[1], frame.shape[0]) > 127

        mask = segmenter.mask(frame, panel)
        assert float((mask[inside] > 127).mean()) < 0.02

    def test_something_standing_in_front_is_found(self):
        segmenter = core.DepthOcclusionSegmenter()
        frame, panel = scene_with_a_panel()
        cv2.rectangle(frame, (430, 0), (470, frame.shape[0]), (70, 70, 75), -1)
        cv2.rectangle(frame, (430, 0), (470, frame.shape[0]), (25, 25, 28), 3)

        inside = core.quad_to_mask(panel, frame.shape[1], frame.shape[0]) > 127
        post = np.zeros(frame.shape[:2], bool)
        post[:, 430:470] = True

        mask = segmenter.mask(frame, panel)
        assert float((mask[post & inside] > 127).mean()) > 0.8
        assert float((mask[inside & ~post] > 127).mean()) < 0.25

    def test_the_crop_keeps_the_mask_near_the_area(self):
        """Running on a crop is what keeps a distant railing big enough to
        register at all; nothing outside it can be marked."""
        segmenter = core.DepthOcclusionSegmenter()
        frame = np.full((360, 640, 3), 128, np.uint8)
        quad = np.float32([[50, 50], [150, 50], [150, 150], [50, 150]])

        mask = segmenter.mask(frame, quad)
        assert int(mask[300:, 500:].sum()) == 0


class TestModelDiscovery:
    def test_missing_model_raises_a_useful_error(self, monkeypatch):
        """The error has to say how to fix it; users will hit this first."""
        monkeypatch.setattr(core, "find_model", lambda name: None)
        with pytest.raises(FileNotFoundError, match="fetch_model.py"):
            core.DepthOcclusionSegmenter()

    def test_availability_is_false_without_the_model(self, monkeypatch):
        monkeypatch.setattr(core, "find_model", lambda name: None)
        assert core.DepthOcclusionSegmenter.is_available() is False

    def test_either_model_can_be_present_without_the_other(self, monkeypatch):
        """They are separate downloads behind separate menu items, and one
        being absent must not switch the other off."""
        monkeypatch.setattr(
            core, "find_model",
            lambda name: ("/models/" + name
                          if name == core.PersonSegmenter.MODEL_FILENAME else None))
        assert core.PersonSegmenter.is_available() is True
        assert core.DepthOcclusionSegmenter.is_available() is False

    def test_loading_a_bogus_model_raises(self, tmp_path):
        bogus = tmp_path / "bogus.onnx"
        bogus.write_bytes(b"not an onnx file")
        with pytest.raises(IOError):
            core.DepthOcclusionSegmenter(model_path=str(bogus))
