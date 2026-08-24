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
        """The scene must carry real depth cues -- sky, a ground plane, a post
        whose ground contact sits well below the panel -- and the artwork must
        be photographic in character. Two earlier versions of this test each
        pinned an accident: a cue-free painted rectangle that only MiDaS fired
        on (Depth Anything correctly read it as flat and failed by being
        right), and giant flat SALE lettering that Depth Anything reads as
        popped out of the panel at any threshold -- a real caution, recorded
        in the README, but a property of cartoon text rather than of the
        detector."""
        segmenter = core.DepthOcclusionSegmenter()
        height, width = 540, 960
        frame = np.full((height, width, 3), 110, np.uint8)
        frame[:200] = (215, 205, 195)                       # sky
        for y in range(340, height):                        # receding ground
            shade = int(58 + (y - 340) * 0.45)
            frame[y, :] = (shade, shade, shade + 6)
        panel = np.float32([[250, 210], [700, 205], [700, 360], [250, 365]])
        # Photographic artwork -- a soft sky with a pale sun. Chosen by
        # measurement, since each model pops a different kind of depicted
        # content: banded gradients read as off-plane to MiDaS (46% falsely
        # marked) and flat lettering to Depth Anything (45%), while both read
        # this at 2-3%.
        art = np.zeros((160, 450, 3), np.uint8)
        for y in range(160):
            t = y / 160
            art[y, :] = (int(180 - 40 * t), int(160 - 30 * t),
                         int(140 + 40 * t))
        cv2.circle(art, (337, 56), 32, (200, 205, 225), -1)
        frame[205:365, 250:700] = cv2.GaussianBlur(art, (31, 31), 9)
        cv2.rectangle(frame, (430, 0), (466, height), (52, 52, 56), -1)

        inside = core.quad_to_mask(panel, width, height) > 127
        post = np.zeros(frame.shape[:2], bool)
        post[:, 430:466] = True

        mask = segmenter.mask(frame, panel)
        assert float((mask[post & inside] > 127).mean()) > 0.8, segmenter.model_name
        assert float((mask[inside & ~post] > 127).mean()) < 0.25, segmenter.model_name

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


class TestThePictureAlreadyOnTheBillboard:
    """The hard case, and the one that shows up as a fault rather than a
    missed obstruction: a hoarding is rarely blank, and a depth network reads
    the photograph on it as the scene it depicts. Where that reading crosses
    the threshold the creative is held back and the old artwork shows through
    -- and because the reading shifts frame to frame it does not show up as a
    steady error but as holes flashing open and shut."""

    def depicted(self, strength):
        """Artwork with depth in it: a shape the network reads as leaning out.

        Expressed as a fraction of the scene's depth range, which is what the
        floor is measured in, so the numbers here mean the same thing as the
        ones measured off real footage.
        """
        depth = surface()
        spread = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        depth[region(slice(70, 140), slice(90, 200))] += strength * spread
        return depth

    def test_artwork_with_depth_in_it_does_not_open_holes(self, interior):
        """Measured off a hoarding showing a road running away under a pan,
        content on the surface reached 0.13 of the scene's depth range at its
        worst. The default floor has to sit clear of that."""
        mask = core.plane_deviation_mask(self.depicted(0.13), QUAD)
        assert marked(mask, np.ones((HEIGHT, WIDTH), bool), interior) < 0.01

    def test_a_real_obstruction_is_still_found(self, interior):
        """The other side of it: a lamppost across the same shot measured
        0.41, railings 0.86. Both have to stay well inside the net."""
        for strength in (0.41, 0.86):
            mask = core.plane_deviation_mask(self.depicted(strength), QUAD)
            found = marked(mask, region(slice(70, 140), slice(90, 200)), interior)
            assert found > 0.9, f"only {found:.0%} of a {strength} obstruction"

    def test_the_default_sits_between_the_two(self):
        assert 0.13 < core.DEPTH_FLOOR < 0.41

    def test_the_dial_trades_one_against_the_other(self, interior):
        """Which way to lean is not something the tool can know -- it depends
        on artwork it has never seen -- so it is a dial rather than a decision.
        Turning it up holds back less, monotonically, which is what makes it
        usable by eye."""
        borderline = self.depicted(0.20)
        held = [marked(core.plane_deviation_mask(borderline, QUAD, floor_rel=f),
                       np.ones((HEIGHT, WIDTH), bool), interior)
                for f in (0.05, 0.15, 0.30, 0.50)]
        assert held == sorted(held, reverse=True), held
        assert held[0] > 0 and held[-1] == 0


class TestTheDials:
    """What the sliders are allowed to be, since a project saved by another
    version has to open with settings this one can actually apply."""

    def test_the_defaults_are_the_measured_ones(self):
        assert core.DepthSettings().floor_rel == core.DEPTH_FLOOR

    def test_every_dial_has_a_range_that_contains_its_default(self):
        fresh = core.DepthSettings()
        for name, title, low, high, step, hint in core.DepthSettings.DIALS:
            value = getattr(fresh, name)
            assert low <= value <= high, f"{name} starts outside its own range"
            assert step > 0 and high > low, name
            assert title and hint, f"{name} has nothing to say for itself"

    def test_it_survives_a_round_trip(self):
        settings = core.DepthSettings(floor_rel=0.31, k=8.5, dilate=7, feather=1.0)
        assert core.DepthSettings.from_dict(settings.to_dict()) == settings

    def test_a_setting_from_beyond_the_dial_is_brought_back_to_it(self):
        restored = core.DepthSettings.from_dict({"floor_rel": 40.0, "k": -3})
        assert restored.floor_rel == 0.60
        assert restored.k == 1.0

    def test_rubbish_leaves_the_default_standing(self):
        restored = core.DepthSettings.from_dict(
            {"floor_rel": "quite a lot", "nonsense": 1})
        assert restored.floor_rel == core.DEPTH_FLOOR

    def test_the_segmenter_is_built_from_them(self):
        settings = core.DepthSettings(floor_rel=0.4, k=9, dilate=6, feather=1.5)
        monkey = core.DepthOcclusionSegmenter.__new__(core.DepthOcclusionSegmenter)
        # Only the numbers are being checked, so the model file is not needed.
        for name, value in settings.to_dict().items():
            setattr(monkey, name, value)
        assert monkey.floor_rel == 0.4 and monkey.k == 9


class TestApproachingTheSurface:
    """The footage this is pointed at is shot walking or driving past, so a
    panel goes from a small distant rectangle to filling the frame inside one
    shot. The threshold's scene-depth term is a fraction of the depth range of
    a padded crop, and that crop's composition changes completely on the way
    in: mostly street at the far end, almost entirely panel at the near one.

    Past that point the term is measuring the panel against itself -- against
    the depth its *artwork* depicts -- and holes open in the creative. So the
    demand is stiffened in proportion to how little of the crop is anything
    else, which is to say the measurement is distrusted exactly as far as it
    has stopped being one.
    """

    def surface_with_depicted_depth(self, strength=0.30):
        """A panel whose own artwork reads as standing off it."""
        depth = surface()
        spread = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        depth[region(slice(70, 140), slice(90, 200))] += strength * spread
        return depth

    def test_a_panel_with_room_around_it_is_judged_as_before(self, interior):
        """Nothing changes while the crop still holds a scene: the quad here
        covers about a fifth of the map."""
        small = np.float32([[120, 90], [200, 92], [198, 150], [118, 148]])
        depth = surface()
        depth[region(slice(100, 130), slice(140, 175))] += 4.0

        loose = core.plane_deviation_mask(depth, small, min_context=0.0)
        stiffened = core.plane_deviation_mask(depth, small)
        assert np.array_equal(loose, stiffened)

    def test_a_panel_filling_the_crop_is_judged_harder(self):
        """The quad covers nearly everything, so there is no scene left to
        measure against and the artwork's own depth is all that is left."""
        filling = np.float32([[1, 1], [WIDTH - 2, 1],
                              [WIDTH - 2, HEIGHT - 2], [1, HEIGHT - 2]])
        depth = self.surface_with_depicted_depth()

        loose = core.plane_deviation_mask(depth, filling, min_context=0.0)
        stiffened = core.plane_deviation_mask(depth, filling)
        assert stiffened.sum() < loose.sum(), \
            "a panel with no context around it was taken at face value"

    def test_something_really_in_front_still_survives_it(self):
        """Stiffening must not amount to switching the feature off up close:
        a real obstruction stands off the panel far further than its artwork
        does, and has to keep clearing the raised bar."""
        filling = np.float32([[1, 1], [WIDTH - 2, 1],
                              [WIDTH - 2, HEIGHT - 2], [1, HEIGHT - 2]])
        depth = self.surface_with_depicted_depth()
        obstruction = region(slice(30, 60), slice(30, 220))
        spread = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        depth[obstruction] += 3.0 * spread

        mask = core.plane_deviation_mask(depth, filling)
        found = float((mask[obstruction] > 127).mean())
        assert found > 0.9, f"only {found:.0%} of a real obstruction survived"

    def test_the_stiffening_grows_as_the_context_goes(self):
        """Not a cliff edge: a shot walks through these as the panel grows, so
        a step would show as the mask lurching on one frame."""
        depth = self.surface_with_depicted_depth()
        removed = []
        for inset in (60, 40, 20, 1):          # ever bigger quad, ever less room
            quad = np.float32([[inset, inset], [WIDTH - inset, inset],
                               [WIDTH - inset, HEIGHT - inset],
                               [inset, HEIGHT - inset]])
            inside = core.quad_to_mask(quad, WIDTH, HEIGHT) > 127
            loose = core.plane_deviation_mask(depth, quad, min_context=0.0)
            stiffened = core.plane_deviation_mask(depth, quad)
            # How much of the false masking the stiffening took away.
            removed.append(float((loose[inside] > 127).mean())
                           - float((stiffened[inside] > 127).mean()))

        assert removed == sorted(removed), f"not gradual: {removed}"
        assert removed[0] == 0.0, "stiffened a panel that still had room around it"
        assert removed[-1] > 0.0, "did nothing to a panel filling the crop"


class TestWhichDepthModelServes:
    """Two files can serve, and the better one is preferred. Selection has to
    be provable without the network, so these drive it through find_model."""

    def test_the_preferred_model_wins_when_both_are_present(self, monkeypatch):
        if not (core.find_model(core.DepthOcclusionSegmenter.PREFERRED_FILENAME)
                and core.find_model(core.DepthOcclusionSegmenter.MODEL_FILENAME)):
            pytest.skip("both depth models needed; run fetch_model.py")
        assert core.DepthOcclusionSegmenter().model_name == "depth-anything-v2-small"

    def test_midas_serves_when_the_preferred_file_is_absent(self, monkeypatch):
        midas = core.find_model(core.DepthOcclusionSegmenter.MODEL_FILENAME)
        if not midas:
            pytest.skip("MiDaS model needed; run fetch_model.py")
        monkeypatch.setattr(core, "find_model",
                            lambda name: midas
                            if name == core.DepthOcclusionSegmenter.MODEL_FILENAME
                            else None)
        assert core.DepthOcclusionSegmenter().model_name == "midas-v21-small"

    def test_a_preferred_file_that_will_not_load_falls_back(self, monkeypatch,
                                                            tmp_path):
        """The expected shape of an OpenCV 4.x machine: the transformer file
        is present but its engine cannot load it. Yesterday's model has to
        keep working rather than the feature dying on an upgrade."""
        midas = core.find_model(core.DepthOcclusionSegmenter.MODEL_FILENAME)
        if not midas:
            pytest.skip("MiDaS model needed; run fetch_model.py")
        bogus = tmp_path / core.DepthOcclusionSegmenter.PREFERRED_FILENAME
        bogus.write_bytes(b"not a network")
        monkeypatch.setattr(core, "find_model",
                            lambda name: str(bogus)
                            if name == core.DepthOcclusionSegmenter.PREFERRED_FILENAME
                            else midas)
        segmenter = core.DepthOcclusionSegmenter()
        assert segmenter.model_name == "midas-v21-small"

    def test_availability_means_either_file(self, monkeypatch):
        monkeypatch.setattr(core, "find_model",
                            lambda name: "/tmp/x.onnx"
                            if name == core.DepthOcclusionSegmenter.PREFERRED_FILENAME
                            else None)
        assert core.DepthOcclusionSegmenter.is_available() is True
        monkeypatch.setattr(core, "find_model", lambda name: None)
        assert core.DepthOcclusionSegmenter.is_available() is False

    def test_the_fetch_list_carries_all_three(self):
        import fetch_model
        names = [m["filename"] for m in fetch_model.MODELS]
        assert core.DepthOcclusionSegmenter.PREFERRED_FILENAME in names
        assert core.DepthOcclusionSegmenter.MODEL_FILENAME in names
        assert core.PersonSegmenter.MODEL_FILENAME in names
        preferred = next(m for m in fetch_model.MODELS
                         if m["filename"] ==
                         core.DepthOcclusionSegmenter.PREFERRED_FILENAME)
        assert preferred["expected_bytes"] == 99060839
