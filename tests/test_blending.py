"""Photometric matching: making an insert belong to the shot it sits in."""

import cv2
import numpy as np
import pytest

import galileo_blend as blend
import galileo_core as core


QUAD = np.float32([[120, 90], [520, 110], [505, 360], [135, 340]])


def make_scene(width=640, height=480, noise=6.0, blur=1.6, gradient=True,
               tint=(1.2, 1.0, 0.85), seed=5):
    """Footage as captured: soft, grainy, unevenly lit and colour-cast."""
    rng = np.random.default_rng(seed)
    small = rng.integers(60, 190, (height // 12, width // 12, 3), dtype=np.uint8)
    scene = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    scene = scene.astype(np.float32)
    for channel, factor in enumerate(tint):
        scene[:, :, channel] *= factor
    if gradient:
        _, xx = np.mgrid[0:height, 0:width]
        scene *= (0.6 + 0.55 * (1 - xx / width))[:, :, None]
    if blur:
        scene = cv2.GaussianBlur(scene, (0, 0), blur)
    if noise:
        scene += rng.normal(0, noise, (height, width, 1))
    return np.clip(scene, 0, 255).astype(np.uint8)


@pytest.fixture
def creative():
    """A clean, sharp, saturated advert, as supplied by an agency."""
    bgr = np.full((260, 420, 3), (25, 40, 215), np.uint8)
    cv2.putText(bgr, "SALE", (40, 150), cv2.FONT_HERSHEY_DUPLEX, 3.0,
                (255, 255, 255), 8)
    return np.dstack([bgr, np.full((260, 420), 255, np.uint8)])


class TestMeasurement:
    def test_noise_estimate_tracks_added_noise(self):
        base = np.full((200, 200, 3), 128, np.uint8)
        quiet = blend.estimate_noise_sigma(base)
        noisy = blend.estimate_noise_sigma(
            blend.add_grain(base, 8.0, seed=1))
        assert quiet < 1.0
        assert 5.0 < noisy < 12.0

    def test_noise_estimate_ignores_texture(self):
        """A variance-based estimate would report the texture as noise."""
        clean_texture = cv2.GaussianBlur(make_scene(noise=0.0), (0, 0), 0.6)
        assert blend.estimate_noise_sigma(clean_texture) < 3.0

    def test_detail_drops_when_blurred(self):
        scene = make_scene(noise=0.0, blur=0.0)
        assert blend.estimate_detail(cv2.GaussianBlur(scene, (0, 0), 3.0)) \
            < blend.estimate_detail(scene)

    def test_empty_input_is_safe(self):
        assert blend.estimate_noise_sigma(None) == 0.0
        assert blend.estimate_detail(np.zeros((0, 0), np.uint8)) == 0.0


class TestRectification:
    def test_base_is_warped_into_creative_space(self):
        scene = make_scene()
        patch = blend.base_in_creative_space(scene, QUAD, (420, 260))
        assert patch.shape[:2] == (260, 420)

    def test_alignment_puts_the_right_pixels_in_the_right_place(self):
        """A mark at the quad's top-left must land at the patch's top-left."""
        scene = np.zeros((480, 640, 3), np.uint8)
        corner = np.int32(QUAD[0])
        cv2.circle(scene, tuple(corner), 18, (255, 255, 255), -1)
        patch = blend.base_in_creative_space(scene, QUAD, (420, 260))
        assert patch[:60, :60].mean() > patch[-60:, -60:].mean()

    def test_degenerate_quad_returns_none(self):
        scene = make_scene()
        flat = np.float32([[0, 0], [10, 0], [20, 0], [30, 0]])
        assert blend.base_in_creative_space(scene, flat, (0, 0)) is None


class TestEffects:
    def test_lighting_transfers_the_gradient(self, creative):
        scene = make_scene(gradient=True, noise=0.0)
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        out = blend.match_lighting(creative[:, :, :3], reference, 1.0)
        # The scene falls off to the right, so the insert should too.
        assert out[:, :80].mean() > out[:, -80:].mean() + 10

    def test_lighting_at_zero_is_a_no_op(self, creative):
        scene = make_scene()
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        out = blend.match_lighting(creative[:, :, :3], reference, 0.0)
        assert np.array_equal(out, creative[:, :, :3])

    def test_colour_shifts_towards_the_reference(self, creative):
        scene = make_scene(tint=(1.6, 1.0, 0.6))
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        out = blend.match_colour(creative[:, :, :3], reference, 1.0)
        before = abs(float(creative[:, :, 0].mean()) - float(reference[:, :, 0].mean()))
        after = abs(float(out[:, :, 0].mean()) - float(reference[:, :, 0].mean()))
        assert after < before

    def test_colour_strength_scales_the_shift(self, creative):
        scene = make_scene(tint=(1.6, 1.0, 0.6))
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        original = creative[:, :, :3]
        gentle = blend.match_colour(original, reference, 0.15)
        hard = blend.match_colour(original, reference, 1.0)

        def shift(img):
            return np.linalg.norm(img.reshape(-1, 3).mean(axis=0)
                                  - original.reshape(-1, 3).mean(axis=0))

        assert shift(gentle) < shift(hard)

    def test_sharpness_softens_a_crisp_creative(self, creative):
        scene = make_scene(blur=3.0, noise=0.0)
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        out = blend.match_sharpness(creative[:, :, :3], reference, 1.0)
        assert blend.estimate_detail(out) < blend.estimate_detail(creative[:, :, :3])

    def test_sharpness_never_sharpens(self, creative):
        """Matching a crisper scene must not invent detail."""
        scene = make_scene(blur=0.0, noise=0.0)
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        out = blend.match_sharpness(creative[:, :, :3], reference, 1.0)
        assert blend.estimate_detail(out) <= blend.estimate_detail(creative[:, :, :3]) + 1

    def test_motion_blur_smears_along_the_direction_of_travel(self, creative):
        out = blend.motion_blur(creative, 20.0, 0.0, 1.0)
        # Horizontal travel: vertical edges soften, horizontal ones survive.
        gx = cv2.Sobel(out[:, :, :3], cv2.CV_32F, 1, 0).var()
        gy = cv2.Sobel(out[:, :, :3], cv2.CV_32F, 0, 1).var()
        original_gx = cv2.Sobel(creative[:, :, :3], cv2.CV_32F, 1, 0).var()
        assert gx < original_gx
        assert gx < gy

    def test_no_motion_is_a_no_op(self, creative):
        assert blend.motion_blur(creative, 0.0, 0.0, 1.0) is creative

    def test_grain_adds_measurable_noise(self):
        flat = np.full((200, 200, 3), 120, np.uint8)
        assert blend.estimate_noise_sigma(blend.add_grain(flat, 6.0, seed=2)) > 3.0

    def test_grain_respects_the_mask(self):
        flat = np.full((200, 200, 3), 120, np.uint8)
        mask = np.zeros((200, 200), np.uint8)
        mask[:100] = 255
        out = blend.add_grain(flat, 10.0, mask, seed=3)
        assert not np.array_equal(out[:100], flat[:100])
        assert np.array_equal(out[100:], flat[100:])

    def test_highlights_can_show_through(self, creative):
        scene = np.zeros((480, 640, 3), np.uint8)
        cv2.circle(scene, (300, 200), 60, (255, 255, 255), -1)
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        out = blend.preserve_highlights(creative[:, :, :3], reference, 1.0)
        assert out.mean() > creative[:, :, :3].mean()


class TestSettings:
    def test_colour_defaults_low_and_the_rest_high(self):
        """Colour is the one control that alters the ad being tested."""
        settings = blend.BlendSettings()
        assert settings.colour <= 0.2
        assert settings.lighting >= 0.5
        assert settings.grain >= 0.5
        assert settings.sharpness >= 0.5

    def test_off_disables_everything(self):
        settings = blend.BlendSettings.off()
        assert not settings.enabled
        assert all(getattr(settings, name) == 0 for name in
                   ("lighting", "colour", "sharpness", "grain", "motion"))

    def test_round_trips_through_a_dict(self):
        settings = blend.BlendSettings(lighting=0.3, grain=0.9)
        restored = blend.BlendSettings.from_dict(settings.to_dict())
        assert restored.lighting == pytest.approx(0.3)
        assert restored.grain == pytest.approx(0.9)

    def test_unknown_keys_are_ignored(self):
        restored = blend.BlendSettings.from_dict({"lighting": 0.4, "nonsense": 1})
        assert restored.lighting == pytest.approx(0.4)


class TestPhotometricMatch:
    def test_alpha_survives_the_match(self, creative):
        scene = make_scene()
        creative[40:80, 40:80, 3] = 0
        out = blend.photometric_match(creative, scene, QUAD)
        assert out.shape[2] == 4
        assert np.all(out[40:80, 40:80, 3] == 0)

    def test_disabled_settings_pass_the_creative_through(self, creative):
        scene = make_scene()
        out = blend.photometric_match(creative, scene, QUAD,
                                      blend.BlendSettings.off())
        assert out is creative

    def test_matching_moves_the_insert_towards_the_scene(self, creative):
        scene = make_scene()
        matched = blend.photometric_match(creative, scene, QUAD)
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))

        def distance(img):
            return abs(float(img[:, :, :3].mean()) - float(reference.mean()))

        assert distance(matched) < distance(creative)

    def test_output_is_finite_and_uint8(self, creative):
        scene = make_scene()
        out = blend.photometric_match(creative, scene, QUAD,
                                      motion=(9.0, 2.0))
        assert out.dtype == np.uint8
        assert np.all(np.isfinite(out))

    def test_offscreen_quad_is_handled(self, creative):
        scene = make_scene()
        away = np.float32([[-900, -900], [-800, -900], [-800, -800], [-900, -800]])
        assert blend.photometric_match(creative, scene, away) is not None

    def test_grain_sigma_follows_the_footage(self, creative):
        quiet = blend.grain_sigma_for(make_scene(noise=0.0), QUAD)
        noisy = blend.grain_sigma_for(make_scene(noise=12.0), QUAD)
        assert noisy > quiet

    def test_grain_sigma_is_zero_when_disabled(self):
        assert blend.grain_sigma_for(make_scene(), QUAD,
                                     blend.BlendSettings.off()) == 0.0


class TestEndToEnd:
    def test_matched_insert_is_closer_to_the_scene_than_a_naive_paste(self, creative):
        """The whole point: a pasted advert stands out, a matched one does not."""
        scene = make_scene()
        region = core.Region(QUAD)

        naive = core.composite_region(scene, creative, region)
        matched_creative = blend.photometric_match(creative, scene, QUAD,
                                                   motion=(6.0, 1.0))
        matched = core.composite_region(scene, matched_creative, region)
        matched = blend.add_grain(
            matched, blend.grain_sigma_for(scene, QUAD),
            core.quad_to_mask(QUAD, 640, 480), seed=1)

        inside = core.quad_to_mask(QUAD, 640, 480) > 0
        outside = ~inside
        scene_mean = scene[outside].mean()

        assert abs(matched[inside].mean() - scene_mean) < \
            abs(naive[inside].mean() - scene_mean)

    def test_matched_insert_is_not_conspicuously_sharp(self, creative):
        scene = make_scene(blur=2.4, noise=0.0)
        matched = blend.photometric_match(creative, scene, QUAD)
        reference = blend.base_in_creative_space(scene, QUAD, (420, 260))
        # Within a factor of the surface's own detail, rather than many times it.
        assert blend.estimate_detail(matched) < blend.estimate_detail(creative) * 0.9
        assert blend.estimate_detail(matched) < blend.estimate_detail(reference) * 8
