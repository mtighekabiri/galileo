"""The artwork cue: the panel's own picture as the occlusion reference.

The depth model has two measured blind spots on drive-up footage: a post
crossing the hoarding between 18% and 45% of frame width is found on 0-7% of
its pixels, and railing bars thinner than about a sixtieth of its crop are
never seen at all. These tests pin the cue that covers both: rectify the
tracked quad to one canonical rectangle, learn the artwork as a per-pixel
median over the shot, and mark whatever disagrees with it.

Everything runs through real written video files, not arrays in memory: codec
compression is part of the problem this has to survive, and an earlier
in-memory measurement flattered the numbers.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

import galileo_core as core
from conftest import write_video

WIDTH, HEIGHT = 960, 540
FRAMES = 32


def artwork(seed=None, w=480, h=270):
    """A perspective scene for the hoarding -- or, seeded, a changing one."""
    if seed is not None:
        rng = np.random.default_rng(seed)
        block = rng.integers(30, 220, (h // 8, w // 8, 3), dtype=np.uint8)
        return cv2.resize(block, (w, h), interpolation=cv2.INTER_NEAREST)
    img = np.full((h, w, 3), (150, 170, 190), np.uint8)
    cv2.fillPoly(img, [np.int32([[0, h], [w, h],
                                 [w // 2 + 40, h // 3], [w // 2 - 40, h // 3]])],
                 (70, 70, 75))
    for t in np.linspace(0.05, 0.95, 9):
        y = int(h - (h - h / 3) * t)
        half = int((w / 2) * (1 - t) + 20)
        cv2.line(img, (w // 2 - half, y), (w // 2 + half, y),
                 (220, 220, 220), max(1, int(4 * (1 - t))))
    cv2.ellipse(img, (int(w * 0.78), int(h * 0.55)), (46, 64), 0, 0, 360,
                (180, 170, 160), -1)
    return img


STATIC_ARTWORK = artwork()


def approach_frame(index, count=FRAMES, screen=False, thin=False,
                   exposure_swing=False):
    """One frame of a drive-up: the panel grows from 10% to 105% of the frame,
    while an obstruction -- nearer, so moving with parallax -- sweeps across."""
    progress = index / (count - 1)
    scale = 0.10 + 0.95 * progress
    bw, bh = WIDTH * scale, WIDTH * scale * 0.56
    cx, cy = WIDTH * 0.5, HEIGHT * 0.45
    quad = np.float32([[cx - bw / 2, cy - bh / 2],
                       [cx + bw / 2, cy - bh / 2 * 0.96],
                       [cx + bw / 2, cy + bh / 2],
                       [cx - bw / 2, cy + bh / 2 * 1.02]])

    frame = np.full((HEIGHT, WIDTH, 3), 110, np.uint8)
    frame[:60] = (200, 200, 205)
    frame[HEIGHT - 90:] = (70, 70, 75)
    for x in range(0, WIDTH, 120):
        cv2.rectangle(frame, (x, 60), (x + 70, HEIGHT - 90),
                      (90 + ((x // 120) % 3) * 18,) * 3, -1)

    picture = artwork(seed=index // 2) if screen else STATIC_ARTWORK
    ah, aw = picture.shape[:2]
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [aw, 0], [aw, ah], [0, ah]]), quad)
    warped = cv2.warpPerspective(picture, matrix, (WIDTH, HEIGHT))
    panel = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillPoly(panel, [quad.astype(np.int32)], 255)
    frame[panel > 0] = warped[panel > 0]

    truth = np.zeros((HEIGHT, WIDTH), np.uint8)
    relative = 0.34 - 0.55 * progress
    if thin:
        for k in range(4):
            x0 = int(cx + (relative + 0.11 * k - 0.15) * bw)
            bar = max(2, int(4 + 10 * progress))
            cv2.rectangle(frame, (x0, 0), (x0 + bar, HEIGHT), (55, 55, 60), -1)
            cv2.rectangle(truth, (x0, 0), (x0 + bar, HEIGHT), 255, -1)
    else:
        x0 = int(cx + relative * bw)
        post = max(6, int(10 + 42 * progress))
        cv2.rectangle(frame, (x0, 0), (x0 + post, HEIGHT), (45, 45, 48), -1)
        cv2.rectangle(truth, (x0, 0), (x0 + post, HEIGHT), 255, -1)

    if exposure_swing:
        frame = np.clip(frame.astype(np.float32) * (0.8 + 0.5 * progress)
                        + 10 * np.sin(index), 0, 255).astype(np.uint8)
    return frame, quad, truth


def history(count=FRAMES, **kwargs):
    return {i: [tuple(map(float, p)) for p in approach_frame(i, count, **kwargs)[1]]
            for i in range(count)}


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("plates")
    paths = {}
    for name, kwargs in (("print", {}), ("screen", {"screen": True}),
                         ("thin", {"thin": True}),
                         ("swing", {"exposure_swing": True})):
        frames = [approach_frame(i, FRAMES, **kwargs)[0] for i in range(FRAMES)]
        paths[name] = write_video(frames, root / f"{name}.mp4")
    return paths


def rates(mask, quad, truth, margin=15):
    interior = core.quad_to_mask(quad, WIDTH, HEIGHT) > 0
    real = (truth > 0) & interior
    clean = interior & ~cv2.dilate(truth, np.ones((margin, margin),
                                                  np.uint8)).astype(bool)
    found = (mask > 127)[real].mean() * 100 if real.sum() else None
    false = (mask > 127)[clean].mean() * 100 if clean.sum() else 0.0
    return found, false


class TestLearningThePlate:
    def test_the_median_sees_through_a_sweeping_obstruction(self, clips):
        """The post crosses the panel during the shot, so no canonical pixel
        is covered in most samples -- the learned plate is the artwork, not
        the artwork with a ghost of the post through it."""
        plate = core.build_surface_plate(clips["print"], history())
        assert plate is not None

        h, w = plate.plate.shape[:2]
        reference = cv2.resize(STATIC_ARTWORK, (w, h)).astype(np.float32)
        difference = np.abs(plate.plate - reference).mean()
        assert difference < 12, f"plate is {difference:.1f} grey levels off the artwork"

    def test_a_screen_playing_content_is_refused(self, clips):
        """The one footage this cue must never touch: every frame disagrees
        with any plate, and accepting one would eat the creative everywhere.
        Falling back to depth is the correct outcome, so build returns None."""
        assert core.build_surface_plate(clips["screen"], history(screen=True)) is None

    def test_an_exposure_swing_does_not_get_it_refused(self, clips):
        """Whole-frame brightness drifting over the shot is the footage's
        ordinary condition, not a moving picture; the gain fit absorbs it."""
        assert core.build_surface_plate(clips["swing"],
                                        history(exposure_swing=True)) is not None

    def test_it_is_deterministic(self, clips):
        """The render rebuilds the plate independently of the preview; they
        must land on identical numbers or the file differs from the screen."""
        first = core.build_surface_plate(clips["print"], history())
        again = core.build_surface_plate(clips["print"], dict(history()))
        assert np.array_equal(first.plate, again.plate)

    def test_too_little_tracking_yields_none(self, clips):
        assert core.build_surface_plate(clips["print"], {}) is None
        assert core.build_surface_plate(clips["print"],
                                        {0: history()[0]}) is None

    def test_an_unreadable_video_yields_none(self):
        assert core.build_surface_plate("no-such-file.mp4", history()) is None


class TestFindingObstructions:
    def test_the_post_is_found_across_the_whole_approach(self, clips):
        """Including 18-45% of frame width, where the depth model finds 0-7%.
        Through codec compression the worst frame measured 53% (a dark post
        over equally dark artwork -- exactly where a depth step is large, so
        the union covers it); assert a little under the measurements."""
        plate = core.build_surface_plate(clips["print"], history())
        capture = cv2.VideoCapture(clips["print"])
        found_rates, false_rates = [], []
        for index in (0, 6, 12, 18, 24, 31):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            assert ok
            _, quad, truth = approach_frame(index)
            found, false = rates(plate.mask(frame, quad), quad, truth, margin=25)
            found_rates.append(found)
            false_rates.append(false)
        capture.release()

        assert min(found_rates) > 45, f"worst frame only {min(found_rates):.0f}%"
        assert np.mean(found_rates) > 70, f"mean {np.mean(found_rates):.0f}%"
        assert max(false_rates) < 3, f"false masking reached {max(false_rates):.1f}%"
        # The depth model's blind stretch specifically: sampled indices 6 and
        # 12 sit at 28% and 47% of frame width.
        assert found_rates[1] > 60 and found_rates[2] > 60

    def test_thin_bars_are_found(self, clips):
        """Bars of 2-14 px: below what a 256-wide depth network can ever see."""
        plate = core.build_surface_plate(clips["thin"], history(thin=True))
        capture = cv2.VideoCapture(clips["thin"])
        found_rates, false_rates = [], []
        for index in (6, 12, 18, 24, 31):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            assert ok
            _, quad, truth = approach_frame(index, thin=True)
            found, false = rates(plate.mask(frame, quad), quad, truth)
            found_rates.append(found)
            false_rates.append(false)
        capture.release()
        assert min(found_rates) > 40
        assert np.mean(found_rates) > 55
        assert max(false_rates) < 3

    def test_tracked_corner_error_is_survived(self, clips):
        """Corners come from the tracker, good to about a pixel. The
        difference is taken as the best over one-pixel shifts of the plate,
        which is what keeps every artwork edge from lighting up."""
        rng = np.random.default_rng(3)
        jittered = {i: [tuple(map(float, p + rng.normal(0, 0.7, 2)))
                        for p in approach_frame(i)[1]]
                    for i in range(FRAMES)}
        plate = core.build_surface_plate(clips["print"], jittered)
        assert plate is not None

        capture = cv2.VideoCapture(clips["print"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, 18)
        ok, frame = capture.read()
        capture.release()
        _, quad, truth = approach_frame(18)
        wobbly = np.float32(jittered[18])
        found, false = rates(plate.mask(frame, wobbly), wobbly, truth, margin=25)
        assert found > 60
        assert false < 15

    def test_the_mask_keeps_the_house_contract(self, clips):
        plate = core.build_surface_plate(clips["print"], history())
        capture = cv2.VideoCapture(clips["print"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, 18)
        ok, frame = capture.read()
        capture.release()
        _, quad, _ = approach_frame(18)
        mask = plate.mask(frame, quad)
        assert mask.shape == frame.shape[:2]
        assert mask.dtype == np.uint8
        # Nothing outside the marked area: the plate knows nothing out there.
        outside = ~cv2.dilate(core.quad_to_mask(quad, WIDTH, HEIGHT),
                              np.ones((21, 21), np.uint8)).astype(bool)
        assert int(mask[outside].sum()) == 0

    def test_a_degenerate_quad_yields_an_empty_mask(self, clips):
        plate = core.build_surface_plate(clips["print"], history())
        frame = np.full((HEIGHT, WIDTH, 3), 100, np.uint8)
        flat = np.float32([[10, 10], [12, 10], [12, 11], [10, 11]])
        assert plate.mask(frame, flat).shape == (HEIGHT, WIDTH)
