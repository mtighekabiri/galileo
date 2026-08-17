"""Photometric matching: making an inserted creative belong to the shot.

A creative pasted in with correct geometry still reads as fake, because it is
*too clean*. The footage around it has been through a lens and a sensor and a
codec: it is unevenly lit, slightly soft, grainy, colour-cast by the ambient
light, and smeared when the camera moves. The insert has none of that.

This module measures those properties from the very pixels the insert is about
to cover and reproduces them on the creative. Each effect is independently
weighted, because how much is right depends on the shot -- and on how much the
research team is willing to alter the creative, which for an ad test is a real
constraint rather than a matter of taste.

Everything here is plain NumPy/OpenCV and testable without a display.
"""

import logging

import cv2
import numpy as np

from galileo_core import as_quad, quad_bounds, to_bgra

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Measuring the footage
# --------------------------------------------------------------------------

def estimate_noise_sigma(image: np.ndarray) -> float:
    """Estimate additive noise, in 0-255 units (Immerkaer's method).

    Convolving with a kernel that annihilates smooth and linear content leaves
    mostly noise, so the mean absolute response scales with the noise level.
    Robust on textured footage, where a plain variance would mostly measure the
    texture instead.
    """
    if image is None or image.size == 0:
        return 0.0
    gray = _gray(image).astype(np.float32)
    if min(gray.shape[:2]) < 3:
        return 0.0

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    response = cv2.filter2D(gray, cv2.CV_32F, kernel)
    # 0.797 = sqrt(pi/2) / 2, converting mean-absolute to a sigma for the
    # kernel's gain of 6.
    return float(np.mean(np.abs(response)) * 0.797 / 6.0)


def estimate_detail(image: np.ndarray) -> float:
    """High-frequency energy, used to compare sharpness between two images."""
    if image is None or image.size == 0:
        return 0.0
    gray = _gray(image).astype(np.float32)
    if min(gray.shape[:2]) < 3:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def base_in_creative_space(base_bgr: np.ndarray, quad, size) -> np.ndarray:
    """Warp the covered patch of footage into the creative's own frame.

    Rectifying the region first means every later comparison is pixel-aligned
    with the creative, so a lighting gradient measured at the top of the
    billboard lands on the top of the creative rather than wherever perspective
    happened to put it.
    """
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return None
    destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    try:
        matrix = cv2.getPerspectiveTransform(as_quad(quad), destination)
    except cv2.error:
        return None
    return cv2.warpPerspective(base_bgr, matrix, (width, height),
                               flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# The individual effects
# --------------------------------------------------------------------------

def match_lighting(creative_bgr: np.ndarray, reference_bgr: np.ndarray,
                   strength: float = 1.0, sigma_fraction: float = 0.12) -> np.ndarray:
    """Carry the surface's *uneven* illumination onto the creative.

    A real billboard is rarely lit flatly: it falls off towards one side, has a
    shadow across a corner, catches a highlight from a shopfront. Only the very
    low frequencies of the covered region describe that, so the reference is
    blurred hard and reduced to a gain field around its own mean. Multiplying
    the creative by it transfers the lighting shape while leaving the creative's
    own artwork untouched.
    """
    if strength <= 0 or creative_bgr is None or reference_bgr is None:
        return creative_bgr

    reference_gray = _gray(reference_bgr).astype(np.float32)
    height, width = reference_gray.shape[:2]
    sigma = max(2.0, sigma_fraction * max(height, width))
    ksize = int(sigma * 4) | 1
    low_frequency = cv2.GaussianBlur(reference_gray, (ksize, ksize), sigma)

    mean = float(low_frequency.mean())
    if mean < 1e-3:
        return creative_bgr
    gain = low_frequency / mean

    # A gain field free to run to zero would punch black holes in the creative.
    gain = np.clip(gain, 0.35, 2.2)
    gain = 1.0 + (gain - 1.0) * float(strength)

    adjusted = creative_bgr.astype(np.float32) * gain[:, :, None]
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def match_colour(creative_bgr: np.ndarray, reference_bgr: np.ndarray,
                 strength: float = 1.0, sigma_fraction: float = 0.16) -> np.ndarray:
    """Pull the creative towards the ambient colour of what it covers.

    The pull is *local* rather than one global shift: both images are reduced
    to heavily blurred low-frequency colour fields and the creative is moved
    by the difference between them. A surface lit warm by a shopfront on one
    side and cool daylight on the other then gets each cast where it belongs,
    instead of the whole insert tinted with the average of the two.

    Only those low frequencies move -- the creative's own detail and contrast
    ride on top untouched. Matching the spread as a full Reinhard transfer
    does will happily crush a punchy creative's contrast to match a dull
    wall, which for an advertising test destroys the thing being tested.
    """
    if strength <= 0 or creative_bgr is None or reference_bgr is None:
        return creative_bgr
    if creative_bgr.size == 0 or reference_bgr.size == 0:
        return creative_bgr

    source = cv2.cvtColor(creative_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(
        reference_bgr[:, :, :3] if reference_bgr.shape[-1] == 4 else reference_bgr,
        cv2.COLOR_BGR2LAB).astype(np.float32)

    height, width = source.shape[:2]
    if min(height, width) < 4 or min(target.shape[:2]) < 4:
        # Too small to carry a field; the plain global shift is all there is.
        shift = (target.reshape(-1, 3).mean(axis=0)
                 - source.reshape(-1, 3).mean(axis=0))
        source += shift[None, None, :] * float(strength)
        return cv2.cvtColor(np.clip(source, 0, 255).astype(np.uint8),
                            cv2.COLOR_LAB2BGR)

    # The reference is normally already rectified into the creative's own
    # frame (base_in_creative_space); align it if a caller hands in another
    # size. The heavy blur below swallows any resampling artefacts.
    if target.shape[:2] != source.shape[:2]:
        target = cv2.resize(target, (width, height),
                            interpolation=cv2.INTER_LINEAR)

    sigma = max(2.0, sigma_fraction * max(height, width))
    ksize = int(sigma * 4) | 1
    low_source = cv2.GaussianBlur(source, (ksize, ksize), sigma)
    low_target = cv2.GaussianBlur(target, (ksize, ksize), sigma)

    shift = (low_target - low_source) * float(strength)
    # A shift free to run away would repaint the creative wherever the
    # covered surface differs sharply from it; cap it at a strong cast.
    source += np.clip(shift, -64.0, 64.0)
    return cv2.cvtColor(np.clip(source, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def match_sharpness(creative_bgr: np.ndarray, reference_bgr: np.ndarray,
                    strength: float = 1.0,
                    candidates=(0.0, 0.4, 0.7, 1.0, 1.4, 1.9, 2.5, 3.2)) -> np.ndarray:
    """Soften a too-crisp creative to the focus of the surface it replaces.

    A screen slightly out of focus, or simply far away, carries much less fine
    detail than a creative supplied at full resolution. The mismatch reads as a
    sticker on the picture. This searches a short ladder of blur radii for the
    one whose high-frequency energy best matches the reference. A search rather
    than a formula because the relationship depends on the content.

    It only ever blurs: sharpening an insert to match crisper footage would
    invent detail that was never in the creative.
    """
    if strength <= 0 or creative_bgr is None or reference_bgr is None:
        return creative_bgr

    target_detail = estimate_detail(reference_bgr)
    if target_detail <= 0:
        return creative_bgr
    if estimate_detail(creative_bgr) <= target_detail:
        return creative_bgr           # already as soft as the footage

    best, best_error = 0.0, None
    for sigma in candidates:
        trial = (creative_bgr if sigma <= 0 else
                 cv2.GaussianBlur(creative_bgr, (0, 0), sigma))
        error = abs(estimate_detail(trial) - target_detail)
        if best_error is None or error < best_error:
            best, best_error = sigma, error

    sigma = best * float(np.clip(strength, 0.0, 1.0))
    if sigma <= 0.05:
        return creative_bgr
    return cv2.GaussianBlur(creative_bgr, (0, 0), sigma)


def motion_blur(image: np.ndarray, dx: float, dy: float,
                strength: float = 1.0) -> np.ndarray:
    """Smear the creative along the direction the surface is travelling.

    When the camera pans, everything in shot is smeared by the exposure. An
    insert that stays razor sharp while the frame around it blurs is one of the
    clearer giveaways in handheld footage.
    """
    length = float(np.hypot(dx, dy)) * float(strength)
    if image is None or length < 1.0:
        return image

    size = int(min(length, 41)) | 1
    if size < 3:
        return image

    kernel = np.zeros((size, size), np.float32)
    centre = size // 2
    angle = np.arctan2(dy, dx)
    for step in range(size):
        offset = step - centre
        x = int(round(centre + offset * np.cos(angle)))
        y = int(round(centre + offset * np.sin(angle)))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    total = kernel.sum()
    if total <= 0:
        return image
    kernel /= total

    if image.ndim == 3 and image.shape[2] == 4:
        # Blur the alpha alongside the colour, or the smear stops dead at the
        # creative's original silhouette.
        return cv2.filter2D(image, -1, kernel)
    return cv2.filter2D(image, -1, kernel)


def add_grain(frame_bgr: np.ndarray, sigma: float, mask: np.ndarray = None,
              seed: int = None) -> np.ndarray:
    """Add sensor/compression noise so the insert is not conspicuously clean.

    Applied in frame space, after the warp: grain added before would be
    resampled along with the creative and end up the wrong size for the shot.
    """
    if frame_bgr is None or sigma <= 0.1:
        return frame_bgr

    rng = np.random.default_rng(seed)

    if mask is None:
        noise = rng.normal(0.0, float(sigma),
                           frame_bgr.shape[:2]).astype(np.float32)
        out = frame_bgr.astype(np.float32) + noise[:, :, None]
        return np.clip(out, 0, 255).astype(np.uint8)

    # The noise is zero wherever the mask is, so the work is confined to the
    # mask's bounding box: a small insert costs its own area, not a full-
    # frame noise field and a full-frame widen to float.
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return frame_bgr

    noise = rng.normal(0.0, float(sigma), (h, w)).astype(np.float32)
    noise *= mask[y:y + h, x:x + w].astype(np.float32) / 255.0
    patch = frame_bgr[y:y + h, x:x + w].astype(np.float32) + noise[:, :, None]
    out = frame_bgr.copy()
    out[y:y + h, x:x + w] = np.clip(patch, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------

class BlendSettings:
    """How hard to push each photometric effect. 0 disables one outright.

    The defaults are deliberately lopsided, and the reason is the use case
    rather than taste. Lighting, grain, softness and motion blur are what sell
    an insert as part of the shot, and none of them touch the creative's hue.
    Colour matching does, and the creative's colour is frequently the very thing
    an ad test is measuring -- pull it hard towards a teal-cast scene and the
    red in the brand's logo is no longer the red under test. Measured on a
    dim, cool-cast test scene, a colour strength of 0.5 moved the creative's
    mean colour by 84 units and turned white type cyan; 0.15 moved it by 27,
    nearly all of that from lighting rather than hue.

    So colour starts low and the rest start high. Raise ``colour`` when the
    insert has to sit in strongly tinted light and the exact hue matters less
    than belonging in the frame.
    """

    __slots__ = ("lighting", "colour", "sharpness", "grain", "motion",
                 "highlights", "enabled")

    def __init__(self, lighting=0.7, colour=0.15, sharpness=0.8, grain=0.9,
                 motion=0.5, highlights=0.0, enabled=True):
        self.lighting = lighting
        self.colour = colour
        self.sharpness = sharpness
        self.grain = grain
        self.motion = motion
        self.highlights = highlights
        self.enabled = enabled

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, data) -> "BlendSettings":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__slots__}
        return cls(**known)

    @classmethod
    def off(cls) -> "BlendSettings":
        return cls(0, 0, 0, 0, 0, 0, enabled=False)

    def __repr__(self):
        return ("BlendSettings(" +
                ", ".join(f"{n}={getattr(self, n)}" for n in self.__slots__) + ")")


def preserve_highlights(creative_bgr: np.ndarray, reference_bgr: np.ndarray,
                        strength: float = 1.0, threshold: int = 235) -> np.ndarray:
    """Let blown-out highlights on the original surface show through.

    A glass screen or a gloss poster throws a specular reflection that belongs
    to the glass, not to what is behind it -- so it should survive a change of
    advert. Off by default: on a matte printed billboard there is no such
    reflection and this would only bleach the creative.
    """
    if strength <= 0 or creative_bgr is None or reference_bgr is None:
        return creative_bgr

    reference = (reference_bgr[:, :, :3] if reference_bgr.shape[-1] == 4
                 else reference_bgr)
    gray = _gray(reference).astype(np.float32)
    highlight = np.clip((gray - threshold) / max(1.0, 255.0 - threshold), 0, 1)
    if not highlight.any():
        return creative_bgr

    highlight = (highlight * float(strength))[:, :, None]
    blended = (creative_bgr.astype(np.float32) * (1.0 - highlight)
               + reference.astype(np.float32) * highlight)
    return np.clip(blended, 0, 255).astype(np.uint8)


def photometric_match(creative, base_bgr: np.ndarray, quad,
                      settings: BlendSettings = None,
                      motion=None) -> np.ndarray:
    """Adapt a creative to the footage it is about to be composited into.

    Returns the creative in its own space, still BGRA, with its alpha intact --
    ready to hand to ``composite_region`` exactly as before.

    Args:
        creative: the BGR/BGRA creative.
        base_bgr: the frame it will be inserted into.
        quad: where it is going, in frame coordinates.
        settings: per-effect strengths.
        motion: ``(dx, dy)`` the surface moved since the previous frame, in
            pixels, used for motion blur.
    """
    if creative is None or base_bgr is None:
        return creative
    settings = settings or BlendSettings()
    if not settings.enabled:
        return creative

    creative = to_bgra(creative)
    colour, alpha = creative[:, :, :3].copy(), creative[:, :, 3]

    height, width = creative.shape[:2]
    reference = base_in_creative_space(base_bgr, quad, (width, height))
    if reference is None:
        return creative

    if settings.colour > 0:
        colour = match_colour(colour, reference, settings.colour)
    if settings.lighting > 0:
        colour = match_lighting(colour, reference, settings.lighting)
    if settings.sharpness > 0:
        colour = match_sharpness(colour, reference, settings.sharpness)
    if settings.highlights > 0:
        colour = preserve_highlights(colour, reference, settings.highlights)

    result = np.dstack([colour, alpha])

    if settings.motion > 0 and motion is not None:
        # Motion is measured in frame pixels; express it in the creative's own
        # scale so a small insert is not smeared as if it were full-frame.
        scale = _creative_scale(quad, width, height)
        result = motion_blur(result, motion[0] * scale, motion[1] * scale,
                             settings.motion)

    return result


def _creative_scale(quad, width, height) -> float:
    """How many creative pixels correspond to one frame pixel."""
    corners = as_quad(quad)
    quad_width = (np.linalg.norm(corners[1] - corners[0])
                  + np.linalg.norm(corners[2] - corners[3])) / 2.0
    quad_height = (np.linalg.norm(corners[3] - corners[0])
                   + np.linalg.norm(corners[2] - corners[1])) / 2.0
    if quad_width < 1e-6 or quad_height < 1e-6:
        return 1.0
    return float(np.clip((width / quad_width + height / quad_height) / 2.0,
                         0.05, 20.0))


def grain_sigma_for(base_bgr: np.ndarray, quad, settings: BlendSettings = None) -> float:
    """How much grain to add over the insert, measured from the footage itself."""
    settings = settings or BlendSettings()
    if not settings.enabled or settings.grain <= 0:
        return 0.0

    height, width = base_bgr.shape[:2]
    box = quad_bounds(quad, width, height)
    if box is None:
        return 0.0
    x0, y0, x1, y1 = box
    region = base_bgr[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return estimate_noise_sigma(region) * float(settings.grain)
