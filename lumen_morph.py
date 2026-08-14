"""Giving the creative a little shape of its own.

A creative goes into the tracked area as a flat rectangle. That is right for a
poster pasted flat on a board, but plenty of real placements are not: a banner
sagging on a fence, a print bowed across a curved panel, an advert on a screen
angled a few degrees away from the one the tracking found.

This bends and tilts the creative *in its own canvas*, before it is warped into
the area. Keeping it separate matters: the tracked quad describes where the
surface is, and should not be nudged about to fake the look of the artwork on
it. Tracking data stays honest, and the same morph can be carried to a
different placement or a different clip.

Two effects, applied in the order someone would do them by hand:

* **Bow** curves the sheet, as though it were wrapped on a cylinder.
* **Tilt** turns it in space -- pitch, yaw and roll -- and projects it through a
  pinhole camera, so the foreshortening is real perspective rather than a
  squash.

Both are exact and cheap. Tilt is a homography; bow is separable, so its
inverse is two one-dimensional lookups rather than a numerical solve.
"""

import logging

import cv2
import numpy as np

from lumen_core import to_bgra

logger = logging.getLogger(__name__)

# How far away the creative is viewed from, in half-widths. This is the only
# free number in the bow: an arc's depth follows from its own curvature, so how
# strongly the near part is magnified depends on nothing else but this. Closer
# than about 2.5 the near edge overtakes its neighbour and the sheet folds.
VIEW_DISTANCE = 4.0


class Morph:
    """How much to bend and tilt a creative. All zero means leave it alone."""

    __slots__ = ("pitch", "yaw", "roll", "bow_h", "bow_v", "perspective",
                 "enabled")

    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0, bow_h=0.0, bow_v=0.0,
                 perspective=0.5, enabled=True):
        """
        Args:
            pitch: degrees to tip the top away from the viewer.
            yaw: degrees to turn the right edge away from the viewer.
            roll: degrees to rotate in the plane.
            bow_h: -1..1, curve about the vertical axis. Positive bulges the
                middle towards the viewer, as a convex panel does.
            bow_v: -1..1, curve about the horizontal axis.
            perspective: 0..1, how much foreshortening a tilt produces. At 0 a
                tilt is an orthographic squash; at 1 the near edge grows and the
                far edge shrinks noticeably.
        """
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.bow_h = bow_h
        self.bow_v = bow_v
        self.perspective = perspective
        self.enabled = enabled

    def is_identity(self, tolerance: float = 1e-4) -> bool:
        """True if this would leave the creative untouched."""
        if not self.enabled:
            return True
        return all(abs(getattr(self, name)) < tolerance
                   for name in ("pitch", "yaw", "roll", "bow_h", "bow_v"))

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}

    @classmethod
    def from_dict(cls, data) -> "Morph":
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})

    def copy(self) -> "Morph":
        return Morph.from_dict(self.to_dict())

    def __repr__(self):
        return ("Morph(" + ", ".join(f"{n}={getattr(self, n)}"
                                     for n in self.__slots__) + ")")


# --------------------------------------------------------------------------
# Bow
# --------------------------------------------------------------------------

def _bow_profile(count: int, amount: float, samples: int = 2048):
    """Where each output position samples from, along one axis.

    Wrapping a sheet on a cylinder and viewing it head-on compresses it towards
    the edges, because those parts have turned away. The forward mapping is
    monotonic, so it can be inverted with a single interpolation rather than a
    solve -- which is the whole reason bow is done as two separate axes.
    """
    if abs(amount) < 1e-6:
        return np.linspace(0.0, count - 1.0, count, dtype=np.float32)

    angle = np.clip(abs(amount), 0.0, 1.0) * 1.15    # radians of half-arc
    source = np.linspace(-1.0, 1.0, samples)

    # The sheet, wrapped on a cylinder of unit half-width and viewed head-on.
    # Sideways position is sign-blind -- a convex and a concave arc of the same
    # curvature cover exactly the same width -- so what separates them is depth,
    # and only the pinhole divide below makes them look different. It is a
    # genuinely subtle difference on real footage, not a strong one.
    across = np.sin(source * angle) / np.sin(angle)
    towards = np.sign(amount) * (np.cos(source * angle) - np.cos(angle)) / np.sin(angle)
    projected = across * VIEW_DISTANCE / (VIEW_DISTANCE - towards)

    # The ends sit at depth zero either way, so they land on +/-1 exactly and
    # the creative keeps filling its canvas however hard it is bowed.
    # projected must increase for the inverse lookup below to be valid; the
    # sort guards VIEW_DISTANCE being lowered past the point where it does.
    order = np.argsort(projected)
    projected, source = projected[order], source[order]

    wanted = np.linspace(-1.0, 1.0, count)
    found = np.interp(wanted, projected, source)
    return ((found + 1.0) * 0.5 * (count - 1.0)).astype(np.float32)


def apply_bow(image: np.ndarray, bow_h: float = 0.0, bow_v: float = 0.0) -> np.ndarray:
    """Curve the creative as though wrapped on a cylinder."""
    if image is None or (abs(bow_h) < 1e-6 and abs(bow_v) < 1e-6):
        return image

    height, width = image.shape[:2]
    columns = _bow_profile(width, bow_h)
    rows = _bow_profile(height, bow_v)
    map_x = np.tile(columns, (height, 1))
    map_y = np.repeat(rows[:, None], width, axis=1)

    return cv2.remap(image, map_x, map_y.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# --------------------------------------------------------------------------
# Tilt
# --------------------------------------------------------------------------

def tilt_corners(width: int, height: int, pitch: float, yaw: float, roll: float,
                 perspective: float = 0.5) -> np.ndarray:
    """Where the creative's corners land after turning it in space.

    The sheet is placed in front of a pinhole camera, rotated, projected, then
    scaled to sit back inside its own canvas -- so a tilt changes the shape
    without changing how much room the creative takes up.
    """
    half_w, half_h = width / 2.0, height / 2.0
    corners = np.float32([[-half_w, -half_h, 0], [half_w, -half_h, 0],
                          [half_w, half_h, 0], [-half_w, half_h, 0]])

    # Image coordinates run y-downwards, which reverses the handedness of the
    # rotation. Negating pitch and yaw restores the usual meaning: a positive
    # angle turns that edge *away* from the viewer, as it does in any 3D tool.
    # Roll is left alone -- clockwise-for-positive is what a screen rotation
    # means everywhere else in the app.
    a, b, c = np.deg2rad([-pitch, -yaw, roll])
    rx = np.float32([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.float32([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.float32([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    rotated = corners @ (rz @ ry @ rx).T

    # A long focal length flattens perspective towards an orthographic squash;
    # a short one exaggerates it.
    strength = float(np.clip(perspective, 0.0, 1.0))
    focal = max(width, height) * (12.0 - 10.5 * strength)
    distance = focal
    depth = rotated[:, 2] + distance
    depth = np.where(np.abs(depth) < 1e-3, 1e-3, depth)
    projected = rotated[:, :2] * (focal / depth)[:, None]

    # Scale back to fit the canvas, keeping the centre put.
    extent = np.abs(projected).max(axis=0)
    limit = np.array([half_w, half_h], np.float32)
    fit = float(np.min(limit / np.maximum(extent, 1e-6)))
    projected = projected * min(fit, 1.0)

    return (projected + np.float32([half_w, half_h])).astype(np.float32)


def apply_tilt(image: np.ndarray, pitch: float = 0.0, yaw: float = 0.0,
               roll: float = 0.0, perspective: float = 0.5) -> np.ndarray:
    """Turn the creative in space, keeping it inside its own canvas."""
    if image is None or (abs(pitch) < 1e-6 and abs(yaw) < 1e-6 and abs(roll) < 1e-6):
        return image

    height, width = image.shape[:2]
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = tilt_corners(width, height, pitch, yaw, roll, perspective)

    try:
        matrix = cv2.getPerspectiveTransform(source, destination)
    except cv2.error:
        logger.debug("tilt skipped: degenerate corners %s", destination.tolist())
        return image

    return cv2.warpPerspective(image, matrix, (width, height),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))


# --------------------------------------------------------------------------
# Both together
# --------------------------------------------------------------------------

def apply_morph(creative, morph: Morph) -> np.ndarray:
    """Bend and tilt a creative, returning BGRA with transparent surround.

    Whatever the morph pushes outside the canvas becomes transparent rather
    than black, so the compositor's alpha handling leaves the footage showing
    through exactly as it does around a creative's own transparency.
    """
    if creative is None or morph is None or morph.is_identity():
        return creative

    result = to_bgra(creative)
    result = apply_bow(result, morph.bow_h, morph.bow_v)
    result = apply_tilt(result, morph.pitch, morph.yaw, morph.roll,
                        morph.perspective)
    return result
