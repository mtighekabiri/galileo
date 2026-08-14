"""Qt-free core for the Galileo Insertion Tool.

Everything in here is plain NumPy/OpenCV so it can be exercised headlessly,
without a display or an event loop. The GUI in ``Galileo_Insertion_Tool_1.0.0.py``
is a thin layer over these functions, which means the preview and the render
run the *same* tracking and compositing code and cannot drift apart.

Conventions used throughout:

* A *quad* is four ``(x, y)`` corners in the base video's pixel coordinates,
  as an array of shape ``(4, 2)``. Corner order is the order the user clicked,
  and it is never silently reordered -- the creative's top-left is mapped to
  the first corner, so click order is how the user controls orientation.
* Images are OpenCV ``uint8`` arrays, BGR or BGRA.
"""

import logging
import os
import shutil
import subprocess

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Quad geometry
# --------------------------------------------------------------------------

def as_quad(points) -> np.ndarray:
    """Coerce any 4-corner representation to a ``(4, 2)`` float32 array."""
    quad = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if quad.shape[0] != 4:
        raise ValueError(f"expected 4 corners, got {quad.shape[0]}")
    return quad


def quad_area(points) -> float:
    """Absolute area of the quad via the shoelace formula."""
    quad = as_quad(points)
    x, y = quad[:, 0], quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _segments_intersect(p1, p2, p3, p4) -> bool:
    """True if open segments p1p2 and p3p4 cross."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def is_simple_quad(points) -> bool:
    """True if the quad does not self-intersect (i.e. is not a bowtie).

    A bowtie quad has no well-defined interior, so both the perspective warp
    and the coverage mask become meaningless. The UI uses this to warn before
    the user renders something that cannot look right.
    """
    q = as_quad(points)
    # Only the two pairs of non-adjacent edges can cross.
    return not (_segments_intersect(q[0], q[1], q[2], q[3])
                or _segments_intersect(q[1], q[2], q[3], q[0]))


def is_valid_quad(points, min_area: float = 4.0) -> bool:
    """True if the quad is finite, non-degenerate and usable for a warp."""
    try:
        q = as_quad(points)
    except (ValueError, TypeError):
        return False
    if not np.all(np.isfinite(q)):
        return False
    if quad_area(q) < min_area:
        return False
    return is_simple_quad(q)


def quad_bounds(points, width: int, height: int, pad: int = 1):
    """Integer bounding box of the quad, clipped to the frame.

    Returns ``(x0, y0, x1, y1)`` as a half-open box, or ``None`` when the quad
    lies entirely outside the frame.
    """
    q = as_quad(points)
    x0 = int(np.floor(q[:, 0].min())) - pad
    y0 = int(np.floor(q[:, 1].min())) - pad
    x1 = int(np.ceil(q[:, 0].max())) + pad
    y1 = int(np.ceil(q[:, 1].max())) + pad

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def quad_to_mask(points, width: int, height: int, feather: float = 0.0,
                 offset=(0, 0)) -> np.ndarray:
    """Anti-aliased coverage mask for a quad.

    Uses ``fillPoly`` rather than ``fillConvexPoly`` so concave quads are
    filled correctly, and fills with sub-pixel precision so the overlay edge
    does not crawl by a whole pixel as the quad moves.
    """
    shift = 3
    scale = 1 << shift
    q = as_quad(points) - np.asarray(offset, dtype=np.float32)
    pts = np.round(q * scale).astype(np.int32).reshape(-1, 1, 2)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA, shift=shift)

    if feather > 0:
        ksize = max(3, int(feather * 4) | 1)  # odd kernel, ~4 sigma wide
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)
    return mask


# --------------------------------------------------------------------------
# Colour adjustment
# --------------------------------------------------------------------------

def apply_brightness_contrast(img: np.ndarray, brightness: float = 0.0,
                              contrast: float = 1.0) -> np.ndarray:
    """Scale/offset the colour channels, leaving any alpha channel untouched."""
    if img is None:
        return img
    if brightness == 0 and contrast == 1.0:
        return img

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = cv2.convertScaleAbs(img[:, :, :3], alpha=contrast, beta=brightness)
        return np.dstack([bgr, img[:, :, 3]])
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def _lab_mean_std(lab: np.ndarray):
    l, a, b = cv2.split(lab)
    return (l.mean(), a.mean(), b.mean()), (l.std(), a.std(), b.std())


def color_transfer(source_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Shift ``source`` to carry ``target``'s colour statistics (Reinhard).

    This is what makes an inserted creative sit in the scene's lighting rather
    than looking pasted on: the mean and spread of each L*a*b* channel are
    matched to the region of the base frame the creative will cover.
    """
    if source_bgr is None or target_bgr is None:
        return source_bgr
    if source_bgr.size == 0 or target_bgr.size == 0:
        return source_bgr

    src_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    s_means, s_stds = _lab_mean_std(src_lab)
    t_means, t_stds = _lab_mean_std(tgt_lab)

    for c in range(3):
        src_lab[..., c] = ((src_lab[..., c] - s_means[c])
                           * (t_stds[c] / (s_stds[c] + 1e-6))
                           + t_means[c])

    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_colourise(overlay_bgra: np.ndarray, base_bgr: np.ndarray,
                    quad, strength: float = 1.0) -> np.ndarray:
    """Colour-match the creative to the base pixels it is about to cover."""
    if overlay_bgra is None or base_bgr is None or strength <= 0:
        return overlay_bgra

    h, w = base_bgr.shape[:2]
    box = quad_bounds(quad, w, h)
    if box is None:
        return overlay_bgra
    x0, y0, x1, y1 = box
    region = base_bgr[y0:y1, x0:x1]
    if region.size == 0:
        return overlay_bgra

    has_alpha = overlay_bgra.ndim == 3 and overlay_bgra.shape[2] == 4
    bgr = overlay_bgra[:, :, :3] if has_alpha else overlay_bgra
    transferred = color_transfer(bgr, region)

    if strength < 1.0:
        transferred = cv2.addWeighted(transferred, strength, bgr, 1.0 - strength, 0)

    if has_alpha:
        return np.dstack([transferred, overlay_bgra[:, :, 3]])
    return transferred


# --------------------------------------------------------------------------
# Compositing
# --------------------------------------------------------------------------

def to_bgra(img: np.ndarray) -> np.ndarray:
    """Normalise a BGR/BGRA/grayscale image to 4-channel BGRA."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    if img.shape[2] == 4:
        return img
    raise ValueError(f"unsupported image with {img.shape[2]} channels")


def load_image_bgra(path: str) -> np.ndarray:
    """Read an image from disk preserving its alpha channel."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"could not read image: {path}")
    return to_bgra(img)


def _apply_occlusion(alpha: np.ndarray, occlusion, box) -> np.ndarray:
    """Hold the insert back wherever something is in front of the surface.

    ``occlusion`` is a frame-sized mask where 255 means "this pixel belongs to
    something nearer than the surface". Scaling the creative's alpha by its
    inverse is what lets a passer-by cross in front of the insert instead of
    having it painted over them.
    """
    if occlusion is None:
        return alpha
    x0, y0, x1, y1 = box
    patch = occlusion[y0:y1, x0:x1]
    if patch.shape[:2] != alpha.shape[:2]:
        patch = cv2.resize(patch, (alpha.shape[1], alpha.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
    return alpha * (1.0 - patch.astype(np.float32) / 255.0)


def composite_overlay(base_bgr: np.ndarray, overlay: np.ndarray, quad,
                      opacity: float = 1.0, feather: float = 0.0,
                      in_place: bool = False, occlusion=None) -> np.ndarray:
    """Warp ``overlay`` into ``quad`` on ``base_bgr`` with real alpha blending.

    The creative's own alpha channel is honoured and multiplied by an
    anti-aliased quad mask, so transparent PNGs stay transparent and edges do
    not stair-step. Work is confined to the quad's bounding box, so cost
    scales with the size of the insert rather than the size of the frame.

    Pass ``occlusion`` (a frame-sized mask, 255 where something is in front of
    the surface) to have the insert render behind it.

    Returns the composited frame (a copy unless ``in_place`` is set).
    """
    if base_bgr is None or overlay is None:
        return base_bgr
    if not is_valid_quad(quad):
        logger.debug("composite skipped: degenerate quad %s", np.asarray(quad).tolist())
        return base_bgr

    out = base_bgr if in_place else base_bgr.copy()
    h, w = out.shape[:2]

    box = quad_bounds(quad, w, h)
    if box is None:
        return out  # entirely off-screen
    x0, y0, x1, y1 = box
    roi_w, roi_h = x1 - x0, y1 - y0

    overlay = to_bgra(overlay)
    oh, ow = overlay.shape[:2]

    src = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
    dst = as_quad(quad)
    try:
        # Exact for a 4-point correspondence; findHomography would fit the
        # same 4 points with a RANSAC loop it has no use for.
        matrix = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        logger.debug("composite skipped: could not solve perspective transform")
        return out

    # Fold the ROI translation into the warp so we only rasterise the box.
    translate = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    warped = cv2.warpPerspective(
        overlay, translate @ matrix, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0))

    mask = quad_to_mask(quad, roi_w, roi_h, feather=feather, offset=(x0, y0))

    alpha = (warped[:, :, 3].astype(np.float32) / 255.0)
    alpha *= mask.astype(np.float32) / 255.0
    if opacity < 1.0:
        alpha *= float(max(0.0, opacity))
    alpha = _apply_occlusion(alpha, occlusion, (x0, y0, x1, y1))
    if not alpha.any():
        return out

    alpha = alpha[:, :, None]
    roi = out[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# Curved insertion regions
# --------------------------------------------------------------------------

UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


class Region:
    """An insertion area: four corners, each edge optionally curved.

    Not every surface worth inserting into is flat. Curved TVs, cylindrical
    pillars, wrapped billboards and banners all need the creative to bend, and
    a straight-edged quad visibly cuts the corners on them.

    Each edge carries two cubic Bezier control points. Their *default*
    positions are exactly one and two thirds along the straight edge, where a
    cubic Bezier reproduces the straight line -- so a region whose controls are
    untouched behaves identically to a plain quad, and ``curved`` can be
    toggled without the insert shifting.

    Curvature is applied as a displacement on top of the perspective mapping
    rather than replacing it, so foreshortening is preserved: a curved insert
    still compresses towards the far edge the way a flat one does.
    """

    __slots__ = ("corners", "curvature", "curved")

    def __init__(self, corners, curvature=None, curved: bool = False):
        """
        Args:
            corners: the four corners, in click order.
            curvature: ``(4, 2, 2)`` bend coefficients -- per edge, per control
                point, as a multiple of the edge vector and of its
                perpendicular. Dimensionless on purpose: expressing the bend in
                the edge's own frame means it rotates and scales with the
                surface as the camera moves, instead of staying pinned to
                screen axes and sliding off a rotating shot.
            curved: whether curvature is applied at all.
        """
        self.corners = as_quad(corners)
        self.curvature = (np.zeros((4, 2, 2), np.float32) if curvature is None
                          else np.asarray(curvature, np.float32).reshape(4, 2, 2))
        self.curved = bool(curved)

    # -- construction ------------------------------------------------------

    @staticmethod
    def default_controls(corners) -> np.ndarray:
        """Control points at the 1/3 and 2/3 marks, i.e. perfectly straight."""
        q = as_quad(corners)
        controls = np.zeros((4, 2, 2), np.float32)
        for i in range(4):
            p0, p1 = q[i], q[(i + 1) % 4]
            controls[i, 0] = p0 + (p1 - p0) / 3.0
            controls[i, 1] = p0 + 2.0 * (p1 - p0) / 3.0
        return controls

    @staticmethod
    def edge_basis(corners):
        """Per-edge ``(edge_vector, perpendicular)``, the frame bends live in."""
        q = as_quad(corners)
        edges = np.stack([q[(i + 1) % 4] - q[i] for i in range(4)])
        normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        return edges, normals

    def copy(self) -> "Region":
        return Region(self.corners.copy(), self.curvature.copy(), self.curved)

    def with_corners(self, corners) -> "Region":
        """The same bend, on new corners.

        Because curvature is held in each edge's own frame, moving, rotating or
        scaling the quad carries the curve along with no conversion at all.
        """
        return Region(corners, self.curvature.copy(), self.curved)

    def scaled(self, factor: float) -> "Region":
        """A copy at a different resolution, for rendering at reduced scale."""
        return Region(self.corners * float(factor), self.curvature.copy(),
                      self.curved)

    # -- geometry ----------------------------------------------------------

    @property
    def controls(self) -> np.ndarray:
        """Absolute Bezier control points implied by the curvature."""
        return self.default_controls(self.corners) + self.control_offsets()

    def control_offsets(self) -> np.ndarray:
        """Control-point displacements from the straight edge, in pixels."""
        if not self.curved:
            return np.zeros((4, 2, 2), np.float32)
        edges, normals = self.edge_basis(self.corners)
        along = self.curvature[:, :, 0:1] * edges[:, None, :]
        across = self.curvature[:, :, 1:2] * normals[:, None, :]
        return (along + across).astype(np.float32)

    def set_control_point(self, edge: int, slot: int, point) -> None:
        """Place one Bezier handle, converting it into the edge's own frame."""
        target = np.asarray(point, np.float32).reshape(2)
        default = self.default_controls(self.corners)[edge, slot]
        edges, normals = self.edge_basis(self.corners)
        basis = np.stack([edges[edge], normals[edge]], axis=1)
        try:
            self.curvature[edge, slot] = np.linalg.solve(
                basis.astype(np.float64), (target - default).astype(np.float64))
        except np.linalg.LinAlgError:
            logger.debug("degenerate edge %d; control point ignored", edge)

    def is_straight(self, tolerance: float = 1e-6) -> bool:
        """True if this region is (or is switched to behave as) a flat quad."""
        if not self.curved:
            return True
        return bool(np.all(np.abs(self.curvature) < tolerance))

    def homography(self) -> np.ndarray:
        """Unit square -> corners, the perspective part of the mapping."""
        return cv2.getPerspectiveTransform(UNIT_SQUARE, self.corners)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "corners": [[float(x), float(y)] for x, y in self.corners],
            "curvature": [[[float(a), float(b)] for a, b in edge]
                          for edge in self.curvature],
            "curved": self.curved,
        }

    @classmethod
    def from_dict(cls, data) -> "Region":
        if isinstance(data, (list, tuple)):   # a bare quad, from older files
            return cls(data)

        region = cls(data["corners"], data.get("curvature"),
                     data.get("curved", False))
        # Older files stored absolute control points; convert them in.
        if "curvature" not in data and data.get("controls") is not None:
            controls = np.asarray(data["controls"], np.float32).reshape(4, 2, 2)
            region.curved = True
            for edge in range(4):
                for slot in range(2):
                    region.set_control_point(edge, slot, controls[edge, slot])
            region.curved = bool(data.get("curved", False))
        return region

    def __repr__(self):
        return f"Region(curved={self.curved}, corners={self.corners.tolist()})"


def _apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project points of shape ``(..., 2)`` through a 3x3 matrix."""
    shape = points.shape[:-1]
    flat = points.reshape(-1, 2).astype(np.float64)
    homogeneous = np.concatenate([flat, np.ones((len(flat), 1))], axis=1) @ matrix.T
    w = homogeneous[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return (homogeneous[:, :2] / w).reshape(*shape, 2)


def _edge_deviation(offsets: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Displacement of one Bezier edge from its chord at parameter ``t``.

    With control points written as chord position plus offset, the cubic's
    straight-line part cancels exactly and only the Bernstein-weighted offsets
    remain -- which is why untouched controls give zero deviation.
    """
    t = t[..., None]
    return 3.0 * (1.0 - t) ** 2 * t * offsets[0] + 3.0 * (1.0 - t) * t ** 2 * offsets[1]


def _edge_deviation_gradient(offsets: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d/dt of :func:`_edge_deviation`, for the Newton solve."""
    t = t[..., None]
    return (3.0 * (1.0 - t) * (1.0 - 3.0 * t) * offsets[0]
            + 3.0 * t * (2.0 - 3.0 * t) * offsets[1])


def _region_deviation(offsets: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Coons interpolation of the four edge deviations across the interior.

    The usual Coons bilinear correction term drops out because every corner
    deviation is zero by construction, leaving a plain sum of the edge terms.
    """
    d_top = _edge_deviation(offsets[0], u)
    d_right = _edge_deviation(offsets[1], v)
    d_bottom = _edge_deviation(offsets[2], 1.0 - u)
    d_left = _edge_deviation(offsets[3], 1.0 - v)
    return (((1.0 - v)[..., None] * d_top) + (v[..., None] * d_bottom)
            + ((1.0 - u)[..., None] * d_left) + (u[..., None] * d_right))


def region_forward(region: Region, u, v) -> np.ndarray:
    """Map surface coordinates ``(u, v)`` in [0,1]^2 to frame pixels."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    base = _apply_homography(region.homography().astype(np.float64),
                             np.stack([u, v], axis=-1))
    if region.is_straight():
        return base
    return base + _region_deviation(region.control_offsets().astype(np.float64), u, v)


def region_boundary(region: Region, samples: int = 24) -> np.ndarray:
    """Trace the region's outline as a polygon, going round the four edges.

    The GUI draws this and the compositor masks with it, so what the user sees
    while dragging a control point is exactly the shape that gets rendered.
    """
    if region.is_straight():
        return region.corners.copy()

    t = np.linspace(0.0, 1.0, max(2, samples), endpoint=False)
    zeros, ones = np.zeros_like(t), np.ones_like(t)
    edges = [
        region_forward(region, t, zeros),          # top:    u 0->1 at v=0
        region_forward(region, ones, t),           # right:  v 0->1 at u=1
        region_forward(region, 1.0 - t, ones),     # bottom: u 1->0 at v=1
        region_forward(region, zeros, 1.0 - t),    # left:   v 1->0 at u=0
    ]
    return np.concatenate(edges, axis=0).astype(np.float32)


def _forward_jacobian(matrix: np.ndarray, offsets: np.ndarray,
                      u: np.ndarray, v: np.ndarray, projected: np.ndarray):
    """Partial derivatives of the forward map, as ``(j00, j01, j10, j11)``.

    Column one is d(x, y)/du and column two is d(x, y)/dv, combining the
    homography's own derivative with the curvature displacement's.
    """
    w = matrix[2, 0] * u + matrix[2, 1] * v + matrix[2, 2]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    x, y = projected[..., 0], projected[..., 1]

    j00 = (matrix[0, 0] - x * matrix[2, 0]) / w
    j01 = (matrix[0, 1] - x * matrix[2, 1]) / w
    j10 = (matrix[1, 0] - y * matrix[2, 0]) / w
    j11 = (matrix[1, 1] - y * matrix[2, 1]) / w

    if offsets is not None:
        d_top = _edge_deviation(offsets[0], u)
        d_right = _edge_deviation(offsets[1], v)
        d_bottom = _edge_deviation(offsets[2], 1.0 - u)
        d_left = _edge_deviation(offsets[3], 1.0 - v)
        g_top = _edge_deviation_gradient(offsets[0], u)
        g_right = _edge_deviation_gradient(offsets[1], v)
        g_bottom = _edge_deviation_gradient(offsets[2], 1.0 - u)
        g_left = _edge_deviation_gradient(offsets[3], 1.0 - v)

        dd_du = ((1.0 - v)[..., None] * g_top - v[..., None] * g_bottom
                 - d_left + d_right)
        dd_dv = (-d_top + d_bottom - (1.0 - u)[..., None] * g_left
                 + u[..., None] * g_right)

        j00 = j00 + dd_du[..., 0]
        j10 = j10 + dd_du[..., 1]
        j01 = j01 + dd_dv[..., 0]
        j11 = j11 + dd_dv[..., 1]

    return j00, j01, j10, j11


def region_is_folded(region: Region, samples: int = 16) -> bool:
    """True if the curvature is so extreme the surface turns inside out.

    Bend an edge far enough and it sweeps back across the patch: the mapping
    stops being one-to-one, so a destination pixel no longer corresponds to a
    single point on the creative and there is nothing sensible to draw. This is
    the curved counterpart of a bowtie quad, and it is detected the same way --
    by looking for the map reversing orientation, i.e. its Jacobian
    determinant changing sign across the patch.
    """
    if region.is_straight():
        return False

    matrix = region.homography().astype(np.float64)
    offsets = region.control_offsets().astype(np.float64)
    t = np.linspace(0.0, 1.0, max(4, samples))
    u, v = np.meshgrid(t, t)

    projected = _apply_homography(matrix, np.stack([u, v], axis=-1))
    j00, j01, j10, j11 = _forward_jacobian(matrix, offsets, u, v, projected)
    determinant = j00 * j11 - j01 * j10
    return bool(determinant.min() <= 0 < determinant.max()
                or np.all(determinant <= 0))


def is_valid_region(region: Region) -> bool:
    """True if the region can actually be rendered."""
    if not is_valid_quad(region.corners):
        return False
    return not region_is_folded(region)


def _region_inverse(region: Region, points: np.ndarray,
                    iterations: int = 6) -> np.ndarray:
    """Solve frame pixels back to surface coordinates ``(u, v)``.

    The forward map is a homography plus a curvature displacement and has no
    closed-form inverse, so it is solved numerically. Simply re-substituting
    the displacement converges only while the bend stays shallow and blows up
    on a strongly curved edge, so this uses Newton's method with the analytic
    Jacobian instead: quadratic convergence, and stable however hard the user
    drags a control point.

    The homography-only inverse is an excellent starting guess, which is what
    keeps the iteration count low.
    """
    matrix = region.homography().astype(np.float64)
    inverse_h = np.linalg.inv(matrix)
    uv = _apply_homography(inverse_h, points)
    if region.is_straight():
        return uv

    offsets = region.control_offsets().astype(np.float64)
    target = points.astype(np.float64)

    for _ in range(iterations):
        u, v = uv[..., 0], uv[..., 1]

        projected = _apply_homography(matrix, uv)
        residual = projected + _region_deviation(offsets, u, v) - target

        j00, j01, j10, j11 = _forward_jacobian(matrix, offsets, u, v, projected)
        determinant = j00 * j11 - j01 * j10
        singular = np.abs(determinant) < 1e-12
        determinant = np.where(singular, 1.0, determinant)

        f0, f1 = residual[..., 0], residual[..., 1]
        step_u = (j11 * f0 - j01 * f1) / determinant
        step_v = (-j10 * f0 + j00 * f1) / determinant
        step = np.stack([np.where(singular, 0.0, step_u),
                         np.where(singular, 0.0, step_v)], axis=-1)

        # Keep the iterate in a bounded neighbourhood of the patch. Pixels
        # that wander outside are not in the region and get masked away, but
        # letting them run to infinity would poison the arithmetic.
        uv = np.clip(uv - step, -1.0, 2.0)

    return uv


def composite_region(base_bgr: np.ndarray, overlay: np.ndarray, region: Region,
                     opacity: float = 1.0, feather: float = 0.0,
                     in_place: bool = False, occlusion=None) -> np.ndarray:
    """Composite a creative into a :class:`Region`, curved edges included.

    Straight regions are handed to the plain homography path, which is both
    faster and exact; only genuinely curved ones pay for the iterative solve.
    """
    if region.is_straight():
        return composite_overlay(base_bgr, overlay, region.corners,
                                 opacity=opacity, feather=feather,
                                 in_place=in_place, occlusion=occlusion)

    if base_bgr is None or overlay is None:
        return base_bgr

    out = base_bgr if in_place else base_bgr.copy()
    h, w = out.shape[:2]

    if region_is_folded(region):
        logger.debug("composite skipped: curved region folds over on itself")
        return out

    polygon = region_boundary(region, samples=32)
    if not np.all(np.isfinite(polygon)):
        logger.debug("composite skipped: curved boundary is not finite")
        return out

    box = quad_bounds(polygon[:4], w, h) if len(polygon) == 4 else None
    if box is None:
        xs, ys = polygon[:, 0], polygon[:, 1]
        x0 = max(0, int(np.floor(xs.min())) - 1)
        y0 = max(0, int(np.floor(ys.min())) - 1)
        x1 = min(w, int(np.ceil(xs.max())) + 1)
        y1 = min(h, int(np.ceil(ys.max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return out
        box = (x0, y0, x1, y1)
    x0, y0, x1, y1 = box
    roi_w, roi_h = x1 - x0, y1 - y0

    overlay = to_bgra(overlay)
    oh, ow = overlay.shape[:2]

    grid_x, grid_y = np.meshgrid(np.arange(x0, x1, dtype=np.float64),
                                 np.arange(y0, y1, dtype=np.float64))
    uv = _region_inverse(region, np.stack([grid_x, grid_y], axis=-1))

    map_x = (uv[..., 0] * ow).astype(np.float32)
    map_y = (uv[..., 1] * oh).astype(np.float32)
    warped = cv2.remap(overlay, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    shift = 3
    pts = np.round((polygon - np.float32([x0, y0])) * (1 << shift)).astype(np.int32)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255, lineType=cv2.LINE_AA, shift=shift)
    if feather > 0:
        ksize = max(3, int(feather * 4) | 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    alpha *= mask.astype(np.float32) / 255.0
    if opacity < 1.0:
        alpha *= float(max(0.0, opacity))
    alpha = _apply_occlusion(alpha, occlusion, (x0, y0, x1, y1))
    if not alpha.any():
        return out

    alpha = alpha[:, :, None]
    roi = out[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1.0 - alpha) + warped[:, :, :3].astype(np.float32) * alpha
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7,
)


class TrackResult:
    """Outcome of a single tracking step."""

    __slots__ = ("ok", "quad", "n_features", "n_inliers", "reason",
                 "following_camera")

    def __init__(self, ok, quad=None, n_features=0, n_inliers=0, reason="",
                 following_camera=False):
        self.ok = ok
        self.quad = quad
        self.n_features = n_features
        self.n_inliers = n_inliers
        self.reason = reason
        #: True when the region is out of shot and this step came from the
        #: camera's own motion rather than from the region's texture.
        self.following_camera = following_camera

    def __repr__(self):
        return (f"TrackResult(ok={self.ok}, n_features={self.n_features}, "
                f"n_inliers={self.n_inliers}, "
                f"following_camera={self.following_camera}, "
                f"reason={self.reason!r})")


#: What a levelled frame is scaled to. Any fixed pair would do; these keep the
#: result comfortably inside 0..255 for ordinary footage, so little is clipped.
LEVEL_MIDDLE, LEVEL_SPREAD = 128.0, 48.0


def level_gray(gray: np.ndarray) -> np.ndarray:
    """Grayscale with the overall brightness and contrast taken out.

    Optical flow rests on one assumption: that a point keeps its brightness
    from one frame to the next. A camera panning across a concourse onto a
    bright screen breaks it on every frame, because its automatic exposure is
    hunting the whole time -- which is not an edge case for this tool but the
    ordinary condition of the footage it is given. Measured on a pan with the
    exposure swinging by 40%, the shape ends 10.5px off its surface without
    this and 0.11px off with it; at 70% it is 21.1px against 0.13px. A steady
    shot pays about three hundredths of a pixel.

    Median and spread-about-the-median, rather than mean and standard
    deviation, because the statistics have to describe the *lighting* and not
    the contents. A large bright object crossing the frame -- or the tracked
    panel itself sliding out of it -- moves a mean and a standard deviation
    enough to fake an exposure change and undo the whole point: on a clip that
    pans a panel out of shot, mean and standard deviation cost 7.6px where the
    median leaves the result untouched.
    """
    # Both statistics come from a 256-bin histogram, which makes them exact
    # over the whole frame for one fast pass. Working from a shrunken copy
    # instead is tempting and wrong: whichever pixels it draws change as the
    # scene moves through them, and the statistics wobble by enough to fake
    # the very exposure change this exists to remove -- 3.9px of it on a clip
    # that pans a panel out of shot and back.
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return gray
    middle = float(np.searchsorted(np.cumsum(histogram), total / 2.0))

    levels = np.arange(256, dtype=np.float32)
    distances = np.bincount(np.abs(levels - middle).astype(np.int64),
                            weights=histogram, minlength=256)
    spread = float(np.searchsorted(np.cumsum(distances), total / 2.0)) * 1.4826
    if spread < 1e-3:                      # a flat frame; nothing to level
        return gray

    # Through a lookup table: the transform is a scalar function of an 8-bit
    # value, so it costs 256 evaluations rather than one per pixel. Doing the
    # arithmetic across a 1080p frame instead ran at 14ms, which playback
    # cannot afford on top of the tracking itself.
    table = np.clip((levels - middle) * (LEVEL_SPREAD / spread) + LEVEL_MIDDLE,
                    0, 255).astype(np.uint8)
    return cv2.LUT(gray, table)


class PlanarTracker:
    """Tracks a planar quad through a video by the texture inside it.

    The corners themselves are deliberately *not* tracked. Hand-placed corners
    tend to land on occluding edges and untextured areas -- the worst possible
    features -- so following them directly is what makes an insert slide off
    its surface. Instead this detects strong features across the quad's
    interior, follows those with pyramidal Lucas-Kanade, discards the ones that
    fail a forward-backward consistency check, and fits a RANSAC homography to
    the survivors. The corners are then carried by that homography.

    Because the fit is over many points, a handful of bad tracks are outliers
    that RANSAC rejects, rather than a quarter of the total signal.
    """

    #: Track the texture inside the region. Right for printed billboards,
    #: posters and any surface whose own markings are fixed to it.
    INTERIOR = "interior"
    #: Track a band around the region instead. Necessary for a digital screen:
    #: its picture changes and slides about independently of the panel, so
    #: features detected on it follow the advert being played rather than the
    #: screen playing it. The bezel, wall and fittings around it are what is
    #: actually rigid.
    SURROUND = "surround"
    #: Both, and let RANSAC arbitrate. A reasonable hedge for a static screen,
    #: but the interior points can outvote the surround if the picture moves.
    BOTH = "both"

    def __init__(self, frame, quad,
                 fb_threshold: float = 1.0,
                 ransac_threshold: float = 3.0,
                 min_features: int = 6,
                 reseed_below: int = 40,
                 max_area_ratio: float = 1.8,
                 max_step_fraction: float = 0.25,
                 feature_source: str = INTERIOR,
                 surround_scale: float = 1.9,
                 anchor_interval: int = 8,
                 anchor_min_inliers: int = 12,
                 anchor_min_ratio: float = 0.60,
                 anchor_max_correction: float = 6.0,
                 offscreen_below: float = 0.30,
                 normalise_illumination: bool = True,
                 feature_params: dict = None,
                 lk_params: dict = None):
        """
        Args:
            frame: the frame the quad was drawn on (BGR or already grayscale).
            quad: the four corners on that frame.
            fb_threshold: max forward-backward reprojection error, in pixels.
            ransac_threshold: RANSAC inlier distance for the homography fit.
            min_features: below this many surviving points, the step fails.
            reseed_below: re-detect features once the pool drops under this.
            max_area_ratio: reject a step that scales the quad more than this.
            max_step_fraction: reject a step moving a corner more than this
                fraction of the frame diagonal.
            feature_source: :attr:`INTERIOR`, :attr:`SURROUND` or :attr:`BOTH`.
            surround_scale: how far the surround band reaches out, as a
                multiple of the region's size about its centre.
            anchor_interval: how often, in frames, to re-fit against the anchor
                frame to clear accumulated drift. 0 disables it.
            anchor_min_inliers: fewest inliers for a correction to be trusted.
            anchor_min_ratio: fewest, as a fraction of the anchor's features,
                that must still match. This is what tells a drift fix from a
                bad one: measured across a pan and a 62-degree swing, every
                correction that improved accuracy scored 0.81 or above, and the
                single one that made things worse scored 0.59. The default sits
                just above that, which also keeps the corrections a
                digital-screen shot depends on; anchor_max_correction is a
                second, independent guard, since the harmful correction also
                moved the shape further (1.6 px) than any helpful one (1.2 px).
            anchor_max_correction: reject a correction that moves a corner
                further than this, in pixels -- it is a drift fix, not a
                relocation, and a big jump means the anchor no longer matches.
            offscreen_below: how much of the region must still be in shot for
                its own texture to drive the tracking. Under this fraction the
                camera's motion is followed instead. 0 disables that.
            normalise_illumination: take each frame's own brightness and
                contrast out before following anything, so a camera hunting
                its exposure does not read as movement. See :func:`level_gray`.
        """
        # Read before the first _gray call, which is inside reset().
        self.normalise_illumination = normalise_illumination
        self.feature_source = feature_source
        self.surround_scale = surround_scale
        self.anchor_interval = anchor_interval
        self.anchor_min_inliers = anchor_min_inliers
        self.anchor_min_ratio = anchor_min_ratio
        self.anchor_max_correction = anchor_max_correction
        self.offscreen_below = offscreen_below
        self.fb_threshold = fb_threshold
        self.ransac_threshold = ransac_threshold
        self.min_features = min_features
        self.reseed_below = reseed_below
        self.max_area_ratio = max_area_ratio
        self.max_step_fraction = max_step_fraction
        self.feature_params = dict(FEATURE_PARAMS, **(feature_params or {}))
        self.lk_params = dict(LK_PARAMS, **(lk_params or {}))

        self.reset(frame, quad)

    # -- lifecycle ---------------------------------------------------------

    def reset(self, frame, quad):
        """Re-anchor the tracker, e.g. after the user nudges a corner."""
        self.prev_gray = self._gray(frame)
        self.anchor_quad = as_quad(quad)
        self.quad = self.anchor_quad.copy()
        self.cumulative_h = np.eye(3, dtype=np.float64)
        self.frame_shape = self.prev_gray.shape[:2]
        self._diagonal = float(np.hypot(*self.frame_shape))
        self.features = None
        # Set once the region has left the shot and the camera is being
        # followed in its place; see :meth:`track`.
        self.following_camera = False
        self.camera_frames = 0
        # Recent centroids, for how fast the region was moving across the
        # frame just before it left it, and what is left of that motion once
        # the camera's own is taken out.
        self._centroids = []
        self._offscreen_drift = None
        self._seed_features()

        # Keep the anchor frame and the features found on it, so the current
        # position can periodically be re-derived from it rather than from an
        # ever-lengthening chain of frame-to-frame steps.
        self.anchor_gray = self.prev_gray.copy()
        self.anchor_features = (self.features.copy()
                                if self.features is not None and len(self.features)
                                else None)
        self._since_anchor = 0
        self.anchor_corrections = 0
        self.anchor_ratio = 0.0
        self.anchor_shift = 0.0

    def _gray(self, frame):
        if frame is None:
            raise ValueError("frame is None")
        if frame.ndim == 2:
            gray = frame
        elif frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return level_gray(gray) if self.normalise_illumination else gray

    def _scaled_quad(self, factor: float) -> np.ndarray:
        centre = self.quad.mean(axis=0)
        return ((self.quad - centre) * factor + centre).astype(np.float32)

    #: How many recent frames to average the region's speed over. Long enough
    #: to shrug off per-frame noise, short enough to still describe what it was
    #: doing at the moment it left rather than earlier in the shot.
    VELOCITY_FRAMES = 6

    #: Below this, in pixels per frame, a region is taken to be fixed to the
    #: scene rather than moving through it. What is left over for a billboard
    #: after the camera is accounted for is measurement noise, well under a
    #: pixel, and carrying that for a long stretch off screen walks the shape
    #: away from where it belongs -- measured at 29px across fifty frames.
    #: Anything genuinely moving on its own runs to many pixels a frame, so the
    #: two are never close.
    OFFSCREEN_DRIFT_FLOOR = 2.0

    def _region_velocity(self) -> np.ndarray:
        """How far the region moved across the frame per frame, lately."""
        if len(self._centroids) < 2:
            return np.zeros(2, np.float32)
        span = self._centroids[-1] - self._centroids[0]
        return (span / (len(self._centroids) - 1)).astype(np.float32)

    def visible_fraction(self) -> float:
        """How much of the region is inside the frame, from 0 to 1."""
        h, w = self.frame_shape
        area = quad_area(self.quad)
        if area <= 0:
            return 0.0
        frame_rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        try:
            overlap, _ = cv2.intersectConvexConvex(
                self.quad.astype(np.float32), frame_rect)
        except cv2.error:
            return 0.0
        return float(np.clip(overlap / area, 0.0, 1.0))

    def _feature_mask(self) -> np.ndarray:
        """Where to look for features, according to ``feature_source``."""
        h, w = self.prev_gray.shape[:2]

        if self.following_camera:
            # Anything that moves with the camera will do, except the region
            # itself: a screen or billboard is usually the strongest texture in
            # shot, so left in it takes most of the detector's budget, and then
            # every one of those points leaves the frame together a moment
            # later. RANSAC throws out whatever else is moving on its own --
            # traffic, pedestrians, other screens.
            mask = np.full((h, w), 255, np.uint8)
            cv2.fillPoly(mask, [self._scaled_quad(1.15).astype(np.int32)], 0)
            return mask

        if self.feature_source in (self.INTERIOR, self.BOTH):
            mask = quad_to_mask(self.quad, w, h)
            # Pull in from the boundary: features sitting on the edge of the
            # region are usually on an occluding contour and track badly.
            eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
            if cv2.countNonZero(eroded) > 100:
                mask = eroded
        else:
            mask = np.zeros((h, w), np.uint8)

        if self.feature_source in (self.SURROUND, self.BOTH):
            band = quad_to_mask(self._scaled_quad(self.surround_scale), w, h)
            # Punch out the region itself, plus a small margin, so the band
            # holds only what surrounds it.
            cv2.fillPoly(band, [self._scaled_quad(1.08).astype(np.int32)], 0)
            mask = cv2.bitwise_or(mask, band)

        mask[mask > 0] = 255
        return mask

    def _seed_features(self):
        """Detect trackable features in whichever area is rigid."""
        mask = self._feature_mask()
        pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        self.features = pts if pts is not None else np.empty((0, 1, 2), np.float32)

    # -- stepping ----------------------------------------------------------

    def track(self, frame) -> TrackResult:
        """Advance the quad to ``frame``.

        On success the tracker's state moves forward. On failure the state is
        left untouched, so the caller can coast on prediction and retry on the
        next frame without the tracker having been corrupted.
        """
        gray = self._gray(frame)

        # Once the surface has left the shot there is nothing of it left to
        # measure, and following the region alone simply stops: the quad
        # freezes against the edge of frame and the insert bunches up there
        # instead of carrying on out of shot. During a pan the rest of the
        # scene is still moving with the camera, so that is followed instead,
        # which keeps the region moving at the right rate, turns it round when
        # the camera does, and leaves it in the right place to be picked up
        # again when it comes back into view.
        following = (self.offscreen_below > 0
                     and self.visible_fraction() < self.offscreen_below)
        if following != self.following_camera:
            self.following_camera = following
            self._offscreen_drift = None    # re-measured on the first step
            self._seed_features()
        if following:
            self.camera_frames += 1

        if self.features is None or len(self.features) < self.min_features:
            self._seed_features()
        if len(self.features) < self.min_features:
            return TrackResult(False, following_camera=following,
                               reason="no trackable features in region")

        p0 = self.features.reshape(-1, 1, 2).astype(np.float32)

        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                 **self.lk_params)
        if p1 is None:
            return TrackResult(False, following_camera=following,
                               reason="optical flow returned nothing")

        # Track back again: a point that does not return to where it started
        # was not really tracked, whatever its reported error.
        p0r, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None,
                                                  **self.lk_params)
        if p0r is None:
            return TrackResult(False, following_camera=following,
                               reason="reverse optical flow returned nothing")

        fb_error = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_error < self.fb_threshold)
        n_good = int(good.sum())
        if n_good < 3:
            return TrackResult(False, n_features=n_good, following_camera=following,
                               reason=f"only {n_good} points survived the flow check")

        src = p0.reshape(-1, 2)[good]
        dst = p1.reshape(-1, 2)[good]

        matrix, inlier_mask = self._fit_transform(src, dst)
        if matrix is None:
            return TrackResult(False, n_features=n_good, following_camera=following,
                               reason="could not fit a transform")

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else n_good
        if n_inliers < 3:
            return TrackResult(False, n_features=n_good, n_inliers=n_inliers,
                               following_camera=following,
                               reason="too few inliers")

        cumulative = matrix @ self.cumulative_h
        new_quad = cv2.perspectiveTransform(
            self.anchor_quad.reshape(-1, 1, 2), cumulative.astype(np.float64)
        ).reshape(4, 2).astype(np.float32)

        if following:
            # The camera's motion is not always the whole story. An advert on
            # the side of a bus is leaving the shot under its own steam, and
            # following the camera alone would park it where it was last seen
            # while the bus drives off. Whatever the region was doing over and
            # above the camera before it left is carried on at the same rate --
            # the only honest guess about something that cannot be seen.
            if self._offscreen_drift is None:
                camera_step = new_quad.mean(axis=0) - self.quad.mean(axis=0)
                drift = self._region_velocity() - camera_step
                if np.linalg.norm(drift) < self.OFFSCREEN_DRIFT_FLOOR:
                    drift = np.zeros(2, np.float32)
                self._offscreen_drift = drift
            shift = np.eye(3, dtype=np.float64)
            shift[0, 2], shift[1, 2] = self._offscreen_drift
            cumulative = shift @ cumulative
            new_quad = cv2.perspectiveTransform(
                self.anchor_quad.reshape(-1, 1, 2), cumulative
            ).reshape(4, 2).astype(np.float32)

        rejection = self._reject_reason(new_quad)
        if rejection:
            return TrackResult(False, n_features=n_good, n_inliers=n_inliers,
                               following_camera=following, reason=rejection)

        # Commit.
        self.cumulative_h = cumulative
        self.quad = new_quad
        self.prev_gray = gray
        if not following:
            # Only positions measured from the region itself are worth
            # averaging: once the camera is being followed the centroid moves
            # by construction, and feeding that back would compound its own
            # guess.
            self._centroids.append(new_quad.mean(axis=0))
            del self._centroids[:-self.VELOCITY_FRAMES]
        kept = dst[inlier_mask.ravel().astype(bool)] if inlier_mask is not None else dst
        self.features = kept.reshape(-1, 1, 2).astype(np.float32)
        if len(self.features) < self.reseed_below:
            self._seed_features()

        self._since_anchor += 1
        # The anchor is a view of the region, so there is nothing for it to
        # match against while the region is out of shot. Its guards would
        # refuse the result anyway; skipping saves the work and removes any
        # chance of a fluke match relocating the shape while it is off screen.
        if (self.anchor_interval and not following
                and self._since_anchor >= self.anchor_interval):
            self._since_anchor = 0
            self._correct_against_anchor(gray)

        return TrackResult(True, quad=self.quad.copy(), n_features=n_good,
                           n_inliers=n_inliers, following_camera=following)

    def _correct_against_anchor(self, gray) -> bool:
        """Re-derive the position from the anchor frame, clearing drift.

        Composing a homography per frame means every step's small error is
        carried forward for the rest of the clip, so the shape slides steadily
        off the surface during a long camera move -- measured at about 0.019 px
        per frame, roughly 0.28% of the distance travelled, which is several
        pixels across a long pan and exactly the "it tracks but it is offset"
        complaint.

        Matching the anchor frame straight to the current one removes the chain
        entirely: the error stops depending on how long tracking has been
        running. The current estimate seeds the search, so the large
        displacement between anchor and now is not a problem, and the result is
        only accepted if it is well supported and close to where incremental
        tracking already thinks the surface is -- this corrects drift, it does
        not relocate the shape.
        """
        if self.anchor_features is None or len(self.anchor_features) < 4:
            return False

        source = self.anchor_features.reshape(-1, 1, 2).astype(np.float32)
        guess = cv2.perspectiveTransform(
            source, self.cumulative_h.astype(np.float64)).astype(np.float32)

        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.anchor_gray, gray, source, guess.copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params)
        if moved is None:
            return False

        # The return trip needs a starting guess as well. Without one it has to
        # find points that may be most of a frame away, which plain LK cannot
        # do, so every point failed the consistency check and no correction
        # was ever accepted after the first few frames.
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.anchor_gray, moved, source.copy(),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params)
        if back is None:
            return False

        error = np.linalg.norm((source - back).reshape(-1, 2), axis=1)
        good = ((status.ravel() == 1) & (status_back.ravel() == 1)
                & (error < self.fb_threshold * 2.0))
        if int(good.sum()) < self.anchor_min_inliers:
            return False

        matrix, mask = cv2.findHomography(
            source.reshape(-1, 2)[good], moved.reshape(-1, 2)[good],
            cv2.RANSAC, self.ransac_threshold)
        if matrix is None or mask is None:
            return False

        # How much of the anchor still matches. This falls as the viewpoint
        # diverges from the anchor, which is precisely when a correction stops
        # being more accurate than the incremental chain.
        self.anchor_ratio = float(mask.sum()) / max(1, len(source))
        if int(mask.sum()) < self.anchor_min_inliers:
            return False
        if self.anchor_ratio < self.anchor_min_ratio:
            logger.debug("anchor correction rejected: only %.0f%% of the anchor "
                         "still matches", self.anchor_ratio * 100)
            return False

        corrected = cv2.perspectiveTransform(
            self.anchor_quad.reshape(-1, 1, 2), matrix.astype(np.float64)
        ).reshape(4, 2).astype(np.float32)

        if not np.all(np.isfinite(corrected)) or not is_simple_quad(corrected):
            return False
        shift = float(np.linalg.norm(corrected - self.quad, axis=1).max())
        self.anchor_shift = shift
        if shift > self.anchor_max_correction:
            # Too far to be drift. The anchor's appearance has probably gone,
            # so keep the incremental estimate rather than jumping.
            logger.debug("anchor correction rejected: %.1fpx shift", shift)
            return False

        self.cumulative_h = matrix.astype(np.float64)
        self.quad = corrected
        self.anchor_corrections += 1
        return True

    def map_from_anchor(self, points) -> np.ndarray:
        """Carry points defined on the anchor frame into the current frame.

        Used for anything pinned to the tracked surface but not part of the
        quad itself -- most importantly the Bezier control points of a curved
        region, which must ride along with the plane they curve over.
        """
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.empty((0, 2), np.float32)
        return cv2.perspectiveTransform(
            pts, self.cumulative_h.astype(np.float64)
        ).reshape(-1, 2).astype(np.float32)

    def _fit_transform(self, src, dst):
        """Fit src->dst, preferring a homography but degrading gracefully.

        A full homography needs a healthy number of well-spread points to be
        stable. With only a few survivors, a 4-degree-of-freedom similarity is
        far better conditioned than an 8-degree-of-freedom projective fit that
        would happily invent extreme perspective from noise.
        """
        if len(src) >= 10:
            matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC,
                                              self.ransac_threshold)
            if matrix is not None:
                return matrix.astype(np.float64), mask
        if len(src) >= 3:
            affine, mask = cv2.estimateAffine2D(
                src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold)
            if affine is not None:
                matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
                return matrix.astype(np.float64), mask
        return None, None

    def _reject_reason(self, new_quad):
        """Why this step should be thrown away, or '' to accept it."""
        if not np.all(np.isfinite(new_quad)):
            return "non-finite corners"
        if not is_simple_quad(new_quad):
            return "quad folded over on itself"

        old_area, new_area = quad_area(self.quad), quad_area(new_quad)
        if new_area < 4.0:
            return "quad collapsed"
        if old_area > 0:
            ratio = new_area / old_area
            if ratio > self.max_area_ratio or ratio < 1.0 / self.max_area_ratio:
                return f"implausible area change ({ratio:.2f}x in one frame)"

        step = np.linalg.norm(new_quad - self.quad, axis=1).max()
        if step > self.max_step_fraction * self._diagonal:
            return f"implausible corner jump ({step:.0f}px in one frame)"
        return ""


# --------------------------------------------------------------------------
# Corner smoothing
# --------------------------------------------------------------------------

class ScreenVerdict:
    """Whether the picture inside a region belongs to the region."""

    __slots__ = ("independent", "worst_gap", "exceeded", "checked")

    def __init__(self, independent, worst_gap=0.0, exceeded=0, checked=0):
        #: True if the inside is playing its own picture rather than carrying
        #: markings fixed to the surface.
        self.independent = bool(independent)
        #: Furthest the two readings disagreed, in pixels.
        self.worst_gap = float(worst_gap)
        self.exceeded = int(exceeded)
        self.checked = int(checked)

    def __repr__(self):
        return (f"ScreenVerdict(independent={self.independent}, "
                f"worst_gap={self.worst_gap:.2f}, "
                f"exceeded={self.exceeded}/{self.checked})")


def looks_like_a_playing_screen(frames, quads, threshold: float = 2.0,
                                samples: int = 12) -> ScreenVerdict:
    """Decide whether a marked area is a screen showing its own picture.

    This is the single most damaging thing that can go wrong with a marked
    area, and the one the user is least able to see coming. Following the
    texture inside a digital screen follows the advert being played rather
    than the panel playing it, and the error is not a wobble -- measured at
    242px by the end of a short clip, with the tracker reporting no failure at
    all, so the export is confidently wrong rather than visibly broken.

    Telling the two apart needs no model of what a screen is. Read the motion
    twice, once from inside the area and once from the surface around it. For
    markings fixed to a panel the two agree to a hundredth of a pixel; for a
    picture running on its own they do not.

    The worst disagreement decides it, not the average: a screen changing
    slides every few seconds agrees perfectly in between and disagrees wildly
    at each change, so an average buries exactly the frames that matter.

    Frames must already be levelled for exposure, which :class:`PlanarTracker`
    does; without it a camera hunting its exposure reads as disagreement and
    an ordinary poster is condemned as a screen.
    """
    gaps = []
    for index in range(min(samples, len(frames) - 1)):
        readings = []
        for source in (PlanarTracker.INTERIOR, PlanarTracker.SURROUND):
            tracker = PlanarTracker(frames[index], quads[index],
                                    feature_source=source, anchor_interval=0)
            if not tracker.track(frames[index + 1]).ok:
                break
            readings.append(tracker.quad)
        if len(readings) == 2:
            gaps.append(float(np.linalg.norm(readings[0] - readings[1],
                                             axis=1).mean()))

    if not gaps:
        return ScreenVerdict(False)
    gaps = np.array(gaps)
    exceeded = int((gaps > threshold).sum())
    return ScreenVerdict(exceeded >= 1, gaps.max(), exceeded, len(gaps))


class SimpleKalmanFilter:
    """Constant-velocity 2D filter used to smooth one tracked corner.

    Measurement noise is set low relative to process noise because the
    measurement now comes from a many-point RANSAC homography rather than a
    single optical-flow point -- it deserves to be trusted. Over-smoothing here
    shows up as the insert lagging behind fast camera moves.
    """

    def __init__(self, initial_x, initial_y, process_noise=0.05,
                 measurement_noise=1.0):
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)
        self.transition_matrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float64)
        self.measurement_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], dtype=np.float64)
        self.process_noise = np.eye(4, dtype=np.float64) * process_noise
        self.measurement_noise = np.eye(2, dtype=np.float64) * measurement_noise
        self.error_covariance = np.eye(4, dtype=np.float64)

    def predict(self):
        self.state = self.transition_matrix @ self.state
        self.error_covariance = (self.transition_matrix @ self.error_covariance
                                 @ self.transition_matrix.T + self.process_noise)
        return self.state[0:2].copy()

    def update(self, measurement):
        measurement = np.asarray(measurement, dtype=np.float64).reshape(2)
        innovation_cov = (self.measurement_matrix @ self.error_covariance
                          @ self.measurement_matrix.T + self.measurement_noise)
        gain = (self.error_covariance @ self.measurement_matrix.T
                @ np.linalg.inv(innovation_cov))
        self.state = self.state + gain @ (measurement - self.measurement_matrix @ self.state)
        self.error_covariance = ((np.eye(4) - gain @ self.measurement_matrix)
                                 @ self.error_covariance)
        return self.state[0:2].copy()


def smooth_quad(filters, quad=None):
    """Run one predict/update cycle over four per-corner filters.

    Pass ``quad`` to fold in a fresh measurement, or ``None`` when tracking
    failed, in which case the filters coast on their prediction and the insert
    keeps moving sensibly through a brief occlusion.
    """
    predicted = [f.predict() for f in filters]
    if quad is None:
        return np.array(predicted, dtype=np.float32)
    measured = as_quad(quad)
    return np.array([f.update(measured[i]) for i, f in enumerate(filters)],
                    dtype=np.float32)


def make_filters(quad, **kwargs):
    """One :class:`SimpleKalmanFilter` per corner of ``quad``."""
    q = as_quad(quad)
    return [SimpleKalmanFilter(float(x), float(y), **kwargs) for x, y in q]


# --------------------------------------------------------------------------
# Locating a target from a reference image
# --------------------------------------------------------------------------

class Detection:
    """Where a reference image was found in a frame."""

    __slots__ = ("quad", "confidence", "n_inliers", "n_matches", "frame_index")

    def __init__(self, quad, confidence, n_inliers, n_matches, frame_index=None):
        self.quad = quad
        self.confidence = confidence
        self.n_inliers = n_inliers
        self.n_matches = n_matches
        self.frame_index = frame_index

    def to_region(self, curved: bool = False) -> Region:
        return Region(self.quad, None, curved)

    def __repr__(self):
        return (f"Detection(frame={self.frame_index}, "
                f"confidence={self.confidence:.2f}, inliers={self.n_inliers})")


def _quads_overlap(a, b, threshold: float = 0.4) -> bool:
    """True if two quads cover much the same ground."""
    corners = np.vstack([as_quad(a), as_quad(b)])
    offset = corners.min(axis=0)
    size = np.ceil(corners.max(axis=0) - offset).astype(int) + 2
    if size.min() <= 0:
        return False
    width, height = int(size[0]), int(size[1])
    if width * height > 4_000_000:      # too big to rasterise; fall back
        return False

    mask_a = quad_to_mask(as_quad(a) - offset, width, height) > 0
    mask_b = quad_to_mask(as_quad(b) - offset, width, height) > 0
    intersection = float(np.count_nonzero(mask_a & mask_b))
    smaller = float(min(np.count_nonzero(mask_a), np.count_nonzero(mask_b)))
    if smaller <= 0:
        return False
    return intersection / smaller > threshold


def _drop_overlapping(detections, threshold: float = 0.4) -> list:
    """Keep the strongest of any detections describing the same panel."""
    ordered = sorted(detections, key=lambda d: d.n_inliers, reverse=True)
    kept = []
    for detection in ordered:
        if any(_quads_overlap(detection.quad, other.quad, threshold)
               for other in kept):
            continue
        kept.append(detection)
    return kept


class ReferenceMatcher:
    """Find a picture of a target inside a video frame.

    Give it a photo or screenshot of the thing you want to insert onto -- a
    billboard, a TV screen, a poster, a shop sign -- and it locates that
    surface in the footage and returns its four corners, ready to hand
    straight to :class:`PlanarTracker`. This replaces clicking the corners by
    hand and, unlike a hand-drawn quad, it lands on the target's true corners
    rather than wherever the click happened to fall.

    It works by matching scale- and rotation-invariant keypoints between the
    reference and the frame, then fitting a RANSAC homography to the matches.
    That homography is what maps the reference's own corners into the frame.

    This finds *planar, textured* targets, which is exactly the class of thing
    this tool inserts into. It is not a face or object detector: a face is
    neither planar nor rigid, and matching one photo of a face against a moving
    one is unreliable. Faces need a dedicated detector, not this.
    """

    def __init__(self, reference, detector: str = "auto", ratio: float = 0.75,
                 min_inliers: int = 12, ransac_threshold: float = 5.0,
                 max_dimension: int = 1000):
        """
        Args:
            reference: the target image (BGR/BGRA array, or a path).
            detector: ``"sift"``, ``"orb"``, or ``"auto"`` to prefer SIFT.
            ratio: Lowe ratio-test threshold; lower keeps only surer matches.
            min_inliers: fewest RANSAC inliers that counts as a real find.
            ransac_threshold: inlier reprojection distance, in pixels.
            max_dimension: reference images are downscaled past this, since
                detail far beyond the video's own resolution only costs time.
        """
        if isinstance(reference, str):
            reference = load_image_bgra(reference)
        if reference is None:
            raise ValueError("reference image is required")

        self.ratio = ratio
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold

        gray = self._gray(reference)
        scale = min(1.0, max_dimension / max(gray.shape[:2]))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        self.reference_gray = gray
        ref_h, ref_w = gray.shape[:2]
        self.reference_corners = np.float32(
            [[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]])

        self.detector_name, self.detector, self.matcher = self._make_detector(detector)
        self.keypoints, self.descriptors = self.detector.detectAndCompute(gray, None)
        if self.descriptors is None or len(self.keypoints) < 4:
            raise ValueError(
                "the reference image has too little detail to match against; "
                "try a sharper or more textured picture of the target")

    @staticmethod
    def _gray(img):
        if img.ndim == 2:
            return img
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _make_detector(name):
        name = (name or "auto").lower()
        if name in ("auto", "sift") and hasattr(cv2, "SIFT_create"):
            return "sift", cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
        if name == "sift":
            raise ValueError("SIFT is not available in this OpenCV build")
        return "orb", cv2.ORB_create(nfeatures=5000), cv2.BFMatcher(cv2.NORM_HAMMING)

    # -- matching ----------------------------------------------------------

    def _good_matches(self, gray):
        """Ratio-tested matches between the reference and one frame."""
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            return [], []
        try:
            knn = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        except cv2.error:
            return [], []

        # Lowe's ratio test: keep a match only when the best candidate is
        # clearly better than the runner-up, which throws out the ambiguous
        # matches that repetitive texture generates.
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        return good, keypoints

    def _fit(self, matches, keypoints, shape, frame_index=None,
             min_inliers=None):
        """Fit one homography to these matches. Returns (detection, inliers)."""
        min_inliers = self.min_inliers if min_inliers is None else min_inliers
        if len(matches) < max(4, min_inliers // 2):
            return None, None

        src = np.float32([self.keypoints[m.queryIdx].pt
                          for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt
                          for m in matches]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold)
        if matrix is None or mask is None:
            return None, None

        inliers = mask.ravel().astype(bool)
        n_inliers = int(inliers.sum())
        if n_inliers < min_inliers:
            return None, None

        quad = cv2.perspectiveTransform(
            self.reference_corners.reshape(-1, 1, 2), matrix).reshape(4, 2)
        if not is_valid_quad(quad, min_area=16.0):
            return None, None

        # A homography fitted to noise loves to produce a sliver stretching
        # off the frame; require something plausibly on screen.
        h, w = shape[:2]
        if quad_area(quad) > 4.0 * w * h:
            return None, None

        detection = Detection(quad.astype(np.float32),
                              n_inliers / float(len(matches)),
                              n_inliers, len(matches), frame_index)
        return detection, inliers

    def locate(self, frame, frame_index: int = None) -> Detection:
        """Find the reference in one frame, or return None."""
        gray = self._gray(frame)
        matches, keypoints = self._good_matches(gray)
        if not matches:
            return None
        detection, _ = self._fit(matches, keypoints, gray.shape, frame_index)
        return detection

    def locate_all(self, frame, max_results: int = 8, frame_index: int = None,
                   min_inliers: int = None) -> list:
        """Find *every* instance of the target in one frame.

        A concourse often carries the same poster on several panels, and a
        single homography can only describe one of them. This fits one, removes
        the keypoints it consumed along with anything else lying inside the
        quad it found, and fits again, until nothing further stands up. The
        results come back strongest first.

        The bar is set a little lower than for a single search, because the
        further instances of a target are usually the smaller, more distant
        ones and carry fewer keypoints. Measured on a frame holding the same
        poster at three sizes, a threshold of 12 found two of them and 8 found
        all three, with no false positives on unrelated footage or against a
        different reference.
        """
        if min_inliers is None:
            min_inliers = max(6, self.min_inliers * 2 // 3)

        gray = self._gray(frame)
        matches, keypoints = self._good_matches(gray)
        if not matches:
            return []

        detections = []
        remaining = list(matches)
        while len(detections) < max_results:
            detection, inliers = self._fit(remaining, keypoints, gray.shape,
                                           frame_index, min_inliers=min_inliers)
            if detection is None:
                break
            detections.append(detection)

            polygon = detection.quad.astype(np.float32)
            survivors = []
            for index, match in enumerate(remaining):
                if inliers is not None and index < len(inliers) and inliers[index]:
                    continue        # consumed by the instance just found
                point = keypoints[match.trainIdx].pt
                if cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])),
                                        False) >= 0:
                    continue        # lies inside it, so belongs to it
                survivors.append(match)

            if len(survivors) == len(remaining):
                break               # made no progress; stop rather than loop
            remaining = survivors

        return _drop_overlapping(detections)

    def scan_video(self, path: str, start: int = 0, max_frames: int = None,
                   step: int = 5, min_confidence: float = 0.3,
                   progress=None) -> Detection:
        """Search a video for the target and return the best sighting.

        The target may not be on screen at the start, so this samples forward
        through the footage rather than giving up on frame one. It stops early
        on a confident hit; otherwise it reports the best it saw, so a marginal
        find can still be offered to the user rather than silently discarded.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"could not open video: {path}")

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            best = None
            index = start
            examined = 0
            step = max(1, int(step))

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if (index - start) % step == 0:
                    detection = self.locate(frame, frame_index=index)
                    if detection is not None:
                        if best is None or detection.n_inliers > best.n_inliers:
                            best = detection
                        if detection.confidence >= min_confidence:
                            return detection
                    examined += 1
                    if progress is not None and total:
                        progress(min(1.0, (index - start) / max(1, total - start)))
                    if max_frames is not None and examined >= max_frames:
                        break
                index += 1
            return best
        finally:
            cap.release()


# --------------------------------------------------------------------------
# Occlusion: letting people pass in front of the insert
# --------------------------------------------------------------------------

MODEL_DIR_NAME = "models"
_MODEL_SEARCH_DIRS = []


def register_model_dir(path: str) -> None:
    """Add a directory to search for bundled model files."""
    if path and os.path.isdir(path) and path not in _MODEL_SEARCH_DIRS:
        _MODEL_SEARCH_DIRS.insert(0, path)


def find_model(filename: str) -> str:
    """Locate a model file beside the app, in ./models, or next to this file."""
    candidates = list(_MODEL_SEARCH_DIRS)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [os.path.join(here, MODEL_DIR_NAME), here, os.getcwd(),
                   os.path.join(os.getcwd(), MODEL_DIR_NAME)]
    for directory in candidates:
        full = os.path.join(directory, filename)
        if os.path.isfile(full):
            return full
    return None


class PersonSegmenter:
    """Finds the people in a frame, so an insert can be drawn behind them.

    In the environments this tool is pointed at -- airports, malls, high
    streets -- somebody is forever walking between the camera and the screen.
    Painting the creative straight over them is the single most obvious tell
    that a shot has been altered, which matters rather a lot when the footage
    is a research stimulus.

    Motion-based approaches cannot help here: a digital screen's own picture
    also moves independently of the panel, so anything keyed on "this does not
    move with the surface" flags the advert being replaced as though it were a
    passer-by. Segmenting people directly sidesteps that.

    Runs a PP-HumanSeg ONNX model through OpenCV's own DNN module, so no extra
    runtime is needed and the model is a single 6 MB file. See fetch_model.py.
    """

    MODEL_FILENAME = "human_segmentation_pphumanseg_2023mar.onnx"
    INPUT_SIZE = (192, 192)
    PERSON_CHANNEL = 1

    def __init__(self, model_path: str = None, threshold: float = 0.5,
                 dilate: int = 3, feather: float = 2.0, pad: float = 0.35):
        """
        Args:
            model_path: the ONNX file; discovered automatically when omitted.
            threshold: probability above which a pixel counts as a person.
            dilate: grow the mask by this many pixels, covering the soft
                boundary the network leaves around hair and shoulders.
            feather: blur the mask edge by this sigma so the insert does not
                cut against a person with a hard, obviously-composited line.
            pad: how far beyond the region to run the model, as a fraction of
                the region's size.
        """
        path = model_path or find_model(self.MODEL_FILENAME)
        if not path:
            raise FileNotFoundError(
                f"Could not find {self.MODEL_FILENAME}. Run fetch_model.py to "
                "download it, or place it in a 'models' folder beside the "
                "application.")
        try:
            self.net = cv2.dnn.readNetFromONNX(path)
        except cv2.error as exc:
            raise IOError(f"Could not load the segmentation model: {exc}") from exc

        self.model_path = path
        self.threshold = threshold
        self.dilate = dilate
        self.feather = feather
        self.pad = pad

    @classmethod
    def is_available(cls) -> bool:
        """True if the model file can be found."""
        return find_model(cls.MODEL_FILENAME) is not None

    def _infer(self, patch_bgr: np.ndarray) -> np.ndarray:
        """Person probability for one patch, at the patch's own size."""
        blob = cv2.dnn.blobFromImage(
            patch_bgr, scalefactor=1 / 127.5, size=self.INPUT_SIZE,
            mean=(127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        output = self.net.forward()
        probability = output[0, self.PERSON_CHANNEL]
        return cv2.resize(probability, (patch_bgr.shape[1], patch_bgr.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    def mask(self, frame_bgr: np.ndarray, quad=None) -> np.ndarray:
        """A frame-sized mask, 255 where a person is.

        When ``quad`` is given the model is run on a padded crop around it
        rather than the whole frame. The network's input is a fixed 192x192,
        so cropping first is what keeps a distant pedestrian large enough to
        be segmented at all in high-resolution footage.
        """
        height, width = frame_bgr.shape[:2]
        mask = np.zeros((height, width), np.uint8)

        if quad is not None:
            corners = as_quad(quad)
            centre = corners.mean(axis=0)
            grown = (corners - centre) * (1.0 + 2.0 * self.pad) + centre
            box = quad_bounds(grown, width, height, pad=0)
            if box is None:
                return mask
        else:
            box = (0, 0, width, height)

        x0, y0, x1, y1 = box
        patch = frame_bgr[y0:y1, x0:x1]
        if patch.size == 0:
            return mask

        probability = self._infer(patch)
        patch_mask = (probability > self.threshold).astype(np.uint8) * 255

        if self.dilate > 0:
            kernel = np.ones((self.dilate * 2 + 1,) * 2, np.uint8)
            patch_mask = cv2.dilate(patch_mask, kernel)
        if self.feather > 0:
            ksize = max(3, int(self.feather * 4) | 1)
            patch_mask = cv2.GaussianBlur(patch_mask, (ksize, ksize), self.feather)

        mask[y0:y1, x0:x1] = patch_mask
        return mask


VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".m4v", ".webm")


class VideoReferenceMatcher:
    """Find a target described by a *clip* of it rather than a single still.

    A short pan around a billboard carries far more than one photograph does:
    several angles, several exposures, and a better chance that one of them
    resembles the shot being searched. Frames are sampled across the clip, each
    becomes its own matcher, and a search tries them all and keeps the strongest
    result. Views too plain to match against are skipped rather than failing
    the whole thing.

    **The clip must be framed on the target**, in exactly the way a reference
    photograph has to be cropped to it. Each sampled frame is taken to *be* the
    target, so its whole rectangle is what gets mapped into the footage. Hand
    over a wide shot in which the billboard occupies a corner and the match
    will succeed while the returned quad describes the entire scene: on a test
    clip where the poster filled about a tenth of the frame, the corners came
    back 234 px from the panel despite 91 inliers and a confident fit.

    :meth:`frame_coverage` reports how much of the base frame the result would
    cover, which is how a caller can spot that mistake and say so.
    """

    def __init__(self, path: str, samples: int = 6, **matcher_kwargs):
        frames = self._sample(path, samples)
        if not frames:
            raise IOError(f"Could not read any frames from: {path}")

        self.views = []
        problems = []
        for index, frame in frames:
            try:
                matcher = ReferenceMatcher(frame, **matcher_kwargs)
            except ValueError as exc:
                problems.append(f"frame {index}: {exc}")
                continue
            self.views.append(matcher)

        if not self.views:
            raise ValueError(
                "None of the sampled frames had enough detail to match "
                "against. Try a clip that holds still on the target.\n"
                + "\n".join(problems[:3]))

        self.detector_name = self.views[0].detector_name
        self.reference_path = path

    @staticmethod
    def _sample(path: str, samples: int):
        """Take frames spread across the clip, skipping unreadable ones."""
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise IOError(f"Could not open the reference video: {path}")
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total <= 0:
                # Some containers do not report a length; read what we can.
                collected = []
                while len(collected) < samples:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    collected.append((len(collected), frame))
                return collected

            wanted = np.linspace(0, max(0, total - 1),
                                 min(samples, total)).astype(int)
            collected = []
            for index in sorted(set(int(i) for i in wanted)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok:
                    collected.append((index, frame))
            return collected
        finally:
            capture.release()

    def locate(self, frame, frame_index: int = None) -> Detection:
        """The best sighting across every sampled view."""
        best = None
        for view in self.views:
            detection = view.locate(frame, frame_index=frame_index)
            if detection is not None and (best is None
                                          or detection.n_inliers > best.n_inliers):
                best = detection
        return best

    def locate_all(self, frame, max_results: int = 8,
                   frame_index: int = None) -> list:
        """Every instance found by any view, with duplicates removed."""
        found = []
        for view in self.views:
            found.extend(view.locate_all(frame, max_results=max_results,
                                         frame_index=frame_index))
        return _drop_overlapping(found)[:max_results]

    def scan_video(self, path: str, **kwargs) -> Detection:
        """Search a video, taking the strongest hit any view produces."""
        best = None
        for view in self.views:
            detection = view.scan_video(path, **kwargs)
            if detection is not None and (best is None
                                          or detection.n_inliers > best.n_inliers):
                best = detection
                if detection.confidence >= kwargs.get("min_confidence", 0.3):
                    break
        return best


def frame_coverage(detection: Detection, frame_shape) -> float:
    """What fraction of the frame a detection's quad covers.

    A result covering most of the picture usually means the reference was a
    wide shot rather than a crop of the target, so the quad describes the whole
    scene instead of the thing to insert onto.
    """
    if detection is None:
        return 0.0
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        return 0.0
    return float(quad_area(detection.quad) / (width * height))


def open_reference(path: str, **kwargs):
    """Build a matcher from either a picture of the target or a clip of it."""
    if os.path.splitext(path)[1].lower() in VIDEO_SUFFIXES:
        samples = kwargs.pop("samples", 6)
        return VideoReferenceMatcher(path, samples=samples, **kwargs)
    kwargs.pop("samples", None)
    return ReferenceMatcher(path, **kwargs)


# --------------------------------------------------------------------------
# Tracking history
# --------------------------------------------------------------------------

def interpolate_tracking(history: dict, start: int = None, end: int = None) -> dict:
    """Fill gaps between tracked frames by interpolating the corners.

    Tracking data is often sparse -- the user tracks a stretch, scrubs, adjusts
    a corner, tracks again. Holding the last known position across those gaps
    makes the insert visibly freeze while the camera keeps moving; interpolating
    keeps it gliding between the frames that were actually solved.

    Only interior gaps are filled. Nothing is extrapolated past the first or
    last tracked frame, since there is no evidence out there.
    """
    tracked = {int(k): as_quad(v) for k, v in history.items()
               if v is not None and len(v) == 4}
    if not tracked:
        return {}

    keys = sorted(tracked)
    lo = keys[0] if start is None else max(keys[0], int(start))
    hi = keys[-1] if end is None else min(keys[-1], int(end))
    if hi < lo:
        return {}

    dense = {}
    for i, key in enumerate(keys[:-1]):
        next_key = keys[i + 1]
        a, b = tracked[key], tracked[next_key]
        span = next_key - key
        for frame in range(key, next_key):
            if lo <= frame <= hi:
                t = (frame - key) / span
                dense[frame] = (a * (1.0 - t) + b * t).astype(np.float32)
    if lo <= keys[-1] <= hi:
        dense[keys[-1]] = tracked[keys[-1]].copy()
    return dense


def history_to_lists(history: dict) -> dict:
    """Convert a history of arrays to JSON-serialisable nested lists."""
    return {int(k): [[float(x), float(y)] for x, y in as_quad(v)]
            for k, v in history.items() if v is not None and len(v) == 4}


# --------------------------------------------------------------------------
# Video output
# --------------------------------------------------------------------------

# Extra directories to search for ffmpeg before falling back to PATH. A
# packaged build registers its own folder here, so dropping ffmpeg.exe next to
# the application is enough to get audio -- no install, no admin rights.
_BINARY_SEARCH_DIRS = []


def register_binary_dir(path: str) -> None:
    """Add a directory to search for bundled helper binaries."""
    if path and os.path.isdir(path) and path not in _BINARY_SEARCH_DIRS:
        _BINARY_SEARCH_DIRS.insert(0, path)


def find_binary(name: str) -> str:
    """Locate a helper binary next to the app, else on PATH."""
    for directory in _BINARY_SEARCH_DIRS:
        for candidate in (name, f"{name}.exe"):
            full = os.path.join(directory, candidate)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return full
    return shutil.which(name)


def has_ffmpeg() -> bool:
    """True if an ``ffmpeg`` binary is available."""
    return find_binary("ffmpeg") is not None


class FrameWriter:
    """Writes rendered frames to a video file, as well as it can.

    OpenCV's VideoWriter is limited to whatever codecs its wheel was built
    with, and in practice that means ``mp4v`` at a fixed, fairly low bitrate:
    measured here at 34 dB PSNR, which is enough to wipe out exactly the
    subtle work that makes an insert convincing. Added film grain fell from a
    sigma of 2.0 to 0.45, and a flat creative suffers worst because it is cheap
    to encode and gets quantised hardest, while the busy footage around it
    keeps its texture. For a research stimulus that is the wrong trade.

    When ffmpeg is available, frames are piped to it and encoded with x264 at a
    visually-lossless CRF, and the source audio is muxed in the same pass. When
    it is not, this falls back to VideoWriter and reports the fact, so the
    render still succeeds and the caveat reaches the user.
    """

    def __init__(self, path: str, fps: float, size, crf: int = 17,
                 audio_source: str = None, audio_start: float = 0.0,
                 duration: float = None):
        self.path = path
        self.fps = float(fps) or 25.0
        self.width, self.height = int(size[0]), int(size[1])
        self.crf = int(crf)
        self.audio_source = audio_source
        self.audio_start = float(audio_start)
        self.duration = duration

        self.process = None
        self.writer = None
        self.caveats = []
        self._frames = 0

        if not self._start_ffmpeg():
            self._start_opencv()

    # -- back ends ---------------------------------------------------------

    def _start_ffmpeg(self) -> bool:
        ffmpeg = find_binary("ffmpeg")
        if not ffmpeg:
            self.caveats.append(
                "encoded with OpenCV's built-in codec; install ffmpeg for "
                "noticeably better quality and to keep the audio")
            return False

        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.width}x{self.height}", "-r", f"{self.fps}",
               "-i", "-"]

        has_audio = bool(self.audio_source) and source_has_audio(self.audio_source)
        if has_audio:
            if self.audio_start:
                cmd += ["-ss", f"{self.audio_start:.6f}"]
            cmd += ["-i", self.audio_source]
            if self.duration:
                cmd += ["-t", f"{self.duration:.6f}"]

        cmd += ["-map", "0:v:0"]
        if has_audio:
            cmd += ["-map", "1:a:0?", "-c:a", "aac", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(self.crf),
                "-pix_fmt", "yuv420p", self.path]

        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Could not start ffmpeg (%s); using OpenCV instead", exc)
            self.process = None
            self.caveats.append("ffmpeg could not be started, so the built-in "
                                "codec was used")
            return False

        if not has_audio and self.audio_source:
            self.caveats.append("the source has no audio track")
        return True

    def _start_opencv(self):
        self.writer = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
            (self.width, self.height))
        if not self.writer.isOpened():
            raise IOError(f"Could not open {self.path} for writing.")

    # -- writing -----------------------------------------------------------

    @property
    def using_ffmpeg(self) -> bool:
        return self.process is not None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.process is not None:
            if self.process.poll() is not None:
                raise IOError("ffmpeg stopped early: " + self._ffmpeg_error())
            try:
                self.process.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
            except (BrokenPipeError, OSError) as exc:
                raise IOError(f"ffmpeg stopped accepting frames: {exc}") from exc
        else:
            self.writer.write(frame_bgr)
        self._frames += 1

    def _ffmpeg_error(self) -> str:
        try:
            return (self.process.stderr.read() or b"").decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            return "no error output"

    def close(self) -> list:
        """Finish the file and return any caveats worth telling the user."""
        if self.process is not None:
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            code = self.process.wait(timeout=600)
            if code != 0:
                self.caveats.append("ffmpeg reported an error while encoding")
                logger.warning("ffmpeg failed: %s", self._ffmpeg_error())
            self.process = None
        elif self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.caveats

    def abort(self) -> None:
        """Give up without waiting, for a cancelled render."""
        if self.process is not None:
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            self.process.terminate()
            self.process = None
        elif self.writer is not None:
            self.writer.release()
            self.writer = None


def remux_audio(video_path: str, source_path: str, out_path: str,
                start_time: float = 0.0, duration: float = None,
                timeout: int = 600) -> bool:
    """Copy the audio from ``source_path`` onto the silent render.

    OpenCV's ``VideoWriter`` has no concept of audio, so a render always comes
    out silent. When ffmpeg is available the original audio is grafted back on,
    offset to match the rendered frame range. Returns True if ``out_path`` was
    written; on any failure the caller keeps the silent render.
    """
    ffmpeg = find_binary("ffmpeg")
    if not ffmpeg:
        logger.info("ffmpeg not found; leaving the render silent")
        return False

    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", video_path]
    if start_time:
        cmd += ["-ss", f"{start_time:.6f}"]
    cmd += ["-i", source_path]
    if duration:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-map", "0:v:0",
        "-map", "1:a:0?",   # '?' => tolerate a source with no audio track
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("audio remux failed to run: %s", exc)
        return False

    if result.returncode != 0:
        logger.warning("audio remux failed: %s",
                       result.stderr.decode("utf-8", "replace").strip())
        return False
    return os.path.exists(out_path)


def source_has_audio(path: str) -> bool:
    """True if ffprobe reports an audio stream (False if ffprobe is absent)."""
    ffprobe = find_binary("ffprobe")
    if ffprobe is None:
        return False
    try:
        result = subprocess.run(
            [ffprobe, "-loglevel", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())
