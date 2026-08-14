"""Qt-free core for the LUMEN Insertion Tool.

Everything in here is plain NumPy/OpenCV so it can be exercised headlessly,
without a display or an event loop. The GUI in ``LUMEN_Insertion_Tool_1.0.0.py``
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


def composite_overlay(base_bgr: np.ndarray, overlay: np.ndarray, quad,
                      opacity: float = 1.0, feather: float = 0.0,
                      in_place: bool = False) -> np.ndarray:
    """Warp ``overlay`` into ``quad`` on ``base_bgr`` with real alpha blending.

    The creative's own alpha channel is honoured and multiplied by an
    anti-aliased quad mask, so transparent PNGs stay transparent and edges do
    not stair-step. Work is confined to the quad's bounding box, so cost
    scales with the size of the insert rather than the size of the frame.

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
                     in_place: bool = False) -> np.ndarray:
    """Composite a creative into a :class:`Region`, curved edges included.

    Straight regions are handed to the plain homography path, which is both
    faster and exact; only genuinely curved ones pay for the iterative solve.
    """
    if region.is_straight():
        return composite_overlay(base_bgr, overlay, region.corners,
                                 opacity=opacity, feather=feather,
                                 in_place=in_place)

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

    __slots__ = ("ok", "quad", "n_features", "n_inliers", "reason")

    def __init__(self, ok, quad=None, n_features=0, n_inliers=0, reason=""):
        self.ok = ok
        self.quad = quad
        self.n_features = n_features
        self.n_inliers = n_inliers
        self.reason = reason

    def __repr__(self):
        return (f"TrackResult(ok={self.ok}, n_features={self.n_features}, "
                f"n_inliers={self.n_inliers}, reason={self.reason!r})")


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

    def __init__(self, frame, quad,
                 fb_threshold: float = 1.0,
                 ransac_threshold: float = 3.0,
                 min_features: int = 6,
                 reseed_below: int = 40,
                 max_area_ratio: float = 1.8,
                 max_step_fraction: float = 0.25,
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
        """
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
        self._seed_features()

    @staticmethod
    def _gray(frame):
        if frame is None:
            raise ValueError("frame is None")
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _seed_features(self):
        """Detect trackable features strictly inside the current quad."""
        h, w = self.prev_gray.shape[:2]
        mask = quad_to_mask(self.quad, w, h)

        # Pull the mask in from the boundary: features sitting on the edge of
        # the region are usually on an occluding contour and track badly.
        eroded = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if cv2.countNonZero(eroded) > 100:
            mask = eroded
        mask[mask > 0] = 255

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

        if self.features is None or len(self.features) < self.min_features:
            self._seed_features()
        if len(self.features) < self.min_features:
            return TrackResult(False, reason="no trackable features in region")

        p0 = self.features.reshape(-1, 1, 2).astype(np.float32)

        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                 **self.lk_params)
        if p1 is None:
            return TrackResult(False, reason="optical flow returned nothing")

        # Track back again: a point that does not return to where it started
        # was not really tracked, whatever its reported error.
        p0r, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None,
                                                  **self.lk_params)
        if p0r is None:
            return TrackResult(False, reason="reverse optical flow returned nothing")

        fb_error = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st_fwd.ravel() == 1) & (st_bwd.ravel() == 1) & (fb_error < self.fb_threshold)
        n_good = int(good.sum())
        if n_good < 3:
            return TrackResult(False, n_features=n_good,
                               reason=f"only {n_good} points survived the flow check")

        src = p0.reshape(-1, 2)[good]
        dst = p1.reshape(-1, 2)[good]

        matrix, inlier_mask = self._fit_transform(src, dst)
        if matrix is None:
            return TrackResult(False, n_features=n_good, reason="could not fit a transform")

        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else n_good
        if n_inliers < 3:
            return TrackResult(False, n_features=n_good, n_inliers=n_inliers,
                               reason="too few inliers")

        cumulative = matrix @ self.cumulative_h
        new_quad = cv2.perspectiveTransform(
            self.anchor_quad.reshape(-1, 1, 2), cumulative.astype(np.float64)
        ).reshape(4, 2).astype(np.float32)

        rejection = self._reject_reason(new_quad)
        if rejection:
            return TrackResult(False, n_features=n_good, n_inliers=n_inliers,
                               reason=rejection)

        # Commit.
        self.cumulative_h = cumulative
        self.quad = new_quad
        self.prev_gray = gray
        kept = dst[inlier_mask.ravel().astype(bool)] if inlier_mask is not None else dst
        self.features = kept.reshape(-1, 1, 2).astype(np.float32)
        if len(self.features) < self.reseed_below:
            self._seed_features()

        return TrackResult(True, quad=new_quad.copy(), n_features=n_good,
                           n_inliers=n_inliers)

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

    def locate(self, frame, frame_index: int = None) -> Detection:
        """Find the reference in one frame, or return None."""
        gray = self._gray(frame)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            return None

        try:
            knn = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        except cv2.error:
            return None

        # Lowe's ratio test: keep a match only when the best candidate is
        # clearly better than the runner-up, which throws out the ambiguous
        # matches that repetitive texture generates.
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        if len(good) < max(4, self.min_inliers // 2):
            return None

        src = np.float32([self.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold)
        if matrix is None or mask is None:
            return None

        n_inliers = int(mask.sum())
        if n_inliers < self.min_inliers:
            return None

        quad = cv2.perspectiveTransform(
            self.reference_corners.reshape(-1, 1, 2), matrix).reshape(4, 2)
        if not is_valid_quad(quad, min_area=16.0):
            return None

        # A homography fitted to noise loves to produce a sliver stretching
        # off the frame; require something plausibly on screen.
        h, w = gray.shape[:2]
        if quad_area(quad) > 4.0 * w * h:
            return None

        confidence = n_inliers / float(len(good))
        return Detection(quad.astype(np.float32), confidence, n_inliers,
                         len(good), frame_index)

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

def has_ffmpeg() -> bool:
    """True if an ``ffmpeg`` binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def remux_audio(video_path: str, source_path: str, out_path: str,
                start_time: float = 0.0, duration: float = None,
                timeout: int = 600) -> bool:
    """Copy the audio from ``source_path`` onto the silent render.

    OpenCV's ``VideoWriter`` has no concept of audio, so a render always comes
    out silent. When ffmpeg is available the original audio is grafted back on,
    offset to match the rendered frame range. Returns True if ``out_path`` was
    written; on any failure the caller keeps the silent render.
    """
    if not has_ffmpeg():
        logger.info("ffmpeg not found; leaving the render silent")
        return False

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path]
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
    if shutil.which("ffprobe") is None:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-loglevel", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())
