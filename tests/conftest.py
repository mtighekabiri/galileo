"""Synthetic footage with known ground truth.

The tracker's whole job is to recover where a surface went, so the tests build
clips where that answer is known exactly: a textured poster is warped into a
quad whose path is generated analytically, then the tracker is asked to find
that path again from the pixels alone.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bend_edge(region, edge, offset, slots=(0, 1)):
    """Push an edge's Bezier handles by an offset in pixels.

    Curvature is stored in each edge's own frame, so tests that want to think
    in screen pixels go through this rather than poking the coefficients.
    """
    for slot in slots:
        current = np.asarray(region.controls[edge, slot], np.float32)
        region.set_control_point(edge, slot, current + np.float32(offset))
    return region


def make_texture(width=320, height=240, seed=0):
    """A busy, high-contrast image with plenty of trackable corners."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 240, np.uint8)

    for _ in range(60):
        x0, y0 = rng.integers(0, width), rng.integers(0, height)
        x1, y1 = rng.integers(0, width), rng.integers(0, height)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        else:
            cv2.line(img, (x0, y0), (x1, y1), color, int(rng.integers(1, 5)))

    for _ in range(40):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, center, int(rng.integers(3, 20)), color, -1)

    return img


def make_background(width=640, height=480, seed=1):
    """A distinct, also-textured backdrop to warp the poster onto."""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 200, (height // 16, width // 16, 3), dtype=np.uint8)
    bg = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(bg, (9, 9), 3)


def quad_path(n_frames, width=640, height=480, drift=(1.6, 0.7),
              scale_per_frame=0.004, rotation_per_frame=0.25):
    """Ground-truth corner positions: a slow pan with drift, zoom and roll."""
    base = np.float32([[180, 130], [430, 120], [440, 330], [170, 340]])
    center = base.mean(axis=0)

    path = []
    for i in range(n_frames):
        angle = np.deg2rad(rotation_per_frame * i)
        scale = 1.0 + scale_per_frame * i
        rot = np.float32([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]]) * scale
        moved = (base - center) @ rot.T + center
        moved += np.float32([drift[0] * i, drift[1] * i])
        path.append(moved.astype(np.float32))
    return path


def render_clip(path_quads, texture=None, background=None,
                width=640, height=480):
    """Warp the texture into each ground-truth quad over the background."""
    texture = make_texture() if texture is None else texture
    background = make_background(width, height) if background is None else background

    th, tw = texture.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])

    frames = []
    for quad in path_quads:
        frame = background.copy()
        matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        warped = cv2.warpPerspective(texture, matrix, (width, height))
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        frame[mask > 0] = warped[mask > 0]
        frames.append(frame)
    return frames


def write_video(frames, path, fps=25.0):
    """Write frames to an mp4 and return the path."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not open a VideoWriter for mp4v")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture(scope="session")
def texture():
    return make_texture()


@pytest.fixture(scope="session")
def clip(texture):
    """20 frames of a textured poster moving on a known path."""
    quads = quad_path(20)
    return render_clip(quads, texture=texture), quads


@pytest.fixture(scope="session")
def clip_video(tmp_path_factory, clip):
    frames, quads = clip
    path = tmp_path_factory.mktemp("clips") / "moving.mp4"
    return write_video(frames, path), quads


@pytest.fixture
def logo_bgra():
    """A translucent creative: opaque disc, soft edge, transparent corners."""
    size = 120
    img = np.zeros((size, size, 4), np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2 - 4, (40, 200, 90, 255), -1)
    cv2.circle(img, (size // 2, size // 2), size // 4, (255, 255, 255, 255), -1)
    return img
