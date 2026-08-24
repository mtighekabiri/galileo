# Working on this repo

## The footage this tool is actually pointed at

Stated by the user, and it should shape design decisions rather than be
discovered again each time:

- **Shot from a moving viewpoint** — driving or walking past. The billboard is
  rarely at a steady distance.
- **Distance changes a great deal within a single shot**, so apparent size,
  angle, sharpness and effective resolution all change with it. A panel can go
  from a small distant rectangle to filling the frame in a few seconds.
- **Occlusion is the ordinary case, not the exception**: railings, lampposts,
  signs, people, passing vehicles.
- Targets are both **printed hoardings and digital screens playing their own
  content**.

**Nothing here should assume a fixed scale.** Anything expressed in pixels, in
frames, or as a fraction of what the camera happens to see needs checking
across an approach, not just on one frame. Two settings measured so far are
known to be scale-dependent -- see below.

## Measured: one obstruction setting does not cover an approach

Driving up to a hoarding, with a post crossing it, at the shipped defaults
(`DEPTH_FLOOR` 0.15, `k` 5). "Fills" is the billboard's width as a share of
the frame's:

| fills | post found | artwork falsely masked | context outside the quad |
| --- | --- | --- | --- |
| 10% | 53% | 18% | 75% |
| 18% | 5% | 0% | 75% |
| 30% | 0% | 0% | 75% |
| 45% | 7% | 0% | 75% |
| 65% | 100% | 16% | 58% |
| 85% | 56% | 25% | 27% |
| 105% | 100% | 46% | 2% |

Two mechanisms, both structural rather than artefacts of the synthetic scene:

1. **Which half of the threshold binds flips with distance.** `max(k * fit
   residual, floor_rel * scene depth spread)` is decided by the `k` term at
   10/30/65/85% and by the floor at 18/45/105%. So *both* dials matter, and
   which one is in charge is not predictable from the shot. Do not describe
   either as "the one that matters most" -- an earlier version of the README
   did, and it is not true here.
2. **Close range starves the normaliser.** The threshold is a fraction of the
   depth range of a padded crop. As the panel fills the frame that crop is
   almost all panel -- 2% of it is anything else at 105% -- so the yardstick
   becomes the artwork's own depicted depth, and 46% of the creative is masked.

Neither is fixed. If this is picked up: the honest options are a threshold
derived per frame from the panel's apparent size, a crop that keeps a minimum
of real context, or an admission in the docs that a drive-up needs the dials
moved partway through.

`STEADY_WINDOW` (11 frames) has the same smell and has *not* been measured
across an approach: a fixed window in frames, smoothing a path in pixels,
while the on-screen motion accelerates.

## Conventions worth keeping

- **Claims in comments, docstrings and the README carry measured numbers.**
  Not "this is better" but "0.57 px became 2.94 px". If a number cannot be
  produced, say what was not measured.
- **The preview and the render call the same functions**, so what is approved
  on screen is what reaches the file. Any new setting must reach both, and the
  render takes a detached copy so a dial moved mid-render cannot change the
  output.
- **Tests assert real quantities against synthetic footage with known ground
  truth** (`tests/conftest.py` builds it). Accuracy is mean corner distance in
  px; steadiness is mean second difference of the corners.
- **A test that passes against the broken code proves nothing.** GUI behaviour
  driven through `QTest` clicks, not by calling slots: a slot call never moves
  focus, so a focus bug survives it. Check a new test fails without its fix.
- Model files are fetched by `fetch_model.py`, never committed.
- Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q`. Tests needing a
  model skip cleanly when it is absent.

## Traps already hit

- Any test leaving the project dirty hangs on the unsaved-work dialog at
  teardown; the `window` fixture marks it clean before closing.
- The left toolbar has to fit ten tools in the 577 px a 1024x640 window
  leaves. An eleventh needs the icons themselves to shrink.
- `tracking_mode` is the auto-tracking switch, *not* whether the drawing tools
  are live (`tracking_overlay.tracking_enabled`). Confusing the two has caused
  bugs more than once.
