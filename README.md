# rembrand — LUMEN Insertion Tool

A PyQt5 desktop tool for inserting a creative (image or video) into a base
video so that it sticks to a surface in the scene as the camera moves.

You load a base video, mark the surface you want to fill — a wall, a screen, a
table, a billboard — pick a creative, and the tool tracks that surface and warps
the creative into it with the right perspective. The per-frame corner positions
are kept as tracking data, which can be saved, reloaded, and exported alongside
the rendered video.

## Running it

```bash
pip install -r requirements.txt
python LUMEN_Insertion_Tool_1.0.0.py
```

Requires a desktop session — it is a windowed Qt application, not a CLI.

Install `ffmpeg` too if you want the rendered video to keep its audio; see
[Audio](#audio) below.

**To give it to someone who does not have Python**, run `python build.py` to
package it into a self-contained folder they can unzip and double-click. No
install, no admin rights. See [BUILD.md](BUILD.md).

Debug output goes to `app_debug.log` in a per-user folder
(`%LOCALAPPDATA%\LUMEN` on Windows, `~/Library/Application Support/LUMEN` on
macOS, `~/.local/share/LUMEN` on Linux).

## Workflow

1. **Load a base video** — hamburger menu → *Load* → *Base Video*
   (`.mp4`, `.avi`, `.mkv`).
2. **Mark the area**, either way round:
   - *By hand*: select the *Draw* tool and click four times to place the
     corners. Drag a corner to move it. The magnifier shows a zoomed view of
     each corner in its own quadrant for precise placement.
   - *From a picture*: hamburger menu → *Find Target from Image…*, then upload a
     photo of the billboard, screen or poster. The tool searches the footage for
     it and drops the quad on its actual corners. See
     [Finding a target from an image](#finding-a-target-from-an-image).
3. **Curve the edges if the surface is not flat** — the *Curve* tool adds two
   square handles per edge; drag them to bend that edge around a curved screen
   or pillar. See [Curved edges](#curved-edges).
4. **Load a creative** — *Load* → *Creative Overlay*. Images (`.png`, `.jpg`,
   `.jpeg`, `.bmp`) and videos (`.mp4`, `.avi`, `.mkv`) are both accepted, and
   several can be loaded at once; each appears as a card in the right-hand panel.
5. **Insert** — click *Insert* on a creative card. It is warped into the area and
   previewed live over the base frame.
6. **Track** — turn the tracking switch on and play forward. The area follows the
   surface and the corner positions are recorded for each frame.
7. **Export** — the *Render* action writes an MP4 over a chosen frame range and
   scale factor, on a background thread with a cancellable progress dialog.

## Controls

| Input | Action |
| --- | --- |
| Click (Draw mode) | Place a corner, up to four |
| Drag a corner | Reposition it |
| Drag a square handle (Curve mode) | Bend that edge |
| `1`–`4` | Select the corresponding corner |
| `5` | Select the whole shape |
| Arrow keys | Nudge the selection by 1 px |
| `D` / `C` | Delete this frame's shape / copy it from the previous frame |
| Tracking switch | Enable/disable tracking during playback |
| Magnifier switch | Show/hide the corner magnifier |

Brightness, contrast and colourise adjustments for the inserted creative are
available from the left toolbar, and all three apply to the render as well as
the preview.

## How the tracking works

The four corners are deliberately *not* tracked directly. Hand-placed corners
tend to land on occluding edges and flat, untextured areas — the worst possible
things to follow — which is what makes an insert slide off its surface.

Instead, `PlanarTracker` detects strong features across the whole interior of
the area, follows them with pyramidal Lucas-Kanade optical flow, discards any
that fail a forward-backward consistency check, and fits a RANSAC homography to
the survivors. The corners are then carried by that homography. Because the fit
is over many points, a few bad tracks are outliers RANSAC rejects rather than a
quarter of the entire signal.

Each step is sanity-checked before it is accepted: a homography that folds the
area over, collapses it, or jumps it implausibly far in one frame is rejected,
and the tracker's state is left untouched so it can recover on the next frame
rather than being corrupted. A constant-velocity Kalman filter per corner
smooths the result and coasts the shape through brief occlusions.

Results accumulate in `tracking_history`, a `{frame_index: [4 corners]}` map
that drives both the preview and the render.

## Finding a target from an image

`ReferenceMatcher` locates a target from a picture of it: SIFT (or ORB)
keypoints are matched between the reference and each frame, and a RANSAC
homography maps the reference's own corners into the footage. If the target is
not on screen at the start, the search samples forward through the video until
it finds it.

This finds **flat, detailed** targets — billboards, TV screens, posters, signs,
shop fronts — which is exactly the class of surface worth inserting onto. It is
not a face or object detector: a face is neither flat nor rigid, so matching one
photo of a face against a moving one is unreliable, and a face is not a surface
you would paste a billboard onto. That needs a dedicated detector, not this.

For the best hit rate, use a sharp, roughly straight-on picture cropped to just
the target.

## Curved edges

Each edge of the area carries two cubic Bezier control points. Their default
positions are exactly one and two thirds along the straight edge, where a cubic
Bezier reproduces a straight line — so *Curve* can be switched on and off
without the insert moving until a handle is actually dragged.

Curvature is applied as a displacement on top of the perspective mapping rather
than replacing it, so a curved insert still foreshortens the way a flat one
does. The bend is stored in each edge's own frame of reference, which means it
rotates and scales with the surface as the camera moves instead of staying
pinned to the screen axes.

Bend an edge far enough and the surface turns inside out — there is then no
single point of the creative behind a given pixel, and nothing sensible to
draw. That case is detected (`region_is_folded`) and the insert is left off the
frame, the same way a self-intersecting bowtie quad is rejected. The outline
turns red when this happens.

## Audio

OpenCV's `VideoWriter` cannot write audio, so a render is silent until the
original audio is copied back onto it. If `ffmpeg` is on `PATH` this happens
automatically, offset to match the rendered frame range. If it is not, the
render still succeeds and the completion message says the result is silent.

## Saving and export formats

| Menu item | Output |
| --- | --- |
| *Save* → *Tracking Points* | JSON, `{frame_index: [[x, y] × 4]}` |
| *Save* → *Project* | JSON: the full tracking history, curvature, creative path and colour settings |
| *Save* → *AOI Geometry* | CSV in `csv_type_3` layout — one row per tracked frame with `START`/`END` times and a `points` column of `x0;y0;…;x3;y3` |
| Render | MP4 (`mp4v`) of the composited result, with audio when ffmpeg is available |

*Load* → *Project* restores a saved session, and reads older project files that
stored only a single quad.

## Layout

`lumen_core.py` holds the algorithms as plain NumPy/OpenCV with no Qt imports.
`LUMEN_Insertion_Tool_1.0.0.py` is the application on top of it. The split
matters for more than tidiness: the preview and the renderer call the *same*
compositing function, so what you approve on screen is what gets written to the
file, and the algorithms can be tested without a display.

| Component | Role |
| --- | --- |
| `lumen_core.PlanarTracker` | Feature-based planar tracking with RANSAC and sanity checks |
| `lumen_core.ReferenceMatcher` | Locates a target in the footage from a reference image |
| `lumen_core.Region` | Four corners plus per-edge curvature |
| `lumen_core.composite_region` | Alpha-correct perspective/curved warp and blend |
| `lumen_core.interpolate_tracking` | Fills the gaps between tracked frames |
| `lumen_core.remux_audio` | Copies the source audio onto a finished render |
| `MainWindow` | Frameless main window, menus, load/save/render actions |
| `CentralPanel` | Video playback, frame stepping, the tracking loop |
| `TrackingOverlay` | The area, corner and curve handles, live preview |
| `MagnifierWidget` | Four-quadrant zoomed view of the corners |
| `RenderSettings` / `RenderWorker` | Off-thread compositing and MP4 export |

## Tests

```bash
pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -q
```

No display required. Tracking accuracy is measured against synthetic footage
whose motion is known exactly, so the tests assert real numbers — sub-pixel
corner error, and that the current tracker beats the four-corner approach it
replaced — rather than merely that the code runs. `tests/test_app_smoke.py`
drives the actual Qt widgets through loading, tracking and a full render.
