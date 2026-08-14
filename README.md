# rembrand — LUMEN Insertion Tool

A PyQt5 desktop tool for inserting a creative (image or video) into a base
video so that it sticks to a surface in the scene as the camera moves.

You load a base video, draw a four-point quad over the surface you want to
fill (a wall, a screen, a table, a billboard), pick a creative, and the tool
tracks that quad frame to frame and warps the creative into it with the right
perspective. The per-frame corner positions are kept as tracking data, which
can be saved, reloaded, and exported alongside the rendered video.

## Running it

```bash
pip install -r requirements.txt
python LUMEN_Insertion_Tool_1.0.0.py
```

Requires a desktop session — it is a windowed Qt application, not a CLI.
Debug output is written to `app_debug.log` in the working directory.

## Workflow

1. **Load a base video** — hamburger menu → *Load* → *Base Video*
   (`.mp4`, `.avi`, `.mkv`).
2. **Draw the area of interest** — select the *Draw* tool, then click four
   times on the frame to place the quad corners. Drag a corner to move it.
   The magnifier panel shows a zoomed view of each corner in its own quadrant
   so the corners can be placed precisely.
3. **Load a creative** — *Load* → *Creative Overlay*. Images
   (`.png`, `.jpg`, `.jpeg`, `.bmp`) and videos (`.mp4`, `.avi`, `.mkv`) are
   both accepted, and several can be loaded at once; each appears as a card in
   the right-hand panel.
4. **Insert** — click *Insert* on a creative card. It is warped into the quad
   via a homography and previewed live over the base frame. For video
   creatives, playback is offset from the frame at which it was inserted.
5. **Track** — turn the tracking switch on and play forward. The quad follows
   the surface, and the corner positions for each frame are recorded.
6. **Export** — the *Render* action writes an MP4 over a chosen frame range
   and scale factor, on a background thread with a cancellable progress dialog.

## Controls

| Input | Action |
| --- | --- |
| Click (Draw mode) | Place a corner, up to four |
| Drag a corner | Reposition it |
| `1`–`4` | Select the corresponding corner |
| `5` | Select the whole shape |
| Arrow keys | Nudge the selection by 1 px |
| Tracking switch | Enable/disable tracking during playback |
| Magnifier switch | Show/hide the corner magnifier |

Brightness, contrast and colourise adjustments for the inserted creative are
available from the left toolbar.

## How the tracking works

Corner positions are propagated with Lucas-Kanade pyramidal optical flow
(`cv2.calcOpticalFlowPyrLK`, 21×21 window, 3 levels) applied to the four
corner points, and each corner is smoothed by a constant-velocity 2D Kalman
filter (`SimpleKalmanFilter`) holding state `[x, y, vx, vy]`. When optical
flow fails or the mean error is too high, the filters' prediction is used
instead, so the quad coasts through brief occlusions.

Results accumulate in `tracking_history`, a `{frame_index: [4 corners]}` map
that drives both the preview and the render.

## Saving and export formats

| Menu item | Output |
| --- | --- |
| *Save* → *Tracking Points* | JSON, `{frame_index: [[x, y] × 4]}` |
| *Load* → *Tracking Points* | Restores a saved tracking history |
| *Save* → *AOI Geometry* | CSV in `csv_type_3` layout — one row per tracked frame with `START`/`END` times and a `points` column of `x0;y0;…;x3;y3` |
| Render | MP4 (`mp4v`) of the composited result |

## Layout

Everything lives in `LUMEN_Insertion_Tool_1.0.0.py`. The significant pieces:

| Component | Role |
| --- | --- |
| `MainWindow` | Frameless main window, menus, load/save/render actions |
| `CentralPanel` | Video playback, frame stepping, the tracking loop |
| `TrackingOverlay` | The quad, corner interaction, live warped preview |
| `MagnifierWidget` | Four-quadrant zoomed view of the corners |
| `RenderWorker` | Off-thread compositing and MP4 export |
| `SimpleKalmanFilter` | Per-corner constant-velocity smoothing |
| `QSwitch`, `HoverButton`, `IconWidget`, `TitleBar`, `LeftColumn` | Custom dark-theme UI widgets |
