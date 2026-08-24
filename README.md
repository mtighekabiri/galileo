# rembrand — Galileo Insertion Tool

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
python Galileo_Insertion_Tool_1.0.0.py
```

Requires a desktop session — it is a windowed Qt application, not a CLI.

Install `ffmpeg` too if you want the rendered video to keep its audio; see
[Audio](#audio) below.

**To give it to someone who does not have Python**, run `python build.py` to
package it into a self-contained folder they can unzip and double-click. No
install, no admin rights. See [BUILD.md](BUILD.md).

Log output goes to `app_debug.log` in a per-user folder
(`%LOCALAPPDATA%\Galileo` on Windows, `~/Library/Application Support/Galileo` on
macOS, `~/.local/share/Galileo` on Linux). Set the environment variable
`GALILEO_DEBUG=1` for full debug-level detail.

## Workflow

1. **Load a base video** — hamburger menu → *Load* → *Base Video*
   (`.mp4`, `.avi`, `.mkv`).
2. **Mark the area**, either way round:
   - *By hand*: select the *Draw* tool and click four times to place the
     corners. Drag a corner to move it. The magnifier gives each handle a
     zoomed view for precise placement — see
     [The magnifier](#the-magnifier).
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
| `T` / `R` / `B` / `L` | Select that edge's bend handles; press again for the second |
| Arrow keys | Nudge the selection by 1 px (Shift for 10, Ctrl for 0.25) |
| Transport buttons | Stepping or scrubbing leaves the drawing keys live |
| `D` / `C` | Delete this frame's shape / copy it from the previous frame (also the *Delete* and *Copy* buttons) |
| Tracking switch | Enable/disable tracking during playback |
| Magnifier switch | Show/hide the magnifier (it also appears while dragging) |
| Double-click the magnifier | Fill the video stage with it, or put it back |
| Scroll on the magnifier | Set its magnification by hand |
| Panes button on the magnifier | One view of the whole area, or a tile per handle |
| Options → Steady the lighting | Even out pulsing from flickering fittings |
| Options → Steady the tracked path | Take the wobble out of the tracked shape |
| Options → Draw behind people | Let passers-by cross in front of the creative |
| Options → Draw behind obstructions | Let railings, posts and signs stay in front of it |
| *Behind* tool (left toolbar) | The same switch, with dials for how readily something counts as being in front |

Brightness, contrast and colourise adjustments for the inserted creative are
available from the left toolbar, and all three apply to the render as well as
the preview.

## Digital screens vs printed billboards

This matters more than it sounds, and getting it wrong makes the insert slide
off the panel entirely.

By default the tracker follows the texture **inside** the marked area. That is
right for a printed billboard or poster, whose artwork is fixed to the surface.

A digital OOH screen is different: it is already playing an advert, and that
picture moves on its own. Features found on it follow the advert rather than
the screen showing it. On a test clip of a screen playing scrolling content,
interior tracking drifts **100 px on average and 175 px by frame 30** — while
reporting no failures at all, so it is confidently wrong rather than visibly
broken.

For those, turn on **Options → Digital screen (track surroundings)**. The
tracker then follows the bezel, wall and fittings around the screen, which are
genuinely rigid. On the same clip that gives **1.0 px** average error.

> When using this mode, leave some room around the screen as you mark it —
> the surrounding detail is what the tracking now depends on.

**You should not have to know any of that**, so as soon as an area is marked
the tool reads the next dozen frames and checks. It reads the motion twice,
once from inside the area and once from the surface around it. Markings fixed
to a panel give two readings that agree to a hundredth of a pixel; a picture
running on its own does not. If they disagree, it explains what it found and
offers the change.

The worst disagreement decides it, not the average — a screen changing slides
every few seconds agrees perfectly in between and wildly at each change, so an
average buries exactly the frames that matter. Measured across an ordinary
poster, one under a hunting exposure, one with a person crossing it, one shot
on a fast pan, a screen scrolling an advert, a screen changing slides, and a
screen changing slides with a person in front, it is right about all seven.

> A screen showing one **still** picture is deliberately *not* flagged. Nothing
> is moving independently, so tracking its inside works perfectly well, and
> there is nothing to warn about.

## Occlusion — people walking in front

In airports, malls and high streets somebody is constantly crossing between the
camera and the screen. Painting the creative over them is the most obvious tell
that a shot has been altered, which matters when the footage is a research
stimulus.

**Options → Draw behind people** segments the people in each frame and holds the
creative back where they are, so they pass in front of it.

This needs a model file, which is not committed to the repository:

```bash
python fetch_model.py      # all three models, once
```

It runs through OpenCV's own DNN module, so there is no extra dependency to
install, and a packaged build bundles it automatically if it was fetched before
building. Without it the menu item warns and stays off.

Note that this segments *people* specifically. For everything else, see below.

## Occlusion — anything else in front

A street is full of things that are not people: railings along a walkway, a
lamppost, a sign, a passing bus. The **Behind** tool in the left toolbar — or
**Options → Draw behind obstructions**, which is the same switch — handles all
of them, and does so without being told what any of them are.

There is no list of objects to recognise, because there could not be one. What
every obstruction has in common is not what it looks like but where it is: in
front of the surface. So that is what gets measured. A depth model estimates
how far away everything in the frame is, a plane is fitted to the area you
marked — the inverse depth of a flat surface is affine in image coordinates, so
a billboard is a tilted plane in that map whatever angle it is seen from — and
whatever stands off that plane *towards the camera* is held back from the
creative.

Two properties fall out of measuring geometry rather than appearance, and both
matter:

- **It works on a digital screen playing its own content.** The test is
  one-sided: only things nearer than the surface are marked, and a picture of a
  landscape on a screen reads as receding. The advert being replaced cannot eat
  into the creative replacing it. This is the same trap the digital-screen
  tracking mode exists for, avoided the same way — nothing here is keyed on
  what moves.
- **Nothing needs to hold still.** The mask is computed from each frame on its
  own, so a moving camera and a moving obstruction are the ordinary case, not a
  hard one. It also means the preview and the render agree by construction:
  scrubbing to a frame gives the same mask as arriving at it in sequence.

The two occlusion options are independent and combine — with both on, a
pedestrian behind a railing is drawn in front of the creative once, from
whichever source found more of them.

Two depth models can serve, and the tool prefers the better one when its file
is present. **Depth Anything V2 small** (Apache-2.0, about 99 MB — the exact
export OpenCV pins in its own dnn test suite) sees what **MiDaS v2.1 small**
(MIT, about 64 MB) structurally cannot. Measured across the drive-up, at the
same dial settings:

| | MiDaS v2.1 small | Depth Anything V2 small |
| --- | --- | --- |
| post crossing the hoarding | dips to 19% at one distance | 100% throughout |
| railing bars of 2–14 px | 0% at every distance | 91–100% |
| cost per frame, four-core desktop | ~66 ms | ~440 ms |

Depth Anything is a transformer and loads on OpenCV 5's dnn engine only; on an
older OpenCV, or when its file has not been fetched, MiDaS serves exactly as
before — the fallback is automatic and logged. Both run through OpenCV's own
DNN module, so neither adds a dependency. The preview slows down noticeably
while the depth cue is on (more so under Depth Anything); renders are
unaffected beyond taking longer.

One caution, measured and mutual: each model misreads a different kind of
*depicted* content as standing off the panel. Giant flat lettering read as
popped to Depth Anything on 45% of a synthetic test panel at any threshold,
and banded gradient artwork read as off-plane to MiDaS on 46% — while
photographic artwork sat at 2–3% for both. The artwork cue above and the
sensitivity dials are the remedies; deleting `depth_anything_v2_small.onnx`
from `models/` forces the MiDaS behaviour if a particular clip prefers it.

### The dials

A hoarding is rarely blank, and a depth network reads the photograph on it as
the scene it depicts rather than as ink on a flat panel. Where that reading
crosses the threshold, the creative is held back and the artwork already on the
billboard shows through. Because the reading shifts frame to frame, it does not
look like a steady error: holes flash open and shut for a frame at a time.

The **Behind** tool in the left toolbar is the switch and its dials in one
place — the same feature as the menu item, and the lamp on the icon follows
whichever one you use. Every dial recomposites the frame as it moves, because
none of them can be judged from a number: the question is always whether the
creative is being eaten by the billboard's own picture, or something really in
front is being painted over, and only the shot answers that.

| Dial | What it does |
| --- | --- |
| **In front by at least** | How far off the surface something must stand to count, as a fraction of the scene's depth range. The one that matters most. |
| **Clear of the noise by** | The same demand in multiples of how ragged the surface's own depth reads — what holds the line where the model is unsure. |
| **Grow the edges** | Widen what is found, in pixels. Depth edges land slightly inside the real object. |
| **Soften the edges** | Blur the mask's edge, which also hides the frame-to-frame wobble in the model's boundaries. |

The first dial starts at **0.15**, which is measured rather than picked.
On a hoarding showing a road running away, under a pan:

| | how far it stands off the surface |
| --- | --- |
| the picture on the billboard, at its worst | 0.13 |
| a lamppost across it | 0.41 |
| railings across it | 0.86 |

At the 0.10 it started out at, the advert crossed the line on 7 frames in 24;
at 0.15, on none. It costs a little of the softest depth edges (railings went
from 90% covered to 88%), which is the right way round: a hole in the creative
is far more obvious than a slightly thin edge on the thing in front of it.

**If the original creative flashes through, raise the first dial.** If
something genuinely in front is being painted over, lower it. Around 0.28 is
clean even on artwork with strong depth in it; below 0.10 is for faint
obstructions on plain artwork. *Restore Defaults* puts every dial back to the
measured setting, and *Cancel* puts back whatever you started the session with
— the dials change the preview as they move, so cancel has real work to do.

### The panel as its own reference

The depth model is one witness; the artwork is another. For a printed
hoarding, whatever is on the panel is fixed to it — so rectifying the tracked
area to one canonical rectangle makes the artwork identical frame after frame,
while anything standing in front of it, being nearer, slides across it as the
viewpoint moves. A per-pixel median over the tracked shot is therefore a clean
plate of the artwork, and wherever a frame disagrees with the plate, something
is in front. The comparison happens at the panel's own on-screen resolution,
with a one-pixel tolerance for tracking error, and it costs no download and no
model.

This cue exists because the depth model measurably cannot cover two things,
and it covers both. Measured on the drive-up, through codec compression:

| | depth model | artwork cue |
| --- | --- | --- |
| post at 18–45% of frame width | 0–7% of its pixels | 75–100% |
| railing bars of 2–14 px | not seen | 55–75% |

Its blind side is the mirror image, which is why the two run together rather
than either replacing the other: an obstruction with nearly the artwork's own
colour under-reads (measured no lower than 53%, a dark post over equally dark
artwork — exactly where a depth step is large), it knows nothing outside the
marked area, and a hard shadow crossing the panel is a photometric change it
will mark as if it were an object.

It refuses footage it cannot serve. A digital screen playing its own content
disagrees with any plate everywhere; `build_surface_plate` measures that
disagreement and returns nothing — measured 1.5 grey levels of median
disagreement for a printed hoarding against 23 for a screen with modest scene
changes, cut at 8 — and a placement in *Digital screen (track surroundings)*
mode is never tried at all. Either way those fall back to depth alone, and a
render says so in its completion notes. The checkbox in the *Behind* dialog
turns the cue off entirely for shots where it misjudges.

The plate is learned once from up to 24 tracked frames (about half a second),
deterministically — the preview and the render each build their own from the
same tracking and land on identical numbers, which is what keeps the file
matching the screen. It refreshes when the tracking it was learned from
changes, but never in the middle of a tracking pass.

### Driving or walking past

The footage this is pointed at is shot from a moving viewpoint, so a panel can
go from a small distant rectangle to filling the frame inside one shot. That
moves the ground under the threshold, because its scene-depth term is a
fraction of the depth range of a padded crop around the panel — and on the way
in, that crop stops being a scene. Measured on an approach with a post crossing
the hoarding, the crop is **75% other things** while the panel is small and
**2%** by the time it fills the frame. At that point the yardstick has quietly
become the depth the *artwork* depicts, and **46%** of the creative was being
masked.

So the demand is stiffened in proportion to how little of the crop is anything
else: below a quarter, up to three times stricter when there is nothing else at
all. On the same approach that takes the worst false masking to **26%**, for
73% of the post still found rather than 100%. Firmer settings keep cutting the
false masking and cost more than they are worth — at four times, 16% masked but
only 61% of the post found. It is gradual rather than a cliff, so a shot walks
through it without the mask lurching on one frame, and a panel with room around
it is judged exactly as before.

**What this does not fix.** The depth model's behaviour still varies with
distance more than one setting can absorb. On that same approach the post was
found on 53% of its pixels at 10% of frame width, essentially missed between
18% and 45%, and found in full at 65%. That middle stretch is the model
failing to see the post as meaningfully nearer at all, not a threshold being
wrong, and no amount of tuning recovers it — measured against four different
ways of normalising the step, the steadiest still swung 18-fold across the
approach. **On a printed hoarding that stretch is now covered by the artwork
cue above**, which found the same post on 75–100% of its pixels at every
distance — and with the Depth Anything model fetched, by the depth cue as
well, which holds 100% throughout the approach where MiDaS dipped to 19%. On
a digital screen playing its own content the artwork cue cannot run, so a
screen on an old-OpenCV machine still carrying only MiDaS keeps the blind
stretch: there, expect to use the dials on the part of the approach you care
about, or to insert across a shorter range.

### Where it stops

Worth knowing before pointing it at footage, since all of these are quiet
failures rather than errors:

- **Very thin railings.** The model sees a crop around the marked area scaled
  to 256×256, and a bar much thinner than a sixtieth of that crop's width is
  gone before the detector runs. Ordinary railings and posts are found; fine
  mesh and chain-link are not.
- **Obstructions covering more than about half the marked area.** Past that the
  obstruction is the majority of what was marked and is taken for the surface.
  The mask then empties out and the creative is painted over — the same result
  as leaving the option off, rather than something worse.
- **Artwork with strong depth in it.** See *Sensitivity* below: this is the
  one that shows up as a fault rather than a miss, and it has a control.
- **An obstruction close to a very distant surface.** Depth separation shrinks
  with distance; something a metre in front of a billboard forty metres away is
  below any honest threshold. A limit of monocular depth, not a setting — and
  see *Driving or walking past* above for what that costs across an approach.
- **Ground and objects just outside the marked area** that genuinely are nearer
  bleed a few pixels into the creative's edge, the same way the person mask
  does.

The mask is recomputed per frame with nothing carried between them, which keeps
preview and render identical but leaves a little shimmer along an obstruction's
edge on marginal cases.

## How the tracking works

The four corners are deliberately *not* tracked directly. Hand-placed corners
tend to land on occluding edges and flat, untextured areas — the worst possible
things to follow — which is what makes an insert slide off its surface.

Instead, `PlanarTracker` detects strong features across a whole region of the
frame — the area's interior by default, or the band around it in digital-screen
mode — follows them with pyramidal Lucas-Kanade optical flow, discards any that
fail a forward-backward consistency check, and fits a RANSAC homography to the
survivors. The corners are then carried by that homography. Because the fit is
over many points, a few bad tracks are outliers RANSAC rejects rather than a
quarter of the entire signal.

Each step is sanity-checked before it is accepted: a homography that folds the
area over, collapses it, or jumps it implausibly far in one frame is rejected,
and the tracker's state is left untouched so it can recover on the next frame
rather than being corrupted. A constant-velocity Kalman filter per corner
smooths the result and coasts the shape through brief occlusions.

The filter weights its own prediction and the tracker's measurement **equally**,
because a many-point RANSAC homography is a good measurement and deserves to be
trusted. Weighted the other way — as it was, by twenty to one in favour of the
prediction — a 0.57 px reading came out as **2.94 px** on a handheld pan, the
insert visibly swimming against the surface whenever the camera moved. That was
worse than useless rather than a poor trade: smoothing is meant to buy
steadiness in exchange for lag, and against a measurement deliberately
corrupted by 1.5 px of noise the old weighting gave 3.17 px where *no filtering
at all* gave 1.97 px, the lag exceeding the jitter it removed. Equal weights
give 0.69 px on real data and 1.78 px on the corrupted one — better on both
counts than either. Coasting through a dropout, which is what the filter is
genuinely for, is unchanged.

Results accumulate in `tracking_history`, a `{frame_index: [4 corners]}` map
that drives both the preview and the render. It saves and reloads **bit for
bit** — corners, curvature and every per-placement setting come back exactly as
they went in, and repeated save-edit-save cycles do not drift. Tracking a clip
is the expensive part of using this tool, and a save that quietly rounds is the
worst kind of fault: the file looks fine and nothing ever reports a problem.

### Corrections made by hand

Corners a person places are marked as corrections and kept apart from the ones
the tracker produced. Two things follow from knowing the difference.

**A later pass will not overwrite them.** It used to: re-tracking a stretch
wrote over every frame in it, so an afternoon spent fixing a difficult clip
frame by frame disappeared on the next pass with nothing on screen to say it
had happened.

**A correction now fixes the frames after it, not just its own.** Reaching a
corrected frame, a pass keeps the shape and re-anchors the tracker to it, so
tracking carries on from where the surface actually is. Fixing drift once and
playing on is the intended way to work; correcting every frame in turn is not.

Projects saved before this record no corrections, which is exactly what they
knew, and load unchanged.

### Steadying the tracked path

**Options → Steady the tracked path** fits a smooth path through the recorded
corners — for the preview and the render alike.

This is aimed squarely at the corrections above. Corners set by hand are right
on average and unsteady in between, because a hand is not accurate to the
pixel, and that unsteadiness is what makes an insert look fidgety after a lot
of careful work. Measured against a known camera move: the move itself
accelerates by **0.007 px** per frame, a tracker having a hard time by about
**0.25 px**, and a path corrected by hand to a typical pixel and a half by
**4.9 px**. The correction fixes where the shape sits and ruins how it moves.

Around each frame a low-order polynomial is fitted through the corners of the
frames nearby and read off at that frame. Reading forwards as well as backwards
is what separates this from the Kalman filter above: that one runs live and can
only see the past, so it lags, it never sees a frame corrected afterwards, and
it is thrown away and restarted from a standstill every time one is. Here the
whole clip is already known, so there is no lag to trade against.

On a hand-corrected path over a handheld shot, that takes the frame-to-frame
acceleration from **4.96 px to 0.64 px** and at the same time brings the shape
**closer** to where it belongs, 1.92 px to 1.08 px — most of what is removed
was never movement, so removing it is not a trade against accuracy.

A quadratic can already describe a pan, a zoom or a steady acceleration, so
real camera movement passes through untouched; only what cannot be described
that way is taken out. Stretches either side of a gap in the recording, or of a
jump too large to be a wobble, are fitted separately, so a deliberate
repositioning stays where it was put instead of being smeared over its
neighbours.

It is off by default, and it is a way of *reading* the recording rather than an
edit to it — the history is untouched, so it can be switched on to see what it
does and off again having cost nothing. Off by default because it is worth a
great deal on a path that was corrected by hand and costs a little on one that
was already steady: on a handheld shot tracked perfectly, fitting over 11
frames pulls the shape about 0.6 px off the surface it is stuck to. Wider
windows keep steadying the path — 21 frames reach 0.31 px of acceleration — at
a growing cost of that kind, which is why the default sits where it does.

Worth knowing about the tracker itself, since it is the other suspect: its own
contribution to fidget is small even on footage it finds difficult, and
steadying an untouched tracked path barely changes it (1.09 px to 1.08 px).
Where the tracker fails on hard footage it fails by being *wrong* rather than
unsteady — on deliberately punishing material, repetitive texture under heavy
blur and noise, it has been measured 33 px from the truth while reporting
success. No amount of smoothing recovers from that; re-marking the area, or
*Digital screen (track surroundings)*, is what helps there.

### Footage whose brightness will not sit still

Optical flow rests on one assumption: that a point keeps its brightness from
one frame to the next. A camera panning across a dim concourse onto a bright
screen breaks it on every frame, because its automatic exposure is hunting the
whole time — not an edge case for this tool but the ordinary condition of the
footage it is given.

Every frame therefore has its own brightness and contrast taken out before
anything is followed. On a pan with the exposure swinging by 40% the shape ends
**10.5 px** off its surface without this and **0.11 px** with it; at 70% it is
**21.1 px** against **0.13 px**. A steady shot pays about three hundredths of a
pixel for it.

Both statistics are the median and the spread about the median, rather than the
mean and standard deviation, because they have to describe the *lighting* and
not the contents. A large bright object crossing the frame — or the tracked
panel itself sliding out of it — moves a mean and a standard deviation enough
to fake an exposure change and undo the whole point: on a clip that pans a panel
out of shot and back, mean and standard deviation cost 7.6 px where the median
leaves the result untouched. They are computed from a 256-bin histogram, which
makes them exact over the whole frame for one pass, and applied through a lookup
table: 0.76 ms on a 1080p frame, against 14 ms doing the arithmetic per pixel.

### When the surface leaves the shot

Pan past a screen and it eventually goes out of frame. Following only the
screen's own texture stops the moment the screen does: the area parks against
the edge of frame and the insert bunches up against it, rather than carrying
on out of shot. Smoothing alone does not save it either — a constant-velocity
prediction cannot turn round, so it keeps sailing the old way while the camera
comes back, and by the time the screen returns it is hundreds of pixels out.

Once less than 30% of the area is still in frame, the tracker follows **the
camera** instead, detecting features across the whole frame *except* the area
itself. That exception matters: a screen or billboard is usually the strongest
texture in shot, so left in it takes most of the detector's budget and then
every one of those points leaves the frame together a moment later.

Whatever the area was doing over and above the camera before it left is
carried on at the same rate, so an advert on the side of a bus keeps moving
after the bus has gone while a fixed billboard does not. For a fixed surface
that leftover is only measurement noise, and carrying it for fifty frames
walks the shape off the panel by some 29px, so below a floor of 2px per frame
it is taken to be nothing.

Measured on a clip that pans a panel out of frame, holds it off, and pans back
onto it, the area stays within **1.7px** throughout and is back on the panel to
**0.5px** when it returns — against 898px for the same clip without this.
`TrackResult.following_camera` says when a step came from the camera rather
than from the surface, and these steps count as measurements rather than
failures, so the smoothing takes them instead of falling back on prediction.

## Footage shot under flickering lighting

*Options → Steady the lighting.*

An office lit by fluorescent or cheap LED fittings is lit by something that
brightens and dims a hundred or a hundred and twenty times a second. A camera
sampling that at 25 or 30fps beats against it, and the footage comes back
pulsing. That matters here beyond looking bad: flicker is one of the strongest
bottom-up attention cues there is, so a flickering fitting pulls fixations
whatever the advert is doing, and it contaminates the AOI data.

Two things can happen, and **the wrong correction does nothing at all**, so
Galileo works out which is present rather than offering one knob:

| | what it is | correction |
|---|---|---|
| **Whole-frame** | mains lighting beating against the frame rate; every pixel brightens and dims together | one gain per frame |
| **Banding** | a rolling shutter reads rows at different moments within one mains cycle, so bands crawl up the picture | one gain per row |

Applying the whole-frame correction to banding measured **11.61% before and
11.62% after** — that is the reason the kind is decided rather than asked. The
same holds for ffmpeg's `deflicker` filter, which is whole-frame only.

Both corrections work the same way: compare each frame, or each row of it,
against a smoothed version of its own history and scale to match. The
measurement is a median rather than a mean for the reason `level_gray` uses
one — somebody walking through the shot must not be mistaken for the lights
changing. Measured on matched clips: whole-frame flicker **20.6% → 0.9%**, and
banding **11.8% → 2.2%** against a floor of 2.3% that is the scene's own
row-to-row variation.

> The per-row correction subsumes the whole-frame one. A row's own level
> carries the global flicker as well as the band, so dividing by a smoothed
> version of it removes both — 20.6% flicker with no banding at all still came
> down to 0.9% under the per-row correction. The cheaper one exists because it
> is cheaper, not because it catches anything the other misses.

**Telling banding from ordinary motion** is the part that needs care. Every
clip's rows vary a little as things move through it, so size alone would
either miss real banding or call a busy scene banded. What separates them is
structure: banding is one crawling sinusoid, so in a spectrum over (time ×
row) its energy piles into a single peak, while motion is broadband and
scatters. Measured on matched clips: **13%** of the energy in one peak for
footage whose only row variation was a walking figure — and the same 13% for
whole-frame flicker, which is not banding either — against **94%** for real
banding.

**How it runs.** Opening a video takes a short look at ninety consecutive
frames from the middle and puts what it found in the menu item's tooltip
(*"Flicker found: 18.4% frame to frame at about 7Hz"*), which costs about
0.7s on a 1080p clip. Ticking the option measures the whole clip properly,
with a progress bar — around 5s for 900 frames of 1080p. The statistics are
taken from a shrunken copy of each frame, which is not a marginal
optimisation: at full size the medians cost four times what decoding does, and
the row gains stretch back to full height with no measurable loss.

The gains are then applied **as each frame is decoded**, not by writing a
corrected copy of the footage. Nothing is re-encoded, so there is no
generation loss and no second file the size of the original, and unticking the
option is free. It is applied before anything else sees the frame — so
tracking, the blend tool's measurements and the preview all work from the same
corrected plate the render will, rather than the advert being matched to
lighting that never reaches the file. Only the setting goes into the project
file; the gains are measured again on load, because per-row gains for a long
clip run to millions of numbers.

Tracking does not need this. The illumination levelling described above
already handles flicker: worst corner error was **1.94px on clean footage and
1.99px with 20% flicker**. This is for the stimulus, not the tracking.

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

### Several matches at once

The same poster often appears on more than one panel. The search finds **every**
instance rather than only the strongest: it fits a homography, removes the
keypoints that instance consumed, and fits again until nothing further stands
up. You are shown what was found with the evidence for each, and every one you
tick becomes its own placement.

The bar for the additional instances is set a little lower than for the first,
because the further ones are usually smaller and carry fewer keypoints.
Measured on a frame holding the same poster at three sizes, a threshold of 12
found two of them and 8 found all three, with no false positives on unrelated
footage or against a different reference.

### Using a clip as the reference

*Find Target from Image…* also accepts a **video**. A short handheld clip of a
billboard carries several angles and exposures, and a better chance that one of
them resembles the shot being searched. Frames are sampled across it, each
becomes a reference view, and the strongest result wins.

> **Frame the clip on the target**, exactly as a reference photo has to be
> cropped to it. Each sampled frame is taken to *be* the target, so its whole
> rectangle is what gets mapped into the footage. Hand over a wide shot in
> which the billboard sits in one corner and the match will succeed while the
> quad describes the entire scene — measured at 234 px off the panel despite a
> confident fit. The tool checks how much of the picture a match covers and
> warns when it looks like this happened.

A clip lands close but not as exactly as a cropped still — around 11 px on a
270 px panel in testing — because a handheld view is never framed to the pixel.
Treat it as a starting point and nudge the corners.

## The magnifier

Marking the area is the one part of the job where single pixels decide the
outcome. The tracker seeds its features from whatever is enclosed, so a corner
left a little *inside* a screen throws away the bezel detail that tracks best,
and one left a little *outside* picks up the wall behind and drags the insert
off during a pan. Neither is visible at video scale.

The magnifier appears whenever a handle is dragged, and can be pinned on with
its switch.

* **Every handle gets a view.** Four corners keep the familiar two-by-two, each
  corner in the part of the box where it actually sits on the video. Switch
  curved edges on and all twelve appear — a row per edge, holding that edge's
  corner and its two bend handles.
* **Dragging one gives it the whole widget.** That is the moment precision is
  wanted, and four thumbnails serve it worse than one clear view.
* **Press the panes button for one view of the whole area.** The grid shows
  each handle closely but never shows the shape they make together, which is
  what tells you whether the outline is following the screen. This frames the
  whole marked area with every handle drawn on it — all twelve once curving is
  on — and follows the outline rather than the corners, so a bend swinging wide
  of them is still inside the view. It fits whatever size the magnifier is, so
  enlarging it or filling the stage is what buys back the magnification.
* **The whole-area view shows the creative in place.** That view is about the
  insert as a whole — whether the advert sits on the screen and stays on it —
  so it magnifies the frame as composited, creatives and all, and updates as
  you drag. The tiles deliberately do not: a tile exists to line one handle up
  against the real edge of the screen underneath, and that edge is exactly what
  the creative covers.
* **Double-click, or press the corner button, to fill the video stage.**
  Twelve handles in a floating box a couple of hundred pixels wide leaves each
  one smaller than the thumbnail it replaced; filling the stage is what makes
  the crowded layouts usable. Double-click again to put it back.
* **Each view picks its own magnification** from how much room it has, so it
  always shows about 28 pixels of footage across its shorter side. A fixed
  number cannot work: a tile is anywhere from 50 to 500 pixels across, and 8×
  put *nine* pixels of footage in a default-sized tile — nothing to align
  against, and blocks so large the picture read as mush. Scroll to override
  it, and click the magnification badge to hand the choice back.
* **It shows true pixels.** Past 4× it stops interpolating and past 12× it
  rules off the source grid, because the question being asked is which pixel
  the edge falls on, and a blurred answer is no answer.
* **Drag the bottom-right corner** to resize it.
* **The area's outline is drawn through each view.** This is what makes the
  bend handles judgeable at all: a handle is placed correctly when the *curve*
  it produces sits on the screen's edge, and the curve is often nowhere near
  the handle itself.

> **The crosshair marks the true position.** The view is centred exactly on the
> handle and never slid back inside the frame. It used to be clamped to the
> frame while the crosshair stayed at the middle of its quadrant, so anywhere
> within half a view of a border — and for any handle carried off screen by
> tracking — it pointed at the wrong pixel.

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

## Several placements at once

A concourse or mall shot usually has more than one screen worth filling, and an
A/B test wants different creatives in them. The **Placements** panel holds as
many as you need. Each carries its own marked area, its own tracking, its own
creative and its own tracker, so one losing its surface cannot disturb the
others — and the digital-screen setting is per placement, since a single shot
can hold a printed poster alongside a digital screen.

Tick a placement to include it in the render; untick to leave it out without
losing its tracking. Editing always applies to the selected one, drawn with
solid handles; the rest stay visible as dashed outlines.

**A tracking pass follows the selected placement only**, which is why the
status reads *Tracking Left* rather than *Tracking on* once there is more than
one. Track them one at a time: mark the first, play through, correct it where
it slipped, then select the second and play through again. The second pass
cannot touch the first placement's tracking — it is read on every frame, so
that placement still moves with the footage on screen, and never written.

> Following every placement on every frame sounds like it would save a pass,
> and it did the opposite. The second play-through re-tracked the first
> placement from wherever its corners happened to be sitting and wrote the
> result over its history, discarding corrections made by hand — an 8px nudge
> came back at 8.3px away from where it had been put, across two thirds of the
> tracked frames. Nothing on screen said so, because the shape was redrawn
> from the same guess that had just replaced it.

The frame count beside each placement is how you see what a pass covered, so it
is brought up to date when the pass ends — switching tracking off, pausing, or
running off the end of the clip. Tracked frames also count as unsaved work now:
a pass used to leave the project looking clean, so tracking a whole clip and
closing the window lost it with no prompt, while nudging one corner by hand
asked to save.

**The library describes the selected placement.** Select a placement, press
*Insert* on a creative, and that creative fills that placement — switch
placement and the library redraws to describe the new one. The same creative
can therefore fill as many placements as you like, which is what an A/B test
against a control needs.

The AOI export writes **one CSV per placement**, named after it, because an
eye-tracking analysis has to tell the adverts apart.

## A creative that is itself a video

A still fills its area on every frame the area is tracked. A video has to be
told which of *its* frames belongs on which frame of the base clip, and the
answer is the simple one: **its first frame lands on the first frame the area
was tracked on**, and it plays forward from there, mapped by time rather than
by frame count so a creative shot at 30 fps into a 25 fps clip plays at its own
speed rather than 20% fast.

> This used to be anchored to wherever the playhead was sitting when *Insert*
> was pressed. In the ordinary order of work — mark the area, track the clip
> through, then fill it — that is the *last* frame of the shot, so every frame
> before it drew nothing and the render came back looking like the untouched
> base video with the advert visible only in its closing moments. Stills were
> never affected, which is what made it read as "video creatives do not
> render". The anchor is now worked out from the tracking, so tracking done
> after the creative went in moves it too.

A creative **shorter than the shot holds on its last frame** rather than
disappearing part-way through, and the render says so when it finishes:

> Completed (the creative video for Screen is shorter than the rendered range,
> so its last frame was held)

Rendering part of a clip picks the creative up where that part of the clip left
it, so frame 900 of the film carries the same frame of the advert whether you
rendered the whole thing or only the last hundred frames — what you scrubbed
past in the preview is what lands in the file.

## Blending the creative into the shot

A creative pasted in with correct geometry still reads as fake, because it is
*too clean*. The footage around it has been through a lens, a sensor and a
codec: unevenly lit, slightly soft, grainy, colour-cast by the ambient light,
and smeared when the camera moves.

The **Blend** tool measures those properties from the very pixels the creative
is about to cover and reproduces them — lighting falloff, colour cast,
softness, grain and motion blur — each on its own slider, updating the preview
live.

> **Colour is deliberately the weakest by default.** It is the one control that
> alters the creative's own colours, which is frequently the thing an ad test is
> measuring. On a cool-cast test scene, a colour strength of 0.5 moved the
> creative's mean colour by 84 units and turned white type cyan; 0.15 moved it
> by 27, nearly all of that from lighting rather than hue. Raise it only when
> belonging in the frame matters more than the exact hue.

## Shaping the creative

A creative goes into the tracked area as a flat rectangle. That is right for a
poster pasted flat on a board, but plenty of real placements are not: a banner
sagging on a fence, a print bowed across a curved panel, an advert on a screen
angled a few degrees away from the one the tracking found.

**Shape**, on the creative's own card in the library, bends and tilts the
artwork *in its own canvas*, before it is warped into the area. Keeping it
separate matters — the tracked area describes where the surface is, and should
not be nudged about to fake the look of the artwork on it. Tracking data stays
honest, and the same shape can be carried to a different placement or clip.

* **Turn**, **Tip** and **Rotate** move it in space. The tilt is projected
  through a pinhole camera, so the foreshortening is real perspective rather
  than a squash — the near edge grows as the far edge shrinks. **Perspective**
  sets how strongly: low is nearly a flat squash, high exaggerates depth.
* **Bow across** and **Bow down** wrap it on a cylinder. Positive bulges the
  middle towards the viewer.
* **Curve shading** darkens the surface as it turns away, by the cosine of the
  angle it has turned through.

> **Shading is what makes a bow read as a curve.** Moving the texture about is
> a weak cue on its own — a compressed edge looks much like a squashed flat
> sheet, and without shading a bowed creative is very nearly indistinguishable
> from a flat one. It is therefore on by default. Drop it to 0 if the
> creative's own colours are what the test is measuring.

> **The two bow directions look more alike than you might expect.** Both
> curvatures turn their ends away by the same angle, so both crowd the artwork
> towards its edges; what separates them is only which part is nearer the
> camera and so magnified. That is the geometry, not a limitation — a convex
> panel and a hollow one really do read similarly head-on.

### A curved screen needs both controls

The two halves of a curved screen are set separately, because they are
separate things:

| What you are describing | Control |
|---|---|
| The screen's **outline** — its edges bow on the footage | **Curved edges** on the tracked area |
| The **artwork** on it — texture crowding towards the edges, light falling away | **Shape → Bow** on the creative |

Bow alone leaves a rectangular silhouette, which is right for a curved panel
seen straight on but not for one seen from above or to the side. Use curved
edges to bend the outline to the screen in the footage, and bow to make the
artwork sit on that curve. Together they read as a barrel-fronted panel;
either alone falls short of it.

**Fill the area** keeps the creative covering every pixel of the tracked area,
which is on by default. A turn shrinks the artwork inside its own canvas, and
what is left over is not empty — it is the tracked surface showing through,
which is the one thing an insert must never do. Filling grows the artwork until
the area is entirely inside it and crops the overhang instead. Measured across
turns, tips, rotations and combinations of all three, that takes the gap from
as much as 72% of the area down to none of it. The cost is the edges of the
artwork, so it can be turned off where those matter more than the seal.

Shaping is applied before brightness, contrast and blending, so what is matched
to the shot is the shape that will actually be laid down.

> **Shaping is not redone frame by frame.** A 1080p creative took 98 ms to
> shape, which capped preview playback at ten frames a second and added the
> same to every frame of a render. Most of that was the curve shading widening
> the whole image to floating point; done a channel at a time through OpenCV's
> saturating multiply it is 47 ms. And for a still creative with settings
> nobody is dragging, the answer is identical every frame, so it is kept — the
> repeat costs 0.002 ms. A creative *video* hands over a new picture each frame
> and so is redone, which is correct: there is genuinely new work to do.

## Audio and output quality

Renders are encoded with **ffmpeg (x264, CRF 17)** when it is available, with
the source audio muxed in the same pass.

Without ffmpeg the render falls back to OpenCV's built-in `mp4v`, which is
silent — `VideoWriter` cannot write audio — and measurably lossy, around 34 dB
PSNR. That is enough to undo the subtler photometric matching: in one measured
render, grain added at a sigma of 2.0 came back at 0.45, and a flat creative
suffers worst because it is cheap to encode and gets quantised hardest. The
render still succeeds and the completion message says which happened.

Installing ffmpeg is therefore worth it for picture quality, not only audio.

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

`galileo_core.py` holds the algorithms as plain NumPy/OpenCV with no Qt imports.
`Galileo_Insertion_Tool_1.0.0.py` is the application on top of it. The split
matters for more than tidiness: the preview and the renderer call the *same*
compositing function, so what you approve on screen is what gets written to the
file, and the algorithms can be tested without a display.

| Component | Role |
| --- | --- |
| `galileo_core.PlanarTracker` | Feature-based planar tracking with RANSAC and sanity checks |
| `galileo_core.ReferenceMatcher` | Locates a target in the footage from a reference image |
| `galileo_core.PersonSegmenter` | Segments people so the insert can go behind them |
| `galileo_core.DepthOcclusionSegmenter` | Finds anything else in front of the surface, by depth |
| `galileo_core.Region` | Four corners plus per-edge curvature |
| `galileo_core.composite_region` | Alpha-correct perspective/curved warp and blend |
| `galileo_core.DepthSettings` | The dials behind *Draw behind obstructions* |
| `galileo_core.SurfacePlate` | The panel's own artwork as an occlusion reference |
| `galileo_core.smooth_tracking` | Fits a smooth path through the recorded corners |
| `galileo_core.interpolate_tracking` | Fills the gaps between tracked frames |
| `galileo_core.remux_audio` | Copies the source audio onto a finished render |
| `MainWindow` | Frameless main window, menus, load/save/render actions |
| `CentralPanel` | Video playback, frame stepping, the tracking loop |
| `TrackingOverlay` | The area, corner and curve handles, live preview |
| `MagnifierWidget` | Zoomed views of every handle, for exact placement |
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
drives the actual Qt widgets through loading, tracking and a full render, and
`tests/test_digital_screens.py` pins down both the animated-screen failure and
its fix. `tests/test_offscreen.py` slides a window over a wide scene so the
camera's motion and the surface's position are both known exactly, and asserts
the area neither stalls at the edge of frame nor runs away while out of it.
`tests/test_creative_video.py` renders a creative whose every frame states
which frame it is, so the frame that landed can be read straight back out of
the file, and pins down where a video creative starts, that a render of part of
a clip picks it up in the right place, and that one shorter than the shot holds
rather than vanishing. `tests/test_deflicker.py` builds both kinds of flicker
over the same footage
and asserts the right correction is chosen for each — including that the
whole-frame one demonstrably does *not* fix banding, which is why the choice
is made rather than offered — and that a preview that has been steadied is
rendered to a file that is steady too.

`tests/test_steadiness.py` measures the wobble itself rather than accuracy —
the mean size of the shape's frame-to-frame acceleration — and pins down both
that a corrected path is hundreds of times less steady than the camera move it
describes and that fitting a path through it removes that without pulling the
shape away from where it belongs. It also asserts that asking for one frame's
steadied corners, which is what the preview does as it goes, gives exactly what
fitting the whole clip gives, which is what the renderer does.

Occlusion tests that need a model file skip cleanly when it has not been
fetched; the compositing side is tested with hand-made masks either way, and
`tests/test_depth_occlusion.py` states what turns a depth map into a mask using
maps built by hand — an obstruction across the middle, a surface seen at a
steep angle, sky showing past a loosely marked edge — so the rules hold whether
or not the 64 MB model has been downloaded.
