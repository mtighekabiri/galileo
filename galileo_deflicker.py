"""Hold the lighting steady in footage shot under flickering lamps.

An office lit by fluorescent or cheap LED fittings is lit by something that
brightens and dims a hundred or a hundred and twenty times a second. A camera
sampling that at 25 or 30 frames per second beats against it, and the footage
comes back pulsing. Two things can happen, and they need different corrections:

``frame``
    Mains lighting beating against the frame rate, so every pixel of a frame
    brightens and dims together.

``rows``
    A rolling shutter reads the sensor's rows at different moments within one
    mains cycle, so bands crawl up the picture instead. A whole-frame
    correction leaves this completely untouched -- measured at 11.61% before
    and 11.62% after -- which is why this module works out which is present
    rather than offering one knob.

Both are corrected the same way: compare each frame, or each row of it,
against a smoothed version of its own history and scale to match. The
measurement is a median rather than a mean for the reason
:func:`galileo_core.level_gray` uses one -- somebody walking through the shot
must not be mistaken for the lights changing.

The per-row correction subsumes the whole-frame one. A row's own level carries
the global flicker as well as the band, so dividing by a smoothed version of
it removes both: on footage with 20.6% whole-frame flicker and no banding at
all, the per-row correction still brought it down to 0.9%. ``frame`` exists
because it is much cheaper, not because it catches anything ``rows`` misses.

Why this is worth doing for research footage: flicker is one of the strongest
bottom-up attention cues there is, so a flickering fitting pulls fixations
whatever the advert is doing. Correcting the plate *before* tracking and
insertion also gives the blend tool a stable reference to measure against,
instead of one chasing the lights.
"""

from collections import namedtuple

import cv2
import numpy as np

#: Frames the reference brightness is averaged over. Long enough to span
#: several flicker cycles, short enough to follow a real lighting change --
#: walking out of a corridor into a bright atrium must still look like that.
DEFAULT_WINDOW = 21

#: Never brighten a frame by more than this. A very dark frame divided into a
#: bright reference asks for a huge gain, which turns sensor noise into
#: confetti; better to leave that frame slightly dark.
DEFAULT_MAX_GAIN = 1.7

#: How much of a frame-to-frame brightness swing counts as flicker, as a
#: percentage of the clip's average brightness. Steady footage measures around
#: a tenth of a percent.
FLICKER_SIZE = 1.5

#: A clip's rows always vary a little as things move through it, so banding is
#: called only when that variation is both sizeable and concentrated into the
#: single peak a crawling sinusoid makes. Measured on matched clips: 13% of the
#: energy in one peak for footage whose only row variation was a walking
#: figure -- and the same 13% for whole-frame flicker, which is not banding
#: either -- against 94% for real banding. The gap is wide enough that the
#: exact threshold hardly matters.
BAND_SIZE, BAND_SHAPE = 3.0, 40.0

#: Frames read to decide whether a clip flickers. They have to be consecutive
#: -- flicker is a property of one frame against the next -- so this is a
#: window, not a spread of samples across the clip. Ninety is three and a half
#: seconds at 25fps, which spans many cycles of anything mains-driven; all
#: three kinds were still named correctly down at forty, so this leaves room
#: for footage less obliging than the test clips while keeping the pause on
#: opening a 1080p video under a second.
SCAN_FRAMES = 90


FlickerReport = namedtuple(
    "FlickerReport",
    "frames fps whole band_size band_shape hz kind")


def _smooth(series, window):
    """Centred moving average along axis 0, reflected at the ends.

    Reflected rather than zero-padded: padding with zeros drags the reference
    towards black at both ends of the clip, and the correction then brightens
    the first and last few frames to compensate.
    """
    series = np.asarray(series, np.float64)
    window = max(3, int(window) | 1)
    pad = window // 2
    widths = [(pad, pad)] + [(0, 0)] * (series.ndim - 1)
    padded = np.pad(series, widths, mode="reflect")
    kernel = np.ones(window) / window
    if series.ndim == 1:
        return np.convolve(padded, kernel, mode="valid")
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid")
                     for i in range(series.shape[1])], axis=1)


#: The statistics are measured on a shrunken copy. Not an optimisation at the
#: margin: on 1080p footage the medians cost four times what decoding the
#: frame does -- 1.74s against 0.44s over 150 frames -- and shrinking first
#: brings the pair down to 0.49s. Nothing is lost, because what is being
#: measured is the lighting: an overall level, and bands that run a few cycles
#: down the frame at most. Enough rows are kept to resolve those many times
#: over, and the row gains are stretched back to the full height afterwards.
STAT_WIDTH, STAT_HEIGHT = 320, 540


def _luminance(frame):
    """One frame's overall brightness and its per-row brightness."""
    height, width = frame.shape[:2]
    if width > STAT_WIDTH or height > STAT_HEIGHT:
        frame = cv2.resize(frame, (min(width, STAT_WIDTH),
                                   min(height, STAT_HEIGHT)),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray)), np.median(gray, axis=1)


def _stretch_rows(gains, height):
    """Resample per-row gains to the footage's real height.

    A gain curve for a band running a few cycles down the frame is smooth by
    construction, so interpolating it back up adds no error worth the name --
    measured at 11.78% banding before and 2.24% after, against 2.23% when the
    same correction was measured at full height.
    """
    gains = np.asarray(gains, np.float64)
    if gains.shape[1] == height:
        return gains
    source = np.linspace(0.0, 1.0, gains.shape[1])
    target = np.linspace(0.0, 1.0, height)
    return np.stack([np.interp(target, source, row) for row in gains])


def whole_frame_amount(frame_lum):
    """Frame-to-frame brightness swing, as a percentage of the average."""
    frame_lum = np.asarray(frame_lum, np.float64)
    if frame_lum.size < 2:
        return 0.0
    return 100.0 * float(np.std(np.diff(frame_lum))) / max(frame_lum.mean(), 1e-6)


def banding_amount(row_lum):
    """How strongly the rows band, and how band-shaped that variation is.

    Two numbers, because size alone cannot tell banding from ordinary motion.
    Every clip has some row-to-row variation -- a figure walking through the
    shot changes the rows it crosses -- so a threshold on size alone either
    misses real banding or calls a busy scene banded.

    What separates them is structure. Banding is one crawling sinusoid: a
    single spatial frequency sliding at a steady rate, so in a spectrum over
    (time x row) its energy piles into one peak, while motion is broadband and
    scatters. The second number is the share of the energy in that peak, and
    it is the one worth trusting.

    The frame's own overall level is divided out first. Without that a purely
    whole-frame flicker reads as banding as well -- every row does brighten
    together, after all -- and the wrong, more expensive correction would be
    chosen for a problem the cheap one solves.
    """
    row_lum = np.asarray(row_lum, np.float64)
    if row_lum.ndim != 2 or row_lum.shape[0] < 8:
        return 0.0, 0.0

    level = np.maximum(row_lum.mean(axis=1, keepdims=True), 1e-6)
    residual = row_lum / level
    residual = residual - residual.mean(axis=0, keepdims=True)
    size = 100.0 * float(residual.std())

    window = (np.hanning(residual.shape[0])[:, None]
              * np.hanning(residual.shape[1]))
    spectrum = np.abs(np.fft.rfft2(residual * window)) ** 2
    spectrum[0, 0] = 0.0
    total = spectrum.sum()
    if total <= 0:
        return size, 0.0
    # A peak is a few bins wide once windowed, so count its neighbourhood.
    peak = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
    rows = slice(max(0, peak[0] - 1), peak[0] + 2)
    cols = slice(max(0, peak[1] - 1), peak[1] + 2)
    return size, 100.0 * float(spectrum[rows, cols].sum() / total)


def dominant_hz(frame_lum, fps):
    """The strongest beat in the brightness, which hints at the cause.

    Mains lighting brightens twice per cycle -- 100Hz on 50Hz mains, 120Hz on
    60Hz -- and aliases down to whatever the frame rate leaves of it. A beat
    at some unrelated rate is more likely a dimmer or a failing ballast.
    """
    frame_lum = np.asarray(frame_lum, np.float64)
    if frame_lum.size < 8:
        return None, 0.0
    signal = frame_lum - frame_lum.mean()
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / max(fps, 1e-6))
    spectrum[0] = 0.0
    total = spectrum.sum()
    peak = int(np.argmax(spectrum))
    return float(freqs[peak]), float(spectrum[peak] / total if total else 0.0)


def classify(frame_lum, row_lum, fps):
    """Name what kind of flicker these measurements describe."""
    whole = whole_frame_amount(frame_lum)
    size, shape = banding_amount(row_lum)
    banded = size > BAND_SIZE and shape > BAND_SHAPE
    flickering = whole > FLICKER_SIZE
    hz, _ = dominant_hz(frame_lum, fps)
    if banded:
        kind = "rows"
    elif flickering:
        kind = "frame"
    else:
        kind = "steady"
    return FlickerReport(len(frame_lum), fps, whole, size, shape,
                         hz if flickering else None, kind)


def scan(path, sample=SCAN_FRAMES):
    """Decide quickly whether a clip flickers, and how.

    Reads a window from the middle rather than the whole clip: this runs when
    a video is opened, and a full pass over a long clip would be a visible
    stall on every load for footage that mostly turns out to be fine. The
    frames have to be consecutive, since flicker is a property of one frame
    against the next.

    Returns a :class:`FlickerReport` whose ``kind`` is ``"steady"`` when there
    is nothing to correct, or ``None`` if the clip could not be read.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample = max(8, int(sample))
        if total > sample:
            capture.set(cv2.CAP_PROP_POS_FRAMES, (total - sample) // 2)

        frame_lum, row_lum = [], []
        for _ in range(sample):
            ok, frame = capture.read()
            if not ok:
                break
            overall, rows = _luminance(frame)
            frame_lum.append(overall)
            row_lum.append(rows)
        if len(frame_lum) < 8:
            return None
        return classify(frame_lum, np.float64(row_lum), fps)
    finally:
        capture.release()


class Deflicker:
    """Gains that hold a clip's lighting steady, ready to apply per frame.

    Measured once over the whole clip and then applied as each frame is
    decoded, rather than by writing a corrected copy: nothing is re-encoded,
    so there is no generation loss, no second file the size of the footage,
    and switching it off costs nothing.
    """

    def __init__(self, gains, mode, height=None):
        self.gains = np.asarray(gains, np.float32)
        self.mode = mode
        #: Which frame height the per-row gains were measured at, so a gain
        #: vector is never stretched across footage it does not describe.
        self.height = height

    def __len__(self):
        return len(self.gains)

    def __repr__(self):
        return f"Deflicker({self.mode!r}, {len(self.gains)} frames)"

    @classmethod
    def measure(cls, path, mode="auto", window=DEFAULT_WINDOW,
                max_gain=DEFAULT_MAX_GAIN, progress=None):
        """Work out the gains for a whole clip.

        ``progress`` is called as ``progress(done, total)`` and may return
        False to cancel, in which case this returns None.
        """
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_lum, row_lum, height = [], [], None
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                height = frame.shape[0]
                overall, rows = _luminance(frame)
                frame_lum.append(overall)
                row_lum.append(rows)
                if progress is not None and not progress(len(frame_lum), total):
                    return None
        finally:
            capture.release()

        if len(frame_lum) < 8:
            return None

        frame_lum = np.float64(frame_lum)
        row_lum = np.float64(row_lum)
        if mode == "auto":
            mode = classify(frame_lum, row_lum, fps).kind
        if mode == "steady":
            return None

        clamp = (1.0 / max(max_gain, 1.0 + 1e-6), max_gain)
        measured = row_lum if mode == "rows" else frame_lum
        gains = np.clip(_smooth(measured, window) / np.maximum(measured, 1e-6),
                        *clamp)
        if mode == "rows" and height is not None:
            gains = _stretch_rows(gains, height)
        return cls(gains, mode, height)

    def apply(self, frame, index):
        """One frame with its gain applied. Anything unmeasured is untouched."""
        if frame is None or not 0 <= index < len(self.gains):
            return frame
        if self.mode == "rows":
            if self.height is not None and frame.shape[0] != self.height:
                # Footage that is not what was measured -- a scaled render, or
                # a different clip. Guessing a mapping would silently stripe it.
                return frame
            column = self.gains[index].reshape(-1, 1, 1)
            return np.clip(frame.astype(np.float32) * column,
                           0, 255).astype(np.uint8)
        return cv2.convertScaleAbs(frame, alpha=float(self.gains[index]))
