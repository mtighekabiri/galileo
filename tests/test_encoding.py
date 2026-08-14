"""Video output quality.

OpenCV's built-in encoder measured 34 dB PSNR on a rendered frame and wiped out
added grain (sigma 2.0 -> 0.45), hitting a flat creative hardest because it is
cheap to encode and gets quantised hardest. For a stimulus whose whole purpose
is that the insert not stand out, that undoes the photometric matching. ffmpeg
is used when present; these tests pin down both paths and the handover.
"""

import os
import subprocess

import cv2
import numpy as np
import pytest

import lumen_core as core


@pytest.fixture
def frames():
    rng = np.random.default_rng(4)
    base = rng.integers(60, 200, (120, 160, 3), dtype=np.uint8)
    return [base.copy() for _ in range(6)]


class TestFallbackPath:
    """Behaviour when ffmpeg is absent, which must always still produce a file."""

    @pytest.fixture(autouse=True)
    def no_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(core, "find_binary", lambda name: None)

    def test_writes_a_readable_file(self, tmp_path, frames):
        path = str(tmp_path / "out.mp4")
        writer = core.FrameWriter(path, 25, (160, 120))
        assert not writer.using_ffmpeg
        for frame in frames:
            writer.write(frame)
        writer.close()

        capture = cv2.VideoCapture(path)
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
        capture.release()

    def test_reports_the_quality_caveat(self, tmp_path, frames):
        writer = core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120))
        caveats = writer.close()
        assert any("ffmpeg" in c for c in caveats)

    def test_abort_leaves_no_handle_open(self, tmp_path, frames):
        writer = core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120))
        writer.write(frames[0])
        writer.abort()
        assert writer.writer is None

    def test_unwritable_path_raises(self, tmp_path):
        with pytest.raises(IOError):
            core.FrameWriter("/nonexistent-dir/out.mp4", 25, (160, 120))


class TestFfmpegCommand:
    """The ffmpeg path cannot be exercised here (no ffmpeg binary available),
    so the command it builds is asserted directly."""

    @pytest.fixture
    def captured(self, monkeypatch, tmp_path):
        monkeypatch.setattr(core, "find_binary",
                            lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)
        monkeypatch.setattr(core, "source_has_audio", lambda path: True)
        recorded = {}

        class FakeProcess:
            returncode = 0

            def __init__(self):
                self.stdin = __import__("io").BytesIO()
                self.stderr = __import__("io").BytesIO(b"")

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

        def fake_popen(cmd, **kwargs):
            recorded["cmd"] = cmd
            return FakeProcess()

        monkeypatch.setattr(core.subprocess, "Popen", fake_popen)
        return recorded

    def test_encodes_with_x264_at_a_high_quality(self, captured, tmp_path):
        core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120), crf=17)
        cmd = captured["cmd"]
        assert "libx264" in cmd
        assert cmd[cmd.index("-crf") + 1] == "17"
        assert "yuv420p" in cmd

    def test_accepts_raw_bgr_frames_of_the_right_size(self, captured, tmp_path):
        core.FrameWriter(str(tmp_path / "o.mp4"), 30, (640, 480))
        cmd = captured["cmd"]
        assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24"
        assert cmd[cmd.index("-s") + 1] == "640x480"

    def test_audio_is_muxed_in_the_same_pass(self, captured, tmp_path):
        """Encoding and audio in one pass replaces a second remux of the file."""
        core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120),
                         audio_source="source.mp4", audio_start=2.0, duration=4.0)
        cmd = captured["cmd"]
        assert "source.mp4" in cmd
        assert "1:a:0?" in cmd
        assert cmd[cmd.index("-ss") + 1].startswith("2.0")
        assert cmd[cmd.index("-t") + 1].startswith("4.0")

    def test_no_audio_requested_means_no_audio_mapping(self, captured, tmp_path):
        core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120))
        assert "1:a:0?" not in captured["cmd"]

    def test_silent_source_is_reported_not_mapped(self, monkeypatch, captured, tmp_path):
        monkeypatch.setattr(core, "source_has_audio", lambda path: False)
        writer = core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120),
                                  audio_source="silent.mp4")
        assert "1:a:0?" not in captured["cmd"]
        assert any("no audio" in c for c in writer.caveats)

    def test_frames_are_piped_to_stdin(self, captured, tmp_path, frames):
        writer = core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120))
        writer.write(frames[0])
        assert writer.process.stdin.getvalue() == frames[0].tobytes()


class TestFfmpegFailureFallsBack:
    def test_popen_failure_uses_opencv(self, monkeypatch, tmp_path, frames):
        monkeypatch.setattr(core, "find_binary", lambda name: "/usr/bin/ffmpeg")

        def boom(cmd, **kwargs):
            raise OSError("no such binary")

        monkeypatch.setattr(core.subprocess, "Popen", boom)
        writer = core.FrameWriter(str(tmp_path / "o.mp4"), 25, (160, 120))

        assert not writer.using_ffmpeg
        for frame in frames:
            writer.write(frame)
        caveats = writer.close()
        assert any("could not be started" in c for c in caveats)
        assert os.path.exists(str(tmp_path / "o.mp4"))


class TestQualityMatters:
    """Documents the measurement that motivated all of the above."""

    def _encode_roundtrip(self, tmp_path, frame, fourcc="mp4v"):
        height, width = frame.shape[:2]
        path = str(tmp_path / f"probe_{fourcc}.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 25,
                                 (width, height))
        assert writer.isOpened()
        for _ in range(8):
            writer.write(frame)
        writer.release()

        capture = cv2.VideoCapture(path)
        for _ in range(5):
            ok, decoded = capture.read()
        capture.release()
        assert ok
        return decoded

    @staticmethod
    def _psnr(a, b):
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        return 10 * np.log10(255 * 255 / max(mse, 1e-9))

    def test_builtin_codec_is_visibly_lossy(self, tmp_path):
        """The measurement that motivated piping through ffmpeg.

        Grain loss depends on how many bits the encoder decides the shot is
        worth overall, so it is not a stable thing to assert on. What is stable
        is that the built-in codec is lossy enough to matter: on a smooth,
        realistic frame it lands around 34 dB, which is well inside the range
        where added grain and gentle photometric matching are degraded.
        """
        rng = np.random.default_rng(11)
        small = rng.integers(60, 190, (20, 27, 3), dtype=np.uint8)
        frame = cv2.resize(small, (320, 240), interpolation=cv2.INTER_CUBIC)
        frame[80:160, 110:210] = (30, 40, 200)      # a flat creative

        decoded = self._encode_roundtrip(tmp_path, frame)
        psnr = self._psnr(frame, decoded)
        assert psnr < 45.0, f"expected a visibly lossy round trip, got {psnr:.1f} dB"

    def test_a_lossless_codec_shows_the_difference_is_the_encoder(self, tmp_path):
        """Rules out the harness: the same frame survives a lossless codec."""
        rng = np.random.default_rng(11)
        small = rng.integers(60, 190, (20, 27, 3), dtype=np.uint8)
        frame = cv2.resize(small, (320, 240), interpolation=cv2.INTER_CUBIC)
        frame[80:160, 110:210] = (30, 40, 200)

        lossy = self._psnr(frame, self._encode_roundtrip(tmp_path, frame, "mp4v"))
        lossless = self._psnr(frame, self._encode_roundtrip(tmp_path, frame, "FFV1"))
        assert lossless > lossy + 20, f"lossless {lossless:.1f} vs lossy {lossy:.1f}"
