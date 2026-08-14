"""Tracking-history interpolation and the audio remux helper."""

import numpy as np
import pytest

import galileo_core as core


def quad_at(x):
    return np.float32([[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]])


class TestInterpolation:
    def test_fills_the_gap_between_two_keyframes(self):
        dense = core.interpolate_tracking({0: quad_at(0), 10: quad_at(100)})
        assert sorted(dense) == list(range(11))
        assert np.allclose(dense[5], quad_at(50), atol=1e-4)

    def test_does_not_extrapolate_past_the_ends(self):
        dense = core.interpolate_tracking({5: quad_at(0), 8: quad_at(30)})
        assert min(dense) == 5 and max(dense) == 8

    def test_keyframes_are_preserved_exactly(self):
        history = {0: quad_at(0), 4: quad_at(17), 9: quad_at(90)}
        dense = core.interpolate_tracking(history)
        for key, quad in history.items():
            assert np.allclose(dense[key], quad, atol=1e-5)

    def test_handles_uneven_spacing(self):
        dense = core.interpolate_tracking({0: quad_at(0), 2: quad_at(20),
                                           20: quad_at(200)})
        assert np.allclose(dense[1], quad_at(10), atol=1e-4)
        assert np.allclose(dense[11], quad_at(110), atol=1e-4)

    def test_range_can_be_narrowed(self):
        dense = core.interpolate_tracking({0: quad_at(0), 20: quad_at(200)},
                                          start=5, end=10)
        assert min(dense) == 5 and max(dense) == 10

    def test_single_keyframe_yields_just_that_frame(self):
        dense = core.interpolate_tracking({7: quad_at(3)})
        assert list(dense) == [7]

    def test_empty_history_is_empty(self):
        assert core.interpolate_tracking({}) == {}

    def test_malformed_entries_are_skipped(self):
        dense = core.interpolate_tracking({0: quad_at(0), 3: None,
                                           5: [[1, 2], [3, 4]], 10: quad_at(100)})
        assert sorted(dense) == list(range(11))

    def test_string_keys_are_accepted(self):
        """Histories loaded from JSON come back with string keys."""
        dense = core.interpolate_tracking({"0": quad_at(0), "4": quad_at(40)})
        assert sorted(dense) == [0, 1, 2, 3, 4]

    def test_interpolation_beats_holding_the_last_value(self):
        """Holding freezes the insert; interpolation keeps it moving."""
        history = {0: quad_at(0), 10: quad_at(100)}
        dense = core.interpolate_tracking(history)
        held = quad_at(0)
        assert np.linalg.norm(dense[5] - quad_at(50)) < np.linalg.norm(held - quad_at(50))


class TestSerialisation:
    def test_history_converts_to_plain_lists(self):
        out = core.history_to_lists({0: quad_at(0), 2: quad_at(20)})
        assert out == {0: [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                       2: [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0]]}
        import json
        json.dumps(out)   # must be serialisable


class TestAudio:
    def test_reports_whether_ffmpeg_is_present(self):
        assert isinstance(core.has_ffmpeg(), bool)

    def test_remux_declines_gracefully_without_ffmpeg(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core.shutil, "which", lambda name: None)
        out = tmp_path / "out.mp4"
        assert core.remux_audio("a.mp4", "b.mp4", str(out)) is False
        assert not out.exists()

    def test_source_has_audio_is_false_without_ffprobe(self, monkeypatch):
        monkeypatch.setattr(core.shutil, "which", lambda name: None)
        assert core.source_has_audio("anything.mp4") is False

    def test_remux_reports_failure_when_ffmpeg_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(core.shutil, "which", lambda name: "/usr/bin/ffmpeg")

        class Result:
            returncode = 1
            stderr = b"boom"

        monkeypatch.setattr(core.subprocess, "run", lambda *a, **k: Result())
        assert core.remux_audio("a.mp4", "b.mp4", str(tmp_path / "o.mp4")) is False

    def test_remux_builds_a_command_with_optional_audio_mapping(self, tmp_path, monkeypatch):
        """'1:a:0?' keeps a silent source from failing the whole render."""
        monkeypatch.setattr(core.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        captured = {}

        class Result:
            returncode = 0
            stderr = b""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return Result()

        monkeypatch.setattr(core.subprocess, "run", fake_run)
        out = tmp_path / "o.mp4"
        out.write_bytes(b"x")

        core.remux_audio("render.mp4", "source.mp4", str(out), start_time=2.0)
        assert "1:a:0?" in captured["cmd"]
        assert "-ss" in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index("-ss") + 1].startswith("2.0")
