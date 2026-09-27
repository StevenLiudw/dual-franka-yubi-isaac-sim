"""Video and joint-state writers for one dual-Franka Isaac Sim episode.

The runner samples at the requested recording rate, 30 Hz by default. Each
video frame and CSV sample share an integer sample index, including sample
zero immediately after reset.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


JOINT_COLUMNS = (
    "episode",
    "sample_index",
    "phase",
    "policy_step",
    "physics_time_s",
    "side",
    "joint_name",
    "joint_kind",
    "position",
    "velocity",
    "position_unit",
    "velocity_unit",
)


class JointStateWriter:
    """Write every named DOF of both arms at each sampled simulator state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=JOINT_COLUMNS)
        self._writer.writeheader()
        self.sample_count = 0
        self.row_count = 0

    def write(self, episode: int, sample_index: int, phase: str, policy_step: int, observation: dict) -> None:
        for side in ("left", "right"):
            robot = observation["robots"][side]
            names = robot["joint_names"]
            positions = robot["joint_positions"]
            velocities = robot["joint_velocities"]
            if not (len(names) == len(positions) == len(velocities)):
                raise ValueError(f"{side} joint names, positions, and velocities have different lengths")
            for name, position, velocity in zip(names, positions, velocities):
                finger = name.startswith("yubi_finger")
                self._writer.writerow(
                    {
                        "episode": episode,
                        "sample_index": sample_index,
                        "phase": phase,
                        "policy_step": policy_step,
                        "physics_time_s": observation["physics_time_s"],
                        "side": side,
                        "joint_name": name,
                        "joint_kind": "gripper" if finger else "arm",
                        "position": float(position),
                        "velocity": float(velocity),
                        "position_unit": "rad",
                        "velocity_unit": "rad/s",
                    }
                )
                self.row_count += 1
        self.sample_count += 1
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "JointStateWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class VideoWriter:
    """Stream rendered RGB frames to an H.264 MP4 through system ffmpeg."""

    def __init__(self, path: Path, width: int, height: int, fps: int) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("Recording video requires ffmpeg on PATH")
        if width <= 0 or height <= 0 or width % 2 or height % 2 or fps <= 0:
            raise ValueError("Video width and height must be positive even integers; fps must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Video already exists: {self.path}")
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self._error_log = tempfile.TemporaryFile(mode="w+b")
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-n",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._error_log,
        )

    def write(self, rgb: object) -> None:
        if rgb is None:
            raise RuntimeError("Isaac Camera returned no RGB frame after rendering")
        frame = np.asarray(rgb)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected RGB frame {(self.height, self.width, 3)}, got {frame.shape}")
        if frame.dtype != np.uint8:
            if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
                raise ValueError(f"RGB frame has invalid dtype/data: {frame.dtype}")
            maximum = float(frame.max())
            frame = np.clip(frame * 255 if maximum <= 1 else frame, 0, 255).astype(np.uint8)
        if int(frame.max()) - int(frame.min()) < 2:
            raise RuntimeError("Isaac Camera rendered a blank frame; check camera pose and RTX renderer")
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg video input is closed")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(f"ffmpeg stopped while encoding {self.path}") from exc
        self.frame_count += 1

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        try:
            return_code = self._process.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            self._process.wait()
            raise RuntimeError(f"ffmpeg did not finish encoding {self.path}") from exc
        self._error_log.seek(0)
        error = self._error_log.read().decode("utf-8", errors="replace").strip()
        self._error_log.close()
        if return_code != 0 or self.frame_count == 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {error or f'exit {return_code}, {self.frame_count} frames'}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
