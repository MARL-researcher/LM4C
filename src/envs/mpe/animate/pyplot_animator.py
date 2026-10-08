"""Matplotlib animation for MPE episodes.

The public constructor intentionally keeps the original eight positional
arguments. Optional generated icons and physical entity sizes can be passed
by keyword, allowing existing callers to keep using ``save_animation(path)``.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import List, Optional, Sequence, Tuple

import numpy as np

import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from .icons import content_frac


@lru_cache(maxsize=4)
def _mp4_encoder_options(ffmpeg):
    """Preserve fine text and color edges without chroma subsampling."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("FFmpeg is required for high-quality MP4 export; run in the pymarl environment") from exc
    encoders = {parts[1] for line in result.stdout.splitlines()
                for parts in [line.split()] if len(parts) >= 2 and parts[0].startswith("V")}
    if "libx264rgb" in encoders:
        # RGB lossless avoids both quantization and RGB-to-YUV conversion.
        return "libx264rgb", ("-crf", "0", "-preset", "medium", "-pix_fmt", "rgb24")
    if "mjpeg" in encoders:
        # The pymarl FFmpeg build lacks x264. Highest-quality, full-color
        # Motion JPEG preserves each frame independently in the MP4 container.
        return "mjpeg", ("-q:v", "1", "-pix_fmt", "yuvj444p")
    raise RuntimeError("FFmpeg needs libx264rgb or mjpeg for high-quality MP4 export")


class MPEAnimator(animation.TimedAnimation):
    """Render one square, world-scaled MPE scene at one frame per step."""

    # A shared multiplier keeps physical predator/prey proportions while
    # making small agents legible at the high-resolution output size. Static
    # landmarks deliberately keep their requested world size.
    _AGENT_ICON_SCALE = 1.7
    _ICON_INTERPOLATION = "lanczos"

    def __init__(
        self,
        agent_positions,
        landmark_positions,
        episode_rewards,
        agent_colors,
        landmark_colors,
        agent_sizes,
        landmark_sizes,
        mask_agents=False,
        agent_icons=None,
        landmark_icons=None,
        agent_world_sizes=None,
        landmark_world_sizes=None,
        field_lim=None,
        icons=None,
        agent_labels=None,
        landmark_labels=None,
        scene_title="MPE",
        fps=1,
        dpi=400,
        trail_length=None,
        show_labels=True,
        frame_offset=0,
    ):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        if not np.isfinite(dpi) or dpi <= 0:
            raise ValueError("dpi must be positive")
        if trail_length is not None and trail_length < 0:
            raise ValueError("trail_length must be nonnegative or None")
        self.fps = float(fps)
        self.scene_title = str(scene_title)
        self.frame_offset = int(frame_offset)
        self.trail_length = trail_length
        self.show_labels = bool(show_labels)
        self.agent_positions = self._coerce_positions(agent_positions, "agent_positions")
        self.landmark_positions = self._coerce_positions(
            landmark_positions, "landmark_positions"
        )
        self.n_agents = self.agent_positions.shape[0]
        self.n_landmarks = self.landmark_positions.shape[0]

        agent_frames = self.agent_positions.shape[1]
        landmark_frames = self.landmark_positions.shape[1]
        if agent_frames and landmark_frames and agent_frames != landmark_frames:
            raise ValueError(
                "agent_positions and landmark_positions must contain the same number of frames"
            )
        self.frames = max(agent_frames, landmark_frames)
        self.lags = max(self.frames, 1)
        self.episode_rewards = (
            np.asarray(episode_rewards, dtype=float).reshape(-1)
            if episode_rewards is not None
            else np.empty(0, dtype=float)
        )
        self.mask_agents = bool(mask_agents)

        self.agent_colors = self._normalise_colors(
            agent_colors, self.n_agents, (0.25, 0.48, 0.80)
        )
        self.landmark_colors = self._normalise_colors(
            landmark_colors, self.n_landmarks, (0.35, 0.38, 0.42)
        )
        self.agent_sizes = self._normalise_numbers(agent_sizes, self.n_agents, 10.0)
        self.landmark_sizes = self._normalise_numbers(
            landmark_sizes, self.n_landmarks, 12.0
        )
        self.agent_world_sizes = self._normalise_numbers(
            agent_world_sizes, self.n_agents, self._default_world_size()
        )
        self.landmark_world_sizes = self._normalise_numbers(
            landmark_world_sizes, self.n_landmarks, self._default_world_size()
        )

        # ``icons`` is a small convenience for callers that have a pair of
        # lists or a mapping. The explicit arguments remain the canonical API.
        if icons is not None:
            agent_icons, landmark_icons = self._unpack_icons(
                icons, agent_icons, landmark_icons
            )
        self.agent_icons = self._normalise_icons(agent_icons, self.n_agents)
        self.landmark_icons = self._normalise_icons(
            landmark_icons, self.n_landmarks
        )
        self._agent_icon_frac = [
            content_frac(icon) if icon is not None else None for icon in self.agent_icons
        ]
        self._landmark_icon_frac = [
            content_frac(icon) if icon is not None else None
            for icon in self.landmark_icons
        ]
        self.field_lim = field_lim
        self.agent_labels = self._normalise_labels(agent_labels, self.n_agents, "Agent")
        self.landmark_labels = self._normalise_labels(landmark_labels, self.n_landmarks, "Landmark")

        # Render text and lines at the final resolution: 3200x3200 at 400 dpi.
        self.fig = plt.figure(figsize=(8, 8), dpi=dpi, facecolor="white")
        self.ax_episode = self.fig.add_subplot(1, 1, 1)
        self.ax_episode.set_facecolor("white")

        # Each artist group has a stable length for the entire animation. In
        # particular, icon artists are moved with set_extent rather than
        # removed/recreated, so blitting never observes a shifted index.
        self._agent_heads = []
        self._trail_lines = []
        self._trail_dots = []
        self._landmark_heads = []
        self._agent_icon_artists = []
        self._landmark_icon_artists = []
        self._agent_texts = []
        self._landmark_texts = []
        self.lines_episode = self._init_episode_animation(self.ax_episode)

        animation.TimedAnimation.__init__(self, self.fig, interval=1000 / self.fps, blit=False)

    @staticmethod
    def _normalise_labels(labels, count, prefix):
        if labels is None:
            return [f"{prefix} {index + 1}" for index in range(count)]
        labels = list(labels)
        if len(labels) != count:
            raise ValueError(f"expected {count} {prefix.lower()} labels, got {len(labels)}")
        return [str(label) for label in labels]

    @staticmethod
    def _coerce_positions(data, name: str) -> np.ndarray:
        array = np.asarray(data, dtype=float)
        if array.size == 0:
            if array.ndim == 3 and array.shape[2] >= 2:
                return array[:, :, :2]
            return np.empty((0, 0, 2), dtype=float)
        if array.ndim == 2 and array.shape[1] >= 2:
            array = array[None, :, :2]
        elif array.ndim == 3 and array.shape[2] >= 2:
            array = array[:, :, :2]
        else:
            raise ValueError(
                f"{name} must have shape (entities, frames, 2), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite coordinates")
        return array

    @staticmethod
    def _is_single_color(value) -> bool:
        if isinstance(value, str):
            return True
        try:
            shape = np.asarray(value).shape
        except (TypeError, ValueError):
            return False
        return len(shape) == 1 and shape[0] in (3, 4) and np.issubdtype(np.asarray(value).dtype, np.number)

    @classmethod
    def _normalise_colors(cls, values, count: int, default) -> List[Tuple[float, ...]]:
        if count == 0:
            return []
        if values is None or cls._is_single_color(values):
            raw = [default if values is None else values] * count
        else:
            try:
                raw = list(values)
            except TypeError:
                raw = [default]
            if not raw:
                raw = [default]
            if len(raw) == 1:
                raw *= count
            elif len(raw) < count:
                raw.extend([default] * (count - len(raw)))
        normalised = []
        for value in raw[:count]:
            try:
                normalised.append(to_rgba(value))
            except (TypeError, ValueError):
                normalised.append(to_rgba(default))
        return normalised

    @staticmethod
    def _normalise_numbers(values, count: int, default: float) -> List[float]:
        if count == 0:
            return []
        if values is None:
            raw = [default] * count
        else:
            try:
                candidate = np.asarray(values)
                if candidate.ndim == 0:
                    raw = [candidate.item()] * count
                else:
                    raw = list(candidate.reshape(-1))
            except (TypeError, ValueError):
                raw = [default]
            if not raw:
                raw = [default]
            if len(raw) == 1:
                raw *= count
            elif len(raw) < count:
                raw.extend([default] * (count - len(raw)))
        normalised = []
        for value in raw[:count]:
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float(default)
            if not np.isfinite(value) or value <= 0:
                value = float(default)
            normalised.append(value)
        return normalised

    def _default_world_size(self) -> float:
        arrays = [
            array.reshape(-1, 2)
            for array in (self.agent_positions, self.landmark_positions)
            if array.size
        ]
        if arrays:
            coordinates = np.concatenate(arrays, axis=0)
            span = float(np.ptp(coordinates, axis=0).max())
            return max(0.04 * max(span, 1.0), 0.04)
        return 0.08

    @staticmethod
    def _unpack_icons(icons, agent_icons, landmark_icons):
        if agent_icons is not None or landmark_icons is not None:
            return agent_icons, landmark_icons
        if isinstance(icons, dict):
            return icons.get("agents", icons.get("agent")), icons.get(
                "landmarks", icons.get("landmark")
            )
        if isinstance(icons, (tuple, list)) and len(icons) == 2:
            return icons[0], icons[1]
        return icons, None

    @staticmethod
    def _normalise_icons(values, count: int) -> List[Optional[np.ndarray]]:
        if count == 0:
            return []
        if values is None:
            return [None] * count
        if isinstance(values, np.ndarray) and values.ndim == 3:
            raw = [values]
        else:
            try:
                raw = list(values)
            except TypeError:
                raw = [values]
        if not raw:
            return [None] * count
        if len(raw) == 1:
            raw *= count
        elif len(raw) != count:
            raise ValueError(
                f"icon list has {len(raw)} entries, expected {count} or one shared icon"
            )
        normalised: List[Optional[np.ndarray]] = []
        for index, icon in enumerate(raw):
            if icon is None:
                normalised.append(None)
                continue
            array = np.asarray(icon)
            if array.ndim != 3 or array.shape[-1] != 4:
                raise ValueError(
                    f"icon {index} must have shape (height, width, 4), got {array.shape}"
                )
            normalised.append(array.astype(np.uint8, copy=False))
        return normalised

    def _limits(self) -> Tuple[float, float, float, float]:
        field = self.field_lim
        if field is not None:
            try:
                values = np.asarray(field, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                values = np.empty(0, dtype=float)
            if values.size == 1 and np.isfinite(values[0]) and values[0] > 0:
                half = float(values[0])
                margin = max(0.4, 0.2 * half)
                return -(half + margin), half + margin, -(half + margin), half + margin
            if values.size == 2 and np.all(np.isfinite(values)) and values[1] > values[0]:
                low, high = map(float, values)
                return self._square_limits(low, high, low, high)
            if values.size == 4 and np.all(np.isfinite(values)):
                return self._square_limits(*map(float, values))

        arrays = []
        for positions, sizes, fractions, scale in (
            (self.agent_positions, self.agent_world_sizes, self._agent_icon_frac, self._AGENT_ICON_SCALE),
            (self.landmark_positions, self.landmark_world_sizes, self._landmark_icon_frac, 1.0),
        ):
            if positions.size:
                radii = np.array([
                    size * scale / max(*(fraction or (1.0, 1.0)), 1.0e-6)
                    for size, fraction in zip(sizes, fractions)
                ])[:, None, None]
                arrays.extend([(positions - radii).reshape(-1, 2), (positions + radii).reshape(-1, 2)])
        if not arrays:
            return -1.0, 1.0, -1.0, 1.0
        positions = np.concatenate(arrays, axis=0)
        x_min, y_min = positions.min(axis=0)
        x_max, y_max = positions.max(axis=0)
        return self._square_limits(float(x_min), float(x_max), float(y_min), float(y_max))

    @staticmethod
    def _square_limits(x_min, x_max, y_min, y_max) -> Tuple[float, float, float, float]:
        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min, 1.0e-6)
        margin = max(0.08 * span, 0.05)
        half = 0.5 * span + margin
        return x_center - half, x_center + half, y_center - half, y_center + half

    def _icon_extent(
        self, x: float, y: float, world_size: float, fraction, scale: float = 1.0
    ) -> Tuple[float, float, float, float]:
        fx, fy = fraction or (1.0, 1.0)
        radius = max(float(world_size) * float(scale), 1.0e-6) / max(
            fx, fy, 1.0e-6
        )
        return x - radius, x + radius, y - radius, y + radius

    def _init_episode_animation(self, ax):
        x_min, x_max, y_min, y_max = self._limits()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X position", fontsize=11)
        ax.set_ylabel("Y position", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.18)
        ax.tick_params(colors="#777777", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{self.scene_title} · step 0", fontsize=14, pad=10)

        for index in range(self.n_agents):
            icon = self.agent_icons[index]
            if icon is None:
                head = ax.plot(
                    [],
                    [],
                    "o",
                    color=self.agent_colors[index],
                    markeredgecolor="#172033",
                    markeredgewidth=0.8,
                    markersize=max(2.0, self.agent_sizes[index]),
                    alpha=1.0,
                    zorder=6,
                )[0]
                self._agent_icon_artists.append(None)
            else:
                head = ax.imshow(
                    icon,
                    extent=self._icon_extent(
                        0.0,
                        0.0,
                        self.agent_world_sizes[index],
                        self._agent_icon_frac[index],
                        self._AGENT_ICON_SCALE,
                    ),
                    zorder=6,
                    interpolation=self._ICON_INTERPOLATION,
                )
                self._agent_icon_artists.append(head)
            if self.mask_agents:
                head.set_visible(False)
            self._agent_heads.append(head)
            text = ax.text(0, 0, "", fontsize=8, color="#334155", ha="center", va="bottom", zorder=8)
            self._agent_texts.append(text)

        for index in range(self.n_agents):
            color = self._lighten(self.agent_colors[index], 0.55)
            trail_line = ax.plot(
                [], [], "-", color=color, alpha=0.55, linewidth=1.8, zorder=3
            )[0]
            trail_dots = ax.plot(
                [],
                [],
                "o",
                color=color,
                alpha=0.28,
                markersize=max(2.0, self.agent_sizes[index] * 0.3),
                zorder=2,
            )[0]
            if self.mask_agents:
                trail_line.set_visible(False)
                trail_dots.set_visible(False)
            self._trail_lines.append(trail_line)
            self._trail_dots.append(trail_dots)

        for index in range(self.n_landmarks):
            icon = self.landmark_icons[index]
            if icon is None:
                head = ax.plot(
                    [],
                    [],
                    "D",
                    color=self.landmark_colors[index],
                    markeredgecolor="#222222",
                    markeredgewidth=0.8,
                    markersize=max(2.0, self.landmark_sizes[index]),
                    alpha=0.85,
                    zorder=4,
                )[0]
                self._landmark_icon_artists.append(None)
            else:
                head = ax.imshow(
                    icon,
                    extent=self._icon_extent(
                        0.0,
                        0.0,
                        self.landmark_world_sizes[index],
                        self._landmark_icon_frac[index],
                    ),
                    zorder=4,
                    interpolation=self._ICON_INTERPOLATION,
                )
                self._landmark_icon_artists.append(head)
            self._landmark_heads.append(head)
            text = ax.text(0, 0, "", fontsize=8, color="#64748b", ha="center", va="bottom", zorder=5)
            self._landmark_texts.append(text)

        return (self._agent_heads + self._trail_lines + self._trail_dots + self._landmark_heads
                + self._agent_texts + self._landmark_texts)

    @staticmethod
    def _lighten(color: Sequence[float], amount: float) -> Tuple[float, float, float, float]:
        rgba = to_rgba(color)
        return tuple(
            min(1.0, channel + (1.0 - channel) * amount) for channel in rgba[:3]
        ) + (rgba[3],)

    def _set_head(self, artist, position, world_size, fraction, scale: float = 1.0):
        if position is None:
            if isinstance(artist, Line2D):
                artist.set_data([], [])
            else:
                artist.set_visible(False)
            return
        x, y = map(float, position[:2])
        if isinstance(artist, Line2D):
            artist.set_data([x], [y])
        else:
            artist.set_visible(True)
            artist.set_extent(self._icon_extent(x, y, world_size, fraction, scale))

    def _draw_frame(self, frame):
        self._draw_was_started = True
        if self.frames:
            frame = min(max(int(frame), 0), self.frames - 1)
            for index, artist in enumerate(self._agent_heads):
                position = (
                    self.agent_positions[index, frame]
                    if frame < self.agent_positions.shape[1]
                    else None
                )
                if not self.mask_agents:
                    self._set_head(
                        artist,
                        position,
                        self.agent_world_sizes[index],
                        self._agent_icon_frac[index],
                        self._AGENT_ICON_SCALE,
                    )
                start = 0 if self.trail_length is None else max(0, frame + 1 - self.trail_length)
                trail = self.agent_positions[index, start : frame + 1]
                self._trail_lines[index].set_data(trail[:, 0], trail[:, 1])
                # Dots are deliberately sparse on long episodes to keep the
                # visual light and the draw cost bounded.
                stride = max(1, len(trail) // 80)
                dots = trail[::stride]
                self._trail_dots[index].set_data(dots[:, 0], dots[:, 1])
                text = self._agent_texts[index]
                text.set_visible(self.show_labels and not self.mask_agents and position is not None)
                if position is not None:
                    text.set_position((position[0], position[1] + self.agent_world_sizes[index] * self._AGENT_ICON_SCALE + 0.04))
                    text.set_text(self.agent_labels[index])
            for index, artist in enumerate(self._landmark_heads):
                position = (
                    self.landmark_positions[index, frame]
                    if frame < self.landmark_positions.shape[1]
                    else None
                )
                self._set_head(
                    artist,
                    position,
                    self.landmark_world_sizes[index],
                    self._landmark_icon_frac[index],
                )
                text = self._landmark_texts[index]
                text.set_visible(self.show_labels and position is not None)
                if position is not None:
                    text.set_position((position[0], position[1] - self.landmark_world_sizes[index] - 0.06))
                    text.set_text(self.landmark_labels[index])
        else:
            frame = 0
            for artist in self._agent_heads + self._landmark_heads:
                self._set_head(artist, None, 0.0, None)
            for line, dots in zip(self._trail_lines, self._trail_dots):
                line.set_data([], [])
                dots.set_data([], [])
            for text in self._agent_texts + self._landmark_texts:
                text.set_visible(False)

        reward = float(self.episode_rewards[:frame + 1].sum())
        self.ax_episode.set_title(f"{self.scene_title} · step {frame + self.frame_offset} · return {reward:.3f}", fontsize=14, pad=10)
        self._drawn_artists = self.lines_episode
        return self._drawn_artists

    def new_frame_seq(self):
        return iter(range(self.frames))

    def _init_draw(self):
        self._draw_was_started = True
        for artist in self.lines_episode:
            if isinstance(artist, Line2D):
                artist.set_data([], [])
            else:
                artist.set_visible(False)
        self.ax_episode.set_title(f"{self.scene_title} · step {self.frame_offset}", fontsize=14, pad=10)
        self._drawn_artists = self.lines_episode

    @staticmethod
    def _output_path(savepath) -> Path:
        raw_value = os.fspath(savepath)
        if not raw_value:
            raise ValueError("savepath must not be empty")
        raw = Path(raw_value)
        if raw.suffix.lower() == ".mp4":
            return raw
        if raw.suffix.lower() in (".gif", ".jpg", ".png"):
            raise ValueError("animation export supports MP4 only; use a .mp4 path")
        # Experiment runners pass a directory rather than an explicit file.
        return raw / "episode.mp4"

    @staticmethod
    def _temporary_path(parent: Path, suffix: str) -> Path:
        handle, name = tempfile.mkstemp(
            prefix=".mpe-animation-", suffix=suffix, dir=str(parent)
        )
        os.close(handle)
        path = Path(name)
        path.unlink()
        return path

    def save_animation(self, savepath="episode"):
        """Write a high-quality full-color MP4, without additional files."""

        if self.frames <= 0:
            raise ValueError("cannot save an animation with no frames")
        # TimedAnimation initializes its artists on the first canvas draw.
        # Complete that event before preparing the first video frame so it
        # cannot clear the icons and labels after we have placed them.
        self.fig.canvas.draw()
        video_path = self._output_path(savepath)
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError("FFmpeg is required for high-quality MP4 export; run in the pymarl environment")
        codec, extra_args = _mp4_encoder_options(animation.FFMpegWriter.bin_path())
        self.video_codec = codec
        video_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_video = self._temporary_path(video_path.parent, ".mp4")
        writer = animation.FFMpegWriter(
            fps=self.fps, codec=codec, bitrate=-1,
            extra_args=list(extra_args) + ["-threads", "2", "-movflags", "+faststart"],
        )
        try:
            with writer.saving(self.fig, str(temporary_video), self.fig.dpi):
                for frame in self.new_frame_seq():
                    self._draw_frame(frame)
                    writer.grab_frame(facecolor="white")
            if not temporary_video.is_file() or temporary_video.stat().st_size == 0:
                raise RuntimeError("FFmpeg completed without producing a valid MP4")
            os.replace(temporary_video, video_path)
        except Exception as exc:
            raise RuntimeError("Unable to write the high-quality MP4 with FFmpeg") from exc
        finally:
            try:
                temporary_video.unlink()
            except FileNotFoundError:
                pass
        self._draw_was_started = True
        return video_path


__all__ = ["MPEAnimator"]
