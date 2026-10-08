"""Small, dependency-free cartoon icons used by the MPE animator.

The artwork is generated at runtime with :mod:`PIL`; no image files or fonts
are needed.  ``make_icon`` returns an immutable-by-convention, cached RGBA
``numpy`` array so the same glyph can be reused by every animation frame.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


RGB = Tuple[int, int, int]
Tint = Optional[Tuple[float, float, float]]


_cache: Dict[Tuple[str, Tint], np.ndarray] = {}


def _canvas() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def _normalise_tint(tint) -> Tint:
    if tint is None:
        return None
    try:
        values = tuple(float(value) for value in tint)
    except (TypeError, ValueError) as exc:
        raise ValueError("tint must be an iterable of three values in [0, 1]") from exc
    if len(values) != 3 or not all(np.isfinite(value) for value in values):
        raise ValueError("tint must be an iterable of three values in [0, 1]")
    return tuple(max(0.0, min(1.0, value)) for value in values)  # type: ignore[return-value]


def _tint_to_255(tint: Tint) -> Optional[RGB]:
    if tint is None:
        return None
    return tuple(int(value * 255) for value in tint)  # type: ignore[return-value]


def _lighten(color: RGB, fraction: float = 0.65) -> RGB:
    return tuple(min(255, int(value + (255 - value) * fraction)) for value in color)  # type: ignore[return-value]


def _tiger(tint: Tint = None) -> np.ndarray:
    image, draw = _canvas()
    color = _tint_to_255(tint) or (255, 153, 51)
    ear = _tint_to_255(tint) or (255, 140, 50)
    dark = (80, 45, 12)
    draw.polygon([(26, 36), (42, 10), (60, 34)], fill=ear)
    draw.polygon([(68, 34), (86, 10), (102, 36)], fill=ear)
    draw.polygon([(32, 32), (42, 18), (52, 30)], fill=_lighten(ear))
    draw.polygon([(76, 30), (86, 18), (96, 32)], fill=_lighten(ear))
    draw.ellipse([(14, 18), (114, 118)], fill=color)
    for x0 in (40, 52, 64, 76, 88):
        draw.line([(x0, 28), (x0 - 8, 44)], fill=dark, width=4)
    for x0 in (30, 40, 50):
        draw.line([(x0, 52), (x0 - 8, 62)], fill=dark, width=4)
    for x0 in (78, 88, 98):
        draw.line([(x0, 52), (x0 + 8, 62)], fill=dark, width=4)
    draw.ellipse([(44, 74), (84, 112)], fill=(255, 226, 180))
    draw.ellipse([(38, 50), (52, 64)], fill=(25, 15, 8))
    draw.ellipse([(76, 50), (90, 64)], fill=(25, 15, 8))
    draw.ellipse([(43, 54), (47, 58)], fill=(255, 255, 255))
    draw.ellipse([(81, 54), (85, 58)], fill=(255, 255, 255))
    draw.polygon([(59, 74), (69, 74), (64, 84)], fill=dark)
    draw.line([(64, 84), (58, 92)], fill=dark, width=2)
    draw.line([(64, 84), (70, 92)], fill=dark, width=2)
    return np.asarray(image)


def _deer(tint: Tint = None) -> np.ndarray:
    image, draw = _canvas()
    fur = _tint_to_255(tint) or (196, 145, 92)
    ear = _tint_to_255(tint) or (200, 150, 100)
    antler = (90, 60, 30)
    draw.line([(52, 20), (40, 34)], fill=antler, width=4)
    draw.line([(52, 20), (64, 32)], fill=antler, width=4)
    draw.line([(52, 24), (52, 44)], fill=antler, width=4)
    for tip in ((40, 34), (64, 32)):
        draw.line([tip, (tip[0] - 10, tip[1] + 4)], fill=antler, width=3)
        draw.line([tip, (tip[0] + 10, tip[1] + 4)], fill=antler, width=3)
    draw.ellipse([(12, 52), (32, 72)], fill=ear)
    draw.ellipse([(96, 52), (116, 72)], fill=ear)
    draw.ellipse([(18, 56), (30, 68)], fill=_lighten(ear))
    draw.ellipse([(98, 56), (110, 68)], fill=_lighten(ear))
    draw.ellipse([(26, 36), (102, 108)], fill=fur)
    draw.ellipse([(50, 78), (78, 106)], fill=(245, 220, 190))
    draw.ellipse([(42, 56), (56, 70)], fill=(30, 20, 10))
    draw.ellipse([(72, 56), (86, 70)], fill=(30, 20, 10))
    draw.ellipse([(46, 59), (52, 65)], fill=(255, 255, 255))
    draw.ellipse([(76, 59), (82, 65)], fill=(255, 255, 255))
    draw.ellipse([(58, 76), (70, 86)], fill=(60, 40, 20))
    return np.asarray(image)


def _robot(tint: Tint = None) -> np.ndarray:
    image, draw = _canvas()
    accent = _tint_to_255(tint) or (125, 136, 148)
    draw.line([(64, 6), (64, 22)], fill=(90, 100, 110), width=3)
    draw.ellipse([(58, 0), (70, 12)], fill=(255, 80, 80))
    draw.rounded_rectangle([(28, 26), (100, 112)], radius=16, fill=accent)
    draw.rounded_rectangle([(38, 42), (90, 92)], radius=10, fill=(205, 214, 222))
    draw.ellipse([(48, 56), (62, 70)], fill=(30, 40, 50))
    draw.ellipse([(66, 56), (80, 70)], fill=(30, 40, 50))
    draw.ellipse([(51, 59), (59, 67)], fill=(120, 220, 255))
    draw.ellipse([(69, 59), (77, 67)], fill=(120, 220, 255))
    draw.rectangle([(52, 78), (76, 84)], fill=(30, 40, 50))
    draw.ellipse([(38, 100), (44, 106)], fill=(90, 100, 110))
    draw.ellipse([(84, 100), (90, 106)], fill=(90, 100, 110))
    return np.asarray(image)


def _target(tint: Tint = None) -> np.ndarray:
    image, draw = _canvas()
    color = _tint_to_255(tint) or (210, 40, 40)
    draw.ellipse([(10, 10), (118, 118)], fill=color)
    draw.ellipse([(28, 28), (100, 100)], fill=(255, 255, 255))
    draw.ellipse([(46, 46), (82, 82)], fill=color)
    draw.ellipse([(58, 58), (70, 70)], fill=(255, 255, 255))
    return np.asarray(image)


def _person(tint: Tint = None) -> np.ndarray:
    image, draw = _canvas()
    shirt = _tint_to_255(tint) or (70, 120, 200)
    draw.ellipse([(44, 14), (84, 54)], fill=(245, 205, 170))
    draw.pieslice([(44, 14), (84, 54)], 180, 360, fill=(60, 40, 25))
    draw.polygon([(38, 60), (90, 60), (102, 108), (26, 108)], fill=shirt)
    draw.line([(38, 64), (20, 94)], fill=shirt, width=8)
    draw.line([(90, 64), (108, 94)], fill=shirt, width=8)
    draw.line([(50, 108), (50, 124)], fill=(50, 70, 110), width=10)
    draw.line([(78, 108), (78, 124)], fill=(50, 70, 110), width=10)
    return np.asarray(image)


def _apple(tint: Tint = None) -> np.ndarray:
    image, draw = _canvas()
    color = _tint_to_255(tint) or (200, 30, 30)
    draw.ellipse([(22, 34), (106, 118)], fill=color)
    draw.ellipse([(44, 54), (58, 68)], fill=(245, 120, 120))
    draw.line([(64, 34), (64, 12)], fill=(90, 60, 25), width=6)
    draw.ellipse([(70, 8), (100, 26)], fill=(90, 160, 60))
    return np.asarray(image)


def _rock(tint: Tint = None) -> np.ndarray:
    image, draw = _canvas()
    color = _tint_to_255(tint) or (140, 142, 148)
    draw.polygon([(24, 84), (30, 44), (52, 26), (84, 28), (104, 50), (102, 84)], fill=color)
    draw.polygon([(24, 84), (30, 44), (60, 48), (58, 84)], fill=_lighten(color, 0.14))
    draw.polygon([(84, 28), (104, 50), (102, 84), (58, 84), (60, 48)], fill=tuple(max(0, value - 20) for value in color))
    return np.asarray(image)


_MAKERS: Dict[str, Callable[[Tint], np.ndarray]] = {
    "tiger": _tiger,
    "deer": _deer,
    "robot": _robot,
    "target": _target,
    "rock": _rock,
    "person": _person,
    "apple": _apple,
    # Semantic aliases make callers independent of a particular predator/prey
    # scenario while retaining the generated artwork from the attachment.
    "predator": _tiger,
    "prey": _deer,
    "landmark": _target,
    "agent": _robot,
    "goal": _target,
    "obstacle": _rock,
}

ICON_KINDS = tuple(sorted(_MAKERS))


def make_icon(kind: str, tint=None) -> np.ndarray:
    """Return a cached 128x128 RGBA icon for ``kind``.

    ``tint`` is an optional three-component RGB value in the range ``[0, 1]``
    and replaces the primary fill where the glyph supports it.  Unknown icon
    names raise a useful ``ValueError`` instead of leaking a dictionary
    ``KeyError`` from the implementation.
    """

    if not isinstance(kind, str):
        raise ValueError(f"Unknown icon kind {kind!r}; choose one of {ICON_KINDS}")
    name = kind.strip().lower()
    if name not in _MAKERS:
        raise ValueError(f"Unknown icon kind {kind!r}; choose one of {ICON_KINDS}")
    normalised_tint = _normalise_tint(tint)
    key = (name, normalised_tint)
    if key not in _cache:
        array = np.asarray(_MAKERS[name](normalised_tint), dtype=np.uint8)
        if array.shape != (128, 128, 4):
            raise ValueError(f"Icon maker {name!r} returned {array.shape}, expected (128, 128, 4)")
        _cache[key] = array
    return _cache[key]


def content_frac(array: np.ndarray) -> Tuple[float, float]:
    """Return the non-transparent artwork fraction along each image axis."""

    array = np.asarray(array)
    if array.ndim != 3 or array.shape[-1] < 4:
        raise ValueError("icon array must have shape (height, width, 4)")
    alpha = array[:, :, 3] > 10
    ys, xs = np.where(alpha)
    if len(xs) == 0:
        return 1.0, 1.0
    return (
        (xs.max() - xs.min() + 1) / array.shape[1],
        (ys.max() - ys.min() + 1) / array.shape[0],
    )


__all__ = ["ICON_KINDS", "content_frac", "make_icon"]
