"""Predator-prey renderer, sharing artist bookkeeping with the CN renderer."""

from .pyplot_animator import MPEAnimator as _GenericMPEAnimator


class MPEAnimator(_GenericMPEAnimator):
    """Render every predator, prey and world landmark without fixed indices."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("scene_title", "Predator–Prey")
        super().__init__(*args, **kwargs)
        if len(self._landmark_heads) != self.n_landmarks:
            raise RuntimeError("predator-prey animator lost a landmark artist")

    def _draw_frame(self, frame):
        artists = super()._draw_frame(frame)
        if self.frames:
            for artist in self._landmark_heads:
                artist.set_visible(True)
        return artists


__all__ = ["MPEAnimator"]
