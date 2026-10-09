"""Random-access interpolation for a single sparse mask sequence."""

import numpy as np
from numpy.typing import NDArray

from mask_interpolation.pair import PairInterpolator
from mask_interpolation.validation import validate_sequence


class SequenceInterpolator:
    """Retain one gap's distance maps while serving planes in any order."""

    def __init__(self, anchors: NDArray[np.generic], *,
                 extrapolate_start: bool = True, extrapolate_end: bool = True) -> None:
        if anchors.ndim != 3 or 0 in anchors.shape:
            raise ValueError("A sequence must have shape (planes, Y, X).")
        self.anchors = anchors
        self.known = np.flatnonzero(np.any(anchors != 0, axis=(1, 2)))
        self.label = validate_sequence(anchors) if len(self.known) else None
        self.start, self.end = extrapolate_start, extrapolate_end
        self._gap: tuple[int, int] | None = None
        self._pair: PairInterpolator | None = None

    def __call__(self, index: int) -> NDArray[np.generic]:
        if not 0 <= index < len(self.anchors):
            raise IndexError("Sequence plane index is outside the sequence.")
        location = int(np.searchsorted(self.known, index))
        if location < len(self.known) and self.known[location] == index:
            return self.anchors[index].copy()
        if not len(self.known):
            return self.anchors[index].copy()
        if location == 0:
            return self.anchors[int(self.known[0]) if self.start else index].copy()
        if location == len(self.known):
            return self.anchors[int(self.known[-1]) if self.end else index].copy()
        left, right = int(self.known[location - 1]), int(self.known[location])
        if self._gap != (left, right):
            assert self.label is not None
            self._pair = PairInterpolator(self.anchors[left], self.anchors[right],
                                          right - left - 1, label=self.label)
            self._gap = left, right
        assert self._pair is not None
        return self._pair.plane(index - left - 1)
