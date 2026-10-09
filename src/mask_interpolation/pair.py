"""Interpolation between two known masks."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt


def interpolate_pair(start_mask: NDArray[np.generic],
                     end_mask: NDArray[np.generic],
                     missing_count: int,
                     *,
                     label: np.generic,
                     ) -> NDArray[np.generic]:
    """
    Interpolate missing 2D masks between two known masks.

    The masks are temporarily aligned by their centroids. Their non-overlapping
    regions are progressively removed or introduced using distance transforms,
    while the centroid follows a linear trajectory between the two anchors.

    Args:
        start_mask: Known mask immediately before the gap.
        end_mask: Known mask immediately after the gap.
        missing_count: Number of intermediate masks to generate.
        label: Nonzero label assigned to every interpolated mask.

    Returns:
        An array of shape ``(missing_count, y, x)`` using the input dtype.
    """
    if missing_count == 0:
        return np.empty((0, *start_mask.shape), dtype=start_mask.dtype)
    prepared = PairInterpolator(start_mask, end_mask, missing_count, label=label)
    masks = np.empty((missing_count, *start_mask.shape), dtype=start_mask.dtype)
    for index in range(missing_count):
        masks[index] = prepared.plane(index)
    return masks


class PairInterpolator:
    """Prepare the distance maps once and evaluate intermediate planes on demand."""

    def __init__(self, start_mask: NDArray[np.generic], end_mask: NDArray[np.generic],
                 missing_count: int, *, label: np.generic) -> None:
        self.dtype, self.label, self.count = start_mask.dtype, label, missing_count
        start_centroid, end_centroid = _centroid(start_mask), _centroid(end_mask)
        center = tuple(size // 2 for size in start_mask.shape)
        self.overlap, self.start_map, self.end_map = _create_shape_transition_maps(
            _relocate(start_mask != 0, center), _relocate(end_mask != 0, center))
        self.centroids = np.linspace(start_centroid, end_centroid, missing_count + 2)[1:-1]

    def plane(self, index: int) -> NDArray[np.generic]:
        if not 0 <= index < self.count:
            raise IndexError("Intermediate plane index is outside the gap.")
        start = _create_intermediate_shape_part(
            self.overlap, self.start_map, index, self.count, transition="remove")
        end = _create_intermediate_shape_part(
            self.overlap, self.end_map, index, self.count, transition="add")
        return np.multiply(_relocate(start | end, self.centroids[index]),
                           self.label, dtype=self.dtype)


def _centroid(mask: NDArray[np.generic]) -> NDArray[np.float64]:
    """Return the centroid of all nonzero pixels."""
    return np.argwhere(mask != 0).mean(axis=0)


def _relocate(mask: NDArray[np.bool_], location: tuple[int, ...] | NDArray[np.float64]) -> NDArray[np.bool_]:
    """Relocate a mask centroid, preserving the historical integer shift."""
    shift = np.asarray(location) - _centroid(mask)
    shift = shift.astype(int)
    points = np.argwhere(mask) + shift
    points = np.clip(points, 0, np.asarray(mask.shape) - 1)
    relocated = np.zeros(mask.shape, dtype=bool)
    relocated[tuple(points.T)] = True
    return relocated


def _create_non_overlap_distance_map(non_overlap: NDArray[np.bool_],
                                     overlap: NDArray[np.bool_],
                                     ) -> NDArray[np.float64]:
    """
    Map each non-overlapping pixel's distance from the shared overlap.
    """
    distance = cast(NDArray[np.float64], distance_transform_edt(~overlap))
    distance[~non_overlap] = 0
    return distance


def _create_shape_transition_maps(start_mask: NDArray[np.bool_],
                                  end_mask: NDArray[np.bool_],
                                  ) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """
    Create the shared overlap and non-overlap distance maps for aligned masks.

    Args:
        start_mask: Centred starting mask.
        end_mask: Centred ending mask.

    Returns:
        Shared overlap followed by distance maps for the start-only and
        end-only regions.
    """
    overlap = start_mask & end_mask
    start_distance_map = _create_non_overlap_distance_map(
        start_mask & ~overlap, overlap)
    end_distance_map = _create_non_overlap_distance_map(
        end_mask & ~overlap, overlap)
    return overlap, start_distance_map, end_distance_map


def _create_intermediate_shape_part(overlap: NDArray[np.bool_],
                                    distance: NDArray[np.float64],
                                    index: int,
                                    missing_count: int,
                                    *,
                                    transition: Literal["remove", "add"],
                                    ) -> NDArray[np.bool_]:
    """
    Create a progressively removed or added part of an intermediate mask.
    """
    maximum = distance.max()
    if maximum == 0:
        return overlap.copy()
    fractions = np.linspace(maximum if transition == "remove" else 0,
                            0 if transition == "remove" else maximum,
                            missing_count + 1,
                            endpoint=False)[1:]
    return overlap | ((distance != 0) & (distance <= fractions[index]))
