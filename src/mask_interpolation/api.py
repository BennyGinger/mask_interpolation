"""Public mask interpolation API."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mask_interpolation.pair import interpolate_pair
from mask_interpolation.validation import validate_array, validate_sequence


def fill_missing_masks(mask_stack: NDArray[np.generic],
                       *,
                       axes: str,
                       interpolation_axis: str,
                       extrapolate_start: bool = True,
                       extrapolate_end: bool = True,
                       ) -> NDArray[np.generic]:
    """
    Fill empty planes along one axis of a sparse mask array.

    Each combination of the non-spatial, non-interpolated axes is processed as
    an independent ``(sequence, y, x)`` mask stack. A plane is considered
    missing when all its values are zero. Gaps enclosed by known planes are
    interpolated. Leading and trailing gaps can optionally be filled by copying
    their nearest known plane.

    Args:
        mask_stack: Mask array with at least three dimensions. Known planes in
            each sequence must contain one common nonzero label.
        axes: Axis labels matching the dimensions of ``mask_stack``, such as
            ``"ZYX"`` or ``"TZYX"``.
        interpolation_axis: Axis along which missing masks will be completed.
            Spatial axes ``Y`` and ``X`` and channel axis ``C`` are not valid.
        extrapolate_start: Copy the first known mask into leading empty planes.
        extrapolate_end: Copy the last known mask into trailing empty planes.

    Returns:
        A completed copy of ``mask_stack`` with the same shape and dtype.

    Raises:
        TypeError: If the axis arguments are not strings.
        ValueError: If the input or axes are invalid, the array contains no
            known masks, or known planes are unsuitable for interpolation.
    """
    normalized_axes, normalized_interpolation_axis = validate_array(
        mask_stack, axes, interpolation_axis)
    sequences, transposed_shape, restore_order = _prepare_sequences(
        mask_stack, normalized_axes, normalized_interpolation_axis)
    completed_sequences = np.empty_like(sequences)

    for index, sequence in enumerate(sequences):
        if _has_known_masks(sequence):
            completed_sequences[index] = _fill_sequence_gaps(
                sequence,
                extrapolate_start=extrapolate_start,
                extrapolate_end=extrapolate_end)
        else: # that is if the sequence is completely empty, leave it unchanged
            completed_sequences[index] = sequence

    return _restore_array(completed_sequences, transposed_shape, restore_order)


def _prepare_sequences(mask_stack: NDArray[np.generic],
                       axes: str,
                       interpolation_axis: str,
                       ) -> tuple[NDArray[np.generic], tuple[int, ...], tuple[int, ...]]:
    """
    Arrange a mask array as independent ``(sequence, y, x)`` stacks.

    Args:
        mask_stack: Validated mask array in its original axis order.
        axes: Normalized labels describing the original axis order.
        interpolation_axis: Axis to place immediately before ``Y`` and ``X``.

    Returns:
        Flattened mask sequences, the transposed shape, and the inverse axis
        order required to restore the original layout.
    """
    interpolation_index = axes.index(interpolation_axis)
    y_index = axes.index("Y")
    x_index = axes.index("X")
    selected = {interpolation_index, y_index, x_index}
    batch_indices = tuple(index for index in range(mask_stack.ndim) if index not in selected)
    order = (*batch_indices, interpolation_index, y_index, x_index)
    transposed = np.transpose(mask_stack, order)
    sequences = transposed.reshape((-1, *transposed.shape[-3:]))
    restore_order = tuple(int(index) for index in np.argsort(order))
    return sequences, transposed.shape, restore_order


def _restore_array(completed_sequences: NDArray[np.generic],
                   transposed_shape: tuple[int, ...],
                   restore_order: tuple[int, ...],
                   ) -> NDArray[np.generic]:
    """
    Restore completed sequences to the input array's shape and axis order.

    Args:
        completed_sequences: Flattened sequences after gap filling.
        transposed_shape: Shape of the prepared array before batch flattening.
        restore_order: Axis permutation that restores the original order.

    Returns:
        Completed mask array in its original layout.
    """
    completed = completed_sequences.reshape(transposed_shape)
    return np.transpose(completed, restore_order)


def _has_known_masks(sequence: NDArray[np.generic]) -> np.bool_:
    """
    Return whether a sequence contains at least one known mask.
    """
    return np.any(sequence != 0)


def _fill_sequence_gaps(mask_stack: NDArray[np.generic],
                        *,
                        extrapolate_start: bool,
                        extrapolate_end: bool,
                        ) -> NDArray[np.generic]:
    """
    Fill gaps in one nonempty sequence of 2D masks.
    """
    label = validate_sequence(mask_stack)
    completed = mask_stack.copy()
    has_mask: NDArray[np.bool_] = np.any(completed != 0, axis=(1, 2))
    known_indices = np.flatnonzero(has_mask)

    first, last = int(known_indices[0]), int(known_indices[-1])
    if extrapolate_start:
        completed[:first] = completed[first]
    if extrapolate_end:
        completed[last + 1:] = completed[last]

    for start, end in zip(known_indices[:-1], known_indices[1:], strict=True):
        missing_count = int(end - start - 1)
        if missing_count:
            completed[start + 1:end] = interpolate_pair(
                completed[start], completed[end], missing_count, label=label)
    return completed
