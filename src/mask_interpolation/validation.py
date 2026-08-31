"""Input validation for mask interpolation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_array(mask_stack: NDArray[np.generic],
                   axes: str,
                   interpolation_axis: str,
                   ) -> tuple[str, str]:
    """
    Validate a mask array and normalize its axis labels.
    
    Returns the normalized axes and interpolation axis in uppercase. 
    
    Raises ValueError if the input is invalid, or TypeError if the axis arguments are not strings.
    """
    if not isinstance(axes, str) or not isinstance(interpolation_axis, str):
        raise TypeError("axes and interpolation_axis must be strings.")
    normalized_axes = axes.upper()
    normalized_interpolation_axis = interpolation_axis.upper()
    if mask_stack.ndim < 3:
        raise ValueError("mask_stack must have at least three dimensions.")
    if 0 in mask_stack.shape:
        raise ValueError("mask_stack must contain at least one plane.")
    if len(normalized_axes) != mask_stack.ndim:
        raise ValueError("axes must contain one label for each mask_stack dimension.")
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("axes cannot contain duplicate labels.")
    if "Y" not in normalized_axes or "X" not in normalized_axes:
        raise ValueError("axes must contain both Y and X spatial axes.")
    if len(normalized_interpolation_axis) != 1:
        raise ValueError("interpolation_axis must be one axis label.")
    if normalized_interpolation_axis not in normalized_axes:
        raise ValueError("interpolation_axis is not present in axes.")
    if normalized_interpolation_axis in "YXC":
        raise ValueError("interpolation_axis cannot be Y, X, or C.")
    if not np.any(mask_stack != 0):
        raise ValueError("mask_stack contains no known masks to interpolate.")
    return normalized_axes, normalized_interpolation_axis


def validate_sequence(mask_stack: NDArray[np.generic]) -> np.generic:
    """
    Validate the known masks in one sequence and return their common label.

    Raises ValueError if the sequence is empty or contains more than one
    nonzero label.
    """
    labels = np.unique(mask_stack[mask_stack != 0])
    if labels.size == 0:
        raise ValueError("A sequence cannot be empty.")
    if labels.size > 1:
        raise ValueError("Known masks in a sequence must use one nonzero label.")
    return labels[0]
