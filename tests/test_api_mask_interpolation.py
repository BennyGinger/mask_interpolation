import numpy as np
import pytest

from mask_interpolation import fill_missing_masks


def mask_at(top: int, left: int) -> np.ndarray:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[top:top + 4, left:left + 4] = 1
    return mask


def test_fill_missing_masks_fills_internal_and_endpoint_gaps():
    stack = np.zeros((7, 16, 16), dtype=np.uint8)
    stack[2] = mask_at(2, 2)
    stack[5] = mask_at(8, 8)
    original = stack.copy()

    result = fill_missing_masks(stack, axes="TYX", interpolation_axis="T")

    assert np.all(np.any(result != 0, axis=(1, 2)))
    assert np.array_equal(result[2], original[2])
    assert np.array_equal(result[5], original[5])
    assert np.array_equal(result[:2], np.broadcast_to(original[2], result[:2].shape))
    assert np.array_equal(result[6], original[5])
    assert np.array_equal(stack, original)


def test_fill_missing_masks_can_leave_endpoint_gaps_empty():
    stack = np.zeros((5, 16, 16), dtype=np.uint8)
    stack[1] = mask_at(2, 2)
    stack[3] = mask_at(8, 8)

    result = fill_missing_masks(
        stack, axes="TYX", interpolation_axis="T",
        extrapolate_start=False, extrapolate_end=False)

    assert not np.any(result[0])
    assert np.any(result[2])
    assert not np.any(result[4])


def test_fill_missing_masks_preserves_complete_input():
    stack = np.stack([mask_at(index, index) for index in range(4)])

    result = fill_missing_masks(stack, axes="TYX", interpolation_axis="T")

    assert np.array_equal(result, stack)
    assert result is not stack


def test_fill_missing_masks_rejects_fully_empty_input():
    with pytest.raises(ValueError, match="no known masks"):
        fill_missing_masks(
            np.zeros((3, 8, 8), dtype=np.uint8), axes="ZYX", interpolation_axis="Z")


def test_fill_missing_masks_interpolates_z_for_every_time_point():
    stack = np.zeros((2, 4, 16, 16), dtype=np.uint8)
    stack[0, 0] = mask_at(1, 1)
    stack[0, 3] = mask_at(7, 7)
    stack[1, 0] = mask_at(3, 3)
    stack[1, 3] = mask_at(9, 9)

    result = fill_missing_masks(stack, axes="TZYX", interpolation_axis="Z")

    assert result.shape == stack.shape
    assert result.dtype == stack.dtype
    assert np.all(np.any(result != 0, axis=(2, 3)))
    assert np.array_equal(result[:, (0, 3)], stack[:, (0, 3)])


def test_fill_missing_masks_interpolates_time_for_every_z_plane():
    stack = np.zeros((4, 2, 16, 16), dtype=np.uint8)
    stack[0, 0] = mask_at(1, 1)
    stack[3, 0] = mask_at(7, 7)

    result = fill_missing_masks(stack, axes="TZYX", interpolation_axis="T")

    assert np.all(np.any(result[:, 0] != 0, axis=(1, 2)))
    assert not np.any(result[:, 1])
