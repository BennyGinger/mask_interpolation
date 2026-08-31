import numpy as np
import pytest

from mask_interpolation.validation import validate_array, validate_sequence


@pytest.mark.parametrize(
    ("stack", "axes", "interpolation_axis", "message"),
    [
        (np.zeros((8, 8)), "YX", "Y", "at least three dimensions"),
        (np.zeros((2, 8, 8)), "YX", "Z", "one label"),
        (np.zeros((2, 8, 8)), "ZYY", "Z", "duplicate"),
        (np.zeros((2, 8, 8)), "ZYX", "C", "not present"),
        (np.zeros((2, 8, 8)), "ZYX", "X", "cannot be"),
    ],
)
def test_validate_array_rejects_invalid_axes(stack, axes, interpolation_axis, message):
    with pytest.raises(ValueError, match=message):
        validate_array(stack, axes, interpolation_axis)


def test_validate_sequence_rejects_empty_sequence():
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_sequence(np.zeros((3, 10, 10), dtype=np.uint8))


def test_validate_sequence_rejects_different_labels():
    sequence = np.zeros((3, 10, 10), dtype=np.uint8)
    sequence[0, 2:5, 2:5] = 1
    sequence[2, 4:7, 4:7] = 2

    with pytest.raises(ValueError, match="one nonzero label"):
        validate_sequence(sequence)
