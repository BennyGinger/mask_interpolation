import numpy as np

from mask_interpolation.pair import interpolate_pair


def square_mask(top: int, left: int, size: int, *, label: int = 1) -> np.ndarray:
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[top:top + size, left:left + size] = label
    return mask


def test_interpolate_pair_preserves_label_shape_and_centroid_trajectory():
    start = square_mask(2, 2, 4, label=7)
    end = square_mask(10, 10, 4, label=7)

    result = interpolate_pair(start, end, 3, label=np.uint16(7))

    assert result.shape == (3, 20, 20)
    assert result.dtype == start.dtype
    assert np.array_equal(np.unique(result), [0, 7])
    centroids = np.array([np.argwhere(mask != 0).mean(axis=0) for mask in result])
    expected = np.linspace([3.5, 3.5], [11.5, 11.5], 5)[1:-1]
    assert np.allclose(centroids, expected, atol=1)
    assert np.all(np.diff(centroids, axis=0) > 0)


def test_interpolate_pair_morphs_between_different_shapes():
    start = square_mask(6, 6, 3)
    end = square_mask(4, 4, 7)

    result = interpolate_pair(start, end, 3, label=np.uint16(1))

    areas = np.count_nonzero(result, axis=(1, 2))
    assert np.all(np.diff(areas) >= 0)
    assert areas[0] >= np.count_nonzero(start)
    assert areas[-1] <= np.count_nonzero(end)
